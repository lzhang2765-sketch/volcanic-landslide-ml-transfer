"""Stage 4E-B execution under the frozen preregistration.

New fitting is limited to XGBoost.  Historical M results and held-out scores for
Logistic/RF/HistGB are reused from Stage 4B and verified before use.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import stage4a_base as B
from . import stage4b as S
from . import stage4d as D
from . import stage4eb_prereg as P
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_b"
CACHE = ROOT / "data/interim/_stage4eb_xgb_cache"
RESULTS = ROOT / "data/interim/_stage4eb_results"
STAGE4B_CACHE = ROOT / "data/interim/_stage4b_cache"
STAGE4B_RESULTS = ROOT / "data/interim/_stage4b_results/PRIMARY.json"
PREREG_FREEZE = OUT / "STAGE4EB_PREREGISTRATION_FREEZE.json"


def verify_prereg() -> dict:
    if not PREREG_FREEZE.exists():
        raise RuntimeError("STOP: Stage 4E-B preregistration is not frozen")
    fr = json.loads(PREREG_FREEZE.read_text())
    for rel, h in fr["files"].items():
        if sha256(ROOT / rel) != h:
            raise RuntimeError(f"STOP: frozen preregistration changed: {rel}")
    return fr


def xgb_module():
    import xgboost
    if xgboost.__version__ != P.XGBOOST_VERSION:
        raise RuntimeError(f"STOP: XGBoost {xgboost.__version__} != frozen {P.XGBOOST_VERSION}")
    return xgboost


def _key(train: list[str], fs: str, cfg: dict) -> str:
    payload = ["stage4eb", sorted(train), fs, cfg, P.XGBOOST_FIXED, "ALL_VALID", "LANDSCAPE_BALANCED"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:28]


def fit_xgb(d: S.Data, train: list[str], fs: str, cfg: dict):
    xgb = xgb_module()
    cols = [d.idx[c] for c in d.fs[fs]]
    samples = [d.sample(l, "ALL_VALID") for l in train]
    X = np.vstack([d.L[l]["X"][r][:, cols] for l, r in zip(train, samples)]).astype(np.float32)
    y = np.concatenate([d.L[l]["y"][r] for l, r in zip(train, samples)])
    ws, audit = S.weights(d, train, "ALL_VALID", "LANDSCAPE_BALANCED")
    w = np.concatenate(ws)
    est = xgb.XGBClassifier(**P.XGBOOST_FIXED, **cfg)
    t0 = time.perf_counter()
    est.fit(X, y, sample_weight=w)
    fit_s = time.perf_counter() - t0
    return est, cols, audit, dict(train_rows=len(y), train_pos=int(y.sum()), fit_seconds=fit_s,
                                  peak_process_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)))


def predict_xgb(d: S.Data, est, cols, lid: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = est.predict_proba(d.L[lid]["X"][:, cols].astype(np.float32))[:, 1].astype(np.float64)
    return p, time.perf_counter() - t0


def cached_predictions(d: S.Data, train: list[str], fs: str, cfg: dict, need: list[str], save_model=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = _key(train, fs, cfg)
    files = {l: CACHE / f"{key}__{l}.npy" for l in need}
    meta_file = CACHE / f"{key}__meta.json"
    model_file = CACHE / f"{key}__model.json"
    if all(p.exists() for p in files.values()) and meta_file.exists() and (not save_model or model_file.exists()):
        return {l: np.load(p) for l, p in files.items()}, json.loads(meta_file.read_text())
    est, cols, waudit, meta = fit_xgb(d, train, fs, cfg)
    timings = {}
    for lid in P.LANDSCAPES:
        if lid in train:
            continue
        pred, sec = predict_xgb(d, est, cols, lid)
        np.save(CACHE / f"{key}__{lid}.npy", pred)
        timings[lid] = sec
    raw = est.get_booster().save_raw(raw_format="ubj")
    meta.update(key=key, train=train, feature_set=fs, config=cfg, fixed=P.XGBOOST_FIXED,
                columns=d.fs[fs], weighting=waudit, inference_seconds=timings,
                serialized_bytes=len(raw), n_trees=len(est.get_booster().get_dump()))
    meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if save_model:
        est.save_model(model_file)
    return {l: np.load(files[l]) for l in need}, meta


def run_xgb_feature_set(d: S.Data, fs: str) -> dict:
    """Nested LOLO for one preregistered XGBoost feature set."""
    verify_prereg()
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS / f"XGBOOST_{fs}.json"
    if result_file.exists():
        return json.loads(result_file.read_text())
    results, inner, meta = [], [], {}
    for held in P.LANDSCAPES:
        train = [l for l in P.LANDSCAPES if l != held]
        scores = []
        for gi, cfg in enumerate(P.XGBOOST_GRID):
            per = {}
            for inner_held in train:
                inner_train = [l for l in train if l != inner_held]
                pp, _ = cached_predictions(d, inner_train, fs, cfg, [inner_held])
                per[inner_held] = float(average_precision_score(d.L[inner_held]["y"], pp[inner_held]))
            mean_ap = float(np.mean(list(per.values())))
            scores.append((mean_ap, gi, per))
            inner.append(dict(model="XGBOOST", feature_set=fs, outer_held_out=held, grid_index=gi,
                              params=json.dumps(cfg, sort_keys=True), inner_mean_pr_auc=mean_ap,
                              **{f"inner_{k}": v for k, v in per.items()}))
        best = max(x[0] for x in scores)
        gi = min(x[1] for x in scores if x[0] >= best - S.TIE_TOL)
        cfg = P.XGBOOST_GRID[gi]
        # Training-only Stage 4B-style calibration and threshold; neither enters architecture selection.
        ps, ys, ws = [], [], []
        for inner_held in train:
            pp, _ = cached_predictions(d, [l for l in train if l != inner_held], fs, cfg, [inner_held])
            y = d.L[inner_held]["y"]
            ps.append(pp[inner_held]); ys.append(y); ws.append(np.full(len(y), 1.0 / (len(y) * len(train))))
        p_in, y_in, w_in = np.concatenate(ps), np.concatenate(ys), np.concatenate(ws)
        cal = S.fit_platt(p_in, y_in, w_in)
        threshold = S.best_threshold(S.apply_platt(p_in, cal), y_in, w_in)
        pp, cmeta = cached_predictions(d, train, fs, cfg, [held], save_model=True)
        raw = pp[held]
        calibrated = S.apply_platt(raw, cal)
        m = S.metrics(d.L[held]["y"], raw, calibrated, threshold)
        raw_i, raw_s = S.calibration_slope_intercept(raw, d.L[held]["y"])
        results.append(dict(analysis="PRIMARY" if fs == "M" else "HISTORICAL_CONTROL", held_out=held,
                            model="XGBOOST", feature_set=fs, params=json.dumps(cfg, sort_keys=True), grid_index=gi,
                            calibrated=cal is not None, calibrator=json.dumps(cal),
                            raw_cal_intercept=raw_i, raw_cal_slope=raw_s, inner_best_mean_pr_auc=best, **m))
        meta[held] = dict(train=train, selected_grid_index=gi, params=cfg, inner_scores=[x[0] for x in scores],
                          calibrator=cal, threshold=threshold, outer_prediction_key=cmeta["key"], complexity=cmeta)
    obj = dict(feature_set=fs, results=results, inner=inner, meta=meta)
    result_file.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return obj


def historical_m() -> tuple[pd.DataFrame, dict]:
    """Exact Stage 4B M rows and raw held-out scores, with full-precision verification."""
    full = json.loads(STAGE4B_RESULTS.read_text())["results"]
    rows, pred = [], {}
    for r in full:
        if r["model"] not in P.MODELS[:3] or r["feature_set"] != "M":
            continue
        held, model = r["held_out"], r["model"]
        train = [l for l in P.LANDSCAPES if l != held]
        params = json.loads(r["params"])
        key = S._key(sorted(train), "M", model, params, "ALL_VALID", "LANDSCAPE_BALANCED")
        p = np.load(STAGE4B_CACHE / f"{key}__{held}.npy")
        with np.load(S.MEG_DIR / f"{held}.npz", allow_pickle=False) as z:
            y = z["y_positive"].astype(bool)
        ap = float(average_precision_score(y, p))
        if abs(ap - r["pr_auc"]) > 1e-12:
            raise RuntimeError(f"STOP: Stage 4B reuse mismatch {held}/{model}: {ap} != {r['pr_auc']}")
        raw_i, raw_s = S.calibration_slope_intercept(p, y)
        row = {k: v for k, v in r.items() if k not in ("analysis",)}
        row.update(analysis="PRIMARY", raw_cal_intercept=raw_i, raw_cal_slope=raw_s, source="STAGE4B_VERIFIED_REUSE")
        rows.append(row)
        cal = json.loads(r["calibrator"]) if r.get("calibrator") not in (None, "null") else None
        pred[(model, held)] = dict(raw=p, cal=S.apply_platt(p, cal), thr=float(r["threshold"]), source="STAGE4B_VERIFIED_REUSE")
    return pd.DataFrame(rows), pred


def primary_state(d: S.Data, xobj: dict):
    hist, pred = historical_m()
    xrows = pd.DataFrame(xobj["results"])
    for r in xobj["results"]:
        held = r["held_out"]
        key = xobj["meta"][held]["outer_prediction_key"]
        raw = np.load(CACHE / f"{key}__{held}.npy")
        cal = json.loads(r["calibrator"]) if r["calibrator"] not in (None, "null") else None
        pred[("XGBOOST", held)] = dict(raw=raw, cal=S.apply_platt(raw, cal), thr=float(r["threshold"]), source="STAGE4EB_NEW_FIT")
    T = {}
    for held in P.LANDSCAPES:
        with np.load(S.MEG_DIR / f"{held}.npz", allow_pickle=False) as z:
            T[held] = dict(y=z["y_positive"].astype(bool), b3=z["spatial_block_3000m"], b6=z["spatial_block_6000m"],
                           cx=z["centroid_x"], cy=z["centroid_y"], ids=z["terrain_unit_id"])
    return pd.concat([hist, xrows], ignore_index=True, sort=False), pred, T


def bootstrap(T: dict, pred: dict) -> dict:
    out = {}
    for li, held in enumerate(P.LANDSCAPES):
        y = T[held]["y"]
        code, nb = D.block_codes(T[held]["b3"])
        sorters = {m: D.Sorted(pred[(m, held)]["raw"], y) for m in P.MODELS}
        rng = np.random.default_rng(P.SEED + li)
        R = {m: np.full((P.BOOTSTRAP["replicates"], 2), np.nan) for m in P.MODELS}
        R["prev"] = np.full(P.BOOTSTRAP["replicates"], np.nan)
        dropped = 0
        for i in range(P.BOOTSTRAP["replicates"]):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code].astype(float)
            if w[y].sum() == 0:
                dropped += 1
                continue
            R["prev"][i] = w[y].sum() / w.sum()
            for m in P.MODELS:
                R[m][i] = sorters[m].ap_auc(w)
        R["dropped"] = dropped
        out[held] = R
    return out


def pct(x) -> tuple[float, float]:
    x = np.asarray(x); x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def bootstrap_summary(lolo: pd.DataFrame, BS: dict) -> pd.DataFrame:
    rows = []
    for held in P.LANDSCAPES:
        prev = float(lolo[lolo.held_out == held].prevalence.iloc[0])
        for m in P.MODELS:
            r = lolo[(lolo.held_out == held) & (lolo.model == m)].iloc[0]
            for j, met in enumerate(("pr_auc", "roc_auc")):
                lo, hi = pct(BS[held][m][:, j])
                rows.append(dict(held_out=held, model=m, feature_set="M", comparison="POINT_INTERVAL", metric=met,
                                 point_estimate=float(r[met]), ci_low=lo, ci_high=hi,
                                 n_bootstrap=P.BOOTSTRAP["replicates"] - BS[held]["dropped"]))
            lo, hi = pct(BS[held][m][:, 0] / BS[held]["prev"])
            rows.append(dict(held_out=held, model=m, feature_set="M", comparison="POINT_INTERVAL", metric="pr_lift",
                             point_estimate=float(r.pr_auc / prev), ci_low=lo, ci_high=hi,
                             n_bootstrap=P.BOOTSTRAP["replicates"] - BS[held]["dropped"]))
            if m != "LOGISTIC_L2":
                delta = BS[held][m][:, 0] - BS[held]["LOGISTIC_L2"][:, 0]
                lo, hi = pct(delta)
                rows.append(dict(held_out=held, model=m, feature_set="M", comparison="PAIRED_VS_LOGISTIC", metric="delta_pr_auc",
                                 point_estimate=float(r.pr_auc - lolo[(lolo.held_out == held) & (lolo.model == "LOGISTIC_L2")].pr_auc.iloc[0]),
                                 ci_low=lo, ci_high=hi, n_bootstrap=P.BOOTSTRAP["replicates"] - BS[held]["dropped"]))
    return pd.DataFrame(rows)


def comparison(lolo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in P.MODELS:
        g = lolo[lolo.model == m].set_index("held_out").reindex(P.LANDSCAPES)
        lift = g.pr_auc / g.prevalence
        rows.append(dict(model=m, feature_set="M", median_pr_auc=float(g.pr_auc.median()), mean_pr_auc=float(g.pr_auc.mean()),
                         median_pr_lift=float(lift.median()), worst_pr_lift=float(lift.min()), worst_landscape=lift.idxmin(),
                         iqr_pr_lift=float(lift.quantile(.75) - lift.quantile(.25)), sd_pr_lift=float(lift.std(ddof=1)),
                         sd_log_pr_lift=float(np.log(lift).std(ddof=1)),
                         robust_score=float(.70 * np.log(lift.median()) + .30 * np.log(lift.min()))))
    return pd.DataFrame(rows)


def selection(lolo: pd.DataFrame, comp: pd.DataFrame, BS: dict) -> tuple[pd.DataFrame, dict]:
    lr = lolo[lolo.model == "LOGISTIC_L2"].set_index("held_out")
    rows = []
    for m in P.MODELS:
        g = lolo[lolo.model == m].set_index("held_out")
        failures, details = [], []
        for held in P.LANDSCAPES:
            lift, llift = g.loc[held, "pr_auc"] / g.loc[held, "prevalence"], lr.loc[held, "pr_auc"] / lr.loc[held, "prevalence"]
            delta = BS[held][m][:, 0] - BS[held]["LOGISTIC_L2"][:, 0]
            lo, hi = pct(delta)
            if held == "COTABATO_DAVAO_2019":
                fail = lift < .80 and hi < 0
            else:
                fail = llift >= 1.25 and hi < 0 and (lift <= 1.0 or lift < .67 * llift)
            if fail:
                failures.append(held)
            details.append(f"{held}:{lift:.6g}/{llift:.6g}/dCI[{lo:.6g},{hi:.6g}]/{'FAIL' if fail else 'PASS'}")
        c = comp[comp.model == m].iloc[0]
        rows.append(dict(model=m, feature_set="M", robust_score=float(c.robust_score), median_pr_auc=float(c.median_pr_auc),
                         median_pr_lift=float(c.median_pr_lift), worst_pr_lift=float(c.worst_pr_lift), catastrophic_fail=bool(failures),
                         catastrophic_landscapes="|".join(failures), guard_details=";".join(details)))
    tab = pd.DataFrame(rows)
    eligible = tab[~tab.catastrophic_fail].copy()
    if eligible.empty:
        decision = dict(development_leader="NO_CLEAR_LEADER", runner_up="", reason="all candidates failed catastrophic guard")
        return tab, decision
    eligible["simplicity"] = eligible.model.map(P.COMPLEXITY_ORDER.index)
    eligible = eligible.sort_values(["robust_score", "simplicity"], ascending=[False, True])
    raw_best = str(eligible.iloc[0].model)
    leader, reasons = raw_best, [f"raw best robust score: {raw_best}"]
    lr_score = float(tab[tab.model == "LOGISTIC_L2"].robust_score.iloc[0])
    best_score = float(tab[tab.model == raw_best].robust_score.iloc[0])
    median_delta = float(lolo[lolo.model == raw_best].set_index("held_out").pr_auc.sub(lr.pr_auc).median())
    boot_med = np.nanmedian(np.column_stack([BS[k][raw_best][:, 0] - BS[k]["LOGISTIC_L2"][:, 0] for k in P.LANDSCAPES]), axis=1)
    med_lo, med_hi = pct(boot_med)
    relative_adv = (best_score - lr_score) / max(abs(lr_score), 1e-12)
    simplicity = False
    if raw_best != "LOGISTIC_L2" and relative_adv <= P.SELECTION["logistic_relative_tolerance"]:
        leader, simplicity = "LOGISTIC_L2", True
        reasons.append(f"5% robust-score simplicity rule activated ({relative_adv:.6g})")
    elif raw_best != "LOGISTIC_L2" and median_delta <= P.SELECTION["tiny_median_pr_auc_delta"] and med_lo <= 0 <= med_hi:
        leader, simplicity = "LOGISTIC_L2", True
        reasons.append(f"tiny uncertain median PR-AUC rule activated (delta {median_delta:.6g}, CI [{med_lo:.6g},{med_hi:.6g}])")
    runner = str(eligible[eligible.model != leader].iloc[0].model) if len(eligible[eligible.model != leader]) else ""
    margin = float(tab[tab.model == leader].robust_score.iloc[0] - tab[tab.model == runner].robust_score.iloc[0]) if runner else np.nan
    decision = dict(development_leader=leader, runner_up=runner, selection_margin=margin, raw_best=raw_best,
                    raw_best_score=best_score, logistic_score=lr_score, relative_score_advantage_vs_logistic=relative_adv,
                    median_pr_auc_delta_vs_logistic=median_delta, median_delta_bootstrap_ci=[med_lo, med_hi],
                    simplicity_tiebreak_activated=simplicity, reason="; ".join(reasons))
    tab["selected_development_leader"] = tab.model.eq(leader)
    tab["runner_up"] = tab.model.eq(runner)
    return tab, decision


def rank_stability(lolo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for omit in P.LANDSCAPES:
        d = lolo[lolo.held_out != omit]
        c = comparison(d)
        c["simplicity"] = c.model.map(P.COMPLEXITY_ORDER.index)
        c = c.sort_values(["robust_score", "simplicity"], ascending=[False, True])
        raw = str(c.iloc[0].model)
        leader = raw
        lr_score = float(c[c.model == "LOGISTIC_L2"].robust_score.iloc[0])
        rel = (float(c.iloc[0].robust_score) - lr_score) / max(abs(lr_score), 1e-12)
        if raw != "LOGISTIC_L2" and rel <= P.SELECTION["logistic_relative_tolerance"]:
            leader = "LOGISTIC_L2"
        for rank, r in enumerate(c.itertuples(), 1):
            rows.append(dict(omitted_landscape=omit, model=r.model, robust_score=r.robust_score, median_pr_auc=r.median_pr_auc,
                             median_pr_lift=r.median_pr_lift, worst_pr_lift=r.worst_pr_lift, raw_rank=rank,
                             selected_architecture=leader, is_selected=r.model == leader))
    return pd.DataFrame(rows)
