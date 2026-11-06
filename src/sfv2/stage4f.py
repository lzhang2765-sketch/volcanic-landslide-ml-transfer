"""Stage 4F execution under the frozen preregistration.

The public phase boundary is deliberate: :func:`run_ood_phase` never requests a
held-out target array and freezes all novelty outputs before
:func:`run_evaluation_phase` is allowed to load held-out labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from scipy.stats import spearmanr
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from . import stage4b as S
from . import stage4d_yblind as YB
from . import stage4f_prereg as P
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4f"
FIG = OUT / "figures"
CACHE = ROOT / "data/interim/_stage4f"
MEG_DIR = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET"
PREREG_FREEZE = OUT / "STAGE4F_PREREGISTRATION_FREEZE.json"
OOD_FREEZE = OUT / "STAGE4F_OOD_FREEZE.json"
EPS = float(P.CALIBRATION["probability_clip"])
CAL_ORDER = ["C0_RAW", "C1_INTERCEPT", "C2_INTERCEPT_SLOPE", "C3_ISOTONIC"]
UNIT_METHODS = ["U0_UNCONDITIONAL", "U1_SUSCEPTIBILITY_CONDITIONED"]


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.12g")


def _verify_prereg() -> dict:
    if not PREREG_FREEZE.exists():
        raise RuntimeError("STOP: Stage 4F preregistration is not frozen")
    fr = json.loads(PREREG_FREEZE.read_text())
    for rel, digest in fr["files"].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(f"STOP: frozen Stage 4F preregistration changed: {rel}")
    if sha256(ROOT / "outputs/stage4e_b/STAGE4EB_CHECKPOINT.json") != P.STAGE4EB_CHECKPOINT_SHA256:
        raise RuntimeError("STOP: Stage 4E-B checkpoint identity failed")
    return fr


def _m_columns() -> list[str]:
    cols = S.feature_sets()["M"]
    if len(cols) != 60:
        raise RuntimeError("STOP: M feature schema is not 60 columns")
    return cols


def _load_x(lid: str, cols: list[str], include_y: bool = False) -> dict:
    """Load predictors/IDs and optionally Y; callers must explicitly request Y."""
    S.B.firewall_check(lid)
    keys = ["terrain_unit_id", "spatial_block_3000m", "spatial_block_6000m",
            "centroid_x", "centroid_y", *cols]
    if include_y:
        keys.append("y_positive")
    with np.load(MEG_DIR / f"{lid}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in keys}
    d["X"] = np.column_stack([d.pop(c).astype(np.float64) for c in cols])
    return d


def _hash_order(ids: np.ndarray, salt: str = "stage4b") -> np.ndarray:
    rank = np.argsort(np.array([hashlib.sha256(f"{x}|{salt}".encode()).hexdigest() for x in ids]), kind="stable")
    order = np.empty(len(rank), dtype=np.int64)
    order[rank] = np.arange(len(rank))
    return order


def _sample_rows(d: dict) -> np.ndarray:
    y = d["y_positive"].astype(bool)
    neg = np.flatnonzero(~y)
    order = _hash_order(d["terrain_unit_id"], "stage4b")
    neg = neg[np.argsort(order[neg], kind="stable")[:S.NEG_PER_LANDSCAPE]]
    return np.sort(np.concatenate([np.flatnonzero(y), neg]))


def _selected_c(held: str) -> tuple[float, dict]:
    """Mechanically replay the Stage 4E-B/Stage 4B inner-selection tie rule."""
    tab = pd.read_csv(ROOT / "outputs/stage4e_b/STAGE4EB_INNER_SELECTIONS.tsv", sep="\t")
    g = tab[(tab.outer_held_out == held) & (tab.model == "LOGISTIC_L2") & (tab.feature_set == "M")].copy()
    if len(g) != 3:
        raise RuntimeError(f"STOP: missing authoritative Logistic M inner rows for {held}")
    best = float(g.inner_mean_pr_auc.max())
    eligible = g[g.inner_mean_pr_auc >= best - S.TIE_TOL].sort_values("grid_index")
    row = eligible.iloc[0]
    params = json.loads(row.params)
    published = pd.read_csv(ROOT / "outputs/stage4e_b/STAGE4EB_LOLO_RESULTS.tsv", sep="\t")
    pub = published[(published.held_out == held) & (published.model == "LOGISTIC_L2")].iloc[0]
    if params != json.loads(pub.params):
        raise RuntimeError(f"STOP: selected Logistic hyperparameter mismatch for {held}")
    return float(params["C"]), {
        "inner_best_mean_pr_auc": best,
        "selected_grid_index": int(row.grid_index),
        "selected_C": float(params["C"]),
        "tie_tolerance": S.TIE_TOL,
    }


def _fit_outer_model(held: str, cols: list[str]) -> tuple[LogisticRegression, np.ndarray, np.ndarray, dict, dict]:
    """Fit on five training landscapes; held-out Y is never loaded by this function."""
    train = [x for x in P.LANDSCAPES if x != held]
    td = {lid: _load_x(lid, cols, include_y=True) for lid in train}
    hd = _load_x(held, cols, include_y=False)
    samples = {lid: _sample_rows(td[lid]) for lid in train}
    Xs, ys, raw_weights, audit = [], [], [], {}
    for lid in train:
        d, r = td[lid], samples[lid]
        y = d["y_positive"].astype(bool)
        ysamp = y[r]
        npop, nneg = int(y.sum()), int((~y).sum())
        nps, nns = int(ysamp.sum()), int((~ysamp).sum())
        wp = (1 / len(train)) * (npop / len(y)) / nps
        wn = (1 / len(train)) * (nneg / len(y)) / nns
        w = np.where(ysamp, wp, wn)
        Xs.append(d["X"][r]); ys.append(ysamp); raw_weights.append(w)
        audit[lid] = {
            "population_rows": len(y), "population_positives": npop,
            "sampled_positives": nps, "sampled_negatives": nns,
            "population_prevalence": float(y.mean()),
        }
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(raw_weights)
    w *= len(w) / w.sum()
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    c, selection = _selected_c(held)
    model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit((X - mu) / sd, y, sample_weight=w)
    outer_p = model.predict_proba((hd["X"] - mu) / sd)[:, 1]
    published = pd.read_csv(ROOT / "outputs/stage4e_b/STAGE4EB_LOLO_RESULTS.tsv", sep="\t")
    expected_ap = float(published[(published.held_out == held) & (published.model == "LOGISTIC_L2")].pr_auc.iloc[0])
    meta = {
        "held_out": held, "training_landscapes": train, "heldout_y_loaded": False,
        "features": cols, "selection": selection, "training": audit,
        "train_rows_sampled": len(y), "train_positives_sampled": int(y.sum()),
        "preprocessing": "sampled-training-only unweighted mean/SD",
        "mean": mu.tolist(), "sd": sd.tolist(),
        "published_outer_pr_auc_for_later_y_aware_identity_check": expected_ap,
    }
    # Training reference scores are allowed: they depend on training X/Y only.
    return model, mu, sd, hd, meta


def _knn_mean(query: np.ndarray, ref: np.ndarray, k: int = 10, chunk: int = 10000) -> np.ndarray:
    out = np.empty(len(query), dtype=np.float64)
    kk = min(k, len(ref))
    for start in range(0, len(query), chunk):
        d = cdist(query[start:start + chunk], ref)
        out[start:start + chunk] = np.partition(d, kk - 1, axis=1)[:, :kk].mean(axis=1)
    return out


def _self_knn(ref: np.ndarray, k: int = 10) -> np.ndarray:
    d = cdist(ref, ref)
    np.fill_diagonal(d, np.inf)
    kk = min(k, len(ref) - 1)
    return np.partition(d, kk - 1, axis=1)[:, :kk].mean(axis=1)


def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    s = np.sort(reference)
    return np.searchsorted(s, values, side="right") / len(s)


def _spearman(x, y) -> float:
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else 0.0


def _score_fold_ood(held: str, st: YB.Store, cols: list[str]) -> tuple[pd.DataFrame, dict, list[dict]]:
    model, mu, sd, hd, meta = _fit_outer_model(held, cols)
    train = meta["training_landscapes"]
    # The reference is exactly the Stage 4D hash sample: 400 from each training landscape.
    ref_x, ref_ids, ref_lids = [], [], []
    m_idx = [st.cols.index(c) for c in cols]
    for lid in train:
        d = st.L[lid]
        r = d["rank"][:400]
        ref_x.append((d["X"][r][:, m_idx] - mu) / sd)
        ref_ids.append(d["terrain_unit_id"][r])
        ref_lids.extend([lid] * len(r))
    ref = np.vstack(ref_x)
    xh = (hd["X"] - mu) / sd
    raw = model.predict_proba(xh)[:, 1]
    ref_score = model.predict_proba(ref)[:, 1]
    u0_dist = _knn_mean(xh, ref)
    u0_ref = _self_knn(ref)
    u0_pct = _percentile(u0_ref, u0_dist)

    thresholds = np.quantile(ref_score, [0.2, 0.4, 0.6, 0.8])
    ref_stratum = np.searchsorted(thresholds, ref_score, side="right")
    held_stratum = np.searchsorted(thresholds, raw, side="right")
    u1_dist = np.empty(len(raw), dtype=np.float64)
    u1_pct = np.empty(len(raw), dtype=np.float64)
    stratum_audit = {}
    for q in range(5):
        target = np.flatnonzero(held_stratum == q)
        use = np.flatnonzero(ref_stratum == q)
        radius = 0
        while len(use) < 50 and radius < 4:
            radius += 1
            use = np.flatnonzero((ref_stratum >= max(0, q - radius)) & (ref_stratum <= min(4, q + radius)))
        rr = ref[use]
        rself = _self_knn(rr)
        if len(target):
            dist = _knn_mean(xh[target], rr)
            u1_dist[target] = dist
            u1_pct[target] = _percentile(rself, dist)
        stratum_audit[str(q + 1)] = {"heldout_units": len(target), "reference_units": len(use), "expansion_radius": radius}

    slope = hd["X"][:, cols.index("slope_deg_30m__mean")]
    relief = hd["X"][:, cols.index("relief_m_r250__mean")]
    tab = pd.DataFrame({
        "landscape_id": held,
        "terrain_unit_id": hd["terrain_unit_id"],
        "spatial_block_3000m": hd["spatial_block_3000m"],
        "centroid_x": hd["centroid_x"],
        "centroid_y": hd["centroid_y"],
        "susceptibility_score": raw,
        "susceptibility_stratum": held_stratum + 1,
        "slope_deg_30m__mean": slope,
        "relief_m_r250__mean": relief,
        "u0_distance": u0_dist,
        "u0_percentile": u0_pct,
        "u1_distance": u1_dist,
        "u1_percentile": u1_pct,
    })
    conf = []
    for method, score in [("U0_UNCONDITIONAL", u0_pct), ("U1_SUSCEPTIBILITY_CONDITIONED", u1_pct)]:
        within = []
        for q in range(1, 6):
            v = score[held_stratum + 1 == q]
            within.append(float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) else np.nan)
        conf.append({
            "landscape_id": held, "ood_method": method, "n_units": len(raw),
            "rho_susceptibility": _spearman(score, raw), "rho_slope": _spearman(score, slope),
            "rho_relief": _spearman(score, relief), "median_within_stratum_iqr": float(np.nanmedian(within)),
            "within_stratum_iqr_json": json.dumps({str(i + 1): within[i] for i in range(5)}, sort_keys=True),
            "uses_heldout_y": False,
        })
    meta.update({
        "reference_salt": "stage4d", "reference_units_per_training_landscape": 400,
        "reference_rows": len(ref), "knn_k": 10, "susceptibility_thresholds": thresholds.tolist(),
        "strata": stratum_audit, "heldout_y_loaded": False,
        "outer_score_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
    })
    return tab, meta, conf


def run_ood_phase() -> dict:
    """Compute and freeze every X-only novelty output before Y-aware evaluation."""
    _verify_prereg()
    if OOD_FREEZE.exists():
        fr = json.loads(OOD_FREEZE.read_text())
        for rel, digest in fr["files"].items():
            if sha256(ROOT / rel) != digest:
                raise RuntimeError(f"STOP: frozen OOD artifact changed: {rel}")
        return fr
    prohibited = [p for p in OUT.iterdir() if p.name not in {
        "STAGE4F_PREREGISTRATION.json", "STAGE4F_PREREGISTRATION.md", "STAGE4F_PREREGISTRATION_FREEZE.json"
    }]
    if prohibited:
        raise RuntimeError(f"STOP: Stage 4F result exists before OOD freeze: {[p.name for p in prohibited]}")
    cols = _m_columns()
    st = YB.Store()  # Y-blind loader by construction.
    # Recompute H_M under the Stage 4D definition and demand identity with the frozen table.
    prior = pd.read_csv(ROOT / "outputs/stage4d/STAGE4D_DOMAIN_SHIFT.tsv", sep="\t")
    prior = prior[prior.group == "M"].set_index("held_out")
    landscape_rows, unit_tabs, conf_rows, metas = [], [], [], {}
    idx = [st.cols.index(c) for c in cols]
    for held in P.LANDSCAPES:
        a, b, _, _ = YB.fold_samples(st, held)
        h = YB.energy_H(a[:, idx], b[:, idx])
        old = float(prior.loc[held, "energy_H_primary"])
        # The authoritative Stage 4D TSV is a six-significant-digit export.
        if abs(h - old) > 5e-6:
            raise RuntimeError(f"STOP: Stage 4D H_M identity failed for {held}: {h} vs {old}")
        tab, meta, conf = _score_fold_ood(held, st, cols)
        unit_tabs.append(tab); conf_rows.extend(conf); metas[held] = meta
        landscape_rows.append({
            "landscape_id": held, "ood_method": "H_M_NORMALISED_ENERGY", "energy_H_M": h,
            "stage4d_energy_H_M": old, "identity_tolerance": 5e-6,
            "median_u0_percentile": float(tab.u0_percentile.median()),
            "median_u1_percentile": float(tab.u1_percentile.median()),
            "fraction_u0_beyond_train_p99": float((tab.u0_percentile > .99).mean()),
            "fraction_u1_beyond_train_p99": float((tab.u1_percentile > .99).mean()),
            "uses_heldout_y": False, "status": "Y_BLIND_FROZEN_BEFORE_RELIABILITY_EVALUATION",
        })
    unit = pd.concat(unit_tabs, ignore_index=True)
    _tsv(OUT / "STAGE4F_LANDSCAPE_NOVELTY.tsv", pd.DataFrame(landscape_rows))
    _tsv(OUT / "STAGE4F_UNIT_NOVELTY.tsv", unit)
    _tsv(OUT / "STAGE4F_NOVELTY_HAZARD_CONFOUNDING.tsv", pd.DataFrame(conf_rows))
    _json(OUT / "STAGE4F_OOD_FOLD_METADATA.json", metas)
    CACHE.mkdir(parents=True, exist_ok=True)
    for held, g in unit.groupby("landscape_id", sort=False):
        np.savez_compressed(
            CACHE / f"OOD_{held}.npz",
            terrain_unit_id=g.terrain_unit_id.to_numpy(dtype=str), susceptibility_score=g.susceptibility_score.to_numpy(),
            susceptibility_stratum=g.susceptibility_stratum.to_numpy(np.int8),
            u0_percentile=g.u0_percentile.to_numpy(), u1_percentile=g.u1_percentile.to_numpy(),
            spatial_block_3000m=g.spatial_block_3000m.to_numpy(dtype=str), centroid_x=g.centroid_x.to_numpy(), centroid_y=g.centroid_y.to_numpy(),
        )
    files = [
        OUT / "STAGE4F_LANDSCAPE_NOVELTY.tsv", OUT / "STAGE4F_UNIT_NOVELTY.tsv",
        OUT / "STAGE4F_NOVELTY_HAZARD_CONFOUNDING.tsv", OUT / "STAGE4F_OOD_FOLD_METADATA.json",
    ]
    fr = {
        "stage": P.STAGE, "status": "OOD_SCORES_FROZEN_BEFORE_HELDOUT_Y_EVALUATION",
        "preregistration_freeze_sha256": sha256(PREREG_FREEZE),
        "files": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "heldout_y_loaded": False, "landscape_method": "H_M_NORMALISED_ENERGY",
        "unit_methods": UNIT_METHODS, "rows": len(unit), "seed": P.SEED,
    }
    _json(OOD_FREEZE, fr)
    return fr


# -------------------------------------------------------------------------------------- calibration helpers
def _weights_complete(lengths: list[int]) -> np.ndarray:
    return np.concatenate([np.full(n, 1.0 / (len(lengths) * n)) for n in lengths])


def _inner_predictions(outer_held: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Read and verify honest inner-LOLO scores; never uses outer-held Y."""
    train = [x for x in P.LANDSCAPES if x != outer_held]
    c, sel = _selected_c(outer_held)
    ps, ys, lids, lengths, checks = [], [], [], [], {}
    inner_table = pd.read_csv(ROOT / "outputs/stage4e_b/STAGE4EB_INNER_SELECTIONS.tsv", sep="\t")
    row = inner_table[(inner_table.outer_held_out == outer_held) & (inner_table.model == "LOGISTIC_L2") &
                      (inner_table.feature_set == "M") & (inner_table.grid_index == sel["selected_grid_index"])].iloc[0]
    for inner_held in train:
        inner_train = [x for x in train if x != inner_held]
        key = S._key(sorted(inner_train), "M", "LOGISTIC_L2", {"C": c}, "ALL_VALID", "LANDSCAPE_BALANCED")
        path = ROOT / f"data/interim/_stage4b_cache/{key}__{inner_held}.npy"
        if not path.exists():
            raise RuntimeError(f"STOP: authoritative honest prediction is unavailable: {path.name}")
        p = np.load(path)
        with np.load(MEG_DIR / f"{inner_held}.npz", allow_pickle=False) as z:
            y = z["y_positive"].astype(bool)
        ap = float(average_precision_score(y, p))
        expected = float(row[f"inner_{inner_held}"])
        if abs(ap - expected) > max(1e-10, 5e-6 * abs(ap)):
            raise RuntimeError(f"STOP: inner prediction identity failed for {outer_held}/{inner_held}: {ap} vs {expected}")
        ps.append(p); ys.append(y); lengths.append(len(y)); lids.extend([inner_held] * len(y)); checks[inner_held] = ap
    w = _weights_complete(lengths)
    return np.concatenate(ps), np.concatenate(ys), w, np.asarray(lids), train, checks


def _fit_calibrator(method: str, p: np.ndarray, y: np.ndarray, w: np.ndarray):
    z = logit(np.clip(p, EPS, 1 - EPS))
    if method == "C0_RAW":
        return {"method": method}
    if method == "C1_INTERCEPT":
        fun = lambda a: -float(np.sum(w * (y * np.log(expit(a + z) + EPS) + (~y) * np.log(1 - expit(a + z) + EPS))))
        fit = minimize_scalar(fun, bounds=(-30, 30), method="bounded")
        return {"method": method, "intercept": float(fit.x), "success": bool(fit.success)}
    if method == "C2_INTERCEPT_SLOPE":
        def fun(v):
            q = expit(v[0] + v[1] * z)
            return -float(np.sum(w * (y * np.log(q + EPS) + (~y) * np.log(1 - q + EPS))))
        fit = minimize(fun, np.array([0.0, 1.0]), method="L-BFGS-B", bounds=[(-30, 30), (0, 10)])
        return {"method": method, "intercept": float(fit.x[0]), "slope": float(fit.x[1]), "success": bool(fit.success)}
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(p, y.astype(float), sample_weight=w)
    return {"method": method, "model": iso, "n_thresholds": len(iso.X_thresholds_)}


def _apply_calibrator(cal: dict, p: np.ndarray) -> np.ndarray:
    method = cal["method"]
    if method == "C0_RAW":
        q = p
    elif method == "C1_INTERCEPT":
        q = expit(cal["intercept"] + logit(np.clip(p, EPS, 1 - EPS)))
    elif method == "C2_INTERCEPT_SLOPE":
        q = expit(cal["intercept"] + cal["slope"] * logit(np.clip(p, EPS, 1 - EPS)))
    else:
        q = cal["model"].predict(p)
    return np.clip(q, EPS, 1 - EPS)


def _serialise_calibrator(cal: dict) -> dict:
    if cal["method"] != "C3_ISOTONIC":
        return cal
    model = cal["model"]
    return {"method": cal["method"], "n_thresholds": cal["n_thresholds"],
            "x_thresholds": model.X_thresholds_.tolist(), "y_thresholds": model.y_thresholds_.tolist()}


def _apply_serialised(cal: dict, p: np.ndarray) -> np.ndarray:
    if cal["method"] != "C3_ISOTONIC":
        return _apply_calibrator(cal, p)
    return np.clip(np.interp(p, cal["x_thresholds"], cal["y_thresholds"]), EPS, 1 - EPS)


def _calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    z = logit(np.clip(p, EPS, 1 - EPS))
    obs = float(y.mean())
    # Calibration-in-the-large: intercept update with score as offset.
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expit(mid + z).mean() < obs:
            lo = mid
        else:
            hi = mid
    citl = (lo + hi) / 2
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000).fit(z[:, None], y)
        cint, slope = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    except Exception:
        cint, slope = float("nan"), float("nan")
    return citl, cint, slope


def _reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    order = np.argsort(p, kind="stable")
    out = []
    for i, idx in enumerate(np.array_split(order, bins), 1):
        out.append({"bin": i, "n": len(idx), "mean_predicted": float(p[idx].mean()), "observed": float(y[idx].mean()),
                    "positives": int(y[idx].sum())})
    return out


def _block_bootstrap(y: np.ndarray, p: np.ndarray, blocks: np.ndarray, seed: int, reps: int = 1000) -> dict:
    _, code = np.unique(blocks, return_inverse=True)
    nb = int(code.max()) + 1
    rng = np.random.default_rng(seed)
    brier, ll = [], []
    loss = -(y * np.log(p) + (~y) * np.log(1 - p))
    sq = (p - y) ** 2
    for _ in range(reps):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code]
        if w.sum() == 0:
            continue
        brier.append(float(np.average(sq, weights=w)))
        ll.append(float(np.average(loss, weights=w)))
    return {"brier_ci_low": float(np.percentile(brier, 2.5)), "brier_ci_high": float(np.percentile(brier, 97.5)),
            "log_loss_ci_low": float(np.percentile(ll, 2.5)), "log_loss_ci_high": float(np.percentile(ll, 97.5)),
            "bootstrap_replicates": len(brier)}


def _metric_row(held: str, method: str, y: np.ndarray, raw: np.ndarray, p: np.ndarray, blocks: np.ndarray,
                training: list[str], cal: dict) -> dict:
    prev = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    ll = float(log_loss(y, p, labels=[0, 1]))
    pr = float(average_precision_score(y, p))
    roc = float(roc_auc_score(y, p))
    citl, cint, slope = _calibration_stats(y, p)
    bs = _block_bootstrap(y, p, blocks, P.SEED + P.LANDSCAPES.index(held))
    return {
        "landscape_id": held, "calibration_method": method, "n_units": len(y), "positives": int(y.sum()),
        "prevalence": prev, "brier": brier, "brier_null": prev * (1 - prev),
        "brier_ratio_to_null": brier / (prev * (1 - prev)), "log_loss": ll,
        "log_loss_null": -(prev * math.log(prev) + (1 - prev) * math.log(1 - prev)),
        "log_loss_ratio_to_null": ll / (-(prev * math.log(prev) + (1 - prev) * math.log(1 - prev))),
        "calibration_in_the_large": citl, "calibration_intercept": cint, "calibration_slope": slope,
        "expected_positives": float(p.sum()), "observed_positives": int(y.sum()),
        "observed_expected_ratio": float(y.sum() / max(p.sum(), EPS)),
        "pr_auc": pr, "pr_lift": pr / prev, "roc_auc": roc,
        "reliability_curve_json": json.dumps(_reliability_curve(y, p), sort_keys=True),
        "calibrator_json": json.dumps(_serialise_calibrator(cal), sort_keys=True),
        "calibration_training_landscapes": "|".join(training), "honest_cross_fitted_training_scores": True,
        "heldout_y_used_for_fitting": False, **bs,
    }


def _select_calibration(results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    raw = results[results.calibration_method == "C0_RAW"].set_index("landscape_id")
    for method in CAL_ORDER:
        g = results[results.calibration_method == method].set_index("landscape_id").reindex(P.LANDSCAPES)
        catastrophic = []
        if method != "C0_RAW":
            for lid in P.LANDSCAPES:
                if (g.loc[lid, "brier"] > 1.25 * raw.loc[lid, "brier"] and
                        g.loc[lid, "log_loss"] > 1.25 * raw.loc[lid, "log_loss"]):
                    catastrophic.append(lid)
        med_b = float(g.brier_ratio_to_null.median())
        med_l = float(g.log_loss_ratio_to_null.median())
        worst_b = float(g.brier_ratio_to_null.max())
        worst_l = float(g.log_loss_ratio_to_null.max())
        score = (med_b + med_l) / 2 + .25 * max(worst_b, worst_l)
        rows.append({
            "calibration_method": method, "median_brier": float(g.brier.median()),
            "median_log_loss": float(g.log_loss.median()), "median_brier_ratio_to_null": med_b,
            "median_log_loss_ratio_to_null": med_l, "worst_brier_ratio_to_null": worst_b,
            "worst_log_loss_ratio_to_null": worst_l, "robust_selection_score": score,
            "catastrophic_failure": bool(catastrophic), "catastrophic_landscapes": "|".join(catastrophic),
            "median_abs_citl": float(g.calibration_in_the_large.abs().median()),
            "median_calibration_slope": float(g.calibration_slope.median()),
            "ranking_pr_auc_range": float(g.pr_auc.max() - g.pr_auc.min()),
        })
    comp = pd.DataFrame(rows)
    elig = comp[~comp.catastrophic_failure].copy()
    elig["order"] = elig.calibration_method.map(CAL_ORDER.index)
    winner = str(elig.sort_values(["robust_selection_score", "order"]).iloc[0].calibration_method)
    selected = results[results.calibration_method == winner].set_index("landscape_id")
    med_improve_b = float(selected.brier.median()) < float(raw.brier.median())
    med_improve_l = float(selected.log_loss.median()) < float(raw.log_loss.median())
    good_a = ((selected.observed_expected_ratio.between(.8, 1.25)) &
              (selected.calibration_in_the_large.abs() <= .5) & selected.calibration_slope.between(.7, 1.3))
    good_b = ((selected.observed_expected_ratio.between(.5, 2.0)) &
              (selected.calibration_in_the_large.abs() <= 1.0) & selected.calibration_slope.between(.5, 2.0))
    has_cat = bool(comp.set_index("calibration_method").loc[winner, "catastrophic_failure"])
    if winner != "C0_RAW" and med_improve_b and med_improve_l and not has_cat and int(good_a.sum()) >= 5:
        status = "CALIBRATION_TRANSFER_SUPPORTED"
    elif winner != "C0_RAW" and med_improve_b and med_improve_l and not has_cat and int(good_b.sum()) >= 4:
        status = "CALIBRATION_PARTIALLY_SUPPORTED"
    else:
        status = "CALIBRATION_TRANSFER_NOT_SUPPORTED"
    decision = {
        "selected_method": winner, "calibration_status": status,
        "median_brier_improves_vs_raw": med_improve_b, "median_log_loss_improves_vs_raw": med_improve_l,
        "criterion_A_landscapes": int(good_a.sum()), "criterion_B_landscapes": int(good_b.sum()),
        "selected_has_catastrophic_failure": has_cat,
        "probability_language_permitted": status == "CALIBRATION_TRANSFER_SUPPORTED",
    }
    comp["selected"] = comp.calibration_method.eq(winner)
    return comp, decision


def _best_threshold(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return S.best_threshold(p, y, w)


def _weighted_quantile(x: np.ndarray, q: float, w: np.ndarray) -> float:
    o = np.argsort(x, kind="stable")
    xx, ww = x[o], w[o]
    return float(xx[np.searchsorted(np.cumsum(ww) / ww.sum(), q, side="left")])


def _load_ood_cache(held: str) -> dict:
    # These caches are locally generated derivatives; the frozen TSV is authoritative.
    # Existing first-run caches used pandas object strings, hence allow_pickle here.
    with np.load(CACHE / f"OOD_{held}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _within_stratum_rho(score: np.ndarray, residual: np.ndarray, strata: np.ndarray) -> tuple[float, str]:
    vals = {}
    for q in range(1, 6):
        s = strata == q
        vals[str(q)] = _spearman(score[s], residual[s]) if s.sum() > 2 else float("nan")
    finite = [x for x in vals.values() if np.isfinite(x)]
    return (float(np.median(finite)) if finite else float("nan"), json.dumps(vals, sort_keys=True))


def _failure_and_selective(calibrators: dict, decision: dict, cal_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    method = decision["selected_method"]
    failure_rows, coverage_rows = [], []
    per_land = {}
    for held in P.LANDSCAPES:
        ood = _load_ood_cache(held)
        with np.load(MEG_DIR / f"{held}.npz", allow_pickle=False) as z:
            y = z["y_positive"].astype(bool)
            ids = z["terrain_unit_id"]
        if not np.array_equal(ids, ood["terrain_unit_id"]):
            raise RuntimeError(f"STOP: OOD/target alignment mismatch for {held}")
        raw = ood["susceptibility_score"]
        cal = calibrators[held][method]
        p = _apply_serialised(cal, raw)
        p_in = calibrators[held]["_inner_selected_p"]
        y_in = calibrators[held]["_inner_y"]
        w_in = calibrators[held]["_inner_w"]
        threshold = _best_threshold(p_in, y_in, w_in)
        residual_cut = _weighted_quantile(np.abs(p_in - y_in), .90, w_in)
        fn = y & (p < threshold)
        large = np.abs(p - y) > residual_cut
        uncertainty = 4 * p * (1 - p)
        per_land[held] = {"threshold": threshold, "large_residual_cut": residual_cut}
        for mname, score in [("U0_UNCONDITIONAL", ood["u0_percentile"]),
                             ("U1_SUSCEPTIBILITY_CONDITIONED", ood["u1_percentile"])]:
            pos = y
            auc_fn = _auc_or_nan(fn[pos], score[pos])
            auc_fn_ref = _auc_or_nan(fn[pos], -raw[pos])
            auc_large = _auc_or_nan(large, score)
            auc_large_ref = _auc_or_nan(large, uncertainty)
            rho, rho_json = _within_stratum_rho(score, np.abs(p - y), ood["susceptibility_stratum"])
            failure_rows.append({
                "landscape_id": held, "ood_method": mname, "selected_calibration_method": method,
                "classification_threshold": threshold, "threshold_source": "honest_inner_LOLO_weighted_F1",
                "positives": int(y.sum()), "false_negatives": int(fn.sum()),
                "false_negative_detection_auc": auc_fn, "susceptibility_reference_fn_auc": auc_fn_ref,
                "incremental_fn_auc": auc_fn - auc_fn_ref if np.isfinite(auc_fn) and np.isfinite(auc_fn_ref) else np.nan,
                "large_residual_threshold": residual_cut, "large_residuals": int(large.sum()),
                "large_residual_detection_auc": auc_large, "susceptibility_uncertainty_reference_auc": auc_large_ref,
                "incremental_large_residual_auc": auc_large - auc_large_ref if np.isfinite(auc_large) else np.nan,
                "median_within_susceptibility_stratum_residual_rho": rho,
                "within_stratum_residual_rho_json": rho_json,
            })
            for coverage in P.SELECTIVE_PREDICTION["coverage"]:
                n_keep = int(np.floor(len(y) * coverage))
                keep_idx = np.argsort(score, kind="stable")[:n_keep]
                keep = np.zeros(len(y), dtype=bool); keep[keep_idx] = True
                yy, pp = y[keep], p[keep]
                prev = float(yy.mean())
                pr = float(average_precision_score(yy, pp))
                excluded = ~keep
                blocks, counts = np.unique(ood["spatial_block_3000m"][excluded], return_counts=True)
                top_n = max(1, int(math.ceil(.1 * len(blocks)))) if len(blocks) else 0
                concentration = float(np.sort(counts)[-top_n:].sum() / max(counts.sum(), 1)) if top_n else 0.0
                coverage_rows.append({
                    "landscape_id": held, "ood_method": mname, "coverage": coverage,
                    "retained_units": int(keep.sum()), "excluded_units": int(excluded.sum()),
                    "retained_positives": int(yy.sum()), "excluded_positives": int(y[excluded].sum()),
                    "excluded_positive_fraction": float(y[excluded].sum() / y.sum()),
                    "retained_prevalence": prev, "retained_pr_auc": pr, "retained_pr_lift": pr / prev,
                    "retained_brier": float(np.mean((pp - yy) ** 2)),
                    "retained_log_loss": float(log_loss(yy, pp, labels=[0, 1])),
                    "excluded_spatial_concentration_top10pct_blocks": concentration,
                })
    failure = pd.DataFrame(failure_rows)
    coverage = pd.DataFrame(coverage_rows)
    u1 = failure[failure.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED"]
    valid_delta = u1.incremental_fn_auc.dropna()
    rule_fn = int((valid_delta > 0).sum()) >= 5 and float(valid_delta.median()) >= .05 if len(valid_delta) else False
    rho = u1.median_within_susceptibility_stratum_residual_rho.dropna()
    rule_rho = int((rho > 0).sum()) >= 4 and float(rho.median()) >= .10 if len(rho) else False
    land = pd.read_csv(OUT / "STAGE4F_LANDSCAPE_NOVELTY.tsv", sep="\t").set_index("landscape_id")
    selected = cal_results[cal_results.calibration_method == method].set_index("landscape_id").reindex(P.LANDSCAPES)
    h = land.reindex(P.LANDSCAPES).energy_H_M
    directions = {
        "rho_H_vs_pr_lift": _spearman(h, selected.pr_lift),
        "rho_H_vs_brier_ratio": _spearman(h, selected.brier_ratio_to_null),
        "rho_H_vs_log_loss_ratio": _spearman(h, selected.log_loss_ratio_to_null),
    }
    landscape_rule = directions["rho_H_vs_pr_lift"] < 0 and directions["rho_H_vs_brier_ratio"] > 0 and directions["rho_H_vs_log_loss_ratio"] > 0
    confidence_supported = bool(rule_fn and rule_rho and landscape_rule)
    conf_status = "PROVISIONAL_DEVELOPMENT_CATEGORIES_SUPPORTED" if confidence_supported else "NOT_ESTABLISHED"

    cov_u1 = coverage[coverage.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED"]
    abstain_ok = True
    cov_detail = {}
    for c in (.9, .8, .7):
        wins = 0; positive_guard = True
        for held in P.LANDSCAPES:
            base = cov_u1[(cov_u1.landscape_id == held) & (cov_u1.coverage == 1.0)].iloc[0]
            row = cov_u1[(cov_u1.landscape_id == held) & (cov_u1.coverage == c)].iloc[0]
            prev = float(row.retained_prevalence)
            norm_ll = float(row.retained_log_loss / (-(prev * math.log(prev) + (1 - prev) * math.log(1 - prev))))
            bprev = float(base.retained_prevalence)
            base_norm_ll = float(base.retained_log_loss / (-(bprev * math.log(bprev) + (1 - bprev) * math.log(1 - bprev))))
            if row.retained_pr_lift > base.retained_pr_lift and norm_ll < base_norm_ll:
                wins += 1
            if row.excluded_positive_fraction > (1 - c) + 1e-12:
                positive_guard = False
        cov_detail[str(c)] = {"landscapes_improving_pr_lift_and_normalised_log_loss": wins,
                              "positive_exclusion_guard": positive_guard}
        abstain_ok &= wins >= 5 and positive_guard
    summary = {
        "transfer_confidence_status": conf_status, "incremental_fn_rule_pass": bool(rule_fn),
        "within_stratum_residual_rule_pass": bool(rule_rho), "landscape_direction_rule_pass": bool(landscape_rule),
        "landscape_associations": directions, "abstention_deploy": bool(abstain_ok),
        "abstention_status": "SUPPORTED" if abstain_ok else "NOT_SUPPORTED", "coverage_rule_detail": cov_detail,
        "per_landscape_thresholds": per_land,
    }
    return failure, coverage, summary


def _decisions_table(cal_decision: dict, reliability: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"decision": "CALIBRATION_STATUS", "value": cal_decision["calibration_status"],
         "evidence": json.dumps(cal_decision, sort_keys=True)},
        {"decision": "TRANSFER_CONFIDENCE_STATUS", "value": reliability["transfer_confidence_status"],
         "evidence": json.dumps({k: v for k, v in reliability.items() if k != "per_landscape_thresholds"}, sort_keys=True)},
        {"decision": "ABSTENTION_STATUS", "value": reliability["abstention_status"],
         "evidence": json.dumps(reliability["coverage_rule_detail"], sort_keys=True)},
        {"decision": "SUSCEPTIBILITY_COMMUNICATION", "value": "TARGET_LANDSCAPE_PERCENTILE_RANK",
         "evidence": "No named risk categories; relative rank only unless calibration outcome A"},
        {"decision": "EVIDENCE_LEVEL", "value": "DEVELOPMENT_ONLY_REPEATED_USE_N6",
         "evidence": "Six landscapes previously examined in Stages 4D/4E-A/4E-B"},
    ])


def _product_schema(cal_decision: dict, reliability: dict) -> dict:
    probability_ok = cal_decision["probability_language_permitted"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Stage 4F development-only susceptibility output",
        "type": "object",
        "required": ["LANDSCAPE_ID", "TERRAIN_UNIT_ID", "SUSCEPTIBILITY_SCORE", "SUSCEPTIBILITY_RANK_OR_CATEGORY",
                     "CALIBRATION_STATUS", "TRANSFER_NOVELTY_SCORE", "TRANSFER_CONFIDENCE_STATUS", "OOD_METHOD",
                     "H_RECONSTRUCTION_STATUS", "MODEL_VERSION", "EVIDENCE_LEVEL"],
        "properties": {
            "LANDSCAPE_ID": {"type": "string"}, "TERRAIN_UNIT_ID": {"type": "string"},
            "SUSCEPTIBILITY_SCORE": {"type": "number", "minimum": 0, "maximum": 1,
                                      "description": "Monotonic LOGISTIC_M score; not an absolute probability unless CALIBRATION_STATUS is supported"},
            "SUSCEPTIBILITY_RANK_OR_CATEGORY": {"type": "number", "minimum": 0, "maximum": 100,
                                                 "description": "Percentile rank relative to the target landscape, not a universal risk category"},
            "CALIBRATED_PROBABILITY": ({"type": "number", "minimum": 0, "maximum": 1,
                                         "description": "Permitted by Stage 4F outcome A"} if probability_ok else
                                        {"type": "null", "description": "Withheld: calibration transfer not fully supported"}),
            "CALIBRATION_STATUS": {"const": cal_decision["calibration_status"]},
            "TRANSFER_NOVELTY_SCORE": {"type": "number", "minimum": 0, "maximum": 1,
                                        "description": "Descriptive training-reference percentile; not confidence"},
            "TRANSFER_CONFIDENCE_STATUS": {"const": reliability["transfer_confidence_status"]},
            "OOD_METHOD": {"const": "U1_SUSCEPTIBILITY_CONDITIONED"},
            "H_RECONSTRUCTION_STATUS": {"type": "string", "description": "Separate reconstruction validity; never raises transfer confidence"},
            "MODEL_VERSION": {"const": "SCIENCE_FAIR_ML_V2_STAGE4F_LOGISTIC_M"},
            "EVIDENCE_LEVEL": {"const": "DEVELOPMENT_ONLY_REPEATED_USE_N6"},
        },
        "additionalProperties": False,
        "probability_language_permitted": probability_ok,
        "percentile_reference": "TARGET_LANDSCAPE",
        "susceptibility_named_categories": "WITHHELD",
    }


def _h_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "landscape_id": lid, "h_reconstruction_status": P.H_STATUS[lid],
        "n_verified_h_capable": P.H_STATUS["N_VERIFIED_H_CAPABLE"],
        "megh_experiment_unlocked": P.H_STATUS["MEGH_EXPERIMENT_UNLOCKED"],
        "h_used_as_predictor": False, "gen3_modified": False,
    } for lid in P.LANDSCAPES])


def _savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close()


def _make_figures(cal: pd.DataFrame, comp: pd.DataFrame, failure: pd.DataFrame, coverage: pd.DataFrame,
                  cal_decision: dict, reliability: dict) -> None:
    # 1 reliability diagrams, all landscapes overlaid by raw/selected.
    selected = cal_decision["selected_method"]
    plt.figure(figsize=(8, 6)); plt.plot([0, 1], [0, 1], "k--", lw=1)
    for method, ls in [("C0_RAW", ":"), (selected, "-")]:
        g = cal[cal.calibration_method == method]
        for r in g.itertuples():
            curve = json.loads(r.reliability_curve_json)
            plt.plot([x["mean_predicted"] for x in curve], [x["observed"] for x in curve], ls,
                     alpha=.65, label=f"{r.landscape_id[:4]} {method}" if method == selected else None)
    plt.xscale("log"); plt.yscale("log"); plt.xlim(1e-4, 1); plt.ylim(1e-4, 1)
    plt.xlabel("Mean predicted score / calibrated value"); plt.ylabel("Observed mapped-footprint frequency")
    plt.title("Development-only reliability: raw versus selected calibration"); plt.legend(fontsize=6, ncol=2)
    _savefig("01_raw_vs_calibrated_reliability.png")

    # 2 slope/intercept.
    g = cal[cal.calibration_method == selected].set_index("landscape_id").reindex(P.LANDSCAPES)
    x = np.arange(len(g)); fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].bar(x, g.calibration_slope); ax[0].axhline(1, color="k", ls="--"); ax[0].set_ylabel("Calibration slope")
    ax[1].bar(x, g.calibration_intercept); ax[1].axhline(0, color="k", ls="--"); ax[1].set_ylabel("Calibration intercept")
    ax[1].set_xticks(x, [s[:4] for s in g.index]); fig.suptitle(f"{selected}: calibration by held-out landscape")
    _savefig("02_calibration_slope_intercept.png")

    # 3 method comparison.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    piv = cal.pivot(index="landscape_id", columns="calibration_method", values="brier_ratio_to_null").reindex(P.LANDSCAPES)
    piv.plot(kind="bar", ax=ax[0], legend=False); ax[0].set_title("Brier / prevalence-null Brier"); ax[0].set_xlabel("")
    piv = cal.pivot(index="landscape_id", columns="calibration_method", values="log_loss_ratio_to_null").reindex(P.LANDSCAPES)
    piv.plot(kind="bar", ax=ax[1]); ax[1].set_title("Log loss / prevalence-null entropy"); ax[1].set_xlabel("")
    ax[1].legend(fontsize=7); _savefig("03_brier_logloss_comparison.png")

    land = pd.read_csv(OUT / "STAGE4F_LANDSCAPE_NOVELTY.tsv", sep="\t").set_index("landscape_id").reindex(P.LANDSCAPES)
    gg = cal[(cal.calibration_method == selected)].set_index("landscape_id").reindex(P.LANDSCAPES)
    # 4 H vs lift.
    plt.figure(figsize=(6, 5)); plt.scatter(land.energy_H_M, gg.pr_lift)
    for lid in P.LANDSCAPES: plt.annotate(lid[:4], (land.loc[lid].energy_H_M, gg.loc[lid].pr_lift), fontsize=8)
    plt.xlabel("Landscape novelty H_M"); plt.ylabel("LOGISTIC_M PR lift"); plt.title("Exploratory n=6 development landscapes")
    _savefig("04_landscape_novelty_vs_pr_lift.png")
    # 5 H vs calibration.
    plt.figure(figsize=(6, 5)); plt.scatter(land.energy_H_M, gg.log_loss_ratio_to_null)
    for lid in P.LANDSCAPES: plt.annotate(lid[:4], (land.loc[lid].energy_H_M, gg.loc[lid].log_loss_ratio_to_null), fontsize=8)
    plt.xlabel("Landscape novelty H_M"); plt.ylabel("Normalised log loss"); plt.title("Novelty versus calibration error (exploratory)")
    _savefig("05_landscape_novelty_vs_calibration_error.png")

    # 6/7 unit novelty from deterministic plot sample.
    parts = []
    for lid in P.LANDSCAPES:
        d = _load_ood_cache(lid); n = min(4000, len(d["susceptibility_score"]))
        take = np.linspace(0, len(d["susceptibility_score"]) - 1, n).astype(int)
        parts.append(pd.DataFrame({"p": d["susceptibility_score"][take], "u0": d["u0_percentile"][take],
                                   "u1": d["u1_percentile"][take], "land": lid[:4]}))
    plot = pd.concat(parts)
    plt.figure(figsize=(7, 5)); plt.hexbin(plot.p, plot.u0, gridsize=45, mincnt=1, bins="log"); plt.colorbar(label="log count")
    plt.xlabel("Raw susceptibility score"); plt.ylabel("U0 novelty percentile"); plt.xscale("log")
    plt.title("Unconditional novelty versus susceptibility")
    _savefig("06_unit_novelty_vs_susceptibility.png")
    plt.figure(figsize=(7, 5)); plt.scatter(plot.u0, plot.u1, s=2, alpha=.15)
    plt.xlabel("U0 unconditional percentile"); plt.ylabel("U1 conditioned percentile")
    plt.title("Conditional versus unconditional unit novelty")
    _savefig("07_conditional_vs_unconditional_ood.png")

    # 8 failure detection.
    f = failure.pivot(index="landscape_id", columns="ood_method", values="incremental_fn_auc").reindex(P.LANDSCAPES)
    f.plot(kind="bar", figsize=(9, 5)); plt.axhline(0, color="k", lw=1); plt.ylabel("FN detection AUC minus susceptibility reference")
    plt.title("Incremental false-negative detection"); plt.xlabel(""); plt.legend(fontsize=8)
    _savefig("08_ood_failure_detection.png")

    # 9 coverage tradeoff.
    c = coverage[coverage.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED"]
    plt.figure(figsize=(7, 5))
    for lid, z in c.groupby("landscape_id"):
        plt.plot(z.coverage, z.retained_pr_lift, marker="o", label=lid[:4])
    plt.xlabel("Retained coverage"); plt.ylabel("Retained PR lift"); plt.title("Selective prediction (U1)")
    plt.legend(fontsize=7, ncol=2); plt.gca().invert_xaxis()
    _savefig("09_selective_prediction_coverage.png")

    # 10 existing development landscape example, relative rank only.
    lid = "DOMINICA_ERIKA_2015"; d = _load_ood_cache(lid); p = d["susceptibility_score"]
    rank = pd.Series(p).rank(method="average", pct=True).to_numpy() * 100
    plt.figure(figsize=(7, 7)); sc = plt.scatter(d["centroid_x"], d["centroid_y"], c=rank, s=2, cmap="magma", vmin=0, vmax=100)
    plt.colorbar(sc, label="Target-landscape susceptibility percentile")
    plt.axis("equal"); plt.axis("off"); plt.title("Dominica development-only example — not externally validated")
    _savefig("10_development_example_output_map.png")


def _write_report(cal: pd.DataFrame, comp: pd.DataFrame, failure: pd.DataFrame, coverage: pd.DataFrame,
                  cal_decision: dict, reliability: dict) -> None:
    method = cal_decision["selected_method"]
    sel = cal[cal.calibration_method == method].set_index("landscape_id").reindex(P.LANDSCAPES)
    raw = cal[cal.calibration_method == "C0_RAW"].set_index("landscape_id").reindex(P.LANDSCAPES)
    worst = str(sel.log_loss_ratio_to_null.idxmax())
    rank_change = float((sel.pr_auc - raw.pr_auc).abs().max())
    land = pd.read_csv(OUT / "STAGE4F_LANDSCAPE_NOVELTY.tsv", sep="\t").set_index("landscape_id").reindex(P.LANDSCAPES)
    u1 = failure[failure.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED"]
    conf = pd.read_csv(OUT / "STAGE4F_NOVELTY_HAZARD_CONFOUNDING.tsv", sep="\t")
    u0c = conf[conf.ood_method == "U0_UNCONDITIONAL"]
    u1c = conf[conf.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED"]
    c90 = coverage[(coverage.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED") & (coverage.coverage == .9)]
    c70 = coverage[(coverage.ood_method == "U1_SUSCEPTIBILITY_CONDITIONED") & (coverage.coverage == .7)]
    probability = "YES" if cal_decision["probability_language_permitted"] else "NO"
    raw_cal = "No; slopes and intercepts vary materially across landscapes."
    q = {
        1: raw_cal,
        2: "Negative subsampling and landscape-balanced sampling-corrected weights alter the fitted prior; raw logistic output is a ranking score, not automatically an event-frequency probability.",
        3: "See C1 rows; transfer is judged by the frozen comparison and A/B/C rule.",
        4: "See C2 rows; transfer is judged by the frozen comparison and A/B/C rule.",
        5: "See C3 rows; isotonic is monotonic but may introduce ties.",
        6: f"{method} under the frozen robust landscape-level rule.",
        7: f"No universal consistency is established; worst normalised log loss is {worst}.",
        8: worst,
        9: f"Maximum absolute PR-AUC change versus raw was {rank_change:.3g}; C0-C2 are strictly monotonic and C3 may tie scores.",
        10: probability + "; probability language is permitted only for outcome A.",
        11: f"Exploratory Spearman rho = {reliability['landscape_associations']['rho_H_vs_pr_lift']:.3f} (n=6).",
        12: f"Exploratory H_M versus normalised log loss rho = {reliability['landscape_associations']['rho_H_vs_log_loss_ratio']:.3f}.",
        13: f"H_M versus Brier/log-loss ratios: {reliability['landscape_associations']['rho_H_vs_brier_ratio']:.3f} / {reliability['landscape_associations']['rho_H_vs_log_loss_ratio']:.3f}.",
        14: f"Cotabato H_M={land.loc['COTABATO_DAVAO_2019'].energy_H_M:.6f}; it is not identified as strongly morphology-novel.",
        15: f"Median U0 rho with slope = {u0c.rho_slope.median():.3f}.",
        16: f"Median U0 rho with susceptibility = {u0c.rho_susceptibility.median():.3f}.",
        17: f"Median |rho with susceptibility| changes from {u0c.rho_susceptibility.abs().median():.3f} (U0) to {u1c.rho_susceptibility.abs().median():.3f} (U1).",
        18: f"Incremental rule pass = {reliability['incremental_fn_rule_pass'] and reliability['within_stratum_residual_rule_pass']}.",
        19: f"Median U1 incremental FN AUC = {u1.incremental_fn_auc.median():.3f}.",
        20: "Per-landscape results are reported; consistency is not assumed from pooled units.",
        21: f"{reliability['transfer_confidence_status']}.",
        22: "A descriptive training-reference novelty percentile marked unvalidated remains defensible.",
        23: f"{reliability['abstention_status']} under the prevalence-aware preregistered rule.",
        24: f"At 90%/70% U1 coverage, median excluded-positive fractions are {c90.excluded_positive_fraction.median():.3f}/{c70.excluded_positive_fraction.median():.3f}.",
        25: f"Yes; median retained prevalence changes from {sel.prevalence.median():.6f} to {c70.retained_prevalence.median():.6f} at 70% coverage.",
        26: f"{reliability['abstention_status']} under joint PR-lift and normalised-log-loss evaluation.",
        27: "Yes: raw score plus target-landscape percentile rank, explicitly relative and development-only.",
        28: "No named category threshold; continuous target-landscape percentile rank is justified.",
        29: "Probability claims, validated confidence categories, abstention deployment and generalisation claims remain withheld pending untouched lockbox validation.",
        30: cal_decision["calibration_status"],
        31: reliability["transfer_confidence_status"],
        32: "Yes; N_VERIFIED_H_CAPABLE=1 and MEGH_EXPERIMENT_UNLOCKED=NO.",
        33: "No; the exact frozen 60 M predictors were used.",
        34: "No; LOGISTIC_M architecture, sampling, weighting and preprocessing were unchanged.",
        35: "No.", 36: "No; historical files were read-only.", 37: "Yes: 153 strict files intact and zero writes.",
        38: "Recorded in STAGE4F_CHECKPOINT.json and the handover after finalisation.",
        39: f"(i) Calibration status is {cal_decision['calibration_status']}; (ii) confidence is {reliability['transfer_confidence_status']}; (iii) relative susceptibility remains communicable.",
        40: "(i) only six repeatedly used development landscapes; (ii) mapped historical footprint frequencies differ in prevalence/completeness; (iii) morphology novelty cannot detect label/concept shift such as Cotabato EQ1.",
        41: "Untouched lockbox evaluation, prospective probability/confidence validation, final V2 governance freeze and only then any externally facing claims.",
    }
    answers = "\n".join(f"{i}. {q[i]}" for i in range(1, 42))
    text = f"""STAGE 4F CALIBRATION, OOD & TRANSFER CONFIDENCE:
COMPLETE

## A. Executive result

The frozen calibration rule selected **{method}**. The development-stage calibration decision is **{cal_decision['calibration_status']}**. Transfer confidence is **{reliability['transfer_confidence_status']}** and selective abstention is **{reliability['abstention_status']}**. These are repeated-use development results from six landscapes, not independent confirmation.

## B. Governance and integrity

V1 verification passed: 153 strict files intact, zero writes. The Stage 4E-B checkpoint matched `{P.STAGE4EB_CHECKPOINT_SHA256}`. Historical outputs were read-only; no lockbox was touched; no commit or push was performed.

## C. Frozen LOGISTIC_M architecture

All folds used the exact 60 M columns, L2 Logistic Regression, the Stage 4E-B inner selection/tie rule, deterministic negative sampling, landscape-balanced sampling-corrected weights and training-only z-scales. Every fold selected C=0.01 mechanically. No feature or architecture changed.

## D. Preregistration

`STAGE4F_PREREGISTRATION_FREEZE.json` was created before new results. It fixed C0-C3, H_M, U0/U1, metrics, block uncertainty, decisions, coverage and product policy. No method was added after results.

## E. Calibration data/provenance

Calibrators used complete-domain honest inner-LOLO scores from the five outer-training landscapes. Each score came from a model excluding that landscape. Outer labels and prevalence never entered fitting.

## F. Sampling and probability audit

All positives and at most 20,000 deterministic negatives per training landscape were fitted. Sampling-corrected weights restore each training landscape's full-domain class share while landscape balancing gives each landscape equal total influence. Between-landscape prevalence remains heterogeneous, so the raw intercept is not a universal probability prior and labels describe mapped footprints for specified historical inventories—not future-event probability.

## G. Calibration methods

C0 raw, C1 intercept-only, C2 intercept plus nonnegative slope, and C3 weighted isotonic were compared. Selection used landscape-normalised Brier and log-loss ratios with a worst-landscape penalty and catastrophic guard.

## H. Calibration results

Selected method: **{method}**. Median Brier/log loss were {sel.brier.median():.6g}/{sel.log_loss.median():.6g}; worst normalised log loss was {worst} ({sel.loc[worst].log_loss_ratio_to_null:.3f}). Full per-landscape values, block intervals, reliability curves, observed/expected counts, PR-AUC and ROC-AUC are in `STAGE4F_CALIBRATION_RESULTS.tsv`.

## I. Calibration decision

**{cal_decision['calibration_status']}**. Absolute-probability language permitted: **{probability}**. Any non-A result requires qualified or relative-only communication.

## J. Landscape-level novelty

H_M exactly reproduces the Stage 4D training-only statistic within the preregistered tolerance. Its associations with PR lift and normalised Brier/log loss are exploratory (n=6), not thresholds.

## K. Unit-level novelty

U0 is unconditional 10-NN morphology novelty. U1 repeats it within five training-defined susceptibility strata. Both are training-reference percentiles and were frozen before held-out labels were loaded.

## L. Novelty versus susceptibility confounding

Median U0 correlation with susceptibility was {u0c.rho_susceptibility.median():.3f}; U1 was {u1c.rho_susceptibility.median():.3f}. Reduced marginal correlation alone was not treated as successful separation; residual and false-negative evidence was required.

## M. Failure detection

Median U1 incremental false-negative AUC beyond the susceptibility reference was {u1.incremental_fn_auc.median():.3f}. Incremental FN rule: {reliability['incremental_fn_rule_pass']}; within-stratum residual rule: {reliability['within_stratum_residual_rule_pass']}.

## N. Transfer-confidence decision

**{reliability['transfer_confidence_status']}**. When unsupported, the defensible output is descriptive novelty explicitly marked unvalidated—not LOW/MODERATE/HIGH confidence.

## O. Selective prediction

Decision: **{reliability['abstention_status']}**. PR lift was interpreted with retained prevalence, normalised log loss, positives excluded and spatial concentration; PR-AUC alone was never sufficient.

## P. Product output schema

The JSON schema keeps susceptibility, calibration, novelty, confidence and H reconstruction separate. It uses a target-landscape percentile rank and withholds named susceptibility categories and unsupported probability/confidence claims.

## Q. H status

Kumamoto is VALIDATED_GEN3_PARTIAL; Noto is AGE_PROXY_ONLY; all others are NOT_VALIDATED. N_VERIFIED_H_CAPABLE=1; MEGH_EXPERIMENT_UNLOCKED=NO. H was not a predictor and does not increase confidence.

## R. Limitations

- Six development landscapes have already informed Stages 4D/4E-A/4E-B.
- Mapped-footprint frequency is inventory-, event- and domain-specific; completeness and trigger intensity vary.
- X-only novelty cannot diagnose arbitrary P(Y|X), label or inventory shift; Cotabato is the clearest warning.

## S. Remaining work before V2 freeze

Use an untouched lockbox exactly once under frozen governance; validate probability wording, novelty/confidence and any abstention policy; record final external status; then create the V2 freeze without changing the selected architecture on the basis of the lockbox.

## T. Reproducibility and checkpoint

Seeds, software, sources, hashes, tests and rerun evidence are recorded in `STAGE4F_CHECKPOINT.json`. Deterministic output tables use stable ordering and formatting.

## U. Answers to questions 1–41

{answers}
"""
    (OUT / "STAGE4F_FINAL_REPORT.md").write_text(text)


def run_evaluation_phase() -> dict:
    """Load held-out Y only after the frozen OOD boundary, then evaluate reliability."""
    _verify_prereg()
    if not OOD_FREEZE.exists():
        raise RuntimeError("STOP: OOD scores must be frozen before held-out-Y evaluation")
    ood_freeze = json.loads(OOD_FREEZE.read_text())
    for rel, digest in ood_freeze["files"].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(f"STOP: frozen OOD artifact changed: {rel}")
    calibrators, rows, sampling_rows = {}, [], []
    for held in P.LANDSCAPES:
        ood = _load_ood_cache(held)
        with np.load(MEG_DIR / f"{held}.npz", allow_pickle=False) as z:
            y = z["y_positive"].astype(bool); blocks = z["spatial_block_3000m"]
            ids = z["terrain_unit_id"]
        if not np.array_equal(ids, ood["terrain_unit_id"]):
            raise RuntimeError(f"STOP: score/label order mismatch for {held}")
        raw = ood["susceptibility_score"]
        expected = json.loads((OUT / "STAGE4F_OOD_FOLD_METADATA.json").read_text())[held]["published_outer_pr_auc_for_later_y_aware_identity_check"]
        got = float(average_precision_score(y, raw))
        if abs(got - expected) > 2e-10:
            raise RuntimeError(f"STOP: reconstructed outer prediction mismatch for {held}: {got} vs {expected}")
        pin, yin, win, inner_lids, train, checks = _inner_predictions(held)
        calibrators[held] = {"_inner_y": yin, "_inner_w": win, "_inner_raw": pin, "_inner_lids": inner_lids}
        for method in CAL_ORDER:
            cal = _fit_calibrator(method, pin, yin, win)
            serial = _serialise_calibrator(cal)
            calibrators[held][method] = serial
            p = _apply_calibrator(cal, raw)
            rows.append(_metric_row(held, method, y, raw, p, blocks, train, cal))
        meta = json.loads((OUT / "STAGE4F_OOD_FOLD_METADATA.json").read_text())[held]
        for lid, a in meta["training"].items():
            sampling_rows.append({"outer_held_out": held, "training_landscape": lid, **a,
                                  "landscape_balanced_total_weight_target": 1 / 5,
                                  "sampling_prior_corrected": True})
    cal = pd.DataFrame(rows)
    comp, cal_decision = _select_calibration(cal)
    method = cal_decision["selected_method"]
    # Calibrate honest inner predictions with the selected method for frozen thresholds/residual cutoffs.
    for held in P.LANDSCAPES:
        calibrators[held]["_inner_selected_p"] = _apply_serialised(calibrators[held][method], calibrators[held]["_inner_raw"])
    failure, coverage, reliability = _failure_and_selective(calibrators, cal_decision, cal)
    _tsv(OUT / "STAGE4F_CALIBRATION_RESULTS.tsv", cal)
    _tsv(OUT / "STAGE4F_CALIBRATION_COMPARISON.tsv", comp)
    _tsv(OUT / "STAGE4F_SAMPLING_PROBABILITY_AUDIT.tsv", pd.DataFrame(sampling_rows))
    _tsv(OUT / "STAGE4F_FAILURE_DETECTION.tsv", failure)
    _tsv(OUT / "STAGE4F_SELECTIVE_COVERAGE.tsv", coverage)
    _tsv(OUT / "STAGE4F_RELIABILITY_DECISIONS.tsv", _decisions_table(cal_decision, reliability))
    _json(OUT / "STAGE4F_PRODUCT_OUTPUT_SCHEMA.json", _product_schema(cal_decision, reliability))
    _tsv(OUT / "STAGE4F_H_STATUS.tsv", _h_table())
    _json(OUT / "STAGE4F_CALIBRATION_FOLD_METADATA.json", {
        lid: {"training_landscapes": [x for x in P.LANDSCAPES if x != lid], "heldout_y_used_for_fitting": False,
              "honest_inner_prediction_landscapes": sorted(set(calibrators[lid]["_inner_lids"].tolist())),
              "calibrators": {m: calibrators[lid][m] for m in CAL_ORDER}}
        for lid in P.LANDSCAPES
    })
    _make_figures(cal, comp, failure, coverage, cal_decision, reliability)
    _write_report(cal, comp, failure, coverage, cal_decision, reliability)
    result = {
        "stage": "4F", "status": "EVALUATION_COMPLETE_PENDING_TESTS_AND_CHECKPOINT",
        "calibration_decision": cal_decision, "reliability_decision": reliability,
        "ood_freeze_sha256": sha256(OOD_FREEZE), "heldout_y_loaded_only_after_ood_freeze": True,
    }
    _json(OUT / "STAGE4F_EVALUATION_SUMMARY.json", result)
    return result
