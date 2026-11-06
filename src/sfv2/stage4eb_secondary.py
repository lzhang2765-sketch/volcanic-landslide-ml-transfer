"""Post-selection Stage 4E-B controls, diagnostics and presentation outputs.

This module refuses to run until the M-only primary decision is frozen.  Nothing
computed here can change the development leader.
"""
from __future__ import annotations

import json
import pickle
import resource
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from . import stage4b as S
from . import stage4d as D
from . import stage4ea_audit as A
from . import stage4eb as E
from . import stage4eb_prereg as P
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_b"
FIG = OUT / "figures"
PRIMARY_FREEZE = OUT / "STAGE4EB_PRIMARY_SELECTION_FREEZE.json"


def require_primary_freeze() -> dict:
    E.verify_prereg()
    if not PRIMARY_FREEZE.exists():
        raise RuntimeError("STOP: primary M selection must be frozen before secondary analysis")
    fr = json.loads(PRIMARY_FREEZE.read_text())
    if fr["status"] != "PRIMARY_M_SELECTION_FROZEN_BEFORE_CONTROLS_OR_CUMULATIVE_SENSITIVITY":
        raise RuntimeError("STOP: invalid primary-selection freeze")
    for rel, digest in fr["files"].items():
        if sha256(ROOT / rel) != digest:
            raise RuntimeError(f"STOP: frozen primary result changed: {rel}")
    return fr


def stable_tsv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.12g")


def primary_state(d: S.Data):
    x = json.loads((E.RESULTS / "XGBOOST_M.json").read_text())
    return E.primary_state(d, x)


def historical_controls(d: S.Data) -> pd.DataFrame:
    """Reuse old controls exactly; add preregistered XGB ME/MEG after the freeze."""
    require_primary_freeze()
    full = json.loads(E.STAGE4B_RESULTS.read_text())["results"]
    old = pd.DataFrame([r for r in full if r["model"] in P.MODELS[:3] and r["feature_set"] in ("M", "ME", "MEG")])
    old["source"] = "STAGE4B_VERIFIED_REUSE"
    new = []
    mobj = json.loads((E.RESULTS / "XGBOOST_M.json").read_text())
    for r in mobj["results"]:
        q = dict(r); q["source"] = "STAGE4EB_FROZEN_PRIMARY"; new.append(q)
    for fs in P.HISTORICAL_CONTROLS:
        obj = E.run_xgb_feature_set(d, fs)
        for r in obj["results"]:
            q = dict(r); q["source"] = "STAGE4EB_POST_SELECTION_CONTROL"; new.append(q)
    out = pd.concat([old, pd.DataFrame(new)], ignore_index=True, sort=False)
    out["pr_lift"] = out.pr_auc / out.prevalence
    out["eligible_for_primary_selection"] = False
    out.loc[out.feature_set.eq("M"), "eligible_for_primary_selection"] = True
    return out.sort_values(["feature_set", "held_out", "model"]).reset_index(drop=True)


def cotabato_labels():
    lid = "COTABATO_DAVAO_2019"
    transform, shape, crs = A.dem_grid(lid)
    u = A.unit_table(lid)
    rr, cc = u.source_grid_row.to_numpy(), u.source_grid_col.to_numpy()

    def hits(layer):
        g = gpd.read_file(f"zip://{A.ZIP}!{layer}.shp").to_crs(crs)
        mask = rasterize([(z.buffer(0), 1) for z in g.geometry if z is not None and not z.is_empty],
                         out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        y = np.zeros(len(u), bool)
        for dr in range(3):
            for dc in range(3):
                y |= mask[np.clip(rr + dr, 0, shape[0] - 1), np.clip(cc + dc, 0, shape[1] - 1)]
        return y, len(g)

    area = gpd.read_file(f"zip://{A.ZIP}!area_study.shp").to_crs(crs)
    am = rasterize([(z, 1) for z in area.geometry], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    inside = am[np.clip(rr + 1, 0, shape[0] - 1), np.clip(cc + 1, 0, shape[1] - 1)]
    labels = {"EQ1_primary": hits("landslides_ott2019"),
              "cumulative_post_EQ3_nov2019": hits("landslides_nov2019"),
              "cumulative_post_EQ4_dic2019": hits("landslides_dic2019")}
    if not np.array_equal(labels["EQ1_primary"][0], u.y_positive.to_numpy(bool)):
        raise RuntimeError("STOP: reconstructed EQ1 target does not reproduce frozen y_positive")
    return u, inside, labels


def cotabato_sensitivity(targets, pred) -> pd.DataFrame:
    """Score frozen M predictions on auxiliary labels; never return a selection value."""
    require_primary_freeze()
    u, inside, labels = cotabato_labels()
    code, nb = D.block_codes(u.spatial_block_3000m.to_numpy())
    rows = []
    lid = "COTABATO_DAVAO_2019"
    for mi, model in enumerate(P.MODELS):
        p = pred[(model, lid)]["raw"]
        sorted_ = {name: D.Sorted(p[inside], y[inside]) for name, (y, _) in labels.items()}
        rng = np.random.default_rng(P.SEED + 500 + mi)
        reps = {name: np.full((P.BOOTSTRAP["replicates"], 2), np.nan) for name in labels}
        for b in range(P.BOOTSTRAP["replicates"]):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code].astype(float)[inside]
            for name in labels:
                reps[name][b] = sorted_[name].ap_auc(w)
        eq1 = reps["EQ1_primary"]
        for name, (yall, polygons) in labels.items():
            y = yall[inside]
            ap = average_precision_score(y, p[inside]); auc = roc_auc_score(y, p[inside])
            apl, aph = E.pct(reps[name][:, 0]); al, ah = E.pct(reps[name][:, 1])
            dl, dh = E.pct(reps[name][:, 0] - eq1[:, 0])
            rows.append(dict(model=model, feature_set="M", label_set=name,
                             role="PRIMARY_OUTCOME" if name == "EQ1_primary" else "POST_SELECTION_SENSITIVITY_ONLY",
                             source_polygons=polygons, units_evaluated=int(inside.sum()), positive_units=int(y.sum()),
                             prevalence=float(y.mean()), pr_auc=float(ap), pr_lift=float(ap / y.mean()),
                             pr_ci_low=apl, pr_ci_high=aph, roc_auc=float(auc), roc_ci_low=al, roc_ci_high=ah,
                             delta_pr_vs_eq1=float(ap - average_precision_score(labels["EQ1_primary"][0][inside], p[inside])),
                             delta_pr_ci_low=dl, delta_pr_ci_high=dh, affects_selection=False,
                             eq1_rule_reproduces_frozen_target=True))
    return pd.DataFrame(rows)


def fit_outer(d: S.Data, model: str, held: str, params: dict):
    train = [x for x in P.LANDSCAPES if x != held]
    t0 = time.perf_counter()
    if model == "XGBOOST":
        est, cols, _, meta = E.fit_xgb(d, train, "M", params)
        prep = None
    else:
        old_jobs = S.FIXED.get(model, {}).get("n_jobs")
        if model == "RANDOM_FOREST":
            S.FIXED[model]["n_jobs"] = 1  # diagnostic-only execution setting; fitted forest is identical
        try:
            est, cols, prep, _, meta = S.fit(d, train, "M", model, params, "ALL_VALID", "LANDSCAPE_BALANCED")
        finally:
            if model == "RANDOM_FOREST":
                S.FIXED[model]["n_jobs"] = old_jobs
    fit_seconds = time.perf_counter() - t0
    # Repeated permutation predictions are deliberately single-threaded.  This
    # changes no fitted tree or score and avoids spawning thousands of short
    # joblib tasks during the interpretation-only audit.
    if model == "RANDOM_FOREST":
        est.n_jobs = 1
    t1 = time.perf_counter()
    p = E.predict_xgb(d, est, cols, held)[0] if model == "XGBOOST" else S.predict(d, est, cols, prep, held, "ALL_VALID")
    inference_seconds = time.perf_counter() - t1
    size = len(pickle.dumps(est, protocol=pickle.HIGHEST_PROTOCOL))
    if model == "LOGISTIC_L2": proxy = int(est.coef_.size + est.intercept_.size)
    elif model == "RANDOM_FOREST": proxy = int(sum(t.tree_.node_count for t in est.estimators_))
    elif model == "HIST_GRADIENT_BOOSTING": proxy = int(sum(len(x) for x in est._predictors))
    else: proxy = int(len(est.get_booster().get_dump()))
    return est, cols, prep, p, dict(train_rows=meta["train_rows"], train_pos=meta["train_pos"],
                                    fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                                    serialized_bytes=size, complexity_proxy=proxy,
                                    peak_process_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)))


def importance_complexity(d: S.Data, lolo: pd.DataFrame, pred: dict, models=None):
    """Post-selection refits used only for computational and interpretation diagnostics."""
    imp, comp, repro = [], [], []
    models = list(models or P.MODELS)
    for mi, model in enumerate(models):
        for li, held in enumerate(P.LANDSCAPES):
            row = lolo[(lolo.model == model) & (lolo.held_out == held)].iloc[0]
            params = json.loads(row.params)
            est, cols, prep, p, cm = fit_outer(d, model, held, params)
            old = pred[(model, held)]["raw"]
            maxdiff = float(np.max(np.abs(p - old)))
            repro.append(dict(model=model, held_out=held, max_abs_prediction_difference=maxdiff,
                              tolerance=1e-12, pass_within_tolerance=maxdiff <= 1e-12))
            comp.append(dict(model=model, held_out=held, feature_set="M", params=row.params,
                             complexity_definition=("standardised coefficients+intercept" if model == "LOGISTIC_L2" else
                                                    "total tree nodes" if model == "RANDOM_FOREST" else
                                                    "boosting predictors" if model == "HIST_GRADIENT_BOOSTING" else "boosted trees"), **cm))
            take = d.sample(held, "ALL_VALID")
            y = d.L[held]["y"][take]
            base = average_precision_score(y, p[take])
            X0 = d.L[held]["X"][take][:, cols].copy()
            if prep is not None:
                X0 = (X0 - np.asarray(prep["mean"])) / np.asarray(prep["sd"])
            native = np.abs(est.coef_[0]) if model == "LOGISTIC_L2" else np.full(len(cols), np.nan)
            for j, name in enumerate(d.fs["M"]):
                vals = []
                for repeat in range(3):
                    seed = P.SEED + mi * 100000 + li * 10000 + j * 10 + repeat
                    rng = np.random.default_rng(seed)
                    X = X0.copy(); X[:, j] = X[rng.permutation(len(X)), j]
                    pp = est.predict_proba(X)[:, 1]
                    vals.append(base - average_precision_score(y, pp))
                imp.append(dict(model=model, held_out=held, feature=name, method="HELD_OUT_PERMUTATION_PR_AUC_DROP",
                                baseline_pr_auc=base, importance_mean=float(np.mean(vals)), importance_sd=float(np.std(vals, ddof=1)),
                                repeats=3, native_abs_standardised_coefficient=float(native[j]) if np.isfinite(native[j]) else np.nan))
    return pd.DataFrame(imp), pd.DataFrame(comp), pd.DataFrame(repro)


def run_diagnostic_models(models: list[str], tag: str) -> dict:
    """Run isolated fold refits; XGBoost and sklearn may use separate runtimes."""
    require_primary_freeze()
    d = S.Data(); lolo, pred, _ = primary_state(d)
    imp, comp, repro = importance_complexity(d, lolo, pred, models=models)
    stable_tsv(OUT/f"_diagnostic_importance_{tag}.tsv", imp)
    stable_tsv(OUT/f"_diagnostic_complexity_{tag}.tsv", comp)
    stable_tsv(OUT/f"_diagnostic_reproduction_{tag}.tsv", repro)
    return dict(tag=tag, models=models, importance_rows=len(imp), complexity_rows=len(comp),
                all_reproduced=bool(repro.pass_within_tolerance.all()))


def ensemble_diagnostic(targets, pred) -> pd.DataFrame:
    rows = []
    for held in P.LANDSCAPES:
        y = targets[held]["y"]
        for i, a in enumerate(P.MODELS):
            for b in P.MODELS[i + 1:]:
                pa, pb = pred[(a, held)]["raw"], pred[(b, held)]["raw"]
                rho = float(spearmanr(pa, pb).statistic)
                ka = max(1, int(.1 * len(pa))); ia = np.argpartition(pa, -ka)[-ka:]; ib = np.argpartition(pb, -ka)[-ka:]
                sa, sb = set(ia.tolist()), set(ib.tolist())
                rows.append(dict(held_out=held, model_a=a, model_b=b, spearman_score_correlation=rho,
                                 top_decile_jaccard=len(sa & sb) / len(sa | sb),
                                 positives_found_only_a=int(y[list(sa - sb)].sum()), positives_found_only_b=int(y[list(sb - sa)].sum()),
                                 role="FEASIBILITY_ONLY_NO_ENSEMBLE_BUILT"))
    return pd.DataFrame(rows)


def h_status() -> pd.DataFrame:
    rows = []
    for lid in P.LANDSCAPES:
        s = P.H_POLICY[lid]
        rows.append(dict(landscape=lid, h_status=s, core_h_available=int("CORE_H_AVAILABLE=1" in s), used_in_stage4eb=False))
    rows.append(dict(landscape="SUMMARY", h_status="N_VERIFIED_H_CAPABLE=1; MEGH_EXPERIMENT_UNLOCKED=NO",
                     core_h_available=1, used_in_stage4eb=False))
    return pd.DataFrame(rows)


def calibration_summary(lolo: pd.DataFrame) -> pd.DataFrame:
    return lolo.groupby("model", as_index=False).agg(median_raw_cal_intercept=("raw_cal_intercept", "median"),
        median_raw_cal_slope=("raw_cal_slope", "median"), min_raw_cal_slope=("raw_cal_slope", "min"),
        max_raw_cal_slope=("raw_cal_slope", "max"), median_raw_brier=("brier_raw", "median"),
        median_raw_log_loss=("log_loss_raw", "median"))


def figures(lolo, comp, robust, stability, importance, cot, cal):
    FIG.mkdir(parents=True, exist_ok=True)
    labels = {"LOGISTIC_L2":"Logistic", "RANDOM_FOREST":"RF", "HIST_GRADIENT_BOOSTING":"HistGB", "XGBOOST":"XGBoost"}
    colors = dict(zip(P.MODELS, ["#255F85", "#2A9D8F", "#E9C46A", "#E76F51"]))
    short = {x:x.split("_")[0].title() for x in P.LANDSCAPES}
    def save(name):
        plt.tight_layout(); plt.savefig(FIG / name, dpi=220, bbox_inches="tight"); plt.close()
    for metric, name, ylabel in [("pr_auc","01_pr_auc_by_landscape.png","PR-AUC"),("pr_lift","02_pr_lift_by_landscape.png","PR lift over prevalence")]:
        fig, ax = plt.subplots(figsize=(10,5.5)); x=np.arange(len(P.LANDSCAPES)); width=.19
        for i,m in enumerate(P.MODELS):
            g=lolo[lolo.model.eq(m)].set_index("held_out").reindex(P.LANDSCAPES)
            ax.bar(x+(i-1.5)*width,g[metric],width,label=labels[m],color=colors[m])
        ax.set_xticks(x, [short[z] for z in P.LANDSCAPES], rotation=20); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0); ax.legend(ncol=4); ax.grid(axis="y",alpha=.25)
        save(name)
    fig,ax=plt.subplots(figsize=(7,5.5))
    for _,r in comp.iterrows(): ax.scatter(r.median_pr_lift,r.worst_pr_lift,s=110,color=colors[r.model],label=labels[r.model])
    ax.set(xlabel="Median PR lift",ylabel="Worst-landscape PR lift"); ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.legend(); ax.grid(alpha=.25); save("03_median_vs_worst_lift.png")
    fig,ax=plt.subplots(figsize=(7,5)); rr=robust.sort_values("robust_score",ascending=False); ax.bar([labels[x] for x in rr.model],rr.robust_score,color=[colors[x] for x in rr.model]); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("Frozen robust-transfer score"); save("04_robust_transfer_score.png")
    leader=json.loads((OUT/"STAGE4EB_DEVELOPMENT_LEADER.json").read_text())["development_leader"]
    boot=pd.read_csv(OUT/"STAGE4EB_SPATIAL_BOOTSTRAP.tsv",sep="\t"); q=boot[(boot.model.eq(leader)) & (boot.comparison.eq("PAIRED_VS_LOGISTIC"))]
    fig,ax=plt.subplots(figsize=(8,5))
    if leader == "LOGISTIC_L2":
        y=np.arange(len(P.LANDSCAPES)); ax.plot(np.zeros(6),y,"o",color=colors[leader]); ax.axvline(0,color="black",lw=1)
        ax.set_yticks(y,[short[x] for x in P.LANDSCAPES]); ax.set_xlim(-.01,.01)
        ax.text(.5,.04,"The selected development leader is Logistic + M;\nits paired difference from itself is exactly zero.",transform=ax.transAxes,ha="center")
    else:
        y=np.arange(len(q)); ax.errorbar(q.point_estimate,y,xerr=[q.point_estimate-q.ci_low,q.ci_high-q.point_estimate],fmt="o",color=colors[leader]); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,[short[x] for x in q.held_out])
    ax.set_xlabel(f"Delta PR-AUC: {labels[leader]} - Logistic"); save("05_leader_vs_logistic_bootstrap.png")
    ranks=lolo.copy(); ranks["rank"]=ranks.groupby("held_out").pr_auc.rank(ascending=False,method="min")
    fig,ax=plt.subplots(figsize=(9,5));
    for m in P.MODELS:
        g=ranks[ranks.model.eq(m)].set_index("held_out").reindex(P.LANDSCAPES); ax.plot(range(6),g["rank"],marker="o",label=labels[m],color=colors[m])
    ax.set_xticks(range(6),[short[x] for x in P.LANDSCAPES],rotation=20); ax.set_yticks([1,2,3,4]); ax.invert_yaxis(); ax.set_ylabel("PR-AUC rank (1 = best)"); ax.legend(ncol=4); save("06_model_rank_by_landscape.png")
    win=stability[stability.is_selected.astype(str).str.lower().eq("true")].groupby("model").size().reindex(P.MODELS,fill_value=0)
    fig,ax=plt.subplots(figsize=(7,5)); ax.bar([labels[x] for x in win.index],win.values,color=[colors[x] for x in win.index]); ax.set_ylim(0,6); ax.set_ylabel("Five-landscape summaries selected (of 6)"); save("07_leave_one_landscape_stability.png")
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(P.MODELS)); ax.bar(x,cal.median_raw_cal_slope,color=[colors[m] for m in cal.model]); ax.axhline(1,color="black",ls="--"); ax.set_xticks(x,[labels[m] for m in cal.model]); ax.set_ylabel("Median raw calibration slope"); save("08_calibration_summary.png")
    top=importance.groupby("feature").importance_mean.median().nlargest(12).index; z=importance[importance.feature.isin(top)].groupby(["model","feature"]).importance_mean.median().unstack(0).reindex(top)
    fig,ax=plt.subplots(figsize=(9,6)); z.plot.barh(ax=ax,color=[colors[m] for m in z.columns]); ax.invert_yaxis(); ax.set_xlabel("Median held-out permutation PR-AUC drop"); ax.legend([labels[m] for m in z.columns],ncol=2); save("09_feature_importance_stability.png")
    fig,ax=plt.subplots(figsize=(9,5)); names=list(cot.label_set.unique()); x=np.arange(len(names)); width=.19
    for i,m in enumerate(P.MODELS):
        g=cot[cot.model.eq(m)].set_index("label_set").reindex(names); ax.bar(x+(i-1.5)*width,g.pr_lift,width,label=labels[m],color=colors[m])
    ax.set_xticks(x,["EQ1 primary","Post-EQ3 cumulative","Post-EQ4 cumulative"],rotation=12); ax.set_ylabel("PR lift"); ax.set_ylim(bottom=0); ax.legend(ncol=4); save("10_cotabato_inventory_sensitivity.png")


def assemble_secondary() -> dict:
    fr = require_primary_freeze()
    d = S.Data(); lolo, pred, targets = primary_state(d)
    controls = historical_controls(d); stable_tsv(OUT/"STAGE4EB_HISTORICAL_CONTROLS.tsv", controls)
    cot = cotabato_sensitivity(targets, pred); stable_tsv(OUT/"STAGE4EB_COTABATO_SENSITIVITY.tsv",cot)
    partial = lambda kind: sorted(OUT.glob(f"_diagnostic_{kind}_*.tsv"))
    for kind in ("importance", "complexity", "reproduction"):
        if not partial(kind):
            raise RuntimeError(f"STOP: missing isolated {kind} diagnostic")
    imp = pd.concat([pd.read_csv(x, sep="\t") for x in partial("importance")], ignore_index=True)
    comp = pd.concat([pd.read_csv(x, sep="\t") for x in partial("complexity")], ignore_index=True)
    repro = pd.concat([pd.read_csv(x, sep="\t") for x in partial("reproduction")], ignore_index=True)
    if set(comp.model) != set(P.MODELS):
        raise RuntimeError(f"STOP: incomplete diagnostic model set: {sorted(comp.model.unique())}")
    stable_tsv(OUT/"STAGE4EB_FEATURE_IMPORTANCE.tsv",imp); stable_tsv(OUT/"STAGE4EB_COMPLEXITY.tsv",comp)
    stable_tsv(OUT/"STAGE4EB_REPRODUCIBILITY.tsv",repro)
    ens=ensemble_diagnostic(targets,pred); stable_tsv(OUT/"STAGE4EB_ENSEMBLE_DIAGNOSTIC.tsv",ens)
    hs=h_status(); stable_tsv(OUT/"STAGE4EB_H_STATUS.tsv",hs)
    cal=calibration_summary(lolo); stable_tsv(OUT/"STAGE4EB_CALIBRATION_SUMMARY.tsv",cal)
    figures(lolo,pd.read_csv(OUT/"STAGE4EB_MODEL_COMPARISON.tsv",sep="\t"),pd.read_csv(OUT/"STAGE4EB_ROBUST_SELECTION.tsv",sep="\t"),
            pd.read_csv(OUT/"STAGE4EB_RANK_STABILITY.tsv",sep="\t"),imp,cot,cal)
    return dict(primary_freeze_sha256=sha256(PRIMARY_FREEZE), controls_rows=len(controls), cotabato_rows=len(cot),
                importance_rows=len(imp), complexity_rows=len(comp), reproducibility_all_pass=bool(repro.pass_within_tolerance.all()),
                figures=len(list(FIG.glob("*.png"))))


def run_secondary() -> dict:
    """Compatibility entry point; diagnostics must first be isolated by runtime."""
    return assemble_secondary()
