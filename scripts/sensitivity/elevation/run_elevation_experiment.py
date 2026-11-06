"""AUSSEF 2026 Stage A.1 — elevation ablation experiment (protocol ELEV_ABLATION_PROTOCOL_v1.0.0).

Reuses the frozen Stage 4B code path (src/sfv2/stage4b.py) unchanged; analyses are added in memory only.
Writes only under AUSSEF_2026_ELEVATION_EXPERIMENT/ and the temporary fit cache given by $ELEV_EXP_CACHE.

Run from the repository root (see README.md):
  PYTHONDONTWRITEBYTECODE=1 ELEV_EXP_CACHE=/some/tmp/dir \
  .venv/bin/python -B Science_Fair_ML_V2/AUSSEF_2026_ELEVATION_EXPERIMENT/05_REPRODUCIBILITY/code/run_elevation_experiment.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib
import json
import os
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
V2 = EXP.parent
sys.path[:0] = [str(V2 / ".ml_site"), str(V2 / "src")]  # identical isolation used by Stage 4B (STAGE4B_CHECKPOINT.json software)

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from sfv2 import stage4b as S
from sfv2 import stage4d as D

RES, PRED = EXP / "02_RESULTS", EXP / "02_RESULTS/predictions"
LOG = EXP / "05_REPRODUCIBILITY/EXECUTION_LOG.txt"
FROZEN_CACHE = V2 / "data/interim/_stage4b_cache"
FROZEN_PRIMARY = V2 / "data/interim/_stage4b_results/PRIMARY.json"
CONTRACT = V2 / "outputs/stage5a/STAGE5A_MODEL_CONTRACT.json"
REPRO_5DE = V2 / "outputs/stage5d_e/STAGE5DE_BASELINE_REPRODUCTION.json"
BOOT_4EB = V2 / "outputs/stage4e_b/STAGE4EB_SPATIAL_BOOTSTRAP.tsv"
COEF_5E = V2 / "outputs/stage5e/STAGE5E_LANDSCAPE_IMPORTANCE.tsv"
PROTOCOL = EXP / "01_PROTOCOL/ELEVATION_EXPERIMENT_PROTOCOL.md"
MANIFEST = EXP / "01_PROTOCOL/PROTOCOL_MANIFEST.json"

ELEV, ELEV_C = "elev_m_30m__mean", "elev_m_30m__mean__landscape_centred"
MODEL, POP, SCHEME = "LOGISTIC_L2", "ALL_VALID", "LANDSCAPE_BALANCED"
CONFIGS = {"B0": "M", "A1": "A1_NO_ABS_ELEV", "A2": "A2_LANDSCAPE_CENTRED_ELEV"}  # config -> feature-set name
BOOT_SEED, REPS = 20261008, 2000
TOL_AP, TOL_PRED, TOL_SUMMARY = 1e-12, 1e-12, 5e-11
L = S.LANDSCAPES


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def robust(lifts) -> float:
    lifts = np.asarray(lifts, float)
    return float(0.70 * np.log(np.median(lifts)) + 0.30 * np.log(np.min(lifts)))


def band(dr: float) -> str:
    return "IDENTICAL" if abs(dr) <= 1e-9 else "NEGLIGIBLE" if abs(dr) < 0.02 else "HIGHER_R" if dr > 0 else "LOWER_R"


def preflight() -> dict:
    m = json.loads(MANIFEST.read_text())
    if sha(PROTOCOL) != m["protocol_sha256"]:
        raise SystemExit("STOP: protocol hash does not match PROTOCOL_MANIFEST.json")
    env = dict(python=sys.version.split()[0], sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__)
    need = dict(python="3.10.9", sklearn="1.7.2", numpy="2.2.6", scipy="1.15.3", pandas="2.3.3")
    bad = {k: (env[k], v) for k, v in need.items() if env[k] != v}
    if bad:
        raise SystemExit(f"STOP: environment differs from Stage 4B record: {bad}")
    if S.LANDSCAPES != ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]:
        raise SystemExit("STOP: landscape order differs from protocol")
    code = {p: sha(V2 / p) for p in ("src/sfv2/stage4b.py", "src/sfv2/stage4d.py", "src/sfv2/stage4a_base.py", "src/sfv2/build_stage4a.py", "src/sfv2/schema.py")}
    if code["src/sfv2/stage4b.py"] != "55c21b564b71f4d3ac7fb267461293689e83ecf81e97f9f066000dad825720a4":
        raise SystemExit("STOP: stage4b.py differs from the hash recorded in STAGE4B_CHECKPOINT.json")
    return dict(protocol_sha256=m["protocol_sha256"], protocol_version=m["protocol_version"], env=env, code=code)


def build_feature_sets(d: S.Data) -> None:
    m = list(d.fs["M"])
    assert len(m) == 60 and m.count(ELEV) == 1
    fp = json.loads((V2 / "outputs/stage4b/STAGE4B_FOLD_PREPROCESSING.json").read_text())
    for lid in L:
        assert fp[f"{lid}|M"]["columns"] == m, "M column order differs from frozen fold preprocessing"
    j = d.idx[ELEV]
    means = {}
    for lid in L:  # unsupervised, label-free, full eligible domain of each landscape
        x = d.L[lid]["X"]
        means[lid] = float(x[:, j].mean())
        d.L[lid]["X"] = np.hstack([x, (x[:, j] - means[lid])[:, None]])
    d.idx[ELEV_C] = len(d.cols)
    d.cols = d.cols + [ELEV_C]
    d.fs["A1_NO_ABS_ELEV"] = [c for c in m if c != ELEV]
    d.fs["A2_LANDSCAPE_CENTRED_ELEV"] = [ELEV_C if c == ELEV else c for c in m]
    for cfg, fs in CONFIGS.items():
        S.ANALYSES[f"ELEV_{cfg}"] = (L, POP, SCHEME, [fs], [MODEL])
    d.centring_means = means


def run(d: S.Data, cfg: str) -> dict:
    fs = CONFIGS[cfg]
    res, inner, meta = S.run_analysis(d, f"ELEV_{cfg}", log=log)
    rows = {r["held_out"]: r for r in res if r["model"] == MODEL}
    out = {}
    for k in L:
        mk = meta[f"ELEV_{cfg}|{k}|{fs}|{MODEL}"]
        p = np.load(S.CACHE / f"{mk['outer_prediction_key']}__{k}.npy")
        # refit the (deterministic) outer model once to read coefficients; must reproduce the cached scores exactly
        est, cols, prep, _, _ = S.fit(d, mk["train"], fs, MODEL, mk["params"], POP, SCHEME)
        p2 = S.predict(d, est, cols, prep, k, POP)
        assert np.array_equal(p, p2), f"refit mismatch {cfg} {k}"
        names = d.fs[fs]
        coef = dict(zip(names, est.coef_[0].tolist()))
        out[k] = dict(row=rows[k], p=p, key=mk["outer_prediction_key"], train=mk["train"], params=mk["params"],
                      inner_scores=mk["inner_scores"], coef=coef, n_iter=int(est.n_iter_[0]))
        np.save(PRED / f"{cfg}__{k}.npy", p)
    return out


def gate(d: S.Data, b0: dict) -> pd.DataFrame:
    frozen = {r["held_out"]: r for r in json.loads(FROZEN_PRIMARY.read_text())["results"] if r["model"] == MODEL and r["feature_set"] == "M"}
    repro = {f["held_out"]: f for f in json.loads(REPRO_5DE.read_text())["folds"]}
    contract = json.loads(CONTRACT.read_text())
    rows = []

    def chk(name, lid, exp, obs, tol=None):
        if tol is None:
            ok, diff = exp == obs, ""
        else:
            diff = abs(float(exp) - float(obs)); ok = diff <= tol
        rows.append(dict(check=name, landscape=lid, expected=exp, observed=obs, abs_difference=diff, tolerance=tol if tol is not None else "EXACT", passed=bool(ok)))

    lifts, aps = [], []
    for k in L:
        r, f, y = b0[k]["row"], frozen[k], d.L[k]["y"]
        fpath = FROZEN_CACHE / f"{b0[k]['key']}__{k}.npy"
        chk("fold_membership_train", k, json.dumps(sorted(l for l in L if l != k)), json.dumps(sorted(b0[k]["train"])))
        chk("cache_key_equals_frozen_path_exists", k, True, fpath.exists())
        if fpath.exists():
            fp = np.load(fpath)
            chk("outer_scores_max_abs_diff", k, 0.0, float(np.max(np.abs(fp - b0[k]["p"]))), TOL_PRED)
            chk("outer_scores_sha256_vs_frozen_cache", k, hashlib.sha256(fp.tobytes()).hexdigest(), hashlib.sha256(b0[k]["p"].tobytes()).hexdigest())
        chk("outer_scores_array_sha256_vs_stage5de_record", k, repro[k]["prediction_sha256"], hashlib.sha256(b0[k]["p"].tobytes()).hexdigest())
        chk("outer_scores_file_sha256_vs_stage5de_B0_file", k, sha(V2 / f"outputs/stage5d_e/predictions/B0__{k}__scores_v1.npy"), sha(PRED / f"B0__{k}.npy"))
        chk("selected_params", k, f["params"], json.dumps(b0[k]["params"]))
        chk("n_predictions", k, int(f["n"]), int(len(b0[k]["p"])))
        chk("positives", k, int(f["positives"]), int(y.sum()))
        chk("ap_vs_frozen_PRIMARY", k, float(f["pr_auc"]), float(r["pr_auc"]), TOL_AP)
        chk("ap_recomputed_sklearn", k, float(f["pr_auc"]), float(average_precision_score(y, b0[k]["p"])), TOL_AP)
        chk("pr_lift_vs_frozen", k, float(f["pr_auc"]) / float(f["prevalence"]), float(r["pr_auc"]) / float(r["prevalence"]), TOL_AP * 1e3)
        aps.append(r["pr_auc"]); lifts.append(r["pr_auc"] / r["prevalence"])
    hist = [h for h in contract["historical_model_comparison"] if h["model"] == MODEL][0]
    chk("median_ap_vs_contract", "ALL", hist["median_pr_auc"], float(np.median(aps)), TOL_SUMMARY)
    chk("median_pr_lift_vs_contract", "ALL", hist["median_pr_lift"], float(np.median(lifts)), TOL_SUMMARY)
    chk("worst_pr_lift_vs_contract", "ALL", hist["worst_pr_lift"], float(np.min(lifts)), TOL_SUMMARY)
    chk("robust_score_vs_contract", "ALL", hist["robust_score"], robust(lifts), TOL_SUMMARY)
    chk("total_predictions", "ALL", 835846, int(sum(len(b0[k]["p"]) for k in L)))
    chk("total_positives", "ALL", 8793, int(sum(d.L[k]["y"].sum() for k in L)))
    # informational (not gating): standardised elevation coefficient vs Stage 5E
    c5 = pd.read_csv(COEF_5E, sep="\t")
    for k in L:
        exp = float(c5[(c5.landscape == k) & (c5.feature == ELEV)].standardized_coefficient.iloc[0])
        rows.append(dict(check="INFO_elev_std_coefficient_vs_stage5e", landscape=k, expected=exp, observed=b0[k]["coef"][ELEV],
                         abs_difference=abs(exp - b0[k]["coef"][ELEV]), tolerance="INFO_ONLY", passed=abs(exp - b0[k]["coef"][ELEV]) < 1e-6))
    return pd.DataFrame(rows)


def bootstrap(d: S.Data, R: dict) -> dict:
    out = {}
    for li, k in enumerate(L):
        with np.load(S.MEG_DIR / f"{k}.npz", allow_pickle=False) as z:
            b3, ids = z["spatial_block_3000m"], z["terrain_unit_id"]
        assert np.array_equal(ids, d.L[k]["ids"])
        y = d.L[k]["y"]
        code, nb = D.block_codes(b3)
        srt = {c: D.Sorted(R[c][k]["p"], y) for c in R}
        rng = np.random.default_rng(BOOT_SEED + li)
        A = {c: np.full((REPS, 2), np.nan) for c in R}
        prev = np.full(REPS, np.nan)
        dropped = 0
        for i in range(REPS):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code].astype(np.float64)
            if w[y].sum() == 0:
                dropped += 1
                continue
            prev[i] = w[y].sum() / w.sum()
            for c in R:
                A[c][i] = srt[c].ap_auc(w)
        out[k] = dict(A=A, prev=prev, dropped=dropped, n_blocks=nb)
        log(f"bootstrap {k}: blocks={nb} dropped={dropped}")
    return out


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def feature_audit(d: S.Data) -> pd.DataFrame:
    """Descriptive, label-free: dependence of each M column on absolute elevation level."""
    from scipy.stats import spearmanr
    defs = pd.read_csv(V2 / "outputs/stage2d/feature_definitions.csv")
    fdef = dict(zip(defs.feature_name, defs.formula_algorithm))
    m = d.fs["M"]
    X = np.vstack([d.L[k]["X"][:, [d.idx[c] for c in m]] for k in L])
    lab = np.concatenate([np.full(len(d.L[k]["y"]), i) for i, k in enumerate(L)])
    w = np.concatenate([np.full(len(d.L[k]["y"]), 1 / len(d.L[k]["y"])) for k in L])
    e = X[:, m.index(ELEV)]
    rng = np.random.default_rng(0)
    sub = np.concatenate([rng.choice(np.flatnonzero(lab == i), min(20000, (lab == i).sum()), replace=False) for i in range(6)])
    rows = []
    for j, c in enumerate(m):
        base = c.rsplit("__", 1)[0]
        x = X[:, j]
        gm = np.average(x, weights=w)
        tot = np.average((x - gm) ** 2, weights=w)
        between = sum((1 / 6) * (x[lab == i].mean() - gm) ** 2 for i in range(6))
        within_rho = [spearmanr(x[lab == i][:20000], e[lab == i][:20000]).statistic for i in range(6)]
        rows.append(dict(feature=c, formula=fdef.get(base, ""),
                         uses_absolute_elevation_level="YES" if c == ELEV else "NO (invariant to a constant elevation offset)",
                         removed_in_A1=c == ELEV, replaced_in_A2=c == ELEV,
                         between_landscape_variance_fraction=round(float(between / tot), 4) if tot > 0 else np.nan,
                         pooled_spearman_with_elev_balanced_sample=round(float(spearmanr(x[sub], e[sub]).statistic), 4),
                         median_within_landscape_spearman_with_elev=round(float(np.nanmedian(within_rho)), 4)))
    # descriptive OLS: how much absolute elevation is linearly recoverable from the other 59 columns
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    others = [i for i, c in enumerate(m) if c != ELEV]
    A_ = np.c_[np.ones(len(sub)), Z[sub][:, others]]
    coef, *_ = np.linalg.lstsq(A_, Z[sub, m.index(ELEV)], rcond=None)
    r2 = 1 - np.var(Z[sub, m.index(ELEV)] - A_ @ coef) / np.var(Z[sub, m.index(ELEV)])
    rows.append(dict(feature="__DESCRIPTIVE_OLS_R2_elev_from_other_59_M__", formula="balanced 20k/landscape sample; not a landslide model",
                     uses_absolute_elevation_level="", removed_in_A1="", replaced_in_A2="", between_landscape_variance_fraction="",
                     pooled_spearman_with_elev_balanced_sample=round(float(r2), 4), median_within_landscape_spearman_with_elev=""))
    return pd.DataFrame(rows)


def main() -> None:
    for p in (RES, PRED, LOG.parent):
        p.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ["ELEV_EXP_CACHE"]).resolve()
    if FROZEN_CACHE.resolve() == cache or V2.resolve() in cache.parents and EXP.resolve() not in cache.parents:
        raise SystemExit("STOP: cache must not be inside frozen project folders")
    S.CACHE = cache
    info = preflight()
    log(f"START protocol={info['protocol_version']} sha={info['protocol_sha256']} env={info['env']} cache={cache}")
    d = S.Data()
    build_feature_sets(d)
    log(f"A2 landscape means (label-free): {json.dumps(d.centring_means)}")

    # ---------------- Phase 3: baseline reproduction gate
    R = {"B0": run(d, "B0")}
    g = gate(d, R["B0"])
    g.to_csv(RES / "BASELINE_REPRODUCTION_AUDIT.csv", index=False)
    gating = g[~g.check.str.startswith("INFO_")]
    if not gating.passed.all():
        (RES / "BASELINE_REPRODUCTION_DIAGNOSTIC.md").write_text(
            "# Baseline reproduction FAILED\n\nFailed checks:\n\n" + gating[~gating.passed].to_string(index=False) + "\n\nA1/A2 were NOT run.\n")
        log("GATE: FAIL — stopping before A1/A2")
        raise SystemExit(2)
    log(f"GATE: PASS ({len(gating)} gating checks)")

    # ---------------- Phase 4: A1, A2
    for cfg in ("A1", "A2"):
        R[cfg] = run(d, cfg)

    # ---------------- Phase 5
    BS = bootstrap(d, R)
    # informational check of the bootstrap implementation against Stage 4E-B LOGISTIC_L2 intervals
    b4 = pd.read_csv(BOOT_4EB, sep="\t")
    fold_rows, diff_rows = [], []
    for k in L:
        y = d.L[k]["y"]; prev = float(y.mean()); bk = BS[k]
        for c in R:
            r = R[c][k]
            ap, roc = float(r["row"]["pr_auc"]), float(r["row"]["roc_auc"])
            lo, hi = pct(bk["A"][c][:, 0]); llo, lhi = pct(bk["A"][c][:, 0] / bk["prev"])
            ref = b4[(b4.held_out == k) & (b4.model == MODEL) & (b4.metric == "pr_auc") & (b4.comparison == "POINT_INTERVAL")]
            fold_rows.append(dict(config=c, feature_set=CONFIGS[c], n_features=len(d.fs[CONFIGS[c]]), held_out=k,
                                  train_landscapes="|".join(r["train"]), n=len(y), positives=int(y.sum()), prevalence=prev,
                                  selected_C=r["params"]["C"], inner_mean_ap_by_grid=json.dumps(r["inner_scores"]),
                                  ap=ap, pr_lift=ap / prev, roc_auc=roc, ap_ci_low=lo, ap_ci_high=hi, pr_lift_ci_low=llo, pr_lift_ci_high=lhi,
                                  std_coef_elev=r["coef"].get(ELEV, np.nan), std_coef_elev_centred=r["coef"].get(ELEV_C, np.nan),
                                  std_coef_elev_sd=r["coef"].get("elev_m_30m__sd", np.nan), lbfgs_n_iter=r["n_iter"],
                                  bootstrap_reps_used=REPS - bk["dropped"], n_blocks_3km=bk["n_blocks"],
                                  b0_ap_ci_matches_stage4eb=(bool(np.isclose(ref.ci_low.iloc[0], lo) and np.isclose(ref.ci_high.iloc[0], hi)) if c == "B0" and len(ref) else ""),
                                  prediction_sha256=sha(PRED / f"{c}__{k}.npy")))
        for c in ("A1", "A2"):
            a, b = R[c][k]["row"], R["B0"][k]["row"]
            dap = bk["A"][c][:, 0] - bk["A"]["B0"][:, 0]
            dlift = dap / bk["prev"]
            dlb = float(b["pr_auc"]) / prev
            dl = float(a["pr_auc"]) / prev - dlb
            rel = dl / dlb
            diff_rows.append(dict(comparison=f"{c}_minus_B0", held_out=k, delta_ap=float(a["pr_auc"]) - float(b["pr_auc"]),
                                  delta_ap_ci_low=pct(dap)[0], delta_ap_ci_high=pct(dap)[1],
                                  delta_pr_lift=dl, delta_pr_lift_ci_low=pct(dlift)[0], delta_pr_lift_ci_high=pct(dlift)[1],
                                  relative_lift_change=rel, delta_roc_auc=float(a["roc_auc"]) - float(b["roc_auc"]),
                                  classification="IMPROVED" if rel >= 0.05 else "WORSENED" if rel <= -0.05 else "UNCHANGED_WITHIN_5PCT",
                                  ap_sign="UP" if float(a["pr_auc"]) > float(b["pr_auc"]) else "DOWN" if float(a["pr_auc"]) < float(b["pr_auc"]) else "EQUAL",
                                  bootstrap_interval_excludes_zero=bool(pct(dap)[0] > 0 or pct(dap)[1] < 0)))
    folds, diffs = pd.DataFrame(fold_rows), pd.DataFrame(diff_rows)
    folds.to_csv(RES / "ELEVATION_ABLATION_FOLD_RESULTS.csv", index=False)
    diffs.to_csv(RES / "PAIRED_PERFORMANCE_DIFFERENCES.csv", index=False)

    lift = {c: folds[folds.config == c].set_index("held_out").reindex(L).pr_lift for c in R}
    rb = {c: robust(lift[c]) for c in R}
    # replicate-wise R (within-landscape sampling uncertainty only)
    rep_lift = {c: np.column_stack([BS[k]["A"][c][:, 0] / BS[k]["prev"] for k in L]) for c in R}
    ok = np.all(np.isfinite(rep_lift["B0"]), axis=1)
    rep_R = {c: 0.70 * np.log(np.median(rep_lift[c][ok], 1)) + 0.30 * np.log(np.min(rep_lift[c][ok], 1)) for c in R}
    summ, rob = [], []
    for c in R:
        g_ = folds[folds.config == c]
        dd = diffs[diffs.comparison == f"{c}_minus_B0"]
        dR = rb[c] - rb["B0"]
        dci = pct(rep_R[c] - rep_R["B0"]) if c != "B0" else ("", "")
        summ.append(dict(config=c, feature_set=CONFIGS[c], n_features=int(g_.n_features.iloc[0]),
                         median_ap=float(g_.ap.median()), mean_ap=float(g_.ap.mean()), median_pr_lift=float(lift[c].median()),
                         min_pr_lift=float(lift[c].min()), worst_landscape=lift[c].idxmin(), median_roc_auc=float(g_.roc_auc.median()),
                         robust_score=rb[c], delta_R_vs_B0=dR, delta_R_band=band(dR) if c != "B0" else "REFERENCE",
                         delta_R_ci_low_within_landscape=dci[0], delta_R_ci_high_within_landscape=dci[1],
                         robust_score_ci_low=pct(rep_R[c])[0], robust_score_ci_high=pct(rep_R[c])[1], bootstrap_reps_used=int(ok.sum()),
                         folds_improved_5pct=int((dd.classification == "IMPROVED").sum()) if c != "B0" else "",
                         folds_worsened_5pct=int((dd.classification == "WORSENED").sum()) if c != "B0" else "",
                         folds_unchanged_5pct=int((dd.classification == "UNCHANGED_WITHIN_5PCT").sum()) if c != "B0" else "",
                         folds_ap_up=int((dd.ap_sign == "UP").sum()) if c != "B0" else "", folds_ap_down=int((dd.ap_sign == "DOWN").sum()) if c != "B0" else ""))
        rob.append(dict(config=c, subset="ALL_SIX", excluded_landscape="", median_pr_lift=float(lift[c].median()), min_pr_lift=float(lift[c].min()),
                        robust_score=rb[c], delta_R_vs_B0_same_subset=dR))
        for x in L:
            s, s0 = lift[c].drop(x), lift["B0"].drop(x)
            rob.append(dict(config=c, subset="LEAVE_ONE_LANDSCAPE_OUT", excluded_landscape=x, median_pr_lift=float(s.median()),
                            min_pr_lift=float(s.min()), robust_score=robust(s), delta_R_vs_B0_same_subset=robust(s) - robust(s0)))
    pd.DataFrame(summ).to_csv(RES / "ELEVATION_ABLATION_SUMMARY.csv", index=False)
    pd.DataFrame(rob).to_csv(RES / "ROBUSTNESS_COMPARISON.csv", index=False)
    feature_audit(d).to_csv(RES / "FEATURE_DERIVATIVE_AUDIT.csv", index=False)
    json.dump(dict(info, centring_means=d.centring_means, robust=rb, bootstrap_reps_used=int(ok.sum())),
              (EXP / "05_REPRODUCIBILITY/RUN_INFO.json").open("w"), indent=2, sort_keys=True)
    log("DONE " + json.dumps({c: dict(R=round(rb[c], 6), median_lift=round(float(lift[c].median()), 6)) for c in R}))


if __name__ == "__main__":
    main()
