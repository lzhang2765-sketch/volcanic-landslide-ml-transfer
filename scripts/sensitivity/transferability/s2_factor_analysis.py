"""Stage C Phases 3-7 — POST HOC EXPLORATORY transferability-factor analysis (plan TRANSFER_FACTOR_PLAN_v1.0.0).

No model is fitted.
- Distribution shift uses predictor values only (no labels from any landscape).
- Associations are Spearman rank correlations on n <= 6 landscapes, with leave-one-landscape-out checks.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wasserstein_distance

OUT = Path(__file__).resolve().parents[2]
V2 = OUT.parent
D = OUT / "02_DATA"
L = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
SHIFT_FEATURES = ["elev_m_30m__mean", "slope_deg_30m__mean", "relief_m_r1000__mean", "rel_elev_local_r1000__mean"]
TAG = {"elev_m_30m__mean": "elev", "slope_deg_30m__mean": "slope", "relief_m_r1000__mean": "relief", "rel_elev_local_r1000__mean": "relelev"}


def log(m):
    print(m, flush=True)
    with (OUT / "05_REPRODUCIBILITY/EXECUTION_LOG.txt").open("a") as f:
        f.write(m + "\n")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ weighted distribution statistics
def wquantile(v_sorted, cw, q):
    return np.interp(q, cw, v_sorted)


def shift_stats(t: np.ndarray, pools: list[np.ndarray]) -> dict:
    p = np.concatenate(pools)
    w = np.concatenate([np.full(len(a), 1.0 / (len(pools) * len(a))) for a in pools])
    mu = np.sum(w * p)
    sd = np.sqrt(np.sum(w * (p - mu) ** 2))
    o = np.argsort(p, kind="stable")
    ps, ws = p[o], w[o]
    cw = np.cumsum(ws) - ws / 2
    lo, hi = wquantile(ps, cw, 0.01), wquantile(ps, cw, 0.99)
    ts = np.sort(t)
    grid = np.unique(np.concatenate([ts, ps]))
    ft = np.searchsorted(ts, grid, side="right") / len(ts)
    cwr = np.cumsum(ws)
    idx = np.searchsorted(ps, grid, side="right") - 1
    fp = np.where(idx >= 0, cwr[np.clip(idx, 0, None)], 0.0)
    return dict(target_mean=float(t.mean()), pool_mean=float(mu), pool_sd=float(sd),
                smd=float((t.mean() - mu) / sd) if sd > 0 else np.nan,
                abs_smd=float(abs(t.mean() - mu) / sd) if sd > 0 else np.nan,
                target_median=float(np.median(t)), pool_median=float(wquantile(ps, cw, 0.5)),
                w1_sd=float(wasserstein_distance(t, p, v_weights=w) / sd) if sd > 0 else np.nan,
                oor=float(((t < lo) | (t > hi)).mean()), oor_above=float((t > hi).mean()), oor_below=float((t < lo).mean()),
                pool_p01=float(lo), pool_p99=float(hi), ks=float(np.max(np.abs(ft - fp))))


# ------------------------------------------------------------------ association helpers
PERMS = np.array(list(itertools.permutations(range(6))))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, len(x), np.nan
    rho = float(spearmanr(x, y).statistic)
    if len(x) == 6:  # exact two-sided permutation p (descriptive only)
        rx, ry = rankdata(x), rankdata(y)
        sims = np.array([np.corrcoef(rx, ry[pp])[0, 1] for pp in PERMS])
        p = float(np.mean(np.abs(sims) >= abs(rho) - 1e-12))
    else:
        p = np.nan
    return rho, int(len(x)), p


def classify(rho, loo, n):
    if n < 5 or not np.isfinite(rho):
        return "NOT_ASSESSABLE", ""
    if abs(rho) < 0.4:
        return "WEAK", ""
    driven = [k for k, r in loo.items() if np.isfinite(r) and (np.sign(r) != np.sign(rho) or abs(r) < 0.2)]
    if driven:
        return "DRIVEN_BY_" + "|".join(driven), "|".join(driven)
    if abs(rho) >= 0.6 and all(np.sign(r) == np.sign(rho) and abs(r) >= 0.4 for r in loo.values() if np.isfinite(r)):
        return "STABLE", ""
    return "MODERATE_UNSTABLE", ""


def evidence(cls):
    return {"STABLE": "DESCRIPTIVELY_ASSOCIATED", "WEAK": "NOT_SUPPORTED"}.get(cls.split("_BY_")[0] if cls.startswith("DRIVEN") else cls, "INDETERMINATE")


def main():
    D.mkdir(parents=True, exist_ok=True)
    man = json.loads((OUT / "01_PROTOCOL/PROTOCOL_MANIFEST.json").read_text())
    assert sha(OUT / "01_PROTOCOL/TRANSFERABILITY_FACTOR_ANALYSIS_PLAN.md") == man["plan_sha256"], "STOP: plan hash mismatch"
    log(f"S2 START plan={man['plan_version']} sha={man['plan_sha256']}")

    # ---------------- frozen performance
    con = {r["held_out"]: r for r in json.loads((V2 / "outputs/stage5a/STAGE5A_MODEL_CONTRACT.json").read_text())["landscape_level_results"]}
    boot = pd.read_csv(V2 / "outputs/stage4e_b/STAGE4EB_SPATIAL_BOOTSTRAP.tsv", sep="\t")
    boot = boot[(boot.model == "LOGISTIC_L2") & (boot.metric == "pr_lift") & (boot.comparison == "POINT_INTERVAL")].set_index("held_out")
    a1 = pd.read_csv(V2 / "AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/ELEVATION_ABLATION_FOLD_RESULTS.csv")
    pdiff = pd.read_csv(V2 / "AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/PAIRED_PERFORMANCE_DIFFERENCES.csv")
    res = {a: {r["held_out"]: r for r in json.loads((V2 / f"data/interim/_stage4b_results/{a}.json").read_text())["results"]
               if r["model"] == "LOGISTIC_L2" and r["feature_set"] == "M"} for a in ("PRIMARY", "ATITLAN_EXCLUDED", "POLYGON_ONLY", "STRICT_DEM_QUALITY")}
    ctx = {r["landscape_id"]: r for r in csv.DictReader((V2 / "outputs/stage3g/STAGE3G_DIVERSITY_METADATA.tsv").open(), delimiter="\t")}
    inv = {r["landscape_id"]: r for r in csv.DictReader((V2 / "AUSSEF_2026_VALIDITY_AUDIT/INVENTORY_PROVENANCE_AUDIT.csv").open())}
    e02 = json.loads((V2 / "AUSSEF_2026_VALIDITY_AUDIT/evidence/E02_LABEL_GEOMETRY_DIAGNOSTICS.json").read_text())
    e03 = pd.read_csv(V2 / "AUSSEF_2026_VALIDITY_AUDIT/evidence/E03_ELEVATION_DISTRIBUTIONS.tsv", sep="\t").set_index("landscape_id")
    ds4a = pd.read_csv(V2 / "outputs/stage4a/STAGE4A_DOMAIN_SHIFT_AUDIT.tsv", sep="\t")
    ds4a = ds4a[ds4a.predictor == "__LANDSCAPE_DISTANCE__"].set_index("landscape_id")
    m_cols = pd.read_csv(V2 / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv", sep="\t").query("role == 'M' and predictor").column.tolist()
    assert len(m_cols) == 60

    # ---------------- Phase 4: distribution shift (predictors only)
    X = {}
    for l in L:
        with np.load(V2 / f"outputs/stage4a/STAGE4A_MEG_DATASET/{l}.npz", allow_pickle=False) as z:
            X[l] = {c: z[c].astype(float) for c in m_cols}  # y_positive deliberately not loaded
    rows = []
    for l in L:
        pool = [o for o in L if o != l]
        for c in m_cols:
            rows.append(dict(held_out=l, training_pool="|".join(pool), feature=c, primary_feature=c in SHIFT_FEATURES,
                             **shift_stats(X[l][c], [X[o][c] for o in pool])))
        log(f"shift done {l}")
    sh = pd.DataFrame(rows)
    sh.to_csv(D / "DISTRIBUTION_SHIFT_RESULTS.csv", index=False)

    # ---------------- Phase 3: master table
    lb = {c: a1[a1.config == c].set_index("held_out") for c in ("B0", "A1", "A2")}
    pdA = {c: pdiff[pdiff.comparison == f"{c}_minus_B0"].set_index("held_out") for c in ("A1", "A2")}
    mmin = {"KUMAMOTO_ASO_2016": 1000.0, "NOTO_PENINSULA_2024": 100.0, "ATITLAN_STAN_2005": 3.6}
    master = []
    for l in L:
        c, s = con[l], sh[sh.held_out == l].set_index("feature")
        r = dict(landscape_id=l, event_year=int(l[-4:]), n_units=int(c["n"]), n_positive=int(c["positives"]), prevalence=float(c["prevalence"]),
                 ap_B0=float(c["pr_auc"]), pr_lift_B0=float(c["pr_lift"]), pr_lift_B0_ci_low=float(boot.loc[l, "ci_low"]),
                 pr_lift_B0_ci_high=float(boot.loc[l, "ci_high"]), roc_auc_B0=float(c["roc_auc"]),
                 geometry="POINT_CENTRE" if l == "KUMAMOTO_ASO_2016" else "POLYGON",
                 label_meaning=inv[l]["source_or_affected"], observation_type=ctx[l]["target_mapping_method"],
                 min_mapping_unit_m2=mmin.get(l, np.nan),
                 completeness_indicator={"ATITLAN_STAN_2005": "93.6% mean / 97.8% pooled (87 grids)", "LOMBOK_RINJANI_2018": "95% estimated (quoted)"}.get(l, "NONE_DOCUMENTED"),
                 overlap_negative_fraction=float(e02[l]["overlap_negative_fraction_of_overlapping"]) if "overlap_negative_fraction_of_overlapping" in e02[l] else np.nan,
                 dem_product="NASADEM_SRTM_2000" if l == "ATITLAN_STAN_2005" else "COPERNICUS_GLO30",
                 tectonic_setting=ctx[l]["tectonic_setting"], volcanic_morphology=ctx[l]["morphology"], erosion_state=ctx[l]["erosion_state"],
                 climate=ctx[l]["climate"], trigger=ctx[l]["trigger"],
                 pr_lift_A1=float(lb["A1"].loc[l, "pr_lift"]), pr_lift_A2=float(lb["A2"].loc[l, "pr_lift"]),
                 rel_dlift_A1=float(pdA["A1"].loc[l, "relative_lift_change"]), rel_dlift_A2=float(pdA["A2"].loc[l, "relative_lift_change"]),
                 within_landscape_elev_auc=float(e03.loc[l, "within_landscape_elev_auc_descriptive"]),
                 std_coef_elev_B0=float(lb["B0"].loc[l, "std_coef_elev"]), std_coef_elev_centred_A2=float(lb["A2"].loc[l, "std_coef_elev_centred"]),
                 stage4a_domain_shift_all=float(ds4a.loc[l, "mean_standardised_median_shift_all"]), stage4a_domain_shift_M=float(ds4a.loc[l, "mean_shift_M"]))
        for f in SHIFT_FEATURES:
            t = TAG[f]
            r.update({f"{t}_smd": s.loc[f, "smd"], f"{t}_abs_smd": s.loc[f, "abs_smd"], f"{t}_w1_sd": s.loc[f, "w1_sd"],
                      f"{t}_oor": s.loc[f, "oor"], f"{t}_ks": s.loc[f, "ks"]})
        r["M_oor_count_gt10pct"] = int((s.loc[m_cols, "oor"] > 0.10).sum())
        r["M_mean_abs_smd"] = float(s.loc[m_cols, "abs_smd"].mean())
        r["M_features_oor_gt10pct"] = "|".join(s.loc[m_cols][s.loc[m_cols, "oor"] > 0.10].index)
        for a, key in (("ATITLAN_EXCLUDED", "pr_lift_atitlan_excluded_training"), ("POLYGON_ONLY", "pr_lift_kumamoto_excluded_training")):
            r[key] = float(res[a][l]["pr_auc"]) / float(res[a][l]["prevalence"]) if l in res[a] else np.nan
            r["rel_d" + key[3:]] = r[key] / r["pr_lift_B0"] - 1 if l in res[a] else np.nan
        master.append(r)
    M = pd.DataFrame(master)
    M.to_csv(D / "LANDSCAPE_FACTOR_MASTER.csv", index=False)

    # ---------------- Phase 5/6: associations
    resp = {"pr_lift_B0": "PRIMARY", "ap_B0": "SECONDARY", "roc_auc_B0": "SECONDARY", "rel_dlift_A1": "SECONDARY", "rel_dlift_A2": "SECONDARY"}
    prim_f = ["prevalence", "n_positive", "n_units", "elev_abs_smd", "slope_abs_smd", "relief_abs_smd"]
    sec_f = ["elev_w1_sd", "elev_oor", "elev_ks", "slope_w1_sd", "slope_oor", "slope_ks", "relief_w1_sd", "relief_oor", "relief_ks",
             "relelev_abs_smd", "M_mean_abs_smd", "M_oor_count_gt10pct", "stage4a_domain_shift_M", "overlap_negative_fraction"]
    pairs = [(f, "pr_lift_B0", "PRIMARY") for f in prim_f] + [(f, "pr_lift_B0", "SECONDARY_SHIFT_OR_INVENTORY") for f in sec_f] + \
            [(f, r_, "SECONDARY_RESPONSE") for r_ in ("ap_B0", "roc_auc_B0", "rel_dlift_A1", "rel_dlift_A2") for f in prim_f] + \
            [("pr_lift_B0", "rel_dlift_A1", "PHASE6_D"), ("pr_lift_B0", "rel_dlift_A2", "PHASE6_D"),
             ("within_landscape_elev_auc", "rel_dlift_A1", "PHASE6_ELEV"), ("within_landscape_elev_auc", "rel_dlift_A2", "PHASE6_ELEV")]
    assoc, sens = [], []
    Mi = M.set_index("landscape_id")
    for f, r_, role in pairs:
        rho, n, p = spearman(Mi[f], Mi[r_])
        loo = {}
        for x in L:
            sub = Mi.drop(x)
            loo[x] = spearman(sub[f], sub[r_])[0] if np.isfinite(Mi.loc[x, f]) else np.nan
            sens.append(dict(factor=f, response=r_, role=role, excluded_landscape=x, n=int(np.isfinite(sub[f]).sum()), spearman_rho=loo[x],
                             skipped="factor missing for excluded landscape" if not np.isfinite(Mi.loc[x, f]) else ""))
        cls, drv = classify(rho, {k: v for k, v in loo.items() if np.isfinite(v)}, n)
        fin = [v for v in loo.values() if np.isfinite(v)]
        assoc.append(dict(factor=f, response=r_, role=role, n=n, spearman_rho=rho, exact_permutation_p_descriptive=p,
                          loo_rho_min=min(fin) if fin else np.nan, loo_rho_max=max(fin) if fin else np.nan,
                          loo_rho_without_atitlan=loo.get("ATITLAN_STAN_2005", np.nan), stability_class=cls, driving_landscapes=drv,
                          evidence_class=evidence(cls), note="exploratory; n<=6; not confirmatory"))
    pd.DataFrame(assoc).to_csv(D / "FACTOR_ASSOCIATIONS.csv", index=False)
    pd.DataFrame(sens).to_csv(D / "ASSOCIATION_SENSITIVITY.csv", index=False)

    # ---------------- Phase 6: training composition (matched landscapes and labels only)
    tc = []
    comps = [("ATITLAN_EXCLUDED_TRAINING", "ATITLAN_EXCLUDED", "training composition"),
             ("KUMAMOTO_EXCLUDED_TRAINING", "POLYGON_ONLY", "training composition (frozen name POLYGON_ONLY)"),
             ("STRICT_DEM_POPULATION", "STRICT_DEM_QUALITY", "evaluated population changes; NOT a composition comparison")]
    for name, a, kind in comps:
        common = [l for l in L if l in res[a]]
        ref, alt = [], []
        for l in common:
            pr, al = res["PRIMARY"][l], res[a][l]
            same_labels = int(pr["positives"]) == int(al["positives"]) and int(pr["n"]) == int(al["n"])
            lr, la = pr["pr_auc"] / pr["prevalence"], al["pr_auc"] / al["prevalence"]
            ref.append(lr); alt.append(la)
            tc.append(dict(comparison=name, kind=kind, landscape=l, eval_rows_reference=int(pr["n"]), eval_rows_alternative=int(al["n"]),
                           identical_evaluation_labels=same_labels, pr_lift_reference_B0=lr, pr_lift_alternative=la, delta_pr_lift=la - lr,
                           relative_change=la / lr - 1, ap_reference=pr["pr_auc"], ap_alternative=al["pr_auc"], selected_params=al["params"]))
        tc.append(dict(comparison=name, kind=kind, landscape="MEDIAN_ON_IDENTICAL_SET:" + "|".join(x[:4] for x in common),
                       pr_lift_reference_B0=float(np.median(ref)), pr_lift_alternative=float(np.median(alt)),
                       delta_pr_lift=float(np.median(alt) - np.median(ref)), relative_change=float(np.median(alt) / np.median(ref) - 1),
                       identical_evaluation_labels=all(t["identical_evaluation_labels"] for t in tc if t["comparison"] == name and "identical_evaluation_labels" in t and t["landscape"] in common)))
    for c in ("A1", "A2"):
        for l in L:
            tc.append(dict(comparison=f"STAGE_A1_{c}", kind="feature representation (same training landscapes)", landscape=l,
                           eval_rows_reference=int(con[l]["n"]), eval_rows_alternative=int(con[l]["n"]), identical_evaluation_labels=True,
                           pr_lift_reference_B0=float(con[l]["pr_lift"]), pr_lift_alternative=float(lb[c].loc[l, "pr_lift"]),
                           delta_pr_lift=float(pdA[c].loc[l, "delta_pr_lift"]), relative_change=float(pdA[c].loc[l, "relative_lift_change"]),
                           ap_reference=float(con[l]["pr_auc"]), ap_alternative=float(lb[c].loc[l, "ap"]), selected_params="C=0.01"))
    pd.DataFrame(tc).to_csv(D / "TRAINING_COMPOSITION_COMPARISONS.csv", index=False)

    # ---------------- Phase 7: inventory heterogeneity (from Stage A + existing label sensitivities)
    cot = pd.read_csv(V2 / "outputs/stage4e_b/STAGE4EB_COTABATO_SENSITIVITY.tsv", sep="\t").query("model == 'LOGISTIC_L2'")
    k = e02["KUMAMOTO_SENSITIVITY_POINTS_VS_PRIMARY_UNITS"]
    ih = [
        dict(item="Kumamoto point-centre inventory", landscapes="KUMAMOTO_ASO_2016", evidence_class="VERIFIED_DIFFERENCE",
             evidence="GSI KML: 1,044 <Point> placemarks = approximate centres; one positive unit per centre; no buffer (Stage A INVENTORY_PROVENANCE_AUDIT.csv)",
             demonstrated_effect="Removing Kumamoto from TRAINING changed median lift on the 5 polygon landscapes 2.243 -> 2.230 (TRAINING_COMPOSITION_COMPARISONS.csv); effect on Kumamoto's OWN lift not testable without a polygon inventory",
             classification_of_effect="DIRECTLY_DEMONSTRATED (training effect, small); UNRESOLVED (own-fold effect)"),
        dict(item="Kumamoto later-mapped centres labelled negative", landscapes="KUMAMOTO_ASO_2016", evidence_class="VERIFIED_DIFFERENCE",
             evidence=f"July {k['SENSITIVITY_JULY']['in_primary_negative_units']} + undated {k['SENSITIVITY_UNDATED']['in_primary_negative_units']} centres in primary negative units (Stage A E02)",
             demonstrated_effect="none (S2/S3 label sensitivity never run)", classification_of_effect="PLAUSIBLE_BUT_UNTESTED (likely lowers Kumamoto AP)"),
        dict(item="Polygon cell-centre rule leaves mapped terrain in negative units", landscapes="NOTO|LOMBOK|COTABATO|ATITLAN|DOMINICA", evidence_class="VERIFIED_DIFFERENCE",
             evidence="overlap-negative share of overlapping units: " + "; ".join(f"{l[:4]} {e02[l]['overlap_negative_fraction_of_overlapping']:.3f}" for l in L if l != "KUMAMOTO_ASO_2016"),
             demonstrated_effect="see FACTOR_ASSOCIATIONS overlap_negative_fraction vs pr_lift_B0 (n=5, descriptive)", classification_of_effect="PLAUSIBLE_BUT_UNTESTED"),
        dict(item="Mapping thresholds differ", landscapes="KUMAMOTO(~1000 m2)|NOTO(~100 m2)|ATITLAN(3.6 m2)|others UNKNOWN", evidence_class="VERIFIED_DIFFERENCE",
             evidence="GSI disclaimers; Machado et al. 2022", demonstrated_effect="none", classification_of_effect="UNRESOLVED (n=3 documented)"),
        dict(item="Deposition areas included (Noto) vs whole affected polygons vs centres", landscapes="NOTO (source+deposit) | others", evidence_class="VERIFIED_DIFFERENCE",
             evidence="GSI Noto disclaimer: 斜面崩壊地や土砂堆積箇所", demonstrated_effect="none", classification_of_effect="PLAUSIBLE_BUT_UNTESTED"),
        dict(item="Label definition changes measured transfer (Cotabato cumulative labels)", landscapes="COTABATO_DAVAO_2019", evidence_class="VERIFIED_DIFFERENCE",
             evidence="; ".join(f"{r.label_set}: positives {r.positive_units}, lift {r.pr_lift:.3f}" for r in cot.itertuples()),
             demonstrated_effect="Same model, same units: lift 1.03 (EQ1) -> 1.49 / 1.42 with cumulative post-EQ3/EQ4 labels (STAGE4EB_COTABATO_SENSITIVITY.tsv; delta PR CI excludes 0)",
             classification_of_effect="DIRECTLY_DEMONSTRATED (label definition alters measured lift; cumulative labels mix later events)"),
        dict(item="Completeness / observation coverage", landscapes="ALL", evidence_class="VERIFIED_DIFFERENCE",
             evidence="Quantitative completeness only Atitlan (93.6%/97.8%); Lombok 95% quoted; others none; all Omega from source mapping extents",
             demonstrated_effect="none", classification_of_effect="UNRESOLVED"),
        dict(item="Possible pre-existing slides in event inventory", landscapes="ATITLAN_STAN_2005 (authors acknowledge)", evidence_class="VERIFIED_DIFFERENCE",
             evidence="Machado et al. 2022", demonstrated_effect="none", classification_of_effect="PLAUSIBLE_BUT_UNTESTED (could inflate Atitlan lift)"),
    ]
    pd.DataFrame(ih).to_csv(D / "INVENTORY_HETEROGENEITY_SUMMARY.csv", index=False)
    log("S2 DONE")


if __name__ == "__main__":
    main()
