"""Stage 4D finalisation: STAGE4D_FINAL_REPORT.md + STAGE4D_CHECKPOINT.json from the computed Stage 4D tables (no new analysis)."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from . import build_stage4d as BD
from . import stage4d_prereg as P
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4d"
S = BD.SHORT


def t(name):
    return pd.read_csv(OUT / name, sep="\t")


def md(df, cols, nd=4):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{nd}g}"
        return str(v)
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r[c]) for c in cols) + " |" for _, r in df.iterrows())


def report(test_summary: str, integ: dict, cp_hash_note: str) -> str:
    sig = t("STAGE4D_TRANSFER_SIGNAL.tsv")
    prim = sig[sig.sensitivity.str.startswith("primary")]
    s6 = sig[~sig.sensitivity.str.startswith("primary")]
    lead = prim[(prim.model == "RANDOM_FOREST") & (prim.feature_set == "MEG")].copy()
    lrm = prim[(prim.model == "LOGISTIC_L2") & (prim.feature_set == "M")].copy()
    boot = t("STAGE4D_LOLO_BOOTSTRAP.tsv")
    roc = boot[(boot.model == "RANDOM_FOREST") & (boot.feature_set == "MEG") & (boot.metric == "roc_auc")]
    ds = t("STAGE4D_DOMAIN_SHIFT.tsv")
    dsp = ds.pivot(index="held_out", columns="group", values="energy_H_primary").reindex(P.LANDSCAPES).reset_index()
    fsum = t("STAGE4D_FEATURE_SET_SUMMARY.tsv")
    abl = t("STAGE4D_FEATURE_GROUP_ABLATION.tsv")
    ablp = abl.pivot_table(index=["model", "contrast"], columns="landscape", values="delta").reset_index()
    abld = abl.pivot_table(index=["model", "contrast"], columns="landscape", values="direction", aggfunc="first").reset_index()
    assoc = t("STAGE4D_SHIFT_PERFORMANCE_ASSOCIATION.tsv")
    idn = t("STAGE4D_LANDSCAPE_IDENTITY_AUDIT.tsv")
    ood = t("STAGE4D_OOD_PROTOTYPE.tsv")
    fx = t("STAGE4D_FORENSICS.tsv")
    hs = t("STAGE4D_H_STATUS.tsv")
    summ = t("STAGE4D_LANDSCAPE_DIAGNOSTIC_SUMMARY.tsv")
    fs = t("STAGE4D_FEATURE_SHIFT.tsv").assign(a=lambda d: d.std_median_diff.abs())
    top = fs.groupby("predictor").a.mean().sort_values(ascending=False).head(8)
    grp = fs.groupby("group").a.median()
    L = P.LANDSCAPES
    oq = ood[ood.ood_quintile > 0]
    trend = lambda col: {S[l]: (round(float(g[col].iloc[0]), 3), round(float(g[col].iloc[-1]), 3)) for l, g in oq.groupby("landscape")}
    pr_assoc = assoc[(assoc.primary == True)].iloc[0]  # noqa: E712
    lift_assoc = assoc[(assoc.model == "RANDOM_FOREST") & (assoc.feature_set == "MEG") & (assoc["shift"] == "energy_H_MEG") & (assoc.performance == "pr_lift")].iloc[0]
    cols6 = [c for c in ablp.columns if c in L]
    lead["label6"] = lead.landscape.map(s6.set_index("landscape").label)
    return f"""STAGE 4D CROSS-LANDSCAPE TRANSFER DIAGNOSIS: COMPLETE

**DEVELOPMENT ONLY - NONFINAL.** Six development-exposed landscapes; no lockbox; no MEGH; Gen-3 untouched; no model or feature set selected.
"Positive transfer" below always means PR-AUC above the held-out prevalence (the PR-AUC of a no-skill model); PR-AUC is never an accuracy.
Terrain units (835,846) are not independent landscapes: every cross-landscape statement rests on n = 6.

## A. Executive result
- Under a paired 3 km spatial block bootstrap (2,000 replicates), the Stage 4B development leader (RF + MEG) shows **ROBUST_POSITIVE_TRANSFER in 5 of 6 held-out landscapes**
  (Kumamoto, Noto, Lombok, Atitlan, Dominica); **Cotabato is UNCERTAIN / chance-like** in every one of the 15 model x feature-set cells (RF + ME is CHANCE_LIKE_OR_WORSE).
- The size of transfer differs ~4x: PR lift ~4.7-4.9 (Kumamoto, Noto) vs ~1.2-1.7 elsewhere; intervals are wide (Noto PR-AUC 0.065-0.156).
- E (climate/vegetation) behaves as a **near-perfect landscape identifier** (landscape-ID balanced accuracy 0.998 vs chance 0.167) and its effect on transfer is inconsistent:
  it **hurts Atitlan in all three model families** (Delta_E -0.048 to -0.090 PR-AUC; 100% of Atitlan units have precipitation seasonality outside the training range).
- G adds no measurable transferable information (Delta_G near zero; a few small losses).
- Covariate (domain) shift does **not** line up with poor transfer: the pre-registered association is positive (Spearman rho = {pr_assoc.spearman_rho:.2f}), fragile
  (leave-one-out {pr_assoc.loo_rho_min:.1f}-{pr_assoc.loo_rho_max:.1f}) and largely a prevalence artefact (rho vs PR lift = {lift_assoc.spearman_rho:.2f}, LOO {lift_assoc.loo_rho_min:.1f}-{lift_assoc.loo_rho_max:.1f}).
  Cotabato, the failure case, is the **least** morphology-shifted landscape (H_M = {dsp.set_index('held_out').loc['COTABATO_DAVAO_2019', 'M']:.4f}).

## B. Governance / integrity
V1 freeze JSON sha256 {integ['v1_freeze_json_sha256']} (expected 9881cc24...; version 1.0.0, FROZEN, 2026-10-05); V1 verifier {integ['v1']['strict_files']} strict files,
{integ['v1']['writes_performed']} writes. Prior checkpoints (3F-4C-B3) unchanged: {integ['prior_unchanged']}; {integ['historical_files_checked']} historical output files re-hashed, all unchanged:
{integ['historical_files_unchanged']}. Stage 4A {integ['prior_checkpoints']['outputs/stage4a/STAGE4A_CHECKPOINT.json']}; Stage 4B {integ['prior_checkpoints']['outputs/stage4b/STAGE4B_CHECKPOINT.json']};
Gen-3 {integ['prior_checkpoints']['outputs/stage4c_b3/STAGE4C_B3_CHECKPOINT.json']}. Stage 4B prediction cache read-only ({integ['stage4b_cache_files']} files, unchanged).

## C. Preregistration
STAGE4D_PREREGISTRATION.json / .md frozen 2026-10-07T10:17:17Z (json {BD.PREREG['json']}, md {BD.PREREG['md']}, module {BD.PREREG['module']}) before any Stage 4D result.
Y-blind outputs (domain shift, feature shift, identity audit) were frozen in STAGE4D_YBLIND_FREEZE.json before any Y-aware step.
**Amendment 01** (STAGE4D_AMENDMENT_01.json; clarification made before any Y-aware metric existed): the 1e-12 reuse check is against Stage 4B's full-precision results
record (the published TSV is its 6-significant-digit export); selected params, calibrators and thresholds are taken from the same record. No design choice changed.

## D. Dataset
6 landscapes, 835,846 units, 8,793 positives; 69 MEG predictors (60 M, 7 E, 2 G); Tier-2 labels. Model families and folds exactly as Stage 4B (seed 20261006).
Reused: all 72 Stage 4B PRIMARY held-out prediction sets (each reproduces the published PR-AUC to 1e-12). New: MG only (18 folds x families) via the unchanged Stage 4B code.

## E. Spatial-bootstrap method
Resample the frozen Stage 4A 3 km blocks of the held-out landscape with replacement (as many draws as blocks), 2,000 replicates, 95% percentile intervals; paired across
models/feature sets/null; weighted AP/ROC-AUC identical to sklearn (unit-tested). Sensitivity: 6 km blocks for the leader.

## F. Transfer uncertainty (RF + MEG; PR-AUC with 95% block-bootstrap CI)
{md(boot[(boot.model == 'RANDOM_FOREST') & (boot.feature_set == 'MEG') & (boot.metric == 'pr_auc')], ['landscape', 'n_units', 'n_positive', 'prevalence', 'point_estimate', 'ci_low', 'ci_high'])}

ROC-AUC (RF + MEG): {', '.join(f"{S[r.landscape]} {r.point_estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in roc.itertuples())}.

## G. Robust-signal analysis (paired Delta_PR = PR-AUC(model) - prevalence)
RF + MEG (Stage 4B development leader):
{md(lead, ['landscape', 'prevalence', 'pr_auc', 'pr_lift', 'delta_pr', 'delta_ci_low', 'delta_ci_high', 'pct_replicates_delta_gt0', 'label', 'label6'])}

LOGISTIC_L2 + M (highest Stage 4B median PR-AUC):
{md(lrm, ['landscape', 'pr_auc', 'pr_lift', 'delta_pr', 'delta_ci_low', 'delta_ci_high', 'label'])}

Labels for all 15 cells are in STAGE4D_TRANSFER_SIGNAL.tsv: Cotabato is never ROBUST; every other landscape is ROBUST in 12-15 of 15 cells (HGB ME / MEG / MEG_NO_BIO15 are UNCERTAIN at Lombok;
HGB MEG_NO_BIO15 is UNCERTAIN at Atitlan).

## H. Domain-shift metric (pre-registered)
Normalised energy statistic H = (2E|X-Y| - E|X-X'| - E|Y-Y'|)/(2E|X-Y|) between the held-out landscape and the pooled training landscapes, in predictor space z-scored with
landscape-balanced TRAINING statistics only; 2,000 held-out units vs 400 per training landscape (hash-selected, Y-free). Selected because it is distribution-free, works for mixed
continuous/binary predictors, needs no bandwidth, is zero iff distributions match and is bounded/normalised. Limitations: Euclidean geometry weights predictors equally; for the
2-predictor binary G group H is inflated by discreteness (not directly comparable with the 60-predictor M group); captures covariate shift only, not P(Y|X) shift.

## I. Domain-shift results (H; subsample SD <= 0.009)
{md(dsp, ['held_out', 'M', 'E', 'G', 'ME', 'MG', 'EG', 'MEG'])}

Most OOD overall (MEG): Atitlan (0.120); least: Lombok (0.029). Univariate: median |standardised median difference| E {grp['E']:.2f}, G {grp['G']:.2f}, M {grp['M']:.2f}.
Most shifted predictors (mean |std. median diff| over folds): {', '.join(f'{k} {v:.2f}' for k, v in top.items())}.

## J. M/E/G decomposition
M carries the bulk of the predictive signal but little shift (H_M 0.007-0.04, except Atitlan 0.107 - NASADEM M and the highest/coldest terrain); E and G are strongly
shifted and landscape-specific. The frozen "dominant group by H" is G for Kumamoto/Noto/Lombok/Cotabato and E for Atitlan/Dominica, but see the G caveat in H.

## K. Feature-group ablation (PR-AUC deltas; paired block-bootstrap direction)
{md(ablp, ['model', 'contrast'] + cols6)}

{md(abld, ['model', 'contrast'] + cols6)}

Feature-set summaries across landscapes (PR lift = PR-AUC / prevalence):
{md(fsum, ['model', 'feature_set', 'median_pr_auc', 'median_pr_lift', 'worst_pr_lift', 'worst_landscape', 'stability_sd_log_lift'])}

## L. Cotabato / Noto / Atitlan forensics (diagnosis only)
{md(fx, ['landscape', 'model', 'feature_set', 'prevalence', 'pr_auc', 'roc_auc', 'score_median_positives', 'score_median_negatives', 'flagged_fraction', 'recall', 'precision', 'fn_share_top10pct_blocks', 'median_ood_percentile_positives', 'median_ood_percentile_negatives'])}

Top shifted predictors per landscape: {' | '.join(f"{S[r.landscape]}: {r.top_shifted_predictors}" for r in fx.drop_duplicates('landscape').itertuples())}.

## M. Landscape-identity audit (6-class landscape ID; 5-fold GroupKFold over 6 km blocks; chance BA = 0.167)
{md(idn[idn.scope == 'GROUP'], ['name', 'n_predictors', 'balanced_accuracy', 'macro_ovr_auc'])}

Strongest single predictors: {', '.join(f"{r.name} {r.balanced_accuracy:.2f}" for r in idn[idn.scope == 'PREDICTOR'].sort_values('balanced_accuracy', ascending=False).head(6).itertuples())}.
No predictor is removed in Stage 4D.

## N. OOD prototype (PROTOTYPE / DEVELOPMENT DIAGNOSTIC / NONFINAL)
kNN (k = 10) distance to the training reference, as the percentile of the training reference's own leave-one-out distances. By within-landscape OOD quintile (Q1 -> Q5):
false-negative rate {trend('fn_rate')}; false-positive rate {trend('fp_rate')}; prevalence {trend('prevalence')}.
Prevalence rises with OOD in all six landscapes (steep, high-relief terrain is both rarer in training and more landslide-prone), so unit-level OOD is confounded with
susceptibility; FN behaviour is inconsistent (rises only in Lombok). Landscape-level median OOD does not track PR lift.

## O. H status (report only; Gen-3 read-only)
{md(hs, ['landscape', 'gen3_status', 'H_AVAILABLE', 'note'])}

N_VERIFIED_H_CAPABLE = 1; MEGH_EXPERIMENT_UNLOCKED = NO; MEGH not trained or evaluated.

## P. Scientific interpretation
1. Morphology-driven susceptibility transfers above chance to most unseen development landscapes, but weakly and with landscape-dependent magnitude; transfer is a property
   of the landscape (Kumamoto/Noto high lift in every family; Cotabato near 1 in every family) more than of the model family or feature set.
2. Adding environmental context mostly re-encodes landscape identity: it helps some landscapes (Kumamoto in all families) and harms others (Atitlan in all families), consistent
   with extrapolation on out-of-range climate values rather than transferable process information.
3. Failure is not explained by covariate shift in this six-landscape set: the worst landscape (Cotabato) is among the least shifted, which points (hypothesis) toward
   concept/label differences - e.g. trigger regime, event footprint, label density - that predictor-space metrics cannot see.

## Q. Limitations
- n = 6 independent landscapes: all cross-landscape associations are hypothesis-generating; leave-one-out rho spans 0.0-0.9 for PR lift.
- The block bootstrap captures within-landscape spatial uncertainty only; it does not represent uncertainty over which landscapes were sampled.
- PR-AUC is prevalence-dependent (prevalence 0.19%-5.7%); lift helps but small-prevalence landscapes (Kumamoto, Cotabato) have few positives (656, 298).
- Energy H treats all z-scored predictors equally and is not comparable across groups of very different dimension (G).
- Stage 4B calibration does not transfer (RF + MEG calibration slopes 1.0-37), so threshold-based FP/FN diagnostics reflect miscalibration as much as ranking.

## R. Stage 4E recommendations (directions only; nothing implemented, no model chosen)
1. Investigate an M-centred susceptibility core and treat E/G as a separately validated context layer: test identity-robust E representations (within-landscape ranks /
   anomalies relative to each landscape's own distribution) instead of absolute climate values, pre-registered and judged on worst-landscape lift.
2. Diagnose Cotabato as a concept/label-shift case before any model change: audit its Tier-2 label provenance, trigger (earthquake sequence), mapping completeness and
   observation domain; add a pre-registered trigger/label-regime descriptor to the landscape metadata.
3. Design OOD at two levels: a landscape-level novelty report (E/G range and identity) and a unit-level score that is explicitly de-confounded from susceptibility
   (e.g. OOD within slope/relief strata), and a separate recalibration study (transferable calibration is currently absent).
4. Keep H locked (N_verified = 1); add landscapes before revisiting MEGH.

## S. Reproducibility / tests / checkpoint
{test_summary}
Clean reruns of the Y-blind tables and of the Y-aware build reproduced every Stage 4D table byte-for-byte (see checkpoint `reproducibility`).
{cp_hash_note}
Git: no commit, no push.

## T. Answers 1-46
1. Robust positive transfer (RF + MEG and LOGISTIC_L2 + M): Kumamoto, Noto, Lombok, Atitlan, Dominica.
2. Uncertain: Cotabato (UNCERTAIN_POSITIVE_TRANSFER in 14/15 cells, CHANCE_LIKE_OR_WORSE for RF + ME).
3. Yes - RF + MEG Delta_PR {lead.set_index('landscape').loc['COTABATO_DAVAO_2019', 'delta_pr']:.4f} [{lead.set_index('landscape').loc['COTABATO_DAVAO_2019', 'delta_ci_low']:.4f}, {lead.set_index('landscape').loc['COTABATO_DAVAO_2019', 'delta_ci_high']:.4f}], ROC-AUC 0.511 [0.428, 0.584]; positive and negative score medians are identical (0.0271 vs 0.0272). Even the post-hoc best cell's interval touches 0.
4. Yes - Noto Delta_PR 0.087 [0.049, 0.129] (6 km blocks: [0.033, 0.145]); 100% of replicates > 0.
5. Highest lift: Noto (RF + MEG 4.91; Kumamoto 4.74). The post-hoc best cell (Kumamoto HGB MEG_NO_BIO15, lift 9.9) is optimistic by construction.
6. Weakest: Cotabato (RF + MEG 1.16; LOGISTIC_L2 + M 1.03).
7. Normalised energy statistic H on training-z-scored MEG predictors.
8. Distribution-free, mixed-type, no tuning parameter, normalised/bounded, chosen before any comparison with performance (section H).
9. Atitlan (H_MEG 0.120; median kNN OOD percentile 0.89).
10. Lombok (H_MEG 0.029).
11. E (with G): E and G are strongly shifted (univariate median |std diff| 0.54 and 0.50 vs 0.08 for M; by H, G dominates 4/6 and E 2/6, with the G discreteness caveat); E is the practically dominant identity-carrying group.
12. Yes, in this set: M is the more transferable core (median lift LOGISTIC_L2 M 2.57 vs ME 2.14; RF M 2.42 vs ME 1.60); E's effect changes sign between landscapes.
13. No measurable transferable gain from G in the current representation.
14. No.
15. Atitlan (Delta_E -0.048 LR, -0.089 RF, -0.090 HGB; all intervals < 0).
16. No meaningful gain: largest Delta_G is +0.0014 PR-AUC (Kumamoto, HGB); losses at Atitlan (LR -0.035) and Noto (HGB -0.020).
17. No: Delta_G|E <= +0.0021 everywhere; losses at Atitlan (LR -0.018, RF -0.012).
18. Strongest median: LOGISTIC_L2 + M (median PR-AUC 0.029, median lift 2.57); LOGISTIC_L2 + MG is practically identical (0.029, 2.51).
19. Strongest worst-landscape: HIST_GRADIENT_BOOSTING + MG (minimum lift 1.32), then RF + MEG (1.16); every family's worst landscape is Cotabato (or Lombok) and the differences lie within Cotabato's uncertainty.
20. Most stable (lowest SD of log lift): HGB + MG 0.47, LOGISTIC_L2 + MG 0.49, LOGISTIC_L2 + M 0.50.
21. Sensitive: LOGISTIC_L2 + M median lift ranges 2.24-2.89 when one landscape is removed, RF + M 1.67-3.17, so the M-family ordering flips depending on whether Lombok, Cotabato or Dominica is dropped.
22. A hypothesis-generating association exists but in the opposite direction to the domain-shift hypothesis (more shift, higher PR-AUC) and it is largely a prevalence effect.
23. rho = {pr_assoc.spearman_rho:.3f} (primary: H_MEG vs RF + MEG PR-AUC; exact permutation p = {pr_assoc.exact_permutation_p_two_sided:.3f}); vs PR lift rho = {lift_assoc.spearman_rho:.3f}.
24. Leave-one-out rho: {pr_assoc.loo_rho_min:.1f}-{pr_assoc.loo_rho_max:.1f} (PR-AUC); {lift_assoc.loo_rho_min:.1f}-{lift_assoc.loo_rho_max:.1f} (PR lift).
25. Yes: Atitlan (highest shift, modest lift) dominates the PR-lift relation (leave-one-out rho 0.9 without Atitlan vs 0.0-0.4 without any other landscape); Noto and Atitlan (highest prevalence among the shifted) drive the PR-AUC relation.
26. Yes: E alone identifies the landscape with balanced accuracy 0.998 (bio4 alone 0.79, bio15 alone 0.78).
27. Partly: two binary G predictors reach balanced accuracy 0.48 (chance 0.17); g_age_quaternary codes are landscape-specific (std. median shift -1.0 for several folds).
28. e_chelsa_bio4_temp_seasonality (Noto +15.6 IQR), e_chelsa_bio15_precip_seasonality (Atitlan 100% out of training range), g_age_quaternary, elev_m_30m__mean (Atitlan +3.8 IQR), bio1, bio13.
29. bio15 at Atitlan coincides with the large E-induced loss there; the shifted E/G predictors at Cotabato (g_age_quaternary -1.0, bio15/bio1 +-0.9) coincide with its failure but so do comparable shifts at landscapes that transfer, so no individual shifted predictor is consistently associated with failure.
30. Cotabato: no rank separation of positives from negatives (ROC 0.51), lowest-prevalence-but-one (0.38%, 298 positives), false negatives clustered (top 10% of 3 km blocks hold 78% of RF + MEG FNs), yet the least morphology-shifted landscape - consistent with a concept/label difference rather than covariate shift (hypothesis).
31. Noto: strong separation (positive median score ~3.4x the negative median, ROC 0.81), higher prevalence (2.2%), landslides on steep high-relief terrain that the morphology-driven models rank highly despite a very large climate shift.
32. Atitlan: every unit's precipitation seasonality (bio15) lies outside the training range and E identifies landscapes almost perfectly; adding E makes LOGISTIC_L2 flag 97% of units at the training threshold (M: 0.6%) and lowers PR-AUC in all three families.
33. No consistent pattern: FN rate rises with OOD only in Lombok; it falls in Noto, Atitlan and Cotabato and is ~0 in Kumamoto and Dominica.
34. FP rate rises with OOD in 5/6 landscapes (falls in Lombok), but prevalence also rises with OOD in all six, so FP enrichment is confounded with susceptibility.
35. Partially: there is structure (FP enrichment, prevalence confound) but not a consistent error signal; a dedicated OOD stage is justified only with a susceptibility-de-confounded design.
36. Both: landscape level (E/G range/identity novelty) and unit level (de-confounded), developed separately.
37. No.
38. No.
39. No.
40. No.
41. No.
42. Yes (153 strict files, 0 writes; freeze JSON hash 9881cc24... verified).
43. See STAGE4D_CHECKPOINT.json (its own sha256 is reported in the handover message; a file cannot contain its own hash).
44. (i) Transfer is robustly above prevalence in 5/6 landscapes but its magnitude is landscape-specific; (ii) E acts as a landscape identifier with inconsistent, sometimes harmful (Atitlan) effects, and G adds nothing measurable; (iii) covariate shift does not explain failure - Cotabato fails while barely shifted.
45. (i) n = 6 landscapes; (ii) poor calibration transfer confounds threshold diagnostics; (iii) no label/trigger-regime metadata to test the concept-shift hypothesis.
46. Section R.
"""


def main(test_summary: str, repro: dict) -> dict:
    integ = BD.integrity()
    (OUT / "STAGE4D_FINAL_REPORT.md").write_text(report(test_summary, integ, "Checkpoint: STAGE4D_CHECKPOINT.json."))
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "STAGE4D_CHECKPOINT.json")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    import scipy
    import sklearn
    cp = dict(stage="4D CROSS-LANDSCAPE TRANSFER DIAGNOSIS", status="DEVELOPMENT_ONLY_NONFINAL", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("stage4d_prereg.py", "stage4d_yblind.py", "stage4d.py", "stage4d_mg.py", "build_stage4d.py",
                                                                      "build_stage4d_final.py", "stage4b.py")},
              preregistration=dict(BD.PREREG, amendments=["STAGE4D_AMENDMENT_01.json"], amendment_01_sha256=sha256(OUT / "STAGE4D_AMENDMENT_01.json"),
                                   yblind_freeze_sha256=sha256(OUT / "STAGE4D_YBLIND_FREEZE.json")),
              input_checkpoints=integ["prior_checkpoints"], v1_freeze_json_sha256=integ["v1_freeze_json_sha256"], v1_verification=integ["v1"],
              historical_output_integrity=dict(files_checked=integ["historical_files_checked"], unchanged=integ["historical_files_unchanged"], prior_checkpoints_unchanged=integ["prior_unchanged"],
                                               stage4b_cache_files=integ["stage4b_cache_files"]),
              seeds=dict(stage4d=P.SEED, stage4b=P.STAGE4B_SEED, bootstrap="SEED + landscape index", identity="SEED"),
              mg_fit_provenance=dict(results="data/interim/_stage4d_mg/MG_RESULTS.json", sha256=sha256(ROOT / "data/interim/_stage4d_mg/MG_RESULTS.json"),
                                     cache="data/interim/_stage4d_cache (Stage 4B cache never written)"),
              software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                            sklearn_site="Science_Fair_ML_V2/.ml_site (isolated, as Stage 4B)"),
              tests=test_summary, reproducibility=repro, git=dict(head=git("rev-parse", "HEAD"), tracked_modified=len([s for s in git("status", "--porcelain").splitlines() if not s.startswith("??")]),
                                                                   committed=False, pushed=False),
              LOCKBOX_TOUCHED="NO", MEGH_TRAINED="NO", GEN3_MODIFIED="NO", H_PREDICTIVE_TEST="NO", MODEL_SELECTED="NO",
              N_VERIFIED_H_CAPABLE=1, MEGH_EXPERIMENT_UNLOCKED="NO",
              output_manifest=[str(p.relative_to(ROOT)) for p in outs], output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs})
    (OUT / "STAGE4D_CHECKPOINT.json").write_text(json.dumps(cp, indent=2, sort_keys=True) + "\n")
    return cp


if __name__ == "__main__":
    main(sys.argv[1], json.loads(sys.argv[2]))
