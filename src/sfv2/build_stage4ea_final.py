"""Stage 4E-A finalisation: STAGE4EA_FINAL_REPORT.md + STAGE4EA_CHECKPOINT.json from the frozen Stage 4E-A tables (no new analysis)."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from . import build_stage4d as BD
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_a"
PRIOR4D = ("outputs/stage4d/STAGE4D_CHECKPOINT.json", "53041df090fb96edaeb6ee414dd49bd9a86a8e64d1683d8b46f567f37fa544fd")
SH = {"KUMAMOTO_ASO_2016": "Kumamoto", "NOTO_PENINSULA_2024": "Noto", "LOMBOK_RINJANI_2018": "Lombok", "COTABATO_DAVAO_2019": "Cotabato",
      "ATITLAN_STAN_2005": "Atitlan", "DOMINICA_ERIKA_2015": "Dominica"}


def t(n):
    return pd.read_csv(OUT / n, sep="\t")


def md(df, cols, nd=4):
    f = lambda v: ("" if not np.isfinite(v) else f"{v:.{nd}g}") if isinstance(v, (float, np.floating)) else str(v)
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r[c]) for c in cols) + " |" for _, r in df.iterrows())


def integrity():
    i = BD.integrity()
    cp4d = json.loads((ROOT / PRIOR4D[0]).read_text())
    i["stage4d_checkpoint_ok"] = sha256(ROOT / PRIOR4D[0]) == PRIOR4D[1]
    i["stage4d_outputs_unchanged"] = all(sha256(ROOT / p) == h for p, h in cp4d["output_hashes"].items())
    return i


def report(tests: str, repro: str, integ: dict) -> str:
    prov, geo, trig, cond, aux, cls = (t(n) for n in ("STAGE4EA_INVENTORY_PROVENANCE_AUDIT.tsv", "STAGE4EA_LABEL_GEOMETRY_AUDIT.tsv", "STAGE4EA_TRIGGER_AUDIT.tsv",
                                                      "STAGE4EA_COTABATO_CONDITIONAL_M_AUDIT.tsv", "STAGE4EA_COTABATO_AUX_INVENTORY_TEST.tsv",
                                                      "STAGE4EA_COTABATO_FAILURE_CLASSIFICATION.tsv"))
    idn, ret, ar, scr, dec, hs = (t(n) for n in ("STAGE4EA_E_IDENTITY_AUDIT.tsv", "STAGE4EA_E_INFORMATION_RETENTION.tsv", "STAGE4EA_ATITLAN_RANGE_TEST.tsv",
                                                 "STAGE4EA_E_TRANSFER_SCREEN.tsv", "STAGE4EA_E_ADVANCEMENT_DECISION.tsv", "STAGE4EA_H_STATUS.tsv"))
    c = cls.set_index("item").evidence
    summ = scr.groupby(["model", "feature_set"]).agg(median_pr_auc=("pr_auc", "median"), median_pr_lift=("pr_lift", "median"), worst_pr_lift=("pr_lift", "min"),
                                                    sd_log_lift=("pr_lift", lambda s: float(np.log(s).std(ddof=1))), median_cal_slope=("cal_slope", "median")).reset_index()
    piv = lambda m, v: scr[scr.model == m].pivot(index="feature_set", columns="landscape", values=v).reset_index()
    at = scr[scr.landscape == "ATITLAN_STAN_2005"][["model", "feature_set", "pr_auc", "flagged_fraction", "cal_slope"]]
    ids = idn[idn.scope == "E_ONLY"].set_index("representation").balanced_accuracy
    return f"""STAGE 4E-A COTABATO FAILURE AUDIT + IDENTITY-ROBUST ENVIRONMENTAL REPRESENTATION DESIGN: COMPLETE

**DEVELOPMENT ONLY - NONFINAL.** Six development-exposed landscapes; n = 6 independent landscapes (terrain units are not independent replicates).
No lockbox, no MEGH, Gen-3 untouched, no new G features, no final model.

## A. Executive result
1. **Cotabato's near-chance transfer is tied to its frozen EQ1 label set, not to covariate shift.** The *same existing* Stage 4B held-out predictions that score ROC-AUC
   0.51 on the 190-polygon EQ1 inventory score 0.70 / 0.68 on the same mapper's cumulative post-EQ3 / post-EQ4 inventories (paired Delta ROC +0.19 [0.11, 0.27] and
   +0.16 [0.08, 0.25]). EQ1 positives sit on *gentler, less rough* terrain than their surroundings (univariate slope AUC 0.46 vs 0.70 in training). Frozen classification:
   **{c['PRIMARY_CLASS']}** (dominant: {c['DOMINANT_COMPONENT']}; secondary: {c['SECONDARY_CLASS']}; confidence **{c['CONFIDENCE']}** by the pre-registered tie rule).
2. **No alternative environmental representation advances.** Within-landscape percentile (E1), landscape-centred (E2) and hybrid (E3) representations remove the
   Atitlan extrapolation failure (units outside the training range 100% -> 0-0.8%; logistic flagged share 97% -> <0.1%) and keep all within-landscape rank information,
   but they still identify the volcano (balanced accuracy 0.85-0.93 vs 0.998) and **reduce transfer** relative to M in both families (median PR-AUC ratio 0.70-0.89;
   Atitlan and Noto lose most). All three fail pre-registered criteria A, B, C, D and F.
3. **Stage 4E-B should centre on M**, with absolute E (ME) and MEG kept only as historical controls.

## B. Governance and integrity
V1 freeze JSON {integ['v1_freeze_json_sha256']} (v1.0.0, FROZEN); verifier {integ['v1']['strict_files']} strict files / {integ['v1']['writes_performed']} writes.
Prior checkpoints 3F-4C-B3 unchanged: {integ['prior_unchanged']}; Stage 4D checkpoint {PRIOR4D[1]} unchanged: {integ['stage4d_checkpoint_ok']}; Stage 4D outputs unchanged:
{integ['stage4d_outputs_unchanged']}; {integ['historical_files_checked']} historical files re-hashed, unchanged: {integ['historical_files_unchanged']}.
Stage 4B cache untouched ({integ['stage4b_cache_files']} files); new fits used data/interim/_stage4ea_cache.

## C. Stage 4D starting evidence
Read from outputs/stage4d (RF+MEG robust in 5/6; Cotabato ROC 0.51, lift 1.16; Noto lift 4.9; Atitlan most shifted, Lombok least; E identity 0.998; E harms Atitlan;
LR+M median lift 2.57; calibration does not transfer). No winner had been selected.

## D. Cotabato provenance audit
{md(prov, ['landscape', 'trigger_class', 'inventory_geometry_type', 'number_original_features', 'mapping_resolution_or_scale', 'post_event_imagery', 'confidence_in_metadata'])}

Missing metadata is written NOT_IN_PROJECT_RECORDS (e.g. completeness statements for 5/6 inventories, Kumamoto/Noto magnitudes). Cotabato: Ferrario et al. (Zenodo 7520726),
manual mapping on 3 m PlanetScope; EQ1 (Mw 6.4, 16 Oct 2019) imagery 24 and 28 Oct, before EQ2.

## E. Label-geometry audit
{md(geo, ['landscape', 'inventory_type', 'prevalence', 'components', 'component_size_max', 'share_positives_in_components_ge10', 'positives_within_1km_p50', 'polygon_area_m2_p50', 'y_area_over_omega', 'polygons_per_km2_omega', 'units_hit_per_polygon_p50'])}

Cotabato has the sparsest polygon inventory (0.28 polygons/km2 of Omega vs 1.6-30.8; mapped area 0.09% of Omega vs 0.29-1.1%), no positive cluster of >= 10 units
(0% of positives vs 27-56% in the other polygon inventories), the fewest positives within 1 km (7), but the LARGEST median polygon (2,377 m2).

## F. Trigger / event audit
{md(trig, ['landscape', 'trigger_class', 'magnitude_or_intensity_in_records', 'epoch_definition', 'training_landscapes_sharing_trigger_when_held_out'])}

Cotabato is seismic like 3 of its 5 training landscapes; its EQ1 is the first and smallest (Mw 6.4) shock of a four-event sequence.

## G. Conditional morphology audit (P(M) vs P(Y | M))
{md(cond, ['predictor', 'cotabato_auc', 'pooled_training_auc', 'cotabato_abs_effect', 'others_median_abs_effect', 'reversed_vs_training', 'cot_pos_vs_train_pos_std_median_diff', 'cot_neg_vs_train_neg_std_median_diff'], 3)}

Cotabato's overall morphology is the least shifted (Stage 4D H_M 0.007), yet its EQ1 positives are gentler than training positives (slope -0.36 training-positive IQR) and its
negatives steeper than training negatives (+0.37): P(M) is similar, P(Y | M) is not. Binned event rates (fig. 3): training rate rises ~30x from the lowest to the highest slope
decile; Cotabato's is flat (0.7-1.3x).

Auxiliary-inventory test (same Stage 4B predictions, no refit):
{md(aux, ['model', 'feature_set', 'label_set', 'source_polygons', 'positive_units', 'prevalence', 'roc_auc', 'roc_ci_low', 'roc_ci_high', 'pr_lift', 'delta_roc_vs_eq1', 'delta_roc_ci_low', 'delta_roc_ci_high'], 3)}

## H. Cotabato failure classification (pre-registered rule, frozen before results)
{md(cls, ['item', 'satisfied', 'evidence'])}

Interpretation (not a change of class): T1/T2 (weak/reversed M -> Y) are measured on the EQ1 labels; the auxiliary test shows that with the same landscape, mapper and
imagery type but the cumulative inventories, the M -> Y association largely returns. The most economical reading is therefore that the EQ1 inventory is a sparse,
event-specific subset (first shock) whose landslides are not the terrain the other inventories label - an inventory/label-semantics difference that manifests as concept shift.
Cotabato is NOT excluded and is not called bad data.

## I. Environmental representation preregistration
STAGE4EA_E_REPRESENTATION_PREREGISTRATION.json/.md frozen after the Cotabato audit freeze and before any E1-E3 result (decision rule A-F, thresholds, max 2 advance).

## J. Candidate E representations
E0 absolute (control); E1 within-landscape percentile; E2 (x - median_L) / IQR_L; E3 = E2 for bio1, bio4, bio13, bio15, treecover SD + absolute log10(bio12) and absolute
treecover mean. E1-E3 use each landscape's unlabeled predictors (DEPLOYMENT-AVAILABLE X-ONLY LANDSCAPE NORMALIZATION). The strict training-reference alternative is an
affine/monotone transform of absolute E and reproduces E0 exactly for both families (reported analytically). All 7 variables passed the X-only validity check.

## K. Landscape-identity audit (E only; Stage 4D architecture; chance 0.167)
{md(idn.drop(columns=['confusion_matrix']), ['representation', 'scope', 'balanced_accuracy', 'recall_KUMA', 'recall_NOTO', 'recall_LOMB', 'recall_COTA', 'recall_ATIT', 'recall_DOMI'], 3)}

## L. Environmental information retention
{md(ret, ['representation', 'column', 'min_distinct_values', 'max_largest_tie_share', 'min_within_landscape_spearman_with_absolute', 'between_landscape_variance_fraction', 'median_within_landscape_spearman_with_elevation', 'passes_rule_E'], 3)}

## M. Limited predictive screen (Stage 4B architecture; LOGISTIC_L2 and RANDOM_FOREST only)
{md(summ, ['model', 'feature_set', 'median_pr_auc', 'median_pr_lift', 'worst_pr_lift', 'sd_log_lift', 'median_cal_slope'], 3)}

PR-AUC by landscape - LOGISTIC_L2:
{md(piv('LOGISTIC_L2', 'pr_auc'), ['feature_set'] + list(SH), 3)}

RANDOM_FOREST:
{md(piv('RANDOM_FOREST', 'pr_auc'), ['feature_set'] + list(SH), 3)}

## N. Atitlan extrapolation test
{md(ar[ar.column == 'ANY_COLUMN'], ['representation', 'atitlan_fraction_outside_training_range'], 3)}

{md(at, ['model', 'feature_set', 'pr_auc', 'flagged_fraction', 'cal_slope'], 3)}

## O. Advancement decision (pre-registered A-F, both families)
{md(dec, ['feature_set', 'identity_balanced_accuracy', 'LOGISTIC_L2_median_ratio', 'LOGISTIC_L2_min_ratio', 'RANDOM_FOREST_median_ratio', 'RANDOM_FOREST_min_ratio', 'D', 'E', 'passes_all', 'advance_to_stage4e_b'], 3)}

**Nothing advances.** E0 remains a historical control.

## P. G status
Existing two G variables (g_fragmental_volcanic, g_age_quaternary) preserved as a control (MEG); no new geology features; no G data error identified in this stage.

## Q. H status
{md(hs, ['landscape', 'gen3_status', 'H_AVAILABLE', 'CORE_H_AVAILABLE'])}

N_VERIFIED_H_CAPABLE = 1; MEGH_EXPERIMENT_UNLOCKED = NO; no Gen-4.

## R. Limitations
- n = 6 landscapes; one Cotabato inventory family; the auxiliary test reuses one landscape's later inventories (which include EQ1 landslides and later shocks).
- Inventory completeness is unreported for 5/6 inventories; trigger intensities (shaking, rainfall) are not in project records.
- CHELSA relative variables are coarse and largely orographic; relative bio1 is almost elevation (within-landscape rho -0.98), so it duplicates M.
- The E screen used two model families only; calibration was not addressed.

## S. Stage 4E-B recommendations (no model chosen)
- Permitted feature sets: **M** (primary); **ME (= M + absolute E0)** and **MEG** as historical controls only; MG may be reported as a control from Stage 4D. No E1/E2/E3.
- Model families: the Stage 4B families (LOGISTIC_L2, RANDOM_FOREST, HIST_GRADIENT_BOOSTING); any additional "stronger" family must be pre-registered with the same LOLO
  architecture, sampling, weighting and a bounded grid.
- Pre-register before 4E-B: the selection rule (median and worst-landscape lift, not maximum median PR-AUC), how Cotabato is scored (frozen EQ1 target remains primary; the
  cumulative inventories may only be a pre-declared sensitivity), the calibration policy, and the bootstrap/uncertainty method.

## T. Reproducibility / checkpoint
{tests}
{repro}
Checkpoint: STAGE4EA_CHECKPOINT.json (its own hash is reported in the handover). Git: no commit, no push.

## U. Answers 1-46
1. No - Cotabato is the least M-shifted landscape (H_M 0.007 vs others' median 0.023).
2. Yes - EQ1 positives are gentler and less rough than training positives (slope -0.36, roughness -0.40 training-positive IQR), while its negatives are steeper than training negatives.
3. Yes - median |AUC - 0.5| over the audited M predictors 0.031 vs 0.18 elsewhere.
4. Weakly: 5 predictors (slope 30/90 m, roughness 270 m, elevation SD, relief 250 m) have AUC 0.46-0.47 against 0.69-0.71 in training.
5. Yes - sparsest polygon inventory (0.28 polygons/km2 Omega; 0.09% mapped area), no clusters >= 10 units, but the largest median polygon.
6. Not alone: prevalence 0.38% is low, but Kumamoto (0.19%) transfers with lift ~4.7; and the same Cotabato area with cumulative labels (prevalence 2.6-3.1%) is ranked well.
7. No - seismic, shared with 3 of the 5 training landscapes; it is however the first and smallest shock of a sequence.
8. No direct evidence: Omega is the source's EQ1 study area (99.4% of mapped area inside); completeness is not reported, so incompleteness cannot be ruled out.
9. Yes, plausibly: EQ1 is an event-specific sparse subset; the same mapper's cumulative inventories carry the expected M -> Y association.
10. {c['PRIMARY_CLASS']} (dominant {c['DOMINANT_COMPONENT']}, secondary {c['SECONDARY_CLASS']}).
11. {c['CONFIDENCE']} (pre-registered rule: label/inventory and trigger/concept each satisfy 2 criteria).
12. Yes - E0 E-only balanced accuracy {ids['E0_ABSOLUTE']:.3f}.
13. E1 within-landscape percentile ({ids['E1_WITHIN_LANDSCAPE_PERCENTILE']:.3f}).
14. Useful normalisation, not destruction (all columns non-constant, within-landscape rank correlation 1.0) - but the joint pattern of the seven relative variables still identifies the volcano.
15. E3 (keeps all within-landscape ranks plus two absolute context variables); E1 and E2 keep identical within-landscape rank information.
16. Yes - all seven pass the validity check in all six landscapes.
17. None fail the pre-registered rule; the coarsest are bio1 (34 distinct values at Noto) and bio15 (45 at Dominica), and relative bio1/bio4 are largely elevation proxies.
18. Yes - IQR > 0 for all seven in all six landscapes.
19. Yes for the failure mode: out-of-range units 100% -> 0-0.8% and logistic flagged share 97% -> 0.01-0.03% at Atitlan.
20. Random forest only (0.092 -> 0.095-0.106); logistic regression no (0.183 -> 0.083-0.086).
21. No - both families stay below M at Atitlan (LR 0.231, RF 0.181).
22. Partly: logistic E2/E3 roughly double Kumamoto PR-AUC (ratio 2.2 / 2.0, E0 2.4); random forest does not (0.91-1.06).
23. No.
24. No (median PR-AUC ratio vs M 0.70-0.89).
25. No (worst-landscape lift below M in both families).
26. Not consistently: with random forest, E2/E3 are slightly less variable across landscapes (SD of log lift 0.50 / 0.48 vs 0.53 for M), but only at lower median and worst-landscape lift; with logistic regression variability increases (0.54-0.78 vs 0.50).
27. No (median lift falls in both families).
28. Qualitatively yes: both families reject every alternative; the Kumamoto gain appears only in logistic regression.
29. The only gains (Kumamoto, logistic) are single-landscape; the losses are dominated by Atitlan and Noto.
30. None.
31. None.
32. Yes - absolute E remains a historical control only.
33. Yes as a control (MEG continuity), not as a primary predictor set.
34. No.
35. No.
36. No.
37. No.
38. No.
39. No.
40. Yes.
41. See STAGE4EA_CHECKPOINT.json (hash in the handover).
42. (i) Cotabato's failure tracks its EQ1 label set: the same predictions rank its cumulative inventories well (ROC 0.70); (ii) relative E removes extrapolation but not volcano identity and lowers transfer; (iii) P(M) similarity does not imply P(Y | M) similarity.
43. (i) n = 6 and a single Cotabato inventory family; (ii) unreported completeness and trigger intensity; (iii) coarse, orographic CHELSA fields limit any E representation.
44. M (primary); ME and MEG as historical controls (MG optional control). No E1/E2/E3.
45. LOGISTIC_L2, RANDOM_FOREST, HIST_GRADIENT_BOOSTING under the Stage 4B architecture; any extra family only if pre-registered.
46. Selection rule (median + worst-landscape lift, stability), primary/secondary Cotabato targets, calibration policy, uncertainty method, model families and grids, and seeds.
"""


def main(tests, repro):
    integ = integrity()
    (OUT / "STAGE4EA_FINAL_REPORT.md").write_text(report(tests, repro, integ))
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "STAGE4EA_CHECKPOINT.json")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    import scipy
    import sklearn
    cp = dict(stage="4E-A COTABATO FAILURE AUDIT + IDENTITY-ROBUST ENVIRONMENTAL REPRESENTATION DESIGN", status="DEVELOPMENT_ONLY_NONFINAL",
              completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              input_checkpoints=dict(integ["prior_checkpoints"], **{PRIOR4D[0]: sha256(ROOT / PRIOR4D[0])}),
              preregistration=dict(cotabato_audit=sha256(OUT / "STAGE4EA_COTABATO_AUDIT_PREREGISTRATION.json"), cotabato_audit_freeze=sha256(OUT / "STAGE4EA_COTABATO_AUDIT_FREEZE.json"),
                                   e_representation_json=sha256(OUT / "STAGE4EA_E_REPRESENTATION_PREREGISTRATION.json"), e_representation_md=sha256(OUT / "STAGE4EA_E_REPRESENTATION_PREREGISTRATION.md")),
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("stage4ea_cotabato_prereg.py", "stage4ea_audit.py", "stage4ea_e_prereg.py", "stage4ea_screen.py",
                                                                      "stage4ea_eaudit.py", "build_stage4ea.py", "build_stage4ea_final.py", "stage4b.py", "stage4d.py")},
              screen_results=dict(path="data/interim/_stage4ea_screen/E_SCREEN_RESULTS.json", sha256=sha256(ROOT / "data/interim/_stage4ea_screen/E_SCREEN_RESULTS.json")),
              seeds=dict(stage4ea=20261008, stage4b=20261006, bootstrap="20261008 + landscape index", identity="20261007 (Stage 4D architecture)"),
              software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
              tests=tests, reproducibility=repro, v1=integ["v1"], v1_freeze_json_sha256=integ["v1_freeze_json_sha256"],
              historical_output_integrity=dict(prior_unchanged=integ["prior_unchanged"], stage4d_unchanged=integ["stage4d_outputs_unchanged"],
                                               files_checked=integ["historical_files_checked"], unchanged=integ["historical_files_unchanged"], stage4b_cache_files=integ["stage4b_cache_files"]),
              git=dict(head=git("rev-parse", "HEAD"), tracked_modified=len([s for s in git("status", "--porcelain").splitlines() if not s.startswith("??")]), committed=False, pushed=False),
              LOCKBOX_TOUCHED="NO", MEGH_TRAINED="NO", GEN3_MODIFIED="NO", H_GATE_MODIFIED="NO", NEW_G_FEATURES_ENGINEERED="NO", FINAL_MODEL_SELECTED="NO",
              E_REPRESENTATIONS_ADVANCED=[], N_VERIFIED_H_CAPABLE=1, MEGH_EXPERIMENT_UNLOCKED="NO",
              output_manifest=[str(p.relative_to(ROOT)) for p in outs], output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs})
    (OUT / "STAGE4EA_CHECKPOINT.json").write_text(json.dumps(cp, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
