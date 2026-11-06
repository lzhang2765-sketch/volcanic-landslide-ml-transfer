# Transferability-Factor Analysis Plan — POST HOC EXPLORATORY ANALYSIS

**Version:** `TRANSFER_FACTOR_PLAN_v1.0.0`
**Status:** frozen by SHA-256 in `PROTOCOL_MANIFEST.json` before any new calculation. Corrections are allowed only as timestamped `AMENDMENT_vX.Y.md` files; this file is never edited.

**Nature of the analysis:**
- Post hoc exploratory and descriptive.
- Six landscape-level observations.
- No model is retrained.
- No causal or population-level claim is made.
- The official model (LOGISTIC_L2 + M) and its post-selection development results are unchanged.

Phase 1 already exists: `SOURCE_EVIDENCE_REGISTER.csv` and `SOURCE_CONSISTENCY_CHECKS.csv`. Every numeric value agrees across sources. Two categorical climate labels differ between the Stage-1 screening record and the production-stage Stage 3G record. **Stage 3G is used**: it is later, production-stage and complete for all six landscapes. The discrepancy is reported.

Paths are relative to `Science_Fair_ML_V2/`.

## 1. Aim

Describe which measurable inventory, morphological-shift, training-composition and context characteristics co-vary with whole-landscape transfer performance of the frozen baseline across the six LOLO folds.

## 2. Responses

| Role | Variable | Definition | Source |
|---|---|---|---|
| **PRIMARY** | `pr_lift_B0` | AP / prevalence of the held-out landscape, frozen baseline | `STAGE5A_MODEL_CONTRACT.json` (`pr_lift`) |
| Secondary | `ap_B0` | Average precision | same |
| Secondary | `roc_auc_B0` | ROC-AUC | same |
| Secondary | `rel_dlift_A1`, `rel_dlift_A2` | (lift_A − lift_B0) / lift_B0 | Stage A.1 `PAIRED_PERFORMANCE_DIFFERENCES.csv` |
| Secondary | `rel_dlift_ATITLAN_EXCLUDED` | Same landscape and labels, Atitlán removed from training (5 landscapes) | `data/interim/_stage4b_results/ATITLAN_EXCLUDED.json` vs PRIMARY |
| Secondary | `rel_dlift_KUMAMOTO_EXCLUDED` | Same landscape and labels, Kumamoto removed from training (5 polygon landscapes; frozen name `POLYGON_ONLY`) | `POLYGON_ONLY.json` vs PRIMARY |

Per-landscape uncertainty for `pr_lift_B0`: frozen 95% 3 km block-bootstrap intervals (`STAGE4EB_SPATIAL_BOOTSTRAP.tsv`). These are within-landscape sampling only.

## 3. Candidate explanatory factors (prespecified)

### Group 1 — Inventory characteristics

| Factor | Definition / units | Use |
|---|---|---|
| `prevalence` | Positives / units (fraction) | PRIMARY association |
| `n_positive` | Positive units (count) | PRIMARY association |
| `n_units` | Evaluated terrain units (count) | PRIMARY association |
| `geometry` | POINT_CENTRE or POLYGON | Categorical; tabulated only (1 vs 5) |
| `label_meaning` | From the Stage A inventory audit | Categorical; tabulated only |
| `min_mapping_unit_m2` | Documented for 3 landscapes only | Tabulated only (n = 3); no association |
| `completeness_indicator` | Documented quantitative completeness (Atitlán only; Lombok quoted) | Tabulated only |
| `overlap_negative_fraction` | Share of polygon-overlapping units labelled negative (Stage A E02) | Secondary descriptive; n = 5 (NA for Kumamoto) |

### Group 2 — Morphological distribution shift (label-free)

**Features:**
- `elev_m_30m__mean` (absolute elevation, m)
- `slope_deg_30m__mean` (°)
- `relief_m_r1000__mean` (m)
- `rel_elev_local_r1000__mean` (0–1)

**Training pool for held-out L:** all units of the other five landscapes. Each training landscape has equal total weight (1/5, spread evenly over its units), mirroring the landscape-balanced training. Only predictor values are used: **no labels from any landscape**, and no outcome statistic. The target landscape's own predictor distribution is used (unlabelled target-domain information).

| Statistic | Status | Definition |
|---|---|---|
| **\|SMD\|** | **PRIMARY** | \|mean_target − mean_pool\| / sd_pool (weighted pool) |
| `W1_sd` | Secondary | 1-Wasserstein distance between the target and the weighted pool, divided by sd_pool |
| `OOR` | Secondary | Fraction of target units outside the pool's weighted 1st–99th percentile |
| `KS` | Secondary | Weighted two-sample Kolmogorov–Smirnov distance (descriptive; no test) |
| `M_oor_count` | Multivariate summary | Number of the 60 M features with OOR > 0.10 |
| `M_mean_abs_smd` | Multivariate summary | Mean \|SMD\| over the 60 M features |

Also reported, without recalculation: the existing Stage 4A domain-shift scores, all / M, for comparison.

### Group 3 — Training-composition dependence

Existing frozen results only:
- ATITLAN_EXCLUDED vs PRIMARY on the 5 common landscapes;
- KUMAMOTO_EXCLUDED (`POLYGON_ONLY`) vs PRIMARY on the 5 common polygon landscapes;
- Stage A.1 A1/A2 vs B0 on all 6.

STRICT_DEM_QUALITY changes the evaluated *population*. It is reported separately and is not treated as a composition comparison. ROW_WEIGHTING has no LOGISTIC_L2 + M result and is excluded. Medians are compared only on identical landscape sets.

### Group 4 — Context (categorical, Stage 3G metadata only)

- `tectonic_setting`
- `morphology` (volcanic landform)
- `erosion_state`
- `climate`
- `trigger`
- `target_mapping_method` (observation type)

These are tabulated only. No new geological classification is created. Trigger (4 earthquake vs 2 cyclone) is summarised by group medians, labelled as confounded with landscape identity, with no claim.

## 4. Methods

1. **Association measure:** Spearman ρ between each PRIMARY factor and `pr_lift_B0` (n = 6), with the exact value reported.
   - An exact permutation p-value (all 720 permutations) is reported as *descriptive only, not confirmatory*. For orientation: with n = 6, |ρ| ≥ 0.886 corresponds to two-sided permutation p ≈ 0.03 or less.
2. **Primary association set (6):** prevalence, n_positive, n_units, |SMD| elevation, |SMD| slope, |SMD| relief.
   - Secondary associations: the same factors against AP, ROC-AUC, rel_dlift_A1 and rel_dlift_A2; the secondary shift statistics against `pr_lift_B0`.
3. **Leave-one-landscape-out sensitivity:** recompute ρ on each of the six 5-landscape subsets.
4. **Stability classification** (applied in this order):
   - `NOT_ASSESSABLE` if fewer than 5 non-missing landscapes.
   - `WEAK` if |ρ_full| < 0.4.
   - `DRIVEN_BY_<L>` if removing landscape L flips the sign of ρ, or makes |ρ_LOO| < 0.2. All such L are listed.
   - `STABLE` if |ρ_full| ≥ 0.6, all LOO ρ share its sign, and every |ρ_LOO| ≥ 0.4.
   - Otherwise `MODERATE_UNSTABLE`.
5. **Not done:** no multivariable regression, no p-value correction claims, no causal language.
6. **Phase 6 questions** (existing results only):
   - **A.** Per-landscape Δ lift, ATITLAN_EXCLUDED vs PRIMARY (5 landscapes).
   - **B.** Per-landscape Δ lift, KUMAMOTO_EXCLUDED vs PRIMARY (5 polygon landscapes); both medians on that identical set.
   - **C.** Spearman of \|SMD\| elevation vs rel_dlift_A1 and rel_dlift_A2.
   - **D.** Spearman of `pr_lift_B0` vs rel_dlift_A1 and rel_dlift_A2.
   - **E.** Counts of landscapes helped and harmed by absolute elevation (from A1/A2).
   - **F.** Fold coefficients (B0 vs A2) and within-landscape elevation AUC from Stage A/A.1, used to classify the SHAP elevation ranking as within- vs between-landscape.
7. **Inventory heterogeneity (Phase 7):** each inventory difference is classified as VERIFIED_DIFFERENCE, DEMONSTRATED_EFFECT (only where an existing sensitivity analysis changes the label set on the same landscape), PLAUSIBLE_IMPLICATION or UNRESOLVED.
   - Demonstrated-effect sources: Cotabato cumulative-label sensitivity (`STAGE4EB_COTABATO_SENSITIVITY.tsv`) and KUMAMOTO_EXCLUDED.

## 5. Missing data

Missing values stay NA and are never imputed. Every association reports its n. Factors documented for fewer than 5 landscapes are tabulated only.

## 6. Multiple-comparison limitation

Six primary associations, and more than 20 secondary ones, are computed on six observations. Several moderate or large |ρ| are expected by chance alone. No association is described as statistically established.

## 7. Interpretation rules (evidence classes)

| Class | Meaning |
|---|---|
| **DIRECTLY_DEMONSTRATED** | Shown by a paired comparison on identical landscapes and labels in existing frozen or Stage A.1 results |
| **DESCRIPTIVELY_ASSOCIATED** | \|ρ_full\| ≥ 0.6 and classified STABLE |
| **PLAUSIBLE_BUT_UNTESTED** | Mechanism consistent with verified differences, but no direct comparison exists |
| **NOT_SUPPORTED** | WEAK, or contradicted by a paired comparison |
| **INDETERMINATE** | DRIVEN_BY a single landscape, MODERATE_UNSTABLE, or NOT_ASSESSABLE |

## 8. Sensitivity analyses

- LOO for every association;
- Atitlán-excluded summaries reported explicitly;
- secondary shift statistics (W1, OOR, KS) checked against the primary \|SMD\| for consistency of ranking.

## 9. Expected outputs

- `02_DATA/`:
  - `LANDSCAPE_FACTOR_MASTER.csv`, `LANDSCAPE_FACTOR_DICTIONARY.csv`
  - `DISTRIBUTION_SHIFT_RESULTS.csv` (all 60 M features × 6 folds)
  - `FACTOR_ASSOCIATIONS.csv`, `ASSOCIATION_SENSITIVITY.csv`
  - `TRAINING_COMPOSITION_COMPARISONS.csv`, `INVENTORY_HETEROGENEITY_SUMMARY.csv`
- `03_FIGURES/`: F1–F6 (PDF, SVG, PNG) plus a source CSV for each.
- `04_INTERPRETATION/`:
  - `TRANSFERABILITY_FACTOR_FINDINGS.md`, `EVIDENCE_STRENGTH_REGISTER.csv`
  - `RESEARCH_QUESTION_ANSWER.md`, `AUSSEF_REPORT_INTEGRATION.md`, `LIMITATIONS.md`
- `05_REPRODUCIBILITY/`: code, source and output manifests, execution log, README.
