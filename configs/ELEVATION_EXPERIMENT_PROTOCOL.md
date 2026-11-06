# Elevation Ablation Experiment — Protocol

**Version:** `ELEV_ABLATION_PROTOCOL_v1.0.0`
**Written:** 2026-10-17 (UTC), before any model fitting for this experiment
**Status:** FROZEN once `PROTOCOL_MANIFEST.json` records its SHA-256. Corrections are allowed only as timestamped amendment files `AMENDMENT_vX.Y.md`. This file is never edited.
**Classification:** NEW exploratory development experiment. It does not replace the official model (LOGISTIC_L2 + M), and none of its results replaces any frozen value.

Paths are relative to `Science_Fair_ML_V2/`.

---

## 1. Research question

Does the raw absolute-elevation predictor contribute to whole-landscape geographical transfer in the frozen six-landscape LOLO design? Or does it partly encode landscape-specific information that reduces generalisation?

## 2. Hypotheses (prospective for this experiment only)

- **H1.** Removing raw absolute elevation (A1) raises the robustness score R relative to B0, if landscape-identity information in elevation produces unstable cross-landscape relationships.
- **H2.** Landscape-centred elevation (A2) raises R relative to B0, if relative vertical position within a landscape transfers better than absolute altitude.
- **H3.** Per-landscape changes are heterogeneous in sign. Some held-out landscapes improve and others worsen.

These hypotheses were not part of the original frozen investigation. Stage A audit observations motivated them (`AUSSEF_2026_VALIDITY_AUDIT/`). Those observations do not establish causation.

## 3. Dataset scope

- **Data:** the frozen Stage 4A MEG dataset, `outputs/stage4a/STAGE4A_MEG_DATASET/{LANDSCAPE}.npz`. It contains six landscapes, 835,846 terrain units and 8,793 positives.
- **Population:** `ALL_VALID` (every row). Labels are `y_positive` as frozen.
- **Landscape order** (used for folds and bootstrap seeds): KUMAMOTO_ASO_2016, NOTO_PENINSULA_2024, LOMBOK_RINJANI_2018, COTABATO_DAVAO_2019, ATITLAN_STAN_2005, DOMINICA_ERIKA_2015. This is `src/sfv2/stage4b.py:LANDSCAPES`.
- **No change** to labels, terrain units, Ω, folds or source data. No new data is acquired.

## 4. Feature definitions

The authoritative M list is the 60 columns with `role == "M"` and `predictor == True` in `outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv`, in manifest order (`stage4b.feature_sets()["M"]`).

| Config | Columns | Definition |
|---|---|---|
| **B0** | 60 | Exact original M. Fitted under feature-set name `M`, so the fold cache keys equal the frozen Stage 4B keys. |
| **A1** | 59 | B0 minus `elev_m_30m__mean` only. Every other column is retained, including `elev_m_30m__sd`. |
| **A2** | 60 | B0 with `elev_m_30m__mean` replaced, at the same position, by `elev_m_30m__mean__landscape_centred`. |

**A2 transformation.** For landscape L:

`elevation_centred = elev_m_30m__mean − mean_L`

`mean_L` is the unweighted arithmetic mean of `elev_m_30m__mean` over **all** ALL_VALID rows of L in the frozen NPZ. It is computed once per landscape and applied to training and held-out landscapes alike. It uses no labels, no prevalence and no performance. This is not a global centring across all six landscapes.

**Assumption, recorded explicitly:** A2 assumes the full eligible DEM-derived terrain domain of the target landscape is available before prediction.

**Derivative audit (to be verified in Phase 1 output).** Raw absolute elevation enters M only through `elev_m_30m__mean`. Every other M column is computed from the same DEM but is invariant to adding a constant to elevation:
- `elev_m_30m__sd`
- slope, aspect, flat flag, curvatures
- roughness, TRI, TPI, relief, relative elevation (r1000)
- flow accumulation, SCA, contributing area
- distance to drainage and ridge
- valley depth, relative slope position
- TWI, SPI

They are **not removed** in A1 or A2.

**Limitation:** this experiment isolates the contribution of the raw elevation feature, not all information correlated with elevation. For example, relief and slope covary with altitude.

## 5. Preprocessing

Unchanged frozen code, `src/sfv2/stage4b.py:fit/predict`:
- z-score of every column using mean and SD of the pooled **training sample only**;
- held-out rows are transformed with the training parameters;
- no imputation (missingness is 0).

The A2 centring happens before this step. It is an unsupervised per-landscape geographical transformation, not a learned preprocessing step.

## 6. Folds and model

- **Outer folds:** leave one whole landscape out (6 folds, unchanged).
- **Inner folds:** leave one of the five training landscapes out.
- **Model:** `sklearn.linear_model.LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)`, scikit-learn 1.7.2.
- **Grid:** C ∈ {0.01, 0.1, 1.0}. Select by maximum mean inner AP; values within 0.005 resolve to the earliest grid entry.
- **Training sample:** all positives plus ≤20,000 negatives per training landscape, ordered by ascending `sha256(terrain_unit_id + "|stage4b")`.
- **Weights:** LANDSCAPE_BALANCED, sampling-corrected, rescaled to mean 1.
- **Calibration and threshold:** Platt and F1 threshold from inner OOF predictions. These are reported only and do not affect AP.
- **Execution:** the identical frozen code path `stage4b.run_analysis`, called with analysis definitions added **in memory only**: universe = six landscapes, `ALL_VALID`, `LANDSCAPE_BALANCED`, one feature set, model `LOGISTIC_L2`. No frozen file is edited. The fit cache is redirected to a temporary directory outside the frozen `data/interim/_stage4b_cache`.

## 7. Seeds

- LogisticRegression/lbfgs is deterministic. The Stage 4B seed 20261006 is recorded for completeness.
- Bootstrap: `numpy.random.default_rng(20261008 + i)`, where i is the landscape index in §3. This is the Stage 4E-B rule.

## 8. Metrics

Per held-out landscape:
- **AP** = `average_precision_score(y, raw score)`;
- **prevalence** = mean(y);
- **PR lift** = AP / prevalence;
- **ROC-AUC**.

Across the six folds:
- median AP;
- median PR lift;
- minimum PR lift;
- robustness score **R = 0.70·ln(median PR lift) + 0.30·ln(minimum PR lift)** (the original formula).

**Primary comparison:** ΔR = R(A1) − R(B0) and ΔR = R(A2) − R(B0), each reported separately.

**Secondary comparisons:**
- Δ median PR lift; Δ minimum PR lift; Δ median AP;
- per-landscape ΔAP and Δ PR lift; ROC-AUC;
- number of folds improved and worsened.

## 9. Comparison rules (fixed now)

- A **higher R means improvement under this criterion only**, not universal superiority.
- **Bands for ΔR:**
  - |ΔR| ≤ 1e-9: IDENTICAL;
  - |ΔR| < 0.02: NEGLIGIBLE;
  - ΔR ≥ 0.02: HIGHER_R;
  - ΔR ≤ −0.02: LOWER_R.
  - (0.02 in R corresponds to roughly a 2% multiplicative change in lift.)
- **Per-landscape classification:**
  - IMPROVED if Δlift / lift_B0 ≥ +0.05;
  - WORSENED if ≤ −0.05;
  - otherwise UNCHANGED_WITHIN_5PCT.
  - The raw sign counts (ΔAP > 0, < 0) are also reported.
- H1/H2 are described as "supported under R" only if ΔR is HIGHER_R. They are additionally described as "robust to single-landscape removal" only if ΔR stays > 0 in all six leave-one-landscape-out recomputations (§10b).
- H3 is supported if at least one fold is IMPROVED and at least one is WORSENED for that configuration.
- No configuration is re-tuned, and no alternative centring or normalisation is tried after results are seen.

## 10. Uncertainty

**a. Within-landscape sampling uncertainty.** A paired 3 km spatial-block bootstrap using the frozen `spatial_block_3000m`:
- 2,000 replicates;
- identical block draws for B0/A1/A2 within each landscape;
- weighted AP and ROC via `src/sfv2/stage4d.py:Sorted`;
- replicates with zero positive weight are dropped and counted.

Reported per landscape:
- 95% percentile intervals of AP, PR lift (AP_rep / prev_rep), ΔAP and Δlift.

Reported across landscapes:
- ΔR per replicate, combining replicate r of every landscape. The draws are independent across landscapes.
- Its 95% interval describes within-landscape sampling uncertainty conditional on the six fitted fold models. It does not include model refitting or between-landscape variation.

**b. Between-landscape heterogeneity.** Descriptive only:
- ΔR recomputed on each 5-landscape subset (leave-one-landscape-out);
- per-landscape changes;
- explicit Atitlán-excluded summary.

**c. Model-selection uncertainty.** Not quantifiable here. B0 was itself selected after outer-fold results were seen, so B0's R is a post-selection development value. A1 and A2 are evaluated once with no selection among variants. No claim of statistical significance at the population level is made (n = 6 landscapes).

## 11. Baseline reproduction gate (Phase 3)

B0 must reproduce the frozen Stage 4B PRIMARY LOGISTIC_L2 + M results:

| Check | Requirement | Reference |
|---|---|---|
| Per-fold AP | \|Δ\| ≤ 1e-12 | `data/interim/_stage4b_results/PRIMARY.json`; `outputs/stage5a/STAGE5A_MODEL_CONTRACT.json` |
| Outer score vectors | max\|Δ\| ≤ 1e-12 per fold; SHA-256 equality reported | frozen `data/interim/_stage4b_cache/{key}__{held}.npy`; hashes in `outputs/stage5d_e/STAGE5DE_BASELINE_REPRODUCTION.json` |
| Selected C per fold | equal (all 0.01) | PRIMARY.json |
| n, positives per fold | equal | PRIMARY.json |
| Fold membership | held-out = one landscape; training = other five | — |
| Median AP, median lift, R | \|Δ\| ≤ 5e-11 (contract reporting tolerance) | contract values 0.0285931177553 / 2.56665260507 / 0.669162938062 |
| Totals | 835,846 predictions, 8,793 positives | — |

**If any check fails: STOP.** Write `02_RESULTS/BASELINE_REPRODUCTION_DIAGNOSTIC.md`, and do not run A1 or A2. A newly trained baseline is never substituted for the authoritative one.

## 12. Exclusion rules

None. All six folds and all ALL_VALID rows are used. ConvergenceWarnings are suppressed exactly as in the frozen code, and `n_iter_` is not altered. A fold whose fit raises an exception aborts the experiment, and the failure is reported.

## 13. Known limitations

- Six development-exposed landscapes.
- B0 is a post-selection estimate.
- A1 and A2 isolate only the raw elevation column; correlated predictors may still carry altitude information.
- A2 requires the full target-landscape DEM domain.
- The bootstrap is within-landscape only.
- Logistic linearity: under a linear model, centring changes only the fitted coefficients (estimated without between-landscape contrast). It does not change within-landscape ranking information.
- Results do not establish any physical or causal role of elevation and are not external validation.

## 14. Expected outputs

Under `AUSSEF_2026_ELEVATION_EXPERIMENT/`:

- `01_PROTOCOL/`: this file, `PROTOCOL_MANIFEST.json`, any `AMENDMENT_*.md`.
- `02_RESULTS/`:
  - `BASELINE_REPRODUCTION_AUDIT.csv`
  - `ELEVATION_ABLATION_FOLD_RESULTS.csv`
  - `ELEVATION_ABLATION_SUMMARY.csv`
  - `PAIRED_PERFORMANCE_DIFFERENCES.csv`
  - `ROBUSTNESS_COMPARISON.csv`
  - `FEATURE_DERIVATIVE_AUDIT.csv`
  - `predictions/{CONFIG}__{LANDSCAPE}.npy` (raw scores in NPZ row order)
- `03_FIGURES/`:
  - `F1_LANDSCAPE_PR_LIFT.{png,pdf}`
  - `F2_PAIRED_AP_DIFFERENCES.{png,pdf}`
  - `F3_ROBUSTNESS_SCORES.{png,pdf}`
  - `F4_ELEVATION_DISTRIBUTIONS.{png,pdf}`
- `04_INTERPRETATION/`: `SCIENTIFIC_FINDINGS.md`, `LIMITATIONS.md`, `AUSSEF_REPORT_IMPLICATIONS.md`.
- `05_REPRODUCIBILITY/`:
  - `code/`
  - `CODE_AND_CONFIG_MANIFEST.tsv`
  - `INPUT_PROVENANCE_MANIFEST.tsv`
  - `OUTPUT_SHA256_MANIFEST.tsv`
  - `EXECUTION_LOG.txt`
- `README.md`.
