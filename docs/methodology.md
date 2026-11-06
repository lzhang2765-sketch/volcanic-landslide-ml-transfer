# Methodology

## Scientific question

The study evaluates whether models trained on several volcanic landscapes can rank landslide-affected terrain in a geographically withheld volcanic landscape. This is geographical transfer of susceptibility ranking, not prediction of event timing or loss.

## Terrain units and labels

Each landscape was represented by a governed 90 m lattice derived from 30 m terrain inputs. The primary population was `ALL_VALID`. The six inventories do not have identical semantics: Kumamoto used affected-area centre points, while the other five used affected-area polygons. A negative target is an operational non-positive under the applicable mapping rule and observation domain, not proof of stable terrain.

## Predictors

The selected feature set `M` contains 60 morphology predictors, including elevation summaries, slope and aspect summaries at multiple scales, curvature, roughness, terrain-position, relief, hydrological accumulation and drainage/ridge-distance measures. The exact archived definitions are in `results/tables/STAGE5A_FEATURE_DICTIONARY.tsv`.

Environmental and geological candidates were investigated during development, but they did not improve the frozen robustness criterion enough to replace `M`. Reconstructed geological-history features were not sufficiently available across landscapes to support a complete predictive comparison.

## Nested leave-one-landscape-out evaluation

For each of six outer folds:

1. One complete landscape was withheld.
2. The other five landscapes supplied training data.
3. Inner leave-one-training-landscape-out folds selected hyperparameters.
4. Scaling, deterministic negative sampling, landscape balancing, calibration diagnostics and model fitting used training data only.
5. Every valid unit in the outer landscape received a score.

The four compared families were L2 logistic regression, random forest, histogram gradient boosting and XGBoost. The exact grids, seeds and fixed arguments are in `configs/STAGE4EB_MODEL_CONFIGURATIONS.tsv` and the frozen preregistration files.

## Metrics

- **Average precision (AP):** a prevalence-sensitive ranking metric for rare positive labels.
- **Precision–recall lift:** `AP / prevalence`; one is the random-ranking baseline for the observed label prevalence.
- **ROC-AUC:** supporting ranking metric.
- **Robustness score:** `0.70 × ln(median lift) + 0.30 × ln(minimum lift)` across the six outer folds.

Calibration and thresholds were diagnostic and did not select the ranking architecture. Raw susceptibility scores should not be interpreted as calibrated landslide probabilities.

## Uncertainty

The primary uncertainty analysis used 2,000 paired resamples of 3 km spatial blocks. This keeps neighbouring units together and measures sensitivity to within-landscape spatial arrangement, conditional on the fitted models and recorded inventories. It does not estimate uncertainty across the population of possible future volcanic landscapes.

## Sensitivity and feasibility analyses

- **Training composition:** post hoc comparisons using frozen alternative training compositions.
- **Elevation representation:** a separately frozen exploratory protocol comparing the 60-feature baseline with removal of mean absolute elevation and within-landscape centring.
- **Cotabato label definition:** post-selection scoring against cumulative later-event inventories; EQ1 remained primary and the sensitivity could not select a model.
- **Geological history:** a feasibility audit of reconstruction support and validation gates. Only a partial Kumamoto reconstruction passed. No V2 `M+E+G+H` predictive model was completed.
