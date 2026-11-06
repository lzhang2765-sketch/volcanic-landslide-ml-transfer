# Stage 4E-B preregistered cross-landscape model selection and transfer robustness

**Status:** FROZEN BEFORE ANY STAGE 4E-B MODEL FIT  
**Frozen UTC:** 2026-10-07T21:24:48Z  
**Primary representation:** M (the frozen 60-column Stage 4A morphology set)  
**Independent transfer sample:** six development landscapes, not 835,846 independent landscapes

## Question

Does additional model complexity improve whole-volcano transfer when the predictor representation is held fixed?

## Models and features

Primary candidates are Logistic L2, Random Forest, HistGradientBoosting and XGBoost, all with M. The first three reuse authoritative Stage 4B predictions. XGBoost uses eight bounded configurations stored verbatim in the JSON and configuration table. ME and MEG are historical controls only; no E1/E2/E3 and no H.

## Validation and training

Outer validation leaves one whole landscape out. Inner validation leaves one of the five outer-training landscapes out. Every positive and at most 20,000 hash-selected negatives per training landscape are used with Stage 4B landscape-balanced, sampling-corrected weights. The outer landscape cannot affect tuning, sampling, weighting, thresholds or calibration.

## Metrics and uncertainty

Primary ranking is held-out PR-AUC interpreted as PR lift over each landscape's EQ1 prevalence. Summaries are landscape-level. A paired 3 km spatial-block bootstrap uses 2,000 replicates. Pooled terrain-unit performance is not the primary metric.

## Frozen selection rule

`ROBUST_SCORE = 0.70 × log(MEDIAN_LIFT) + 0.30 × log(WORST_LIFT)`.

Candidates must pass the numerical catastrophic-failure guard in the JSON. The raw winner has the greatest score. Logistic is selected instead when the complex winner's score advantage is within 5% of the Logistic score, or when its median PR-AUC gain is no more than 0.002 and the paired-bootstrap interval for the median gain includes zero. Exact ties use Logistic < Random Forest < HistGradientBoosting < XGBoost.

## Cotabato, calibration and controls

Cotabato EQ1 remains primary and included. Cumulative EQ3/EQ4 labels are sensitivity-only and may be scored only after the primary decision is frozen. Calibration is reported but cannot select the ranking architecture. XGBoost ME/MEG are secondary historical-control fits performed only after the primary selection freeze.

## Interpretation and prohibited actions

Permutation importance and coefficient summaries cannot alter features. Ensemble work is a correlation/complementarity diagnostic only, with no weights or averaging. No future lockbox, MEGH, Gen-4, H-gate change, new predictor, post-result feature engineering or rule alteration is permitted.
