"""Frozen Stage 4F design: calibration, Y-blind novelty and transfer reliability.

This module is declarative.  It contains no Stage 4F result and is hashed into the
preregistration freeze before any new calibration or novelty score is computed.
"""
from __future__ import annotations

STAGE = "4F"
AS_OF = "2026-10-08"
SEED = 20261008
STAGE4B_SEED = 20261006
LANDSCAPES = [
    "KUMAMOTO_ASO_2016",
    "NOTO_PENINSULA_2024",
    "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019",
    "ATITLAN_STAN_2005",
    "DOMINICA_ERIKA_2015",
]

V1_FREEZE_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
STAGE4EB_CHECKPOINT_SHA256 = "d0d02641b1298262f6f68e80d48b73bb71016b6bbd6c8ee57eec0fd4b3ebb140"
INPUT_CHECKPOINTS = {
    "stage4d": "53041df090fb96edaeb6ee414dd49bd9a86a8e64d1683d8b46f567f37fa544fd",
    "stage4e_a": "e6e72d30fb7e7bb52e4cdede02809911eb0a63343f481c1f9b9acbcebccd2bf6",
    "stage4e_b": STAGE4EB_CHECKPOINT_SHA256,
}

MODEL = {
    "name": "LOGISTIC_M",
    "implementation": "sklearn LogisticRegression L2, lbfgs, max_iter=5000",
    "features": "exact frozen Stage 4A M columns (60); no E, G or H",
    "hyperparameter_grid": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "inner_selection": (
        "maximise unweighted mean inner-landscape PR-AUC; configurations within 0.005 "
        "of the maximum choose the earliest/strongest-regularisation entry"
    ),
    "sampling": (
        "per training landscape, all positives plus at most 20,000 negatives selected "
        "by ascending sha256(terrain_unit_id+'|stage4b')"
    ),
    "weighting": (
        "each training landscape has equal total weight; within landscape, sampled positive "
        "and negative weights reproduce the complete-domain prevalence; weights normalised to mean one"
    ),
    "preprocessing": "training-sample-only mean and SD z-score for all 60 M predictors",
    "outer_validation": "nested leave-one-landscape-out; complete held-out observation domain",
    "prohibition": "no architecture, feature, seed, weighting, sampling or preprocessing change",
}

CALIBRATION = {
    "candidates": {
        "C0_RAW": "unchanged LOGISTIC_M probability-scale output",
        "C1_INTERCEPT": "weighted logistic recalibration with logit(raw) offset and one fitted intercept",
        "C2_INTERCEPT_SLOPE": (
            "weighted logistic recalibration intercept plus non-negative slope on logit(raw); "
            "slope constrained to [0, 10] to preserve score order"
        ),
        "C3_ISOTONIC": (
            "weighted isotonic regression on raw score with increasing=True, out_of_bounds='clip'; "
            "chosen as the single flexible monotonic candidate"
        ),
    },
    "training": (
        "for each outer fold, fit each calibrator only to complete-domain honest inner-LOLO predictions "
        "from models trained on the other four outer-training landscapes; each calibration landscape "
        "has equal total weight; held-out Y and held-out prevalence are unavailable"
    ),
    "probability_clip": 1e-7,
    "metrics": [
        "Brier score", "log loss", "calibration-in-the-large", "calibration intercept",
        "calibration slope", "observed/expected positives", "10-bin equal-count reliability curve",
        "PR-AUC", "PR lift", "ROC-AUC",
    ],
    "spatial_uncertainty": (
        "paired 3 km spatial-block bootstrap, 1,000 replicates, seed + landscape index; "
        "95% percentile intervals for Brier and log loss"
    ),
    "selection": (
        "rank candidates by the mean of their median landscape-normalised Brier and log-loss ratios, "
        "where denominators are the held-out prevalence-only Brier and entropy log loss; add 0.25 times "
        "the worse of the two worst-landscape ratios; lowest score wins; exact ties prefer C0<C1<C2<C3"
    ),
    "catastrophic_failure": (
        "a calibrated candidate is ineligible if on any landscape both Brier and log loss exceed C0 by >25%"
    ),
    "decision_A": (
        "selected non-raw method improves median Brier and median log loss versus C0; no catastrophic "
        "failure; at least 5/6 landscapes have observed/expected in [0.8,1.25], absolute calibration-"
        "in-the-large <=0.5 and slope in [0.7,1.3]"
    ),
    "decision_B": (
        "selected non-raw method improves median Brier and median log loss versus C0; no catastrophic "
        "failure; at least 4/6 landscapes have observed/expected in [0.5,2.0], absolute calibration-"
        "in-the-large <=1.0 and slope in [0.5,2.0]"
    ),
    "decision_C": "all other outcomes",
    "labels": [
        "CALIBRATION_TRANSFER_SUPPORTED",
        "CALIBRATION_PARTIALLY_SUPPORTED",
        "CALIBRATION_TRANSFER_NOT_SUPPORTED",
    ],
}

OOD = {
    "landscape_primary": (
        "Stage 4D normalised energy statistic H_M, recomputed with the identical training-only "
        "landscape-balanced z-scale and deterministic 2,000 held-out / 400-per-training-landscape samples"
    ),
    "landscape_secondary": "none; avoiding a post-result distance search is preferred to another weak n=6 comparison",
    "unit_methods": {
        "U0_UNCONDITIONAL": (
            "mean Euclidean distance in training-z M space to k=10 neighbours in a deterministic "
            "400-unit-per-training-landscape reference; percentile against reference leave-one-out kNN distances"
        ),
        "U1_SUSCEPTIBILITY_CONDITIONED": (
            "the same kNN score within five strata defined by quintiles of LOGISTIC_M scores on the "
            "outer-training reference; held-out units use frozen training thresholds; if a stratum has "
            "fewer than 50 references, expand symmetrically to adjacent strata until >=50"
        ),
    },
    "fit_rule": (
        "scaling, reference sampling, score strata and reference distance distributions use outer-training "
        "predictors and (only for conditioning) the outer-training-fitted susceptibility model; never held-out Y"
    ),
    "freeze_rule": (
        "hash landscape and unit novelty tables in STAGE4F_OOD_FREEZE.json before any Stage 4F code loads "
        "held-out labels for reliability evaluation"
    ),
    "hazard_confounding": [
        "Spearman correlation with raw susceptibility", "Spearman correlation with slope_deg_30m__mean",
        "Spearman correlation with relief_m_r250__mean", "score distributions by susceptibility quintile",
        "within-quintile novelty IQR", "within-quintile association with absolute residual at evaluation",
    ],
}

FAILURE_DETECTION = {
    "threshold": (
        "maximise landscape-balanced weighted F1 on honest inner-LOLO calibrated predictions using the "
        "selected calibration method; apply unchanged to outer landscape"
    ),
    "large_residual": (
        "absolute error greater than the weighted 90th percentile of honest inner-LOLO absolute errors"
    ),
    "analyses": [
        "landscape H_M versus PR lift, Brier ratio and log-loss ratio (Spearman; exploratory n=6)",
        "ROC-AUC for identifying false negatives among positives",
        "ROC-AUC for identifying preregistered large residuals among all units",
        "within-susceptibility-quintile Spearman association with absolute residual",
    ],
    "reference": "susceptibility-only: negative raw score for false negatives; absolute selected-calibration residual for residual ranking",
    "incremental_rule": (
        "an OOD method provides incremental evidence only if its false-negative AUC exceeds the susceptibility "
        "reference in >=5/6 landscapes with median delta >=0.05, and its within-stratum residual association "
        "is positive in >=4/6 with median rho >=0.10"
    ),
    "confidence_release": (
        "LOW/MODERATE/HIGH categories require the incremental rule and landscape H_M directions of lower PR "
        "lift plus higher Brier and log-loss ratios; because n=6 is development-only, any passing categories "
        "remain provisional. Otherwise TRANSFER_CONFIDENCE=NOT_ESTABLISHED and novelty is descriptive/unvalidated"
    ),
}

SELECTIVE_PREDICTION = {
    "coverage": [1.0, 0.9, 0.8, 0.7],
    "ordering": "retain lowest novelty first; thresholds are target-X quantiles and never use target Y",
    "methods": ["U0_UNCONDITIONAL", "U1_SUSCEPTIBILITY_CONDITIONED"],
    "metrics": [
        "retained PR-AUC", "retained prevalence", "retained PR lift", "Brier", "log loss",
        "excluded-positive fraction", "fraction of excluded units in the most-used 10% of 3 km blocks",
    ],
    "deployment_rule": (
        "deploy abstention only if U1 improves both PR lift and normalised log loss versus 100% coverage in "
        ">=5/6 landscapes at every reduced coverage and never excludes a larger positive fraction than unit fraction"
    ),
}

PRODUCT = {
    "susceptibility_output": (
        "always provide raw monotonic susceptibility score and target-landscape percentile rank; only call "
        "a calibrated output probability when calibration decision A is reached"
    ),
    "categories": (
        "no named LOW/MEDIUM/HIGH susceptibility categories are selected in Stage 4F; percentile rank is the "
        "primary relative communication and is explicitly relative to the target landscape"
    ),
    "confidence": "release categories only under FAILURE_DETECTION.confidence_release; otherwise NOT_ESTABLISHED",
    "evidence": "DEVELOPMENT_ONLY_REPEATED_USE_N6",
}

H_STATUS = {
    "KUMAMOTO_ASO_2016": "VALIDATED_GEN3_PARTIAL",
    "NOTO_PENINSULA_2024": "AGE_PROXY_ONLY",
    "LOMBOK_RINJANI_2018": "NOT_VALIDATED",
    "COTABATO_DAVAO_2019": "NOT_VALIDATED",
    "ATITLAN_STAN_2005": "NOT_VALIDATED",
    "DOMINICA_ERIKA_2015": "NOT_VALIDATED",
    "N_VERIFIED_H_CAPABLE": 1,
    "MEGH_EXPERIMENT_UNLOCKED": "NO",
}

PROHIBITIONS = [
    "no future lockbox access", "no Gen-4", "no MEGH", "no H-gate or Gen-3 modification",
    "no new M/E/G features", "no LOGISTIC_M architecture change", "no post-result method invention",
    "no external-validation claim", "no historical output write", "no universal percentage-risk claim",
]

