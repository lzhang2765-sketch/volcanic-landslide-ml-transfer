"""Frozen Stage 4E-B design: model complexity with the M representation held fixed.

Pure data only.  This module contains no result and must be frozen before the first
XGBoost fit.  Stage 4B Logistic/RF/HistGB predictions are reused read-only.
"""
from __future__ import annotations

STAGE = "4E-B"
AS_OF = "2026-10-08"
SEED = 20261008
STAGE4B_SEED = 20261006
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
              "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
MODELS = ["LOGISTIC_L2", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING", "XGBOOST"]
COMPLEXITY_ORDER = MODELS
PRIMARY_FEATURE_SET = "M"
HISTORICAL_CONTROLS = ["ME", "MEG"]

INPUT_CHECKPOINTS = {
    "stage3f": "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a",
    "stage3g": "a4f8035123658c063f1572bdfd3276906053ba313735ae3371c442e45cbe4664",
    "stage3h": "f1ea33e848559bcc97f6c8fb7bc7a78c8fca9d004fe924d8c0136d1f488f2d1f",
    "stage4a": "fd25accb50578bd4f4f28042e42504a89ef936c904d5ed262a0f65fc7fa2452c",
    "stage4b": "2685d1b6d565de7945e138d9b014d9aa4b53657511552299c19ebc1f4df946b8",
    "stage4c_a": "09463f7d76cc865b08ad602a5518532c8a4cc2a5131952b73618082bed09bfa7",
    "stage4c_b": "90835a71a53cb8a544a960ae7d1e36f5a3ea44e524e730cd6f62db22c6e4bd7c",
    "stage4c_b2": "f548ceb0c501560b603ee7835a6a18b8350899beef92b527dc4d9571fb6aa1f7",
    "stage4c_b3": "d1ad15c66bb1df312f6738c2c1aeb5d25dc868025641bd92661631408c8d7807",
    "stage4d": "53041df090fb96edaeb6ee414dd49bd9a86a8e64d1683d8b46f567f37fa544fd",
    "stage4e_a": "e6e72d30fb7e7bb52e4cdede02809911eb0a63343f481c1f9b9acbcebccd2bf6",
}
V1_FREEZE_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"

# Ordered from strongest regularisation / shallowest structure toward greater flexibility.
XGBOOST_VERSION = "2.1.4"
XGBOOST_FIXED = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                     max_bin=256, n_jobs=8, random_state=SEED, verbosity=0)
XGBOOST_GRID = [
    dict(max_depth=2, learning_rate=0.03, n_estimators=300, min_child_weight=10, subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0, reg_alpha=1.0),
    dict(max_depth=2, learning_rate=0.05, n_estimators=200, min_child_weight=5,  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,  reg_alpha=0.5),
    dict(max_depth=3, learning_rate=0.03, n_estimators=300, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, reg_alpha=1.0),
    dict(max_depth=3, learning_rate=0.05, n_estimators=200, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,  reg_alpha=0.5),
    dict(max_depth=4, learning_rate=0.03, n_estimators=300, min_child_weight=10, subsample=0.8, colsample_bytree=0.7, reg_lambda=10.0, reg_alpha=1.0),
    dict(max_depth=4, learning_rate=0.05, n_estimators=200, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,  reg_alpha=0.5),
    dict(max_depth=3, learning_rate=0.05, n_estimators=300, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,  reg_alpha=0.0),
    dict(max_depth=4, learning_rate=0.05, n_estimators=300, min_child_weight=1,  subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,  reg_alpha=0.0),
]

VALIDATION = dict(
    outer="leave one whole landscape out; train on five; evaluate every unit of the held-out landscape",
    inner="within each outer fold, leave one of the five training landscapes out; the outer landscape is absent from all tuning",
    inner_selection="maximum unweighted mean inner-landscape PR-AUC; configurations within 0.005 use the earliest grid entry",
    sampling="Stage 4B exact rule: every positive plus at most 20,000 negatives per training landscape, selected by the frozen terrain-unit hash order",
    weighting="Stage 4B landscape-balanced sampling-corrected weights; every training landscape has equal total weight and retains its population prevalence",
    preprocessing="none for tree models; Logistic historical predictions retain training-only Stage 4B z-scaling",
    held_out_rule="no held-out Y, X summary, sampling, threshold, calibration or score may affect training or tuning",
)

METRICS = dict(
    primary="PR-AUC (average precision) from raw held-out scores",
    primary_transfer="PR_LIFT = PR-AUC / held-out EQ1 prevalence; six landscapes are six transfer experiments",
    secondary=["ROC-AUC", "raw Brier", "raw log loss", "raw calibration intercept/slope",
               "precision", "recall", "F1 at the training-only Stage 4B-style threshold", "prediction-score distribution"],
    aggregation=["median PR-AUC", "median PR lift", "minimum PR lift", "IQR PR lift", "SD log(PR lift)", "per-landscape rank"],
    prohibited="pooled terrain-unit performance is never the primary transfer metric and PR-AUC is never called accuracy",
)

BOOTSTRAP = dict(block="frozen Stage 4A spatial_block_3000m", replicates=2000,
                 method="resample held-out 3 km blocks with replacement; identical draws across models within landscape",
                 interval="95% percentile", paired=True,
                 seed_rule="numpy default_rng(SEED + landscape index)",
                 outputs=["PR-AUC", "PR lift", "ROC-AUC", "paired candidate-minus-comparator differences"],
                 limitation="within-landscape spatial uncertainty only; not uncertainty over the population of volcanoes")

SELECTION = dict(
    score="ROBUST_SCORE = 0.70*log(MEDIAN_LIFT) + 0.30*log(WORST_LIFT); all lifts must be finite and > 0",
    raw_ranking="among candidates passing the catastrophic guard, greatest ROBUST_SCORE; exact ties (1e-12) use COMPLEXITY_ORDER",
    logistic_relative_tolerance=0.05,
    logistic_rule="if the raw-best complex model's score advantage over LOGISTIC_L2 is <= 5% of abs(LOGISTIC_L2 score), select LOGISTIC_L2",
    tiny_median_pr_auc_delta=0.002,
    uncertainty_rule="also select LOGISTIC_L2 when raw-best median PR-AUC improvement is <=0.002 AND the 95% paired bootstrap interval for the median six-landscape PR-AUC difference includes zero",
    runner_up="highest-ranked eligible model other than the selected leader after applying the same score and exact-tie ordering",
    selection_margin="selected ROBUST_SCORE minus RUNNER_UP ROBUST_SCORE; may be negative when the preregistered simplicity override selects Logistic",
    no_clear_leader="only if every non-null candidate fails the catastrophic guard or a required metric is non-finite; otherwise exactly one leader",
)

CATASTROPHIC_GUARD = dict(
    reference="LOGISTIC_L2 + M",
    non_cotabato=("where Logistic PR lift >=1.25, a candidate fails if paired 95% upper CI for Delta PR(candidate-Logistic) < 0 and either "
                   "candidate lift <=1.0 or candidate lift <0.67*Logistic lift"),
    cotabato=("EQ1 remains included but is already near-null: a candidate fails there only if candidate lift <0.80 and paired 95% upper CI for "
              "Delta PR(candidate-Logistic) < 0"),
    purpose="prevent an easy-landscape gain from hiding a major new transfer collapse without making the shared Cotabato EQ1 failure disqualifying",
)

COTABATO = dict(primary="COTABATO_DAVAO_2019 EQ1 frozen y_positive; never excluded or replaced",
                sensitivity=["cumulative_post_EQ3_nov2019", "cumulative_post_EQ4_dic2019"],
                timing="score cumulative labels only after primary M selection metrics and decision are frozen",
                prohibition="sensitivity labels cannot tune, select, change features, alter the rule or replace EQ1")

CONTROLS = dict(
    historical="reuse authoritative Stage 4B M/ME/MEG results for Logistic/RF/HistGB; never refit solely to create new values",
    xgboost="after the primary XGBOOST+M result and selection freeze, fit XGBOOST+ME and XGBOOST+MEG with the same nested architecture as secondary historical controls",
    interpretation="E/G controls cannot become the primary architecture and cannot trigger feature engineering",
)

CALIBRATION = dict(selection="never used for primary architecture selection",
                   reporting="raw Brier/log loss, calibration intercept/slope and reliability summaries; legacy Stage 4B-style inner Platt/F1 threshold only for continuity",
                   next_stage="final recalibration, categories and reliability are explicitly deferred")
IMPORTANCE = dict(method="held-out permutation PR-AUC drop on all positives plus 20,000 deterministic negatives; three repeats per fold",
                  logistic="standardised coefficients from training-only scaling, summarised across folds",
                  trees="model-agnostic permutation importance; native impurity/gain may be a labelled secondary diagnostic only",
                  selection="importance cannot alter features or select the model")
STABILITY = dict(method="remove one of the six already-computed outer results, recompute summaries/ranking without retraining",
                 report="five-landscape wins, leader changes, driving landscape, Noto and Atitlan dependence")
COMPLEXITY = dict(report=["configuration/proxy", "fit time", "inference time", "serialized bytes", "peak memory if practical"],
                  role="secondary; performance governs unless the frozen simplicity tolerance activates")
ENSEMBLE = dict(status="feasibility diagnostic only after individual results freeze",
                analyses=["held-out score Spearman correlation", "rank disagreement", "error complementarity"],
                prohibition="no ensemble fitting, averaging, weight optimisation or leader replacement in Stage 4E-B")
OOD = "no final OOD thresholds, adjustment or OOD-based selection; Stage 4D confounding finding retained"
H_POLICY = dict(KUMAMOTO_ASO_2016="CORE_H_AVAILABLE=1", NOTO_PENINSULA_2024="AGE_PROXY_ONLY; CORE_H_AVAILABLE=0",
                LOMBOK_RINJANI_2018="CORE_H_AVAILABLE=0", COTABATO_DAVAO_2019="CORE_H_AVAILABLE=0",
                ATITLAN_STAN_2005="CORE_H_AVAILABLE=0", DOMINICA_ERIKA_2015="CORE_H_AVAILABLE=0",
                N_VERIFIED_H_CAPABLE=1, MEGH_EXPERIMENT_UNLOCKED="NO", MEGH_TRAINED="NO", GEN4="NO")
LOCKBOX = "no future lockbox is named, loaded, inspected or scored; all six landscapes are development-exposed"
NO_FEATURE_ENGINEERING = "no post-result feature engineering; frozen 60-column M; no new M/E/G; no E1/E2/E3; H absent"

