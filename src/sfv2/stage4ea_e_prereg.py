"""Stage 4E-A environmental-representation pre-registration (frozen AFTER the Cotabato audit freeze and BEFORE any E1/E2/E3 predictive result).

Pure data.  Candidate set, formulas, safeguards, screen design and the advancement rule are fixed here; no transformation may be invented after results.
"""
from __future__ import annotations

SEED = 20261008
STAGE4B_SEED = 20261006
E_SOURCE = ["e_chelsa_bio1_mean_annual_temp", "e_chelsa_bio4_temp_seasonality", "e_chelsa_bio12_annual_precip", "e_chelsa_bio13_wettest_month_precip",
            "e_chelsa_bio15_precip_seasonality", "e_hansen_treecover2000_pct__mean", "e_hansen_treecover2000_pct__sd"]
FORBIDDEN_SOURCES = ["trigger intensity", "Hansen lossyear", "SoilGrids", "ESA CCI", "any post-event layer", "any Y-derived quantity"]

VALIDITY_CHECK = dict(
    rule="a variable may be relatively normalised in a landscape only if its within-landscape IQR > 0, it has >= 20 distinct values and its most common value covers <= 50% "
         "of units; a variable failing in ANY landscape is excluded from E1/E2/E3 for ALL landscapes (consistent columns)",
    result_before_freeze="X-only check run before this freeze: all 7 variables pass in all 6 landscapes (IQR > 0; 34-46,773 distinct values; largest tie share <= 0.23)",
    caveat="CHELSA fields are ~1 km cells: within-landscape variation is spatially blocky and largely orographic (expected to correlate with elevation)",
)
REPRESENTATIONS = {
    "E0_ABSOLUTE": dict(formula="x (Stage 4A values)", role="historical control (Stage 4B ME)", columns=7, transductive=False),
    "E1_WITHIN_LANDSCAPE_PERCENTILE": dict(formula="p_L(x) = (rank_L(x) - 0.5) / n_L with average ranks for ties, computed over ALL units of landscape L",
                                           role="candidate", columns=7, transductive=True),
    "E2_LANDSCAPE_CENTERED_ROBUST": dict(formula="(x - median_L(x)) / IQR_L(x) over all units of landscape L", role="candidate", columns=7, transductive=True),
    "E3_HYBRID_CONTEXT_RELATIVE": dict(formula="E2 for bio1, bio4, bio13, bio15, treecover_sd + ABSOLUTE log10(bio12) and ABSOLUTE treecover_mean (%)",
                                       role="candidate", columns=7, transductive="partly (5 relative columns)",
                                       justification="annual precipitation total and canopy cover are physically interpretable absolute controls (hydrologic loading, root "
                                                     "reinforcement) and were the least landscape-identifying absolute E variables in the Y-free Stage 4D identity audit "
                                                     "(single-variable balanced accuracy 0.217 and 0.305 vs 0.78-0.79 for bio15/bio4); seasonality/temperature variables, the "
                                                     "strongest identifiers, enter only relatively. Fixed before any E1/E2 result."),
}
MAX_ALTERNATIVES = 3  # E1, E2, E3 (limit 4)
TRANSDUCTIVE = dict(
    label="DEPLOYMENT-AVAILABLE X-ONLY LANDSCAPE NORMALIZATION",
    statement="E1/E2 (and the relative part of E3) use the unlabeled predictor distribution of each landscape, including the held-out one. A new landscape's predictor "
              "rasters exist before inference, so this is available at deployment. No Y, no performance, deterministic. It is NOT training-only scaling.",
    strict_training_reference_sensitivity=("Normalising with pooled TRAINING-landscape statistics is an affine (E2-type) or monotone (E1-type) transform of the absolute values "
                                           "applied identically to every landscape. Logistic regression already z-scores on the training sample and trees are invariant to "
                                           "monotone transforms, so this variant reproduces E0 predictions exactly; it is therefore reported analytically and in the identity "
                                           "audit, not re-fitted."),
)
LEAKAGE = ["no Y in any transform", "per-landscape statistics use all units of that landscape (X only)", "no held-out label, prediction or metric influences a transform",
           "M unchanged; G not added to the screen"]

SCREEN = dict(
    architecture="sfv2.stage4b.run_analysis unchanged (outer LOLO, inner leave-one-training-landscape-out tuning, all positives + 20,000 hash-chosen negatives per landscape, "
                 "landscape-balanced weights, Stage 4B grids, Platt calibration, F1 threshold, seed 20261006); cache data/interim/_stage4ea_cache",
    model_families=["LOGISTIC_L2", "RANDOM_FOREST"], excluded=["HIST_GRADIENT_BOOSTING (not needed for this question)", "XGBoost", "LightGBM", "CatBoost", "neural networks", "ensembles"],
    feature_sets={"M": "reused Stage 4B", "M+E0": "reused Stage 4B ME", "M+E1": "new", "M+E2": "new", "M+E3": "new"},
    metrics=["PR-AUC (primary)", "PR lift", "ROC-AUC", "Brier", "calibration slope/intercept", "median and worst landscape"],
    uncertainty="paired 3 km block bootstrap (Stage 4D method, 2,000 replicates) for Delta PR-AUC vs M",
)
IDENTITY = dict(architecture="identical to Stage 4D: HistGradientBoosting, 3,000 hash-chosen units per landscape, 5-fold GroupKFold over 6 km blocks",
                sets=["E-only for E0, E1, E2, E3 (primary)", "M + E-representation (secondary)"], report=["balanced accuracy", "confusion matrix", "per-landscape recall", "chance 1/6"])
RETENTION = ["nonconstant fraction (IQR > 0) per landscape", "distinct values", "largest tie share", "within-landscape Spearman with the absolute variable",
             "between-landscape variance fraction", "within-landscape Spearman with elev_m_30m__mean (redundancy with M)"]
ATITLAN_TEST = ["fraction of Atitlan units outside the training range per column (max over columns)", "LOGISTIC_L2 fraction of Atitlan units flagged at the Stage 4B-style threshold",
                "Atitlan PR-AUC vs M and vs E0"]

DECISION_RULE = dict(
    per_family="evaluated separately for LOGISTIC_L2 and RANDOM_FOREST; a representation advances only if it passes A-F in BOTH",
    A="median over the 6 held-out landscapes of PR-AUC(rep) / PR-AUC(M) >= 0.95 (no material median loss; 5% tolerance ~ Stage 4B within-family spread)",
    B="for every landscape PR-AUC(rep) / PR-AUC(M) >= 0.80 (no catastrophic loss; E0 reaches 0.47 (RF) / 0.79 (LR) at Atitlan)",
    C="worst-landscape PR lift(rep) >= 0.90 x worst-landscape PR lift(M)",
    D="E-only landscape-ID balanced accuracy <= E0's minus 0.20",
    E="every column nonconstant (IQR > 0) in all 6 landscapes and largest tie share <= 0.5 (not information destruction)",
    F="criterion A still holds when any single landscape is left out (gain/no-loss not driven by one volcano)",
    max_advance=2, tie_break="if more than 2 pass: higher mean (LR, RF) worst-landscape lift ratio; then fewer absolute columns (E1/E2 before E3)",
    controls="M is mandatory in Stage 4E-B; E0 remains a historical control regardless",
    not_used="maximum median PR-AUC alone is never the criterion",
)
G_POLICY = "existing two G variables preserved as control (MEG); no new geology features"
H_POLICY = dict(N_VERIFIED_H_CAPABLE=1, MEGH_EXPERIMENT_UNLOCKED="NO")
