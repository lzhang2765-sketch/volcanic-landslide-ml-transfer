"""Stage 4D pre-registration: cross-landscape transfer diagnosis, domain shift and feature robustness (DEVELOPMENT ONLY).

Pure data, frozen before any Stage 4D diagnostic result is computed.  No model-family or feature-set winner is selected; no H predictive test;
no future lockbox; V1 and every authoritative V2 output are read-only.
"""
from __future__ import annotations

STAGE = "4D"
AS_OF = "2026-10-07"
SEED = 20261007
STAGE4B_SEED = 20261006
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]

INPUTS = dict(
    stage4a="outputs/stage4a/STAGE4A_MEG_DATASET (835,846 units; 60 M + 7 E + 2 G admitted predictors)",
    stage4b_results="outputs/stage4b/STAGE4B_LANDSCAPE_RESULTS.tsv (PRIMARY analysis: selected params, Platt calibrator, threshold per fold)",
    stage4b_predictions=("held-out raw probabilities re-read (read-only) from the Stage 4B prediction cache data/interim/_stage4b_cache using the Stage 4B key "
                         "(training landscapes, feature set, model, selected params, ALL_VALID, LANDSCAPE_BALANCED); every reconstructed fold must reproduce the "
                         "published Stage 4B PR-AUC to 1e-12 or Stage 4D STOPs"),
    new_fits="MG only (missing ablation): sfv2.stage4b.run_analysis unchanged (same folds, inner LOLO tuning, sampling, weighting, grids, seed, calibration, "
             "threshold), feature set MG = M + G, cache redirected to data/interim/_stage4d_cache (the Stage 4B cache is never written)",
)
MODELS = ["NULL_PREVALENCE", "LOGISTIC_L2", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING"]
FEATURE_SETS = ["M", "ME", "MG", "MEG", "MEG_NO_BIO15"]

# ------------------------------------------------------------------------------------------------ A/B metrics
METRICS = dict(primary="PR-AUC (average precision) on raw held-out probabilities, all rows of the held-out landscape (Stage 4B convention)",
               secondary=["ROC-AUC (raw)", "Brier score (calibrated, Stage 4B convention)", "log loss (calibrated)", "calibration intercept and slope",
                          "precision / recall / F1 at the Stage 4B training-chosen threshold", "PR_LIFT = PR-AUC / held-out prevalence",
                          "Delta_PR = PR-AUC(model) - PR-AUC(null); the prevalence null's PR-AUC equals the held-out prevalence"],
               wording="PR-AUC is never called accuracy; it is always read against prevalence")

# ------------------------------------------------------------------------------------------------ C bootstrap
BOOTSTRAP = dict(
    unit="frozen Stage 4A spatial_block_3000m (deterministic 3 km grid cells from unit coordinates; Y-free; the same blocks Stage 4B recorded)",
    method="within each held-out landscape, resample its 3 km blocks with replacement (as many draws as blocks); each unit's weight = its block's draw count; "
           "metrics are computed with those weights (weighted average precision with tied scores grouped exactly as sklearn; weighted ROC-AUC with half-credit ties; "
           "weighted Brier)",
    replicates=2000, interval="95% percentile", paired="identical block draws for every model, feature set and the null within a landscape (paired differences)",
    seed_rule="numpy default_rng(SEED + landscape index)", degenerate="replicates with no positive are dropped and counted",
    sensitivity="6 km blocks (spatial_block_6000m) for the Stage 4B development leader only, reported as sensitivity",
    rationale="terrain units are spatially autocorrelated; a row bootstrap would overstate precision; 3 km exceeds the 450-1500 m M focal windows",
)
TRANSFER_LABELS = dict(
    ROBUST_POSITIVE_TRANSFER="paired 95% interval for Delta_PR entirely > 0",
    UNCERTAIN_POSITIVE_TRANSFER="point Delta_PR > 0 but the paired 95% interval includes 0",
    CHANCE_LIKE_OR_WORSE="point Delta_PR <= 0",
    landscape_label_model="labels are computed for every model x feature set; the landscape summary label uses the Stage 4B DEVELOPMENT_LEADER (RANDOM_FOREST, MEG); "
                          "the per-landscape 'best existing' cell is reported as a post-hoc maximum (optimistic, flagged)",
    note="descriptive diagnostics, not universal significance claims",
)

# ------------------------------------------------------------------------------------------------ D/E domain shift (Y-blind)
DOMAIN_SHIFT = dict(
    primary="normalised energy statistic H = (2 E|X-Y| - E|X-X'| - E|Y-Y'|) / (2 E|X-Y|) in [0, 1] (Szekely & Rizzo energy coefficient), Euclidean distance",
    groups=["M", "E", "G", "ME", "MG", "EG", "MEG"],
    preprocessing="z-score every predictor with the landscape-balanced mean and SD of the outer TRAINING landscapes only (fold-specific); the held-out landscape never "
                  "enters the scaling; binary G predictors are z-scored the same way",
    sampling="held-out: 2,000 units; training pool: 400 units from each of the 5 training landscapes (2,000); units chosen by ascending sha256(terrain_unit_id + '|stage4d') "
             "(Y-free, deterministic)",
    why_selected="distribution-free, defined for mixed continuous/binary predictors, zero iff the distributions are equal, no kernel bandwidth to tune, bounded and "
                 "comparable across predictor groups of different dimension (normalised), chosen conceptually before any relation to performance is examined",
    limitations="Euclidean geometry weights all z-scored predictors equally (M has 60 vs E 7, G 2); subsample noise (reported via 20 subsample-seed replicates); "
                "describes marginal-joint predictor shift only, not concept shift P(Y|X)",
    secondary=["sliced Wasserstein-1 (100 random unit directions, seed SEED) on the same samples", "subsample-replicate SD of H (20 replicates with seeds SEED+r)"],
    univariate=["standardised median difference = (median_heldout - median_train) / IQR_train", "Wasserstein-1 on the training-z scale", "KS statistic",
                "binary G: prevalence difference"],
    y_blind_order="all domain-shift and identity-audit outputs are computed and hashed (STAGE4D_YBLIND_FREEZE.json) BEFORE any Y-aware Stage 4D analysis",
)
ASSOCIATION = dict(
    primary="Spearman rho between H_MEG (held-out vs training) and held-out PR-AUC of the Stage 4B DEVELOPMENT_LEADER (RANDOM_FOREST, MEG), n = 6 landscapes",
    secondary=["Spearman rho vs PR_LIFT and ROC-AUC (prevalence-independent)", "the same for LOGISTIC_L2 M and every model x MEG", "Pearson r (secondary)",
               "leave-one-landscape-out rho (6 values, n = 5 each)", "exact permutation distribution of rho over all 720 orderings (descriptive)"],
    language="hypothesis-generating association only; never causal; n = 6 independent landscapes regardless of terrain-unit counts",
)

# ------------------------------------------------------------------------------------------------ F feature-group ablation
ABLATION = dict(
    deltas={"Delta_E": "ME - M", "Delta_G": "MG - M", "Delta_G_given_E": "MEG - ME", "Delta_E_given_G": "MEG - MG"},
    per="model family x held-out landscape; PR-AUC; paired block bootstrap interval",
    summaries=dict(median_transfer="median held-out PR-AUC and median PR_LIFT across 6 landscapes",
                   worst_landscape="minimum PR_LIFT across 6 landscapes", stability="SD of log(PR_LIFT) across 6 landscapes (lower = more stable)",
                   consistency="number of landscapes with Delta interval > 0, < 0, or spanning 0",
                   dominance="a gain is 'dominated by one landscape' if removing that landscape flips the sign of the median Delta",
                   loo="rankings recomputed leaving each landscape out"),
    no_selection="no feature set or model family is selected in Stage 4D",
)

# ------------------------------------------------------------------------------------------------ G identity audit / H OOD prototype / forensics
IDENTITY = dict(
    task="predict landscape_id (6 classes) from predictors; diagnostic only (never a susceptibility model; never touches lockboxes)",
    model="HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)",
    sample="3,000 units per landscape by ascending sha256(terrain_unit_id + '|stage4d-id') (Y-free)",
    cv="5-fold GroupKFold over spatial_block_6000m (blocks never split between train and test)",
    metrics=["balanced accuracy (chance = 1/6)", "macro one-vs-rest ROC-AUC"], groups=["M", "E", "G", "MEG"],
    per_predictor="single-predictor balanced accuracy with the same model and CV",
    policy="identity predictability is reported, never used to remove a predictor in Stage 4D",
)
OOD = dict(
    status="PROTOTYPE / DEVELOPMENT DIAGNOSTIC / NONFINAL",
    score="for each held-out unit: mean Euclidean distance to its k = 10 nearest neighbours in the training reference sample (same z-scaling and sample as the "
          "primary shift metric), expressed as the percentile of the training reference's own leave-one-out 10-NN distances",
    analyses=["false-negative rate among positives by OOD quintile (positives classified with the Stage 4B threshold)",
              "false-positive rate among negatives by OOD quintile", "PR_LIFT within OOD quintiles", "landscape-level: median OOD score vs PR_LIFT"],
    no_thresholds="no OOD threshold is defined, tuned or integrated; nothing is tuned to held-out PR-AUC",
)
FORENSICS = dict(landscapes=["COTABATO_DAVAO_2019", "NOTO_PENINSULA_2024", "ATITLAN_STAN_2005"],
                 model="Stage 4B DEVELOPMENT_LEADER (RANDOM_FOREST, MEG) plus LOGISTIC_L2 M/ME for the Atitlan E question",
                 items=["prevalence", "prediction distribution", "positive vs negative score distributions", "calibration (reliability deciles)",
                        "false-positive / false-negative clusters (3 km blocks) at the Stage 4B threshold", "feature shift vs training", "OOD enrichment",
                        "most shifted M/E/G predictors", "identity risk of E/G"],
                 timing="only after STAGE4D_YBLIND_FREEZE.json exists; diagnosis only, no model change")
MULTIPLICITY = ("many landscape x model x feature-set cells are examined; no p-value is used for a claim; intervals are descriptive; conclusions require consistency "
                "across landscapes and model families, and single-landscape effects are always reported alongside the others")
H_POLICY = dict(N_VERIFIED_H_CAPABLE=1, MEGH_EXPERIMENT_UNLOCKED="NO", rule="no H predictive test; Gen-3 is read-only and reported only")
LOCKBOX = "no future lockbox is accessed (stage4a_base.firewall_check on every load; development landscapes only)"
NO_WINNER = "no model-family or feature-set winner is selected in Stage 4D"
