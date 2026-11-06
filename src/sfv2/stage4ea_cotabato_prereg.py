"""Stage 4E-A Cotabato failure-audit pre-registration (frozen before any label-geometry, conditional-morphology or auxiliary-inventory result).

Pure data.  Diagnosis only: no model is changed, no landscape is excluded, nothing here feeds a predictor.
"""
from __future__ import annotations

SEED = 20261008
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
POLYGON_INVENTORIES = ["NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
POINT_INVENTORIES = {"KUMAMOTO_ASO_2016": "point occurrence centres: excluded from polygon-specific metrics; unit-level metrics reported with a POINT flag"}

LABEL_GEOMETRY = dict(
    unit_level=["prevalence", "8-connected positive-unit components on the 90 m unit grid: count, median / P90 / max size, share of positives in components >= 10 units",
                "nearest-positive distance (m) P50/P90", "local positive density = positives within 1 km of a positive (P50)",
                "Gini of positives over 3 km blocks; share of positives in the top 10% of 3 km blocks", "slope_deg_90m__mean and relief_m_r250__mean of positives (P25/P50/P75)"],
    polygon_level=["source polygon count", "polygon area (m2) P10/P50/P90", "Y area / Omega area", "polygons per km2 of Omega",
                   "30 m cell centres and 90 m units hit per polygon (P50/P90; polygon rasterised on the landscape's 30 m grid, overlaps resolved by order)",
                   "share of polygons hitting no 30 m cell centre"],
)
AUX_INVENTORY_TEST = dict(
    purpose="separate EQ1-specific label semantics from a landscape-wide P(Y|M) difference",
    labels=["PRIMARY EQ1 (landslides_ott2019; frozen target)", "cumulative post-EQ3 (landslides_nov2019; auxiliary)", "cumulative post-EQ4 (landslides_dic2019; auxiliary)"],
    rule="same 9-cell-centre rule as the frozen target, applied to the frozen Cotabato Stage 4A units; auxiliary inventories evaluated only on units whose centroid lies inside "
         "area_study (their mapping extent)",
    predictions="EXISTING Stage 4B held-out Cotabato predictions (RANDOM_FOREST MEG and LOGISTIC_L2 M), read-only; no refit",
    metrics="ROC-AUC and PR lift with the paired 3 km block bootstrap (2,000 replicates, seed SEED); Delta ROC-AUC (aux - EQ1) on the common units",
    status="Y-aware diagnostic; auxiliary labels never become targets or predictors",
)
CONDITIONAL_M = dict(
    predictors="the 10 M predictors with the highest mean Stage 4B permutation PR-AUC drop (RANDOM_FOREST, MEG, individual predictors, mean over the six outer folds) "
               "from outputs/stage4b/STAGE4B_FEATURE_INTERPRETATION.tsv; plus slope_deg_90m__mean, relief_m_r250__mean, tpi_m_r450__mean, "
               "roughness_sd_m_90m__mean, curv_total_1pm_90m__mean, rel_elev_local_r1000__mean if not already included",
    metrics=["univariate ROC-AUC of the predictor for Y within each landscape (0.5 = none; < 0.5 = reversed relative to > 0.5)",
             "pooled training ROC-AUC (landscape-balanced weights, Cotabato excluded)", "standardised median difference of Cotabato positives vs pooled training positives "
             "(training-positive IQR)", "the same for negatives", "binned event rate of Cotabato vs training in deciles of the pooled-training distribution (slope, relief)"],
)
CLASSES = ["COVARIATE_SHIFT_DOMINATED", "LABEL_OR_INVENTORY_SHIFT_PLAUSIBLE", "TRIGGER_OR_CONCEPT_SHIFT_PLAUSIBLE", "MIXED_SHIFT", "NO_CLEAR_EXPLANATION"]
CRITERIA = {
    "COVARIATE": ["C1: Cotabato H_M (Stage 4D) >= median H_M of the other five landscapes",
                  "C2: median Stage 4D OOD percentile of Cotabato positives >= 0.90"],
    "LABEL_OR_INVENTORY": ["L1: Cotabato is the minimum among the five polygon inventories in >= 2 of {Y area / Omega area, polygons per km2 Omega, median polygon area}",
                           "L2: Stage 4B RF+MEG ROC-AUC on an auxiliary cumulative Cotabato inventory exceeds its EQ1 ROC-AUC by >= 0.05 with a paired 95% interval > 0",
                           "L3: median 90 m units hit per source polygon is the minimum among the five polygon inventories AND <= 1"],
    "TRIGGER_OR_CONCEPT": ["T1: median |univariate ROC-AUC - 0.5| over the audited M predictors in Cotabato < 0.5 x the median of the same statistic over the other five landscapes",
                           "T2: >= 3 audited M predictors whose Cotabato direction is reversed relative to the pooled training direction (|AUC - 0.5| >= 0.03 in both)",
                           "T3: Cotabato's trigger class (Phase C) is shared by <= 1 of the five training landscapes"],
}
DECISION = dict(
    counting="a class is SUPPORTED if >= 1 of its criteria holds",
    primary="the supported class with the most satisfied criteria (ties: LABEL_OR_INVENTORY before TRIGGER_OR_CONCEPT before COVARIATE, because inventory evidence is the most directly observable)",
    mixed="if >= 2 classes are supported -> PRIMARY_CLASS = MIXED_SHIFT; DOMINANT and SECONDARY components reported",
    none="no class supported -> NO_CLEAR_EXPLANATION",
    confidence=dict(HIGH="dominant class has >= 2 satisfied criteria from independent evidence types and no competing class has more than 1",
                    MEDIUM="dominant class has >= 1 criterion and strictly more than any competitor", LOW="otherwise"),
    not_permitted="the classification never removes or down-weights Cotabato; any future exclusion needs an independent data-quality reason",
)
