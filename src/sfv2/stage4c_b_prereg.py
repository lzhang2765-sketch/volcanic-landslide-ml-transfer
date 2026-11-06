"""Stage 4C-B pre-registration: every operational choice of the production H reconstruction, frozen BEFORE any surface is fitted.

This module is pure data (no imports, no file access).  It operationalises the frozen Stage 4C-A specification
(stage4c_a_spec.py, STAGE4C_A_*.md/.tsv) where 4C-A left an implementation detail open.  Nothing here was chosen from
landslide occurrence (Y), Omega labels, Stage 4B residuals/predictions, or any H predictive result.  The validation gates
are copied verbatim from Stage 4C-A and are never relaxed.
"""
from __future__ import annotations

AS_OF = "2026-10-07"
SEED = 4302

LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
LOCKBOXES = ["VANUATU_HAROLD_2020", "POPOCATEPETL_PUEBLA_2017", "CHUUK_CHATAAN_2002", "MOUNT_ELGON"]

# ------------------------------------------------------------------------------------------------ inputs (Y-free allow-list)
# Modern DEM = the exact DEM from which frozen M was computed; geology = the frozen index raster + its ordered mapping table.
SOURCES = {
    "KUMAMOTO_ASO_2016": dict(dir="stage2c", dem="dem_cop30_30m.tif", geo="geology_seamless_index_30m.tif",
                              mapping="outputs/stage2c/geology_mapping_KUMAMOTO_ASO_2016_seamless.csv"),
    "NOTO_PENINSULA_2024": dict(dir="stage2c", dem="dem_cop30_30m.tif", geo="geology_seamless_index_30m.tif",
                                mapping="outputs/stage2c/geology_mapping_NOTO_PENINSULA_2024_seamless.csv"),
    "LOMBOK_RINJANI_2018": dict(dir="stage2f2", dem="dem_cop30_30m.tif", geo="geology_geomap_index_30m.tif",
                                mapping="outputs/stage2f2/geology_mapping_LOMBOK_geomap_primary.csv"),
    "COTABATO_DAVAO_2019": dict(dir="stage3d", dem="dem_cop30_30m.tif", geo="geology_mgb_index_30m.tif",
                                mapping="outputs/stage3d/COTABATO_GEOLOGY_MAPPING.csv"),
    "ATITLAN_STAN_2005": dict(dir="stage3f", dem="dem_primary_30m.tif", geo="geology_official_index_30m.tif",
                              mapping="outputs/stage3f/ATITLAN_GEOLOGY_HARMONISATION.csv"),
    "DOMINICA_ERIKA_2015": dict(dir="stage3f", dem="dem_primary_30m.tif", geo="geology_official_index_30m.tif",
                                mapping="outputs/stage3f/DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv"),
}
MAPPING_COLUMNS = ["source_unit_code", "source_age", "harmonised_class"]  # explicit allow-list (some mapping tables also carry Y-area columns)
FORBIDDEN_PATH_PREFIX = "y_"  # any path component starting with y_ (target geometries) is refused
FORBIDDEN_PATH_TOKENS = ("target", "omega", "stage4b", "landslide", "prediction", "residual", "positive")
ASO_CALDERA_SOURCE = "data/interim/_stage2c_aso/geo_A.shp"  # GSJ Aso Volcano 1:50,000 geological map (registered in Stage 2B)

# ------------------------------------------------------------------------------------------------ Part I: public centres
GVP = dict(
    publisher="Smithsonian Institution, Global Volcanism Program (GVP)", dataset="Volcanoes of the World (VOTW), live database via GVP GeoServer WFS",
    layers={"HOLOCENE": "GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes", "PLEISTOCENE": "GVP-VOTW:Smithsonian_VOTW_Pleistocene_Volcanoes"},
    url="https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0&request=GetFeature&typeName={layer}&outputFormat=application/json",
    licence="Public; Smithsonian Terms of Use; citation required: Global Volcanism Program, Volcanoes of the World (Smithsonian Institution)",
    coordinate_semantics="GVP summit/representative location of the volcano (GVP 'Latitude'/'Longitude'; decimal degrees, WGS84)",
    selection="all catalogue entries inside the landscape DEM bounding box +0.1 deg; only development landscapes; nothing else is retained",
)
RADIAL_CENTRES = {  # registered centres used to build radial flanks (from GVP; never from Y).  Others are vent-core exclusions only.
    "LOMBOK_RINJANI_2018": dict(rule="Holocene GVP centres inside the window", names=["Rinjani"]),
    "DOMINICA_ERIKA_2015": dict(rule="Holocene GVP centres (young Pleistocene-Recent constructional centres)",
                                names=["Diables, Morne aux", "Diablotins, Morne", "Trois Pitons, Morne", "Watt, Morne", "Plat Pays, Morne"]),
    "COTABATO_DAVAO_2019": dict(rule="registered GVP centre(s) within the window; radial only if CENTRE_GEOMETRY_GATE passes", names=["Apo"]),
}

# ------------------------------------------------------------------------------------------------ Part II: family assignment
FAMILY = {
    "KUMAMOTO_ASO_2016": dict(primary="PRESERVED_REMNANT_ENVELOPE", caldera="CALDERA_RIM_CONSTRAINED_MASK", semantics="MINIMUM_REMOVAL"),
    "NOTO_PENINSULA_2024": dict(primary="AGE_HISTORY_PROXY", caldera=None, semantics="NONE"),
    "LOMBOK_RINJANI_2018": dict(primary="RADIAL_MULTI_CENTRE_FLANK", caldera="CALDERA_RIM_CONSTRAINED_MASK", semantics="ESTIMATED_REMOVAL"),
    "COTABATO_DAVAO_2019": dict(primary="RADIAL_MULTI_CENTRE_FLANK", fallback="PRESERVED_REMNANT_ENVELOPE", caldera=None, semantics="ESTIMATED_REMOVAL|MINIMUM_REMOVAL (fallback)"),
    "ATITLAN_STAN_2005": dict(primary="PRESERVED_REMNANT_ENVELOPE", caldera=None, semantics="MINIMUM_REMOVAL"),
    "DOMINICA_ERIKA_2015": dict(primary="RADIAL_MULTI_CENTRE_FLANK+PRESERVED_REMNANT_ENVELOPE", caldera=None, semantics="ESTIMATED_REMOVAL (young) | MINIMUM_REMOVAL (older)"),
}

# Family-applicable mapped units (= Stage 4C-A indicative-support rules; frozen).  domain -> rule on the mapping table.
#   kind: "age_contains" (source_age substring AND harmonised class), "codes" (explicit source codes)
APPLICABLE = {
    "KUMAMOTO_ASO_2016": {"ENVELOPE": dict(kind="age_contains", age="Late Pleistocene", classes=["PYROCLASTIC"],
                                           note="caldera-forming Late-Pleistocene pyroclastic-flow units (outer plateau); temporal target = their depositional top")},
    "LOMBOK_RINJANI_2018": {"RADIAL": dict(kind="codes", codes=["Qhv", "Qhv1", "Qvl"], note="strictly Quaternary young-edifice units (Sheet 1807)")},
    "COTABATO_DAVAO_2019": {"RADIAL_OR_ENVELOPE": dict(kind="codes", codes=["I-QV", "I-QVP"], note="Pliocene-Quaternary volcanic units (MGB)")},
    "ATITLAN_STAN_2005": {"ENVELOPE": dict(kind="codes", codes=["Qp", "Tv"], note="all mapped volcanic highland / pumice-fill units (minimum removal)")},
    "DOMINICA_ERIKA_2015": {"RADIAL": dict(kind="codes", codes=["YPR_BLOCK_ASH_APRON", "YPR_IGNIMBRITE", "PELEAN_DOME_RED"], note="younger Pleistocene-Recent centres"),
                            "ENVELOPE": dict(kind="codes", codes=["OP_BLOCK_ASH_AND_LAVA", "PLIO_ASSORTED_VOLCANIC", "PLIO_PELEAN_DOME", "MIOCENE_ASSORTED_VOLCANIC"],
                                             note="older Miocene-Older-Pleistocene units (minimum removal)")},
}
# Mapped units younger than the temporal target (deposits that may legitimately lie above the reference).  Support already excludes
# them by geology; the list is recorded so the negative-removal gate is computed "outside mapped young deposits" explicitly.
YOUNG_DEPOSIT_CODES = {lid: [] for lid in LANDSCAPES}

# ------------------------------------------------------------------------------------------------ Part III: control extraction
WINDOW_BUFFER_M = 3000.0          # reconstruction window = bbox(Stage 4A unit centroids) + buffer; per-cell Omega/Y labels never used
CONTROL = dict(
    resolution_m=30.0,             # candidates detected on the 30 m M DEM (TPI exactly as frozen M: annulus focal mean)
    tpi_radius_m=450.0, crest_tpi_min_m=10.0, crest_min_cells=3,
    plateau_slope_max_deg=8.0, plateau_min_cells=25, plateau_tpi1500_min_m=0.0,   # plateau tops must be elevated (not valley floors)
    connectivity=8,
    thinning_block_m=1000.0,       # one control per 1 km block: cells inside a block are NOT independent controls
    block_min_cells=3, dominant_unit_share_min=0.60,   # below -> Q3 (ambiguous unit support)
    summit_percentiles=(90, 95, 99), central_percentile=95,
    coherence_radius_m=3000.0, coherence_min_neighbours=5, tau_fraction=0.05, tau_min_m=20.0,  # accordance with neighbours (Q2 -> Q3 if violated)
    radial_sector_deg=45.0,        # outward-decreasing check along each centre sector (radial families only)
    vent_core_radius_m=500.0, rim_band_m=2000.0, summit_search_m=1000.0,
    min_controls=12,
)
QUALITY = {"Q1": "registered-centre summits; caldera-rim crests; plateau tops of mapped depositional units",
           "Q2": "accordant interfluve crests on mapped constructional units",
           "Q3": "crests failing unit dominance / accordance / outward-decrease (diagnostic only; never fitted)",
           "Q4": "catalogue/literature elevations only (never fitted)"}
CALDERA = {
    "KUMAMOTO_ASO_2016": dict(method="GSJ_ASO_POST_CALDERA_UNITS", exclude_symbols_legend05e=["Air-fall ash"], exclude_legend04e=["Kuju Volcano"],
                              legend04e="Aso Volcano Post-caldera volcanoes and deposits", closing_m=500.0, buffer_m=500.0,
                              note="caldera interior = union of mapped post-caldera volcanoes/deposits (air-fall ash excluded because it mantles the outer flanks), closed, holes filled, +500 m to cover the lower inner wall"),
    "LOMBOK_RINJANI_2018": dict(method="WBM_LAKE_RAY_RIM", search_km=10.0, rim_max_km=7.0, n_rays=72, wbm="cop30_wbm_30m.tif", lake_class=2,
                                note="Segara Anak/Samalas caldera: centre = centroid of the largest Copernicus WBM lake (class 2) within 10 km of the registered Rinjani centre; "
                                     "rim = highest DEM cell along each of 72 rays within 7 km; interior = inside the rim on every ray"),
}

# ------------------------------------------------------------------------------------------------ Part IV: control-geometry verdict
CONTROL_GEOMETRY = dict(
    radial=dict(sectors=8, min_occupied_sectors=4, min_per_occupied_sector=2, max_sector_share=0.50),
    envelope=dict(block_m=2000.0, min_blocks=6, max_block_share=0.25),
    any_control_in_excluded_zone="FAIL",
)
# Cotabato centre-geometry gate (4C-A left it qualitative; frozen here before any fit).  All must hold for RADIAL; else ENVELOPE.
CENTRE_GEOMETRY_GATE = dict(flank_radius_km=30.0, min_unit_fraction_within=0.75, max_spearman_elev_vs_distance=-0.5,
                            min_sectors_with_3_controls=4,
                            meaning="the applicable units lie on one registered centre's flank; preserved crests decrease outward from it; "
                                    "angular control coverage is sufficient for leave-one-sector-out validation")

# ------------------------------------------------------------------------------------------------ Part V: scenario ensembles (equal weights)
ENVELOPE_SCENARIOS = dict(trend_order=(1, 2), summit_percentile=(90, 95, 99), tolerance_offset_sigma=(-1.0, 0.0, 1.0),
                          control_subsample_replicates=8, subsample_fraction=0.8)
RADIAL_SCENARIOS = dict(p=(1.0, 1.5, 2.0, 2.5), radius_factor=(1.1, 1.3, 1.6), summit_percentile=(90, 95, 99), tolerance_offset_sigma=(-1.0, 0.0, 1.0),
                        coalescence=("UPPER_ENVELOPE", "ADDITIVE_SUPERPOSITION"), control_subsample_replicates=2, subsample_fraction=0.8,
                        radius_definition="R_k = radius_factor x P95 radial distance of the fit controls assigned to centre k",
                        asymmetry="H(theta) = c0 + c1 cos(theta) + c2 sin(theta), fitted only for centres with >= 6 controls; |(c1,c2)|/c0 <= 0.5 or the scenario is rejected",
                        model="Z = B + max_k / sum_k [ (1 - r_k/R_k)_+^p * H_k(theta) ]  (shared base B; linear least squares given p, R, assignment); single-centre landscapes have no coalescence dimension")
FIT = dict(huber_k=1.345, irls_iterations=20, sigma="1.4826 x MAD of in-sample residuals (floor 1 m)", upper_envelope_iterations=5,
           tolerance_offset="scenario surface shifted by offset x sigma (control tolerance; 4C-A 'control tolerance' / 'surface tolerance')",
           coordinates="envelope polynomials in centred coordinates scaled by the control-set half-diagonal")

# ------------------------------------------------------------------------------------------------ Part VII: support
SUPPORT = dict(max_control_distance_m=3000.0, radial_max_r="P95 radial distance of fit controls of the assigned centre",
               excludes=["caldera interior", "vent cores (0.5 km of every registered centre)", "non-applicable geology", "invalid DEM"],
               grid="90 m unit grid of the full reconstruction window (Stage 4A unit cells are a subset)",
               zero_fill="FORBIDDEN: H outside support is missing")

# ------------------------------------------------------------------------------------------------ Part VIII: validation (gates verbatim from 4C-A)
VALIDATION = dict(radial="leave-one-sector-out: 8 x 45 deg sectors about each control's nearest registered centre",
                  envelope="leave-one-2-km-block-out (blocks on a fixed 2,000 m grid)",
                  hybrid="each component validated with its own scheme; held-out residuals pooled for the landscape gates",
                  residual="z_control(P95 summit) - median_j Zref_j(x) of scenarios refitted without the held-out group",
                  bias="mean held-out residual", band="held-out z_control inside [P05, P95] over scenarios",
                  negative_removal="fraction of supported Stage 4A units (outside mapped young deposits) with median_j(Zref_j - Zmod) < -tau_L, "
                                   "tau_L = median scenario control tolerance (sigma)",
                  slope_statistic="maximum over supported Stage 4A units of median_j slope(Zref_j) (90 m Horn; the same quantity entering H_SLOPE_CHANGE_DEG); P99 also reported")
GATES = dict(median_abs_residual_fraction_max=0.10, abs_bias_fraction_max=0.05, holdout_band_fraction_min=0.60, min_admissible_controls=12,
             min_support_fraction_of_units=0.25, negative_removal_fraction_max_outside_mapped_young_deposits=0.10, max_reconstructed_slope_deg_constructional=40.0)
FAILURE_CODES = ["INSUFFICIENT_CONTROLS", "POOR_CONTROL_GEOMETRY", "HIGH_RESIDUAL", "HIGH_BIAS", "LOW_SCENARIO_COVERAGE", "LOW_SUPPORT",
                 "EXCESS_NEGATIVE_REMOVAL", "IMPLAUSIBLE_SLOPE", "FAMILY_MISMATCH", "INSUFFICIENT_GEOLOGICAL_EVIDENCE", "OTHER"]
STATUS_RULE = dict(VERIFIED_FULL="every gate passes and supported unit fraction >= 0.90", VERIFIED_PARTIAL="every gate passes and 0.25 <= supported fraction < 0.90",
                   AGE_PROXY_ONLY="family AGE_HISTORY_PROXY (no palaeosurface)", FAILED_H="any gate fails, the control-geometry verdict is FAIL, or every scenario is rejected")

# ------------------------------------------------------------------------------------------------ Part X: CORE_H operators
H_OPERATORS = dict(grid="90 m unit grid (Stage 4A 3x3 blocks)", z_modern="mean of the valid 30 m M-DEM cells of the unit",
                   slope="Horn gradient at 90 m spacing, identical operator for modern and every reference scenario",
                   relief450="max - min in a 450 m-radius disk on the 90 m grid, identical operator for modern and reference",
                   R_L="P98 - P02 of the scenario-median reference over the support mask (90 m window grid)",
                   Lambda_L="sqrt(support-mask area / pi)", control_distance="Euclidean distance from the unit centre to the nearest fitted (Q1/Q2) control",
                   column_prefix="h_ (Stage 4A H contract); h_available = supported unit with all six CORE_H finite")
CONFIDENCE_QA = dict(HIGH="h_control_distance_fraction <= 0.25 and h_scenario_spread_fraction <= 0.05",
                     LOW="h_control_distance_fraction > 0.5 or h_scenario_spread_fraction > 0.15", MEDIUM="otherwise")
REDUNDANCY = dict(method="pooled OLS R^2 of each CORE_H on the 60 frozen M predictors (standardised), H-available rows of verified landscapes; no Y",
                  threshold=0.90)
