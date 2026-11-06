"""Stage 4C-A: frozen H (landscape-history) specification and feasibility for the six development landscapes.

Designed from geological/geomorphological principles and prior evidence (Stage 2C/2D/2E/2F.2/3C/3G records, V1 forensic audit).
No landslide target (Y), no Stage 4B residual, prediction or feature-importance result was used to define anything in this module.
"""
from __future__ import annotations

LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
STATUSES = ("FULL_RECONSTRUCTION", "PARTIAL_RECONSTRUCTION", "AGE_PROXY_ONLY", "H_UNAVAILABLE")

FAMILIES = [  # extends the Stage 2E Gen-II family registry; names kept where they already existed
    dict(family_id="RADIAL_MULTI_CENTRE_FLANK", stage2e_family="RADIAL_FLANK", applies_to="young constructional cones/composite centres with surviving flanks",
         reference_surface="radial macro-surface per independently registered centre (Z = B + H(theta) F(r/R)), combined by upper envelope or nearest scaled centre",
         controls="registered centre(s) and summit elevations, preserved flank-remnant crest envelopes on mapped young constructional units",
         semantics="ESTIMATED_REMOVAL relative to the constructional flank surface", tweed_concepts="radial profiles; centre-constrained surface; remnant crests; angular support"),
    dict(family_id="CALDERA_RIM_CONSTRAINED", stage2e_family="CALDERA_RIM_CONSTRAINED", applies_to="caldera complexes (combined with an outer-flank family)",
         reference_surface="outer flank/plateau surface with the rim crest as a one-sided lower bound; caldera interior is MASKED (collapse is not erosion)",
         controls="DEM-derived rim crest on mapped caldera-wall units, rim elevations", semantics="caldera interior excluded; outside rim per partner family",
         tweed_concepts="core/caldera exclusion; remnant controls"),
    dict(family_id="PRESERVED_REMNANT_ENVELOPE", stage2e_family="PRESERVED_REMNANT_ENVELOPE", applies_to="dissected plateaux, ignimbrite sheets and eroded edifices without a defensible single centre",
         reference_surface="smooth low-order envelope through accordant interfluve summits of one mapped constructional/depositional unit group",
         controls="accordant summits on mapped constructional units, plateau tops", semantics="MINIMUM_REMOVAL (envelope is a lower bound on the former surface)",
         tweed_concepts="remnant/plateau controls; erosion depth (as a minimum); uncertainty"),
    dict(family_id="AGE_HISTORY_PROXY", stage2e_family="AGE_HISTORY_PROXY", applies_to="old, faulted or uplifted volcanic piles with no surviving constructional surface",
         reference_surface="NONE (no palaeosurface claimed)", controls="mapped unit ages only", semantics="age/history fields only; all surface-based H missing",
         tweed_concepts="geological classification only"),
]

# Per-landscape taxonomy: evidence sources are V2 records (paths) or explicitly labelled literature/general knowledge to be verified in 4C-B.
TAXONOMY = [
    dict(landscape_id="KUMAMOTO_ASO_2016", volcano_type="Caldera complex (Aso) with outer pyroclastic-flow plateau and post-caldera central cones",
         tectonic_context="Subduction arc (SW Japan)", relevant_age="Caldera-forming eruptions Aso-1..4 (Middle-Late Pleistocene; Aso-4 ~90 ka per literature); post-caldera cones younger",
         erosional_state="Mixed: young central cones; moderately dissected ignimbrite plateau", structural_complications="Caldera collapse; multiple post-caldera centres; 2016 Futagawa fault system",
         caldera_history="Yes: caldera collapse must be masked, not treated as erosion", radial_plausible="NO for the domain as a whole (Stage 2E: single-centre radial rejected)",
         family="PRESERVED_REMNANT_ENVELOPE (+ CALDERA_RIM_CONSTRAINED mask)", temporal_target="Depositional top of the youngest mapped caldera-forming pyroclastic-flow unit outside the caldera",
         confidence="MEDIUM", evidence_paths="outputs/stage2c/geology_mapping_KUMAMOTO_ASO_2016_aso_1_50000.csv; outputs/stage2e/gen2/reconstruction_method.json"),
    dict(landscape_id="NOTO_PENINSULA_2024", volcano_type="Old eroded Oligocene-Miocene volcanic terrane (no edifice)", tectonic_context="Japan arc / back-arc crustal; active reverse faulting and uplift",
         relevant_age="Late Oligocene-Early Miocene volcanic units (Seamless ages)", erosional_state="Old, deeply eroded; uplifted marine terraces (non-volcanic surfaces)",
         structural_complications="Uplift and faulting since emplacement; no constructional geometry survives", caldera_history="None identified", radial_plausible="NO",
         family="AGE_HISTORY_PROXY", temporal_target="NONE (no palaeosurface)", confidence="HIGH",
         evidence_paths="outputs/stage2c/h_feasibility.csv; outputs/stage2e/gen2/h_missingness.json"),
    dict(landscape_id="LOMBOK_RINJANI_2018", volcano_type="Young stratovolcano complex (Rinjani, Pusuk, Nangi) with Samalas caldera (1257 CE)", tectonic_context="Subduction arc (Sunda)",
         relevant_age="Holocene-Late Pleistocene edifice units (Qhv, Qvl); older Tertiary units on the southern margin", erosional_state="Young-to-mature; flanks largely constructional",
         structural_complications="1257 CE caldera collapse; multiple centres; older dissected Tertiary units", caldera_history="Yes (Segara Anak caldera)",
         radial_plausible="YES for young edifice flanks only", family="RADIAL_MULTI_CENTRE_FLANK (+ CALDERA_RIM_CONSTRAINED mask)",
         temporal_target="Constructional flank surface of the mapped young edifice units (pre-1257 cone NOT restored in the primary reconstruction)", confidence="MEDIUM",
         evidence_paths="outputs/stage2f2/h_feasibility_update.json; outputs/stage2f2/sheet_legend_evidence.csv"),
    dict(landscape_id="COTABATO_DAVAO_2019", volcano_type="Dissected Pliocene-Quaternary volcanic arc terrane near composite centres", tectonic_context="Arc / collision / subduction margin (Mindanao)",
         relevant_age="Pliocene-Quaternary volcanic units (MGB); minor Oligocene-Miocene", erosional_state="Mature dissected", structural_complications="Active faulting (2019 sequence); coarse regional geology (scale unstated)",
         caldera_history="Not established", radial_plausible="UNCERTAIN: depends on whether the domain lies on one centre's constructional flank (to verify with a registered centre catalogue)",
         family="RADIAL_MULTI_CENTRE_FLANK if centre geometry passes, else PRESERVED_REMNANT_ENVELOPE", temporal_target="Constructional flank surface of mapped Pliocene-Quaternary units",
         confidence="LOW", evidence_paths="outputs/stage3d/COTABATO_GEOLOGY_MAPPING.csv; outputs/stage3c/STAGE3C_CANDIDATE_AUDIT.tsv"),
    dict(landscape_id="ATITLAN_STAN_2005", volcano_type="Caldera complex (Atitlan) with Tertiary volcanic highland and Quaternary pumice fills", tectonic_context="Subduction arc (Central America)",
         relevant_age="Tertiary (Mio-Pliocene) volcanics; Quaternary pumice-ash fills", erosional_state="Mature dissected; valleys re-incised into pumice fills",
         structural_complications="Two-stage history (highland dissection, pumice infill, re-incision); coarse 1:500,000 geology with 2 units; NASADEM M",
         caldera_history="Yes (caldera lies largely outside the production domain; any interior is masked)", radial_plausible="NO",
         family="PRESERVED_REMNANT_ENVELOPE", temporal_target="Accordant-summit envelope of the mapped volcanic highland/fill units (minimum-removal)", confidence="LOW",
         evidence_paths="outputs/stage3e/ATITLAN_GEOLOGY_AUDIT.json; outputs/stage3f/ATITLAN_GEOLOGY_HARMONISATION.csv"),
    dict(landscape_id="DOMINICA_ERIKA_2015", volcano_type="Volcanic island complex with several Pleistocene-Recent centres and older Pliocene-Miocene units", tectonic_context="Subduction arc (Lesser Antilles)",
         relevant_age="Age-coded map: Miocene, Pliocene, older and younger Pleistocene-Recent", erosional_state="Young-to-dissected",
         structural_complications="Multiple centres and domes; digitised map (RMS ~24 m); deep tropical weathering", caldera_history="Not a single caldera system",
         radial_plausible="YES for young centres only", family="RADIAL_MULTI_CENTRE_FLANK (young centres) + PRESERVED_REMNANT_ENVELOPE (older units)",
         temporal_target="Constructional surfaces of the mapped Pleistocene-Recent centres; envelope (minimum-removal) on older units", confidence="MEDIUM",
         evidence_paths="outputs/stage3f/DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv; outputs/stage3f/DOMINICA_ERIKA_GEOREFERENCE_QA.json"),
]

FEASIBILITY = {  # expected status after 4C-B construction + Y-independent validation; nothing is reconstructed yet
    "KUMAMOTO_ASO_2016": ("PARTIAL_RECONSTRUCTION", "Outer pyroclastic-flow plateau only; caldera interior and post-caldera cones outside support; Stage 2E numeric-control gate (>=12 admissible height controls) must pass"),
    "NOTO_PENINSULA_2024": ("AGE_PROXY_ONLY", "No surviving constructional surface; uplift/faulting; marine terraces are not volcanic palaeosurfaces"),
    "LOMBOK_RINJANI_2018": ("PARTIAL_RECONSTRUCTION", "Young edifice flanks; Samalas caldera interior masked; older Tertiary margin outside support"),
    "COTABATO_DAVAO_2019": ("PARTIAL_RECONSTRUCTION", "Conditional: centre-geometry gate and coarse-geology risk; status falls to AGE_PROXY_ONLY if the gate fails"),
    "ATITLAN_STAN_2005": ("PARTIAL_RECONSTRUCTION", "Minimum-removal envelope only; two-unit 1:500,000 geology; status falls to AGE_PROXY_ONLY if accordant-control gate fails"),
    "DOMINICA_ERIKA_2015": ("PARTIAL_RECONSTRUCTION", "Young centres (radial) + older units (envelope); centre registry required"),
}

EVIDENCE = [  # (landscape, evidence, kind OBSERVED_CONSTRAINT/MODEL_ASSUMPTION, availability, provenance)
    ("KUMAMOTO_ASO_2016", "GSJ Aso 1:50,000 unit polygons (caldera wall, pyroclastic-flow units, post-caldera cones)", "OBSERVED_CONSTRAINT", "AVAILABLE", "outputs/stage2c/geology_mapping_KUMAMOTO_ASO_2016_aso_1_50000.csv"),
    ("KUMAMOTO_ASO_2016", "Accordant interfluve summits on mapped caldera-forming pyroclastic-flow units (DEM)", "OBSERVED_CONSTRAINT", "EXTRACTABLE_Y_BLIND", "Copernicus DEM (Stage 2C)"),
    ("KUMAMOTO_ASO_2016", "Caldera rim crest from DEM on mapped wall units", "OBSERVED_CONSTRAINT", "EXTRACTABLE_Y_BLIND", "Copernicus DEM + Aso map"),
    ("KUMAMOTO_ASO_2016", "Envelope smoothness order (1st/2nd-order trend) and summit percentile", "MODEL_ASSUMPTION", "PRE-REGISTERED_ALTERNATIVES", "this specification"),
    ("KUMAMOTO_ASO_2016", "Numeric eruption ages", "OBSERVED_CONSTRAINT", "MISSING (literature only)", "Stage 2E temporal-target note"),
    ("NOTO_PENINSULA_2024", "Seamless unit ages (Oligocene-Miocene)", "OBSERVED_CONSTRAINT", "AVAILABLE", "outputs/stage2c/geology_mapping_NOTO_PENINSULA_2024_seamless.csv"),
    ("NOTO_PENINSULA_2024", "Preserved constructional surfaces", "OBSERVED_CONSTRAINT", "ABSENT", "outputs/stage2c/h_feasibility.csv"),
    ("LOMBOK_RINJANI_2018", "Sheet-1807 edifice units Qhv(r/p/n), Qvl; relative stratigraphy", "OBSERVED_CONSTRAINT", "AVAILABLE", "outputs/stage2f2/sheet_legend_evidence.csv"),
    ("LOMBOK_RINJANI_2018", "Caldera rim (DEM morphology) and summit elevations", "OBSERVED_CONSTRAINT", "EXTRACTABLE_Y_BLIND", "Copernicus DEM"),
    ("LOMBOK_RINJANI_2018", "Registered volcano centre coordinates (Rinjani, Pusuk, Nangi)", "OBSERVED_CONSTRAINT", "TO_ACQUIRE (public volcano catalogue)", "4C-B task"),
    ("LOMBOK_RINJANI_2018", "Radial profile exponent p, radius R, asymmetry, centre coalescence rule", "MODEL_ASSUMPTION", "PRE-REGISTERED_BOUNDS", "this specification"),
    ("LOMBOK_RINJANI_2018", "Numeric unit ages", "OBSERVED_CONSTRAINT", "MISSING", "outputs/stage2f2/h_feasibility_update.json"),
    ("COTABATO_DAVAO_2019", "MGB regional geology (Pliocene-Quaternary vs Oligocene-Miocene)", "OBSERVED_CONSTRAINT", "AVAILABLE (scale unstated)", "outputs/stage3d/COTABATO_GEOLOGY_MAPPING.csv"),
    ("COTABATO_DAVAO_2019", "Registered centre coordinates and summit elevations", "OBSERVED_CONSTRAINT", "TO_ACQUIRE", "4C-B task"),
    ("COTABATO_DAVAO_2019", "Flank-remnant crest envelopes", "OBSERVED_CONSTRAINT", "EXTRACTABLE_Y_BLIND", "Copernicus DEM"),
    ("ATITLAN_STAN_2005", "MAGA 1:500,000 geology (Qp pumice fills, Tv volcanics)", "OBSERVED_CONSTRAINT", "AVAILABLE", "outputs/stage3e/ATITLAN_GEOLOGY_AUDIT.json"),
    ("ATITLAN_STAN_2005", "Accordant summits on volcanic highland/fill surfaces", "OBSERVED_CONSTRAINT", "EXTRACTABLE_Y_BLIND", "NASADEM"),
    ("ATITLAN_STAN_2005", "Upper bound on former surface", "OBSERVED_CONSTRAINT", "ABSENT (minimum-removal only)", "V1 SF-4F audit finding"),
    ("DOMINICA_ERIKA_2015", "Age-coded 1:100,000 geology incl. dome units (centre indicators)", "OBSERVED_CONSTRAINT", "AVAILABLE", "outputs/stage3f/DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv"),
    ("DOMINICA_ERIKA_2015", "Registered centre coordinates", "OBSERVED_CONSTRAINT", "TO_ACQUIRE", "4C-B task"),
    ("DOMINICA_ERIKA_2015", "Flank crests on Pleistocene-Recent units; accordant summits on older units", "OBSERVED_CONSTRAINT", "EXTRACTABLE_Y_BLIND", "Copernicus DEM"),
    ("ALL", "Published cross sections / palaeogeomorphic interpretations", "OBSERVED_CONSTRAINT", "NOT_ASSEMBLED (optional validation)", "literature search in 4C-B"),
]

# Candidate H families (Part VIII): evaluated before any H exists.  redundancy = expectation from physics, to be measured Y-free in 4C-B.
CANDIDATES = [
    # name, physical meaning, transferability, size_dependence, family_dependence, redundancy, uncertainty_sensitivity, leakage, availability, decision
    ("reconstructed_elevation_m", "Absolute height of the reference surface", "LOW", "HIGH (metres)", "HIGH", "MOSTLY_REDUNDANT", "MEDIUM", "NONE", "all reconstructed", "REJECTED"),
    ("removed_thickness_m (Z_ref - Z_mod)", "Material removed (or added) since the temporal target", "LOW", "HIGH (scales with edifice)", "MEDIUM (minimum vs estimated)", "PARTLY_REDUNDANT", "HIGH", "NONE", "all reconstructed", "REJECTED_AS_RAW (normalised form is CORE)"),
    ("removed_fraction_of_relief", "Removed thickness divided by landscape reconstructed relief", "MEDIUM-HIGH", "LOW", "MEDIUM", "HISTORY_UNIQUE", "HIGH", "NONE", "all reconstructed", "CORE"),
    ("reference_relative_height", "Position on the reconstructed surface between its base and top (0-1)", "HIGH", "LOW", "LOW", "PARTLY_REDUNDANT", "MEDIUM", "NONE", "all reconstructed", "CORE"),
    ("radial_position_norm", "Distance from registered centre / reconstructed edifice radius", "HIGH within radial families", "LOW", "HIGH (radial only)", "PARTLY_REDUNDANT", "MEDIUM", "centre must not use Y", "radial families only", "OPTIONAL"),
    ("reconstructed_local_slope_deg", "Steepness of the reference surface", "MEDIUM", "LOW", "HIGH (cone vs plateau)", "PARTLY_REDUNDANT", "MEDIUM", "NONE", "all reconstructed", "OPTIONAL"),
    ("reconstructed_local_relief", "Relief of the smooth reference surface in a window", "LOW", "HIGH", "HIGH", "MOSTLY_REDUNDANT", "LOW", "NONE", "all reconstructed", "REJECTED"),
    ("reconstructed_curvature", "Curvature of a smooth macro-surface (near zero by construction)", "LOW", "MEDIUM", "HIGH", "MOSTLY_REDUNDANT", "HIGH", "NONE", "all reconstructed", "REJECTED"),
    ("slope_change_deg (modern - reconstructed)", "Steepening of the present slope relative to the constructional surface", "MEDIUM-HIGH", "LOW (angles)", "MEDIUM", "PARTLY_REDUNDANT", "MEDIUM", "NONE", "all reconstructed", "CORE"),
    ("relief_change_norm", "Dissection: (modern - reconstructed local relief, 450 m) / landscape relief", "MEDIUM", "LOW", "MEDIUM", "PARTLY_REDUNDANT", "MEDIUM", "NONE", "all reconstructed", "CORE"),
    ("curvature_change", "Modern minus reference curvature (~ modern curvature)", "LOW", "MEDIUM", "LOW", "MOSTLY_REDUNDANT", "HIGH", "NONE", "all reconstructed", "REJECTED"),
    ("local_surface_difference_500m (V1)", "Removed thickness minus its 500 m mean (~ -TPI of the modern surface)", "LOW", "HIGH", "LOW", "MOSTLY_REDUNDANT", "MEDIUM", "NONE", "all reconstructed", "REJECTED"),
    ("scenario_spread_norm", "IQR of the reference surface across ensemble scenarios / landscape relief", "MEDIUM", "LOW", "MEDIUM", "HISTORY_UNIQUE", "n/a (is uncertainty)", "NONE", "all reconstructed", "CORE"),
    ("control_distance_norm", "Distance to nearest admissible control / landscape size scale", "MEDIUM", "LOW", "LOW", "HISTORY_UNIQUE", "n/a (is support)", "NONE (controls are Y-blind)", "all reconstructed", "CORE"),
    ("parameter_vs_structural_spread", "Decomposed uncertainty components", "LOW", "LOW", "HIGH", "HISTORY_UNIQUE", "n/a", "NONE", "multi-scenario families", "OPTIONAL"),
    ("flank_sector_sin_cos", "Orientation of the reconstructed flank", "LOW", "LOW", "HIGH (radial only)", "MOSTLY_REDUNDANT (aspect in M)", "LOW", "NONE", "radial only", "REJECTED"),
    ("time_since_emplacement_years", "Exposure/weathering time from numeric unit ages", "HIGH", "LOW", "LOW", "HISTORY_UNIQUE", "HIGH (broad ages)", "NONE", "only where numeric ages exist (currently none)", "OPTIONAL"),
    ("external_expected_removal_anomaly", "Removal relative to an external analogue population (Stage 2E DeltaD_anomaly)", "HIGH", "LOW", "LOW", "HISTORY_UNIQUE", "HIGH", "self-reference forbidden", "requires external analogue population (absent)", "OPTIONAL_DEFERRED"),
]

CORE_H = [  # 6 predictors; frozen here; never redefined by later predictive results
    dict(name="H_REMOVED_FRACTION", group="HISTORICAL_CHANGE", formula="median_j(Z_ref_j - Z_mod) / R_L",
         definition="Material removed (positive) or added (negative) since the landscape's temporal target, as a fraction of the landscape's reconstructed relief",
         units="dimensionless", normalization="R_L = P98 - P02 of the scenario-median reference surface inside the support mask",
         meaning="How much of the constructional landscape has been stripped at this location", inputs="reference-surface ensemble; modern DEM (same as M)",
         availability="inside support mask; family semantics flag recorded (MINIMUM_REMOVAL for envelope families)", uncertainty="P05/P95 kept as QA columns",
         redundancy="HISTORY_UNIQUE", leakage="PASS (Y-blind controls; pre-event DEM)"),
    dict(name="H_SLOPE_CHANGE_DEG", group="HISTORICAL_CHANGE", formula="slope(Z_mod, 90 m) - median_j slope(Z_ref_j, 90 m)",
         definition="Present slope minus the slope of the reconstructed surface at the same place", units="degrees",
         normalization="angles are scale-free; no further normalisation", meaning="Over-steepening by incision relative to the original flank",
         inputs="reference-surface ensemble; modern DEM", availability="inside support mask", uncertainty="member-wise then summarised",
         redundancy="PARTLY_REDUNDANT", leakage="PASS"),
    dict(name="H_RELIEF_CHANGE_FRACTION", group="HISTORICAL_CHANGE", formula="(relief450(Z_mod) - median_j relief450(Z_ref_j)) / R_L",
         definition="Local dissection: increase of 450 m-window relief over the reconstructed surface, as a fraction of landscape relief", units="dimensionless",
         normalization="R_L", meaning="How strongly the original surface has been dissected around the unit", inputs="reference ensemble; modern DEM",
         availability="inside support mask", uncertainty="member-wise", redundancy="PARTLY_REDUNDANT", leakage="PASS"),
    dict(name="H_REFERENCE_RELATIVE_HEIGHT", group="RECONSTRUCTED_POSITION", formula="(median_j Z_ref_j - P02_L) / R_L",
         definition="Normalised position on the reconstructed surface (0 = its base, 1 = its top)", units="dimensionless", normalization="P02_L and R_L of the scenario-median surface",
         meaning="Palaeogeomorphic position (summit-flank-toe) independent of volcano size", inputs="reference ensemble", availability="inside support mask",
         uncertainty="P05/P95 kept as QA", redundancy="PARTLY_REDUNDANT", leakage="PASS"),
    dict(name="H_SCENARIO_SPREAD_FRACTION", group="UNCERTAINTY_SUPPORT", formula="IQR_j(Z_ref_j) / R_L",
         definition="Disagreement among plausible reconstruction scenarios, as a fraction of landscape relief", units="dimensionless", normalization="R_L",
         meaning="How weakly the earlier surface is constrained here (scenario spread, not a probability)", inputs="reference ensemble",
         availability="inside support mask", uncertainty="is the uncertainty term", redundancy="HISTORY_UNIQUE", leakage="PASS"),
    dict(name="H_CONTROL_DISTANCE_FRACTION", group="UNCERTAINTY_SUPPORT", formula="d(x, nearest admissible control) / Lambda_L",
         definition="Distance to the nearest admissible reconstruction control, in units of the landscape size scale", units="dimensionless",
         normalization="Lambda_L = sqrt(area(support mask) / pi)", meaning="Extrapolation distance from direct evidence", inputs="control set; support mask",
         availability="inside support mask", uncertainty="support term", redundancy="HISTORY_UNIQUE", leakage="PASS (controls never selected with Y)"),
]
OPTIONAL_H = [c for c in CANDIDATES if c[-1].startswith("OPTIONAL")]
REJECTED_H = [c for c in CANDIDATES if c[-1].startswith("REJECTED") and "CORE" not in c[-1]]

SELECTED_H_RULE = dict(
    principle="Y-independent; decided from reconstruction evidence only; never 'keep what improves held-out PR-AUC'",
    feature_rule=("A CORE_H feature enters SELECTED_H only if (a) its pooled linear R^2 on the 60 M predictors (computed without Y, on H-capable rows) is < 0.90, "
                  "(b) it is available in every H-capable landscape, and (c) it is not flagged by the reconstruction-validation gate"),
    row_rule="SELECTED_H rows: H_CONTROL_DISTANCE_FRACTION <= 0.5 (within half a landscape size scale of direct evidence); the same rows are used for MEG and MEGH",
    landscape_rule="A landscape is H-capable only if its reconstruction passes every Y-independent validation gate in STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md",
    frozen_before_any_h_result=True,
)

VALIDATION_GATES = dict(
    holdout="Spatially blocked holdout of admissible controls: leave-one-sector-out (radial, 8 sectors) or leave-one-2-km-block-out (envelope); fit without the held-out controls",
    metrics=["median |vertical residual| / R_L", "signed bias / R_L", "fraction of held-out controls inside the scenario P05-P95 band", "residual by control quality class and distance"],
    gates=dict(median_abs_residual_fraction_max=0.10, abs_bias_fraction_max=0.05, holdout_band_fraction_min=0.60, min_admissible_controls=12, min_support_fraction_of_units=0.25,
               negative_removal_fraction_max_outside_mapped_young_deposits=0.10, max_reconstructed_slope_deg_constructional=40.0),
    geometric_checks=["no reference surface below preserved remnants beyond the control tolerance", "no unsupported summit higher than the highest admissible control + tolerance",
                      "caldera interiors masked, never scored as erosion", "edifice dimensions within the range of the landscape's own controls",
                      "removed-volume total reported with scenario P05-P95 (no probability claim)"],
    forbidden=["landslide occurrence or Y in any form", "Stage 4B residuals, predictions or feature importance", "tuning to performance"],
)

TWEED_CONCEPTS = {  # applicability per landscape: YES / PARTIAL / NO
    "radial_profiles": dict(KUMAMOTO_ASO_2016="NO", NOTO_PENINSULA_2024="NO", LOMBOK_RINJANI_2018="YES", COTABATO_DAVAO_2019="PARTIAL", ATITLAN_STAN_2005="NO", DOMINICA_ERIKA_2015="PARTIAL"),
    "centre_constrained_surface": dict(KUMAMOTO_ASO_2016="NO", NOTO_PENINSULA_2024="NO", LOMBOK_RINJANI_2018="YES", COTABATO_DAVAO_2019="PARTIAL", ATITLAN_STAN_2005="NO", DOMINICA_ERIKA_2015="PARTIAL"),
    "preserved_ridge_remnant_controls": dict(KUMAMOTO_ASO_2016="YES", NOTO_PENINSULA_2024="NO", LOMBOK_RINJANI_2018="YES", COTABATO_DAVAO_2019="YES", ATITLAN_STAN_2005="YES", DOMINICA_ERIKA_2015="YES"),
    "geological_classification": {l: "YES" for l in LANDSCAPES},
    "erosion_depth": dict(KUMAMOTO_ASO_2016="YES (minimum)", NOTO_PENINSULA_2024="NO", LOMBOK_RINJANI_2018="YES", COTABATO_DAVAO_2019="YES", ATITLAN_STAN_2005="YES (minimum)", DOMINICA_ERIKA_2015="YES"),
    "reconstruction_uncertainty": dict(KUMAMOTO_ASO_2016="YES", NOTO_PENINSULA_2024="NO", LOMBOK_RINJANI_2018="YES", COTABATO_DAVAO_2019="YES", ATITLAN_STAN_2005="YES", DOMINICA_ERIKA_2015="YES"),
    "angular_support": dict(KUMAMOTO_ASO_2016="NO", NOTO_PENINSULA_2024="NO", LOMBOK_RINJANI_2018="YES", COTABATO_DAVAO_2019="PARTIAL", ATITLAN_STAN_2005="NO", DOMINICA_ERIKA_2015="PARTIAL"),
}
