"""Stage 3A: landscape-space coverage matrix, gap analysis, transparent candidate priority scoring, tiers and lockbox planning.

Every attribute is either copied from earlier V2 candidate metadata or set by an explicit CURATED record that carries its
evidence.  Anything not evidenced stays UNKNOWN/UNCERTAIN.  No score uses Y prevalence or expected model performance.
"""
from __future__ import annotations

import csv

from .schema import ROOT

AS_OF = "2026-10-06"
PRODUCTION = ("KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018")
DASHBOARD = ROOT / "outputs/stage2f2/landscape_state_dashboard.csv"
CANDIDATES = ROOT / "metadata/stage2a2_landscape_candidates.csv"

AXES = ("morphology", "erosion_state", "climate", "tectonic_setting", "trigger", "continent", "material")
MORPHOLOGY = ("STRATOVOLCANO", "SHIELD", "CALDERA", "VOLCANIC_COMPLEX", "VOLCANIC_PLATEAU", "OLD_MASSIF_REMNANT", "MIXED_OTHER")
EROSION = ("YOUNG_INTACT", "MODERATELY_DISSECTED", "STRONGLY_ERODED", "ANCIENT_REMNANT", "UNCERTAIN")
CLIMATE = ("TROPICAL_WET", "TROPICAL_SEASONAL", "TEMPERATE_WET", "TEMPERATE_SEASONAL", "DRY_SEMI_ARID", "HIGH_ALTITUDE_ALPINE",
           "OTHER_UNCERTAIN")
TECTONIC = ("SUBDUCTION_ARC", "HOTSPOT_OCEAN_ISLAND", "CONTINENTAL_HOTSPOT", "CONTINENTAL_RIFT", "BACK_ARC", "INTRAPLATE",
            "OTHER_UNCERTAIN")
TRIGGER = ("EARTHQUAKE", "EXTREME_RAINFALL", "CYCLONE_STORM", "PROLONGED_RAINFALL", "VOLCANIC_ACTIVITY", "MIXED", "UNCLEAR")
TIER = ("TIER_1", "TIER_2", "BOTH_AVAILABLE", "UNCLEAR")
MATERIAL = ("BASALTIC", "ANDESITIC", "DACITIC_RHYOLITIC", "PYROCLASTIC_DOMINATED", "MIXED_VOLCANIC", "HYDROTHERMALLY_ALTERED",
            "SEDIMENTARY_INTERBEDDED", "OTHER_UNCERTAIN")
H_CLASSES = ("FULL_RECONSTRUCTION", "PARTIAL_RECONSTRUCTION", "AGE_PROXY_ONLY", "H_UNAVAILABLE", "NOT_ASSESSED")
LOCKBOX = ("POTENTIAL_LOCKBOX", "DEVELOPMENT_PREFERRED", "UNDECIDED", "INELIGIBLE_V1_HISTORICAL")
UNKNOWNS = {"UNCERTAIN", "OTHER_UNCERTAIN", "UNCLEAR", "UNKNOWN", "MIXED_OTHER"}

# Gate status vocabulary -> 0..3 quality points (used for B, C, E).  FAIL is a hard blocker.
STATUS_POINTS = {"PASS": 3, "PROVISIONAL_DECISION_REQUIRED": 2, "PUBLIC_AVAILABLE_UNAUDITED": 2, "USER_ACCESS_REQUIRED": 1,
                 "REQUEST_REQUIRED": 1, "UNCERTAIN": 1, "BLOCKED_NO_ROUTE": 0, "FAIL": 0}
M_POINTS = {"PASS": 3, "COP30_PRE_EVENT_FEASIBLE": 3, "PRE_EVENT_ALT_SOURCE_REQUIRED": 1, "MULTI_TEMPORAL_UNRESOLVED": 1,
            "UNCERTAIN": 1, "TEMPORAL_FAIL": 0}
ACCESS_POINTS = {"COMPLETE": 3, "PUBLIC": 3, "USER_SIGNIN": 2, "REQUEST": 1, "NONE": 0}
TIER_A_MIN_SCORE = 18  # fixed a priori: half of the 36-point maximum
WEIGHTS = {"A_diversity": 3, "B_target": 2, "C_omega": 2, "D_m": 1, "E_geology": 1, "F_access": 1, "G_time": 1, "H_validation": 1,
           "I_redundancy": 2}

# V1 contact registry (read-only listing of Science_Fair_Extension/01_Data_Raw, 2026-10-06).  Any landscape whose raw data V1
# held (core folders or SF1_verification) is V1-historical and can never be an unseen lockbox (Stage 2E/2F precedent).
V1_CORE = {"ATITLAN_STAN_2005", "CINCHONA_POAS_2009", "KURIKOMA_2008", "LOMBOK_RINJANI_2018", "POHNPEI_CUMULATIVE"}
V1_SF1_VERIFICATION = {"CAPELLADES_TURRIALBA_2016", "CINCHONA_POAS_2009", "DOMINICA_MARIA_2017", "EL_SALVADOR_2001_JAN",
                       "GUATEMALA_1976", "LOMBOK_RINJANI_2018", "MATZACO_PUEBLA_2017", "PUERTO_RICO_MARIA_2017"}
V1_US_CATALOGUE_CONTACT_COUNTRY = "USA"  # SF1 held the national US landslide inventory v3 catalogue

# Landscape groups: one independent prediction domain each (events inside a group are replicates, not new landscapes).
GROUPS = {
    "ASO_2012_RAIN": "ASO_CALDERA", "KUMAMOTO_ASO_2016": "ASO_CALDERA",
    "MOUNT_ELGON_UGANDA": "MOUNT_ELGON", "MBALE_ELGON_2022": "MOUNT_ELGON",
    "KIVU_RIFT_NTK_2000_2019": "KIVU_RIFT", "KIVU_KALEHE_RIFT_FLANK": "KIVU_RIFT", "RWANDA_VIRUNGA_NW": "KIVU_RIFT",
    "BANKS_PENINSULA_2021": "BANKS_PENINSULA", "CHRISTCHURCH_PORT_HILLS_2011": "BANKS_PENINSULA",
    "EL_SALVADOR_1986": "EL_SALVADOR_CENTRAL", "EL_SALVADOR_2001_JAN": "EL_SALVADOR_CENTRAL", "EL_SALVADOR_2001_FEB": "EL_SALVADOR_CENTRAL",
    "SAO_MIGUEL_AZORES": "SAO_MIGUEL", "SETE_CIDADES_AZORES": "SAO_MIGUEL",
    "REUNION_BDMVT": "REUNION", "SALAZIE_RESEARCH": "REUNION",
    "NZ_GABRIELLE_EAST_COAST": "NZ_EAST_COAST_NONVOLCANIC",
}

# Explicit evidence-backed records (override/extend earlier metadata).  y/omega/g use STATUS_POINTS vocabulary; m uses M_POINTS.
CURATED: dict[str, dict] = {
    "KUMAMOTO_ASO_2016": dict(morphology="CALDERA", erosion_state="UNCERTAIN", climate="TEMPERATE_WET", tectonic_setting="SUBDUCTION_ARC",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="PYROCLASTIC_DOMINATED", y="PASS", omega="PASS", m="PASS", g="PASS", access="COMPLETE",
        e="PENDING_BOUNDED_ACCESS", h="PARTIAL_RECONSTRUCTION", h_evidence="caldera wall, radial flanks; Gen-II control gate failed (no surface)",
        lockbox="DEVELOPMENT_PREFERRED", blocker="NONE", next_action="None (production-ready); E allowlist pending",
        evidence="Stage 2D production; geology classes PYROCLASTIC 187,121 / MIXED 130,956 / VOLCANICLASTIC 36,046 units; Koppen Cfa"),
    "NOTO_PENINSULA_2024": dict(morphology="OLD_MASSIF_REMNANT", erosion_state="STRONGLY_ERODED", climate="TEMPERATE_WET",
        tectonic_setting="OTHER_UNCERTAIN", trigger="EARTHQUAKE", target_tier="TIER_2", material="MIXED_VOLCANIC", y="PASS", omega="PASS",
        m="PASS", g="PASS", access="COMPLETE", e="PENDING_BOUNDED_ACCESS", h="AGE_PROXY_ONLY", h_evidence="Oligocene-Miocene unit ages only",
        lockbox="DEVELOPMENT_PREFERRED", blocker="NONE", next_action="None (production-ready); E allowlist pending",
        evidence="Stage 2D production; MIXED_VOLCANIC 96,756 units; Late Oligocene-Early Miocene volcanics"),
    "LOMBOK_RINJANI_2018": dict(morphology="VOLCANIC_COMPLEX", erosion_state="UNCERTAIN", climate="TROPICAL_SEASONAL",
        tectonic_setting="SUBDUCTION_ARC", trigger="EARTHQUAKE", target_tier="TIER_2", material="MIXED_VOLCANIC", y="PASS", omega="PASS",
        m="PASS", g="PASS", access="COMPLETE", e="PENDING_BOUNDED_ACCESS", h="PARTIAL_RECONSTRUCTION",
        h_evidence="sheet 1807 edifice units Qhv(r/p/n), caldera rim symbol, relative stratigraphy; no numeric ages",
        lockbox="INELIGIBLE_V1_HISTORICAL", blocker="NONE (ALL_VALID vs STRICT_DEM_QUALITY must stay separate: 49.4% of positives touch edited DEM)",
        next_action="None (production-ready)", evidence="Stage 2F.2 production; 208,443 units; erosion state recorded as MIXED"),
    "PUERTO_RICO_MARIA_2017": dict(morphology="OLD_MASSIF_REMNANT", erosion_state="ANCIENT_REMNANT", climate="TROPICAL_WET",
        tectonic_setting="SUBDUCTION_ARC", trigger="CYCLONE_STORM", target_tier="TIER_1", material="SEDIMENTARY_INTERBEDDED",
        y="PASS", omega="PROVISIONAL_DECISION_REQUIRED", m="COP30_PRE_EVENT_FEASIBLE", g="PUBLIC_AVAILABLE_UNAUDITED", access="PUBLIC",
        e="GLOBAL_ROUTE_PENDING", h="H_UNAVAILABLE", h_evidence="Cretaceous-Eocene fossil island arc; no edifice surfaces",
        lockbox="INELIGIBLE_V1_HISTORICAL", hard_blocker="",
        blocker="OMEGA: no exact imagery-footprint/visibility geometry (FEMA S3 prefix now empty; NOAA covers ~10%); documented island-wide "
                "search scope requires an explicit scientific decision. DOMAIN: frozen Cenozoic rule removes Cretaceous volcanics",
        next_action="Decision D-PR-OMEGA + request coverage/unmapped-area documentation from USGS (W. Schulz); then harmonise OFR 98-38 geology",
        evidence="Hughes et al. 2019 metadata: FEMA 15 cm ~97% + DigitalGlobe ~99% of main island, overlapping imagery used against "
                 "cloud/shadow, small islands not examined; OFR 98-38 screen: Cenozoic volcanic-text units <=940 km2 (~14,860 of 71,431 "
                 "headscarps), Cretaceous volcanic 3,040 km2 (24,716) excluded by frozen DOMAIN_RULE; fossil arc, Paleocene-Eocene units"),
    "TALAKMAU_PASAMAN_2022": dict(morphology="STRATOVOLCANO", erosion_state="YOUNG_INTACT", climate="TROPICAL_WET",
        tectonic_setting="SUBDUCTION_ARC", trigger="EARTHQUAKE", target_tier="TIER_2", material="OTHER_UNCERTAIN",
        y="PROVISIONAL_DECISION_REQUIRED", omega="PUBLIC_AVAILABLE_UNAUDITED", m="COP30_PRE_EVENT_FEASIBLE", g="USER_ACCESS_REQUIRED",
        access="USER_SIGNIN", e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION", h_evidence="young intact Quaternary cone; edifice geometry",
        lockbox="UNDECIDED",
        blocker="Y is a preliminary 10 m Sentinel-2 rapid map (dissolved polygon, not field-validated) needing a Y-quality decision; "
                "geology needs GeoMap district SHP (sign-in)",
        next_action="User GeoMap sign-in for Pasaman / Pasaman Barat geology; then Y-quality decision D-TAL-Y and fast path",
        evidence="UNOSAT LS20220308IDN acquired: AnalysisExtent 881.7 km2, CloudObstruction 30.0 km2, LandslideExtent 6.24 km2 "
                 "(Sentinel-2 2022-03-04; Mw 6.1 2022-02-25); COP30 TanDEM-X 2011-2014 pre-event"),
    "BANKS_PENINSULA_2021": dict(morphology="SHIELD", erosion_state="STRONGLY_ERODED", climate="OTHER_UNCERTAIN",
        tectonic_setting="CONTINENTAL_HOTSPOT", trigger="EXTREME_RAINFALL", target_tier="TIER_2", material="BASALTIC",
        y="REQUEST_REQUIRED", omega="REQUEST_REQUIRED", m="COP30_PRE_EVENT_FEASIBLE", g="PUBLIC_AVAILABLE_UNAUDITED", access="REQUEST",
        e="GLOBAL_ROUTE_PENDING", h="FULL_RECONSTRUCTION", h_evidence="Lyttelton/Akaroa shield planezes, radial dykes, published 40Ar/39Ar ages",
        lockbox="POTENTIAL_LOCKBOX",
        blocker="Inventory + 115 km2 mapped boundary available only on reasonable request; loess mantle may map as surficial (domain risk)",
        next_action="Send request to corresponding author (Stage 2F RP02, refreshed in STAGE3A_HUMAN_ACTIONS)",
        evidence="Article 10.1080/00288306.2025.2458507: >1,300 landslides, 16 Dec 2021, 115 km2 Eastern Bays damage area; "
                 "Macrostrat: Akaroa Volcanics basaltic/trachytic (Tortonian)"),
    "COROMANDEL_GABRIELLE_2023": dict(morphology="VOLCANIC_COMPLEX", erosion_state="STRONGLY_ERODED", climate="TEMPERATE_WET",
        tectonic_setting="SUBDUCTION_ARC", trigger="CYCLONE_STORM", target_tier="UNCLEAR", material="ANDESITIC",
        y="REQUEST_REQUIRED", omega="REQUEST_REQUIRED", m="COP30_PRE_EVENT_FEASIBLE", g="PUBLIC_AVAILABLE_UNAUDITED", access="REQUEST",
        e="GLOBAL_ROUTE_PENDING", h="AGE_PROXY_ONLY", h_evidence="deeply eroded Miocene-Pliocene arc; no constructional surfaces",
        lockbox="UNDECIDED",
        blocker="Public Gabrielle extract (OSF 8sbrj, CC BY 4.0) covers only 337 training cells with NO volcanic geology class; full "
                "V1.0 inventory + priority mapping cells not yet public",
        next_action="Request V1.0 polygons + priority-cell grid for Coromandel and Waitakere volcanic terrain from GNS (batched with Auckland)",
        evidence="Massey et al. 2025 (Landslides): 5x2.5 km priority mapping cells, 337 training cells; GeolCodes = Quaternary soils, "
                 "Neogene sedimentary, Cretaceous-Paleogene sedimentary, intrusive, Torlesse"),
    "AUCKLAND_WAITAKERE_2023": dict(morphology="VOLCANIC_COMPLEX", erosion_state="STRONGLY_ERODED", climate="TEMPERATE_WET",
        tectonic_setting="SUBDUCTION_ARC", trigger="CYCLONE_STORM", target_tier="UNCLEAR", material="ANDESITIC",
        y="REQUEST_REQUIRED", omega="REQUEST_REQUIRED", m="COP30_PRE_EVENT_FEASIBLE", g="PUBLIC_AVAILABLE_UNAUDITED", access="REQUEST",
        e="GLOBAL_ROUTE_PENDING", h="AGE_PROXY_ONLY", h_evidence="Miocene Waitakere arc remnant", lockbox="UNDECIDED",
        blocker="Same GNS V1.0 access route as Coromandel; Auckland was a mapped Gabrielle region (Massey et al. 2025)",
        next_action="Batch into the GNS Gabrielle request", evidence="Massey et al. 2025 regions: Hawke's Bay, Tairawhiti, Auckland"),
    "PNG_PORGERA_2018": dict(morphology="MIXED_OTHER", erosion_state="UNCERTAIN", climate="TROPICAL_WET", tectonic_setting="OTHER_UNCERTAIN",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="SEDIMENTARY_INTERBEDDED", y="PUBLIC_AVAILABLE_UNAUDITED", omega="FAIL",
        m="COP30_PRE_EVENT_FEASIBLE", g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="NOT_ASSESSED", h_evidence="",
        lockbox="UNDECIDED", hard_blocker="OMEGA_Y_DERIVED_PUBLIC_EXTENT",
        blocker="Public mapping_extent is described as the area encompassing all landslides (Y-derived); volcanic share of the "
                "fold-thrust-belt epicentral area uncertain",
        next_action="Optional request to H. Tanyas for an independent searched-area/visibility mask; do not build until then",
        evidence="ScienceBase 61f04680 (Tanyas et al. 2022): only repository item with mapping_extent; derivation Y-dependent"),
    "KIVU_RIFT_NTK_2000_2019": dict(morphology="VOLCANIC_PLATEAU", erosion_state="UNCERTAIN", climate="TROPICAL_WET",
        tectonic_setting="CONTINENTAL_RIFT", trigger="MIXED", target_tier="UNCLEAR", material="BASALTIC", y="PUBLIC_AVAILABLE_UNAUDITED",
        omega="REQUEST_REQUIRED", m="MULTI_TEMPORAL_UNRESOLVED", g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING",
        h="AGE_PROXY_ONLY", h_evidence="Neogene-Quaternary rift basalts; no edifice surfaces", lockbox="POTENTIAL_LOCKBOX",
        blocker="Public Zenodo 5027004 points (2000-2019, Google Earth) lack a year-specific search mask; multi-temporal M; much of NTK is "
                "Precambrian basement",
        next_action="Request year-specific imagery/search mask + source polygons from RMCA (Depicker/Dewitte); batch with Kalehe",
        evidence="Stage 2F public-route audit; Zenodo 5027004 (CC BY 4.0)"),
    "MOUNT_ELGON_UGANDA": dict(morphology="SHIELD", erosion_state="STRONGLY_ERODED", climate="TROPICAL_WET",
        tectonic_setting="CONTINENTAL_HOTSPOT", trigger="MIXED", target_tier="UNCLEAR", material="OTHER_UNCERTAIN",
        y="UNCERTAIN", omega="FAIL", m="TEMPORAL_FAIL", g="REQUEST_REQUIRED", access="REQUEST", e="GLOBAL_ROUTE_PENDING",
        h="PARTIAL_RECONSTRUCTION", h_evidence="extinct Miocene shield (4,321 m)", lockbox="UNDECIDED",
        hard_blocker="OMEGA_IMPOSSIBLE_BY_CONSTRUCTION",
        blocker="Authors state the field-based inventory is not complete for the study area (some areas never or once surveyed); "
                "landslides span decades with partly unknown dates -> no observation domain and no pre-event M",
        next_action="STOP: an email cannot create a systematic search domain that was never surveyed",
        evidence="Broeckx et al. 2019 (Landslides 16:297), open PDF: 653 landslides from field surveys + Google Earth; 'we do not "
                 "consider it to be complete for the entire study area'"),
    "GUATEMALA_1976": dict(morphology="MIXED_OTHER", erosion_state="UNCERTAIN", climate="TROPICAL_SEASONAL", tectonic_setting="SUBDUCTION_ARC",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="PYROCLASTIC_DOMINATED", y="PUBLIC_AVAILABLE_UNAUDITED", omega="REQUEST_REQUIRED",
        m="TEMPORAL_FAIL", g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="AGE_PROXY_ONLY", h_evidence="Quaternary pumice fill",
        lockbox="INELIGIBLE_V1_HISTORICAL", hard_blocker="M_TEMPORAL_FAIL_NO_PRE_EVENT_DEM",
        blocker="Earliest usable DEMs (SRTM 2000, COP30 2011-2014) post-date the 1976 event by 24-38 years; Omega only on printed Plate 1",
        next_action="STOP under V2 temporal rules; do not waive", evidence="Harp et al. 1981 (USGS PP 1204-A); held in V1 SF1_verification"),
    "KIVU_KALEHE_RIFT_FLANK": dict(morphology="VOLCANIC_PLATEAU", erosion_state="UNCERTAIN", climate="TROPICAL_WET",
        tectonic_setting="CONTINENTAL_RIFT", trigger="EXTREME_RAINFALL", target_tier="UNCLEAR", material="OTHER_UNCERTAIN",
        y="PROVISIONAL_DECISION_REQUIRED", omega="PUBLIC_AVAILABLE_UNAUDITED", m="COP30_PRE_EVENT_FEASIBLE", g="UNCERTAIN",
        access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="H_UNAVAILABLE", h_evidence="rift flank, largely Precambrian", lockbox="UNDECIDED",
        blocker="New public UNOSAT 2023 route: AnalysisExtent 144.8 km2 but 54.0 km2 (37%) cloud; Y is one dissolved 0.9 km2 "
                "Sentinel-2 polygon, not field-validated; volcanic share of Kalehe flank unverified; NHESS 2023 inventory request-only",
        next_action="Hold; optional joint RMCA request with Kivu NTK", evidence="UNOSAT FL20230508COD acquired (CC BY-SA)"),
    "MIRAVALLES_OTTO_2016": dict(morphology="STRATOVOLCANO", erosion_state="UNCERTAIN", climate="TROPICAL_WET",
        tectonic_setting="SUBDUCTION_ARC", trigger="MIXED", target_tier="UNCLEAR", material="ANDESITIC", y="REQUEST_REQUIRED",
        omega="REQUEST_REQUIRED", m="COP30_PRE_EVENT_FEASIBLE", g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING",
        h="PARTIAL_RECONSTRUCTION", h_evidence="Miravalles stratovolcano + Guayabo caldera", lockbox="POTENTIAL_LOCKBOX",
        blocker="942 landslides from a compound Jul-2016 Mw 5.4 earthquake + Nov-2016 Hurricane Otto; no public polygons with trigger "
                "attribution or mapped area",
        next_action="Send request (Stage 2F RP07) asking for trigger-attributed polygons + mapped area",
        evidence="Quesada-Roman et al. 2019, Landslides (10.1007/s10346-019-01209-4)"),
    "ATITLAN_STAN_2005": dict(morphology="CALDERA", erosion_state="MODERATELY_DISSECTED", climate="TROPICAL_SEASONAL",
        tectonic_setting="SUBDUCTION_ARC", trigger="CYCLONE_STORM", target_tier="TIER_2", material="PYROCLASTIC_DOMINATED", y="PASS",
        omega="PASS", m="COP30_PRE_EVENT_FEASIBLE", g="UNCERTAIN", access="USER_SIGNIN", e="GLOBAL_ROUTE_PENDING",
        h="PARTIAL_RECONSTRUCTION", h_evidence="Atitlan III caldera (~84 ka) + stratocones", lockbox="INELIGIBLE_V1_HISTORICAL",
        blocker="No reusable authoritative geology: MAGA Solola study PDFs return HTTP 403 to scripts; Macrostrat has no large-scale unit",
        next_action="User opens the MAGA PDF links in a browser; if they load, save them for uncertainty-aware digitisation of the cited "
                    "IGN 1:50,000 geology quadrangles",
        evidence="Stage 2F internal resolution; Macrostrat query 14.69N 91.19W returned no units (2026-10-06)"),
    "KURIKOMA_2008": dict(morphology="VOLCANIC_COMPLEX", erosion_state="UNCERTAIN", climate="TEMPERATE_WET", tectonic_setting="SUBDUCTION_ARC",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="OTHER_UNCERTAIN", y="PASS", omega="BLOCKED_NO_ROUTE", m="PASS", g="UNCERTAIN",
        access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION", h_evidence="radial flank + plateau", lockbox="INELIGIBLE_V1_HISTORICAL",
        blocker="No coordinate-supported mapped extent: GSI Material 4 not georeferenced; the alternative USGS-repository inventory "
                "(Yagi et al. 2009) answers 'Is the boundary of the mapped area indicated? No'",
        next_action="Optional single GSI request; otherwise stop", evidence="ScienceBase 5b59ec98 Yagi_methods_summary form, item 4"),
    "HOKKAIDO_IBURI_2018": dict(morphology="MIXED_OTHER", erosion_state="UNCERTAIN", climate="OTHER_UNCERTAIN", tectonic_setting="SUBDUCTION_ARC",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="PYROCLASTIC_DOMINATED", y="PASS", omega="PASS", m="PASS", g="FAIL",
        access="COMPLETE", e="GLOBAL_ROUTE_PENDING", h="H_UNAVAILABLE", h_evidence="tephra cover over Neogene sediments", lockbox="UNDECIDED",
        hard_blocker="VOLCANIC_DOMAIN_FAIL_FROZEN_RULE",
        blocker="Tephra-mantled Neogene sedimentary terrain fails the frozen volcanic-domain rule (not weakened)",
        next_action="None; tephra-sensitivity use only", evidence="Stage 2C/2D BLOCKED_VOLCANIC_DOMAIN"),
    "WILSONS_RIVER_NSW_2022": dict(morphology="VOLCANIC_PLATEAU", erosion_state="STRONGLY_ERODED", climate="TEMPERATE_WET",
        tectonic_setting="CONTINENTAL_HOTSPOT", trigger="EXTREME_RAINFALL", target_tier="TIER_2", material="BASALTIC", y="PASS",
        omega="REQUEST_REQUIRED", m="UNCERTAIN", g="PUBLIC_AVAILABLE_UNAUDITED", access="REQUEST", e="GLOBAL_ROUTE_PENDING",
        h="PARTIAL_RECONSTRUCTION", h_evidence="Tweed/Lamington shield planeze remnants", lockbox="UNDECIDED",
        blocker="Candidate-generation/search mask and pre-2022 DEM epoch unresolved; catchment mixes volcanic and sedimentary units; lies in "
                "the Lamington Volcanics (Tweed volcanic province, the parent project's application region) -> lockbox use needs a decision",
        next_action="Send Stage 2F second-wave request for search mask, rejected-area mask and imagery/DEM epochs",
        evidence="Stage 2F second-wave audit: 3,689 validated affected-area polygons"),
    "DOMINICA_MARIA_2017": dict(morphology="MIXED_OTHER", erosion_state="UNCERTAIN", climate="TROPICAL_WET", tectonic_setting="SUBDUCTION_ARC",
        trigger="CYCLONE_STORM", target_tier="BOTH_AVAILABLE", material="ANDESITIC", y="USER_ACCESS_REQUIRED", omega="REQUEST_REQUIRED",
        m="COP30_PRE_EVENT_FEASIBLE", g="UNCERTAIN", access="USER_SIGNIN", e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION",
        h_evidence="multi-edifice volcanic island", lockbox="INELIGIBLE_V1_HISTORICAL",
        blocker="Restricted Zenodo polygons (login + access request); Cloud.shp alone is not a search domain",
        next_action="Optional: request restricted access on Zenodo 15224820 and ask for the imagery/search footprint",
        evidence="V1 SF1_verification holds the open Zenodo 15182980 record"),
    "ASO_2012_RAIN": dict(morphology="CALDERA", erosion_state="UNCERTAIN", climate="TEMPERATE_WET", tectonic_setting="SUBDUCTION_ARC",
        trigger="EXTREME_RAINFALL", target_tier="UNCLEAR", material="PYROCLASTIC_DOMINATED", y="UNCERTAIN", omega="UNCERTAIN", m="UNCERTAIN",
        g="PASS", access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION", h_evidence="same Aso caldera as Kumamoto",
        lockbox="DEVELOPMENT_PREFERRED", hard_blocker="SAME_LANDSCAPE_AS_PRODUCTION",
        blocker="Same prediction domain as KUMAMOTO_ASO_2016: a temporal/trigger replicate, not a new landscape",
        next_action="Keep as possible rainfall-trigger replicate only", evidence="Landscape group ASO_CALDERA"),
    "CINCHONA_POAS_2009": dict(morphology="STRATOVOLCANO", erosion_state="UNCERTAIN", climate="TROPICAL_WET", tectonic_setting="SUBDUCTION_ARC",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="ANDESITIC", y="PASS", omega="BLOCKED_NO_ROUTE", m="PRE_EVENT_ALT_SOURCE_REQUIRED",
        g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION", h_evidence="Poas stratovolcano",
        lockbox="INELIGIBLE_V1_HISTORICAL", blocker="No mapping boundary in V1 or in the USGS repository item (Ruiz et al. 2019)",
        next_action="Hold", evidence="ScienceBase 61ead71b: no extent file"),
    "CAPELLADES_TURRIALBA_2016": dict(morphology="STRATOVOLCANO", erosion_state="UNCERTAIN", climate="TROPICAL_WET",
        tectonic_setting="SUBDUCTION_ARC", trigger="EARTHQUAKE", target_tier="TIER_2", material="ANDESITIC", y="PASS",
        omega="BLOCKED_NO_ROUTE", m="COP30_PRE_EVENT_FEASIBLE", g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING",
        h="PARTIAL_RECONSTRUCTION", h_evidence="Turrialba stratovolcano", lockbox="INELIGIBLE_V1_HISTORICAL",
        blocker="No mapping extent in the USGS repository item (Ruiz et al. 2020); M5.3 inventory small", next_action="Hold",
        evidence="ScienceBase 61f031a0: no extent file"),
    "MADEIRA_2010": dict(morphology="SHIELD", erosion_state="STRONGLY_ERODED", climate="TEMPERATE_SEASONAL",
        tectonic_setting="HOTSPOT_OCEAN_ISLAND", trigger="EXTREME_RAINFALL", target_tier="UNCLEAR", material="BASALTIC",
        y="REQUEST_REQUIRED", omega="REQUEST_REQUIRED", m="PRE_EVENT_ALT_SOURCE_REQUIRED", g="UNCERTAIN", access="REQUEST",
        e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION", h_evidence="deeply incised shield", lockbox="POTENTIAL_LOCKBOX",
        blocker="Lira et al. 2013 (NHESS 13:709): 1,257 landslides over a 178 km2 study area (Funchal + Ribeira Brava basins) but no public "
                "vectors; Feb-2010 event predates COP30 acquisitions -> frozen-M source decision needed",
        next_action="Optional request to NHESS 13:709 corresponding author for inventory + basin study area",
        evidence="NHESS 13:709 (2013) abstract"),
    "SAO_MIGUEL_AZORES": dict(morphology="MIXED_OTHER", erosion_state="UNCERTAIN", climate="TEMPERATE_WET", tectonic_setting="OTHER_UNCERTAIN",
        trigger="UNCLEAR", target_tier="UNCLEAR", material="OTHER_UNCERTAIN", y="REQUEST_REQUIRED", omega="REQUEST_REQUIRED", m="UNCERTAIN",
        g="UNCERTAIN", access="REQUEST", e="GLOBAL_ROUTE_PENDING", h="PARTIAL_RECONSTRUCTION", h_evidence="calderas, multi-edifice island",
        lockbox="POTENTIAL_LOCKBOX", blocker="Occurrence vectors/support not verified on the regional portal",
        next_action="Optional custodian request (Stage 2F second wave)", evidence="Stage 2F second-wave audit"),
    "IFFI_ITALIAN_VOLCANIC_DISTRICTS": dict(morphology="VOLCANIC_COMPLEX", erosion_state="UNCERTAIN", climate="TEMPERATE_SEASONAL",
        tectonic_setting="OTHER_UNCERTAIN", trigger="MIXED", target_tier="BOTH_AVAILABLE", material="OTHER_UNCERTAIN",
        y="PUBLIC_AVAILABLE_UNAUDITED", omega="UNCERTAIN", m="MULTI_TEMPORAL_UNRESOLVED", g="PUBLIC_AVAILABLE_UNAUDITED", access="PUBLIC",
        e="GLOBAL_ROUTE_PENDING", h="NOT_ASSESSED", h_evidence="", lockbox="UNDECIDED",
        blocker="Cumulative, undated IFFI polygons: no event and no systematic search support -> Omega/M temporal unresolved",
        next_action="Targeted district audit only if Europe remains empty after the primary queue", evidence="Stage 2F second wave"),
    "EL_SALVADOR_2001_JAN": dict(morphology="MIXED_OTHER", erosion_state="UNCERTAIN", climate="TROPICAL_SEASONAL", tectonic_setting="SUBDUCTION_ARC",
        trigger="EARTHQUAKE", target_tier="TIER_2", material="PYROCLASTIC_DOMINATED", y="PUBLIC_AVAILABLE_UNAUDITED", omega="BLOCKED_NO_ROUTE",
        m="PRE_EVENT_ALT_SOURCE_REQUIRED", g="UNCERTAIN", access="PUBLIC", e="GLOBAL_ROUTE_PENDING", h="NOT_ASSESSED", h_evidence="",
        lockbox="INELIGIBLE_V1_HISTORICAL", blocker="Points only, reporting-biased; no mapping extent in USGS repository item",
        next_action="Hold", evidence="ScienceBase 5874cce7: no extent file"),
    "NICARAGUA_MITCH_1998": dict(morphology="VOLCANIC_COMPLEX", erosion_state="UNCERTAIN", climate="TROPICAL_SEASONAL",
        tectonic_setting="SUBDUCTION_ARC", trigger="CYCLONE_STORM", target_tier="BOTH_AVAILABLE", material="OTHER_UNCERTAIN",
        y="PUBLIC_AVAILABLE_UNAUDITED", omega="UNCERTAIN", m="TEMPORAL_FAIL", g="UNCERTAIN", access="PUBLIC", e="GLOBAL_ROUTE_PENDING",
        h="NOT_ASSESSED", h_evidence="", lockbox="UNDECIDED", hard_blocker="M_TEMPORAL_FAIL_NO_PRE_EVENT_DEM",
        blocker="1998 event predates SRTM; ~200 m positional accuracy", next_action="Stop", evidence="Stage 2A.2 limitations"),
}
_CUMULATIVE = dict(trigger="MIXED", m="MULTI_TEMPORAL_UNRESOLVED", hard_blocker="CUMULATIVE_UNDATED_INVENTORY_NO_PRE_EVENT_M",
                   next_action="Stop: undated cumulative deposits have no event, so pre-event M is undefined and mapped morphology can encode Y")
for _lid, _why in {
        "OREGON_CASCADES_SLIDO": "SLIDO: cumulative morphology-mapped deposits; Stage 2A.2 flagged M may encode Y; no event timing",
        "WASHINGTON_CASCADES_WASLID": "WASLID: cumulative morphology-mapped deposits; no event timing",
        "COLUMBIA_RIVER_GORGE_BASALT": "Cumulative lidar-mapped deposits; no event timing",
        "IFFI_ITALIAN_VOLCANIC_DISTRICTS": "IFFI: cumulative, undated; no systematic event search support"}.items():
    CURATED[_lid] = {**CURATED.get(_lid, {}), **_CUMULATIVE, "blocker": _why, "evidence": "Stage 2A.2 / 2F second-wave limitations"}
CURATED["PATAGONIA_PALI"] = dict(hard_blocker="Y_PREDICTIONS_NOT_GROUND_TRUTH", trigger="MIXED", m="MULTI_TEMPORAL_UNRESOLVED",
    blocker="Zenodo supplies a small manual training subset plus model predictions; predictions are never Y", next_action="Stop unless full-area manual labels exist",
    evidence="Stage 2F second wave: FAST_FAIL_UNLESS_MANUAL_SUPPORT_PROVIDED")
TIER_A_GEOLOGY = {"PASS", "PUBLIC_AVAILABLE_UNAUDITED", "USER_ACCESS_REQUIRED"}

MORPH_MAP = {"STRATOVOLCANO": "STRATOVOLCANO", "SHIELD": "SHIELD", "CALDERA": "CALDERA", "VOLCANIC_COMPLEX": "VOLCANIC_COMPLEX",
             "VOLCANIC_PLATEAU": "VOLCANIC_PLATEAU"}
EROSION_MAP = {"YOUNG_INTACT": "YOUNG_INTACT", "MODERATE": "MODERATELY_DISSECTED", "DEEPLY_ERODED": "STRONGLY_ERODED"}
CLIMATE_MAP = {"TROPICAL_WET": "TROPICAL_WET", "TROPICAL_SEASONAL": "TROPICAL_SEASONAL", "ARID_SEMI_ARID": "DRY_SEMI_ARID",
               "HIGH_ALTITUDE": "HIGH_ALTITUDE_ALPINE"}
TECTONIC_MAP = {"SUBDUCTION_ARC": "SUBDUCTION_ARC", "INTRAPLATE_HOTSPOT_OCEANIC_ISLAND": "HOTSPOT_OCEAN_ISLAND",
                "INTRAPLATE_HOTSPOT_CONTINENTAL": "CONTINENTAL_HOTSPOT", "CONTINENTAL_RIFT": "CONTINENTAL_RIFT"}
TRIGGER_MAP = {"EARTHQUAKE": "EARTHQUAKE", "EXTREME_RAINFALL_CYCLONE": "EXTREME_RAINFALL", "RAINFALL": "EXTREME_RAINFALL",
               "MIXED_BACKGROUND": "MIXED"}
GATE_MAP = {"PASS": "PUBLIC_AVAILABLE_UNAUDITED", "UNCERTAIN": "UNCERTAIN", "FAIL": "FAIL"}
ACCESS_MAP = {"OPEN_DOWNLOAD": "PUBLIC", "OPEN_PORTAL_UNVERIFIED": "PUBLIC", "LOGIN_REQUIRED": "USER_SIGNIN", "REQUEST_REQUIRED": "REQUEST",
              "VERIFIED_LOCAL_V1": "PUBLIC", "PAPER_ONLY": "NONE", "NOT_FOUND": "NONE", "UNKNOWN": "NONE"}


def _rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _m_default(year: str, m_gate: str) -> str:
    try:
        y = int(year[:4])
    except ValueError:
        return "UNCERTAIN"
    if y < 2000:
        return "TEMPORAL_FAIL"
    if y < 2015:
        return "PRE_EVENT_ALT_SOURCE_REQUIRED"
    return "COP30_PRE_EVENT_FEASIBLE" if m_gate == "PASS" else "UNCERTAIN"


def _lockbox(lid: str, country: str, curated: str | None) -> str:
    if lid in V1_CORE | V1_SF1_VERIFICATION:
        return "INELIGIBLE_V1_HISTORICAL"
    if curated:
        return curated
    return "UNDECIDED"


def build_matrix() -> list[dict]:
    dash = {r["landscape_id"]: r for r in _rows(DASHBOARD)}
    cand = {r["landscape_id"]: r for r in _rows(CANDIDATES)}
    out = []
    for lid in sorted(dash):
        d, c, k = dash[lid], cand.get(lid, {}), CURATED.get(lid, {})
        erosion_src = c.get("erosional_state", "UNKNOWN")
        rec = dict(
            landscape_id=lid, country=d["country"], continent=d["continent"], landscape_group=GROUPS.get(lid, lid),
            morphology=k.get("morphology") or MORPH_MAP.get(c.get("volcano_type", ""), "MIXED_OTHER"),
            morphology_source_value=c.get("volcano_type", d.get("morphology", "")),
            erosion_state=k.get("erosion_state") or EROSION_MAP.get(erosion_src, "UNCERTAIN"), erosion_source_value=erosion_src,
            climate=k.get("climate") or CLIMATE_MAP.get(c.get("climate_class", d["climate"]), "OTHER_UNCERTAIN"),
            climate_source_value=c.get("climate_class", d["climate"]),
            tectonic_setting=k.get("tectonic_setting") or TECTONIC_MAP.get(d["tectonic_setting"], "OTHER_UNCERTAIN"),
            tectonic_source_value=d["tectonic_setting"],
            trigger=k.get("trigger") or TRIGGER_MAP.get(c.get("inventory_trigger", ""), "UNCLEAR"),
            target_tier=k.get("target_tier") or (c.get("inventory_tier") if c.get("inventory_tier") in TIER else "UNCLEAR"),
            material=k.get("material", "OTHER_UNCERTAIN"),
            y=k.get("y") or GATE_MAP.get(c.get("y_gate", ""), "UNCERTAIN"),
            omega=k.get("omega") or GATE_MAP.get(c.get("omega_gate", ""), "UNCERTAIN"),
            m=k.get("m") or _m_default(c.get("inventory_year", "UNKNOWN"), c.get("m_gate", "")),
            g=k.get("g") or ("PUBLIC_AVAILABLE_UNAUDITED" if c.get("geology_available") == "TRUE" else "UNCERTAIN"),
            e=k.get("e", "GLOBAL_ROUTE_PENDING"), h_feasibility=k.get("h", "NOT_ASSESSED"), h_evidence=k.get("h_evidence", ""),
            temporal_validity=("PASS" if k.get("m") in ("PASS", "COP30_PRE_EVENT_FEASIBLE") else
                               "FAIL" if (k.get("m") or _m_default(c.get("inventory_year", "UNKNOWN"), c.get("m_gate", ""))) == "TEMPORAL_FAIL" else "UNRESOLVED"),
            access=k.get("access") or ACCESS_MAP.get(c.get("data_accessibility", "UNKNOWN"), "NONE"),
            licence=c.get("licence", "") or "UNKNOWN",
            production_ready=lid in PRODUCTION, prior_state=d["state"],
            v1_historical=("V1_CORE" if lid in V1_CORE else "V1_SF1_VERIFICATION_REFERENCE" if lid in V1_SF1_VERIFICATION else
                           "V1_US_CATALOGUE_CONTACT" if d["country"] == V1_US_CATALOGUE_CONTACT_COUNTRY else "NOT_HISTORICAL"),
            hard_blocker=k.get("hard_blocker", "PRIOR_STAGE_REJECT" if d["state"] == "REJECT" else ""),
            main_blocker=k.get("blocker") or d["primary_blocker"], next_action=k.get("next_action") or "No action unless primary queue fails",
            evidence=k.get("evidence") or f"Stage 2A.2 metadata; limitations: {c.get('major_limitations', '')}",
            curated=bool(k),
        )
        rec["lockbox_status"] = _lockbox(lid, d["country"], k.get("lockbox"))
        if rec["v1_historical"] == "V1_US_CATALOGUE_CONTACT" and rec["lockbox_status"] == "POTENTIAL_LOCKBOX":
            rec["lockbox_status"] = "UNDECIDED"
        if rec["omega"] == "FAIL" and not rec["hard_blocker"]:
            rec["hard_blocker"] = "OMEGA_FAIL"
        if rec["m"] == "TEMPORAL_FAIL" and not rec["hard_blocker"]:
            rec["hard_blocker"] = "M_TEMPORAL_FAIL"
        if rec["morphology_source_value"] == "NON_VOLCANIC_REFERENCE":
            rec["hard_blocker"] = "NON_VOLCANIC"
        out.append(rec)
    return out


def production_coverage(matrix: list[dict]) -> dict[str, set]:
    prod = [r for r in matrix if r["production_ready"]]
    return {a: {r[a] for r in prod} for a in AXES}


def novel_axes(rec: dict, coverage: dict[str, set]) -> list[str]:
    return [f"{a}={rec[a]}" for a in AXES if rec[a] not in UNKNOWNS and rec[a] not in coverage[a]]


def score(rec: dict, coverage: dict[str, set], matrix: list[dict]) -> dict:
    new = novel_axes(rec, coverage)
    a = 0 if not new else 1 if len(new) <= 2 else 2 if len(new) <= 4 else 3
    b = STATUS_POINTS[rec["y"]] if rec["target_tier"] != "UNCLEAR" or rec["y"] == "PASS" else min(STATUS_POINTS[rec["y"]], 1)
    c = STATUS_POINTS[rec["omega"]]
    d = M_POINTS[rec["m"]]
    e = STATUS_POINTS[rec["g"]]
    f = ACCESS_POINTS[rec["access"]]
    open_gates = sum(rec[g] != "PASS" for g in ("y", "omega", "g")) + (rec["m"] not in ("PASS", "COP30_PRE_EVENT_FEASIBLE"))
    g = 0 if rec["hard_blocker"] else {0: 3, 1: 3, 2: 2, 3: 1}.get(open_gates, 0)
    h = {"POTENTIAL_LOCKBOX": 3, "UNDECIDED": 2, "DEVELOPMENT_PREFERRED": 2, "INELIGIBLE_V1_HISTORICAL": 1}[rec["lockbox_status"]]
    prod = [r for r in matrix if r["production_ready"] and r["landscape_id"] != rec["landscape_id"]]
    same_group = any(p["landscape_group"] == rec["landscape_group"] for p in prod)
    max_match = max((sum(rec[x] == p[x] and rec[x] not in UNKNOWNS for x in AXES) for p in prod), default=0)
    i = -3 if same_group else -2 if max_match >= 5 else -1 if max_match >= 4 else 0
    parts = dict(A_diversity=a, B_target=b, C_omega=c, D_m=d, E_geology=e, F_access=f, G_time=g, H_validation=h, I_redundancy=i)
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return dict(parts, total=total, novel_categories="|".join(new), max_category_match_with_production=max_match)


def tier(rec: dict, s: dict) -> tuple[str, str]:
    if rec["production_ready"]:
        return "PRODUCTION", "Already production-ready"
    if rec["hard_blocker"]:
        return "C", f"Hard blocker: {rec['hard_blocker']}"
    if rec["access"] == "NONE":
        return "C", "No known data route (inventory paper-only / not found)"
    tier_a = (s["total"] >= TIER_A_MIN_SCORE and rec["access"] in ("PUBLIC", "USER_SIGNIN") and s["D_m"] == 3
              and STATUS_POINTS[rec["y"]] >= 2 and STATUS_POINTS[rec["omega"]] >= 2 and rec["g"] in TIER_A_GEOLOGY)
    if tier_a:
        return "A", (f"Score {s['total']} >= {TIER_A_MIN_SCORE}; Y/Omega evidence in hand; pre-event M feasible; geology route known; "
                     "remaining steps need no external custodian")
    if s["total"] >= TIER_A_MIN_SCORE or s["A_diversity"] >= 2:
        return "B", f"Score {s['total']} / diversity {s['A_diversity']}: valuable but blocked by {rec['access']} access or an unresolved gate"
    return "C", f"Score {s['total']} < {TIER_A_MIN_SCORE} with diversity {s['A_diversity']}: low near-term value"


def rank(matrix: list[dict]) -> list[dict]:
    cov = production_coverage(matrix)
    out = []
    for r in matrix:
        s = score(r, cov, matrix)
        t, why = tier(r, s)
        out.append(dict(landscape_id=r["landscape_id"], landscape_group=r["landscape_group"], priority_tier=t, tier_reason=why, **s,
                        main_blocker=r["main_blocker"], next_action=r["next_action"], lockbox_status=r["lockbox_status"]))
    order = {"PRODUCTION": 0, "A": 1, "B": 2, "C": 3}
    return sorted(out, key=lambda x: (order[x["priority_tier"]], -x["total"], x["landscape_id"]))
