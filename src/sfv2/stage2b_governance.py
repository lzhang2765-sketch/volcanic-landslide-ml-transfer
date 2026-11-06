"""Stage 2B governance records: E allowlist proposal, external request queue, Kurikoma transfer rules, Stage-2A.2 erratum."""
from __future__ import annotations

# ---------------------------------------------------------------------------------------------------------------
# E allowlist PROPOSAL (nothing downloaded).  Licences marked VERIFY_AT_FREEZE were not confirmed in this stage.
# (variable, purpose, dataset, version, resolution, period, licence, leakage, consistent_all_landscapes, recommendation)
E_ALLOWLIST = [
    ("MAP_mm", "Mean annual precipitation: long-term wetness control", "CHELSA", "V2.1 bio12", "30 arcsec (~1 km)", "1981-2010",
     "VERIFY_AT_FREEZE", "LOW: climatology predates/ignores events", "YES (global)", "CORE"),
    ("PRECIP_SEASONALITY", "Rainfall seasonality (CV)", "CHELSA", "V2.1 bio15", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE", "LOW", "YES", "CORE"),
    ("PRECIP_WETTEST_MONTH_mm", "Wet-season loading", "CHELSA", "V2.1 bio13", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE", "LOW", "YES", "CORE"),
    ("MAT_C", "Mean annual temperature (weathering/freeze-thaw)", "CHELSA", "V2.1 bio1", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE", "LOW", "YES", "CORE"),
    ("TEMP_SEASONALITY", "Temperature seasonality", "CHELSA", "V2.1 bio4", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE", "LOW", "YES", "CORE"),
    ("TREE_COVER_2000_pct", "Pre-event vegetation root reinforcement proxy", "Hansen Global Forest Change", "v1.11 treecover2000", "1 arcsec (~30 m)",
     "Year 2000 baseline", "CC BY 4.0", "LOW for 2016/2018/2024 events (baseline predates them); stale for Noto (24 yr)", "YES", "CORE_WITH_CAVEAT"),
    ("LAND_COVER_PRE_EVENT", "Land use/cover class", "ESA CCI Land Cover / C3S", "Annual v2.0.7 / v2.1.1", "300 m", "Year before each event",
     "VERIFY_AT_FREEZE", "MODERATE: must use the year before each event; ESA WorldCover 2020/2021 is POST-event for Kumamoto/Hokkaido and is excluded",
     "YES (epoch-matched)", "OPTIONAL"),
    ("SOIL_CLAY_SAND_0_30cm", "Soil texture", "SoilGrids", "2.0", "250 m", "Static model", "CC BY 4.0",
     "MODERATE: modelled from terrain covariates; check collinearity with M", "YES", "OPTIONAL"),
    ("ANTECEDENT_SOIL_MOISTURE", "Event antecedent wetness", "ERA5-Land", "hourly", "0.1 deg", "Event window", "Copernicus licence",
     "HIGH heterogeneity: event-specific and trigger-dependent", "PARTIAL", "EXCLUDE_FROM_CORE"),
    ("TRIGGER_INTENSITY", "Event trigger (PGA/MMI for earthquakes; event rainfall for storms)", "USGS ShakeMap / GPM IMERG", "event versions",
     "~1 km / 0.1 deg", "Event window", "Public domain / NASA open", "Not comparable across trigger types; landscape-level constant for small domains",
     "NO (trigger-specific)", "SEPARATE_TRIGGER_BLOCK"),
]

# ---------------------------------------------------------------------------------------------------------------
# External-data request queue (prepared only; nobody is contacted).  Scores 1-5.
WEIGHTS = {"resolve_probability": 0.30, "scientific_quality": 0.20, "diversity": 0.20, "h_potential": 0.10, "unit_yield": 0.10, "response_speed": 0.10}
# (landscape_id, custodian, exact request, resolve, quality, diversity, h, yield, speed, note)
REQUESTS = [
    ("PNG_PORGERA_2018", "Tanyas et al. / USGS GHSC", "Definition of mapping_extent (searched imagery vs. envelope); cloud/no-image gaps as polygons; imagery footprints and dates",
     5, 4, 5, 3, 5, 3, "Y already open; one clarification could promote to READY_CORE"),
    ("BANKS_PENINSULA_2021", "University of Canterbury / custodians (Yates et al.)", "Source and runout polygons plus the 115 km2 mapping boundary as vectors",
     4, 5, 4, 5, 3, 3, "Gates already PASS on paper; intraplate shield H"),
    ("KIVU_RIFT_NTK_2000_2019", "Depicker & Dewitte (RMCA)", "Source-area polygons; Google Earth imagery-availability/search mask by year; volcanic-subset geology",
     3, 3, 5, 2, 4, 3, "Africa/rift diversity; open points already held"),
    ("COROMANDEL_GABRIELLE_2023", "GNS Science (registration)", "USER DECISION: authorise registered-user access; then the mapping-cell support layer",
     4, 4, 3, 3, 4, 4, "Not an email request; needs your authorisation to register"),
    ("GUATEMALA_1976", "USGS (Harp et al. 1981)", "Digital mapped-area boundary for PP1204-A, if held; otherwise permission to digitise plates",
     3, 4, 2, 2, 4, 4, "M leakage justification still required"),
    ("AUCKLAND_WAITAKERE_2023", "Auckland Council", "2023 landslide vectors for the Waitakere Ranges with mapping coverage, imagery/LiDAR epochs and method",
     3, 4, 3, 3, 4, 2, "Miocene arc volcanics"),
    ("MIRAVALLES_OTTO_2016", "Quesada-Roman et al.", "Landslide polygons, mapped area and imagery dates; separation of earthquake vs. Otto failures",
     3, 3, 2, 4, 3, 3, "Compound trigger must be separated"),
    ("KIVU_KALEHE_RIFT_FLANK", "Maki Mateso / Dewitte et al.", "Inventory vectors, study boundary, visibility/fieldwork mask, epoch per landslide",
     3, 3, 4, 2, 3, 3, "Overlaps KIVU_RIFT_NTK_2000_2019; request jointly"),
    ("CHRISTCHURCH_PORT_HILLS_2011", "GNS Science / Christchurch City Council", "Source-area (cliff-collapse/rockfall) polygons and surveyed-area boundaries",
     3, 3, 4, 5, 2, 2, "Rockfall-dominated; Lyttelton H"),
    ("MOUNT_ELGON_UGANDA", "KU Leuven / VUB (Broeckx, Jacobs)", "Landslide inventory vectors with mapping area and dates",
     2, 4, 5, 5, 3, 2, "High diversity; low access probability"),
]


def request_rows() -> list[dict]:
    rows = []
    for lid, who, what, *scores, note in REQUESTS:
        s = dict(zip(WEIGHTS, scores))
        rows.append(dict(landscape_id=lid, custodian=who, exact_request=what, **s,
                         weighted_score=round(sum(WEIGHTS[k] * v for k, v in s.items()), 2), note=note, contacted="NO"))
    rows.sort(key=lambda r: (-r["weighted_score"], r["landscape_id"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Kurikoma V1 -> V2 provenance transfer: eligible = raw inputs only.  Excluded prefixes are never transferable.
KURIKOMA_V1_PREFIX = "Science_Fair_Extension/01_Data_Raw/Kurikoma/"
KURIKOMA_ROLE = {"Y/": "Y", "M/": "DEM", "G/": "GEOLOGY", "E/": "ENVIRONMENT", "Domain/": "TRIGGER_CONTEXT", "Metadata/": "Y_METHOD"}
KURIKOMA_EXCLUDED = {
    "Science_Fair_Extension/02_Data_Processed/": "V1-derived product; V2 must rebuild from raw under V2 rules (V1 catchment is not V2 Omega)",
    "Science_Fair_Extension/03_Reconstruction/": "V1 H reconstruction; V2 H must be rebuilt independently",
    "Science_Fair_Extension/04_Feature_Engineering/": "V1 features",
    "Science_Fair_Extension/05_ML_Dataset/": "V1 ML dataset (holdout) - never transferable",
    "Science_Fair_Extension/08_Predictions/": "V1 external predictions - never transferable (results known)",
    "Science_Fair_Extension/00_Governance/": "Governance records are cited, not copied",
    "Science_Fair_Extension/15_Manifests/": "Protocol records are cited, not copied",
}
KURIKOMA_NOTES = ("Kurikoma V2 Omega must come from the GSI aerial-photo interpretation range (S2EXT_KURIKOMA_GSI), which is not part of V1 "
                  "and must be acquired separately; the V1 processing catchment is excluded.")

# ---------------------------------------------------------------------------------------------------------------
# Erratum for Stage 2A.2 transient_inspection_manifest.json: hashes were mispaired (shasum argument order vs. ls order).
# Correct values recomputed from the retained scratch copies and, for GSI files, confirmed by byte-identical Stage-2B re-downloads.
STAGE2A2_ERRATUM = [
    ("https://zenodo.org/records/5027004/files/landslide_data.zip", "60a7aed1f049371bfd61c48bab385aabf2a43e3b7d548f6793e61fa9db1c5079",
     "58c747ca16f93cd3f9a7d2378cee52c6b4c2e5243490eff472e5761a28e443e4"),
    ("https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/noto_suzu-wajimaEAST_20240106.zip",
     "cf41e0c9c876c157380f005dafed8ae2edcf351a209992046dd384a9080ed57d", "60a7aed1f049371bfd61c48bab385aabf2a43e3b7d548f6793e61fa9db1c5079"),
    ("https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/noto_wajimaCENTRAL_20240115.zip",
     "56fe42587d5d6e2d494ecdd2c5cac3d71f760d5f4a389a26a4104a8616835d0e", "bf208c57272c6b65dbd9d5ae6b12815818e0f8a89a395f06e433f357c482dea3"),
    ("https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/noto_wajimaWEST_20240122.zip",
     "0b44ba10b98d3be69ff92fc27b3690a8889a36d9e05368372a1542b9e6b0bb5f", "550d515b91502567d48e284467a8495d818449dd42b9ce2c0a4af2609dadbff3"),
    ("https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/noto_anamizu_20240118.zip",
     "550d515b91502567d48e284467a8495d818449dd42b9ce2c0a4af2609dadbff3", "2eedbbdd47f2ff0b7428178f073f1367bd876ec0b73df6d673e41087d990cabb"),
    ("https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/noto_nanao_20240118.zip",
     "7a661bcfcccd478834c1d9696c2671b5f9cdaf8f449d8a4f2136165551893fb4", "408ca6c555abea36169e9a71f6daef62f8ccc9e830d5b258eb1add290cbef908"),
    ("https://unosat-maps.web.cern.ch/ID/LS20220308IDN/LS20220308IDN_SHP.zip", "ffef0abbd46ff824fea6af55cc96f7be6a00918b9b9461ff5451789f7cd04879",
     "7a661bcfcccd478834c1d9696c2671b5f9cdaf8f449d8a4f2136165551893fb4"),
    ("https://unosat.org/static/unosat_filesystem/4013/TC20241028PHL_SHP.zip", "1293c7b49d95071d2d4528e95e00f8d6449dc752e811c406739b9dcff9138c1a",
     "cf41e0c9c876c157380f005dafed8ae2edcf351a209992046dd384a9080ed57d"),
    ("https://www.sciencebase.gov/catalog/file/get/61f04680d34e8b818adc32a8 (mapping_extent.shp)",
     "ce19a17099acb2234dc112c6d5640bbaa33b1a2cb360f8ba2d0261fc0bd3cdd0", "56fe42587d5d6e2d494ecdd2c5cac3d71f760d5f4a389a26a4104a8616835d0e"),
]
