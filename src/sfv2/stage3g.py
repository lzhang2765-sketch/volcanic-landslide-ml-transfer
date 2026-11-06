"""Stage 3G gap-closure and validation-design audit.

This stage is deliberately metadata-only.  It registers public source bytes,
audits landscape independence and validation arithmetic, and writes no terrain
units, features, targets, models, predictions, or H products.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

from .schema import ROOT


AS_OF = "2026-10-06"
RETRIEVED = "2026-10-06T17:21:22+10:00"
OUT = ROOT / "outputs/stage3g"
RAW = ROOT / "data/raw/stage3g"

PRIOR = {
    "outputs/stage2f2/STAGE2F2_CHECKPOINT.json": "5357dd80f3f5430cb7a34388c658d5d7d84794bd341c97587c652c7b1b2c7b91",
    "outputs/stage3a/STAGE3A_CHECKPOINT.json": "c2d2af66034765171a687bb778255a66b83ccda921071e8b0ef2a65b8e591351",
    "outputs/stage3b/STAGE3B_CHECKPOINT.json": "73ea92c57e4699172916765c8d66de94e2d84c37aa4217ddf2f6dbddae9e0dc4",
    "outputs/stage3c/STAGE3C_CHECKPOINT.json": "c1f2b3a17634cd65c583afd8035f93ca8114caddeab0a732aca6431143d4ebcc",
    "outputs/stage3d/STAGE3D_CHECKPOINT.json": "ad13ed7ca649f328e1f01e51314b1da9b6e7d4f949f259577cfb94b348e00c38",
    "outputs/stage3e/STAGE3E_CHECKPOINT.json": "24efe4647b8212f85f8bc39045a42b574c5fe3f18e2823ab9245d43ecee70257",
    "outputs/stage3f/STAGE3F_CHECKPOINT.json": "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a",
}

PRODUCTION = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
]

LANDSCAPES = [
    dict(landscape_id="KUMAMOTO_ASO_2016", physical_volcanic_system="ASO_CALDERA",
         event="2016_KUMAMOTO_EARTHQUAKE", country="JAPAN", continent="ASIA",
         tectonic_setting="SUBDUCTION_ARC", morphology="CALDERA_COMPLEX",
         erosion_state="MIXED_YOUNG_TO_DISSECTED", climate="HUMID_SUBTROPICAL",
         elevation_class="MONTANE", trigger="EARTHQUAKE",
         geology_composition="MIXED_VOLCANIC_PYROCLASTIC_VOLCANICLASTIC",
         target_mapping_method="GSI_ORTHOPHOTO_VISUAL_MAPPING_POINT_CENTRES",
         target_source_institution="GEOSPATIAL_INFORMATION_AUTHORITY_OF_JAPAN",
         dem_source="COPERNICUS_GLO30", geology_source="GSJ_SEAMLESS_PLUS_ASO_1_50000",
         v1_history="V1_UNSEEN", physical_independence="PASS", observational_independence="PASS",
         methodological_similarity="SHARES_GSI_AND_COP30_WITH_NOTO; NOT_DISQUALIFYING"),
    dict(landscape_id="NOTO_PENINSULA_2024", physical_volcanic_system="NOTO_CENOZOIC_VOLCANIC_TERRANE",
         event="2024_NOTO_PENINSULA_EARTHQUAKE", country="JAPAN", continent="ASIA",
         tectonic_setting="JAPAN_ARC_BACKARC_CRUSTAL", morphology="OLD_ERODED_VOLCANIC_TERRANE",
         erosion_state="OLD_DEEPLY_ERODED", climate="HUMID_TEMPERATE",
         elevation_class="LOW_TO_MID_RELIEF", trigger="EARTHQUAKE",
         geology_composition="MIXED_VOLCANIC_DOMINANT",
         target_mapping_method="GSI_MULTI_DATE_ORTHOPHOTO_POLYGON_MAPPING",
         target_source_institution="GEOSPATIAL_INFORMATION_AUTHORITY_OF_JAPAN",
         dem_source="COPERNICUS_GLO30", geology_source="GSJ_SEAMLESS_1_200000",
         v1_history="V1_UNSEEN", physical_independence="PASS", observational_independence="PASS",
         methodological_similarity="SHARES_GSI_AND_COP30_WITH_ASO; DIFFERENT_EVENT_AND_SYSTEM"),
    dict(landscape_id="LOMBOK_RINJANI_2018", physical_volcanic_system="RINJANI_SAMALAS_VOLCANIC_COMPLEX",
         event="2018_LOMBOK_EARTHQUAKE_SEQUENCE", country="INDONESIA", continent="ASIA",
         tectonic_setting="SUBDUCTION_ARC", morphology="YOUNG_STRATOVOLCANO_COMPLEX",
         erosion_state="YOUNG_TO_MATURE", climate="TROPICAL_SEASONAL",
         elevation_class="HIGH_MOUNTAIN", trigger="EARTHQUAKE",
         geology_composition="MIXED_VOLCANIC",
         target_mapping_method="MANUAL_SATELLITE_POLYGON_MAPPING_SOURCE_PLUS_RUNOUT",
         target_source_institution="USGS_GROUND_FAILURE_REPOSITORY_RESEARCH_TEAM",
         dem_source="COPERNICUS_GLO30", geology_source="BADAN_GEOLOGI_GEOMAP_DISTRICT_SHP",
         v1_history="V1_HISTORICAL", physical_independence="PASS", observational_independence="PASS",
         methodological_similarity="SHARES_EARTHQUAKE_TRIGGER_AND_COP30; DISTINCT_CAMPAIGN"),
    dict(landscape_id="COTABATO_DAVAO_2019", physical_volcanic_system="COTABATO_DAVAO_VOLCANIC_TERRANES",
         event="2019_COTABATO_DAVAO_EARTHQUAKE_SEQUENCE", country="PHILIPPINES", continent="ASIA",
         tectonic_setting="ARC_COLLISION_SUBDUCTION_MARGIN", morphology="DISSECTED_VOLCANIC_COMPLEX",
         erosion_state="MATURE_DISSECTED", climate="TROPICAL_WET",
         elevation_class="MONTANE", trigger="EARTHQUAKE",
         geology_composition="MIXED_VOLCANIC_VOLCANICLASTIC",
         target_mapping_method="MANUAL_SATELLITE_POLYGON_MAPPING",
         target_source_institution="ZENODO_7520726_RESEARCH_TEAM",
         dem_source="COPERNICUS_GLO30", geology_source="PHILIPPINES_MGB_OFFICIAL_GEOLOGY",
         v1_history="V1_UNSEEN", physical_independence="PASS", observational_independence="PASS",
         methodological_similarity="SHARES_EARTHQUAKE_AND_COP30; DISTINCT_COUNTRY_CAMPAIGN"),
    dict(landscape_id="ATITLAN_STAN_2005", physical_volcanic_system="ATITLAN_CALDERA_COMPLEX",
         event="HURRICANE_STAN_2005", country="GUATEMALA", continent="NORTH_AMERICA",
         tectonic_setting="SUBDUCTION_ARC", morphology="CALDERA_COMPLEX",
         erosion_state="MATURE_DISSECTED", climate="TROPICAL_HIGHLAND",
         elevation_class="HIGH_ALTITUDE", trigger="CYCLONE_STORM_PRECIPITATION",
         geology_composition="PYROCLASTIC_MIXED_VOLCANIC",
         target_mapping_method="PUBLISHED_MANUAL_POLYGON_INVENTORY",
         target_source_institution="PUBLISHED_ATITLAN_RESEARCH_INVENTORY",
         dem_source="NASADEM_SRTM_2000", geology_source="GUATEMALA_MAGA_OFFICIAL_GEOLOGY",
         v1_history="V1_HISTORICAL", physical_independence="PASS", observational_independence="PASS",
         methodological_similarity="SHARES_CYCLONE_POLYGONS_WITH_DOMINICA; DISTINCT_SYSTEM_CAMPAIGN"),
    dict(landscape_id="DOMINICA_ERIKA_2015", physical_volcanic_system="DOMINICA_VOLCANIC_ISLAND",
         event="TROPICAL_STORM_ERIKA_2015", country="DOMINICA", continent="NORTH_AMERICA",
         tectonic_setting="SUBDUCTION_ARC", morphology="VOLCANIC_ISLAND_COMPLEX",
         erosion_state="YOUNG_TO_DISSECTED", climate="TROPICAL_WET",
         elevation_class="MONTANE_ISLAND", trigger="CYCLONE_STORM_PRECIPITATION",
         geology_composition="INTERMEDIATE_LAVA_PYROCLASTIC_MIXED_VOLCANIC",
         target_mapping_method="MULTI_DATE_SATELLITE_MANUAL_POLYGON_INVENTORY",
         target_source_institution="NHESS_RESEARCH_DATASET",
         dem_source="COPERNICUS_GLO30", geology_source="DOMINICA_GOVERNMENT_1_100000_MAP",
         v1_history="V1_HISTORICAL", physical_independence="PASS", observational_independence="PASS",
         methodological_similarity="SHARES_CYCLONE_POLYGONS_WITH_ATITLAN; DISTINCT_SYSTEM_CAMPAIGN"),
]

DIVERSITY_AXES = ["continent", "climate", "trigger", "tectonic_setting", "morphology",
                  "erosion_state", "elevation_class", "geology_composition", "target_mapping_method"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_state() -> dict:
    hashes = {p: sha256(ROOT / p) if (ROOT / p).exists() else "MISSING" for p in PRIOR}
    sf = json.loads((ROOT / "outputs/stage3f/STAGE3F_CHECKPOINT.json").read_text())
    internal_errors = []
    for group in ("artifact_hashes", "production_artifact_hashes"):
        for rel, expected in sf[group].items():
            path = ROOT / rel
            if not path.exists() or sha256(path) != expected:
                internal_errors.append(rel)
    return {
        "prior_checkpoint_hashes": hashes,
        "prior_checkpoint_mismatches": [p for p in PRIOR if hashes[p] != PRIOR[p]],
        "stage3f_internal_artifact_mismatches": internal_errors,
        "production_landscape_list": sf["production_landscape_list"],
        "production_landscape_count": sf["production_landscape_count"],
        "unique_physical_landscape_count": sf["unique_physical_landscape_count"],
        "terrain_units": sf["production_terrain_units_total"],
        "positive_units": sf["production_positive_units_total"],
        "negative_units": sf["production_negative_units_total"],
        "dominica_maria_status": sf["gate_outcomes"]["DOMINICA_MARIA_2017"],
        "atitlan_status": sf["gate_outcomes"]["ATITLAN_STAN_2005"],
        "dominica_erika_status": sf["gate_outcomes"]["DOMINICA_ERIKA_2015"],
        "stage3f_unchanged": not internal_errors and hashes["outputs/stage3f/STAGE3F_CHECKPOINT.json"] == PRIOR["outputs/stage3f/STAGE3F_CHECKPOINT.json"],
    }


def acquisition_specs() -> list[dict]:
    cems = "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR325/"
    specs = [
        ("RWANDA_RUTSIRO_2023", "Y_OMEGA_VECTOR_ARCHIVE", "UNOSAT_PRODUCT_3579",
         "https://unosat.org/static/unosat_filesystem/3579/FL20230504RWA_SHP.zip", "PUBLIC_UNOSAT_PRODUCT; REUSE_TERMS_NOT_STATED_ON_PRODUCT_PAGE",
         "RWANDA_RUBAVU_2023/SRC_UNOSAT_FL20230504RWA/FL20230504RWA_SHP.zip"),
        ("RWANDA_RUTSIRO_2023", "METHOD_REPORT", "UNOSAT_PRODUCT_3579",
         "https://unosat.org/static/unosat_filesystem/3579/UNOSAT_Preliminary_Assessment_Report_FL20230504RWA_Rubavu_Rwanda_9May2023.pdf", "PUBLIC_UNOSAT_PRODUCT; REUSE_TERMS_NOT_STATED_ON_PRODUCT_PAGE",
         "RWANDA_RUBAVU_2023/SRC_UNOSAT_FL20230504RWA/UNOSAT_Preliminary_Assessment_Report_FL20230504RWA_Rubavu_Rwanda_9May2023.pdf"),
        ("CHUUK_CHATAAN_2002", "METHOD_REPORT", "USGS_OFR_2004_1348",
         "https://pubs.usgs.gov/of/2004/1348/ofr20041348_pamphlet_Revision.pdf", "US_GOVERNMENT_WORK_PUBLIC_DOMAIN",
         "CHUUK_CHATAAN_2002/SRC_USGS_OFR2004_1348/ofr20041348_pamphlet_Revision.pdf"),
        ("GLOBAL_NASA_COOLR_2019_2023", "MULTI_EVENT_Y_POINTS", "FIGSHARE_26972467",
         "https://figshare.com/ndownloader/files/54363830", "CC_BY_4_0",
         "GLOBAL_NASA_COOLR_2019_2023/SRC_FIGSHARE_26972467/nasa_coolr_new_events.zip"),
        ("REUNION_SALAZIE_1978_1984", "Y_AND_CIRQUE_BOUNDARY", "MENDELEY_GVKJPP87XJ_V3",
         "https://data.mendeley.com/public-api/zip/gvkjpp87xj/download/3", "CC_BY_4_0",
         "REUNION_SALAZIE_1978_1984/SRC_MENDELEY_GVKJPP87XJ_V3/gvkjpp87xj_v3.zip"),
        ("MADEIRA_FUNCHAL_2010", "Y_POLYGONS", "CM_FUN_CHAL_ARCGIS_DPE_RISCOS_LAYER6",
         "https://sig.cm-funchal.pt/arcgis/rest/services/Intranet/DPE_RISCOS/MapServer/6/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=pjson", "PUBLIC_OFFICIAL_SERVICE; LICENCE_NOT_STATED",
         "MADEIRA_FUNCHAL_2010/SRC_CM_FUN_CHAL_ARCGIS_DPE_RISCOS/layer6_features.json"),
        ("MADEIRA_FUNCHAL_2010", "SERVICE_METADATA", "CM_FUN_CHAL_ARCGIS_DPE_RISCOS_LAYER6",
         "https://sig.cm-funchal.pt/arcgis/rest/services/Intranet/DPE_RISCOS/MapServer/6?f=pjson", "PUBLIC_OFFICIAL_SERVICE; LICENCE_NOT_STATED",
         "MADEIRA_FUNCHAL_2010/SRC_CM_FUN_CHAL_ARCGIS_DPE_RISCOS/layer6_metadata.json"),
    ]
    for name in [
        "EMSR325_01KUFUWEST_01DELINEATION_MAP_v1_vector.zip",
        "EMSR325_01KUFUWEST_02GRADING_MAP_v1_vector.zip",
        "EMSR325_02KUFUEAST_01DELINEATION_MAP_v1_vector.zip",
        "EMSR325_02KUFUEAST_02GRADING_MAP_v1_vector.zip",
        "EMSR325_03BUSHIKA_02GRADING_MAP_v1_vector.zip",
    ]:
        specs.append(("MOUNT_ELGON_BAKALASI_2018", "Y_OMEGA_RESPONSE_MAPPING", "COPERNICUS_EMSR325",
                      cems + name, "COPERNICUS_EMS_EU_REUSE_WITH_ATTRIBUTION",
                      "MOUNT_ELGON_BAKALASI_2018/SRC_COPERNICUS_EMSR325/" + name))
    rows = []
    for candidate, role, source, url, licence, relative in specs:
        path = RAW / relative
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({"candidate": candidate, "artifact_role": role, "source": source,
                     "url_or_doi": url, "licence": licence, "retrieval_timestamp": RETRIEVED,
                     "relative_path": str(path.relative_to(ROOT)), "filename": path.name,
                     "bytes": path.stat().st_size, "sha256": sha256(path)})
    return sorted(rows, key=lambda r: (r["candidate"], r["relative_path"]))


def candidates() -> list[dict]:
    base = [
        ("ST_LUCIA_TOMAS_2010", "PROMISING_ONE_BLOCKER", "NORTH_AMERICA", "CYCLONE_STORM", "SUBDUCTION_ARC", "VOLCANIC_ISLAND_COMPLEX", "PASS", "BLOCKED", "PASS_ROUTE", "PASS", "PASS_ROUTE", "PASS", "PASS", "PASS", 10, "TRUE", "HIGH", "Exact target-independent 2011 searched/visible polygon and cloud-shadow-fringe exclusions"),
        ("VANUATU_HAROLD_2020", "PROMISING_REQUEST_ONLY", "OCEANIA", "CYCLONE_STORM", "SUBDUCTION_ARC", "VOLCANIC_ISLAND_ARC", "REQUEST", "REQUEST", "PASS", "PASS", "PASS_ROUTE", "PASS", "PASS", "PASS", 7, "TRUE", "HIGH", "Original 1,481 polygons, mapped boundary, provenance and reuse licence"),
        ("POPOCATEPETL_PUEBLA_2017", "PROMISING_REQUEST_ONLY", "NORTH_AMERICA", "EARTHQUAKE", "SUBDUCTION_ARC", "YOUNG_STRATOVOLCANO", "REQUEST", "REQUEST", "PASS", "PASS", "PASS_ROUTE", "PASS", "PASS", "PASS", 7, "TRUE", "HIGH", "Study Area C polygons, search boundary, cloud/snow mask, provenance, CRS and licence"),
        ("CHUUK_CHATAAN_2002", "PROMISING_ONE_BLOCKER", "OCEANIA", "CYCLONE_STORM", "HOTSPOT_INTRA_PLATE", "OCEANIC_HOTSPOT_ISLANDS", "PASS", "BLOCKED", "PASS_ROUTE", "PASS", "PASS_ROUTE", "PASS", "PASS", "PASS", 10, "TRUE", "VERY_HIGH_IF_RESCUED", "No systematic target-independent aerial-survey coverage geometry; Y-derived convex hull rejected"),
        ("MOUNT_ELGON_BAKALASI_2018", "HOLD", "AFRICA", "RAINFALL", "EAST_AFRICAN_RIFT_ASSOCIATED", "ERODED_SHIELD_VOLCANO", "FAIL_LANDSCAPE_SCALE", "PASS_RESPONSE_AOIS", "PASS_ROUTE", "PASS", "PASS_ROUTE", "LIKELY", "FAIL_RESPONSE_IMPACT_ONLY", "PASS", 12, "TRUE", "HIGH_IF_FULL_INVENTORY_FOUND", "EMSR325 contains only 9 mass-movement polygons: 3 mudflows and 6 landslides; not an exhaustive landscape inventory"),
        ("RWANDA_RUTSIRO_2023", "REJECT", "AFRICA", "RAINFALL", "EAST_AFRICAN_RIFT_REGION", "KIVU_HIGHLANDS", "FAIL_ONE_MULTIPART", "PASS_PRODUCT_AOI", "PASS_ROUTE", "PASS", "PASS_ROUTE", "FAIL_OUTSIDE_VOLCANIC_DOMAIN", "FAIL_FLOOD_CODED_PRELIMINARY", "PASS", 12, "TRUE", "LOW", "Delivered landslide geometry is at Kibingo/Rutsiro outside the required Virunga volcanic domain and is flood-coded/preliminary"),
        ("REUNION_SALAZIE_1978_1984", "HOLD", "AFRICA", "CYCLONE_DOMINANT_INTERVAL", "HOTSPOT_INTRA_PLATE", "ERODED_SHIELD_CIRQUE", "PASS_WITH_EVENT_AMBIGUITY", "BLOCKED_VISIBILITY_EXCLUSIONS", "BLOCKED_PRE_EVENT_M", "FAIL", "PASS_ROUTE", "PASS", "HOLD_DEBRIS_FLOW_DOMINANT_INTERVAL", "PASS", 10, "TRUE", "VERY_HIGH_IF_RESOLVED", "1,440 polygons are mapped for 1978-1984 and mainly Hyacinthe; missing distributed 1978 terrain and cloud/shadow masks"),
        ("MADEIRA_FUNCHAL_2010", "HOLD", "EUROPE", "EXTREME_RAINFALL", "HOTSPOT_INTRA_PLATE", "ERODED_SHIELD_ISLAND", "PASS_243_OFFICIAL_POLYGONS", "BLOCKED_SYSTEMATIC_SEARCH_SUPPORT", "PASS_ROUTE_NASADEM", "PASS", "PASS_ROUTE", "LIKELY", "UNCERTAIN_EVENT_PROVENANCE", "PASS", 10, "TRUE", "HIGH_IF_RESOLVED", "Official layer is parented to 2010 but does not document exhaustive mapping, exact searched area, or sufficiently explicit event attribution/reuse terms"),
        ("COOLR_KIVU_RIFT_2023", "REJECT", "AFRICA", "RAINFALL", "EAST_AFRICAN_RIFT_REGION", "KIVU_HIGHLANDS", "FAIL_POINT_ONLY", "BLOCKED", "PASS_ROUTE", "PASS", "PASS_ROUTE", "FAIL_MOSTLY_NON_VOLCANIC", "FAIL_POINT_TARGET", "PASS", 9, "TRUE", "LOW", "COOLR Rwanda/DRC entries are points, lack independent Omega, and inspected bounds are mostly outside volcanic terrain"),
    ]
    fields = ["candidate_id", "classification", "continent", "trigger", "tectonic_setting", "morphology",
              "Y_gate", "Omega_gate", "M_gate", "temporal_gate", "G_gate", "volcanic_domain_gate",
              "target_semantics_gate", "independent_group_gate", "machine_accessibility", "V1_unseen",
              "lockbox_diversity_value", "blocker"]
    return [dict(zip(fields, row)) for row in base]


def search_log() -> list[dict]:
    rows = [
        (1, "AFRICA_RIFT", "UNOSAT Rwanda products", "Public Rubavu package acquired; delivered landslide layer is one Rutsiro multipart feature outside Virunga volcanic domain"),
        (1, "AFRICA_RIFT", "NASA COOLR Figshare 26972467", "Public 15,274-point archive acquired; Rwanda 156 and DRC Masisi 22 manual points lack Omega and mostly miss volcanic terrain"),
        (1, "AFRICA_RIFT", "Copernicus EMSR325 Mount Elgon", "Five vector packages acquired; independent AOIs exist but only nine mass-movement impacts, not a landscape inventory"),
        (1, "AFRICA_RIFT", "Zenodo/Figshare/PANGAEA/OSF/Dryad/GFZ/NHESS/USGS/HDX/RMCA repositories", "No public event polygon Y plus independent Omega meeting all gates found for Virunga, Kivu, Ethiopian Rift or Cameroon Volcanic Line"),
        (2, "HOTSPOT_SHIELD", "Mendeley Salazie gvkjpp87xj v3", "CC BY archive acquired: 1,440 1978-1984 mainly-Hyacinthe polygons and cirque boundary; temporal M and visibility/event purity remain blocked"),
        (2, "HOTSPOT_SHIELD", "Funchal municipal ArcGIS DPE_RISCOS layer 6", "243 official polygons acquired; service proves public geometry but not exhaustive search support or exact event provenance"),
        (2, "HOTSPOT_SHIELD", "Madeira NHESS 2013", "Event inventory publication identified; semi-automatic/model-derived polygons are not admitted as frozen labels and no reusable vectors were found"),
        (2, "HOTSPOT_SHIELD", "Hawaii/Azores/Canary/Cape Verde/Samoa/Galapagos", "No stronger public event-specific polygon Y plus independent Omega combination found"),
        (3, "CHUUK_RESCUE", "USGS OFR 2004-1348 report", "Report acquired; broad aerial reconnaissance followed by detailed work at selected high-impact sites does not establish systematic coverage"),
        (3, "CHUUK_RESCUE", "USGS P14VTB5D and 2003-2022 inventory", "New/revised public data confirm affected-area mapping and later imagery, not a target-independent exhaustive 2002 observation mask"),
        (4, "REQUEST_BLOCKED", "Saint Lucia / Vanuatu / Popocatepetl", "No genuinely new missing artifacts found; Stage 3E states preserved"),
    ]
    return [dict(zip(["round", "axis", "route", "outcome"], r)) for r in rows]


def diversity_products() -> tuple[list[dict], list[dict], list[dict], dict]:
    long_rows, matrix_rows = [], []
    by_id = {r["landscape_id"]: r for r in LANDSCAPES}
    for a in PRODUCTION:
        row = {"landscape_id": a}
        for b in PRODUCTION:
            if a == b:
                distance, differing = 0.0, []
            else:
                differing = [axis for axis in DIVERSITY_AXES if by_id[a][axis] != by_id[b][axis]]
                distance = len(differing) / len(DIVERSITY_AXES)
            row[b] = f"{distance:.6f}"
            if a < b:
                long_rows.append({"landscape_a": a, "landscape_b": b,
                                  "different_axes": len(differing), "total_axes": len(DIVERSITY_AXES),
                                  "distance": f"{distance:.6f}", "differing_axes": ";".join(differing)})
        matrix_rows.append(row)
    ordered = sorted(long_rows, key=lambda r: (float(r["distance"]), r["landscape_a"], r["landscape_b"]))
    summary = {"method": "NORMALISED_UNWEIGHTED_HAMMING_DISTANCE_ON_NINE_PRESPECIFIED_METADATA_AXES",
               "descriptive_only": True, "axes": DIVERSITY_AXES,
               "most_redundant_pair": ordered[0], "most_complementary_pair": ordered[-1]}
    return LANDSCAPES, matrix_rows, long_rows, summary


def validation_arithmetic() -> list[dict]:
    ability = {4: "VERY_FRAGILE", 5: "FRAGILE", 6: "MINIMUM_EXPLORATORY", 7: "LIMITED",
               8: "MODERATE", 9: "MODERATE_PLUS", 10: "IMPROVED", 11: "IMPROVED",
               12: "STRONGEST_IN_RANGE_BUT_STILL_FEW_GROUPS"}
    rows = []
    for n in range(4, 13):
        share = 100 / n
        fragility = "HIGH" if n <= 5 else "MATERIAL" if n <= 7 else "MODERATE" if n <= 9 else "REDUCED_NOT_ELIMINATED"
        rows.append({"independent_landscapes": n, "LOLO_outer_folds": n,
                     "training_landscapes_per_outer_fold": n - 1,
                     "plausible_inner_landscape_folds": n - 1,
                     "inner_training_landscapes_per_fold": n - 2,
                     "between_landscape_variance_ability": ability[n],
                     "reserve_1_lockbox_leaves_development": n - 1,
                     "reserve_2_lockboxes_leaves_development": n - 2,
                     "reserve_3_lockboxes_leaves_development": n - 3,
                     "one_anomalous_landscape_share_pct": f"{share:.3f}",
                     "anomalous_landscape_fragility": fragility,
                     "model_fitting_performed": "FALSE"})
    return rows


def stopping_audit() -> dict:
    options = [
        {"option": "A_MINIMUM_DEFENSIBLE", "development": 6, "lockboxes": 2, "total": 8,
         "claim": "Evidence of transfer across several independent volcanic landscapes; exploratory generalisation only, with wide between-landscape uncertainty."},
        {"option": "B_STRONG_RECOMMENDED", "development": 8, "lockboxes": 3, "total": 11,
         "claim": "A credible multi-setting generalisation experiment with nested landscape-level selection, per-landscape reporting and three complementary untouched external tests."},
        {"option": "C_IDEAL", "development": 10, "lockboxes": 3, "total": 13,
         "claim": "Broader and less fragile cross-setting generalisation evidence, still bounded to represented domains rather than universal volcanic applicability."},
    ]
    return {
        "finding": "THE_OLD_11_TOTAL_RULE_IS_JUSTIFIED_AS_THE_RECOMMENDED_8_PLUS_3_DESIGN_NOT_AS_A_MAGIC_COUNT",
        "formal_amendment_recommended": True,
        "amendment": "Replace count-only stopping with count plus mandatory coverage and untouched-lockbox requirements.",
        "minimum_development": 6, "recommended_development": 8,
        "minimum_lockboxes": 2, "recommended_lockboxes": 3,
        "why_one_lockbox_is_weak": "It cannot separate a model failure from an idiosyncratic test landscape and cannot expose heterogeneity across external settings.",
        "what_three_adds": "A third complementary external setting reduces dependence on either single test and permits a basic assessment of external-performance heterogeneity.",
        "quantity_vs_diversity": "Both are necessary; redundant landscapes do not substitute for missing domain axes.",
        "recommended_composition": {
            "trigger": ["EARTHQUAKE", "PRECIPITATION_INCLUDING_CYCLONE"],
            "tectonics": ["SUBDUCTION", "NON_SUBDUCTION", "AT_LEAST_ONE_HOTSPOT_SHIELD", "AT_LEAST_ONE_CONTINENTAL_RIFT_ASSOCIATED"],
            "morphology": ["CALDERA_OR_COMPLEX", "STRATOVOLCANO", "OLD_ERODED_TERRAIN", "ISLAND_OR_SHIELD"],
            "climate": ["TROPICAL_WET", "TEMPERATE_OR_DRIER", "HIGH_ALTITUDE"],
            "geography": ["AT_LEAST_3_CONTINENTS_OR_MAJOR_REGIONS"],
            "validation": ["8_DEVELOPMENT", "3_V1_UNSEEN_V2_UNTOUCHED_COMPLEMENTARY_LOCKBOXES"],
        },
        "options": options,
        "literature_limit": "No cited method paper supplies a universal 6, 8, 11 or 13 landscape threshold; these are transparent design judgments from fold arithmetic, clustered-validation principles, lockbox purpose and required domain coverage.",
        "recommended_option": "B_STRONG_RECOMMENDED",
    }


def references() -> list[dict]:
    return [
        {"reference": "Roberts et al. (2017), Cross-validation strategies for structured data", "doi": "10.1111/ecog.02881", "url": "https://doi.org/10.1111/ecog.02881", "audit_use": "Blocking must match spatial/hierarchical dependence; supports landscape-group splits."},
        {"reference": "Varoquaux (2018), Cross-validation failure: small samples lead to large error bars", "doi": "10.1016/j.neuroimage.2017.06.061", "url": "https://doi.org/10.1016/j.neuroimage.2017.06.061", "audit_use": "Small validation samples yield unstable error estimates; fold-to-fold SE is not full uncertainty."},
        {"reference": "Varma and Simon (2006), Bias in error estimation with model selection", "doi": "10.1186/1471-2105-7-91", "url": "https://doi.org/10.1186/1471-2105-7-91", "audit_use": "Model and feature selection must be nested inside validation."},
        {"reference": "Riley et al. (2016), External validation using multiple datasets", "doi": "10.1136/bmj.i3140", "url": "https://www.bmj.com/content/353/bmj.i3140", "audit_use": "Multiple external datasets are needed to assess heterogeneity and generalisability."},
        {"reference": "Debray et al. (2023), TRIPOD-Cluster explanation and elaboration", "doi": "10.1136/bmj-2022-071058", "url": "https://www.bmj.com/content/380/bmj-2022-071058", "audit_use": "Internal-external CV omits each cluster in turn and reports heterogeneity; formal minimum cluster guidance is lacking."},
        {"reference": "Gulrajani and Lopez-Paz (2021), In Search of Lost Domain Generalization", "doi": "OPENREVIEW:lQdXeXDoWtI", "url": "https://openreview.net/pdf?id=lQdXeXDoWtI", "audit_use": "Domain-generalisation model selection must be explicit; an algorithm without a selection criterion is incomplete."},
        {"reference": "Yu et al. (2024), Rethinking the Evaluation Protocol of Domain Generalization", "doi": "CVPR_2024", "url": "https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Rethinking_the_Evaluation_Protocol_of_Domain_Generalization_CVPR_2024_paper.pdf", "audit_use": "Multiple test domains reduce dependence and leakage risk relative to a single target domain."},
    ]


def lockbox_audit(candidate_rows: list[dict]) -> list[dict]:
    rows = []
    for item in LANDSCAPES:
        rows.append({"landscape_id": item["landscape_id"], "state": "CURRENT_PRODUCTION",
                     "V1_UNSEEN": str(item["v1_history"] == "V1_UNSEEN").upper(),
                     "V2_UNUSED_FOR_DEVELOPMENT": "FALSE", "LOCKBOX_ELIGIBLE": "FALSE",
                     "LOCKBOX_DIVERSITY_VALUE": "NONE_AFTER_DEVELOPMENT_USE",
                     "reason": "Already admitted to V2 production/development; cannot be retroactively hidden."})
    for item in candidate_rows:
        eligible = item["V1_unseen"] == "TRUE" and item["classification"] != "REJECT"
        rows.append({"landscape_id": item["candidate_id"], "state": item["classification"],
                     "V1_UNSEEN": item["V1_unseen"], "V2_UNUSED_FOR_DEVELOPMENT": "TRUE",
                     "LOCKBOX_ELIGIBLE": "CONDITIONAL" if eligible else "FALSE",
                     "LOCKBOX_DIVERSITY_VALUE": item["lockbox_diversity_value"],
                     "reason": item["blocker"]})
    return rows


def lockbox_portfolios() -> dict:
    return {
        "not_frozen": True,
        "two_lockboxes": [
            {"candidate_id": "CHUUK_CHATAAN_2002", "value": "Oceania, cyclone, hotspot/non-subduction oceanic islands", "data_ready": False, "blocker": "independent systematic Omega"},
            {"candidate_id": "POPOCATEPETL_PUEBLA_2017", "value": "North America, earthquake, young high-altitude stratovolcano", "data_ready": False, "blocker": "promised Study Area C Y/Omega/cloud files"},
        ],
        "three_lockboxes": [
            {"candidate_id": "CHUUK_CHATAAN_2002", "value": "hotspot/cyclone/Oceania", "data_ready": False, "blocker": "Omega"},
            {"candidate_id": "POPOCATEPETL_PUEBLA_2017", "value": "subduction/earthquake/high-altitude North America", "data_ready": False, "blocker": "Y/Omega package"},
            {"candidate_id": "MOUNT_ELGON_BAKALASI_2018_FULL_INVENTORY", "value": "Africa/East-African-rift-associated eroded shield/rainfall", "data_ready": False, "blocker": "landscape-scale event inventory; EMSR325 has only nine response impacts"},
        ],
        "warning": "These are candidate portfolios only. No member is frozen and none is currently production-ready.",
    }


def route_to_final() -> dict:
    return {
        "current": {"development": 6, "lockboxes": 0, "total": 6},
        "minimum_8": {"target": "6_DEVELOPMENT_PLUS_2_LOCKBOXES", "sequence": ["CHUUK_CHATAAN_2002_AFTER_OMEGA", "POPOCATEPETL_PUEBLA_2017_AFTER_REQUEST"], "ready_now": False},
        "recommended_11": {"target": "8_DEVELOPMENT_PLUS_3_LOCKBOXES", "development_additions": ["ST_LUCIA_TOMAS_2010_AFTER_OMEGA", "REUNION_SALAZIE_1978_1984_AFTER_TEMPORAL_M_VISIBILITY_AND_EVENT_PURITY"], "lockboxes": ["CHUUK_CHATAAN_2002_AFTER_OMEGA", "POPOCATEPETL_PUEBLA_2017_AFTER_REQUEST", "MOUNT_ELGON_BAKALASI_2018_FULL_INVENTORY_AFTER_RECOVERY"], "ready_now": False},
        "ideal_13": {"target": "10_DEVELOPMENT_PLUS_3_LOCKBOXES", "additional_development_classes": ["VANUATU_HAROLD_2020_AFTER_REQUEST", "MADEIRA_FUNCHAL_2010_AFTER_OMEGA_AND_PROVENANCE"], "inherits_recommended_sequence": True, "ready_now": False},
        "can_continue_without_waiting": True,
        "qualification": "Machine search, source QA and blocker rescue can continue, but immediate production cannot: zero new candidates have complete raw gates.",
        "critical_user_actions": ["Send the already prepared exact-artifact requests for Saint Lucia, Vanuatu and Popocatepetl", "If pursuing the recommended route, request systematic coverage/full event inventory support for Chuuk and Mount Elgon"],
    }


def africa_audit() -> dict:
    return {
        "production_feasible_african_volcanic_landscape_found": False,
        "production_feasible_continental_rift_landscape_found": False,
        "best_lead": "MOUNT_ELGON_BAKALASI_2018",
        "best_lead_result": "Official public vectors and independent response AOIs acquired, but only nine mass-movement impact polygons; frozen target/landscape-scale gates fail.",
        "rwanda_result": "UNOSAT package acquired; one Rutsiro multipart landslide is outside required Virunga volcanic terrain and preliminary/flood-coded.",
        "coolr_result": "Public points add event leads but no admissible polygon Y or independent Omega; Rwanda and Masisi bounds do not rescue volcanic-domain coverage.",
        "verdict": "NO_READY_AFRICA_OR_CONTINENTAL_RIFT_CANDIDATE",
    }


def hotspot_audit() -> dict:
    return {
        "production_feasible_hotspot_or_shield_found": False,
        "salazie": {"features": 1440, "independent_cirque_boundary": True, "verdict": "HOLD", "blockers": ["event interval/mainly-Hyacinthe attribution", "cloud-shadow visibility exclusions", "pre-1978 terrain M"]},
        "madeira": {"official_polygons": 243, "verdict": "HOLD", "blockers": ["systematic searched-area evidence", "explicit event provenance", "reuse terms"]},
        "chuuk": {"verdict": "PROMISING_ONE_BLOCKER", "blocker": "independent systematic Omega"},
        "verdict": "NO_READY_HOTSPOT_SHIELD_CANDIDATE",
    }


def chuuk_audit() -> dict:
    return {
        "candidate_id": "CHUUK_CHATAAN_2002", "omega_rescued": False,
        "new_report_acquired": True, "source": "USGS_OFR_2004_1348",
        "evidence": ["Field reconnaissance occurred 27 July through 7 August 2002", "Aerial overflight provided a broad overview", "Detailed field work focused on selected high-impact sites", "At least 269 slides were inventoried on nine islands"],
        "fatal_gap": "No flight lines, photo footprints or other geometry prove a complete systematically inspected area; public metadata describes oblique photographs of landslide-affected areas.",
        "rejected": ["NHESS Y-derived mapped-within convex hull", "island outlines without proof of systematic inspection", "Y buffer"],
        "verdict": "OMEGA_REMAINS_BLOCKED",
    }


def report(state, candidate_rows, acquisitions, arithmetic, stopping, diversity_summary, tests, v1_status) -> str:
    gate_lines = "\n".join(f"- `{r['candidate_id']}` — **{r['classification']}**; Y={r['Y_gate']}, Omega={r['Omega_gate']}, M={r['M_gate']}, G={r['G_gate']}; blocker: {r['blocker']}" for r in candidate_rows)
    fold_lines = "\n".join(f"- N={r['independent_landscapes']}: {r['LOLO_outer_folds']} outer folds; {r['training_landscapes_per_outer_fold']} training groups/fold; {r['plausible_inner_landscape_folds']} inner folds; variance={r['between_landscape_variance_ability']}; one anomaly={r['one_anomalous_landscape_share_pct']}%." for r in arithmetic)
    independence = "\n".join(f"- `{r['landscape_id']}`: physical={r['physical_independence']}, observational={r['observational_independence']}; {r['methodological_similarity']}" for r in LANDSCAPES)
    matrix = ["| landscape | " + " | ".join(PRODUCTION) + " |", "|---|" + "---:|" * len(PRODUCTION)]
    _, matrix_rows, _, _ = diversity_products()
    for row in matrix_rows:
        matrix.append("| " + row["landscape_id"] + " | " + " | ".join(row[x] for x in PRODUCTION) + " |")
    return f"""STAGE 3G GLOBAL GAP-CLOSURE + STOPPING-RULE AUDIT: COMPLETE

## A. EXECUTIVE RESULT

- Current production remains **6 independent landscapes, 835,846 terrain units and 8,793 positives, Tier 2 only**.
- Four serious new named landscape leads were added (Rutsiro, Mount Elgon, Salazie and Madeira), plus the multi-event COOLR source; **6 public source packages / {len(acquisitions)} original artifacts** were acquired and checksum-registered.
- Newly production-ready candidates: **0**. Africa/rift: no. Hotspot/shield: no. Chuuk Omega: not rescued.
- Recommended stopping rule: **8 development + 3 complementary untouched lockboxes = 11 total, with mandatory coverage**. Eleven is retained as a recommended design, not a magic count.

## B. CURRENT SIX-LANDSCAPE INDEPENDENCE AUDIT

{independence}

All six pass physical/event independence. GSI/Copernicus/manual-mapping similarities are methodological similarities, not physical leakage. The validation group remains physical landscape/event system. All six have already entered V2 development, so none is a future final lockbox.

## C. GLOBAL GAP-CLOSURE RESULTS

- Machine-first searches covered the requested repositories and event/volcano routes. Public leads were downloaded, inspected and failed closed where Ω, temporal M, target semantics or volcanic-domain support was absent.
- Saint Lucia, Vanuatu and Popocatépetl retain their exact Stage 3E states because no new missing artifacts were found.

## D. AFRICA / RIFT RESULTS

- **No production-feasible African volcanic or continental-rift landscape was found.** EMSR325 closes access but not target validity: Kufu/Bushika contains only three mudflows and six landslides. UNOSAT Rutsiro is outside the required Virunga volcanic domain. COOLR provides points without Ω.

## E. HOTSPOT / SHIELD RESULTS

- Salazie is the strongest newly downloaded lead: 1,440 polygons and an independent cirque boundary, but the product is 1978–1984 “mainly” Hyacinthe, lacks distributed pre-1978 terrain M, and lacks visibility exclusions.
- Madeira provides 243 official municipal polygons, but not evidence of an exhaustive searched area or sufficiently explicit event provenance. Neither candidate is ready.

## F. CHUUK RESCUE

- **Not rescued.** The USGS report documents broad overflight reconnaissance followed by detailed work at selected high-impact sites; no public flight-line/photo-footprint support establishes a complete systematic Ω. The Y-derived convex hull remains rejected.

## G. NEW PUBLIC ACQUISITIONS

- {len(acquisitions)} original files across UNOSAT Rwanda, Copernicus EMSR325, USGS Chuuk, NASA COOLR/Figshare, Mendeley Salazie and the Funchal municipal ArcGIS service. Full URLs, licences/reuse status, byte sizes and SHA256 values are in `STAGE3G_PUBLIC_ACQUISITIONS.tsv`.

## H. CANDIDATE GATE TABLE

{gate_lines}

## I. VALIDATION-STRUCTURE AUDIT

{fold_lines}

## J. STOPPING-RULE EVIDENCE

- Structured data require group-aware blocking; all selection must be nested; small numbers of independent groups produce unstable performance estimates; multiple external settings are needed to assess heterogeneity. No authoritative source supplies a universal “11” threshold.
- One lockbox is weak because its result is inseparable from that landscape's idiosyncrasy. Two are minimally defensible; three allow complementary external tests and a basic heterogeneity check.

## K. MINIMUM / RECOMMENDED / IDEAL OPTIONS

- **Minimum:** 6 development + 2 lockboxes = **8**. Supports only an exploratory claim of transfer across several represented volcanic landscapes.
- **Recommended:** 8 development + 3 lockboxes = **11**. Supports a credible multi-setting generalisation experiment with nested selection and multiple external settings.
- **Ideal:** 10 development + 3 lockboxes = **13**. Reduces group fragility and broadens represented settings, but still does not justify universal claims.

## L. RECOMMENDED COMPOSITION-BASED RULE

- Use **count + coverage**, not count alone: seismic and precipitation triggers; subduction and non-subduction; at least one hotspot/shield and one continental-rift-associated landscape; caldera/complex, stratovolcano, old-eroded and island/shield morphology; tropical-wet, temperate/drier and high-altitude climate; at least three continents/major regions; and three V1-unseen, V2-untouched complementary lockboxes.

## M. CURRENT SIX-LANDSCAPE DIVERSITY MATRIX

{chr(10).join(matrix)}

- Distance is descriptive normalised Hamming distance across nine predeclared metadata axes. Most redundant: `{diversity_summary['most_redundant_pair']['landscape_a']}`–`{diversity_summary['most_redundant_pair']['landscape_b']}` ({diversity_summary['most_redundant_pair']['distance']}). Most complementary: `{diversity_summary['most_complementary_pair']['landscape_a']}`–`{diversity_summary['most_complementary_pair']['landscape_b']}` ({diversity_summary['most_complementary_pair']['distance']}).

## N. LOCKBOX PORTFOLIO OPTIONS

- Two: Chuuk–Chataan + Popocatépetl–Puebla. This contrasts hotspot/cyclone/Oceania with young high-altitude subduction/earthquake/North America.
- Three: add a full Mount Elgon–Bakalasi 2018 landscape inventory to test Africa, rainfall and rift-associated eroded shield terrain. None is data-ready and nothing is frozen.

## O. NEXT PRODUCTION BATCH

- **Empty (0 rows).** No candidate has complete raw Y/Ω/M/G/temporal/target gates; a batch was not manufactured.

## P. FASTEST ROUTE FROM 6

- Minimum 8: preserve the six as development; rescue Chuuk Ω and obtain Popocatépetl Study Area C as two untouched lockboxes.
- Recommended 11: additionally resolve Saint Lucia for development, resolve Salazie temporal-M/visibility/event purity for development, and recover a full Mount Elgon–Bakalasi landscape inventory as the third lockbox.
- Ideal 13: the recommended route plus Vanuatu–Harold and Madeira–Funchal after their blockers are resolved.

## Q. REQUEST DEPENDENCIES

- Acquisition/search can continue without waiting, but **production expansion cannot currently proceed**. The prepared Saint Lucia, Vanuatu and Popocatépetl requests are now genuine critical-path actions unless equally strong machine-ready replacements appear. Chuuk coverage and full Mount Elgon inventory support are also critical to the recommended composition.

## R. TESTS / CHECKPOINT / V1 / GIT

- Stage 3F and all prior checkpoints: unchanged; production remains exactly 6 / 835,846 / 8,793. Acquisitions are checksum-registered. No new production rows, ML, LOLO, H, Y-derived Ω, Tier mixing or duplicate landscape counting.
- Tests: {tests}. V1: {v1_status}. Git: dirty pre-existing worktree inspected; no commit and no push.

1. Did we find a production-feasible African volcanic landscape? **No.**
2. Did we find a production-feasible continental-rift landscape? **No.**
3. Did we find a production-feasible hotspot/shield landscape? **No.**
4. Was Chuuk's Omega rescued? **No.**
5. How many completely machine-ready new candidates now exist? **Zero.**
6. What should Stage 3H process? **No production batch; only a separately authorised acquisition/rescue stage for the named blockers.**
7. Is the old 11-landscape stopping rule scientifically justified? **Yes as the recommended 8-development + 3-lockbox, coverage-constrained design; no as an unconditional magic count.**
8. What is the minimum defensible number of development landscapes? **Six.**
9. What is the recommended number of development landscapes? **Eight.**
10. How many untouched final lockboxes are minimally defensible? **Two.**
11. How many final lockboxes are recommended? **Three.**
12. What total landscape count is MINIMUM DEFENSIBLE? **Eight.**
13. What total landscape count is RECOMMENDED? **Eleven.**
14. What total landscape count is IDEAL? **Thirteen.**
15. Which diversity dimensions are mandatory in the recommended rule? **Trigger, tectonics, morphology, erosion state, climate/elevation, geography, geology/observation-method breadth, hotspot/shield and continental-rift representation, and three complementary untouched lockboxes.**
16. What claims could the minimum design support? **Exploratory transfer across several independent represented volcanic landscapes, with explicit wide uncertainty.**
17. What additional claims could the recommended design support? **A credible multi-setting cross-landscape generalisation experiment across the represented axes, not universal volcanic-landscape generalisation.**
18. What is the strongest candidate 2-lockbox portfolio? **Chuuk–Chataan + Popocatépetl–Puebla, conditional on their blockers.**
19. What is the strongest candidate 3-lockbox portfolio? **Chuuk–Chataan + Popocatépetl–Puebla + a full Mount Elgon–Bakalasi 2018 landscape inventory, conditional on blockers.**
20. Can acquisition continue without waiting for user-sent requests? **Yes for search and source QA, but not for immediate production.**
21. What exact candidates form the fastest route from 6 to the recommended
    dataset? **Development: Saint Lucia–Tomas and Réunion–Salazie; untouched lockboxes: Chuuk–Chataan, Popocatépetl–Puebla and full Mount Elgon–Bakalasi 2018, all conditional on listed blockers.**
22. Are any user actions now on the critical path? **Yes: send the already prepared Saint Lucia, Vanuatu and Popocatépetl requests; Chuuk/Mount Elgon coverage requests are also required for the recommended route unless replacements are found.**
23. Are we ready to train ML right now? **NO.** The audited rule is not met: only six development landscapes exist; there are zero frozen untouched lockboxes, no production-ready hotspot/shield or continental-rift landscape, and insufficient geographic/tectonic breadth.
"""


def main(tests: str = "PENDING", v1_status: str = "PENDING") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    state = verify_state()
    if (state["prior_checkpoint_mismatches"] or state["stage3f_internal_artifact_mismatches"] or
            state["production_landscape_list"] != PRODUCTION or state["production_landscape_count"] != 6 or
            state["unique_physical_landscape_count"] != 6 or state["terrain_units"] != 835846 or
            state["positive_units"] != 8793):
        raise RuntimeError(f"Frozen state verification failed: {state}")
    write_json(OUT / "STATE_VERIFICATION.json", state)
    acquisitions = acquisition_specs()
    write_tsv(OUT / "STAGE3G_PUBLIC_ACQUISITIONS.tsv", acquisitions)
    candidate_rows = candidates()
    write_tsv(OUT / "STAGE3G_CANDIDATE_GATES.tsv", candidate_rows)
    write_tsv(OUT / "STAGE3G_SEARCH_LOG.tsv", search_log())
    metadata, matrix, long_matrix, diversity_summary = diversity_products()
    write_tsv(OUT / "STAGE3G_CURRENT_INDEPENDENCE.tsv", metadata)
    write_tsv(OUT / "STAGE3G_DIVERSITY_METADATA.tsv", metadata)
    write_tsv(OUT / "STAGE3G_PAIRWISE_DIVERSITY_MATRIX.tsv", matrix,
              ["landscape_id", *PRODUCTION])
    write_tsv(OUT / "STAGE3G_PAIRWISE_DIVERSITY_LONG.tsv", long_matrix)
    write_json(OUT / "STAGE3G_DIVERSITY_SUMMARY.json", diversity_summary)
    write_tsv(OUT / "STAGE3G_LOCKBOX_AUDIT.tsv", lockbox_audit(candidate_rows))
    write_json(OUT / "STAGE3G_LOCKBOX_PORTFOLIOS.json", lockbox_portfolios())
    write_json(OUT / "AFRICA_RIFT_AUDIT.json", africa_audit())
    write_json(OUT / "HOTSPOT_SHIELD_AUDIT.json", hotspot_audit())
    write_json(OUT / "CHUUK_RESCUE_AUDIT.json", chuuk_audit())
    arithmetic = validation_arithmetic()
    write_tsv(OUT / "STAGE3G_VALIDATION_ARITHMETIC.tsv", arithmetic)
    stopping = stopping_audit()
    write_json(OUT / "STAGE3G_STOPPING_RULE_AUDIT.json", stopping)
    write_tsv(OUT / "STAGE3G_STOPPING_RULE_REFERENCES.tsv", references())
    route = route_to_final()
    write_json(OUT / "STAGE3G_ROUTE_TO_FINAL.json", route)
    next_fields = ["rank", "candidate_id", "continent", "trigger", "tectonic_setting", "morphology",
                   "Y_gate", "Omega_gate", "M_gate", "G_gate", "temporal_gate", "target_gate",
                   "machine_accessibility", "V1_unseen", "lockbox_potential", "diversity_gain",
                   "raw_complete", "recommended_action"]
    write_tsv(OUT / "STAGE3G_NEXT_BATCH_PRODUCTION.tsv", [], next_fields)
    diagnostics = {
        "stage": "3G", "production_rows_added": 0, "machine_ready_new_candidates": 0,
        "ml_trained": False, "exploratory_ml_run": False, "lolo_predictions_run": False,
        "h_generated": False, "susceptibility_maps_built": False, "y_derived_omega_accepted": False,
        "tier_mixing": False, "duplicate_physical_landscape_counting": False,
        "emails_sent": False, "commit_performed": False, "push_performed": False,
        "request_states_preserved": {"ST_LUCIA_TOMAS_2010": "PROMISING_ONE_BLOCKER",
                                     "VANUATU_HAROLD_2020": "PROMISING_REQUEST_ONLY",
                                     "POPOCATEPETL_PUEBLA_2017": "PROMISING_REQUEST_ONLY"},
    }
    write_json(OUT / "STAGE3G_DIAGNOSTICS.json", diagnostics)
    (OUT / "STAGE3G_REPORT.md").write_text(report(state, candidate_rows, acquisitions, arithmetic, stopping,
                                                   diversity_summary, tests, v1_status))
    artifact_names = sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name != "STAGE3G_CHECKPOINT.json")
    checkpoint = {
        "stage": "3G", "as_of": AS_OF, "input_checkpoint_hashes": state["prior_checkpoint_hashes"],
        "acquisition_hashes": {r["relative_path"]: r["sha256"] for r in acquisitions},
        "candidate_states": {r["candidate_id"]: r["classification"] for r in candidate_rows},
        "current_production_state": {"landscapes": PRODUCTION, "count": 6, "independent_physical_landscapes": 6,
                                     "terrain_units": 835846, "positive_units": 8793, "target_tier": "TIER_2_ONLY"},
        "diversity_state": {"current_major_gaps": ["C1_COUNT", "C2_LOCKBOXES", "C6_NON_SUBDUCTION_BREADTH", "C8_GEOGRAPHY", "C10_HOTSPOT_AND_RIFT_PRODUCTION"],
                            "pairwise_summary": diversity_summary},
        "validation_arithmetic": arithmetic, "stopping_rule_audit": stopping,
        "recommended_stopping_rule_option": "B_STRONG_RECOMMENDED_8_DEVELOPMENT_PLUS_3_LOCKBOXES_TOTAL_11_WITH_COVERAGE",
        "next_production_batch": [], "route_to_final": route,
        "tests": tests, "v1_state": v1_status,
        "git_state": "DIRTY_PRE_EXISTING_WORKTREE_INSPECTED; NO_COMMIT; NO_PUSH",
        "prohibitions": diagnostics,
        "artifact_hashes": {f"outputs/stage3g/{name}": sha256(OUT / name) for name in artifact_names},
    }
    write_json(OUT / "STAGE3G_CHECKPOINT.json", checkpoint)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="PENDING")
    parser.add_argument("--v1-status", default="PENDING")
    args = parser.parse_args()
    main(args.tests, args.v1_status)
