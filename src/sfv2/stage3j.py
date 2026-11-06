"""Stage 3J blocker consolidation and machine-accessible rescue audit.

This module only writes Stage 3J audit/checkpoint products.  It cannot run ML,
construct production H, generate terrain units, admit a candidate, or alter a
prior checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .schema import ROOT


AS_OF = "2026-10-06"
RETRIEVED = "2026-10-06T19:10:00+10:00"
OUT = ROOT / "outputs/stage3j"
RAW = ROOT / "data/raw/stage3j"
CHECKPOINT_HASHES = {
    "stage3f": "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a",
    "stage3g": "a4f8035123658c063f1572bdfd3276906053ba313735ae3371c442e45cbe4664",
    "stage3h": "f1ea33e848559bcc97f6c8fb7bc7a78c8fca9d004fe924d8c0136d1f488f2d1f",
    "stage3i": "90072d3562da197c46a0dee0bdc0e1fe752a242c520ee9f44a09bd1b09bc9ea1",
}
PRODUCTION = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def state_verification() -> dict:
    observed = {
        stage: sha256(ROOT / f"outputs/{stage}/STAGE{stage[5:].upper()}_CHECKPOINT.json")
        for stage in CHECKPOINT_HASHES
    }
    stage3g = json.loads((ROOT / "outputs/stage3g/STAGE3G_CHECKPOINT.json").read_text())
    production = stage3g["current_production_state"]
    exact = (
        production["landscapes"] == PRODUCTION
        and production["count"] == 6
        and production["terrain_units"] == 835846
        and production["positive_units"] == 8793
        and production["terrain_units"] - production["positive_units"] == 827053
        and production["target_tier"] == "TIER_2_ONLY"
    )
    return {
        "checkpoint_hashes_expected": CHECKPOINT_HASHES,
        "checkpoint_hashes_observed": observed,
        "all_prior_checkpoint_hashes_unchanged": observed == CHECKPOINT_HASHES,
        "production_state": {
            "landscapes": production["landscapes"], "count": production["count"],
            "terrain_units": production["terrain_units"], "positive_units": production["positive_units"],
            "negative_units": production["terrain_units"] - production["positive_units"],
            "target_tier": production["target_tier"],
        },
        "production_state_exact": exact,
    }


def blocker_ledger() -> list[dict]:
    fields = [
        "candidate_id", "physical_landscape", "country", "region", "volcano_type",
        "tectonic_setting", "trigger", "target_tier", "Y_status", "Omega_status", "M_status",
        "G_status", "H_status", "temporal_status", "target_completeness_status",
        "development_or_lockbox_role", "V1_unseen", "V2_exposed", "lockbox_eligible",
        "exact_remaining_blocker", "human_request_status", "machine_rescue_possible", "priority", "next_action",
    ]
    data = [
        ("TALAKMAU_2022", "Mount Talamau/Talakmau", "Indonesia", "West Sumatra", "STRATOVOLCANO", "SUBDUCTION_ARC", "EARTHQUAKE", "TIER_2", "HOLD_PRELIMINARY_TO_BE_EVALUATED_NOT_FIELD_VALIDATED", "PASS_ANALYSIS_EXTENT_MINUS_CLOUD_INDEPENDENT_OF_Y", "PASS_COP30_PRE_EVENT", "HOLD_OFFICIAL_GEOMAP_ATTRIBUTE_GEOGRAPHY_MISMATCH", "PARTIAL_RECONSTRUCTION_FEASIBLE_NOT_BUILT", "PASS", "HOLD_PRELIMINARY_INVENTORY", "DEVELOPMENT", "TRUE", "TRUE", "FALSE", "Independent Y validation; official corrected geology/crosswalk/service restoring geometry-to-unit identity", "NO_REQUEST_RECORDED_PARKED", "NO_NEW_AUTHORITATIVE_ROUTE_FOUND", "PARKED", "Wait for genuinely new Badan Geologi correction and independent inventory validation; do not retry failing raster portal"),
        ("ST_LUCIA_TOMAS_2010", "Saint Lucia volcanic island", "Saint Lucia", "Island-wide Tomas mapping", "VOLCANIC_ISLAND_COMPLEX", "SUBDUCTION_ARC", "CYCLONE_STORM", "TIER_2", "PASS_2011_TOMAS_POLYGONS", "BLOCKED_EXACT_SEARCH_VISIBILITY_GEOMETRY", "PASS_ROUTE_PRE_EVENT_SRTM_ASTER", "PASS_ROUTE_BGS_CHARIM", "RECONSTRUCTION_FEASIBLE_NOT_BUILT", "PASS", "PASS", "DEVELOPMENT", "TRUE", "TRUE", "FALSE", "Exact independent 2011 searched/visible polygon with cloud-shadow-fringe exclusions", "SENT_BY_USER", "NO_AFTER_FINAL_PUBLIC_SWEEP", "REQUEST_PENDING", "Ingest reply; QA visibility support; admit only in a later authorized production stage"),
        ("VANUATU_HAROLD_2020", "Pentecost and nearby Harold mapping domain", "Vanuatu", "Pentecost", "VOLCANIC_ISLAND_ARC", "SUBDUCTION_ARC", "CYCLONE_STORM", "TIER_2", "REQUEST_ORIGINAL_1481_POLYGONS", "REQUEST_MAPPED_BOUNDARY", "PASS_COPERNICUS_PRE_EVENT", "PASS_ROUTE_NOT_FULLY_AUDITED", "RECONSTRUCTION_FEASIBLE_NOT_BUILT", "PASS", "REQUEST_PROVENANCE_AND_COMPLETENESS", "LOCKBOX", "TRUE", "METADATA_ONLY_NO_MODEL_CONTACT", "TRUE_CONDITIONAL", "Original 1,481 polygons, mapped boundary, provenance and reuse licence", "SENT_BY_USER", "NO_AFTER_FINAL_PUBLIC_SWEEP", "REQUEST_PENDING", "Ingest and QA reply while preserving model isolation; later human lockbox designation required"),
        ("POPOCATEPETL_PUEBLA_2017", "Popocatépetl Study Area C", "Mexico", "Puebla", "STRATOVOLCANO", "SUBDUCTION_ARC", "EARTHQUAKE", "TIER_2", "REQUEST_STUDY_AREA_C_754_POLYGONS", "REQUEST_365_KM2_BOUNDARY_AND_MASKS", "PASS_COPERNICUS_PRE_EVENT", "PASS_ROUTE_NOT_FULLY_AUDITED", "RECONSTRUCTION_FEASIBLE_NOT_BUILT", "PASS", "REQUEST_MAPPING_PACKAGE", "LOCKBOX", "TRUE", "METADATA_ONLY_NO_MODEL_CONTACT", "TRUE_CONDITIONAL", "Study Area C polygons, exact boundary, cloud/snow masks, provenance, CRS and licence", "SENT_BY_USER", "NO_AFTER_FINAL_PUBLIC_SWEEP", "REQUEST_PENDING", "Ingest and QA reply while preserving model isolation; later human lockbox designation required"),
        ("CHUUK_CHATAAN_2002", "Chuuk high islands / Tonoas", "Federated States of Micronesia", "Chuuk State", "ERODED_OCEANIC_SHIELD_REMNANT", "INTRAPLATE_HOTSPOT_OCEANIC_ISLAND", "EXTREME_RAINFALL", "TIER_2", "AVAILABLE_264_ORIGINAL_POLYGONS; 52_REVISED_TONOAS_FEATURES", "BLOCKED_NO_SYSTEMATIC_OBSERVATION_SUPPORT", "PASS_ROUTE_2000_DEM", "PASS_ROUTE", "NOT_ASSESSED_NEVER_ZERO", "PASS_PRE_2002", "FAIL_USGS_EXPLICIT_NOT_COMPLETE_OR_EXHAUSTIVE", "LOCKBOX", "TRUE", "RAW_QA_ONLY_NO_MODEL_CONTACT", "TRUE_CONDITIONAL", "Target-independent photo/search support plus explicit completeness within that support", "USER_HANDLED_OUTSIDE_CODEX", "NO_NEW_USGS_PRODUCT_RESOLVES_EITHER_BLOCKER", "MULTI_BLOCKER", "Ingest custodian reply; do not use modeled susceptibility as Omega or revised known-site mapping as exhaustive Y"),
        ("REUNION_SALAZIE_1980", "Cirque de Salazie", "France", "Réunion", "ERODED_SHIELD_VOLCANO", "INTRAPLATE_HOTSPOT_OCEANIC_ISLAND", "CYCLONE_STORM", "TIER_2", "PASS_1440_HYACINTHE_POLYGONS_DATASET_LEVEL", "BLOCKED_VALID_OVERLAP_VISIBILITY_MASK", "BLOCKED_1978_DSM_NOT_DISTRIBUTED", "PASS_ROUTE", "PARTIAL_RECONSTRUCTION_NOT_BUILT", "DOCUMENTED_PRE_EVENT_1978", "CONDITIONAL_ON_VISIBLE_TERRAIN", "DEVELOPMENT", "TRUE", "TRUE", "FALSE", "Author-produced 1978 DSM with datum/QA plus valid-overlap, cloud/shadow, failed-photogrammetry/interpreted-area mask", "USER_HANDLED_OUTSIDE_CODEX", "NO_AFTER_REPOSITORY_FORENSICS", "MULTI_BLOCKER", "Ingest reply; QA DSM timing/datum/coverage and observation support; no photogrammetric reconstruction in this stage"),
        ("MOUNT_ELGON_BUKALASI_2018", "Mount Elgon / Bukalasi-Bushika", "Uganda", "Bududa District", "ERODED_MIOCENE_SHIELD", "RIFT_ASSOCIATED_CONTINENTAL_INTRAPLATE", "EXTREME_RAINFALL", "TIER_2", "BLOCKED_NO_COMPLETE_EVENT_INVENTORY", "PARTIAL_PUBLIC_OAM_VALID_DATA_SUPPORT_NOT_SYSTEMATIC_Y_DOMAIN", "PASS_ROUTE_NASADEM_PRE_EVENT", "BLOCKED_AUTHORITATIVE_VECTOR_AND_VOLCANIC_OVERLAP", "PARTIAL_RECONSTRUCTION_NOT_BUILT", "PASS_PRE_2018", "FAIL_RESPONSE_IMPACTS_NOT_EXHAUSTIVE", "LOCKBOX", "TRUE", "RAW_QA_ONLY_NO_MODEL_CONTACT", "TRUE_CONDITIONAL", "Complete event Y; documented systematic interpretation domain; authoritative geology/volcanic overlap; requested World Bank DSM/DTM/AOIs remain absent", "USER_HANDLED_OUTSIDE_CODEX", "PARTIAL_ORTHOMOSAIC_AND_VALID_MASK_RECOVERED_NEEDS_QA", "MULTI_BLOCKER", "Ingest reply and compare with OAM footprint; do not treat response impacts as exhaustive"),
        ("MADEIRA_FUNCHAL_2010", "Funchal / Ribeira Brava basins", "Portugal", "Madeira", "ERODED_SHIELD_ISLAND", "INTRAPLATE_HOTSPOT_OCEANIC_ISLAND", "EXTREME_RAINFALL", "TIER_2_EXPECTED", "HOLD_243_OFFICIAL_POLYGONS_UNCERTAIN_EVENT_PROVENANCE", "BLOCKED_SYSTEMATIC_SEARCH_SUPPORT", "PASS_ROUTE_NASADEM_PRE_EVENT", "PASS_ROUTE", "PARTIAL_RECONSTRUCTION_NOT_BUILT", "PASS_ROUTE", "BLOCKED_EXHAUSTIVENESS_AND_EVENT_ATTRIBUTION", "DEVELOPMENT_IF_RESOLVED", "TRUE", "TRUE", "FALSE", "Explicit 2010 event provenance, exhaustive searched area/visibility support and reuse terms", "NOT_SENT_NOT_IN_STAGE3J_REQUEST_SET", "LOW_NO_NEW_TARGETED_SWEEP_AUTHORIZED", "PARKED", "Keep parked; pursue only after higher-priority replies or a new official metadata release"),
    ]
    return [dict(zip(fields, row)) for row in data]


def request_spec() -> list[dict]:
    common = "SHA-256 every received file before and after copy; require byte identity; never overwrite existing hash-addressed raw objects"
    metadata = "source/sender; licence/reuse permission; acquisition/mapping date; CRS; method; completeness/coverage statement"
    raw = "data/raw/request_responses/{candidate_id}/{received_at}/{sha256_prefix}/"
    rows = [
        ("ST_LUCIA_TOMAS_2010", "SENT_BY_USER", "Exact 2011 Tomas searched/visible polygon; image footprints; cloud/shadow/fringe exclusions", "ZIP;SHP;GPKG;GeoJSON;TIFF;PDF;KML;KMZ", metadata, common, raw, "Omega", "Geometry validity; CRS; independence from Y; exact image epoch; valid/search coverage; cloud-shadow exclusions"),
        ("VANUATU_HAROLD_2020", "SENT_BY_USER", "Original 1,481 polygons; mapped boundary; provenance; licence", "ZIP;SHP;GPKG;GeoJSON;PDF;KML;KMZ", metadata, common, raw, "Y;Omega;target completeness;access/licence", "Feature count/geometry; CRS; PlanetScope date; boundary independence; completeness wording; licence"),
        ("POPOCATEPETL_PUEBLA_2017", "SENT_BY_USER", "Study Area C 754 polygons; 365 km2 boundary; cloud/snow mask; provenance; CRS; licence", "ZIP;SHP;GPKG;GeoJSON;TIFF;PDF;KML;KMZ", metadata, common, raw, "Y;Omega;target completeness;access/licence", "Counts/area; geometry; CRS; imagery dates; masks; boundary independence; licence"),
        ("REUNION_SALAZIE_1980", "USER_HANDLED_OUTSIDE_CODEX", "1978 SfM DSM; vertical datum/QA; orthophoto/valid-overlap/visibility/cloud-shadow/failed-photogrammetry masks", "ZIP;SHP;GPKG;GeoJSON;TIFF;PDF;KML;KMZ", metadata, common, raw, "M;Omega;target completeness", "Raster CRS/resolution/nodata/vertical datum; pre-event timing; uncertainty; mask coverage; independence from Y"),
        ("MOUNT_ELGON_BUKALASI_2018", "USER_HANDLED_OUTSIDE_CODEX", "2018 UAV ortho/DSM/DTM/AOIs/flight plans plus exhaustive event polygons and authoritative geology", "ZIP;SHP;GPKG;GeoJSON;TIFF;PDF;CSV;KML;KMZ", metadata, common, raw, "Y;Omega;M support;G;volcanic domain;target completeness", "Event purity; exhaustive mapping claim; raster validity; AOI independence; CRS/dates; geology authority; Y-G overlap"),
        ("CHUUK_CHATAAN_2002", "USER_HANDLED_OUTSIDE_CODEX", "Aerial-photo index/footprints/flight lines/interpreted coverage/mapping sheets and explicit sector completeness statement", "ZIP;SHP;GPKG;GeoJSON;TIFF;PDF;CSV;KML;KMZ", metadata, common, raw, "Omega;target completeness", "Target-independent support; image dates; systematic interpretation; visibility; explicit completeness; no hull/buffer derivation"),
    ]
    fields = ["candidate_id", "request_status", "expected_artifact", "accepted_file_types", "minimum_metadata_required", "checksum_procedure", "raw_preservation_location", "gates_artifact_could_resolve", "QA_required_before_admission"]
    result = [dict(zip(fields, row)) for row in rows]
    for record in result:
        record["raw_preservation_location"] = record["raw_preservation_location"].replace(
            "{candidate_id}", record["candidate_id"]
        )
    return result


def machine_results() -> list[dict]:
    fields = [
        "candidate_id", "search_scope", "source_chain_inspected", "new_machine_accessible_artifact",
        "artifact_finding", "Y_effect", "Omega_effect", "M_effect", "G_effect",
        "target_completeness_effect", "classification", "remaining_blocker", "stop_rule",
    ]
    data = [
        ("REUNION_SALAZIE_1980", "1978 DSM, masks, supplements and repositories", "JGR article/supporting-information record; Mendeley v3; HAL; BRGM/ReNovRisk; Recherche Data Gouv; Zenodo; IGN archive route", "NO", "The JGR data statement deposits the landslide shapefile and points to raw IGN photographs but does not deposit the 1978 DSM, orthophoto, valid-overlap or visibility mask; supporting information is a PDF, not a raster repository", "UNCHANGED_PASS_EVENT_LAYER", "UNCHANGED_BLOCKED", "UNCHANGED_BLOCKED", "UNCHANGED_PASS_ROUTE", "UNCHANGED_CONDITIONAL", "MULTI_BLOCKER", "1978 DSM/datum/QA and machine-readable valid-overlap/visibility support", "REQUEST_DEPENDENT_STOP_PUBLIC_REPOSITORY_SWEEP"),
        ("MOUNT_ELGON_BUKALASI_2018", "2018 UAV products, AOIs, event Y and repositories", "World Bank/GFDRR reports; OPM chain; 3DroneMapping; Uganda Flying Labs; OpenAerialMap API; HOT references; Copernicus EMSR325; UNOSAT", "YES_OPENAERIALMAP_ORTHOMOSAIC_AND_METADATA", "CC-BY 4.0 post-event orthomosaic acquired 2018-10-15; EPSG:32636; 5.72 cm GSD; bbox 0.664062 km2; internal valid mask 0.378691 km2 (57.0265%); no public DSM/DTM or exhaustive event polygons", "UNCHANGED_BLOCKED", "PARTIAL_INDEPENDENT_RASTER_SUPPORT_NEEDS_MAPPING_PROTOCOL_QA", "UNCHANGED_PASS_ROUTE", "UNCHANGED_BLOCKED", "UNCHANGED_FAIL", "PARTIAL_SUPPORT_FOUND_NOT_RAW_GATE_RESOLVED", "Complete 2018 event Y, systematic interpretation statement, broader AOI/DSM/DTM requested products, authoritative geology", "REQUEST_DEPENDENT_AFTER_PUBLIC_OAM_RECOVERY"),
        ("CHUUK_CHATAAN_2002", "2002 photo/search support plus 2025/2026 USGS updates", "OFR 2004-1348; ScienceBase project tree; P14VTB5D files/FGDC XML; 2003-2022 inventory; 2024 field observations; P14LUYRC model release/readme/XML; 2026 methods article", "YES_REVISED_TONOAS_AND_MODEL_METADATA", "P14VTB5D has no child items and explicitly says not complete or exhaustive. Tonoas re-mapping starts from 52 known locations, reports 51 after merging two, while the distributed layer contains 52 polygon records. P14LUYRC contains modeled source/runout/inundation surfaces, not 2002 observation support", "UNCHANGED_AVAILABLE_BUT_INCOMPLETE", "UNCHANGED_BLOCKED", "UNCHANGED_PASS_ROUTE", "UNCHANGED_PASS_ROUTE", "UNCHANGED_FAIL", "MULTI_BLOCKER", "Systematic target-independent image/search domain and explicit target completeness within it", "REQUEST_DEPENDENT_STOP_MODEL_OUTPUTS_NOT_OBSERVATION_SUPPORT"),
        ("TALAKMAU_2022", "official corrected geology/service plus independent Y validation", "Badan Geologi GeoMap/Geolindo/library/event geology/geoheritage pages; ESDM official-domain search; BMKG/BNPB event sources; 2026 independent LandTrendr paper", "YES_INDEPENDENT_RESEARCH_PAPER_ONLY", "No corrected GeoMap SHP, unambiguous crosswalk, public official WMS/WFS/ArcGIS layer, or independent authoritative inventory was found. The 2026 LandTrendr study reports independent Landsat detections but uses UNOSAT as reference, so it is corroboration rather than independent validation", "UNCHANGED_HOLD", "UNCHANGED_PASS", "UNCHANGED_PASS", "UNCHANGED_HOLD", "UNCHANGED_HOLD", "PARKED_MULTI_BLOCKER", "Independent Y validation and official corrected geology identity", "PARKED_DO_NOT_RETRY_FAILING_RASTER"),
    ]
    return [dict(zip(fields, row)) for row in data]


def acquisitions() -> list[dict]:
    specs = [
        ("MOUNT_ELGON_BUKALASI_2018", "OPENAERIALMAP_METADATA", "OpenAerialMap / HOT", "CC-BY 4.0", "https://oin-hotosm-temp.s3.amazonaws.com/5bd01c0ae5203200052af198/0/5bd01c0ae5203200052af199_meta.json", "MOUNT_ELGON_BUKALASI_2018/SRC_OPENAERIALMAP_BUDUDA_20181015/5bd01c0ae5203200052af199_meta.json", "Exact acquisition time, CRS, resolution, provider and footprint for public post-event UAV imagery"),
        ("MOUNT_ELGON_BUKALASI_2018", "POST_EVENT_UAV_ORTHOMOSAIC", "OpenAerialMap / MapUganda / Uganda Flying Labs / Uganda Red Cross", "CC-BY 4.0", "https://oin-hotosm-temp.s3.amazonaws.com/5bd01c0ae5203200052af198/0/5bd01c0ae5203200052af199.tif", "MOUNT_ELGON_BUKALASI_2018/SRC_OPENAERIALMAP_BUDUDA_20181015/5bd01c0ae5203200052af199.tif", "Independent post-event image/valid-data support candidate; no target mapping inferred"),
        ("CHUUK_CHATAAN_2002", "SCIENCEBASE_ITEM_METADATA", "U.S. Geological Survey", "CC0-1.0", "https://www.sciencebase.gov/catalog/item/686e955fd4be020e5c0a67db?format=json", "CHUUK_CHATAAN_2002/SRC_USGS_P14VTB5D/sciencebase_item_686e955fd4be020e5c0a67db.json", "Authoritative file inventory, purpose, rights and non-exhaustiveness statement"),
        ("CHUUK_CHATAAN_2002", "REVISED_TONOAS_POLYGONS", "U.S. Geological Survey", "CC0-1.0", "https://www.sciencebase.gov/catalog/file/get/686e955fd4be020e5c0a67db?f=__disk__49%2Fc5%2F6b%2F49c56b3f3fe51fa633c0f7ea142dd4e836d2a8a8", "CHUUK_CHATAAN_2002/SRC_USGS_P14VTB5D/Tonoas_Chataan_landslides_revised.zip", "Audit revised Tonoas geometry, attributes, count and whether a systematic subset exists"),
        ("CHUUK_CHATAAN_2002", "SUSCEPTIBILITY_RELEASE_METADATA", "U.S. Geological Survey", "CC0-1.0", "https://www.sciencebase.gov/catalog/item/68717827d4be026e1750f36b?format=json", "CHUUK_CHATAAN_2002/SRC_USGS_P14LUYRC/sciencebase_item_68717827d4be026e1750f36b.json", "Distinguish modeled susceptibility/runout products from observation-support geometry"),
        ("CHUUK_CHATAAN_2002", "SUSCEPTIBILITY_FGDC_METADATA", "U.S. Geological Survey", "CC0-1.0", "https://www.sciencebase.gov/catalog/file/get/68717827d4be026e1750f36b?f=__disk__0a%2F4d%2F4f%2F0a4d4fc2c8a025bf29b7fbaca687ffb37bf92471", "CHUUK_CHATAAN_2002/SRC_USGS_P14LUYRC/Landslide_Susceptibility_and_Runout_Results_Chuuk.xml", "Audit model purpose, inputs, process and completeness language"),
        ("CHUUK_CHATAAN_2002", "SUSCEPTIBILITY_README", "U.S. Geological Survey", "CC0-1.0", "https://www.sciencebase.gov/catalog/file/get/68717827d4be026e1750f36b?f=__disk__db%2F03%2Fac%2Fdb03ac34ac120158835838a6a045c5e4d5edefa9", "CHUUK_CHATAAN_2002/SRC_USGS_P14LUYRC/readme.txt", "Confirm model-layer semantics and reject them as 2002 observation support"),
        ("TALAKMAU_2022", "INDEPENDENT_CHANGE_DETECTION_PAPER", "MDPI Applied Sciences / Universitas Padjadjaran-led authors", "CC-BY 4.0", "https://mdpi-res.com/d_attachment/applsci/applsci-16-03090/article_deploy/applsci-16-03090.pdf", "TALAKMAU_2022/SRC_MDPI_LANDTRENDR_2026/applsci-16-03090.pdf", "Evaluate whether independent Landsat change detection validates the preliminary UNOSAT Y"),
    ]
    fields = ["candidate_id", "artifact_role", "publisher", "licence", "URL", "retrieval_date", "byte_size", "SHA256", "relative_path", "scientific_purpose"]
    rows = []
    for candidate, role, publisher, licence, url, relative, purpose in specs:
        path = RAW / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(dict(zip(fields, [candidate, role, publisher, licence, url, RETRIEVED, path.stat().st_size,
                                      sha256(path), str(path.relative_to(ROOT)), purpose])))
    return rows


def stopping_rule() -> list[dict]:
    fields = ["record_type", "id", "target_development", "target_lockboxes", "target_total", "current_development", "current_lockboxes", "current_total", "status", "candidates", "coverage_requirement", "blocking_reason"]
    data = [
        ("THRESHOLD", "MINIMUM", 6, 2, 8, 6, 0, 6, "NOT_MET", "VANUATU_HAROLD_2020;POPOCATEPETL_PUEBLA_2017", "Two complementary untouched lockboxes after QA and human designation", "Both lockbox packages remain request-pending"),
        ("THRESHOLD", "RECOMMENDED", 8, 3, 11, 6, 0, 6, "NOT_MET", "ST_LUCIA_TOMAS_2010;REUNION_SALAZIE_1980;VANUATU_HAROLD_2020;POPOCATEPETL_PUEBLA_2017;MOUNT_ELGON_BUKALASI_2018", "COUNT + COVERAGE: 8 development across morphology/erosion/climate/tectonic/trigger space plus 3 complementary lockboxes", "Two development additions and all three lockboxes remain unresolved"),
        ("THRESHOLD", "IDEAL", 10, 3, 13, 6, 0, 6, "NOT_MET", "RECOMMENDED_ROUTE;MADEIRA_FUNCHAL_2010;ONE_ADDITIONAL_COMPLETE_HOTSPOT_OR_RIFT_DEVELOPMENT", "Recommended coverage plus two additional development groups without duplication", "Madeira remains parked and no second qualifying development addition is file-backed"),
        ("CATEGORY", "READY", "", "", "", 6, 0, 6, "NONE", "NONE", "All raw gates passed; later production still separately authorized", "No unresolved candidate is ready"),
        ("CATEGORY", "POTENTIALLY_RESOLVABLE_MACHINE", "", "", "", 6, 0, 6, "NONE_AFTER_SWEEP", "NONE", "A public artifact would have to plausibly close every remaining raw gate pending QA", "OpenAerialMap closes neither Mount Elgon Y/completeness nor geology"),
        ("CATEGORY", "REQUEST_PENDING", "", "", "", 6, 0, 6, "ACTIVE", "ST_LUCIA_TOMAS_2010;VANUATU_HAROLD_2020;POPOCATEPETL_PUEBLA_2017", "Reply artifacts must pass candidate-specific QA", "Requests already sent by user"),
        ("CATEGORY", "MULTI_BLOCKER", "", "", "", 6, 0, 6, "ACTIVE", "CHUUK_CHATAAN_2002;REUNION_SALAZIE_1980;MOUNT_ELGON_BUKALASI_2018", "Every independent blocker must pass without gate weakening", "Requests are handled by user outside Codex"),
        ("CATEGORY", "PARKED", "", "", "", 6, 0, 6, "ACTIVE", "TALAKMAU_2022;MADEIRA_FUNCHAL_2010", "Only genuinely new authoritative evidence reopens a parked candidate", "No qualifying new authority in this sweep"),
        ("CATEGORY", "REJECTED", "", "", "", 6, 0, 6, "NONE_NEW", "NONE", "Rejected leads are not counted", "No new-candidate search was performed"),
    ]
    return [dict(zip(fields, row)) for row in data]


def report(state: dict, acquisitions_rows: list[dict], tests: str, v1: str) -> str:
    return f"""STAGE 3J BLOCKER CONSOLIDATION + MACHINE-ACCESSIBLE RESCUE SWEEP: COMPLETE

## A. EXECUTIVE RESULT

No candidate became raw-gate resolved or potentially raw-gate resolved. One useful public artifact was recovered: a CC-BY 4.0, 2018-10-15 OpenAerialMap Bududa orthomosaic with exact footprint and internal valid-data mask. It improves Mount Elgon observation-support evidence but does not supply complete event Y, a systematic interpretation statement, the requested DSM/DTM, or authoritative geology. Production remains 6 landscapes / 835,846 units / 8,793 positives / 827,053 negatives, Tier 2 only.

## B. MASTER BLOCKER STATE

The authoritative eight-candidate ledger is `STAGE3J_MASTER_BLOCKER_LEDGER.tsv`. Clean request-queue items are Saint Lucia, Vanuatu and Popocatépetl. Chuuk, Réunion and Mount Elgon remain request-dependent multi-blocker candidates. Talakmau and Madeira are parked. No candidate is READY.

## C. HUMAN REQUEST STATE

The user has already sent Saint Lucia, Vanuatu and Popocatépetl requests. Réunion, Mount Elgon and Chuuk requests are being handled by the user outside Codex. No request was sent or duplicated here. `STAGE3J_REQUEST_INGESTION_SPEC.tsv` records exact expected artifacts, metadata, hashes, raw destinations, affected gates and QA.

## D. RÉUNION MACHINE RESCUE

The Rault/Dewez/Aunay chain was traced through the JGR article and supporting-information record, Mendeley, HAL, BRGM/ReNovRisk, Recherche Data Gouv, Zenodo and IGN. The article's data statement publishes the shapefile and points to raw IGN photographs; it does not deposit the author-produced 1978 DSM, orthophoto, valid-overlap or visibility mask. M and Omega remain blocked.

## E. MOUNT ELGON MACHINE RESCUE

OpenAerialMap item `5bd13ced007b5600073957e8` was recovered and preserved. The orthomosaic is EPSG:32636, 21,512 × 9,427 pixels, 5.7224 cm GSD, and acquired 15 October 2018. Its 0.664062 km² bounding rectangle contains 0.378691 km² of valid pixels (57.0265%). This is target-independent raster support that could contribute to Omega for a separately documented exhaustive mapping, but no such Y or systematic interpretation statement was found. World Bank/OPM DSM, DTM and later survey AOIs remain non-public.

## F. CHUUK MACHINE RESCUE

The complete public ScienceBase project tree was audited. P14VTB5D has no child items and explicitly says the data are not complete or exhaustive. The process starts from 52 known Tonoas locations and reports 51 after merging two indistinguishable landslides; the distributed shapefile nevertheless contains 52 valid polygon records, an internal count discrepancy that requires custodian clarification. P14LUYRC is a susceptibility/runout model release, not 2002 image/search support. Neither Omega nor target completeness is resolved, and no Tonoas-only systematic domain is documented.

## G. TALAKMAU FINAL MACHINE RESCUE

No official corrected GeoMap vector, unambiguous geometry-to-unit crosswalk, or public authoritative geology service was found. A 2026 Landsat/LandTrendr study independently detects Pasaman changes but uses UNOSAT as its reference, so it cannot independently validate the preliminary inventory. Talakmau remains `PARKED_MULTI_BLOCKER`; the failing raster portal was not retried.

## H. NEW PUBLIC ACQUISITIONS

{len(acquisitions_rows)} files were checksum-registered in `STAGE3J_PUBLIC_ACQUISITIONS.tsv`: OpenAerialMap image/metadata; USGS Tonoas revision and current ScienceBase/model metadata; and the open-access Talakmau change-detection paper. Public licences are CC-BY 4.0 or USGS CC0-1.0. Duplicate Stage 3E Chuuk files are not counted as new acquisitions.

## I. RESPONSE-INGESTION SYSTEM

`src/sfv2/stage3j_ingest.py` plus `config/stage3j_response_routes.json` supports ZIP, SHP, GPKG, GeoJSON, TIFF/GeoTIFF, PDF, CSV, KML and KMZ. It inventories, hashes, metadata-inspects, candidate-associates, compares filenames/types with requested artifacts, copies bytes to hash-addressed read-only raw storage, verifies the copy, and routes it to candidate QA. It never admits gates or writes production outputs.

## J. UPDATED ROUTE TO 8 / 11 / 13

- Minimum 8: retain six development landscapes; recover, QA and later freeze Vanuatu and Popocatépetl as two complementary lockboxes.
- Recommended 11: add Saint Lucia as development #7 and Réunion as development #8; protect Vanuatu, Popocatépetl and a genuinely complete Mount Elgon event as three lockboxes.
- Ideal 13: inherit the recommended route, resolve Madeira as development #9, and add one further complete hotspot/rift development landscape as #10.

Every route remains COUNT + COVERAGE, not count alone. Current confirmed development = 6; current untouched frozen lockboxes = 0.

## K. WHAT CAN PROCEED WITHOUT HUMAN REPLIES

The response-ingestion system and deterministic gate-QA routing are ready. Existing public evidence can be revalidated and official release metadata can be monitored without touching production. No further repeated portal search, candidate admission, terrain-unit generation or model work is justified without new artifacts.

## L. WHAT NOW STRICTLY REQUIRES HUMAN REPLIES

Saint Lucia, Vanuatu and Popocatépetl are purely request-queue dependent. Réunion, Mount Elgon and Chuuk also require human replies, with multiple independent gates preserved. Talakmau requires genuinely new authoritative official geology and independent Y validation, not another download retry.

## M. TESTS / CHECKPOINT / V1 / GIT

Stage 3F `{state['checkpoint_hashes_observed']['stage3f']}`, Stage 3G `{state['checkpoint_hashes_observed']['stage3g']}`, Stage 3H `{state['checkpoint_hashes_observed']['stage3h']}` and Stage 3I `{state['checkpoint_hashes_observed']['stage3i']}` are unchanged. Tests: {tests}. V1: {v1}. `STAGE3J_CHECKPOINT.json` hashes all Stage 3J outputs. Dirty pre-existing worktree preserved; no email, commit or push.

1. Did any candidate become potentially raw-gate resolved? **NO.**
2. Was the Réunion 1978 DSM publicly recovered? **NO.**
3. Was a defensible Réunion visibility mask recovered? **NO.**
4. Were Mount Elgon 2018 UAV products publicly recovered? **PARTIALLY—one post-event orthomosaic plus exact footprint/valid mask; no DSM, DTM, full AOIs or flight plans.**
5. Was a complete Mount Elgon event Y recovered? **NO.**
6. Was a defensible Mount Elgon Omega recovered? **NO.** A useful independent raster-support candidate was recovered, but no source documents exhaustive mapping across it.
7. Did newer USGS Chuuk products resolve Omega? **NO.**
8. Did newer USGS Chuuk products resolve target completeness? **NO.**
9. Could a defensible Tonoas-only subset be constructed without Y-derived Omega? **NO.**
10. Was corrected authoritative Talakmau geology recovered? **NO.**
11. Was Talakmau Y independently validated? **NO.**
12. Is Talakmau still parked? **YES—`PARKED_MULTI_BLOCKER`.**
13. Which candidates are now purely request-dependent? **Saint Lucia, Vanuatu and Popocatépetl are clean request-queue items; Réunion, Mount Elgon and Chuuk are request-dependent with multiple gates.**
14. Which candidates still have multiple independent blockers? **Talakmau, Chuuk, Réunion, Mount Elgon and Madeira.**
15. What is the single fastest path to development landscape #7? **Receive and QA Saint Lucia's exact Tomas observation-support geometry, then productionise only in a later authorized stage.**
16. What is the single fastest path to development landscape #8? **After Saint Lucia, receive and QA Réunion's 1978 DSM plus visibility/valid-overlap support, then productionise only in a later authorized stage.**
17. What are the best three potential final lockboxes? **Vanuatu–Harold, Popocatépetl–Puebla Study Area C, and a genuinely complete Mount Elgon 2018 event package.**
18. What work can Codex continue automatically while replies are pending? **Checksum/inventory automation, non-mutating release monitoring, and candidate-specific QA preparation; reply QA can start immediately when files arrive.**
19. Is the response-ingestion system ready? **YES.**
20. Are production totals unchanged? **YES—6 / 835,846 / 8,793 / 827,053, Tier 2 only.**
21. Are we ready to train ML? **NO.** The recommended 8-development + 3-lockbox COUNT + COVERAGE rule is not satisfied.
"""


def main(tests: str = "PENDING", v1_status: str = "PENDING") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    state = state_verification()
    if not state["all_prior_checkpoint_hashes_unchanged"] or not state["production_state_exact"]:
        raise RuntimeError(f"Frozen input state failed: {state}")
    ledger = blocker_ledger()
    requests = request_spec()
    rescue = machine_results()
    acquired = acquisitions()
    stopping = stopping_rule()
    diagnostics = {
        "stage": "3J", "new_candidates_searched": 0, "new_raw_gates_resolved": 0,
        "potentially_raw_gate_resolved": 0, "production_state_changed": False,
        "terrain_units_generated": False, "production_tables_generated": False,
        "ml_trained": False, "exploratory_ml_run": False, "lolo_run": False,
        "h_generated": False, "gate_weakened": False, "tier_mixing": False,
        "y_derived_omega_accepted": False, "preliminary_inventory_promoted": False,
        "emails_sent": False, "requests_duplicated": False, "commit_performed": False,
        "push_performed": False, "auth_bypassed": False, "access_controls_circumvented": False,
        "terms_accepted": False, "talakmau_failing_raster_retried": False,
        "response_ingestion_ready": True,
        "human_request_state": {
            "SENT_BY_USER": ["ST_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020", "POPOCATEPETL_PUEBLA_2017"],
            "USER_HANDLED_OUTSIDE_CODEX": ["REUNION_SALAZIE_1980", "MOUNT_ELGON_BUKALASI_2018", "CHUUK_CHATAAN_2002"],
        },
    }
    write_json(OUT / "STATE_VERIFICATION.json", state)
    write_tsv(OUT / "STAGE3J_MASTER_BLOCKER_LEDGER.tsv", ledger)
    write_tsv(OUT / "STAGE3J_REQUEST_INGESTION_SPEC.tsv", requests)
    write_tsv(OUT / "STAGE3J_MACHINE_RESCUE_RESULTS.tsv", rescue)
    write_tsv(OUT / "STAGE3J_STOPPING_RULE_STATUS.tsv", stopping)
    write_tsv(OUT / "STAGE3J_PUBLIC_ACQUISITIONS.tsv", acquired)
    write_json(OUT / "STAGE3J_DIAGNOSTICS.json", diagnostics)
    (OUT / "STAGE3J_REPORT.md").write_text(report(state, acquired, tests, v1_status))
    artifact_names = sorted(path.name for path in OUT.iterdir() if path.is_file() and path.name != "STAGE3J_CHECKPOINT.json")
    checkpoint = {
        "stage": "3J", "as_of": AS_OF, "prior_checkpoint_hashes": state["checkpoint_hashes_observed"],
        "production_totals": state["production_state"], "master_blocker_ledger": ledger,
        "request_ingestion_specification": requests, "machine_rescue_results": rescue,
        "new_public_acquisitions": acquired, "stopping_rule_status": stopping,
        "response_ingestion": {
            "script": "src/sfv2/stage3j_ingest.py",
            "config": "config/stage3j_response_routes.json",
            "supported_types": ["ZIP", "SHP", "GPKG", "GeoJSON", "TIFF/GeoTIFF", "PDF", "CSV", "KML", "KMZ"],
            "production_writes": False,
        },
        "tests": tests, "v1_state": v1_status,
        "git_state": {"head_at_audit": "4958b65982e3788e5777c5a4f14e87e67ad95a0a",
                      "worktree": "DIRTY_PRE_EXISTING_WORKTREE_PRESERVED",
                      "commit_performed": False, "push_performed": False},
        "diagnostics": diagnostics,
        "artifact_hashes": {f"outputs/stage3j/{name}": sha256(OUT / name) for name in artifact_names},
    }
    write_json(OUT / "STAGE3J_CHECKPOINT.json", checkpoint)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="PENDING")
    parser.add_argument("--v1-status", default="PENDING")
    args = parser.parse_args()
    main(args.tests, args.v1_status)
