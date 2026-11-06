"""Stage 3E acquisition, rescue, and gate-screening checkpoint.

This module intentionally produces no terrain units, model inputs, H surfaces,
or predictions.  It records source artifacts and fail-closed candidate gates.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .schema import ROOT


AS_OF = "2026-10-06"
RETRIEVED = "2026-10-06T05:30:00Z"
OUT = ROOT / "outputs/stage3e"

PRIOR = {
    "outputs/stage2f2/STAGE2F2_CHECKPOINT.json": "5357dd80f3f5430cb7a34388c658d5d7d84794bd341c97587c652c7b1b2c7b91",
    "outputs/stage3a/STAGE3A_CHECKPOINT.json": "c2d2af66034765171a687bb778255a66b83ccda921071e8b0ef2a65b8e591351",
    "outputs/stage3b/STAGE3B_CHECKPOINT.json": "73ea92c57e4699172916765c8d66de94e2d84c37aa4217ddf2f6dbddae9e0dc4",
    "outputs/stage3c/STAGE3C_CHECKPOINT.json": "c1f2b3a17634cd65c583afd8035f93ca8114caddeab0a732aca6431143d4ebcc",
    "outputs/stage3d/STAGE3D_CHECKPOINT.json": "ad13ed7ca649f328e1f01e51314b1da9b6e7d4f949f259577cfb94b348e00c38",
}


ACQUISITIONS = [
    {
        "source": "Scientific Reports",
        "doi_or_url": "https://doi.org/10.1038/s41598-021-89015-8",
        "licence": "CC BY 4.0 (article)",
        "relative_path": "data/raw/stage3e/POPOCATEPETL_PUEBLA_2017/SRC_SCIENTIFIC_REPORTS/s41598-021-89015-8.pdf",
        "sha256": "2a6a4dc4cdbf5cae4d8aad1b3aedfaec1ab82b4d4e287b38b9e043878991f738",
        "bytes": 4828066,
        "role": "PRIMARY_ARTICLE_AND_DATA_AVAILABILITY_EVIDENCE",
    },
    {
        "source": "USGS ScienceBase data release P14VTB5D",
        "doi_or_url": "https://doi.org/10.5066/P14VTB5D",
        "licence": "CC0 1.0",
        "relative_path": "data/raw/stage3e/CHUUK_CHATAAN_2002/SRC_USGS_P14VTB5D/OFR2004_1348_Chataan_landslides.zip",
        "sha256": "114b3519ed5864dc14d4362743975bb82a4e834b151eda515ae73bf5ce6d11bc",
        "bytes": 110723,
        "role": "EVENT_LANDSLIDE_Y",
    },
    {
        "source": "USGS ScienceBase data release P14VTB5D",
        "doi_or_url": "https://doi.org/10.5066/P14VTB5D",
        "licence": "CC0 1.0",
        "relative_path": "data/raw/stage3e/CHUUK_CHATAAN_2002/SRC_USGS_P14VTB5D/Location_length_height_Chataan_landslides_Chuuk_FSM.xml",
        "sha256": "e67322cf2739aec7e7aed0290a86dd6cd541fcd0d610c44b8b4cca3177d183db",
        "bytes": 18374,
        "role": "Y_METADATA_AND_COMPLETENESS_LIMIT",
    },
    {
        "source": "NHESS 22, 1129-1152 supplementary data",
        "doi_or_url": "https://nhess.copernicus.org/articles/22/1129/2022/nhess-22-1129-2022-supplement.zip",
        "licence": "CC BY 4.0",
        "relative_path": "data/raw/stage3e/SRC_NHESS_22_1129_2022/nhess-22-1129-2022-supplement.zip",
        "sha256": "7400106d0510d956b2f3396fd7774c401eb3bac971f4631c98a7afda54516656",
        "bytes": 71984138,
        "role": "MULTI_INVENTORY_ARCHIVE; DOMINICA_ERIKA_Y_OMEGA_CLOUDS",
    },
    {
        "source": "CDEMA/BGS Saint Lucia landslide report",
        "doi_or_url": "https://www.cdema.org/virtuallibrary/images/SLULandslideReport.pdf",
        "licence": "PUBLIC_REPORT; reuse licence not stated",
        "relative_path": "data/raw/stage3e/ST_LUCIA_TOMAS_2010/SRC_CDEMA_BGS/SLULandslideReport.pdf",
        "sha256": "b9628a065c0a8f4e9275d10ae7b107167d9cdf03d30facd51d10549a8333c907",
        "bytes": 19175900,
        "role": "METHODOLOGY_VISIBILITY_AND_COUNT_EVIDENCE",
    },
    {
        "source": "MAGA Guatemala Geoportal",
        "doi_or_url": "https://geoportal.maga.gob.gt/api/wms?service=WMS&version=1.3.0&request=GetCapabilities",
        "licence": "PUBLIC_OFFICIAL; WMS_FEES_NONE; ACCESS_CONSTRAINTS_NONE",
        "relative_path": "data/raw/stage3e/ATITLAN_STAN_2005/SRC_MAGA_GEOPORTAL/wms_getcapabilities.xml",
        "sha256": "b7f03982760313696586fe09febfaa2ebaa26dfde9a120502d043099e264addb",
        "bytes": 336682,
        "role": "G_SERVICE_CAPABILITIES_AND_ACCESS_TERMS",
    },
    {
        "source": "MAGA Guatemala Geoportal",
        "doi_or_url": "https://geoportal.maga.gob.gt/api/metadata?layer=SUELOS%3Ageologia_nacional",
        "licence": "PUBLIC_OFFICIAL; WMS_FEES_NONE; ACCESS_CONSTRAINTS_NONE",
        "relative_path": "data/raw/stage3e/ATITLAN_STAN_2005/SRC_MAGA_GEOPORTAL/geologia_nacional_metadata.json",
        "sha256": "0e1b8251bcf9499d95c5dcdce075cb9f55a8262656a7fb3204ab2153afc09106",
        "bytes": 873,
        "role": "G_METADATA",
    },
    {
        "source": "MAGA Guatemala Geoportal",
        "doi_or_url": "https://geoportal.maga.gob.gt/api/download?workspace=SUELOS&layer=geologia_nacional&format=application/json",
        "licence": "PUBLIC_OFFICIAL; WMS_FEES_NONE; ACCESS_CONSTRAINTS_NONE",
        "relative_path": "data/raw/stage3e/ATITLAN_STAN_2005/SRC_MAGA_GEOPORTAL/geologia_nacional.geojson",
        "sha256": "7de729e9567cecff1eed4127eed422151b44e9091da041dc9f9b0e2acc31363b",
        "bytes": 6835445,
        "role": "OFFICIAL_G_VECTOR",
    },
    {
        "source": "NASA NTRS",
        "doi_or_url": "https://ntrs.nasa.gov/api/citations/20220008839/downloads/Manuscript_Global_Landslide_Forecasting_System_Accepted.pdf",
        "licence": "PUBLIC_NASA_ACCEPTED_MANUSCRIPT; dataset reuse not established",
        "relative_path": "data/raw/stage3e/VANUATU_HAROLD_2020/SRC_NASA_NTRS/Manuscript_Global_Landslide_Forecasting_System_Accepted.pdf",
        "sha256": "51dc514b5946eb056deccaa350916a9eb735b417133575f006914c41995bdb05",
        "bytes": 3641857,
        "role": "PRIMARY_METHODS_AND_DATASET_DOCUMENTATION",
    },
]


CANDIDATES = [
    {
        "candidate_id": "ATITLAN_STAN_2005", "status": "NEXT_BATCH_READY", "newly_discovered": False,
        "continent": "NORTH_AMERICA", "trigger": "CYCLONE_STORM", "morphology": "VOLCANIC_CALDERA_COMPLEX",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "PASS", "Omega_gate": "PASS",
        "M_gate": "PASS", "G_gate": "PASS", "Y_score": 3, "Omega_score": 3, "M_score": 3, "G_score": 3,
        "machine_accessibility": 12, "v1_history": "V1_HISTORICAL", "lockbox_potential": False,
        "blocker": "NONE; full Stage 3F production QA remains",
    },
    {
        "candidate_id": "DOMINICA_ERIKA_2015", "status": "NEXT_BATCH_READY", "newly_discovered": False,
        "continent": "NORTH_AMERICA", "trigger": "CYCLONE_STORM", "morphology": "VOLCANIC_ISLAND_COMPLEX",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "PASS", "Omega_gate": "PASS",
        "M_gate": "PASS", "G_gate": "PASS_ROUTE", "Y_score": 3, "Omega_score": 3, "M_score": 3, "G_score": 3,
        "machine_accessibility": 12, "v1_history": "V1_HISTORICAL", "lockbox_potential": False,
        "blocker": "NONE at raw-input gate; official geology georeferencing remains Stage 3F",
    },
    {
        "candidate_id": "ST_LUCIA_TOMAS_2010", "status": "SAINT_LUCIA_ONE_BLOCKER", "newly_discovered": False,
        "continent": "NORTH_AMERICA", "trigger": "CYCLONE_STORM", "morphology": "VOLCANIC_ISLAND_COMPLEX",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "PASS", "Omega_gate": "BLOCKED",
        "M_gate": "PASS_ROUTE", "G_gate": "PASS_ROUTE", "Y_score": 3, "Omega_score": 1, "M_score": 3, "G_score": 3,
        "machine_accessibility": 10, "v1_history": "V1_UNSEEN", "lockbox_potential": True,
        "blocker": "Exact 2011 Tomas searched/visible polygon or image footprints plus cloud-shadow-fringe exclusions",
    },
    {
        "candidate_id": "VANUATU_HAROLD_2020", "status": "REQUEST_REQUIRED_PUBLIC_ITEM_REMOVED", "newly_discovered": False,
        "continent": "OCEANIA", "trigger": "CYCLONE_STORM", "morphology": "VOLCANIC_ISLAND_ARC",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "REQUEST", "Omega_gate": "REQUEST",
        "M_gate": "PASS", "G_gate": "PASS_ROUTE", "Y_score": 1, "Omega_score": 1, "M_score": 3, "G_score": 2,
        "machine_accessibility": 7, "v1_history": "V1_UNSEEN", "lockbox_potential": True,
        "blocker": "Original 1,481 polygons, mapped boundary, provenance, and reuse licence",
    },
    {
        "candidate_id": "POPOCATEPETL_PUEBLA_2017", "status": "REQUEST_REQUIRED_PROMISED_FILES_ABSENT", "newly_discovered": False,
        "continent": "NORTH_AMERICA", "trigger": "EARTHQUAKE", "morphology": "YOUNG_STRATOVOLCANO",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "REQUEST", "Omega_gate": "REQUEST",
        "M_gate": "PASS", "G_gate": "PASS_ROUTE", "Y_score": 1, "Omega_score": 1, "M_score": 3, "G_score": 2,
        "machine_accessibility": 7, "v1_history": "V1_UNSEEN", "lockbox_potential": True,
        "blocker": "Study Area C 754 polygons, 365 km2 boundary, cloud/snow mask, provenance, CRS, and reuse licence",
    },
    {
        "candidate_id": "CHUUK_CHATAAN_2002", "status": "OMEGA_REQUEST_REQUIRED", "newly_discovered": True,
        "continent": "OCEANIA", "trigger": "CYCLONE_STORM", "morphology": "OCEANIC_HOTSPOT_VOLCANIC_ISLANDS",
        "tectonic_setting": "HOTSPOT_INTRA_PLATE", "target_tier": "TIER_2", "Y_gate": "PASS", "Omega_gate": "BLOCKED",
        "M_gate": "PASS_ROUTE", "G_gate": "PASS_ROUTE", "Y_score": 3, "Omega_score": 1, "M_score": 3, "G_score": 3,
        "machine_accessibility": 10, "v1_history": "V1_UNSEEN", "lockbox_potential": False,
        "blocker": "USGS metadata says mapping is not complete/exhaustive; no independent complete observation domain",
    },
]


NEXT_BATCH = [
    {
        "rank": 1, "candidate_id": "ATITLAN_STAN_2005", "landscape_group": "ATITLAN_STAN_2005",
        "continent": "NORTH_AMERICA", "trigger": "CYCLONE_STORM", "morphology": "VOLCANIC_CALDERA_COMPLEX",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "PASS",
        "Omega_gate": "PASS", "M_gate": "PASS_NASADEM_2000", "G_gate": "PASS_MAGA_OFFICIAL_VECTOR",
        "temporal_gate": "PASS", "volcanic_domain_gate": "PASS_RAPID_SPATIAL_AUDIT", "machine_accessibility": 12,
        "V1_history": "V1_HISTORICAL", "lockbox_potential": "NO", "raw_inputs_complete": "YES",
        "recommended_action": "STAGE3F_FULL_PRODUCTION_GATES_FIRST",
    },
    {
        "rank": 2, "candidate_id": "DOMINICA_ERIKA_2015", "landscape_group": "DOMINICA",
        "continent": "NORTH_AMERICA", "trigger": "CYCLONE_STORM", "morphology": "VOLCANIC_ISLAND_COMPLEX",
        "tectonic_setting": "SUBDUCTION_ARC", "target_tier": "TIER_2", "Y_gate": "PASS_EVENT_SPECIFIC_1548",
        "Omega_gate": "PASS_DATE_AWARE_FOOTPRINT_MINUS_CLOUD", "M_gate": "PASS_COP30_PRE_EVENT",
        "G_gate": "PASS_ROUTE_OFFICIAL_DOMINICA_MAP", "temporal_gate": "PASS",
        "volcanic_domain_gate": "PASS_ROUTE_REQUIRES_STAGE3F_SPATIAL_AUDIT", "machine_accessibility": 12,
        "V1_history": "V1_HISTORICAL", "lockbox_potential": "NO", "raw_inputs_complete": "YES",
        "recommended_action": "STAGE3F_FULL_PRODUCTION_GATES_SECOND; KEEP_MARIA_BLOCKED",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def machine_score(candidate: dict) -> int:
    return sum(int(candidate[key]) for key in ("Y_score", "Omega_score", "M_score", "G_score"))


def verify_state() -> dict:
    got = {path: sha256(ROOT / path) if (ROOT / path).exists() else "MISSING" for path in PRIOR}
    stage3d = json.loads((ROOT / "outputs/stage3d/STAGE3D_CHECKPOINT.json").read_text())
    raw_errors = []
    for item in ACQUISITIONS:
        path = ROOT / item["relative_path"]
        digest = sha256(path) if path.exists() else "MISSING"
        size = path.stat().st_size if path.exists() else -1
        if digest != item["sha256"] or size != item["bytes"]:
            raw_errors.append({"relative_path": item["relative_path"], "sha256": digest, "bytes": size})
    return {
        "prior_checkpoint_hashes": got,
        "prior_checkpoint_mismatches": [p for p, expected in PRIOR.items() if got[p] != expected],
        "production_landscapes": stage3d["production_landscape_count"],
        "terrain_units": stage3d["production_terrain_units_total"],
        "positive_units": stage3d["production_positive_units_total"],
        "positive_units_by_landscape": {"KUMAMOTO_ASO_2016": 656, "NOTO_PENINSULA_2024": 2148,
                                        "LOMBOK_RINJANI_2018": 3102, "COTABATO_DAVAO_2019": 298},
        "dominica_maria_verdict": stage3d["production_gate_outcomes"]["DOMINICA_MARIA_2017"],
        "new_raw_files": len(ACQUISITIONS), "new_raw_hash_or_size_errors": raw_errors,
        "all_inputs_verified": not raw_errors and all(got[p] == PRIOR[p] for p in PRIOR),
    }

