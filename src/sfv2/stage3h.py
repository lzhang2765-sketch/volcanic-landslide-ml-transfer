"""Stage 3H targeted blocker rescue and acquisition audit.

This module is intentionally acquisition/metadata-only.  It cannot create
terrain units, model features, labels, predictions, or production H.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .schema import ROOT


AS_OF = "2026-10-06"
RETRIEVED = "2026-10-06T17:49:00+10:00"
OUT = ROOT / "outputs/stage3h"
RAW = ROOT / "data/raw/stage3h"
STAGE3G_HASH = "a4f8035123658c063f1572bdfd3276906053ba313735ae3371c442e45cbe4664"
STAGE3F_HASH = "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a"

PRODUCTION = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
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


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_state() -> dict:
    p3g = ROOT / "outputs/stage3g/STAGE3G_CHECKPOINT.json"
    p3f = ROOT / "outputs/stage3f/STAGE3F_CHECKPOINT.json"
    g = json.loads(p3g.read_text())
    expected_states = {
        "ST_LUCIA_TOMAS_2010": "PROMISING_ONE_BLOCKER",
        "VANUATU_HAROLD_2020": "PROMISING_REQUEST_ONLY",
        "POPOCATEPETL_PUEBLA_2017": "PROMISING_REQUEST_ONLY",
        "CHUUK_CHATAAN_2002": "PROMISING_ONE_BLOCKER",
        "MOUNT_ELGON_BAKALASI_2018": "HOLD",
        "REUNION_SALAZIE_1978_1984": "HOLD",
        "MADEIRA_FUNCHAL_2010": "HOLD",
    }
    observed = {k: g["candidate_states"].get(k) for k in expected_states}
    return {
        "stage3g_checkpoint_sha256": sha256(p3g),
        "stage3g_checkpoint_expected_sha256": STAGE3G_HASH,
        "stage3g_checkpoint_unchanged": sha256(p3g) == STAGE3G_HASH,
        "stage3f_checkpoint_sha256": sha256(p3f),
        "stage3f_checkpoint_expected_sha256": STAGE3F_HASH,
        "stage3f_checkpoint_unchanged": sha256(p3f) == STAGE3F_HASH,
        "stage3g_candidate_states_expected": expected_states,
        "stage3g_candidate_states_observed": observed,
        "stage3g_candidate_states_unchanged": observed == expected_states,
        "production_landscapes": g["current_production_state"]["landscapes"],
        "production_count": g["current_production_state"]["count"],
        "terrain_units": g["current_production_state"]["terrain_units"],
        "positive_units": g["current_production_state"]["positive_units"],
        "target_tier": g["current_production_state"]["target_tier"],
    }


def acquisitions() -> list[dict]:
    specs = [
        dict(candidate_id="REUNION_SALAZIE_1978_1984", artifact_role="EVENT_METHOD_AND_TERRAIN_PAPER",
             source_title="Landslide Processes Involved in Volcano Dismantling From Past to Present",
             source_institution="BRGM / Universite de La Reunion", authors="Claire Rault et al.",
             publication="Journal of Geophysical Research: Earth Surface (2022)", DOI="10.1029/2021JF006257",
             URL="https://univ-reunion.hal.science/hal-03674815/document", licence="CC_BY_NC_ND_4_0",
             reuse_status="PUBLIC_FULL_TEXT; CITE; NO_DERIVATIVES",
             filename="REUNION_SALAZIE_1978_1984/SRC_HAL_03674815/Rault_et_al_2022_JGR.pdf",
             original_format="PDF", CRS_if_known="NOT_APPLICABLE",
             notes="Primary methods source; supports dataset-level Hyacinthe attribution and identifies 1978/1984 imagery limitations."),
        dict(candidate_id="REUNION_SALAZIE_1978_1984", artifact_role="PRE_EVENT_DSM_VALIDATION_PAPER",
             source_title="Structure-from-Motion Processing of Aerial Archive Photographs",
             source_institution="BRGM / ISPRS", authors="Claire Rault; Thomas J. B. Dewez; Bertrand Aunay",
             publication="ISPRS Annals V-2-2020, 773-780", DOI="10.5194/isprs-annals-V-2-2020-773-2020",
             URL="https://isprs-annals.copernicus.org/articles/V-2-2020/773/2020/isprs-annals-V-2-2020-773-2020.pdf",
             licence="CC_BY_4_0", reuse_status="OPEN_ACCESS_WITH_ATTRIBUTION",
             filename="REUNION_SALAZIE_1978_1984/SRC_ISPRS_RAULT_2020/isprs-annals-V-2-2020-773-2020.pdf",
             original_format="PDF", CRS_if_known="RGR92_UTM_40S_FOR_STUDY_PRODUCTS",
             notes="Documents a validated 1978 SfM DSM; the DSM raster itself is not distributed with the article."),
        dict(candidate_id="REUNION_SALAZIE_1978_1984", artifact_role="HISTORICAL_PHOTOGRAMMETRY_METHOD_PAPER",
             source_title="How Can the Morphometric Characteristics and Failure Conditions of a Historic Gully Be Reconstructed?",
             source_institution="BRGM / MDPI", authors="Claire Rault et al.", publication="Earth 3(1), 324-344 (2022)",
             DOI="10.3390/earth3010020", URL="https://mdpi-res.com/d_attachment/earth/earth-03-00020/article_deploy/earth-03-00020.pdf",
             licence="CC_BY_4_0", reuse_status="OPEN_ACCESS_WITH_ATTRIBUTION",
             filename="REUNION_SALAZIE_1978_1984/SRC_EARTH_RAULT_2022/earth-03-00020.pdf",
             original_format="PDF", CRS_if_known="RGR92_UTM_40S_FOR_STUDY_PRODUCTS",
             notes="Corroborates 60 1978 and 79 1984 archive photos and photogrammetric DSM/orthophoto workflow."),
        dict(candidate_id="MOUNT_ELGON_BAKALASI_2018", artifact_role="REGIONAL_INVENTORY_METHOD_PAPER",
             source_title="Landslide susceptibility and mobilization rates in the Mount Elgon region, Uganda",
             source_institution="ETH public repository / KU Leuven-VUB research team", authors="Jente Broeckx et al.",
             publication="Landslides 16, 571-584 (2019)", DOI="10.1007/s10346-018-1085-y",
             URL="https://iacweb.ethz.ch/doc/publications/Broeckx_etal_2019_LS.pdf", licence="NOT_STATED_IN_REPOSITORY_FILE",
             reuse_status="PUBLIC_READ_AND_CITE; DATA_REUSE_REQUIRES_PERMISSION",
             filename="MOUNT_ELGON_BAKALASI_2018/SRC_ETH_BROECKX_2019/Broeckx_etal_2019_LS.pdf",
             original_format="PDF", CRS_if_known="NOT_APPLICABLE",
             notes="653 calibration landslides plus an independent 400+ validation set; long-term field inventory, not a specific event inventory."),
        dict(candidate_id="MOUNT_ELGON_BAKALASI_2018", artifact_role="POST_EVENT_UAV_ACQUISITION_REPORT",
             source_title="Drones and the Response to the 2018 Uganda Landslide", source_institution="World Bank / Uganda OPM",
             authors="World Bank", publication="World Bank case study", DOI="",
             URL="https://documents1.worldbank.org/curated/en/099121923145119796/pdf/P1719901c0d7ef098193ad17183bbb30219.pdf",
             licence="LICENCE_NOT_STATED_ON_DOCUMENT_LANDING", reuse_status="PUBLIC_READ_AND_CITE",
             filename="MOUNT_ELGON_BAKALASI_2018/SRC_WORLD_BANK_DRONES_2018/P171990_drones_response_2018_uganda.pdf",
             original_format="PDF", CRS_if_known="NOT_STATED",
             notes="Confirms OPM/World Bank commissioned Bukalasi and Bushika orthophotos, DSM, DTM and video on 27 Nov and 7 Dec 2018."),
        dict(candidate_id="MOUNT_ELGON_BAKALASI_2018", artifact_role="REGIONAL_INVENTORY_PROJECT_REPORT",
             source_title="AfReSlide Final Report", source_institution="BELSPO / AfReSlide consortium",
             authors="AfReSlide consortium", publication="BRAIN-be final report", DOI="",
             URL="https://www.belspo.be/belspo/brain-be/projects/FinalReports/AfReSlide_FinRep_AD.pdf",
             licence="LICENCE_NOT_STATED", reuse_status="PUBLIC_READ_AND_CITE; VECTOR_REUSE_REQUIRES_PERMISSION",
             filename="MOUNT_ELGON_BAKALASI_2018/SRC_BELSPO_AFRESLIDE/AfReSlide_Final_Report.pdf",
             original_format="PDF", CRS_if_known="NOT_APPLICABLE",
             notes="Documents a 233-polygon Mount Elgon field inventory completed in October 2016; not event-specific and no vector download found."),
        dict(candidate_id="MOUNT_ELGON_BAKALASI_2018", artifact_role="EVENT_IMPACT_MAP",
             source_title="Landslides/Mudflow Assessment in Mbale District, Eastern Region, Uganda",
             source_institution="UNITAR-UNOSAT", authors="UNITAR-UNOSAT", publication="LS20181012UGA, 16 Oct 2018", DOI="",
             URL="https://unosat.org/static/unosat_filesystem/2687/UNOSAT_A3_Landslides_LS20181012UGA_Bukalasi.pdf",
             licence="CC_BY_NC_3_0", reuse_status="NONCOMMERCIAL_REUSE_WITH_ATTRIBUTION",
             filename="MOUNT_ELGON_BAKALASI_2018/SRC_UNOSAT_LS20181012UGA/UNOSAT_A3_Landslides_LS20181012UGA_Bukalasi.pdf",
             original_format="PDF", CRS_if_known="WGS_1984_UTM_ZONE_36N",
             notes="Preliminary 0.5 m Pleiades impact map; one landslide/mudflow extent, not exhaustive landscape Y."),
    ]
    rows = []
    for spec in specs:
        relative = spec.pop("filename")
        path = RAW / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        row = dict(spec)
        row.update(retrieval_timestamp=RETRIEVED, filename=path.name, byte_size=path.stat().st_size,
                   SHA256=sha256(path), relative_path=str(path.relative_to(ROOT)))
        rows.append(row)
    fields = ["candidate_id", "artifact_role", "source_title", "source_institution", "authors", "publication",
              "DOI", "URL", "licence", "reuse_status", "retrieval_timestamp", "filename", "byte_size", "SHA256",
              "original_format", "CRS_if_known", "notes", "relative_path"]
    return [{k: r[k] for k in fields} for r in rows]


def chuuk_audit() -> dict:
    return {
        "candidate_id": "CHUUK_CHATAAN_2002", "omega_recovered": False,
        "target_completeness_resolved": False, "final_status": "BOTH_BLOCKED",
        "source_chain": [
            "USGS OFR 2004-1348: broad aerial overview followed by detailed work at selected high-impact sites",
            "USGS P14VTB5D: original mapping was from oblique photographs of landslide-affected areas",
            "USGS P14VTB5D: revised Tonoas mapping starts from the original 52 known landslides and refines 51",
        ],
        "decisive_metadata_language": "These data should not be considered complete or exhaustive.",
        "omega_finding": "No flight lines, interpreted-photo footprints, visibility mask, or systematic mapped-area polygon was found.",
        "completeness_finding": "Complete for the information presented is an archive-completeness statement, not complete landslide ascertainment.",
        "subset_audit": {"defensible_subset_found": False,
                         "reason": "Tonoas remapping refines previously known locations; it does not document a systematic island-wide search."},
        "rejected_omega": ["whole-island outlines", "administrative boundaries", "Y convex hull", "Y buffer", "affected-area photo footprint inference"],
        "remaining_artifacts": ["systematically interpreted image/flight support", "source confirmation of exhaustive target mapping within that support"],
    }


def reunion_audits() -> tuple[dict, dict, dict]:
    event = {
        "candidate_id": "REUNION_SALAZIE_1978_1984", "hyacinthe_specific_y_defensible": True,
        "features": 1440, "crs": "EPSG:2975 RGR92 / UTM zone 40S",
        "basis": [
            "The distributed layer is named Landslides_Salazie_TC_hyacinthe.",
            "Embedded metadata purpose says it maps landslides triggered by Cyclone Hyacinthe (1980).",
            "The primary JGR paper treats >1,400 changes mapped between 1978 and 1984 as triggered by TC Hyacinthe.",
        ],
        "attribute_fields": ["Id", "id_pinchin", "Surface", "Slope_coun", "Slope_sum", "Slope_mean", "ID_bis", "Type_Ls"],
        "per_feature_date_or_trigger_field": False,
        "interpretation": "Event membership is a dataset-level source classification, not a spatial or per-feature inference.",
        "event_attribution_status": "RESOLVED",
    }
    observation = {
        "candidate_id": "REUNION_SALAZIE_1978_1984", "cirque_boundary_independent": True,
        "systematic_mapping_documented": True, "full_cirque_omega_defensible": False,
        "imagery": {"1978_frames": 60, "1984_frames": 79, "method": "historical aerial-photo SfM-MVS orthophotos/DSMs plus visual change interpretation"},
        "fatal_gap": "The paper excludes areas where change could not be seen because of shade/cloud/photogrammetric gaps, but no machine-readable valid-overlap/visibility mask is distributed.",
        "omega_status": "BLOCKED_VISIBILITY_MASK",
        "prohibited_inference": "The cirque boundary cannot be used alone as Omega_obs.",
    }
    terrain = {
        "candidate_id": "REUNION_SALAZIE_1978_1984", "pre_event_surface_exists_in_research_workflow": True,
        "distributed_machine_readable_surface_found": False, "event_date": 1980, "surface_date": 1978,
        "documented_surface": "1978 SfM DSM from IGN aerial photographs",
        "validation": "0.66 m/pixel proof-of-concept DSM had median deviation -1.39 m ± 3.34 m MAD against 2015 lidar DTM; JGR analysis used harmonized 5 m products.",
        "public_inputs_route": "IGN Remonter le Temps archive photographs are publicly downloadable; reconstruction is nontrivial and requires GCP, camera, datum and uncertainty QA.",
        "m_status": "BLOCKED_DISTRIBUTED_1978_SURFACE",
        "minimal_artifact": "Author-produced 1978 DSM with CRS/vertical datum, valid-data footprint and uncertainty/provenance metadata.",
    }
    return event, observation, terrain


def mount_elgon_audits() -> tuple[dict, dict]:
    inventory = {
        "candidate_id": "MOUNT_ELGON_BAKALASI_2018", "complete_event_inventory_found": False,
        "event_products": [
            {"source": "Copernicus EMSR325", "features": 9, "reason_rejected": "impact-response mapping: 3 mudflows and 6 landslides, not landscape-exhaustive"},
            {"source": "UNITAR-UNOSAT LS20181012UGA", "features": 1, "reason_rejected": "preliminary combined landslide/mudflow impact extent"},
        ],
        "regional_near_misses": [
            {"source": "Broeckx et al. 2019", "features": "653 calibration + >400 validation", "reason_rejected": "long-term mixed-age field inventory; local-reported locations; no specific event attribution"},
            {"source": "AfReSlide", "features": 233, "reason_rejected": "field inventory completed in 2016, predating the 2018 event; vectors not publicly distributed"},
            {"source": "Nakileza and Nedala 2020", "features": 284, "reason_rejected": "recent and historical records mixed; polygons and/or points; no event-isolatable public vectors"},
            {"source": "Knapen/Claessens regional work", "features": 81, "reason_rejected": "older multi-temporal susceptibility calibration inventory"},
        ],
        "uav_lead": "World Bank/Uganda OPM commissioned post-event orthophotos, DSM, DTM and video over Bukalasi and Bushika; no systematic event landslide layer was found publicly.",
        "inventory_status": "BLOCKED_COMPLETE_EVENT_Y",
    }
    omega_geology = {
        "candidate_id": "MOUNT_ELGON_BAKALASI_2018", "omega_defensible": False,
        "response_aois_independent": True,
        "reason": "EMSR/UNOSAT AOIs support their limited impact products only; no exhaustive event Y is tied to a systematically searched footprint.",
        "authoritative_geology_route": "Uganda DGSM 1:100,000/2014 geology is cited by local studies; Uganda NSDI lists a public Minerals shapefile interface, but an authenticated-free downloadable Mount Elgon geology artifact could not be verified in this audit.",
        "volcanic_setting": "Mount Elgon is an extinct Miocene shield volcano, but the regional study area also contains Precambrian basement and carbonatite/fenitised units.",
        "volcanic_overlap_quantified": False,
        "volcanic_domain_status": "CONDITIONAL_MIXED_DOMAIN_REQUIRES_Y_OMEGA_AND_G_OVERLAY",
        "omega_status": "BLOCKED_WITH_Y",
    }
    return inventory, omega_geology


def replacement_search() -> list[dict]:
    data = [
        (1, "LIMBE_MOUNT_CAMEROON_2001", "AFRICA; HOTSPOT/CVL", "57 scars + 6 multi-slide zones in published map", "No reusable vectors or explicit machine Omega found", "REQUEST_DEPENDENT", "NOT_BETTER"),
        (2, "BAMBOUTOS_CAMEROON_MULTITEMPORAL", "AFRICA; CVL", "AfReSlide reports 666 field polygons", "Not event-specific; vectors and Omega not public", "REQUEST_DEPENDENT", "NOT_BETTER"),
        (3, "SAO_MIGUEL_AZORES_2005_2016", "HOTSPOT/RIFT", "442 rainfall slides in published multi-period inventory", "Multiple periods; no event-specific public Y+Omega route", "SPECULATIVE", "NOT_BETTER"),
        (4, "SAO_MIGUEL_AZORES_ARCHIVE", "HOTSPOT/RIFT", "9,890 scar archive with dated-event research", "Archive is multi-temporal; event polygon subset/coverage not public", "SPECULATIVE", "NOT_BETTER"),
        (5, "KAUAI_APRIL_2018", "OCEANIA/PACIFIC; HOTSPOT", "Well-documented rainfall disaster", "No exhaustive event polygon inventory; NASA catalog is event/point reporting", "SPECULATIVE", "NOT_BETTER"),
        (6, "MADEIRA_FUNCHAL_2010", "HOTSPOT/SHIELD", "243 official polygons already acquired in Stage 3G", "Event provenance and systematic search support remain blocked", "CARRIED_FORWARD", "NOT_BETTER"),
        (7, "RWENZORI_AFRESLIDE", "AFRICA; RIFT_REGION", "450 field polygons + 770 park polygons reported", "Not volcanic target and not single-event inventory", "REJECT", "NOT_BETTER"),
        (8, "COOLR_HAWAII_AND_OCEANIA", "OCEANIA/PACIFIC; HOTSPOT", "Machine-readable event reports", "Points/report footprints, no systematic polygon Y+Omega", "REJECT", "NOT_BETTER"),
        (9, "IBURI_LANDSLIDE4SENSE_2018", "OCEANIA", "Open segmentation labels", "Benchmark/model label provenance and volcanic-domain/event Omega do not meet frozen target rules", "REJECT", "NOT_BETTER"),
        (10, "MOUNT_CAMEROON_REGIONAL_INVENTORIES", "AFRICA; HOTSPOT/CVL", "Systematic field inventory publications exist", "Mixed events/ages and no accessible event-specific vector package", "REQUEST_DEPENDENT", "NOT_BETTER"),
    ]
    fields = ["lead_number", "candidate_id", "gap_closed", "positive_evidence", "fatal_gap", "access_class", "verdict"]
    return [dict(zip(fields, row)) for row in data]


def rescue_rows() -> list[dict]:
    fields = ["candidate_id", "scientific_value", "Y_status", "Y_completeness", "Omega_status", "Omega_independence",
              "M_status", "M_temporal_validity", "G_status", "volcanic_domain_status", "target_semantics",
              "machine_accessibility", "V1_unseen", "development_or_lockbox_preference", "remaining_blockers", "rescue_verdict"]
    data = [
        ("CHUUK_CHATAAN_2002", "VERY_HIGH_OCEANIA_HOTSPOT", "AVAILABLE_250_PLUS_POLYGONS", "FAIL_EXPLICIT_NOT_COMPLETE_OR_EXHAUSTIVE", "BLOCKED_NO_SYSTEMATIC_SEARCH_GEOMETRY", "FAIL", "PASS_ROUTE_2000_DEM", "PASS_PRE_2002", "PASS_ROUTE", "PASS", "PASS_EVENT", "PARTIAL", "TRUE", "LOCKBOX", "systematic mapped support; defensible target completeness within it", "MULTIPLE_BLOCKERS_REMAIN"),
        ("REUNION_SALAZIE_1978_1984", "VERY_HIGH_HOTSPOT_SHIELD", "PASS_1440_HYACINTHE_POLYGONS", "CONDITIONAL_ON_VISIBLE_TERRAIN", "BLOCKED_MISSING_VALID_OVERLAP_MASK", "CIRQUE_INDEPENDENT_BUT_INSUFFICIENT", "BLOCKED_1978_DSM_NOT_DISTRIBUTED", "DOCUMENTED_PRE_EVENT", "PASS_ROUTE", "PASS", "PASS_HYACINTHE_DATASET_LEVEL", "PARTIAL", "TRUE", "DEVELOPMENT", "1978 DSM with datum/QA; interpreted-area and visibility mask", "MULTIPLE_BLOCKERS_REMAIN"),
        ("MOUNT_ELGON_BAKALASI_2018", "VERY_HIGH_AFRICA_RIFT", "FAIL_ONLY_IMPACT_PRODUCTS", "FAIL_NOT_LANDSCAPE_EXHAUSTIVE", "BLOCKED_NO_EVENT_SEARCH_SUPPORT", "FAIL_FOR_REQUIRED_Y", "PASS_ROUTE_NASADEM", "PASS_PRE_2018", "PASS_REFERENCE_ROUTE_NOT_ACQUIRED", "CONDITIONAL_MIXED_DOMAIN", "FAIL_RESPONSE_IMPACT_ONLY", "PARTIAL", "TRUE", "LOCKBOX", "complete event polygons; searched-area/visibility support; authoritative geology vector and overlap", "MULTIPLE_BLOCKERS_REMAIN"),
        ("ST_LUCIA_TOMAS_2010", "HIGH_CARIBBEAN", "PASS_CARRIED_FORWARD", "PASS_CARRIED_FORWARD", "BLOCKED_EXACT_SEARCH_VISIBILITY_POLYGON", "PENDING_REQUEST", "PASS_ROUTE", "PASS", "PASS_ROUTE", "PASS", "PASS", "PARTIAL", "TRUE", "DEVELOPMENT", "exact independent searched/visible polygon", "ONE_BLOCKER_REMAINS"),
        ("VANUATU_HAROLD_2020", "HIGH_OCEANIA", "REQUEST", "REQUEST", "REQUEST", "REQUEST", "PASS", "PASS", "PASS_ROUTE", "PASS", "PASS", "REQUEST", "TRUE", "LOCKBOX", "original 1481 polygons; mapped boundary; provenance; licence", "REQUEST_REQUIRED"),
        ("POPOCATEPETL_PUEBLA_2017", "HIGH_YOUNG_STRATOVOLCANO", "REQUEST", "REQUEST", "REQUEST", "REQUEST", "PASS", "PASS", "PASS_ROUTE", "PASS", "PASS", "REQUEST", "TRUE", "LOCKBOX", "Study Area C polygons; search and cloud/snow masks; CRS; licence", "REQUEST_REQUIRED"),
    ]
    return [dict(zip(fields, row)) for row in data]


def requests_markdown() -> str:
    return """# STAGE 3H HIGH-VALUE ARTIFACT REQUESTS

No message was sent. These are drafts for the user.

## CHUUK_CHATAAN_2002 — USGS (Dianne L. Brien / Mark E. Reid; alternative: USGS Landslide Hazards Program)

- Why: they authored the current release and/or original mapping.
- Exact artifact: flight lines or photograph footprints explicitly and systematically interpreted for Chata'an landslides, visibility exclusions, mapped-area polygon, and a statement of completeness/detection threshold within that support.
- Scientific need: independent Omega_obs and defensible absence labels; the current release explicitly says it is not complete/exhaustive.
- Reference: Harp et al. (2004), OFR 2004-1348; Brien, Reid & Harp, USGS DOI 10.5066/P14VTB5D.

> Dear Ms Brien and Dr Reid, I am conducting non-commercial student research on event landslides in volcanic terrain using the Chata'an 2002 USGS inventory (OFR 2004-1348; DOI 10.5066/P14VTB5D). Could you share any flight lines, aerial-photo footprints, systematically interpreted-area polygon, visibility exclusions, or mapping notes that define where landslides were searched for? Could you also clarify whether any documented sector (including Tonoas) was exhaustively mapped above a stated detection threshold? I need these to define an observation domain independently of the landslide locations. I would cite the USGS sources and any supplied artifact exactly.

## REUNION_SALAZIE_1978_1984 — Claire Rault / Yannick Thiery, BRGM (alternative: Bertrand Aunay / BRGM data services)

- Why: the authors produced the historical SfM products and the 1,440-feature inventory.
- Exact artifact: author-produced 1978 DSM, 1978 and 1984 orthophoto valid-data footprints, cloud/shadow/photogrammetric-failure masks, systematic interpreted-area polygon, CRS and vertical datum, uncertainty/QA, provenance and reuse terms.
- Scientific need: a pre-Hyacinthe terrain surface and target-independent Omega_obs.
- Reference: DOI 10.1029/2021JF006257; DOI 10.5194/isprs-annals-V-2-2020-773-2020; Mendeley DOI 10.17632/gvkjpp87xj.3.

> Dear Dr Rault and Dr Thiery, I am using your CC BY Salazie landslide inventory for non-commercial student research. The embedded metadata and JGR paper support a Cyclone Hyacinthe target, but I cannot define defensible non-landslide observations without the valid interpreted support. Could you share the 1978 DSM used in the study, the 1978/1984 orthophoto or DSM valid-data footprints, cloud/shadow/photogrammetric-gap masks, and any polygon defining the area systematically interpreted for Hyacinthe landslides? I would also appreciate the CRS, vertical datum, uncertainty/QA, provenance, and permitted research reuse terms. I will cite all three sources above.

## MOUNT_ELGON_BAKALASI_2018 — Uganda Office of the Prime Minister / World Bank (alternative: 3DroneMapping)

- Why: the documented response commissioned Bukalasi/Bushika orthophotos, DSM, DTM and video after the 2018 event.
- Exact artifact: UAV flight/AOI boundaries, post-event orthomosaics, pre/post imagery if held, and any human-mapped event landslide polygons with completeness/visibility documentation.
- Scientific need: exhaustive event Y and independent observation support; impact-response extents alone are insufficient.
- Reference: World Bank, *Drones and the Response to the 2018 Uganda Landslide*.

> Dear Uganda OPM / World Bank data team, I am conducting non-commercial student research on the 11 October 2018 Bukalasi landslide event. Your case study documents UAV orthophotos, DSMs, DTMs and video acquired over Bukalasi and Bushika on 27 November and 7 December 2018. Could you share the flight/AOI footprints and derived products, plus any systematically mapped event-landslide polygons and documentation of visibility/completeness? I need to distinguish a complete searched terrain domain from impact-only mapping and will cite the World Bank/OPM project.

## MOUNT_ELGON REGIONAL INVENTORY — Jente Broeckx / Liesbet Jacobs (alternative: AfReSlide / KU Leuven-VUB)

- Why: the team compiled 653 calibration landslides, a 400+ validation set, and a separate AfReSlide Mount Elgon field inventory.
- Exact artifact: polygons, dates/triggers/source confidence, mapped/search boundary, mapping protocol and licence; specifically ask whether a single-event subset with systematic coverage exists.
- Scientific need: determine whether a defensible event-specific Tier-2 target can be isolated; the published regional inventory is mixed-age.
- Reference: DOI 10.1007/s10346-018-1085-y; AfReSlide final report.

> Dear Dr Broeckx and Dr Jacobs, I am studying rainfall-triggered landslides in volcanic terrain. Could you share the Mount Elgon inventory geometries with event date/trigger/source fields, mapped-area boundaries, mapping protocol and research reuse terms? In particular, does the inventory contain any single rainfall-event polygon subset mapped systematically across an independently defined search area? The published mixed-age regional inventory cannot by itself support event absence labels. I will cite DOI 10.1007/s10346-018-1085-y and the AfReSlide project.
"""


def route_update() -> dict:
    return {
        "current": {"confirmed_development": 6, "raw_gates_resolved_unprocessed": 0, "untouched_lockboxes": 0},
        "classification_buckets": {
            "CONFIRMED": PRODUCTION,
            "RAW_GATES_RESOLVED": [],
            "REQUEST_DEPENDENT": ["ST_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020", "POPOCATEPETL_PUEBLA_2017", "REUNION_SALAZIE_1978_1984", "CHUUK_CHATAAN_2002", "MOUNT_ELGON_BAKALASI_2018"],
            "SPECULATIVE": ["LIMBE_MOUNT_CAMEROON_2001", "MADEIRA_FUNCHAL_2010"],
        },
        "minimum_8": {
            "target": "6 development + 2 untouched lockboxes",
            "fastest_realistic": ["POPOCATEPETL_PUEBLA_2017 after existing reply", "VANUATU_HAROLD_2020 after existing reply"],
            "qualification": "Fastest count route; still requires raw QA, human lockbox designation and a separate production stage. Chuuk is no longer a one-blocker fast path.",
            "ready_now": False,
        },
        "recommended_11": {
            "target": "8 development + 3 complementary untouched lockboxes with coverage",
            "development": ["ST_LUCIA_TOMAS_2010 after Omega reply", "REUNION_SALAZIE_1978_1984 after DSM/visibility artifacts"],
            "lockboxes": ["POPOCATEPETL_PUEBLA_2017 after reply", "VANUATU_HAROLD_2020 after reply", "MOUNT_ELGON event after complete Y/Omega recovery"],
            "substitute": "Chuuk can substitute only if both Omega and target-completeness evidence are recovered.",
            "ready_now": False,
        },
        "ideal_13": {
            "target": "10 development + 3 complementary untouched lockboxes",
            "inherits_recommended": True,
            "additional_development": ["MADEIRA_FUNCHAL_2010 after provenance/Omega", "one newly resolved hotspot/rift candidate such as Limbe only after full raw gates"],
            "ready_now": False,
        },
        "stage3i_without_replies": False,
    }


def report(state: dict, rows: list[dict], new: list[dict], tests: str, v1: str) -> str:
    verdict = {r["candidate_id"]: r["rescue_verdict"] for r in rows}
    return f"""STAGE 3H TARGETED BLOCKER RESCUE & ACQUISITION: COMPLETE

## A. EXECUTIVE RESULT

No candidate has all raw gates resolved. Production remains exactly 6 landscapes, 835,846 terrain units and 8,793 positives. Stage 3H performed no production or ML.

## B. CHUUK Ω FORENSIC RESULT

Not recovered. The source chain documents reconnaissance and oblique photographs of landslide-affected areas, not a target-independent systematically searched and visible domain.

## C. CHUUK TARGET-COMPLETENESS RESULT

Not resolved. Current USGS metadata explicitly says the data are not complete or exhaustive. The 51-feature Tonoas revision refines known locations and does not establish a complete island subset. Final forensic class: `BOTH_BLOCKED`.

## D. RÉUNION EVENT-ATTRIBUTION RESULT

Resolved at dataset level: the 1,440-feature layer is explicitly named and described in its embedded metadata as Cyclone Hyacinthe (1980) landslides, corroborated by the primary paper. No spatial inference was used; individual features do not carry event fields.

## E. RÉUNION OBSERVATION-SUPPORT RESULT

Not yet defensible. The cirque boundary is independent and the 1978/1984 imagery workflow is systematic, but shaded/cloud/failed-photogrammetry terrain was not mappable and no valid-overlap/visibility mask is distributed.

## F. RÉUNION PRE-EVENT TERRAIN RESULT

A scientifically defensible 1978 SfM DSM is documented and validated in the research, but the raster, vertical datum and valid-data mask were not found in public distribution. Public IGN photos are inputs, not a ready terrain surface. M remains blocked.

## G. MOUNT ELGON INVENTORY RESULT

No complete event inventory was found. EMSR325 has 9 mass-movement impacts and UNOSAT has one preliminary combined extent. The 653 + 400+, 233, 284 and 81-feature research inventories are regional/mixed-age rather than specific-event exhaustive Y.

## H. MOUNT ELGON Ω / GEOLOGY RESULT

No event-linked Omega exists because no qualifying event Y exists. Mount Elgon is an eroded Miocene shield, but surrounding terrain includes basement and carbonatite/fenitised units; overlap cannot be quantified without Y/Omega and an authoritative geology vector. Uganda DGSM/NSDI is the appropriate geology route.

## I. REPLACEMENT SEARCH

Ten focused leads were screened (within the 15-lead cap). Limbe 2001 is the best new African/hotspot lead but only a published 57-scar + 6-zone map was located; no reusable vector Y plus independent Omega. No better ready replacement was found.

## J. NEW ACQUISITIONS

Seven public evidence PDFs were preserved and checksum-registered: three Réunion methods/terrain papers and four Mount Elgon inventory/UAV/event sources. No derived or protected data were acquired.

## K. RESCUE DECISION TABLE

- Chuuk: `{verdict['CHUUK_CHATAAN_2002']}`
- Réunion: `{verdict['REUNION_SALAZIE_1978_1984']}`
- Mount Elgon: `{verdict['MOUNT_ELGON_BAKALASI_2018']}`
- Saint Lucia: `{verdict['ST_LUCIA_TOMAS_2010']}`
- Vanuatu: `{verdict['VANUATU_HAROLD_2020']}`
- Popocatépetl: `{verdict['POPOCATEPETL_PUEBLA_2017']}`

## L. LOCKBOX IMPLICATIONS

Keep Chuuk, Popocatépetl, Vanuatu and any future qualifying Mount Elgon event untouched. No model contact occurred. Réunion and Saint Lucia remain preferred development additions if their gates close.

## M. HUMAN REQUESTS STILL REQUIRED

High-value drafts are in `STAGE3H_REQUESTS.md`: USGS for Chuuk systematic support/completeness; BRGM authors for Réunion 1978 DSM and visibility support; Uganda OPM/World Bank for 2018 UAV support/products; Broeckx/Jacobs for dated Mount Elgon vectors and search support. Existing Saint Lucia, Vanuatu and Popocatépetl requests remain pending and were not duplicated.

## N. UPDATED ROUTE TO 8 / 11 / 13

- 8: fastest realistic count route is the already-requested Popocatépetl and Vanuatu packages as two untouched lockboxes, after QA and explicit human designation.
- 11: add Saint Lucia and Réunion as development; protect Popocatépetl, Vanuatu and a recovered full Mount Elgon event as complementary lockboxes. Chuuk is a conditional substitute only after both blockers close.
- 13: inherit the 11 route, then resolve Madeira plus one genuinely complete hotspot/rift development candidate. This remains speculative.

## O. TESTS / CHECKPOINT / V1 / GIT

Stage 3G hash `{state['stage3g_checkpoint_sha256']}` and Stage 3F hash `{state['stage3f_checkpoint_sha256']}` are unchanged. Tests: {tests}. V1: {v1}. Dirty pre-existing worktree preserved; no commit or push.

1. Was Chuuk's independent Ω_obs recovered? **NO.**
2. Was Chuuk's target-completeness problem resolved? **NO.**
3. Is Chuuk RAW_GATES_RESOLVED? **NO.** Both Omega and completeness remain blocked.
4. Can a defensible Cyclone Hyacinthe-specific Réunion Y be isolated? **YES.** The distributed 1,440-polygon layer is explicitly classified as Hyacinthe by its own metadata and primary paper.
5. Is Réunion observation support defensible? **NO, not yet.** The machine-readable visibility/valid-overlap exclusions are missing.
6. Was defensible pre-event terrain found for Réunion? **DOCUMENTED, BUT NOT ACQUIRED.** A validated 1978 DSM exists in the research workflow; the raster and required datum/QA package are not distributed.
7. Is Réunion RAW_GATES_RESOLVED? **NO.**
8. Was a complete/systematically mapped Mount Elgon event inventory found? **NO.**
9. Is Mount Elgon Ω_obs defensible? **NO.**
10. Does the Mount Elgon candidate lie in admissible volcanic terrain? **NOT YET QUANTIFIABLE.** The volcanic edifice is real, but the study region is geologically mixed and qualifying Y/Omega are absent.
11. Is Mount Elgon RAW_GATES_RESOLVED? **NO.**
12. Did we find a better replacement hotspot/shield candidate? **NO.**
13. Did we find a better replacement African/rift candidate? **NO.**
14. How many candidates now have all raw gates resolved? **ZERO.**
15. Which candidates should be DEVELOPMENT candidates? **If resolved: Saint Lucia–Tomas and Réunion–Salazie.** Current production remains the six confirmed development landscapes.
16. Which candidates should be protected as potential LOCKBOXES? **Chuuk–Chata'an, Popocatépetl–Puebla, Vanuatu–Harold, and any future complete Mount Elgon event.**
17. What exact human requests remain necessary? **Chuuk flight/interpreted footprints plus completeness threshold; Réunion 1978 DSM plus valid-data/cloud-shadow/interpreted-area masks and datum/QA; Mount Elgon 2018 UAV/AOI products plus exhaustive event polygons; dated regional Mount Elgon polygons/search support; and the already-sent Saint Lucia, Vanuatu and Popocatépetl requests.**
18. What is now the fastest realistic route from 6 to 8? **Wait for and QA the existing Popocatépetl and Vanuatu packages, then—only in an authorised production stage—freeze both as untouched lockboxes.**
19. What is now the fastest realistic route from 6 to 11? **Resolve Saint Lucia and Réunion for development, and resolve Popocatépetl, Vanuatu and a full Mount Elgon event for three complementary lockboxes.**
20. Can we proceed to a production Stage 3I without waiting for replies? **NO.** Search/QA can continue, but zero candidates currently have all raw gates.
21. Are we ready to train ML? **NO.** The recommended 8-development + 3-lockbox count-and-coverage rule and mandatory composition requirements are not satisfied.
"""


def main(tests: str = "PENDING", v1_status: str = "PENDING") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    state = verify_state()
    if not all([state["stage3g_checkpoint_unchanged"], state["stage3f_checkpoint_unchanged"],
                state["stage3g_candidate_states_unchanged"]]) or state["production_landscapes"] != PRODUCTION or (
                state["production_count"], state["terrain_units"], state["positive_units"], state["target_tier"]) != (
                6, 835846, 8793, "TIER_2_ONLY"):
        raise RuntimeError(f"Frozen input state failed: {state}")
    write_json(OUT / "STATE_VERIFICATION.json", state)
    new = acquisitions()
    write_tsv(OUT / "STAGE3H_SOURCE_REGISTRY.tsv", new)
    chuuk = chuuk_audit()
    event, observation, terrain = reunion_audits()
    elgon_inventory, elgon_omega = mount_elgon_audits()
    replacements = replacement_search()
    rescue = rescue_rows()
    route = route_update()
    write_json(OUT / "CHUUK_FORENSIC_AUDIT.json", chuuk)
    write_json(OUT / "REUNION_EVENT_ATTRIBUTION_AUDIT.json", event)
    write_json(OUT / "REUNION_OBSERVATION_SUPPORT_AUDIT.json", observation)
    write_json(OUT / "REUNION_PRE_EVENT_TERRAIN_AUDIT.json", terrain)
    write_json(OUT / "MOUNT_ELGON_INVENTORY_AUDIT.json", elgon_inventory)
    write_json(OUT / "MOUNT_ELGON_OMEGA_GEOLOGY_AUDIT.json", elgon_omega)
    write_tsv(OUT / "STAGE3H_REPLACEMENT_SEARCH.tsv", replacements)
    write_json(OUT / "STAGE3H_REPLACEMENT_SEARCH.json", replacements)
    write_tsv(OUT / "STAGE3H_RESCUE_STATUS.tsv", rescue)
    (OUT / "STAGE3H_REQUESTS.md").write_text(requests_markdown())
    write_json(OUT / "STAGE3H_ROUTE_UPDATE.json", route)
    diagnostics = {
        "stage": "3H", "new_raw_gates_resolved": 0, "production_rows_added": 0,
        "production_state_changed": False, "ml_trained": False, "exploratory_ml_run": False,
        "lolo_run": False, "h_generated": False, "terrain_units_generated": False,
        "production_tables_generated": False, "y_derived_omega_accepted": False,
        "tier_mixing": False, "candidate_double_counting": False, "emails_sent": False,
        "accounts_created": False, "auth_bypassed": False, "terms_accepted": False,
        "commit_performed": False, "push_performed": False,
        "raw_files_overwritten": False, "replacement_leads_screened": len(replacements),
        "replacement_search_cap_respected": len(replacements) <= 15,
        "deterministic_classification": True,
        "legacy_regression_note": "The broad Stage3D-3H run initially found the Cotabato Stage3D features gzip truncated (EOF and hash mismatch). The pre-existing Stage3D deterministic-replay test regenerated it to the exact frozen checkpoint hash f38ceebda61b236d0d17e3a297f50ea5bb6a0b1a1e149847a7587e87fc497b52; Stage3D, Stage3F and Stage3G checkpoint hashes did not change. Focused frozen-hash recheck then passed.",
        "user_email_candidate_states": {"ST_LUCIA_TOMAS_2010": "ONE_BLOCKER_REMAINS",
                                        "VANUATU_HAROLD_2020": "REQUEST_REQUIRED",
                                        "POPOCATEPETL_PUEBLA_2017": "REQUEST_REQUIRED"},
    }
    write_json(OUT / "STAGE3H_DIAGNOSTICS.json", diagnostics)
    (OUT / "STAGE3H_REPORT.md").write_text(report(state, rescue, new, tests, v1_status))
    artifact_names = sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name != "STAGE3H_CHECKPOINT.json")
    checkpoint = {
        "stage": "3H", "as_of": AS_OF, "stage3g_input_hash": STAGE3G_HASH,
        "production_state": {"landscapes": PRODUCTION, "count": 6, "terrain_units": 835846,
                             "positive_units": 8793, "target_tier": "TIER_2_ONLY"},
        "acquisitions": new, "chuuk_forensic_result": chuuk,
        "reunion_forensic_result": {"event": event, "observation": observation, "pre_event_terrain": terrain},
        "mount_elgon_result": {"inventory": elgon_inventory, "omega_geology": elgon_omega},
        "replacement_search_result": replacements,
        "rescue_table": rescue, "request_package": "outputs/stage3h/STAGE3H_REQUESTS.md",
        "route_update": route, "tests": tests, "v1_state": v1_status,
        "git_state": "DIRTY_PRE_EXISTING_WORKTREE_PRESERVED; NO_COMMIT; NO_PUSH",
        "diagnostics": diagnostics,
        "artifact_hashes": {f"outputs/stage3h/{name}": sha256(OUT / name) for name in artifact_names},
    }
    write_json(OUT / "STAGE3H_CHECKPOINT.json", checkpoint)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="PENDING")
    parser.add_argument("--v1-status", default="PENDING")
    args = parser.parse_args()
    main(args.tests, args.v1_status)
