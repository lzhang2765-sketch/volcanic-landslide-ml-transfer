"""Build Stage 2E audits, access products, Gen-2 contracts, and checkpoint.

No acquisition bytes, account actions, messages, terrain-unit production, H
surface generation, or machine-learning operations occur in this stage.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .raw_store import sha256
from .schema import ROOT
from .stage2c_governance import REQUEST_PACKAGES
from .stage2e_gen2 import (AS_OF, CONTROL_FIELDS, FAMILY_REGISTRY, H_STATES, SEED, aso_controls,
                           ensemble_members, h_feature_ontology, method_spec, software)

OUT = ROOT / "outputs/stage2e"
REPO = ROOT.parent
FREEZE = REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE"
SCRIPT = Path(__file__)
GEN2_SCRIPT = Path(__file__).with_name("stage2e_gen2.py")
STAMP = "2026-10-06T00:00:00+10:00"


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False, lineterminator="\n")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def _v1_verify() -> dict:
    p = subprocess.run([sys.executable, "-m", "tweed_science_fair.ml_v1_freeze", "verify"], cwd=REPO,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")}, capture_output=True, text=True, check=True)
    result = json.loads(p.stdout)
    if not result["v1_intact"] or result["writes_performed"]:
        raise RuntimeError(f"V1 integrity failure: {result}")
    return result


def internal_resolution() -> list[dict]:
    rows = [
        {
            "landscape_id": "LOMBOK_RINJANI_2018", "authoritative_source": "Pusat Survei Geologi GeoMap",
            "publisher": "Geological Agency, Ministry of Energy and Mineral Resources, Indonesia", "scale": "1:100,000 vector catalogue; 1:250,000 regional alternatives",
            "format": "Shapefile behind GeoMap sign-in; BIG ArcGIS polygon service inspected only", "crs": "GeoMap item-specific; BIG service EPSG:4326",
            "unit_attributes": "GeoMap item-specific; BIG: FORMASI,SANDI,PENYUSUN,ENDAPAN,age fields,KRONOSTRAT",
            "licence": "GeoMap open licence: use/download with PSG attribution, no resale; selected file still requires account access",
            "coverage": "Lombok administrative area expected in GeoMap; BIG national regional layer covers Lombok",
            "access_route": "User sign-in at https://geologi.esdm.go.id/geomap then Province=Nusa Tenggara Barat and Kabupaten=Lombok Utara/Timur/Tengah/Barat; prefer 1:100k SHP",
            "evidence_urls": "https://psg.geologi.esdm.go.id/index.php/layanan-geomap/|https://geologi.esdm.go.id/geomap|https://geoservices.big.go.id/gis/rest/services/PTRA/Atlas_250K_Geologi/MapServer/15",
            "raw_unit_mapping": "NOT_PERFORMED_NO_AUTHORISED_VECTOR_BYTES", "omega_analysis": "NOT_CONSTRUCTED",
            "y_gate": "PASS_PROVISIONAL", "omega_gate": "PASS_PROVISIONAL", "m_gate": "PASS_PROVISIONAL_DEM_QUALITY_PENDING", "geology_gate": "BLOCKED_ACCESS",
            "terrain_units_generated": False, "production_ready": False,
            "exact_blocker": "Authoritative reusable vector is behind user sign-in; BIG service is explicitly for atlas.big.go.id and is not substituted.",
            "h_feasibility": "LIKELY_MULTI_EDIFICE_AFTER_GEOLOGY_AND_TEMPORAL_CONTROLS",
        },
        {
            "landscape_id": "ATITLAN_STAN_2005", "authoritative_source": "INSIVUMEH/IGN mapping and USGS Atitlan publications",
            "publisher": "INSIVUMEH / Instituto Geografico Nacional Guatemala / USGS", "scale": "1:250,000 IGN sheet and published Atitlan maps; no reusable authoritative vector located",
            "format": "PDF/raster publications; no first-party polygon GIS found", "crs": "Not available for a reusable vector source",
            "unit_attributes": "Published classes include Quaternary alluvium, volcaniclastic and pyroclastic deposits, Tertiary rocks, intrusive rocks, limestone/metasediments, stratovolcanoes",
            "licence": "USGS public-domain reports; Guatemala source reuse terms not bound to a vector dataset",
            "coverage": "Lake Atitlan/caldera and surrounding highlands in published maps",
            "access_route": "Request the digital source for IGN ND-15-8-G or equivalent polygons from INSIVUMEH/IGN; do not vectorise a small publication figure",
            "evidence_urls": "https://www.insivumeh.gob.gt/insivumeh/revista-geocienciagt-insivumeh_nov2021.pdf|https://www.usgs.gov/publications/recent-geologic-history-lake-atitlan-a-caldera-lake-western-guatemala|https://www.usgs.gov/publications/volcanic-hazards-atitlan-volcano-guatemala",
            "raw_unit_mapping": "NOT_PERFORMED_NO_AUTHORITATIVE_DIGITAL_UNITS", "omega_analysis": "NOT_CONSTRUCTED",
            "y_gate": "PASS_PROVISIONAL", "omega_gate": "PASS_PROVISIONAL", "m_gate": "PASS_PROVISIONAL_DEM_QUALITY_PENDING", "geology_gate": "BLOCKED_SOURCE",
            "terrain_units_generated": False, "production_ready": False,
            "exact_blocker": "No authoritative reusable geological polygons and attributes were located; global/third-party maps are too coarse or insufficiently sourced.",
            "h_feasibility": "LIKELY_CALDERA_RIM_FLANK_AFTER_GEOLOGY_AND_CONTROLS",
        },
        {
            "landscape_id": "KURIKOMA_2008", "authoritative_source": "GSI 2008 emergency aerial-photo interpretation campaign",
            "publisher": "Geospatial Information Authority of Japan", "scale": "Aerial photos 1:10,000; interpretation-range attachment small-scale PDF",
            "format": "PDF without GeoPDF registration; no associated range GIS located", "crs": "Not documented in attachment",
            "unit_attributes": "Not applicable to Omega", "licence": "GSI terms; no new bytes acquired", "coverage": "Published photo-interpretation range",
            "access_route": "Ask GSI Geographic Department for the digital boundary/base-map registration associated with Material 4 from 18 June 2008",
            "evidence_urls": "https://www.gsi.go.jp/johosystem/johosystem60029.html|https://www.gsi.go.jp/REPORT/JIHO/vol117-main117-abstract117.htm",
            "raw_unit_mapping": "NOT_APPLICABLE", "omega_analysis": "NOT_CONSTRUCTED",
            "y_gate": "PASS_PROVISIONAL", "omega_gate": "BLOCKED", "m_gate": "PASS_PROVISIONAL", "geology_gate": "NOT_SUBSET_WITHOUT_OMEGA",
            "terrain_units_generated": False, "production_ready": False,
            "exact_blocker": "No coordinate-supported official interpretation-range boundary; arbitrary PDF matching remains prohibited.",
            "h_feasibility": "DEFER_UNTIL_OMEGA",
        },
    ]
    _csv(OUT / "internal/internal_resolution.csv", rows)
    atitlan_units = [
        ("Quaternary stratovolcano", "MIXED_VOLCANIC", "VOLCANIC_EDIFICE", "CONDITIONAL_TRUE", "Geometry/age attribution required"),
        ("Quaternary pyroclastic deposits", "PYROCLASTIC", "CALDERA_RELATED_DEPOSIT", "CONDITIONAL_TRUE", "Exclude unrelated regional deposits after geometry review"),
        ("Quaternary pumiceous volcaniclastic deposits", "VOLCANICLASTIC", "CALDERA_RELATED_DEPOSIT", "CONDITIONAL_TRUE", "Origin and primary/reworked status unresolved"),
        ("Ash-flow tuffs from ancient caldera", "PYROCLASTIC", "CALDERA_RELATED_DEPOSIT", "CONDITIONAL_TRUE", "Cenozoic age and source geometry required"),
        ("San Jorge / Antigua caldera fill", "VOLCANICLASTIC", "CALDERA_RELATED_DEPOSIT", "CONDITIONAL_TRUE", "May include reworked/sedimentary fill; do not force without attributes"),
        ("Maria Tecun ash-flow tuffs", "PYROCLASTIC", "CALDERA_RELATED_DEPOSIT", "CONDITIONAL_TRUE", "Age and mapped polygon required"),
        ("Hypabyssal and plutonic rocks", "INTRUSIVE", "AMBIGUOUS_FOR_SURFACE_VOLCANIC_DOMAIN", "FALSE", "Frozen ontology does not treat intrusive units as volcanic analysis domain"),
        ("Limestone", "SEDIMENTARY_BASEMENT", "BASEMENT_NON_VOLCANIC", "FALSE", "Exclude"),
        ("Metasediments", "METAMORPHIC_BASEMENT", "BASEMENT_NON_VOLCANIC", "FALSE", "Exclude"),
        ("Quaternary alluvium and lacustrine deposits", "UNCONSOLIDATED_SURFICIAL", "SURFICIAL", "FALSE", "Exclude from bedrock volcanic domain"),
        ("Inferred stratovolcano", "UNKNOWN", "AMBIGUOUS", "FALSE", "Do not force inferred geometry into production domain"),
    ]
    _csv(OUT / "internal/ATITLAN_published_unit_mapping.csv", [
        {"published_map_class": a, "frozen_v2_class": b, "explicit_group": c, "domain_eligible_if_vector_obtained": d,
         "mapping_confidence": "PRELIMINARY_PUBLICATION_LEGEND_ONLY", "notes": e, "production_use": "PROHIBITED_UNTIL_AUTHORITATIVE_VECTOR_BOUND"}
        for a, b, c, d, e in atitlan_units
    ])
    return rows


def puerto_rico_plan() -> list[dict]:
    hashes = json.loads((FREEZE / "ML_V1_HASH_MANIFEST.json").read_text())["files"]
    with (FREEZE / "ML_V1_FILE_MANIFEST.tsv").open(newline="") as f:
        manifest = {r["path"]: r for r in csv.DictReader(f, delimiter="\t")}
    prefix = "Science_Fair_Extension/01_Data_Raw/SF1_verification/puerto_rico_2017_maria_hughes/"
    rows = []
    for p in sorted(k for k in hashes if k.startswith(prefix)):
        h, m = hashes[p], manifest[p]
        if h["role"] != "RAW_INPUT_DATA" or m["role"] != "RAW_INPUT_DATA":
            continue
        name = Path(p).name
        role = "Y" if name == "landslides.zip" else "Y_DOCUMENTATION"
        rows.append({
            "landscape_id": "PUERTO_RICO_MARIA_2017", "v1_path": p, "v1_sha256": h["sha256"],
            "v1_bytes": h["size_bytes"], "v1_stage": h["stage"], "v1_role": h["role"], "scientific_role": role,
            "planned_v2_destination": f"data/raw/stage2e/PUERTO_RICO_MARIA_2017/V1_TRANSFER/{role}/{name}",
            "eligibility": "ELIGIBLE_RAW_SOURCE_ONLY", "transfer_status": "PLANNED_NOT_EXECUTED",
            "v2_unseen_test_eligibility": "INELIGIBLE_V1_HISTORICAL_REFERENCE",
        })
    _csv(OUT / "internal/PUERTO_RICO_MARIA_2017_transfer_plan.csv", rows)
    audit = {
        "eligible_raw_files": len(rows), "eligible_bytes": sum(int(r["v1_bytes"]) for r in rows),
        "Y": "71,431 point locations at headscarp centres; UTM 19N/WGS84; likely POINT_HEADSCARP_TIER1, kept separate from other target families",
        "Omega": "Not an eligible vector file in V1 freeze. Documentation reports overlapping imagery coverage (about 97-99% main island), but exact observable footprint/cloud-shadow exclusions must be acquired.",
        "M": "No eligible V1 M file. Acquire 2015-2016 USGS 3DEP lidar or defensible pre-event fallback with quality masks.",
        "geology": "No eligible V1 geology file. Must acquire USGS/Puerto Rico authoritative digital geology and intersect Cretaceous-Eocene island-arc volcanic units without assuming island-wide volcanics.",
        "expected_blockers": ["exact Omega/visibility geometry", "authoritative volcanic-domain mapping", "pre-event DEM tiling/quality", "Irma/Maria temporal ambiguity"],
        "target_family": "POINT_HEADSCARP_TIER1", "effort": "HIGH: Y is strong; Omega/G/M acquisition and island-scale processing remain",
        "authorised_to_execute": False, "transfer_executed": False,
    }
    _json(OUT / "internal/PUERTO_RICO_MARIA_2017_transfer_audit.json", audit)
    return rows


def coromandel_action() -> None:
    text = """# Coromandel / Cyclone Gabrielle — exact user action (not performed)

1. Open the official [New Zealand Landslide Database](https://www.gns.cri.nz/data-and-resources/new-zealand-landslide-database/) and follow **Explore the webmap**.
2. Select free public registration, use your own email, verify it, and sign in. Do not share the password. Unregistered access exposes points only; registered access exposes all feature classes, including polygons and tables.
3. Filter/search dataset, event, trigger and date for **Cyclone Gabrielle (February 2023)** and **Coromandel**. Confirm the selected dataset's custodian/citation rather than assuming every visible feature belongs to that event.
4. Inspect/download or record:
   - source polygons, affected-area/deposit polygons and debris trails as separate feature classes;
   - stable feature and dataset IDs; landslide type; trigger/event/date; source-versus-runout semantics;
   - mapping basemap/imagery, mapping scale, locational accuracy and categorical quality rank;
   - the independently defined searched/mapped area, imagery footprints, cloud/shadow/unmapped exclusions, and inventory-completeness statement;
   - native geometry format, native CRS and any export reprojection; prefer GeoPackage, otherwise Shapefile/GeoJSON;
   - the exact dataset licence/terms, required citation, non-commercial restriction, and whether bulk export is enabled.
5. Save the data dictionary and dataset metadata beside any later download. Do not treat the database-wide disclaimer or a point display as event-specific completeness evidence.

Accept polygon Y only when event attribution and source/deposit semantics are documented. Accept Omega only when it is an independent systematically inspected area with visibility exclusions. A hull/buffer of positives, administrative Coromandel outline, screenshot, or absence of displayed points is not Omega.

If polygon or mapping-domain export is unavailable, send the retained request (not sent in Stage 2E):

> To: [institutional data custodian; address removed from public source copy]  
> Subject: Student research request — Cyclone Gabrielle 2023 Coromandel polygons and mapped domain  
> I am a student building a non-commercial, multi-landscape volcanic-terrain landslide study. Could you provide the event-specific Coromandel Cyclone Gabrielle source/affected-area polygons and the independent area systematically mapped, including imagery footprints, cloud/visibility exclusions, stable IDs, trigger/date, mapping method, positional-quality fields, native CRS, licence and citation? GeoPackage is preferred. I will acknowledge Earth Sciences New Zealand and comply with the NZLD non-commercial terms. Thank you.
"""
    p = OUT / "access/COROMANDEL_USER_ACTION.md"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)


def request_scoring() -> list[dict]:
    # Scores are workflow priorities, not evidence-gate substitutes.  0-3; risk
    # is adverse and subtracted.  Existing order is retained because no new
    # concrete evidence changed the expected-value ordering.
    score = {
        "RP01": (3, 3, 1, 3, 2, 3, 1, 3, 1), "RP02": (2, 3, 3, 3, 3, 3, 3, 2, 1),
        "RP03": (2, 2, 1, 3, 2, 3, 2, 2, 1), "RP04": (2, 2, 1, 2, 2, 3, 3, 1, 1),
        "RP05": (1, 2, 2, 2, 2, 2, 1, 2, 0), "RP07": (1, 3, 2, 2, 3, 2, 3, 1, 1),
        "RP06": (1, 2, 1, 2, 2, 3, 2, 1, 1),
    }
    prior = ["RP01", "RP02", "RP03", "RP04", "RP05", "RP07", "RP06"]
    by_id = {x[0]: x for x in REQUEST_PACKAGES}
    rows = []
    request_text = ["# Ready-to-send external requests (not sent)", ""]
    for rank, pid in enumerate(prior, 1):
        x = by_id[pid]
        vals = score[pid]
        total = 4 * vals[0] + sum(vals[1:7]) + 2 * vals[7] - 2 * vals[8]
        rows.append({
            "rank": rank, "package_id": pid, "landscape_id": x[1], "blocker_resolution_probability": vals[0],
            "y_quality": vals[1], "omega_quality": vals[2], "m_feasibility": vals[3], "volcanic_domain_confidence": vals[4],
            "diversity_contribution": vals[5], "h_potential": vals[6], "expected_response_speed": vals[7],
            "licensing_access_risk": vals[8], "priority_score": total, "recipient_custodian": x[3],
            "exact_missing_artifact": x[4], "preferred_format": x[6], "licence_question": x[8],
            "ranking_decision": "RETAIN_STAGE2D_ORDER_NO_NEW_CONCRETE_EVIDENCE",
        })
        request_text += [f"## {rank}. {x[1]} — {pid}", "", f"To: {x[3]}", "", f"Missing: {x[4]}", "", x[9], "",
                         f"Please also confirm native CRS, mapping/search support, imagery dates, licence, citation and permission for non-commercial student research. Preferred format: {x[6]}.", ""]
    _csv(OUT / "access/external_request_scoring.csv", rows)
    (OUT / "access/READY_TO_SEND_REQUESTS.md").write_text("\n".join(request_text))
    return rows


def second_wave() -> list[dict]:
    rows = [
        {"rank": 1, "landscape_id": "WILSONS_RIVER_NSW_2022", "diversity": "Oceania; old continental hotspot volcanics; H-capable plateau",
         "dataset_first_evidence": "3,689 manually validated affected-area polygons; catchment stated", "y": "PASS_WITH_LINEAGE_QUALIFICATION", "omega": "UNRESOLVED_TERRAIN_FILTER_SUPPORT",
         "m": "UNRESOLVED_PRE2022_LIDAR_EPOCH", "geology_h": "NSW geology likely; remnant envelope feasible", "access_action": "Ask authors for candidate-generation/search mask, rejected-area mask and exact imagery/DEM epochs",
         "decision": "TARGETED_REQUEST"},
        {"rank": 2, "landscape_id": "DOMINICA_MARIA_2017", "diversity": "Caribbean volcanic island; multi-edifice; non-Asian H",
         "dataset_first_evidence": "Restricted Zenodo polygons separate scarp/transport/accumulation and include Cloud.shp", "y": "LIKELY_PASS_AFTER_ACCESS", "omega": "CLOUD_MASK_ALONE_INSUFFICIENT",
         "m": "GLOBAL_PRE_EVENT_FALLBACK_FEASIBLE", "geology_h": "volcanic island/multi-edifice feasible", "access_action": "User login/request for restricted files plus complete imagery/search footprint and unmapped exclusions",
         "decision": "TARGETED_ACCESS"},
        {"rank": 3, "landscape_id": "IFFI_ITALIAN_VOLCANIC_DISTRICT", "diversity": "Europe; Etna/Vesuvius/Phlegraean volcanic terrains",
         "dataset_first_evidence": "Official ISPRA IFFI polygons downloadable/OGC; cumulative inventory", "y": "PASS_CUMULATIVE_GEOMETRY", "omega": "UNRESOLVED_SYSTEMATIC_SUPPORT_AND_EPOCH",
         "m": "TEMPORAL_PAIRING_DIFFICULT", "geology_h": "authoritative Italian geology; family depends on selected district", "access_action": "Select one homogeneous volcanic district and request/source project completeness, mapping epoch and searched domain",
         "decision": "TARGETED_DISTRICT_AUDIT"},
        {"rank": 4, "landscape_id": "SAO_MIGUEL_AZORES", "diversity": "Europe; oceanic hotspot island; calderas; H-capable",
         "dataset_first_evidence": "Official Azores portal has landslide-risk cartography but occurrence vectors/support not verified", "y": "UNRESOLVED", "omega": "UNRESOLVED",
         "m": "LIKELY_FEASIBLE", "geology_h": "strong caldera/multi-edifice potential", "access_action": "Ask regional portal custodian for occurrence polygons, inventory method, searched domain and date—not hazard zones",
         "decision": "TARGETED_REQUEST"},
        {"rank": 5, "landscape_id": "PATAGONIA_PALI", "diversity": "South America; cold high-relief arc and volcanic subset",
         "dataset_first_evidence": "Zenodo supplies manual training polygons, predicted polygons and a study area separately", "y": "MANUAL_SUBSET_TOO_SMALL_PREDICTIONS_NOT_GROUND_TRUTH", "omega": "STUDY_AREA_EXISTS_FOR_TEST_TILE_ONLY",
         "m": "CHRONOLOGY_UNRESOLVED", "geology_h": "volcanic intersection likely small; H low", "access_action": "Ask whether full-area manual labels and systematic validation/search support exist; never use predictions as Y",
         "decision": "FAST_FAIL_UNLESS_MANUAL_SUPPORT_PROVIDED"},
        {"rank": 6, "landscape_id": "RWANDA_VIRUNGA_NW", "diversity": "Africa; continental rift volcanic margin",
         "dataset_first_evidence": "National compilation data available only on request and privacy-limited; Kivu points already in primary queue", "y": "UNRESOLVED_MIXED_SOURCES", "omega": "UNRESOLVED",
         "m": "10m DEM exists but access/epoch unresolved", "geology_h": "volcanic subset uncertain", "access_action": "Request reusable polygons and per-source mapped domain only if RP03 does not resolve African-rift need",
         "decision": "CONDITIONAL_BACKUP"},
        {"rank": 7, "landscape_id": "REUNION_BDMVT", "diversity": "Africa; oceanic hotspot shield; excellent H morphology",
         "dataset_first_evidence": "Official downloadable BRGM/Georisques cumulative reports from heterogeneous sources", "y": "POINT_REPORTS", "omega": "FAIL_NO_SYSTEMATIC_SEARCH_DOMAIN",
         "m": "FEASIBLE", "geology_h": "high H potential", "access_action": "Do not request BDMvt itself; pursue only a named systematic campaign with polygons and its mapped domain",
         "decision": "FAST_FAIL_CURRENT_DATASET"},
    ]
    _csv(OUT / "access/second_wave_access.csv", rows)
    return rows


def dashboard(internal: list[dict]) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "outputs/stage2d/landscape_state_dashboard.csv")
    state_map = {r["landscape_id"]: ("BLOCKED_ACCESS" if r["landscape_id"] == "LOMBOK_RINJANI_2018" else r["production_ready"] and "READY_FOR_TERRAIN_UNITS" or "") for r in internal}
    for i, r in d.iterrows():
        lid = r["landscape_id"]
        if lid == "LOMBOK_RINJANI_2018": d.at[i, "state"] = "BLOCKED_ACCESS"
        if lid in ("ATITLAN_STAN_2005", "HOKKAIDO_IBURI_2018"): d.at[i, "state"] = "BLOCKED_VOLCANIC_DOMAIN"
        if lid == "KURIKOMA_2008": d.at[i, "state"] = "BLOCKED_OMEGA"
    d["stage2e_h_state"] = "H_UNAVAILABLE"
    d["stage2e_h_scientific_status"] = "NOT_AUDITED"
    for lid, hstate, status in [
        ("KUMAMOTO_ASO_2016", "H_UNAVAILABLE", "GEN2_CONTROL_GATE_FAILED_NO_SURFACE"),
        ("NOTO_PENINSULA_2024", "AGE_PROXY_ONLY", "NO_FULL_EDIFICE_RECONSTRUCTION"),
        ("LOMBOK_RINJANI_2018", "H_UNAVAILABLE", "LIKELY_AFTER_GEOLOGY"),
        ("ATITLAN_STAN_2005", "H_UNAVAILABLE", "LIKELY_AFTER_GEOLOGY"),
        ("KURIKOMA_2008", "H_UNAVAILABLE", "DEFER_UNTIL_OMEGA"),
    ]:
        ix = d.index[d.landscape_id.eq(lid)]
        d.loc[ix, ["stage2e_h_state", "stage2e_h_scientific_status"]] = [hstate, status]
    d["harmonisation_status"] = d["state"].map(lambda x: "COMPLETE" if x == "READY_FOR_TERRAIN_UNITS" else ("IN_PROGRESS_BLOCKED" if str(x).startswith("BLOCKED") else "NOT_STARTED"))
    _csv(OUT / "landscape_state_dashboard.csv", d)
    ready = d[d.state.eq("READY_FOR_TERRAIN_UNITS")]
    summary = {
        "candidate_count": int(len(d)), "production_ready_count": int(len(ready)),
        "harmonisation_in_progress_count": int(d.harmonisation_status.eq("IN_PROGRESS_BLOCKED").sum()),
        "state_counts": d.state.value_counts().sort_index().to_dict(),
        "h_capable_count_conceptual": 4, "h_surface_passed_qc_count": 0,
        "production_ready_continents": ready.continent.value_counts().to_dict(),
        "production_ready_tectonic_settings": ready.tectonic_setting.value_counts().to_dict(),
        "production_ready_climates": ready.climate.value_counts().to_dict(),
        "production_ready_target_families": ready.target_family.value_counts().to_dict(),
        "v1_historical_status_counts": d.v1_historical_status.value_counts().to_dict(),
        "unseen_test_eligibility_counts": d.final_unseen_eligibility.value_counts().to_dict(),
        "warning": "Production-ready means all harmonisation gates and production terrain units passed; metadata-ready is insufficient.",
    }
    _json(OUT / "landscape_state_summary.json", summary)
    return d


def write_gen2() -> tuple[pd.DataFrame, dict]:
    gdir = OUT / "gen2"; gdir.mkdir(parents=True, exist_ok=True)
    controls, controls_geo = aso_controls(sha256)
    _csv(gdir / "kumamoto_reconstruction_controls.csv", controls)
    controls_geo.to_file(gdir / "kumamoto_reconstruction_controls.geojson", driver="GeoJSON")
    spec = method_spec(controls); _json(gdir / "reconstruction_method.json", spec)
    _json(gdir / "family_registry.json", {"families": FAMILY_REGISTRY})
    members = ensemble_members(); _csv(gdir / "ensemble_candidate_parameters.csv", members)
    _csv(gdir / "h_feature_ontology.csv", h_feature_ontology())
    _json(gdir / "h_missingness.json", {
        "allowed_states": H_STATES, "zero_fill_forbidden": True,
        "KUMAMOTO_ASO_2016": {"state": "H_UNAVAILABLE", "reason": "No surface passed the control/temporal-target gate"},
        "NOTO_PENINSULA_2024": {"state": "AGE_PROXY_ONLY", "reason": "Mapped age/history may be used; full palaeo-edifice surface unsupported"},
        "rules": {"FULL_RECONSTRUCTION": "all core surface outputs plus uncertainty available",
                  "PARTIAL_RECONSTRUCTION": "surface valid only inside explicit support mask; outside remains missing",
                  "AGE_PROXY_ONLY": "age/history fields may exist; Z_palaeo and change geometry remain missing",
                  "H_UNAVAILABLE": "all H numeric values missing, never zero"},
    })
    _json(gdir / "external_reference_requirement.json", {
        "feature": "DeltaD_anomaly", "implemented": False,
        "requirement": "Expected morphology must be trained/estimated from an external empirical reference population excluded from the target landscape.",
        "forbidden": "Using Aso's own reconstructed surface to define expected Aso geometry and then scoring Aso against itself.",
        "future_data": ["external volcanic morphology population", "morphology/age/climate matching variables", "leave-landscape-out reference predictions", "reference uncertainty"],
    })
    return controls, spec


def class_imbalance() -> None:
    rows = [
        {"landscape_id": "KUMAMOTO_ASO_2016", "target_family": "POINT_EVENT_TIER2", "valid_units": 354123, "positive": 656, "negative": 353467, "prevalence": 656/354123},
        {"landscape_id": "NOTO_PENINSULA_2024", "target_family": "POLYGON_AFFECTED_TIER2", "valid_units": 96812, "positive": 2148, "negative": 94664, "prevalence": 2148/96812},
    ]
    for r in rows:
        r.update(balance_policy="NATURAL_PREVALENCE_UNCHANGED", forbidden="SMOTE|OVERSAMPLE|UNDERSAMPLE|LABEL_CHANGE", future_primary_metric="PR_AUC_NO_EVALUATION_STAGE2E", grouping="LANDSCAPE_ID")
    _csv(OUT / "class_imbalance_metadata.csv", rows)


def report(internal, requests, wave, controls, spec, dashboard_df) -> None:
    ctrl_counts = controls.decision.value_counts().to_dict()
    lines = [
        "# Stage 2E — internal rescue, targeted access, and Gen-2 control prototype", "",
        "## Scope and hard stops", "",
        "No ML was trained; no LOLO, model metric, probability, SHAP, lockbox assignment, or susceptibility product was created. No account was created, no request was sent, no scientific data were downloaded, no V1 file was modified, and no H surface was generated.", "",
        "## Track A — internal", "",
    ]
    for r in internal:
        lines += [f"### {r['landscape_id']}", "", f"- Authoritative route: {r['authoritative_source']} ({r['scale']}).",
                  f"- Gates: Y `{r['y_gate']}`; Omega `{r['omega_gate']}`; M `{r['m_gate']}`; geology `{r['geology_gate']}`.",
                  f"- Terrain units generated: **no**. Production-ready: **no**.", f"- Exact blocker: {r['exact_blocker']}", f"- H: {r['h_feasibility']}.", ""]
    lines += [
        "### Puerto Rico", "", "A controlled three-file raw-only transfer plan is complete but was not executed. The frozen source contains a 71,431-point headscarp-centre inventory and documentation, but no exact Omega vector, pre-event M, or geology. It remains a high-effort historical/reference candidate and ineligible for a future unseen test.", "",
        "## Track B — expansion", "", "- Coromandel: exact free-registration and post-login inspection/export instructions are final; fallback request retained and unsent.",
        "- Seven existing request packages were scored on the nine requested dimensions. No new concrete evidence justified changing the Stage 2D order: PNG Porgera, Banks Peninsula, Kivu NTK, Mount Elgon, Guatemala 1976, Miravalles/Otto, Kivu Kalehe.",
        f"- Second wave: {len(wave)} tightly bounded candidates, including explicit fast-fails; no broad rediscovery.",
        "- Realistic path to 5: current 2 + Coromandel + two successful primary requests, or one request plus a fully resolved internal landscape.",
        "- Realistic path to 10: current 2 + Coromandel + Lombok/Atitlan/Kurikoma only if their gates resolve + four successful primary requests.",
        "- Realistic path to 15: the primary queue and internal rescues are insufficient alone; it requires strong primary-request success plus several second-wave successes. Gates must not be weakened.",
        "- Largest missing diversity: South America, Africa/continental rift, Europe, oceanic hotspot islands, arid/high-altitude terrain, and non-Asian H-capable morphology.", "",
        "## Track C — Gen-2", "", f"- Control database: {len(controls)} records; decisions {json.dumps(ctrl_counts, sort_keys=True)}. No occurrence target is a control.",
        "- Conceptual family: hybrid caldera-rim + preserved-remnant/flank + stratigraphic constraints. A single radial cone is rejected for Aso's multi-centre, caldera-discontinuous history.",
        "- Ensemble architecture: 48 deterministic candidate parameter combinations across thin-plate RBF, tension spline and constrained-envelope families. They are parameter designs only; all surface executions are gated off.",
        "- Uncertainty model: within-member parameter spread, between-family disagreement, structural scenarios, control distance/support, and median/IQR/SD outputs.",
        "- Validation design: spatially blocked withheld independent geological/topographic controls, vertical residual/bias/IQR coverage, distance-stratified residuals, cross-sections and geological rejection checks; never Y.",
        f"- QC: `{spec['scientific_qc']}`. The temporal target is unresolved and there are {spec['control_gate']['available_admissible_numeric_height_controls']} admissible numeric height controls versus a predeclared minimum of {spec['control_gate']['required_minimum_numeric_height_controls']}.",
        "- No small pilot extraction was generated because a computational demonstration cannot substitute for missing palaeoelevation evidence. Production H is not authorised for the next stage.",
        "- The common ontology distinguishes core, candidate, family-specific, sensitivity-only, rejected and deferred/external-reference features. H missingness is explicit; no zero filling.", "",
        "## Global", "", "- Production-ready: 2 landscapes (Kumamoto/Aso and Noto); 450,935 production terrain units unchanged.",
        "- Conceptually H-capable: 4; H surfaces that passed scientific QC: 0. Noto remains `AGE_PROXY_ONLY`.",
        f"- Dashboard: {len(dashboard_df)} candidates; production-ready settings remain Asia-only.",
        "- Scientific downloads: 0 bytes. Web research inspected metadata/access pages only.",
        "- Natural class prevalence is preserved. Landscape ID remains the independence/grouping unit.", "",
        "## Explicit final answers", "",
        "1. **2** landscapes are genuinely production-ready.",
        "2. **No.** Lombok remains blocked by signed-in access to authoritative vector geology and DEM-quality binding.",
        "3. **No.** Atitlan remains blocked by the absence of authoritative reusable geology polygons and DEM-quality binding.",
        "4. **Potentially, but not from current public artifacts.** Kurikoma needs GSI's coordinate-supported digital range/base-map registration; until then it remains blocked.",
        "5. Register for NZLD and inspect/export Coromandel; send the seven prepared requests in ranked order; sign in to Indonesian GeoMap and obtain Lombok 1:100k SHP; request Atitlan digital geology and Kurikoma's digital interpretation boundary; then pursue only the bounded second wave.",
        "6. **The Gen-2 architecture is scientifically defensible; an Aso surface is not yet.** Its evidence gate correctly stopped the reconstruction.",
        "7. **Yes at architecture level.** The family registry and common ontology cover calderas, radial edifices, remnant envelopes, stratigraphic surfaces, external analogues and age-only cases.",
        "8. **No.** Production H generation is not justified until Aso's temporal target, explicit rim/remnant geometry, numeric ages and palaeoelevation controls pass holdout QC.",
        "9. **D — combination**, led by (A) access/acquisition and (C) environmental/geological completion; continue only control acquisition/design for (B), not production H.",
        "10. **No.** The project is not ready for any ML training.",
    ]
    (OUT / "STAGE2E_REPORT.md").write_text("\n".join(lines) + "\n")


def lineage_and_checkpoint(before, after, controls, spec) -> dict:
    excluded = {"lineage.json", "STAGE2E_CHECKPOINT.json"}
    artifacts = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded)
    scripts = {str(SCRIPT.relative_to(ROOT)): sha256(SCRIPT), str(GEN2_SCRIPT.relative_to(ROOT)): sha256(GEN2_SCRIPT)}
    input_paths = [ROOT / "outputs/stage2d/STAGE2D_CHECKPOINT.json", ROOT / "outputs/stage2d/landscape_state_dashboard.csv",
                   ROOT / "data/interim/_stage2c_aso/geo_A.shp", ROOT / "data/interim/_stage2c_aso/geo_L.shp",
                   ROOT / "data/interim/_stage2c_aso/ol1.shp", ROOT / "data/interim/_stage2c_aso/pnt.shp",
                   FREEZE / "ML_V1_HASH_MANIFEST.json", FREEZE / "ML_V1_FILE_MANIFEST.tsv"]
    inputs = [{"path": str(p.relative_to(REPO)), "sha256": sha256(p)} for p in input_paths]
    rows = []
    for p in artifacts:
        rows.append({"output": str(p.relative_to(ROOT)), "output_sha256": sha256(p), "operation": "STAGE2E_DETERMINISTIC_AUDIT_OR_SPECIFICATION",
                     "parameters": {"as_of": AS_OF, "seed": SEED, "surface_generation": False, "scientific_download_bytes": 0},
                     "software": software(), "scripts": scripts, "inputs": inputs, "timestamp": STAMP})
    _json(OUT / "lineage.json", rows)
    output_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in artifacts}
    output_hashes["outputs/stage2e/lineage.json"] = sha256(OUT / "lineage.json")
    cp = {
        "stage": "2E", "as_of": AS_OF, "prior_checkpoint": "outputs/stage2d/STAGE2D_CHECKPOINT.json",
        "prior_checkpoint_sha256": sha256(ROOT / "outputs/stage2d/STAGE2D_CHECKPOINT.json"),
        "production_ready_landscapes": ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024"], "production_terrain_units": 450935,
        "lombok_production_ready": False, "atitlan_production_ready": False, "kurikoma_production_ready": False,
        "h_reconstruction_surface_generated": False, "h_production_authorised": False, "gen2_scientific_qc": spec["scientific_qc"],
        "training": False, "lolo": False, "model_metrics": False, "lockbox_assignment": False, "probability_calibration": False,
        "natural_class_balance_preserved": True, "scientific_download_bytes": 0, "external_messages_sent": 0, "accounts_created": 0,
        "v1_integrity_before": before, "v1_integrity_after": after, "output_hashes": output_hashes,
        "provenance_hash": output_hashes["outputs/stage2e/lineage.json"],
    }
    _json(OUT / "STAGE2E_CHECKPOINT.json", cp)
    return cp


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    before = _v1_verify()
    internal = internal_resolution(); puerto_rico_plan(); coromandel_action()
    requests = request_scoring(); wave = second_wave(); d = dashboard(internal)
    controls, spec = write_gen2(); class_imbalance(); report(internal, requests, wave, controls, spec, d)
    after = _v1_verify()
    diagnostics = {"v1_before": before, "v1_after": after, "control_records": len(controls),
                   "control_schema": CONTROL_FIELDS, "ensemble_seed": SEED, "ensemble_members": 48,
                   "h_states": H_STATES, "surface_generation": False, "scientific_download_bytes": 0}
    _json(OUT / "STAGE2E_DIAGNOSTICS.json", diagnostics)
    cp = lineage_and_checkpoint(before, after, controls, spec)
    return cp


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
