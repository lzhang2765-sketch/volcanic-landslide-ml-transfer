"""Build Stage 2A.2 (dataset-first discovery) without mutating Stage 1, Stage 2A or Stage 2A-extension outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .build_stage1 import priority, score
from .build_stage2a_extension import ACQ_COLUMNS, EXT_COLUMNS, INV_COLUMNS, SOURCE_COLUMNS
from .schema import COLUMNS, CONFIG, FUTURE_FIELDS, RATING_FIELDS, ROOT
from .stage2a2_records import (D, DATASETS, DATE, EXISTING_GATES, NEW_CANDIDATES, NEW_SOURCES, NOTO_ACQUISITION, NOTO_GEOLOGY_SAMPLE,
                               ORIGIN, SEARCH_LOG, TRANSIENT_INSPECTIONS, UNSEEN_INELIGIBLE)

META = ROOT / "metadata"
OUT = ROOT / "outputs" / "stage2a2"
NOTES = ROOT / "screening" / "evidence" / "stage2a2"
S2A2_COLUMNS = EXT_COLUMNS + [
    "discovery_origin", "v2_unseen_test_eligibility", "stage2a2_screened", "stage2a2_readiness", "stage2a2_fast_fail_step",
    "stage2a2_volcanic_intersection", "stage2a2_volcanic_evidence", "stage2a2_blocker", "stage2a2_y_resolution", "stage2a2_omega_resolution",
    "stage2a2_m_resolution", "stage2a2_m_leakage_risk", "stage2a2_h_resolution", "stage2a2_source_ids", "stage2a2_date"]
DATASET_COLUMNS = ["dataset_id", "dataset_title", "publisher", "url", "landscape_id", "step_a_spatial_y", "step_b_omega", "step_c_volcanic",
                   "step_d_temporal_m", "fast_fail_outcome", "volcanic_intersection", "reason", "verification", "access_date"]
SEARCH_COLUMNS = ["search_id", "repository_or_collection", "query_or_route", "dataset_ids", "outcome", "date"]
PRIOR = ["metadata/landscape_candidates.csv", "outputs/screening/STAGE_1_SCREENING_CHECKPOINT.json",
         "metadata/stage2a_landscape_candidates.csv", "metadata/stage2a_source_manifest.csv", "metadata/stage2a_inventory_manifest.csv",
         "outputs/stage2a/STAGE2A_CHECKPOINT.json", "metadata/stage2a_extension_landscape_candidates.csv",
         "metadata/stage2a_extension_source_manifest.csv", "metadata/stage2a_extension_inventory_manifest.csv",
         "outputs/stage2a_extension/STAGE2A_EXTENSION_CHECKPOINT.json"]
READY = ("READY_CORE", "READY_CORE_H")


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in data)


def canon(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage1_row(rec: dict) -> dict:
    """Apply the frozen Stage-1 scoring/priority rules to a new dataset-first candidate."""
    rec = dict(rec)
    rec.update(experimental_role="UNASSIGNED", lockbox_status="UNASSIGNED", lockbox_assignment_stage="NOT_ASSIGNED",
               lockbox_notes="No experimental or final-lockbox assignment is permitted during Stage 2A.2")
    ratings = dict(zip(CONFIG["score_components"], rec.pop("ratings")))
    for comp, col in RATING_FIELDS.items():
        rec[col] = ratings[comp]
    rec["screening_score"] = score(ratings)
    pc, why = priority(rec)
    rec.update(priority_class=pc, screen_status=CONFIG["status_from_priority"][pc])
    if pc in ("REJECT", "HOLD") and rec["exclusion_reason"] == "NONE":
        rec["exclusion_reason"] = why
    for f in FUTURE_FIELDS:
        rec[f] = ""
    missing, extra = set(COLUMNS) - set(rec), set(rec) - set(COLUMNS)
    if missing or extra:
        raise ValueError(f"{rec['landscape_id']}: missing {missing} extra {extra}")
    row = {c: rec[c] for c in COLUMNS}
    row.update(stage2a_screened="FALSE", stage2a_wave="NOT_SCREENED", stage2a_readiness="NOT_SCREENED",
               stage2a_extension_investigated="FALSE", stage2a_extension_readiness="NOT_SCREENED")
    return row


def build_candidates() -> list[dict]:
    data = rows(META / "stage2a_extension_landscape_candidates.csv")
    for row in data:
        row["discovery_origin"] = "STAGE_1"
    data += [dict(_stage1_row(rec), discovery_origin=ORIGIN) for rec in NEW_CANDIDATES]
    for row in data:
        lid = row["landscape_id"]
        row["v2_unseen_test_eligibility"] = ("INELIGIBLE_V1_HISTORICAL: " + UNSEEN_INELIGIBLE[lid]) if lid in UNSEEN_INELIGIBLE else "UNDETERMINED"
        d = D.get(lid)
        if not d:
            row.update(stage2a2_screened="FALSE", stage2a2_readiness=row["stage2a_extension_readiness"])
            continue
        if lid in EXISTING_GATES:
            row.update(EXISTING_GATES[lid])
        row["inventory_tier"] = d["target_tier"]
        row.update(stage2a2_screened="TRUE", stage2a2_readiness=d["readiness"], stage2a2_fast_fail_step=d["fast_fail_step"],
                   stage2a2_volcanic_intersection=d["volcanic_intersection"], stage2a2_volcanic_evidence=d["volcanic_evidence"],
                   stage2a2_blocker=d["blocker"], stage2a2_y_resolution=d["y_resolution"], stage2a2_omega_resolution=d["omega_resolution"],
                   stage2a2_m_resolution=d["m_resolution"], stage2a2_m_leakage_risk=d["m_leakage_risk"], stage2a2_h_resolution=d["h_resolution"],
                   stage2a2_source_ids="|".join(d["source_ids"]), stage2a2_date=DATE)
    return data


def build_sources() -> list[dict]:
    data = rows(META / "stage2a_extension_source_manifest.csv")
    for sid, lid, role, title, org, stype, url, doi, mfile in NEW_SOURCES:
        path = ROOT / mfile if mfile else None
        data.append(dict(source_id=sid, landscape_id=lid, data_role=role, source_title=title, source_organisation=org, source_type=stype,
                         citation=title, doi=doi, url=url, access_date=DATE, downloadable="TRUE" if stype.endswith("DATASET") else "UNKNOWN",
                         file_format=path.suffix.lstrip(".").upper() if path else "UNKNOWN", licence="See candidate record",
                         spatial_coverage="See Stage-2A.2 evidence note", temporal_coverage="See Stage-2A.2 evidence note", resolution_or_scale="UNKNOWN",
                         tier_relevance="See gate table", omega_obs_evidence="See gate table", h_relevance="See H audit", verified="VERIFIED",
                         notes="Stage-2A.2 dataset-first evidence", authors="UNKNOWN", publication_year="UNKNOWN", access_route=url,
                         stage2a_verified="TRUE", metadata_file=mfile, metadata_sha256=sha(path) if path and path.exists() else ""))
    return data


def build_inventories(candidates: list[dict]) -> list[dict]:
    data = rows(META / "stage2a_extension_inventory_manifest.csv")
    first_y = {}
    for sid, lid, role, *_ in NEW_SOURCES:
        if "Y" in role.split("|"):
            first_y.setdefault(lid, sid)
    existing = {r["landscape_id"] for r in data}
    for c in candidates:
        lid = c["landscape_id"]
        if c["discovery_origin"] != ORIGIN or lid in existing:
            continue
        sid = first_y.get(lid, "UNKNOWN")
        omega_sid = sid if any(s[0] == sid and "OMEGA" in s[2] for s in NEW_SOURCES) else "UNKNOWN"
        data.append(dict(inventory_id=f"INV_{lid}_01", landscape_id=lid, inventory_name=c["inventory_name"], inventory_source_id=sid,
                         inventory_year=c["inventory_year"], inventory_trigger=c["inventory_trigger"], inventory_mapping_method=c["inventory_mapping_method"],
                         geometry_type=c["inventory_geometry_type"], target_tier=c["inventory_tier"], feature_count_estimate=c["inventory_count_estimate"],
                         mapped_area=c["inventory_area"], observation_domain_source_id=omega_sid,
                         observation_domain_establishable=c["observation_domain_establishable"], verification_status=c["y_verification"],
                         downloadable=c["inventory_downloadable"], access_route=c["inventory_url_or_identifier"], licence=c["licence"],
                         notes="Stage-2A.2 dataset-first inventory record", stage2a_role="TARGET_Y_CANDIDATE", scientific_usability=c["stage2a2_readiness"],
                         mapping_bias=c["stage2a2_blocker"]))
    return data


def acquisition_rows() -> list[dict]:
    base = "data/raw/stage2b/NOTO_PENINSULA_2024"
    out = [dict(acquisition_id=aid, landscape_id="NOTO_PENINSULA_2024", source_id="S2A2_NOTO_GSI", data_role="Y|OMEGA",
                dataset_name="GSI 2024 Noto slope-failure/deposition GeoJSON with interpretation range",
                source_url="https://www.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html",
                download_url=f"https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/{fn}", expected_format="ZIP/GeoJSON", expected_size=size,
                licence="GSI content terms; attribution required", temporal_coverage="Orthophotos 2-17 Jan 2024",
                spatial_coverage="Published district interpretation range", resolution="Features >= ~100 m2", target_destination=f"{base}/S2A2_NOTO_GSI/{fn}",
                original_filename=fn, checksum_algorithm="SHA256", checksum="", download_status="PLANNED", verification_status="READY_TO_DOWNLOAD",
                notes="Retain combined Tier-2 semantics, 判読範囲 and cloud_poly layers; Stage-2A.2 screening hash recorded in transient_inspection_manifest.json")
           for aid, fn, size in NOTO_ACQUISITION]
    out.append(dict(acquisition_id="ACQ_NOTO_DEM", landscape_id="NOTO_PENINSULA_2024", source_id="S2_GSI_HISTORICAL_DEM", data_role="DEM",
                    dataset_name="GSI Fundamental Geospatial Data historical DEM", source_url="https://service.gsi.go.jp/kiban/",
                    download_url="https://service.gsi.go.jp/kiban/app/map/", expected_format="ZIP/JPGIS-GML", expected_size="UNKNOWN_UNTIL_MESH_SELECTION",
                    licence="GSI Fundamental Geospatial Data terms", temporal_coverage="Creation/source date must be <= 2023-12-31",
                    spatial_coverage="Meshes intersecting the union interpretation range", resolution="Prefer 5 m pre-event; 10 m fallback",
                    target_destination=f"{base}/S2_GSI_HISTORICAL_DEM/", original_filename="TBD_BY_OFFICIAL_MESH_QUERY", checksum_algorithm="SHA256",
                    checksum="", download_status="PLANNED_SELECTION_GATE", verification_status="REFUSE_IF_SOURCE_EPOCH_POST_EVENT_OR_UNDOCUMENTED",
                    notes="GSI post-event laser DEM explicitly excluded from M"))
    out.append(dict(acquisition_id="ACQ_NOTO_G", landscape_id="NOTO_PENINSULA_2024", source_id="SRC_GEOLOGY_GSJ_SEAMLESS", data_role="GEOLOGY|AGE",
                    dataset_name="Seamless Digital Geological Map of Japan V2", source_url="https://gbank.gsj.jp/seamless/",
                    download_url="https://gbank.gsj.jp/seamless/v2/api/", expected_format="Vector/API export", expected_size="UNKNOWN_UNTIL_SUBSET",
                    licence="GSJ/AIST terms", temporal_coverage="Version to freeze at download", spatial_coverage="Noto interpretation range",
                    resolution="1:200,000", target_destination=f"{base}/SRC_GEOLOGY_GSJ_SEAMLESS/", original_filename="TBD_SUBSET_EXPORT",
                    checksum_algorithm="SHA256", checksum="", download_status="PLANNED_SELECTION_GATE", verification_status="VERSION_AND_EXPORT_ROUTE_TO_BIND",
                    notes="Defines the volcanic subdomain independently of Y; no H claim"))
    return out


QUEUE = [
    (1, "KUMAMOTO_ASO_2016", "ACQUIRE", "Unchanged Stage-2A-extension order"),
    (2, "HOKKAIDO_IBURI_2018", "ACQUIRE", "Unchanged Stage-2A-extension order"),
    (3, "KURIKOMA_2008", "ACQUIRE_AFTER_PROVENANCE_AUTHORISATION", "V1 historical evidence; never an unseen V2 test"),
    (4, "NOTO_PENINSULA_2024", "ACQUIRE", "New READY_CORE; ~0.6 MB Y/Omega plus gated DEM/geology"),
    (5, "PNG_PORGERA_2018", "RESOLVE_OMEGA", "Ask the authors/USGS how mapping_extent was drawn and for cloud gaps; Y already open"),
    (6, "GUATEMALA_1976", "RESOLVE_OMEGA_AND_M", "Digitise the PP1204-A mapped-area boundary; write and review an explicit post-event DEM leakage justification"),
    (7, "TALAKMAU_PASAMAN_2022", "RESOLVE_Y", "Validate the preliminary UNOSAT extent against an independent high-resolution inventory"),
    (8, "KIVU_RIFT_NTK_2000_2019", "RESOLVE_OMEGA", "Request imagery-availability/search mask from the authors; quantify the volcanic subset by geology"),
    (9, "BANKS_PENINSULA_2021", "REQUEST_DATA", "Existing BLOCKED: custodian request (Y/Omega/M/H already PASS on paper)"),
    (10, "AUCKLAND_WAITAKERE_2023", "REQUEST_DATA", "Auckland Council catalogue and mapping support"),
    (11, "CHRISTCHURCH_PORT_HILLS_2011", "REQUEST_DATA", "GNS source-area data and survey support"),
    (12, "MIRAVALLES_OTTO_2016", "REQUEST_DATA", "Author GIS request"),
    (13, "KIVU_KALEHE_RIFT_FLANK", "REQUEST_DATA", "Author GIS request; overlaps KIVU_RIFT_NTK_2000_2019"),
    (14, "COROMANDEL_GABRIELLE_2023", "DECIDE_REGISTRATION", "Existing BLOCKED: registered-user access needs explicit authorisation"),
]


def _note(row: dict, sidx: dict) -> str:
    links = []
    for sid in filter(None, row["stage2a2_source_ids"].split("|")):
        s = sidx[sid]
        links.append(f"- `{sid}` — [{s['source_title']}]({s['url']})" if s["url"].startswith("http") else f"- `{sid}` — {s['source_title']}")
    return "\n".join([f"# {row['landscape_name']} — Stage 2A.2", "", f"Origin: `{row['discovery_origin']}`. Decision: **{row['stage2a2_readiness']}** "
                      f"(fast-fail: `{row['stage2a2_fast_fail_step']}`).", "",
                      f"- Y `{row['y_gate']}` / `{row['inventory_tier']}`: {row['stage2a2_y_resolution']}",
                      f"- Omega `{row['omega_gate']}`: {row['stage2a2_omega_resolution']}",
                      f"- Volcanic terrain `{row['stage2a2_volcanic_intersection']}`: {row['stage2a2_volcanic_evidence']}",
                      f"- M `{row['m_gate']}` / leakage `{row['stage2a2_m_leakage_risk']}`: {row['stage2a2_m_resolution']}",
                      f"- H `{row['h_status']}`: {row['stage2a2_h_resolution']}", "", f"**Exact remaining blocker:** {row['stage2a2_blocker']}", "",
                      "## Sources", "", *links, "", "No experimental/lockbox role or future ML field was assigned.", ""])


def _report(c, newly, ready, datasets, n_new) -> str:
    ready_h = [r for r in ready if r["stage2a2_readiness"] == "READY_CORE_H"]
    screened = [r for r in c if r["stage2a2_screened"] == "TRUE"]
    volc = sum(d["volcanic_intersection"] in ("YES", "PARTIAL") for d in datasets)
    t = ["# Science Fair ML V2 — Stage 2A.2: dataset-first global discovery", "", "## Executive decision", "",
         f"{len(datasets)} landslide datasets were screened dataset-first across {len(SEARCH_LOG)} repository/collection routes. "
         f"{volc} intersect volcanic terrain. {n_new} new candidates were created. Exactly one new landscape reached READY: "
         f"`{'`, `'.join(r['landscape_id'] for r in newly)}` (READY_CORE). No new READY_CORE_H was found.", "",
         f"READY pool: **{len(ready)}** ({len(ready_h)} READY_CORE_H). Continents: {', '.join(sorted({r['continent'] for r in ready}))}. "
         "The target of ~15 defensible landscapes is **not** reachable from open data alone; the remaining path runs through custodian data requests "
         "and Omega/M resolution, not through more searching.", "", "## Stage-2A.2 decisions", "",
         "| Landscape | Origin | Decision | Fast-fail | Y / tier | Omega | Volcanic | M / leakage | H |", "|---|---|---|---|---|---|---|---|---|"]
    t += [f"| `{r['landscape_id']}` | {r['discovery_origin']} | **{r['stage2a2_readiness']}** | {r['stage2a2_fast_fail_step']} | {r['y_gate']} / {r['inventory_tier']} | "
          f"{r['omega_gate']} | {r['stage2a2_volcanic_intersection']} | {r['m_gate']} / {r['stage2a2_m_leakage_risk']} | {r['h_status']} |" for r in screened]
    t += ["", "## Why most datasets stopped", "",
          "- **Omega (step B)** is the binding constraint. Rapid-mapping products (UNOSAT, CEMS) often ship an explicit analysis extent, but their Y is preliminary, "
          "10 m, or dissolved. Research inventories with strong Y (PNG 2018, Kivu) ship envelopes or no search mask.",
          "- **Access** blocks most non-Asian volcanic inventories: Auckland, Port Hills, Kalehe, Miravalles, Elgon, Ethiopia and the Azores are request-only or paper-only.",
          "- **Volcanic terrain (step C)** removed several open, well-documented inventories (Concepción, Karongi, Limón 1991, Aisén 2007, Haiti 2021).",
          "- Global compilations (UGLC, Sen12Landslides, HR-GLDD, COOLR) are derivative and do not preserve mapping support.", "",
          "## Noto 2024 evidence", "",
          "GSI distributes, per district, the landslide polygons, the 判読範囲 interpretation-range polygon and cloud masks. In a seeded 200-point GSJ Seamless "
          f"sample inside the range, {NOTO_GEOLOGY_SAMPLE['counts']['VOLCANIC']} of 158 onshore points ({NOTO_GEOLOGY_SAMPLE['volcanic_share_of_onshore_samples']:.0%}) "
          "fall on Oligo-Miocene andesite/dacite lavas and pyroclastics. The volcanic subdomain must be defined from geology before sampling. "
          "H is not claimed. Noto is the fourth Japanese READY landscape, which increases rather than reduces the Japan concentration.", "",
          "## KURIKOMA_2008", "", "Kurikoma remains READY_CORE_H as historical V1 evidence. `v2_unseen_test_eligibility` marks it, and every KNOWN_V1 landscape, "
          "as ineligible for any unseen V2 final test.", "", "## Recommended next step", "",
          "Broad Stage 2B is **not** authorised: the READY pool spans only Asia and North America. Bounded acquisition can extend to Noto (4th in queue). "
          "Material expansion now requires an access-request phase (Banks Peninsula, Auckland Council, GNS Port Hills, Kalehe/Miravalles authors, PNG extent "
          "clarification) and a decision on registered-user access (Coromandel/Gabrielle). That phase needs explicit user authorisation.", "",
          "No training, bulk download, package installation, lockbox assignment, commit or push occurred.", ""]
    return "\n".join(t)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    before = {rel: sha(ROOT / rel) for rel in PRIOR}
    candidates, sources = build_candidates(), build_sources()
    inventories, acq = build_inventories(candidates), acquisition_rows()
    sidx = {s["source_id"]: s for s in sources}
    datasets = [dict(zip(DATASET_COLUMNS, d + (DATE,))) for d in DATASETS]
    search = [dict(zip(SEARCH_COLUMNS, s + (DATE,))) for s in SEARCH_LOG]

    write_csv(META / "stage2a2_landscape_candidates.csv", candidates, S2A2_COLUMNS)
    write_csv(META / "stage2a2_source_manifest.csv", sources, SOURCE_COLUMNS)
    write_csv(META / "stage2a2_inventory_manifest.csv", inventories, INV_COLUMNS)
    screened = [r for r in candidates if r["stage2a2_screened"] == "TRUE"]
    new = [r for r in candidates if r["discovery_origin"] == ORIGIN]
    ready = sorted((r for r in candidates if r["stage2a2_readiness"] in READY), key=lambda r: r["landscape_id"])
    newly = [r for r in screened if r["stage2a2_readiness"] in READY and r["stage2a_extension_readiness"] not in READY]

    write_csv(OUT / "new_candidates.csv", new, ["landscape_id", "landscape_name", "country", "continent", "tectonic_setting", "volcano_type", "climate_class",
                                               "inventory_trigger", "inventory_tier", "data_accessibility", "screening_score", "priority_class", "stage2a2_readiness"])
    write_csv(OUT / "dataset_first_search_log.csv", search, SEARCH_COLUMNS)
    write_csv(OUT / "fast_fail_table.csv", datasets, DATASET_COLUMNS)
    write_csv(OUT / "gate_table.csv", screened, ["landscape_id", "discovery_origin", "y_gate", "omega_gate", "m_gate", "h_status", "inventory_tier",
                                                 "stage2a2_volcanic_intersection", "stage2a2_m_leakage_risk", "stage2a2_readiness"])
    write_csv(OUT / "acquisition_readiness.csv", candidates, ["landscape_id", "discovery_origin", "country", "continent", "tectonic_setting", "volcano_type",
                                                            "y_gate", "omega_gate", "m_gate", "h_status", "stage2a2_readiness", "v2_unseen_test_eligibility"])
    write_csv(OUT / "h_capability_audit.csv", [r for r in candidates if r["stage2a2_readiness"] in READY or r["stage2a2_screened"] == "TRUE"],
              ["landscape_id", "stage2a2_readiness", "h_status", "h_quality", "h_method_candidate", "stage2a2_h_resolution"])
    write_csv(OUT / "unresolved_candidates.csv", [r for r in screened if r["stage2a2_readiness"] in ("PARTIAL", "BLOCKED")],
              ["landscape_id", "stage2a2_readiness", "stage2a2_fast_fail_step", "stage2a2_blocker"])
    write_csv(OUT / "rejected_candidates.csv", [r for r in screened if r["stage2a2_readiness"] == "REJECT"],
              ["landscape_id", "stage2a2_fast_fail_step", "y_gate", "omega_gate", "stage2a2_blocker"])
    write_csv(OUT / "acquisition_queue.csv", [dict(order=o, landscape_id=l, action=a, notes=n) for o, l, a, n in QUEUE],
              ["order", "landscape_id", "action", "notes"])
    write_csv(OUT / "noto_acquisition_manifest.csv", acq, ACQ_COLUMNS)
    canon(OUT / "noto_geology_sample.json", NOTO_GEOLOGY_SAMPLE)
    canon(OUT / "transient_inspection_manifest.json", [dict(url=u, bytes=b, sha256=h, landscape_id=l, purpose=p, retained_in_repository=False)
                                                       for u, b, h, l, p in TRANSIENT_INSPECTIONS])
    for row in screened:
        (NOTES / f"{row['landscape_id']}.md").write_text(_note(row, sidx))
    dims = ["continent", "tectonic_setting", "volcano_type", "climate_class", "erosional_state", "inventory_tier", "h_status"]
    diversity = {"ready_count": len(ready), "ready_h_count": sum(r["stage2a2_readiness"] == "READY_CORE_H" for r in ready),
                 "ready": [r["landscape_id"] for r in ready], "countries": dict(Counter(r["country"] for r in ready)),
                 **{f"ready_by_{d}": dict(Counter(r[d] for r in ready)) for d in dims},
                 "screened_by_continent": dict(Counter(r["continent"] for r in screened)),
                 "remaining_gaps": ["Africa", "South America", "Europe", "Oceania", "continental rift", "intraplate hotspot", "arid/high-altitude",
                                    "non-Asian H-capable landscape beyond Atitlan"], "japan_concentration": sum(r["country"] == "Japan" for r in ready),
                 "broad_acquisition_ready": False}
    canon(OUT / "global_diversity_audit.json", diversity)
    (OUT / "READY_POOL_DIVERSITY.md").write_text("\n".join(
        ["# READY pool after Stage 2A.2", "", "| Landscape | Country | Continent | Tectonic setting | Volcano type | Tier | H | Unseen-test eligibility |",
         "|---|---|---|---|---|---|---|---|"] +
        [f"| {r['landscape_id']} | {r['country']} | {r['continent']} | {r['tectonic_setting']} | {r['volcano_type']} | {r['inventory_tier']} | {r['h_status']} | "
         f"{r['v2_unseen_test_eligibility'].split(':')[0]} |" for r in ready]) + "\n")
    (OUT / "STAGE2A2_REPORT.md").write_text(_report(candidates, newly, ready, datasets, len(new)))
    downloads = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p), "role": "bounded metadata/method/landing-page evidence"}
                 for p in sorted((ROOT / "data/metadata/stage2a2").glob("*"))]
    canon(OUT / "download_manifest.json", downloads)
    from .validate_stage2a2 import validate
    validation = validate(candidates, sources, inventories, datasets, acq, before)
    canon(OUT / "validation.json", validation)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    after = {rel: sha(ROOT / rel) for rel in PRIOR}
    generated = sorted(p for p in list(OUT.glob("*")) + list(NOTES.glob("*.md")) +
                       [META / "stage2a2_landscape_candidates.csv", META / "stage2a2_source_manifest.csv", META / "stage2a2_inventory_manifest.csv"]
                       if p.is_file() and not p.name.startswith("STAGE2A2_CHECKPOINT"))
    cp = {"schema": "SFV2_STAGE2A2_CHECKPOINT_V1", "date": DATE, "datasets_examined": len(datasets), "search_routes": len(SEARCH_LOG),
          "volcanic_intersections": sum(d["volcanic_intersection"] in ("YES", "PARTIAL") for d in datasets), "new_candidates": len(new),
          "screened": len(screened), "newly_ready": [r["landscape_id"] for r in newly], "promoted_ready_core": sum(r["stage2a2_readiness"] == "READY_CORE" for r in newly),
          "promoted_ready_core_h": sum(r["stage2a2_readiness"] == "READY_CORE_H" for r in newly), "total_ready": len(ready),
          "total_ready_h": diversity["ready_h_count"], "broad_stage2b_ready": False, "discovery_productive": "DIMINISHING_RETURNS",
          "limited_acquisition_order": [q[1] for q in QUEUE if q[2].startswith("ACQUIRE")], "prior_hashes": before, "prior_preserved": before == after,
          "downloads": downloads, "output_hashes": {str(p.relative_to(ROOT)): sha(p) for p in generated}, "training": False, "bulk_download": False,
          "lockbox_assignment": False}
    canon(OUT / "STAGE2A2_CHECKPOINT.json", cp)
    (OUT / "STAGE2A2_CHECKPOINT.md").write_text(
        f"# Stage 2A.2 checkpoint\n\n- Datasets examined: {cp['datasets_examined']}\n- New candidates: {cp['new_candidates']}\n"
        f"- Newly READY: {', '.join(cp['newly_ready'])}\n- Total READY: {cp['total_ready']} (READY_CORE_H {cp['total_ready_h']})\n"
        f"- Broad Stage 2B ready: false\n- Prior checkpoints preserved: {str(cp['prior_preserved']).lower()}\n")
    return cp


if __name__ == "__main__":
    main()
