"""Build the deterministic Stage-2A evidence-screening checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .schema import COLUMNS, FUTURE_FIELDS, INVENTORY_COLUMNS, ROOT, SOURCE_COLUMNS
from .stage2a_records import DATE, NEW_SOURCES, R, WAVE1, WAVE2

META = ROOT / "metadata"
OUT = ROOT / "outputs" / "stage2a"
NOTES = ROOT / "screening" / "evidence" / "stage2a"
STAGE1 = ROOT / "outputs" / "screening"

STAGE2_COLUMNS = COLUMNS + [
    "stage2a_screened", "stage2a_wave", "stage2a_readiness", "stage2a_y_summary",
    "stage2a_omega_summary", "stage2a_m_summary", "stage2a_e_summary", "stage2a_g_summary",
    "stage2a_h_summary", "stage2a_unresolved", "stage2a_source_ids",
    "stage2a_estimated_burden", "stage2a_target_tier_note", "stage2a_verified_date",
]
STAGE2_SOURCE_COLUMNS = SOURCE_COLUMNS + ["authors", "publication_year", "access_route", "stage2a_verified", "metadata_file", "metadata_sha256"]
STAGE2_INVENTORY_COLUMNS = INVENTORY_COLUMNS + ["stage2a_role", "scientific_usability", "mapping_bias"]
READINESS = {"READY_CORE", "READY_CORE_H", "PARTIAL", "BLOCKED", "REJECT", "NOT_SCREENED"}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)

def canonical(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_candidates() -> list[dict]:
    rows = read_csv(META / "landscape_candidates.csv")
    wave1, wave2 = set(WAVE1), set(WAVE2)
    for row in rows:
        identifier = row["landscape_id"]
        rec = R.get(identifier)
        if rec:
            row.update(y_gate=rec["y_gate"], omega_gate=rec["omega_gate"], m_gate=rec["m_gate"], h_status=rec["h_status"])
            row.update(
                stage2a_screened="TRUE", stage2a_wave="WAVE_1" if identifier in wave1 else "WAVE_2",
                stage2a_readiness=rec["readiness"], stage2a_y_summary=rec["y_summary"],
                stage2a_omega_summary=rec["omega_summary"], stage2a_m_summary=rec["m_summary"],
                stage2a_e_summary=rec["e_summary"], stage2a_g_summary=rec["g_summary"],
                stage2a_h_summary=rec["h_summary"], stage2a_unresolved=rec["unresolved"],
                stage2a_source_ids="|".join(rec["source_ids"]), stage2a_estimated_burden=rec["estimated_burden"],
                stage2a_target_tier_note=rec["target_tier_note"], stage2a_verified_date=DATE,
            )
        else:
            row.update(stage2a_screened="FALSE", stage2a_wave="NOT_SCREENED", stage2a_readiness="NOT_SCREENED",
                       stage2a_y_summary="", stage2a_omega_summary="", stage2a_m_summary="", stage2a_e_summary="",
                       stage2a_g_summary="", stage2a_h_summary="", stage2a_unresolved="",
                       stage2a_source_ids="", stage2a_estimated_burden="", stage2a_target_tier_note="", stage2a_verified_date="")
    return rows

def _new_source_rows() -> list[dict]:
    rows = []
    years = {"S2_BANKS_PAPER":"2023","S2_GLOBAL_EQ_DB_META":"2026","S2_HOKKAIDO_GSI":"2018",
             "S2_KUMAMOTO_GSI":"2016","S2_ASO_GEOLOGY":"1985","S2_PALI":"2022","S2_WILSON_ZENODO":"2026"}
    downloadable = {"S2_BANKS_PAPER":"FALSE","S2_GLOBAL_EQ_DB_META":"FALSE","S2_SLIDO":"TRUE","S2_RI45":"TRUE",
                    "S2_EL_SALVADOR_USGS":"TRUE","S2_HOKKAIDO_GSI":"TRUE","S2_GSI_HISTORICAL_DEM":"TRUE",
                    "S2_KUMAMOTO_GSI":"TRUE","S2_ASO_GEOLOGY":"TRUE","S2_IFFI":"TRUE","S2_PALI":"TRUE","S2_WILSON_ZENODO":"TRUE"}
    licences = {"S2_PALI":"CC BY 4.0","S2_WILSON_ZENODO":"CC BY 4.0"}
    for source_id, landscape, role, title, org, url, doi, kind, note in NEW_SOURCES:
        metadata_file = ""
        metadata_hash = ""
        if source_id == "S2_GLOBAL_EQ_DB_META":
            metadata_file = "data/metadata/stage2a/zenodo_20157141.json"; metadata_hash = "5e8405e8ab05761248daca5a7d96885b059015475a15721c13effd5d3f687acf"
        if source_id == "S2_WILSON_ZENODO":
            metadata_file = "data/metadata/stage2a/zenodo_22105158.json"; metadata_hash = "79abe916c7c1f8762c7abf6b6cd83b95ffab5ac018bbfb3bfda48e06c8a35f59"
        rows.append(dict(source_id=source_id, landscape_id=landscape, data_role=role, source_title=title,
                         source_organisation=org, source_type=kind, citation=title, doi=doi, url=url,
                         access_date=DATE, downloadable=downloadable.get(source_id,"UNKNOWN"), file_format="UNKNOWN", licence=licences.get(source_id,"UNKNOWN"),
                         spatial_coverage="See evidence note", temporal_coverage="See evidence note",
                         resolution_or_scale="UNKNOWN", tier_relevance="See evidence note", omega_obs_evidence="See evidence note",
                         h_relevance="See evidence note", verified="VERIFIED", notes=note, authors="UNKNOWN",
                         publication_year=years.get(source_id,"UNKNOWN"), access_route=url, stage2a_verified="TRUE",
                         metadata_file=metadata_file, metadata_sha256=metadata_hash))
    return rows

def build_sources() -> list[dict]:
    inherited = read_csv(META / "source_manifest.csv")
    for row in inherited:
        row.update(authors="UNKNOWN", publication_year="UNKNOWN", access_route=row["url"],
                   stage2a_verified="FALSE", metadata_file="", metadata_sha256="")
    return inherited + _new_source_rows()

def build_inventories() -> list[dict]:
    rows = read_csv(META / "inventory_manifest.csv")
    for row in rows:
        rec = R.get(row["landscape_id"])
        row.update(stage2a_role="PRIMARY_Y_CANDIDATE" if rec else "NOT_SCREENED",
                   scientific_usability=rec["readiness"] if rec else "NOT_SCREENED",
                   mapping_bias=rec["unresolved"] if rec else "NOT_SCREENED")
    additions = [
        dict(inventory_id="INV_HOKKAIDO_IBURI_2018_GSI_OFFICIAL", landscape_id="HOKKAIDO_IBURI_2018", inventory_name="GSI slope-failure/deposition distribution", inventory_source_id="S2_HOKKAIDO_GSI", inventory_year="2018", inventory_trigger="EARTHQUAKE", inventory_mapping_method="Aerial-photograph interpretation", geometry_type="MIXED", target_tier="TIER_2", feature_count_estimate="UNKNOWN", mapped_area="Official interpretation range", observation_domain_source_id="S2_HOKKAIDO_GSI", observation_domain_establishable="TRUE", verification_status="VERIFIED", downloadable="TRUE", access_route="https://www.gsi.go.jp/BOUSAI/H30-hokkaidoiburi-east-earthquake-index.html", licence="Government open data terms; confirm at acquisition", notes="Official event product; combined failure/deposition semantics retained", stage2a_role="PRIMARY_Y_CANDIDATE", scientific_usability="READY_CORE", mapping_bias="Aerial-photo detectability and combined geometry"),
        dict(inventory_id="INV_KUMAMOTO_ASO_2016_GSI_OFFICIAL", landscape_id="KUMAMOTO_ASO_2016", inventory_name="GSI Kumamoto earthquake landslide distribution", inventory_source_id="S2_KUMAMOTO_GSI", inventory_year="2016", inventory_trigger="EARTHQUAKE", inventory_mapping_method="Aerial-photograph interpretation", geometry_type="POINT", target_tier="TIER_2", feature_count_estimate="UNKNOWN", mapped_area="Official interpretation range", observation_domain_source_id="S2_KUMAMOTO_GSI", observation_domain_establishable="TRUE", verification_status="VERIFIED", downloadable="TRUE", access_route="https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html", licence="Government open data terms; confirm at acquisition", notes="Official event product; points represent centres of mapped major/minor failed areas, not source polygons", stage2a_role="PRIMARY_Y_CANDIDATE", scientific_usability="READY_CORE_H", mapping_bias="Aerial-photo detectability and approximately 0.1 ha lower mapped-size class"),
        dict(inventory_id="INV_PATAGONIA_PALI_MODEL_PREDICTIONS", landscape_id="PATAGONIA_PALI", inventory_name="PALI model predictions", inventory_source_id="S2_PALI", inventory_year="CUMULATIVE", inventory_trigger="MIXED", inventory_mapping_method="Machine-learning predictions", geometry_type="POLYGON", target_tier="UNCLEAR", feature_count_estimate="UNKNOWN", mapped_area="PALI study area", observation_domain_source_id="S2_PALI", observation_domain_establishable="TRUE", verification_status="VERIFIED", downloadable="TRUE", access_route="https://doi.org/10.5281/zenodo.7057656", licence="CC BY 4.0", notes="Separated explicitly from manual labels", stage2a_role="DIAGNOSTIC_ONLY_NOT_Y", scientific_usability="REJECT_AS_Y", mapping_bias="Model outputs cannot be ground-truth Y"),
        dict(inventory_id="INV_OREGON_SLIDO_HISTORIC_POINTS", landscape_id="OREGON_CASCADES_SLIDO", inventory_name="SLIDO historic landslide points", inventory_source_id="S2_SLIDO", inventory_year="CUMULATIVE", inventory_trigger="MIXED", inventory_mapping_method="Historical compilation", geometry_type="POINT", target_tier="TIER_1", feature_count_estimate="UNKNOWN", mapped_area="Oregon mapped projects", observation_domain_source_id="S2_SLIDO", observation_domain_establishable="UNCERTAIN", verification_status="VERIFIED", downloadable="TRUE", access_route="https://www.oregongeology.org/slido/", licence="Government open data terms; confirm", notes="Must remain separate from LiDAR morphology polygons", stage2a_role="ALTERNATE_Y_CANDIDATE", scientific_usability="PARTIAL", mapping_bias="Heterogeneous historic reporting and coverage"),
    ]
    return rows + additions

def _gate_changes(stage1: list[dict], stage2: list[dict]) -> list[dict]:
    old = {row["landscape_id"]: row for row in stage1}
    out = []
    for row in stage2:
        if row["stage2a_screened"] != "TRUE": continue
        prior = old[row["landscape_id"]]
        for gate in ("y_gate", "omega_gate", "m_gate", "h_status"):
            if row[gate] != prior[gate]:
                out.append(dict(landscape_id=row["landscape_id"], field=gate, stage1=prior[gate], stage2a=row[gate], evidence=row[f"stage2a_{'h' if gate == 'h_status' else gate.replace('_gate','')}_summary"]))
    return out

def _note(row: dict, source_index: dict) -> str:
    source_lines = []
    for sid in row["stage2a_source_ids"].split("|"):
        if sid:
            source = source_index[sid]
            if source["url"] == "UNKNOWN":
                source_lines.append(f"- `{sid}` — {source['source_title']} (access route unresolved)")
            else:
                source_lines.append(f"- `{sid}` — [{source['source_title']}]({source['url']})")
    return "\n".join([
        f"# {row['landscape_name']} — Stage 2A evidence note", "",
        f"**Landscape ID:** `{row['landscape_id']}`  ", f"**Wave:** `{row['stage2a_wave']}`  ",
        f"**Decision:** `{row['stage2a_readiness']}`", "", "## Verified gate evidence", "",
        f"- **Y ({row['y_gate']}):** {row['stage2a_y_summary']}",
        f"- **Omega ({row['omega_gate']}):** {row['stage2a_omega_summary']}",
        f"- **M ({row['m_gate']}):** {row['stage2a_m_summary']}",
        f"- **E:** {row['stage2a_e_summary']}", f"- **G:** {row['stage2a_g_summary']}",
        f"- **H ({row['h_status']}):** {row['stage2a_h_summary']}", "", "## Unresolved", "",
        row["stage2a_unresolved"], "", "## Traceable sources", "", *source_lines, "",
        "No experimental/lockbox role was assigned and no future ML metric was populated.", "",
    ])

def _summary(rows: list[dict], field: str) -> list[dict]:
    return [{"category": key, "count": value} for key, value in sorted(Counter(r[field] for r in rows).items())]

def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True); NOTES.mkdir(parents=True, exist_ok=True)
    stage1_candidates = read_csv(META / "landscape_candidates.csv")
    stage1_paths = [META / "landscape_candidates.csv", META / "source_manifest.csv", META / "inventory_manifest.csv",
                    STAGE1 / "STAGE_1_SCREENING_CHECKPOINT.json", STAGE1 / "STAGE_1_SCREENING_CHECKPOINT.md"]
    stage1_hashes = {str(path.relative_to(ROOT)): sha(path) for path in stage1_paths}
    candidates, sources, inventories = build_candidates(), build_sources(), build_inventories()
    source_index = {row["source_id"]: row for row in sources}
    write_csv(META / "stage2a_landscape_candidates.csv", candidates, STAGE2_COLUMNS)
    write_csv(META / "stage2a_source_manifest.csv", sources, STAGE2_SOURCE_COLUMNS)
    write_csv(META / "stage2a_inventory_manifest.csv", inventories, STAGE2_INVENTORY_COLUMNS)
    screened = [row for row in candidates if row["stage2a_screened"] == "TRUE"]
    changes = _gate_changes(stage1_candidates, candidates)
    write_csv(OUT / "stage2a_gate_changes.csv", changes, ["landscape_id","field","stage1","stage2a","evidence"])
    readiness_cols = ["landscape_id","landscape_name","stage2a_wave","y_gate","omega_gate","m_gate","h_status","stage2a_readiness","stage2a_unresolved","stage2a_estimated_burden"]
    write_csv(OUT / "stage2a_acquisition_readiness.csv", screened, readiness_cols)
    unresolved = [row for row in screened if row["stage2a_readiness"] in {"PARTIAL","BLOCKED","REJECT"}]
    write_csv(OUT / "stage2a_unresolved_evidence.csv", unresolved, ["landscape_id","stage2a_readiness","stage2a_unresolved"])
    ready = [row for row in screened if row["stage2a_readiness"].startswith("READY_")]
    order = {"KUMAMOTO_ASO_2016":1,"HOKKAIDO_IBURI_2018":2,"LOMBOK_RINJANI_2018":3,"ATITLAN_STAN_2005":4,"PUERTO_RICO_MARIA_2017":5}
    queue = sorted(ready, key=lambda row: order[row["landscape_id"]])
    queue_rows = [dict(acquisition_order=order[r["landscape_id"]], landscape_id=r["landscape_id"], readiness=r["stage2a_readiness"], burden=r["stage2a_estimated_burden"], note="New acquisition" if order[r["landscape_id"]] <= 2 else "Already represented in frozen V1/reference; do not copy V1 automatically") for r in queue]
    write_csv(OUT / "stage2a_ready_acquisition_queue.csv", queue_rows, ["acquisition_order","landscape_id","readiness","burden","note"])
    for row in screened: (NOTES / f"{row['landscape_id']}.md").write_text(_note(row, source_index))
    summaries = {
        "readiness": _summary(screened,"stage2a_readiness"), "target_tier": _summary(screened,"inventory_tier"),
        "dem_temporal": _summary(screened,"m_gate"), "omega": _summary(screened,"omega_gate"),
        "h_capability": _summary(screened,"h_status"), "continent_ready": _summary(ready,"continent"),
        "tectonic_ready": _summary(ready,"tectonic_setting"), "volcano_type_ready": _summary(ready,"volcano_type"),
    }
    for name, data in summaries.items(): canonical(OUT / f"stage2a_{name}_summary.json", data)
    diversity = {
        "ready_count": len(ready), "continents": sorted({r["continent"] for r in ready}),
        "countries": sorted({r["country"] for r in ready}), "tectonic_settings": sorted({r["tectonic_setting"] for r in ready}),
        "volcano_types": sorted({r["volcano_type"] for r in ready}), "climates": sorted({r["climate_class"] for r in ready}),
        "gaps": ["Africa", "South America", "Europe", "continental rift", "intraplate continental", "arid/high-altitude", "open H-capable non-Asian candidates"],
        "wave2_triggered": True, "wave2_reason": "Wave 1 produced only two ready landscapes and an Asia/Japan-heavy pool.",
    }
    canonical(OUT / "stage2a_global_diversity_audit.json", diversity)
    downloads = [{"path":"data/metadata/stage2a/zenodo_20157141.json","bytes":6381,"sha256":"5e8405e8ab05761248daca5a7d96885b059015475a15721c13effd5d3f687acf","role":"Restricted global earthquake-inventory record metadata"},
                 {"path":"data/metadata/stage2a/zenodo_22105158.json","bytes":5411,"sha256":"79abe916c7c1f8762c7abf6b6cd83b95ffab5ac018bbfb3bfda48e06c8a35f59","role":"Wilsons River inventory record metadata"}]
    canonical(OUT / "stage2a_download_manifest.json", downloads)
    report = ["# Science Fair ML V2 — Stage 2A deep evidence screening", "",
              "## Executive decision", "",
              "Stage 2A deeply screened 31 landscapes in two waves. Five meet the data gates; only Kumamoto/Aso and Hokkaido require new Stage-2B acquisition. The ready pool remains too small and geographically narrow for the intended 15–30-landscape experiment, so broad acquisition is not yet authorised by this checkpoint.", "",
              "## Readiness", "", "| Status | Count |", "| --- | ---: |"]
    for item in summaries["readiness"]: report.append(f"| {item['category']} | {item['count']} |")
    report += ["", "Ready landscapes: " + ", ".join(f"`{r['landscape_id']}`" for r in queue), "",
               "Strongest partials: `WILSONS_RIVER_NSW_2022`, `CAPELLADES_TURRIALBA_2016`, `CHUUK_CHATAAN_2002`, `EL_SALVADOR_2001_JAN`, `OREGON_CASCADES_SLIDO`, `COLUMBIA_RIVER_GORGE_BASALT`, `MATZACO_PUEBLA_2017`.", "",
               "## Gate findings", "", f"Gate changes from Stage 1: {len(changes)}. M is PASS for {sum(r['m_gate']=='PASS' for r in screened)} screened landscapes; Omega is PASS for {sum(r['omega_gate']=='PASS' for r in screened)}; Y is PASS for {sum(r['y_gate']=='PASS' for r in screened)}; H is PASS for {sum(r['h_status']=='PASS' for r in screened)}.", "",
               "Transitions: Y has 7 UNCERTAIN→PASS, 2 PASS→UNCERTAIN and 1 UNCERTAIN→FAIL; Omega has 5 UNCERTAIN→PASS and 3 UNCERTAIN→FAIL; M has 13 UNCERTAIN→PASS; H has 8 UNCERTAIN→PASS and 4 UNCERTAIN→FAIL.", "",
               "A dataset release date was never substituted for acquisition epoch. NASADEM is treated as February 2000 terrain. Japanese candidates require tile-level binding to a historical pre-event GSI DEM edition during acquisition. Post-event or same-lineage morphology remains unresolved where it could encode Y.", "",
               "Omega was accepted only from an independent mapping/interpretation extent or a previously frozen support record. Catalogue bounds, positive bounding boxes and positive convex hulls were not accepted.", "",
               "## Environment, geology and H", "", "CHELSA/WorldClim provide a consistent global climatology path; event rainfall or shaking remains event-specific. Geology must be harmonised after acquisition. H was not forced: Puerto Rico and Hokkaido are core-only, while Kumamoto/Aso, Lombok and Atitlan have credible bounded H paths.", "",
               "The source manifest grew from 70 to 91 rows. The inventory child manifest grew from 49 to 53 rows, including official Hokkaido and Kumamoto products, a separately typed PALI prediction product that is prohibited as Y, and separate SLIDO historic points.", "",
               "## Diversity and next action", "", "Wave 2 was required because Wave 1 yielded only two ready Japanese landscapes. It added three previously verified core landscapes, but the five-member pool still lacks Africa, South America, Europe, continental-rift, arid/high-altitude and open non-Asian H-capable representation.", "",
               "The exact new-acquisition order is (1) Kumamoto/Aso and (2) Hokkaido Iburi. Lombok, Atitlan and Puerto Rico follow only after experimental-governance review because their evidence is already represented in frozen V1/reference records; do not copy the frozen V1 tree. A further evidence wave should resolve the high-value partials before broad Stage 2B acquisition. Do not train models.", "",
               "## Governance", "", "All experimental and lockbox roles remain UNASSIGNED. Future ML metrics remain empty. No bulk raster/vector data was downloaded; only two small Zenodo metadata JSON records were saved (11,792 bytes total). No package was installed and no model was trained.", ""]
    (OUT / "STAGE2A_DEEP_EVIDENCE_SCREENING.md").write_text("\n".join(report))
    from .validate_stage2a import validate
    validation = validate(candidates, sources, inventories, stage1_hashes)
    canonical(OUT / "stage2a_validation.json", validation)
    if not validation["valid"]: raise ValueError("Stage2A validation failed: " + "; ".join(validation["errors"]))
    outputs = [META / "stage2a_landscape_candidates.csv", META / "stage2a_source_manifest.csv", META / "stage2a_inventory_manifest.csv"] + sorted(OUT.glob("*")) + sorted(NOTES.glob("*.md"))
    checkpoint = {"schema":"SFV2_STAGE2A_CHECKPOINT_V1","created":DATE,"deeply_screened":len(screened),
                  "wave1":len(WAVE1),"wave2":len(WAVE2),"readiness_counts":dict(Counter(r["stage2a_readiness"] for r in screened)),
                  "stage1_hashes":stage1_hashes,"stage1_preserved":all(sha(ROOT / rel)==digest for rel,digest in stage1_hashes.items()),
                  "manifest_growth":{"sources":{"stage1":70,"stage2a":len(sources)},"inventories":{"stage1":49,"stage2a":len(inventories)}},
                  "downloads":downloads,
                  "output_hashes":{str(path.relative_to(ROOT)):sha(path) for path in outputs if path.is_file() and path.name != "STAGE2A_CHECKPOINT.json"},
                  "decision":"NOT_READY_FOR_BROAD_STAGE2B_ACQUISITION; TWO_NEW_LANDSCAPES_INDIVIDUALLY_READY", "model_training":False,"bulk_acquisition":False,"lockbox_assigned":False}
    canonical(OUT / "STAGE2A_CHECKPOINT.json", checkpoint)
    (OUT / "STAGE2A_CHECKPOINT.md").write_text("# Stage 2A checkpoint\n\n- Deeply screened: 31\n- READY_CORE: 2\n- READY_CORE_H: 3\n- Stage 1 preserved: true\n- Broad Stage 2B acquisition: not ready\n- Individually ready new acquisitions: Kumamoto/Aso, then Hokkaido Iburi\n")
    return checkpoint

if __name__ == "__main__": main()
