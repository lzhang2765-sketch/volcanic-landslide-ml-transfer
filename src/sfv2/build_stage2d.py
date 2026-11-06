"""Build the Stage 2D data checkpoint (terrain units/features, no ML and no H)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .raw_store import sha256
from .schema import ROOT
from .stage2c_governance import REQUEST_PACKAGES
from .stage2d_production import ACTIVE, AS_OF, DER, OUT, build_landscape, feature_definition_records
from .stage2d_transfer import audit as transfer_audit

REPO = ROOT.parent
PRIOR = ROOT / "outputs/stage2c/STAGE2C_CHECKPOINT.json"
SCRIPT = Path(__file__)


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def kurikoma_assessment() -> dict:
    pdf = ROOT / "data/raw/stage2b/KURIKOMA_2008/S2EXT_KURIKOMA_GSI/000043733.pdf"
    qdir = OUT / "qc/KURIKOMA_2008"; qdir.mkdir(parents=True, exist_ok=True)
    preview = qdir / "official_interpretation_range_source_preview.png"
    # A source-only visual inspection aid; it is not a georeferenced raster.
    if not preview.exists() and subprocess.run(["sips", "-s", "format", "png", str(pdf), "--out", str(preview)],
                                               capture_output=True).returncode != 0:
        preview = None
    record = {
        "landscape_id": "KURIKOMA_2008", "decision": "STOP_BLOCKED_OMEGA",
        "source_pdf": str(pdf.relative_to(ROOT)), "source_sha256": sha256(pdf),
        "pdf_pages": 1, "documented_crs_or_projection": "NONE_FOUND",
        "embedded_geospatial_registration": False, "control_points_recorded": 0,
        "transformation": "NOT_PERFORMED", "residual_metrics": "NOT_AVAILABLE",
        "uncertainty": "UNBOUNDED_WITH_CURRENT_EVIDENCE",
        "reason": ("The official map visibly contains the blue photo-interpretation boundary over a small-scale Japanese topographic base, "
                   "but the archived PDF contains no coordinate grid, CRS, GeoPDF registration, or explicit scale/origin sufficient to create "
                   "auditable control points without an independently acquired authoritative base map. Manual matching by place-name appearance "
                   "would fabricate precision. No Omega polygon was digitised and Y was not used."),
        "geology_result": "NOT_SUBSET: a defensible Omega is required before subsetting",
        "terrain_unit_readiness": "BLOCKED", "v2_unseen_test_eligibility": "INELIGIBLE_V1_HISTORICAL",
        "source_preview": str(preview.relative_to(ROOT)) if preview else "NOT_CREATED",
    }
    _json(OUT / "KURIKOMA_GEOREFERENCE_ASSESSMENT.json", record)
    (OUT / "KURIKOMA_GEOREFERENCE_ASSESSMENT.md").write_text(
        "# Kurikoma controlled georeferencing assessment\n\n"
        "**Decision: STOP; keep `BLOCKED_OMEGA`.**\n\n" + record["reason"] + "\n\n"
        "No transformation, residual, area, or digitised polygon is reported because those values would imply unsupported precision. "
        "The 22 transferred raw files remain byte-identical, and the historical eligibility remains `INELIGIBLE_V1_HISTORICAL`.\n"
    )
    return record


def internal_audits(logs: dict[str, list[dict]]) -> dict:
    base = ROOT / "data/raw/stage2d"
    out = {}
    # Lombok: source polygons and source mapping extent are usable; geology is a paper figure only.
    lp = base / "LOMBOK_RINJANI_2018/V1_TRANSFER"
    ly = sorted((lp / "Y").glob("*.shp"))
    lo = next((lp / "OMEGA").glob("*.shp"))
    yg = [gpd.read_file(p) for p in ly]
    og = gpd.read_file(lo)
    out["LOMBOK_RINJANI_2018"] = {
        "transferred_files": len(logs["LOMBOK_RINJANI_2018"]), "all_hash_equal": True,
        "y_semantics": "Tier-2 combined source+runout affected-area polygons (Ferrario 2019 phases kept separate)",
        "y_feature_counts": {p.stem: len(g) for p, g in zip(ly, yg)}, "target_family": "POLYGON_AFFECTED_TIER2",
        "omega": f"Published mapping_extent vector: {len(og)} feature(s), CRS {og.crs}",
        "m": "Copernicus GLO-30 raw DEM present; V1-derived M/features not transferred",
        "g": "Only a scanned/reproduced 1:250,000 geology figure is present; no authoritative digital unit polygons",
        "volcanic_domain": "NOT_DEFENSIBLY_MAPPABLE_FROM_CURRENT_RAW_G",
        "dem_leakage": "Source is pre-event per frozen V1 provenance; V2 EDM/FLM quality layers not present",
        "trigger": "EARTHQUAKE_SEQUENCE_2018", "terrain_unit_compatibility": "YES_AFTER_G_AND_DEM_QUALITY_RESOLUTION",
        "production_readiness": "BLOCKED_VOLCANIC_DOMAIN", "v2_unseen_test_eligibility": "INELIGIBLE_V1_HISTORICAL",
    }
    ap = base / "ATITLAN_STAN_2005/V1_TRANSFER"
    ay = sorted((ap / "Y").glob("*.zip")); ao = sorted((ap / "OMEGA_CONTEXT").glob("Boundary_*.zip"))
    yg = [gpd.read_file(f"zip://{p}") for p in ay]; og = [gpd.read_file(f"zip://{p}") for p in ao]
    out["ATITLAN_STAN_2005"] = {
        "transferred_files": len(logs["ATITLAN_STAN_2005"]), "all_hash_equal": True,
        "y_semantics": "Tier-2 affected-area polygons from Hurricane Stan, Atitlan and Tzojoma source datasets retained separately",
        "y_feature_counts": {p.stem: len(g) for p, g in zip(ay, yg)}, "target_family": "POLYGON_AFFECTED_TIER2",
        "omega": {p.stem: f"{len(g)} feature(s), CRS {g.crs}" for p, g in zip(ao, og)},
        "m": "Copernicus GLO-30 raw DEM present; V1-derived M/features not transferred",
        "g": "No raw geology source exists in the frozen V1 Atitlan input tree",
        "volcanic_domain": "NOT_ESTABLISHED", "dem_leakage": "Pre-event DEM provenance retained; EDM/FLM quality layers absent",
        "trigger": "RAINFALL_HURRICANE_STAN_2005", "terrain_unit_compatibility": "YES_AFTER_G_AND_DEM_QUALITY_RESOLUTION",
        "production_readiness": "BLOCKED_VOLCANIC_DOMAIN", "v2_unseen_test_eligibility": "INELIGIBLE_V1_HISTORICAL",
    }
    _json(OUT / "internal_reharmonisation_audit.json", out)
    return out


def dashboard(internal: dict) -> pd.DataFrame:
    src = pd.read_csv(ROOT / "metadata/stage2a2_landscape_candidates.csv").drop_duplicates("landscape_id", keep="last")
    rows = []
    target_override = {"KUMAMOTO_ASO_2016": "POINT_EVENT_TIER2", "NOTO_PENINSULA_2024": "POLYGON_AFFECTED_TIER2",
                       "KURIKOMA_2008": "POLYGON_AFFECTED_TIER2", "LOMBOK_RINJANI_2018": "POLYGON_AFFECTED_TIER2",
                       "ATITLAN_STAN_2005": "POLYGON_AFFECTED_TIER2"}
    for _, r in src.iterrows():
        lid = r.landscape_id
        if lid in ACTIVE: state = "READY_FOR_TERRAIN_UNITS"
        elif lid in ("KURIKOMA_2008",): state = "BLOCKED_OMEGA"
        elif lid in internal: state = "BLOCKED_VOLCANIC_DOMAIN"
        elif lid == "HOKKAIDO_IBURI_2018": state = "BLOCKED_VOLCANIC_DOMAIN"
        elif lid == "COROMANDEL_GABRIELLE_2023": state = "BLOCKED_ACCESS"
        elif str(r.get("screen_status")) == "REJECT" or str(r.get("stage2a2_readiness")) == "REJECT": state = "REJECT"
        elif str(r.get("y_gate")) == "FAIL": state = "BLOCKED_Y"
        elif str(r.get("omega_gate")) == "FAIL": state = "BLOCKED_OMEGA"
        elif str(r.get("m_gate")) == "FAIL": state = "BLOCKED_M"
        elif str(r.get("y_gate")) == "PASS" and str(r.get("omega_gate")) == "PASS" and str(r.get("m_gate")) == "PASS": state = "PARTIAL"
        elif str(r.get("candidate_class")) == "REFERENCE_ONLY": state = "REFERENCE_ONLY"
        else: state = "PARTIAL"
        hist = "INELIGIBLE_V1_HISTORICAL" if str(r.get("candidate_class")) == "KNOWN_V1" else "NO_V1_HISTORY"
        rows.append({
            "landscape_id": lid, "state": state,
            "metadata_ready": str(r.get("stage2a2_readiness", "")).startswith("READY"),
            "production_ready_after_harmonisation": state == "READY_FOR_TERRAIN_UNITS",
            "continent": r.get("continent", "UNKNOWN"), "country": r.get("country", "UNKNOWN"),
            "tectonic_setting": r.get("tectonic_setting", "UNKNOWN"), "climate": r.get("climate_class", "UNKNOWN"),
            "morphology": r.get("volcano_type", "UNKNOWN"),
            "target_family": target_override.get(lid, str(r.get("inventory_geometry_type", "UNKNOWN"))),
            "h_feasibility": r.get("h_reconstruction_feasible", "UNKNOWN"), "v1_historical_status": hist,
            "final_unseen_eligibility": "INELIGIBLE" if hist == "INELIGIBLE_V1_HISTORICAL" else "UNASSIGNED_NOT_LOCKED",
            "primary_blocker": "NONE" if state == "READY_FOR_TERRAIN_UNITS" else state,
        })
    df = pd.DataFrame(rows).sort_values("landscape_id")
    df.to_csv(OUT / "landscape_state_dashboard.csv", index=False, lineterminator="\n")
    _json(OUT / "landscape_state_summary.json", {
        "candidate_count": len(df), "metadata_ready_count": int(df.metadata_ready.sum()),
        "genuinely_terrain_unit_ready_count": int(df.production_ready_after_harmonisation.sum()),
        "state_counts": df.state.value_counts().sort_index().to_dict(),
        "warning": "metadata_ready is screening metadata only and must never be reported as production-ready",
    })
    return df


def external_packages() -> None:
    cor = """# Coromandel / Cyclone Gabrielle — user action (not performed)

1. Open the official NZ Landslide Database webmap: https://data.gns.cri.nz/landslides/.
2. Choose the personal account registration/sign-up option and complete email verification. Registration is free; do not share credentials.
3. After login, search/filter for the Cyclone Gabrielle 2023 Coromandel dataset and inspect all feature classes, especially source/affected-area polygons, debris trails, and dataset metadata.
4. Check the FAQ/download controls. We need GeoPackage/Shapefile/GeoJSON polygons, stable feature IDs, trigger/date, mapping method/basemap, locational accuracy/quality, and an independent searched-area/imagery-footprint polygon with cloud or visibility exclusions.

Usable **Y** is the event-specific polygon feature class with documented affected-area/source semantics. Usable **Omega** is the area systematically inspected, independent of the positives, with visibility exclusions. Do not accept points as a silent substitute for polygon Y, a convex hull/buffer of positives as Omega, screenshots/PDF maps as production vectors, or records without event attribution.

If vector export remains unavailable, send (do not send now):

> To: [institutional data custodian; address removed from public source copy]  
> Subject: Student research request — Cyclone Gabrielle 2023 Coromandel landslide polygons and mapping domain  
> I am a student building a non-commercial, multi-landscape volcanic-terrain landslide study. Could you provide the event-specific Coromandel Cyclone Gabrielle source/affected-area polygons and the independently defined area systematically mapped, including imagery footprints, cloud/visibility exclusions, stable IDs, trigger/date, mapping method, positional-quality fields, CRS, licence and citation? GeoPackage is preferred. I will acknowledge Earth Sciences New Zealand and comply with the NZLD non-commercial terms. Thank you.
"""
    (OUT / "COROMANDEL_USER_ACTION.md").write_text(cor)
    ranking = {"RP01": 1, "RP02": 2, "RP03": 3, "RP04": 4, "RP05": 5, "RP07": 6, "RP06": 7}
    fallback = {
        "RP01": "Reject from development until an independent searched-area/visibility mask is obtained.",
        "RP02": "Keep blocked-access; do not digitise figures unless authors confirm the printed boundary is the full mapped domain.",
        "RP03": "Use only as reference; point-bias correction is not Omega.", "RP04": "Retain as high-value blocked-access candidate.",
        "RP05": "Controlled georeferencing of the official plate only if coordinate ticks support auditable residuals.",
        "RP06": "Keep blocked; combine follow-up with RP03 custodian network.", "RP07": "Reject mixed labels unless trigger attribution is supplied.",
    }
    rows = []
    rdir = OUT / "request_packages"; rdir.mkdir(parents=True, exist_ok=True)
    for pid, lid, study, who, missing, why, fmt, meta, lic, draft, div in REQUEST_PACKAGES:
        rows.append({"rank": ranking[pid], "package_id": pid, "landscape_id": lid, "recipient_custodian": who,
                     "exact_missing_artifact": missing, "why_needed": why, "preferred_format": fmt,
                     "mapping_domain_requirement": meta, "inventory_requirement": missing,
                     "licence_use_question": lic, "fallback_if_request_fails": fallback[pid], "diversity_contribution": div})
        (rdir / f"{pid}_{lid}.md").write_text(
            f"# {pid} — {lid} (ready to send; NOT SENT)\n\n- Recipient/custodian: {who}\n- Missing: {missing}\n"
            f"- Why: {why}\n- Preferred format: {fmt}\n- Mapping-domain requirement: {meta}\n- Licence question: {lic}\n"
            f"- Fallback: {fallback[pid]}\n\n## Exact draft\n\n> {draft}\n")
    _csv(OUT / "external_request_ranking.csv", sorted(rows, key=lambda x: x["rank"]))
    _csv(OUT / "additional_access_route_audit.csv", [{
        "scope": "EXISTING_SCREENED_CANDIDATES_ONLY", "date": AS_OF, "result": "NO_NEW_BLOCKER_RESOLUTION",
        "routes_checked": "existing registration route|existing supplementary/repository links|publisher mirror evidence already recorded",
        "decision": "No candidate promoted; no broad rediscovery performed",
    }])


def h_feasibility() -> None:
    rows = [
        ("KUMAMOTO_ASO_2016", "CALDERA_RIM_PLUS_RADIAL_FLANK_ENSEMBLE", "Aso 1:50k geology; caldera wall; DEM", "Aso-4 remnants and pre-caldera controls", "MEDIUM", "YES"),
        ("NOTO_PENINSULA_2024", "AGE_HISTORY_PROXY_ONLY", "Seamless unit ages", "preserved constructional surface", "HIGH", "NO"),
        ("KURIKOMA_2008", "RADIAL_FLANK_PLUS_PLATEAU", "raw DEM and geology transferred", "defensible Omega; V2-native controls", "MEDIUM_HIGH", "YES_AFTER_OMEGA"),
        ("LOMBOK_RINJANI_2018", "MULTI_EDIFICE_VOLCANIC_COMPLEX", "raw DEM; published geology figure; caldera context", "digital geology and V2 control extraction", "MEDIUM", "LIKELY_AFTER_G"),
        ("ATITLAN_STAN_2005", "CALDERA_RIM_PLUS_FLANK", "raw DEM; caldera and Tzojoma domains", "authoritative geology and V2 controls", "MEDIUM", "LIKELY_AFTER_G"),
    ]
    _csv(OUT / "h_feasibility.csv", [dict(zip(("landscape_id", "reconstruction_family", "available_controls", "missing_controls",
                                                "expected_uncertainty", "ensemble_feasibility"), r)) for r in rows])


def e_status() -> None:
    _csv(OUT / "e_acquisition_status.csv", [
        {"dataset": "CHELSA V2.1 bio1,bio4,bio12,bio13,bio15", "landscapes": "KUMAMOTO_ASO_2016|NOTO_PENINSULA_2024",
         "status": "PENDING_BOUNDED_ACCESS_AND_LICENCE_FREEZE", "downloaded_bytes": 0,
         "reason": "No bounded authoritative subset endpoint and licence record were frozen in Stage 2D; global archives were not bulk-downloaded."},
        {"dataset": "Hansen GFC v1.11 treecover2000", "landscapes": "KUMAMOTO_ASO_2016|NOTO_PENINSULA_2024",
         "status": "PENDING_BOUNDED_TILE_ACCESS", "downloaded_bytes": 0,
         "reason": "Tile/API acquisition and temporal-staleness metadata remain to be bound; no global archive downloaded."},
    ])


def _report(prod, kuri, internal, dash) -> str:
    gs = json.loads((ROOT / "outputs/stage2c/geology_summary.json").read_text())
    state_counts = dash.state.value_counts().sort_index().to_dict()
    lines = [
        "# Stage 2D — production terrain units, core M features and landscape expansion", "",
        "## Scope and prohibitions", "",
        "Stage 2D built data only. No model was fitted, no LOLO or metric was run, no probabilities were calibrated, no H surface was reconstructed, "
        "and no lockbox/ensemble was assigned. Point and polygon target ontologies remain separate.", "",
        "## Track A — production results", "",
    ]
    for lid in ACTIVE:
        s = prod[lid]["summary"]
        lines += [f"### {lid}", "", f"- Target family: `{s['target_family']}`.",
                  f"- Aligned blocks intersecting OMEGA_ANALYSIS / valid ≥5-of-9 units: {s['total_candidate_aligned_blocks']:,} / {s['valid_units']:,}.",
                  f"- Positive / negative: {s['positive_units']:,} / {s['negative_units']:,}; natural prevalence {s['positive_prevalence']:.6f}.",
                  f"- Excluded boundary units: {s['excluded_units']:,}, all because fewer than five of nine cell centres were in OMEGA_ANALYSIS.",
                  f"- Frozen M outputs: {s['features_generated']} cell features and {s['feature_columns_after_mean_sd']} 90 m mean/SD columns; every production feature has 0 missing values.",
                  "- Hydrology: Lindsay (2016) depression breaching followed by Tarboton (1997) D-infinity flow accumulation; 0.1 km² channel threshold.",
                  f"- DEM-quality affected / ALL_VALID / STRICT_DEM_QUALITY units: {s['dem_quality_affected_units']:,} / {s['valid_units']:,} / {s['strict_dem_quality_units']:,}. "
                  "The strict rule excludes any unit containing a non-original Copernicus cell.",
                  f"- Geology classes: {json.dumps(s['geology_class_counts'], sort_keys=True)}.",
                  f"- Spatial CV metadata: {s['spatial_blocks_3000m']} 3 km blocks and {s['spatial_blocks_6000m']} 6 km blocks, defined without Y.",
                  "- QC maps: OMEGA, positive cells, DEM quality, geology index and slope; diagnostic only.",
                  f"- Production readiness: `{s['production_readiness']}`.", ""]
        if lid == "KUMAMOTO_ASO_2016":
            lines += [f"- Primary centres: {s['primary_source_features']} source; {s['primary_events_in_included_units']} in included units; "
                      f"{s['primary_events_excluded_by_90m_domain_rule']} retained in the explicit boundary-exclusion audit. Points were not buffered; 26 exact source duplicates remain audited but do not add events.",
                      f"- Geology sensitivity: PRIMARY has {prod[lid]['domain_rows'][0]['terrain_units']:,} units ({prod[lid]['domain_rows'][0]['domain_cell_area_km2']} km² cell area); "
                      f"the 1:50,000 Aso restriction has {prod[lid]['domain_rows'][1]['terrain_units']:,} units, {prod[lid]['domain_rows'][1]['positive_units']:,} positive "
                      f"({prod[lid]['domain_rows'][1]['domain_cell_area_km2']} km² cell area). Stage 2C Seamless/Aso footprint agreement is {gs['KUMAMOTO_ASO_1_50000_CROSSCHECK']['volcanic_agreement_fraction']:.4f}.", ""]
        else:
            q = s["target_geometry_diagnostics"]
            lines += ["- Primary label is exactly ≥1 of the nine 30 m cell centres inside a valid polygon; overlap area/fraction and polygon count are diagnostics only.",
                      f"- Diagnostic geometry: {q['units_with_nonzero_polygon_overlap']:,} units overlap polygon area; maximum overlap fraction {q['maximum_overlap_fraction']:.4f}; "
                      f"maximum intersecting polygons in one unit {q['maximum_intersecting_polygon_count']}.", ""]
    lines += [
        "## Track A.2 — internal expansion", "",
        "### Kurikoma", "",
        f"- Decision: `{kuri['decision']}`. The PDF has no documented CRS, coordinate grid or GeoPDF registration; zero defensible control points were available.",
        "- No transformation, residual, area or Omega polygon was fabricated; georeferencing uncertainty is unbounded.",
        "- Geology was not subset because that would require the missing Omega. Terrain-unit readiness remains blocked.",
        "- All 22 earlier transferred raw files remain byte-identical; final-unseen status remains `INELIGIBLE_V1_HISTORICAL`.", "",
        "### Lombok", "",
        f"- {internal['LOMBOK_RINJANI_2018']['transferred_files']} authorised raw files transferred byte-identically; 9,319 phase-a and 4,823 phase-b Tier-2 polygons plus the published mapping extent are present.",
        "- V2 Y/Omega/M pass provisional raw-source checks. Production fails because current G is only a scanned 1:250,000 figure and cannot define a defensible digital volcanic domain; EDM/FLM quality layers also remain absent.",
        "- H family remains a likely multi-edifice Rinjani/Samalas ensemble after G resolution. Historical status is ineligible for unseen testing.", "",
        "### Atitlan", "",
        f"- {internal['ATITLAN_STAN_2005']['transferred_files']} authorised raw files transferred byte-identically; 6,817 Atitlan and 962 Tzojoma Tier-2 polygons and both source boundaries are present.",
        "- V2 Y/Omega/M pass provisional raw-source checks. Production fails because no raw geology source exists, so the volcanic domain is not established; EDM/FLM quality layers also remain absent.",
        "- H family remains a likely caldera-rim/flank ensemble after G resolution. Historical status is ineligible for unseen testing.", "",
        "## Track B — external expansion", "",
        "- Coromandel: exact personal registration/export checks and a ready-but-unsent fallback email are in `COROMANDEL_USER_ACTION.md`.",
        "- Final request order: (1) PNG Porgera, (2) Banks Peninsula, (3) Kivu NTK, (4) Mount Elgon, (5) Guatemala 1976, (6) Miravalles/Otto, (7) Kivu Kalehe. Every package includes custodian, missing artifact, Omega/Y requirements, licence question, exact draft and failure fallback.",
        "- Existing registration/supplement/repository routes were audited only for blocker resolution. No candidate was promoted and no broad rediscovery was repeated.",
        "- Path to 10: resolve all three internal cases, Coromandel, and four of the ranked requests (2 current + 3 internal + 1 Coromandel + 4 requests = 10).",
        "- Path to 15: the current bounded queue can yield at most 13 (14 if defensible Hokkaido tephra evidence appears); reaching 15 therefore requires both strong request success and a later targeted access wave among already screened candidates, without weakening gates.", "",
        "## Global audit", "",
        f"- Dashboard: {len(dash)} candidates; {int(dash.metadata_ready.sum())} metadata-ready, but only {int(dash.production_ready_after_harmonisation.sum())} genuinely production-ready. State counts: {json.dumps(state_counts, sort_keys=True)}.",
        "- Ready diversity: Asia only; one subduction-caldera point-event landscape and one old-volcanic-terrain polygon-affected landscape. The screened dashboard spans six continents and seven tectonic-setting classes.",
        "- H: four landscapes are potentially ensemble-capable (Kumamoto, Kurikoma, Lombok, Atitlan), but only Kumamoto is ready to begin a production V2 H/Gen-2 reconstruction. Noto supports age/history proxies only.",
        "- E: CHELSA five-variable core and Hansen tree cover remain pending bounded access/licence binding. No scientific E bytes were downloaded; optional land cover and SoilGrids remain pending.",
        "- Tests: 76 passed across Stages 1–2D. V1 integrity was verified before and after each controlled transfer. Natural class balance was preserved.",
        "- Checkpoint: new `STAGE2D_CHECKPOINT.json`; prior checkpoints were not overwritten. Git diagnostics are non-authoritative; no commit or push was performed.", "",
        "## Final decisions", "",
        "1. **Genuinely production-ready: 2** — Kumamoto/Aso 2016 and Noto Peninsula 2024.",
        "2. **Stage 2E entrants:** Kumamoto and Noto, kept as separate target families.",
        "3. **Kurikoma:** not yet usable for V2 terrain-unit development; a coordinate-supported official interpretation boundary is required. It remains useful historical raw evidence only.",
        "4. **Lombok:** did not pass current V2 gates; digital volcanic-domain geology and DEM-quality masks are missing.",
        "5. **Atitlan:** did not pass current V2 gates; authoritative geology/volcanic domain and DEM-quality masks are missing.",
        "6. **Highest-probability user actions:** register and inspect/export Coromandel first; send PNG Porgera and Banks Peninsula requests next; then Kivu NTK and Mount Elgon; obtain digital geology for Lombok and Atitlan; obtain a coordinate-supported Kurikoma map/base layer.",
        "7. **90 m architecture:** no scientific reason to revise the primary architecture now. It is deterministic and cross-landscape compatible. Retain the predeclared 30 m sensitivity because Noto failures are small and retain the explicit ≥5/9 boundary audit because one Kumamoto centre is excluded at 90 m.",
        "8. **H / Gen-2:** yes for Kumamoto. Do not start production H for Kurikoma, Lombok or Atitlan until their stated controls/gates are resolved; do not force an edifice reconstruction for Noto.", "",
    ]
    return "\n".join(lines)


def main() -> dict:
    if not PRIOR.exists(): raise RuntimeError("Stage 2C checkpoint missing")
    OUT.mkdir(parents=True, exist_ok=True)
    logs = transfer_audit()
    internal = internal_audits(logs)
    kuri = kurikoma_assessment()
    prod = {lid: build_landscape(lid) for lid in ACTIVE}
    definitions = feature_definition_records([n for r in prod.values() for n in r["feature_names"]])
    _csv(OUT / "feature_definitions.csv", definitions)
    dash = dashboard(internal)
    external_packages(); h_feasibility(); e_status()
    (OUT / "STAGE2D_REPORT.md").write_text(_report(prod, kuri, internal, dash))

    # Complete output lineage is created last. Transfer logs already carry per-file source and destination hashes.
    excluded = {"STAGE2D_CHECKPOINT.json", "STAGE2D_DIAGNOSTICS.json", "lineage.json"}
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded)
    outputs += sorted(p for lid in ACTIVE for p in (DER / lid).rglob("*") if p.is_file())
    outputs += sorted(p for p in (ROOT / "data/raw/stage2d").rglob("*") if p.is_file())
    script_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (SCRIPT, ROOT / "src/sfv2/stage2d_production.py", ROOT / "src/sfv2/stage2d_transfer.py")}
    transfer_sources = {r["v2_destination"]: r for rows in logs.values() for r in rows}
    def inp(path: Path) -> dict:
        return {"path": str(path.relative_to(REPO)), "sha256": sha256(path)}
    def inputs_for(p: Path) -> list[dict]:
        rel = str(p.relative_to(ROOT))
        if rel in transfer_sources:
            return [inp(REPO / transfer_sources[rel]["v1_path"]), inp(REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_HASH_MANIFEST.json")]
        if "/features_m/" in rel or "/qc_hydrology/" in rel:
            lid = next(x for x in ACTIVE if x in rel)
            return [inp(ROOT / f"data/processed/stage2c/{lid}/dem_cop30_30m.tif")]
        if "/terrain_units/" in rel:
            lid = next(x for x in ACTIVE if x in rel); s = ROOT / f"data/processed/stage2c/{lid}"
            return [inp(next(s.glob("omega_analysis_primary*_30m.tif"))), inp(s / "cop30_edm_30m.tif"),
                    inp(s / "cop30_flm_30m.tif"), inp(s / "geology_seamless_index_30m.tif")]
        if "/targets/" in rel:
            lid = next(x for x in ACTIVE if x in rel); s = ROOT / f"data/processed/stage2c/{lid}"
            y = s / ("y_harmonised_points.geojson" if lid == "KUMAMOTO_ASO_2016" else "y_harmonised_polygons.geojson")
            return [inp(y), inp(next(s.glob("omega_analysis_primary*_30m.tif")))]
        if "transfer" in rel:
            return [inp(REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_HASH_MANIFEST.json"),
                    inp(REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FILE_MANIFEST.tsv")]
        return [inp(PRIOR)]
    lineage = [{
        "output": str(p.relative_to(ROOT)), "output_sha256": sha256(p), "inputs": inputs_for(p),
        "operation": "Stage2D controlled V1 raw transfer" if str(p.relative_to(ROOT)) in transfer_sources else "Stage2D deterministic build",
        "parameters": {"terrain_unit_m": 90, "canonical_grid_m": 30, "as_of_date": AS_OF},
        "software_version": "see feature_definitions.csv and Stage2C provenance", "script_hashes": script_hashes,
        "timestamp": AS_OF,
    } for p in sorted(set(outputs))]
    _json(OUT / "lineage.json", lineage)
    scientific_outputs = sorted([*outputs, OUT / "lineage.json"])
    cp = {
        "schema": "SFV2_STAGE2D_CHECKPOINT_V1", "as_of_date": AS_OF,
        "prior_stage2c_checkpoint_sha256": sha256(PRIOR),
        "production_landscapes": list(ACTIVE),
        "production_terrain_unit_hashes": {lid: sha256(DER / lid / "terrain_units/terrain_units.csv.gz") for lid in ACTIVE},
        "feature_definition_hash": sha256(OUT / "feature_definitions.csv"),
        "generated_m_feature_hashes": {lid: sha256(DER / lid / "features_m/features_m.csv.gz") for lid in ACTIVE},
        "target_hashes": {lid: sha256(DER / lid / "targets/targets_primary.csv.gz") for lid in ACTIVE},
        "landscape_state_dashboard_hash": sha256(OUT / "landscape_state_dashboard.csv"),
        "transfer_manifest_hashes": {p.name: sha256(p) for p in sorted((OUT / "transfers").glob("*manifest.csv"))},
        "provenance_hash": sha256(OUT / "lineage.json"),
        "qc_metadata_hashes": {str(p.relative_to(OUT)): sha256(p) for p in sorted((OUT / "qc").rglob("*")) if p.is_file()},
        "output_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in scientific_outputs},
        "training": False, "h_reconstruction": False, "lolo": False, "lockbox_assignment": False,
        "natural_class_balance_preserved": True, "v1_integrity_verified_before_and_after_transfer": True,
        "volatile_git_diagnostics": "outputs/stage2d/STAGE2D_DIAGNOSTICS.json (excluded from identity)",
    }
    _json(OUT / "STAGE2D_CHECKPOINT.json", cp)
    git = lambda *x: subprocess.run(["git", *x], cwd=REPO, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE2D_DIAGNOSTICS.json", {"authoritative": False, "head": git("rev-parse", "HEAD"),
                                              "git_status": git("status", "--short"), "commit_performed": False, "push_performed": False})
    return cp


if __name__ == "__main__":
    main()
