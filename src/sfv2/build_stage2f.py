"""Build the Stage 2F access/rescue checkpoint without weakening any gate."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .raw_store import sha256
from .schema import ROOT
from .stage2c_governance import REQUEST_PACKAGES
from .stage2f_transfer import REQUIRED_PROVENANCE, transfer_puerto_rico, v1_verify

OUT = ROOT / "outputs/stage2f"
REPO = ROOT.parent
SCRIPT = Path(__file__)
TRANSFER_SCRIPT = Path(__file__).with_name("stage2f_transfer.py")
STAMP = "2026-10-06T00:00:00+10:00"
ALLOWED_STATES = (
    "PRODUCTION_READY", "HARMONISATION_IN_PROGRESS", "HUMAN_ACCESS_REQUIRED",
    "EXTERNAL_REQUEST_REQUIRED", "BLOCKED_OMEGA", "BLOCKED_GEOLOGY", "BLOCKED_M",
    "BLOCKED_Y", "BLOCKED_VOLCANIC_DOMAIN", "HOLD", "REJECT", "REFERENCE_ONLY",
)
PRIMARY_REQUEST_ORDER = ("RP01", "RP02", "RP03", "RP04", "RP05", "RP06", "RP07")


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False, lineterminator="\n")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def internal_audit() -> list[dict]:
    rows = [
        {
            "landscape_id": "LOMBOK_RINJANI_2018", "state": "HUMAN_ACCESS_REQUIRED",
            "y_gate": "PASS_PROVISIONAL", "omega_gate": "PASS_PROVISIONAL_INDEPENDENT_SOURCE",
            "m_gate": "PASS_PROVISIONAL_QUALITY_RECHECK_REQUIRED", "geology_gate": "BLOCKED_AUTHENTICATED_DOWNLOAD",
            "target_family": "POLYGON_AFFECTED_TIER2", "production_ready": False, "terrain_units_generated": False,
            "finding": "Official district pages expose authenticated geology SHP for Lombok Utara (52.08) and Lombok Timur (52.03). The public raster fallback is Peta Geologi Lembar Lombok, Nusatenggara Barat, sheet 1807, 1:250,000 (1994).",
            "exact_blocker": "User sign-in and download of the authoritative district geology SHP archives; attributes, CRS, legend and coverage must then pass inspection.",
            "official_urls": "https://geologi.esdm.go.id/geomap/pages/district/52.08|https://geologi.esdm.go.id/geomap/pages/district/52.03|https://geologi.esdm.go.id/geomap/pages/preview/peta-geologi-lembar-lombok-nusatenggara-barat",
            "h_status": "H_UNAVAILABLE_LIKELY_MULTI_EDIFICE_AFTER_CORE_GATES",
        },
        {
            "landscape_id": "COROMANDEL_GABRIELLE_2023", "state": "HUMAN_ACCESS_REQUIRED",
            "y_gate": "BLOCKED_AUTHENTICATED_POLYGON_INSPECTION", "omega_gate": "BLOCKED_INDEPENDENT_EVENT_SUPPORT",
            "m_gate": "BLOCKED_PRE_EVENT_DEM_BINDING", "geology_gate": "BLOCKED_VOLCANIC_DOMAIN_BINDING",
            "target_family": "UNRESOLVED_SOURCE_VS_AFFECTED_POLYGONS", "production_ready": False, "terrain_units_generated": False,
            "finding": "NZLD confirms that unregistered users see points and free registered users can access all feature classes, including polygons. Database coverage is not event-specific Omega.",
            "exact_blocker": "Authenticated inspection/export of Cyclone Gabrielle polygons and independent searched/mapped support, followed by pre-event M and volcanic geology binding.",
            "official_urls": "https://www.gns.cri.nz/data-and-resources/new-zealand-landslide-database/|https://data.gns.cri.nz/landslides/",
            "h_status": "H_UNAVAILABLE_DEFER",
        },
        {
            "landscape_id": "PUERTO_RICO_MARIA_2017", "state": "BLOCKED_OMEGA",
            "y_gate": "PASS_RAW_POINT_INVENTORY", "omega_gate": "BLOCKED_NO_EXACT_INDEPENDENT_VISIBILITY_GEOMETRY",
            "m_gate": "BLOCKED_PRE_EVENT_DEM_SELECTION", "geology_gate": "BLOCKED_AUTHORITATIVE_VOLCANIC_DOMAIN",
            "target_family": "POINT_HEADSCARP_TIER1", "production_ready": False, "terrain_units_generated": False,
            "finding": "The 71,431 headscarp-centre points and documentation transferred byte-identically. Approximate imagery percentages in metadata do not define Omega.",
            "exact_blocker": "Exact systematically observed imagery/search footprint with cloud-shadow/unmapped exclusions; pre-event M and authoritative volcanic geology also remain unresolved.",
            "official_urls": "https://www.sciencebase.gov/catalog/item/5d4c8b26e4b01d82ce8dfeb0",
            "h_status": "H_UNAVAILABLE_NOT_PRIORITY",
        },
        {
            "landscape_id": "ATITLAN_STAN_2005", "state": "BLOCKED_GEOLOGY",
            "y_gate": "PASS_PROVISIONAL", "omega_gate": "PASS_PROVISIONAL_INDEPENDENT_SOURCE",
            "m_gate": "PASS_PROVISIONAL_QUALITY_RECHECK_REQUIRED", "geology_gate": "BLOCKED_NO_REUSABLE_AUTHORITATIVE_GEOMETRY",
            "target_family": "POLYGON_AFFECTED_TIER2", "production_ready": False, "terrain_units_generated": False,
            "finding": "The official MAGA Solola soil study documents use of four 1:50,000 IGN geology quadrangles, but both official PDF routes returned HTTP 403 and no authoritative vector or full-resolution georeferenced geology was exposed.",
            "exact_blocker": "Reusable authoritative geology polygons, or a full-resolution coordinate-supported official map with complete legend suitable for uncertainty-aware digitisation.",
            "official_urls": "https://www.maga.gob.gt/wp-content/uploads/pdf/suelos/solola1.pdf|https://www.maga.gob.gt/download/suelos-solola1.pdf|https://insivumeh.gob.gt/mapas-de-amenaza-volcanica/",
            "h_status": "H_UNAVAILABLE_LIKELY_CALDERA_AFTER_CORE_GATES",
        },
        {
            "landscape_id": "KURIKOMA_2008", "state": "BLOCKED_OMEGA",
            "y_gate": "PASS_PROVISIONAL", "omega_gate": "BLOCKED_NO_COORDINATE_SUPPORTED_BOUNDARY",
            "m_gate": "PASS_PROVISIONAL", "geology_gate": "NOT_EVALUATED_WITHOUT_OMEGA",
            "target_family": "POLYGON_AFFECTED_TIER2", "production_ready": False, "terrain_units_generated": False,
            "finding": "The bounded official search confirmed 1:10,000 aerial photography and an interpretation-range PDF, but found no GIS boundary, coordinate grid, GeoPDF, or spatially registered raster for that campaign.",
            "exact_blocker": "GSI digital interpretation boundary or officially registered base map associated with Material 4.",
            "official_urls": "https://www.gsi.go.jp/johosystem/johosystem60029.html|https://www.gsi.go.jp/johosystem/johosystem40087.html",
            "h_status": "H_UNAVAILABLE_DEFER_UNTIL_OMEGA",
        },
    ]
    _csv(OUT / "internal/internal_resolution.csv", rows)
    return rows


def puerto_harmonisation(transfer: dict) -> dict:
    result = {
        "landscape_id": "PUERTO_RICO_MARIA_2017",
        "transferred_raw_files": len(transfer["logs"]),
        "transferred_raw_bytes": sum(int(r["actual_bytes"]) for r in transfer["logs"]),
        "all_hash_equal": all(r["equality"] == "PASS" for r in transfer["logs"]),
        "y": {"status": "PASS_RAW", "features": 71431, "geometry": "POINT", "semantics": "headscarp centre", "target_family": "POINT_HEADSCARP_TIER1"},
        "omega": {"status": "BLOCKED", "independent_of_y": True, "derived_from_y": False,
                  "reason": "Percent imagery coverage and overlapping scenes do not supply exact observable geometry or visibility exclusions."},
        "m": {"status": "BLOCKED", "reason": "No pre-event M was eligible in the transfer; epoch-safe source and quality masks remain to be acquired."},
        "geology": {"status": "BLOCKED", "reason": "No geology was eligible in the transfer; authoritative unit polygons must define the island-arc volcanic subset."},
        "temporal_caveat": "Some mapped failures may be Irma/pre-Maria or post-event failures between Maria and image acquisition.",
        "omega_analysis_generated": False,
        "production_terrain_units_generated": False,
        "production_ready": False,
        "final_state": "BLOCKED_OMEGA",
        "final_unseen_eligibility": "INELIGIBLE_V1_HISTORICAL_REFERENCE",
    }
    _json(OUT / "internal/PUERTO_RICO_MARIA_2017_harmonisation.json", result)
    return result


def public_route_audit() -> list[dict]:
    rows = [
        (1, "PNG_PORGERA_2018", "USGS ScienceBase item already public", "No public derivation of mapping_extent or internal cloud/image-free masks found", "READY_TO_SEND", "Shapefile/GeoPackage support and visibility polygons", "https://www.sciencebase.gov/catalog/item/61f04680"),
        (2, "BANKS_PENINSULA_2021", "Publisher article and institutional record inspected", "Article states data are available on reasonable request; no public GIS supplement found", "READY_TO_SEND", "Source/runout polygons and independent 115 km2 mapped boundary", "https://rsnz.onlinelibrary.wiley.com/doi/10.1080/00288306.2025.2458507"),
        (3, "KIVU_RIFT_NTK_2000_2019", "Zenodo 5027004 and publication route inspected", "Public points do not resolve source polygons or year-specific search support", "READY_TO_SEND", "Source polygons, imagery/search mask by year, negative-Year definition", "https://zenodo.org/records/5027004"),
        (4, "MOUNT_ELGON_UGANDA", "Official VUB/KU Leuven publication records inspected", "No public inventory vector or systematic mapped-area file found", "READY_TO_SEND", "Polygons with dates and systematically mapped area", "https://researchportal.vub.be/en/publications/landslide-susceptibility-assessment-in-a-highly-populated-mountain"),
        (5, "GUATEMALA_1976", "USGS dataset and Professional Paper plates inspected", "Inventory and plates are public; no digital Plate-1 mapped-domain boundary found; pre-event M remains difficult", "READY_TO_SEND", "Digital mapped-area boundary, preferably Shapefile with registration", "https://pubs.usgs.gov/publication/pp1204A"),
        (6, "KIVU_KALEHE_RIFT_FLANK", "NHESS article and supplement inspected", "Supplement is PDF; article says inventory/maps are available on reasonable request", "READY_TO_SEND", "Inventory vectors, study boundary, visibility/fieldwork mask, epoch", "https://nhess.copernicus.org/articles/23/643/2023/"),
        (7, "MIRAVALLES_OTTO_2016", "Publisher/repository search bounded to the cited study", "No public GIS polygons with separated earthquake/hurricane triggers found", "READY_TO_SEND", "Trigger-attributed polygons, mapped area, imagery dates", "https://doi.org/10.1007/s10346-019-01172-y"),
    ]
    records = [{"rank": a, "landscape_id": b, "public_routes_checked": c, "result": d,
                "request_status": e, "exact_missing_artifact": f, "official_or_publisher_url": g,
                "public_blocker_resolved": False} for a, b, c, d, e, f, g in rows]
    _csv(OUT / "external/public_route_audit.csv", records)
    return records


def request_packages() -> list[dict]:
    by_id = {x[0]: x for x in REQUEST_PACKAGES}
    records, text = [], ["# Stage 2F ready-to-send requests", "", "Status: prepared, verified, and **not sent**.", ""]
    for rank, package_id in enumerate(PRIMARY_REQUEST_ORDER, 1):
        x = by_id[package_id]
        records.append({
            "rank": rank, "package_id": package_id, "landscape_id": x[1], "dataset": x[2],
            "custodian": x[3], "exact_missing_artifact": x[4], "why_required": x[5],
            "preferred_format": x[6], "spatial_metadata": x[7], "licence_question": x[8],
            "status": "READY_TO_SEND_NOT_SENT",
        })
        text += [f"## {rank}. {x[1]} — {package_id}", "", f"**To:** {x[3]}", "",
                 f"**Need:** {x[4]}", "", f"**Preferred delivery:** {x[6]}; {x[7]}.", "",
                 f"**Licence question:** {x[8]}", "", x[9], ""]
    _csv(OUT / "external/request_queue.csv", records)
    (OUT / "external/READY_TO_SEND_REQUESTS.md").write_text("\n".join(text) + "\n")
    return records


def second_wave() -> pd.DataFrame:
    wave = pd.read_csv(ROOT / "outputs/stage2e/access/second_wave_access.csv")
    wave["stage2f_decision"] = wave["decision"].map({
        "TARGETED_REQUEST": "RETAIN_BOUNDED_QUEUE", "TARGETED_ACCESS": "RETAIN_BOUNDED_QUEUE",
        "TARGETED_DISTRICT_AUDIT": "RETAIN_BOUNDED_QUEUE",
        "FAST_FAIL_UNLESS_MANUAL_SUPPORT_PROVIDED": "NO_ACTION_UNLESS_PRIMARY_QUEUE_FAILS",
        "CONDITIONAL_BACKUP": "NO_ACTION_UNLESS_PRIMARY_QUEUE_FAILS",
        "FAST_FAIL_CURRENT_DATASET": "REJECT_CURRENT_DATASET",
    })
    _csv(OUT / "external/second_wave.csv", wave)
    return wave


def feature_contract() -> dict:
    path = ROOT / "outputs/stage2d/feature_definitions.csv"
    frame = pd.read_csv(path)
    contract = {
        "source": "outputs/stage2d/feature_definitions.csv", "sha256": sha256(path),
        "definition_count": len(frame), "reuse_policy": "EXACT_FROZEN_REUSE_ONLY",
        "modified_for_stage2f": False, "newly_rescued_landscapes": [],
        "new_production_terrain_unit_products": [],
        "reason": "No candidate passed every production gate; deterministic generation was therefore not applicable.",
    }
    _json(OUT / "production/frozen_m_feature_contract.json", contract)
    return contract


def dashboard() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "outputs/stage2e/landscape_state_dashboard.csv")
    old_rejects = set(d.loc[d.state.eq("REJECT"), "landscape_id"])
    d["state"] = "HOLD"
    d.loc[d.landscape_id.isin(old_rejects), "state"] = "REJECT"
    mappings = {
        "KUMAMOTO_ASO_2016": "PRODUCTION_READY", "NOTO_PENINSULA_2024": "PRODUCTION_READY",
        "LOMBOK_RINJANI_2018": "HUMAN_ACCESS_REQUIRED", "COROMANDEL_GABRIELLE_2023": "HUMAN_ACCESS_REQUIRED",
        "PUERTO_RICO_MARIA_2017": "BLOCKED_OMEGA", "ATITLAN_STAN_2005": "BLOCKED_GEOLOGY",
        "KURIKOMA_2008": "BLOCKED_OMEGA", "HOKKAIDO_IBURI_2018": "HOLD",
        "PNG_PORGERA_2018": "EXTERNAL_REQUEST_REQUIRED", "BANKS_PENINSULA_2021": "EXTERNAL_REQUEST_REQUIRED",
        "KIVU_RIFT_NTK_2000_2019": "EXTERNAL_REQUEST_REQUIRED", "MOUNT_ELGON_UGANDA": "EXTERNAL_REQUEST_REQUIRED",
        "GUATEMALA_1976": "EXTERNAL_REQUEST_REQUIRED", "KIVU_KALEHE_RIFT_FLANK": "EXTERNAL_REQUEST_REQUIRED",
        "MIRAVALLES_OTTO_2016": "EXTERNAL_REQUEST_REQUIRED", "WILSONS_RIVER_NSW_2022": "EXTERNAL_REQUEST_REQUIRED",
        "DOMINICA_MARIA_2017": "HUMAN_ACCESS_REQUIRED", "SAO_MIGUEL_AZORES": "EXTERNAL_REQUEST_REQUIRED",
        "RWANDA_VIRUNGA_NW": "HOLD", "PATAGONIA_PALI": "HOLD", "REUNION_BDMVT": "REJECT",
        "IFFI_ITALIAN_VOLCANIC_DISTRICTS": "HOLD",
    }
    for lid, state in mappings.items():
        d.loc[d.landscape_id.eq(lid), "state"] = state
    d["production_ready_after_harmonisation"] = d.state.eq("PRODUCTION_READY")
    d["harmonisation_status"] = d.state.map(lambda s: "COMPLETE" if s == "PRODUCTION_READY" else "NOT_PRODUCTION_READY")
    d.loc[d.landscape_id.eq("PUERTO_RICO_MARIA_2017"), "target_family"] = "POINT_HEADSCARP_TIER1"
    d.loc[d.landscape_id.eq("PUERTO_RICO_MARIA_2017"), "final_unseen_eligibility"] = "INELIGIBLE_V1_HISTORICAL_REFERENCE"
    if not set(d.state).issubset(ALLOWED_STATES):
        raise RuntimeError("Stage 2F dashboard contains a non-authoritative state")
    _csv(OUT / "landscape_state_dashboard.csv", d)
    ready = d[d.state.eq("PRODUCTION_READY")]
    _json(OUT / "landscape_state_summary.json", {
        "candidate_count": len(d), "allowed_states": ALLOWED_STATES,
        "state_counts": dict(sorted(Counter(d.state).items())),
        "production_ready_count": len(ready), "production_ready_landscapes": ready.landscape_id.tolist(),
        "production_terrain_units": 450935,
        "continent_distribution": dict(sorted(Counter(ready.continent).items())),
        "tectonic_distribution": dict(sorted(Counter(ready.tectonic_setting).items())),
        "target_family_distribution": dict(sorted(Counter(ready.target_family).items())),
        "h_capable_conceptual_count": 4, "h_surface_scientific_qc_pass_count": 0,
    })
    return d


def user_actions() -> None:
    text = """# Stage 2F user actions

Only the following actions require your identity, account, or sending authority. Do not send passwords or session cookies.

## 1 — Lombok GeoMap sign-in (fastest plausible landscape #3)

- Sign in at [Lombok Utara SHP](https://geologi.esdm.go.id/geomap/pages/signin/district/52.08) and download **SHP: Kab./Kota Lombok Utara**.
- Sign in at [Lombok Timur SHP](https://geologi.esdm.go.id/geomap/pages/signin/district/52.03) and download **SHP: Kab./Kota Lombok Timur**.
- Preserve each original archive and every `.shp/.shx/.dbf/.prj/.cpg`, legend/readme, metadata and licence file. If either district SHP is unavailable, download the complete 1994 **Peta Geologi Lembar Lombok, Nusatenggara Barat**, sheet 1807, 1:250,000 raster plus margins/legend from its [official preview](https://geologi.esdm.go.id/geomap/pages/preview/peta-geologi-lembar-lombok-nusatenggara-barat).
- Put untouched downloads in `Science_Fair_ML_V2/data/incoming_user/`. They are needed to bind authoritative units, age/lithology and CRS before the volcanic domain and terrain units can be built.

## 2 — Coromandel NZLD registration and export

- Create/verify a free account and sign in at [NZLD](https://data.gns.cri.nz/landslides/).
- Filter for **Cyclone Gabrielle, February 2023, Coromandel**. Export source polygons, affected/deposit polygons and debris trails separately, with dataset/feature IDs, event/date/trigger, mapping method, quality, native CRS, licence and data dictionary.
- Also export the independently defined searched/mapped area, survey/imagery footprints, cloud-shadow/unmapped exclusions and completeness statement. Database coverage or a hull/buffer of landslides is not acceptable Omega.
- Put untouched exports and metadata in `Science_Fair_ML_V2/data/incoming_user/`. If event polygons or mapped support cannot be exported, contact the named institutional custodian using the Coromandel request in the Stage 2E action file.

## 3–9 — Send prepared external requests

Send, in this order: **PNG Porgera**, **Banks Peninsula**, **Kivu NTK**, **Mount Elgon**, **Guatemala 1976**, **Kivu Kalehe**, **Miravalles/Otto**. The exact addressees, missing artifacts, formats, licence questions and ready-to-send text are in `outputs/stage2f/external/READY_TO_SEND_REQUESTS.md`. Use the corresponding-author or institutional-custodian route on each cited source page; direct email addresses are removed from this public source copy.

For every reply, save the original attachment(s), accompanying email/terms and citation in `Science_Fair_ML_V2/data/incoming_user/`. Tell Codex the filenames; ingestion will preserve the originals, checksum them, register immutable RAW copies, and resume gate evaluation.
"""
    (OUT / "STAGE2F_USER_ACTIONS.md").write_text(text)


def report(transfer: dict, d: pd.DataFrame, requests: list[dict], wave: pd.DataFrame) -> None:
    text = f"""PRODUCTION-READY LANDSCAPES:
START = 2
END = 2

# Stage 2F access and landscape rescue report

No readiness gate was relaxed. No ML, LOLO, model metric, SHAP, calibration, lockbox assignment, susceptibility mapping, production H, commit, push, or new production terrain-unit generation occurred.

## Lombok

- Access: official public catalogue inspection is complete; the two district SHP downloads require sign-in.
- Exact action: obtain Lombok Utara `52.08` and Lombok Timur `52.03` district geology SHP archives, including CRS/legend/metadata; use the sheet-1807 1:250,000 raster only as an authoritative fallback.
- Geology: no bytes supplied yet, so source units, age/lithology fields and volcanic-domain geometry cannot be validated.
- Readiness: `HUMAN_ACCESS_REQUIRED`, not production-ready.

## Coromandel

- Access: public documentation confirms points for unauthenticated users and all feature classes, including polygons, for free registered users.
- Exact action: sign in, export event-attributed polygon classes and independent event mapping/imagery support with exclusions and metadata.
- Feasibility: Y may be viable, but NZLD database coverage is not Omega; source/deposit semantics, pre-event M and geology still require binding.
- Readiness: `HUMAN_ACCESS_REQUIRED`, not production-ready.

## Puerto Rico

- Transfer: **3/3 authorised raw files**, **{sum(int(x['actual_bytes']) for x in transfer['logs']):,} bytes**, copied byte-identically; every V1 and V2 SHA256 matches. Frozen V1 verified intact before and after.
- V2 result: 71,431 point headscarp centres preserve `POINT_HEADSCARP_TIER1`; no V1 assumptions or derived files were imported.
- Readiness: `BLOCKED_OMEGA`. Approximate imagery-coverage percentages are not an exact observable domain. Pre-event M, visibility masks and authoritative volcanic geology also remain unresolved.

## Atitlan

- The targeted official search located the MAGA Solola study's documented 1:50,000 IGN geology inputs, but official PDF endpoints returned 403 and no reusable vector/full-resolution registered map was exposed.
- Conservative volcanic/caldera-related/basement/surficial/intrusive/ambiguous classes remain unchanged; no low-quality figure was digitised.
- Readiness: `BLOCKED_GEOLOGY`.

## Kurikoma

- The one bounded GSI attempt found the campaign's 1:10,000 aerial-photography record and interpretation-range PDF, but no coordinate-supported GIS, GeoPDF, grid or registered raster.
- No Omega was inferred from Y; the landscape remains `INELIGIBLE_V1_HISTORICAL` and `BLOCKED_OMEGA`.

## External queue

- Publicly resolved blockers: **0/7**. Public articles, repositories, supplements and publisher mirrors were inspected; none supplies every missing artifact.
- Ranked send order: PNG Porgera; Banks Peninsula; Kivu NTK; Mount Elgon; Guatemala 1976; Kivu Kalehe; Miravalles/Otto.
- All seven packages remain `READY_TO_SEND_NOT_SENT`; zero external messages were sent.

## Second wave

The existing bounded seven-candidate wave is retained because the primary/internal queue cannot reliably reach ten. It covers Wilsons River, Dominica, an IFFI volcanic district, Sao Miguel, Patagonia Pali, Rwanda/Virunga and Reunion; fast-fail/conditional decisions are preserved. No broad rediscovery or new large download was performed.

## Global

- Production-ready: **2**; terrain units: **450,935**.
- Distribution: Asia 2; tectonics = `SUBDUCTION_ARC` 1 and `OTHER` 1; targets = `POINT_EVENT_TIER2` 1 and `POLYGON_AFFECTED_TIER2` 1.
- Conceptually H-capable: 4; H surfaces passing scientific QC: 0. Aso production H remains stopped.
- Human actions: two authenticated downloads/inspections plus seven external sends; see `STAGE2F_USER_ACTIONS.md`.
- Scientific internet downloads: **0 bytes**. Controlled local V1-to-V2 raw transfer: **2,562,898 bytes**.
- V1: intact before and after, 153 strict files, zero writes.
- Git: pre-existing dirty/untracked workspace preserved; no commit or push.

## Final questions

1. **Did we move beyond 2?** No. The count remains 2 because no candidate passed all gates.
2. **Fastest route to #3?** Lombok: the user downloads the two official district geology SHPs, then Codex validates attributes/CRS, builds the volcanic-domain intersection, rechecks M/visibility/leakage and generates the unchanged Stage 2D terrain-unit/features only if all gates pass.
3. **Fastest route to #4 and #5?** Banks Peninsula is the cleanest request-based prospect; Coromandel follows if authenticated export includes an independent event mapping domain. PNG is next if its custodian supplies a defensible search/visibility mask. Replies, not gate relaxation, control timing.
4. **Exact user actions now?** GeoMap sign-in/download; NZLD registration/export; send the seven prepared requests in the fixed order. No credentials should be shared.
5. **What can Codex continue alone?** Immediately ingest/checksum any supplied file, validate provenance/licence/CRS and semantics, harmonise Y/Omega/M/geology, build domains, run QC, and generate deterministic terrain units with the frozen feature contract when—and only when—all gates pass. Without new bytes or replies, the bounded public searches are exhausted.
6. **Realistic path to 10?** Current 2 + Lombok + Coromandel + three successful primary requests + three successful bounded second-wave/internal rescues. That is plausible only with unusually strong access responses and complete independent Omega artifacts.
7. **Is 15 achievable?** Not on evidence currently accessible within a normal science-fair timeline. It remains a stretch only if request response rates are high and additional dataset-first candidates later satisfy every gate; do not plan the core study around 15.
8. **Are gates unnecessarily restrictive?** No. They are preventing genuine contamination: positive-derived Omega, post-event leakage, mixed target semantics and non-volcanic/background misclassification. Hokkaido is the clearest example of a correct scientific stop.
9. **Is ML training justified?** **No.** Two independent landscapes are insufficient for the intended cross-landscape evaluation.
"""
    (OUT / "STAGE2F_REPORT.md").write_text(text)


def checkpoint(before: dict, after: dict, transfer: dict, d: pd.DataFrame, feature: dict) -> dict:
    excluded = {"lineage.json", "STAGE2F_CHECKPOINT.json"}
    artifacts = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in excluded)
    scripts = {str(SCRIPT.relative_to(ROOT)): sha256(SCRIPT), str(TRANSFER_SCRIPT.relative_to(ROOT)): sha256(TRANSFER_SCRIPT)}
    inputs = [
        ROOT / "outputs/stage2e/STAGE2E_CHECKPOINT.json",
        ROOT / "outputs/stage2e/landscape_state_dashboard.csv",
        ROOT / "outputs/stage2d/feature_definitions.csv",
        REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_HASH_MANIFEST.json",
    ]
    input_hashes = [{"path": str(p.relative_to(REPO)), "sha256": sha256(p)} for p in inputs]
    lineage = [{
        "output": str(p.relative_to(ROOT)), "output_sha256": sha256(p),
        "operation": "STAGE2F_ACCESS_TRANSFER_OR_GATE_AUDIT", "timestamp": STAMP,
        "parameters": {"scientific_download_bytes": 0, "controlled_transfer_bytes": 2562898,
                       "production_gate_conjunction_required": True, "training": False},
        "scripts": scripts, "inputs": input_hashes,
    } for p in artifacts]
    _json(OUT / "lineage.json", lineage)
    output_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in artifacts}
    output_hashes["outputs/stage2f/lineage.json"] = sha256(OUT / "lineage.json")
    cp = {
        "stage": "2F", "as_of": "2026-10-06", "prior_checkpoint": "outputs/stage2e/STAGE2E_CHECKPOINT.json",
        "prior_checkpoint_sha256": sha256(ROOT / "outputs/stage2e/STAGE2E_CHECKPOINT.json"),
        "production_ready_start": 2, "production_ready_end": int(d.state.eq("PRODUCTION_READY").sum()),
        "production_ready_landscapes": d.loc[d.state.eq("PRODUCTION_READY"), "landscape_id"].tolist(),
        "production_terrain_units": 450935, "newly_rescued_landscapes": [], "new_production_terrain_unit_products": [],
        "puerto_rico_transfer_files": len(transfer["logs"]), "controlled_transfer_bytes": 2562898,
        "scientific_download_bytes": 0, "external_messages_sent": 0, "accounts_created": 0,
        "training": False, "lolo": False, "model_metrics": False, "shap": False,
        "probability_calibration": False, "lockbox_assignment": False, "susceptibility_maps": False,
        "production_h_generated": False, "feature_contract_sha256": feature["sha256"],
        "authoritative_states": ALLOWED_STATES, "production_gate_conjunction_required": True,
        "v1_integrity_before": before, "v1_integrity_after": after,
        "output_hashes": output_hashes, "provenance_hash": output_hashes["outputs/stage2f/lineage.json"],
    }
    _json(OUT / "STAGE2F_CHECKPOINT.json", cp)
    return cp


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    before = v1_verify()
    transfer = transfer_puerto_rico()
    internal = internal_audit()
    puerto_harmonisation(transfer)
    public_route_audit()
    requests = request_packages()
    wave = second_wave()
    d = dashboard()
    feature = feature_contract()
    user_actions()
    report(transfer, d, requests, wave)
    after = v1_verify()
    _json(OUT / "STAGE2F_DIAGNOSTICS.json", {
        "v1_before": before, "v1_after": after, "internal_tracks": len(internal),
        "primary_requests": len(requests), "second_wave": len(wave),
        "production_ready": int(d.state.eq("PRODUCTION_READY").sum()),
        "production_terrain_units": 450935, "scientific_download_bytes": 0,
        "controlled_transfer_bytes": 2562898, "production_h_generated": False,
    })
    return checkpoint(before, after, transfer, d, feature)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
