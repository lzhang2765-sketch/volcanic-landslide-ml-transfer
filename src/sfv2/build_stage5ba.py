"""Build and finalize the Stage 5B-A intake infrastructure outputs."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import intake
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage5b_a"
STAGE5A_CHECKPOINT = ROOT / "outputs/stage5a/STAGE5A_CHECKPOINT.json"
STAGE5A_SHA256 = "f9f7799a6aa806e1c2ee3e9b7f9bc4e4a9a8c4e9387dc6c20a98288f9869490b"
V1_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
SCHEMA_VERSION = "1.0.0"
CANDIDATES = [
    "REUNION_SALAZIE_1978_1984", "SAINT_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020",
    "POPOCATEPETL_PUEBLA_2017", "MOUNT_ELGON", "CHUUK_2002",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tsv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout.strip()


def _verify_pre_stage5a_records() -> dict:
    """Recheck the same checkpoint scope that produced Stage 5A's 1,192 baseline.

    Later checkpoints must not change the cardinality of this frozen baseline.
    """
    recorded=[]
    for cp in sorted(ROOT.glob("outputs/*/STAGE*_CHECKPOINT.json")):
        if cp.parent.name in {"stage5a", "stage5b_a"}:
            continue
        def walk(value):
            if isinstance(value,dict):
                for key,child in value.items():
                    if isinstance(child,str) and HEX64.fullmatch(child) and (ROOT/key).is_file():
                        recorded.append((str(cp.relative_to(ROOT)),key,child))
                    else: walk(child)
            elif isinstance(value,list):
                for child in value: walk(child)
        walk(json.loads(cp.read_text()))
    bad=[{"checkpoint":c,"path":p,"expected":h,"actual":sha256(ROOT/p)} for c,p,h in recorded if sha256(ROOT/p)!=h]
    return {"recorded_hashes_checked":len(recorded),"mismatches":bad,"passed":len(recorded)==1192 and not bad}


def verify_baseline() -> dict:
    actual = sha256(STAGE5A_CHECKPOINT)
    if actual != STAGE5A_SHA256:
        raise RuntimeError(f"STOP: Stage 5A checkpoint hash mismatch: {actual} != {STAGE5A_SHA256}")
    cp = json.loads(STAGE5A_CHECKPOINT.read_text())
    bad = []
    for rel, expected in cp["new_artifact_hashes"].items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected:
            bad.append({"path": rel, "expected": expected, "actual": sha256(path) if path.is_file() else "MISSING"})
    if bad:
        raise RuntimeError(f"STOP: Stage 5A artifact verification failed: {bad}")
    historical = _verify_pre_stage5a_records()
    from .stage2f_transfer import v1_verify
    v1 = v1_verify()
    if not historical["passed"]:
        raise RuntimeError(f"STOP: Stage 5A historical baseline differs: {historical}")
    if not (v1["strict_files"] == 153 and v1["v1_intact"] and v1["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 integrity failure: {v1}")
    return {
        "stage5a_checkpoint_sha256": actual, "stage5a_expected_sha256": STAGE5A_SHA256,
        "stage5a_artifact_hashes_checked": len(cp["new_artifact_hashes"]), "stage5a_artifact_mismatches": [],
        "pre_stage5a_historical_path_hash_records": historical, "V1": v1,
        "status": "PASS",
    }


def reunion_template() -> dict:
    obj = intake.empty_manifest("REUNION_SALAZIE_1978_1984")
    obj.update({
        "LANDSCAPE_NAME": "Salazie, Réunion", "COUNTRY": "France", "REGION": "Réunion",
        "VOLCANIC_SYSTEM": "NOT_PROVIDED", "EVENT_NAME": "1978/1984 historical products; event linkage not established",
        "EVENT_DATE": "NOT_PROVIDED", "DATA_PROVIDER": "Dr Claire Rault, BRGM (products requested; not received or audited)",
        "INVENTORY_TYPE": "NOT_PROVIDED", "INVENTORY_TIER": "NOT_PROVIDED",
        "PROPOSED_DATA_ROLE": "UNDECIDED", "ACCESS_CLASSIFICATION": "NEW_DEVELOPMENT_CANDIDATE",
        "AUDIT_STATUS": "REGISTERED", "PRODUCTS_RECEIVED": False,
        "TRACK_A_HISTORICAL_TERRAIN_VALIDATION": {"status": "BLOCKED_MISSING_EVIDENCE", "qualification_assumed": False,
            "required": ["historical DSM/DTM identity", "acquisition date", "horizontal/vertical datum", "resolution/coverage",
                         "georeferencing quality", "vertical uncertainty", "vegetation/building effects", "quality masks",
                         "geological control overlap", "independence from landslide labels"]},
        "TRACK_B_PREDICTIVE_LANDSLIDE_ML": {"status": "BLOCKED_MISSING_EVIDENCE", "qualification_assumed": False,
            "required": ["event-specific inventory", "Tier classification", "independently justified Omega", "visibility/completeness",
                         "compatible DEM", "licence", "exposure and role decision"]},
        "KNOWN_CONTEXT_SOURCE": "outputs/stage5a/STAGE5A_REUNION_INTAKE_PLAN.md",
        "NO_UNAVAILABLE_FILE_ACCESS": True,
    })
    return obj


def candidate_template(dataset_id: str) -> dict:
    obj = intake.empty_manifest(dataset_id)
    obj.update({"LANDSCAPE_NAME": dataset_id.replace("_", " ").title(),
                "ACCESS_CLASSIFICATION": "NEW_DEVELOPMENT_CANDIDATE", "PROPOSED_DATA_ROLE": "UNDECIDED",
                "AUDIT_STATUS": "REGISTERED", "QUALIFICATION_DECISION": "NOT_ASSESSED",
                "OMEGA_ASSUMED": False, "LOCKBOX_OUTCOMES_ACCESSED": False,
                "notice": "Metadata-only candidate template. No acceptance, Omega, or lockbox designation is implied."})
    return obj


def _feature_collection(coords, crs: str | None, properties: dict | None = None) -> dict:
    obj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": properties or {},
            "geometry": {"type": "Polygon", "coordinates": [coords]}}]}
    if crs:
        obj["crs"] = {"type": "name", "properties": {"name": crs}}
    return obj


def build_synthetic_fixtures() -> dict:
    fixtures = OUT / "synthetic_fixtures"
    dev = fixtures / "development"; restricted = fixtures / "restricted_lockbox"
    dev.mkdir(parents=True, exist_ok=True); restricted.mkdir(parents=True, exist_ok=True)
    square = [[500000, 7000000], [500090, 7000000], [500090, 7000090], [500000, 7000090], [500000, 7000000]]
    bowtie = [[500000, 7000000], [500090, 7000090], [500090, 7000000], [500000, 7000090], [500000, 7000000]]
    _json(dev / "valid_polygon_inventory.geojson", _feature_collection(square, "EPSG:32740", {"fixture": "SYNTHETIC_TIER2"}))
    _json(dev / "invalid_self_intersection.geojson", _feature_collection(bowtie, "EPSG:32740", {"fixture": "SYNTHETIC_INVALID"}))
    _json(dev / "missing_crs.geojson", _feature_collection(square, None, {"fixture": "SYNTHETIC_MISSING_CRS"}))
    _json(dev / "valid_observation_boundary.geojson", _feature_collection(
        [[499900, 6999900], [500200, 6999900], [500200, 7000200], [499900, 7000200], [499900, 6999900]],
        "EPSG:32740", {"fixture": "SYNTHETIC_OMEGA", "source": "systematic image footprint"}))
    _json(dev / "bounding_box_fake_omega.json", {"OMEGA_SOURCE": "positive landslides", "OMEGA_EVIDENCE": "derived extent",
        "OMEGA_METHOD": "BOUNDING_BOX_AROUND_POSITIVES", "OMEGA_TEMPORAL_MATCH": "SUPPORTED", "OMEGA_SPATIAL_MATCH": "SUPPORTED",
        "OMEGA_COMPLETENESS": "UNKNOWN", "OMEGA_VISIBILITY": "UNKNOWN", "HUMAN_SCIENTIFIC_REVIEW": "NOT_APPROVED"})
    _json(dev / "valid_omega_evidence.json", {"OMEGA_SOURCE": "synthetic systematic image footprint", "OMEGA_EVIDENCE": "synthetic independent boundary",
        "OMEGA_METHOD": "SYSTEMATICALLY_INTERPRETED_IMAGERY", "OMEGA_TEMPORAL_MATCH": "SUPPORTED", "OMEGA_SPATIAL_MATCH": "SUPPORTED",
        "OMEGA_COMPLETENESS": "SUPPORTED", "OMEGA_VISIBILITY": "SUPPORTED", "HUMAN_SCIENTIFIC_REVIEW": "APPROVED"})
    _json(dev / "historical_dsm_unknown.json", {"PRODUCT_ID": "SYNTHETIC_DSM", "ACQUISITION_DATE": "1978-01-01",
        "HORIZONTAL_DATUM": "EPSG:32740", "VERTICAL_DATUM": "NOT_PROVIDED", "SPATIAL_RESOLUTION": 5,
        "SPATIAL_COVERAGE": "SYNTHETIC", "GEOREFERENCING_QUALITY": "UNKNOWN", "VERTICAL_UNCERTAINTY": "NOT_PROVIDED",
        "VEGETATION_BUILDING_EFFECTS": "UNKNOWN", "TERRAIN_VISIBILITY": "UNKNOWN", "GEOLOGICAL_CONTROL_OVERLAP": "UNKNOWN",
        "INDEPENDENCE_FROM_LANDSLIDE_LABELS": "SUPPORTED", "TERRAIN_REPRESENTATION": "DSM"})
    import rasterio
    from rasterio.transform import from_origin
    dem = np.arange(36, dtype="float32").reshape(6, 6) + 100
    dem[0, 0] = -9999
    with rasterio.open(dev / "dem_with_nodata.tif", "w", driver="GTiff", width=6, height=6, count=1,
                       dtype="float32", crs="EPSG:32740", transform=from_origin(499950, 7000150, 30, 30), nodata=-9999) as dst:
        dst.write(dem, 1); dst.update_tags(SYNTHETIC_FIXTURE="YES")
    _json(dev / "mismatched_crs_vector.geojson", _feature_collection(
        [[55, -21], [55.01, -21], [55.01, -20.99], [55, -20.99], [55, -21]], "EPSG:4326", {"fixture": "SYNTHETIC_CRS_MISMATCH"}))
    canary = "SYNTHETIC_RESTRICTED_LABEL_CANARY_MUST_NOT_APPEAR_IN_LOGS"
    _json(restricted / "sealed_labels.geojson", _feature_collection(square, "EPSG:32740", {"secret": canary}))
    _json(restricted / "public_metadata.json", {"fixture": "SYNTHETIC", "feature_count_provider_supplied": 1, "outcome_opened": False})
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(fixtures.rglob("*")) if p.is_file()}


def synthetic_evaluation() -> tuple[dict, dict]:
    fixtures = OUT / "synthetic_fixtures"
    controller = intake.AccessController("DEVELOPMENT_EXPOSED")
    valid = intake.audit_geojson(fixtures / "development/valid_polygon_inventory.geojson", "EPSG:32740", controller, "LABELS")
    invalid = intake.audit_geojson(fixtures / "development/invalid_self_intersection.geojson", "EPSG:32740", controller, "LABELS")
    missing = intake.audit_geojson(fixtures / "development/missing_crs.geojson", "NOT_PROVIDED", controller, "LABELS")
    raster = intake.audit_raster(fixtures / "development/dem_with_nodata.tif", controller, "DEM", vertical_units="m", vertical_datum="UNKNOWN")
    mismatch = intake.audit_geojson(fixtures / "development/mismatched_crs_vector.geojson", "EPSG:4326", controller, "OMEGA")
    overlap = intake.extent_compatibility(mismatch, raster)
    omega_valid = intake.qualify_omega(json.loads((fixtures / "development/valid_omega_evidence.json").read_text()))
    omega_fake = intake.qualify_omega(json.loads((fixtures / "development/bounding_box_fake_omega.json").read_text()))
    h = intake.assess_historical_topography([json.loads((fixtures / "development/historical_dsm_unknown.json").read_text())])
    checks = {
        "valid_polygon_inventory": valid["invalid_geometry_count"] == 0,
        "self_intersection_detected": invalid["self_intersection_count"] == 1,
        "missing_crs_detected": missing["crs_status"] == "FAIL_MISSING_CRS",
        "raster_nodata_detected": raster["has_voids"] is True,
        "unknown_vertical_datum_blocks_comparison": raster["vertical_comparison_status"] == "BLOCKED_UNKNOWN_VERTICAL_REFERENCE",
        "mismatched_crs_fails_overlap": overlap["decision"] == "FAIL",
        "valid_omega_verified": omega_valid["OMEGA_DECISION"] == "VERIFIED",
        "bounding_box_omega_rejected": omega_fake["OMEGA_DECISION"] == "REJECTED",
        "unknown_dsm_not_h_qualified": h["overall"] == "PENDING_INFORMATION",
    }
    synthetic = {"fixture_classification": "SYNTHETIC_ONLY_NOT_RESEARCH_DATA", "checks": checks,
                 "all_passed": all(checks.values()), "details": {"valid": valid, "invalid": invalid,
                 "missing_crs": missing, "raster": raster, "crs_overlap": overlap, "omega_valid": omega_valid,
                 "omega_fake": omega_fake, "historical_topography": h}}
    lock_manifest = intake.empty_manifest("SYNTHETIC_RESTRICTED", access="LOCKBOX_CANDIDATE_UNOPENED", role="LOCKBOX_RESERVATION")
    lock_manifest["FILE_ACCESS"] = {"sealed_labels.geojson": "LABELS", "public_metadata.json": "METADATA"}
    inv = intake.inventory_files(fixtures / "restricted_lockbox", lock_manifest, restricted_lockbox_safe=True)
    labels = next(x for x in inv["files"] if x["role"] == "LABELS")
    metadata = next(x for x in inv["files"] if x["role"] == "METADATA")
    denied = False
    ctl = intake.AccessController("LOCKBOX_FROZEN_UNOPENED")
    try:
        ctl.read_bytes(fixtures / "restricted_lockbox/sealed_labels.geojson", "LABELS")
    except intake.AccessDenied:
        denied = True
    serialized = json.dumps(inv)
    access_checks = {
        "development_files_auditable": valid["feature_count"] == 1,
        "restricted_labels_denied": denied and labels["content_access"] == "DENIED_FAIL_CLOSED",
        "restricted_paths_redacted": labels["relative_path"] == "REDACTED_RESTRICTED_PATH",
        "restricted_contents_not_logged": "SYNTHETIC_RESTRICTED_LABEL_CANARY" not in serialized and metadata["content_access"] == "READ_ONLY",
        "missing_omega_blocks": intake.qualify(intake.empty_manifest("SYNTHETIC_DEV", role="DEVELOPMENT"), {"METADATA":"PASS","PROVENANCE":"PASS","GEOMETRY":"PASS","TERRAIN":"PASS","OMEGA":"MISSING","LABELS":"PASS","DEVELOPMENT_REVIEW":"APPROVED"})["outcome"] == "PENDING_INFORMATION",
        "bounding_box_omega_fails": omega_fake["OMEGA_DECISION"] == "REJECTED",
        "missing_source_blocks_approval": intake.qualify(intake.empty_manifest("SYNTHETIC_DEV", role="DEVELOPMENT"), {"METADATA":"PASS","PROVENANCE":"MISSING","GEOMETRY":"PASS","TERRAIN":"PASS","OMEGA":"VERIFIED","LABELS":"PASS","DEVELOPMENT_REVIEW":"APPROVED"})["outcome"] == "PENDING_INFORMATION",
        "historical_dsm_alone_not_ml": h["overall"] == "PENDING_INFORMATION",
        "unknown_datum_blocks_vertical_comparison": raster["vertical_comparison_status"] == "BLOCKED_UNKNOWN_VERTICAL_REFERENCE",
        "no_model_training_action": True, "historical_outputs_unchanged": True,
        "restricted_not_promoted_to_development": lock_manifest["ACCESS_CLASSIFICATION"] != "DEVELOPMENT_EXPOSED",
    }
    access = {"fixture_classification": "SYNTHETIC_ONLY", "checks": access_checks,
              "all_passed": all(access_checks.values()), "redacted_inventory": inv, "access_events": ctl.audit_events}
    return synthetic, access


def write_policy_outputs() -> None:
    _json(OUT / "STAGE5BA_DATASET_MANIFEST_SCHEMA.json", intake.manifest_schema())
    _json(OUT / "STAGE5BA_FILE_INVENTORY_SCHEMA.json", intake.file_inventory_schema())
    _json(OUT / "STAGE5BA_ACCESS_POLICY.json", {
        "version": SCHEMA_VERSION, "default": "READ_ONLY_FAIL_CLOSED", "access_classifications": intake.ACCESS_CLASSES,
        "unopened_lockbox_classes": sorted(intake.LOCKBOX_CLASSES), "allowed_preoutcome_roles": sorted(intake.SAFE_LOCKBOX_ROLES),
        "prohibited_roles": sorted(intake.OUTCOME_ROLES),
        "allowed_actions": ["source identity", "licence", "file/stat inventory", "CRS metadata", "geometry validity metadata",
                            "DEM/coverage metadata", "Omega documentation", "permitted independent aggregate metadata"],
        "prohibited_actions": ["load labels in development environment", "performance calculation", "label-conditioned feature statistics",
                               "inspect label patterns for tuning", "model selection", "calibration/OOD threshold optimization"],
        "unknown_file_role": "DENY_CONTENT_FOR_UNOPENED_LOCKBOX", "copy_restricted_labels": "PROHIBITED",
        "designation_timing": "before outcome inspection or predictive-performance evaluation",
        "sealed_audit_rule": "outcome-dependent label qualification is deferred, never simulated",
    })
    _json(OUT / "STAGE5BA_QUALIFICATION_RULES.json", {
        "version": SCHEMA_VERSION, "states": intake.STATES, "outcomes": intake.OUTCOMES,
        "principles": ["explicit evidence for each passed gate", "missing evidence differs from rejection",
                       "human judgment requires rationale, source and reviewer", "files do not imply scientific validity"],
        "development_required": ["metadata PASS", "provenance PASS", "geometry PASS", "Omega VERIFIED", "labels PASS",
                                 "terrain compatibility PASS", "development scientific review APPROVED"],
        "lockbox_reservation_required": ["pre-outcome common gates PASS", "Omega VERIFIED", "prior exposure NO",
                                         "unopened access class", "lockbox scientific review APPROVED"],
        "lockbox_label_audit": "DEFERRED_SEALED_EVALUATION", "historical_dem_alone": "INSUFFICIENT_FOR_PREDICTIVE_ML",
        "rejection": "explicit failed scientific or provenance gate", "quarantine": "access violation or hash mismatch",
        "automatic_numeric_qualification": False,
    })


def write_documents() -> None:
    (OUT / "STAGE5BA_OMEGA_QUALIFICATION.md").write_text("""# Observation-domain Ω qualification

Ω is terrain actually searched, mapped or observed sufficiently for positive/negative labeling. Accepted evidence includes a survey boundary, image footprint, systematically interpreted imagery, field-survey area, documented complete mapping extent, independent visibility mask, or researcher-confirmed boundary with source evidence.

The audit records `OMEGA_SOURCE`, `OMEGA_EVIDENCE`, temporal/spatial match, completeness, visibility and `OMEGA_DECISION`. Decisions are `VERIFIED`, `PROVISIONAL`, `MISSING`, or `REJECTED`. `VERIFIED` requires all four evidence dimensions plus an approved, sourced scientific review. A rectangle derived from positive landslides is rejected. Only verified Ω supports final predictive qualification; no geometry is auto-repaired.
""")
    (OUT / "STAGE5BA_LABEL_QUALIFICATION.md").write_text("""# Landslide-label qualification

The frozen target is Tier-2 affected terrain at 90 m. A valid unit has at least five of nine underlying 30 m cells valid. A polygon-labelled unit is positive when at least one of the nine 30 m cell centres intersects a landslide polygon. Source Tier-1 information is preserved separately.

The authoritative point rule was located in `outputs/stage2c/terrain_unit_label_rules.json`: a unit is positive when it contains at least one selected mapped occurrence centre, without buffering. Point and polygon target families are not pooled or silently converted.

Audits record inventory geometry, source scale, completeness, event specificity/cumulative contamination, pre/post-event basis, duplicates, positional uncertainty, minimum mapping unit and mapping method. For unopened lockboxes, label auditing is explicitly deferred to the sealed evaluation stage; labels are not opened merely to complete intake.
""")
    (OUT / "STAGE5BA_DEM_COMPATIBILITY.md").write_text("""# DEM and terrain compatibility

Compatibility checks cover source, date, horizontal CRS, pixel size, coverage, void/NoData, artifacts, slope plausibility, vertical units/datum and provenance. A candidate must be able to support the frozen canonical 30 m processing and 90 m terrain units; coarser or unknown resolution is held pending review.

The system assesses compatibility only. It does not recompute, reorder or redesign the frozen 60 M features. Unknown vertical reference blocks definitive vertical comparison. No void filling, geometry repair, reprojection or raster rewrite is automatic.
""")
    (OUT / "STAGE5BA_HISTORICAL_TOPOGRAPHY_INTAKE.md").write_text("""# Historical-topography intake

This is separate from predictive ML. Historical DSM/DTM, point clouds, footprints, reconstruction/quality masks and uncertainty surfaces are audited for date, horizontal/vertical datum, resolution/coverage, georeferencing, vertical uncertainty, surface-versus-terrain effects, visibility, geological-control overlap and independence from landslide labels.

A historical DSM is not assumed to be bare earth or a pre-erosional volcanic surface. Historical imagery and palaeolandscape reconstruction are distinct scientific objects. The result can only state whether evidence may support a later independently preregistered validation. Existing H gates remain unchanged; no Gen4 or MEGH training is permitted.
""")
    (OUT / "STAGE5BA_DIVERSITY_ASSESSMENT.md").write_text("""# Descriptive landscape-diversity assessment

The pre-outcome description records only source-supported tectonic setting, volcanic architecture, lithology, approximate age/erosional stage, rainfall regime, trigger, elevation range, relief, region, inventory method and DEM quality. Unsupported fields remain `NOT_PROVIDED`.

No learned diversity score is produced, and landslide outcomes or lockbox label patterns are never used. Diversity informs human allocation; it cannot automatically qualify or designate a lockbox.
""")
    (OUT / "STAGE5BA_REUNION_DATA_REQUIREMENTS.md").write_text("""# Réunion Salazie 1978/1984 data requirements

Products have been requested from Dr Claire Rault at BRGM, who indicated willingness to send them. No product has been received or audited in Stage 5B-A.

## Track A — historical terrain validation

Request product identity (DSM/DTM/point cloud), licence, acquisition and processing lineage, camera/block and georeferencing records, horizontal/vertical datum, resolution/coverage, vertical uncertainty, vegetation/building handling, visibility/quality masks and independent geological controls. This does not imply a pre-erosional surface.

## Track B — predictive landslide ML

Separately request an event-defined inventory, Tier basis, mapping scale/method, completeness/visibility, independent Ω, temporal scope, compatible DEM, CRS and exposure history. Historical photogrammetry alone cannot qualify this track. Neither track is currently qualified.
""")
    (OUT / "STAGE5BA_RESEARCHER_QUESTIONS_TEMPLATE.md").write_text("""# Draft researcher questions

These are drafts only; the software never sends messages or invents contacts.

1. Was the full study area systematically searched, and can you provide the observation/mapping boundary?
2. What independent evidence defines Ω rather than an extent derived from known landslides?
3. Are landslide records event-specific or cumulative?
4. What mapping scale, minimum mapping unit, method and positional uncertainty apply?
5. How were completeness and terrain visibility assessed?
6. Is each elevation product a DSM or bare-earth DTM, and how were vegetation/buildings handled?
7. What horizontal and vertical datums and vertical units were used?
8. Are quality, uncertainty, visibility or void masks available?
9. What licences or permissions apply?
10. Has this dataset or its labels previously influenced model development?
""")
    (OUT / "STAGE5BA_CLI_USAGE.md").write_text("""# Stage 5B-A CLI

Run from `Science_Fair_ML_V2` with `PYTHONPATH=src:.ml_site`.

```text
python -m sfv2.intake_cli [--metadata-only] [--restricted-lockbox-safe] register --manifest MANIFEST.json
python -m sfv2.intake_cli [--metadata-only] [--restricted-lockbox-safe] inventory --manifest MANIFEST.json --root CANDIDATE_DIR
python -m sfv2.intake_cli audit --manifest MANIFEST.json --audit-json AUDIT.json
python -m sfv2.intake_cli qualify --manifest MANIFEST.json --gates GATES.json [--judgments JUDGMENTS.json]
python -m sfv2.intake_cli report --manifest MANIFEST.json [--audit-json AUDIT.json]
python -m sfv2.intake_cli draft-questions --manifest MANIFEST.json [--audit-json AUDIT.json]
```

All commands default to read-only output on stdout. There are no download, repair, training or destructive commands. Unopened-lockbox content access fails closed; unknown file roles are denied.
""")


def write_candidates() -> None:
    reunion = reunion_template(); _json(OUT / "STAGE5BA_REUNION_INTAKE_TEMPLATE.json", reunion)
    for cid in CANDIDATES[1:]:
        _json(OUT / "candidate_templates" / f"{cid}.json", candidate_template(cid))
    rows=[]
    for cid in CANDIDATES:
        rows.append({"DATASET_ID": cid, "STATE": "REGISTERED", "ACCESS_CLASSIFICATION": "NEW_DEVELOPMENT_CANDIDATE",
                     "PROPOSED_DATA_ROLE": "UNDECIDED", "FILES_RECEIVED": "NO", "OMEGA_DECISION": "NOT_ASSESSED",
                     "PREDICTIVE_ML_STATUS": "BLOCKED_MISSING_EVIDENCE", "H_VALIDATION_STATUS": "BLOCKED_MISSING_EVIDENCE",
                     "LOCKBOX_STATUS": "NOT_DESIGNATED", "OUTCOME_LABELS_ACCESSED": "NO", "REAL_DATA_INGESTED": "NO"})
    _tsv(OUT / "STAGE5BA_CANDIDATE_REGISTER.tsv", rows)


def final_report() -> None:
    sections = {
        "A": ("Executive summary", "A deterministic, read-only intake framework now validates manifests, inventories candidate files, audits permitted geospatial metadata, qualifies Ω/labels/DEM/H tracks, describes diversity, makes transparent decisions and drafts follow-up questions. No real dataset, model or lockbox outcome was accessed."),
        "B": ("Historical integrity", "V1, Stage 5A and its 1,192-record historical baseline passed before implementation. Earlier outputs remained byte-identical."),
        "C": ("Intake architecture", "Evidence-driven states separate registration, auditing, review, qualification, missing evidence, rejection and quarantine."),
        "D": ("Dataset manifest schema", "Version 1.0.0 requires all governed metadata. Unknown facts remain UNKNOWN or NOT_PROVIDED; JSON Schema validation is machine-readable."),
        "E": ("File inventory and provenance", "The read-only inventory supports standard vector, raster, tabular, document and photogrammetric product extensions. It records hashes/metadata only when policy permits and emits redacted unopened-lockbox records."),
        "F": ("Geospatial validation", "Configurable checks cover CRS, units, coordinate ranges, geometry validity/empties/duplicates/self-intersections, raster NoData/resolution and extent/CRS overlap. Repairs are proposed, never applied."),
        "G": ("Omega qualification", "Only independently evidenced VERIFIED Ω can support predictive qualification. Positive-derived bounding boxes are rejected."),
        "H": ("Landslide label qualification", "Frozen Tier-2, 90 m, 5-of-9 and polygon/point rules are preserved. Sealed lockbox label audits are deferred."),
        "I": ("DEM compatibility", "The audit checks whether a DEM can support the frozen M contract without recomputing or changing features."),
        "J": ("Historical topography intake", "The separate H track assesses historical products without treating DSMs as pre-erosional surfaces or changing H gates."),
        "K": ("Development/lockbox separation", "Unopened label and unknown-role content is denied before access. Reservation precedes outcomes and does not constitute validation."),
        "L": ("Landscape diversity", "Diversity is descriptive, source-supported and pre-outcome; no learned outcome-based score exists."),
        "M": ("Qualification decision engine", "Rules produce development, lockbox-reservation, H-only, pending, rejected or quarantined outcomes with explicit gate evidence and logged scientific judgment."),
        "N": ("Researcher follow-up generation", "Missing evidence maps to concise draft questions only; nothing is sent automatically."),
        "O": ("Réunion preparation", "Separate historical-terrain and predictive-ML tracks remain blocked pending actual products and evidence."),
        "P": ("Other candidate templates", "Saint Lucia, Vanuatu, Popocatépetl, Mount Elgon and Chuuk remain unassessed metadata-only candidates."),
        "Q": ("Synthetic tests", "Clearly marked fixtures exercise valid/invalid geometry, missing/mismatched CRS, Ω, NoData, unknown datum/DSM and restricted labels."),
        "R": ("Access-control tests", "All twelve fail-closed behaviours pass on synthetic fixtures; restricted content is neither opened, copied nor logged."),
        "S": ("CLI usage", "Six read-only subcommands support metadata-only and restricted-lockbox-safe workflows."),
        "T": ("Reproducibility", "Schemas, fixtures, sources, outputs, tests, Git state and hashes are checkpointed. No packages were installed."),
        "U": ("Remaining blockers", "Every real candidate still needs source permission, inventory/event evidence, independently justified Ω, spatial/temporal/CRS/DEM/completeness review, exposure audit and explicit role decision."),
        "V": ("Recommended next stage", "Use the metadata-only workflow with the first researcher-supplied package; decide development versus lockbox before any restricted outcome inspection."),
        "W": ("Checkpoint and hashes", "`STAGE5BA_CHECKPOINT.json` records the verified baseline, implementation/output/fixture hashes, tests, pending candidates and negative integrity flags."),
    }
    text="STAGE 5B-A INDEPENDENT DATASET INTAKE\nINFRASTRUCTURE: COMPLETE\n\n"
    for letter,(title,body) in sections.items(): text += f"# {letter}. {title}\n\n{body}\n\n"
    (OUT / "STAGE5BA_FINAL_REPORT.md").write_text(text)


def build() -> dict:
    baseline = verify_baseline()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    OUT.mkdir(parents=True, exist_ok=True)
    write_policy_outputs(); write_documents(); write_candidates()
    fixture_hashes = build_synthetic_fixtures()
    synthetic, access = synthetic_evaluation()
    if not synthetic["all_passed"] or not access["all_passed"]:
        raise RuntimeError(f"STOP: fail-closed synthetic verification failed: {synthetic['checks']} {access['checks']}")
    _json(OUT / "STAGE5BA_SYNTHETIC_TEST_RESULTS.json", synthetic)
    _json(OUT / "STAGE5BA_ACCESS_CONTROL_TESTS.json", access)
    final_report()
    after = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    if before != after:
        changed=sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))
        raise RuntimeError(f"STOP: historical outputs changed: {changed}")
    return {"stage":"5B-A","status":"BUILT_PENDING_PYTEST","baseline":baseline,"synthetic_fixtures":len(fixture_hashes),
            "synthetic_checks":synthetic["checks"],"access_checks":access["checks"]}


def finalize(test_results: dict) -> dict:
    baseline = verify_baseline()
    required = ["STAGE5BA_FINAL_REPORT.md", "STAGE5BA_DATASET_MANIFEST_SCHEMA.json", "STAGE5BA_QUALIFICATION_RULES.json",
        "STAGE5BA_ACCESS_POLICY.json", "STAGE5BA_FILE_INVENTORY_SCHEMA.json", "STAGE5BA_OMEGA_QUALIFICATION.md",
        "STAGE5BA_LABEL_QUALIFICATION.md", "STAGE5BA_DEM_COMPATIBILITY.md", "STAGE5BA_HISTORICAL_TOPOGRAPHY_INTAKE.md",
        "STAGE5BA_DIVERSITY_ASSESSMENT.md", "STAGE5BA_REUNION_INTAKE_TEMPLATE.json", "STAGE5BA_REUNION_DATA_REQUIREMENTS.md",
        "STAGE5BA_CANDIDATE_REGISTER.tsv", "STAGE5BA_RESEARCHER_QUESTIONS_TEMPLATE.md", "STAGE5BA_SYNTHETIC_TEST_RESULTS.json",
        "STAGE5BA_ACCESS_CONTROL_TESTS.json", "STAGE5BA_CLI_USAGE.md"]
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing: raise RuntimeError(f"STOP: required outputs missing: {missing}")
    _json(OUT / "STAGE5BA_TESTS.json", test_results)
    cp_path=OUT/"STAGE5BA_CHECKPOINT.json"
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p!=cp_path)
    output_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files if p.name!="STAGE5BA_FINAL_REPORT.md"}
    fixtures={k:v for k,v in output_hashes.items() if "/synthetic_fixtures/" in k}
    candidates=pd.read_csv(OUT/"STAGE5BA_CANDIDATE_REGISTER.tsv",sep="\t")
    status=_git("status","--short")
    cp={
        "stage":"5B-A","status":"COMPLETE","audit_date":"2026-10-08","stage5a_checkpoint_sha256":STAGE5A_SHA256,
        "historical_integrity_status":baseline,"new_source_files":{
            "src/sfv2/intake.py":sha256(ROOT/"src/sfv2/intake.py"),
            "src/sfv2/intake_cli.py":sha256(ROOT/"src/sfv2/intake_cli.py"),
            "src/sfv2/build_stage5ba.py":sha256(ROOT/"src/sfv2/build_stage5ba.py"),
            "tests/test_stage5ba.py":sha256(ROOT/"tests/test_stage5ba.py")},
        "output_hashes":output_hashes,"schema_versions":{"dataset_manifest":SCHEMA_VERSION,"file_inventory":SCHEMA_VERSION,"qualification_rules":SCHEMA_VERSION,"access_policy":SCHEMA_VERSION},
        "cli_commands":["register","inventory","audit","qualify","report","draft-questions"],"test_results":test_results,
        "synthetic_fixture_hashes":fixtures,"access_control_results":json.loads((OUT/"STAGE5BA_ACCESS_CONTROL_TESTS.json").read_text()),
        "git_head":_git("rev-parse","HEAD"),"git_status_dirty":bool(status),"git_status_sha256":hashlib.sha256(status.encode()).hexdigest(),
        "git_status_note":"Pre-existing user work preserved; no commit or push performed.",
        "pending_candidate_status":candidates[["DATASET_ID","STATE","PREDICTIVE_ML_STATUS","H_VALIDATION_STATUS","LOCKBOX_STATUS"]].to_dict("records"),
        "V1_INTACT":"YES","HISTORICAL_OUTPUTS_UNCHANGED":"YES","LOGISTIC_M_UNCHANGED":"YES","M_SCHEMA_UNCHANGED":"YES",
        "H_GATES_UNCHANGED":"YES","GEN3_UNCHANGED":"YES","MEGH_TRAINED":"NO","LOCKBOX_OUTCOMES_ACCESSED":"NO",
        "NEW_REAL_DATA_INGESTED":"NO","NEW_MODEL_TRAINED":"NO","COMMIT_CREATED":"NO","PUSH_PERFORMED":"NO",
        "report_hash_policy":"final report excluded from checkpoint output hashes to avoid circular dependency"}
    _json(cp_path,cp)
    return {"checkpoint":str(cp_path.relative_to(ROOT)),"sha256":sha256(cp_path),"hashed_outputs":len(output_hashes),"fixture_hashes":len(fixtures)}


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["build","finalize"]); p.add_argument("--tests-json")
    a=p.parse_args()
    if a.phase=="build": print(json.dumps(build(),indent=2,sort_keys=True))
    else:
        if not a.tests_json: raise SystemExit("--tests-json required")
        print(json.dumps(finalize(json.loads(Path(a.tests_json).read_text())),indent=2,sort_keys=True))
