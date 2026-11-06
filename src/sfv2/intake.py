"""Scientific dataset-intake primitives for Stage 5B-A.

The module is deliberately model-free.  It audits metadata and candidate files,
enforces access policy before content access, and returns evidence-driven states.
It never repairs or rewrites supplied files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

STATES = [
    "REGISTERED", "FILES_RECEIVED", "METADATA_AUDITED", "PROVENANCE_VERIFIED",
    "GEOMETRY_AUDITED", "OBSERVATION_DOMAIN_VERIFIED", "LABELS_AUDITED",
    "TERRAIN_COMPATIBILITY_VERIFIED", "QUALIFICATION_REVIEW", "QUALIFIED_DEVELOPMENT",
    "QUALIFIED_LOCKBOX_CANDIDATE", "QUALIFIED_H_VALIDATION_ONLY",
    "BLOCKED_MISSING_EVIDENCE", "REJECTED", "QUARANTINED",
]
ACCESS_CLASSES = [
    "DEVELOPMENT_EXPOSED", "NEW_DEVELOPMENT_CANDIDATE", "LOCKBOX_CANDIDATE_UNOPENED",
    "LOCKBOX_FROZEN_UNOPENED", "EXTERNAL_VALIDATION_OPENED",
]
OUTCOMES = [
    "QUALIFIED_FOR_DEVELOPMENT", "ELIGIBLE_FOR_LOCKBOX_RESERVATION",
    "QUALIFIED_FOR_H_VALIDATION_ONLY", "PENDING_INFORMATION", "REJECTED", "QUARANTINED",
]
MANIFEST_REQUIRED = [
    "DATASET_ID", "LANDSCAPE_NAME", "COUNTRY", "REGION", "VOLCANIC_SYSTEM", "EVENT_NAME",
    "EVENT_DATE", "EVENT_TRIGGER_TYPE", "DATA_PROVIDER", "PRIMARY_PUBLICATION", "DATA_LICENSE",
    "DATA_RECEIPT_DATE", "INVENTORY_TYPE", "INVENTORY_TIER", "INVENTORY_TEMPORAL_SCOPE",
    "OBSERVATION_DOMAIN_SOURCE", "OBSERVATION_DOMAIN_METHOD", "DEM_SOURCE", "DEM_DATE",
    "DEM_RESOLUTION", "CRS", "SPATIAL_EXTENT", "KNOWN_COMPLETENESS_LIMITATIONS",
    "VISIBILITY_LIMITATIONS", "PRIOR_DEVELOPMENT_EXPOSURE", "PROPOSED_DATA_ROLE",
    "ACCESS_CLASSIFICATION", "SOURCE_FILE_HASHES", "AUDIT_STATUS",
]
UNKNOWN = {"UNKNOWN", "NOT_PROVIDED"}
LOCKBOX_CLASSES = {"LOCKBOX_CANDIDATE_UNOPENED", "LOCKBOX_FROZEN_UNOPENED"}
SAFE_LOCKBOX_ROLES = {"METADATA", "DOCUMENTATION", "DEM", "OMEGA", "HISTORICAL_TOPOGRAPHY", "QUALITY_MASK"}
OUTCOME_ROLES = {"LABELS", "OUTCOMES", "PERFORMANCE", "LABEL_DERIVED"}


class AccessDenied(PermissionError):
    """Raised before restricted outcome content is opened."""


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def manifest_schema() -> dict:
    properties = {key: {"type": ["string", "number", "object", "array"]} for key in MANIFEST_REQUIRED}
    for key in MANIFEST_REQUIRED:
        properties[key]["description"] = "Use UNKNOWN or NOT_PROVIDED when evidence is absent; never infer a value."
    properties["DATASET_ID"].update({"type": "string", "pattern": "^[A-Z0-9][A-Z0-9_-]+$"})
    properties["ACCESS_CLASSIFICATION"] = {"type": "string", "enum": ACCESS_CLASSES}
    properties["PROPOSED_DATA_ROLE"] = {"type": "string", "enum": [
        "DEVELOPMENT", "LOCKBOX_RESERVATION", "H_VALIDATION_ONLY", "UNDECIDED"]}
    properties["AUDIT_STATUS"] = {"type": "string", "enum": STATES}
    properties["SOURCE_FILE_HASHES"] = {
        "type": "object", "additionalProperties": {"type": "string", "pattern": "^(UNKNOWN|NOT_PROVIDED|REDACTED|[0-9a-f]{64})$"}}
    properties["FILE_ACCESS"] = {"type": "object", "additionalProperties": {
        "type": "string", "enum": sorted(SAFE_LOCKBOX_ROLES | OUTCOME_ROLES | {"UNKNOWN"})}}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "sfv2://stage5ba/dataset-manifest/v1.0.0",
        "title": "Science Fair ML V2 candidate dataset manifest", "type": "object",
        "required": MANIFEST_REQUIRED, "properties": properties, "additionalProperties": True,
        "x-unknown-policy": "Missing knowledge is represented as UNKNOWN or NOT_PROVIDED, never an invented value.",
        "x-schema-version": "1.0.0",
    }


def validate_manifest(manifest: dict, schema: dict | None = None) -> dict:
    schema = schema or manifest_schema()
    errors: list[str] = []
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
        errors.extend(e.message for e in sorted(validator.iter_errors(manifest), key=lambda x: list(x.path)))
    except ImportError:
        errors.extend(f"missing required field: {k}" for k in MANIFEST_REQUIRED if k not in manifest)
    if manifest.get("ACCESS_CLASSIFICATION") in LOCKBOX_CLASSES and manifest.get("PRIOR_DEVELOPMENT_EXPOSURE") not in {"NO", "UNKNOWN", "NOT_PROVIDED"}:
        errors.append("lockbox candidate has prior development exposure")
    return {"valid": not errors, "errors": errors, "schema_version": schema["x-schema-version"]}


@dataclass
class AccessController:
    access_classification: str
    restricted_safe: bool = True
    audit_events: list[dict] = field(default_factory=list)

    @property
    def sealed(self) -> bool:
        return self.access_classification in LOCKBOX_CLASSES

    def permitted(self, role: str) -> bool:
        role = role.upper()
        if not self.sealed:
            return True
        return role in SAFE_LOCKBOX_ROLES

    def require(self, role: str, action: str = "READ_CONTENT") -> None:
        allowed = self.permitted(role)
        self.audit_events.append({"action": action, "role": role.upper(), "decision": "ALLOW" if allowed else "DENY",
                                  "content_logged": False})
        if not allowed:
            raise AccessDenied(f"access denied by unopened-lockbox policy for role {role.upper()}")

    def read_bytes(self, path: Path, role: str) -> bytes:
        self.require(role)
        return path.read_bytes()


FORMAT_MAP = {
    ".gpkg": "GEOPACKAGE", ".shp": "SHAPEFILE", ".shx": "SHAPEFILE_COMPONENT",
    ".dbf": "SHAPEFILE_COMPONENT", ".prj": "SHAPEFILE_COMPONENT", ".cpg": "SHAPEFILE_COMPONENT",
    ".geojson": "GEOJSON", ".json": "JSON", ".tif": "GEOTIFF", ".tiff": "GEOTIFF",
    ".nc": "NETCDF", ".csv": "CSV", ".tsv": "TSV", ".xlsx": "EXCEL", ".xls": "EXCEL",
    ".pdf": "PDF_DOCUMENT", ".txt": "TEXT_METADATA", ".md": "TEXT_METADATA",
    ".las": "POINT_CLOUD", ".laz": "POINT_CLOUD", ".vrt": "RASTER_VIRTUAL",
}


def file_inventory_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "sfv2://stage5ba/file-inventory/v1.0.0",
        "type": "object", "required": ["schema_version", "dataset_id", "access_classification", "files"],
        "properties": {
            "schema_version": {"const": "1.0.0"}, "dataset_id": {"type": "string"},
            "access_classification": {"enum": ACCESS_CLASSES}, "redacted": {"type": "boolean"},
            "files": {"type": "array", "items": {"type": "object", "required": [
                "relative_path", "extension", "bytes", "sha256", "detected_format", "role",
                "content_access", "readable_metadata", "crs", "raster_dimensions", "vector_layers"]}},
        }, "additionalProperties": False, "x-schema-version": "1.0.0",
    }


def _geojson_metadata(path: Path) -> dict:
    obj = json.loads(path.read_text())
    features = obj.get("features", []) if obj.get("type") == "FeatureCollection" else []
    types = sorted({(f.get("geometry") or {}).get("type", "EMPTY") for f in features})
    crs = (obj.get("crs") or {}).get("properties", {}).get("name", "NOT_PROVIDED")
    return {"feature_count": len(features), "geometry_types": types, "crs": crs, "layers": ["default"]}


def _gpkg_metadata(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT table_name, data_type, srs_id FROM gpkg_contents ORDER BY table_name").fetchall()
    return {"layers": [r[0] for r in rows], "layer_types": [r[1] for r in rows], "srs_ids": [r[2] for r in rows]}


def _raster_metadata(path: Path) -> dict:
    import rasterio
    with rasterio.open(path) as src:
        return {"crs": str(src.crs) if src.crs else "NOT_PROVIDED", "dimensions": [src.height, src.width, src.count],
                "pixel_size": [abs(src.transform.a), abs(src.transform.e)], "nodata": src.nodata,
                "bounds": list(src.bounds), "dtypes": list(src.dtypes)}


def _ogr_metadata(path: Path) -> dict:
    import pyogrio
    layers = pyogrio.list_layers(path).tolist()
    details=[]
    for name, geometry_type in layers:
        info=pyogrio.read_info(path, layer=name)
        details.append({"name":name,"geometry_type":geometry_type,"feature_count":int(info["features"]),
                        "crs":str(info["crs"]) if info["crs"] else "NOT_PROVIDED",
                        "fields":list(map(str,info["fields"])),"bounds":list(map(float,info["total_bounds"]))})
    crs_values=sorted({x["crs"] for x in details})
    return {"layers":[x["name"] for x in details],"layer_details":details,
            "crs":crs_values[0] if len(crs_values)==1 else crs_values}


def _netcdf_metadata(path: Path) -> dict:
    from scipy.io import netcdf_file
    with netcdf_file(path,"r",mmap=False) as src:
        return {"dimensions":{str(k):int(v) for k,v in src.dimensions.items()},
                "variables":sorted(map(str,src.variables)),"crs":"REQUIRES_CF_METADATA_REVIEW"}


def _readable_metadata(path: Path, fmt: str) -> dict:
    if fmt == "GEOJSON":
        return _geojson_metadata(path)
    if fmt == "GEOPACKAGE":
        try:
            return _ogr_metadata(path)
        except Exception:
            return _gpkg_metadata(path)
    if fmt == "SHAPEFILE":
        return _ogr_metadata(path)
    if fmt == "GEOTIFF":
        return _raster_metadata(path)
    if fmt == "NETCDF":
        return _netcdf_metadata(path)
    if fmt in {"CSV", "TSV"}:
        with path.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f, delimiter="\t" if fmt == "TSV" else ","), [])
        return {"columns": header}
    if fmt == "EXCEL":
        import openpyxl
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        names = book.sheetnames; book.close()
        return {"sheet_names": names}
    if fmt in {"PDF_DOCUMENT", "TEXT_METADATA", "JSON"}:
        return {"document": True}
    return {"reader_status": "METADATA_READER_NOT_AVAILABLE"}


def inventory_files(root: Path, manifest: dict, *, metadata_only: bool = False,
                    restricted_lockbox_safe: bool = True) -> dict:
    root = root.resolve()
    access = AccessController(manifest["ACCESS_CLASSIFICATION"], restricted_lockbox_safe)
    roles = manifest.get("FILE_ACCESS", {})
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        role = str(roles.get(rel, "UNKNOWN")).upper()
        fmt = FORMAT_MAP.get(path.suffix.lower(), "UNKNOWN")
        allowed = access.permitted(role)
        if access.sealed and not allowed:
            access.audit_events.append({"action": "INVENTORY_CONTENT", "role": role, "decision": "DENY", "content_logged": False})
            digest, metadata, content_access = "REDACTED", {}, "DENIED_FAIL_CLOSED"
        elif metadata_only:
            digest, metadata, content_access = "NOT_COMPUTED_METADATA_ONLY", {}, "METADATA_ONLY"
        else:
            access.require(role, "INVENTORY_CONTENT")
            digest = sha256(path)
            try:
                metadata = _readable_metadata(path, fmt)
            except Exception as exc:
                metadata = {"reader_status": "UNREADABLE_METADATA", "error_type": type(exc).__name__}
            content_access = "READ_ONLY"
        rows.append({
            "relative_path": rel if allowed or not access.sealed else "REDACTED_RESTRICTED_PATH",
            "extension": path.suffix.lower(), "bytes": path.stat().st_size, "sha256": digest,
            "detected_format": fmt, "role": role, "content_access": content_access,
            "readable_metadata": metadata,
            "crs": metadata.get("crs", "NOT_INSPECTED") if metadata else "REDACTED",
            "raster_dimensions": metadata.get("dimensions", []) if metadata else [],
            "vector_layers": metadata.get("layers", []) if metadata else [],
            "permission_access_classification": manifest["ACCESS_CLASSIFICATION"],
        })
    return {"schema_version": "1.0.0", "dataset_id": manifest["DATASET_ID"],
            "access_classification": manifest["ACCESS_CLASSIFICATION"], "redacted": access.sealed,
            "files": rows, "access_events": access.audit_events}


def audit_geojson(path: Path, declared_crs: str, controller: AccessController, role: str) -> dict:
    controller.require(role, "GEOMETRY_AUDIT")
    obj = json.loads(path.read_text())
    from shapely.geometry import shape
    from shapely.validation import explain_validity
    features = obj.get("features", [])
    geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
    wkbs = [g.wkb_hex for g in geoms]
    invalid = [{"record": i, "reason": explain_validity(g)} for i, g in enumerate(geoms) if not g.is_valid]
    bounds = [g.bounds for g in geoms if not g.is_empty]
    combined = [min(x[0] for x in bounds), min(x[1] for x in bounds), max(x[2] for x in bounds), max(x[3] for x in bounds)] if bounds else []
    geographic = "4326" in str(declared_crs).upper() or "WGS84" in str(declared_crs).upper()
    suspicious = bool(combined and geographic and (combined[0] < -180 or combined[2] > 180 or combined[1] < -90 or combined[3] > 90))
    return {
        "crs_status": "PASS" if declared_crs not in UNKNOWN else "FAIL_MISSING_CRS",
        "crs": declared_crs, "coordinate_units": "GEOGRAPHIC_DEGREES" if geographic else "PROJECTED_OR_UNKNOWN",
        "feature_count": len(features), "geometry_types": sorted({g.geom_type for g in geoms}),
        "empty_geometry_count": sum(g.is_empty for g in geoms), "invalid_geometry_count": len(invalid),
        "invalid_geometry_details": invalid, "duplicate_geometry_count": len(wkbs) - len(set(wkbs)),
        "self_intersection_count": sum("Self-intersection" in x["reason"] for x in invalid),
        "bounds": combined, "suspicious_coordinate_ranges": suspicious,
        "repair_performed": False, "proposed_repairs": ["repair only in a versioned copy after scientific review"] if invalid else [],
    }


def audit_raster(path: Path, controller: AccessController, role: str, *, vertical_units: str = "UNKNOWN",
                 vertical_datum: str = "UNKNOWN") -> dict:
    controller.require(role, "RASTER_AUDIT")
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1, masked=True)
        valid = a.compressed().astype(float)
        res = [abs(src.transform.a), abs(src.transform.e)]
        pixel_ok = max(res) <= 30.000001
        nodata_fraction = float(np.ma.getmaskarray(a).sum() / a.size)
        slope_status = "NOT_VERIFIABLE_VERTICAL_UNITS_UNKNOWN"
        slope_p99 = None
        if valid.size and vertical_units.upper() in {"M", "METRE", "METRES", "METER", "METERS"}:
            filled = a.filled(np.nan).astype(float)
            gy, gx = np.gradient(filled, res[1], res[0])
            slopes = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
            slope_p99 = float(np.nanpercentile(slopes, 99)) if np.isfinite(slopes).any() else None
            slope_status = "PASS_PLAUSIBLE" if slope_p99 is not None and slope_p99 <= 89.9 else "FLAG_REVIEW"
        return {
            "crs": str(src.crs) if src.crs else "NOT_PROVIDED", "crs_status": "PASS" if src.crs else "FAIL_MISSING_CRS",
            "dimensions": [src.height, src.width, src.count], "pixel_size": res,
            "resolution_compatible_with_30m_contract": pixel_ok, "nodata": src.nodata,
            "nodata_fraction": nodata_fraction, "has_voids": nodata_fraction > 0,
            "bounds": list(src.bounds), "vertical_units": vertical_units, "vertical_datum": vertical_datum,
            "vertical_comparison_status": "PASS_METADATA_PRESENT" if vertical_units not in UNKNOWN and vertical_datum not in UNKNOWN else "BLOCKED_UNKNOWN_VERTICAL_REFERENCE",
            "slope_plausibility": slope_status, "slope_p99_deg": slope_p99,
            "artifact_review": "REQUIRES_HUMAN_REVIEW", "repair_performed": False,
        }


def extent_compatibility(vector_audit: dict, raster_audit: dict) -> dict:
    vb, rb = vector_audit.get("bounds", []), raster_audit.get("bounds", [])
    same_crs = vector_audit.get("crs") == raster_audit.get("crs")
    overlap = False
    if same_crs and len(vb) == len(rb) == 4:
        overlap = max(vb[0], rb[0]) < min(vb[2], rb[2]) and max(vb[1], rb[1]) < min(vb[3], rb[3])
    return {"crs_consistent": same_crs, "bounds_overlap": overlap,
            "decision": "PASS" if same_crs and overlap else "FAIL", "repair_performed": False}


def qualify_omega(evidence: dict | None) -> dict:
    if not evidence:
        return {"OMEGA_SOURCE": "NOT_PROVIDED", "OMEGA_EVIDENCE": "NOT_PROVIDED", "OMEGA_TEMPORAL_MATCH": "UNKNOWN",
                "OMEGA_SPATIAL_MATCH": "UNKNOWN", "OMEGA_COMPLETENESS": "UNKNOWN", "OMEGA_VISIBILITY": "UNKNOWN", "OMEGA_DECISION": "MISSING"}
    method = str(evidence.get("OMEGA_METHOD", evidence.get("OBSERVATION_DOMAIN_METHOD", "NOT_PROVIDED"))).upper()
    if method in {"POSITIVE_DERIVED_BOUNDING_BOX", "LANDSLIDE_BOUNDING_BOX", "BOUNDING_BOX_AROUND_POSITIVES"}:
        decision = "REJECTED"
    else:
        supported = all(str(evidence.get(k, "UNKNOWN")).upper() in {"SUPPORTED", "VERIFIED"} for k in
                        ("OMEGA_TEMPORAL_MATCH", "OMEGA_SPATIAL_MATCH", "OMEGA_COMPLETENESS", "OMEGA_VISIBILITY"))
        source_ok = str(evidence.get("OMEGA_SOURCE", "NOT_PROVIDED")).upper() not in UNKNOWN
        evidence_ok = str(evidence.get("OMEGA_EVIDENCE", "NOT_PROVIDED")).upper() not in UNKNOWN
        approved = evidence.get("HUMAN_SCIENTIFIC_REVIEW") == "APPROVED"
        decision = "VERIFIED" if supported and source_ok and evidence_ok and approved else "PROVISIONAL"
    return {k: evidence.get(k, "NOT_PROVIDED") for k in (
        "OMEGA_SOURCE", "OMEGA_EVIDENCE", "OMEGA_TEMPORAL_MATCH", "OMEGA_SPATIAL_MATCH", "OMEGA_COMPLETENESS", "OMEGA_VISIBILITY")} | {
        "OMEGA_DECISION": decision, "positive_derived_domain_prohibited": True}


def qualify_labels(manifest: dict, label_evidence: dict | None, *, point_rule_located: bool) -> dict:
    if manifest["ACCESS_CLASSIFICATION"] in LOCKBOX_CLASSES:
        return {"decision": "DEFERRED_SEALED_EVALUATION", "labels_opened": False,
                "reason": "outcome-dependent label audit prohibited before lockbox opening"}
    inventory_type = str((label_evidence or {}).get("INVENTORY_TYPE", manifest.get("INVENTORY_TYPE", "UNKNOWN"))).upper()
    if inventory_type.startswith("POINT") and not point_rule_located:
        return {"decision": "BLOCKED", "labels_opened": False, "reason": "authoritative point conversion rule not located"}
    if not label_evidence:
        return {"decision": "MISSING", "labels_opened": False, "reason": "label evidence not provided"}
    required = ["SOURCE_MAPPING_SCALE", "INVENTORY_COMPLETENESS", "EVENT_SPECIFICITY", "CUMULATIVE_EVENT_CONTAMINATION",
                "PRE_POST_EVENT_STATUS", "DUPLICATE_MAPPING", "POSITIONAL_UNCERTAINTY", "MINIMUM_MAPPING_UNIT", "MAPPING_METHOD"]
    missing = [x for x in required if str(label_evidence.get(x, "NOT_PROVIDED")).upper() in UNKNOWN]
    tier = str(label_evidence.get("INVENTORY_TIER", manifest.get("INVENTORY_TIER", "UNKNOWN"))).upper()
    return {"decision": "PASS" if not missing and tier in {"TIER_1", "TIER_2"} else "MISSING_EVIDENCE",
            "inventory_type": inventory_type, "source_tier_preserved": tier, "target": "TIER_2_AFFECTED_TERRAIN",
            "terrain_resolution": "90m", "valid_unit_rule": ">=5 of 9 underlying 30m cells valid",
            "polygon_positive_rule": ">=1 of nine 30m cell centres intersects a landslide polygon",
            "point_rule": "contains >=1 selected mapped occurrence centre; no buffer" if inventory_type.startswith("POINT") else "NOT_APPLICABLE",
            "missing_evidence": missing, "labels_opened": True}


def qualify_dem(manifest: dict, evidence: dict | None) -> dict:
    evidence = evidence or {}
    required = ["COVERAGE", "TEMPORAL_SUITABILITY", "VOIDS", "RASTER_ARTIFACTS", "SLOPE_PLAUSIBILITY",
                "VERTICAL_UNITS", "VERTICAL_DATUM", "SOURCE_PROVENANCE"]
    missing = [k for k in required if str(evidence.get(k, "NOT_PROVIDED")).upper() in UNKNOWN]
    try:
        resolution = float(manifest["DEM_RESOLUTION"])
        resolution_ok = resolution <= 30.000001
    except (TypeError, ValueError):
        resolution_ok = False
    crs_ok = str(manifest.get("CRS", "UNKNOWN")).upper() not in UNKNOWN
    return {"decision": "COMPATIBLE" if not missing and resolution_ok and crs_ok and evidence.get("HUMAN_SCIENTIFIC_REVIEW") == "APPROVED" else "PENDING",
            "supports_frozen_60_feature_contract": bool(not missing and resolution_ok and crs_ok),
            "resolution_compatible": resolution_ok, "crs_present": crs_ok, "missing_evidence": missing,
            "feature_recomputation_performed": False, "feature_redesign_performed": False}


def assess_historical_topography(products: list[dict] | None) -> dict:
    products = products or []
    required = ["ACQUISITION_DATE", "HORIZONTAL_DATUM", "VERTICAL_DATUM", "SPATIAL_RESOLUTION", "SPATIAL_COVERAGE",
                "GEOREFERENCING_QUALITY", "VERTICAL_UNCERTAINTY", "VEGETATION_BUILDING_EFFECTS", "TERRAIN_VISIBILITY",
                "GEOLOGICAL_CONTROL_OVERLAP", "INDEPENDENCE_FROM_LANDSLIDE_LABELS"]
    assessments = []
    for item in products:
        missing = [k for k in required if str(item.get(k, "NOT_PROVIDED")).upper() in UNKNOWN]
        terrain_known = str(item.get("TERRAIN_REPRESENTATION", "UNKNOWN")).upper() in {"DTM", "BARE_EARTH"}
        assessments.append({"PRODUCT_ID": item.get("PRODUCT_ID", "UNKNOWN"), "missing_evidence": missing,
                            "vertical_comparison_blocked": "VERTICAL_DATUM" in missing or "VERTICAL_UNCERTAINTY" in missing,
                            "terrain_representation_known": terrain_known,
                            "decision": "POTENTIALLY_SUITABLE" if not missing and terrain_known and item.get("HUMAN_SCIENTIFIC_REVIEW") == "APPROVED" else "PENDING_INFORMATION"})
    return {"track": "HISTORICAL_TOPOGRAPHY_SEPARATE_FROM_PREDICTIVE_ML", "products": assessments,
            "overall": "POTENTIALLY_SUITABLE" if assessments and all(x["decision"] == "POTENTIALLY_SUITABLE" for x in assessments) else "PENDING_INFORMATION",
            "not_preerosional_surface": True, "H_gates_changed": False, "Gen4_created": False, "MEGH_trained": False}


def assess_diversity(attributes: dict | None) -> dict:
    keys = ["TECTONIC_SETTING", "VOLCANIC_ARCHITECTURE", "DOMINANT_LITHOLOGY", "AGE_EROSIONAL_STAGE", "RAINFALL_REGIME",
            "TRIGGER_TYPE", "ELEVATION_RANGE", "RELIEF", "GEOGRAPHIC_REGION", "INVENTORY_METHOD", "DEM_QUALITY"]
    attributes = attributes or {}
    rows = [{"attribute": k, "value": attributes.get(k, "NOT_PROVIDED"),
             "source": attributes.get(f"{k}_SOURCE", "NOT_PROVIDED"),
             "status": "SUPPORTED" if str(attributes.get(k, "NOT_PROVIDED")).upper() not in UNKNOWN and str(attributes.get(f"{k}_SOURCE", "NOT_PROVIDED")).upper() not in UNKNOWN else "MISSING_EVIDENCE"} for k in keys]
    return {"assessment_type": "DESCRIPTIVE_PRE_OUTCOME", "attributes": rows, "learned_score": None,
            "landslide_outcomes_used": False, "supported_count": sum(x["status"] == "SUPPORTED" for x in rows)}


def qualify(manifest: dict, gates: dict, human_judgments: list[dict] | None = None) -> dict:
    judgments = human_judgments or []
    unlogged = [j for j in judgments if not all(str(j.get(k, "")).strip() for k in ("JUDGMENT", "RATIONALE", "SOURCE", "REVIEWER"))]
    if unlogged:
        return {"outcome": "PENDING_INFORMATION", "state": "BLOCKED_MISSING_EVIDENCE", "reason": "human judgment lacks rationale/source/reviewer", "gate_record": gates}
    if gates.get("ACCESS_VIOLATION") or gates.get("HASH_MISMATCH"):
        return {"outcome": "QUARANTINED", "state": "QUARANTINED", "reason": "access or integrity violation", "gate_record": gates}
    if gates.get("PROVENANCE") == "REJECTED" or gates.get("OMEGA") == "REJECTED":
        return {"outcome": "REJECTED", "state": "REJECTED", "reason": "explicit scientific gate rejection", "gate_record": gates}
    role = manifest.get("PROPOSED_DATA_ROLE", "UNDECIDED")
    common = all(gates.get(k) == "PASS" for k in ("METADATA", "PROVENANCE", "GEOMETRY", "TERRAIN"))
    if role == "LOCKBOX_RESERVATION":
        eligible = common and gates.get("OMEGA") == "VERIFIED" and gates.get("PRIOR_EXPOSURE") == "NO" and gates.get("LOCKBOX_REVIEW") == "APPROVED" and manifest.get("ACCESS_CLASSIFICATION") in LOCKBOX_CLASSES
        if eligible:
            return {"outcome": "ELIGIBLE_FOR_LOCKBOX_RESERVATION", "state": "QUALIFIED_LOCKBOX_CANDIDATE", "reason": "pre-outcome gates and reservation review pass; external validation not performed", "gate_record": gates}
    if role == "DEVELOPMENT":
        eligible = common and gates.get("OMEGA") == "VERIFIED" and gates.get("LABELS") == "PASS" and gates.get("DEVELOPMENT_REVIEW") == "APPROVED"
        if eligible:
            return {"outcome": "QUALIFIED_FOR_DEVELOPMENT", "state": "QUALIFIED_DEVELOPMENT", "reason": "all predictive gates and scientific review pass", "gate_record": gates}
    if role == "H_VALIDATION_ONLY" and gates.get("HISTORICAL_TOPOGRAPHY") == "POTENTIALLY_SUITABLE" and gates.get("H_REVIEW") == "APPROVED":
        return {"outcome": "QUALIFIED_FOR_H_VALIDATION_ONLY", "state": "QUALIFIED_H_VALIDATION_ONLY", "reason": "historical-topography track only; no predictive qualification", "gate_record": gates}
    missing = sorted(k for k, v in gates.items() if v in {"MISSING", "NOT_PROVIDED", "PENDING", "UNKNOWN", "PROVISIONAL", "DEFERRED_SEALED_EVALUATION"})
    return {"outcome": "PENDING_INFORMATION", "state": "BLOCKED_MISSING_EVIDENCE", "reason": "qualification evidence incomplete", "missing_or_deferred_gates": missing, "gate_record": gates}


QUESTION_MAP = {
    "OBSERVATION_DOMAIN_SOURCE": "Was the full study area systematically searched, and can you provide the mapped observation boundary?",
    "OBSERVATION_DOMAIN_METHOD": "What independent survey, imagery, or field method defines the observation domain Omega?",
    "INVENTORY_TEMPORAL_SCOPE": "Are the landslide records event-specific or cumulative across multiple events?",
    "DEM_DATE": "What is the DEM acquisition date, and is it temporally suitable for this event?",
    "VERTICAL_DATUM": "What vertical datum and vertical units were used?",
    "TERRAIN_REPRESENTATION": "Is the elevation product a DSM or bare-earth DTM, and how were vegetation/buildings handled?",
    "QUALITY_MASK": "Are quality, uncertainty, visibility, or void masks available?",
    "DATA_LICENSE": "What licence or written permission governs scientific use and redistribution of derived metadata?",
    "INVENTORY_COMPLETENESS": "How was inventory completeness assessed across the observation domain?",
}


def draft_questions(manifest: dict, audit: dict | None = None) -> list[str]:
    audit = audit or {}
    missing = {k for k, v in manifest.items() if str(v).upper() in UNKNOWN}
    missing.update(audit.get("missing_evidence", []))
    questions = [QUESTION_MAP[k] for k in QUESTION_MAP if k in missing]
    if audit.get("OMEGA_DECISION") in {"MISSING", "PROVISIONAL", "REJECTED"}:
        questions.append("Can you supply an independently justified searched/mapped boundary rather than a box derived from positive landslides?")
    return sorted(set(questions))


def empty_manifest(dataset_id: str, *, access: str = "NEW_DEVELOPMENT_CANDIDATE", role: str = "UNDECIDED") -> dict:
    obj = {k: "NOT_PROVIDED" for k in MANIFEST_REQUIRED}
    obj.update({"DATASET_ID": dataset_id, "LANDSCAPE_NAME": dataset_id, "ACCESS_CLASSIFICATION": access,
                "PROPOSED_DATA_ROLE": role, "SOURCE_FILE_HASHES": {}, "AUDIT_STATUS": "REGISTERED", "FILE_ACCESS": {}})
    return obj
