"""Fail-closed, deterministic validation for Science Fair ML V2 Stage 1."""

from __future__ import annotations

import json
from collections import Counter

from .inventory_manifest import build_inventory_rows
from .schema import COLUMNS, FIELDS, FUTURE_FIELDS, INVENTORY_COLUMNS, SOURCE_COLUMNS, VOCAB
from .sources import all_sources

UNKNOWN = "UNKNOWN"
CRITICAL_GATES = ("y_gate", "omega_gate", "m_gate")
UNVERIFIED_EPOCH_MARKERS = ("to be checked", "epoch validity", "epoch unknown", "date unknown", "unverified")
POSITIVE_BOUNDS_MARKERS = ("positive polygon bounds", "convex hull of positives", "target bounds")
SOURCE_ROLES = {"Y", "OMEGA", "DEM", "GEOLOGY", "ENVIRONMENT", "AGE", "DISCOVERY_CATALOGUE"}


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def validate_stage1(candidates: list[dict], sources: list[dict], inventories: list[dict]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    field_names = [field[0] for field in FIELDS]
    duplicate_fields = _duplicates(field_names)
    if duplicate_fields:
        errors.append(f"duplicate schema fields: {duplicate_fields}")
    if field_names != COLUMNS:
        errors.append("COLUMNS does not match ordered FIELDS")

    candidate_ids = [row.get("landscape_id", "") for row in candidates]
    source_ids = [row.get("source_id", "") for row in sources]
    inventory_ids = [row.get("inventory_id", "") for row in inventories]
    for label, values in (("landscape_id", candidate_ids), ("source_id", source_ids), ("inventory_id", inventory_ids)):
        duplicates = _duplicates(values)
        if duplicates:
            errors.append(f"duplicate {label}: {duplicates}")

    candidate_set = set(candidate_ids)
    source_set = set(source_ids)
    for index, row in enumerate(candidates):
        missing = [field for field in COLUMNS if field not in row]
        extra = sorted(set(row) - set(COLUMNS))
        if missing or extra:
            errors.append(f"candidate[{index}] schema mismatch missing={missing} extra={extra}")
            continue
        lid = row["landscape_id"]
        for field, _, vocabulary, required, _ in FIELDS:
            value = row[field]
            if required and (value is None or value == ""):
                errors.append(f"{lid}: required field {field} is empty")
            allowed = VOCAB.get(vocabulary)
            if isinstance(allowed, list) and value not in allowed:
                errors.append(f"{lid}: {field}={value!r} outside {vocabulary}")
        if any(row[field] not in ("", None) for field in FUTURE_FIELDS):
            errors.append(f"{lid}: future ML fields must remain empty")
        coverage = str(row["dem_coverage"]).lower()
        if row["m_gate"] == "PASS" and any(marker in coverage for marker in UNVERIFIED_EPOCH_MARKERS):
            errors.append(f"{lid}: M_GATE PASS with unverified DEM epoch")
        if row["y_gate"] == "PASS":
            if row["inventory_available"] != "TRUE" or row["y_evidence_stage"] not in (
                "DATA_EXISTS", "DATA_DOWNLOADABLE", "DATA_SCIENTIFICALLY_USABLE"
            ):
                errors.append(f"{lid}: Y_GATE PASS without usable inventory evidence")
        if row["omega_gate"] == "PASS":
            evidence = " ".join(str(row[field]) for field in (
                "observation_domain_source", "observation_domain_type", "observation_domain_notes"
            )).lower()
            if row["observation_domain_establishable"] != "TRUE":
                errors.append(f"{lid}: OMEGA_GATE PASS but observation domain is not establishable")
            if row["observation_domain_source"] == UNKNOWN or row["observation_domain_type"] == UNKNOWN:
                errors.append(f"{lid}: OMEGA_GATE PASS lacks traceable source/type")
            if any(marker in evidence for marker in POSITIVE_BOUNDS_MARKERS):
                errors.append(f"{lid}: OMEGA_GATE PASS appears derived from target bounds")
            linked = [s for s in sources if s["landscape_id"] == lid and s["data_role"] == "OMEGA" and s["omega_obs_evidence"] != UNKNOWN]
            if not linked:
                errors.append(f"{lid}: OMEGA_GATE PASS has no source-manifest evidence link")
        if row["h_available"] == "FALSE" and any(row[field] in (0, 0.0, "0", "0.0") for field in ("h_quality", "h_notes")):
            errors.append(f"{lid}: unavailable H represented as numerical zero")
        if row["priority_class"] == "REJECT" and row["exclusion_reason"] in ("", "NONE", UNKNOWN):
            errors.append(f"{lid}: REJECT lacks reason")
        if any(row[gate] == "FAIL" for gate in CRITICAL_GATES) and row["candidate_class"] not in ("KNOWN_V1", "REFERENCE_ONLY"):
            if row["priority_class"] != "REJECT":
                errors.append(f"{lid}: hard-gate failure overridden by score/priority")
        if row["experimental_role"] != "UNASSIGNED" or row["lockbox_status"] != "UNASSIGNED":
            errors.append(f"{lid}: Stage-1 experimental/lockbox role must remain UNASSIGNED")
        if not any(source["landscape_id"] == lid for source in sources):
            warnings.append(f"{lid}: no source-manifest row")

    for index, row in enumerate(sources):
        missing = [field for field in SOURCE_COLUMNS if field not in row]
        extra = sorted(set(row) - set(SOURCE_COLUMNS))
        if missing or extra:
            errors.append(f"source[{index}] schema mismatch missing={missing} extra={extra}")
        if row.get("landscape_id") != "GLOBAL" and row.get("landscape_id") not in candidate_set:
            errors.append(f"{row.get('source_id')}: unknown landscape foreign key {row.get('landscape_id')}")
        if row.get("data_role") not in SOURCE_ROLES:
            errors.append(f"{row.get('source_id')}: uncontrolled data_role={row.get('data_role')}")

    for index, row in enumerate(inventories):
        missing = [field for field in INVENTORY_COLUMNS if field not in row]
        extra = sorted(set(row) - set(INVENTORY_COLUMNS))
        if missing or extra:
            errors.append(f"inventory[{index}] schema mismatch missing={missing} extra={extra}")
            continue
        iid = row["inventory_id"]
        if row["landscape_id"] not in candidate_set:
            errors.append(f"{iid}: unknown landscape foreign key")
        for field in ("inventory_source_id", "observation_domain_source_id"):
            if row[field] != UNKNOWN and row[field] not in source_set:
                errors.append(f"{iid}: unknown {field}={row[field]}")
        if row["inventory_trigger"] not in VOCAB["inventory_trigger"]:
            errors.append(f"{iid}: uncontrolled inventory_trigger={row['inventory_trigger']}")
        if row["geometry_type"] not in VOCAB["inventory_geometry_type"]:
            errors.append(f"{iid}: uncontrolled geometry_type={row['geometry_type']}")
        if row["target_tier"] not in VOCAB["inventory_tier"]:
            errors.append(f"{iid}: uncontrolled target_tier={row['target_tier']}")
        if row["verification_status"] not in VOCAB["verification"]:
            errors.append(f"{iid}: uncontrolled verification_status={row['verification_status']}")
        if row["downloadable"] not in VOCAB["tristate"]:
            errors.append(f"{iid}: uncontrolled downloadable={row['downloadable']}")

    return {
        "schema": "SFV2_STAGE1_VALIDATION_V1",
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": sorted(errors),
        "warnings": sorted(warnings),
        "counts": {
            "candidates": len(candidates),
            "sources": len(sources),
            "inventories": len(inventories),
        },
    }


def current_validation() -> dict:
    from .build_stage1 import build_rows

    candidates = build_rows()
    sources = all_sources()
    inventories = build_inventory_rows(candidates, sources)
    return validate_stage1(candidates, sources, inventories)


def main() -> None:
    result = current_validation()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
