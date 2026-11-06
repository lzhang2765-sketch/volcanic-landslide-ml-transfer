"""Deterministic Stage-2A.2 validator: identity, vocabularies, gate implications, provenance and prior-checkpoint integrity."""
from __future__ import annotations

import hashlib

from .schema import FUTURE_FIELDS, ROOT, VOCAB
from .validate_stage2a_extension import LEAKAGE_RISK, READINESS, SOURCE_ROLES

STEPS = {"PASSED_A_TO_D", "STOPPED_AT_A_ACCESS", "STOPPED_AT_A_QUALITY", "STOPPED_AT_B", "STOPPED_AT_C_D", "FAILED_A", "FAILED_B", "FAILED_C"}
OUTCOMES = {"CANDIDATE_READY", "CANDIDATE_PARTIAL", "CANDIDATE_BLOCKED", "REJECT_A", "REJECT_B", "REJECT_C", "BLOCKED_ACCESS", "NOT_INDEPENDENT"}
STEP_VALUES = {"PASS", "FAIL", "UNCERTAIN", "PARTIAL", "UNKNOWN"}
VOLCANIC = {"YES", "PARTIAL", "NO", "UNKNOWN"}
CRITICAL = ("y_gate", "omega_gate", "m_gate")


def validate(candidates, sources, inventories, datasets, acq, prior_hashes):
    errors = []
    ids = [r["landscape_id"] for r in candidates]
    for label, vals in (("landscape", ids), ("source", [s["source_id"] for s in sources]), ("inventory", [i["inventory_id"] for i in inventories]),
                        ("dataset", [d["dataset_id"] for d in datasets]), ("acquisition", [a["acquisition_id"] for a in acq])):
        if len(vals) != len(set(vals)):
            errors.append(f"duplicate {label} IDs")
    idset, sidset = set(ids), {s["source_id"] for s in sources}
    for s in sources:
        if s["landscape_id"] not in idset | {"GLOBAL"}:
            errors.append(f"bad source FK {s['source_id']}")
        if set(s["data_role"].split("|")) - SOURCE_ROLES:
            errors.append(f"uncontrolled source role {s['source_id']}")
        if s.get("metadata_file") and not s.get("metadata_sha256"):
            errors.append(f"metadata file without hash {s['source_id']}")
    for i in inventories:
        if i["landscape_id"] not in idset:
            errors.append(f"bad inventory FK {i['inventory_id']}")
        if i["inventory_source_id"] not in sidset | {"UNKNOWN"} or i["observation_domain_source_id"] not in sidset | {"UNKNOWN"}:
            errors.append(f"bad inventory source FK {i['inventory_id']}")
        if i["target_tier"] not in VOCAB["inventory_tier"]:
            errors.append(f"uncontrolled inventory tier {i['inventory_id']}")
    for d in datasets:
        if d["landscape_id"] and d["landscape_id"] not in idset:
            errors.append(f"dataset references unknown landscape {d['dataset_id']}")
        if d["fast_fail_outcome"] not in OUTCOMES or d["volcanic_intersection"] not in VOLCANIC or d["verification"] not in VOCAB["verification"]:
            errors.append(f"uncontrolled dataset vocabulary {d['dataset_id']}")
        if {d[k] for k in ("step_a_spatial_y", "step_b_omega", "step_c_volcanic", "step_d_temporal_m")} - STEP_VALUES:
            errors.append(f"uncontrolled fast-fail step {d['dataset_id']}")
        if d["fast_fail_outcome"] == "CANDIDATE_READY" and any(d[k] != "PASS" for k in ("step_a_spatial_y", "step_b_omega", "step_c_volcanic", "step_d_temporal_m")):
            errors.append(f"READY dataset with non-PASS fast-fail step {d['dataset_id']}")
        if not d["url"].startswith("http"):
            errors.append(f"dataset without URL {d['dataset_id']}")
    for r in candidates:
        lid, status = r["landscape_id"], r["stage2a2_readiness"]
        for f in ("y_gate", "omega_gate", "m_gate", "h_status"):
            if r[f] not in VOCAB["gate"]:
                errors.append(f"uncontrolled {f} {lid}")
        if r["inventory_tier"] not in VOCAB["inventory_tier"] or status not in READINESS:
            errors.append(f"uncontrolled tier/readiness {lid}")
        if status in ("READY_CORE", "READY_CORE_H") and any(r[g] != "PASS" for g in CRITICAL):
            errors.append(f"READY with non-PASS critical gate {lid}")
        if status == "READY_CORE_H" and r["h_status"] != "PASS":
            errors.append(f"READY_CORE_H without H PASS {lid}")
        if any(r[f] for f in FUTURE_FIELDS):
            errors.append(f"future metric populated {lid}")
        if r["experimental_role"] != "UNASSIGNED" or r["lockbox_status"] != "UNASSIGNED":
            errors.append(f"role assigned {lid}")
        if r["candidate_class"] == "KNOWN_V1" and not r["v2_unseen_test_eligibility"].startswith("INELIGIBLE_V1_HISTORICAL"):
            errors.append(f"KNOWN_V1 landscape not marked unseen-ineligible {lid}")
        if r["stage2a2_screened"] == "TRUE":
            if r["stage2a2_fast_fail_step"] not in STEPS or r["stage2a2_volcanic_intersection"] not in VOLCANIC:
                errors.append(f"uncontrolled fast-fail step {lid}")
            if r["stage2a2_m_leakage_risk"] not in LEAKAGE_RISK:
                errors.append(f"uncontrolled leakage risk {lid}")
            if status == "REJECT" and not any(r[g] == "FAIL" for g in CRITICAL):
                errors.append(f"REJECT without a critical-gate FAIL {lid}")
            if status in ("READY_CORE", "READY_CORE_H") and r["stage2a2_volcanic_intersection"] != "YES":
                errors.append(f"READY without verified volcanic terrain {lid}")
            for sid in filter(None, r["stage2a2_source_ids"].split("|")):
                if sid not in sidset:
                    errors.append(f"missing Stage-2A.2 claim source {lid}:{sid}")
        elif r["stage2a2_readiness"] != r["stage2a_extension_readiness"]:
            errors.append(f"unscreened row changed readiness {lid}")
    by_id = {r["landscape_id"]: r for r in candidates}
    if by_id.get("KURIKOMA_2008", {}).get("stage2a2_readiness") != "READY_CORE_H":
        errors.append("Kurikoma readiness changed without new evidence")
    for guard, field, value in (("WILSONS_RIVER_NSW_2022", "stage2a_extension_lineage_risk", "HIGH"), ("MATZACO_PUEBLA_2017", "omega_gate", "FAIL")):
        if by_id.get(guard, {}).get(field) != value:
            errors.append(f"prior safeguard weakened {guard}")
    for a in acq:
        if a["checksum"] or a["download_status"] == "COMPLETE":
            errors.append(f"acquisition row claims execution {a['acquisition_id']}")
    for rel, digest in prior_hashes.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
            errors.append(f"prior checkpoint/table changed {rel}")
    return {"valid": not errors, "error_count": len(errors), "errors": errors,
            "checks": {"candidates": len(candidates), "sources": len(sources), "inventories": len(inventories), "datasets": len(datasets),
                       "screened": sum(r["stage2a2_screened"] == "TRUE" for r in candidates), "acquisition_rows": len(acq),
                       "prior_preserved": not any(e.startswith("prior checkpoint") for e in errors)}}
