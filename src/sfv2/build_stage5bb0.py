"""Build the read-only Stage 5B-B0 real-data arrival audit.

The audit reads governance metadata and filenames only.  If no explicit authorized
candidate-inbox location is documented, it performs no candidate-filesystem scan.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5b_b0"
STAGE5BA_OUT = ROOT / "outputs/stage5b_a"
V1_FREEZE = ROOT.parent / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json"
V1_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
STAGE5DA_CHECKPOINT = ROOT / "outputs/stage5d_a/STAGE5DA_CHECKPOINT.json"
STAGE5DA_CHECKPOINT_SHA256 = "b08ed1b452ff23c52a559c28d174ccab23da3d5f69b4b59694a346279f19ca4d"
STAGE5DA_PREREGISTRATION = ROOT / "outputs/stage5d_a/STAGE5DA_EXPERIMENT_PREREGISTRATION.json"
STAGE5DA_PREREGISTRATION_SHA256 = "e03af55eacc87832d8e644d4a78c201b330fd4bbd459aef7d426cf5de43728a1"
STAGE5C_SCIENTIFIC = {
    "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
    "outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json": "bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
    "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1",
}
CANDIDATES = [
    "REUNION_SALAZIE_1978_1984", "SAINT_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020",
    "POPOCATEPETL_PUEBLA_2017", "MOUNT_ELGON", "CHUUK_2002",
]
SAFE_GOVERNANCE_FILES = [
    STAGE5BA_OUT / "STAGE5BA_CANDIDATE_REGISTER.tsv",
    STAGE5BA_OUT / "STAGE5BA_ACCESS_POLICY.json",
    STAGE5BA_OUT / "STAGE5BA_CLI_USAGE.md",
    STAGE5BA_OUT / "STAGE5BA_FINAL_REPORT.md",
    STAGE5BA_OUT / "STAGE5BA_CHECKPOINT.json",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def _json(path: Path, value: object) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def historical_hash_audit() -> dict:
    """Verify path/hash references without imposing a brittle exact record count."""
    records: list[tuple[str, str, str]] = []
    for checkpoint in sorted(ROOT.glob("outputs/*/STAGE*_CHECKPOINT.json")):
        if checkpoint.parent == OUT:
            continue

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, str) and HEX64.fullmatch(child) and (ROOT / key).is_file():
                        records.append((str(checkpoint.relative_to(ROOT)), key, child))
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(json.loads(checkpoint.read_text()))
    mismatches = []
    for checkpoint, rel, expected in records:
        actual = sha256(ROOT / rel)
        if actual != expected:
            mismatches.append({"checkpoint": checkpoint, "path": rel, "expected": expected, "actual": actual})
    return {
        "records_checked": len(records), "unique_paths_checked": len({rel for _, rel, _ in records}),
        "mismatches": mismatches, "passed": bool(records) and not mismatches,
        "cardinality_policy": "record count is reported, not hard-coded; every discoverable path/hash reference must match",
    }


def verify_baseline() -> dict:
    from .stage2f_transfer import v1_verify

    v1 = v1_verify()
    if sha256(V1_FREEZE) != V1_SHA256 or not (v1["v1_intact"] and v1["strict_files"] == 153 and v1["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 integrity failure: {v1}")
    if sha256(STAGE5DA_CHECKPOINT) != STAGE5DA_CHECKPOINT_SHA256:
        raise RuntimeError("STOP: Stage 5D-A checkpoint-file hash mismatch")
    stage5da = json.loads(STAGE5DA_CHECKPOINT.read_text())
    bad = [rel for rel, digest in stage5da["new_artifact_hashes"].items() if not (ROOT / rel).is_file() or sha256(ROOT / rel) != digest]
    if bad:
        raise RuntimeError(f"STOP: Stage 5D-A recorded artifacts differ: {bad}")
    if sha256(STAGE5DA_PREREGISTRATION) != STAGE5DA_PREREGISTRATION_SHA256 or stage5da["preregistration_hash"] != STAGE5DA_PREREGISTRATION_SHA256:
        raise RuntimeError("STOP: Stage 5D-A preregistration identity mismatch")
    freeze = json.loads((ROOT / "outputs/stage5c_a/STAGE5CA_PREREGISTRATION_FREEZE.json").read_text())
    for rel, expected in STAGE5C_SCIENTIFIC.items():
        if freeze["files"].get(rel) != expected or sha256(ROOT / rel) != expected:
            raise RuntimeError(f"STOP: frozen Stage 5C-A scientific hash mismatch: {rel}")
    historical = historical_hash_audit()
    if not historical["passed"]:
        raise RuntimeError(f"STOP: historical manifest mismatch: {historical}")
    return {
        "status": "PASS", "V1": v1, "V1_freeze_sha256": sha256(V1_FREEZE),
        "stage5da_checkpoint_sha256": sha256(STAGE5DA_CHECKPOINT),
        "stage5da_artifacts_checked": len(stage5da["new_artifact_hashes"]),
        "stage5da_preregistration_sha256": sha256(STAGE5DA_PREREGISTRATION),
        "stage5c_scientific_hashes": STAGE5C_SCIENTIFIC, "historical_manifest": historical,
    }


def authorized_inbox_audit() -> dict:
    """Inspect only named governance documents for a structured inbox declaration."""
    patterns = [
        re.compile(r'"(?:AUTHORIZED_)?CANDIDATE_INBOX"\s*:\s*"([^"\n]+)"', re.I),
        re.compile(r'"AUTHORIZED_INBOX_PATH"\s*:\s*"([^"\n]+)"', re.I),
    ]
    declarations = []
    for path in SAFE_GOVERNANCE_FILES:
        text = path.read_text()
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and value not in {"UNKNOWN", "NOT_PROVIDED"}:
                    declarations.append({"source": str(path.relative_to(ROOT)), "path": value})
    # Generic CLI placeholder CANDIDATE_DIR is not an authorization or concrete location.
    if declarations:
        raise RuntimeError("STOP: an authorized inbox declaration exists but Stage 5B-B0 has no approved path resolver")
    return {
        "status": "NO_DOCUMENTED_AUTHORIZED_INBOX", "declaration_sources_checked": [str(p.relative_to(ROOT)) for p in SAFE_GOVERNANCE_FILES],
        "documented_locations": [], "filesystem_locations_enumerated": [], "recursive_scans_performed": 0,
        "email_accessed": False, "network_accessed": False,
        "interpretation": "No broader search is permitted; existing project templates are not researcher deliveries.",
    }


def arrival_rows(inbox: dict) -> list[dict]:
    register = pd.read_csv(STAGE5BA_OUT / "STAGE5BA_CANDIDATE_REGISTER.tsv", sep="\t").set_index("DATASET_ID")
    rows = []
    for candidate in CANDIDATES:
        row = register.loc[candidate]
        template = STAGE5BA_OUT / ("STAGE5BA_REUNION_INTAKE_TEMPLATE.json" if candidate == CANDIDATES[0] else f"candidate_templates/{candidate}.json")
        next_action = (
            "USER_DEFINE_AUTHORIZED_INBOX_THEN_REQUEST_METADATA_ONLY_H_TRACK_DELIVERY"
            if candidate == "REUNION_SALAZIE_1978_1984" else
            "USER_DEFINE_AUTHORIZED_INBOX_THEN_REQUEST_METADATA_ONLY_INTAKE_PACKAGE"
        )
        rows.append({
            "DATASET_ID": candidate, "STAGE5BA_STATE": row["STATE"], "CURRENT_ACCESS_CLASSIFICATION": row["ACCESS_CLASSIFICATION"],
            "CURRENT_PROPOSED_ROLE": row["PROPOSED_DATA_ROLE"], "LOCKBOX_STATUS": row["LOCKBOX_STATUS"],
            "AUTHORIZED_INBOX_STATUS": inbox["status"], "AUTHORIZED_INBOX_PATH": "UNKNOWN",
            "ARRIVAL_STATUS": "NOT_RECEIVED", "FILES_PRESENT_IN_AUTHORIZED_INBOX": "NOT_VERIFIED",
            "PROJECT_GENERATED_TEMPLATE_PRESENT": "YES" if template.is_file() else "NO",
            "SUPPLIED_MANIFEST": "NOT_RECEIVED", "RESEARCHER_METADATA_DOCUMENTS": "NOT_RECEIVED",
            "SOURCE_PROVENANCE": "NOT_VERIFIED", "KNOWN_ACCESS_RESTRICTIONS": "UNKNOWN",
            "OBSERVATION_DOMAIN_DOCUMENTED": "NOT_RECEIVED", "EVENT_SPECIFIC_LANDSLIDE_INVENTORY": "NOT_RECEIVED",
            "SUITABLE_DEM_COVERAGE_DOCUMENTED": "NOT_RECEIVED", "HISTORICAL_TOPOGRAPHIC_PRODUCTS": "NOT_RECEIVED",
            "OUTCOME_LABEL_CONTENT_OPENED": "NO", "NEXT_PERMISSIBLE_ACTION": next_action,
            "BASIS": "Stage 5B-A FILES_RECEIVED=NO; no documented authorized inbox; project template is governance metadata, not delivery",
        })
    return rows


def missing_evidence_rows() -> list[dict]:
    requirements = [
        ("AUTHORIZED_INBOX_AND_RECEIPT_RECORD", "NOT_VERIFIED", "Define one explicit authorized candidate-inbox path; record receipt date and metadata-only file list."),
        ("SOURCE_IDENTITY_PERMISSION_LICENSE", "NOT_RECEIVED", "Provide source/provider identity, permission, licence, citation and permitted research uses."),
        ("INVENTORY_EVENT_DEFINITION", "NOT_RECEIVED", "Define trigger, event date/window, event-specific versus cumulative status and inventory lineage."),
        ("INVENTORY_COMPLETENESS_VISIBILITY", "NOT_RECEIVED", "Provide mapping method/scale, minimum mapping unit, completeness assessment and visibility/quality masks."),
        ("OBSERVATION_DOMAIN_OMEGA", "NOT_RECEIVED", "Provide independently justified searched/mapped domain and its provenance; not a landslide-derived extent."),
        ("CRS_COORDINATE_INTEGRITY", "NOT_RECEIVED", "Provide horizontal/vertical CRS, units, datums, accuracy and coordinate-integrity documentation."),
        ("DEM_COVERAGE_TEMPORAL_SUITABILITY", "NOT_RECEIVED", "Provide DEM source/date/resolution/coverage, NoData/quality information and event-time suitability."),
        ("EXPOSURE_AND_ROLE_HISTORY", "NOT_RECEIVED", "Document prior label/model exposure and propose development, reserve-lockbox, H-only or reject role before outcomes."),
    ]
    rows = []
    for candidate in CANDIDATES:
        candidate_requirements = list(requirements)
        if candidate == "REUNION_SALAZIE_1978_1984":
            candidate_requirements.append(("HISTORICAL_DSM_DTM_METADATA_UNCERTAINTY", "NOT_RECEIVED", "For Track A provide product identity, acquisition/processing lineage, datum, uncertainty, vegetation/building handling, masks and independent controls."))
            candidate_requirements.append(("SEPARATE_PREDICTIVE_ML_EVIDENCE", "NOT_RECEIVED", "Track B still requires event-specific labels, Omega, compatible DEM, completeness and role evidence; a historical DSM is insufficient."))
        for item, status, request in candidate_requirements:
            rows.append({
                "DATASET_ID": candidate, "EVIDENCE_ITEM": item, "STATUS": status,
                "QUALIFICATION_IMPACT": "BLOCKS_CONTROLLED_STAGE5B_B_QUALIFICATION",
                "NEXT_REQUEST": request, "OUTCOME_CONTENT_REQUIRED_AT_B0": "NO",
            })
    return rows


def write_documents(arrivals: list[dict], inbox: dict) -> None:
    pd.DataFrame(arrivals).to_csv(OUT / "STAGE5BB0_CANDIDATE_ARRIVAL_REGISTER.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame(missing_evidence_rows()).to_csv(OUT / "STAGE5BB0_MISSING_EVIDENCE_REGISTER.tsv", sep="\t", index=False, lineterminator="\n")
    _write(OUT / "STAGE5BB0_REUNION_READINESS.md", """# Réunion Salazie readiness

## Arrival finding

`REUNION_SALAZIE_1978_1984` is `NOT_RECEIVED`. Stage 5B-A records that products were requested and the researcher indicated willingness to provide them, but no product was received or audited. No authorized inbox is currently documented, so no broader filesystem search was performed.

## Track A — historical topography validation

Status: `BLOCKED_MISSING_EVIDENCE`; not ready for Stage 5B-B qualification. A later metadata-first intake would require product identity (DSM/DTM/point cloud), acquisition and processing lineage, licence, horizontal/vertical datum, resolution and coverage, georeferencing accuracy, vertical uncertainty, vegetation/building treatment, visibility and quality masks, independent geological controls, and independence from landslide labels.

A historical DSM is an observed historical surface. It is not automatically bare earth and must not be interpreted as the original pre-erosional volcano surface. Receipt would support only a later H-track qualification audit under unchanged H gates; it would not create Gen4 or unlock MEGH.

## Track B — predictive landslide ML

Status: `BLOCKED_MISSING_EVIDENCE`. Historical topography alone does not provide an event-specific landslide inventory, defensible observation domain Ω, mapping completeness evidence, or a compatible predictive target. Track B requires those items independently plus DEM, CRS, licence, exposure and role evidence.

## Next permissible action

The user must define an explicit authorized inbox, then request a metadata-only delivery manifest and Track A documentation. Request Track B evidence separately. Do not inspect label contents or perform qualification in Stage 5B-B0.
""")
    _write(OUT / "STAGE5BB0_RESEARCHER_FOLLOWUPS.md", """# Draft researcher follow-ups

These messages are drafts only. Nothing was sent and no response is assumed.

## Common metadata-first draft

Subject: Metadata needed before controlled volcanic-landscape dataset intake

Thank you for considering sharing the dataset. Before any scientific qualification or outcome inspection, could you please provide a metadata-only manifest covering:

1. the mapped/searched observation boundary and independent basis for Ω;
2. the triggering event and exact inventory time window, including whether mapping is event-specific or cumulative;
3. mapping method, scale, minimum mapping unit, positional uncertainty, visibility and completeness limits;
4. DEM source, acquisition date, resolution, coverage, CRS, horizontal/vertical datum, units, NoData and quality masks;
5. source/provider, citation, licence, permission and permitted research uses; and
6. any prior use of the inventory or labels in model development or evaluation.

Please do not send outcome-label files until an authorized intake location and dataset role have been confirmed.

Apply this draft separately to Saint Lucia Tomas 2010, Vanuatu Harold 2020, Popocatépetl Puebla 2017, Mount Elgon and Chuuk 2002; preserve each candidate's event-specific wording and do not infer missing answers.

## Réunion Track A addition

For each 1978/1984 historical product, please identify whether it is a DSM, DTM, point cloud or derivative; supply acquisition and processing lineage, camera/block and georeferencing records where applicable, coverage/resolution, horizontal and vertical datums, vertical uncertainty, vegetation/building treatment, visibility/quality masks and usage permission. Please also describe any independent geological controls. These products will not be assumed to represent a pre-erosional surface.

## Réunion Track B addition

If predictive landslide ML use is proposed, please document the event-specific inventory and its independently justified Ω separately. Historical photogrammetry alone cannot qualify the predictive track.
""")
    _write(OUT / "STAGE5BB0_NEXT_INTAKE_ACTIONS.md", """# Next controlled intake actions

1. **User action required:** designate one explicit, project-scoped authorized candidate-inbox path. Do not use a broad personal, school or cloud root.
2. Send the draft metadata-first questions outside this software; Stage 5B-B0 does not send messages.
3. Ask providers to deliver a manifest and non-outcome documentation first. Keep label/outcome files sealed until role governance permits access.
4. On receipt, record immutable filenames, sizes, dates and hashes without interpreting scientific content. Apply the Stage 5B-A access policy.
5. Audit provenance, licence, event definition, Ω, completeness, CRS and DEM suitability. For Réunion, audit Track A and Track B independently.
6. Make an explicit role decision before outcome inspection: development candidate, reserve lockbox, H-only, pending or reject. Newness alone never creates a lockbox.
7. Only after qualification and an outcome-blind development-membership freeze may Stage 5D-B readiness be reconsidered.

Current action boundary: no candidate file has arrived; do not run intake, qualification, W1, D1, E1 or G1.
""")
    readiness = [
        {"ITEM": "CURRENT_DEVELOPMENT_LANDSCAPES", "VALUE": "6", "STATUS": "VERIFIED", "BASIS": "Stage 5D-A frozen development set"},
        {"ITEM": "ADDITIONAL_QUALIFIED_DEVELOPMENT_LANDSCAPES", "VALUE": "0", "STATUS": "BLOCKED", "BASIS": "all six candidates NOT_RECEIVED"},
        {"ITEM": "PREFERRED_DEVELOPMENT_COUNT", "VALUE": ">=8", "STATUS": "NOT_MET", "BASIS": "preferred design condition, not a power guarantee"},
        {"ITEM": "FROZEN_LOCKBOX_COUNT", "VALUE": "0", "STATUS": "VERIFIED", "BASIS": "Stage 5C-A current_frozen_lockboxes=[] and no promotion in B0"},
        {"ITEM": "DEVELOPMENT_MEMBERSHIP_FREEZE", "VALUE": "NO", "STATUS": "BLOCKED", "BASIS": "new-data allocation unresolved"},
        {"ITEM": "W1_D1_E1_EXECUTION", "VALUE": "NOT_RUN", "STATUS": "BLOCKED", "BASIS": "data and membership conditions unmet"},
        {"ITEM": "G1", "VALUE": "DEFERRED", "STATUS": "DEFERRED", "BASIS": "Stage 5D-A decision unchanged"},
        {"ITEM": "ADVANCED_ML_EXPERIMENT_READINESS", "VALUE": "NO", "STATUS": "BLOCKED", "BASIS": "no additional qualified development landscapes; membership not frozen"},
    ]
    pd.DataFrame(readiness).to_csv(OUT / "STAGE5BB0_STAGE5D_READINESS.tsv", sep="\t", index=False, lineterminator="\n")
    _json(OUT / "STAGE5BB0_AUDIT_SCOPE.json", inbox)


def final_report() -> None:
    _write(OUT / "STAGE5BB0_FINAL_REPORT.md", """STAGE 5B-B0 REAL-DATA ARRIVAL AUDIT: COMPLETE

NO NEW CANDIDATE DATASETS RECEIVED.
STAGE 5D EXECUTION REMAINS BLOCKED.

# 1. Historical integrity status

V1, the frozen Stage 5D-A checkpoint/preregistration, all three Stage 5C-A scientific files and every discoverable historical path/hash reference passed verification. Historical artifacts were not modified.

# 2. Candidate files actually present

The project contains Stage 5B-A-generated templates for Réunion, Saint Lucia, Vanuatu, Popocatépetl, Mount Elgon and Chuuk. Those files are governance templates, not researcher deliveries. Stage 5B-A records `FILES_RECEIVED=NO` for every candidate. No explicit authorized candidate-inbox path is documented, so zero candidate directories were enumerated and no unrelated filesystem was scanned.

# 3. Unreceived datasets

All six candidates are `NOT_RECEIVED`: `REUNION_SALAZIE_1978_1984`, `SAINT_LUCIA_TOMAS_2010`, `VANUATU_HAROLD_2020`, `POPOCATEPETL_PUEBLA_2017`, `MOUNT_ELGON`, and `CHUUK_2002`.

# 4. Missing metadata

Every candidate lacks a supplied manifest, verified provenance/licence, event/inventory definition, completeness/visibility evidence, defensible Ω, CRS documentation, suitable DEM evidence and exposure/role history. Réunion additionally lacks the historical DSM/DTM lineage and uncertainty package.

# 5. Controlled qualification readiness

No candidate is ready for Stage 5B-B qualification. Réunion remains separately blocked on both Track A historical-topography evidence and Track B predictive-ML evidence. A historical DSM alone would not qualify Track B and would not represent a pre-erosional surface.

# 6. Stage 5D readiness

The project still has six development landscapes, zero additional qualified development landscapes and zero frozen lockboxes. Development membership is not frozen. W1, D1 and E1 were not executed; G1 remains deferred; LOGISTIC_M is unchanged.

# 7. Precise next user action

Define one narrow, explicit authorized candidate-inbox path inside the project workflow. Then use the draft follow-up questions to request metadata-only manifests and documentation—starting with Ω, event-specific inventory provenance, DEM suitability, historical-product uncertainty where relevant, and research-use permission. Do not send or inspect outcome labels until role and access governance are resolved.

# Integrity boundary

No outcome content was opened, no real data was ingested, no model was trained, no candidate was promoted, no message was sent, no network/email access occurred, and no commit or push was performed.
""")


REQUIRED = [
    "STAGE5BB0_FINAL_REPORT.md", "STAGE5BB0_CANDIDATE_ARRIVAL_REGISTER.tsv",
    "STAGE5BB0_MISSING_EVIDENCE_REGISTER.tsv", "STAGE5BB0_REUNION_READINESS.md",
    "STAGE5BB0_RESEARCHER_FOLLOWUPS.md", "STAGE5BB0_NEXT_INTAKE_ACTIONS.md",
    "STAGE5BB0_STAGE5D_READINESS.tsv", "STAGE5BB0_AUDIT_SCOPE.json",
]


def build() -> dict:
    baseline = verify_baseline()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    inbox = authorized_inbox_audit()
    arrivals = arrival_rows(inbox)
    OUT.mkdir(parents=True, exist_ok=True)
    write_documents(arrivals, inbox)
    final_report()
    after = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    if before != after:
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise RuntimeError(f"STOP: historical outputs changed: {changed}")
    return {"stage": "5B-B0", "status": "BUILT_PENDING_TESTS", "baseline": baseline, "inbox_audit": inbox, "candidate_count": len(arrivals)}


def finalize(test_results: dict) -> dict:
    baseline = verify_baseline()
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"STOP: missing Stage 5B-B0 artifacts: {missing}")
    if not test_results.get("passed"):
        raise RuntimeError(f"STOP: focused tests failed: {test_results}")
    _json(OUT / "STAGE5BB0_TESTS.json", test_results)
    checkpoint_path = OUT / "STAGE5BB0_CHECKPOINT.json"
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path != checkpoint_path)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    arrivals = pd.read_csv(OUT / "STAGE5BB0_CANDIDATE_ARRIVAL_REGISTER.tsv", sep="\t")
    readiness = pd.read_csv(OUT / "STAGE5BB0_STAGE5D_READINESS.tsv", sep="\t").set_index("ITEM")["STATUS"].to_dict()
    git_status = _git("status", "--short")
    checkpoint = {
        "stage": "5B-B0", "status": "COMPLETE", "audit_date": "2026-10-08",
        "historical_verification": baseline, "new_artifact_hashes": hashes,
        "arrival_summary": {"candidate_count": len(arrivals), "received_count": int((arrivals["ARRIVAL_STATUS"] != "NOT_RECEIVED").sum()), "not_received_count": int((arrivals["ARRIVAL_STATUS"] == "NOT_RECEIVED").sum())},
        "audit_scope": json.loads((OUT / "STAGE5BB0_AUDIT_SCOPE.json").read_text()),
        "readiness_statuses": readiness, "test_results": test_results,
        "source_code_changes": ["src/sfv2/build_stage5bb0.py", "tests/test_stage5bb0.py"],
        "code_hashes": {"src/sfv2/build_stage5bb0.py": sha256(ROOT / "src/sfv2/build_stage5bb0.py"), "tests/test_stage5bb0.py": sha256(ROOT / "tests/test_stage5bb0.py")},
        "software_versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "platform": platform.platform()},
        "git_head": _git("rev-parse", "HEAD"), "git_working_tree_dirty": bool(git_status), "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_note": "Pre-existing user work preserved; no commit or push.",
        "next_user_action": "Define one narrow explicit authorized candidate-inbox path, then request metadata-only manifests/documentation; do not provide or open outcome labels yet.",
        "remaining_blockers": ["no documented authorized candidate inbox", "all six candidates not received", "no candidate qualified", "development membership not frozen", "preferred >=8 development landscapes not met"],
        "V1_INTACT": "YES", "HISTORICAL_OUTPUTS_UNCHANGED": "YES", "STAGE5DA_CHECKPOINT_UNCHANGED": "YES",
        "STAGE5DA_PREREGISTRATION_UNCHANGED": "YES", "STAGE5CA_PROTOCOL_UNCHANGED": "YES",
        "STAGE5CA_PREREGISTRATION_UNCHANGED": "YES", "STAGE5CA_SUCCESS_RULES_UNCHANGED": "YES",
        "OUTCOME_LABEL_CONTENT_OPENED": "NO", "NEW_REAL_DATA_INGESTED": "NO", "NEW_MODEL_TRAINED": "NO",
        "CANDIDATE_AUTOMATICALLY_PROMOTED": "NO", "LOCKBOX_OPENED": "NO", "EMAIL_ACCESSED": "NO",
        "NETWORK_DATA_DOWNLOAD": "NO", "UNAUTHORIZED_FILESYSTEM_SCAN": "NO", "COMMIT_CREATED": "NO", "PUSH_PERFORMED": "NO",
        "hash_policy": "all Stage 5B-B0 outputs except the checkpoint itself are hashed",
    }
    _json(checkpoint_path, checkpoint)
    return {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "sha256": sha256(checkpoint_path), "hashed_outputs": len(hashes)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "finalize"])
    parser.add_argument("--tests-json")
    args = parser.parse_args()
    if args.phase == "build":
        print(json.dumps(build(), indent=2, sort_keys=True))
    else:
        if not args.tests_json:
            raise SystemExit("--tests-json required")
        print(json.dumps(finalize(json.loads(Path(args.tests_json).read_text())), indent=2, sort_keys=True))
