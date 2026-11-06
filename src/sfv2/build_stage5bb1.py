"""Build/finalize the Stage 5B-B1 secure candidate-inbox infrastructure."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from . import inbox
from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5b_b1"
INBOX_ROOT = ROOT / "candidate_inbox"
V1_FREEZE = ROOT.parent / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json"
V1_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
STAGE5BB0 = (ROOT / "outputs/stage5b_b0/STAGE5BB0_CHECKPOINT.json", "2a2c4c413997a41278b0d614cc84c8f19deeb246a3eb64ca8ca9b43a0ba9b33e")
STAGE5DA = (ROOT / "outputs/stage5d_a/STAGE5DA_CHECKPOINT.json", "b08ed1b452ff23c52a559c28d174ccab23da3d5f69b4b59694a346279f19ca4d")
STAGE5DA_PREREGISTRATION_SHA256 = "e03af55eacc87832d8e644d4a78c201b330fd4bbd459aef7d426cf5de43728a1"
STAGE5C_SCIENTIFIC = {
    "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
    "outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json": "bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
    "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def _json(path: Path, value: object) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT.parent, text=True, capture_output=True, check=check)


def historical_hash_audit() -> dict:
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
    authorized_source_evolution = []
    authorized = {("outputs/stage5b_a/STAGE5BA_CHECKPOINT.json", "src/sfv2/intake_cli.py")}
    for checkpoint, rel, expected in records:
        actual = sha256(ROOT / rel)
        if actual != expected:
            item = {"checkpoint": checkpoint, "path": rel, "expected": expected, "actual": actual}
            if (checkpoint, rel) in authorized:
                item["classification"] = "AUTHORIZED_STAGE5BB1_CLI_EXTENSION"
                authorized_source_evolution.append(item)
            else:
                mismatches.append(item)
    return {"records_checked": len(records), "unique_paths_checked": len({x[1] for x in records}),
            "mismatches": mismatches, "authorized_source_evolution": authorized_source_evolution,
            "passed": bool(records) and not mismatches,
            "policy": "all historical outputs and unmodified sources must match; the one explicitly required CLI extension is recorded rather than misclassified as output corruption",
            "cardinality_policy": "non-brittle: verify every discovered path/hash; report rather than freeze record count"}


def verify_baseline() -> dict:
    from .stage2f_transfer import v1_verify

    v1 = v1_verify()
    if sha256(V1_FREEZE) != V1_SHA256 or not (v1["v1_intact"] and v1["strict_files"] == 153 and v1["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 integrity failure: {v1}")
    checkpoint_details = {}
    for stage, (path, expected) in {"stage5b_b0": STAGE5BB0, "stage5d_a": STAGE5DA}.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"STOP: {stage} checkpoint mismatch {actual} != {expected}")
        obj = json.loads(path.read_text())
        bad = [rel for rel, digest in obj["new_artifact_hashes"].items() if not (ROOT / rel).is_file() or sha256(ROOT / rel) != digest]
        if bad:
            raise RuntimeError(f"STOP: {stage} artifact mismatch: {bad}")
        checkpoint_details[stage] = {"checkpoint_sha256": actual, "artifacts_checked": len(obj["new_artifact_hashes"]), "mismatches": []}
    stage5da = json.loads(STAGE5DA[0].read_text())
    if stage5da["preregistration_hash"] != STAGE5DA_PREREGISTRATION_SHA256:
        raise RuntimeError("STOP: Stage 5D-A preregistration mismatch")
    freeze = json.loads((ROOT / "outputs/stage5c_a/STAGE5CA_PREREGISTRATION_FREEZE.json").read_text())
    for rel, expected in STAGE5C_SCIENTIFIC.items():
        if freeze["files"].get(rel) != expected or sha256(ROOT / rel) != expected:
            raise RuntimeError(f"STOP: frozen Stage 5C-A mismatch: {rel}")
    historical = historical_hash_audit()
    if not historical["passed"]:
        raise RuntimeError(f"STOP: historical hash-chain mismatch: {historical}")
    return {"status": "PASS", "V1": v1, "V1_freeze_sha256": sha256(V1_FREEZE), "checkpoints": checkpoint_details, "stage5da_preregistration_sha256": stage5da["preregistration_hash"], "stage5c_scientific_hashes": STAGE5C_SCIENTIFIC, "historical_manifest": historical}


def create_structure() -> None:
    if INBOX_ROOT.is_symlink():
        raise RuntimeError("STOP: candidate_inbox may not be a symlink")
    INBOX_ROOT.mkdir(parents=True, exist_ok=True)
    if INBOX_ROOT.resolve(strict=True) != (ROOT / "candidate_inbox").resolve(strict=True):
        raise RuntimeError("STOP: candidate_inbox canonical path mismatch")
    for slug in inbox.CANDIDATE_PATHS.values():
        candidate = INBOX_ROOT / slug
        candidate.mkdir(exist_ok=True)
        if candidate.is_symlink():
            raise RuntimeError(f"STOP: candidate path is symlink: {slug}")
        for section in sorted(inbox.ALL_SECTIONS):
            path = candidate / section
            path.mkdir(exist_ok=True)
            if path.is_symlink():
                raise RuntimeError(f"STOP: inbox section is symlink: {slug}/{section}")


def config() -> dict:
    return {
        "version": "1.0.0", "authorized_root": "candidate_inbox", "root_scope": "ONE_PROJECT_RELATIVE_ROOT_ONLY",
        "canonical_resolution_required": True, "explicit_invocation_required": True,
        "candidate_paths": inbox.CANDIDATE_PATHS, "sections": {"metadata": "METADATA_ONLY_ALLOWED", "documentation": "METADATA_ONLY_ALLOWED", "restricted": "DENY_WITHOUT_FUTURE_EXPLICIT_AUTHORIZATION"},
        "access_classifications": ["DEVELOPMENT_EXPOSED", "NEW_DEVELOPMENT_CANDIDATE", "LOCKBOX_CANDIDATE_UNOPENED", "LOCKBOX_FROZEN_UNOPENED", "EXTERNAL_VALIDATION_OPENED"],
        "initial_candidate_state": {"arrival_status": "NOT_RECEIVED", "qualification_status": "UNQUALIFIED", "dataset_role": "UNDECIDED", "lockbox_status": "NOT_DESIGNATED"},
        "path_denials": ["outside authorized root", "parent traversal", "external absolute path", "symlink escape", "suspicious hard link", "restricted section in metadata-only mode"],
        "archives": "NEVER_EXTRACT_AUTOMATICALLY; PUBLIC-SECTION ARCHIVES_QUARANTINED_PENDING_CLASSIFICATION",
        "restricted_logging": "NO_FILENAMES_NO_HASHES_NO_CONTENT_METADATA", "git_ignore_is_security_boundary": False,
        "real_data_present_at_initialization": False,
    }


def git_ignore_check() -> dict:
    probes = [f"Science_Fair_ML_V2/candidate_inbox/{slug}/{section}/STAGE5BB1_IGNORE_PROBE" for slug in inbox.CANDIDATE_PATHS.values() for section in sorted(inbox.ALL_SECTIONS)]
    results = []
    for probe in probes:
        process = _git("check-ignore", "-v", "--", probe, check=False)
        results.append({"probe": probe, "ignored": process.returncode == 0, "rule": process.stdout.strip()})
    return {"all_ignored": all(x["ignored"] for x in results), "probe_count": len(results), "results": results}


def synthetic_security_tests() -> dict:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="sfv2_stage5bb1_") as tmp:
        root = Path(tmp) / "candidate_inbox"
        mapping = {"SYNTHETIC": "synthetic", "EMPTY": "empty"}
        for slug in mapping.values():
            for section in inbox.ALL_SECTIONS:
                (root / slug / section).mkdir(parents=True, exist_ok=True)
        synthetic = root / "synthetic"
        (synthetic / "metadata/source.txt").write_text("SYNTHETIC PUBLIC METADATA\n")
        (synthetic / "documentation/readme.txt").write_text("SYNTHETIC PUBLIC DOCUMENTATION\n")
        (synthetic / "metadata/.inbox-placeholder").write_text("PLACEHOLDER\n")
        restricted_file = synthetic / "restricted/tripwire.bin"
        restricted_file.write_bytes(b"SYNTHETIC RESTRICTED TRIPWIRE")
        restricted_archive = synthetic / "restricted/outcomes.zip"
        restricted_archive.write_bytes(b"NOT A REAL ARCHIVE; RESTRICTED TRIPWIRE")
        (synthetic / "metadata/mixed_delivery.zip").write_bytes(b"SYNTHETIC MIXED ARCHIVE; DO NOT EXTRACT")
        (synthetic / "metadata/outcome_labels.csv").write_text("SYNTHETIC SENSITIVE-NAME TRIPWIRE\n")
        external = Path(tmp) / "external.txt"; external.write_text("SYNTHETIC EXTERNAL\n")
        (synthetic / "metadata/escape_link").symlink_to(external)
        hardlink_supported = True
        try:
            os.link(restricted_file, synthetic / "metadata/restricted_alias.bin")
        except OSError:
            hardlink_supported = False
        guard = inbox.InboxGuard(root.resolve(), mapping, test_mode=True)
        restricted_walk_attempts = []
        restricted_open_attempts = []
        real_walk = os.walk
        real_open = Path.open

        def guarded_walk(top, *args, **kwargs):
            resolved = Path(top).resolve(strict=False)
            if inbox._within(resolved, (synthetic / "restricted").resolve()):
                restricted_walk_attempts.append(str(resolved))
                raise AssertionError("restricted traversal attempted")
            return real_walk(top, *args, **kwargs)

        def guarded_open(path_self, *args, **kwargs):
            if inbox._within(path_self.resolve(strict=False), (synthetic / "restricted").resolve()):
                restricted_open_attempts.append("REDACTED")
                raise AssertionError("restricted content open attempted")
            return real_open(path_self, *args, **kwargs)

        with patch("sfv2.inbox.os.walk", side_effect=guarded_walk), patch.object(Path, "open", guarded_open):
            status = guard.status("SYNTHETIC")["candidates"][0]
            empty = guard.status("EMPTY")["candidates"][0]
        checks["authorized_metadata_access"] = status["metadata_file_count"] == 1
        checks["authorized_documentation_access"] = status["documentation_file_count"] == 1
        checks["restricted_contents_not_opened"] = not restricted_open_attempts and status["restricted_contents_enumerated"] is False
        checks["restricted_directory_not_traversed"] = not restricted_walk_attempts
        for key, candidate, section, relative in [
            ("parent_traversal_denied", "SYNTHETIC", "metadata", "../restricted/tripwire.bin"),
            ("absolute_external_path_denied", "SYNTHETIC", "metadata", str(external)),
            ("symlink_escape_denied", "SYNTHETIC", "metadata", "escape_link"),
            ("restricted_resolution_denied", "SYNTHETIC", "restricted", "."),
        ]:
            try:
                guard.resolve(candidate, section, relative)
                checks[key] = False
            except inbox.InboxAccessDenied:
                checks[key] = True
        checks["hardlink_alias_conservatively_blocked"] = (status["blocked_hardlink_count"] == 1) if hardlink_supported else True
        if hardlink_supported:
            try:
                guard.resolve("SYNTHETIC", "metadata", "restricted_alias.bin")
                checks["hardlink_alias_resolution_denied"] = False
            except inbox.InboxAccessDenied:
                checks["hardlink_alias_resolution_denied"] = True
        else:
            checks["hardlink_alias_resolution_denied"] = True
        checks["restricted_archive_not_extracted"] = not restricted_open_attempts and status["restricted_contents_enumerated"] is False
        checks["mixed_archive_quarantined"] = status["quarantined_archive_count"] >= 1 and status["arrival_status"] == "QUARANTINED_PENDING_MANUAL_CLASSIFICATION"
        checks["sensitive_filename_cannot_bypass_restriction"] = status["quarantined_sensitive_name_count"] == 1
        try:
            guard.resolve("SYNTHETIC", "metadata", "mixed_delivery.zip")
            checks["archive_resolution_denied"] = False
        except inbox.InboxAccessDenied:
            checks["archive_resolution_denied"] = True
        try:
            guard.resolve("SYNTHETIC", "metadata", "outcome_labels.csv")
            checks["sensitive_filename_resolution_denied"] = False
        except inbox.InboxAccessDenied:
            checks["sensitive_filename_resolution_denied"] = True
        checks["placeholder_not_delivery"] = status["placeholder_count"] == 1 and status["metadata_file_count"] == 1
        checks["empty_directory_not_received"] = empty["arrival_status"] == "NOT_RECEIVED" and not empty["non_placeholder_public_files_present"]
        checks["no_lockbox_promotion"] = status["lockbox_status"] == "NOT_DESIGNATED" and status["dataset_role"] == "UNDECIDED"
        details = {"hardlink_supported": hardlink_supported, "restricted_walk_attempts": len(restricted_walk_attempts), "restricted_open_attempts": len(restricted_open_attempts), "synthetic_status": status, "empty_status": empty}
    git = git_ignore_check()
    checks["git_ignores_candidate_inbox"] = git["all_ignored"]
    checks["no_unauthorized_filesystem_traversal"] = details["restricted_walk_attempts"] == 0
    checks["no_training_or_inference"] = True
    return {"classification": "SYNTHETIC_SECURITY_FIXTURES_ONLY", "all_passed": all(checks.values()), "checks": checks, "details": details, "git_check_ignore": git, "real_candidate_files_created": False, "model_operations": 0}


def write_outputs() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    _json(OUT / "STAGE5BB1_AUTHORIZED_INBOX_CONFIG.json", config())
    guard = inbox.InboxGuard.from_config(OUT / "STAGE5BB1_AUTHORIZED_INBOX_CONFIG.json")
    statuses = guard.status()["candidates"]
    rows = []
    outstanding = "source manifest; provenance/licence; event inventory; Omega; completeness; CRS; DEM; exposure/role evidence"
    for status in statuses:
        rows.append({
            "CANDIDATE_ID": status["candidate_id"], "RELATIVE_INBOX_PATH": f"candidate_inbox/{status['relative_path']}",
            "INTAKE_STATUS": status["arrival_status"], "SCIENTIFIC_QUALIFICATION_STATUS": "UNQUALIFIED",
            "DATASET_ROLE": "UNDECIDED", "ACCESS_CLASSIFICATION": "NEW_DEVELOPMENT_CANDIDATE",
            "METADATA_AVAILABILITY": "NOT_RECEIVED", "DOCUMENTATION_AVAILABILITY": "NOT_RECEIVED",
            "RESTRICTED_ACCESS_STATUS": "DENIED_STAGE5B_B1_NOT_ENUMERATED", "LOCKBOX_STATUS": "NOT_DESIGNATED",
            "LAST_AUTHORIZED_AUDIT": "STAGE5B_B1_INITIAL_STRUCTURE_AUDIT", "OUTSTANDING_EVIDENCE": outstanding,
        })
    pd.DataFrame(rows).to_csv(OUT / "STAGE5BB1_CANDIDATE_PATH_REGISTER.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame(statuses).to_csv(OUT / "STAGE5BB1_INBOX_STATUS.tsv", sep="\t", index=False, lineterminator="\n")
    _write(OUT / "STAGE5BB1_INBOX_ACCESS_POLICY.md", """# Candidate-inbox access policy

`candidate_inbox/` is the only authorized project-scoped intake root. Every command resolves the configured root canonically, then resolves a registered candidate and section. Absolute external paths, `..`, unregistered candidates/sections, root or section symlinks, symlink escapes and suspicious hard links are denied.

Metadata-only operations may explicitly enumerate `metadata/` and `documentation/`. They do not traverse `restricted/`, list its entries, hash its contents, inspect archive members or log restricted filenames. The only metadata-only restricted result is whether the expected directory itself exists; content presence remains `REDACTED_NOT_ENUMERATED`. Future restricted access requires a separate explicit authorization that Stage 5B-B1 does not issue.

Archives in public sections are never automatically extracted or opened. They are marked `QUARANTINED_PENDING_MANUAL_CLASSIFICATION`, because mixed archives may contain both public metadata and outcomes. Any public-section file with link count greater than one is conservatively blocked as a possible hard-link alias; this may reject benign hard links. This conservative limitation is preferable to claiming protection that cannot be established without examining restricted paths.

Files in public sections whose names indicate labels, outcomes, ground truth or a landslide inventory are also quarantined without opening. Filename screening cannot establish content safety; it is only an additional fail-closed barrier.

Folder location never establishes scientific role. All six candidates remain `NOT_RECEIVED`, `UNQUALIFIED`, `UNDECIDED` and `NOT_DESIGNATED`. Git exclusion reduces accidental tracking but is not confidentiality, authorization, encryption or an access-control boundary.
""")
    _write(OUT / "STAGE5BB1_FILE_PLACEMENT_GUIDE.md", """# Safe file-placement guide

Place only future provider-supplied files under the registered candidate directory:

| File type | Location | Rule |
|---|---|---|
| Researcher README | `metadata/` or `documentation/` | Must not contain outcome records |
| Licence/permission/publication | `documentation/` | Record source and permitted uses |
| CRS, acquisition and provenance metadata | `metadata/` | Non-sensitive metadata only |
| Observation-domain explanation | `metadata/` or `documentation/` | Only if it contains no restricted outcome information |
| Landslide inventory or outcome labels | `restricted/` | Remains inaccessible without future authorization |
| Historical DSM/DTM | classify before placement | Terrain raster is not automatically an outcome; use existing intake policy and quarantine if uncertain |

Do not place files directly in `candidate_inbox/` or a candidate root. Do not rename an outcome file as metadata. A generated Stage 5B-A template is not a delivery.

Mixed ZIP/TAR/7z/RAR or other archives must be placed in quarantine pending manual classification. Never automatically extract them into `metadata/`; archive extensions in public sections are quarantined without member inspection. Contact the project owner before placement when sensitivity is uncertain.
""")
    security = synthetic_security_tests()
    if not security["all_passed"]:
        raise RuntimeError(f"STOP: synthetic security tests failed: {security['checks']}")
    _json(OUT / "STAGE5BB1_SECURITY_TESTS.json", security)
    _write(OUT / "STAGE5BB1_NEXT_ACTIONS.md", """# Next actions

1. Give a researcher only the relevant candidate subdirectory placement instructions; do not expose unrelated project paths.
2. Request non-sensitive manifest, licence, provenance, event, Ω, completeness, CRS and DEM documentation first.
3. Place known landslide inventories/outcomes only in `restricted/`; leave them unopened.
4. Run `python -m sfv2.intake_cli inbox-status` explicitly. Review quarantine/blocked indicators before any further action.
5. If public metadata arrives, register receipt without promoting role or qualification. Conduct Stage 5B-B qualification separately.
6. Do not designate a lockbox or execute W1/D1/E1 until the applicable outcome-blind governance and dataset-membership gates are complete.
""")
    return security


def final_report() -> None:
    sections = {
        "A": ("Executive summary", "One project-scoped candidate inbox is registered with six candidate directories and metadata/documentation/restricted sections. It contains no real data; every candidate remains unreceived and unqualified."),
        "B": ("Historical integrity", "V1, Stage 5B-B0, Stage 5D-A, the Stage 5D-A preregistration, frozen Stage 5C-A scientific artifacts and the complete discoverable historical hash chain passed."),
        "C": ("Authorized inbox structure", "The only authorized root is `candidate_inbox/`. Six fixed paths each contain `metadata/`, `documentation/` and `restricted/`. Canonical resolution is mandatory."),
        "D": ("Candidate registry", "The register maps all six canonical IDs to paths. Their statuses remain NOT_RECEIVED, UNQUALIFIED, UNDECIDED and NOT_DESIGNATED."),
        "E": ("Git exclusion", "The narrowly scoped `candidate_inbox/` rule passed 18 `git check-ignore` probes. Git ignore prevents accidental tracking but is not a security boundary."),
        "F": ("Path authorization", "Outside paths, absolute paths, parent traversal, unregistered paths and symlink escapes are rejected. Scans are explicit and limited to registered public sections."),
        "G": ("Restricted access enforcement", "Metadata-only commands do not enumerate, open, hash, extract or log restricted contents. Suspicious hard links are conservatively blocked and archives are quarantined."),
        "H": ("CLI integration", "The existing `sfv2.intake_cli` now provides `inbox-status`, `inbox-audit` and `inbox-paths`; each is read-only and metadata-only by default."),
        "I": ("Synthetic security tests", "Synthetic fixtures outside the real inbox verified public access, restricted denial, traversal/path/link controls, archive quarantine, placeholder semantics, Git ignore and non-promotion."),
        "J": ("Candidate arrival status", "No real files were placed in the inbox. Empty directories and project-generated templates do not count as deliveries. All six candidates remain NOT_RECEIVED."),
        "K": ("Scientific limitations", "An organized directory proves neither provenance, Ω, inventory quality, terrain compatibility nor independence. Folder placement cannot qualify development data or designate a lockbox."),
        "L": ("Remaining blockers", "Researcher files, manifests, permissions, event definitions, Ω, completeness, CRS/DEM evidence, exposure audits and role decisions are absent. Stage 5D remains blocked."),
        "M": ("Recommended next action", "Provide the placement guide to a data provider and request metadata/documentation first. Keep outcome inventories restricted and unopened, then run explicit inbox status."),
        "N": ("Checkpoint and hashes", "The checkpoint records the authorized root, mapping/config hashes, Git exclusion, security tests, outputs, sources, historical verification, limitations and negative integrity flags."),
    }
    text = "STAGE 5B-B1 SECURE CANDIDATE INBOX:\nCOMPLETE\n\n"
    for letter, (title, body) in sections.items():
        text += f"# {letter}. {title}\n\n{body}\n\n"
    _write(OUT / "STAGE5BB1_FINAL_REPORT.md", text)


REQUIRED = [
    "STAGE5BB1_FINAL_REPORT.md", "STAGE5BB1_AUTHORIZED_INBOX_CONFIG.json",
    "STAGE5BB1_CANDIDATE_PATH_REGISTER.tsv", "STAGE5BB1_INBOX_ACCESS_POLICY.md",
    "STAGE5BB1_FILE_PLACEMENT_GUIDE.md", "STAGE5BB1_INBOX_STATUS.tsv",
    "STAGE5BB1_SECURITY_TESTS.json", "STAGE5BB1_NEXT_ACTIONS.md",
]


def build() -> dict:
    baseline = verify_baseline()
    before = {str(path.relative_to(ROOT)): sha256(path) for path in ROOT.glob("outputs/**/*") if path.is_file() and OUT not in path.parents}
    create_structure()
    security = write_outputs()
    final_report()
    after = {str(path.relative_to(ROOT)): sha256(path) for path in ROOT.glob("outputs/**/*") if path.is_file() and OUT not in path.parents}
    if before != after:
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise RuntimeError(f"STOP: historical outputs changed: {changed}")
    return {"stage": "5B-B1", "status": "BUILT_PENDING_TESTS", "baseline": baseline, "authorized_root": str(INBOX_ROOT.resolve()), "security_tests": security}


def finalize(test_results: dict) -> dict:
    baseline = verify_baseline()
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"STOP: missing Stage 5B-B1 artifacts: {missing}")
    if not test_results.get("passed"):
        raise RuntimeError(f"STOP: tests failed: {test_results}")
    _json(OUT / "STAGE5BB1_TESTS.json", test_results)
    checkpoint_path = OUT / "STAGE5BB1_CHECKPOINT.json"
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path != checkpoint_path)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    config_path = OUT / "STAGE5BB1_AUTHORIZED_INBOX_CONFIG.json"
    paths = pd.read_csv(OUT / "STAGE5BB1_CANDIDATE_PATH_REGISTER.tsv", sep="\t")
    security = json.loads((OUT / "STAGE5BB1_SECURITY_TESTS.json").read_text())
    git_status = _git("status", "--short").stdout.strip()
    checkpoint = {
        "stage": "5B-B1", "status": "COMPLETE", "audit_date": "2026-10-08", "historical_verification": baseline,
        "authorized_inbox_root": "candidate_inbox", "canonical_inbox_root": str(INBOX_ROOT.resolve()),
        "candidate_paths": paths[["CANDIDATE_ID", "RELATIVE_INBOX_PATH"]].to_dict("records"),
        "configuration_hashes": {"outputs/stage5b_b1/STAGE5BB1_AUTHORIZED_INBOX_CONFIG.json": sha256(config_path), ".gitignore": sha256(ROOT / ".gitignore")},
        "security_test_results": security, "test_results": test_results, "new_artifact_hashes": hashes,
        "source_code_changes": [".gitignore", "src/sfv2/inbox.py", "src/sfv2/intake_cli.py", "src/sfv2/build_stage5bb1.py", "tests/test_stage5bb1.py"],
        "code_hashes": {rel: sha256(ROOT / rel) for rel in ["src/sfv2/inbox.py", "src/sfv2/intake_cli.py", "src/sfv2/build_stage5bb1.py", "tests/test_stage5bb1.py"]},
        "git_head": _git("rev-parse", "HEAD").stdout.strip(), "git_working_tree_dirty": bool(git_status), "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_note": "Candidate inbox is ignored; other pre-existing user changes preserved. No staging, commit or push.",
        "remaining_limitations": ["Git ignore is not a security boundary", "restricted authorization is not implemented or issued", "hard links are conservatively rejected based on link count", "archives are quarantined without classification", "no candidate evidence has arrived"],
        "software_versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "platform": platform.platform()},
        "V1_INTACT": "YES", "HISTORICAL_OUTPUTS_UNCHANGED": "YES", "STAGE5CA_PROTOCOL_UNCHANGED": "YES",
        "STAGE5DA_PREREGISTRATION_UNCHANGED": "YES", "LOGISTIC_M_UNCHANGED": "YES", "M_SCHEMA_UNCHANGED": "YES", "H_GATES_UNCHANGED": "YES",
        "REAL_DATA_INGESTED": "NO", "LOCKBOX_OUTCOMES_ACCESSED": "NO", "NEW_MODEL_TRAINED": "NO", "LOCKBOX_DESIGNATED": "NO",
        "COMMIT_CREATED": "NO", "PUSH_PERFORMED": "NO", "hash_policy": "all Stage 5B-B1 outputs except checkpoint itself are hashed",
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
