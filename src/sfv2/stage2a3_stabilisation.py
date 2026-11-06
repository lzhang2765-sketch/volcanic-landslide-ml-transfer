"""Stage 2A.3: record the checkpoint-stabilisation transition (archived pre-fix checkpoints vs current, field by field).

The fix itself lives in build_stage1.py (schema 1.2.0): volatile git state is written to STAGE_1_GIT_DIAGNOSTICS.json,
which is non-authoritative and never hashed, instead of inside STAGE_1_SCREENING_CHECKPOINT.json.
"""
from __future__ import annotations

import hashlib
import json

from .schema import ROOT

ARCHIVE = ROOT / "outputs" / "checkpoint_history" / "pre_stage2a3"
OUT = ROOT / "outputs" / "stage2a3"
CHECKPOINTS = {"STAGE_1_SCREENING_CHECKPOINT.json": "outputs/screening", "STAGE2A_CHECKPOINT.json": "outputs/stage2a",
               "STAGE2A_EXTENSION_CHECKPOINT.json": "outputs/stage2a_extension", "STAGE2A2_CHECKPOINT.json": "outputs/stage2a2"}
# Every key allowed to differ across the transition; anything else is a scientific change and fails the audit.
EXPECTED_PREFIXES = ("git_context.", "volatile_diagnostics.", "schema_version", "builder.version",
                     "authoritative_file_hashes.src/sfv2/build_stage1.py.", "stage1_hashes.outputs/screening/STAGE_1_SCREENING_CHECKPOINT",
                     "prior_hashes.outputs/screening/STAGE_1_SCREENING_CHECKPOINT", "prior_hashes.outputs/stage2a/STAGE2A_CHECKPOINT",
                     "prior_hashes.outputs/stage2a_extension/STAGE2A_EXTENSION_CHECKPOINT")


def _flat(d, prefix=""):
    if not isinstance(d, dict):
        return {prefix: d}
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}.{k}" if prefix else k))
    return out


def transition_audit() -> dict:
    report = {}
    for name, folder in CHECKPOINTS.items():
        old, new = _flat(json.loads((ARCHIVE / name).read_text())), _flat(json.loads((ROOT / folder / name).read_text()))
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        report[name] = {"archived_sha256": hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest(),
                        "current_sha256": hashlib.sha256((ROOT / folder / name).read_bytes()).hexdigest(),
                        "changed_keys": changed, "unexpected_changes": [k for k in changed if not k.startswith(EXPECTED_PREFIXES)]}
    return report


def main() -> dict:
    report = {"schema": "SFV2_STAGE2A3_TRANSITION_AUDIT_V1", "stage1_schema_from": "1.1.0", "stage1_schema_to": "1.2.0",
              "checkpoints": transition_audit()}
    report["scientific_content_unchanged"] = not any(v["unexpected_changes"] for v in report["checkpoints"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CHECKPOINT_TRANSITION_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    main()
