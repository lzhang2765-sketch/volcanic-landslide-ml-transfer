"""Controlled V1 -> V2 transfer for the Stage 2D Lombok and Atitlan audit.

Only files labelled RAW_INPUT_DATA in the frozen V1 manifest are eligible.  The
source is opened read-only, both frozen manifests are checked, and a destination
is created exclusively and made read-only.  No V1 processed/H/feature/ML output
can enter the transfer plan.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .raw_store import RawIntegrityError, sha256
from .schema import ROOT

REPO = ROOT.parent
FREEZE = REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE"
HASH_MANIFEST = FREEZE / "ML_V1_HASH_MANIFEST.json"
FILE_MANIFEST = FREEZE / "ML_V1_FILE_MANIFEST.tsv"
OUT = ROOT / "outputs/stage2d/transfers"
RAW = ROOT / "data/raw/stage2d"

LANDSCAPE_PREFIXES = {
    "LOMBOK_RINJANI_2018": (
        "Science_Fair_Extension/01_Data_Raw/Lombok/",
        "Science_Fair_Extension/01_Data_Raw/SF1_verification/lombok_2018_ferrario/",
    ),
    "ATITLAN_STAN_2005": ("Science_Fair_Extension/01_Data_Raw/Atitlan/",),
}


def _v1_verify() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "tweed_science_fair.ml_v1_freeze", "verify"],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)
    if not record["v1_intact"] or record["writes_performed"]:
        raise RawIntegrityError(f"STOP: frozen V1 integrity failed: {record}")
    return record


def _role(path: str) -> str:
    bits = Path(path).parts
    if "SF1_verification" in bits:
        name = Path(path).name.lower()
        if name.startswith("mapping_extent"):
            return "OMEGA"
        if name.startswith("ferrario_2019"):
            return "Y"
        return "Y_METHOD"
    for token, role in (("Y", "Y"), ("M", "DEM"), ("G", "GEOLOGY"),
                        ("E", "ENVIRONMENT"), ("Domain", "OMEGA_CONTEXT"),
                        ("Metadata", "DOCUMENTATION")):
        if token in bits:
            return role
    raise RawIntegrityError(f"STOP: no controlled role for {path}")


def plan(landscape_id: str) -> list[dict]:
    frozen = json.loads(HASH_MANIFEST.read_text())["files"]
    with FILE_MANIFEST.open(newline="") as f:
        tab = {r["path"]: r for r in csv.DictReader(f, delimiter="\t")}
    paths = sorted(p for p in frozen if any(p.startswith(q) for q in LANDSCAPE_PREFIXES[landscape_id]))
    rows = []
    for p in paths:
        h, meta = frozen[p], tab.get(p)
        if not meta or h.get("role") != "RAW_INPUT_DATA" or meta.get("role") != "RAW_INPUT_DATA":
            continue
        role = _role(p)
        # Prefix the verification-subtree files to prevent basename collisions.
        sub = "SF1_VERIFICATION_" + Path(p).name if "SF1_verification" in p else Path(p).name
        dest = f"data/raw/stage2d/{landscape_id}/V1_TRANSFER/{role}/{sub}"
        rows.append({
            "landscape_id": landscape_id,
            "v1_path": p,
            "v1_sha256": h["sha256"],
            "v1_bytes": h["size_bytes"],
            "v1_stage": h["stage"],
            "v1_role": h["role"],
            "v2_destination": dest,
            "scientific_role": role,
            "eligibility": "ELIGIBLE_RAW_SOURCE_ONLY",
            "v2_unseen_test_eligibility": "INELIGIBLE_V1_HISTORICAL",
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def transfer(landscape_id: str) -> list[dict]:
    _v1_verify()
    frozen = json.loads(HASH_MANIFEST.read_text())["files"]
    planned = plan(landscape_id)
    if not planned:
        raise RawIntegrityError(f"STOP: empty transfer plan for {landscape_id}")
    _write_csv(OUT / f"{landscape_id}_transfer_manifest.csv", planned)
    log_path = OUT / f"{landscape_id}_transfer_log.csv"
    prior = {}
    if log_path.exists():
        with log_path.open(newline="") as f:
            prior = {r["v1_path"]: r for r in csv.DictReader(f)}
    log = []
    for r in planned:
        src, dest = REPO / r["v1_path"], ROOT / r["v2_destination"]
        before = sha256(src)
        if before != r["v1_sha256"] or before != frozen[r["v1_path"]]["sha256"]:
            raise RawIntegrityError(f"STOP: V1 hash disagreement for {r['v1_path']}")
        if dest.exists():
            if sha256(dest) != before:
                raise RawIntegrityError(f"STOP: existing V2 destination differs: {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as fin, dest.open("xb") as fout:
                shutil.copyfileobj(fin, fout)
            dest.chmod(0o444)
        after = sha256(dest)
        if after != before:
            raise RawIntegrityError(f"STOP: transferred hash mismatch for {r['v1_path']}")
        log.append({
            **r,
            "v2_sha256": after,
            "equality": "PASS",
            "transfer_method": "COPY_VERIFIED_READ_ONLY_SOURCE_EXCLUSIVE_DESTINATION_CHMOD_0444",
            "transfer_timestamp_utc": prior.get(r["v1_path"], {}).get("transfer_timestamp_utc")
                                      or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    _v1_verify()
    _write_csv(log_path, log)
    return log


def audit() -> dict[str, list[dict]]:
    return {lid: transfer(lid) for lid in LANDSCAPE_PREFIXES}
