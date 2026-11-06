"""Authorised (Stage 2C) controlled V1 -> V2 raw transfer for KURIKOMA_2008.

Implements docs/KURIKOMA_V1_TO_V2_TRANSFER_PROCEDURE.md exactly: verify V1 -> hash source against both frozen V1
manifests -> copy (source opened read-only) -> hash destination -> chmod 0444 -> register -> verify V1 again.
Any hash disagreement STOPs the transfer (RawIntegrityError).  Only the 22 prepared raw files are eligible.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

from .raw_store import RawIntegrityError, load_registry, save_registry, sha256
from .schema import ROOT

REPO = ROOT.parent
PREPARED = ROOT / "outputs" / "stage2b" / "kurikoma_transfer_manifest.csv"
V1_HASHES = REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_HASH_MANIFEST.json"
LOG = ROOT / "outputs" / "stage2c" / "kurikoma_transfer_log.csv"
FORBIDDEN = ("02_Data_Processed", "03_Reconstruction", "04_Feature_Engineering", "05_ML_Dataset", "06_Models", "08_Predictions")


def _v1_verify() -> dict:
    out = subprocess.run([sys.executable, "-m", "tweed_science_fair.ml_v1_freeze", "verify"], cwd=REPO,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=True, capture_output=True, text=True).stdout
    result = json.loads(out)
    if not result["v1_intact"] or result["writes_performed"]:
        raise RawIntegrityError(f"STOP: V1 freeze not intact: {result}")
    return result


def run() -> list[dict]:
    with PREPARED.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 22:
        raise RawIntegrityError(f"STOP: expected 22 prepared files, found {len(rows)}")
    frozen = json.loads(V1_HASHES.read_text())["files"]
    _v1_verify()
    registry, log = load_registry(), []
    for r in rows:
        if any(tag in r["v1_path"] for tag in FORBIDDEN) or not r["v1_path"].startswith("Science_Fair_Extension/01_Data_Raw/Kurikoma/"):
            raise RawIntegrityError(f"STOP: ineligible V1 path {r['v1_path']}")
        src, dest = REPO / r["v1_path"], ROOT / r["v2_destination"]
        source_sha = sha256(src)
        if not (source_sha == r["v1_sha256"] == frozen[r["v1_path"]]["sha256"]):
            raise RawIntegrityError(f"STOP: V1 hash disagreement for {r['v1_path']}")
        acq = f"ACQ_KURIKOMA_V1T_{r['v2_role']}_{os.path.basename(dest).replace('.', '_')}"
        prior = next((e for e in registry if e["acquisition_id"] == acq), None)
        if dest.exists():
            if sha256(dest) != source_sha:
                raise RawIntegrityError(f"STOP: existing destination differs for {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as fin, dest.open("xb") as fout:  # read-only source, exclusive-create destination
                shutil.copyfileobj(fin, fout)
            dest.chmod(0o444)
        dest_sha = sha256(dest)
        equal = dest_sha == source_sha
        if not equal:
            dest.chmod(0o644)
            dest.unlink()
            raise RawIntegrityError(f"STOP: transferred hash mismatch for {r['v1_path']}")
        ts = prior["acquisition_timestamp_utc"] if prior else dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not prior:
            registry.append(dict(
                acquisition_id=acq, landscape_id="KURIKOMA_2008", source_id="V1_FREEZE_TRANSFER", data_role=r["v2_role"],
                dataset_name=f"V1 frozen raw input ({r['v1_role']})", original_filename=os.path.basename(dest), source_url=f"repo://{r['v1_path']}",
                download_url=f"repo://{r['v1_path']}", publisher="Frozen ML V1 (original publishers per V1 provenance records)", access_date="2026-10-05",
                licence="As recorded in V1 provenance for the original source", expected_format=os.path.splitext(dest)[1].lstrip(".").upper() or "UNKNOWN",
                actual_format=os.path.splitext(dest)[1].lstrip(".").upper() or "UNKNOWN", expected_size=r["v1_bytes"], actual_bytes=dest.stat().st_size,
                sha256=dest_sha, acquisition_timestamp_utc=ts, temporal_coverage="As per V1 source", spatial_coverage="Kurikoma V1 study window",
                resolution="As per V1 source", raw_path=r["v2_destination"], verification_status="TRANSFERRED_BYTE_IDENTICAL",
                notes=f"Controlled V1->V2 transfer; v1_path={r['v1_path']}; v1_sha256={r['v1_sha256']}; method=COPY_VERIFIED; "
                      "v2_unseen_test_eligibility=INELIGIBLE_V1_HISTORICAL",
                v1_path=r["v1_path"], v1_sha256=r["v1_sha256"], v2_unseen_test_eligibility="INELIGIBLE_V1_HISTORICAL"))
        log.append(dict(v1_path=r["v1_path"], v1_frozen_sha256=r["v1_sha256"], v2_destination=r["v2_destination"], transferred_sha256=dest_sha,
                        transfer_method="COPY_VERIFIED (read-only source, exclusive-create destination, chmod 0444)", timestamp_utc=ts,
                        role=r["v2_role"], equality="PASS" if equal else "FAIL", v2_unseen_test_eligibility="INELIGIBLE_V1_HISTORICAL"))
    save_registry(registry)
    _v1_verify()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(log)
    return log


if __name__ == "__main__":
    run()
