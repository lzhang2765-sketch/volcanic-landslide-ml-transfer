"""Stage 2F controlled raw transfer and incoming-file registration.

The Puerto Rico transfer is deliberately narrow: exactly the three raw source
files authorised in the Stage 2E plan.  Frozen V1 is verified before and after
the copy and is never opened for writing.  User-supplied files are copied (not
moved) from ``data/incoming_user`` into a write-once raw store.
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

from .raw_store import RawIntegrityError, _sniff, sha256
from .schema import ROOT

REPO = ROOT.parent
FREEZE = REPO / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE"
PLAN = ROOT / "outputs/stage2e/internal/PUERTO_RICO_MARIA_2017_transfer_plan.csv"
OUT = ROOT / "outputs/stage2f/transfers"
INCOMING = ROOT / "data/incoming_user"
INCOMING_RAW = ROOT / "data/raw/stage2f"
STAMP = "2026-10-06T00:00:00Z"

REQUIRED_PROVENANCE = (
    "acquisition_id", "landscape_id", "source_id", "data_role", "dataset_name",
    "original_filename", "source_url", "download_url", "publisher", "access_date",
    "licence", "expected_format", "actual_format", "expected_size", "actual_bytes",
    "sha256", "acquisition_timestamp_utc", "temporal_coverage", "spatial_coverage",
    "resolution", "crs", "raw_path", "verification_status", "acquisition_method",
    "v1_source_path", "v1_unseen_eligibility", "notes",
)


def v1_verify() -> dict:
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


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or tuple(rows[0]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _puerto_provenance(row: dict, destination: Path, digest: str) -> dict:
    name = row["original_filename"]
    downloads = {
        "landslides.zip": "https://www.sciencebase.gov/catalog/file/get/5d4c8b26e4b01d82ce8dfeb0?f=__disk__c9%2Fe9%2F25%2Fc9e925899f8ad6b000ca1178cb2333bef7ccc802",
        "landslide_shapefile_metadata.xml": "https://www.sciencebase.gov/catalog/file/get/5d4c8b26e4b01d82ce8dfeb0?f=__disk__31%2F12%2F06%2F311206616b720af34dab081a1aeb9bb279e5a2be",
        "sciencebase_item_5d4c8b26e4b01d82ce8dfeb0.json": "https://www.sciencebase.gov/catalog/item/5d4c8b26e4b01d82ce8dfeb0?format=json",
    }
    formats = {"landslides.zip": "ZIP", "landslide_shapefile_metadata.xml": "XML",
               "sciencebase_item_5d4c8b26e4b01d82ce8dfeb0.json": "JSON"}
    return {
        "acquisition_id": f"S2F_PR_{name.upper().replace('.', '_')}",
        "landscape_id": "PUERTO_RICO_MARIA_2017",
        "source_id": "USGS_SCIENCEBASE_5D4C8B26E4B01D82CE8DFEB0",
        "data_role": row["scientific_role"],
        "dataset_name": "Map data showing concentration of landslides caused by Hurricane Maria in Puerto Rico",
        "original_filename": name,
        "source_url": "https://www.sciencebase.gov/catalog/item/5d4c8b26e4b01d82ce8dfeb0",
        "download_url": downloads[name],
        "publisher": "U.S. Geological Survey",
        "access_date": "2026-10-06",
        "licence": "PUBLIC_DOMAIN_NO_USE_RESTRICTION",
        "expected_format": formats[name],
        "actual_format": formats[name],
        "expected_size": int(row["v1_bytes"]),
        "actual_bytes": destination.stat().st_size,
        "sha256": digest,
        "acquisition_timestamp_utc": STAMP,
        "temporal_coverage": "2017-09-25/2017-11; post-Hurricane-Maria imagery, with documented Irma/inter-image ambiguity",
        "spatial_coverage": "Puerto Rico main island; documentation reports approximately 97-99% imagery coverage, not an exact Omega polygon",
        "resolution": "Inventory mapped primarily at 1:1,000; generally approximately 3 m positional accuracy",
        "crs": "EPSG:32619 (WGS 84 / UTM zone 19N) for inventory geometry; documentation files non-spatial",
        "raw_path": str(destination.relative_to(ROOT)),
        "verification_status": "HASH_EQUAL_TO_FROZEN_V1_RAW_SOURCE",
        "acquisition_method": "CONTROLLED_V1_TO_V2_RAW_ONLY_COPY",
        "v1_source_path": row["v1_path"],
        "v1_unseen_eligibility": "INELIGIBLE_V1_HISTORICAL_REFERENCE",
        "notes": "Raw source only; no V1-derived mask, feature, model, H surface, or processed product transferred.",
    }


def transfer_puerto_rico() -> dict:
    """Execute the authorised three-file transfer, idempotently and fail-closed."""
    before = v1_verify()
    with PLAN.open(newline="") as f:
        plan = list(csv.DictReader(f))
    if len(plan) != 3 or {Path(r["v1_path"]).name for r in plan} != {
        "landslides.zip", "landslide_shapefile_metadata.xml",
        "sciencebase_item_5d4c8b26e4b01d82ce8dfeb0.json",
    }:
        raise RawIntegrityError("STOP: Stage 2E Puerto Rico plan is not the authorised three-file set")

    frozen = json.loads((FREEZE / "ML_V1_HASH_MANIFEST.json").read_text())["files"]
    logs, provenance = [], []
    for planned in sorted(plan, key=lambda r: r["v1_path"]):
        src = REPO / planned["v1_path"]
        dest = ROOT / planned["planned_v2_destination"]
        expected = planned["v1_sha256"]
        if frozen.get(planned["v1_path"], {}).get("role") != "RAW_INPUT_DATA":
            raise RawIntegrityError(f"STOP: non-raw V1 role: {planned['v1_path']}")
        if sha256(src) != expected or frozen[planned["v1_path"]]["sha256"] != expected:
            raise RawIntegrityError(f"STOP: source hash mismatch: {planned['v1_path']}")
        if dest.exists():
            if sha256(dest) != expected:
                raise RawIntegrityError(f"STOP: immutable destination collision: {dest}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as fin, dest.open("xb") as fout:
                shutil.copyfileobj(fin, fout)
            dest.chmod(0o444)
        digest = sha256(dest)
        if digest != expected:
            raise RawIntegrityError(f"STOP: transfer equality failed: {dest}")
        record = {
            **planned,
            "original_filename": dest.name,
            "v2_destination": str(dest.relative_to(ROOT)),
            "v2_sha256": digest,
            "actual_bytes": dest.stat().st_size,
            "equality": "PASS",
            "destination_status": "PRESENT_IMMUTABLE_HASH_EQUAL",
            "transfer_method": "COPY_READ_ONLY_SOURCE_EXCLUSIVE_DESTINATION_CHMOD_0444",
            "transfer_timestamp_utc": STAMP,
        }
        logs.append(record)
        provenance.append(_puerto_provenance(record, dest, digest))

    after = v1_verify()
    _write_csv(OUT / "PUERTO_RICO_MARIA_2017_transfer_manifest.csv", logs)
    _write_csv(OUT / "PUERTO_RICO_MARIA_2017_transfer_log.csv", logs)
    _write_csv(OUT / "PUERTO_RICO_MARIA_2017_raw_provenance.csv", provenance, REQUIRED_PROVENANCE)
    return {"before": before, "after": after, "logs": logs, "provenance": provenance}


def register_incoming(source: Path, landscape_id: str, data_role: str, metadata: dict,
                      raw_root: Path = INCOMING_RAW) -> dict:
    """Copy one user drop into immutable raw storage; preserve the original."""
    source = source.resolve()
    incoming_root = INCOMING.resolve()
    if incoming_root not in source.parents or not source.is_file():
        raise RawIntegrityError("STOP: source must be a file below data/incoming_user")
    digest = sha256(source)
    dest = raw_root / landscape_id / data_role / source.name
    if dest.exists():
        if sha256(dest) != digest:
            raise RawIntegrityError(f"STOP: immutable destination collision: {dest}")
        newly_written = False
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as fin, dest.open("xb") as fout:
            shutil.copyfileobj(fin, fout)
        dest.chmod(0o444)
        newly_written = True
    required_user = ("source_id", "dataset_name", "source_url", "publisher", "access_date", "licence",
                     "temporal_coverage", "spatial_coverage", "resolution", "crs")
    missing = [k for k in required_user if not metadata.get(k)]
    if missing:
        if newly_written:
            dest.chmod(0o644)
            dest.unlink()
        raise RawIntegrityError(f"STOP: incomplete incoming provenance: {missing}")
    return {
        "acquisition_id": metadata.get("acquisition_id", f"S2F_USER_{landscape_id}_{digest[:12]}"),
        "landscape_id": landscape_id,
        "source_id": metadata["source_id"],
        "data_role": data_role,
        "dataset_name": metadata["dataset_name"],
        "original_filename": source.name,
        "source_url": metadata["source_url"],
        "download_url": metadata.get("download_url", metadata["source_url"]),
        "publisher": metadata["publisher"],
        "access_date": metadata["access_date"],
        "licence": metadata["licence"],
        "expected_format": metadata.get("expected_format", _sniff(source)),
        "actual_format": _sniff(dest),
        "expected_size": metadata.get("expected_size", source.stat().st_size),
        "actual_bytes": dest.stat().st_size,
        "sha256": digest,
        "acquisition_timestamp_utc": metadata.get("acquisition_timestamp_utc", STAMP),
        "temporal_coverage": metadata["temporal_coverage"],
        "spatial_coverage": metadata["spatial_coverage"],
        "resolution": metadata["resolution"],
        "crs": metadata["crs"],
        "raw_path": str(dest.relative_to(ROOT)) if ROOT in dest.parents else str(dest),
        "verification_status": "USER_DROP_HASHED_AND_COPIED_IMMUTABLY",
        "acquisition_method": "USER_AUTHENTICATED_DOWNLOAD_THEN_CONTROLLED_COPY",
        "v1_source_path": "NOT_APPLICABLE",
        "v1_unseen_eligibility": metadata.get("v1_unseen_eligibility", "ELIGIBILITY_REQUIRES_LINEAGE_REVIEW"),
        "notes": metadata.get("notes", "Incoming original preserved; raw copy is immutable."),
    }
