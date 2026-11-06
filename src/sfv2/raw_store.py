"""Immutable, checksum-registered raw-data store for Stage 2B.

Rules: a raw file is written once, then made read-only; re-acquiring an existing file must reproduce the registered
SHA256 or acquisition STOPs (RawIntegrityError); every file has one registry entry carrying full provenance.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .schema import ROOT

RAW = ROOT / "data" / "raw" / "stage2b"
STAGING = ROOT / "data" / "interim" / "_stage2b_downloads"
REGISTRY = ROOT / "metadata" / "stage2b_raw_registry.json"
REQUIRED = ("acquisition_id", "landscape_id", "source_id", "data_role", "dataset_name", "original_filename", "source_url", "download_url",
            "publisher", "access_date", "licence", "expected_format", "actual_format", "expected_size", "actual_bytes", "sha256",
            "acquisition_timestamp_utc", "temporal_coverage", "spatial_coverage", "resolution", "raw_path", "verification_status", "notes")


class RawIntegrityError(RuntimeError):
    """A raw file would be overwritten or a re-download disagrees with its registered checksum."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path = REGISTRY) -> list[dict]:
    return json.loads(path.read_text())["entries"] if path.exists() else []


def save_registry(entries: list[dict], path: Path = REGISTRY) -> None:
    ids = [e["acquisition_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise RawIntegrityError("duplicate acquisition_id in registry")
    for e in entries:
        missing = [k for k in REQUIRED if e.get(k) in (None, "")]
        if missing:
            raise RawIntegrityError(f"{e.get('acquisition_id')}: missing provenance {missing}")
    path.write_text(json.dumps({"schema": "SFV2_STAGE2B_RAW_REGISTRY_V1", "entries": sorted(entries, key=lambda e: e["acquisition_id"])},
                               indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def place_immutable(staged: Path, dest: Path) -> tuple[str, bool]:
    """Move a staged file into the raw store. Returns (sha256, newly_written). Never overwrites."""
    digest = sha256(staged)
    if dest.exists():
        existing = sha256(dest)
        staged.unlink()
        if existing != digest:
            raise RawIntegrityError(f"STOP: re-download of {dest} has sha256 {digest} but stored file has {existing}")
        return existing, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), dest)
    dest.chmod(0o444)
    return digest, True


def acquire(entry: dict, registry: list[dict]) -> dict:
    """Download entry['download_url'] into the raw store and register it (idempotent, fail-closed)."""
    dest = RAW / entry["landscape_id"] / entry["source_id"] / entry["original_filename"]
    prior = next((e for e in registry if e["acquisition_id"] == entry["acquisition_id"]), None)
    STAGING.mkdir(parents=True, exist_ok=True)
    staged = STAGING / f"{entry['acquisition_id']}.part"
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-A", "Mozilla/5.0", "-o", str(staged), entry["download_url"]], check=True)
    digest, new = place_immutable(staged, dest)
    if prior:
        if prior["sha256"] != digest:
            raise RawIntegrityError(f"STOP: {entry['acquisition_id']} hash changed from {prior['sha256']} to {digest}")
        return prior
    record = dict(entry, sha256=digest, actual_bytes=dest.stat().st_size, raw_path=str(dest.relative_to(ROOT)),
                  acquisition_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  actual_format=_sniff(dest))
    registry.append(record)
    return record


def _sniff(path: Path) -> str:
    raw = path.read_bytes()[:512]
    head = raw[3:11] if raw.startswith(b"\xef\xbb\xbf") else raw[:8]
    if b"<html" in raw.lower():
        return "HTML"
    if head.startswith(b"PK"):
        return "ZIP"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "GeoTIFF"
    text = head.decode("latin-1").lstrip().lower()
    if text.startswith(("<?xm", "<kml")):
        return "XML/KML"
    if text.startswith(("<!do", "<htm")):
        return "HTML"
    if text.startswith("%pdf"):
        return "PDF"
    if text.startswith("{"):
        return "JSON"
    return "UNKNOWN_BINARY"
