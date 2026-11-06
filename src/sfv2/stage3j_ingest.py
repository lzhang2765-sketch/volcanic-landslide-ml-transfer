"""Fail-closed response ingestion for Stage 3J custodian replies.

This module inventories and preserves received files.  It deliberately stops
before gate admission, production transforms, terrain-unit generation, or ML.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any

from .schema import ROOT


CONFIG = ROOT / "config/stage3j_response_routes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _base_metadata(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "byte_size": path.stat().st_size,
        "sha256": sha256(path),
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def _zip_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        unsafe = [m.filename for m in members if Path(m.filename).is_absolute() or ".." in Path(m.filename).parts]
        bad_crc = archive.testzip()
        names = [m.filename for m in members]
    shp_stems = {Path(n).stem for n in names if Path(n).suffix.lower() == ".shp"}
    complete = []
    for stem in sorted(shp_stems):
        suffixes = {Path(n).suffix.lower() for n in names if Path(n).stem == stem}
        complete.append({"stem": stem, "required_components_present": {".shp", ".shx", ".dbf"} <= suffixes})
    return {
        "zip_member_count": len(names),
        "zip_members": names,
        "zip_crc_ok": bad_crc is None,
        "zip_unsafe_paths": unsafe,
        "shapefile_bundles": complete,
    }


def _csv_metadata(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return {"csv_columns": header, "csv_data_rows": rows}


def _raster_metadata(path: Path) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError:
        return {"spatial_inspector": "rasterio unavailable; route to raster QA"}
    with rasterio.open(path) as dataset:
        return {
            "raster_width": dataset.width,
            "raster_height": dataset.height,
            "raster_bands": dataset.count,
            "raster_dtypes": list(dataset.dtypes),
            "crs": str(dataset.crs) if dataset.crs else None,
            "bounds": [_json_safe(v) for v in dataset.bounds],
            "resolution": [_json_safe(v) for v in dataset.res],
            "nodata": _json_safe(dataset.nodata),
            "mask_flags": [[str(flag) for flag in flags] for flags in dataset.mask_flag_enums],
        }


def _vector_metadata(path: Path) -> dict[str, Any]:
    try:
        import geopandas as gpd
    except ImportError:
        return {"spatial_inspector": "geopandas unavailable; route to vector QA"}
    try:
        frame = gpd.read_file(path)
    except Exception as exc:  # malformed or driver unavailable must remain fail-closed
        return {"vector_read_error": f"{type(exc).__name__}: {exc}"}
    geometry = frame.geometry
    return {
        "feature_count": len(frame),
        "crs": str(frame.crs) if frame.crs else None,
        "bounds": [_json_safe(v) for v in frame.total_bounds],
        "geometry_types": sorted(str(v) for v in geometry.geom_type.dropna().unique()),
        "null_geometry_count": int(geometry.isna().sum()),
        "empty_geometry_count": int(geometry.is_empty.sum()),
        "invalid_geometry_count": int((~geometry.is_valid & ~geometry.isna()).sum()),
        "attribute_fields": [str(column) for column in frame.columns if column != frame.geometry.name],
    }


def inspect_file(path: Path) -> dict[str, Any]:
    result = _base_metadata(path)
    extension = result["extension"]
    if extension in {".zip", ".kmz"}:
        result.update(_zip_metadata(path))
    if extension == ".csv":
        result.update(_csv_metadata(path))
    if extension in {".tif", ".tiff"}:
        result.update(_raster_metadata(path))
    if extension in {".shp", ".gpkg", ".geojson", ".kml"}:
        result.update(_vector_metadata(path))
    return result


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text())


def match_requested_artifacts(metadata: dict[str, Any], route: dict[str, Any]) -> list[str]:
    filename = metadata["filename"].lower()
    extension = metadata["extension"]
    matches = []
    for artifact in route["requested_artifacts"]:
        extension_ok = extension in artifact["accepted_extensions"]
        term_ok = any(term.lower() in filename for term in artifact["filename_terms"])
        if extension_ok and term_ok:
            matches.append(artifact["artifact_id"])
    return matches


def inventory_folder(incoming: Path, candidate_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    if candidate_id not in config["candidates"]:
        raise ValueError(f"Unknown candidate_id: {candidate_id}")
    if not incoming.is_dir():
        raise NotADirectoryError(incoming)
    route = config["candidates"][candidate_id]
    allowed = set(config["allowed_extensions"])
    records = []
    for path in sorted(p for p in incoming.rglob("*") if p.is_file()):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not admitted: {path}")
        metadata = inspect_file(path)
        metadata["relative_incoming_path"] = str(path.relative_to(incoming))
        shapefile_sidecars = {".cpg", ".dbf", ".prj", ".shx"}
        has_sibling_shp = (path.parent / f"{path.stem}.shp").is_file()
        metadata["supported"] = metadata["extension"] in allowed and (
            metadata["extension"] not in shapefile_sidecars or has_sibling_shp
        )
        metadata["matched_artifact_ids"] = match_requested_artifacts(metadata, route)
        metadata["qa_route"] = route["qa_route"]
        records.append(metadata)
    return {
        "schema_version": config["schema_version"],
        "candidate_id": candidate_id,
        "incoming_folder": str(incoming.resolve()),
        "minimum_metadata_required": config["minimum_metadata"],
        "qa_route": route["qa_route"],
        "requested_artifacts": route["requested_artifacts"],
        "files": records,
        "file_count": len(records),
        "all_extensions_supported": all(record["supported"] for record in records),
        "admission_decision": "NOT_EVALUATED_ROUTE_TO_CANDIDATE_QA",
        "production_authorized": False,
    }


def preserve_immutable(
    inventory: dict[str, Any],
    incoming: Path,
    received_at: str,
    raw_root: Path | None = None,
) -> Path:
    """Copy a received bundle to hash-addressed, read-only raw storage."""
    config = load_config()
    raw_root = raw_root or ROOT / config["raw_root"]
    safe_time = received_at.replace(":", "").replace("+", "p").replace("-", "")
    destination = raw_root / inventory["candidate_id"] / safe_time
    if destination.exists():
        raise FileExistsError(f"Immutable destination already exists: {destination}")
    destination.mkdir(parents=True)
    for record in inventory["files"]:
        source = incoming / record["relative_incoming_path"]
        target = destination / record["sha256"][:16] / source.name
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
        if sha256(target) != record["sha256"]:
            raise IOError(f"Checksum mismatch after copy: {source}")
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        record["preserved_raw_path"] = str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)
    manifest = destination / "INGESTION_MANIFEST.json"
    manifest.write_text(json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.chmod(manifest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--received-at", help="ISO-8601 receipt time; required with --execute")
    parser.add_argument("--execute", action="store_true", help="Copy to immutable raw storage; default is inventory-only")
    args = parser.parse_args(argv)
    inventory = inventory_folder(args.incoming, args.candidate)
    if args.execute:
        if not args.received_at:
            parser.error("--received-at is required with --execute")
        destination = preserve_immutable(inventory, args.incoming, args.received_at)
        inventory["preserved_bundle"] = str(destination)
    print(json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
