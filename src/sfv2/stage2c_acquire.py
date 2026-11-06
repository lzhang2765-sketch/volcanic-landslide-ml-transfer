"""Stage 2C bounded geology acquisition: GSJ Seamless Digital Geological Map of Japan V2 (1:200,000), official API 1.2.1.

No vector-extract endpoint exists, so geology is acquired as the official z13 geology-layer tiles (layer=g, palette PNG,
~17 m pixels) plus the official bounding-box legend (unit symbol, colour, lithology, age).  Each tile is an immutable raw
file; a tile-set manifest (itself immutable and registered) lists every tile with its SHA256.  Decoding colour -> unit is a
DERIVED operation (stage2c_harmonise.py) validated against independent point-legend queries.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import subprocess

from .raw_store import RAW, STAGING, acquire, load_registry, place_immutable, save_registry, sha256
from .schema import ROOT

API = "https://gbank.gsj.jp/seamless/v2/api/1.2.1"
Z = 13
SOURCE_ID = "S2C_GSJ_SEAMLESS_V2_API"
# Omega bounding boxes (from Stage-2B QC of the official interpretation ranges) + 0.01 deg buffer: (south, west, north, east)
BOXES = {"KUMAMOTO_ASO_2016": (32.42, 130.10, 33.405, 131.532), "HOKKAIDO_IBURI_2018": (42.532, 141.673, 42.963, 142.135),
         "NOTO_PENINSULA_2024": (37.003, 136.67, 37.543, 137.361)}


def tile_range(box):
    s, w, n, e = box
    N = 2 ** Z
    tx = lambda lon: int((lon + 180) / 360 * N)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * N)
    return [(x, y) for x in range(tx(w), tx(e) + 1) for y in range(ty(n), ty(s) + 1)]


def legend_strips(w: float, e: float, max_px: int = 8000) -> list[tuple[float, float]]:
    px_per_deg = 256 * 2 ** Z / 360
    n = math.ceil((e - w) * px_per_deg / max_px)
    step = (e - w) / n
    return [(round(w + i * step, 6), round(w + (i + 1) * step, 6)) for i in range(n)]


def _entry(acq, lid, role, name, fn, url, fmt, notes):
    return dict(acquisition_id=acq, landscape_id=lid, source_id=SOURCE_ID, data_role=role, dataset_name=name, original_filename=fn,
                source_url=f"{API}/", download_url=url, publisher="Geological Survey of Japan (AIST)", access_date="2026-10-05",
                licence="Government Standard Terms of Use (Version 2.0); attribution required", expected_format=fmt, expected_size="UNKNOWN",
                temporal_coverage="Seamless V2 as served by API 1.2.1 at access date", spatial_coverage=f"Bounding box of {lid} Omega",
                resolution="1:200,000 source map; z13 tiles (~17 m pixels)", verification_status="DOWNLOADED_PENDING_QC", notes=notes)


def _fetch_tile(lid, x, y):
    dest = RAW / lid / SOURCE_ID / "tiles_z13_layer_g" / f"{Z}_{y}_{x}.png"
    if dest.exists():
        return dict(z=Z, x=x, y=y, file=str(dest.relative_to(ROOT)), sha256=sha256(dest), bytes=dest.stat().st_size)
    STAGING.mkdir(parents=True, exist_ok=True)
    part = STAGING / f"{lid}_{Z}_{y}_{x}.part"
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-o", str(part), f"{API}/tiles/{Z}/{y}/{x}.png?layer=g"], check=True)
    digest, _ = place_immutable(part, dest)
    return dict(z=Z, x=x, y=y, file=str(dest.relative_to(ROOT)), sha256=digest, bytes=dest.stat().st_size)


def run() -> dict:
    registry, summary = load_registry(), {}
    for lid, box in BOXES.items():
        s, w, n, e = box
        for k, (ws, es) in enumerate(legend_strips(w, e)):  # API limit: box width < 8182 px at z13
            for lang in ("en", "ja"):
                acquire(_entry(f"ACQ_{lid.split('_')[0]}_GSJ_LEGEND_{lang.upper()}_S{k}", lid, "GEOLOGY_LEGEND",
                               f"GSJ Seamless V2 legend for Omega box strip {k} ({lang})", f"legend_box_strip{k}_{lang}.json",
                               f"{API}/legend.json?box={s},{ws},{n},{es}&lang={lang}", "JSON",
                               "Official legend (symbol, RGB value, lithology, formation age) for all units in the strip"), registry)
        tiles = tile_range(box)
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            rows = sorted(pool.map(lambda t: _fetch_tile(lid, *t), tiles), key=lambda r: (r["x"], r["y"]))
        manifest = {"schema": "SFV2_TILESET_MANIFEST_V1", "landscape_id": lid, "url_template": f"{API}/tiles/{{z}}/{{y}}/{{x}}.png?layer=g",
                    "zoom": Z, "box_south_west_north_east": box, "tile_count": len(rows), "tiles": rows}
        STAGING.mkdir(parents=True, exist_ok=True)
        part = STAGING / f"{lid}_tileset.part"
        part.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        fn = "tileset_manifest_z13_layer_g.json"
        dest = RAW / lid / SOURCE_ID / fn
        digest, _ = place_immutable(part, dest)
        acq = f"ACQ_{lid.split('_')[0]}_GSJ_TILESET_Z13"
        if not any(r["acquisition_id"] == acq for r in registry):
            entry = _entry(acq, lid, "GEOLOGY", "GSJ Seamless V2 geology-layer tiles z13 (tile-set manifest)", fn, manifest["url_template"],
                           "JSON manifest + PNG tiles", f"{len(rows)} tiles; manifest lists per-tile SHA256")
            import datetime as dt
            registry.append(dict(entry, sha256=digest, actual_bytes=dest.stat().st_size, raw_path=str(dest.relative_to(ROOT)), actual_format="JSON",
                                 acquisition_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 tileset_tile_count=len(rows), tileset_total_bytes=sum(r["bytes"] for r in rows)))
        summary[lid] = dict(tiles=len(rows), bytes=sum(r["bytes"] for r in rows))
        save_registry(registry)
    return summary


def acquire_point_validation(n: int = 60, seed: int = 20261005) -> dict:
    """Independent check of tile decoding: official legend.json point queries at seeded random points inside each Omega."""
    import datetime as dt
    import numpy as np
    from shapely.geometry import Point
    from . import build_stage2b as b2
    registry, out = load_registry(), {}
    for lid in BOXES:
        acq = f"ACQ_{lid.split('_')[0]}_GSJ_POINT_VALIDATION"
        if any(e["acquisition_id"] == acq for e in registry):
            continue
        omega = b2.landscape_qc(lid, b2.stage2b_scope(registry))["omega"]
        rng, pts = np.random.default_rng(seed), []
        minx, miny, maxx, maxy = omega.bounds
        while len(pts) < n:
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if omega.contains(Point(x, y)):
                pts.append((round(y, 6), round(x, 6)))
        results = []
        for lat, lon in pts:
            txt = subprocess.run(["curl", "-sSL", "--fail", f"{API}/legend.json?point={lat},{lon}"], check=False, capture_output=True, text=True).stdout
            j = json.loads(txt) if txt.strip() else {}
            results.append(dict(lat=lat, lon=lon, symbol=j.get("symbol") if isinstance(j, dict) else None, response=j))
        STAGING.mkdir(parents=True, exist_ok=True)
        part = STAGING / f"{lid}_pointval.part"
        part.write_text(json.dumps({"schema": "SFV2_GSJ_POINT_VALIDATION_V1", "seed": seed, "n": n, "points": results}, indent=2, sort_keys=True,
                                   ensure_ascii=False) + "\n")
        fn = "point_validation_legend_en.json"
        dest = RAW / lid / SOURCE_ID / fn
        digest, _ = place_immutable(part, dest)
        registry.append(dict(_entry(acq, lid, "GEOLOGY_VALIDATION", "GSJ legend point queries for decoding validation", fn,
                                    f"{API}/legend.json?point=<lat>,<lon>", "JSON", f"{n} seeded points inside official Omega (independent of Y)"),
                             sha256=digest, actual_bytes=dest.stat().st_size, raw_path=str(dest.relative_to(ROOT)), actual_format="JSON",
                             acquisition_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        save_registry(registry)
        out[lid] = sum(r["symbol"] is not None for r in results)
    return out


if __name__ == "__main__":
    print(run())
