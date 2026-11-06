"""Stage 4A environmental (E) predictors: governed Stage 2C allowlist only (CHELSA V2.1 bio1/4/12/13/15; Hansen treecover2000).

Raw handling: source-native WINDOWS of the official global rasters are read over HTTP range requests and written with the
same grid, dtype, scale/offset and no resampling (byte content of the pixels is unchanged; only the extent is cropped).
Each window is hash-registered as an immutable raw object.  Unit features are then sampled, never interpolated:
  * CHELSA (30 arcsec ~ 0.9 km, much coarser than 90 m): value of the cell containing the unit centroid -> CONTEXTUAL; no SD.
  * Hansen (1 arcsec ~ 30 m): nearest pixel at each of the 9 frozen 30 m cell centres -> unit mean and SD (LOCAL).
"""
from __future__ import annotations

import datetime as dt
import math
import subprocess

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds

from .raw_store import STAGING, load_registry, place_immutable, save_registry
from .schema import ROOT

AS_OF = "2026-10-06"
RAW = ROOT / "data/raw/stage4a"
CHELSA_URL = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/CHELSA_{v}_1981-2010_V.2.1.tif"
HANSEN_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_treecover2000_{tile}.tif"
CHELSA_VARS = {  # CHELSA V2.1 technical specification (Karger et al.); scale/offset are read from the file itself
    "bio1": ("e_chelsa_bio1_mean_annual_temp", "degC", "Mean annual near-surface air temperature"),
    "bio4": ("e_chelsa_bio4_temp_seasonality", "degC/100", "Temperature seasonality (SD of monthly means x 100)"),
    "bio12": ("e_chelsa_bio12_annual_precip", "kg m-2 yr-1", "Annual precipitation amount"),
    "bio13": ("e_chelsa_bio13_wettest_month_precip", "kg m-2 month-1", "Precipitation of the wettest month"),
    "bio15": ("e_chelsa_bio15_precip_seasonality", "percent (CV)", "Precipitation seasonality (coefficient of variation)"),
}
HANSEN_TILES = {"KUMAMOTO_ASO_2016": "40N_130E", "NOTO_PENINSULA_2024": "40N_130E", "LOMBOK_RINJANI_2018": "00N_110E",
                "COTABATO_DAVAO_2019": "10N_120E", "ATITLAN_STAN_2005": "20N_100W", "DOMINICA_ERIKA_2015": "20N_070W"}
LICENCE = {"CHELSA": "CC BY 4.0 (CHELSA V2.1; cite Karger et al. 2017 Sci Data 4:170122 and CHELSA V2.1 data release)",
           "HANSEN": "CC BY 4.0 (Hansen et al. 2013 Science 342:850; GFC-2024-v1.12)"}
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
PAD_DEG = 0.03


def unit_lonlat(terrain: pd.DataFrame, offsets: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unit centroids (offsets=False) or the 9 frozen 30 m cell centres (offsets=True, shape n x 9) in EPSG:4326."""
    crs = terrain.projected_crs.iloc[0]
    x, y = terrain.centroid_x.to_numpy(), terrain.centroid_y.to_numpy()
    if offsets:
        d = np.array([-30.0, 0.0, 30.0])
        dx, dy = np.meshgrid(d, d)
        x = (x[:, None] + dx.ravel()[None, :]).ravel()
        y = (y[:, None] + dy.ravel()[None, :]).ravel()
    lon, lat = warp_transform(crs, "EPSG:4326", x, y)
    lon, lat = np.asarray(lon), np.asarray(lat)
    return (lon.reshape(-1, 9), lat.reshape(-1, 9)) if offsets else (lon, lat)


def _head(url: str) -> dict:
    out = subprocess.run(["curl", "-sI", "-L", "-m", "60", url], capture_output=True, text=True).stdout
    h = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in out.splitlines() if ":" in ln)}
    return dict(etag=h.get("etag", ""), content_length=h.get("content-length", ""), last_modified=h.get("last-modified", ""))


def extract_window(lid: str, source: str, var: str, url: str, bounds: tuple) -> dict:
    """Crop a source-native window (no resampling) and register it immutably; idempotent and fail-closed."""
    acq = f"ACQ_E_{source}_{var.upper()}_{lid}"
    registry = load_registry()
    prior = next((e for e in registry if e["acquisition_id"] == acq), None)
    dest = RAW / lid / source / f"{url.rsplit('/', 1)[1][:-4]}__window.tif"
    if prior and dest.exists():
        return prior
    with rasterio.Env(**ENV), rasterio.open(f"/vsicurl/{url}") as src:
        f = from_bounds(*bounds, transform=src.transform)
        c0, r0 = math.floor(f.col_off), math.floor(f.row_off)
        w = Window(c0, r0, math.ceil(f.col_off + f.width) - c0, math.ceil(f.row_off + f.height) - r0)  # outward, whole pixels
        arr = src.read(1, window=w)
        prof = src.profile.copy()
        prof.update(width=int(w.width), height=int(w.height), transform=src.window_transform(w), tiled=False, compress="deflate",
                    blockxsize=None, blockysize=None)
        for k in ("blockxsize", "blockysize"):
            prof.pop(k, None)
        meta = dict(scale=float(src.scales[0]), offset=float(src.offsets[0]), nodata=src.nodata, crs=src.crs.to_string(),
                    res_deg=float(src.transform.a), source_width=src.width, source_height=src.height, dtype=str(arr.dtype),
                    window=dict(col_off=int(w.col_off), row_off=int(w.row_off), width=int(w.width), height=int(w.height)))
    STAGING.mkdir(parents=True, exist_ok=True)
    staged = STAGING / f"{acq}.part.tif"
    with rasterio.open(staged, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.scales, dst.offsets = (meta["scale"],), (meta["offset"],)
        dst.update_tags(SOURCE_URL=url, EXTRACTION="SOURCE_NATIVE_WINDOW_NO_RESAMPLING", WINDOW=str(meta["window"]))
    digest, _ = place_immutable(staged, dest)
    head = _head(url)
    rec = dict(acquisition_id=acq, landscape_id=lid, source_id=f"SRC_{source}", data_role="E", dataset_name=f"{source} {var} source-native window",
               original_filename=dest.name, source_url=url, download_url=url, publisher="WSL / CHELSA" if source == "CHELSA" else "University of Maryland GLAD (Hansen et al.)",
               version="CHELSA V2.1, climatology 1981-2010" if source == "CHELSA" else "GFC-2024-v1.12 treecover2000",
               access_date=AS_OF, licence=LICENCE[source], expected_format="GeoTIFF", actual_format="GeoTIFF", sha256=digest,
               actual_bytes=dest.stat().st_size, raw_path=str(dest.relative_to(ROOT)), source_etag=head["etag"],
               source_content_length=head["content_length"], source_last_modified=head["last_modified"],
               acquisition_timestamp_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               extraction="SOURCE_NATIVE_WINDOW_NO_RESAMPLING", expected_size="UNKNOWN_WINDOW",
               temporal_coverage="1981-2010 climatology" if source == "CHELSA" else "Year 2000 tree canopy cover",
               spatial_coverage=f"{lid} unit-centroid bounding box + {PAD_DEG} deg", resolution="30 arcsec" if source == "CHELSA" else "1 arcsec",
               verification_status="WINDOW_EXTRACTED_HASHED",
               notes="Window crop of the official global GeoTIFF; pixel values, grid, dtype, scale/offset unchanged; no resampling", **meta)
    if prior and prior["sha256"] != digest:
        raise RuntimeError(f"STOP: {acq} hash changed")
    if not prior:
        registry.append(rec)
        save_registry(registry)
    return prior or rec


def acquire(landscapes: dict[str, pd.DataFrame]) -> list[dict]:
    out = []
    for lid, t in landscapes.items():
        lon, lat = unit_lonlat(t, offsets=False)
        b = (lon.min() - PAD_DEG, lat.min() - PAD_DEG, lon.max() + PAD_DEG, lat.max() + PAD_DEG)
        for v in CHELSA_VARS:
            out.append(extract_window(lid, "CHELSA", v, CHELSA_URL.format(v=v), b))
        out.append(extract_window(lid, "HANSEN", "treecover2000", HANSEN_URL.format(tile=HANSEN_TILES[lid]), b))
    return out


def _sample(path, lon, lat) -> tuple[np.ndarray, float, float]:
    with rasterio.open(path) as r:
        arr = r.read(1)
        rows, cols = rasterio.transform.rowcol(r.transform, lon.ravel(), lat.ravel(), op=np.floor)
        rows, cols = np.asarray(rows).astype(np.int64), np.asarray(cols).astype(np.int64)
        if rows.min() < 0 or cols.min() < 0 or rows.max() >= arr.shape[0] or cols.max() >= arr.shape[1]:
            raise RuntimeError(f"STOP: sample outside window {path}")
        return arr[rows, cols].reshape(lon.shape).astype("float64"), float(r.scales[0]), float(r.offsets[0])


def features(lid: str, terrain: pd.DataFrame) -> pd.DataFrame:
    reg = {e["acquisition_id"]: e for e in load_registry()}
    out = pd.DataFrame({"terrain_unit_id": terrain.terrain_unit_id.to_numpy()})
    lon, lat = unit_lonlat(terrain, offsets=False)
    for v, (name, _, _) in CHELSA_VARS.items():
        raw, scale, offset = _sample(ROOT / reg[f"ACQ_E_CHELSA_{v.upper()}_{lid}"]["raw_path"], lon, lat)
        out[name] = raw * scale + offset  # CHELSA has no nodata value; land/sea both carry values
    lon9, lat9 = unit_lonlat(terrain, offsets=True)
    tc, _, _ = _sample(ROOT / reg[f"ACQ_E_HANSEN_TREECOVER2000_{lid}"]["raw_path"], lon9, lat9)
    tc = np.where(tc > 100, np.nan, tc)  # valid range 0-100 %; anything else is invalid, never zero-filled
    out["e_hansen_treecover2000_pct__mean"] = np.nanmean(tc, axis=1)
    out["e_hansen_treecover2000_pct__sd"] = np.nanstd(tc, axis=1)
    return out
