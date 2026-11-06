"""Stage 2D production terrain units and frozen M-feature engine.

This module deliberately contains no estimator, split, resampling, balancing, or
model metric code.  Predictors are functions of the canonical DEM only; targets
are joined afterwards in a separate table.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import platform
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scipy
from rasterio.features import rasterize
from scipy import ndimage, signal
from shapely import box, intersection
from shapely.ops import unary_union
from whitebox.whitebox_tools import WhiteboxTools

from .raw_store import sha256
from .schema import ROOT
from .stage2c_geology import aso_mapping

STAGE2C = ROOT / "data/processed/stage2c"
DER = ROOT / "data/processed/stage2d"
OUT = ROOT / "outputs/stage2d"
ACTIVE = ("KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024")
TARGET_FAMILY = {
    "KUMAMOTO_ASO_2016": "POINT_EVENT_TIER2",
    "NOTO_PENINSULA_2024": "POLYGON_AFFECTED_TIER2",
}
RES = 30.0
UNIT = 3
STRICT_MAX_NONORIGINAL_FRACTION = 0.0
CHANNEL_AREA_M2 = 100_000.0
AS_OF = "2026-10-06"
SCRIPT = Path(__file__)
SOFTWARE = {
    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    "rasterio": rasterio.__version__, "scipy": scipy.__version__,
    "whitebox_tools": "2.4.0", "algorithm_reference": "Tarboton (1997) D-infinity",
}


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def _csv_gz(path: Path, df: pd.DataFrame) -> None:
    """Deterministic gzip CSV (no timestamp or source filename in header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            df.to_csv(gz, index=False, lineterminator="\n", float_format="%.7g")


def _write_tif(path: Path, arr: np.ndarray, profile: dict, nodata=-9999.0, dtype="float32") -> None:
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


def _block(arr: np.ndarray, reducer: str, h3: int, w3: int) -> np.ndarray:
    x = arr[:h3, :w3].reshape(h3 // 3, 3, w3 // 3, 3)
    with np.errstate(invalid="ignore"):
        if reducer == "mean": return np.nanmean(x, axis=(1, 3))
        if reducer == "sd": return np.nanstd(x, axis=(1, 3))
        if reducer == "sum": return np.nansum(x, axis=(1, 3))
    raise ValueError(reducer)


def _disk(radius_cells: float, annulus: bool = False) -> np.ndarray:
    r = int(math.ceil(radius_cells))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    k = (xx * xx + yy * yy) <= radius_cells * radius_cells
    if annulus:
        k[r, r] = False
    return k


def _focal_mean(z: np.ndarray, valid: np.ndarray, radius_cells: float, annulus=False) -> np.ndarray:
    k = _disk(radius_cells, annulus).astype(np.float32)
    num = signal.fftconvolve(np.where(valid, z, 0).astype(np.float32), k, mode="same")
    den = signal.fftconvolve(valid.astype(np.float32), k, mode="same")
    return np.divide(num, den, out=np.full(z.shape, np.nan, np.float32), where=den > 0)


def _horn(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Horn derivatives; y is positive north because raster rows increase south.
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / (8 * RES)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32) / (8 * RES)
    p = ndimage.convolve(z, kx, mode="nearest")
    q = ndimage.convolve(z, ky, mode="nearest")
    slope = np.degrees(np.arctan(np.hypot(p, q))).astype(np.float32)
    return p, q, slope


def _curvature(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, q, _ = _horn(z)
    r = ndimage.convolve1d(z, np.array([1, -2, 1], np.float32) / RES ** 2, axis=1, mode="nearest")
    t = ndimage.convolve1d(z, np.array([1, -2, 1], np.float32) / RES ** 2, axis=0, mode="nearest")
    s = ndimage.convolve(z, np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], np.float32) / (4 * RES ** 2), mode="nearest")
    g2 = p * p + q * q
    profile = np.divide(-(r * p * p + 2 * s * p * q + t * q * q), g2 * np.power(1 + g2, 1.5),
                        out=np.zeros_like(z, np.float32), where=g2 > 1e-12)
    plan = np.divide(r * q * q - 2 * s * p * q + t * p * p, np.power(g2, 1.5),
                     out=np.zeros_like(z, np.float32), where=g2 > 1e-12)
    return profile.astype(np.float32), plan.astype(np.float32), (r + t).astype(np.float32)


def _roughness(z: np.ndarray, n: int) -> np.ndarray:
    a = ndimage.uniform_filter(z, size=n, mode="nearest")
    b = ndimage.uniform_filter(z * z, size=n, mode="nearest")
    return np.sqrt(np.maximum(b - a * a, 0)).astype(np.float32)


def _tri(z: np.ndarray) -> np.ndarray:
    total = np.zeros_like(z, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                total += np.abs(ndimage.shift(z, (dy, dx), order=0, mode="nearest", prefilter=False) - z)
    return total / 8.0


def _hydrology(lid: str, dem_path: Path, dem: np.ndarray, valid: np.ndarray, profile: dict) -> tuple[dict[str, np.ndarray], list[Path]]:
    hdir = DER / lid / "qc_hydrology"
    hdir.mkdir(parents=True, exist_ok=True)
    breached, accum = hdir / "dem_breached_30m.tif", hdir / "dinf_sca_30m.tif"
    inv, inv_breached, inv_accum = hdir / "dem_inverted_30m.tif", hdir / "dem_inverted_breached_30m.tif", hdir / "dinf_sca_inverted_30m.tif"
    inv_arr = np.where(valid, -dem, -9999.0)
    _write_tif(inv, inv_arr, profile)
    wbt = WhiteboxTools(); wbt.set_verbose_mode(False); wbt.set_max_procs(4)
    for source, conditioned, out in ((dem_path, breached, accum), (inv, inv_breached, inv_accum)):
        if conditioned.exists() and out.exists():
            continue
        # Lindsay (2016) hybrid breach-first algorithm.  No depth/length cap is
        # imposed, so all depressions are resolved; single-cell pits are raised.
        if wbt.breach_depressions(str(source), str(conditioned), fill_pits=True) != 0:
            raise RuntimeError(f"Whitebox breach failed for {source}")
        if wbt.d_inf_flow_accumulation(str(conditioned), str(out), out_type="sca", log=False, clip=False) != 0:
            raise RuntimeError(f"Whitebox D-infinity accumulation failed for {source}")
    def read(p):
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
            return np.where((a == src.nodata) | ~valid | ~np.isfinite(a), np.nan, a)
    sca, inv_sca = read(accum), read(inv_accum)
    contributing_area = sca * RES
    flow_cells = contributing_area / (RES * RES)
    channel = contributing_area > CHANNEL_AREA_M2
    ridge = inv_sca * RES > CHANNEL_AREA_M2
    dist_channel, channel_ix = ndimage.distance_transform_edt(~channel, sampling=RES, return_indices=True)
    dist_ridge, ridge_ix = ndimage.distance_transform_edt(~ridge, sampling=RES, return_indices=True)
    valley_surface = dem[channel_ix[0], channel_ix[1]]
    ridge_surface = dem[ridge_ix[0], ridge_ix[1]]
    valley_depth = np.maximum(dem - valley_surface, 0)
    below_ridge = np.maximum(ridge_surface - dem, 0)
    rsp = np.divide(valley_depth, valley_depth + below_ridge, out=np.full(dem.shape, .5, np.float32), where=(valley_depth + below_ridge) > 0)
    return {
        "flow_acc_cells_30m": flow_cells, "contrib_area_m2_per_m_30m": sca,
        "contrib_area_m2_30m": contributing_area,
        "dist_drainage_m_30m": dist_channel.astype(np.float32), "dist_ridge_m_30m": dist_ridge.astype(np.float32),
        "valley_depth_m_30m": valley_depth.astype(np.float32), "rel_slope_position_30m": rsp.astype(np.float32),
    }, [breached, accum, inv, inv_breached, inv_accum]


def feature_arrays(lid: str, dem_path: Path) -> tuple[dict[str, np.ndarray], dict, np.ndarray, list[Path]]:
    with rasterio.open(dem_path) as src:
        raw, profile = src.read(1), src.profile
        valid = (raw != src.nodata) & np.isfinite(raw)
    # Nearest filling is used only outside valid DEM to avoid contaminating focal windows with -9999.
    ix = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    z = raw[ix[0], ix[1]].astype(np.float32)
    p, q, slope = _horn(z)
    slope90 = _horn(ndimage.uniform_filter(z, 3, mode="nearest"))[2]
    aspect = np.arctan2(q, -p)
    flat = slope < 1.0
    curvp, curvl, curvt = _curvature(z)
    curvp90, curvl90, curvt90 = _curvature(ndimage.uniform_filter(z, 3, mode="nearest"))
    features: dict[str, np.ndarray] = {
        "elev_m_30m": z, "slope_deg_30m": slope, "slope_deg_90m": slope90,
        "aspect_sin_30m": np.where(flat, 0, np.sin(aspect)).astype(np.float32),
        "aspect_cos_30m": np.where(flat, 0, np.cos(aspect)).astype(np.float32),
        "flat_flag_30m": flat.astype(np.float32),
        "curv_profile_1pm_30m": curvp, "curv_plan_1pm_30m": curvl, "curv_total_1pm_30m": curvt,
        "curv_profile_1pm_90m": curvp90, "curv_plan_1pm_90m": curvl90, "curv_total_1pm_90m": curvt90,
        "roughness_sd_m_90m": _roughness(z, 3), "roughness_sd_m_270m": _roughness(z, 9),
        "tri_m_30m": _tri(z),
    }
    for radius in (150, 450, 1500):
        features[f"tpi_m_r{radius}"] = (z - _focal_mean(z, valid, radius / RES, annulus=True)).astype(np.float32)
    for radius in (250, 1000):
        footprint = _disk(radius / RES)
        hi = ndimage.maximum_filter(z, footprint=footprint, mode="nearest")
        lo = ndimage.minimum_filter(z, footprint=footprint, mode="nearest")
        features[f"relief_m_r{radius}"] = (hi - lo).astype(np.float32)
        if radius == 1000:
            features["rel_elev_local_r1000"] = np.divide(z - lo, hi - lo, out=np.zeros_like(z), where=(hi - lo) > 0).astype(np.float32)
    hydro, hydro_paths = _hydrology(lid, dem_path, z, valid, profile)
    features.update(hydro)
    beta = np.radians(np.maximum(slope, .1))
    a = np.maximum(features["contrib_area_m2_per_m_30m"], RES)
    features["twi_30m"] = np.log(a / np.tan(beta)).astype(np.float32)
    features["spi_m_30m"] = (a * np.tan(beta)).astype(np.float32)
    for k in features:
        features[k] = np.where(valid, features[k], np.nan).astype(np.float32)
    return features, profile, valid, hydro_paths


def _aso_sensitivity(primary: np.ndarray, profile: dict) -> np.ndarray:
    work = ROOT / "data/interim/_stage2c_aso"
    aso = gpd.read_file(work / "geo_A.shp")
    amap = aso_mapping(aso.drop(columns="geometry").drop_duplicates("SYMBOL").to_dict("records"))
    eligible = {r["source_unit_code"] for r in amap if r["domain_eligible"] == "TRUE"}
    a = aso.to_crs(profile["crs"])
    volcanic = rasterize([(g, 1) for g, code in zip(a.geometry, a["SYMBOL"]) if code in eligible],
                         out_shape=primary.shape, transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    return primary & volcanic


def _unit_geometries(rows: np.ndarray, cols: np.ndarray, transform) -> np.ndarray:
    x0 = transform.c + cols * RES; y1 = transform.f - rows * RES
    return np.array([box(x, y - 90, x + 90, y) for x, y in zip(x0, y1)], dtype=object)


def build_landscape(lid: str) -> dict:
    srcdir, d = STAGE2C / lid, DER / lid
    for sub in ("terrain_units", "features_m", "targets", "metadata", "qc"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    dem_path = srcdir / "dem_cop30_30m.tif"
    omega_path = next(srcdir.glob("omega_analysis_primary*_30m.tif"))
    with rasterio.open(omega_path) as src:
        omega, profile = src.read(1).astype(bool), src.profile
    with rasterio.open(srcdir / "cop30_edm_30m.tif") as src: edm = src.read(1)
    with rasterio.open(srcdir / "cop30_flm_30m.tif") as src: flm = src.read(1)
    with rasterio.open(srcdir / "geology_seamless_index_30m.tif") as src: geo = src.read(1)
    h3, w3 = omega.shape[0] // 3 * 3, omega.shape[1] // 3 * 3
    counts = _block(omega.astype(np.float32), "sum", h3, w3).astype(np.uint8)
    valid_ix = np.argwhere(counts >= 5)
    ur, uc = valid_ix[:, 0], valid_ix[:, 1]
    rows, cols = ur * 3, uc * 3
    unit_ids = np.array([f"{lid}_{r:05d}_{c:05d}" for r, c in zip(rows, cols)])
    cx = profile["transform"].c + (cols + 1.5) * RES
    cy = profile["transform"].f - (rows + 1.5) * RES
    edited = _block((edm != 1).astype(np.float32), "sum", h3, w3)[ur, uc] / 9
    filled = _block(((flm != 2) & (flm != 255)).astype(np.float32), "sum", h3, w3)[ur, uc] / 9
    def mode9(r, c):
        vals, n = np.unique(geo[r:r+3, c:c+3][omega[r:r+3, c:c+3] & (geo[r:r+3, c:c+3] > 0)], return_counts=True)
        return int(vals[np.argmax(n)]) if len(vals) else 0
    geo_code = np.array([mode9(r, c) for r, c in zip(rows, cols)], dtype=np.uint16)
    mapping = list(csv.DictReader((ROOT / f"outputs/stage2c/geology_mapping_{lid}_seamless.csv").open()))
    gm = {i + 1: row for i, row in enumerate(mapping)}
    terrain = pd.DataFrame({
        "terrain_unit_id": unit_ids, "landscape_id": lid, "target_family": TARGET_FAMILY[lid],
        "centroid_x": cx, "centroid_y": cy, "projected_crs": profile["crs"].to_string(), "cell_area_m2": 8100,
        "source_grid_row": rows, "source_grid_col": cols, "valid_fraction": counts[ur, uc] / 9,
        "volcanic_domain_flag": True, "observation_domain_flag": True, "dem_valid_flag": True,
        "dem_edited_fraction": edited, "dem_filled_fraction": filled,
        "dem_quality_flag": np.where(edited == 0, "ORIGINAL", "EDITED_OR_FILLED"),
        "dem_source_quality_class": np.where(edited == 0, "COP30_NOT_EDITED", "COP30_NON_ORIGINAL_PRESENT"),
        "analysis_population_all_valid": True,
        "analysis_population_strict_dem_quality": edited <= STRICT_MAX_NONORIGINAL_FRACTION,
        "spatial_block_3000m": [f"B3_{math.floor(x/3000)}_{math.floor(y/3000)}" for x, y in zip(cx, cy)],
        "spatial_block_6000m": [f"B6_{math.floor(x/6000)}_{math.floor(y/6000)}" for x, y in zip(cx, cy)],
        "geology_index": geo_code,
        "source_unit_code": [gm.get(int(i), {}).get("source_unit_code", "UNMAPPED") for i in geo_code],
        "source_lithology": [gm.get(int(i), {}).get("source_lithology", "UNKNOWN") for i in geo_code],
        "source_age": [gm.get(int(i), {}).get("source_age", "UNKNOWN") for i in geo_code],
        "harmonised_geology_class": [gm.get(int(i), {}).get("harmonised_class", "UNKNOWN") for i in geo_code],
        "geology_mapping_confidence": [gm.get(int(i), {}).get("mapping_confidence", "LOW") for i in geo_code],
    })
    if lid == "KUMAMOTO_ASO_2016":
        sens = _aso_sensitivity(omega, profile)
        terrain["primary_domain"] = True
        terrain["high_res_geology_sensitivity_domain"] = _block(sens.astype(np.float32), "sum", h3, w3)[ur, uc] >= 5
        p = d / "metadata/high_res_geology_sensitivity_domain_30m.tif"
        _write_tif(p, sens.astype(np.uint8), profile, nodata=255, dtype="uint8")
    else:
        terrain["primary_domain"] = True
        terrain["high_res_geology_sensitivity_domain"] = False
    terrain_path = d / "terrain_units/terrain_units.csv.gz"; _csv_gz(terrain_path, terrain)

    # Predictor computation happens before and independently of target construction.
    features, _, dem_valid, hydro_paths = feature_arrays(lid, dem_path)
    feature_table = pd.DataFrame({"terrain_unit_id": unit_ids})
    for name, arr in features.items():
        feature_table[name + "__mean"] = _block(arr, "mean", h3, w3)[ur, uc]
        feature_table[name + "__sd"] = _block(arr, "sd", h3, w3)[ur, uc]
    features_path = d / "features_m/features_m.csv.gz"; _csv_gz(features_path, feature_table)

    if lid == "KUMAMOTO_ASO_2016":
        y = gpd.read_file(srcdir / "y_harmonised_points.geojson")
        primary = y[(y.target_set == "PRIMARY_APRIL_EQ") & y.in_primary_analysis.astype(bool)]
        pr, pc = rasterio.transform.rowcol(profile["transform"], primary.geometry.x, primary.geometry.y)
        block_key = list(zip(np.asarray(pr) // 3 * 3, np.asarray(pc) // 3 * 3))
        event = pd.Series(block_key).value_counts()
        valid_keys = set(zip(rows, cols))
        excluded_primary = primary[["geometry"]].copy()
        excluded_primary["source_grid_row"] = np.asarray(pr)
        excluded_primary["source_grid_col"] = np.asarray(pc)
        excluded_primary["terrain_unit_row"] = np.asarray(pr) // 3 * 3
        excluded_primary["terrain_unit_col"] = np.asarray(pc) // 3 * 3
        excluded_primary = excluded_primary[
            [k not in valid_keys for k in zip(excluded_primary.terrain_unit_row, excluded_primary.terrain_unit_col)]
        ].copy()
        excluded_primary["exclusion_reason"] = "PARENT_90M_UNIT_HAS_LESS_THAN_5_OF_9_CELLS_IN_OMEGA_ANALYSIS"
        excluded_primary.drop(columns="geometry").to_csv(d / "targets/primary_target_exclusions.csv", index=True,
                                                           index_label="source_feature_index", lineterminator="\n")
        duplicates = y[y.target_set == "EXCLUDED_DUPLICATE"]
        dr, dc = rasterio.transform.rowcol(profile["transform"], duplicates.geometry.x, duplicates.geometry.y)
        dup = pd.Series(list(zip(np.asarray(dr)//3*3, np.asarray(dc)//3*3))).value_counts()
        counts_y = np.array([event.get((r, c), 0) for r, c in zip(rows, cols)])
        label_qc = np.zeros(omega.shape, dtype=bool)
        label_qc[np.asarray(pr), np.asarray(pc)] = True
        targets = pd.DataFrame({
            "terrain_unit_id": unit_ids, "y_positive": counts_y > 0, "event_count_in_unit": counts_y,
            "duplicate_point_count": [dup.get((r, c), 0) for r, c in zip(rows, cols)],
            "target_source": "GSI_2016_MAPPED_LANDSLIDE_AREA_CENTRES", "target_tier": "TIER_2",
            "trigger_class": "EARTHQUAKE", "photo_date_class": "APRIL_EQ",
            "label_rule": "CONTAINS_AT_LEAST_ONE_UNBUFFERED_VALID_CENTRE",
            "background_valid_reason": np.where(counts_y > 0, "POSITIVE", "INSIDE_OMEGA_DEM_VOLCANIC_VISIBILITY_QC_AND_NO_PRIMARY_POINT"),
        })
        sensitivity = y[y.target_set.isin(["SENSITIVITY_JULY", "SENSITIVITY_UNDATED"])][["target_set", "epoch", "photo_date", "geometry"]]
        sensitivity.to_file(d / "targets/sensitivity_points.geojson", driver="GeoJSON")
        overlap_area = np.zeros(len(terrain)); n_inter = np.zeros(len(terrain), dtype=int)
        source_target_count, included_target_count, excluded_target_count = len(primary), int(counts_y.sum()), len(excluded_primary)
    else:
        y = gpd.read_file(srcdir / "y_harmonised_polygons.geojson")
        label_grid = rasterize([(g, 1) for g in y.geometry], out_shape=omega.shape, transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
        label_qc = label_grid
        counts_y = _block(label_grid.astype(np.float32), "sum", h3, w3)[ur, uc].astype(int)
        geoms = _unit_geometries(rows, cols, profile["transform"])
        yu = unary_union(y.geometry)
        overlap_area = np.asarray([g.area for g in intersection(geoms, yu)])
        pairs = y.sindex.query(geoms, predicate="intersects")
        n_inter = np.bincount(pairs[0], minlength=len(geoms)) if pairs.size else np.zeros(len(geoms), int)
        targets = pd.DataFrame({
            "terrain_unit_id": unit_ids, "y_positive": counts_y > 0, "event_count_in_unit": "",
            "duplicate_point_count": 0, "target_source": "GSI_2024_AFFECTED_AREA_POLYGONS", "target_tier": "TIER_2",
            "trigger_class": "EARTHQUAKE", "photo_date_class": "POST_EVENT_INTERPRETATION",
            "label_rule": "POSITIVE_IF_AT_LEAST_ONE_OF_NINE_30M_CELL_CENTRES_IS_INSIDE_VALID_POLYGON",
            "landslide_overlap_area_m2": overlap_area, "landslide_overlap_fraction": overlap_area / 8100,
            "number_of_intersecting_polygons": n_inter,
            "background_valid_reason": np.where(counts_y > 0, "POSITIVE", "INSIDE_OMEGA_DEM_VOLCANIC_VISIBILITY_QC_AND_NO_POLYGON_CELL_CENTRE"),
        })
        source_target_count = included_target_count = excluded_target_count = None
    target_path = d / "targets/targets_primary.csv.gz"; _csv_gz(target_path, targets)

    pos = targets.y_positive.astype(bool).to_numpy()
    domain_rows = []
    if lid == "KUMAMOTO_ASO_2016":
        for domain, keep, cell_mask in (("PRIMARY_DOMAIN", np.ones(len(terrain), bool), omega),
                                        ("HIGH_RES_GEOLOGY_SENSITIVITY_DOMAIN", terrain.high_res_geology_sensitivity_domain.to_numpy(bool), sens)):
            domain_rows.append({"landscape_id": lid, "analysis_domain_version": domain,
                                "domain_cell_area_km2": round(cell_mask.sum()*0.0009, 4),
                                "included_unit_footprint_km2": round(keep.sum()*0.0081, 4),
                                "boundary_rule": "unit included when >=5/9 cell centres are in domain",
                                "terrain_units": int(keep.sum()), "positive_units": int((pos & keep).sum()),
                                "negative_units": int((~pos & keep).sum())})
        _csv(d / "metadata/geology_sensitivity_summary.csv", domain_rows)
    missing = {c: round(float(feature_table[c].isna().mean()), 7) for c in feature_table if c != "terrain_unit_id"}
    summary = {
        "landscape_id": lid, "target_family": TARGET_FAMILY[lid], "total_candidate_aligned_blocks": int((counts > 0).sum()),
        "valid_units": len(terrain), "positive_units": int(pos.sum()), "negative_units": int((~pos).sum()),
        "positive_prevalence": round(float(pos.mean()), 8),
        "primary_source_features": source_target_count,
        "primary_events_in_included_units": included_target_count,
        "primary_events_excluded_by_90m_domain_rule": excluded_target_count,
        "target_geometry_diagnostics": ({
            "units_with_nonzero_polygon_overlap": int((overlap_area > 0).sum()),
            "units_with_intersecting_polygons": int((n_inter > 0).sum()),
            "maximum_overlap_fraction": round(float((overlap_area / 8100).max()), 7),
            "maximum_intersecting_polygon_count": int(n_inter.max()),
            "primary_rule_unchanged_by_diagnostics": True,
        } if lid == "NOTO_PENINSULA_2024" else {
            "duplicate_source_points": 26, "duplicates_not_used_as_extra_events": True,
            "point_buffer_m": 0,
        }),
        "excluded_units": int(((counts > 0) & (counts < 5)).sum()),
        "exclusion_reasons": {"INTERSECTS_OMEGA_BUT_LESS_THAN_5_OF_9_CELLS_IN_OMEGA_ANALYSIS": int(((counts > 0) & (counts < 5)).sum())},
        "dem_quality_affected_units": int((terrain.dem_edited_fraction > 0).sum()),
        "strict_dem_quality_units": int(terrain.analysis_population_strict_dem_quality.sum()),
        "strict_rule": "dem_edited_fraction == 0.0 (no non-original Copernicus cell in the 3x3 unit)",
        "features_generated": len(features), "feature_columns_after_mean_sd": len(features) * 2,
        "feature_missing_rates": missing, "geology_class_counts": terrain.harmonised_geology_class.value_counts().sort_index().to_dict(),
        "spatial_blocks_3000m": int(terrain.spatial_block_3000m.nunique()), "spatial_blocks_6000m": int(terrain.spatial_block_6000m.nunique()),
        "production_readiness": "READY_FOR_STAGE_2E_DATA_ASSEMBLY",
    }
    _json(d / "metadata/production_summary.json", summary)
    _qc(lid, omega, label_qc & omega, edm != 1, geo, features["slope_deg_30m"], d / "qc")
    artifacts = [terrain_path, features_path, target_path, d / "metadata/production_summary.json", *hydro_paths]
    return {"summary": summary, "artifacts": artifacts, "feature_names": sorted(features), "domain_rows": domain_rows}


def _qc(lid, omega, label, edited, geology, slope, qdir):
    qdir.mkdir(parents=True, exist_ok=True)
    layers = [("OMEGA_ANALYSIS", omega), ("PRIMARY_POSITIVE_CELL", label), ("DEM_NON_ORIGINAL", edited & omega),
              ("GEOLOGY_INDEX", np.where(omega, geology, np.nan)), ("SLOPE_DEG", np.where(omega, slope, np.nan))]
    for name, arr in layers:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        ax.imshow(arr[::3, ::3], interpolation="nearest", cmap="viridis")
        ax.set(title=f"{lid} — {name} (QC)", xlabel="canonical-grid column / 3", ylabel="canonical-grid row / 3")
        fig.savefig(qdir / f"{name.lower()}.png", dpi=140); plt.close(fig)


def feature_definition_records(feature_names: list[str]) -> list[dict]:
    rows = []
    for n in sorted(set(feature_names)):
        if n.startswith("elev_"): group, alg, unit, scale = "ELEVATION", "canonical DEM z", "m", "30m"
        elif n.startswith("slope_"): group, alg, unit, scale = "SLOPE", "Horn (1981) 3x3 gradient; atan magnitude", "degree", n.rsplit("_", 1)[-1]
        elif n.startswith("aspect_") or n.startswith("flat_"): group, alg, unit, scale = "ASPECT", "sin/cos Horn aspect; flat slope<1 degree -> 0,0 and flag", "dimensionless", "30m"
        elif n.startswith("curv_"): group, alg, unit, scale = "CURVATURE", "Zevenbergen-Thorne second derivatives", "1/m", n.rsplit("_", 1)[-1]
        elif n.startswith("roughness_"): group, alg, unit, scale = "ROUGHNESS", "population SD of elevation in square window", "m", n.rsplit("_", 1)[-1]
        elif n.startswith("tri_"): group, alg, unit, scale = "TRI", "Riley mean absolute centre-neighbour elevation difference", "m", "30m"
        elif n.startswith("tpi_"): group, alg, unit, scale = "TPI", "z minus circular annulus mean", "m", n.split("r")[-1] + "m radius"
        elif n.startswith("relief_"): group, alg, unit, scale = "RELIEF", "circular-window maximum minus minimum", "m", n.split("r")[-1] + "m radius"
        elif n.startswith("rel_elev_"): group, alg, unit, scale = "RELATIVE_ELEVATION", "(z-min)/(max-min) in circular window", "0-1", "1000m radius"
        elif "flow_acc" in n: group, alg, unit, scale = "HYDROLOGY", "WhiteboxTools D-infinity after Lindsay (2016) hybrid depression breach", "cells", "30m"
        elif "contrib_area_m2_per_m" in n: group, alg, unit, scale = "HYDROLOGY", "WhiteboxTools D-infinity specific contributing area", "m2/m", "30m"
        elif "contrib_area_m2" in n: group, alg, unit, scale = "HYDROLOGY", "D-infinity SCA times 30m contour width", "m2", "30m"
        elif n.startswith("twi_"): group, alg, unit, scale = "HYDROLOGY", "ln(a/tan(max(slope,0.1deg)))", "dimensionless", "30m"
        elif n.startswith("spi_"): group, alg, unit, scale = "HYDROLOGY", "a*tan(max(slope,0.1deg))", "m", "30m"
        elif n.startswith("dist_drainage"): group, alg, unit, scale = "HYDROLOGY", "Euclidean distance to D-infinity contributing-area >0.1km2 cells", "m", "30m"
        elif n.startswith("dist_ridge"): group, alg, unit, scale = "HYDROLOGY", "Euclidean distance to inverted-DEM D-infinity divide network", "m", "30m"
        elif n.startswith("valley_depth"): group, alg, unit, scale = "HYDROLOGY", "height above nearest channel-network elevation surface, clipped >=0", "m", "30m"
        else: group, alg, unit, scale = "HYDROLOGY", "valley_depth/(valley_depth+height_below_nearest_divide)", "0-1", "30m"
        rows.append({"feature_name": n, "feature_group": group, "formula_algorithm": alg,
                     "source_raster": "stage2c/dem_cop30_30m.tif", "units": unit, "spatial_scale": scale,
                     "nodata_rule": "NaN if canonical DEM invalid; production unit requires valid DEM and >=5/9 Omega cells",
                     "edge_rule": "compute on Stage2C 1.5km buffered grid; emit rows only inside OMEGA_ANALYSIS",
                     "software_library": json.dumps(SOFTWARE, sort_keys=True),
                     "parameters": json.dumps({"resolution_m": 30, "hydrology_breach_max_depth": "unlimited",
                                                "hydrology_breach_max_length": "unlimited", "fill_single_cell_pits": True,
                                                "channel_area_m2": 100000}, sort_keys=True),
                     "output_dtype": "float32 cell grid; deterministic mean and population-SD at 90m unit"})
    return rows
