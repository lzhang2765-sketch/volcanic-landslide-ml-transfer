"""Parameterised mirror of the FROZEN Stage-2D polygon-target production branch (stage2d_production.build_landscape).

stage2d_production.py is hash-bound by the Stage-2D checkpoint, so it is not edited.  This module re-states only the
polygon-target path with input locations and target-source labels as parameters, and calls the frozen feature engine
(stage2d_production.feature_arrays, _block, _csv_gz, _json, _qc, _unit_geometries) unchanged.  Equivalence is proven by
regenerating NOTO_PENINSULA_2024 and requiring byte-identical terrain/feature/target tables (tests/test_stage2f2.py).
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import intersection
from shapely.ops import unary_union

from . import stage2d_production as P


@contextlib.contextmanager
def _hydrology_root(path: Path):
    """feature_arrays() writes Whitebox intermediates under P.DER/<lid>/qc_hydrology; redirect only that root."""
    old = P.DER
    P.DER = path
    try:
        yield
    finally:
        P.DER = old


def build_polygon_landscape(lid: str, srcdir: Path, outdir: Path, mapping: list[dict], y: gpd.GeoDataFrame, target_source: str,
                            trigger_class: str = "EARTHQUAKE", photo_date_class: str = "POST_EVENT_INTERPRETATION",
                            omega_name: str | None = None, geology_name: str = "geology_seamless_index_30m.tif",
                            hydrology_root: Path | None = None, target_family: str = "POLYGON_AFFECTED_TIER2") -> dict:
    d = outdir / lid
    for sub in ("terrain_units", "features_m", "targets", "metadata", "qc"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    dem_path = srcdir / "dem_cop30_30m.tif"
    omega_path = srcdir / omega_name if omega_name else next(srcdir.glob("omega_analysis_primary*_30m.tif"))
    with rasterio.open(omega_path) as src:
        omega, profile = src.read(1).astype(bool), src.profile
    with rasterio.open(srcdir / "cop30_edm_30m.tif") as src: edm = src.read(1)
    with rasterio.open(srcdir / "cop30_flm_30m.tif") as src: flm = src.read(1)
    with rasterio.open(srcdir / geology_name) as src: geo = src.read(1)
    h3, w3 = omega.shape[0] // 3 * 3, omega.shape[1] // 3 * 3
    counts = P._block(omega.astype(np.float32), "sum", h3, w3).astype(np.uint8)
    valid_ix = np.argwhere(counts >= 5)
    ur, uc = valid_ix[:, 0], valid_ix[:, 1]
    rows, cols = ur * 3, uc * 3
    unit_ids = np.array([f"{lid}_{r:05d}_{c:05d}" for r, c in zip(rows, cols)])
    cx = profile["transform"].c + (cols + 1.5) * P.RES
    cy = profile["transform"].f - (rows + 1.5) * P.RES
    edited = P._block((edm != 1).astype(np.float32), "sum", h3, w3)[ur, uc] / 9
    filled = P._block(((flm != 2) & (flm != 255)).astype(np.float32), "sum", h3, w3)[ur, uc] / 9

    def mode9(r, c):
        vals, n = np.unique(geo[r:r+3, c:c+3][omega[r:r+3, c:c+3] & (geo[r:r+3, c:c+3] > 0)], return_counts=True)
        return int(vals[np.argmax(n)]) if len(vals) else 0
    geo_code = np.array([mode9(r, c) for r, c in zip(rows, cols)], dtype=np.uint16)
    gm = {i + 1: row for i, row in enumerate(mapping)}
    terrain = pd.DataFrame({
        "terrain_unit_id": unit_ids, "landscape_id": lid, "target_family": target_family,
        "centroid_x": cx, "centroid_y": cy, "projected_crs": profile["crs"].to_string(), "cell_area_m2": 8100,
        "source_grid_row": rows, "source_grid_col": cols, "valid_fraction": counts[ur, uc] / 9,
        "volcanic_domain_flag": True, "observation_domain_flag": True, "dem_valid_flag": True,
        "dem_edited_fraction": edited, "dem_filled_fraction": filled,
        "dem_quality_flag": np.where(edited == 0, "ORIGINAL", "EDITED_OR_FILLED"),
        "dem_source_quality_class": np.where(edited == 0, "COP30_NOT_EDITED", "COP30_NON_ORIGINAL_PRESENT"),
        "analysis_population_all_valid": True,
        "analysis_population_strict_dem_quality": edited <= P.STRICT_MAX_NONORIGINAL_FRACTION,
        "spatial_block_3000m": [f"B3_{math.floor(x/3000)}_{math.floor(y/3000)}" for x, y in zip(cx, cy)],
        "spatial_block_6000m": [f"B6_{math.floor(x/6000)}_{math.floor(y/6000)}" for x, y in zip(cx, cy)],
        "geology_index": geo_code,
        "source_unit_code": [gm.get(int(i), {}).get("source_unit_code", "UNMAPPED") for i in geo_code],
        "source_lithology": [gm.get(int(i), {}).get("source_lithology", "UNKNOWN") for i in geo_code],
        "source_age": [gm.get(int(i), {}).get("source_age", "UNKNOWN") for i in geo_code],
        "harmonised_geology_class": [gm.get(int(i), {}).get("harmonised_class", "UNKNOWN") for i in geo_code],
        "geology_mapping_confidence": [gm.get(int(i), {}).get("mapping_confidence", "LOW") for i in geo_code],
    })
    terrain["primary_domain"] = True
    terrain["high_res_geology_sensitivity_domain"] = False
    terrain_path = d / "terrain_units/terrain_units.csv.gz"; P._csv_gz(terrain_path, terrain)

    # Predictors first and independently of targets (frozen engine, unchanged).
    with _hydrology_root(hydrology_root or outdir):
        features, _, dem_valid, hydro_paths = P.feature_arrays(lid, dem_path)
    feature_table = pd.DataFrame({"terrain_unit_id": unit_ids})
    for name, arr in features.items():
        feature_table[name + "__mean"] = P._block(arr, "mean", h3, w3)[ur, uc]
        feature_table[name + "__sd"] = P._block(arr, "sd", h3, w3)[ur, uc]
    features_path = d / "features_m/features_m.csv.gz"; P._csv_gz(features_path, feature_table)

    label_grid = rasterize([(g, 1) for g in y.geometry], out_shape=omega.shape, transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    counts_y = P._block(label_grid.astype(np.float32), "sum", h3, w3)[ur, uc].astype(int)
    geoms = P._unit_geometries(rows, cols, profile["transform"])
    yu = unary_union(y.geometry)
    overlap_area = np.asarray([g.area for g in intersection(geoms, yu)])
    pairs = y.sindex.query(geoms, predicate="intersects")
    n_inter = np.bincount(pairs[0], minlength=len(geoms)) if pairs.size else np.zeros(len(geoms), int)
    targets = pd.DataFrame({
        "terrain_unit_id": unit_ids, "y_positive": counts_y > 0, "event_count_in_unit": "",
        "duplicate_point_count": 0, "target_source": target_source, "target_tier": "TIER_2",
        "trigger_class": trigger_class, "photo_date_class": photo_date_class,
        "label_rule": "POSITIVE_IF_AT_LEAST_ONE_OF_NINE_30M_CELL_CENTRES_IS_INSIDE_VALID_POLYGON",
        "landslide_overlap_area_m2": overlap_area, "landslide_overlap_fraction": overlap_area / 8100,
        "number_of_intersecting_polygons": n_inter,
        "background_valid_reason": np.where(counts_y > 0, "POSITIVE", "INSIDE_OMEGA_DEM_VOLCANIC_VISIBILITY_QC_AND_NO_POLYGON_CELL_CENTRE"),
    })
    target_path = d / "targets/targets_primary.csv.gz"; P._csv_gz(target_path, targets)
    pos = targets.y_positive.astype(bool).to_numpy()
    missing = {c: round(float(feature_table[c].isna().mean()), 7) for c in feature_table if c != "terrain_unit_id"}
    summary = {
        "landscape_id": lid, "target_family": target_family, "total_candidate_aligned_blocks": int((counts > 0).sum()),
        "valid_units": len(terrain), "positive_units": int(pos.sum()), "negative_units": int((~pos).sum()),
        "positive_prevalence": round(float(pos.mean()), 8),
        "primary_source_features": None, "primary_events_in_included_units": None, "primary_events_excluded_by_90m_domain_rule": None,
        "target_geometry_diagnostics": {
            "units_with_nonzero_polygon_overlap": int((overlap_area > 0).sum()),
            "units_with_intersecting_polygons": int((n_inter > 0).sum()),
            "maximum_overlap_fraction": round(float((overlap_area / 8100).max()), 7),
            "maximum_intersecting_polygon_count": int(n_inter.max()),
            "primary_rule_unchanged_by_diagnostics": True,
        },
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
    P._json(d / "metadata/production_summary.json", summary)
    P._qc(lid, omega, label_grid & omega, edm != 1, geo, features["slope_deg_30m"], d / "qc")
    return {"summary": summary, "artifacts": [terrain_path, features_path, target_path, d / "metadata/production_summary.json", *hydro_paths],
            "feature_names": sorted(features), "terrain": terrain, "targets": targets, "features": feature_table, "label_grid": label_grid,
            "omega": omega, "edm": edm}
