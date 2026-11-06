"""Parameterised mirror of the FROZEN Stage-2D point-target production branch (stage2d_production.build_landscape, Kumamoto path).

stage2d_production.py is hash-bound by the Stage-2D checkpoint and is not edited.  This module re-states only the point-target
path with inputs, target-source labels and the optional sensitivity domain as parameters, and calls the frozen engine
(_block, feature_arrays, _csv_gz, _json, _qc, _write_tif) unchanged.  Equivalence is proven by regenerating KUMAMOTO_ASO_2016 and
requiring byte-identical terrain/feature/target/exclusion tables (tests/test_stage3b.py).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from . import stage2d_production as P
from .stage2f2_production import _hydrology_root


def build_point_landscape(lid: str, srcdir: Path, outdir: Path, mapping: list[dict], primary: gpd.GeoDataFrame, target_source: str,
                          target_tier: str, target_family: str, trigger_class: str, photo_date_class: str,
                          duplicates: gpd.GeoDataFrame | None = None, sensitivity_points: gpd.GeoDataFrame | None = None,
                          sensitivity_domain: Callable | None = None, omega_name: str | None = None,
                          geology_name: str = "geology_seamless_index_30m.tif", hydrology_root: Path | None = None) -> dict:
    """`primary` must already be the frozen target set (points inside the analysis grid); duplicates are never extra events."""
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
    if sensitivity_domain is not None:
        sens = sensitivity_domain(omega, profile)
        terrain["high_res_geology_sensitivity_domain"] = P._block(sens.astype(np.float32), "sum", h3, w3)[ur, uc] >= 5
        P._write_tif(d / "metadata/high_res_geology_sensitivity_domain_30m.tif", sens.astype(np.uint8), profile, nodata=255, dtype="uint8")
    else:
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

    pr, pc = rasterio.transform.rowcol(profile["transform"], primary.geometry.x, primary.geometry.y)
    block_key = list(zip(np.asarray(pr) // 3 * 3, np.asarray(pc) // 3 * 3))
    event = pd.Series(block_key).value_counts()
    valid_keys = set(zip(rows, cols))
    excluded = primary[["geometry"]].copy()
    excluded["source_grid_row"] = np.asarray(pr)
    excluded["source_grid_col"] = np.asarray(pc)
    excluded["terrain_unit_row"] = np.asarray(pr) // 3 * 3
    excluded["terrain_unit_col"] = np.asarray(pc) // 3 * 3
    excluded = excluded[[k not in valid_keys for k in zip(excluded.terrain_unit_row, excluded.terrain_unit_col)]].copy()
    excluded["exclusion_reason"] = "PARENT_90M_UNIT_HAS_LESS_THAN_5_OF_9_CELLS_IN_OMEGA_ANALYSIS"
    excluded.drop(columns="geometry").to_csv(d / "targets/primary_target_exclusions.csv", index=True,
                                             index_label="source_feature_index", lineterminator="\n")
    dups = duplicates if duplicates is not None else primary.iloc[0:0]
    dr, dc = rasterio.transform.rowcol(profile["transform"], dups.geometry.x, dups.geometry.y) if len(dups) else ([], [])
    dup = pd.Series(list(zip(np.asarray(dr) // 3 * 3, np.asarray(dc) // 3 * 3)), dtype=object).value_counts()
    counts_y = np.array([event.get((r, c), 0) for r, c in zip(rows, cols)])
    label_qc = np.zeros(omega.shape, dtype=bool)
    label_qc[np.asarray(pr), np.asarray(pc)] = True
    targets = pd.DataFrame({
        "terrain_unit_id": unit_ids, "y_positive": counts_y > 0, "event_count_in_unit": counts_y,
        "duplicate_point_count": [dup.get((r, c), 0) for r, c in zip(rows, cols)],
        "target_source": target_source, "target_tier": target_tier,
        "trigger_class": trigger_class, "photo_date_class": photo_date_class,
        "label_rule": "CONTAINS_AT_LEAST_ONE_UNBUFFERED_VALID_CENTRE",
        "background_valid_reason": np.where(counts_y > 0, "POSITIVE", "INSIDE_OMEGA_DEM_VOLCANIC_VISIBILITY_QC_AND_NO_PRIMARY_POINT"),
    })
    if sensitivity_points is not None:
        sensitivity_points.to_file(d / "targets/sensitivity_points.geojson", driver="GeoJSON")
    target_path = d / "targets/targets_primary.csv.gz"; P._csv_gz(target_path, targets)
    pos = targets.y_positive.astype(bool).to_numpy()
    missing = {c: round(float(feature_table[c].isna().mean()), 7) for c in feature_table if c != "terrain_unit_id"}
    summary = {
        "landscape_id": lid, "target_family": target_family, "total_candidate_aligned_blocks": int((counts > 0).sum()),
        "valid_units": len(terrain), "positive_units": int(pos.sum()), "negative_units": int((~pos).sum()),
        "positive_prevalence": round(float(pos.mean()), 8), "primary_source_features": len(primary),
        "primary_events_in_included_units": int(counts_y.sum()), "primary_events_excluded_by_90m_domain_rule": len(excluded),
        "target_geometry_diagnostics": {"duplicate_source_points": len(dups), "duplicates_not_used_as_extra_events": True, "point_buffer_m": 0},
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
    P._qc(lid, omega, label_qc & omega, edm != 1, geo, features["slope_deg_30m"], d / "qc")
    return {"summary": summary, "artifacts": [terrain_path, features_path, target_path, d / "targets/primary_target_exclusions.csv",
                                              d / "metadata/production_summary.json", *hydro_paths], "feature_names": sorted(features)}


def kumamoto_inputs() -> dict:
    """Exactly the frozen Kumamoto inputs (Stage 2C derived products), for the equivalence proof."""
    lid = "KUMAMOTO_ASO_2016"
    src = P.STAGE2C / lid
    y = gpd.read_file(src / "y_harmonised_points.geojson")
    return dict(lid=lid, srcdir=src, mapping=list(csv.DictReader((P.ROOT / f"outputs/stage2c/geology_mapping_{lid}_seamless.csv").open())),
                primary=y[(y.target_set == "PRIMARY_APRIL_EQ") & y.in_primary_analysis.astype(bool)],
                duplicates=y[y.target_set == "EXCLUDED_DUPLICATE"],
                sensitivity_points=y[y.target_set.isin(["SENSITIVITY_JULY", "SENSITIVITY_UNDATED"])][["target_set", "epoch", "photo_date", "geometry"]],
                target_source="GSI_2016_MAPPED_LANDSLIDE_AREA_CENTRES", target_tier="TIER_2", target_family=P.TARGET_FAMILY[lid],
                trigger_class="EARTHQUAKE", photo_date_class="APRIL_EQ", sensitivity_domain=P._aso_sensitivity)


def kumamoto_equivalence(outdir: Path) -> dict:
    k = kumamoto_inputs()
    lid = k.pop("lid")
    r = build_point_landscape(lid, k.pop("srcdir"), outdir, k.pop("mapping"), k.pop("primary"), hydrology_root=outdir, **k)
    import gzip
    rels = ("terrain_units/terrain_units.csv.gz", "features_m/features_m.csv.gz", "targets/targets_primary.csv.gz", "targets/primary_target_exclusions.csv")
    read = lambda p: gzip.open(p).read() if p.suffix == ".gz" else p.read_bytes()
    result = {rel: read(outdir / lid / rel) == read(P.DER / lid / rel) for rel in rels}
    frozen = __import__("json").loads((P.DER / lid / "metadata/production_summary.json").read_text())
    result["summary_counts_equal"] = all(r["summary"][k] == frozen[k] for k in ("valid_units", "positive_units", "negative_units", "excluded_units",
                                                                                 "strict_dem_quality_units", "geology_class_counts"))
    return dict(result, units=r["summary"]["valid_units"], positives=r["summary"]["positive_units"])


EQUIVALENCE_JSON = P.ROOT / "outputs/stage3b/POINT_MIRROR_KUMAMOTO_EQUIVALENCE.json"


def record_equivalence(scratch: Path) -> dict:
    """Run the proof into a scratch directory (never the frozen Stage 2D directory) and record the result."""
    import json
    r = kumamoto_equivalence(scratch)
    r["frozen_reference"] = "data/processed/stage2d/KUMAMOTO_ASO_2016 (Stage 2D checkpoint)"
    r["all_byte_identical"] = all(v for k, v in r.items() if k.endswith((".gz", ".csv")))
    EQUIVALENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EQUIVALENCE_JSON.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    return r


if __name__ == "__main__":
    import sys
    record_equivalence(Path(sys.argv[1]))
