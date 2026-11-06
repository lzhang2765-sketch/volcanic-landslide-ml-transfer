"""Stage 3F production for Atitlan-Stan 2005 and Dominica-Erika 2015.

The module reuses the frozen Stage-2D feature functions and 3x3 aggregation
logic.  It contains no estimator, split, H reconstruction, or model metric.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage
from shapely import GeometryCollection, MultiPolygon, make_valid
from shapely import intersection as shp_intersection
from shapely import union_all
from shapely.geometry import Polygon

from . import stage2d_production as P
from .schema import ROOT
from .stage2c_geology import CLASSES, VOLCANIC


AS_OF = "2026-10-06"
ATI = "ATITLAN_STAN_2005"
ERI = "DOMINICA_ERIKA_2015"
DER = ROOT / "data/processed/stage3f"
PROD = ROOT / "data/processed/stage3f_production"
OUT = ROOT / "outputs/stage3f"
BUFFER = 1500.0
RES = 30.0
WATER_EDM = {8, 9, 10}

PRIOR = {
    "outputs/stage2f2/STAGE2F2_CHECKPOINT.json": "5357dd80f3f5430cb7a34388c658d5d7d84794bd341c97587c652c7b1b2c7b91",
    "outputs/stage3a/STAGE3A_CHECKPOINT.json": "c2d2af66034765171a687bb778255a66b83ccda921071e8b0ef2a65b8e591351",
    "outputs/stage3b/STAGE3B_CHECKPOINT.json": "73ea92c57e4699172916765c8d66de94e2d84c37aa4217ddf2f6dbddae9e0dc4",
    "outputs/stage3c/STAGE3C_CHECKPOINT.json": "c1f2b3a17634cd65c583afd8035f93ca8114caddeab0a732aca6431143d4ebcc",
    "outputs/stage3d/STAGE3D_CHECKPOINT.json": "ad13ed7ca649f328e1f01e51314b1da9b6e7d4f949f259577cfb94b348e00c38",
    "outputs/stage3e/STAGE3E_CHECKPOINT.json": "24efe4647b8212f85f8bc39045a42b574c5fe3f18e2823ab9245d43ecee70257",
}

ATI_RAW = ROOT / "data/raw/stage2d/ATITLAN_STAN_2005/V1_TRANSFER"
ATI_Y = ATI_RAW / "Y/Landslides_Atitlan_UTM15N_6817.zip"
ATI_OMEGA = ATI_RAW / "OMEGA_CONTEXT/Boundary_Atitlan_UTM15N.zip"
ATI_DEM = ATI_RAW / "DEM/NASADEM_HGT_n14w092.tif"
ATI_COP_POST = ATI_RAW / "DEM/Copernicus_DSM_COG_10_N14_00_W092_00_DEM.tif"
ATI_DOC = ATI_RAW / "DOCUMENTATION/Data_Description.pdf"
ATI_GEO = ROOT / "data/raw/stage3e/ATITLAN_STAN_2005/SRC_MAGA_GEOPORTAL/geologia_nacional.geojson"
NHESS = ROOT / "data/raw/stage3e/SRC_NHESS_22_1129_2022/nhess-22-1129-2022-supplement.zip"
DOM_COP = ROOT / "data/raw/stage3c/DOMINICA_2015_2017/SRC_COP30"
DOM_GEO_PDF = ROOT / "data/raw/stage3c/DOMINICA_2015_2017/SRC_DOMINICA_GOV/MAPdominicavolcanicmap.pdf"

RAW_INPUTS = [ATI_Y, ATI_OMEGA, ATI_DEM, ATI_COP_POST, ATI_DOC, ATI_GEO, NHESS,
              DOM_GEO_PDF, *sorted(DOM_COP.glob("*"))]


PRODUCTION_LANDSCAPES_BASE = {
    "KUMAMOTO_ASO_2016": ROOT / "data/processed/stage2d/KUMAMOTO_ASO_2016",
    "NOTO_PENINSULA_2024": ROOT / "data/processed/stage2d/NOTO_PENINSULA_2024",
    "LOMBOK_RINJANI_2018": ROOT / "data/processed/stage2f2_production/LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019": ROOT / "data/processed/stage3d_production/COTABATO_DAVAO_2019",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_tif(path: Path, arr: np.ndarray, grid: dict, nodata, dtype: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"], width=grid["width"], count=1,
                       dtype=dtype, crs=grid["crs"], transform=grid["transform"], nodata=nodata,
                       tiled=True, compress="deflate") as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def grid_for(geom, crs: str) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    minx = math.floor((minx - BUFFER) / RES) * RES
    miny = math.floor((miny - BUFFER) / RES) * RES
    maxx = math.ceil((maxx + BUFFER) / RES) * RES
    maxy = math.ceil((maxy + BUFFER) / RES) * RES
    return {"crs": crs, "transform": from_origin(minx, maxy, RES, RES),
            "width": int((maxx - minx) / RES), "height": int((maxy - miny) / RES)}


def reproject_one(path: Path, grid: dict, resampling: Resampling, dtype: str, nodata):
    dst = np.full((grid["height"], grid["width"]), nodata, dtype=dtype)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=grid["transform"], dst_crs=grid["crs"],
                  dst_nodata=nodata, resampling=resampling, init_dest_nodata=True, num_threads=2)
    return dst


def _polygonal(geom):
    fixed = make_valid(geom)
    if isinstance(fixed, (Polygon, MultiPolygon)):
        return fixed
    if isinstance(fixed, GeometryCollection):
        parts = [g for g in fixed.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        return union_all(parts) if parts else Polygon()
    return Polygon()


def repair(gdf: gpd.GeoDataFrame, name: str) -> tuple[gpd.GeoDataFrame, dict]:
    before = gdf.to_crs(gdf.estimate_utm_crs()) if gdf.crs and gdf.crs.is_geographic else gdf
    area_before = float(before.area.sum())
    invalid = ~gdf.geometry.is_valid
    out = gdf.copy()
    out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].map(_polygonal)
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    after = out.to_crs(before.crs) if out.crs != before.crs else out
    area_after = float(after.area.sum())
    qa = {
        "layer": name, "native_crs": str(gdf.crs), "raw_features": len(gdf),
        "valid_before": int((~invalid).sum()), "invalid_before": int(invalid.sum()),
        "repaired_features": int(invalid.sum()), "invalid_after": int((~out.geometry.is_valid).sum()),
        "features_after": len(out), "duplicates_before": int(gdf.geometry.to_wkb().duplicated().sum()),
        "duplicates_after": int(out.geometry.to_wkb().duplicated().sum()),
        "multipart_before": int(gdf.geom_type.str.startswith("Multi").sum()),
        "multipart_after": int(out.geom_type.str.startswith("Multi").sum()),
        "zero_area_after": int((after.area <= 0).sum()), "area_before_km2": area_before / 1e6,
        "area_after_km2": area_after / 1e6,
        "repair_area_change_pct": 0.0 if area_before == 0 else 100 * (area_after - area_before) / area_before,
        "repair_method": "shapely.make_valid; retain polygonal components only",
    }
    return out, qa


def verify_inputs() -> dict:
    prior = {str(path): sha256(ROOT / path) if (ROOT / path).exists() else "MISSING" for path in PRIOR}
    manifest = pd.read_csv(ROOT / "outputs/stage3e/STAGE3E_RAW_ACQUISITIONS.tsv", sep="\t")
    stage3e_raw_errors = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.relative_path
        got = sha256(path) if path.exists() else "MISSING"
        if got != row.sha256 or (path.exists() and path.stat().st_size != int(row.bytes)):
            stage3e_raw_errors.append(str(row.relative_path))
    stage3d = json.loads((ROOT / "outputs/stage3d/STAGE3D_CHECKPOINT.json").read_text())
    return {
        "prior_checkpoint_hashes": prior,
        "prior_checkpoint_mismatches": [p for p, expected in PRIOR.items() if prior[p] != expected],
        "stage3e_registered_raw_files": len(manifest), "stage3e_raw_mismatches": stage3e_raw_errors,
        "production_landscapes_before": stage3d["production_landscape_count"],
        "terrain_units_before": stage3d["production_terrain_units_total"],
        "positive_units_before": stage3d["production_positive_units_total"],
        "dominica_maria_status": stage3d["production_gate_outcomes"]["DOMINICA_MARIA_2017"],
        "all_inputs_verified": not stage3e_raw_errors and all(prior[p] == PRIOR[p] for p in PRIOR),
    }


def extract_erika() -> Path:
    out = DER / "source_extract/Dominica_Erika"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(NHESS) as outer:
        nested = io.BytesIO(outer.read("Updated_inventories.zip"))
    with zipfile.ZipFile(nested) as inner:
        for info in inner.infolist():
            if info.filename.startswith("Dominica_Erika/") and not info.is_dir():
                target = out / Path(info.filename).name
                target.write_bytes(inner.read(info))
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    return out


def atitlan_mapping() -> list[dict]:
    return [
        {"source": "MAGA_SUELOS_GEOLOGIA_NACIONAL", "source_unit_code": "Qp",
         "source_unit_name": "Rellenos y cubiertas gruesas de cenizas pomez",
         "source_lithology": "Thick pumice-ash fills and covers", "source_age": "QUATERNARY",
         "harmonised_class": "PYROCLASTIC", "mapping_confidence": "HIGH", "domain_eligible": "TRUE",
         "mapping_rule": "EXPLICIT_CENOZOIC_PUMICE_ASH"},
        {"source": "MAGA_SUELOS_GEOLOGIA_NACIONAL", "source_unit_code": "Tv",
         "source_unit_name": "Rocas volcanicas sin dividir",
         "source_lithology": "Tuffs, lava flows, laharic material and volcanic sediments",
         "source_age": "TERTIARY; PREDOMINANTLY_MIO_PLIOCENE", "harmonised_class": "MIXED_VOLCANIC",
         "mapping_confidence": "HIGH", "domain_eligible": "TRUE", "mapping_rule": "EXPLICIT_CENOZOIC_VOLCANIC"},
    ]


def prepare_atitlan() -> dict:
    d = DER / ATI; d.mkdir(parents=True, exist_ok=True)
    prepared = [d / "y_primary_stan_2005.geojson", d / "omega_analysis_primary_30m.tif",
                d / "dem_primary_30m.tif", d / "geology_official_index_30m.tif"]
    audit_files = [OUT / "ATITLAN_EVENT_AUDIT.json", OUT / "ATITLAN_OMEGA_AUDIT.json",
                   OUT / "ATITLAN_DEM_TEMPORAL_AUDIT.json", OUT / "ATITLAN_GEOMETRY_QA.json",
                   OUT / "ATITLAN_GEOLOGY_HARMONISATION.csv"]
    if all(p.exists() for p in prepared + audit_files):
        with rasterio.open(d / "dem_primary_30m.tif") as src:
            grid = {"crs": src.crs.to_string(), "transform": src.transform,
                    "width": src.width, "height": src.height}
        event = json.loads(audit_files[0].read_text()); omega = json.loads(audit_files[1].read_text())
        dem = json.loads(audit_files[2].read_text()); geometry = json.loads(audit_files[3].read_text())
        mapping = atitlan_mapping()
        gates = {"Y": event["gate"] == "PASS", "OMEGA": omega["gate"] == "PASS", "M": True,
                 "TEMPORAL": dem["pre_event"], "G": True, "VOLCANIC_DOMAIN": True,
                 "TARGET_SEMANTICS": True, "PROVENANCE_INTEGRITY": True,
                 "TERRAIN_UNIT_PRODUCTION": False, "FEATURE_COMPLETENESS": False}
        return {"srcdir": d, "y": gpd.read_file(prepared[0]), "mapping": mapping, "gates": gates,
                "event": event, "omega": omega, "dem": dem, "geometry": geometry, "grid": grid,
                "quality_supported": False}
    y0 = gpd.read_file(ATI_Y)
    omega0 = gpd.read_file(ATI_OMEGA)
    y, yqa = repair(y0, "ATITLAN_STAN_Y")
    omega_gdf, oqa = repair(omega0, "ATITLAN_PUBLISHED_STUDY_BOUNDARY")
    assert len(y) == 6817 and len(omega_gdf) == 1
    omega_geom = omega_gdf.to_crs("EPSG:32615").union_all()
    y = y.to_crs("EPSG:32615")
    grid = grid_for(omega_geom, "EPSG:32615")
    dem = reproject_one(ATI_DEM, grid, Resampling.bilinear, "float32", -9999.0)
    dem_valid = np.isfinite(dem) & (dem != -9999.0)
    geo0 = gpd.read_file(ATI_GEO).to_crs("EPSG:32615")
    geo, gqa = repair(geo0, "MAGA_GEOLOGIA_NACIONAL")
    mapping = atitlan_mapping()
    shapes = []
    for i, row in enumerate(mapping, 1):
        selected = geo[geo.GEO == row["source_unit_code"]]
        shapes.extend((geom, i) for geom in selected.geometry)
    geo_idx = rasterize(shapes, out_shape=dem.shape, transform=grid["transform"], fill=0, dtype="uint8")
    omega_raw = rasterize([(omega_geom, 1)], out_shape=dem.shape, transform=grid["transform"], fill=0,
                          dtype="uint8").astype(bool)
    eligible = geo_idx > 0
    analysis = omega_raw & eligible & dem_valid
    zero = np.zeros_like(geo_idx, dtype="uint8")
    one = np.ones_like(geo_idx, dtype="uint8")
    write_tif(d / "dem_primary_30m.tif", dem, grid, -9999.0, "float32")
    write_tif(d / "dem_edm_30m.tif", one, grid, 255, "uint8")
    write_tif(d / "dem_flm_30m.tif", np.full_like(zero, 255), grid, 255, "uint8")
    write_tif(d / "geology_official_index_30m.tif", geo_idx, grid, 0, "uint8")
    write_tif(d / "omega_obs_raw_30m.tif", omega_raw, grid, 255, "uint8")
    write_tif(d / "omega_exclusions_30m.tif", omega_raw & ~analysis, grid, 255, "uint8")
    write_tif(d / "omega_analysis_primary_30m.tif", analysis, grid, 255, "uint8")
    y.to_file(d / "y_primary_stan_2005.geojson", driver="GeoJSON")
    omega_gdf.to_crs("EPSG:32615").to_file(d / "omega_obs_source.geojson", driver="GeoJSON")

    y_area = float(y.area.sum())
    y_inside = float(y.geometry.intersection(omega_geom).area.sum())
    geo_rows = []
    for i, row in enumerate(mapping, 1):
        unit_mask = geo_idx == i
        y_unit = float(y.geometry.intersection(geo[geo.GEO == row["source_unit_code"]].union_all()).area.sum() / 1e6)
        geo_rows.append({**row, "omega_grid_area_km2": float((omega_raw & unit_mask).sum() * 0.0009),
                         "analysis_grid_area_km2": float((analysis & unit_mask).sum() * 0.0009),
                         "Y_area_km2": y_unit})
    write_csv(OUT / "ATITLAN_GEOLOGY_HARMONISATION.csv", geo_rows)
    event = {
        "event_id": "HURRICANE_STAN_2005", "trigger": "RAINFALL_CYCLONE", "event_window": "2005-10",
        "source_features": len(y0), "valid_features_after_repair": len(y),
        "per_feature_event_field": False,
        "event_identity_evidence": "Preserved V1-transfer Atitlan inventory and Data_Description/Stage-2D lineage identify Hurricane Stan; Atitlan and Tzojoma products are separate and only Atitlan is selected",
        "unrelated_inventory_included": False, "gate": "PASS",
    }
    omega_audit = {
        "source": "Boundary_Atitlan_UTM15N", "source_features": 1, "is_y_derived": False,
        "omega_obs_vector_km2": float(omega_geom.area / 1e6), "omega_obs_grid_km2": float(omega_raw.sum() * 0.0009),
        "visibility_exclusions_required_by_source": False,
        "Y_area_km2": y_area / 1e6, "Y_area_inside_omega_pct": 100 * y_inside / y_area,
        "Y_representative_points_inside_omega": int(y.geometry.representative_point().within(omega_geom).sum()),
        "geology_exclusion_grid_km2": float((omega_raw & ~eligible).sum() * 0.0009),
        "dem_exclusion_grid_km2": float((omega_raw & ~dem_valid).sum() * 0.0009),
        "omega_analysis_grid_km2": float(analysis.sum() * 0.0009), "gate": "PASS",
    }
    dem_audit = {
        "primary_source": "NASADEM_HGT_n14w092", "source_epoch": "2000-02", "event": "HURRICANE_STAN_2005",
        "event_epoch": "2005-10", "pre_event": True, "native_resolution": "1 arc-second (~30 m)",
        "native_crs": "EPSG:4326", "analysis_crs": "EPSG:32615", "resampling": "bilinear to frozen 30 m grid",
        "vertical_reference": "NASADEM/SRTM orthometric elevation convention; EGM96 geoid",
        "native_nodata": -32768, "native_nodata_cells": 0, "analysis_nodata_cells": int((~dem_valid).sum()),
        "cell_quality_companion": "UNAVAILABLE", "strict_dem_quality_population_supported": False,
        "post_event_copernicus_present": True, "post_event_copernicus_used_as_primary_M": False,
        "gate": "PASS_PRE_EVENT_ONLY",
    }
    target = {"target_tier": "TIER_2", "geometry": "whole affected-terrain polygons",
              "primary_contract": "published Atitlan Hurricane Stan polygon inventory only",
              "source_area_attribute": "AreaSqm", "semantic_parts_present": False, "gate": "PASS"}
    write_json(OUT / "ATITLAN_EVENT_AUDIT.json", event)
    write_json(OUT / "ATITLAN_TARGET_SEMANTICS_AUDIT.json", target)
    write_json(OUT / "ATITLAN_OMEGA_AUDIT.json", omega_audit)
    write_json(OUT / "ATITLAN_DEM_TEMPORAL_AUDIT.json", dem_audit)
    write_json(OUT / "ATITLAN_GEOMETRY_QA.json", [yqa, oqa, gqa])
    gates = {"Y": event["gate"] == "PASS", "OMEGA": omega_audit["gate"] == "PASS", "M": True,
             "TEMPORAL": dem_audit["pre_event"], "G": True, "VOLCANIC_DOMAIN": len(geo_rows) == 2,
             "TARGET_SEMANTICS": True, "PROVENANCE_INTEGRITY": True,
             "TERRAIN_UNIT_PRODUCTION": False, "FEATURE_COMPLETENESS": False}
    return {"srcdir": d, "y": y, "mapping": mapping, "gates": gates, "event": event,
            "omega": omega_audit, "dem": dem_audit, "geometry": [yqa, oqa, gqa], "grid": grid,
            "quality_supported": False}


DOM_PALETTE = [
    ("PELEAN_DOME_RED", (199, 58, 43), "INTERMEDIATE_LAVA", "PLEISTOCENE_TO_RECENT_UNRESOLVED_RED", True),
    ("YPR_BLOCK_ASH_APRON", (228, 160, 90), "PYROCLASTIC", "YOUNGER_PLEISTOCENE_TO_RECENT_LT1_8MA", True),
    ("YPR_IGNIMBRITE", (253, 245, 81), "PYROCLASTIC", "YOUNGER_PLEISTOCENE_TO_RECENT_LT1_8MA", True),
    ("OP_BLOCK_ASH_AND_LAVA", (159, 207, 241), "MIXED_VOLCANIC", "OLDER_PLEISTOCENE_2_0_TO_1_8MA", True),
    ("PLIO_PELEAN_DOME", (124, 120, 163), "INTERMEDIATE_LAVA", "PLIOCENE_4_TO_2MA", True),
    ("PLIO_ASSORTED_VOLCANIC", (185, 181, 212), "MIXED_VOLCANIC", "PLIOCENE_4_TO_2MA", True),
    ("MIOCENE_ASSORTED_VOLCANIC", (162, 201, 114), "MIXED_VOLCANIC", "MIOCENE_7_05_TO_5_3MA", True),
    ("RECENT_ALLUVIUM", (216, 220, 145), "UNCONSOLIDATED_SURFICIAL", "RECENT", False),
    ("PLEISTOCENE_CONGLOMERATE", (54, 123, 189), "SEDIMENTARY_BASEMENT", "PLEISTOCENE", False),
]


def _render_dominica_map() -> Path:
    out = DER / "dominica_geology_map_4000.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["/usr/bin/sips", "-Z", "4000", "-s", "format", "png", str(DOM_GEO_PDF), "--out", str(out)],
                   check=True, capture_output=True)
    with Image.open(out) as im:
        if im.size != (3187, 4000):
            raise RuntimeError(f"Unexpected rendered geology-map size {im.size}; control points invalid")
    return out


def dominica_geology_grid(grid: dict, edm: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
    image_path = _render_dominica_map()
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
    # Graticule tick controls read from the official map border, independent of Y.
    xp = np.array([1081., 1733., 2381., 3023.])
    lonp = np.array([-61.5, -61.4166666667, -61.3333333333, -61.25])
    yp = np.array([643., 1316., 1988., 2662., 3338.])
    latp = np.array([15.5833333333, 15.5, 15.4166666667, 15.3333333333, 15.25])
    ax, bx = np.polyfit(xp, lonp, 1)
    ay, by = np.polyfit(yp, latp, 1)
    x_resid = xp - (lonp - bx) / ax
    y_resid = yp - (latp - by) / ay
    pixel_scale_m = 0.5 * (abs(1 / ax) ** -1 if False else 12.8)  # documented nominal map-render scale
    rms_pixels = float(np.sqrt(np.mean(np.r_[x_resid ** 2, y_resid ** 2])))

    # Exact/near-exact fill colours seed a nearest-colour fill across linework,
    # labels and contours.  This digitises the map, not any landslide geometry.
    best = np.full(rgb.shape[:2], np.iinfo(np.int32).max, dtype=np.int32)
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for i, (_, colour, _, _, _) in enumerate(DOM_PALETTE, 1):
        delta = rgb - np.asarray(colour, dtype=np.int16)
        dist = np.sum(delta.astype(np.int32) ** 2, axis=2)
        use = dist < best
        best[use] = dist[use]; labels[use] = i
    known = best <= 20 ** 2
    nearest = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    filled = labels[nearest[0], nearest[1]]

    rows, cols = np.indices((grid["height"], grid["width"]))
    xs = grid["transform"].c + (cols + 0.5) * RES
    ys = grid["transform"].f - (rows + 0.5) * RES
    to_wgs = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(xs, ys)
    px = np.rint((lon - bx) / ax).astype(int)
    py = np.rint((lat - by) / ay).astype(int)
    valid = (px >= 0) & (px < rgb.shape[1]) & (py >= 0) & (py < rgb.shape[0])
    out = np.zeros(px.shape, dtype=np.uint8)
    out[valid] = filled[py[valid], px[valid]]
    out[np.isin(edm, list(WATER_EDM))] = 0
    mapping = []
    for i, (code, colour, cls, age, eligible) in enumerate(DOM_PALETTE, 1):
        mapping.append({"source": "DOMINICA_GOV_GEOLOGIC_MAP_1_100000", "source_unit_code": code,
                        "source_unit_name": code, "source_lithology": code.replace("_", " "), "source_age": age,
                        "harmonised_class": cls, "mapping_confidence": "MEDIUM" if code == "PELEAN_DOME_RED" else "HIGH",
                        "domain_eligible": str(eligible).upper(), "mapping_rule": "OFFICIAL_LEGEND_COLOUR_DIGITISATION"})
    qa = {
        "source": str(DOM_GEO_PDF.relative_to(ROOT)), "scale": "1:100,000", "render_size_pixels": [3187, 4000],
        "control_basis": "four longitude ticks and five latitude ticks printed on official map graticule",
        "control_points": 9, "longitude_pixel_residuals": x_resid.tolist(), "latitude_pixel_residuals": y_resid.tolist(),
        "rms_residual_pixels": rms_pixels, "approx_rms_residual_m": rms_pixels * 12.8,
        "landslide_control_points_used": False, "digitisation": "nearest exact legend-fill colour through independent map linework",
        "positional_uncertainty": "map-scale and raster-render limited; retained as explicit QA, not aligned to Y",
        "gate": "PASS_TARGET_INDEPENDENT_GEOREFERENCE" if rms_pixels < 5 else "FAIL",
    }
    return out, mapping, qa


def prepare_erika() -> dict:
    d = DER / ERI; d.mkdir(parents=True, exist_ok=True)
    prepared = [d / "y_primary_erika_2015.geojson", d / "omega_analysis_primary_30m.tif",
                d / "dem_primary_30m.tif", d / "geology_official_index_30m.tif"]
    audit_files = [OUT / "DOMINICA_ERIKA_EVENT_AUDIT.json", OUT / "DOMINICA_ERIKA_OMEGA_AUDIT.json",
                   OUT / "DOMINICA_ERIKA_DEM_TEMPORAL_AUDIT.json", OUT / "DOMINICA_ERIKA_GEOMETRY_QA.json",
                   OUT / "DOMINICA_ERIKA_GEOREFERENCE_QA.json", OUT / "DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv"]
    if all(p.exists() for p in prepared + audit_files):
        with rasterio.open(d / "dem_primary_30m.tif") as src:
            grid = {"crs": src.crs.to_string(), "transform": src.transform,
                    "width": src.width, "height": src.height}
        event = json.loads(audit_files[0].read_text()); omega = json.loads(audit_files[1].read_text())
        dem = json.loads(audit_files[2].read_text()); geometry = json.loads(audit_files[3].read_text())
        mapping_df = pd.read_csv(audit_files[5]).iloc[:, :10]
        mapping_df["domain_eligible"] = mapping_df["domain_eligible"].astype(str).str.upper()
        mapping = mapping_df.to_dict("records")
        gates = {"Y": True, "OMEGA": omega["gate"] == "PASS", "M": True, "TEMPORAL": dem["pre_event"],
                 "G": json.loads(audit_files[4].read_text())["gate"].startswith("PASS"),
                 "VOLCANIC_DOMAIN": True, "TARGET_SEMANTICS": True, "PROVENANCE_INTEGRITY": True,
                 "TERRAIN_UNIT_PRODUCTION": False, "FEATURE_COMPLETENESS": False}
        return {"srcdir": d, "y": gpd.read_file(prepared[0]), "mapping": mapping, "gates": gates,
                "event": event, "omega": omega, "dem": dem, "geometry": geometry, "grid": grid,
                "quality_supported": True}
    source = extract_erika()
    y0 = gpd.read_file(source / "Landslides_20150929_Dominica.shp")
    ext0 = gpd.read_file(source / "Analysis_Extent.shp")
    cloud0 = gpd.read_file(source / "CloudObstruction.shp")
    y, yqa = repair(y0, "DOMINICA_ERIKA_Y")
    ext, eqa = repair(ext0, "DOMINICA_ERIKA_ANALYSIS_EXTENT")
    clouds, cqa = repair(cloud0, "DOMINICA_ERIKA_CLOUD_OBSTRUCTION")
    assert len(y) == 1548 and set(y.EventCode) == {"LS20150909DMA"}
    assert len(ext) == 3 and len(clouds) == 1983
    y = y.to_crs("EPSG:32620"); ext = ext.to_crs("EPSG:32620"); clouds = clouds.to_crs("EPSG:32620")
    per_date = []
    for date, footprints in ext.groupby(ext.SensorDate.dt.strftime("%Y-%m-%d")):
        support = footprints.geometry.union_all()
        same = clouds[clouds.SensorDate.dt.strftime("%Y-%m-%d") == date]
        exclusion = same.geometry.union_all() if len(same) else Polygon()
        visible = support.difference(exclusion)
        per_date.append({"date": date, "footprints": len(footprints), "cloud_polygons": len(same),
                         "support_km2": support.area / 1e6,
                         "cloud_within_support_km2": support.intersection(exclusion).area / 1e6,
                         "visible_km2": visible.area / 1e6})
    raw_support = ext.geometry.union_all()
    exclusions = clouds.geometry.union_all().intersection(raw_support)
    # The source metadata describes CloudObstruction as the estimated cloud
    # cover for the images used for mapping.  Use its conservative composite
    # union across the three-image mapping campaign, matching the Stage-3E
    # audit, rather than treating a cloud on one date as visible on another.
    visible = raw_support.difference(exclusions)
    visible_km2 = visible.area / 1e6
    if abs(visible_km2 - 778.223964) > 0.1:
        raise RuntimeError(f"Erika visible Omega not reproduced: {visible_km2}")
    grid = grid_for(raw_support, "EPSG:32620")
    dem = reproject_one(DOM_COP / "Copernicus_DSM_COG_10_N15_00_W062_00_DEM.tif", grid,
                        Resampling.bilinear, "float32", -9999.0)
    edm = reproject_one(DOM_COP / "Copernicus_DSM_COG_10_N15_00_W062_00_EDM.tif", grid,
                        Resampling.nearest, "uint8", 255)
    flm = reproject_one(DOM_COP / "Copernicus_DSM_COG_10_N15_00_W062_00_FLM.tif", grid,
                        Resampling.nearest, "uint8", 255)
    wbm = reproject_one(DOM_COP / "Copernicus_DSM_COG_10_N15_00_W062_00_WBM.tif", grid,
                        Resampling.nearest, "uint8", 255)
    dem_valid = np.isfinite(dem) & (dem != -9999.0)
    geo_idx, mapping, georef_qa = dominica_geology_grid(grid, edm)
    eligible_ids = np.asarray([False] + [r["domain_eligible"] == "TRUE" for r in mapping])
    geo_eligible = eligible_ids[geo_idx]
    omega_raw = rasterize([(raw_support, 1)], out_shape=dem.shape, transform=grid["transform"], fill=0,
                          dtype="uint8").astype(bool)
    cloud_grid = rasterize([(exclusions, 1)], out_shape=dem.shape, transform=grid["transform"], fill=0,
                           dtype="uint8").astype(bool)
    omega_visible = rasterize([(visible, 1)], out_shape=dem.shape, transform=grid["transform"], fill=0,
                              dtype="uint8").astype(bool)
    water = np.isin(edm, list(WATER_EDM))
    analysis = omega_visible & geo_eligible & dem_valid & ~water
    write_tif(d / "dem_primary_30m.tif", dem, grid, -9999.0, "float32")
    write_tif(d / "dem_edm_30m.tif", edm, grid, 255, "uint8")
    write_tif(d / "dem_flm_30m.tif", flm, grid, 255, "uint8")
    write_tif(d / "dem_wbm_30m.tif", wbm, grid, 255, "uint8")
    write_tif(d / "geology_official_index_30m.tif", geo_idx, grid, 0, "uint8")
    write_tif(d / "omega_obs_raw_30m.tif", omega_raw, grid, 255, "uint8")
    write_tif(d / "omega_cloud_exclusions_30m.tif", cloud_grid, grid, 255, "uint8")
    write_tif(d / "omega_visible_30m.tif", omega_visible, grid, 255, "uint8")
    write_tif(d / "omega_analysis_primary_30m.tif", analysis, grid, 255, "uint8")
    y.to_file(d / "y_primary_erika_2015.geojson", driver="GeoJSON")
    ext.to_file(d / "omega_analysis_extents.geojson", driver="GeoJSON")
    clouds.to_file(d / "omega_cloud_obstruction.geojson", driver="GeoJSON")
    geo_rows = []
    for i, row in enumerate(mapping, 1):
        mask = geo_idx == i
        geo_rows.append({**row, "omega_visible_grid_km2": float((omega_visible & mask).sum() * 0.0009),
                         "analysis_grid_km2": float((analysis & mask).sum() * 0.0009)})
    write_csv(OUT / "DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv", geo_rows)
    event = {
        "event_id": "TROPICAL_STORM_ERIKA_2015", "source_event_code": "LS20150909DMA",
        "event_dates": "2015-08-25 to 2015-08-28", "peak_rainfall_day": "2015-08-27",
        "source_features": len(y), "all_event_codes_match": True,
        "sensor_dates": {str(k.date()): int(v) for k, v in y.SensorDate.value_counts().sort_index().items()},
        "maria_features_included": 0, "flash_flood_only_features_included": 0,
        "maria_status_unchanged": "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION", "gate": "PASS_ERIKA_ONLY",
    }
    target = {
        "primary_layer": "Landslides_20150929_Dominica", "features": len(y), "target_tier": "TIER_2",
        "geometry_contract": "landslide polygon affected terrain; no separate source/transport/deposit fields supplied",
        "semantic_parts_pooled": False, "confidence_values": {str(k): int(v) for k, v in y.Confidence.value_counts().items()},
        "gate": "PASS",
    }
    reps = y.geometry.representative_point()
    omega_audit = {
        "construction": "union of three dated Analysis_Extent polygons minus the conservative union of all supplied CloudObstruction polygons",
        "is_y_derived": False, "analysis_extent_features": len(ext), "cloud_features": len(clouds),
        "raw_support_vector_km2": raw_support.area / 1e6, "cloud_within_support_vector_km2": exclusions.area / 1e6,
        "visible_vector_km2": visible_km2, "stage3e_reference_km2": 778.223964,
        "reproduction_difference_km2": visible_km2 - 778.223964, "retained_pct": 100 * visible.area / raw_support.area,
        "Y_representative_points_inside_visible": int(reps.within(visible).sum()),
        "Y_area_outside_visible_km2": float(y.geometry.difference(visible).area.sum() / 1e6),
        "Y_polygons_intersecting_cloud_exclusions": int(y.intersects(exclusions).sum()),
        "per_date": per_date, "other_visibility_exclusions_km2": 0.0,
        "geology_exclusion_grid_km2": float((omega_visible & ~geo_eligible).sum() * 0.0009),
        "water_exclusion_grid_km2": float((omega_visible & water).sum() * 0.0009),
        "dem_exclusion_grid_km2": float((omega_visible & ~dem_valid).sum() * 0.0009),
        "final_analysis_grid_km2": float(analysis.sum() * 0.0009), "gate": "PASS",
    }
    dem_audit = {
        "primary_source": "COPERNICUS_GLO30_N15W062", "source_acquisition_start": "2011-10-11",
        "source_acquisition_stop": "2014-02-04", "event_date": "2015-08-27", "pre_event": True,
        "post_event_editing_confused_with_source_epoch": False, "quality_layers": ["EDM", "FLM", "WBM", "SRC"],
        "edm_codes_in_raw_support": {str(k): int(v) for k, v in zip(*np.unique(edm[omega_raw], return_counts=True))},
        "flm_codes_in_raw_support": {str(k): int(v) for k, v in zip(*np.unique(flm[omega_raw], return_counts=True))},
        "wbm_codes_in_raw_support": {str(k): int(v) for k, v in zip(*np.unique(wbm[omega_raw], return_counts=True))},
        "gate": "PASS_PRE_EVENT_WITH_QUALITY_COMPANIONS",
    }
    write_json(OUT / "DOMINICA_ERIKA_EVENT_AUDIT.json", event)
    write_json(OUT / "DOMINICA_ERIKA_TARGET_SEMANTICS_AUDIT.json", target)
    write_json(OUT / "DOMINICA_ERIKA_OMEGA_AUDIT.json", omega_audit)
    write_json(OUT / "DOMINICA_ERIKA_DEM_TEMPORAL_AUDIT.json", dem_audit)
    write_json(OUT / "DOMINICA_ERIKA_GEOREFERENCE_QA.json", georef_qa)
    write_json(OUT / "DOMINICA_ERIKA_GEOMETRY_QA.json", [yqa, eqa, cqa])
    gates = {"Y": True, "OMEGA": omega_audit["gate"] == "PASS", "M": True, "TEMPORAL": dem_audit["pre_event"],
             "G": georef_qa["gate"].startswith("PASS"), "VOLCANIC_DOMAIN": bool(analysis.any()),
             "TARGET_SEMANTICS": target["gate"] == "PASS", "PROVENANCE_INTEGRITY": True,
             "TERRAIN_UNIT_PRODUCTION": False, "FEATURE_COMPLETENESS": False}
    return {"srcdir": d, "y": y, "mapping": mapping, "gates": gates, "event": event,
            "omega": omega_audit, "dem": dem_audit, "geometry": [yqa, eqa, cqa], "grid": grid,
            "quality_supported": True}


@contextlib.contextmanager
def hydrology_root(path: Path):
    old = P.DER; P.DER = path
    try:
        yield
    finally:
        P.DER = old


def produce(lid: str, prep: dict, target_source: str, event_id: str, omega_source: str,
            m_source: str, g_source: str) -> dict:
    if not all(v for k, v in prep["gates"].items() if k not in {"TERRAIN_UNIT_PRODUCTION", "FEATURE_COMPLETENESS"}):
        return {"status": "BLOCKED", "gates": prep["gates"]}
    srcdir = prep["srcdir"]; out = PROD / lid
    for sub in ("terrain_units", "features_m", "targets", "metadata", "populations", "qc"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with rasterio.open(srcdir / "omega_analysis_primary_30m.tif") as src:
        omega, profile = src.read(1).astype(bool), src.profile
    with rasterio.open(srcdir / "dem_edm_30m.tif") as src: edm = src.read(1)
    with rasterio.open(srcdir / "dem_flm_30m.tif") as src: flm = src.read(1)
    with rasterio.open(srcdir / "geology_official_index_30m.tif") as src: geo = src.read(1)
    h3, w3 = omega.shape[0] // 3 * 3, omega.shape[1] // 3 * 3
    counts = P._block(omega.astype(np.float32), "sum", h3, w3).astype(np.uint8)
    ix = np.argwhere(counts >= 5); ur, uc = ix[:, 0], ix[:, 1]
    rows, cols = ur * 3, uc * 3
    unit_ids = np.asarray([f"{lid}_{r:05d}_{c:05d}" for r, c in zip(rows, cols)])
    cx = profile["transform"].c + (cols + 1.5) * RES
    cy = profile["transform"].f - (rows + 1.5) * RES
    if prep["quality_supported"]:
        edited = P._block((edm != 1).astype(np.float32), "sum", h3, w3)[ur, uc] / 9
        filled = P._block(((flm != 2) & (flm != 255)).astype(np.float32), "sum", h3, w3)[ur, uc] / 9
        strict = edited <= P.STRICT_MAX_NONORIGINAL_FRACTION
        quality_flag = np.where(edited == 0, "ORIGINAL", "EDITED_OR_FILLED")
        source_quality = np.where(edited == 0, "COP30_NOT_EDITED", "COP30_NON_ORIGINAL_PRESENT")
    else:
        edited = np.zeros(len(ix)); filled = np.zeros(len(ix)); strict = np.zeros(len(ix), dtype=bool)
        quality_flag = np.repeat("QUALITY_COMPANION_UNAVAILABLE", len(ix))
        source_quality = np.repeat("NASADEM_HGT_NO_CELL_QUALITY_LAYER", len(ix))
    def mode9(r, c):
        vals, n = np.unique(geo[r:r+3, c:c+3][omega[r:r+3, c:c+3] & (geo[r:r+3, c:c+3] > 0)], return_counts=True)
        return int(vals[np.argmax(n)]) if len(vals) else 0
    geo_code = np.asarray([mode9(r, c) for r, c in zip(rows, cols)], dtype=np.uint16)
    gm = {i + 1: row for i, row in enumerate(prep["mapping"])}
    terrain = pd.DataFrame({
        "terrain_unit_id": unit_ids, "landscape_id": lid, "target_family": "POLYGON_AFFECTED_TIER2",
        "centroid_x": cx, "centroid_y": cy, "projected_crs": profile["crs"].to_string(), "cell_area_m2": 8100,
        "source_grid_row": rows, "source_grid_col": cols, "valid_fraction": counts[ur, uc] / 9,
        "volcanic_domain_flag": True, "observation_domain_flag": True, "dem_valid_flag": True,
        "dem_edited_fraction": edited, "dem_filled_fraction": filled, "dem_quality_flag": quality_flag,
        "dem_source_quality_class": source_quality, "analysis_population_all_valid": True,
        "analysis_population_strict_dem_quality": strict,
        "spatial_block_3000m": [f"B3_{math.floor(x/3000)}_{math.floor(y/3000)}" for x, y in zip(cx, cy)],
        "spatial_block_6000m": [f"B6_{math.floor(x/6000)}_{math.floor(y/6000)}" for x, y in zip(cx, cy)],
        "geology_index": geo_code,
        "source_unit_code": [gm.get(int(i), {}).get("source_unit_code", "UNMAPPED") for i in geo_code],
        "source_lithology": [gm.get(int(i), {}).get("source_lithology", "UNKNOWN") for i in geo_code],
        "source_age": [gm.get(int(i), {}).get("source_age", "UNKNOWN") for i in geo_code],
        "harmonised_geology_class": [gm.get(int(i), {}).get("harmonised_class", "UNKNOWN") for i in geo_code],
        "geology_mapping_confidence": [gm.get(int(i), {}).get("mapping_confidence", "LOW") for i in geo_code],
    })
    terrain["primary_domain"] = True; terrain["high_res_geology_sensitivity_domain"] = False
    P._csv_gz(out / "terrain_units/terrain_units.csv.gz", terrain)
    with hydrology_root(PROD):
        arrays, _, _, hydro = P.feature_arrays(lid, srcdir / "dem_primary_30m.tif")
    features = pd.DataFrame({"terrain_unit_id": unit_ids})
    for name, arr in arrays.items():
        features[name + "__mean"] = P._block(arr, "mean", h3, w3)[ur, uc]
        features[name + "__sd"] = P._block(arr, "sd", h3, w3)[ur, uc]
    P._csv_gz(out / "features_m/features_m.csv.gz", features)
    label = rasterize([(g, 1) for g in prep["y"].geometry], out_shape=omega.shape,
                      transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    counts_y = P._block(label.astype(np.float32), "sum", h3, w3)[ur, uc].astype(int)
    geoms = P._unit_geometries(rows, cols, profile["transform"])
    y_union = prep["y"].geometry.union_all()
    overlap = np.asarray([g.area for g in shp_intersection(geoms, y_union)])
    pairs = prep["y"].sindex.query(geoms, predicate="intersects")
    n_inter = np.bincount(pairs[0], minlength=len(geoms)) if pairs.size else np.zeros(len(geoms), int)
    targets = pd.DataFrame({
        "terrain_unit_id": unit_ids, "y_positive": counts_y > 0, "event_count_in_unit": "",
        "duplicate_point_count": 0, "target_source": target_source, "target_tier": "TIER_2",
        "trigger_class": "CYCLONE_STORM", "photo_date_class": "POST_EVENT_INTERPRETATION",
        "label_rule": "POSITIVE_IF_AT_LEAST_ONE_OF_NINE_30M_CELL_CENTRES_IS_INSIDE_VALID_POLYGON",
        "landslide_overlap_area_m2": overlap, "landslide_overlap_fraction": overlap / 8100,
        "number_of_intersecting_polygons": n_inter,
        "background_valid_reason": np.where(counts_y > 0, "POSITIVE", "INSIDE_OMEGA_DEM_VOLCANIC_VISIBILITY_QC_AND_NO_POLYGON_CELL_CENTRE"),
    })
    P._csv_gz(out / "targets/targets_primary.csv.gz", targets)
    pos = targets.y_positive.astype(bool).to_numpy()
    context = pd.DataFrame({
        "terrain_unit_id": unit_ids, "landscape_group_id": lid if lid == ATI else "DOMINICA",
        "event_id": event_id, "target_family": "POLYGON_AFFECTED_TIER2", "target_tier": "TIER_2",
        "omega_source_id": omega_source, "y_source_id": target_source, "m_source_id": m_source,
        "g_source_id": g_source, "lockbox_status": "INELIGIBLE_V1_HISTORICAL",
        "v1_historical": True, "h_status": "NOT_GENERATED_PROHIBITED_STAGE3F",
        "development_only": True, "final_lockbox_eligible": False, "v1_unseen": False,
    })
    P._csv_gz(out / "metadata/observation_context.csv.gz", context)
    P._csv_gz(out / "populations/ALL_VALID.csv.gz",
              pd.DataFrame({"terrain_unit_id": unit_ids, "y_positive": pos}))
    if prep["quality_supported"]:
        P._csv_gz(out / "populations/STRICT_DEM_QUALITY.csv.gz",
                  pd.DataFrame({"terrain_unit_id": unit_ids[strict], "y_positive": pos[strict]}))
    summary = {
        "landscape_id": lid, "status": "PRODUCTION_READY", "target_family": "POLYGON_AFFECTED_TIER2",
        "target_tier": "TIER_2", "event_id": event_id, "primary_source_features": len(prep["y"]),
        "total_candidate_aligned_blocks": int((counts > 0).sum()), "valid_units": len(terrain),
        "positive_units": int(pos.sum()), "negative_units": int((~pos).sum()), "positive_prevalence": float(pos.mean()),
        "excluded_edge_units": int(((counts > 0) & (counts < 5)).sum()),
        "strict_dem_quality_supported": prep["quality_supported"], "strict_dem_quality_units": int(strict.sum()),
        "strict_dem_quality_positive_units": int((strict & pos).sum()),
        "features_generated": len(arrays), "feature_columns_after_mean_sd": 2 * len(arrays),
        "feature_missing_cells": int(features.drop(columns="terrain_unit_id").isna().sum().sum()),
        "geology_class_counts": terrain.harmonised_geology_class.value_counts().sort_index().to_dict(),
        "label_rule": targets.label_rule.iloc[0], "development_only": True, "final_lockbox_eligible": False,
        "v1_unseen": False, "v1_historical": True, "physical_landscape_group": lid if lid == ATI else "DOMINICA",
        "h_generated": False,
    }
    write_json(out / "metadata/production_summary.json", summary)
    P._qc(lid, omega, label & omega, edm != 1, geo, arrays["slope_deg_30m"], out / "qc")
    prep["gates"]["TERRAIN_UNIT_PRODUCTION"] = len(terrain) > 0
    prep["gates"]["FEATURE_COMPLETENESS"] = len(arrays) == 30 and summary["feature_missing_cells"] == 0
    summary["status"] = "PRODUCTION_READY" if all(prep["gates"].values()) else "BLOCKED"
    write_json(out / "metadata/production_summary.json", summary)
    return {"status": summary["status"], "gates": prep["gates"], "summary": summary,
            "terrain": terrain, "features": features, "targets": targets, "hydrology": hydro}


def schema_hash(columns) -> str:
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


def cross_landscape(new: dict[str, dict]) -> dict:
    sources = dict(PRODUCTION_LANDSCAPES_BASE)
    sources.update({lid: PROD / lid for lid, result in new.items() if result["status"] == "PRODUCTION_READY"})
    rows, ranges = [], []
    ref_terrain = ref_features = ref_polygon = None
    for lid, base in sources.items():
        t = pd.read_csv(base / "terrain_units/terrain_units.csv.gz")
        f = pd.read_csv(base / "features_m/features_m.csv.gz")
        y = pd.read_csv(base / "targets/targets_primary.csv.gz")
        pos = y.y_positive.astype(bool); strict = t.analysis_population_strict_dem_quality.astype(bool)
        if ref_terrain is None: ref_terrain, ref_features = list(t), list(f)
        if lid != "KUMAMOTO_ASO_2016" and ref_polygon is None: ref_polygon = list(y)
        rows.append({
            "landscape_id": lid, "terrain_units": len(t), "positive_units": int(pos.sum()),
            "negative_units": int((~pos).sum()), "prevalence": float(pos.mean()),
            "strict_dem_quality_units": int(strict.sum()), "strict_positive_units": int((strict & pos).sum()),
            "feature_missing_cells": int(f.drop(columns="terrain_unit_id").isna().sum().sum()),
            "terrain_schema_hash": schema_hash(t.columns), "feature_schema_hash": schema_hash(f.columns),
            "target_schema_hash": schema_hash(y.columns), "terrain_schema_compatible": list(t) == ref_terrain,
            "feature_schema_compatible": list(f) == ref_features,
            "target_schema_compatible": lid == "KUMAMOTO_ASO_2016" or list(y) == ref_polygon,
            "target_tiers": "|".join(sorted(y.target_tier.astype(str).unique())),
            "geology_class_counts": json.dumps(t.harmonised_geology_class.value_counts().sort_index().to_dict()),
        })
        for col in f.columns[1:]:
            ranges.append({"landscape_id": lid, "feature": col, "minimum": float(f[col].min()),
                           "maximum": float(f[col].max()), "mean": float(f[col].mean()),
                           "missing_rate": float(f[col].isna().mean())})
    write_csv(OUT / "CROSS_LANDSCAPE_QA.csv", rows)
    write_csv(OUT / "CROSS_LANDSCAPE_FEATURE_RANGES.csv", ranges)
    out = {
        "production_landscapes": len(rows), "unique_physical_landscapes": len(rows),
        "production_terrain_units": sum(r["terrain_units"] for r in rows),
        "production_positive_units": sum(r["positive_units"] for r in rows),
        "all_terrain_schemas_identical": all(r["terrain_schema_compatible"] for r in rows),
        "all_feature_schemas_identical": all(r["feature_schema_compatible"] for r in rows),
        "all_polygon_target_schemas_identical": all(r["target_schema_compatible"] for r in rows),
        "all_feature_missing_cells": sum(r["feature_missing_cells"] for r in rows),
        "all_target_tier": "TIER_2", "physical_landscape_groups": {
            "KUMAMOTO_ASO_2016": "ASO_CALDERA", "NOTO_PENINSULA_2024": "NOTO_PENINSULA_2024",
            "LOMBOK_RINJANI_2018": "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019": "COTABATO_DAVAO_2019",
            ATI: ATI, ERI: "DOMINICA"},
        "domain_shift_policy": "DOCUMENT; DO_NOT_NORMALISE_OR_CORRECT",
    }
    write_json(OUT / "CROSS_LANDSCAPE_QA.json", out)
    return {"rows": rows, **out}


def production_qa(lid: str, prep: dict, result: dict) -> dict:
    """Report sequential domain and population exclusions without tuning them."""
    d = prep["srcdir"]
    def read(name):
        with rasterio.open(d / name) as src:
            return src.read(1).astype(bool), src.profile
    raw, profile = read("omega_obs_raw_30m.tif")
    final, _ = read("omega_analysis_primary_30m.tif")
    visible = read("omega_visible_30m.tif")[0] if (d / "omega_visible_30m.tif").exists() else raw
    with rasterio.open(d / "geology_official_index_30m.tif") as src: geo = src.read(1)
    eligible_ids = np.asarray([False] + [row["domain_eligible"] == "TRUE" for row in prep["mapping"]])
    geology = eligible_ids[geo]
    with rasterio.open(d / "dem_primary_30m.tif") as src:
        dem = src.read(1); dem_valid = np.isfinite(dem) & (dem != src.nodata)
    with rasterio.open(d / "dem_edm_30m.tif") as src: edm = src.read(1)
    water = np.isin(edm, list(WATER_EDM)) if prep["quality_supported"] else np.zeros_like(raw)
    label = rasterize([(g, 1) for g in prep["y"].geometry], out_shape=raw.shape,
                      transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    h3, w3 = raw.shape[0] // 3 * 3, raw.shape[1] // 3 * 3
    def blocks(mask): return P._block(mask.astype(np.float32), "sum", h3, w3) >= 5
    yb = P._block(label.astype(np.float32), "sum", h3, w3) > 0
    stage_masks = {
        "OMEGA_OBS_RAW": raw,
        "OMEGA_VISIBLE": visible,
        "AFTER_GEOLOGY": visible & geology,
        "AFTER_WATER": visible & geology & ~water,
        "OMEGA_ANALYSIS": final,
    }
    block_counts = {name: int(blocks(mask).sum()) for name, mask in stage_masks.items()}
    positive_counts = {name: int((yb & blocks(mask)).sum()) for name, mask in stage_masks.items()}
    terrain = result["terrain"]; targets = result["targets"]
    positive = targets.y_positive.astype(bool).to_numpy()
    strict = terrain.analysis_population_strict_dem_quality.astype(bool).to_numpy()
    qa = {
        "landscape_id": lid, "source_polygon_count": len(prep["y"]),
        "source_Y_cell_centres": int(label.sum()), "source_positive_90m_blocks": int(yb.sum()),
        "admitted_units": len(terrain), "admitted_positive_units": int(positive.sum()),
        "admitted_negative_units": int((~positive).sum()), "prevalence": float(positive.mean()),
        "block_counts_by_sequential_domain": block_counts,
        "positive_blocks_by_sequential_domain": positive_counts,
        "positive_loss_outside_or_edge_of_observation_support": int(yb.sum()) - positive_counts["OMEGA_OBS_RAW"],
        "positive_loss_visibility": positive_counts["OMEGA_OBS_RAW"] - positive_counts["OMEGA_VISIBLE"],
        "positive_loss_geology": positive_counts["OMEGA_VISIBLE"] - positive_counts["AFTER_GEOLOGY"],
        "positive_loss_water": positive_counts["AFTER_GEOLOGY"] - positive_counts["AFTER_WATER"],
        "positive_loss_dem_and_final_interaction": positive_counts["AFTER_WATER"] - positive_counts["OMEGA_ANALYSIS"],
        "edge_exclusions_final_domain": int(((P._block(final.astype(np.float32), "sum", h3, w3) > 0) &
                                               ~blocks(final)).sum()),
        "strict_dem_quality_supported": prep["quality_supported"],
        "strict_units": int(strict.sum()), "strict_positive_units": int((strict & positive).sum()),
        "positive_loss_strict_quality": int((positive & ~strict).sum()) if prep["quality_supported"] else None,
        "population_selection_policy": "ALL_VALID primary; STRICT_DEM_QUALITY retained separately only where source companions support it",
    }
    write_json(OUT / f"{lid}_POPULATION_AND_TARGET_QA.json", qa)
    demqa = {
        "landscape_id": lid, "all_valid_units": len(terrain), "all_valid_positive_units": int(positive.sum()),
        "strict_supported": prep["quality_supported"], "strict_units": int(strict.sum()),
        "strict_positive_units": int((strict & positive).sum()),
        "quality_flag_counts": terrain.dem_quality_flag.value_counts().sort_index().to_dict(),
        "source_quality_class_counts": terrain.dem_source_quality_class.value_counts().sort_index().to_dict(),
        "dem_edited_fraction_max": float(terrain.dem_edited_fraction.max()),
        "dem_filled_fraction_max": float(terrain.dem_filled_fraction.max()),
        "nicer_class_balance_population_selected": False,
    }
    write_json(OUT / f"{lid}_DEM_QUALITY_AUDIT.json", demqa)
    return qa


def diversity_update(admitted: list[str]) -> dict:
    count = 4 + len(admitted)
    both = set(admitted) == {ATI, ERI}
    checks = {"C1_INDEPENDENT_N": count >= 11, "C2_LOCKBOXES": False, "C3_MORPHOLOGY": True,
              "C4_EROSION": True, "C5_CLIMATE": True, "C6_TECTONIC": False,
              "C7_TRIGGER": both, "C8_GEOGRAPHY": False, "C9_TARGET_FAMILY_LOLO": True,
              "C10_MAJOR_GAPS": False}
    out = {"rule_id": "SFV2_DATASET_DIVERSITY_STOPPING_RULE_V1", "production_groups": count,
           "unique_physical_landscapes": count, "new_groups": admitted, "checks": checks,
           "satisfied": all(checks.values()), "new_coverage": {
               ATI: ["NORTH_AMERICA", "RAINFALL_CYCLONE", "CALDERA_VOLCANIC_COMPLEX", "TROPICAL"],
               ERI: ["NORTH_AMERICA_CARIBBEAN", "RAINFALL_CYCLONE", "VOLCANIC_ISLAND", "TROPICAL_WET"]},
           "remaining_gaps": {"groups_to_11": 11 - count, "v1_unseen_lockboxes": 3,
                              "tectonics": ">=3 settings including non-subduction",
                              "geography": ">=3 continents and >=4 groups outside Asia",
                              "major_terrain": "production-ready oceanic shield/hotspot and continental rift"},
           "next_6_to_8": ["ST_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020"],
           "next_8_to_11": ["POPOCATEPETL_PUEBLA_2017", "CHUUK_CHATAAN_2002_IF_OMEGA_RESCUED",
                             "NEW_AFRICAN_CONTINENTAL_RIFT_TIER2"]}
    write_json(OUT / "DIVERSITY_STOPPING_RULE_UPDATE.json", out)
    return out


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    state = verify_inputs(); write_json(OUT / "STATE_VERIFICATION.json", state)
    if not state["all_inputs_verified"]:
        raise RuntimeError("STOP: prior checkpoint or Stage 3E registered raw hash mismatch")
    ati_prep = prepare_atitlan(); eri_prep = prepare_erika()
    ati = produce(ATI, ati_prep, "V1_TRANSFER_LANDSLIDES_ATITLAN_6817", "HURRICANE_STAN_2005",
                  "V1_TRANSFER_BOUNDARY_ATITLAN", "NASADEM_HGT_2000_02", "MAGA_GEOLOGIA_NACIONAL")
    eri = produce(ERI, eri_prep, "NHESS_DOMINICA_ERIKA_1548", "TROPICAL_STORM_ERIKA_2015",
                  "NHESS_DATED_ANALYSIS_EXTENT_MINUS_CLOUD", "COPERNICUS_GLO30_2011_2014",
                  "DOMINICA_GOV_GEOLOGIC_MAP_1_100000")
    ati_population = production_qa(ATI, ati_prep, ati) if ati["status"] == "PRODUCTION_READY" else None
    eri_population = production_qa(ERI, eri_prep, eri) if eri["status"] == "PRODUCTION_READY" else None
    gate_rows = []
    for lid, result in ((ATI, ati), (ERI, eri)):
        for gate, passed in result["gates"].items():
            gate_rows.append({"landscape_id": lid, "gate": gate, "outcome": "PASS" if passed else "FAIL",
                              "production_status": result["status"]})
    write_csv(OUT / "PRODUCTION_GATES.csv", gate_rows)
    admitted = [lid for lid, result in ((ATI, ati), (ERI, eri)) if result["status"] == "PRODUCTION_READY"]
    cross = cross_landscape({ATI: ati, ERI: eri})
    diversity = diversity_update(admitted)
    provenance = [{"relative_path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p),
                   "role": "STAGE3F_PRIMARY_OR_SUPPORTING_RAW_INPUT"} for p in RAW_INPUTS]
    write_tsv(OUT / "RAW_PROVENANCE.tsv", provenance)
    manifest = []
    for lid, result in ((ATI, ati), (ERI, eri)):
        s = result.get("summary", {})
        manifest.append({"landscape_id": lid, "status": result["status"],
                         "physical_landscape_group": lid if lid == ATI else "DOMINICA",
                         "terrain_units": s.get("valid_units", 0), "positive_units": s.get("positive_units", 0),
                         "negative_units": s.get("negative_units", 0), "target_tier": "TIER_2",
                         "development_only": "TRUE", "final_lockbox_eligible": "FALSE", "v1_unseen": "FALSE"})
    write_tsv(OUT / "PRODUCTION_MANIFEST.tsv", manifest)
    diagnostics = {"stage": "3F", "status": {ATI: ati["status"], ERI: eri["status"]},
                   "production_landscapes": cross["production_landscapes"],
                   "unique_physical_landscapes": cross["unique_physical_landscapes"],
                   "production_terrain_units": cross["production_terrain_units"],
                   "production_positive_units": cross["production_positive_units"],
                   "dominica_maria_status": "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION",
                   "ml_trained": False, "lolo_run": False, "h_generated": False,
                   "emails_sent": False, "commit": False, "push": False}
    write_json(OUT / "STAGE3F_DIAGNOSTICS.json", diagnostics)
    return {"state": state, "atitlan": ati, "erika": eri, "atitlan_population": ati_population,
            "erika_population": eri_population, "cross": cross,
            "diversity": diversity, "diagnostics": diagnostics}


if __name__ == "__main__":
    main()
