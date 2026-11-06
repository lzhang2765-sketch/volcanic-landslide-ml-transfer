"""Stage 2C core harmonisation (deterministic; RAW is read-only; all products go to data/processed/stage2c).

Per landscape: canonical 30 m UTM grid -> DEM (single bilinear reprojection) + Copernicus EDM/FLM masks (nearest) ->
GSJ Seamless geology (tile decode -> legend index -> mode resampling) -> OMEGA_ANALYSIS = official Omega minus cloud,
minus DEM nodata/water, restricted to mapped volcanic geology (each step quantified) -> harmonised Y (source semantics
kept) -> DEM-quality flags -> small terrain-unit construction pilot (no features, no modelling).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import shape
from shapely.ops import unary_union

from . import build_stage2b as b2
from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage2c_geology import CLASSES, VOLCANIC, aso_mapping, seamless_mapping

AS_OF = "2026-10-05"
OUT = ROOT / "outputs" / "stage2c"
DER = ROOT / "data" / "processed" / "stage2c"
RES = 30.0
BUFFER_M = 1500.0
LANDSCAPES = {"KUMAMOTO_ASO_2016": "EPSG:32652", "HOKKAIDO_IBURI_2018": "EPSG:32654", "NOTO_PENINSULA_2024": "EPSG:32653"}
EDM_LEGEND = {0: "Void", 1: "Not edited", 2: "Infill of external elevation data", 3: "Interpolated", 4: "Smoothed", 5: "Airport editing",
              6: "Raised negative elevation", 7: "Flattened", 8: "Ocean", 9: "Lake", 10: "River", 11: "Shoreline", 12: "Morphed", 13: "Shifted"}
FLM_LEGEND = {0: "Void", 1: "Edited (except filled)", 2: "Not edited / not filled (TanDEM-X)", 3: "ASTER", 4: "SRTM90", 5: "SRTM30", 6: "GMTED2010",
              7: "SRTM30plus", 8: "TerraSAR-X radargrammetric DEM", 9: "AW3D30"}
FILL_SOURCE_EPOCH = {3: "ASTER GDEM (2000-2013)", 4: "SRTM (Feb 2000)", 5: "SRTM (Feb 2000)", 6: "GMTED2010 (pre-2010 sources)",
                     7: "SRTM30plus (pre-2010)", 8: "TerraSAR-X radargrammetry (2007-2015 mission period)", 9: "AW3D30 (2006-2011)"}
WATER_EDM = {8, 9, 10}
MERC = 20037508.342789244
UNIT_CELLS = 3  # 90 m terrain unit = 3 x 3 canonical 30 m cells
SOFTWARE = {"python": platform.python_version(), "numpy": np.__version__, "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
            "geopandas": gpd.__version__, "shapely": shapely.__version__, "pandas": pd.__version__}
SCRIPT_SHA = sha256(Path(__file__))
LINEAGE: list[dict] = []


# ------------------------------------------------------------------------------------------------- provenance
def lineage(output: Path, inputs: list[Path], operation: str, params: dict) -> None:
    LINEAGE.append(dict(output=str(output.relative_to(ROOT)), output_sha256=sha256(output),
                        inputs=[{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sorted(set(inputs))],
                        operation=operation, parameters=params, software=SOFTWARE, script="src/sfv2/stage2c_harmonise.py",
                        script_sha256=SCRIPT_SHA, as_of_date=AS_OF))


def _raw(registry, acq) -> Path:
    return ROOT / next(e for e in registry if e["acquisition_id"] == acq)["raw_path"]


def _write_tif(path: Path, arr: np.ndarray, grid: dict, nodata, dtype) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype, crs=grid["crs"],
                       transform=grid["transform"], nodata=nodata, compress="deflate", tiled=True) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _write_geojson(path: Path, gdf: gpd.GeoDataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, driver="GeoJSON", COORDINATE_PRECISION=3)
    return path


# ------------------------------------------------------------------------------------------------- grid
def grid_spec(omega4326, crs: str) -> dict:
    g = gpd.GeoSeries([omega4326], crs="EPSG:4326").to_crs(crs).iloc[0]
    minx, miny, maxx, maxy = g.bounds
    minx = math.floor((minx - BUFFER_M) / RES) * RES
    miny = math.floor((miny - BUFFER_M) / RES) * RES
    maxx = math.ceil((maxx + BUFFER_M) / RES) * RES
    maxy = math.ceil((maxy + BUFFER_M) / RES) * RES
    return dict(crs=crs, transform=from_origin(minx, maxy, RES, RES), width=int((maxx - minx) / RES), height=int((maxy - miny) / RES),
                bounds=(minx, miny, maxx, maxy))


def _reproject_tiles(paths: list[Path], grid: dict, resampling: Resampling, dtype, nodata) -> np.ndarray:
    dst = np.full((grid["height"], grid["width"]), nodata, dtype=dtype)
    for p in paths:
        with rasterio.open(p) as src:
            tmp = np.full_like(dst, nodata)
            reproject(source=rasterio.band(src, 1), destination=tmp, dst_transform=grid["transform"], dst_crs=grid["crs"],
                      dst_nodata=nodata, resampling=resampling, src_nodata=src.nodata)
            fill = (tmp != nodata) & (dst == nodata)
            dst[fill] = tmp[fill]
    return dst


# ------------------------------------------------------------------------------------------------- geology tiles
def legend_for(registry, lid: str) -> list[dict]:
    units = {}
    for e in sorted(registry, key=lambda e: e["acquisition_id"]):
        if e["landscape_id"] == lid and e["data_role"] == "GEOLOGY_LEGEND" and e["acquisition_id"].split("_")[-2] == "EN":
            for u in json.loads((ROOT / e["raw_path"]).read_text()):
                units.setdefault(u["symbol"], u)
        if e["landscape_id"] == lid and e["data_role"] == "GEOLOGY_LEGEND" and e["acquisition_id"].split("_")[-2] == "JA":
            for u in json.loads((ROOT / e["raw_path"]).read_text()):
                units.setdefault("__ja__" + u["symbol"], u)
    en = [u for k, u in units.items() if not k.startswith("__ja__")]
    for u in en:
        ja = units.get("__ja__" + u["symbol"], {})
        u.setdefault("lithology_ja", ja.get("lithology_ja", u.get("lithology_ja", "")))
    return sorted(en, key=lambda u: u["symbol"])


def geology_mosaic(registry, lid: str, legend: list[dict]) -> tuple[np.ndarray, rasterio.Affine, dict]:
    manifest_path = _raw(registry, f"ACQ_{lid.split('_')[0]}_GSJ_TILESET_Z13")
    man = json.loads(manifest_path.read_text())
    colour_to_index, ambiguous = {}, set()
    for i, u in enumerate(legend, start=1):
        c = int(u["value"], 16)
        if c in colour_to_index:
            ambiguous.add(c)
        colour_to_index.setdefault(c, i)
    xs = sorted({t["x"] for t in man["tiles"]})
    ys = sorted({t["y"] for t in man["tiles"]})
    arr = np.zeros(((ys[-1] - ys[0] + 1) * 256, (xs[-1] - xs[0] + 1) * 256), dtype=np.uint16)
    unknown_px = transparent_px = 0
    for t in man["tiles"]:
        p = ROOT / t["file"]
        if sha256(p) != t["sha256"]:
            raise RuntimeError(f"tile checksum mismatch {p}")
        rgba = np.array(Image.open(p).convert("RGBA"))
        code = (rgba[..., 0].astype(np.int64) << 16) | (rgba[..., 1].astype(np.int64) << 8) | rgba[..., 2]
        idx = np.zeros(code.shape, dtype=np.uint16)
        opaque = rgba[..., 3] == 255
        transparent_px += int((~opaque).sum())
        vals, inv = np.unique(code[opaque], return_inverse=True)
        lut = np.array([colour_to_index.get(int(v), 65535) for v in vals], dtype=np.uint16)
        idx[opaque] = lut[inv]
        unknown_px += int((idx == 65535).sum())
        idx[idx == 65535] = 0
        r0, c0 = (t["y"] - ys[0]) * 256, (t["x"] - xs[0]) * 256
        arr[r0:r0 + 256, c0:c0 + 256] = idx
    ts = 2 * MERC / 2 ** man["zoom"]
    transform = from_origin(-MERC + xs[0] * ts, MERC - ys[0] * ts, ts / 256, ts / 256)
    qc = dict(tiles=len(man["tiles"]), pixels=int(arr.size), transparent_pixels=transparent_px, undecodable_pixels=unknown_px,
              ambiguous_colours=len(ambiguous), legend_units=len(legend), manifest=str(manifest_path.relative_to(ROOT)))
    return arr, transform, qc


def geology_validation(registry, lid, legend, arr, transform) -> dict:
    e = next((e for e in registry if e["acquisition_id"] == f"ACQ_{lid.split('_')[0]}_GSJ_POINT_VALIDATION"), None)
    if not e:
        return {"status": "NOT_ACQUIRED"}
    sym_by_index = {i: u["symbol"] for i, u in enumerate(legend, start=1)}
    pts = json.loads((ROOT / e["raw_path"]).read_text())["points"]
    agree = n = 0
    for p in pts:
        x, y = gpd.GeoSeries.from_xy([p["lon"]], [p["lat"]], crs=4326).to_crs(3857).iloc[0].coords[0]
        col, row = ~transform * (x, y)
        r, c = int(row), int(col)
        if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
            continue
        decoded = sym_by_index.get(int(arr[r, c]), "NONE")
        truth = p.get("symbol") or "NONE"
        n += 1
        agree += decoded == truth
    return dict(status="VALIDATED", points=n, agreement=agree, agreement_rate=round(agree / n, 4) if n else None)


# ------------------------------------------------------------------------------------------------- landscape
def harmonise(lid: str, registry: list[dict]) -> dict:
    crs = LANDSCAPES[lid]
    d = DER / lid
    d.mkdir(parents=True, exist_ok=True)
    res = b2.landscape_qc(lid, b2.stage2b_scope(registry))
    omega4326 = res["omega"]
    grid = grid_spec(omega4326, crs)
    gspec = dict(landscape_id=lid, native_crs="EPSG:4326 (Copernicus GLO-30, 1 arcsec)", working_crs=crs, resolution_m=RES,
                 width=grid["width"], height=grid["height"], bounds=[round(v, 1) for v in grid["bounds"]], alignment="origin snapped to 30 m multiples",
                 buffer_m=BUFFER_M, elevation_resampling="bilinear (single reprojection from native)", categorical_resampling="nearest (EDM/FLM/WBM); mode (geology)",
                 vertical_units="metres, EGM2008 geoid", nodata={"dem": -9999.0, "masks": 255, "geology": 0})
    # ---------------- M
    by_name = {Path(e["raw_path"]).name: ROOT / e["raw_path"] for e in registry if e["landscape_id"] == lid}
    tiles = b2.COP_TILES[lid]
    dem_paths = [by_name[f"Copernicus_DSM_COG_10_{t}_DEM.tif"] for t in tiles]
    dem = _reproject_tiles(dem_paths, grid, Resampling.bilinear, np.float32, -9999.0)
    p_dem = _write_tif(d / "dem_cop30_30m.tif", dem, grid, -9999.0, "float32")
    lineage(p_dem, dem_paths, "reproject+mosaic", {"resampling": "bilinear", **{k: str(v) for k, v in gspec.items() if k in ("working_crs", "resolution_m")}})
    masks = {}
    for kind in ("EDM", "FLM", "WBM"):
        paths = [by_name[f"Copernicus_DSM_COG_10_{t}_{kind}.tif"] for t in tiles]
        masks[kind] = _reproject_tiles(paths, grid, Resampling.nearest, np.uint8, 255)
        p = _write_tif(d / f"cop30_{kind.lower()}_30m.tif", masks[kind], grid, 255, "uint8")
        lineage(p, paths, "reproject+mosaic (categorical)", {"resampling": "nearest"})
    # ---------------- G
    legend = legend_for(registry, lid)
    gmap = seamless_mapping(legend)
    garr, gtrans, gqc = geology_mosaic(registry, lid, legend)
    gqc["point_validation"] = geology_validation(registry, lid, legend, garr, gtrans)
    geo = np.zeros((grid["height"], grid["width"]), dtype=np.uint16)
    reproject(source=garr, destination=geo, src_transform=gtrans, src_crs="EPSG:3857", dst_transform=grid["transform"], dst_crs=crs,
              src_nodata=0, dst_nodata=0, resampling=Resampling.mode)
    p_geo = _write_tif(d / "geology_seamless_index_30m.tif", geo, grid, 0, "uint16")
    tile_inputs = [_raw(registry, f"ACQ_{lid.split('_')[0]}_GSJ_TILESET_Z13")] + [ROOT / e["raw_path"] for e in registry
                                                                                   if e["landscape_id"] == lid and e["data_role"] == "GEOLOGY_LEGEND"]
    lineage(p_geo, tile_inputs, "decode tile colours via official legend -> unit index; reproject", {"resampling": "mode", "zoom": 13})
    order = {u["symbol"]: i for i, u in enumerate(legend, start=1)}
    gmap_by_index = {order[m["source_unit_code"]]: m for m in gmap}
    cls_by_index = np.array(["UNKNOWN"] + [gmap_by_index[i]["harmonised_class"] for i in range(1, len(legend) + 1)])
    elig_by_index = np.array([False] + [gmap_by_index[i]["domain_eligible"] == "TRUE" for i in range(1, len(legend) + 1)])
    geo_cls = cls_by_index[geo]
    volcanic_geo = elig_by_index[geo] & (geo > 0)
    geo_uncertain = (geo == 0) | (geo_cls == "UNKNOWN")
    # ---------------- Omega masks (official ranges; never from Y)
    burn = lambda geom: rasterize([(gpd.GeoSeries([geom], crs=4326).to_crs(crs).iloc[0], 1)], out_shape=dem.shape, transform=grid["transform"],
                                  fill=0, dtype="uint8").astype(bool) if geom is not None and not geom.is_empty else np.zeros(dem.shape, bool)
    parts = res["omega_parts"]
    omega_obs = burn(omega4326)
    domains = {}
    variants = {"PRIMARY": omega_obs}
    if lid == "KUMAMOTO_ASO_2016":
        april = burn(unary_union([g for n, g in parts.items() if n.startswith("201604")]))
        july = burn(unary_union([g for n, g in parts.items() if n.startswith("201607")]))
        variants = {"PRIMARY_APRIL_EQ": april, "SENSITIVITY_APRIL_PLUS_JULY": omega_obs}
        domains["april_july_overlap_km2"] = round(float((april & july).sum()) * RES * RES / 1e6, 3)
        domains["july_only_km2"] = round(float((july & ~april).sum()) * RES * RES / 1e6, 3)
    cloud = burn(res["cloud_geom"]) if res.get("cloud_geom") is not None else np.zeros(dem.shape, bool)
    dem_bad = dem == -9999.0
    water = np.isin(masks["EDM"], list(WATER_EDM))
    steps_out = {}
    for name, base in variants.items():
        cur = base.copy()
        steps = [("official_omega_obs", int(cur.sum()))]
        for label, remove in (("cloud_or_visibility_mask", cloud), ("dem_nodata", dem_bad), ("dem_water_ocean_lake_river", water),
                              ("geology_uncertain_or_unmapped", geo_uncertain), ("non_volcanic_or_pre_cenozoic_geology", ~volcanic_geo & ~geo_uncertain)):
            removed = int((cur & remove).sum())
            cur &= ~remove
            steps.append((label, removed))
        steps_out[name] = dict(steps=[dict(step=s, cells=c, km2=round(c * RES * RES / 1e6, 3)) for s, c in steps],
                               analysis_cells=int(cur.sum()), analysis_km2=round(cur.sum() * RES * RES / 1e6, 3),
                               retained_fraction=round(float(cur.sum()) / max(int(base.sum()), 1), 4))
        p = _write_tif(d / f"omega_analysis_{name.lower()}_30m.tif", cur.astype(np.uint8), grid, 255, "uint8")
        lineage(p, [p_dem, p_geo, d / "cop30_edm_30m.tif"] + [ROOT / e["raw_path"] for e in registry if e["landscape_id"] == lid and "OMEGA" in e["data_role"]],
                "OMEGA_ANALYSIS = official Omega - cloud - DEM nodata - water ; restricted to mapped volcanic geology", {"variant": name, "rasterize": "cell-centre"})
        domains[name] = cur
    # ---------------- Y
    y_summary, y_out = harmonise_y(lid, res, crs, grid, domains, masks, d, registry)
    # ---------------- DEM quality over primary analysis domain
    primary = next(iter(domains[k] for k in variants))
    dq = dem_quality(lid, primary, masks, y_out, grid)
    # ---------------- pilot
    pilot = pilot_units(lid, primary, y_out, grid, d)
    # ---------------- geology summary
    obs_cells = int(omega_obs.sum())
    cls_counts = {c: int(((geo_cls == c) & omega_obs & (geo > 0)).sum()) for c in CLASSES}
    geo_summary = dict(omega_cells=obs_cells, geology_mapped_fraction=round(float(((geo > 0) & omega_obs).sum()) / obs_cells, 4),
                       class_fraction_in_omega={c: round(v / obs_cells, 4) for c, v in cls_counts.items() if v},
                       volcanic_units_included=sorted({m["source_unit_code"] for m in gmap if m["domain_eligible"] == "TRUE"}),
                       units_excluded=sorted({m["source_unit_code"] for m in gmap if m["domain_eligible"] == "FALSE"}), tile_qc=gqc)
    return dict(grid=gspec, geology_mapping=gmap, geology=geo_summary, omega_analysis=steps_out,
                omega_extra={k: v for k, v in domains.items() if not isinstance(v, np.ndarray)}, y=y_summary, dem_quality=dq, pilot=pilot,
                geo_cls=geo_cls, grid_obj=grid, primary=primary)


def harmonise_y(lid, res, crs, grid, domains, masks, d, registry):
    raw_y = [ROOT / e["raw_path"] for e in registry if e["landscape_id"] == lid and "Y" in e["data_role"].split("|")]
    out = {}
    primary_name = next(k for k in domains if isinstance(domains[k], np.ndarray))
    if lid == "KUMAMOTO_ASO_2016":
        y = res["y_frame"].copy()
        y["photo_date"] = y["description"].fillna("").str.extract(r"(\d{4}/\d{2}/\d{2}[^<]*)")[0].fillna("").str.strip()
        y["epoch"] = np.where(y["photo_date"].str.startswith("2016/04"), "APRIL_EQ", np.where(y["photo_date"].str.startswith("2016/07"), "JULY", "UNKNOWN"))
        y = y.to_crs(crs)
        y["xy_key"] = y.geometry.apply(lambda g: f"{g.x:.3f},{g.y:.3f}")
        y["duplicate_group_size"] = y.groupby("xy_key")["xy_key"].transform("size")
        y["is_duplicate_drop"] = y.duplicated("xy_key", keep="first")
        april_omega = gpd.GeoSeries([unary_union([g for n, g in res["omega_parts"].items() if n.startswith("201604")])], crs=4326).to_crs(crs).iloc[0]
        july_omega = gpd.GeoSeries([unary_union([g for n, g in res["omega_parts"].items() if n.startswith("201607")])], crs=4326).to_crs(crs).iloc[0]
        y["in_april_range"], y["in_july_range"] = y.within(april_omega), y.within(july_omega)
        y["target_set"] = np.where(y["is_duplicate_drop"], "EXCLUDED_DUPLICATE",
                                   np.where((y["epoch"] == "APRIL_EQ") & y["in_april_range"], "PRIMARY_APRIL_EQ",
                                            np.where(y["epoch"] == "JULY", "SENSITIVITY_JULY", "SENSITIVITY_UNDATED")))
        rows, cols = rasterio.transform.rowcol(grid["transform"], y.geometry.x.values, y.geometry.y.values)
        y["grid_row"], y["grid_col"] = rows, cols
        y["in_primary_analysis"] = [bool(domains[primary_name][r, c]) for r, c in zip(rows, cols)]
        y["edm_code"] = [int(masks["EDM"][r, c]) for r, c in zip(rows, cols)]
        y["representation"] = "POINT_OCCURRENCE_CENTRE"
        y["target_tier"] = "TIER_2"
        keep = y.drop(columns=[c for c in ("description", "timestamp", "begin", "end", "altitudeMode", "tessellate", "extrude", "visibility",
                                          "drawOrder", "icon", "id") if c in y.columns])
        p = _write_geojson(d / "y_harmonised_points.geojson", keep)
        lineage(p, raw_y, "parse photo epoch; flag exact-duplicate locations; assign target sets (no geometry change)", {"duplicate_key": "UTM xy to 1 mm"})
        out = dict(representation="POINT_OCCURRENCE_CENTRE", total=len(y), duplicates_dropped=int(y["is_duplicate_drop"].sum()),
                   epoch_counts=dict(y["epoch"].value_counts().sort_index()), target_set_counts=dict(y["target_set"].value_counts().sort_index()),
                   primary_in_analysis=int(((y["target_set"] == "PRIMARY_APRIL_EQ") & y["in_primary_analysis"]).sum()),
                   undated_in_april_only=int(((y["epoch"] == "UNKNOWN") & y["in_april_range"] & ~y["in_july_range"]).sum()),
                   undated_in_july_range=int(((y["epoch"] == "UNKNOWN") & y["in_july_range"]).sum()))
        return out, y
    y = res["y_frame"].copy()
    invalid = ~y.is_valid
    y["source_geometry_valid"] = ~invalid
    y["geometry"] = y.geometry.make_valid()
    y = y.to_crs(crs)
    y["y_id"] = [f"{lid}_Y{i:05d}" for i in range(len(y))]
    y["representation"] = "POLYGON_AFFECTED_AREA"
    y["target_tier"] = "TIER_2"
    analysis_vec = analysis_polygons(domains[primary_name], grid)
    inter = y.geometry.intersection(analysis_vec)
    y["area_m2"] = y.area
    y["area_in_primary_analysis_m2"] = inter.area
    burned = rasterize([(g, 1) for g in y.geometry], out_shape=domains[primary_name].shape, transform=grid["transform"], fill=0, dtype="uint8").astype(bool)
    keep = y[[c for c in y.columns if not c.startswith("_")]]
    p = _write_geojson(d / "y_harmonised_polygons.geojson", keep)
    lineage(p, raw_y, "make_valid (derived only); reproject; compute area inside OMEGA_ANALYSIS", {"repair": "shapely.make_valid", "crs": crs})
    out = dict(representation="POLYGON_AFFECTED_AREA", total=len(y), invalid_source_geometries_repaired_in_derived=int(invalid.sum()),
               intersecting_primary_analysis=int((y["area_in_primary_analysis_m2"] > 0).sum()),
               area_km2=round(float(y["area_m2"].sum()) / 1e6, 4), area_in_primary_analysis_km2=round(float(y["area_in_primary_analysis_m2"].sum()) / 1e6, 4),
               y_cells_in_primary_analysis=int((burned & domains[primary_name]).sum()))
    y.attrs["burned"] = burned
    return out, y


def analysis_polygons(mask: np.ndarray, grid):
    geoms = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=grid["transform"]) if v == 1]
    return unary_union(geoms) if geoms else shapely.Polygon()


def dem_quality(lid, primary, masks, y, grid) -> dict:
    n = int(primary.sum())
    edm = masks["EDM"][primary]
    flm = masks["FLM"][primary]
    out = dict(analysis_cells=n, edm_fraction={EDM_LEGEND.get(int(k), str(k)): round(int(v) / n, 5) for k, v in zip(*np.unique(edm, return_counts=True))},
               flm_fraction={FLM_LEGEND.get(int(k), str(k)): round(int(v) / n, 5) for k, v in zip(*np.unique(flm, return_counts=True))},
               non_original_fraction=round(float((edm != 1).sum()) / n, 5) if n else None,
               fill_sources_present={FILL_SOURCE_EPOCH[int(k)]: int(v) for k, v in zip(*np.unique(flm, return_counts=True)) if int(k) in FILL_SOURCE_EPOCH})
    if "grid_row" in y.columns:
        sel = y[(y["target_set"] == "PRIMARY_APRIL_EQ") & y["in_primary_analysis"]]
        out["y_on_non_original_cells"] = int((sel["edm_code"] != 1).sum())
        out["y_total_considered"] = len(sel)
    else:
        burned = y.attrs["burned"] & primary
        out["y_cells_on_non_original"] = int((burned & (masks["EDM"] != 1)).sum())
        out["y_cells_total"] = int(burned.sum())
    out["post_event_material_possible"] = False
    out["leakage_assessment"] = ("LOW: every fill source predates the event and editing operations (interpolation/smoothing/water flattening) "
                                 "derive from pre-event data; no post-event elevation source exists in the FLM legend")
    return out


def pilot_units(lid, primary, y, grid, d) -> dict:
    """Construction check only: 5 km x 5 km of 90 m units centred on the analysis-domain centroid (independent of Y)."""
    rows, cols = np.nonzero(primary)
    if not len(rows):
        return {"status": "NO_ANALYSIS_DOMAIN"}
    rc, cc = int(np.median(rows)), int(np.median(cols))
    half = (5000 // int(RES)) // 2
    r0, c0 = (rc - half) // UNIT_CELLS * UNIT_CELLS, (cc - half) // UNIT_CELLS * UNIT_CELLS
    if "grid_row" in y.columns:
        pts = y[y["target_set"] == "PRIMARY_APRIL_EQ"]
        label_grid = np.zeros(primary.shape, bool)
        label_grid[pts["grid_row"].clip(0, primary.shape[0] - 1), pts["grid_col"].clip(0, primary.shape[1] - 1)] = True
        rule = "unit positive if it contains >=1 PRIMARY_APRIL_EQ mapped centre (point-event formulation)"
    else:
        label_grid = y.attrs["burned"]
        rule = "unit positive if >=1 of its 9 cell centres lies inside a Tier-2 affected-area polygon"
    out_rows = []
    for i in range(0, 2 * half, UNIT_CELLS):
        for j in range(0, 2 * half, UNIT_CELLS):
            r, c = r0 + i, c0 + j
            if r < 0 or c < 0 or r + UNIT_CELLS > primary.shape[0] or c + UNIT_CELLS > primary.shape[1]:
                continue
            block = primary[r:r + UNIT_CELLS, c:c + UNIT_CELLS]
            x, yy = rasterio.transform.xy(grid["transform"], r + 1, c + 1)
            out_rows.append(dict(unit_id=f"{lid}_U{r:05d}_{c:05d}", row=r, col=c, centre_x=round(x, 1), centre_y=round(yy, 1),
                                 analysis_cells=int(block.sum()), in_analysis=bool(block.sum() >= 5),
                                 label=int(label_grid[r:r + UNIT_CELLS, c:c + UNIT_CELLS].any()) if block.sum() >= 5 else ""))
    df = pd.DataFrame(out_rows)
    p = d / "pilot_terrain_units_90m.csv"
    df.to_csv(p, index=False, lineterminator="\n")
    lineage(p, [], "pilot unit construction (no features, no modelling)", {"unit_cells": UNIT_CELLS, "window_m": 5000, "label_rule": rule})
    ins = df[df["in_analysis"]]
    return dict(status="CONSTRUCTED", units=len(df), units_in_analysis=len(ins), positive_units=int((ins["label"] == 1).sum()), label_rule=rule,
                unique_ids=bool(df["unit_id"].is_unique))
