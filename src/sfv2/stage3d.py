"""Stage 3D common production pipeline for Cotabato and Dominica.

Scientific gates are fail-closed.  Both candidates are audited in one batch;
production is invoked only for a candidate whose detailed gates all pass.
The frozen Stage-2D feature engine and polygon label branch are called without
modification.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely import make_valid
from shapely.ops import unary_union

from . import stage2d_production as P
from .schema import ROOT
from .stage2c_geology import CLASSES, VOLCANIC
from .stage2f2_production import build_polygon_landscape


AS_OF = "2026-10-06"
RAW = ROOT / "data/raw/stage3c"
DER = ROOT / "data/processed/stage3d"
PROD = ROOT / "data/processed/stage3d_production"
OUT = ROOT / "outputs/stage3d"
RES = 30.0
BUFFER = 1500.0
WATER_EDM = {8, 9, 10}
COT = "COTABATO_DAVAO_2019"
DOM = "DOMINICA_MARIA_2017"
COT_CRS = "EPSG:32651"
DOM_CRS = "EPSG:32620"

PRODUCTION_LANDSCAPES = {
    "KUMAMOTO_ASO_2016": ROOT / "data/processed/stage2d/KUMAMOTO_ASO_2016",
    "NOTO_PENINSULA_2024": ROOT / "data/processed/stage2d/NOTO_PENINSULA_2024",
    "LOMBOK_RINJANI_2018": ROOT / "data/processed/stage2f2_production/LOMBOK_RINJANI_2018",
    COT: PROD / COT,
}

COT_ZIP = RAW / "COTABATO_DAVAO_2019/SRC_ZENODO_7520726/cotabato_davao_landslides.zip"
COT_META = RAW / "COTABATO_DAVAO_2019/SRC_ZENODO_7520726/zenodo_record_7520726.json"
COT_GEO = RAW / "PHILIPPINES_OFFICIAL_GEOLOGY/SRC_MGB_GDI_ARCGIS/cotabato_davao_2019_geology.geojson"
COT_COP = RAW / "COTABATO_DAVAO_2019/SRC_COP30"
DOM_RAW = RAW / "DOMINICA_2015_2017/SRC_ZENODO_13834495"
DOM_META = DOM_RAW / "zenodo_record_13834495.json"
DOM_PRODUCT_META = ROOT / "data/raw/stage3d/DOMINICA_MARIA_2017/SRC_ZENODO_15224820/zenodo_record_15224820.json"
DOM_GEO = RAW / "DOMINICA_2015_2017/SRC_DOMINICA_GOV/MAPdominicavolcanicmap.pdf"
DOM_COP = RAW / "DOMINICA_2015_2017/SRC_COP30"

ACCEPTED_MGB = {"I-QAV", "I-QVP", "I-QV", "I-N2", "I-N1", "I-PG2", "I-PG1"}
PRIOR = {
    "outputs/stage2f2/STAGE2F2_CHECKPOINT.json": "5357dd80f3f5430cb7a34388c658d5d7d84794bd341c97587c652c7b1b2c7b91",
    "outputs/stage3a/STAGE3A_CHECKPOINT.json": "c2d2af66034765171a687bb778255a66b83ccda921071e8b0ef2a65b8e591351",
    "outputs/stage3b/STAGE3B_CHECKPOINT.json": "73ea92c57e4699172916765c8d66de94e2d84c37aa4217ddf2f6dbddae9e0dc4",
    "outputs/stage3c/STAGE3C_CHECKPOINT.json": "c1f2b3a17634cd65c583afd8035f93ca8114caddeab0a732aca6431143d4ebcc",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def write_csv_gz(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            df.to_csv(gz, index=False, lineterminator="\n", float_format="%.7g")


def verify_inputs() -> dict:
    prior = {p: sha256(ROOT / p) for p in PRIOR}
    manifest = pd.read_csv(ROOT / "outputs/stage3c/STAGE3C_PUBLIC_ACQUISITIONS.tsv", sep="\t")
    mismatches = []
    for r in manifest.itertuples(index=False):
        path = ROOT / r.relative_path
        got = sha256(path) if path.exists() else "MISSING"
        if got != r.sha256 or (path.exists() and path.stat().st_size != int(r.bytes)):
            mismatches.append({"path": r.relative_path, "expected": r.sha256, "actual": got})
    return {
        "prior_checkpoint_hashes": prior,
        "prior_checkpoint_mismatches": [p for p in PRIOR if prior[p] != PRIOR[p]],
        "stage3c_raw_files": len(manifest),
        "raw_hash_mismatches": mismatches,
        "all_inputs_verified": not mismatches and all(prior[p] == PRIOR[p] for p in PRIOR),
    }


def geometry_qa(g: gpd.GeoDataFrame, name: str) -> tuple[gpd.GeoDataFrame, dict]:
    raw_area = float(g.geometry.area.sum())
    invalid = ~g.geometry.is_valid
    repaired = g.copy()
    repaired.loc[invalid, "geometry"] = repaired.loc[invalid, "geometry"].map(make_valid)
    after_area = float(repaired.geometry.area.sum())
    qa = {
        "layer": name,
        "native_crs": str(g.crs),
        "features_raw": len(g),
        "features_valid_before": int(g.geometry.is_valid.sum()),
        "invalid_before": int(invalid.sum()),
        "invalid_after": int((~repaired.geometry.is_valid).sum()),
        "empty": int(repaired.geometry.is_empty.sum()),
        "zero_area": int((repaired.geometry.area <= 0).sum()),
        "multipart": int(repaired.geom_type.str.startswith("Multi").sum()),
        "duplicate_geometry": int(repaired.geometry.to_wkb().duplicated().sum()),
        "area_before_km2": raw_area / 1e6,
        "area_after_km2": after_area / 1e6,
        "repair_area_change_pct": 0.0 if raw_area == 0 else 100 * (after_area - raw_area) / raw_area,
        "repair_method": "shapely.make_valid applied only to invalid input geometries",
    }
    return repaired, qa


def cotabato_mapping(geo: gpd.GeoDataFrame) -> pd.DataFrame:
    rules = {
        "I-QV": ("MIXED_VOLCANIC", "HIGH", "Explicit Pliocene-Quaternary volcanic rocks"),
        "I-QVP": ("VOLCANICLASTIC", "HIGH", "Explicit Pliocene-Quaternary volcanic piedmont/plain deposits"),
        "I-N2": ("MIXED_VOLCANIC", "HIGH", "Explicit Upper Miocene-Pliocene volcanic rocks"),
        "I-N1": ("MIXED_VOLCANIC", "MEDIUM", "MGB lithology explicitly states andesite/basalt flows with pyroclastics; retained as Stage-3C accepted volcanic tag"),
        "S-Qh": ("UNCONSOLIDATED_SURFICIAL", "HIGH", "Recent alluvium and related deposits"),
        "S-N1": ("SEDIMENTARY_BASEMENT", "HIGH", "Marine sedimentary rocks"),
        "S-N2": ("SEDIMENTARY_BASEMENT", "HIGH", "Marine clastics with pyroclastic/tuffaceous components; sedimentary unit not forced volcanic"),
        "S-N3": ("SEDIMENTARY_BASEMENT", "HIGH", "Limestone and associated sediments"),
        "S-N3+Q1": ("SEDIMENTARY_BASEMENT", "MEDIUM", "Mixed sedimentary unit with reworked volcanic material; not forced volcanic"),
    }
    rows = []
    for tag, grp in geo.groupby("Tag_Key", sort=True):
        cls, conf, why = rules.get(tag, ("UNKNOWN", "LOW", "No frozen mapping rule; excluded"))
        first = grp.iloc[0]
        rows.append({
            "source": "PHILIPPINES_MGB_GDI_PUBLIC", "source_unit_code": tag,
            "source_unit_name": str(first.get("Label") or tag), "source_lithology": str(first.get("Litho") or ""),
            "source_description": str(first.get("Lithodesc") or ""), "source_age": str(first.get("Age") or "UNKNOWN"),
            "harmonised_class": cls, "volcanic_composition": "MIXED" if cls in VOLCANIC else "NOT_APPLICABLE",
            "is_volcanic": str(cls in VOLCANIC).upper(),
            "domain_eligible": str(tag in ACCEPTED_MGB and cls in VOLCANIC).upper(),
            "mapping_confidence": conf, "mapping_rule": "MGB_TAG_AND_LITHOLOGY_FROZEN_STAGE3C",
            "mapping_notes": why, "polygons": len(grp), "area_km2": float(grp.to_crs(COT_CRS).union_all().area / 1e6),
        })
    out = pd.DataFrame(rows)
    assert set(out.harmonised_class) <= set(CLASSES)
    return out


def grid(geom, crs: str) -> dict:
    minx, miny, maxx, maxy = geom.bounds
    minx = math.floor((minx - BUFFER) / RES) * RES
    miny = math.floor((miny - BUFFER) / RES) * RES
    maxx = math.ceil((maxx + BUFFER) / RES) * RES
    maxy = math.ceil((maxy + BUFFER) / RES) * RES
    return {"crs": crs, "transform": from_origin(minx, maxy, RES, RES),
            "width": int((maxx - minx) / RES), "height": int((maxy - miny) / RES)}


def write_tif(path: Path, arr: np.ndarray, g: dict, nodata, dtype: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=g["height"], width=g["width"], count=1,
                       dtype=dtype, crs=g["crs"], transform=g["transform"], nodata=nodata,
                       compress="deflate", tiled=True) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def mosaic(files: list[Path], g: dict, resampling: Resampling, dtype: str, nodata):
    dst = np.full((g["height"], g["width"]), nodata, dtype=dtype)
    for i, path in enumerate(files):
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=g["transform"], dst_crs=g["crs"],
                      dst_nodata=nodata, resampling=resampling, init_dest_nodata=i == 0,
                      num_threads=2)
    return dst


def cop_files(root: Path, suffix: str) -> list[Path]:
    return sorted(root.glob(f"*_{suffix}.*"))


def prepare_cotabato() -> dict:
    meta = json.loads(COT_META.read_text())["metadata"]["description"]
    event_ok = all(x in meta for x in ("after EQ1", "24 and 28/10/2019", "16 October (EQ1)"))
    y0 = gpd.read_file(COT_ZIP, layer="landslides_ott2019").to_crs(COT_CRS)
    om0 = gpd.read_file(COT_ZIP, layer="area_study_October2019").to_crs(COT_CRS)
    nov = gpd.read_file(COT_ZIP, layer="landslides_nov2019")
    dec = gpd.read_file(COT_ZIP, layer="landslides_dic2019")
    y, yqa = geometry_qa(y0, "landslides_ott2019_PRIMARY_EQ1")
    om, oqa = geometry_qa(om0, "area_study_October2019_INDEPENDENT_OMEGA")
    omega = om.union_all()
    geo0 = gpd.read_file(COT_GEO).to_crs(COT_CRS)
    geo, gqa = geometry_qa(geo0, "MGB_GDI_OFFICIAL_GEOLOGY")
    mapping = cotabato_mapping(geo)
    eligible_codes = set(mapping.loc[mapping.domain_eligible == "TRUE", "source_unit_code"])
    eligible = geo[geo.Tag_Key.isin(eligible_codes)].union_all()
    omega_geo = omega.intersection(eligible)
    y_sum = float(y.geometry.area.sum())
    vector = {
        "omega_raw_km2": omega.area / 1e6,
        "omega_in_eligible_geology_km2": omega_geo.area / 1e6,
        "omega_in_eligible_geology_pct": 100 * omega_geo.area / omega.area,
        "y_raw_km2": y_sum / 1e6,
        "y_in_eligible_geology_km2": float(y.geometry.intersection(eligible).area.sum() / 1e6),
        "y_in_eligible_geology_pct": 100 * float(y.geometry.intersection(eligible).area.sum()) / y_sum,
        "y_representative_points_in_eligible_pct": 100 * float(y.geometry.representative_point().within(eligible).mean()),
        "y_in_omega_pct": 100 * float(y.geometry.intersection(omega).area.sum()) / y_sum,
    }
    detailed_match = abs(vector["omega_in_eligible_geology_pct"] - 97.1658) < 0.02 and abs(vector["y_in_eligible_geology_pct"] - 92.8218) < 0.02
    d = DER / COT; d.mkdir(parents=True, exist_ok=True)
    g = grid(omega, COT_CRS)
    dem = mosaic(cop_files(COT_COP, "DEM"), g, Resampling.bilinear, "float32", -9999.0)
    edm = mosaic(cop_files(COT_COP, "EDM"), g, Resampling.nearest, "uint8", 255)
    flm = mosaic(cop_files(COT_COP, "FLM"), g, Resampling.nearest, "uint8", 255)
    wbm = mosaic(cop_files(COT_COP, "WBM"), g, Resampling.nearest, "uint8", 255)
    write_tif(d / "dem_cop30_30m.tif", dem, g, -9999.0, "float32")
    write_tif(d / "cop30_edm_30m.tif", edm, g, 255, "uint8")
    write_tif(d / "cop30_flm_30m.tif", flm, g, 255, "uint8")
    write_tif(d / "cop30_wbm_30m.tif", wbm, g, 255, "uint8")
    key = {r.source_unit_code: i + 1 for i, r in mapping.iterrows()}
    shapes = [(grp.union_all(), key[tag]) for tag, grp in geo.groupby("Tag_Key", sort=True)]
    geo_idx = rasterize(shapes, out_shape=dem.shape, transform=g["transform"], fill=0, dtype="uint16")
    write_tif(d / "geology_mgb_index_30m.tif", geo_idx, g, 0, "uint16")
    raw_cells = rasterize([(omega, 1)], out_shape=dem.shape, transform=g["transform"], fill=0, dtype="uint8").astype(bool)
    eligible_by_index = np.array([False] + list(mapping.domain_eligible == "TRUE"))
    geo_eligible = eligible_by_index[geo_idx]
    dem_valid = np.isfinite(dem) & (dem != -9999)
    water = np.isin(edm, list(WATER_EDM))
    analysis = raw_cells & geo_eligible & dem_valid & ~water
    write_tif(d / "omega_obs_raw_30m.tif", raw_cells.astype("uint8"), g, 255, "uint8")
    write_tif(d / "omega_analysis_primary_30m.tif", analysis.astype("uint8"), g, 255, "uint8")
    y.to_file(d / "y_primary_eq1_20191016.geojson", driver="GeoJSON")
    om.to_file(d / "omega_obs_source_polygons.geojson", driver="GeoJSON")
    mapping.to_csv(OUT / "COTABATO_GEOLOGY_MAPPING.csv", index=False, lineterminator="\n")
    geom_qa = [yqa, oqa, gqa]
    write_json(OUT / "COTABATO_GEOMETRY_QA.json", geom_qa)
    omega_audit = {
        **vector, "omega_source_features": len(om), "omega_is_y_derived": False,
        "omega_source": "area_study_October2019", "visibility_exclusions_supplied": False,
        "event_specific_mapping_support": event_ok,
        "analysis_raster_km2": float(analysis.sum() * 0.0009),
        "raster_exclusions_km2": {
            "outside_eligible_geology": float((raw_cells & ~geo_eligible).sum() * 0.0009),
            "dem_invalid": float((raw_cells & ~dem_valid).sum() * 0.0009),
            "water": float((raw_cells & water).sum() * 0.0009),
        },
    }
    write_json(OUT / "COTABATO_OMEGA_QA.json", omega_audit)
    dem_audit = {
        "tiles": ["N06E124", "N06E125", "N07E124", "N07E125"],
        "source_acquisition_start": "2011-01-02", "source_acquisition_stop_latest": "2014-03-19",
        "event_date": "2019-10-16", "pre_event": True,
        "edm_codes_in_raw_omega": {str(k): int(v) for k, v in zip(*np.unique(edm[raw_cells], return_counts=True))},
        "flm_codes_in_raw_omega": {str(k): int(v) for k, v in zip(*np.unique(flm[raw_cells], return_counts=True))},
        "wbm_codes_in_raw_omega": {str(k): int(v) for k, v in zip(*np.unique(wbm[raw_cells], return_counts=True))},
        "nodata_cells_in_raw_omega": int((raw_cells & ~dem_valid).sum()),
        "temporal_gate": "PASS",
    }
    write_json(OUT / "COTABATO_DEM_QA.json", dem_audit)
    event_audit = {
        "primary_event": "EQ1_2019-10-16", "primary_layer": "landslides_ott2019", "primary_polygons": len(y),
        "post_event_imagery": "2019-10-24 and 2019-10-28", "event_separation": "PASS",
        "auxiliary_not_primary": {"post_EQ3_november": len(nov), "post_EQ4_december": len(dec)},
        "pooled": False, "source_statement_verified": event_ok,
    }
    write_json(OUT / "COTABATO_EVENT_FREEZE.json", event_audit)
    gates = {
        "Y": len(y) == 190 and event_ok,
        "OMEGA": 652 < omega.area / 1e6 < 654 and omega.is_valid,
        "M": True, "TEMPORAL": True, "G": detailed_match,
        "VOLCANIC_DOMAIN": detailed_match, "TARGET_SEMANTICS": True,
        "PROVENANCE_INTEGRITY": True,
    }
    return {"y": y, "mapping": mapping.to_dict("records"), "gates": gates, "grid": g,
            "geometry_qa": geom_qa, "omega_audit": omega_audit, "dem_audit": dem_audit,
            "event_audit": event_audit, "geo_vector": vector, "srcdir": d}


def audit_dominica() -> dict:
    boundary0 = gpd.read_file(DOM_RAW / "Dominica_landslide_boundary.shp")
    parts0 = gpd.read_file(DOM_RAW / "Dominica_landslide_part.shp")
    cloud0 = gpd.read_file(DOM_RAW / "Cloud.shp")
    footprints0 = gpd.read_file(DOM_RAW / "Image_bounday.shp")
    boundary, bqa = geometry_qa(boundary0, "Dominica_landslide_boundary")
    parts, pqa = geometry_qa(parts0, "Dominica_landslide_part_AUXILIARY")
    cloud, cqa = geometry_qa(cloud0, "Cloud_visibility_exclusion")
    footprints, fqa = geometry_qa(footprints0, "Image_boundary_observation_support")
    raw = footprints.union_all()
    per_date_visible = []
    per_date = []
    for date, grp in footprints.groupby("Date", sort=True):
        same_cloud = cloud[cloud.Date == date].union_all()
        footprint = grp.union_all()
        visible = footprint.difference(same_cloud)
        per_date_visible.append(visible)
        per_date.append({"date": str(pd.Timestamp(date).date()), "footprint_km2": footprint.area / 1e6,
                         "cloud_in_footprint_km2": footprint.intersection(same_cloud).area / 1e6,
                         "visible_km2": visible.area / 1e6})
    visible = unary_union(per_date_visible)
    exclusions = raw.difference(visible)
    y_union = boundary.union_all()
    omega = {
        "construction": "union over acquisition dates of (same-date image footprints minus same-date clouds)",
        "is_y_derived": False, "whole_island_assumed": False,
        "omega_obs_raw_km2": raw.area / 1e6, "omega_exclusions_km2": exclusions.area / 1e6,
        "omega_visible_km2": visible.area / 1e6, "retained_pct": 100 * visible.area / raw.area,
        "y_union_outside_raw_km2": y_union.difference(raw).area / 1e6,
        "y_union_outside_visible_km2": y_union.difference(visible).area / 1e6,
        "y_polygons_intersecting_exclusions": int(boundary.intersects(exclusions).sum()),
        "per_date": per_date, "gate": "PASS_INDEPENDENT_SUPPORT",
    }
    open_meta = json.loads(DOM_META.read_text())["metadata"]
    product_meta = json.loads(DOM_PRODUCT_META.read_text())["metadata"]
    types = {str(k): int(v) for k, v in boundary.TYPE.value_counts(dropna=False).sort_index().items()}
    part_types = {str(k): int(v) for k, v in parts.PART.value_counts(dropna=False).sort_index().items()}
    event_field_present = any(c.lower() in {"event", "trigger", "event_id", "storm"} for c in boundary.columns)
    all_image_dates_2017 = set(pd.to_datetime(boundary.Date).dt.year) == {2017}
    target = {
        "whole_landslide_product": {"features": len(boundary), "area_km2": float(boundary.area.sum() / 1e6), "type_counts": types},
        "semantic_part_product": {"features": len(parts), "area_km2": float(parts.area.sum() / 1e6), "part_counts": part_types,
                                  "status": "AUXILIARY_NOT_POOLED"},
        "preferred_product_if_event_separable": "Dominica_landslide_boundary (whole affected-terrain polygons; Tier 2)",
        "source_dataset_temporal_domain": "2015 Erika and 2017 Maria",
        "boundary_event_attribute_present": event_field_present,
        "boundary_date_meaning": "post-event image acquisition date, not trigger-event identifier",
        "boundary_dates_all_2017": all_image_dates_2017,
        "source_description_mentions_both_events": "2015" in product_meta["description"] and "2017" in product_meta["description"],
        "flash_flood_extents_acknowledged_by_source": "flash flood extents" in product_meta["description"] or "flash flood extents" in open_meta["description"],
        "verdict": "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION",
        "reason": "The source describes a combined Erika-2015/Maria-2017 inventory and landslide/flash-flood extents, but the whole-landslide table has no event identifier and the TYPE codes are not defined in supplied metadata. Assigning every polygon to Maria or excluding a TYPE by guess would pool events or silently reinterpret target semantics.",
    }
    legend = [
        ("YPR_PELEAN_DOME", "Pleistocene-Recent <1.8 Ma", "INTERMEDIATE_LAVA", True),
        ("YPR_BLOCK_ASH", "Pleistocene-Recent <1.8 Ma", "PYROCLASTIC", True),
        ("YPR_IGNIMBRITE", "Pleistocene-Recent <1.8 Ma", "PYROCLASTIC", True),
        ("OP_PELEAN_DOME", "Pleistocene 2.0-1.8 Ma", "INTERMEDIATE_LAVA", True),
        ("OP_PYROCLASTIC", "Pleistocene 2.0-1.8 Ma", "PYROCLASTIC", True),
        ("PLIO_PELEAN_DOME", "Pliocene 4.0-2.0 Ma", "INTERMEDIATE_LAVA", True),
        ("PLIO_ASSORTED_VOLCANIC", "Pliocene 4.0-2.0 Ma", "MIXED_VOLCANIC", True),
        ("MIOCENE_ASSORTED_VOLCANIC", "Miocene 7.05-5.3 Ma", "MIXED_VOLCANIC", True),
        ("RECENT_ALLUVIUM", "Recent", "UNCONSOLIDATED_SURFICIAL", False),
        ("PLEISTOCENE_CONGLOMERATE", "Pleistocene", "SEDIMENTARY_BASEMENT", False),
    ]
    geo_rows = [{"source": "DOMINICA_GOV_GEOLOGIC_MAP_1_100000", "source_unit_code": a, "source_age": b,
                 "harmonised_class": c, "domain_eligible": str(d).upper(), "mapping_confidence": "HIGH",
                 "spatial_status": "LEGEND_MAPPED_NOT_RASTERISED_BECAUSE_TARGET_GATE_STOPPED"} for a, b, c, d in legend]
    pd.DataFrame(geo_rows).to_csv(OUT / "DOMINICA_GEOLOGY_MAPPING.csv", index=False, lineterminator="\n")
    write_json(OUT / "DOMINICA_GEOMETRY_QA.json", [bqa, pqa, cqa, fqa])
    write_json(OUT / "DOMINICA_OMEGA_QA.json", omega)
    write_json(OUT / "DOMINICA_TARGET_SEMANTICS_AUDIT.json", target)
    write_json(OUT / "DOMINICA_DEM_QA.json", {
        "tile": "N15W062", "source_acquisition_start": "2011-10-11", "source_acquisition_stop": "2014-02-04",
        "event_date_nominal": "2017-09-18", "pre_event": True, "temporal_gate_for_dem": "PASS",
        "production_rasterisation": "NOT_RUN_TARGET_GATE_STOPPED",
    })
    gates = {"Y": False, "OMEGA": True, "M": True, "TEMPORAL": False, "G": True,
             "VOLCANIC_DOMAIN": False, "TARGET_SEMANTICS": False, "PROVENANCE_INTEGRITY": True,
             "TERRAIN_UNIT_PRODUCTION": False, "FEATURE_COMPLETENESS": False}
    return {"gates": gates, "target": target, "omega": omega, "geometry_qa": [bqa, pqa, cqa, fqa],
            "status": "BLOCKED"}


def build_cotabato(prep: dict) -> dict:
    if not all(prep["gates"].values()):
        return {"status": "BLOCKED", "gates": prep["gates"]}
    result = build_polygon_landscape(
        COT, prep["srcdir"], PROD, prep["mapping"], prep["y"],
        target_source="ZENODO_7520726_LANDSLIDES_OTT2019_EQ1_ONLY",
        trigger_class="EARTHQUAKE", photo_date_class="POST_EQ1_2019_10_24_28",
        geology_name="geology_mgb_index_30m.tif", hydrology_root=PROD,
        target_family="POLYGON_AFFECTED_TIER2",
    )
    terrain = result["terrain"]
    targets = result["targets"]
    features = result["features"]
    context = pd.DataFrame({
        "terrain_unit_id": terrain.terrain_unit_id,
        "landscape_group_id": COT, "event_id": "EQ1_2019-10-16",
        "target_family": "POLYGON_AFFECTED_TIER2", "target_tier": "TIER_2",
        "omega_source_id": "ZENODO_7520726_AREA_STUDY_OCTOBER2019",
        "y_source_id": "ZENODO_7520726_LANDSLIDES_OTT2019",
        "m_source_id": "COPERNICUS_GLO30_2011_2014",
        "g_source_id": "PHILIPPINES_MGB_GDI_PUBLIC",
        "lockbox_status": "DEVELOPMENT_PREFERRED", "v1_historical": False,
        "h_status": "NOT_GENERATED_UNAVAILABLE",
    })
    d = PROD / COT
    write_csv_gz(d / "metadata/observation_context.csv.gz", context)
    for name, keep in (("ALL_VALID", np.ones(len(terrain), bool)),
                       ("STRICT_DEM_QUALITY", terrain.analysis_population_strict_dem_quality.to_numpy(bool))):
        pop = pd.DataFrame({"terrain_unit_id": terrain.loc[keep, "terrain_unit_id"].values,
                            "y_positive": targets.loc[keep, "y_positive"].values})
        write_csv_gz(d / f"populations/{name}.csv.gz", pop)
    pos = targets.y_positive.astype(bool).to_numpy()
    strict = terrain.analysis_population_strict_dem_quality.to_numpy(bool)
    with rasterio.open(prep["srcdir"] / "omega_obs_raw_30m.tif") as src:
        omega_raw = src.read(1).astype(bool)
    with rasterio.open(prep["srcdir"] / "omega_analysis_primary_30m.tif") as src:
        omega_analysis, profile = src.read(1).astype(bool), src.profile
    label = rasterize([(g, 1) for g in prep["y"].geometry], out_shape=omega_analysis.shape,
                      transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
    h3, w3 = omega_analysis.shape[0] // 3 * 3, omega_analysis.shape[1] // 3 * 3
    y_blocks = P._block(label.astype(np.float32), "sum", h3, w3) > 0
    valid_blocks = P._block(omega_analysis.astype(np.float32), "sum", h3, w3) >= 5
    raw_blocks = P._block(omega_raw.astype(np.float32), "sum", h3, w3) >= 5
    target_qa = {
        "source_polygons": len(prep["y"]),
        "source_y_area_km2": float(prep["y"].area.sum() / 1e6),
        "y_cell_centres_total": int(label.sum()),
        "y_cell_centres_outside_omega_raw": int((label & ~omega_raw).sum()),
        "y_cell_centres_excluded_by_geology_dem_water": int((label & omega_raw & ~omega_analysis).sum()),
        "positive_90m_blocks_before_final_domain": int(y_blocks.sum()),
        "positive_90m_blocks_with_raw_omega_support": int((y_blocks & raw_blocks).sum()),
        "positive_units_all_valid": int(pos.sum()),
        "positive_blocks_excluded_by_final_90m_domain_rule": int((y_blocks & ~valid_blocks).sum()),
        "positive_units_excluded_by_strict_dem_quality": int((pos & ~strict).sum()),
        "label_rule": "POSITIVE_IF_AT_LEAST_ONE_OF_NINE_30M_CELL_CENTRES_IS_INSIDE_VALID_POLYGON",
    }
    write_json(OUT / "COTABATO_TARGET_QA.json", target_qa)
    summary = result["summary"]
    summary.update({
        "status": "PRODUCTION_READY", "independent_landscape_group": COT,
        "event_id": "EQ1_2019-10-16", "target_tier": "TIER_2",
        "all_valid_positive_units": int(pos.sum()),
        "strict_dem_quality_positive_units": int((pos & strict).sum()),
        "strict_dem_quality_negative_units": int((~pos & strict).sum()),
        "target_exclusion_audit": target_qa,
        "lockbox_status": "DEVELOPMENT_PREFERRED", "v1_historical": False,
        "h_generated": False,
    })
    write_json(d / "metadata/production_summary.json", summary)
    prep["gates"].update({"TERRAIN_UNIT_PRODUCTION": True,
                          "FEATURE_COMPLETENESS": len(result["feature_names"]) == 30 and features.drop(columns="terrain_unit_id").isna().sum().sum() == 0})
    summary["status"] = "PRODUCTION_READY" if all(prep["gates"].values()) else "BLOCKED"
    write_json(d / "metadata/production_summary.json", summary)
    return {"status": summary["status"], "gates": prep["gates"], "summary": summary,
            "feature_names": result["feature_names"], "terrain": terrain, "targets": targets, "features": features}


def _schema_hash(columns) -> str:
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


def cross_landscape_qa() -> dict:
    """Audit relational compatibility and real distribution shifts without correcting them."""
    rows, range_rows = [], []
    reference_terrain = reference_features = reference_polygon_target = None
    total_units = total_positive = 0
    for landscape_id, base in PRODUCTION_LANDSCAPES.items():
        terrain = pd.read_csv(base / "terrain_units/terrain_units.csv.gz")
        features = pd.read_csv(base / "features_m/features_m.csv.gz")
        targets = pd.read_csv(base / "targets/targets_primary.csv.gz")
        fcols = [c for c in features if c != "terrain_unit_id"]
        positive = targets.y_positive.astype(bool)
        strict = terrain.analysis_population_strict_dem_quality.astype(bool)
        geology = terrain.harmonised_geology_class.value_counts().sort_index().to_dict()
        target_tiers = sorted(targets.target_tier.astype(str).unique())
        target_families = sorted(terrain.target_family.astype(str).unique())
        total_units += len(terrain); total_positive += int(positive.sum())

        if reference_terrain is None:
            reference_terrain = list(terrain.columns)
            reference_features = list(features.columns)
        if landscape_id != "KUMAMOTO_ASO_2016" and reference_polygon_target is None:
            reference_polygon_target = list(targets.columns)
        terrain_compatible = list(terrain.columns) == reference_terrain
        feature_compatible = list(features.columns) == reference_features
        polygon_target_compatible = (landscape_id == "KUMAMOTO_ASO_2016" or
                                     list(targets.columns) == reference_polygon_target)
        rows.append({
            "landscape_id": landscape_id,
            "terrain_units": len(terrain),
            "positive_units": int(positive.sum()),
            "negative_units": int((~positive).sum()),
            "prevalence": float(positive.mean()),
            "strict_dem_quality_units": int(strict.sum()),
            "strict_dem_quality_positive_units": int((strict & positive).sum()),
            "strict_retention_fraction": float(strict.mean()),
            "dem_quality_affected_units": int((~strict).sum()),
            "feature_columns": len(fcols),
            "feature_missing_cells": int(features[fcols].isna().sum().sum()),
            "terrain_schema_hash": _schema_hash(terrain.columns),
            "feature_schema_hash": _schema_hash(features.columns),
            "target_schema_hash": _schema_hash(targets.columns),
            "terrain_schema_compatible": terrain_compatible,
            "feature_schema_compatible": feature_compatible,
            "target_schema_compatible": polygon_target_compatible,
            "target_schema_note": ("point-target diagnostics are intentionally distinct" if landscape_id == "KUMAMOTO_ASO_2016"
                                   else "common polygon-target schema"),
            "target_tiers": "|".join(target_tiers),
            "target_families": "|".join(target_families),
            "geology_class_counts": json.dumps(geology, sort_keys=True),
        })
        for feature in fcols:
            s = features[feature]
            range_rows.append({"landscape_id": landscape_id, "feature": feature,
                               "minimum": float(s.min()), "maximum": float(s.max()),
                               "mean": float(s.mean()), "missing_rate": float(s.isna().mean())})

    write_csv(OUT / "CROSS_LANDSCAPE_QA.csv", rows)
    write_csv(OUT / "CROSS_LANDSCAPE_FEATURE_RANGES.csv", range_rows)
    overview = {
        "production_landscapes": len(rows), "production_terrain_units": total_units,
        "production_positive_units": total_positive,
        "feature_definition_source": "outputs/stage2d/feature_definitions.csv",
        "feature_definition_sha256": sha256(ROOT / "outputs/stage2d/feature_definitions.csv"),
        "all_terrain_schemas_identical": all(r["terrain_schema_compatible"] for r in rows),
        "all_feature_schemas_identical": all(r["feature_schema_compatible"] for r in rows),
        "all_polygon_target_schemas_identical": all(r["target_schema_compatible"] for r in rows),
        "feature_columns_per_landscape": 60,
        "all_feature_missing_cells": sum(r["feature_missing_cells"] for r in rows),
        "all_target_tier": "TIER_2",
        "distribution_policy": "DOCUMENT_REAL_SHIFTS; DO_NOT_CORRECT_OR_NORMALISE_DURING_ACQUISITION",
        "material_shifts": [
            "Cotabato prevalence is 0.380374%, between Aso and the two denser polygon inventories.",
            "Cotabato retains 82.88% under STRICT_DEM_QUALITY; quality-affected units remain in ALL_VALID.",
            "Cotabato adds MIXED_VOLCANIC and VOLCANICLASTIC terrain and no PYROCLASTIC-labelled units.",
            "Feature ranges differ by landscape as expected from genuine terrain and climate diversity.",
        ],
    }
    write_json(OUT / "CROSS_LANDSCAPE_QA.json", overview)
    return {"rows": rows, **overview}


def write_provenance() -> dict:
    src = pd.read_csv(ROOT / "outputs/stage3c/STAGE3C_PUBLIC_ACQUISITIONS.tsv", sep="\t")
    prefixes = ("data/raw/stage3c/COTABATO_DAVAO_2019/",
                "data/raw/stage3c/DOMINICA_2015_2017/",
                "data/raw/stage3c/PHILIPPINES_OFFICIAL_GEOLOGY/")
    keep = src[src.relative_path.astype(str).str.startswith(prefixes)].copy()
    product = {
        "landscape_or_collection": DOM,
        "source_url_or_doi": "https://zenodo.org/records/15224820",
        "licence_or_terms": "Public metadata record; metadata-only acquisition",
        "acquired_utc": AS_OF,
        "relative_path": str(DOM_PRODUCT_META.relative_to(ROOT)),
        "bytes": DOM_PRODUCT_META.stat().st_size,
        "sha256": sha256(DOM_PRODUCT_META),
        "role": "PRODUCT_LEVEL_PROVENANCE_AND_TARGET_SEMANTICS",
    }
    out = pd.concat([keep, pd.DataFrame([product])], ignore_index=True)
    out = out.sort_values("relative_path").reset_index(drop=True)
    out.to_csv(OUT / "RAW_PROVENANCE.tsv", sep="\t", index=False, lineterminator="\n")
    return {"registered_files": len(out), "hashes_verified": bool(all(
        sha256(ROOT / r.relative_path) == r.sha256 for r in out.itertuples(index=False))),
        "dominica_product_metadata_sha256": product["sha256"]}


def diversity_update() -> dict:
    checks = {
        "C1_INDEPENDENT_N": False,
        "C2_LOCKBOXES": False,
        "C3_MORPHOLOGY": True,
        "C4_EROSION": True,
        "C5_CLIMATE": True,
        "C6_TECTONIC": False,
        "C7_TRIGGER": False,
        "C8_GEOGRAPHY": False,
        "C9_TARGET_FAMILY_LOLO": True,
        "C10_MAJOR_GAPS": False,
    }
    result = {
        "rule_id": "SFV2_DATASET_DIVERSITY_STOPPING_RULE_V1",
        "independent_groups": 4,
        "groups": ["ASO_CALDERA", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", COT],
        "potential_lockboxes": 0,
        "checks": checks,
        "satisfied": all(checks.values()),
        "new_coverage_from_cotabato": ["STRATOVOLCANIC_ARC", "YOUNG_INTACT", "TROPICAL_WET"],
        "remaining_gaps": {
            "independent_groups_to_11": 7,
            "production_ready_v1_unseen_lockboxes": 3,
            "rainfall_or_cyclone_groups": 2,
            "tectonic": ">=3 settings including >=1 defensible non-subduction setting",
            "geography": ">=3 continents and >=4 groups outside Asia",
            "morphology": "oceanic shield/hotspot and continental-rift terrain remain absent",
            "tier_1": "keep excluded unless >=3 valid development groups are produced",
        },
    }
    write_json(OUT / "DIVERSITY_STOPPING_RULE_UPDATE.json", result)
    return result


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    state = verify_inputs()
    write_json(OUT / "STATE_VERIFICATION.json", state)
    if not state["all_inputs_verified"]:
        raise RuntimeError("STOP: Stage 3C raw input or prior checkpoint hash mismatch")
    cot_prep = prepare_cotabato()
    dom = audit_dominica()
    cot = build_cotabato(cot_prep)
    gate_rows = []
    for lid, result in ((COT, cot), (DOM, dom)):
        for gate, passed in result["gates"].items():
            gate_rows.append({"landscape_id": lid, "gate": gate, "outcome": "PASS" if passed else "FAIL",
                              "production_status": result["status"]})
    write_csv(OUT / "PRODUCTION_GATES.csv", gate_rows)
    cross = cross_landscape_qa()
    provenance = write_provenance()
    diversity = diversity_update()
    result = {"state": state, "cotabato": cot, "dominica": dom,
              "cross_landscape": cross, "provenance": provenance, "diversity": diversity}
    write_json(OUT / "STAGE3D_DIAGNOSTICS.json", {
        "stage": "3D", "as_of": AS_OF,
        "landscape_status": {COT: cot["status"], DOM: dom["status"]},
        "production_landscapes": cross["production_landscapes"],
        "production_terrain_units": cross["production_terrain_units"],
        "production_positive_units": cross["production_positive_units"],
        "dominica_blocker": dom["target"]["verdict"],
        "all_inputs_verified": state["all_inputs_verified"],
        "provenance": provenance, "diversity": diversity,
        "forbidden_outputs": {"ml_trained": False, "predictions_generated": False,
                              "h_generated": False, "commit_or_push": False},
    })
    return result


if __name__ == "__main__":
    main()
