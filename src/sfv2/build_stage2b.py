"""Stage 2B: verify the immutable raw store and run non-destructive raw/spatial QC (no harmonisation, no features, no ML).

Deterministic: reads data/raw/stage2b + metadata/stage2b_raw_registry.json, extracts archive members into the disposable
QC workspace data/interim/_stage2b_qc (never into RAW), and writes outputs/stage2b.  Volatile repository state is written
to STAGE2B_DIAGNOSTICS.json, which is non-authoritative and excluded from checkpoint identity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import rasterio.mask
from shapely import polygonize
from shapely.ops import unary_union

from .raw_store import REGISTRY, REQUIRED, load_registry, sha256
from .schema import ROOT
from .stage2b_governance import (E_ALLOWLIST, KURIKOMA_EXCLUDED, KURIKOMA_NOTES, KURIKOMA_ROLE, KURIKOMA_V1_PREFIX, STAGE2A2_ERRATUM,
                                 WEIGHTS, request_rows)
from .stage2b_plan import COP_TILES, DATE, EVENT_DATES, FIXED, ORDER, cop_entries

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs" / "stage2b"
QCW = ROOT / "data" / "interim" / "_stage2b_qc"
EQUAL_AREA = "EPSG:6933"
V1_MANIFEST = ROOT.parent / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FILE_MANIFEST.tsv"
PRIOR = ["outputs/screening/STAGE_1_SCREENING_CHECKPOINT.json", "outputs/stage2a/STAGE2A_CHECKPOINT.json",
         "outputs/stage2a_extension/STAGE2A_EXTENSION_CHECKPOINT.json", "outputs/stage2a2/STAGE2A2_CHECKPOINT.json"]
# Archive members by role, matched on decoded names (GSI archives use cp932 names).
MEMBERS = {
    "KUMAMOTO_ASO_2016": {"Y": [r"20160728_houkaichi\.kml$"], "Y_DUPLICATE_SYMBOL_LAYER": [r"houkaichi_kigou\.kml$"],
                          "OMEGA": [r"handokuhani\.kml$"], "CLOUD": []},
    "HOKKAIDO_IBURI_2018": {"Y": [r"斜面崩壊・堆積範囲\.geojson$"], "OMEGA": [r"(?<!未)判読範囲\.geojson$"], "CLOUD": [r"雲による未判読範囲\.geojson$"]},
    "NOTO_PENINSULA_2024": {"Y": [r"hokai.*\.geojson$"], "OMEGA": [r"判読範囲.*\.geojson$"], "CLOUD": [r"cloud_poly.*\.geojson$"]},
}
TIER = {"KUMAMOTO_ASO_2016": ("TIER_2", "Occurrence-centre points of combined slope failure/landslide/debris-flow areas; GSI: centres, not shapes or counts"),
        "HOKKAIDO_IBURI_2018": ("TIER_2", "GSI: bare ground from failure, transport and deposition plotted as one polygon"),
        "NOTO_PENINSULA_2024": ("TIER_2", "GSI: slope-failure and sediment-deposition areas combined")}


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)


def _canon(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


# ------------------------------------------------------------------------------------------------- registry
STAGE2B_IDS = {e["acquisition_id"] for e in FIXED} | {e["acquisition_id"] for lid in ORDER for e in cop_entries(lid)}


def stage2b_scope(registry: list[dict]) -> list[dict]:
    """The registry is shared and append-only; Stage 2B owns exactly its planned acquisitions (Stage 2C fix)."""
    return [e for e in registry if e["acquisition_id"] in STAGE2B_IDS]


def scope_digest(entries: list[dict]) -> str:
    return _sha_text(json.dumps(sorted(entries, key=lambda e: e["acquisition_id"]), sort_keys=True, ensure_ascii=False))


def verify_registry(registry: list[dict]) -> list[str]:
    errors = []
    ids = [e["acquisition_id"] for e in registry]
    if len(ids) != len(set(ids)):
        errors.append("duplicate acquisition IDs")
    for e in registry:
        missing = [k for k in REQUIRED if e.get(k) in (None, "")]
        if missing:
            errors.append(f"{e['acquisition_id']}: missing provenance {missing}")
        p = ROOT / e["raw_path"]
        if not p.exists():
            errors.append(f"{e['acquisition_id']}: raw file missing")
            continue
        if sha256(p) != e["sha256"] or p.stat().st_size != e["actual_bytes"]:
            errors.append(f"{e['acquisition_id']}: checksum/size mismatch")
        if p.stat().st_mode & 0o222:
            errors.append(f"{e['acquisition_id']}: raw file is writable (immutability violated)")
        if not e["raw_path"].startswith(f"data/raw/stage2b/{e['landscape_id']}/{e['source_id']}/"):
            errors.append(f"{e['acquisition_id']}: raw path outside canonical layout")
    return errors


# ------------------------------------------------------------------------------------------------- extraction / vector QC
def _members(zip_path: Path, lid: str, tag: str) -> list[tuple[str, Path]]:
    """Copy archive members to the QC workspace with ASCII names; returns (decoded_name, workspace_path)."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for k, info in enumerate(z.infolist()):
            if info.is_dir():
                continue
            name = info.filename
            try:
                name = name.encode("cp437").decode("cp932")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            suffix = Path(name).suffix.lower()
            dest = QCW / lid / tag / f"m{k:03d}{suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
            out.append((name, dest))
    return out


def _read(path: Path) -> gpd.GeoDataFrame:
    layers = pyogrio.list_layers(path)[:, 0]
    frames = [gpd.read_file(path, layer=layer) for layer in layers]
    gdf = frames[0] if len(frames) == 1 else gpd.GeoDataFrame(np.concatenate([f.values for f in frames]), columns=frames[0].columns, crs=frames[0].crs)
    return gdf


def vector_qc(gdf: gpd.GeoDataFrame, label: str) -> dict:
    wkb = gdf.geometry.apply(lambda g: g.wkb_hex if g is not None else "")
    rec = dict(layer=label, feature_count=len(gdf), geometry_types="|".join(f"{k}:{v}" for k, v in sorted(gdf.geom_type.value_counts().items())),
               crs=str(gdf.crs), bounds="|".join(f"{v:.5f}" for v in gdf.total_bounds) if len(gdf) else "",
               invalid_geometries=int((~gdf.is_valid).sum()), empty_geometries=int(gdf.geometry.is_empty.sum() + gdf.geometry.isna().sum()),
               duplicate_geometries=int(wkb.duplicated().sum()), attribute_schema="|".join(c for c in gdf.columns if c != "geometry"))
    if len(gdf) and gdf.geom_type.str.contains("Polygon").any():
        rec["area_km2"] = round(float(gdf.to_crs(EQUAL_AREA).area.sum() / 1e6), 4)
    return rec


def _omega_union(omega: gpd.GeoDataFrame):
    geoms = list(omega.geometry)
    if all(g.geom_type in ("LineString", "MultiLineString") for g in geoms):
        polys = list(polygonize(geoms).geoms)
        closed = all(g.is_closed for g in geoms if g.geom_type == "LineString")
        return unary_union(polys) if polys else None, f"LINE_BOUNDARY_POLYGONISED_IN_QC_ONLY(closed={closed})"
    return unary_union([g for g in geoms if g.geom_type in ("Polygon", "MultiPolygon")]), "POLYGON"


def _area(geom) -> float:
    return float(gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(EQUAL_AREA).area.iloc[0] / 1e6) if geom is not None and not geom.is_empty else 0.0


# ------------------------------------------------------------------------------------------------- DEM QC
def _xml_epoch(xml: Path) -> dict:
    t = xml.read_text(errors="ignore")
    grab = lambda tag: (re.search(rf"<{tag}>([^<]+)</{tag}>", t) or [None, ""])[1]
    return dict(tandemx_start=grab("tsxx:tsxx_startTime")[:10], tandemx_stop=grab("tsxx:tsxx_stopTime")[:10],
                generation_time=grab("tsxx:generationTime")[:10], date_of_analysis=grab("tsxx:DateofAnalysis")[:10])


def _src_dates(kml: Path) -> tuple[str, str, int]:
    text = re.sub(r"<[^>]+>", " ", kml.read_text(errors="ignore"))
    dates = sorted(set(re.findall(r"Acquisition Date:\s*(\d{4}-\d{2}-\d{2})", text)))
    return (dates[0], dates[-1], len(dates)) if dates else ("", "", 0)


def _masked(path: Path, geom) -> np.ndarray:
    with rasterio.open(path) as r:
        if not gpd.GeoSeries([geom], crs="EPSG:4326").intersects(gpd.GeoSeries.from_xy([0], [0]).envelope.set_crs(4326).iloc[0]).iloc[0] and \
                not geom.intersects(_tile_box(r)):
            return np.array([])
        data, _ = rasterio.mask.mask(r, [geom.intersection(_tile_box(r))], crop=True, filled=False)
        return data[0].compressed()


def _tile_box(r):
    from shapely.geometry import box
    return box(*r.bounds)


def dem_qc(lid: str, registry: list[dict], omega) -> tuple[list[dict], dict]:
    rows, event = [], EVENT_DATES[lid].split("/")[0]
    by_file = {Path(e["raw_path"]).name: e for e in registry if e["landscape_id"] == lid}
    covered, nodata_px, total_px, fill_counts, edit_counts = [], 0, 0, {}, {}
    for t in COP_TILES[lid]:
        dem = ROOT / by_file[f"Copernicus_DSM_COG_10_{t}_DEM.tif"]["raw_path"]
        xml = ROOT / by_file[f"Copernicus_DSM_10_{t}.xml"]["raw_path"]
        src = ROOT / by_file[f"Copernicus_DSM_10_{t}_SRC.kml"]["raw_path"]
        flm = ROOT / by_file[f"Copernicus_DSM_COG_10_{t}_FLM.tif"]["raw_path"]
        edm = ROOT / by_file[f"Copernicus_DSM_COG_10_{t}_EDM.tif"]["raw_path"]
        ep, (s0, s1, ns) = _xml_epoch(xml), _src_dates(src)
        with rasterio.open(dem) as r:
            meta = dict(crs=str(r.crs), width=r.width, height=r.height, res_deg=f"{r.res[0]:.10f}x{r.res[1]:.10f}", nodata=str(r.nodata),
                        dtype=r.dtypes[0], band_count=r.count, bounds="|".join(f"{v:.5f}" for v in r.bounds))
            covered.append(_tile_box(r))
        part = omega.intersection(covered[-1]) if omega is not None else None
        if part is not None and not part.is_empty:
            with rasterio.open(dem) as r:
                d, _ = rasterio.mask.mask(r, [part], crop=True, filled=False)
                arr = d[0]
                total_px += int((~arr.mask).sum())
                nod = np.isnan(arr.data) if r.nodata is None or np.isnan(r.nodata) else (arr.data == r.nodata)
                nodata_px += int((nod & ~arr.mask).sum())
            for p, store in ((flm, fill_counts), (edm, edit_counts)):
                with rasterio.open(p) as r:
                    d, _ = rasterio.mask.mask(r, [part], crop=True, filled=False)
                    vals, cnt = np.unique(d[0].compressed(), return_counts=True)
                    for v, c in zip(vals.tolist(), cnt.tolist()):
                        store[str(v)] = store.get(str(v), 0) + c
        pre = bool(ep["tandemx_stop"]) and ep["tandemx_stop"] < event and (not s1 or s1 < event)
        rows.append(dict(landscape_id=lid, tile=t, dem_product="Copernicus DEM GLO-30 (DGED COG)", provider="ESA/Airbus via AWS Open Data",
                         nominal_resolution="1 arcsec (~30 m)", horizontal_datum="WGS84 (EPSG:4326)", vertical_datum="EGM2008 geoid heights (product specification)",
                         tandemx_start=ep["tandemx_start"], tandemx_stop=ep["tandemx_stop"], src_scene_first=s0, src_scene_last=s1, src_scene_dates=ns,
                         editing_date_of_analysis=ep["date_of_analysis"], editing_after_event="TRUE" if ep["date_of_analysis"] >= event else "FALSE",
                         event_date=event, pre_event="TRUE" if pre else "FALSE",
                         leakage_risk="LOW" if pre else "HIGH_OR_UNVERIFIED", sha256=by_file[dem.name]["sha256"], **meta))
    union_tiles = unary_union(covered)
    summary = dict(omega_area_km2=round(_area(omega), 3), omega_area_covered_by_tiles_km2=round(_area(omega.intersection(union_tiles)), 3) if omega is not None else 0,
                   omega_dem_valid_pixels=total_px, omega_dem_nodata_pixels=nodata_px, flm_value_counts_in_omega=fill_counts,
                   edm_value_counts_in_omega=edit_counts, all_tiles_pre_event=all(r["pre_event"] == "TRUE" for r in rows))
    summary["omega_dem_coverage_fraction"] = round(summary["omega_area_covered_by_tiles_km2"] / summary["omega_area_km2"], 6) if summary["omega_area_km2"] else 0
    return rows, summary


# ------------------------------------------------------------------------------------------------- landscape QC
def landscape_qc(lid: str, registry: list[dict]) -> dict:
    yo = [e for e in registry if e["landscape_id"] == lid and e["data_role"] == "Y|OMEGA"]
    roles = {k: [] for k in MEMBERS[lid]}
    omega_parts = {}
    vrows, notes = [], []
    for e in sorted(yo, key=lambda e: e["acquisition_id"]):
        for name, path in _members(ROOT / e["raw_path"], lid, e["acquisition_id"]):
            if path.suffix == ".txt":
                continue
            for role, pats in MEMBERS[lid].items():
                if any(re.search(p, name) for p in pats):
                    gdf = _read(path)
                    roles[role].append(gdf)
                    if role == "OMEGA":
                        omega_parts[name] = _omega_union(gdf.to_crs("EPSG:4326"))[0]
                    vrows.append(dict(landscape_id=lid, acquisition_id=e["acquisition_id"], member=name, role=role, **vector_qc(gdf, role)))
    cat = lambda fs: gpd.GeoDataFrame(np.concatenate([f.values for f in fs]) if fs else [], columns=fs[0].columns if fs else ["geometry"],
                                      crs="EPSG:4326") if fs else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    y = gpd.GeoDataFrame(geometry=list(np.concatenate([f.geometry.values for f in roles["Y"]])), crs="EPSG:4326")
    omega_df = gpd.GeoDataFrame(geometry=list(np.concatenate([f.geometry.values for f in roles["OMEGA"]])), crs="EPSG:4326")
    cloud = unary_union(list(np.concatenate([f.geometry.values for f in roles["CLOUD"]]))) if roles["CLOUD"] else None
    omega, omega_kind = _omega_union(omega_df)
    is_point = y.geom_type.str.contains("Point").all()
    inside = y.geometry.within(omega) if is_point else y.geometry.representative_point().within(omega)
    sc = dict(landscape_id=lid, y_features=len(y), y_geometry="POINT" if is_point else "POLYGON", omega_geometry_source=omega_kind,
              omega_area_km2=round(_area(omega), 3), y_outside_omega_count=int((~inside).sum()))
    if not is_point:
        ya = unary_union(list(y.geometry.buffer(0)))
        sc.update(y_area_km2=round(_area(ya), 4), y_area_outside_omega_km2=round(_area(ya.difference(omega)), 4))
    if cloud is not None:
        sc.update(cloud_area_km2=round(_area(cloud), 4), cloud_area_inside_omega_km2=round(_area(cloud.intersection(omega)), 4),
                  y_intersecting_cloud_count=int(y.geometry.intersects(cloud).sum()))
    if lid == "KUMAMOTO_ASO_2016":
        ydf = roles["Y"][0]
        dup = roles["Y_DUPLICATE_SYMBOL_LAYER"][0]
        sc["symbol_layer_identical_geometry"] = bool(sorted(ydf.geometry.apply(lambda g: g.wkb_hex)) == sorted(dup.geometry.apply(lambda g: g.wkb_hex)))
        sc["y_class_counts"] = dict(ydf["Name"].value_counts().sort_index())
        sc["y_photo_window_counts"] = dict(ydf["description"].str.extract(r"(\d{4}/\d{2}/\d{2}[^<]*)")[0].fillna("UNKNOWN").value_counts().sort_index())
        notes.append("July 2016 photography postdates the 20-21 June 2016 heavy rain; July-interpreted points may include rainfall-triggered failures (trigger-mixture flag).")
    dem_rows, dem_summary = dem_qc(lid, registry, omega)
    sc.update({f"dem_{k}": v for k, v in dem_summary.items()})
    tier, tier_evidence = TIER[lid]
    y_frame = gpd.GeoDataFrame(pd.concat([f.to_crs("EPSG:4326") for f in roles["Y"]], ignore_index=True), crs="EPSG:4326")
    return dict(vector_rows=vrows, consistency=sc, dem_rows=dem_rows, tier=tier, tier_evidence=tier_evidence, notes=notes, omega=omega,
                omega_parts=omega_parts, y_frame=y_frame, cloud_geom=cloud)


def geology_qc(registry: list[dict], omega, omega_parts: dict) -> tuple[list[dict], dict]:
    e = next(e for e in registry if e["acquisition_id"] == "ACQ_KUMAMOTO_G_H")
    rows, summary = [], {}
    members = _members(ROOT / e["raw_path"], "KUMAMOTO_ASO_2016", "geology")
    shp_dir = QCW / "KUMAMOTO_ASO_2016" / "geology_named"
    if shp_dir.exists():
        shutil.rmtree(shp_dir)
    shp_dir.mkdir(parents=True)
    for name, path in members:  # shapefiles need sibling names; rebuild with original basenames in the QC workspace
        shutil.copy(path, shp_dir / Path(name).name)
    for shp in sorted(shp_dir.glob("*.shp")):
        g = gpd.read_file(shp)
        rows.append(dict(landscape_id="KUMAMOTO_ASO_2016", acquisition_id=e["acquisition_id"], member=shp.name, role="GEOLOGY", **vector_qc(g, shp.stem)))
        if shp.stem == "geo_A":
            g4 = g.to_crs(4326)
            union = unary_union(list(g4.geometry.buffer(0)))
            summary = dict(geology_polygons=len(g), geology_crs=str(g.crs), geology_columns="|".join(c for c in g.columns if c != "geometry"),
                           omega_area_covered_by_geology_km2=round(_area(omega.intersection(union)), 3), omega_area_km2=round(_area(omega), 3))
            summary["omega_geology_coverage_fraction"] = round(summary["omega_area_covered_by_geology_km2"] / summary["omega_area_km2"], 4)
            summary["coverage_by_interpretation_range"] = {n: round(_area(part.intersection(union)) / _area(part), 4) for n, part in sorted(omega_parts.items())}
    readme = next((p for n, p in members if n.endswith(".txt")), None)
    summary["readme_present"] = readme is not None
    summary["legend_workbook_present"] = any(n.endswith(".xlsx") for n, _ in members)
    return rows, summary


# ------------------------------------------------------------------------------------------------- Kurikoma transfer manifest (prepared only)
def kurikoma_manifest() -> tuple[list[dict], list[dict]]:
    with V1_MANIFEST.open(newline="") as f:
        v1 = [r for r in csv.DictReader(f, delimiter="\t") if "kurikoma" in r["path"].lower()]
    eligible, excluded = [], []
    for r in v1:
        if r["path"].startswith(KURIKOMA_V1_PREFIX):
            sub = r["path"][len(KURIKOMA_V1_PREFIX):]
            role = next(v for k, v in KURIKOMA_ROLE.items() if sub.startswith(k))
            current = sha256(ROOT.parent / r["path"])  # read-only verification
            eligible.append(dict(v1_path=r["path"], v1_sha256=r["sha256"], v1_bytes=r["size_bytes"], current_sha256_verified=current,
                                 v1_hash_match="TRUE" if current == r["sha256"] else "FALSE", v1_stage=r["stage"], v1_role=r["role"],
                                 v2_role=role, v2_destination=f"data/raw/stage2b/KURIKOMA_2008/V1_TRANSFER/{role}/{Path(sub).name}",
                                 transfer_method="COPY_VERIFIED: hash source -> copy -> hash destination -> chmod 0444 -> register",
                                 transfer_status="PREPARED_NOT_EXECUTED", transfer_timestamp_utc="", v2_unseen_test_eligibility="INELIGIBLE_V1_HISTORICAL",
                                 scientific_role="Development/diagnostics only after authorisation; never an unseen V2 final test or lockbox"))
        else:
            reason = next((v for k, v in KURIKOMA_EXCLUDED.items() if r["path"].startswith(k)), "Not a raw input")
            excluded.append(dict(v1_path=r["path"], v1_sha256=r["sha256"], v1_role=r["role"], exclusion_reason=reason))
    return eligible, excluded


# ------------------------------------------------------------------------------------------------- main
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = stage2b_scope(load_registry())
    errors = verify_registry(registry)
    if errors:
        raise RuntimeError("; ".join(errors))
    results, vector_rows, dem_rows, consistency = {}, [], [], []
    for lid in ORDER:
        res = landscape_qc(lid, registry)
        results[lid] = res
        vector_rows += res["vector_rows"]
        dem_rows += res["dem_rows"]
        consistency.append(res["consistency"])
    geo_rows, geo_summary = geology_qc(registry, results["KUMAMOTO_ASO_2016"]["omega"], results["KUMAMOTO_ASO_2016"]["omega_parts"])
    vector_rows += geo_rows
    raster_rows = []
    for e in registry:
        if e["raw_path"].endswith(".tif"):
            with rasterio.open(ROOT / e["raw_path"]) as r:
                raster_rows.append(dict(landscape_id=e["landscape_id"], acquisition_id=e["acquisition_id"], file=Path(e["raw_path"]).name, readable="TRUE",
                                        crs=str(r.crs), width=r.width, height=r.height, res_deg=f"{r.res[0]:.10f}", nodata=str(r.nodata), dtype=r.dtypes[0],
                                        band_count=r.count, bounds="|".join(f"{v:.5f}" for v in r.bounds), sha256=e["sha256"]))
    # acquisition manifest (registry + QC-derived CRS/verification)
    crs_by_acq = {}
    for v in vector_rows:
        crs_by_acq.setdefault(v["acquisition_id"], set()).add(v["crs"])
    for r in raster_rows:
        crs_by_acq.setdefault(r["acquisition_id"], set()).add(r["crs"])
    manifest = []
    for e in sorted(registry, key=lambda e: (ORDER.index(e["landscape_id"]), e["acquisition_id"])):
        manifest.append(dict(e, crs="|".join(sorted(crs_by_acq.get(e["acquisition_id"], {"N/A"}))), checksum_algorithm="SHA256",
                             qc_verification_status="RAW_VERIFIED_QC_PASSED"))
    man_cols = ["landscape_id", "acquisition_id", "source_id", "data_role", "dataset_name", "original_filename", "source_url", "download_url", "publisher",
                "access_date", "licence", "expected_format", "actual_format", "expected_size", "actual_bytes", "checksum_algorithm", "sha256",
                "acquisition_timestamp_utc", "temporal_coverage", "spatial_coverage", "resolution", "crs", "raw_path", "verification_status",
                "qc_verification_status", "notes"]
    _write_csv(OUT / "acquisition_manifest.csv", manifest, man_cols)
    _write_csv(OUT / "raw_qc_vectors.csv", vector_rows, ["landscape_id", "acquisition_id", "member", "role", "layer", "feature_count", "geometry_types",
                                                         "crs", "bounds", "invalid_geometries", "empty_geometries", "duplicate_geometries", "area_km2",
                                                         "attribute_schema"])
    _write_csv(OUT / "raw_qc_rasters.csv", raster_rows)
    _write_csv(OUT / "dem_temporal_audit.csv", dem_rows)
    _canon(OUT / "spatial_consistency.json", {"landscapes": consistency, "kumamoto_geology": geo_summary})
    _write_csv(OUT / "e_allowlist_proposal.csv", [dict(zip(["variable", "purpose", "dataset", "version", "resolution", "period", "licence",
                                                             "leakage_concern", "consistent_all_landscapes", "recommendation"], x)) for x in E_ALLOWLIST])
    req = request_rows()
    _write_csv(OUT / "external_request_queue.csv", req, ["rank", "landscape_id", "custodian", "exact_request", *WEIGHTS, "weighted_score", "note", "contacted"])
    eligible, excluded = kurikoma_manifest()
    _write_csv(OUT / "kurikoma_transfer_manifest.csv", eligible)
    _write_csv(OUT / "kurikoma_transfer_exclusions.csv", excluded)
    transient = json.loads((ROOT / "outputs/stage2a2/transient_inspection_manifest.json").read_text())
    recorded = {t["url"]: t["sha256"] for t in transient}
    reg_by_url = {e["download_url"]: e["sha256"] for e in registry}
    erratum = [dict(url=u, stage2a2_recorded_sha256=bad, corrected_sha256=good, recorded_value_matches_stage2a2=str(recorded.get(u) == bad).upper(),
                    confirmed_by_stage2b_redownload=str(reg_by_url.get(u) == good).upper() if u in reg_by_url else "NOT_REDOWNLOADED")
               for u, bad, good in STAGE2A2_ERRATUM]
    _write_csv(OUT / "stage2a2_transient_hash_erratum.csv", erratum)

    status = {}
    for lid in ORDER:
        sc, res = next(c for c in consistency if c["landscape_id"] == lid), results[lid]
        reg = [e for e in registry if e["landscape_id"] == lid]
        status[lid] = dict(
            y_acquired=True, y_feature_count=sc["y_features"], y_geometry=sc["y_geometry"], target_tier=res["tier"], tier_evidence=res["tier_evidence"],
            omega_acquired=True, omega_geometry_source=sc["omega_geometry_source"], omega_area_km2=sc["omega_area_km2"],
            omega_independent_of_y="TRUE: official GSI interpretation-range product distributed separately from the landslide layer",
            m_acquired=True, m_source="Copernicus DEM GLO-30 (SRC_GLOBAL_COP30)", m_tiles=COP_TILES[lid], m_epoch_pre_event=sc["dem_all_tiles_pre_event"],
            m_leakage="LOW" if sc["dem_all_tiles_pre_event"] else "UNRESOLVED", m_preferred_gsi_5_10m="BLOCKED_REGISTRATION_REQUIRED_AWAITING_USER_AUTHORISATION",
            g_acquired=lid == "KUMAMOTO_ASO_2016",
            g_status="ACQUIRED_GSJ_ASO_1:50000_SHAPEFILE" if lid == "KUMAMOTO_ASO_2016" else "PENDING_HARMONISATION_STAGE (GSJ Seamless export route not bounded in manifest)",
            downloaded_bytes=sum(e["actual_bytes"] for e in reg), files=len(reg), notes=res["notes"])
    report = _report(status, consistency, geo_summary, req, eligible, erratum)
    (OUT / "STAGE2B_REPORT.md").write_text(report)
    prior = {rel: sha256(ROOT / rel) for rel in PRIOR}
    generated = sorted(p for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("STAGE2B_CHECKPOINT", "STAGE2B_DIAGNOSTICS")))
    checkpoint = {"schema": "SFV2_STAGE2B_CHECKPOINT_V1", "date": DATE, "scope": ORDER, "raw_registry_scope_digest": scope_digest(registry), "registry_scope_rule": "Stage-2B planned acquisition IDs only (registry is shared and append-only)",
                  "raw_files": {e["raw_path"]: {"sha256": e["sha256"], "bytes": e["actual_bytes"]} for e in sorted(registry, key=lambda e: e["raw_path"])},
                  "total_raw_bytes": sum(e["actual_bytes"] for e in registry), "output_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in generated},
                  "prior_checkpoint_hashes": prior, "landscape_status": {k: {kk: vv for kk, vv in v.items() if kk != "notes"} for k, v in status.items()},
                  "volatile_diagnostics": {"path": "outputs/stage2b/STAGE2B_DIAGNOSTICS.json", "authoritative": False, "excluded_from_checkpoint_identity": True},
                  "training": False, "features": False, "harmonisation": False, "lockbox_assignment": False, "e_downloaded": False,
                  "kurikoma_transfer_executed": False}
    _canon(OUT / "STAGE2B_CHECKPOINT.json", checkpoint)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _canon(OUT / "STAGE2B_DIAGNOSTICS.json", {"authoritative": False, "head": git("rev-parse", "HEAD"),
                                              "untracked_entries": len([l for l in git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if l.startswith("??")])})
    return checkpoint


def _report(status, consistency, geo, req, eligible, erratum) -> str:
    L = ["# Stage 2B — bounded core acquisition and raw QC", ""]
    for lid, s in status.items():
        sc = next(c for c in consistency if c["landscape_id"] == lid)
        L += [f"## {lid}", "", f"- Y: {s['y_feature_count']} {s['y_geometry']} features, {s['target_tier']} — {s['tier_evidence']}",
              f"- Omega: {s['omega_geometry_source']}, {s['omega_area_km2']} km2; Y outside Omega: {sc['y_outside_omega_count']}",
              f"- M: {s['m_source']} tiles {', '.join(s['m_tiles'])}; pre-event {s['m_epoch_pre_event']}; leakage {s['m_leakage']}; "
              f"Omega coverage {sc['dem_omega_dem_coverage_fraction']}", f"- G: {s['g_status']}", f"- Raw bytes: {s['downloaded_bytes']:,}",
              *[f"- Note: {n}" for n in s["notes"]], ""]
    L += ["## Kumamoto geology", "", f"Coverage of the union interpretation range by the 1:50,000 Aso map: {geo.get('omega_geology_coverage_fraction')} "
          "(the Aso map covers the volcano, not the whole April range).", "",
          "## Governance", "", f"- External request queue: {len(req)} prepared entries; nobody contacted.",
          f"- Kurikoma transfer: {len(eligible)} raw V1 files prepared, {sum(r['v1_hash_match'] == 'TRUE' for r in eligible)} hash-verified; NOT executed.",
          f"- Stage-2A.2 transient hash erratum: {len(erratum)} rows corrected (mispaired hashes); Stage-2A.2 checkpoint not rewritten.",
          "- No E download, harmonisation, features, H reconstruction, training or lockbox assignment.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
