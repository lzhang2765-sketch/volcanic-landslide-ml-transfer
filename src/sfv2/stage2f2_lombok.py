"""Stage 2F.2: Lombok four-district geology, official sheet-1807 legend resolution, target freeze, domain and gates.

RAW is read-only; derived rasters go to data/processed/stage2f2/LOMBOK_RINJANI_2018.  Omega is the unchanged published
mapping extent and is never derived from or trimmed to Y or geology.  Production is delegated to stage2f2_production.py
and runs only when every gate passes.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import Point
from shapely.ops import unary_union

from .raw_store import acquire, load_registry, save_registry, sha256
from .schema import ROOT
from .stage2c_geology import CLASSES, VOLCANIC
from .stage2f_transfer import INCOMING, register_incoming

LID = "LOMBOK_RINJANI_2018"
UTM = "EPSG:32750"
RES, BUFFER = 30.0, 1500.0
OUT = ROOT / "outputs/stage2f2"
DER = ROOT / "data/processed/stage2f2" / LID
V1T = ROOT / "data/raw/stage2d/LOMBOK_RINJANI_2018/V1_TRANSFER"
AS_OF = "2026-10-06"
WATER_EDM = {8, 9, 10}
GEOMAP_LICENCE = ("GeoMap Lisensi Terbuka: use permitted with attribution to Pusat Survei Geologi (PSG), Badan Geologi; resale prohibited "
                  "(GeoMap usage policy, accessed 2026-10-06)")
DISTRICTS = {
    "UTARA": dict(acq="ACQ_LOMBOK_GEOMAP_UTARA_SHP", code="52.08", layer="lombok_utara/geologi_ar_100k_lombok_utara.shp"),
    "TIMUR": dict(acq="ACQ_LOMBOK_GEOMAP_TIMUR_SHP", code="52.03", layer="lombok_timur/geologi_ar_100k_lombok_timur.shp"),
    "BARAT": dict(acq="ACQ_LOMBOK_GEOMAP_BARAT_SHP", code="52.01", layer="lombok_barat/geologi_ar_100k_lombok_barat.shp",
                  file="Peta_SHP_Lombok_Barat_9e74e6d683b2578f7ad2a05802bbfac3.zip", district="Kabupaten Lombok Barat"),
    "TENGAH": dict(acq="ACQ_LOMBOK_GEOMAP_TENGAH_SHP", code="52.02", layer="lombok_tengah/geologi_ar_100k_lombok_tengah.shp",
                   file="Peta_SHP_Lombok_Tengah_efa49209abd877789e0e8dc6b79c8eab.zip", district="Kabupaten Lombok Tengah"),
}
SHEET_FILE = "Peta_Geologi_Lembar_Lombok,_Nusatenggara_Barat_6c7aafffa4db07b2417cb8a905a15a6c.jpg"
SHEET_ACQ = "ACQ_LOMBOK_GEOMAP_SHEET1807_1994_RASTER"
ESURF_ACQ = "ACQ_LOMBOK_DOC_BURROWS2025_ESURF"

# ------------------------------------------------------------------------------------------------- official sheet-1807 legend
# Transcribed verbatim from the official raster (KETERANGAN/EXPLANATION, full-resolution crop x5780-7420, y2900-4250) and the
# KORELASI SATUAN PETA (correlation) chart (x5650-7700, y380-2600).  Only legible text was transcribed.
SHEET_LEGEND = {
    "Qa": ("ALUVIUM / ALLUVIUM", "Pebble, granule, sand, clay, peat and fragments of coral", "Holocene", "SURFICIAL_DEPOSITS"),
    "Qhv": ("BATUAN GUNUNGAPI TAK TERPISAHKAN / UNDIFFERENTIATED VOLCANIC ROCKS (p=Pusuk, n=Nangi, r=Rinjani)", "Lava, breccia and tuff",
            "Holocene", "VOLCANIC_ROCKS"),
    "Qvl": ("FORMASI LEKOPIKO / LEKOPIKO FORMATION", "Pumiceous tuff, laharic breccia and lava", "Pleistocene", "VOLCANIC_ROCKS"),
    "TQb": ("FORMASI KALIBABAK / KALIBABAK FORMATION", "Breccia and lava", "Pliocene-Pleistocene", "VOLCANIC_ROCKS"),
    "TQp": ("FORMASI KALIPALUNG / KALIPALUNG FORMATION", "Alternating calcareous breccia and lava", "Pliocene-Pleistocene", "VOLCANIC_ROCKS"),
    "TQs": ("ANGGOTA SELAYAR FORMASI KALIPALUNG / SELAYAR MEMBER", "Tuffaceous sandstone, tuffaceous claystone with thin carbon intercalations",
            "Pliocene-Pleistocene", "VOLCANIC_ROCKS (member)"),
    "Tme": ("FORMASI EKAS / EKAS FORMATION", "Limestone/calcarenite, crystalline locally", "Middle-Late Miocene", "SEDIMENTARY_ROCKS"),
    "Tomp": ("FORMASI PENGULUNG / PENGULUNG FORMATION", "Breccia, lava, tuff with lenses of limestone, containing sulphide minerals and quartz veins",
             "Late Oligocene-Early Miocene", "VOLCANIC_ROCKS (interfingering with Tomk)"),
    "Tomk": ("FORMASI KAWANGAN / KAWANGAN FORMATION", "Alternating quartz sandstone, claystone and breccia", "Late Oligocene-Early Miocene",
             "SEDIMENTARY_ROCKS"),
    "Tmi": ("BATUAN TEROBOSAN / INTRUSIVE ROCKS", "Dacite and basalt", "Middle Miocene", "IGNEOUS_INTRUSIVE_ROCKS"),
}
# District SHP code -> official sheet unit -> frozen ontology (class, confidence, rationale)
UNIT_MAP = {
    "Qa": ("Qa", "UNCONSOLIDATED_SURFICIAL", "HIGH", "Official legend: alluvium"),
    "Qhv1": ("Qhv", "MIXED_VOLCANIC", "HIGH", "SHP REMARK 'Batuan Gunungapi Tidak Terdiferensiasi' = official Qhv; lava, breccia and tuff undivided"),
    "Qhv": ("Qhv", "MIXED_VOLCANIC", "MEDIUM", "SHP name 'Youngers Volcanic / Batuan Gunungapi Muda' not on sheet legend; code matches official Qhv (volcanic)"),
    "Qvl": ("Qvl", "MIXED_VOLCANIC", "HIGH", "Official legend: pumiceous tuff, laharic breccia and lava (volcanic-rocks column)"),
    "TQb": ("TQb", "MIXED_VOLCANIC", "HIGH", "Official legend: breccia and lava (volcanic-rocks column)"),
    "TQp": ("TQp", "MIXED_VOLCANIC", "MEDIUM", "Official legend: alternating calcareous breccia and lava (volcanic-rocks column)"),
    "Tomp": ("Tomp", "MIXED_VOLCANIC", "HIGH", "Official legend: breccia, lava, tuff with limestone lenses"),
    "Tme": ("Tme", "SEDIMENTARY_BASEMENT", "HIGH", "Official legend: limestone/calcarenite"),
    "Tomk": ("Tomk", "SEDIMENTARY_BASEMENT", "HIGH", "Official legend: quartz sandstone, claystone and breccia"),
    "Tmi": ("Tmi", "INTRUSIVE", "HIGH", "Official legend: intrusive dacite and basalt"),
}
ANOMALIES = {
    "Qhvb": "Name 'Batur Volcanics' (Bali) absent from the sheet-1807 legend; geometry lies on the Rinjani/Samalas caldera floor",
    "Qv": "Name 'Volcanic of Lesong-Pohen-Sengayang' (Bali) absent from the sheet-1807 legend",
    "Gr4": "Name 'Palu Granite' (Sulawesi, Jurassic) absent from the sheet-1807 legend; tiny coastal polygons outside Omega",
}
ERA = {"Kuarter": "Cenozoic Quaternary", "Neogen": "Cenozoic Neogene", "Jura": "Mesozoic Jurassic"}


# ------------------------------------------------------------------------------------------------- registration
def register_new() -> list[dict]:
    registry, recs = load_registry(), []
    items = [(k, d["file"], "GEOLOGY", d["acq"], f"GeoMap SHP Peta Geologi 1:100,000 district extract - {d['district']} ({d['code']})",
              f"https://geologi.esdm.go.id/geomap/pages/district/{d['code']}", "1:100,000 vector", "EPSG:4326 (per .prj)", "ZIP/ESRI Shapefile",
              f"{d['district']} district extent") for k, d in DISTRICTS.items() if "file" in d]
    items.append(("SHEET", SHEET_FILE, "GEOLOGY_LEGEND_MAP", SHEET_ACQ,
                  "Peta Geologi Lembar Lombok, Nusatenggara Barat, sheet 1807, 1:250,000 (Mangga, Atmawinata, Hermanto, Setyogroho & Amin, 1994)",
                  "https://geologi.esdm.go.id/geomap/pages/preview/peta-geologi-lembar-lombok-nusatenggara-barat", "1:250,000 scanned map, 300 dpi",
                  "NONE (non-georeferenced JPEG; graticule 115d45'-116d45'E, 8d-9dS printed)", "JPEG", "Lombok island (sheet 1807)"))
    for key, fname, role, acq, name, url, res, crs, fmt, cov in items:
        meta = dict(acquisition_id=acq, source_id="S2F1_GEOMAP_PSG_DISTRICT_GEOLOGY" if role == "GEOLOGY" else "S2F2_GEOMAP_PSG_SHEET_1807",
                    dataset_name=name, source_url=url, publisher="Pusat Survei Geologi / Puslitbang Geologi, Badan Geologi (Indonesia GeoMap V.2)",
                    access_date="2026-10-06", licence=GEOMAP_LICENCE,
                    temporal_coverage="GeoMap V.2 vector compiled 2021-09-26" if role == "GEOLOGY" else "Published 1994 (geology mapped 1988)",
                    spatial_coverage=cov, resolution=res, crs=crs, expected_format=fmt,
                    v1_unseen_eligibility="NOT_APPLICABLE_NEW_V2_GEOLOGY (landscape remains INELIGIBLE_V1_HISTORICAL)",
                    notes="User-authenticated GeoMap download; original preserved in data/incoming_user; raw copy immutable")
        rec = register_incoming(INCOMING / fname, LID, role, meta)
        rec["user_authenticated_acquisition"] = True
        prior = next((e for e in registry if e["acquisition_id"] == acq), None)
        if prior and prior["sha256"] != rec["sha256"]:
            raise RuntimeError(f"STOP: hash changed for {acq}")
        if not prior:
            registry.append(rec)
        recs.append(rec)
    save_registry(registry)
    registry = load_registry()
    acquire(dict(acquisition_id=ESURF_ACQ, landscape_id=LID, source_id="DOC_BURROWS_MILLEDGE_FERRARIO_2025", data_role="SOURCE_DOC",
                 dataset_name="Burrows, Milledge & Ferrario (2025) Earth Surf. Dynam. 13:1039 (landing page)", original_filename="esurf-13-1039-2025.html",
                 source_url="https://doi.org/10.5194/esurf-13-1039-2025", download_url="https://esurf.copernicus.org/articles/13/1039/2025/",
                 publisher="Copernicus Publications (EGU)", access_date="2026-10-06", licence="CC BY 4.0", expected_format="HTML", expected_size="UNKNOWN",
                 temporal_coverage="2025", spatial_coverage="Lombok", resolution="N/A", verification_status="DOWNLOADED_SOURCE_DOC",
                 notes="Documents Ferrario (2019) inventories: 4823 (4.88 km2) after 5 Aug 2018; 9319 (10.25 km2) after 19 Aug 2018, cumulative"), registry)
    save_registry(registry)
    return recs


def _raw(acq: str) -> Path:
    return ROOT / next(e["raw_path"] for e in load_registry() if e["acquisition_id"] == acq)


# ------------------------------------------------------------------------------------------------- geology
def load_geology() -> gpd.GeoDataFrame:
    frames = []
    for key, d in DISTRICTS.items():
        g = gpd.read_file(f"/vsizip/{_raw(d['acq'])}/{d['layer']}")
        g["source_district"], g["source_crs"] = key, str(g.crs)
        g["source_acquisition_id"] = d["acq"]
        frames.append(g.to_crs(UTM))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=UTM)


def unit_mapping(geo: gpd.GeoDataFrame, mode: str = "PRIMARY") -> pd.DataFrame:
    """mode PRIMARY: anomalies UNKNOWN.  mode SENSITIVITY: anomalies on official volcanic units (diagnostic check) as MIXED_VOLCANIC."""
    rows = []
    for (code, name, age, remark), grp in geo.groupby(["SIMOBJ", "NAMOBJ", "UMUROBJ", "REMARK"], dropna=False):
        era = ERA.get(age, f"UNMAPPED:{age}")
        if code in ANOMALIES:
            official, cls, conf, why = "NONE_ON_SHEET", "UNKNOWN", "LOW", ANOMALIES[code]
            if mode == "SENSITIVITY" and code in ("Qhvb", "Qv"):
                cls, conf, why = "MIXED_VOLCANIC", "LOW", why + "; SENSITIVITY: diagnostic sheet overlay shows official volcanic units beneath"
        elif code in UNIT_MAP:
            official, cls, conf, why = UNIT_MAP[code]
        else:
            official, cls, conf, why = "NOT_FOUND", "UNKNOWN", "LOW", "Code not on sheet legend; not forced"
        leg = SHEET_LEGEND.get(official, ("", "NOT_IN_OFFICIAL_LEGEND", "", ""))
        rows.append(dict(source_unit_code=code, source_unit_name=name, source_remark_id=remark, source_age=f"{era} ({age})",
                         source_districts="|".join(sorted(set(grp.source_district))), official_sheet_unit=official, official_unit_name=leg[0],
                         source_lithology=leg[1], official_age=leg[2], official_rock_group=leg[3], harmonised_class=cls,
                         is_volcanic=str(cls in VOLCANIC).upper(), domain_eligible=str(cls in VOLCANIC and era.startswith("Cenozoic")).upper(),
                         mapping_confidence=conf, mapping_rule="SHEET1807_LEGEND" if code in UNIT_MAP else "ANOMALY_OR_UNMATCHED", rationale=why,
                         evidence=f"GeoMap district SHP ({'|'.join(sorted(set(grp.source_district)))}) + sheet-1807 legend raster ({SHEET_ACQ})",
                         polygons=len(grp), area_km2=round(grp.union_all().area / 1e6, 4)))
    df = pd.DataFrame(rows).sort_values(["source_unit_code", "source_unit_name"]).reset_index(drop=True)
    assert set(df.harmonised_class) <= set(CLASSES)
    return df


def seam_qc(geo: gpd.GeoDataFrame) -> list[dict]:
    rows, keys = [], list(DISTRICTS)
    unions = {k: geo[geo.source_district == k].union_all() for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = unions[a].boundary.intersection(unions[b].boundary)
            if shared.is_empty or shared.length < 100:
                continue
            near = shared.buffer(50)
            ga, gb = geo[(geo.source_district == a) & geo.intersects(near)], geo[(geo.source_district == b) & geo.intersects(near)]
            pairs = collections.Counter()
            for _, x in ga.iterrows():
                for _, y in gb[gb.intersects(x.geometry.buffer(50))].iterrows():
                    pairs[(x.SIMOBJ, y.SIMOBJ)] += x.geometry.buffer(50).intersection(y.geometry.buffer(50)).intersection(near).area
            tot = sum(pairs.values()) or 1
            gap = unions[a].buffer(25).intersection(unions[b].buffer(25)).difference(unions[a]).difference(unions[b])
            parts = [p for p in getattr(gap, "geoms", [gap]) if not p.is_empty]
            rows.append(dict(seam=f"{a}|{b}", shared_boundary_m=round(shared.length, 1), overlap_m2=round(unions[a].intersection(unions[b]).area, 2),
                             gap_within_25m_m2=round(gap.area, 1), slivers_lt_5000m2=sum(p.area < 5000 for p in parts),
                             same_code_contact_fraction=round(sum(v for (x, y), v in pairs.items() if x == y) / tot, 4),
                             code_changes_at_seam="|".join(f"{x}>{y}:{round(v)}" for (x, y), v in sorted(pairs.items()) if x != y),
                             crs_identical=len(set(geo[geo.source_district.isin([a, b])].source_crs)) == 1))
    return rows


def anomaly_overlay(geo: gpd.GeoDataFrame) -> dict:
    """DIAGNOSTIC ONLY: affine georeference of the sheet neatline (115.75-116.75E, 8-9S) and nearest-legend-swatch colour votes."""
    Image.MAX_IMAGE_PIXELS = None
    im = np.asarray(Image.open(_raw(SHEET_ACQ)).convert("RGB")).astype(float)
    L, R, T, B = 303, 5513, 465, 5682
    sw = {"Qa": (5890, 2980), "Qhv": (5840, 3105), "Qvl": (5840, 3245), "TQb": (5840, 3365), "TQp": (5840, 3525), "TQs": (5840, 3660),
          "Tme": (5840, 3805), "Tomp": (5840, 3930), "Tomk": (5840, 4085), "Tmi": (5840, 4185), "SEA": (1500, 1000)}
    names = list(sw)
    C = np.array([np.median(im[y - 6:y + 6, x - 6:x + 6].reshape(-1, 3), 0) for x, y in sw.values()])
    g4 = geo.to_crs(4326)
    out, rng = {}, np.random.default_rng(20261006)
    for code, grp in g4.groupby("SIMOBJ"):
        geom = grp.union_all().buffer(-0.002)
        if geom.is_empty:
            out[code] = {"status": "TOO_SMALL_FOR_OVERLAY"}
            continue
        minx, miny, maxx, maxy = geom.bounds
        votes, n, tries = collections.Counter(), 0, 0
        while n < 300 and tries < 30000:
            tries += 1
            lo, la = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if not geom.contains(Point(lo, la)):
                continue
            x, y = int(L + (lo - 115.75) * (R - L)), int(T + (-8.0 - la) * (B - T))
            p = np.median(im[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3), 0)
            votes[names[int(np.argmin(((C - p) ** 2).sum(1)))]] += 1
            n += 1
        out[code] = dict(points=n, votes=dict(votes.most_common()), official_volcanic_fraction=round(
            sum(v for k, v in votes.items() if k in ("Qhv", "Qvl", "TQb", "TQp", "Tomp")) / max(n, 1), 3))
    return out


# ------------------------------------------------------------------------------------------------- grid / rasters
def grid(omega) -> dict:
    minx, miny, maxx, maxy = omega.bounds
    minx, miny = math.floor((minx - BUFFER) / RES) * RES, math.floor((miny - BUFFER) / RES) * RES
    maxx, maxy = math.ceil((maxx + BUFFER) / RES) * RES, math.ceil((maxy + BUFFER) / RES) * RES
    return dict(crs=UTM, transform=from_origin(minx, maxy, RES, RES), width=int((maxx - minx) / RES), height=int((maxy - miny) / RES))


def _write(path: Path, arr, g, nodata, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype, crs=g["crs"],
                       transform=g["transform"], nodata=nodata, compress="deflate", tiled=True) as d:
        d.write(arr.astype(dtype), 1)
    return path


def _reproject(src_path: Path, g, resampling, dtype, nodata):
    dst = np.full((g["height"], g["width"]), nodata, dtype=dtype)
    with rasterio.open(src_path) as s:
        reproject(source=rasterio.band(s, 1), destination=dst, dst_transform=g["transform"], dst_crs=g["crs"], dst_nodata=nodata,
                  resampling=resampling, src_nodata=s.nodata)
    return dst


def build_domain(geo, mapping_primary, mapping_sens, omega) -> dict:
    g = grid(omega)
    lineage = []
    dem_src = V1T / "DEM/Copernicus_DSM_COG_10_S09_00_E116_00_DEM.tif"
    dem = _reproject(dem_src, g, Resampling.bilinear, np.float32, -9999.0)
    p_dem = _write(DER / "dem_cop30_30m.tif", dem, g, -9999.0, "float32")
    lineage.append((p_dem, [dem_src], "reproject (bilinear, single)"))
    masks = {}
    for k in ("EDM", "FLM", "WBM"):
        src = _raw(f"ACQ_LOMBOK_COP30_S09_E116_{k}")
        masks[k] = _reproject(src, g, Resampling.nearest, np.uint8, 255)
        lineage.append((_write(DER / f"cop30_{k.lower()}_30m.tif", masks[k], g, 255, "uint8"), [src], "reproject (nearest, categorical)"))
    # geology index raster: cell-centre rasterisation of unit polygons, index = 1-based row of the primary mapping table
    key = {(r.source_unit_code, r.source_unit_name): i + 1 for i, r in mapping_primary.iterrows()}
    shapes = [(geom, key[(c, n)]) for geom, c, n in zip(geo.geometry, geo.SIMOBJ, geo.NAMOBJ)]
    geo_idx = rasterize(shapes, out_shape=(g["height"], g["width"]), transform=g["transform"], fill=0, dtype="uint16")
    p_geo = _write(DER / "geology_geomap_index_30m.tif", geo_idx, g, 0, "uint16")
    lineage.append((p_geo, [_raw(d["acq"]) for d in DISTRICTS.values()], "rasterise unit polygons at cell centres (categorical; no interpolation)"))
    omega_cells = rasterize([(omega, 1)], out_shape=geo_idx.shape, transform=g["transform"], fill=0, dtype="uint8").astype(bool)
    dem_bad = dem == -9999.0
    water = np.isin(masks["EDM"], list(WATER_EDM))
    out = {}
    for mode, mp in (("PRIMARY", mapping_primary), ("SENSITIVITY_ANOMALIES_INCLUDED", mapping_sens)):
        cls = np.array(["UNMAPPED"] + list(mp.harmonised_class))[geo_idx]
        elig = np.array([False] + list(mp.domain_eligible == "TRUE"))[geo_idx]
        cur = omega_cells.copy()
        steps = [("official_omega_obs", int(cur.sum()))]
        for label, remove in (("visibility_mask_none_supplied", np.zeros_like(cur)), ("dem_nodata", dem_bad), ("dem_water_ocean_lake_river", water),
                              ("geology_unmapped", geo_idx == 0), ("geology_unknown_or_anomaly", cls == "UNKNOWN"),
                              ("surficial_geology", cls == "UNCONSOLIDATED_SURFICIAL"), ("intrusive_geology", cls == "INTRUSIVE"),
                              ("sedimentary_basement_geology", cls == "SEDIMENTARY_BASEMENT"), ("other_non_eligible", ~elig)):
            removed = int((cur & remove).sum())
            cur &= ~remove
            steps.append((label, removed))
        p = _write(DER / f"omega_analysis_{mode.lower()}_30m.tif", cur.astype(np.uint8), g, 255, "uint8")
        lineage.append((p, [p_dem, p_geo, DER / "cop30_edm_30m.tif", V1T / "OMEGA/SF1_VERIFICATION_mapping_extent.shp"],
                        f"OMEGA_ANALYSIS {mode}: official Omega - DEM nodata - water, restricted to Cenozoic volcanic geology"))
        class_area = {c: round(float(((cls == c) & omega_cells & ~water).sum()) * 9e-4, 3) for c in sorted(set(cls[omega_cells]))}
        out[mode] = dict(mask=cur, steps=[dict(step=s, cells=c, km2=round(c * 9e-4, 3)) for s, c in steps], analysis_km2=round(cur.sum() * 9e-4, 3),
                         retained_fraction=round(cur.sum() / omega_cells.sum(), 5), class_area_in_land_omega_km2=class_area,
                         eligible_volcanic_km2_in_land_omega=round(float((elig & omega_cells & ~water).sum()) * 9e-4, 3))
    return dict(grid=g, masks=masks, geo_idx=geo_idx, omega_cells=omega_cells, domains=out, lineage=lineage, water=water, dem_bad=dem_bad)


def y_audit(y: gpd.GeoDataFrame, omega, covered, dom, mapping) -> dict:
    g = dom["grid"]
    pts = y.geometry.representative_point()
    r, c = rasterio.transform.rowcol(g["transform"], pts.x.values, pts.y.values)
    r, c = np.clip(np.asarray(r), 0, g["height"] - 1), np.clip(np.asarray(c), 0, g["width"] - 1)
    cls = np.array(["UNMAPPED"] + list(mapping.harmonised_class))[dom["geo_idx"][r, c]]
    elig = np.array([False] + list(mapping.domain_eligible == "TRUE"))[dom["geo_idx"][r, c]]
    in_omega = pts.within(omega).values
    in_geo = pts.within(covered).values
    primary = dom["domains"]["PRIMARY"]["mask"]
    in_final = primary[r, c]
    burn = rasterize([(gm, 1) for gm in y.geometry], out_shape=primary.shape, transform=g["transform"], fill=0, dtype="uint8").astype(bool)
    reasons = collections.Counter()
    for io, ig, cl, el, fin, rr, cc in zip(in_omega, in_geo, cls, elig, in_final, r, c):
        if fin:
            continue
        if not io:
            reasons["outside_omega_obs"] += 1
        elif dom["dem_bad"][rr, cc]:
            reasons["dem_invalid"] += 1
        elif dom["water"][rr, cc]:
            reasons["dem_water"] += 1
        elif not ig or cl == "UNMAPPED":
            reasons["outside_geology_coverage"] += 1
        elif cl == "UNKNOWN":
            reasons["unknown_or_anomaly_geology"] += 1
        elif cl == "UNCONSOLIDATED_SURFICIAL":
            reasons["surficial_geology"] += 1
        elif cl in ("SEDIMENTARY_BASEMENT", "INTRUSIVE"):
            reasons["non_volcanic_geology"] += 1
        else:
            reasons["other"] += 1
    return dict(original=len(y), inside_omega_obs=int(in_omega.sum()), inside_geology_coverage=int((in_omega & in_geo).sum()),
                inside_eligible_volcanic_domain=int((in_omega & elig).sum()), inside_final_omega_analysis=int(in_final.sum()),
                excluded=dict(sorted(reasons.items())), area_km2=round(float(y.area.sum()) / 1e6, 4),
                y_cells_in_omega_analysis=int((burn & primary).sum()), rule="representative point per polygon for audit; labels use 9-cell-centre rule")
