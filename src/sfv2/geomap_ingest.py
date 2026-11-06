"""Generic Indonesian GeoMap (Badan Geologi) district-SHP ingestion, generalised from the Lombok Stage 2F.1/2F.2 pipeline.

Parameterised by a district table, so any landscape (e.g. Talakmau) reuses the exact Lombok-validated logic:
registration of untouched user downloads -> archive integrity -> geology-layer discovery -> per-district load with source
preservation -> seam QC -> unit inventory and legend-evidence template.  Unit CLASSIFICATION is deliberately not done here:
it requires the official sheet legend (as for Lombok sheet 1807) and is recorded per landscape with evidence.
"""
from __future__ import annotations

import collections
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .raw_store import load_registry, save_registry, sha256
from .schema import ROOT
from .stage2f_transfer import INCOMING, register_incoming

GEOMAP_PUBLISHER = "Pusat Survei Geologi / Puslitbang Geologi, Badan Geologi (Indonesia GeoMap V.2)"
GEOMAP_LICENCE = ("GeoMap Lisensi Terbuka: use permitted with attribution to Pusat Survei Geologi (PSG), Badan Geologi; resale prohibited "
                  "(terms accepted by the user at authenticated download)")
GEOLOGY_LAYER = re.compile(r"geologi_ar[^/]*\.shp$", re.I)
REQUIRED_PARTS = (".shp", ".shx", ".dbf", ".prj")
UNIT_FIELDS = ("SIMOBJ", "NAMOBJ", "UMUROBJ", "REMARK")


def find_incoming(pattern: str) -> list[Path]:
    rx = re.compile(pattern)
    return sorted(p for p in INCOMING.glob("*") if p.is_file() and rx.fullmatch(p.name))


def register_district(path: Path, landscape_id: str, acquisition_id: str, district_name: str, district_code: str) -> dict:
    meta = dict(acquisition_id=acquisition_id, source_id="GEOMAP_PSG_DISTRICT_GEOLOGY",
                dataset_name=f"GeoMap SHP Peta Geologi 1:100,000 district extract - {district_name} ({district_code})",
                source_url=f"https://geologi.esdm.go.id/geomap/pages/district/{district_code}", publisher=GEOMAP_PUBLISHER, access_date="USER_DOWNLOAD",
                licence=GEOMAP_LICENCE, temporal_coverage="GeoMap V.2 vector compilation", spatial_coverage=f"{district_name} district extent",
                resolution="1:100,000 vector", crs="per .prj (verified at load)", expected_format="ZIP/ESRI Shapefile",
                notes="User-authenticated GeoMap download; original preserved in data/incoming_user; raw copy immutable")
    return _register(path, landscape_id, "GEOLOGY", meta)


def register_sheet(path: Path, landscape_id: str, acquisition_id: str, title: str, preview_slug: str) -> dict:
    meta = dict(acquisition_id=acquisition_id, source_id="GEOMAP_PSG_SHEET_250K", dataset_name=title,
                source_url=f"https://geologi.esdm.go.id/geomap/pages/preview/{preview_slug}", publisher=GEOMAP_PUBLISHER, access_date="USER_DOWNLOAD",
                licence=GEOMAP_LICENCE, temporal_coverage="per sheet imprint", spatial_coverage=title, resolution="1:250,000 scanned map",
                crs="NONE (non-georeferenced raster; graticule printed)", expected_format="JPEG", notes="Legend/explanation used to resolve district units")
    return _register(path, landscape_id, "GEOLOGY_LEGEND_MAP", meta)


def _register(path: Path, landscape_id: str, role: str, meta: dict) -> dict:
    registry = load_registry()
    rec = register_incoming(path, landscape_id, role, meta)
    rec["user_authenticated_acquisition"] = True
    prior = next((e for e in registry if e["acquisition_id"] == rec["acquisition_id"]), None)
    if prior and prior["sha256"] != rec["sha256"]:
        raise RuntimeError(f"STOP: registered hash changed for {rec['acquisition_id']}")
    if not prior:
        registry.append(rec)
        save_registry(registry)
    return prior or rec


def archive_audit(zip_path: Path) -> dict:
    zf = zipfile.ZipFile(zip_path)
    names = [i.filename for i in zf.infolist() if not i.is_dir()]
    layers = [n for n in names if GEOLOGY_LAYER.search(n)]
    stem = layers[0][:-4] if layers else ""
    return dict(path=str(zip_path.relative_to(ROOT)) if ROOT in zip_path.parents else str(zip_path), sha256=sha256(zip_path), members=len(names),
                crc_ok=zf.testzip() is None, geology_layers=layers, required_parts_present=bool(stem) and all(stem + s in names for s in REQUIRED_PARTS))


def load_districts(districts: dict[str, dict], utm: str) -> gpd.GeoDataFrame:
    """districts: {key: {acq: acquisition id, layer: optional explicit member path}}; preserves source district and CRS."""
    reg = {e["acquisition_id"]: e for e in load_registry()}
    frames = []
    for key, d in districts.items():
        zp = ROOT / reg[d["acq"]]["raw_path"]
        layer = d.get("layer") or archive_audit(zp)["geology_layers"][0]
        g = gpd.read_file(f"/vsizip/{zp}/{layer}")
        g["source_district"], g["source_crs"], g["source_acquisition_id"] = key, str(g.crs), d["acq"]
        frames.append(g.to_crs(utm))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=utm)


def seam_qc(geo: gpd.GeoDataFrame, keys: list[str], unit_field: str = "SIMOBJ") -> list[dict]:
    """Identical algorithm to the Lombok Stage 2F.2 seam QC (validated by reproducing its frozen output)."""
    rows = []
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
                    pairs[(x[unit_field], y[unit_field])] += x.geometry.buffer(50).intersection(y.geometry.buffer(50)).intersection(near).area
            tot = sum(pairs.values()) or 1
            gap = unions[a].buffer(25).intersection(unions[b].buffer(25)).difference(unions[a]).difference(unions[b])
            parts = [p for p in getattr(gap, "geoms", [gap]) if not p.is_empty]
            rows.append(dict(seam=f"{a}|{b}", shared_boundary_m=round(shared.length, 1), overlap_m2=round(unions[a].intersection(unions[b]).area, 2),
                             gap_within_25m_m2=round(gap.area, 1), slivers_lt_5000m2=sum(p.area < 5000 for p in parts),
                             same_code_contact_fraction=round(sum(v for (x, y), v in pairs.items() if x == y) / tot, 4),
                             code_changes_at_seam="|".join(f"{x}>{y}:{round(v)}" for (x, y), v in sorted(pairs.items()) if x != y),
                             crs_identical=len(set(geo[geo.source_district.isin([a, b])].source_crs)) == 1))
    return rows


def unit_inventory(geo: gpd.GeoDataFrame, omega=None) -> pd.DataFrame:
    """One row per (unit code, name, age, remark) with districts, polygon count and area (inside Omega when given).
    harmonised_class is left PENDING_SHEET_LEGEND: classification requires the official legend evidence."""
    g = geo.copy()
    g["area_km2"] = (g.geometry.intersection(omega) if omega is not None else g.geometry).area / 1e6
    fields = [f for f in UNIT_FIELDS if f in g.columns]
    out = (g.groupby(fields, dropna=False).agg(polygons=("geometry", "size"), area_km2=("area_km2", "sum"),
                                               districts=("source_district", lambda s: "|".join(sorted(set(s))))).reset_index())
    out["area_km2"] = out.area_km2.round(4)
    out["harmonised_class"] = "PENDING_SHEET_LEGEND"
    out["mapping_rule"] = "NOT_CLASSIFIED_BEFORE_LEGEND_EVIDENCE"
    return out.sort_values(fields).reset_index(drop=True)
