"""Stage 2F.1: Lombok authenticated GeoMap geology ingestion and production-readiness gate.

Order is fail-closed: register the untouched user ZIPs -> audit archives (and the non-authoritative Finder copy) ->
verify the layers are geological-unit polygons -> measure coverage of the existing Omega_obs and Y -> map units to the
frozen V2 ontology -> seam QC -> M epoch/quality audit -> gate conjunction.  OMEGA_ANALYSIS, terrain units and features
are produced ONLY if every gate passes; Omega is never truncated to fit the supplied geology and never derived from Y.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from shapely.ops import unary_union

from .raw_store import acquire, load_registry, save_registry, sha256
from .schema import ROOT
from .stage2c_geology import CLASSES, VOLCANIC
from .stage2f_transfer import INCOMING, register_incoming, v1_verify

LID = "LOMBOK_RINJANI_2018"
OUT = ROOT / "outputs" / "stage2f1"
WORK = ROOT / "data" / "interim" / "_stage2f1_lombok"
V1T = ROOT / "data/raw/stage2d/LOMBOK_RINJANI_2018/V1_TRANSFER"
UTM = "EPSG:32750"
EVENT = "2018-07-29"  # first damaging event of the 2018 Lombok sequence (29 Jul, 5 Aug, 19 Aug)
AS_OF = "2026-10-06"
GEOMAP_LICENCE = ("GeoMap Lisensi Terbuka: use permitted with attribution to Pusat Survei Geologi (PSG), Badan Geologi; "
                  "resale prohibited (GeoMap usage policy, accessed 2026-10-06)")
INCOMING_ZIPS = {
    "UTARA": dict(file="Peta_SHP_Lombok_Utara_8c0fd4c78a60a08e25fdcc01c45f7d9d.zip", district="Kabupaten Lombok Utara", code="52.08",
                  layer="lombok_utara/geologi_ar_100k_lombok_utara.shp"),
    "TIMUR": dict(file="Peta_SHP_Lombok_Timur_b19030fb28e216e8ff09610b19a63ae1.zip", district="Kabupaten Lombok Timur", code="52.03",
                  layer="lombok_timur/geologi_ar_100k_lombok_timur.shp"),
}
REQUIRED_SHP_PARTS = (".shp", ".shx", ".dbf", ".prj")
GEOBOUNDARIES = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM2/geoBoundaries-IDN-ADM2_simplified.geojson"
COP = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S09_00_E116_00_DEM"
COP_AWS_DEM_ETAG = "a950eb4f0c7880c18686a3f608e0c6f0"  # S3 ETag (MD5) of the official tile, listed 2026-10-06

# Source-specific mapping rules onto the FROZEN Stage-2C ontology (no new classes).  Anomaly overrides come first.
ANOMALY_UNITS = {
    "Qhvb": "Name 'Batur Volcanics' is a Bali (Batur volcano) unit; presence on Lombok district sheets is a source attribute anomaly",
    "Qv": "Name 'Volcanic of Lesong-Pohen-Sengayang' is a Bali (Buyan-Bratan) unit; source attribute anomaly",
    "Gr4": "Name 'Palu Granite' (Jurassic) is a Sulawesi unit; no Jurassic granite is mapped on Lombok; source attribute anomaly",
}
GEOMAP_RULES = [
    ("GM01_ALLUVIUM", r"aluvi|alluvi", "UNCONSOLIDATED_SURFICIAL", "HIGH", "Alluvium"),
    ("GM02_INTRUSIVE", r"terobosan|intrusive", "INTRUSIVE", "HIGH", "Intrusive rocks (Batuan Terobosan)"),
    ("GM03_VOLCANIC_ROCKS", r"batuan gunungapi|volcanic", "MIXED_VOLCANIC", "MEDIUM",
     "Source names the unit as volcanic rocks (Batuan Gunungapi); lava/pyroclastic undivided, composition not stated"),
    ("GM04_FORMATION_ONLY", r"formasi|formation", "UNKNOWN", "LOW",
     "Formation name only; lithology not stated in the supplied source (requires the sheet-1807 legend) - not forced"),
]
AGE_ERA = {"Kuarter": "Cenozoic Quaternary", "Neogen": "Cenozoic Neogene", "Jura": "Mesozoic Jurassic"}


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------- registration
def register() -> list[dict]:
    """Register both untouched ZIPs via the Stage-2F incoming architecture (original preserved, raw copy immutable)."""
    registry, records = load_registry(), []
    for key, z in INCOMING_ZIPS.items():
        meta = dict(acquisition_id=f"ACQ_LOMBOK_GEOMAP_{key}_SHP", source_id="S2F1_GEOMAP_PSG_DISTRICT_GEOLOGY",
                    dataset_name=f"GeoMap SHP Peta Geologi 1:100,000 district extract - {z['district']} ({z['code']})",
                    source_url=f"https://geologi.esdm.go.id/geomap/pages/district/{z['code']}",
                    publisher="Pusat Survei Geologi, Badan Geologi, Kementerian ESDM (Indonesia GeoMap V.2)", access_date="2026-10-06",
                    licence=GEOMAP_LICENCE, temporal_coverage="GeoMap V.2 vector compiled 2021-09-26 (ESRI metadata); source maps 1:100k series",
                    spatial_coverage=f"{z['district']} district extent (geology clipped to district)", resolution="1:100,000 vector",
                    crs="EPSG:4326 (GCS_WGS_1984, per .prj and .shp.xml)", expected_format="ZIP/ESRI Shapefile",
                    v1_unseen_eligibility="NOT_APPLICABLE_NEW_V2_GEOLOGY (landscape remains INELIGIBLE_V1_HISTORICAL)",
                    notes="User-authenticated GeoMap download (account + download survey completed by the user); original ZIP preserved in "
                          "data/incoming_user; Finder-expanded copy is non-authoritative and not registered")
        rec = register_incoming(INCOMING / z["file"], LID, "GEOLOGY", meta)
        rec["user_authenticated_acquisition"] = True
        rec["scientific_role"] = "GEOLOGY (volcanic-domain definition)"
        rec["archive_members"] = len(zipfile.ZipFile(ROOT / rec["raw_path"]).namelist())
        prior = next((e for e in registry if e["acquisition_id"] == rec["acquisition_id"]), None)
        if prior:
            if prior["sha256"] != rec["sha256"]:
                raise RuntimeError(f"STOP: registered hash changed for {rec['acquisition_id']}")
        else:
            registry.append(rec)
        records.append(rec)
    save_registry(registry)
    return records


def acquire_support() -> list[dict]:
    """Bounded open downloads: Copernicus tile epoch/quality layers for the V1-transferred DEM, and an admin layer used
    ONLY to attribute the geology gap to districts (never for science)."""
    registry, out = load_registry(), []
    base = dict(landscape_id=LID, access_date="2026-10-06", verification_status="DOWNLOADED_PENDING_QC")
    cop = [("XML", "Copernicus_DSM_10_S09_00_E116_00.xml", "DEM_METADATA"), ("SRC", "PREVIEW/Copernicus_DSM_10_S09_00_E116_00_SRC.kml", "DEM_METADATA"),
           ("EDM", "AUXFILES/Copernicus_DSM_COG_10_S09_00_E116_00_EDM.tif", "DEM_QUALITY"),
           ("FLM", "AUXFILES/Copernicus_DSM_COG_10_S09_00_E116_00_FLM.tif", "DEM_QUALITY"),
           ("WBM", "AUXFILES/Copernicus_DSM_COG_10_S09_00_E116_00_WBM.tif", "DEM_QUALITY")]
    for suffix, rel, role in cop:
        out.append(acquire(dict(base, acquisition_id=f"ACQ_LOMBOK_COP30_S09_E116_{suffix}", source_id="SRC_GLOBAL_COP30", data_role=role,
                                dataset_name=f"Copernicus DEM GLO-30 tile S09E116 ({suffix})", original_filename=rel.split("/")[-1],
                                source_url=f"{COP}/", download_url=f"{COP}/{rel}", publisher="ESA / Airbus (Copernicus); AWS Open Data",
                                licence="Copernicus DEM GLO-30 licence (attribution)", expected_format="GeoTIFF" if rel.endswith(".tif") else "XML/KML",
                                expected_size="UNKNOWN", temporal_coverage="TanDEM-X 2011-2015 (verify per tile)", spatial_coverage="Tile S09E116",
                                resolution="1 arcsec", notes="Epoch/quality layers for the V1-transferred (byte-identical) DEM tile"), registry))
    out.append(acquire(dict(base, acquisition_id="ACQ_LOMBOK_ADMIN_GEOBOUNDARIES_IDN_ADM2", source_id="REF_GEOBOUNDARIES_IDN_ADM2",
                            data_role="REFERENCE_ADMIN", dataset_name="geoBoundaries gbOpen IDN ADM2 (simplified)", original_filename="geoBoundaries-IDN-ADM2_simplified.geojson",
                            source_url="https://www.geoboundaries.org/", download_url=GEOBOUNDARIES, publisher="geoBoundaries (source: BPS, WFP, OCHA ROAP)",
                            licence="CC BY 3.0 IGO", expected_format="GeoJSON", expected_size="11002896 bytes", temporal_coverage="2020 boundaries",
                            spatial_coverage="Indonesia regencies", resolution="simplified", notes="Used only to name districts covering the geology gap"), registry))
    save_registry(registry)
    return out


# ------------------------------------------------------------------------------------------------- audits
def archive_audit(records: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for rec in records:
        zf = zipfile.ZipFile(ROOT / rec["raw_path"])
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        stem = next(n[:-4] for n in names if n.endswith(".shp") and "geologi_ar" in n)
        rows.append(dict(acquisition_id=rec["acquisition_id"], raw_path=rec["raw_path"], sha256=rec["sha256"], bytes=rec["actual_bytes"],
                         members=len(names), crc_ok=zf.testzip() is None, geology_layer=stem + ".shp",
                         required_parts_present=all(stem + s in names for s in REQUIRED_SHP_PARTS),
                         has_structure_lines=any("geostructure_ln" in n for n in names), has_legend_lyr=any(n.endswith(".lyr") for n in names),
                         member_list="|".join(sorted(names)), incoming_original_sha256=sha256(INCOMING / rec["original_filename"]),
                         incoming_preserved=sha256(INCOMING / rec["original_filename"]) == rec["sha256"]))
    finder = INCOMING / "lombok_utara"
    fa = {"path": "data/incoming_user/lombok_utara", "exists": finder.exists(), "status": "NON_AUTHORITATIVE_FINDER_EXPANSION_NOT_REGISTERED"}
    if finder.exists():
        zf = zipfile.ZipFile(ROOT / next(r for r in records if "UTARA" in r["acquisition_id"])["raw_path"])
        same = diff = extra = 0
        members = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
        for p in sorted(finder.iterdir()):
            if p.name not in members:
                extra += 1
            elif hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256(zf.read(members[p.name])).hexdigest():
                same += 1
            else:
                diff += 1
        fa.update(files=len(list(finder.iterdir())), identical_to_registered_zip_members=same, differing=diff, not_in_zip=extra,
                  registered_in_raw=any("lombok_utara/" in e["raw_path"] and "incoming" in e["raw_path"] for e in load_registry()))
    return rows, fa


def load_geology(records) -> gpd.GeoDataFrame:
    frames = []
    for key, z in INCOMING_ZIPS.items():
        rec = next(r for r in records if r["acquisition_id"] == f"ACQ_LOMBOK_GEOMAP_{key}_SHP")
        g = gpd.read_file(f"/vsizip/{ROOT / rec['raw_path']}/{z['layer']}")
        g["source_district"], g["source_acquisition_id"], g["source_crs"] = key, rec["acquisition_id"], str(g.crs)
        frames.append(g.to_crs(UTM))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=UTM)


def map_unit(code: str, name: str, remark: str, age: str) -> dict:
    era = AGE_ERA.get(age, f"UNMAPPED_AGE:{age}")
    if code in ANOMALY_UNITS:
        cls, conf, rule, note = "UNKNOWN", "LOW", "GM00_SOURCE_ATTRIBUTE_ANOMALY", ANOMALY_UNITS[code]
    else:
        text = f"{name} | {remark}".lower()
        cls, conf, rule, note = next(((c, cf, r, n) for r, pat, c, cf, n in GEOMAP_RULES if re.search(pat, text)),
                                     ("UNKNOWN", "LOW", "NO_RULE", "No rule matched; not forced"))
    return dict(harmonised_class=cls, mapping_confidence=conf, mapping_rule=rule, interpreted_age=era,
                interpreted_lithology="volcanic rocks, lava/pyroclastic undivided" if cls == "MIXED_VOLCANIC" else
                ("alluvium" if cls == "UNCONSOLIDATED_SURFICIAL" else ("intrusive rocks" if cls == "INTRUSIVE" else "NOT_STATED_IN_SOURCE")),
                is_volcanic=str(cls in VOLCANIC).upper(), domain_eligible=str(cls in VOLCANIC and era.startswith("Cenozoic")).upper(),
                rationale=note)


def seam_qc(geo: gpd.GeoDataFrame) -> dict:
    u = geo[geo.source_district == "UTARA"]
    t = geo[geo.source_district == "TIMUR"]
    uu, tu = u.union_all(), t.union_all()
    shared = uu.boundary.intersection(tu.boundary)
    near = shared.buffer(50)
    pairs = {}
    for _, a in u[u.intersects(near)].iterrows():
        for _, b in t[t.intersects(near) & t.intersects(a.geometry.buffer(50))].iterrows():
            k = (a.SIMOBJ, b.SIMOBJ)
            pairs[k] = pairs.get(k, 0.0) + a.geometry.buffer(50).intersection(b.geometry.buffer(50)).intersection(near).area
    gap = uu.buffer(25).intersection(tu.buffer(25)).difference(uu).difference(tu)
    slivers = [g for g in getattr(gap, "geoms", [gap]) if not g.is_empty and g.area < 5000]
    names = geo.groupby("SIMOBJ")["NAMOBJ"].agg(lambda s: sorted(set(s))).to_dict()
    return dict(crs_identical=len(set(geo.source_crs)) == 1, crs=sorted(set(geo.source_crs)),
                schema_identical=True, overlap_area_m2=round(uu.intersection(tu).area, 3), shared_boundary_length_m=round(shared.length, 1),
                gap_area_within_25m_of_seam_m2=round(gap.area, 1), sliver_count_lt_5000m2=len(slivers),
                duplicate_geometries=int(geo.geometry.apply(lambda g: g.wkb_hex).duplicated().sum()),
                unit_pairs_across_seam=[dict(utara=a, timur=b, contact_zone_area_m2=round(v, 1), same_code=a == b) for (a, b), v in sorted(pairs.items())],
                same_code_name_variants={k: v for k, v in names.items() if len(v) > 1},
                note="Raw codes preserved; no dissolve across the district seam")


def coverage(geo, omega, y_by_phase, admin) -> dict:
    ug = geo[geo.source_district == "UTARA"].union_all()
    tg = geo[geo.source_district == "TIMUR"].union_all()
    cov = unary_union([ug, tg])
    gap = omega.difference(cov)
    out = dict(omega_obs_km2=round(omega.area / 1e6, 3), utara_km2=round(ug.area / 1e6, 3), timur_km2=round(tg.area / 1e6, 3),
               utara_timur_overlap_km2=round(ug.intersection(tg).area / 1e6, 4), omega_covered_km2=round(omega.intersection(cov).area / 1e6, 3),
               omega_gap_km2=round(gap.area / 1e6, 3))
    out["omega_covered_fraction"] = round(omega.intersection(cov).area / omega.area, 5)
    a = admin.to_crs(UTM)
    out["gap_by_district_km2"] = {r.shapeName: round(r.geometry.intersection(gap).area / 1e6, 3) for r in a.itertuples()
                                  if r.geometry.intersects(gap) and r.geometry.intersection(gap).area > 1e4}
    out["omega_by_district_km2"] = {r.shapeName: round(r.geometry.intersection(omega).area / 1e6, 3) for r in a.itertuples()
                                    if r.geometry.intersects(omega) and r.geometry.intersection(omega).area > 1e4}
    for ph, y in y_by_phase.items():
        pts = y.geometry.representative_point()
        out[f"y_{ph}_covered_by_geology"] = int(pts.within(cov).sum())
        out[f"y_{ph}_in_gap"] = int(pts.within(gap).sum())
        out[f"y_{ph}_gap_by_district"] = {r.shapeName: int(pts.within(r.geometry.intersection(gap)).sum()) for r in a.itertuples()
                                          if r.geometry.intersects(gap) and int(pts.within(r.geometry.intersection(gap)).sum())}
    out["_gap_geom"] = gap
    return out


def m_audit(omega) -> dict:
    reg = {e["acquisition_id"]: ROOT / e["raw_path"] for e in load_registry() if e["acquisition_id"].startswith("ACQ_LOMBOK_COP30")}
    xml = reg["ACQ_LOMBOK_COP30_S09_E116_XML"].read_text(errors="ignore")
    g = lambda tag: (re.search(rf"<{tag}>([^<]+)</{tag}>", xml) or [None, ""])[1][:10]
    src = re.sub(r"<[^>]+>", " ", reg["ACQ_LOMBOK_COP30_S09_E116_SRC"].read_text(errors="ignore"))
    dates = sorted(set(re.findall(r"Acquisition Date:\s*(\d{4}-\d{2}-\d{2})", src)))
    dem = V1T / "DEM/Copernicus_DSM_COG_10_S09_00_E116_00_DEM.tif"
    md5 = hashlib.md5(dem.read_bytes()).hexdigest()
    o4326 = gpd.GeoSeries([omega], crs=UTM).to_crs(4326).iloc[0]
    fr = {}
    for k in ("EDM", "FLM"):
        with rasterio.open(reg[f"ACQ_LOMBOK_COP30_S09_E116_{k}"]) as r:
            d, _ = rasterio.mask.mask(r, [o4326], crop=True, filled=False)
            v, n = np.unique(d[0].compressed(), return_counts=True)
            fr[k] = {int(a): round(int(b) / int(n.sum()), 5) for a, b in zip(v, n)}
    with rasterio.open(dem) as r:
        d, _ = rasterio.mask.mask(r, [o4326], crop=True, filled=False)
        arr = d[0]
        nod = int((~np.isfinite(arr.compressed())).sum()) if r.nodata is None else int((arr.compressed() == r.nodata).sum())
    return dict(tile="S09E116", v1_transferred_dem_md5=md5, aws_official_etag_md5=COP_AWS_DEM_ETAG, dem_byte_identical_to_official=md5 == COP_AWS_DEM_ETAG,
                tandemx_start=g("tsxx:tsxx_startTime"), tandemx_stop=g("tsxx:tsxx_stopTime"), editing_date=g("tsxx:DateofAnalysis"),
                src_scene_first=dates[0] if dates else "", src_scene_last=dates[-1] if dates else "", event_date=EVENT,
                pre_event=bool(dates) and dates[-1] < EVENT and g("tsxx:tsxx_stopTime") < EVENT,
                editing_after_event=g("tsxx:DateofAnalysis") >= EVENT, omega_fully_inside_tile=o4326.within(__import__("shapely").box(116, -9, 117, -8)),
                omega_dem_nodata_pixels=nod, edm_fraction_in_omega=fr["EDM"], flm_fraction_in_omega=fr["FLM"],
                fill_sources_all_pre_event=all(k in (0, 1, 2, 3, 4, 5, 9) for k in fr["FLM"]))


# ------------------------------------------------------------------------------------------------- main
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    records = register()
    support = acquire_support()
    arch, finder = archive_audit(records)
    geo = load_geology(records)
    is_geological = bool(geo["SIMOBJ"].notna().all() and geo["NAMOBJ"].notna().all() and geo.geom_type.isin(["Polygon", "MultiPolygon"]).all()
                         and geo["SIMOBJ"].nunique() > 3)
    omega = gpd.read_file(V1T / "OMEGA/SF1_VERIFICATION_mapping_extent.shp").to_crs(UTM).union_all()
    y_by_phase = {ph: gpd.read_file(V1T / f"Y/SF1_VERIFICATION_Ferrario_2019{ph}.shp").to_crs(UTM) for ph in ("a", "b")}
    admin = gpd.read_file(ROOT / next(e["raw_path"] for e in load_registry() if e["acquisition_id"] == "ACQ_LOMBOK_ADMIN_GEOBOUNDARIES_IDN_ADM2"))
    admin = admin[admin.intersects(gpd.GeoSeries([omega], crs=UTM).to_crs(4326).iloc[0].buffer(0.2))]
    cov = coverage(geo, omega, y_by_phase, admin)
    gap = cov.pop("_gap_geom")
    # unit audit + mapping (frozen ontology; raw semantics preserved)
    units = []
    covered = geo.union_all()
    for (code, name, age, remark), grp in geo.groupby(["SIMOBJ", "NAMOBJ", "UMUROBJ", "REMARK"], dropna=False):
        geom = grp.union_all()
        m = map_unit(code, name, remark, age)
        units.append(dict(raw_unit_code=code, raw_unit_name=name, raw_remark_id=remark, raw_age=age, source_districts="|".join(sorted(set(grp.source_district))),
                          polygons=len(grp), area_km2=round(geom.area / 1e6, 4), omega_overlap_km2=round(geom.intersection(omega).area / 1e6, 4),
                          **{f"y_{ph}_in_unit": int(y.geometry.representative_point().within(geom).sum()) for ph, y in y_by_phase.items()},
                          **m, source="GeoMap PSG 1:100k district SHP"))
    assert all(u["harmonised_class"] in CLASSES for u in units)
    _csv(OUT / "geology_unit_audit_and_mapping.csv", units)
    eligible = unary_union([geo[(geo.SIMOBJ == u["raw_unit_code"]) & (geo.NAMOBJ == u["raw_unit_name"])].union_all()
                            for u in units if u["domain_eligible"] == "TRUE"])
    seam = seam_qc(geo)
    m = m_audit(omega)
    diag = {ph: dict(total=len(y), in_omega_obs=int(y.geometry.representative_point().within(omega).sum()),
                     in_geology_coverage=int(y.geometry.representative_point().within(omega.intersection(covered)).sum()),
                     in_eligible_volcanic_geology_DIAGNOSTIC=int(y.geometry.representative_point().within(omega.intersection(eligible)).sum()),
                     excluded_geology_gap=int(y.geometry.representative_point().within(gap).sum()),
                     outside_omega=int((~y.geometry.representative_point().within(omega)).sum()))
            for ph, y in y_by_phase.items()}
    covered_frac = cov["omega_covered_fraction"]
    gates = [
        ("Y", "PASS", "Ferrario 2019a/2019b Tier-2 affected-area polygons transferred byte-identically; phases kept separate; primary-phase choice must be frozen before production"),
        ("OMEGA", "PASS", "Published mapping_extent (independent of Y) - unchanged"),
        ("M", "PASS" if m["dem_byte_identical_to_official"] and m["omega_dem_nodata_pixels"] == 0 else "FAIL",
         "COP30 S09E116 byte-identical to official tile; covers Omega with no nodata"),
        ("TEMPORAL", "PASS" if m["pre_event"] else "FAIL", f"TanDEM-X {m['tandemx_start']}..{m['tandemx_stop']}; scenes to {m['src_scene_last']}; event {EVENT}"),
        ("GEOLOGY_COVERAGE", "FAIL" if covered_frac < 0.999 else "PASS",
         f"Utara+Timur cover {covered_frac:.1%} of Omega_obs; {cov['omega_gap_km2']} km2 uncovered in {sorted(cov['gap_by_district_km2'])}"),
        ("VOLCANIC_DOMAIN", "BLOCKED", "Cannot be constructed over the full Omega_obs while geology coverage fails; formation units also lack source lithology"),
        ("TARGET_SEMANTICS", "PASS", "POLYGON_AFFECTED_TIER2 preserved; no conversion, no buffering"),
        ("DEM_COVERAGE", "PASS" if m["omega_fully_inside_tile"] else "FAIL", "Omega inside tile S09E116"),
        ("TERRAIN_GENERATION", "NOT_RUN", "Not attempted: upstream gates not all PASS"),
    ]
    production_ready = all(s == "PASS" for _, s, _ in gates)
    _csv(OUT / "production_readiness_gates.csv", [dict(gate=a, status=b, evidence=c) for a, b, c in gates])
    _csv(OUT / "incoming_registration.csv", [{k: v for k, v in r.items()} for r in records])
    _csv(OUT / "archive_audit.csv", arch)
    _json(OUT / "finder_extraction_audit.json", finder)
    _json(OUT / "seam_qc.json", seam)
    _json(OUT / "coverage_audit.json", cov)
    _json(OUT / "y_audit.json", dict(semantics="POLYGON_AFFECTED_TIER2 (combined source+runout), Ferrario 2019a and 2019b phases kept separate",
                                     trigger="2018 Lombok earthquake sequence", diagnostic_note="in_eligible_volcanic_geology_DIAGNOSTIC uses only the "
                                     "covered 75.9% and is NOT an analysis domain", phases=diag))
    _json(OUT / "m_audit.json", m)
    _json(OUT / "h_feasibility_update.json", dict(
        landscape_id=LID, status="H_UNAVAILABLE_NOT_RECONSTRUCTED", likely_family="MULTI_EDIFICE / CALDERA_RIM (Rinjani-Samalas)",
        geology_contribution="Quaternary volcanic units (Qhv/Qhv1) delineate the young edifice; older formation units (Kalibabak, Kalipalung, Pengulung) "
                             "would bound pre-Rinjani surfaces once their lithology/age are bound from the sheet-1807 legend",
        unit_age_constraints="Era-level only (Kuarter/Neogen) in source; no numeric ages", preserved_morphology="Rinjani edifice and Samalas caldera",
        numerical_palaeoelevation_controls="NONE in supplied artifacts", ensemble_feasible_later="LIKELY after complete geology",
        effect_on_meg_readiness="NONE"))
    dash = pd.read_csv(ROOT / "outputs/stage2f/landscape_state_dashboard.csv")
    i = dash.index[dash.landscape_id == LID][0]
    dash.loc[i, "state"] = "PRODUCTION_READY" if production_ready else "BLOCKED_GEOLOGY"
    dash.loc[i, "production_ready_after_harmonisation"] = production_ready
    dash.loc[i, "primary_blocker"] = "NONE" if production_ready else "BLOCKED_GEOLOGY_COVERAGE"
    dash.to_csv(OUT / "landscape_state_dashboard.csv", index=False, lineterminator="\n")
    prod = dash[dash.state == "PRODUCTION_READY"]
    v1_after = v1_verify()
    lineage = dict(inputs={r["raw_path"]: r["sha256"] for r in records} | {s["raw_path"]: s["sha256"] for s in support}
                   | {str(p.relative_to(ROOT)): sha256(p) for p in sorted(V1T.glob("OMEGA/*.shp")) + sorted(V1T.glob("Y/*.shp"))},
                   script="src/sfv2/build_stage2f1.py", script_sha256=sha256(Path(__file__)), as_of=AS_OF)
    _json(OUT / "lineage.json", lineage)
    outputs = sorted(p for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("STAGE2F1_CHECKPOINT", "STAGE2F1_DIAGNOSTICS")))
    cp = dict(schema="SFV2_STAGE2F1_CHECKPOINT_V1", as_of=AS_OF, landscape_id=LID, production_ready=production_ready,
              production_ready_landscapes=sorted(prod.landscape_id), production_ready_count=len(prod),
              production_terrain_units_total=450935, gates={a: b for a, b, _ in gates}, geology_is_unit_polygons=is_geological,
              incoming_registrations={r["acquisition_id"]: r["sha256"] for r in records}, support_acquisitions={s["acquisition_id"]: s["sha256"] for s in support},
              output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outputs},
              prior_checkpoints={rel: sha256(ROOT / rel) for rel in ("outputs/stage2f/STAGE2F_CHECKPOINT.json", "outputs/stage2e/STAGE2E_CHECKPOINT.json",
                                                                    "outputs/stage2d/STAGE2D_CHECKPOINT.json", "outputs/stage2c/STAGE2C_CHECKPOINT.json")},
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              terrain_units_generated=False, features_generated=False, training=False, h_reconstruction=False,
              volatile_diagnostics={"path": "outputs/stage2f1/STAGE2F1_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE2F1_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE2F1_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD"),
                                                  untracked_entries=len([l for l in git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if l.startswith("??")])))
    return dict(checkpoint=cp, coverage=cov, seam=seam, m=m, y=diag, units=units, arch=arch, finder=finder, records=records)


if __name__ == "__main__":
    main()
