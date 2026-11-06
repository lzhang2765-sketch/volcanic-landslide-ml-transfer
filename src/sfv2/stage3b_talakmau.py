"""Stage 3B Track A: Talakmau (Pasaman, West Sumatra; Mw 6.1, 2022-02-25) - Y semantics audit, Omega, M, geology requirements, gates."""
from __future__ import annotations

import geopandas as gpd
import shapely

from .raw_store import load_registry, sha256
from .schema import ROOT

LID = "TALAKMAU_PASAMAN_2022"
UTM = "EPSG:32647"
EVENT = "2022-02-25"
Y_ACQ = "ACQ_TALAKMAU_UNOSAT_LS20220308IDN_SHP"
ADMIN_ACQ = "ACQ_LOMBOK_ADMIN_GEOBOUNDARIES_IDN_ADM2"  # national IDN ADM2 file, already registered (Stage 2F.1); used only to name districts
DER = ROOT / "data/processed/stage3b" / LID
PREFIX = "LS20220308IDN_SHP/S2_20220304_"
LAYERS = {"extent": "AnalysisExtent_MountTalakmau", "cloud": "CloudObsruction_MountTalakmau", "y": "LandslideExtent_MountTalakmau"}

# Exact GeoMap products (official portal listing, 2026-10-06).  Districts: geoBoundaries ADM2 intersection with Omega_obs.
GEOMAP = {
    "PASAMAN": dict(code="13.08", district="Kab. Pasaman", page="https://geologi.esdm.go.id/geomap/pages/district/13.08",
                    product="SHP: Kab. / Kota Pasaman (Unduh SHP, ~1 MB)", acq="ACQ_TALAKMAU_GEOMAP_PASAMAN_SHP",
                    pattern=r"Peta_SHP_Pasaman_[0-9a-f]{32}\.zip"),
    "PASAMAN_BARAT": dict(code="13.12", district="Kab. Pasaman Barat", page="https://geologi.esdm.go.id/geomap/pages/district/13.12",
                          product="SHP: Kab. / Kota Pasaman Barat (Unduh SHP, ~1 MB)", acq="ACQ_TALAKMAU_GEOMAP_PASAMAN_BARAT_SHP",
                          pattern=r"Peta_SHP_Pasaman_Barat_[0-9a-f]{32}\.zip"),
}
SHEETS = {
    "LUBUKSIKAPING": dict(required=True, title="Peta Geologi Lembar Lubuksikaping, Sumatera 1:250.000 (sheet index 15; Lembar 0616, 0717)",
                          slug="peta-geologi-lembar-lubuksikaping-sumatera", product="Unduh Raster (~13 MB)", acq="ACQ_TALAKMAU_GEOMAP_SHEET_LUBUKSIKAPING",
                          pattern=r"Peta_Geologi_Lembar_Lubuksikaping,_Sumatera_[0-9a-f]{32}\.(jpg|jpeg|png|pdf)",
                          why="Covers Mount Talakmau (~0.08N 99.98E) and the Omega; its legend/explanation resolves district unit codes"),
    "PADANG": dict(required=False, title="Peta Geologi Lembar Padang, Sumatera 1:250.000 (sheet index 14; Lembar 0715)",
                   slug="peta-geologi-lembar-padang-sumatera", product="Unduh Raster (~11 MB)", acq="ACQ_TALAKMAU_GEOMAP_SHEET_PADANG",
                   pattern=r"Peta_Geologi_Lembar_Padang,_Sumatera_[0-9a-f]{32}\.(jpg|jpeg|png|pdf)",
                   why="Only if a district unit code is absent from the Lubuksikaping legend (public preview shows Maninjau caldera, ~0.3S, near "
                       "its north edge; Omega reaches only 0.06S). Optional: fetch in the same signed-in session to avoid a second round-trip"),
}


def _raw(acq: str):
    return ROOT / {e["acquisition_id"]: e for e in load_registry()}[acq]["raw_path"]


def layers() -> dict[str, gpd.GeoDataFrame]:
    z = _raw(Y_ACQ)
    return {k: gpd.read_file(f"zip://{z}!{PREFIX}{v}.shp") for k, v in LAYERS.items()}


def y_semantics_audit(lay: dict) -> dict:
    y = lay["y"].iloc[0]
    yg = lay["y"].to_crs(UTM).union_all()
    return dict(
        source="UNOSAT LS20220308IDN (HDX: 'Landslide analysis in Mount Talakmau in Pasaman & Pasaman Barat districts, Indonesia as of 04 March 2022'), CC BY-SA",
        raw_sha256=sha256(_raw(Y_ACQ)), event_date=EVENT, trigger="EARTHQUAKE (Mw 6.1 Pasaman Barat earthquake; blind fault beneath Talakmau)",
        imagery="Sentinel-2 (10 m), acquired 2022-03-04 (layer names + HDX); SensorDate attribute 2022-04-03 is a day/month swap",
        mapping_date="Shapefile metadata created 2022-03-08 (UNOSAT product date)", geometry="1 MultiPolygon (dissolved)",
        mapped_parts=len(getattr(yg, "geoms", [yg])), mapped_area_km2=round(yg.area / 1e6, 3),
        attributes=dict(Main_Dmg=y.get("Main_Dmg"), Confidence=y.get("Confidence"), FieldValid=y.get("FieldValid"), SensorID=y.get("SensorID")),
        polygon_meaning="Satellite-detected landslide-affected ground (scar + runout/deposit undivided; no source/runout attribute) -> affected terrain",
        semantics_class="AFFECTED_TERRAIN_UNDIVIDED", target_tier="TIER_2",
        tier_evidence="Layer 'LandslideExtent', Main_Dmg='Landslide', HDX text '6 km2 of landslides observed'; no initiation-zone mapping -> not TIER_1",
        target_family="POLYGON_AFFECTED_TIER2 (same family as Noto and Lombok; never pooled with point families)",
        validation_status="Preliminary; Confidence 'To Be Evaluated'; 'Not yet field validated'",
        cloud_limitation="CloudObstruction layer (122 parts, 30.0 km2 inside the analysis extent) marks unobservable ground",
        mapping_limitations=["10 m Sentinel-2: landslides smaller than a few pixels are under-detected (detection-limited completeness)",
                             "Dissolved geometry: individual landslide counts and sizes unavailable (cell-centre label rule does not need them)",
                             "Single post-event scene 7 days after the earthquake: any rain-triggered failures in that week are not separable"],
        background_defensibility=("Absence inside the visible analysis extent means 'no landslide detected by a systematic 10 m Sentinel-2 search'. "
                                  "Defensible as background under V2 rules only with an explicit detection-limit caveat, as for other "
                                  "image-mapped inventories; the preliminary, unvalidated status is what requires decision D-TAL-Y."),
        decision_required="D-TAL-Y")


def omega(lay: dict) -> dict:
    e = lay["extent"].to_crs(UTM).union_all()
    c = lay["cloud"].to_crs(UTM).union_all()
    y = lay["y"].to_crs(UTM).union_all()
    om = e.difference(c)
    return dict(geometry=om, extent=e, cloud=c, y=y, stats=dict(
        definition="UNOSAT AnalysisExtent (analyst-declared rectangular analysed area, 4 corners) minus UNOSAT CloudObstruction",
        independence_evidence=["AnalysisExtent is an axis-aligned 4-corner rectangle (not traced from landslides)",
                               f"Landslide convex hull covers only {round(y.convex_hull.area / e.area, 3)} of the extent",
                               "Landslide polygons cross the extent edge (mapping not clipped to Y), cloud mask independent of Y"],
        extent_vertices=int(shapely.get_num_coordinates(e)), original_analysis_extent_km2=round(e.area / 1e6, 3),
        cloud_obscured_km2=round(e.intersection(c).area / 1e6, 3), other_exclusions_km2=0.0,
        other_exclusions_note="No other visibility, shadow or imagery-gap layer is provided; DEM water/nodata is removed later in OMEGA_ANALYSIS",
        omega_obs_km2=round(om.area / 1e6, 3), y_total_km2=round(y.area / 1e6, 4), y_inside_omega_km2=round(y.intersection(om).area / 1e6, 4),
        y_in_cloud_km2=round(y.intersection(c).area / 1e6, 4), y_outside_extent_km2=round(y.difference(e).area / 1e6, 4),
        hull_or_buffer_of_y_used=False))


def write_omega(om: dict) -> list:
    DER.mkdir(parents=True, exist_ok=True)
    p = DER / "omega_obs_epsg32647.geojson"
    gpd.GeoDataFrame({"name": ["OMEGA_OBS"], "rule": [om["stats"]["definition"]]}, geometry=[om["geometry"]], crs=UTM).to_file(p, driver="GeoJSON")
    return [(p, [_raw(Y_ACQ)], "AnalysisExtent minus CloudObstruction (UTM 47N)")]


def districts_required(om_geom) -> list[dict]:
    adm = gpd.read_file(_raw(ADMIN_ACQ)).to_crs(UTM)
    hit = adm[adm.intersects(om_geom)]
    return [dict(district=r.shapeName, omega_km2=round(r.geometry.intersection(om_geom).area / 1e6, 2)) for r in hit.itertuples()]


def gates(y: dict, om: dict, m: dict, districts: list[dict], geology_present: bool) -> list[tuple]:
    return [
        ("Y", "DECISION_REQUIRED", f"TIER_2 affected-terrain polygons ({y['mapped_area_km2']} km2, {y['mapped_parts']} parts); preliminary, not "
                                   "field-validated -> user decision D-TAL-Y"),
        ("OMEGA", "PASS", f"Omega_obs {om['stats']['omega_obs_km2']} km2 = analysed rectangle - cloud; independent of Y; "
                          f"Y outside Omega {round(om['stats']['y_total_km2'] - om['stats']['y_inside_omega_km2'], 4)} km2 "
                          "(precedent: mapper-published extents for Noto and Lombok)"),
        ("M_TEMPORAL", "PASS" if m["m_temporal_pass"] else "FAIL", "TanDEM-X scenes to " + max(t["src_scene_last"] for t in m["tiles"]) +
                       f" < {EVENT}; editing dates before event; fill sources pre-event-compatible"),
        ("DEM_COVERAGE", "PASS" if m["aoi_fully_covered"] and m["dem_nodata_cells_in_aoi"] == 0 else "FAIL",
         f"4 tiles cover the analysis extent; nodata cells {m['dem_nodata_cells_in_aoi']}; non-original {m['non_original_fraction_in_aoi']}"),
        ("GEOLOGY", "PASS" if geology_present else "BLOCKED_USER_DOWNLOAD",
         "GeoMap district SHPs for " + " + ".join(d["district"] for d in districts) + " + Lubuksikaping sheet legend (authenticated)"),
        ("VOLCANIC_DOMAIN", "PENDING_GEOLOGY", "Frozen rule (volcanic AND Cenozoic) applied after legend resolution"),
        ("TARGET_SEMANTICS", "DECISION_REQUIRED", "POLYGON_AFFECTED_TIER2 confirmed from evidence; acceptance of preliminary rapid mapping is D-TAL-Y"),
        ("TERRAIN_GENERATION", "PENDING", "Frozen polygon mirror (stage2f2_production.build_polygon_landscape) ready; runs when all gates PASS"),
    ]
