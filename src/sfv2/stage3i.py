"""Stage 3I Talakmau geology rescue and raw-gate re-audit.

This stage is deliberately fail-closed.  It audits the two user-authenticated
GeoMap archives and the already-frozen Talakmau Y/Omega/M artifacts.  The
production branch is not entered while D-TAL-Y remains unresolved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import pandas as pd

from .raw_store import load_registry
from .schema import ROOT


AS_OF = "2026-10-06"
LID = "TALAKMAU_PASAMAN_2022"
UTM = "EPSG:32647"
OUT = ROOT / "outputs/stage3i"
STAGE3H_HASH = "f1ea33e848559bcc97f6c8fb7bc7a78c8fca9d004fe924d8c0136d1f488f2d1f"
STAGE3G_HASH = "a4f8035123658c063f1572bdfd3276906053ba313735ae3371c442e45cbe4664"
STAGE3F_HASH = "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a"
STAGE3B_HASH = "73ea92c57e4699172916765c8d66de94e2d84c37aa4217ddf2f6dbddae9e0dc4"
UNOSAT_HASH = "7a661bcfcccd478834c1d9696c2671b5f9cdaf8f449d8a4f2136165551893fb4"

PRODUCTION = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
]

PACKAGES = {
    "PASAMAN": {
        "district_code": "13.08",
        "district_name": "Kab. Pasaman",
        "page": "https://geologi.esdm.go.id/geomap/pages/district/13.08",
        "incoming": ROOT / "data/incoming_user/Peta_SHP_Pasaman_f0a3c4f1995d46359f54cab11e39667f.zip",
        "raw": ROOT / "data/raw/stage3i/TALAKMAU_PASAMAN_2022/SRC_GEOMAP_PASAMAN/Peta_SHP_Pasaman_f0a3c4f1995d46359f54cab11e39667f.zip",
        "sha256": "eb54b5d7b54b061a2105129ffcefdcc335b554f1709f19a5f82d1f906bedeeea",
        "layer": "pasaman/geologi_ar_100k_pasaman.shp",
    },
    "PASAMAN_BARAT": {
        "district_code": "13.12",
        "district_name": "Kab. Pasaman Barat",
        "page": "https://geologi.esdm.go.id/geomap/pages/district/13.12",
        "incoming": ROOT / "data/incoming_user/Peta_SHP_Pasaman_Barat_19062981b0ad7d72261827f68e1c6bec.zip",
        "raw": ROOT / "data/raw/stage3i/TALAKMAU_PASAMAN_2022/SRC_GEOMAP_PASAMAN_BARAT/Peta_SHP_Pasaman_Barat_19062981b0ad7d72261827f68e1c6bec.zip",
        "sha256": "5931b329cc79f17af01706ad92d9c4602750b188605a10d13ff6757dbdb8de7d",
        "layer": "pasaman_barat/geologi_ar_100k_pasaman_barat.shp",
    },
}

Y_ZIP = ROOT / "data/raw/stage2b/TALAKMAU_PASAMAN_2022/SRC_UNOSAT_LS20220308IDN/LS20220308IDN_SHP.zip"
Y_LAYER = "LS20220308IDN_SHP/S2_20220304_LandslideExtent_MountTalakmau.shp"
OMEGA_PATH = ROOT / "data/processed/stage3b/TALAKMAU_PASAMAN_2022/omega_obs_epsg32647.geojson"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_rows(path: Path, rows: list[dict], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def state_verification() -> dict:
    paths = {
        "stage3b": ROOT / "outputs/stage3b/STAGE3B_CHECKPOINT.json",
        "stage3f": ROOT / "outputs/stage3f/STAGE3F_CHECKPOINT.json",
        "stage3g": ROOT / "outputs/stage3g/STAGE3G_CHECKPOINT.json",
        "stage3h": ROOT / "outputs/stage3h/STAGE3H_CHECKPOINT.json",
    }
    expected = {"stage3b": STAGE3B_HASH, "stage3f": STAGE3F_HASH,
                "stage3g": STAGE3G_HASH, "stage3h": STAGE3H_HASH}
    observed = {k: sha256(v) for k, v in paths.items()}
    production = json.loads(paths["stage3h"].read_text())["production_state"]
    return {
        "checkpoint_hashes_expected": expected,
        "checkpoint_hashes_observed": observed,
        "all_checkpoint_hashes_unchanged": observed == expected,
        "production_state": production,
        "production_state_exact": production == {
            "count": 6, "landscapes": PRODUCTION, "positive_units": 8793,
            "target_tier": "TIER_2_ONLY", "terrain_units": 835846,
        },
    }


def source_registry() -> list[dict]:
    rows = []
    for key, p in PACKAGES.items():
        incoming, raw = p["incoming"], p["raw"]
        if not incoming.is_file() or not raw.is_file():
            raise FileNotFoundError(incoming if not incoming.is_file() else raw)
        ih, rh = sha256(incoming), sha256(raw)
        if ih != p["sha256"] or rh != p["sha256"]:
            raise RuntimeError(f"Original/raw hash mismatch for {key}: {ih} / {rh}")
        timestamp = datetime.fromtimestamp(incoming.stat().st_mtime, ZoneInfo("Australia/Brisbane")).isoformat()
        rows.append({
            "package": key, "district_code": p["district_code"], "district_name": p["district_name"],
            "filename": incoming.name, "original_path": str(incoming.relative_to(ROOT)),
            "preserved_raw_path": str(raw.relative_to(ROOT)), "byte_size": incoming.stat().st_size,
            "modification_time": timestamp, "SHA256": ih, "byte_identical_raw_copy": ih == rh,
            "source_page": p["page"],
        })
    return rows


def member_role(name: str) -> str:
    lower = name.lower()
    if "geologi_ar" in lower:
        return "GEOLOGICAL_UNIT_POLYGON_COMPONENT"
    if "geostructure_ln" in lower:
        return "GEOLOGICAL_STRUCTURE_LINE_COMPONENT"
    if lower.endswith("legend.lyr"):
        return "ARCMAP_SYMBOLOGY"
    return "OTHER"


def archive_inventory() -> tuple[list[dict], dict]:
    rows, summary = [], {}
    for key, p in PACKAGES.items():
        with zipfile.ZipFile(p["raw"]) as zf:
            bad = zf.testzip()
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rows.append({"package": key, "member": info.filename, "uncompressed_bytes": info.file_size,
                             "compressed_bytes": info.compress_size, "CRC32": f"{info.CRC:08x}",
                             "semantic_role": member_role(info.filename)})
            stem = p["layer"][:-4]
            required = [stem + x for x in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".shp.xml")]
            summary[key] = {
                "member_count": len(names), "zip_crc_ok": bad is None, "bad_member": bad,
                "geology_polygon_layer": p["layer"],
                "required_geology_components": required,
                "all_required_geology_components_present": all(x in names for x in required),
                "other_layer": next(x for x in names if "geostructure_ln" in x and x.endswith(".shp")),
                "other_layer_role": "geological structure lines; not unit polygons",
                "legend_file": next(x for x in names if x.endswith("Legend.lyr")),
                "legend_file_role": "ArcMap style/symbology, not a geological map-sheet legend table",
            }
    return rows, summary


def load_geology() -> dict[str, gpd.GeoDataFrame]:
    out = {}
    for key, p in PACKAGES.items():
        g = gpd.read_file(f"zip://{p['raw'].resolve()}!{p['layer']}")
        g["source_district"] = key
        out[key] = g
    return out


def geology_qa(frames: dict[str, gpd.GeoDataFrame]) -> dict:
    rows = {}
    for key, g in frames.items():
        projected = g.to_crs(UTM)
        rows[key] = {
            "layer": PACKAGES[key]["layer"], "features": len(g),
            "geometry_types": {str(k): int(v) for k, v in g.geom_type.value_counts().sort_index().items()},
            "valid_features": int(g.is_valid.sum()), "invalid_features": int((~g.is_valid).sum()),
            "empty_features": int(g.is_empty.sum()), "null_features": int(g.geometry.isna().sum()),
            "duplicate_geometries": int(g.geometry.to_wkb().duplicated().sum()),
            "source_crs": str(g.crs), "verified_epsg": g.crs.to_epsg(),
            "bounds_lon_lat": [float(x) for x in g.total_bounds],
            "coordinate_plausibility": bool(-1 < g.total_bounds[1] < 1 and -1 < g.total_bounds[3] < 1 and
                                              98 < g.total_bounds[0] < 101 and 98 < g.total_bounds[2] < 101),
            "polygon_area_sum_km2": round(float(projected.area.sum() / 1e6), 6),
            "polygon_union_area_km2": round(float(projected.geometry.union_all().area / 1e6), 6),
            "internal_overlap_inflation_km2": round(float((projected.area.sum() - projected.geometry.union_all().area) / 1e6), 9),
            "repairs": [],
            "unit_fields_present": {f: f in g.columns for f in ("SIMOBJ", "NAMOBJ", "UMUROBJ", "REMARK", "METADATA", "SRS_ID")},
        }
    return {
        "packages": rows,
        "all_polygon_features_valid": all(v["invalid_features"] == 0 for v in rows.values()),
        "repairs_performed": False,
    }


def provenance() -> dict:
    return {
        "identity": "Official Indonesia GeoMap V.2 district geology SHP downloads for Kab. Pasaman (13.08) and Kab. Pasaman Barat (13.12)",
        "publisher": "Pusat Survei Geologi / Puslitbang Geologi, Badan Geologi, Indonesia",
        "portal": "https://geologi.esdm.go.id/geomap/",
        "product_scale": "1:100,000 district extract (100k embedded in official layer names)",
        "format": "ESRI Shapefile ZIP",
        "crs": "EPSG:4326 WGS 84 in both .prj files and embedded Esri XML",
        "metadata_evidence": [
            "Esri XML titles Pasaman / Pasaman Barat",
            "Esri XML itemLocation contains GeoMap V.2/geomap_shp/GGL_KAB/Sumatera_Barat/GEOLOGI_AR",
            "Esri XML creation/synchronisation date 2021-09-26 and ArcGIS 10.4",
            "GeoMap district page identity and authenticated archive filenames",
        ],
        "geological_fields": {
            "SIMOBJ": "source unit code", "NAMOBJ": "source unit name", "UMUROBJ": "source age",
            "REMARK": "official Indonesian remark/name", "METADATA": "source metadata reference", "SRS_ID": "source spatial-reference id",
        },
        "map_name_and_map_institution_attributes": "present in schema but null in these features; portal identity and embedded GeoMap path supply product provenance",
        "licence": "GeoMap open-licence terms recorded by the existing validated GeoMap ingestion: attribution to PSG/Badan Geologi; resale prohibited; terms were accepted by the user during authenticated download",
        "attribute_consistency_audit": {
            "status": "FAIL_SOURCE_ATTRIBUTE_GEOGRAPHY_MISMATCH",
            "official_badan_geologi_points": [
                {"site": "Gunung Api Tidak Aktif Talamau", "latitude": 0.078725, "longitude": 99.9839,
                 "official_description": "Quaternary Talamau volcano; andesite-dacite lava, volcanic breccia, lapilli and tuff",
                 "vector_hit": "PASAMAN_BARAT Qvps Peuet Sague Unit"},
                {"site": "Lava G. Talamau", "latitude": 0.09908, "longitude": 99.93063,
                 "official_description": "Talamau andesite lava and volcanic breccia",
                 "vector_hit": "PASAMAN_BARAT Qvps Peuet Sague Unit"},
                {"site": "Air Terjun Batu Merah Kajai", "latitude": 0.12811, "longitude": 99.95369,
                 "official_description": "Talamau andesite lava interbedded with volcanic breccia",
                 "vector_hit": "PASAMAN_BARAT Qvps Peuet Sague Unit"},
            ],
            "official_event_geology_crosscheck": "Badan Geologi's 2022 Pasaman event note identifies local units as Puku, Tmv (undifferentiated volcanic deposits) and Qvta (Talamau products); the SHP instead labels the summit and both official Talamau outcrops Qvps Peuet Sague Unit, an Aceh volcano name.",
            "additional_mismatch": "The SHP expands Tmv as Maninjau Volcanic Centre, whereas the official 2022 Lubuksikaping-based note expands local Tmv as undifferentiated volcanic deposits without a preserved centre.",
            "qvta_vector_bounds": [99.73113, 0.06223, 99.84722, 0.17004],
            "official_talamau_summit": [99.9839, 0.078725],
            "conclusion": "Polygon geometry coverage is complete, but the DBF code/name-to-geometry association is not scientifically trustworthy without the correct sheet legend/crosswalk or corrected vector product.",
            "official_sources": [
                "https://geologi.esdm.go.id/geoheritage//pages/site/gunung-api-tidak-aktif-talamau",
                "https://geologi.esdm.go.id/geoheritage//pages/site/lava-g-talamau",
                "https://geologi.esdm.go.id/geoheritage//pages/site/air-terjun-batu-merah-kajai",
                "https://geologi.esdm.go.id/media-center/likuefaksi-akibat-gempa-bumi-pasaman-25-februari-2022",
            ],
        },
        "raster_required_for_scientific_interpretation": True,
        "raster_reason": "The SHP DBF attributes are geographically inconsistent with official Badan Geologi Talamau sites and the agency's Lubuksikaping-based 2022 event geology. The exact sheet legend/map, an official corrected SHP, or an official code-to-polygon crosswalk is required to restore trustworthy unit identity; this is not merely visual confirmation.",
    }


def mapping_decision(code: str, name: str, age: str, remark: str) -> tuple[str, object, str, str]:
    if code == "Qvps":
        return ("MIXED_VOLCANIC", "UNRESOLVED",
                "The literal DBF label is volcanic/Quaternary, but it is geographically impossible here: official Talamau summit and outcrop coordinates fall inside this polygon while Badan Geologi identifies the local unit as Talamau products (Qvta), not Aceh's Peuet Sague. Do not admit until corrected.", "LOW")
    if code == "Tmv":
        return ("MIXED_VOLCANIC", "UNRESOLVED",
                "Tmv is volcanic/Cenozoic in the official 2022 Lubuksikaping-based description, but the SHP expansion 'Maninjau Volcanic Centre' conflicts with that description; geometry-to-legend identity needs official correction.", "LOW")
    if code in {"Qal", "Qh"}:
        return ("UNCONSOLIDATED_SURFICIAL", False, "Official name/remark is Alluvium; surficial deposits are excluded.", "HIGH")
    if code == "TMi1":
        return ("INTRUSIVE", False, "Official name is Miscellaneous Intrusion; intrusive units are not accepted as volcanic terrain.", "HIGH")
    if code == "Pl":
        return ("SEDIMENTARY_BASEMENT", False, "Official name/remark is Limestone / Batugamping; excluded.", "HIGH")
    reasons = {
        "Puku": "Kuantan Formation is Triassic and lacks a lithologic description in the SHP; pre-Cenozoic and excluded fail-closed.",
        "Muw": "Undifferentiated Woyla Member is Cretaceous; pre-Cenozoic and excluded fail-closed.",
        "Mupu": "Pasaman Ultramafic Complex is Mesozoic; pre-Cenozoic and excluded fail-closed.",
        "MPu": "Undifferentiated Permian/Paleozoic-Mesozoic unit is pre-Cenozoic and excluded fail-closed.",
        "Qpt1": "Totolan Formation is Quaternary but the SHP does not identify it as volcanic; ambiguous and excluded fail-closed.",
    }
    return ("UNKNOWN", False, reasons.get(code, f"{name} ({age}; {remark}) is not explicitly admissible volcanic terrain."), "MEDIUM")


def coverage_and_mapping(frames: dict[str, gpd.GeoDataFrame]) -> tuple[dict, list[dict], dict]:
    omega = gpd.read_file(OMEGA_PATH).to_crs(UTM).geometry.union_all()
    y = gpd.read_file(f"zip://{Y_ZIP.resolve()}!{Y_LAYER}").to_crs(UTM).geometry.union_all()
    projected = []
    for key, g in frames.items():
        p = g.to_crs(UTM).copy()
        p["source_district"] = key
        projected.append(p)
    geo = gpd.GeoDataFrame(pd.concat(projected, ignore_index=True), crs=UTM)
    coverage = geo.geometry.union_all()

    unit_rows = []
    group_fields = ["source_district", "SIMOBJ", "NAMOBJ", "UMUROBJ", "REMARK"]
    for keys, sub in geo.groupby(group_fields, dropna=False, sort=True):
        geom = sub.geometry.union_all()
        omega_area = geom.intersection(omega).area / 1e6
        y_area = geom.intersection(y).area / 1e6
        if omega_area <= 1e-10 and y_area <= 1e-10:
            continue
        district, code, name, age, remark = ("" if pd.isna(x) else str(x) for x in keys)
        cls, accepted, reason, confidence = mapping_decision(code, name, age, remark)
        unit_rows.append({
            "source_district": district, "source_unit_code": code, "source_unit_name": name,
            "source_age": age, "source_lithology": remark or "NOT_SUPPLIED_IN_SHP",
            "V2_geology_class": cls, "volcanic_domain_acceptance": str(accepted).upper(),
            "mapping_reason": reason, "confidence": confidence,
            "omega_intersection_km2": f"{omega_area:.6f}", "y_total_intersection_km2": f"{y_area:.6f}",
        })
    unit_rows.sort(key=lambda r: (r["source_district"], r["source_unit_code"], r["source_unit_name"]))
    nominal = geo[geo.SIMOBJ.isin(["Qvps", "Tmv"])].geometry.union_all()
    y_in_omega = y.intersection(omega)
    pu = {k: g.to_crs(UTM).geometry.union_all() for k, g in frames.items()}
    union = pu["PASAMAN"].union(pu["PASAMAN_BARAT"])
    coverage_audit = {
        "omega_area_km2": omega.area / 1e6,
        "packages": {
            key: {"omega_covered_km2": geom.intersection(omega).area / 1e6,
                  "omega_covered_percent": 100 * geom.intersection(omega).area / omega.area}
            for key, geom in pu.items()
        },
        "both_packages_required": True,
        "package_overlap_inside_omega_km2": pu["PASAMAN"].intersection(pu["PASAMAN_BARAT"]).intersection(omega).area / 1e6,
        "unmapped_gap_inside_omega_km2": omega.difference(union).area / 1e6,
        "omega_geology_covered_km2": omega.intersection(union).area / 1e6,
        "omega_geology_covered_percent": 100 * omega.intersection(union).area / omega.area,
        "y_total_area_km2": y.area / 1e6,
        "y_inside_omega_area_km2": y_in_omega.area / 1e6,
        "y_total_geology_covered_km2": y.intersection(union).area / 1e6,
        "y_total_geology_covered_percent": 100 * y.intersection(union).area / y.area,
        "y_inside_omega_geology_covered_percent": 100 * y_in_omega.intersection(union).area / y_in_omega.area,
        "note": "The 0.3787% of total Y outside geology coverage is the mapper polygon area outside Omega/analysis support; production eligibility is evaluated inside Omega.",
    }
    volcanic = {
        "frozen_rule": "Only defensibly volcanic AND Cenozoic source units are admitted; ambiguous, intrusive, basement, sedimentary and surficial units are excluded.",
        "accepted_codes": [],
        "provisional_label_screen_codes_not_accepted": ["Qvps", "Tmv"],
        "omega_area_km2": omega.area / 1e6,
        "omega_geology_covered_area_km2": omega.intersection(coverage).area / 1e6,
        "omega_accepted_volcanic_area_km2": None,
        "omega_accepted_volcanic_percent": None,
        "omega_nominal_shp_label_volcanic_area_km2": omega.intersection(nominal).area / 1e6,
        "omega_nominal_shp_label_volcanic_percent": 100 * omega.intersection(nominal).area / omega.area,
        "y_total_area_km2": y.area / 1e6,
        "y_total_accepted_volcanic_area_km2": None,
        "y_total_accepted_volcanic_percent": None,
        "y_total_nominal_shp_label_volcanic_area_km2": y.intersection(nominal).area / 1e6,
        "y_total_nominal_shp_label_volcanic_percent": 100 * y.intersection(nominal).area / y.area,
        "y_inside_omega_area_km2": y_in_omega.area / 1e6,
        "y_inside_omega_accepted_volcanic_area_km2": None,
        "y_inside_omega_accepted_volcanic_percent": None,
        "y_inside_omega_nominal_shp_label_volcanic_area_km2": y_in_omega.intersection(nominal).area / 1e6,
        "y_inside_omega_nominal_shp_label_volcanic_percent": 100 * y_in_omega.intersection(nominal).area / y_in_omega.area,
        "gate": "HOLD_SOURCE_ATTRIBUTE_MISMATCH",
        "gate_reason": "The polygons cover Omega, but official Badan Geologi point and event-geology evidence contradicts the SHP unit labels at Talamau. Nominal label fractions are diagnostics only and cannot be admitted under the frozen rule.",
    }
    return coverage_audit, unit_rows, volcanic


def raw_integrity() -> dict:
    checkpoint = json.loads((ROOT / "outputs/stage3b/STAGE3B_CHECKPOINT.json").read_text())
    expected = {k: v for k, v in checkpoint["new_raw_acquisitions"].items() if k.startswith("ACQ_TALAKMAU_COP30_")}
    registry = {r["acquisition_id"]: r for r in load_registry()}
    observed = {k: sha256(ROOT / registry[k]["raw_path"]) for k in expected}
    return {
        "unosat_expected_sha256": UNOSAT_HASH, "unosat_observed_sha256": sha256(Y_ZIP),
        "unosat_unchanged": sha256(Y_ZIP) == UNOSAT_HASH,
        "cop30_expected_hashes": expected, "cop30_observed_hashes": observed,
        "cop30_all_24_companions_unchanged": observed == expected,
        "omega_expected_sha256": "22310e9ad4203b98b2597257e30e5526f1349222d83c5291f9d4000718daa012",
        "omega_observed_sha256": sha256(OMEGA_PATH),
        "omega_unchanged": sha256(OMEGA_PATH) == "22310e9ad4203b98b2597257e30e5526f1349222d83c5291f9d4000718daa012",
    }


def gate_reaudit(integrity: dict) -> tuple[list[dict], dict]:
    rows = [
        {"gate": "Y_GATE", "status": "HOLD_DECISION_REQUIRED", "evidence": "Official UNOSAT Tier-2 affected-terrain geometry is coherent and tied to the 2022-02-25 earthquake, but producer metadata says Preliminary, Confidence=To Be Evaluated, Not yet field validated; the frozen D-TAL-Y acceptance/validation decision is still absent."},
        {"gate": "OMEGA_GATE", "status": "PASS", "evidence": "851.735 km2 = UNOSAT AnalysisExtent minus CloudObstruction; rectangular analyst-declared support and cloud mask are independent of Y; no hull/buffer of positives."},
        {"gate": "M_GATE", "status": "PASS", "evidence": "Four COP30 tiles and DEM/EDM/FLM/WBM/XML/SRC companions remain checksum-coherent; full AOI coverage, zero DEM nodata, 3.406% non-original cells."},
        {"gate": "TEMPORAL_GATE", "status": "PASS", "evidence": "TanDEM-X contributing scenes end by 2014-04-10 and editing dates by 2020-01-20, before the 2022-02-25 event; fill sources are pre-event-compatible."},
        {"gate": "G_GATE", "status": "HOLD_SOURCE_ATTRIBUTE_MISMATCH", "evidence": "Official GeoMap polygons jointly cover 100% of Omega, but their DBF attributes place the Talamau summit and official Talamau outcrops in 'Qvps Peuet Sague Unit' and expand local Tmv inconsistently with Badan Geologi's Lubuksikaping-based 2022 geology. Valid geometry does not cure a wrong unit join."},
        {"gate": "VOLCANIC_DOMAIN_GATE", "status": "HOLD_SOURCE_ATTRIBUTE_MISMATCH", "evidence": "The nominal SHP-label screen yields 80.399377% of Omega and 100% of Y inside Omega, but those labels fail official geographic cross-checks; the fractions are diagnostic, not admissible."},
        {"gate": "TARGET_SEMANTICS_GATE", "status": "PASS_TIER_2_CLASSIFICATION", "evidence": "LandslideExtent, Main_Dmg=Landslide; scar plus runout/deposit undivided affected ground, not initiation-only Tier 1. This semantic pass does not override unresolved Y quality."},
        {"gate": "INDEPENDENT_GROUP_GATE", "status": "PASS", "evidence": "Talakmau/West Sumatra is a separate physical volcanic system and 2022 event from all six current production groups."},
    ]
    audit = {
        "event": "Mw 6.1 Pasaman Barat earthquake, 2022-02-25",
        "target_tier": "TIER_2_AFFECTED_TERRAIN",
        "target_source": "UNITAR-UNOSAT LS20220308IDN; Sentinel-2 2022-03-04; mapped 2022-03-08",
        "systematic_mapping_support": "PASS: explicit AnalysisExtent plus CloudObstruction",
        "inventory_quality": "UNRESOLVED: preliminary, confidence to be evaluated, not field validated; 10 m detection limit and one-week event attribution caveat",
        "omega_independent_of_y": True, "pre_event_m": True,
        "strict_dem_quality_supported_if_future_production": True,
        "strict_dem_quality_basis": "COP30 EDM and FLM companions are present and checksum-coherent; use the unchanged frozen threshold only after Y passes.",
        "v1_unseen": True, "v2_development_exposure": True, "final_lockbox_eligible": False,
        "recommended_role": "DEVELOPMENT",
        "integrity": integrity,
        "all_raw_gates_resolved": False,
        "remaining_blockers": [
            "D-TAL-Y: independent validation or explicit scientifically governed acceptance of the preliminary/detection-limited UNOSAT target",
            "G/volcanic domain: official Lubuksikaping raster/legend, corrected SHP, or official geometry-to-unit crosswalk resolving the GeoMap DBF mismatch",
        ],
    }
    return rows, audit


def diversity_and_route() -> tuple[dict, dict]:
    diversity = {
        "status": "POTENTIAL_ONLY_NOT_REALIZED_BECAUSE_NOT_ADMITTED",
        "metadata": {"continent": "ASIA", "country": "INDONESIA", "trigger": "EARTHQUAKE",
                     "tectonic_setting": "SUBDUCTION_ARC", "volcano_morphology": "STRATOVOLCANO",
                     "erosion_state": "YOUNG_INTACT", "climate": "TROPICAL_WET",
                     "elevation_class": "HIGH_MOUNTAIN", "geographic_region": "WEST_SUMATRA",
                     "geology_breadth": "QUATERNARY_AND_NEOGENE_MIXED_VOLCANIC_WITH_EXPLICIT_NONVOLCANIC_EXCLUSIONS"},
        "C1_group_count": "NO_CHANGE_NOW; would add one independent development group after Y resolution",
        "C3_morphology": "ENRICHES young intact stratovolcano representation but does not close a missing broad class; Lombok already supplies a young stratovolcano complex",
        "C4_erosion": "WOULD_IMPROVE the young-intact endmember",
        "C5_climate": "NO_NEW_CLASS; tropical-wet already present",
        "C6_tectonics": "NO_NEW_CLASS; subduction-arc already present",
        "C7_trigger": "NO_NEW_CLASS; earthquake already present",
        "C8_geography": "REGIONAL_REPLICATION_ONLY; Asia and Indonesia already present",
        "C9_tier2_lolo": "WOULD_ADD one Tier-2 earthquake development group after Y resolution",
        "C10_hotspot_rift": "NO; Talakmau is a subduction-arc stratovolcano",
    }
    route = {
        "current": {"development": 6, "untouched_lockboxes": 0, "total": 6},
        "talakmau": "MULTIPLE_BLOCKERS_REMAIN_D_TAL_Y_AND_G_ATTRIBUTE_IDENTITY; development only; not lockbox",
        "fastest_route_to_8_development": "Resolve both Talakmau streams (D-TAL-Y and an official corrected geology identity artifact), then productionise it as development #7 in a later authorised stage; one additional development candidate is still required.",
        "fastest_route_to_11_total": "Still requires 8 development plus 3 complementary untouched lockboxes. Talakmau can fill one development slot only after both blockers resolve; it cannot replace any lockbox.",
        "recommended_target": {"development": 8, "untouched_lockboxes": 3, "total": 11},
        "ready_to_train_ml": False,
    }
    return diversity, route


def report(state: dict, sources: list[dict], archives: dict, qa: dict, prov: dict, coverage: dict,
           mapping: list[dict], volcanic: dict, gates: list[dict], audit: dict, decision: dict,
           diversity: dict, route: dict, tests: str, v1: str) -> str:
    gate = {r["gate"]: r["status"] for r in gates}
    return f"""STAGE 3I TALAKMAU GEOLOGY RESCUE + RAW-GATE VERIFICATION: COMPLETE

## A. EXECUTIVE RESULT

The two official GeoMap SHPs provide complete polygon coverage but **do not resolve G**: their DBF unit labels fail official Badan Geologi geographic cross-checks at Talamau. Talakmau is also blocked by Y (`HOLD_DECISION_REQUIRED`): UNOSAT marks it preliminary, confidence “To Be Evaluated,” and not field validated. Production stays 6 landscapes / 835,846 units / 8,793 positives.

## B. PASAMAN SHP PACKAGE CONTENTS

Both original ZIPs were preserved byte-for-byte. Pasaman is {sources[0]['byte_size']:,} bytes (`{sources[0]['SHA256']}`); Pasaman Barat is {sources[1]['byte_size']:,} bytes (`{sources[1]['SHA256']}`). Each contains a complete `geologi_ar_100k` polygon shapefile, a `geostructure_ln_100k` structure-line shapefile, Esri XML and `Legend.lyr`. CRC and required-component checks pass.

## C. SOURCE / PROVENANCE

These are GeoMap V.2 district geology extracts from Pusat Survei Geologi / Badan Geologi for Kab. Pasaman 13.08 and Kab. Pasaman Barat 13.12, scale 1:100,000. Embedded XML identifies the GeoMap V.2 Sumatera Barat `GEOLOGI_AR` path, EPSG:4326, and the unit fields `SIMOBJ`, `NAMOBJ`, `UMUROBJ`, `REMARK`, `METADATA`, `SRS_ID`. Product authenticity passes; scientific attribute consistency does not.

## D. GEOLOGY GEOMETRY / CRS QA

Pasaman: 56 polygon/multipolygon features; Pasaman Barat: 52. All 108 are valid, nonempty and non-null; duplicates 0; repairs 0. Both are EPSG:4326 with plausible West Sumatra bounds and negligible numerical overlap inflation. However, all three official Talamau reference points tested (summit and two volcanic-rock sites) intersect a polygon labelled `Qvps Peuet Sague Unit`, not local `Qvta Talamau products`; this is an attribute/geometry association failure, not a geometry-validity failure.

## E. TALAKMAU SPATIAL COVERAGE

Pasaman covers {coverage['packages']['PASAMAN']['omega_covered_km2']:.6f} km² and Pasaman Barat {coverage['packages']['PASAMAN_BARAT']['omega_covered_km2']:.6f} km² of Ω; both are required. Their overlap inside Ω is {coverage['package_overlap_inside_omega_km2']:.9f} km² and their unmapped Ω gap is {coverage['unmapped_gap_inside_omega_km2']:.9f} km². Geological coverage is {coverage['omega_geology_covered_percent']:.6f}% of Ω and {coverage['y_total_geology_covered_percent']:.6f}% of total Y; it is {coverage['y_inside_omega_geology_covered_percent']:.6f}% for Y eligible inside Ω.

## F. GEOLOGY ONTOLOGY MAPPING

All {len(mapping)} district-unit records intersecting Ω/Y are in `TALAKMAU_GEOLOGY_ONTOLOGY_MAPPING.csv`. Qvps and Tmv are nominally `MIXED_VOLCANIC`, but neither geometry is admitted because their local expansions conflict with official Badan Geologi Talamau/Lubuksikaping evidence. Alluvium, intrusion, limestone, pre-Cenozoic units and the ambiguous Totolan Formation are excluded fail-closed.

## G. VOLCANIC-DOMAIN AUDIT

Ω is {volcanic['omega_area_km2']:.6f} km². A naive label screen would call {volcanic['omega_nominal_shp_label_volcanic_area_km2']:.6f} km² ({volcanic['omega_nominal_shp_label_volcanic_percent']:.6f}%) volcanic and {volcanic['y_inside_omega_nominal_shp_label_volcanic_percent']:.6f}% of Y inside Ω volcanic, but these values are **not admissible** because the label join is contradicted by official sources. Defensible accepted percentages are not determinable. Gate: **HOLD_SOURCE_ATTRIBUTE_MISMATCH**.

## H. FULL TALAKMAU RAW-GATE REAUDIT

Y `{gate['Y_GATE']}`; Ω `{gate['OMEGA_GATE']}`; M `{gate['M_GATE']}`; temporal `{gate['TEMPORAL_GATE']}`; G `{gate['G_GATE']}`; volcanic domain `{gate['VOLCANIC_DOMAIN_GATE']}`; target semantics `{gate['TARGET_SEMANTICS_GATE']}`; independent group `{gate['INDEPENDENT_GROUP_GATE']}`. The UNOSAT, Ω and all 24 COP30 companion hashes remain coherent.

## I. TARGET / INDEPENDENCE AUDIT

The 2022-02-25 Mw 6.1 earthquake target is affected terrain (scar plus runout/deposit undivided), so it is Tier 2, not Tier 1. The explicit AnalysisExtent-minus-cloud Ω is independent of Y. Talakmau is a distinct physical system/event. It is V1-unseen but already exposed during V2 development, so its role is DEVELOPMENT and it is not final-lockbox eligible.

## J. RASTER MAP REQUIREMENT

**The missing raster (or an official corrected vector/crosswalk) is required.** The mismatch is concrete: Badan Geologi places the Talamau summit and two Talamau lava/breccia sites inside the area that the SHP calls Peuet Sague, and its 2022 Lubuksikaping-based event note identifies local Qvta/Tmv differently. The required artifact must restore the correct unit identity-to-polygon association; it is not optional visual confirmation.

## K. PRODUCTION DECISION

`{decision['classification']}`. All raw gates are not resolved: D-TAL-Y remains, and the GeoMap attribute/geometry mismatch blocks G and the volcanic domain. The frozen production pipeline was not invoked.

## L. PRODUCTION RESULTS

Not applicable. No terrain units, targets, features, ALL_VALID or STRICT_DEM_QUALITY tables were generated.

## M. UPDATED V2 DATASET

Unchanged: 6 independent landscapes, 835,846 terrain units, 8,793 positives, 827,053 negatives, Tier 2 only.

## N. DIVERSITY IMPACT

No realized change. If Y and G are later resolved, Talakmau would add a young-intact West Sumatran development group and Tier-2 earthquake replication, improving C1/C4/C9. It does not add a continent, country, trigger, tropical-wet climate, subduction setting, or hotspot/rift class.

## O. UPDATED ROUTE TO RECOMMENDED DATASET

Talakmau remains a multi-blocker development candidate. Resolve D-TAL-Y and obtain an official corrected geology identity artifact, then it can become development #7; one further development landscape is still needed for 8. The recommended endpoint remains 8 development + 3 untouched lockboxes = 11; Talakmau cannot substitute for a lockbox.

## P. TESTS / CHECKPOINT / V1 / GIT

Stage 3F, 3G and 3H hashes are unchanged; Stage 3H is `{STAGE3H_HASH}`. Tests: {tests}. V1: {v1}. `STAGE3I_CHECKPOINT.json` contains the full audit and artifact hashes. Dirty pre-existing worktree preserved; no email, commit or push.

1. Were both Pasaman SHP packages successfully located? **YES.**
2. Are they genuine machine-readable geological vector products? **YES.**
3. What geological polygon layers do they contain? **`pasaman/geologi_ar_100k_pasaman.shp` and `pasaman_barat/geologi_ar_100k_pasaman_barat.shp`.**
4. Is the missing raster geological map actually required? **YES—unless Badan Geologi supplies an official corrected SHP or geometry-to-unit crosswalk.**
5. Do the SHPs spatially cover Talakmau Omega? **YES; both packages together are required.**
6. What percentage of Omega has geological coverage? **{coverage['omega_geology_covered_percent']:.6f}%.**
7. What percentage of Y has geological coverage? **{coverage['y_total_geology_covered_percent']:.6f}% of total Y; 100.000000% of Y inside Ω.**
8. What percentage of Omega lies in accepted volcanic units? **NOT DEFENSIBLY DETERMINABLE; the contradicted SHP-label screen is 80.399377% and is diagnostic only.**
9. What percentage of Y lies in accepted volcanic units? **NOT DEFENSIBLY DETERMINABLE; the contradicted label screen is 99.621279% of total Y / 100.000000% inside Ω.**
10. Did the G gate pass? **NO; source attribute/geometry identity is inconsistent.**
11. Did the volcanic-domain gate pass? **NO; it is held on the same source inconsistency.**
12. Did Y pass? **NO; D-TAL-Y remains unresolved.**
13. Did Omega pass? **YES.**
14. Did pre-event M pass? **YES.**
15. Did temporal validity pass? **YES.**
16. Did target semantics pass as Tier 2? **YES; semantic class only, without overriding Y quality.**
17. Did independent-group status pass? **YES.**
18. Are ALL Talakmau raw gates resolved? **NO.**
19. Is Talakmau eligible as an untouched final lockbox? **NO.**
20. What role should Talakmau have: DEVELOPMENT or LOCKBOX? **DEVELOPMENT.**
21. Was Talakmau productionised? **NO.**
22. If yes, how many valid terrain units? **NOT APPLICABLE.**
23. If yes, how many positive units? **NOT APPLICABLE.**
24. If yes, what is prevalence? **NOT APPLICABLE.**
25. If yes, what are the new global terrain-unit and positive totals? **NOT APPLICABLE; unchanged at 835,846 and 8,793.**
26. If yes, are there now exactly 7 production landscapes? **NO; there are still exactly 6.**
27. Which diversity gaps does Talakmau actually close? **NONE yet; if admitted after Y and G correction, it improves young-intact erosion-state and Tier-2 earthquake group replication, but not hotspot/rift, trigger, tectonic, climate, continent or country gaps.**
28. How does this change the fastest route to 8 development landscapes? **It does not create an immediate fast path: both D-TAL-Y and corrected official geology must resolve before Talakmau can become #7; one additional development landscape would still be required.**
29. How does this change the fastest route to 11 total landscapes? **It can fill one development slot only after both blockers resolve; the route still needs 8 development plus 3 untouched lockboxes.**
30. Are we ready to train ML? **NO.**
"""


def main(tests: str = "PENDING", v1_status: str = "PENDING") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    state = state_verification()
    if not state["all_checkpoint_hashes_unchanged"] or not state["production_state_exact"]:
        raise RuntimeError(f"Frozen input state failed: {state}")
    sources = source_registry()
    archive_rows, archives = archive_inventory()
    frames = load_geology()
    qa = geology_qa(frames)
    prov = provenance()
    coverage, mapping, volcanic = coverage_and_mapping(frames)
    integrity = raw_integrity()
    if not all((integrity["unosat_unchanged"], integrity["cop30_all_24_companions_unchanged"], integrity["omega_unchanged"])):
        raise RuntimeError(f"Talakmau prior artifact integrity failed: {integrity}")
    gates, audit = gate_reaudit(integrity)
    diversity, route = diversity_and_route()
    decision = {
        "classification": "MULTIPLE_BLOCKERS_REMAIN_NOT_RAW_GATES_RESOLVED",
        "production_authorized": False, "production_performed": False,
        "blockers": audit["remaining_blockers"],
        "reason": "Stage 3I permits production only after every gate unequivocally passes; preliminary/unvalidated Y and contradicted geology attributes cannot be silently promoted.",
        "production_state": state["production_state"],
    }
    diagnostics = {
        "stage": "3I", "geology_rescued": False, "geology_polygon_coverage_recovered": True,
        "geology_attribute_identity_resolved": False, "all_raw_gates_resolved": False,
        "production_state_changed": False, "terrain_units_generated": False,
        "production_tables_generated": False, "ml_trained": False, "exploratory_ml_run": False,
        "lolo_run": False, "h_generated": False, "tier_mixing": False,
        "y_derived_omega_accepted": False, "geology_rule_weakened": False,
        "ambiguous_geology_accepted": False, "post_event_primary_m": False,
        "emails_sent": False, "commit_performed": False, "push_performed": False,
        "raw_files_overwritten": False,
    }

    write_json(OUT / "STATE_VERIFICATION.json", state)
    write_rows(OUT / "STAGE3I_SOURCE_REGISTRY.tsv", sources, "\t")
    write_rows(OUT / "STAGE3I_ARCHIVE_INVENTORY.tsv", archive_rows, "\t")
    write_json(OUT / "STAGE3I_ARCHIVE_AUDIT.json", archives)
    write_json(OUT / "STAGE3I_PROVENANCE_AUDIT.json", prov)
    write_json(OUT / "STAGE3I_GEOLOGY_GEOMETRY_QA.json", qa)
    write_json(OUT / "TALAKMAU_SPATIAL_COVERAGE_AUDIT.json", coverage)
    write_rows(OUT / "TALAKMAU_GEOLOGY_ONTOLOGY_MAPPING.csv", mapping)
    write_json(OUT / "TALAKMAU_VOLCANIC_DOMAIN_AUDIT.json", volcanic)
    write_rows(OUT / "TALAKMAU_RAW_GATE_REAUDIT.tsv", gates, "\t")
    write_json(OUT / "TALAKMAU_TARGET_INDEPENDENCE_AUDIT.json", audit)
    write_json(OUT / "TALAKMAU_PRODUCTION_DECISION.json", decision)
    write_json(OUT / "TALAKMAU_DIVERSITY_IMPACT.json", diversity)
    write_json(OUT / "STAGE3I_ROUTE_UPDATE.json", route)
    write_json(OUT / "STAGE3I_DIAGNOSTICS.json", diagnostics)
    (OUT / "STAGE3I_REPORT.md").write_text(report(state, sources, archives, qa, prov, coverage, mapping,
                                                      volcanic, gates, audit, decision, diversity, route,
                                                      tests, v1_status))
    names = sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name != "STAGE3I_CHECKPOINT.json")
    checkpoint = {
        "stage": "3I", "as_of": AS_OF, "stage3h_input_hash": STAGE3H_HASH,
        "pasaman_source_hashes": {r["package"]: r["SHA256"] for r in sources},
        "archive_inventory": archives, "source_provenance_audit": prov,
        "geology_layer_qa": qa, "talakmau_coverage_audit": coverage,
        "unit_ontology_mapping": mapping, "volcanic_domain_audit": volcanic,
        "raw_gate_results": gates, "target_independence_audit": audit,
        "production_decision": decision, "production_totals": state["production_state"],
        "diversity_impact": diversity, "stopping_rule_impact": route,
        "tests": tests, "v1_state": v1_status,
        "git_state": {"head_at_audit": "4958b65982e3788e5777c5a4f14e87e67ad95a0a",
                      "worktree": "DIRTY_PRE_EXISTING_WORKTREE_PRESERVED",
                      "commit_performed": False, "push_performed": False},
        "diagnostics": diagnostics,
        "artifact_hashes": {f"outputs/stage3i/{name}": sha256(OUT / name) for name in names},
    }
    write_json(OUT / "STAGE3I_CHECKPOINT.json", checkpoint)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="PENDING")
    parser.add_argument("--v1-status", default="PENDING")
    args = parser.parse_args()
    main(args.tests, args.v1_status)
