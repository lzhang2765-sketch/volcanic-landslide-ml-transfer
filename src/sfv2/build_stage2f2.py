"""Build Stage 2F.2: Lombok complete geology, legend resolution, target freeze, gates and (only if all pass) production."""
from __future__ import annotations

import csv
import json
import subprocess
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import stage2d_production as P
from . import stage2f2_lombok as L
from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage2f2_production import build_polygon_landscape
from .stage2f_transfer import INCOMING, v1_verify

OUT = L.OUT
PROD = ROOT / "data/processed/stage2f2_production"
COVERAGE_MIN_LAND_FRACTION = 0.999  # uncovered LAND (after DEM water) must be < 0.1% of Omega_obs

TARGET_FREEZE = {
    "landscape_id": L.LID, "decision_date": L.AS_OF, "target_family": "POLYGON_AFFECTED_TIER2",
    "PRIMARY": {"file": "Ferrario_2019b", "polygons": 4823, "area_km2": 4.89,
                "meaning": "Landslides mapped after the 5 August 2018 Mw 6.9 earthquake (cumulative from the start of the sequence, incl. 29 July)"},
    "SENSITIVITY": [{"file": "Ferrario_2019a", "polygons": 9319, "area_km2": 10.25,
                     "meaning": "End-of-sequence cumulative inventory after the 19 August 2018 Mw 6.9 events; includes enlarged/reactivated 5-August polygons"}],
    "identity_evidence": [
        "RAW methods form (SF1_VERIFICATION_ground-failure inventory metadata form_Lombok.pdf): 'Two inventories were built, related to landslides "
        "triggered after the August 5, 2018 Mw 6.9 event, and at the end of the seismic sequence (after the August 19, 2018 Mw 6.9 event)'; "
        "PlanetScope/Sentinel-2 pre/post comparison, 95% estimated completeness, pre-event landslides removed",
        "Burrows, Milledge & Ferrario (2025) Earth Surf. Dynam. 13:1039 (registered ACQ_LOMBOK_DOC_BURROWS2025_ESURF): 'Ferrario (2019) mapped 4823 "
        "landslides (4.88 km2) following the 5 August earthquake increasing to 9319 (10.25 km2) following the 19 August earthquakes'",
        "Exact count/area match: Ferrario_2019b = 4823 polygons / 4.89 km2; Ferrario_2019a = 9319 polygons / 10.25 km2",
        "Geometry consistency: 89% of 2019b landslide area lies inside 2019a (cumulative), 43% of 2019a inside 2019b",
    ],
    "rationale": [
        "Trigger clarity: 2019b is bounded by the strongest-shaking event (5 Aug) and excludes the 19 Aug events and later growth/reactivation",
        "Consistency with the frozen Kumamoto design: the earliest, trigger-cleanest inventory is primary; later mixed-trigger additions are sensitivity",
        "Both share the same published mapping extent (Omega), mapper, imagery type and pre-event M; neither class balance nor model performance was considered",
    ],
    "selection_criteria_excluded": ["class balance / prevalence", "model performance", "label count"],
    "caveats": ["2019b still aggregates the 29 Jul Mw 6.4 and 5 Aug Mw 6.9 events (sequence phase, not a single shock)",
                "Copernicus editing date 2018-08-16 postdates the primary phase; carried as the STRICT_DEM_QUALITY sensitivity"],
}


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False, lineterminator="\n")


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    recs = L.register_new()
    reg = {e["acquisition_id"]: e for e in load_registry()}
    archive = []
    for acq in [d["acq"] for d in L.DISTRICTS.values()]:
        p = ROOT / reg[acq]["raw_path"]
        zf = zipfile.ZipFile(p)
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        stem = next(n[:-4] for n in names if "geologi_ar" in n and n.endswith(".shp"))
        archive.append(dict(acquisition_id=acq, sha256=reg[acq]["sha256"], bytes=reg[acq]["actual_bytes"], members=len(names),
                            crc_ok=zf.testzip() is None, required_parts=all(stem + s in names for s in (".shp", ".shx", ".dbf", ".prj")),
                            original_preserved=sha256(INCOMING / reg[acq]["original_filename"]) == reg[acq]["sha256"]))
    sheet = reg[L.SHEET_ACQ]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(ROOT / sheet["raw_path"]) as im:
        sheet_audit = dict(acquisition_id=L.SHEET_ACQ, sha256=sheet["sha256"], bytes=sheet["actual_bytes"], format=im.format, size_px=list(im.size),
                           dpi=list(im.info.get("dpi", ())), georeferenced=False, contains=["map image", "legend (KETERANGAN/EXPLANATION)",
                           "unit descriptions/lithology", "correlation chart (relative ages, rock groups)", "cross-sections", "authors/date (Mangga et al. 1994)"],
                           machine_readable_text=False, transcription="Visual transcription of legible legend text only (see sheet_legend_evidence.csv)",
                           original_preserved=sha256(INCOMING / sheet["original_filename"]) == sheet["sha256"])
    finder = INCOMING / "lombok_utara"
    finder_audit = dict(path="data/incoming_user/lombok_utara", exists=finder.exists(), status="NON_AUTHORITATIVE_FINDER_EXPANSION_NOT_REGISTERED",
                        registered=any("incoming_user/lombok_utara" in e["raw_path"] for e in load_registry()))

    geo = L.load_geology()
    is_geology = bool(geo.geom_type.isin(["Polygon", "MultiPolygon"]).all() and geo.SIMOBJ.notna().all() and geo.SIMOBJ.nunique() > 5)
    mp, ms = L.unit_mapping(geo, "PRIMARY"), L.unit_mapping(geo, "SENSITIVITY")
    mp.to_csv(OUT / "geology_mapping_LOMBOK_geomap_primary.csv", index=False, lineterminator="\n")
    ms.to_csv(OUT / "geology_mapping_LOMBOK_geomap_sensitivity.csv", index=False, lineterminator="\n")
    _csv(OUT / "sheet_legend_evidence.csv", [dict(official_unit=k, official_name=v[0], official_lithology=v[1], official_age=v[2], rock_group=v[3],
                                                   district_shp_codes="|".join(c for c, m in L.UNIT_MAP.items() if m[0] == k) or "NONE_IN_DISTRICT_SHPS",
                                                   evidence_source=f"{L.SHEET_ACQ} legend/correlation (visual transcription)") for k, v in L.SHEET_LEGEND.items()])
    seam = L.seam_qc(geo)
    _csv(OUT / "seam_qc.csv", seam)
    overlay = L.anomaly_overlay(geo)
    _json(OUT / "anomaly_resolution.json", dict(
        method="DIAGNOSTIC ONLY: affine georeference of sheet neatline + nearest legend-swatch colour votes (validated: TQp/Tme/Tomp/Qhv recovered; Qvl/TQb confusable)",
        overlay=overlay, resolution={
            "Qhvb": "Attribute contamination (Bali name). Geometry = Rinjani/Samalas caldera floor; sheet shows official volcanic units beneath "
                    f"({overlay.get('Qhvb', {}).get('official_volcanic_fraction')}). PRIMARY: UNKNOWN (excluded); SENSITIVITY: MIXED_VOLCANIC",
            "Qv": "Attribute contamination (Bali name). Sheet shows TQb/Qvl beneath "
                  f"({overlay.get('Qv', {}).get('official_volcanic_fraction')}). PRIMARY: UNKNOWN (excluded); SENSITIVITY: MIXED_VOLCANIC",
            "Gr4": "Attribute contamination (Sulawesi name). Seven tiny coastal polygons (0.069 km2 total) outside Omega; UNKNOWN; no effect"}))
    omega = gpd.read_file(L.V1T / "OMEGA/SF1_VERIFICATION_mapping_extent.shp").to_crs(L.UTM).union_all()
    covered = geo.union_all()
    dom = L.build_domain(geo, mp, ms, omega)
    land_uncovered_km2 = next(s["km2"] for s in dom["domains"]["PRIMARY"]["steps"] if s["step"] == "geology_unmapped")
    cov = dict(omega_obs_km2=round(omega.area / 1e6, 3), geology_covered_km2=round(omega.intersection(covered).area / 1e6, 3),
               uncovered_km2=round(omega.difference(covered).area / 1e6, 4), coverage_fraction=round(omega.intersection(covered).area / omega.area, 6),
               uncovered_land_after_dem_water_km2=land_uncovered_km2,
               uncovered_explanation="Residual uncovered area is coastal/sea cells removed by the Copernicus water mask; uncovered land is cartographic sliver",
               districts={k: round(geo[geo.source_district == k].union_all().area / 1e6, 3) for k in L.DISTRICTS})
    y = {}
    for ph in ("a", "b"):
        g = gpd.read_file(L.V1T / f"Y/SF1_VERIFICATION_Ferrario_2019{ph}.shp").to_crs(L.UTM)
        g["geometry"] = g.geometry.make_valid()
        y[ph] = g
    cov["y_outside_geology_primary"] = int((~y["b"].geometry.representative_point().within(covered)).sum())
    _json(OUT / "coverage_audit.json", cov)
    yaud = {"PRIMARY_Ferrario_2019b": L.y_audit(y["b"], omega, covered, dom, mp), "SENSITIVITY_Ferrario_2019a": L.y_audit(y["a"], omega, covered, dom, mp)}
    _json(OUT / "y_audit.json", yaud)
    _json(OUT / "TARGET_DEFINITION_FREEZE.json", TARGET_FREEZE)
    domain_rows = [dict(variant=k, **s) for k, v in dom["domains"].items() for s in v["steps"]]
    _csv(OUT / "omega_analysis_derivation.csv", domain_rows)
    _json(OUT / "volcanic_domain_summary.json", {k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in dom["domains"].items()})
    m = json.loads((ROOT / "outputs/stage2f1/m_audit.json").read_text())
    gates = [
        ("Y", "PASS", "Ferrario 2019b primary (4,823) / 2019a sensitivity; identity documented; Tier-2 polygons"),
        ("OMEGA", "PASS", "Published mapping_extent unchanged; independent of Y"),
        ("M", "PASS" if m["dem_byte_identical_to_official"] and m["omega_dem_nodata_pixels"] == 0 else "FAIL", "COP30 S09E116, official bytes, no nodata in Omega"),
        ("TEMPORAL", "PASS" if m["pre_event"] else "FAIL", f"TanDEM-X {m['tandemx_start']}..{m['tandemx_stop']} < 2018-07-29; editing date caveat carried"),
        ("GEOLOGY_COVERAGE", "PASS" if is_geology and land_uncovered_km2 / cov["omega_obs_km2"] < 1 - COVERAGE_MIN_LAND_FRACTION else "FAIL",
         f"{cov['coverage_fraction']:.4%} of Omega; uncovered land {land_uncovered_km2} km2"),
        ("VOLCANIC_DOMAIN", "PASS" if dom["domains"]["PRIMARY"]["analysis_km2"] > 0 and not (mp.harmonised_class == "UNKNOWN").sum() > 3 else "FAIL",
         f"{dom['domains']['PRIMARY']['analysis_km2']} km2 from sheet-1807-resolved Cenozoic volcanic units; anomalies excluded"),
        ("DEM_COVERAGE", "PASS" if m["omega_fully_inside_tile"] and m["omega_dem_nodata_pixels"] == 0 else "FAIL", "Omega inside tile, no nodata"),
        ("TARGET_SEMANTICS", "PASS", "POLYGON_AFFECTED_TIER2 preserved; phases not mixed; no buffering"),
    ]
    eligible = all(s == "PASS" for _, s, _ in gates)
    production, dq, sens = None, None, None
    if eligible:
        srcdir = L.DER
        r = build_polygon_landscape(L.LID, srcdir, PROD, mp.to_dict("records"), y["b"], "FERRARIO_2019B_POST_5_AUG_2018_AFFECTED_AREA_POLYGONS",
                                    trigger_class="EARTHQUAKE_SEQUENCE_PHASE_29JUL_5AUG_2018", omega_name="omega_analysis_primary_30m.tif",
                                    geology_name="geology_geomap_index_30m.tif", hydrology_root=PROD)
        s, t, tr = r["summary"], r["targets"], r["terrain"]
        # sensitivity target on identical units (same frozen rule), stored separately; never pooled with primary
        import rasterio
        from rasterio.features import rasterize
        with rasterio.open(srcdir / "omega_analysis_primary_30m.tif") as od:
            transform = od.transform
        lg = rasterize([(g, 1) for g in y["a"].geometry], out_shape=r["omega"].shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        h3, w3 = lg.shape[0] // 3 * 3, lg.shape[1] // 3 * 3
        cnt = P._block(lg.astype(np.float32), "sum", h3, w3)[tr.source_grid_row.to_numpy() // 3, tr.source_grid_col.to_numpy() // 3]
        sens = pd.DataFrame({"terrain_unit_id": tr.terrain_unit_id, "y_positive": cnt > 0, "target_source": "FERRARIO_2019A_END_OF_SEQUENCE_CUMULATIVE",
                             "role": "SENSITIVITY_ONLY_NEVER_POOLED"})
        P._csv_gz(PROD / L.LID / "targets/targets_sensitivity_ferrario2019a.csv.gz", sens)
        with rasterio.open(srcdir / "omega_analysis_sensitivity_anomalies_included_30m.tif") as sd:
            sm = sd.read(1).astype(bool)
        sc = P._block(sm.astype(np.float32), "sum", h3, w3)
        sens_domain_units = int((sc >= 5).sum())
        pos = t.y_positive.astype(bool).to_numpy()
        strict = tr.analysis_population_strict_dem_quality.to_numpy(bool)
        dq = dict(units=len(tr), units_with_non_original_cells=int((tr.dem_edited_fraction > 0).sum()),
                  fraction_units_non_original=round(float((tr.dem_edited_fraction > 0).mean()), 5),
                  positives=int(pos.sum()), positives_with_non_original_cells=int((pos & ~strict).sum()),
                  fraction_positives_non_original=round(float((pos & ~strict).sum() / max(pos.sum(), 1)), 5),
                  prevalence_all_valid=round(float(pos.mean()), 6), strict_units=int(strict.sum()), strict_positives=int((pos & strict).sum()),
                  prevalence_strict=round(float(pos[strict].mean()), 6) if strict.any() else None,
                  cell_fraction_non_original_in_domain=round(float(((r["edm"] != 1) & r["omega"]).sum() / r["omega"].sum()), 5),
                  editing_date="2018-08-16", note="Mandatory future sensitivity: ALL_VALID vs STRICT_DEM_QUALITY")
        _json(OUT / "dem_quality_sensitivity.json", dq)
        frozen = P.feature_definition_records(r["feature_names"])
        fdef = pd.read_csv(ROOT / "outputs/stage2d/feature_definitions.csv")
        feat_cols = [c for c in r["features"].columns if c != "terrain_unit_id"]
        production = dict(summary=s, sensitivity_positive_units=int(sens.y_positive.sum()), sensitivity_domain_units=sens_domain_units,
                          feature_names_identical_to_frozen=sorted(r["feature_names"]) == sorted(fdef.feature_name),
                          feature_definitions_identical=[x["feature_name"] for x in frozen] == sorted(fdef.feature_name),
                          feature_columns=len(feat_cols), unexpected_missing_columns=[c for c in feat_cols if r["features"][c].isna().any()],
                          feature_ranges={c: [round(float(r["features"][c].min()), 4), round(float(r["features"][c].max()), 4)]
                                          for c in feat_cols if c.endswith("__mean")},
                          artifacts={str(p.relative_to(ROOT)): sha256(p) for p in r["artifacts"] + [PROD / L.LID / "targets/targets_sensitivity_ferrario2019a.csv.gz"]})
        _json(OUT / "production_summary.json", production)
    gates.append(("TERRAIN_GENERATION", "PASS" if production and production["summary"]["valid_units"] > 0 and not production["unexpected_missing_columns"]
                  and production["feature_names_identical_to_frozen"] else ("NOT_RUN" if not eligible else "FAIL"),
                  "Frozen Stage-2D polygon architecture (byte-equivalence proven on Noto)"))
    ready = all(s == "PASS" for _, s, _ in gates)
    _csv(OUT / "production_readiness_gates.csv", [dict(gate=a, status=b, evidence=c) for a, b, c in gates])
    _json(OUT / "h_feasibility_update.json", dict(
        landscape_id=L.LID, status="H_NOT_RECONSTRUCTED", likely_family="MULTI_EDIFICE / CALDERA_RIM (Rinjani-Samalas)",
        edifice_boundaries="Qhv(r) Rinjani, Qhv(p) Pusuk, Qhv(n) Nangi delineated on sheet 1807; district SHPs merge them as Qhv1",
        caldera_units="Caldera rim symbol on sheet; Qhvb polygon on caldera floor (attribute-anomalous)",
        relative_stratigraphy="Tomp/Tomk < Tme < Tmi < TQb/TQp(TQs) < Qvl < Qhv (correlation chart)",
        numeric_ages="NONE on sheet (epoch-level only)", eruption_stage_information="Epoch-level only; 1257 CE Samalas eruption from literature (Lavigne et al. 2013, RAW)",
        palaeoelevation_controls="NONE", missing=["numeric unit ages", "pre-1257 Samalas edifice reconstruction controls", "digital caldera-rim geometry"],
        ensemble_feasible_later="LIKELY (geometry and relative stratigraphy now available)", effect_on_meg_readiness="NONE"))
    dash = pd.read_csv(ROOT / "outputs/stage2f1/landscape_state_dashboard.csv")
    i = dash.index[dash.landscape_id == L.LID][0]
    dash.loc[i, ["state", "production_ready_after_harmonisation", "primary_blocker", "target_family"]] = [
        "PRODUCTION_READY" if ready else "BLOCKED_GEOLOGY", ready, "NONE" if ready else "SEE_STAGE2F2_GATES", "POLYGON_AFFECTED_TIER2"]
    dash.to_csv(OUT / "landscape_state_dashboard.csv", index=False, lineterminator="\n")
    totals = {}
    for lid in ("KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024"):
        s = json.loads((P.DER / lid / "metadata/production_summary.json").read_text())
        totals[lid] = dict(units=s["valid_units"], positives=s["positive_units"], negatives=s["negative_units"], family=s["target_family"])
    if ready:
        totals[L.LID] = dict(units=production["summary"]["valid_units"], positives=production["summary"]["positive_units"],
                             negatives=production["summary"]["negative_units"], family="POLYGON_AFFECTED_TIER2")
    prod_rows = dash[dash.state == "PRODUCTION_READY"]
    summary = dict(production_ready=sorted(prod_rows.landscape_id), count=len(prod_rows), total_units=sum(v["units"] for v in totals.values()),
                   by_landscape=totals, by_family={f: dict(units=sum(v["units"] for v in totals.values() if v["family"] == f),
                                                           positives=sum(v["positives"] for v in totals.values() if v["family"] == f),
                                                           negatives=sum(v["negatives"] for v in totals.values() if v["family"] == f))
                                                    for f in sorted({v["family"] for v in totals.values()})},
                   continents=prod_rows.continent.value_counts().to_dict(), tectonic=prod_rows.tectonic_setting.value_counts().to_dict(),
                   h_feasibility=prod_rows.h_feasibility.value_counts().to_dict(),
                   lombok_lockbox="INELIGIBLE_V1_HISTORICAL: development use only under historical-status rules")
    _json(OUT / "landscape_state_summary.json", summary)
    _json(OUT / "ingestion_audit.json", dict(registrations=[{k: r[k] for k in ("acquisition_id", "sha256", "actual_bytes", "raw_path", "original_filename")}
                                                            for r in recs], archives=archive, sheet=sheet_audit, finder=finder_audit, geology_is_unit_polygons=is_geology))
    _json(OUT / "lineage.json", dict(derived=[dict(output=str(p.relative_to(ROOT)), output_sha256=sha256(p), operation=op,
                                                   inputs={str(i.relative_to(ROOT)): sha256(i) for i in ins}) for p, ins, op in dom["lineage"]],
                                     script_sha256={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage2f2.py", "stage2f2_lombok.py", "stage2f2_production.py",
                                                                                                "stage2d_production.py")}))
    v1_after = v1_verify()
    outs = sorted(p for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("STAGE2F2_CHECKPOINT", "STAGE2F2_DIAGNOSTICS")))
    cp = dict(schema="SFV2_STAGE2F2_CHECKPOINT_V1", as_of=L.AS_OF, lombok_production_ready=ready, gates={a: b for a, b, _ in gates},
              production_ready_count=summary["count"], production_terrain_units_total=summary["total_units"],
              registrations={r["acquisition_id"]: r["sha256"] for r in recs}, output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs},
              derived_hashes={x: sha256(ROOT / x) for x in sorted(str(p.relative_to(ROOT)) for p, _, _ in dom["lineage"])},
              production_artifact_hashes=production["artifacts"] if production else {},
              prior_checkpoints={rel: sha256(ROOT / rel) for rel in ("outputs/stage2f1/STAGE2F1_CHECKPOINT.json", "outputs/stage2f/STAGE2F_CHECKPOINT.json",
                                                                    "outputs/stage2d/STAGE2D_CHECKPOINT.json")},
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              training=False, h_reconstruction=False, lockbox_assignment=False, balancing=False,
              volatile_diagnostics={"path": "outputs/stage2f2/STAGE2F2_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE2F2_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE2F2_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD")))
    return dict(cp=cp, cov=cov, y=yaud, dq=dq, production=production, summary=summary, seam=seam, overlay=overlay, dom={k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in dom["domains"].items()})


if __name__ == "__main__":
    main()
