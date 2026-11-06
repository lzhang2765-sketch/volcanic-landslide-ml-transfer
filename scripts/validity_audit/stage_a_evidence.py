"""AUSSEF 2026 Stage A validity audit — read-only evidence extraction.

Reads frozen raw metadata, frozen production tables and the frozen Stage 4A feature matrix.
Writes ONLY into Science_Fair_ML_V2/AUSSEF_2026_VALIDITY_AUDIT/evidence/.
No model is fitted, no label/feature is changed, nothing outside the audit folder is written.

Run from the repository root:
    .venv/bin/python -I Science_Fair_ML_V2/AUSSEF_2026_VALIDITY_AUDIT/scripts/stage_a_evidence.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

V2 = Path(__file__).resolve().parents[2]
OUT = V2 / "AUSSEF_2026_VALIDITY_AUDIT" / "evidence"
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
              "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
EVENT_DATE = {  # first trigger date of the PRIMARY target, from STAGE5A_DATASET_REGISTER / target freezes
    "KUMAMOTO_ASO_2016": "2016-04-14", "NOTO_PENINSULA_2024": "2024-01-01", "LOMBOK_RINJANI_2018": "2018-07-29",
    "COTABATO_DAVAO_2019": "2019-10-16", "ATITLAN_STAN_2005": "2005-10-01", "DOMINICA_ERIKA_2015": "2015-08-25"}
COP = {
    "KUMAMOTO_ASO_2016": ("data/raw/stage2b/KUMAMOTO_ASO_2016/SRC_GLOBAL_COP30", None),
    "NOTO_PENINSULA_2024": ("data/raw/stage2b/NOTO_PENINSULA_2024/SRC_GLOBAL_COP30", None),
    "LOMBOK_RINJANI_2018": ("data/raw/stage2b/LOMBOK_RINJANI_2018/SRC_GLOBAL_COP30", "data/raw/stage2d/LOMBOK_RINJANI_2018/V1_TRANSFER/DEM"),
    "COTABATO_DAVAO_2019": ("data/raw/stage3c/COTABATO_DAVAO_2019/SRC_COP30", None),
    "DOMINICA_ERIKA_2015": ("data/raw/stage3c/DOMINICA_2015_2017/SRC_COP30", None),
}
PROD = {
    "KUMAMOTO_ASO_2016": "data/processed/stage2d/KUMAMOTO_ASO_2016",
    "NOTO_PENINSULA_2024": "data/processed/stage2d/NOTO_PENINSULA_2024",
    "LOMBOK_RINJANI_2018": "data/processed/stage2f2_production/LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019": "data/processed/stage3d_production/COTABATO_DAVAO_2019",
    "ATITLAN_STAN_2005": "data/processed/stage3f_production/ATITLAN_STAN_2005",
    "DOMINICA_ERIKA_2015": "data/processed/stage3f_production/DOMINICA_ERIKA_2015",
}
MEG = V2 / "outputs/stage4a/STAGE4A_MEG_DATASET"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _tag(x: str, tag: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", m).strip() for m in re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", x, re.S)]


def dem_chronology() -> list[dict]:
    rows = []
    for lid, (meta_dir, dem_dir) in COP.items():
        md = V2 / meta_dir
        for xml in sorted(md.glob("Copernicus_DSM_10_*.xml")):
            tile = xml.name.replace("Copernicus_DSM_10_", "").replace(".xml", "")
            x = xml.read_text(encoding="utf-8", errors="replace")
            kml = (md / xml.name.replace(".xml", "_SRC.kml")).read_text(encoding="utf-8", errors="replace")
            acq = sorted(re.findall(r"Acquisition Date:</td><td>([0-9T:\-]+)", kml))
            sets = pd.Series(re.findall(r"Acquisition Set:</td><td>([^<]+)", kml)).value_counts().to_dict()
            dem = (V2 / (dem_dir or meta_dir)) / f"Copernicus_DSM_COG_10_{tile}_DEM.tif"
            with rasterio.open(dem) as r:
                crs, res = str(r.crs), r.res
            rows.append(dict(
                landscape_id=lid, tile=tile, dem_file=str(dem.relative_to(V2)), dem_sha256=sha256(dem),
                native_crs=crs, native_res_deg=f"{res[0]:.10f}",
                xml_temporal_start=_tag(x, "tsxx:tsxx_startTime")[0][:10], xml_temporal_stop=_tag(x, "tsxx:tsxx_stopTime")[0][:10],
                xml_number_of_used_acquisitions=_tag(x, "tsxx:numberOfUsedAcquisitions")[0],
                xml_tandemx_dem_generation_time=_tag(x, "tsxx:generationTime")[0][:10],
                xml_date_of_analysis_editing=_tag(x, "tsxx:DateofAnalysis")[0][:10],
                xml_edition_date=_tag(x, "gco:Date")[1][:10] if len(_tag(x, "gco:Date")) > 1 else "",
                xml_metadata_datestamp=_tag(x, "gco:Date")[0][:10],
                src_kml_scene_count=len(acq), src_kml_first=acq[0][:10], src_kml_last=acq[-1][:10],
                src_kml_acquisition_sets=json.dumps(sets, ensure_ascii=False),
                event_date=EVENT_DATE[lid],
                all_scenes_before_event=bool(acq[-1][:10] < EVENT_DATE[lid]),
                xml_stop_equals_kml_last=bool(_tag(x, "tsxx:tsxx_stopTime")[0][:10] == acq[-1][:10]),
            ))
    # Atitlan: NASADEM tile header only (no acquisition metadata file archived)
    for name in ("NASADEM_HGT_n14w092.tif", "Copernicus_DSM_COG_10_N14_00_W092_00_DEM.tif"):
        p = V2 / "data/raw/stage2d/ATITLAN_STAN_2005/V1_TRANSFER/DEM" / name
        with rasterio.open(p) as r:
            tags = r.tags()
            crs, res, nod = str(r.crs), r.res, r.nodata
            arr = r.read(1)
        rows.append(dict(landscape_id="ATITLAN_STAN_2005", tile=name, dem_file=str(p.relative_to(V2)), dem_sha256=sha256(p),
                         native_crs=crs, native_res_deg=f"{res[0]:.10f}", geotiff_tags=json.dumps(tags)[:500],
                         native_nodata=str(nod), native_nodata_cells=int((arr == nod).sum()) if nod is not None else 0,
                         role="PRIMARY_M" if name.startswith("NASADEM") else "NOT_USED_POST_EVENT_COPERNICUS",
                         event_date=EVENT_DATE["ATITLAN_STAN_2005"]))
    return rows


def label_diagnostics() -> dict:
    out = {}
    for lid, d in PROD.items():
        t = pd.read_csv(V2 / d / "targets/targets_primary.csv.gz", low_memory=False)
        y = t.y_positive.astype(str).str.upper().eq("TRUE")
        r = dict(units=len(t), positives=int(y.sum()), label_rule=t.label_rule.iloc[0], target_source=t.target_source.iloc[0])
        if "landslide_overlap_area_m2" in t:
            ov = pd.to_numeric(t.landslide_overlap_area_m2, errors="coerce").fillna(0)
            frac = pd.to_numeric(t.landslide_overlap_fraction, errors="coerce").fillna(0)
            r.update(units_with_any_polygon_overlap=int((ov > 0).sum()),
                     overlap_but_labelled_negative=int(((ov > 0) & ~y).sum()),
                     overlap_negative_fraction_of_overlapping=round(float(((ov > 0) & ~y).sum() / max((ov > 0).sum(), 1)), 4),
                     overlap_negative_area_km2=round(float(ov[(ov > 0) & ~y].sum() / 1e6), 4),
                     positive_area_overlap_km2=round(float(ov[y].sum() / 1e6), 4),
                     positive_overlap_fraction_median=round(float(frac[y].median()), 4),
                     negatives_with_overlap_ge_0p25=int(((frac >= 0.25) & ~y).sum()),
                     positives_with_overlap_lt_0p10=int(((frac < 0.10) & y).sum()))
        else:
            ec = pd.to_numeric(t.event_count_in_unit, errors="coerce").fillna(0)
            r.update(points_in_included_units=int(ec.sum()), units_with_ge2_points=int((ec >= 2).sum()),
                     max_points_per_unit=int(ec.max()))
        out[lid] = r
    # Kumamoto: July / undated sensitivity centres that fall in PRIMARY negative units
    d = V2 / PROD["KUMAMOTO_ASO_2016"]
    tu = pd.read_csv(d / "terrain_units/terrain_units.csv.gz", usecols=["terrain_unit_id", "source_grid_row", "source_grid_col"])
    tg = pd.read_csv(d / "targets/targets_primary.csv.gz", usecols=["terrain_unit_id", "y_positive"])
    m = tu.merge(tg, on="terrain_unit_id")
    key = {(int(a), int(b)): str(v).upper() == "TRUE" for a, b, v in zip(m.source_grid_row, m.source_grid_col, m.y_positive)}
    import geopandas as gpd
    sp = gpd.read_file(d / "targets/sensitivity_points.geojson")
    with rasterio.open(V2 / "data/processed/stage2c/KUMAMOTO_ASO_2016/omega_analysis_primary_april_eq_30m.tif") as r:
        assert str(sp.crs.to_epsg()) == str(r.crs.to_epsg())
        rr, cc = rasterio.transform.rowcol(r.transform, sp.geometry.x, sp.geometry.y)
    blocks = [(int(a) // 3 * 3, int(b) // 3 * 3) for a, b in zip(rr, cc)]
    res = {}
    for ts in sorted(sp.target_set.unique()):
        sel = [b for b, s in zip(blocks, sp.target_set) if s == ts]
        in_units = [b for b in sel if b in key]
        res[ts] = dict(points=len(sel), in_primary_valid_units=len(in_units),
                       in_primary_negative_units=sum(1 for b in in_units if not key[b]),
                       distinct_negative_units=len({b for b in in_units if not key[b]}))
    out["KUMAMOTO_SENSITIVITY_POINTS_VS_PRIMARY_UNITS"] = res
    return out


def elevation_diagnostics() -> dict:
    data = {}
    for lid in LANDSCAPES:
        with np.load(MEG / f"{lid}.npz", allow_pickle=False) as z:
            data[lid] = dict(e=z["elev_m_30m__mean"].astype(float), y=z["y_positive"].astype(bool),
                             ids=z["terrain_unit_id"], x=z["centroid_x"], yy=z["centroid_y"], crs=str(z["projected_crs"][0]))
    rows, ident = [], {}
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    for lid in LANDSCAPES:
        e, y = data[lid]["e"], data[lid]["y"]
        train = np.concatenate([data[o]["e"] for o in LANDSCAPES if o != lid])
        lo, hi = train.min(), train.max()
        p1, p99 = np.percentile(train, [1, 99])
        # descriptive within-landscape rank association (Mann-Whitney AUC) of elevation with the frozen label; not a model
        from scipy.stats import rankdata
        rk = rankdata(e)
        n1, n0 = y.sum(), (~y).sum()
        auc = (rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        rows.append(dict(landscape_id=lid, units=len(e), positives=int(y.sum()),
                         **{f"elev_q{k:02d}": round(float(v), 1) for k, v in zip(q, np.percentile(e, q))},
                         elev_median_positives=round(float(np.median(e[y])), 1), elev_median_negatives=round(float(np.median(e[~y])), 1),
                         within_landscape_elev_auc_descriptive=round(float(auc), 4),
                         other_five_min=round(float(lo), 1), other_five_max=round(float(hi), 1),
                         other_five_p01=round(float(p1), 1), other_five_p99=round(float(p99), 1),
                         frac_outside_other_five_minmax=round(float(((e < lo) | (e > hi)).mean()), 5),
                         frac_outside_other_five_p01_p99=round(float(((e < p1) | (e > p99)).mean()), 5),
                         frac_above_other_five_p99=round(float((e > p99).mean()), 5),
                         positives_frac_outside_other_five_p01_p99=round(float(((e[y] < p1) | (e[y] > p99)).mean()), 5)))
    # Landscape-identity statistics for elevation (same rule Stage 4A applied to E/G; M was only reported there)
    allv = np.concatenate([data[l]["e"] for l in LANDSCAPES])
    lab = np.concatenate([np.full(len(data[l]["e"]), i) for i, l in enumerate(LANDSCAPES)])
    w = np.concatenate([np.full(len(data[l]["e"]), 1.0 / len(data[l]["e"])) for l in LANDSCAPES])
    edges = np.unique(np.quantile(allv, np.linspace(0, 1, 101)))
    b = np.clip(np.searchsorted(edges, allv, side="right") - 1, 0, len(edges) - 2)
    W = np.zeros((len(edges) - 1, 6))
    np.add.at(W, (b, lab), w)
    tot = W.sum(1)
    pure = (W.max(1) / np.where(tot > 0, tot, 1)) >= 0.9
    ident["purity90_coverage_landscape_balanced"] = round(float(W[pure].sum() / W.sum()), 4)
    gm = np.average(allv, weights=w)
    between = sum((1 / 6) * (data[l]["e"].mean() - gm) ** 2 for l in LANDSCAPES)
    total = np.average((allv - gm) ** 2, weights=w)
    ident["between_landscape_variance_fraction"] = round(float(between / total), 4)
    ident["pure_bin_dominant_landscapes"] = {LANDSCAPES[i]: int(c) for i, c in zip(*np.unique(W[pure].argmax(1), return_counts=True))}
    # cross-landscape identifier / duplicate checks
    ids = np.concatenate([data[l]["ids"] for l in LANDSCAPES])
    dup = {l: int(pd.DataFrame({"x": data[l]["x"], "y": data[l]["yy"]}).duplicated().sum()) for l in LANDSCAPES}
    checks = dict(total_units=int(len(ids)), unique_ids=int(len(np.unique(ids))),
                  duplicate_centroids_within_landscape=dup, crs={l: data[l]["crs"] for l in LANDSCAPES},
                  positives=int(sum(data[l]["y"].sum() for l in LANDSCAPES)))
    # fold-specific standardisation and coefficients already frozen by Stage 4B / 5E
    fp = json.loads((V2 / "outputs/stage4b/STAGE4B_FOLD_PREPROCESSING.json").read_text())
    li = pd.read_csv(V2 / "outputs/stage5e/STAGE5E_LANDSCAPE_IMPORTANCE.tsv", sep="\t")
    pi = pd.read_csv(V2 / "outputs/stage5e/STAGE5E_PERMUTATION_IMPORTANCE.tsv", sep="\t")
    folds = []
    for lid in LANDSCAPES:
        f = fp[f"{lid}|M"]
        j = f["columns"].index("elev_m_30m__mean")
        mu, sd = f["mean"][j], f["sd"][j]
        r = li[(li.landscape == lid) & (li.feature == "elev_m_30m__mean")].iloc[0]
        p = pi[(pi.landscape == lid) & (pi.name == "elev_m_30m__mean") & (pi.level == "FEATURE")].iloc[0]
        e = data[lid]["e"]
        folds.append(dict(held_out=lid, train_elev_mean=round(mu, 2), train_elev_sd=round(sd, 2), fitted_on=f["fitted_on"],
                          heldout_median_z=round(float((np.median(e) - mu) / sd), 3),
                          standardized_coefficient=round(float(r.standardized_coefficient), 4),
                          mean_abs_shap=round(float(r.mean_abs_shap_log_odds), 4), signed_mean_shap=round(float(r.signed_mean_shap_log_odds), 4),
                          shap_rank=int(r.rank_mean_abs_shap), permutation_ap_drop=round(float(p.importance_mean), 5),
                          permutation_rank=int(p.rank_within_landscape)))
    return dict(per_landscape=rows, identity=ident, integrity=checks, folds=folds)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dem = dem_chronology()
    pd.DataFrame(dem).to_csv(OUT / "E01_DEM_TILE_CHRONOLOGY.tsv", sep="\t", index=False)
    lab = label_diagnostics()
    (OUT / "E02_LABEL_GEOMETRY_DIAGNOSTICS.json").write_text(json.dumps(lab, indent=2, sort_keys=True) + "\n")
    el = elevation_diagnostics()
    pd.DataFrame(el["per_landscape"]).to_csv(OUT / "E03_ELEVATION_DISTRIBUTIONS.tsv", sep="\t", index=False)
    pd.DataFrame(el["folds"]).to_csv(OUT / "E04_ELEVATION_FOLD_DIAGNOSTICS.tsv", sep="\t", index=False)
    (OUT / "E05_IDENTITY_AND_INTEGRITY.json").write_text(json.dumps({"elevation_identity": el["identity"], "integrity": el["integrity"]},
                                                                    indent=2, sort_keys=True) + "\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    sys.exit(main())
