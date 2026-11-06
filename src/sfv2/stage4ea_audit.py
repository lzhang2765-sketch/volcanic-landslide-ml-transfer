"""Stage 4E-A Phases A-E: Cotabato provenance, label geometry, trigger, conditional morphology, failure classification (diagnosis only)."""
from __future__ import annotations

import json
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from . import stage4a_base as B
from . import stage4b as S
from . import stage4c_b_prereg as P1
from . import stage4d as D
from . import stage4ea_cotabato_prereg as C
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_a"
FIG = OUT / "figures"
MEG = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET"
UNIT_KM2 = 0.0081
ZIP = ROOT / "data/raw/stage3c/COTABATO_DAVAO_2019/SRC_ZENODO_7520726/cotabato_davao_landslides.zip"
Y_SOURCES = {
    "NOTO_PENINSULA_2024": ROOT / "data/processed/stage2c/NOTO_PENINSULA_2024/y_harmonised_polygons.geojson",
    "LOMBOK_RINJANI_2018": ROOT / "data/raw/stage2d/LOMBOK_RINJANI_2018/V1_TRANSFER/Y/SF1_VERIFICATION_Ferrario_2019b.shp",
    "COTABATO_DAVAO_2019": f"zip://{ZIP}!landslides_ott2019.shp",
    "ATITLAN_STAN_2005": ROOT / "data/processed/stage3f/ATITLAN_STAN_2005/y_primary_stan_2005.geojson",
    "DOMINICA_ERIKA_2015": ROOT / "data/processed/stage3f/DOMINICA_ERIKA_2015/y_primary_erika_2015.geojson",
    "KUMAMOTO_ASO_2016": ROOT / "data/processed/stage2c/KUMAMOTO_ASO_2016/y_harmonised_points.geojson",
}
OMEGA_RAW_KM2 = {"KUMAMOTO_ASO_2016": 5713.634, "NOTO_PENINSULA_2024": None, "LOMBOK_RINJANI_2018": 1798.686, "COTABATO_DAVAO_2019": 653.18,
                 "ATITLAN_STAN_2005": 221.01, "DOMINICA_ERIKA_2015": 778.22}


# ------------------------------------------------------------------------------------------------ Phase A + C (records)
def provenance() -> pd.DataFrame:
    NR = "NOT_IN_PROJECT_RECORDS"
    rows = [
        dict(landscape="KUMAMOTO_ASO_2016", event="2016 Kumamoto earthquakes (April sequence)", event_date="2016-04-16 (photography 16, 19, 20 April 2016)",
             trigger_class="SEISMIC", trigger_subtype="earthquake sequence; primary target restricted to April EQ epoch (July rainfall-epoch points held out)",
             inventory_source="Geospatial Information Authority of Japan (GSI) aerial-photo landslide distribution", inventory_mapping_method="aerial-photo interpretation",
             inventory_geometry_type="POINT (occurrence centres)", label_tier="TIER_2 (POINT_EVENT_TIER2)", number_original_features="1,044 points (717 primary APRIL_EQ after de-duplication; 678 in analysis)",
             observation_domain_source="GSI April interpretation range", omega_definition="April interpretation range intersected with eligible volcanic geology, DEM-valid, non-water",
             inventory_completeness_statement=NR, mapping_resolution_or_scale="aerial photography (scale not recorded)", source_imagery="aerial photography",
             pre_event_imagery=NR, post_event_imagery="2016-04-16, 04-19, 04-20 (primary); July 2016 (sensitivity)", minimum_mapping_unit_if_known=NR,
             landslide_types_included=NR, source_area_vs_full_polygon_semantics="point = occurrence centre (no footprint)", runout_or_deposit_included="NOT_APPLICABLE (points)",
             cloud_or_visibility_limitations="none supplied", temporal_window="April 2016", known_inventory_biases="point labels mark one 90 m unit per landslide: footprint size not represented; July epoch excluded for rainfall contamination",
             citation_or_source_record="outputs/stage2c/KUMAMOTO_TRIGGER_DECISION.md; outputs/stage2c/y_harmonisation_summary.json; metadata/stage2b_raw_registry.json", confidence_in_metadata="MEDIUM"),
        dict(landscape="NOTO_PENINSULA_2024", event="2024 Noto Peninsula earthquake", event_date="2024-01-01", trigger_class="SEISMIC", trigger_subtype="single mainshock (magnitude " + NR + ")",
             inventory_source="GSI 2024 Noto slope-failure/deposition interpretation (GeoJSON)", inventory_mapping_method="orthophoto interpretation", inventory_geometry_type="POLYGON",
             label_tier="TIER_2 (POLYGON_AFFECTED_TIER2)", number_original_features="2,345 polygons (1,350 intersecting the analysis domain)", observation_domain_source="GSI interpretation range",
             omega_definition="interpretation range intersected with eligible volcanic geology, DEM-valid, non-water", inventory_completeness_statement=NR,
             mapping_resolution_or_scale="features >= ~100 m2", source_imagery="orthophotos", pre_event_imagery=NR, post_event_imagery="2024-01-02 to 2024-01-17",
             minimum_mapping_unit_if_known="~100 m2", landslide_types_included="slope failure + deposition (combined 'affected terrain')", source_area_vs_full_polygon_semantics="affected-terrain polygons (failure and deposit not separated)",
             runout_or_deposit_included="YES (deposition included)", cloud_or_visibility_limitations="none supplied", temporal_window="January 2024",
             known_inventory_biases="deposit areas labelled positive alongside source areas", citation_or_source_record="outputs/stage2c/y_harmonisation_summary.json; metadata/stage2b_raw_registry.json", confidence_in_metadata="MEDIUM"),
        dict(landscape="LOMBOK_RINJANI_2018", event="2018 Lombok earthquake sequence (primary phase)", event_date="2018-07-29 (Mw 6.4) + 2018-08-05 (Mw 6.9)", trigger_class="SEISMIC",
             trigger_subtype="sequence phase (two events) up to 5 Aug 2018; 19 Aug events excluded", inventory_source="Ferrario (2019b)", inventory_mapping_method="PlanetScope / Sentinel-2 pre/post comparison; pre-event landslides removed",
             inventory_geometry_type="POLYGON", label_tier="TIER_2", number_original_features="4,823 polygons (4.89 km2)", observation_domain_source="published mapping extent",
             omega_definition="mapping extent intersected with eligible volcanic geology, DEM-valid, non-water", inventory_completeness_statement="95% estimated completeness (source methods form)",
             mapping_resolution_or_scale="3-10 m imagery", source_imagery="PlanetScope, Sentinel-2", pre_event_imagery="YES (pre/post comparison)", post_event_imagery="after 2018-08-05",
             minimum_mapping_unit_if_known=NR, landslide_types_included=NR, source_area_vs_full_polygon_semantics="whole landslide polygons", runout_or_deposit_included=NR,
             cloud_or_visibility_limitations="none supplied", temporal_window="29 Jul - ~16 Aug 2018", known_inventory_biases="aggregates two shocks; Copernicus DEM edits post-date the event (STRICT sensitivity)",
             citation_or_source_record="outputs/stage2f2/TARGET_DEFINITION_FREEZE.json; outputs/stage2f2/y_audit.json", confidence_in_metadata="HIGH"),
        dict(landscape="COTABATO_DAVAO_2019", event="2019 Cotabato - Davao del Sur seismic sequence, EQ1 only", event_date="2019-10-16 (Mw 6.4, EQ1)", trigger_class="SEISMIC",
             trigger_subtype="first shock of a four-earthquake sequence (EQ2 Mw 6.6 29 Oct; EQ3 Mw 6.5 31 Oct; EQ4 Mw 6.8 15 Dec); primary target = EQ1 only",
             inventory_source="Ferrario et al., Zenodo 10.5281/zenodo.7520726 (landslides_ott2019)", inventory_mapping_method="manual mapping on 3 m PlanetScope images",
             inventory_geometry_type="POLYGON", label_tier="TIER_2", number_original_features="190 polygons (0.603 km2); auxiliary cumulative inventories 4,737 (post-EQ3) and 5,666 (post-EQ4)",
             observation_domain_source="area_study_October2019 (inventory-1 study area)", omega_definition="EQ1 study area intersected with eligible volcanic geology, DEM-valid, non-water",
             inventory_completeness_statement=NR, mapping_resolution_or_scale="3 m PlanetScope", source_imagery="PlanetScope", pre_event_imagery=NR,
             post_event_imagery="2019-10-24 and 2019-10-28 (before EQ2)", minimum_mapping_unit_if_known=NR, landslide_types_included=NR,
             source_area_vs_full_polygon_semantics="polygonal landslide inventory (parts not separated)", runout_or_deposit_included=NR, cloud_or_visibility_limitations="no visibility exclusions supplied",
             temporal_window="16-28 Oct 2019", known_inventory_biases="first, smallest shock of a sequence; very small and sparse inventory relative to the later cumulative inventories",
             citation_or_source_record="outputs/stage3d/COTABATO_EVENT_FREEZE.json; COTABATO_TARGET_QA.json; COTABATO_OMEGA_QA.json; Zenodo record 7520726 (raw)", confidence_in_metadata="HIGH"),
        dict(landscape="ATITLAN_STAN_2005", event="Hurricane Stan", event_date="2005-10", trigger_class="PRECIPITATION_CYCLONE", trigger_subtype="tropical cyclone rainfall",
             inventory_source="Machado et al. (2022) Atitlan Hurricane Stan inventory (V1 transfer lineage)", inventory_mapping_method="manual digitising on 0.5 m orthophotos (screening record)",
             inventory_geometry_type="POLYGON", label_tier="TIER_2", number_original_features="6,817 polygons (2.44 km2)", observation_domain_source="Boundary_Atitlan_UTM15N",
             omega_definition="published boundary (221 km2), all eligible", inventory_completeness_statement=NR, mapping_resolution_or_scale="0.5 m orthophotos (screening record)",
             source_imagery="orthophotos", pre_event_imagery=NR, post_event_imagery=NR, minimum_mapping_unit_if_known=NR, landslide_types_included=NR,
             source_area_vs_full_polygon_semantics="whole affected-terrain polygons", runout_or_deposit_included=NR, cloud_or_visibility_limitations="none required by source",
             temporal_window="October 2005", known_inventory_biases="M from NASADEM (no DEM-quality companion)", citation_or_source_record="outputs/stage3f/ATITLAN_EVENT_AUDIT.json; ATITLAN_TARGET_SEMANTICS_AUDIT.json; ATITLAN_OMEGA_AUDIT.json; metadata/inventory_manifest.csv",
             confidence_in_metadata="MEDIUM"),
        dict(landscape="DOMINICA_ERIKA_2015", event="Tropical Storm Erika", event_date="2015-08-25 to 2015-08-28 (peak rainfall 27 Aug)", trigger_class="PRECIPITATION_CYCLONE",
             trigger_subtype="tropical storm rainfall; Maria 2017 and flash-flood-only features excluded", inventory_source="Dominica landslide inventory, event code LS20150909DMA",
             inventory_mapping_method="expert interpretation of satellite imagery (screening record)", inventory_geometry_type="POLYGON", label_tier="TIER_2", number_original_features="1,548 polygons",
             observation_domain_source="three dated analysis extents minus cloud obstruction", omega_definition="union of dated extents minus all cloud polygons (87.4% retained)",
             inventory_completeness_statement=NR, mapping_resolution_or_scale=NR, source_imagery="satellite (sensor dates 2015-09-03, 09-05, 09-15)", pre_event_imagery=NR,
             post_event_imagery="2015-09-03 / 09-05 / 09-15", minimum_mapping_unit_if_known=NR, landslide_types_included=NR,
             source_area_vs_full_polygon_semantics="landslide polygon affected terrain; no source/transport/deposit fields", runout_or_deposit_included=NR,
             cloud_or_visibility_limitations="1,983 cloud polygons removed (112 km2)", temporal_window="Sep 2015", known_inventory_biases="digitised map geology; cloud-masked observation domain",
             citation_or_source_record="outputs/stage3f/DOMINICA_ERIKA_EVENT_AUDIT.json; DOMINICA_ERIKA_TARGET_SEMANTICS_AUDIT.json; DOMINICA_ERIKA_OMEGA_AUDIT.json", confidence_in_metadata="HIGH"),
    ]
    return pd.DataFrame(rows)


def trigger_audit() -> pd.DataFrame:
    t = [("KUMAMOTO_ASO_2016", "SEISMIC", "mainshock sequence 14-16 Apr 2016 (magnitudes NOT_IN_PROJECT_RECORDS)", "April epoch only; rainfall epoch excluded"),
         ("NOTO_PENINSULA_2024", "SEISMIC", "single mainshock 1 Jan 2024 (magnitude NOT_IN_PROJECT_RECORDS)", "one epoch"),
         ("LOMBOK_RINJANI_2018", "SEISMIC", "Mw 6.4 (29 Jul) + Mw 6.9 (5 Aug 2018)", "two shocks aggregated"),
         ("COTABATO_DAVAO_2019", "SEISMIC", "Mw 6.4 EQ1 (16 Oct 2019), first of four Mw > 6 shocks", "EQ1 only; later shocks' landslides deliberately excluded"),
         ("ATITLAN_STAN_2005", "PRECIPITATION_CYCLONE", "Hurricane Stan rainfall (intensity NOT_IN_PROJECT_RECORDS)", "one storm"),
         ("DOMINICA_ERIKA_2015", "PRECIPITATION_CYCLONE", "Tropical Storm Erika rainfall, peak 27 Aug 2015 (intensity NOT_IN_PROJECT_RECORDS)", "one storm")]
    rows = []
    for lid, cls, mag, epoch in t:
        train = [x for x in t if x[0] != lid]
        rows.append(dict(landscape=lid, trigger_class=cls, magnitude_or_intensity_in_records=mag, epoch_definition=epoch,
                         training_landscapes_sharing_trigger_when_held_out=sum(x[1] == cls for x in train),
                         spatially_uniform_triggering_known="NOT_IN_PROJECT_RECORDS (no shaking / rainfall field acquired)",
                         trigger_used_as_predictor="NO", note="diagnostic only; trigger intensity is not a Stage 4E-A predictor"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Phase B
def unit_table(lid):
    t = B.load(lid, "terrain_units/terrain_units", usecols=["terrain_unit_id", "source_grid_row", "source_grid_col"])
    with np.load(MEG / f"{lid}.npz", allow_pickle=False) as z:
        assert (z["terrain_unit_id"] == t.terrain_unit_id.to_numpy()).all()
        d = {k: z[k] for k in ("y_positive", "centroid_x", "centroid_y", "spatial_block_3000m", "slope_deg_90m__mean", "relief_m_r250__mean")}
    t = t.assign(**d)
    t["ur"], t["uc"] = t.source_grid_row // 3, t.source_grid_col // 3
    return t


def dem_grid(lid):
    s = P1.SOURCES[lid]
    with rasterio.open(ROOT / "data/processed" / s["dir"] / lid / s["dem"]) as r:
        return r.transform, r.shape, r.crs


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else np.nan


def label_geometry() -> tuple[pd.DataFrame, dict]:
    rows, keep = [], {}
    for lid in C.LANDSCAPES:
        u = unit_table(lid)
        y = u.y_positive.to_numpy(bool)
        g = np.zeros((u.ur.max() + 1, u.uc.max() + 1), bool)
        g[u.ur[y], u.uc[y]] = True
        lab, n = ndimage.label(g, np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        xy = u.loc[y, ["centroid_x", "centroid_y"]].to_numpy()
        tree = cKDTree(xy)
        nn = tree.query(xy, k=2)[0][:, 1] if len(xy) > 1 else np.array([np.nan])
        dens = np.array([len(v) - 1 for v in tree.query_ball_point(xy, 1000)])
        blk = u.groupby("spatial_block_3000m").y_positive.sum().sort_values(ascending=False)
        row = dict(landscape=lid, inventory_type="POINT" if lid in C.POINT_INVENTORIES else "POLYGON", units=len(u), positive_units=int(y.sum()), prevalence=float(y.mean()),
                   analysis_area_km2=len(u) * UNIT_KM2, components=int(n), component_size_p50=float(np.median(sizes)), component_size_p90=float(np.percentile(sizes, 90)),
                   component_size_max=int(sizes.max()), share_positives_in_components_ge10=float(sizes[sizes >= 10].sum() / y.sum()),
                   nearest_positive_m_p50=float(np.median(nn)), nearest_positive_m_p90=float(np.percentile(nn, 90)), positives_within_1km_p50=float(np.median(dens)),
                   gini_over_3km_blocks=gini(blk.values), share_positives_top10pct_blocks=float(blk.head(max(1, len(blk) // 10)).sum() / blk.sum()),
                   pos_slope90_p25_p50_p75="|".join(f"{v:.1f}" for v in np.percentile(u.slope_deg_90m__mean[y], [25, 50, 75])),
                   pos_relief250_p25_p50_p75="|".join(f"{v:.0f}" for v in np.percentile(u.relief_m_r250__mean[y], [25, 50, 75])))
        if lid in C.POLYGON_INVENTORIES:
            T, shape, crs = dem_grid(lid)
            p = gpd.read_file(Y_SOURCES[lid]).to_crs(crs)
            p = p[p.geometry.notna() & ~p.geometry.is_empty].reset_index(drop=True)
            p["geometry"] = p.geometry.buffer(0)
            rp = p.geometry.representative_point()
            col = ((rp.x - T.c) / 30).astype(int)
            rr = ((T.f - rp.y) / 30).astype(int)
            inunit = set(zip(u.ur, u.uc))
            sel = np.array([(r // 3, c // 3) in inunit for r, c in zip(rr, col)])
            p = p[sel].reset_index(drop=True)
            idx = rasterize([(geom, i + 1) for i, geom in enumerate(p.geometry)], out_shape=shape, transform=T, fill=0, dtype="int32")
            rr_, cc_ = np.nonzero(idx)
            pid = idx[rr_, cc_]
            cells = np.bincount(pid, minlength=len(p) + 1)[1:]
            units_hit = pd.DataFrame(dict(p=pid, b=(rr_ // 3) * 100000 + cc_ // 3)).drop_duplicates().groupby("p").size().reindex(range(1, len(p) + 1), fill_value=0).to_numpy()
            area = p.geometry.area.to_numpy()
            omega = len(u) * UNIT_KM2
            row.update(source_polygons_in_domain=len(p), polygon_area_m2_p10=float(np.percentile(area, 10)), polygon_area_m2_p50=float(np.median(area)),
                       polygon_area_m2_p90=float(np.percentile(area, 90)), y_area_over_omega=float(area.sum() / 1e6 / omega), polygons_per_km2_omega=float(len(p) / omega),
                       cells_hit_per_polygon_p50=float(np.median(cells)), units_hit_per_polygon_p50=float(np.median(units_hit)), units_hit_per_polygon_p90=float(np.percentile(units_hit, 90)),
                       share_polygons_hitting_no_cell_centre=float((cells == 0).mean()))
            keep[lid] = dict(area=area, units_hit=units_hit)
        rows.append(row)
    return pd.DataFrame(rows), keep


# ------------------------------------------------------------------------------------------------ Phase D
def audited_predictors() -> list[str]:
    fi = pd.read_csv(ROOT / "outputs/stage4b/STAGE4B_FEATURE_INTERPRETATION.tsv", sep="\t")
    f = fi[(fi.kind == "feature") & ~fi.unit.str.startswith(("e_", "g_"))]
    top = f.groupby("unit").permutation_pr_auc_drop.mean().sort_values(ascending=False).head(10).index.tolist()
    extra = ["slope_deg_90m__mean", "relief_m_r250__mean", "tpi_m_r450__mean", "roughness_sd_m_90m__mean", "curv_total_1pm_90m__mean", "rel_elev_local_r1000__mean"]
    return top + [e for e in extra if e not in top]


def conditional_m() -> tuple[pd.DataFrame, dict]:
    preds = audited_predictors()
    data = {}
    for lid in C.LANDSCAPES:
        with np.load(MEG / f"{lid}.npz", allow_pickle=False) as z:
            data[lid] = dict(y=z["y_positive"].astype(bool), X=np.column_stack([z[c].astype(float) for c in preds]))
    rows = []
    k = "COTABATO_DAVAO_2019"
    train = [l for l in C.LANDSCAPES if l != k]
    for j, c in enumerate(preds):
        auc = {l: float(roc_auc_score(data[l]["y"], data[l]["X"][:, j])) for l in C.LANDSCAPES}
        Xt = np.concatenate([data[l]["X"][:, j] for l in train])
        yt = np.concatenate([data[l]["y"] for l in train])
        wt = np.concatenate([np.full(len(data[l]["y"]), 1 / len(data[l]["y"])) for l in train])
        pooled = float(roc_auc_score(yt, Xt, sample_weight=wt))
        tp = np.concatenate([data[l]["X"][data[l]["y"], j] for l in train])
        tn = np.concatenate([data[l]["X"][~data[l]["y"], j] for l in train])
        iq = lambda a: (np.subtract(*np.percentile(a, [75, 25])) or np.std(a) or 1.0)
        cp, cn = data[k]["X"][data[k]["y"], j], data[k]["X"][~data[k]["y"], j]
        rev = (abs(auc[k] - 0.5) >= 0.03) and (abs(pooled - 0.5) >= 0.03) and (np.sign(auc[k] - 0.5) != np.sign(pooled - 0.5))
        rows.append(dict(predictor=c, stage4b_top10=c in preds[:10], cotabato_auc=auc[k], pooled_training_auc=pooled, **{f"auc_{l[:4]}": v for l, v in auc.items() if l != k},
                         cotabato_abs_effect=abs(auc[k] - 0.5), others_median_abs_effect=float(np.median([abs(auc[l] - 0.5) for l in train])),
                         reversed_vs_training=bool(rev), cot_pos_vs_train_pos_std_median_diff=float((np.median(cp) - np.median(tp)) / iq(tp)),
                         cot_neg_vs_train_neg_std_median_diff=float((np.median(cn) - np.median(tn)) / iq(tn)),
                         cot_pos_median=float(np.median(cp)), cot_neg_median=float(np.median(cn)), train_pos_median=float(np.median(tp)), train_neg_median=float(np.median(tn))))
    return pd.DataFrame(rows), data


def binned_rates(data, preds, names=("slope_deg_90m__mean", "relief_m_r250__mean")) -> pd.DataFrame:
    rows = []
    k = "COTABATO_DAVAO_2019"
    train = [l for l in C.LANDSCAPES if l != k]
    for c in names:
        j = preds.index(c)
        edges = np.unique(np.quantile(np.concatenate([data[l]["X"][:, j] for l in train]), np.linspace(0, 1, 11)))
        for who, ls in (("COTABATO", [k]), ("TRAINING_POOLED_BALANCED", train)):
            for b in range(len(edges) - 1):
                rates = []
                for l in ls:
                    x, y = data[l]["X"][:, j], data[l]["y"]
                    m = (x >= edges[b]) & ((x < edges[b + 1]) | (b == len(edges) - 2))
                    rates.append((y[m].mean() / y.mean()) if m.sum() else np.nan)
                rows.append(dict(predictor=c, group=who, decile=b + 1, lo=edges[b], hi=edges[b + 1], relative_event_rate=float(np.nanmean(rates))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ auxiliary inventory test
def aux_inventory_test() -> pd.DataFrame:
    lid = "COTABATO_DAVAO_2019"
    T, shape, crs = dem_grid(lid)
    u = unit_table(lid)
    r0, c0 = u.source_grid_row.to_numpy(), u.source_grid_col.to_numpy()

    def unit_hits(layer):
        g = gpd.read_file(f"zip://{ZIP}!{layer}.shp").to_crs(crs)
        m = rasterize([(geom, 1) for geom in g.geometry.buffer(0) if geom is not None and not geom.is_empty], out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)
        hit = np.zeros(len(u), bool)
        for dr in range(3):
            for dc in range(3):
                hit |= m[np.clip(r0 + dr, 0, shape[0] - 1), np.clip(c0 + dc, 0, shape[1] - 1)]
        return hit, len(g)
    area = gpd.read_file(f"zip://{ZIP}!area_study.shp").to_crs(crs)
    am = rasterize([(geom, 1) for geom in area.geometry], out_shape=shape, transform=T, fill=0, dtype="uint8").astype(bool)
    inside = am[np.clip(r0 + 1, 0, shape[0] - 1), np.clip(c0 + 1, 0, shape[1] - 1)]
    y1, n1 = unit_hits("landslides_ott2019")
    labels = {"EQ1_primary": (y1, n1), "cumulative_post_EQ3_nov2019": unit_hits("landslides_nov2019"), "cumulative_post_EQ4_dic2019": unit_hits("landslides_dic2019")}
    yref = u.y_positive.to_numpy(bool)
    full = json.loads((ROOT / "data/interim/_stage4b_results/PRIMARY.json").read_text())["results"]
    rows = []
    for model, fs in (("RANDOM_FOREST", "MEG"), ("LOGISTIC_L2", "M")):
        r = next(x for x in full if x["held_out"] == lid and x["model"] == model and x["feature_set"] == fs)
        key = S._key(sorted([l for l in C.LANDSCAPES if l != lid]), fs, model, json.loads(r["params"]), "ALL_VALID", "LANDSCAPE_BALANCED")
        p = np.load(ROOT / f"data/interim/_stage4b_cache/{key}__{lid}.npy")
        code, nb = D.block_codes(u.spatial_block_3000m.to_numpy())
        m = inside
        srt = {name: D.Sorted(p[m], lab[m]) for name, (lab, _) in labels.items()}
        rng = np.random.default_rng(C.SEED)
        reps = {name: [] for name in labels}
        for _ in range(2000):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code].astype(float)[m]
            for name in labels:
                reps[name].append(srt[name].ap_auc(w))
        base = np.array([a[1] for a in reps["EQ1_primary"]])
        for name, (lab, nfeat) in labels.items():
            y = lab[m]
            arr = np.array(reps[name])
            dd = arr[:, 1] - base
            rows.append(dict(model=model, feature_set=fs, label_set=name, source_polygons=nfeat, units_evaluated=int(m.sum()), positive_units=int(y.sum()), prevalence=float(y.mean()),
                             roc_auc=float(roc_auc_score(y, p[m])), roc_ci_low=float(np.nanpercentile(arr[:, 1], 2.5)), roc_ci_high=float(np.nanpercentile(arr[:, 1], 97.5)),
                             pr_lift=float(D.average_precision_score(y, p[m]) / y.mean()),
                             delta_roc_vs_eq1=float(roc_auc_score(y, p[m]) - roc_auc_score(labels["EQ1_primary"][0][m], p[m])),
                             delta_roc_ci_low=float(np.nanpercentile(dd, 2.5)), delta_roc_ci_high=float(np.nanpercentile(dd, 97.5)),
                             eq1_rule_reproduces_frozen_target=bool((labels["EQ1_primary"][0] == yref).all()), eq1_mismatch_units=int((labels["EQ1_primary"][0] != yref).sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Phase E
def classify(geo: pd.DataFrame, cond: pd.DataFrame, aux: pd.DataFrame, trig: pd.DataFrame) -> pd.DataFrame:
    ds = pd.read_csv(ROOT / "outputs/stage4d/STAGE4D_DOMAIN_SHIFT.tsv", sep="\t")
    hm = ds[ds.group == "M"].set_index("held_out").energy_H_primary
    fx = pd.read_csv(ROOT / "outputs/stage4d/STAGE4D_FORENSICS.tsv", sep="\t")
    k = "COTABATO_DAVAO_2019"
    pos_ood = float(fx[(fx.landscape == k) & (fx.model == "RANDOM_FOREST")].median_ood_percentile_positives.iloc[0])
    pg = geo[geo.inventory_type == "POLYGON"].set_index("landscape")
    mins = sum(pg[c].idxmin() == k for c in ("y_area_over_omega", "polygons_per_km2_omega", "polygon_area_m2_p50"))
    rf = aux[(aux.model == "RANDOM_FOREST") & (aux.label_set != "EQ1_primary")]
    l2 = bool(((rf.delta_roc_vs_eq1 >= 0.05) & (rf.delta_roc_ci_low > 0)).any())
    l3 = bool(pg.units_hit_per_polygon_p50.idxmin() == k and pg.loc[k, "units_hit_per_polygon_p50"] <= 1)
    t1 = bool(cond.cotabato_abs_effect.median() < 0.5 * cond.others_median_abs_effect.median())
    t2 = int(cond.reversed_vs_training.sum())
    t3 = int(trig.set_index("landscape").loc[k, "training_landscapes_sharing_trigger_when_held_out"])
    crit = {
        "C1": (bool(hm[k] >= hm.drop(k).median()), f"H_M {hm[k]:.4f} vs others' median {hm.drop(k).median():.4f}"),
        "C2": (pos_ood >= 0.90, f"median OOD percentile of positives {pos_ood:.3f}"),
        "L1": (mins >= 2, f"Cotabato is the minimum in {mins}/3 inventory-density metrics"),
        "L2": (l2, "; ".join(f"{r.label_set}: dROC {r.delta_roc_vs_eq1:+.3f} [{r.delta_roc_ci_low:+.3f}, {r.delta_roc_ci_high:+.3f}]" for r in rf.itertuples())),
        "L3": (l3, f"median units hit per polygon {pg.loc[k, 'units_hit_per_polygon_p50']:.1f}; minimum is {pg.units_hit_per_polygon_p50.idxmin()}"),
        "T1": (t1, f"Cotabato median |AUC-0.5| {cond.cotabato_abs_effect.median():.3f} vs others {cond.others_median_abs_effect.median():.3f}"),
        "T2": (t2 >= 3, f"{t2} reversed predictors"),
        "T3": (t3 <= 1, f"{t3} training landscapes share the SEISMIC trigger"),
    }
    sup = {"COVARIATE": sum(crit[c][0] for c in ("C1", "C2")), "LABEL_OR_INVENTORY": sum(crit[c][0] for c in ("L1", "L2", "L3")),
           "TRIGGER_OR_CONCEPT": sum(crit[c][0] for c in ("T1", "T2", "T3"))}
    order = ["LABEL_OR_INVENTORY", "TRIGGER_OR_CONCEPT", "COVARIATE"]
    supported = [c for c in order if sup[c] > 0]
    name = {"COVARIATE": "COVARIATE_SHIFT_DOMINATED", "LABEL_OR_INVENTORY": "LABEL_OR_INVENTORY_SHIFT_PLAUSIBLE", "TRIGGER_OR_CONCEPT": "TRIGGER_OR_CONCEPT_SHIFT_PLAUSIBLE"}
    if not supported:
        primary, dom, sec = "NO_CLEAR_EXPLANATION", "", ""
    else:
        ranked = sorted(supported, key=lambda c: (-sup[c], order.index(c)))
        dom = ranked[0]
        sec = ranked[1] if len(ranked) > 1 else ""
        primary = "MIXED_SHIFT" if len(supported) >= 2 else name[dom]
    comp = max([sup[c] for c in sup if c != dom] or [0]) if dom else 0
    conf = "LOW"
    if dom and sup[dom] >= 2 and comp <= 1:
        conf = "HIGH"
    elif dom and sup[dom] >= 1 and sup[dom] > comp:
        conf = "MEDIUM"
    rows = [dict(item=c, satisfied=v[0], evidence=v[1]) for c, v in crit.items()]
    rows.append(dict(item="SUPPORT_COUNTS", satisfied=None, evidence=json.dumps(sup)))
    rows.append(dict(item="PRIMARY_CLASS", satisfied=None, evidence=primary))
    rows.append(dict(item="DOMINANT_COMPONENT", satisfied=None, evidence=name.get(dom, "")))
    rows.append(dict(item="SECONDARY_CLASS", satisfied=None, evidence=name.get(sec, "")))
    rows.append(dict(item="CONFIDENCE", satisfied=None, evidence=conf))
    rows.append(dict(item="POLICY", satisfied=None, evidence=C.DECISION["not_permitted"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures 1-4
def figures(geo, keep, cond, data, bins, prov, trig, aux):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    meta = {"Software": None}
    sh = {l: l.split("_")[0].title() for l in C.LANDSCAPES}
    col = lambda l: "tab:red" if l.startswith("COTABATO") else "tab:blue"
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    g = geo.set_index("landscape").reindex(C.LANDSCAPES)
    for a, (c, t) in zip(ax.ravel(), [("prevalence", "Positive-unit prevalence"), ("positives_within_1km_p50", "Median positives within 1 km of a positive"),
                                     ("share_positives_top10pct_blocks", "Share of positives in top 10% of 3 km blocks"), ("component_size_p50", "Median positive-cluster size (units)"),
                                     ("y_area_over_omega", "Mapped landslide area / Omega area (polygons)"), ("units_hit_per_polygon_p50", "Median 90 m units hit per polygon")]):
        a.bar([sh[l] for l in C.LANDSCAPES], g[c], color=[col(l) for l in C.LANDSCAPES])
        a.set_title(t); a.tick_params(axis="x", rotation=30); a.set_ylim(0, None)
    fig.suptitle("Cotabato (red) vs other development inventories: label geometry (Kumamoto = point labels; polygon metrics not applicable)")
    fig.savefig(FIG / "fig1_label_geometry.png", dpi=110, bbox_inches="tight", metadata=meta); plt.close(fig)
    preds = cond.predictor.tolist()
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    k = "COTABATO_DAVAO_2019"
    for a, c in zip(ax.ravel(), preds[:8]):
        j = preds.index(c)
        x, y = data[k]["X"][:, j], data[k]["y"]
        lo, hi = np.percentile(x, [0.5, 99.5])
        b = np.linspace(lo, hi, 40)
        a.hist(x[~y], b, density=True, alpha=.6, label="negatives"); a.hist(x[y], b, density=True, alpha=.6, label="positives")
        a.set_title(f"{c}\nCotabato AUC {cond.set_index('predictor').loc[c, 'cotabato_auc']:.2f} (training {cond.set_index('predictor').loc[c, 'pooled_training_auc']:.2f})", fontsize=8)
    ax[0, 0].legend()
    fig.suptitle("Cotabato: positive vs negative distributions of the top Stage 4B morphology predictors")
    fig.savefig(FIG / "fig2_cotabato_pos_neg_morphology.png", dpi=110, bbox_inches="tight", metadata=meta); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for a, c in zip(ax[:2], ("slope_deg_90m__mean", "relief_m_r250__mean")):
        for who, st in (("COTABATO", "o-"), ("TRAINING_POOLED_BALANCED", "s--")):
            b = bins[(bins.predictor == c) & (bins.group == who)]
            a.plot(b.decile, b.relative_event_rate, st, label=who)
        a.axhline(1, color="grey", lw=1); a.set_xlabel(f"decile of pooled-training {c}"); a.set_ylabel("event rate / landscape prevalence"); a.set_ylim(0, None); a.legend()
        a.set_title(f"P(Y | {c}) shape")
    cc = cond.set_index("predictor")
    ax[2].scatter(cc.pooled_training_auc, cc.cotabato_auc)
    for p_, r in cc.iterrows():
        ax[2].annotate(p_.replace("__mean", ""), (r.pooled_training_auc, r.cotabato_auc), fontsize=6)
    ax[2].plot([0.3, 0.8], [0.3, 0.8], "k--", lw=1); ax[2].axhline(0.5, color="grey", lw=1); ax[2].axvline(0.5, color="grey", lw=1)
    ax[2].set_xlabel("univariate AUC, pooled training"); ax[2].set_ylabel("univariate AUC, Cotabato"); ax[2].set_title("M -> Y association: Cotabato vs training")
    fig.suptitle("P(M) can be similar while P(Y | M) differs: Cotabato vs pooled training landscapes")
    fig.savefig(FIG / "fig3_conditional_morphology.png", dpi=110, bbox_inches="tight", metadata=meta); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    mat = pd.DataFrame({sh[r.landscape]: [r.trigger_class, r.inventory_geometry_type, r.mapping_resolution_or_scale[:28], r.number_original_features.split("(")[0][:22],
                                          r.post_event_imagery[:28]] for r in prov.itertuples()}, index=["trigger", "geometry", "resolution", "features", "post-event imagery"])
    ax[0].axis("off")
    tb = ax[0].table(cellText=mat.values, rowLabels=mat.index, colLabels=mat.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 2)
    ax[0].set_title("Inventory / trigger comparison (project records)")
    a2 = aux[aux.model == "RANDOM_FOREST"]
    ax[1].errorbar(range(len(a2)), a2.roc_auc, yerr=[a2.roc_auc - a2.roc_ci_low, a2.roc_ci_high - a2.roc_auc], fmt="o", capsize=5)
    ax[1].set_xticks(range(len(a2)), [f"{s}\n({n} polygons)" for s, n in zip(a2.label_set, a2.source_polygons)], fontsize=8)
    ax[1].axhline(0.5, color="grey", ls="--"); ax[1].set_ylim(0.3, 0.9); ax[1].set_ylabel("ROC-AUC of the existing Stage 4B RF+MEG held-out predictions")
    ax[1].set_title("Same predictions, different Cotabato inventories (diagnostic)")
    fig.savefig(FIG / "fig4_trigger_inventory.png", dpi=110, bbox_inches="tight", metadata=meta); plt.close(fig)
