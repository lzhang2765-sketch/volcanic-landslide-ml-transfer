"""Build Stage 4A: six-landscape leakage-safe MEG development dataset (Y + frozen M + governed E + harmonised G).

No model is trained, no LOLO is run, no H is built, and no future lockbox is read.  Prior checkpoints and production tables are
read only.  Predictor admission rules are fixed in POLICY before any audit statistic is computed.
"""
from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import stage4a_base as B
from . import stage4a_e as E
from . import stage4a_g as G
from .build_stage3a import scan_ml_h_artifacts
from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

AS_OF = "2026-10-06"
OUT = ROOT / "outputs/stage4a"
MEG_DIR = OUT / "STAGE4A_MEG_DATASET"
PRIOR = {"stage3f": "STAGE3F_CHECKPOINT.json", "stage3g": "STAGE3G_CHECKPOINT.json", "stage3h": "STAGE3H_CHECKPOINT.json",
         "stage3i": "STAGE3I_CHECKPOINT.json", "stage3j": "STAGE3J_CHECKPOINT.json"}
FROZEN_M_CONTRACT = ROOT / "outputs/stage2f/production/frozen_m_feature_contract.json"

POLICY = dict(  # fixed a priori (before computing any statistic)
    identity_rule=("A candidate E/G predictor is EXCLUDED from the primary MEG set if, under landscape-balanced weights, >=90% of weight lies in "
                   "value bins (100 pooled quantile bins, or categories) where one landscape holds >=90% of the bin weight; 50-90% -> admitted "
                   "with IDENTITY_RISK flag; frozen M is never modified, only reported"),
    support_rule="A categorical G level must occur in >=3 development landscapes, otherwise the representation is rejected",
    missing_rule="No zero-fill, no imputation; NaN is preserved and audited",
    e_resolution_rule="CHELSA (~0.9 km) is CONTEXTUAL: value of the containing cell, no SD; Hansen (~30 m) LOCAL: 9-cell mean and SD",
    target_rule="Y, Omega, terrain units and M are frozen and read only",
)
IDENTITY_EXCLUDE, IDENTITY_FLAG, CONTEXT_FLAG = 0.90, 0.50, 0.95


def identity_flag(st: dict) -> str:
    flags = []
    if st["identity_purity90_coverage"] >= IDENTITY_EXCLUDE: flags.append("EXCLUDED_IDENTITY")
    elif st["identity_purity90_coverage"] >= IDENTITY_FLAG: flags.append("IDENTITY_RISK")
    if st["between_landscape_variance_fraction"] >= CONTEXT_FLAG: flags.append("LANDSCAPE_CONSTANT_CONTEXT")
    return "|".join(flags) or "NONE"


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def _tsv(p: Path, df) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(p, sep="\t", index=False, lineterminator="\n", float_format="%.6g")


# ------------------------------------------------------------------------------------------------ typed columnar storage
def write_columns(path: Path, df: pd.DataFrame) -> None:
    """Deterministic typed column archive (.npz readable by numpy.load): one .npy per column, fixed zip timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for col in df.columns:
            arr = df[col].to_numpy()
            if arr.dtype == object:
                arr = arr.astype(str)
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{col}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buf.getvalue())


def read_columns(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame({k: z[k] for k in z.files})


# ------------------------------------------------------------------------------------------------ Part I: base
def load_base() -> tuple[dict, dict]:
    tables, checks = {}, {}
    contract = json.loads(FROZEN_M_CONTRACT.read_text())
    m_cols = None
    for lid in B.DEVELOPMENT:
        t = B.load(lid, "terrain_units/terrain_units")
        f = B.load(lid, "features_m/features_m")
        y = B.load(lid, "targets/targets_primary")
        assert (t.terrain_unit_id.values == f.terrain_unit_id.values).all() and (t.terrain_unit_id.values == y.terrain_unit_id.values).all()
        m_cols = m_cols or [c for c in f.columns if c != "terrain_unit_id"]
        assert [c for c in f.columns if c != "terrain_unit_id"] == m_cols
        strict = np.where(t.analysis_population_strict_dem_quality.astype(bool), "IN", "OUT").astype(object)
        pop_dir = B.path(lid, "populations")
        if pop_dir.exists() and (pop_dir / "STRICT_DEM_QUALITY.csv.gz").exists():
            s = pd.read_csv(pop_dir / "STRICT_DEM_QUALITY.csv.gz")
            assert set(s.terrain_unit_id) == set(t.terrain_unit_id[t.analysis_population_strict_dem_quality.astype(bool)])
        if lid in B.DEM_SOURCE:  # no defensible DEM-quality companion: never fabricate a STRICT population
            strict = np.full(len(t), "NOT_AVAILABLE", dtype=object)
        unit_ok = bool(((t.source_grid_row % 3 == 0) & (t.source_grid_col % 3 == 0)).all() and (t.cell_area_m2 == 8100).all()
                       and (t.valid_fraction >= 5 / 9).all() and t.terrain_unit_id.is_unique)
        tables[lid] = dict(t=t, f=f, y=y, strict=strict)
        checks[lid] = dict(rows=len(t), positives=int(y.y_positive.sum()), negatives=int((~y.y_positive.astype(bool)).sum()),
                           target_tiers=sorted(y.target_tier.unique()), target_family=t.target_family.iloc[0], unit_geometry_ok=unit_ok,
                           strict_units=int((strict == "IN").sum()), strict_available=lid not in B.DEM_SOURCE,
                           dem_source=B.DEM_SOURCE.get(lid, "COPERNICUS_GLO30_WITH_EDM_FLM"))
    tot = dict(rows=sum(c["rows"] for c in checks.values()), positives=sum(c["positives"] for c in checks.values()),
               negatives=sum(c["negatives"] for c in checks.values()))
    return tables, dict(per_landscape=checks, totals=tot, matches_expected=tot == B.EXPECTED, m_columns=m_cols,
                        m_contract_sha256=sha256(ROOT / "outputs/stage2d/feature_definitions.csv"),
                        m_contract_matches=sha256(ROOT / "outputs/stage2d/feature_definitions.csv") in json.dumps(contract),
                        tier2_only=all(c["target_tiers"] == ["TIER_2"] for c in checks.values()))


# ------------------------------------------------------------------------------------------------ identity / domain statistics
def _weights(lids: np.ndarray) -> np.ndarray:
    u, inv, n = np.unique(lids, return_inverse=True, return_counts=True)
    return 1.0 / (n[inv] * len(u))


def identity_stats(x: pd.Series, lids: np.ndarray, categorical: bool) -> dict:
    ok = x.notna().to_numpy()
    xv, lv = x.to_numpy()[ok], lids[ok]
    w = _weights(lv)
    if categorical:
        bins = pd.factorize(xv)[0]
    else:
        edges = np.unique(np.quantile(xv.astype(float), np.linspace(0, 1, 101)))
        bins = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, max(len(edges) - 2, 0))
    df = pd.DataFrame({"b": bins, "l": lv, "w": w}).groupby(["b", "l"]).w.sum().unstack(fill_value=0.0)
    bw = df.sum(axis=1)
    purity = df.max(axis=1) / bw
    coverage = float(bw[purity >= 0.9].sum() / bw.sum())
    xs = pd.Series(xv.astype(float))
    grand = np.average(xs, weights=w)
    means = pd.DataFrame({"x": xs, "l": lv, "w": w}).groupby("l").apply(lambda d: np.average(d.x, weights=d.w), include_groups=False)
    between = float(np.mean((means - grand) ** 2))
    total = float(np.average((xs - grand) ** 2, weights=w))
    return dict(identity_purity90_coverage=round(coverage, 4), between_landscape_variance_fraction=round(between / total, 4) if total > 0 else 1.0)


# ------------------------------------------------------------------------------------------------ main
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    prior = {f"outputs/{k}/{v}": sha256(ROOT / f"outputs/{k}/{v}") for k, v in PRIOR.items()}  # keyed by path, as in every V2 checkpoint
    j = json.loads((ROOT / "outputs/stage3j/STAGE3J_CHECKPOINT.json").read_text())
    prior_ok = all(prior[f"outputs/{k}/{PRIOR[k]}"] == j["prior_checkpoint_hashes"][k] for k in ("stage3f", "stage3g", "stage3h", "stage3i"))
    tables, base = load_base()
    e_acq = E.acquire({lid: d["t"] for lid, d in tables.items()})  # idempotent: windows already registered are reused

    frames, xwalks, gaudit = [], [], {}
    for lid, d in tables.items():
        t, f, y = d["t"], d["f"], d["y"]
        ef = E.features(lid, t)
        gf = G.features(t)
        ga = G.unit_audit(lid, t)
        assert (ga.qa_geology_recomputed_index.values == t.geology_index.values).all(), f"STOP: frozen geology rule not reproduced for {lid}"
        xwalks.append(G.crosswalk(lid, t))
        gaudit[lid] = dict(units=len(t), missing_geology=int((t.geology_index == 0).sum()), boundary_fraction=round(float(ga.qa_geology_boundary_unit.mean()), 5),
                           tie_fraction=round(float(ga.qa_geology_tie_unit.mean()), 6), coverage_fraction=round(float((t.geology_index > 0).mean()), 6),
                           frozen_rule_reproduced=True)
        pub, prod, scale, kind, crs = G.SOURCE[lid]
        df = pd.DataFrame({
            "landscape_id": lid, "terrain_unit_id": t.terrain_unit_id.to_numpy(), "spatial_block_3000m": t.spatial_block_3000m.to_numpy(),
            "spatial_block_6000m": t.spatial_block_6000m.to_numpy(), "centroid_x": t.centroid_x.to_numpy(), "centroid_y": t.centroid_y.to_numpy(),
            "projected_crs": t.projected_crs.to_numpy(), "y_positive": y.y_positive.astype(bool).to_numpy(), "target_tier": y.target_tier.to_numpy(),
            "target_family": t.target_family.to_numpy(), "target_source": y.target_source.to_numpy(),
            "population_all_valid": t.analysis_population_all_valid.astype(bool).to_numpy(), "population_strict_dem_quality": d["strict"]})
        df = pd.concat([df, f.drop(columns="terrain_unit_id").reset_index(drop=True), ef.drop(columns="terrain_unit_id"),
                        gf.drop(columns="terrain_unit_id"), ga.drop(columns=["terrain_unit_id", "qa_geology_recomputed_index"])], axis=1)
        df["qa_dem_edited_fraction"] = t.dem_edited_fraction.to_numpy()
        df["qa_dem_source"] = B.DEM_SOURCE.get(lid, "COPERNICUS_GLO30_WITH_EDM_FLM")
        df["qa_harmonised_geology_class"] = t.harmonised_geology_class.to_numpy()
        df["qa_source_unit_code"] = t.source_unit_code.to_numpy()
        df["qa_geology_mapping_confidence"] = t.geology_mapping_confidence.to_numpy()
        df["qa_geology_source_quality"] = G.quality_class(scale, kind)
        frames.append(df)
    meg = pd.concat(frames, ignore_index=True)
    lids = meg.landscape_id.to_numpy()
    m_cols = base["m_columns"]
    e_cand = [c for c in meg.columns if c.startswith("e_")]
    g_cand = [c for c in meg.columns if c.startswith("g_")]

    # ---- identity / admission (rules fixed in POLICY)
    ident = {}
    for c in m_cols + e_cand + g_cand:
        ident[c] = identity_stats(meg[c], lids, categorical=c.startswith("g_"))
    e_adm = [c for c in e_cand if ident[c]["identity_purity90_coverage"] < IDENTITY_EXCLUDE]
    g_adm = [c for c in g_cand if ident[c]["identity_purity90_coverage"] < IDENTITY_EXCLUDE]
    e_excl = [c for c in e_cand if c not in e_adm]
    g_excl = [c for c in g_cand if c not in g_adm]

    # ---- manifest
    def role(c):
        if c in m_cols: return "M"
        if c in e_adm: return "E"
        if c in g_adm: return "G"
        if c == "y_positive": return "TARGET"
        if c in ("landscape_id", "terrain_unit_id"): return "IDENTIFIER"
        if c.startswith(("qa_",)) or c in e_excl or c in g_excl: return "QA_ONLY"
        return "NON_PREDICTOR"
    manifest = pd.DataFrame([dict(column=c, role=role(c), predictor=role(c) in ("M", "E", "G"), dtype=str(meg[c].dtype),
                                  note=(identity_flag(ident[c]) if c in ident else "")) for c in meg.columns])
    pred = [c for c in meg.columns if role(c) in ("M", "E", "G")]

    # ---- storage (partitioned by landscape; typed)
    files = {}
    for lid in B.DEVELOPMENT:
        p = MEG_DIR / f"{lid}.npz"
        write_columns(p, meg[meg.landscape_id == lid].reset_index(drop=True))
        files[str(p.relative_to(ROOT))] = sha256(p)
    _json(MEG_DIR / "SCHEMA.json", dict(format="numpy .npz (zip of .npy columns; numpy.load compatible; deterministic)", partition="landscape_id",
                                        rows=len(meg), columns={c: str(meg[c].dtype) for c in meg.columns},
                                        reader="sfv2.build_stage4a.read_columns(path) -> pandas.DataFrame",
                                        csv_export="not produced (typed archive is authoritative; CSV adds ~0.3 GB without information)"))

    # ---- dictionaries, crosswalk, temporal audit
    _tsv(OUT / "STAGE4A_FEATURE_MANIFEST.tsv", manifest)
    reg = {e["acquisition_id"]: e for e in load_registry()}
    e_dict = []
    for v, (name, unit, meaning) in E.CHELSA_VARS.items():
        e_dict.append(dict(feature_name=name, source="CHELSA V2.1 (WSL)", native_resolution="30 arcsec (~0.9 km)", unit=unit, temporal_basis="1981-2010 climatology",
                           aggregation="value of the containing CHELSA cell at the unit centroid (no interpolation, no SD)", scale_class="CONTEXTUAL",
                           scientific_interpretation=meaning, leakage_status="PASS", admitted=name in e_adm,
                           identity_purity90_coverage=ident[name]["identity_purity90_coverage"],
                           between_landscape_variance_fraction=ident[name]["between_landscape_variance_fraction"], identity_flag=identity_flag(ident[name])))
    for suf, agg in (("__mean", "mean of the 9 frozen 30 m cell-centre samples (nearest native pixel)"), ("__sd", "SD (ddof=0) of the same 9 samples")):
        n = "e_hansen_treecover2000_pct" + suf
        e_dict.append(dict(feature_name=n, source="Hansen GFC-2024-v1.12 treecover2000", native_resolution="1 arcsec (~30 m)", unit="percent canopy cover",
                           temporal_basis="year 2000", aggregation=agg, scale_class="LOCAL", scientific_interpretation="Pre-event vegetation / root reinforcement proxy",
                           leakage_status="PASS", admitted=n in e_adm, identity_purity90_coverage=ident[n]["identity_purity90_coverage"],
                           between_landscape_variance_fraction=ident[n]["between_landscape_variance_fraction"], identity_flag=identity_flag(ident[n])))
    _tsv(OUT / "STAGE4A_E_FEATURE_DICTIONARY.tsv", e_dict)
    temporal = []
    for lid, ev in B.EVENT_DATE.items():
        for v in E.CHELSA_VARS:
            contemp = ev[:4] <= "2010"
            temporal.append(dict(landscape_id=lid, source_dataset="CHELSA V2.1", variable=v, reference_period="1981-2010", event_date=ev,
                                 status="CONTEMPORANEOUS_LONG_TERM_CLIMATOLOGY" if contemp else "PRE_EVENT",
                                 justification=("30-year background climatology; the event month is 1 of 360 monthly fields, so the layer describes "
                                                "background climate, not the event's consequences" if contemp else "Reference period ends before the event"),
                                 admissible=True))
        temporal.append(dict(landscape_id=lid, source_dataset="Hansen GFC v1.12", variable="treecover2000", reference_period="2000", event_date=ev,
                             status="PRE_EVENT", justification="Year-2000 canopy precedes every development event (earliest 2005-10)", admissible=True))
        for src, var, why in (("Hansen GFC v1.12", "lossyear", "Post-2000 loss can include the event and its scars: post-event information"),
                              ("ESA CCI Land Cover", "annual LC", "Optional; landscape-specific year choice, 300 m categorical, overlaps treecover2000; not admitted to the compact primary set"),
                              ("SoilGrids 2.0", "soil properties", "Optional; 2020 machine-learning product whose covariates post-date most events and include terrain (redundant with M); not admitted"),
                              ("Event rainfall / shaking", "trigger intensity", "Not authorised as a predictor in Stage 4A")):
            temporal.append(dict(landscape_id=lid, source_dataset=src, variable=var, reference_period="various", event_date=ev,
                                 status="NOT_ADMITTED", justification=why, admissible=False))
    _tsv(OUT / "STAGE4A_TEMPORAL_AUDIT.tsv", temporal)
    _tsv(OUT / "STAGE4A_GEOLOGY_CROSSWALK.tsv", pd.concat(xwalks, ignore_index=True))
    g_support = {c: int(meg.groupby("landscape_id")[c].apply(lambda s: s.notna().any() and s.nunique() > 0).sum()) for c in g_cand}
    g_dict = [
        dict(feature_name="g_fragmental_volcanic", definition="1 if frozen class is PYROCLASTIC or VOLCANICLASTIC; 0 if MIXED_VOLCANIC or *_LAVA; NaN otherwise",
             rationale="Fragmental (tephra/volcaniclastic) vs coherent/undivided volcanic material: governed material contrast relevant to slope strength",
             landscapes_with_level_1=int((meg.groupby("landscape_id").g_fragmental_volcanic.max() == 1).sum()), landscapes_represented=g_support["g_fragmental_volcanic"],
             missingness=float(meg.g_fragmental_volcanic.isna().mean()), identity_purity90_coverage=ident["g_fragmental_volcanic"]["identity_purity90_coverage"],
             decision="ADMITTED" if "g_fragmental_volcanic" in g_adm else "EXCLUDED_IDENTITY"),
        dict(feature_name="g_age_quaternary", definition="1 if source formation age is strictly Quaternary; 0 if any pre-Quaternary component (e.g. Pliocene-Quaternary); NaN if unresolved",
             rationale="Broad age: younger, less consolidated volcanic deposits vs older weathered/compacted units",
             landscapes_with_level_1=int((meg.groupby("landscape_id").g_age_quaternary.max() == 1).sum()), landscapes_represented=g_support["g_age_quaternary"],
             missingness=float(meg.g_age_quaternary.isna().mean()), identity_purity90_coverage=ident["g_age_quaternary"]["identity_purity90_coverage"],
             decision="ADMITTED" if "g_age_quaternary" in g_adm else "EXCLUDED_IDENTITY"),
        dict(feature_name="(rejected) harmonised class one-hot", definition="Frozen class levels", rationale="Ontology class",
             landscapes_with_level_1="INTERMEDIATE_LAVA 1; VOLCANICLASTIC 2; PYROCLASTIC 4 (Noto 56 units)", landscapes_represented=6, missingness=0.0,
             identity_purity90_coverage="INTERMEDIATE_LAVA level = Dominica only", decision="REJECTED_SUPPORT_RULE (kept as qa_harmonised_geology_class)"),
        dict(feature_name="(rejected) source unit code / name", definition="Local map-sheet code", rationale="No meaning outside its source map",
             landscapes_with_level_1="each code in 1 landscape", landscapes_represented=6, missingness=0.0, identity_purity90_coverage=1.0,
             decision="REJECTED_LANDSCAPE_IDENTITY (qa_source_unit_code)"),
        dict(feature_name="(QA only) geology boundary / dominant fraction", definition="Distinct units among the 9 cells", rationale="Local heterogeneity",
             landscapes_with_level_1=6, landscapes_represented=6, missingness=0.0,
             identity_purity90_coverage="map-scale artefact: boundary frequency 0.9% (Lombok) to 17.9% (Dominica digitised)", decision="QA_ONLY_SOURCE_SCALE_CONFOUND"),
        dict(feature_name="(QA only) mapping confidence / source quality", definition="Mapping-process descriptors", rationale="Encodes the mapping process",
             landscapes_with_level_1=6, landscapes_represented=6, missingness=0.0, identity_purity90_coverage="source identifier", decision="QA_ONLY"),
    ]
    _tsv(OUT / "STAGE4A_G_FEATURE_DICTIONARY.tsv", g_dict)
    g_sources = [dict(landscape_id=lid, publisher=s[0], product=s[1], scale=s[2], mapping_resolution=s[3], crs=s[4],
                      geology_source_quality=G.quality_class(s[2], s[3]), **gaudit[lid]) for lid, s in G.SOURCE.items()]
    _tsv(OUT / "STAGE4A_GEOLOGY_SOURCE_AUDIT.tsv", g_sources)

    # ---- missingness
    miss = []
    for c in pred + e_excl + g_excl:
        row = dict(predictor=c, role=role(c), global_missing=round(float(meg[c].isna().mean()), 6),
                   missing_y1=round(float(meg.loc[meg.y_positive, c].isna().mean()), 6), missing_y0=round(float(meg.loc[~meg.y_positive, c].isna().mean()), 6),
                   nonfinite=int((~np.isfinite(meg[c].to_numpy(dtype=float)) & meg[c].notna().to_numpy()).sum()))
        for lid in B.DEVELOPMENT:
            row[f"missing_{lid}"] = round(float(meg.loc[meg.landscape_id == lid, c].isna().mean()), 6)
        miss.append(row)
    miss = pd.DataFrame(miss)
    _tsv(OUT / "STAGE4A_MISSINGNESS_AUDIT.tsv", miss)

    # ---- domain shift (descriptive; no Y)
    shift_rows, dist = [], {}
    cont = [c for c in pred if not c.startswith("g_")]
    for c in cont + g_adm:
        for lid in B.DEVELOPMENT:
            s = meg.loc[meg.landscape_id == lid, c]
            if c.startswith("g_"):
                shift_rows.append(dict(predictor=c, landscape_id=lid, kind="categorical", proportion_1=round(float(s.mean()), 4), median=None, iqr=None, p05=None, p95=None))
            else:
                q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                shift_rows.append(dict(predictor=c, landscape_id=lid, kind="continuous", proportion_1=None, median=q[0.5], iqr=q[0.75] - q[0.25], p05=q[0.05], p95=q[0.95]))
    pooled_iqr = {c: float(meg[c].quantile(0.75) - meg[c].quantile(0.25)) or 1.0 for c in cont}
    for lid in B.DEVELOPMENT:
        a, b = meg[meg.landscape_id == lid], meg[meg.landscape_id != lid]
        per = {c: abs(float(a[c].median()) - float(b[c].median())) / pooled_iqr[c] for c in cont}
        per.update({c: abs(float(a[c].mean()) - float(b[c].mean())) for c in g_adm})
        top = sorted(per.items(), key=lambda kv: -kv[1])[:6]
        dist[lid] = dict(mean_standardised_median_shift_all=round(float(np.mean(list(per.values()))), 4),
                         mean_shift_M=round(float(np.mean([per[c] for c in m_cols])), 4),
                         mean_shift_E=round(float(np.mean([per[c] for c in e_adm])), 4) if e_adm else None,
                         top_features="|".join(f"{k}:{round(v, 3)}" for k, v in top))
    shift = pd.DataFrame(shift_rows)
    shift = pd.concat([shift, pd.DataFrame([dict(predictor="__LANDSCAPE_DISTANCE__", landscape_id=k, kind="summary", **v) for k, v in dist.items()])], ignore_index=True)
    ident_rows = pd.DataFrame([dict(predictor="__IDENTITY__", landscape_id="ALL", kind=role(c), feature=c, **v) for c, v in ident.items()])
    _tsv(OUT / "STAGE4A_DOMAIN_SHIFT_AUDIT.tsv", pd.concat([shift, ident_rows], ignore_index=True))
    strongest = max(dist, key=lambda k: dist[k]["mean_standardised_median_shift_all"])

    # ---- leakage audit
    leak = []
    for c in pred + e_excl + g_excl + ["qa_dem_edited_fraction", "qa_source_unit_code", "qa_geology_boundary_unit", "spatial_block_3000m", "centroid_x"]:
        r = role(c) if c in meg.columns else "QA_ONLY"
        if c in m_cols:
            a = dict(post_event="NO (pre-event DEM: COP30 2010-2015 scenes; Atitlan NASADEM 2000)", from_y="NO", mapping_process="NO", omega="NO (computed on full DEM)",
                     identity_artefact="FLAG: Atitlan M from NASADEM (different DEM source)", dem_quality="FLAG: COP30 edits; STRICT sensitivity available (5/6)",
                     status="PASS_FROZEN_M_WITH_4B_SENSITIVITIES")
        elif c.startswith("e_"):
            a = dict(post_event="NO" if "chelsa" in c else "NO (year 2000)", from_y="NO", mapping_process="NO", omega="NO",
                     identity_artefact=("NO artefact (uniform global product); environmental identity measured: purity90 coverage "
                                        f"{ident[c]['identity_purity90_coverage']}"), dem_quality="NO",
                     status="PASS" if c in e_adm else "EXCLUDED_IDENTITY_RULE")
        elif c.startswith("g_"):
            a = dict(post_event="NO (pre-existing official geology)", from_y="NO", mapping_process="LOW (binary aggregate of frozen classes)",
                     omega="NO (Omega restricts rows, not values)", identity_artefact=f"purity90 coverage {ident[c]['identity_purity90_coverage']}",
                     dem_quality="NO", status="PASS" if c in g_adm else "EXCLUDED_IDENTITY_RULE")
        else:
            a = dict(post_event="NO", from_y="NO", mapping_process="YES" if "geology" in c or "unit" in c else "NO", omega="NO",
                     identity_artefact="YES", dem_quality="YES" if "dem" in c else "NO", status="EXCLUDED_NON_PREDICTOR")
        leak.append(dict(predictor=c, role=r, **a))
    leak = pd.DataFrame(leak)
    _tsv(OUT / "STAGE4A_LEAKAGE_AUDIT.tsv", leak)
    assert leak[leak.role.isin(["M", "E", "G"])].status.str.startswith("PASS").all()

    # ---- firewall, 4B spec, H contract
    firewall = dict(development_landscapes=list(B.DEVELOPMENT), protected_future_lockboxes=B.PROTECTED_LOCKBOX_CANDIDATES,
                    rules=["Only development landscapes are loaded (stage4a_base.firewall_check raises otherwise)",
                           "No E acquisition, feature engineering decision, distribution or target inspection for protected landscapes",
                           "Feature-admission rules fixed before computing statistics; computed on development rows only",
                           "Protected landscapes may not appear in any MEG partition, manifest statistic or Stage 4B fold"],
                    meg_landscapes=sorted(meg.landscape_id.unique()), protected_rows_in_meg=0,
                    e_acquisitions_landscapes=sorted({r["landscape_id"] for r in e_acq}))
    assert set(firewall["meg_landscapes"]) == set(B.DEVELOPMENT) and not set(firewall["e_acquisitions_landscapes"]) - set(B.DEVELOPMENT)
    _json(OUT / "STAGE4A_LOCKBOX_FIREWALL.json", firewall)
    (OUT / "STAGE4B_VALIDATION_SPEC.md").write_text(spec_4b(pred, m_cols, e_adm, g_adm))
    _json(OUT / "STAGE4A_H_CONTRACT.json", dict(
        feature_sets={"M": m_cols, "ME": m_cols + e_adm, "MEG": m_cols + e_adm + g_adm, "MEGH": "MEG + h_* columns (reserved)", "MEG+selected_H": "MEG + pre-registered H subset"},
        reserved_prefix="h_", availability_flag="h_available (bool per row)", missing_h_rule="NEVER zero-fill; rows without H are excluded from MEGH comparisons, not imputed",
        comparison_rule="MEG vs MEGH evaluated on identical rows and identical LOLO folds restricted to rows where H exists",
        v1_lesson="V1: H improved internal validation but harmed Kurikoma external transfer; H is an empirical hypothesis, not an assumed gain"))

    # ---- report + checkpoint
    v1_after = v1_verify()
    ml_h = scan_ml_h_artifacts()
    summary = dict(base=base, e_admitted=e_adm, e_excluded=e_excl, g_admitted=g_adm, g_excluded=g_excl, n_m=len(m_cols), n_e=len(e_adm), n_g=len(g_adm),
                   n_pred=len(pred), global_missing=round(float(meg[pred].isna().to_numpy().mean()), 8), strongest_shift=strongest, distances=dist,
                   identity=ident, g_audit=gaudit, prior_ok=prior_ok, prior=prior, files=files, ml_h=ml_h, e_acq=len(e_acq))
    (OUT / "STAGE4A_REPORT.md").write_text(report(summary))
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith(("STAGE4A_CHECKPOINT", "STAGE4A_DIAGNOSTICS")))
    cp = dict(schema="SFV2_STAGE4A_CHECKPOINT_V1", as_of=AS_OF, development_landscapes=list(B.DEVELOPMENT), totals=base["totals"],
              tier2_only=base["tier2_only"], predictors=dict(M=len(m_cols), E=len(e_adm), G=len(g_adm), total=len(pred)),
              e_admitted=e_adm, e_excluded=e_excl, g_admitted=g_adm, policy=POLICY, prior_checkpoints=prior, prior_checkpoints_unchanged=prior_ok,
              stage3j_checkpoint_sha256=prior["outputs/stage3j/STAGE3J_CHECKPOINT.json"], e_raw_windows={r["acquisition_id"]: r["sha256"] for r in e_acq},
              meg_partitions=files, output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs},
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage4a.py", "stage4a_base.py", "stage4a_e.py", "stage4a_g.py")},
              ml_trained=False, lolo_executed=False, h_generated=False, lockbox_accessed=False, ml_or_h_artifacts_found=ml_h,
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              volatile_diagnostics={"path": "outputs/stage4a/STAGE4A_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE4A_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    st = git("status", "--porcelain").splitlines()
    _json(OUT / "STAGE4A_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD"), modified_tracked=sum(not s.startswith("??") for s in st)))
    return dict(cp=cp, summary=summary, manifest=manifest, miss=miss, leak=leak)


def spec_4b(pred, m_cols, e_adm, g_adm) -> str:
    return f"""# Stage 4B development validation specification (DESIGN ONLY — not executed in Stage 4A)

## Data
- Development landscapes (6): {', '.join(B.DEVELOPMENT)}. Protected future lockboxes are never loaded (STAGE4A_LOCKBOX_FIREWALL.json).
- Rows: the STAGE4A_MEG_DATASET partitions. Target `y_positive` is Tier 2 only. Only columns whose manifest role is M/E/G are predictors.
- Feature-set arms on identical rows/folds: **M** ({len(m_cols)}), **ME** ({len(m_cols) + len(e_adm)}), **MEG** ({len(pred)}). MEGH is reserved (STAGE4A_H_CONTRACT.json).

## Primary outer validation: Leave-One-Landscape-Out
- 6 outer folds; each holds out one entire landscape; training = the other 5. No random row split is ever a primary result.
- Hyperparameters: chosen by inner leave-one-landscape-out across the 5 training landscapes only (5 inner folds).
- Feature scaling/encoding: fitted on training landscapes only, inside each fold.
- Calibration (if any): fitted on inner out-of-fold predictions of training landscapes only.
- Thresholds (for precision/recall/F1): chosen on training landscapes only (pre-registered rule: maximise inner-fold F1), then applied unchanged to the held-out landscape.

## Metrics
- Primary: PR-AUC (average precision) on each held-out landscape.
- Secondary: ROC-AUC, Brier score, log loss, precision, recall, F1 at the training-chosen threshold, calibration intercept and slope (logistic recalibration of held-out predictions; reported, never refitted for scoring).
- Always report the per-landscape value with each landscape's prevalence (PR-AUC baseline), then the unweighted mean across held-out landscapes. Pooled metrics are secondary.

## Weighting options (to be fixed before any outer result is viewed)
1. Unweighted rows (Kumamoto = 42% of rows dominates).
2. Landscape-balanced: each training landscape contributes equal total weight (design-proposed primary, because row counts differ 13x; confirm before execution).
3. Landscape-balanced + class-balanced within landscape (sensitivity).
The choice must be recorded before outer-fold results exist and must never be made from held-out performance.

## Required sensitivities (pre-registered)
- Population: ALL_VALID (primary) vs STRICT_DEM_QUALITY on the 5 landscapes that have a quality companion; Atitlan has none (NOT_AVAILABLE) and is reported separately.
- Target family: pooled Tier 2 (primary) vs polygon-family only (excluding Kumamoto POINT_EVENT_TIER2), because point and polygon labels differ in semantics.
- DEM source: excluding Atitlan (NASADEM M) vs all.
- Identity-flagged E predictors: ME vs M quantifies whether coarse climate adds transferable information or only landscape-level context.

## Benchmark models (compact; no model zoo)
1. Null: training-landscape prevalence (landscape-balanced).
2. L2-regularised logistic regression (standardised predictors; missing indicators only if a governed missingness treatment is adopted).
3. Random Forest (balanced class weights optional as a sensitivity).
4. Gradient-boosted trees (scikit-learn HistGradientBoostingClassifier, which handles NaN natively).
Purpose: test whether nonlinear models materially improve unseen-landscape transfer.

## Environment prerequisite
scikit-learn is NOT installed in the project venv. Stage 4B needs the user's permission to install it (and only it).

## Not allowed in 4B
Final model freezing, final hyperparameters, lockbox access, using held-out landscapes for any choice, zero-filling H.
"""


def report(s: dict) -> str:
    b = s["base"]
    rows = "\n".join(f"| {k} | {v['rows']:,} | {v['positives']:,} | {v['negatives']:,} | {v['target_family']} | {v['strict_units'] if v['strict_available'] else 'NOT_AVAILABLE'} | {v['dem_source']} |"
                     for k, v in b["per_landscape"].items())
    ident = "\n".join(f"| {k} | {v['identity_purity90_coverage']} | {v['between_landscape_variance_fraction']} | {identity_flag(v)} |"
                      for k, v in s["identity"].items() if k.startswith(("e_", "g_")))
    dist = "\n".join(f"| {k} | {v['mean_standardised_median_shift_all']} | {v['mean_shift_M']} | {v['mean_shift_E']} | {v['top_features']} |" for k, v in s["distances"].items())
    return f"""# Stage 4A report — six-landscape MEG development dataset

| Landscape | Units | Positives | Negatives | Target family | STRICT units | M DEM source |
|---|---|---|---|---|---|---|
{rows}
| **Total** | **{b['totals']['rows']:,}** | **{b['totals']['positives']:,}** | **{b['totals']['negatives']:,}** | Tier 2 only: {b['tier2_only']} | | |

Predictors: M {s['n_m']} (frozen), E {s['n_e']} ({', '.join(s['e_admitted']) or 'none'}), G {s['n_g']} ({', '.join(s['g_admitted'])}); total {s['n_pred']}.
Excluded by the a-priori identity rule: E {s['e_excluded'] or 'none'}; G {s['g_excluded'] or 'none'}.
Global predictor missingness: {s['global_missing']}. Strongest domain shift: {s['strongest_shift']}.

## Identity statistics (E/G candidates)
| Feature | purity90 coverage | between-landscape variance fraction | flag |
|---|---|---|---|
{ident}

## Domain shift (descriptive; |median_L - median_rest| / pooled IQR; binary G: |mean difference|)
| Landscape | all | M | E | top features |
|---|---|---|---|---|
{dist}

## Policy (fixed before statistics)
{json.dumps(POLICY, indent=1)}

Outputs: STAGE4A_MEG_DATASET/ (typed .npz per landscape + SCHEMA.json), FEATURE_MANIFEST, E/G dictionaries, GEOLOGY_CROSSWALK, GEOLOGY_SOURCE_AUDIT,
TEMPORAL_AUDIT, MISSINGNESS_AUDIT, DOMAIN_SHIFT_AUDIT, LEAKAGE_AUDIT, LOCKBOX_FIREWALL, H_CONTRACT, STAGE4B_VALIDATION_SPEC.md, CHECKPOINT.
No model trained, no LOLO executed, no H generated, no lockbox accessed.
"""


if __name__ == "__main__":
    main()
