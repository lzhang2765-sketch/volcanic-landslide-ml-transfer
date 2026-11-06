"""Stage 4A geology (G) predictors from the FROZEN V2 ontology (Stages 2C-3F); no new classification is invented.

Unit assignment = the frozen production rule already stored in terrain_units (dominant geology code among the unit's 30 m cells
that are inside OMEGA_ANALYSIS; ties resolved by the frozen np.unique/argmax rule).  This module re-derives that rule from the
30 m rasters only to AUDIT it (exact agreement required) and to record boundary/ambiguity QA.  Missing geology is never inferred.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import rasterio

from .schema import ROOT
from .stage2c_geology import VOLCANIC

PROC = ROOT / "data/processed"
OUT_PRE = ROOT / "outputs"
RASTERS = {  # (30 m geology index raster, OMEGA_ANALYSIS raster) used by the frozen production run
    "KUMAMOTO_ASO_2016": ("stage2c/KUMAMOTO_ASO_2016/geology_seamless_index_30m.tif", "stage2c/KUMAMOTO_ASO_2016/omega_analysis_primary_april_eq_30m.tif"),
    "NOTO_PENINSULA_2024": ("stage2c/NOTO_PENINSULA_2024/geology_seamless_index_30m.tif", "stage2c/NOTO_PENINSULA_2024/omega_analysis_primary_30m.tif"),
    "LOMBOK_RINJANI_2018": ("stage2f2/LOMBOK_RINJANI_2018/geology_geomap_index_30m.tif", "stage2f2/LOMBOK_RINJANI_2018/omega_analysis_primary_30m.tif"),
    "COTABATO_DAVAO_2019": ("stage3d/COTABATO_DAVAO_2019/geology_mgb_index_30m.tif", "stage3d/COTABATO_DAVAO_2019/omega_analysis_primary_30m.tif"),
    "ATITLAN_STAN_2005": ("stage3f/ATITLAN_STAN_2005/geology_official_index_30m.tif", "stage3f/ATITLAN_STAN_2005/omega_analysis_primary_30m.tif"),
    "DOMINICA_ERIKA_2015": ("stage3f/DOMINICA_ERIKA_2015/geology_official_index_30m.tif", "stage3f/DOMINICA_ERIKA_2015/omega_analysis_primary_30m.tif"),
}
MAPPING = {
    "KUMAMOTO_ASO_2016": "stage2c/geology_mapping_KUMAMOTO_ASO_2016_seamless.csv", "NOTO_PENINSULA_2024": "stage2c/geology_mapping_NOTO_PENINSULA_2024_seamless.csv",
    "LOMBOK_RINJANI_2018": "stage2f2/geology_mapping_LOMBOK_geomap_primary.csv", "COTABATO_DAVAO_2019": "stage3d/COTABATO_GEOLOGY_MAPPING.csv",
    "ATITLAN_STAN_2005": "stage3f/ATITLAN_GEOLOGY_HARMONISATION.csv", "DOMINICA_ERIKA_2015": "stage3f/DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv",
}
SOURCE = {  # geology source audit (from Stage 2C-3F provenance); QA only, never a predictor
    "KUMAMOTO_ASO_2016": ("GSJ / AIST", "Seamless Digital Geological Map of Japan V2 (Aso 1:50,000 used only as sensitivity)", "1:200,000", "vector, official", "EPSG:4326 source -> UTM 52N"),
    "NOTO_PENINSULA_2024": ("GSJ / AIST", "Seamless Digital Geological Map of Japan V2", "1:200,000", "vector, official", "EPSG:4326 source -> UTM 53N"),
    "LOMBOK_RINJANI_2018": ("Badan Geologi (PSG)", "GeoMap district SHP (Lombok Utara/Timur/Barat/Tengah) + sheet 1807 legend", "1:100,000 (legend 1:250,000)", "vector, official", "EPSG:4326 -> UTM 50S"),
    "COTABATO_DAVAO_2019": ("Mines and Geosciences Bureau (Philippines)", "GDI Regional Geology public ArcGIS service", "UNSTATED (regional compilation)", "vector, official service", "EPSG:3857 -> UTM 51N"),
    "ATITLAN_STAN_2005": ("MAGA Guatemala", "SUELOS geologia_nacional", "1:500,000", "vector, official", "-> UTM 15N"),
    "DOMINICA_ERIKA_2015": ("Government of Dominica (Physical Planning)", "Geologic map of Dominica (PDF), legend-colour digitisation, RMS 23.7 m", "1:100,000", "raster digitised from official map", "-> UTM 20N"),
}
FRAGMENTAL = {"PYROCLASTIC", "VOLCANICLASTIC"}
COHERENT_OR_MIXED = {"MIXED_VOLCANIC", "BASALTIC_LAVA", "INTERMEDIATE_LAVA", "FELSIC_LAVA"}
QUAT = r"pleistocene|holocene|quaternary|kuarter|recent|gelasian|calabrian"
PREQ = r"pliocene|miocene|oligocene|eocene|paleocene|neogene|neogen\b|paleogene|tertiary|messinian|tortonian|langhian|burdigalian|aquitanian|chattian"


def quality_class(scale: str, kind: str) -> str:
    m = re.search(r"1:([\d,]+)", scale)
    n = int(m.group(1).replace(",", "")) if m else None
    q = "FINE" if n and n <= 100000 else "MEDIUM" if n and n <= 250000 else "COARSE_OR_UNSTATED"
    return q + ("_DIGITISED" if "digitised" in kind else "")


def age_quaternary(age: str) -> float:
    a = str(age).lower()
    q, p = bool(re.search(QUAT, a)), bool(re.search(PREQ, a))
    return 1.0 if q and not p else 0.0 if p else np.nan  # spans Pliocene-Quaternary -> 0 (not strictly Quaternary); unresolved -> NaN


def fragmental(cls: str) -> float:
    return 1.0 if cls in FRAGMENTAL else 0.0 if cls in COHERENT_OR_MIXED else np.nan


def unit_audit(lid: str, terrain: pd.DataFrame) -> pd.DataFrame:
    gp, op = RASTERS[lid]
    with rasterio.open(PROC / gp) as g, rasterio.open(PROC / op) as o:
        geo, om = g.read(1), o.read(1).astype(bool)
    r = terrain.source_grid_row.to_numpy()
    c = terrain.source_grid_col.to_numpy()
    blocks_g = np.stack([geo[r + i, c + j] for i in range(3) for j in range(3)], axis=1).astype(np.int64)
    blocks_o = np.stack([om[r + i, c + j] for i in range(3) for j in range(3)], axis=1)
    valid = blocks_o & (blocks_g > 0)
    n_units, dom_frac, tie, mode = [], [], [], []
    for gv, vv in zip(blocks_g, valid):
        vals, n = np.unique(gv[vv], return_counts=True)
        if len(vals) == 0:
            n_units.append(0); dom_frac.append(np.nan); tie.append(False); mode.append(0); continue
        k = int(np.argmax(n))
        mode.append(int(vals[k])); n_units.append(len(vals)); dom_frac.append(n[k] / n.sum()); tie.append(int((n == n[k]).sum()) > 1)
    return pd.DataFrame({"terrain_unit_id": terrain.terrain_unit_id.to_numpy(), "qa_geology_units_in_9cells": n_units,
                         "qa_geology_dominant_fraction": dom_frac, "qa_geology_boundary_unit": np.asarray(n_units) > 1,
                         "qa_geology_tie_unit": tie, "qa_geology_recomputed_index": mode})


def features(terrain: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"terrain_unit_id": terrain.terrain_unit_id.to_numpy(),
                         "g_fragmental_volcanic": terrain.harmonised_geology_class.map(fragmental).astype("float64"),
                         "g_age_quaternary": terrain.source_age.map(age_quaternary).astype("float64")})


def crosswalk(lid: str, terrain: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(OUT_PRE / MAPPING[lid])
    reason_col = next(c for c in ("mapping_notes", "rationale", "mapping_rule") if c in m.columns)
    m = m.rename(columns={reason_col: "mapping_reason"})
    used = terrain.groupby(["geology_index", "source_unit_code", "source_lithology", "source_age", "harmonised_geology_class",
                            "geology_mapping_confidence"]).size().rename("terrain_units").reset_index()
    rows = []
    for u in used.itertuples():
        mr = m.iloc[int(u.geology_index) - 1] if 0 < u.geology_index <= len(m) else None
        consistent = mr is not None and mr["source_unit_code"] == u.source_unit_code and mr["harmonised_class"] == u.harmonised_geology_class
        rows.append(dict(landscape_id=lid, source_unit_code=u.source_unit_code,
                         source_unit_name=(mr.get("source_unit_name") if mr is not None else "") or "", source_age=u.source_age,
                         source_lithology=u.source_lithology, v2_geology_class=u.harmonised_geology_class, confidence=u.geology_mapping_confidence,
                         mapping_reason=(f"{mr.get('mapping_rule', '')}: {mr['mapping_reason']}" if mr is not None else "INDEX_NOT_IN_MAPPING"),
                         is_volcanic=u.harmonised_geology_class in VOLCANIC, g_fragmental_volcanic=fragmental(u.harmonised_geology_class),
                         g_age_quaternary=age_quaternary(u.source_age), terrain_units=int(u.terrain_units), mapping_table_consistent=bool(consistent),
                         mapping_table=f"outputs/{MAPPING[lid]}"))
    return pd.DataFrame(rows)
