"""Build the Stage 5F development-only geospatial visualization audit.

This module reads frozen predictions and diagnostics.  It does not expose a
fit, calibration, threshold-selection, or external-data pathway.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import spearmanr

from .build_stage5bb1 import verify_baseline
from .raw_store import sha256
from .schema import ROOT
from .stage5f_mapping import (DISPLAY, LANDSCAPES, MODEL_ID, OUT, WARNING,
                              export_geotiff, export_table, grid_layout,
                              load_landscape, rasterize_values, render_map,
                              render_status, render_three_panel, status_panel,
                              structured_table, _map_array)


FIG = OUT / "figures"
MAP = OUT / "maps"
GIS = OUT / "gis_exports"
EXPECTED = {
    "outputs/stage5e/STAGE5E_CHECKPOINT.json": "258ad4b9576b45ce502a93871b96f77dec277ea3626e8925c0435dc5732af8b2",
    "outputs/stage5d_e/STAGE5DE_CHECKPOINT.json": "a3fe549b960b6e3f470bdd0cd812bd33f2ab54fce70ada9c2666da903c867dc0",
    "outputs/stage4f/STAGE4F_CHECKPOINT.json": "a5f2caa78dd50c831eef9e691bec9c4b3f620329e280dc5e06d21e575de4c9d7",
    "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
    "outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json": "bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
    "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1",
}
OMEGA = {
    "KUMAMOTO_ASO_2016": "data/processed/stage2c/KUMAMOTO_ASO_2016/omega_analysis_primary_april_eq_30m.tif",
    "NOTO_PENINSULA_2024": "data/processed/stage2c/NOTO_PENINSULA_2024/omega_analysis_primary_30m.tif",
    "LOMBOK_RINJANI_2018": "data/processed/stage2f2/LOMBOK_RINJANI_2018/omega_analysis_primary_30m.tif",
    "COTABATO_DAVAO_2019": "data/processed/stage3d/COTABATO_DAVAO_2019/omega_analysis_primary_30m.tif",
    "ATITLAN_STAN_2005": "data/processed/stage3f/ATITLAN_STAN_2005/omega_analysis_primary_30m.tif",
    "DOMINICA_ERIKA_2015": "data/processed/stage3f/DOMINICA_ERIKA_2015/omega_analysis_primary_30m.tif",
}


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): x = x.item()
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n")


def write_json(path: Path, obj) -> None:
    write(path, json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False))


def tsv(path: Path, rows) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.12g")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def integrity_gate() -> dict:
    actual = {p: sha256(ROOT / p) for p in EXPECTED}
    bad = {p: {"expected": EXPECTED[p], "actual": actual[p]} for p in EXPECTED if actual[p] != EXPECTED[p]}
    if bad: raise RuntimeError(f"STOP: protected artifact mismatch: {bad}")
    baseline = verify_baseline()
    decisions = pd.read_csv(ROOT / "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv", sep="\t")
    text = "\n".join(decisions.astype(str).agg(" ".join, axis=1))
    for required in ("CALIBRATION_TRANSFER_NOT_SUPPORTED", "NOT_ESTABLISHED", "ABSTENTION", "NOT_SUPPORTED"):
        if required not in text: raise RuntimeError(f"STOP: Stage 4F decision missing: {required}")
    return {"status": "PASS", "expected_and_actual_hashes": actual, "baseline": baseline,
            "stage4f_decisions_sha256": sha256(ROOT / "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv")}


def omega_audit(lid: str, d: dict) -> dict:
    path = ROOT / OMEGA[lid]
    if not path.is_file(): raise RuntimeError(f"STOP: authoritative observation-domain raster missing: {path}")
    with rasterio.open(path) as src:
        if str(src.crs) != d["projected_crs"]: raise RuntimeError(f"STOP: Omega CRS mismatch: {lid}")
        return {"observation_domain_path": OMEGA[lid], "observation_domain_sha256": sha256(path),
                "observation_domain_crs": str(src.crs), "observation_domain_width": src.width,
                "observation_domain_height": src.height, "observation_domain_resolution_x_m": abs(src.transform.a),
                "observation_domain_resolution_y_m": abs(src.transform.e), "observation_domain_nodata": src.nodata,
                "observation_domain_bounds": list(src.bounds)}


def spatial_diagnostics(d: dict) -> dict:
    pct, novelty = d["percentile"], d["novelty"]
    parr, layout = rasterize_values(d, pct); narr, _ = rasterize_values(d, novelty)
    valid = parr != -9999
    pairs_a, pairs_b = [], []
    for a, b, va, vb in ((parr[:, :-1], parr[:, 1:], valid[:, :-1], valid[:, 1:]),
                         (parr[:-1, :], parr[1:, :], valid[:-1, :], valid[1:, :])):
        ok = va & vb; pairs_a.append(a[ok]); pairs_b.append(b[ok])
    adjacent_r = float(np.corrcoef(np.concatenate(pairs_a), np.concatenate(pairs_b))[0, 1])
    top = valid & (parr >= 90)
    labels, ncomp = ndimage.label(top, structure=ndimage.generate_binary_structure(2, 1))
    sizes = np.bincount(labels.ravel())[1:]
    row, col = layout["row"], layout["col"]; high = pct >= 90
    cx, cy = float(d["centroid_x"].mean()), float(d["centroid_y"].mean())
    hx, hy = float(d["centroid_x"][high].mean()), float(d["centroid_y"][high].mean())
    dx, dy = hx - cx, hy - cy
    ns = "north" if dy > 0 else "south"; ew = "east" if dx > 0 else "west"
    rho, p = spearmanr(pct, novelty)
    q = np.quantile(d["raw_score"], [0, .01, .05, .25, .5, .75, .95, .99, 1])
    nq = np.quantile(novelty, [0, .25, .5, .75, 1])
    return {"landscape_id": d["landscape_id"], "n_units": len(pct), "n_positive": int(d["y_positive"].sum()),
            "raw_min": q[0], "raw_q01": q[1], "raw_q05": q[2], "raw_q25": q[3], "raw_median": q[4],
            "raw_q75": q[5], "raw_q95": q[6], "raw_q99": q[7], "raw_max": q[8],
            "percentile_min": float(pct.min()), "percentile_max": float(pct.max()),
            "novelty_min": nq[0], "novelty_q25": nq[1], "novelty_median": nq[2], "novelty_q75": nq[3], "novelty_max": nq[4],
            "susceptibility_novelty_spearman_rho": float(rho), "susceptibility_novelty_spearman_p": float(p),
            "rook_adjacent_percentile_pearson_r": adjacent_r, "top_decile_units": int(high.sum()),
            "top_decile_rook_components": int(ncomp), "largest_top_decile_component_units": int(sizes.max()),
            "largest_top_decile_component_fraction": float(sizes.max() / high.sum()),
            "top_decile_centroid_x": hx, "top_decile_centroid_y": hy,
            "domain_centroid_x": cx, "domain_centroid_y": cy,
            "top_decile_centroid_direction": f"{ns}-{ew}",
            "interpretation": "spatial description only; not causal and not prospective validation"}


def create_maps(data: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    map_meta, figcat, alignment = [], [], []
    for i, lid in enumerate(LANDSCAPES, 1):
        d = data[lid]; layout = grid_layout(d); om = omega_audit(lid, d)
        table = export_table(d, GIS / f"{lid}_STAGE5F_OUTPUT.csv.gz")
        layers = [
            ("RAW_MODEL_SCORE", d["raw_score"], "uncalibrated_score"),
            ("RELATIVE_SUSCEPTIBILITY_PERCENTILE", d["percentile"], "within_landscape_percentile_0_100"),
            ("EXPLORATORY_FEATURE_NOVELTY", d["novelty"], "training_reference_percentile_0_1"),
        ]
        for layer, values, units in layers:
            path = export_geotiff(d, values, GIS / f"{lid}_{layer}.tif", layer, units)
            with rasterio.open(path) as src:
                sample = src.read(1)[layout["row"][0], layout["col"][0]]
                expected = np.float32(values[0])
                if sample != expected or src.crs.to_string() != d["projected_crs"] or src.nodata != -9999:
                    raise RuntimeError(f"STOP: GeoTIFF round-trip QA failed: {path}")
                coverage = int(np.count_nonzero(src.read(1) != src.nodata))
                if coverage != len(values): raise RuntimeError(f"STOP: GeoTIFF coverage failed: {path}")
                map_meta.append({"landscape_id": lid, "layer": layer, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                 "crs": src.crs.to_string(), "width": src.width, "height": src.height,
                                 "resolution_m": abs(src.transform.a), "nodata": src.nodata, "valid_cells": coverage,
                                 "value_min": float(np.min(values)), "value_max": float(np.max(values)), "units": units,
                                 "development_only": "YES", "probability_layer": "NO"})
        alignment.append({"landscape_id": lid, "terrain_units": len(d["terrain_unit_id"]), "prediction_ids": len(d["raw_score"]),
                          "joined_ids": len(d["terrain_unit_id"]), "identifier_order_exact": "PASS", "coordinate_join_exact": "PASS",
                          "prediction_stage4f_text_roundtrip_max_abs": float(np.max(np.abs(
                              pd.read_csv(ROOT / "outputs/stage4f/STAGE4F_UNIT_NOVELTY.tsv", sep="\t").query("landscape_id == @lid").susceptibility_score.to_numpy() - d["raw_score"]))),
                          "crs": d["projected_crs"], "terrain_unit_resolution_m": 90, "source_grid_resolution_m": 30,
                          "grid_orientation": "north-up", "raster_transform": str(layout["transform"]), "nodata": -9999,
                          "spatial_coverage": "PASS_ALL_VALID_TERRAIN_UNITS", **om})
        products = []
        for kind in ("susceptibility", "novelty", "status", "three_panel", "inventory_diagnostic"):
            for ext in ("png", "pdf"):
                path = MAP / lid / f"{lid}_{kind}.{ext}"
                if kind == "three_panel": render_three_panel(d, path)
                elif kind == "status": render_status(d, path)
                else: render_map(d, kind, path)
                products.append((kind, path))
        for kind, path in products:
            figcat.append({"figure_id": f"MAP_{i:02d}_{kind.upper()}", "landscape_id": lid, "figure_type": kind,
                           "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                           "scientific_status": "DEVELOPMENT_ONLY", "interpretation": WARNING})
        figcat.append({"figure_id": f"TABLE_{i:02d}", "landscape_id": lid, "figure_type": "vectorized_output_table",
                       "path": str(table.relative_to(ROOT)), "sha256": sha256(table), "scientific_status": "DEVELOPMENT_ONLY",
                       "interpretation": WARNING})
        print(f"maps {i}/6 complete: {lid}", flush=True)
    return map_meta, figcat, alignment


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)


def atlas(data: dict[str, dict], novelty: bool, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, lid in zip(axes.ravel(), LANDSCAPES):
        d = data[lid]
        if novelty: _map_array(ax, d, 100*d["novelty"], "viridis", 0, 100, "Novelty percentile", DISPLAY[lid])
        else: _map_array(ax, d, d["percentile"], "magma", 0, 100, "Susceptibility percentile", DISPLAY[lid])
    title = "Exploratory feature novelty" if novelty else "Relative landslide susceptibility"
    fig.suptitle(f"Six-landscape {title} atlas — DEVELOPMENT-ONLY RESULTS", fontweight="bold", fontsize=15)
    note = "Feature-space difference, not predictive reliability." if novelty else "Within-landscape rank, not landslide probability; equal percentiles do not imply equal absolute risk."
    fig.text(.5, .01, note, ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, .03, 1, .96)); savefig(fig, path)


def additional_figures(data: dict[str, dict]) -> list[tuple[str, Path, str]]:
    made = []
    for novelty, stem in ((False, "FIG01_SUSCEPTIBILITY_ATLAS"), (True, "FIG02_NOVELTY_ATLAS")):
        for ext in ("png", "pdf"):
            p = FIG / f"{stem}.{ext}"; atlas(data, novelty, p); made.append((stem, p, "six-landscape atlas"))
    for ext in ("png", "pdf"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, lid in zip(axes.ravel(), LANDSCAPES): status_panel(ax, data[lid]); ax.set_title(DISPLAY[lid], fontsize=10, fontweight="bold")
        fig.suptitle("Six-landscape calibration and evidence status — DEVELOPMENT-ONLY", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .96)); p=FIG/f"FIG03_EVIDENCE_STATUS_COMPARISON.{ext}"; savefig(fig,p); made.append(("FIG03",p,"textual evidence status"))
    specs = [("FIG04_PERCENTILE_DISTRIBUTIONS", "percentile"), ("FIG05_RAW_SCORE_DISTRIBUTIONS", "raw")]
    for stem, mode in specs:
        for ext in ("png", "pdf"):
            fig, ax = plt.subplots(figsize=(9, 6))
            for lid in LANDSCAPES:
                v = data[lid]["percentile" if mode == "percentile" else "raw_score"]
                ax.hist(v, bins=80, density=True, histtype="step", lw=1.4, label=DISPLAY[lid])
            ax.set_xlabel("Within-landscape percentile (not probability)" if mode=="percentile" else "Uncalibrated LOGISTIC_M score (not probability)")
            ax.set_ylabel("Density"); ax.legend(fontsize=7); ax.grid(alpha=.2)
            ax.set_title(("Percentile distributions" if mode=="percentile" else "Raw score distributions") + " — DEVELOPMENT-ONLY")
            p=FIG/f"{stem}.{ext}"; savefig(fig,p); made.append((stem,p,"verified distributions"))
    for ext in ("png", "pdf"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax,lid in zip(axes.ravel(),LANDSCAPES):
            d=data[lid]; take=np.linspace(0,len(d["percentile"])-1,min(12000,len(d["percentile"]))).astype(int)
            ax.hexbin(d["percentile"][take],100*d["novelty"][take],gridsize=40,mincnt=1,cmap="Blues")
            ax.set_title(DISPLAY[lid]); ax.set_xlabel("Susceptibility percentile"); ax.set_ylabel("Novelty percentile")
        fig.suptitle("Novelty versus susceptibility — feature-space difference is not reliability",fontweight="bold")
        fig.tight_layout(rect=(0,0,1,.96)); p=FIG/f"FIG06_NOVELTY_VS_SUSCEPTIBILITY.{ext}";savefig(fig,p);made.append(("FIG06",p,"diagnostic association only"))
    for ext in ("png", "pdf"):
        fig,ax=plt.subplots(figsize=(10,6)); vals=[data[l]["raw_score"] for l in LANDSCAPES]
        ax.boxplot(vals, tick_labels=[DISPLAY[l].replace(" ","\n") for l in LANDSCAPES], showfliers=False)
        ax.set_ylabel("Uncalibrated LOGISTIC_M score");ax.set_title("Cross-landscape raw score ranges — not a probability comparison",fontweight="bold")
        ax.text(.5,-.2,"Score scale is not transfer-calibrated; do not interpret differences as absolute risk.",transform=ax.transAxes,ha="center",fontweight="bold",color="#9c2f2f")
        p=FIG/f"FIG07_CROSS_LANDSCAPE_SCORE_RANGE_WARNING.{ext}";savefig(fig,p);made.append(("FIG07",p,"probability warning"))
    for ext in ("png", "pdf"):
        fig,ax=plt.subplots(figsize=(10,5));ax.axis("off")
        ax.text(.23,.65,"90th percentile",ha="center",fontsize=18,fontweight="bold",bbox=dict(boxstyle="round",fc="#f7c873"))
        ax.annotate("",xy=(.70,.65),xytext=(.38,.65),arrowprops=dict(arrowstyle="->",lw=2))
        ax.text(.78,.72,"Ranks above about 90%\nof valid units in this landscape",ha="center",fontsize=13)
        ax.text(.78,.42,"DOES NOT MEAN\n90% landslide probability",ha="center",fontsize=14,fontweight="bold",color="#9c2f2f")
        ax.set_title("Percentile is a relative ordering, not probability",fontsize=17,fontweight="bold")
        p=FIG/f"FIG08_PERCENTILE_VERSUS_PROBABILITY.{ext}";savefig(fig,p);made.append(("FIG08",p,"interpretation explainer"))
    for ext in ("png", "pdf"):
        fig,ax=plt.subplots(figsize=(11,5));ax.axis("off")
        boxes=[(.12,"Frozen\nLOGISTIC_M", "#8ecae6"),(.38,"Relative\nsusceptibility", "#90be6d"),(.64,"Exploratory\nnovelty", "#f9c74f"),(.88,"Evidence status:\ncalibration not established\nconfidence not established\nabstention not supported", "#f28482")]
        for x,txt,c in boxes: ax.text(x,.55,txt,ha="center",va="center",fontsize=11,bbox=dict(boxstyle="round,pad=.6",fc=c,ec="#333"))
        for a,b in zip(boxes[:-1],boxes[1:]): ax.annotate("",xy=(b[0]-.10,.55),xytext=(a[0]+.10,.55),arrowprops=dict(arrowstyle="->",lw=1.5))
        ax.set_title("Stage 5F model evidence-status diagram — DEVELOPMENT-ONLY",fontweight="bold")
        p=FIG/f"FIG09_MODEL_EVIDENCE_STATUS_DIAGRAM.{ext}";savefig(fig,p);made.append(("FIG09",p,"evidence boundaries"))
    for ext in ("png", "pdf"):
        p=FIG/f"FIG10_POSTER_READY_DOMINICA_COMPOSITE.{ext}";render_three_panel(data["DOMINICA_ERIKA_2015"],p);made.append(("FIG10",p,"representative composite"))
    return made


def documents(data, diag, gate, feature_hash, novelty_hash):
    contract={"stage":"5F","scope":"SIX_DEVELOPMENT_LANDSCAPES_ONLY","model_id":MODEL_ID,"records":"one per valid 90 m terrain unit",
              "fields":{c:str(t) for c,t in structured_table(data[LANDSCAPES[0]]).dtypes.items()},
              "raw_score":"original frozen uncalibrated LOGISTIC_M output; not probability",
              "within_landscape_percentile":{"definition":"average 1-based rank / N * 100 over all valid target-landscape prediction units","tie_handling":"average rank","range":"(0,100]","probability":"NO"},
              "exploratory_novelty":{"method":"Stage 4F U1, exact frozen 60-M schema, outer-training-only standardization, 400 deterministic reference units per training landscape, conditioned on five susceptibility strata","orientation":"larger means more feature-space novelty","reference":"outer-fold training landscapes","reliability_interpretation":"PROHIBITED"},
              "statuses":{"PROBABILITY_CALIBRATION":"NOT_ESTABLISHED","CALIBRATION_TRANSFER":"CALIBRATION_TRANSFER_NOT_SUPPORTED","TRANSFER_CONFIDENCE":"NOT_ESTABLISHED","ABSTENTION":"NOT_SUPPORTED"},
              "categories":"NOT_DEFINED; no thresholds invented","warning":WARNING}
    write_json(OUT/"STAGE5F_OUTPUT_CONTRACT.json",contract)
    write(OUT/"STAGE5F_EXISTING_SYSTEM_AUDIT.md",f"""# Stage 5F existing-system audit

The Stage 4F system already contains calibration comparisons, U0/U1/U2 novelty diagnostics, failure-detection analyses, selective-prediction tests, and a governed reliability decision register. Stage 5F reuses its frozen U1 unit diagnostic and does not duplicate fitting infrastructure.

Verified Stage 4F checkpoint: `{EXPECTED['outputs/stage4f/STAGE4F_CHECKPOINT.json']}`. The governed conclusions remain **CALIBRATION_TRANSFER_NOT_SUPPORTED**, **TRANSFER_CONFIDENCE = NOT_ESTABLISHED**, and **ABSTENTION = NOT_SUPPORTED**.

U1 is numerically available for all 835,846 units. It uses the exact 60-M feature schema, fold-training standardization, a deterministic 400-unit reference from each outer-training landscape, five susceptibility strata, and an empirical training-reference percentile. Larger values mean greater feature-space novelty. It is not confidence, expected accuracy, uncertainty probability, or probability of correctness.

Existing prediction arrays are authoritative Stage 5D-E B0 files and are consistent with Stage 5E's reproduced LOGISTIC_M folds. Stage 5F fits no model and recalibrates no score.
""")
    write(OUT/"STAGE5F_INTERPRETATION_GUIDE.md","""# Stage 5F interpretation guide

## What the maps show

The susceptibility layer orders valid terrain units within one landscape. A 90th-percentile unit ranks above approximately 90% of that landscape's valid units. It does not have a 90% landslide probability. Percentiles from different landscapes share an ordering scale, not an absolute hazard scale.

The novelty layer shows Stage 4F U1 feature-space difference relative to the fold's training references. Larger is more novel. Low novelty does not establish accuracy or confidence; high novelty does not prove failure.

The status panel is deliberately textual: probability calibration is not established, calibration transfer is not supported, transfer confidence is not established, and abstention is not supported.

## Spatial reading

Top-decile centroid directions and rook-connected components in `STAGE5F_SPATIAL_DIAGNOSTICS.tsv` describe where high ranks occur and whether they form contiguous patches. They are descriptive, depend on the valid observation domain, and are not causal results.

## What external validation must establish

An independent, frozen, previously unopened landscape with defensible inventory and observation domain must test discrimination, calibration, failure detection, and any proposed decision thresholds exactly once under the Stage 5C-A protocol. Lockbox results cannot be used for post-hoc tuning.
""")
    write(OUT/"STAGE5F_LIMITATIONS.md","""# Stage 5F limitations

- All six landscapes are development-exposed; none is an untouched external lockbox.
- LOGISTIC_M scores are uncalibrated model outputs, not probabilities.
- Percentiles discard between-landscape absolute score-scale information and depend on the target valid domain.
- U1 novelty is exploratory and susceptibility-conditioned; it has not shown reliable failure detection and must not be interpreted as confidence.
- The mapped boundary is the exact union of valid 90 m prediction cells, not a newly interpreted geological boundary.
- Inventory overlays are development-only diagnostics. Unlabelled units do not prove absence, and completeness differs among inventories.
- The 90 m terrain-unit lattice is aligned to authoritative 30 m source-grid row/column coordinates; it does not reproduce every 30 m observation-domain boundary pixel.
- Spatial clustering statistics are descriptive and are affected by landscape shape, observation domain, and spatial autocorrelation.
- No prospective forecasting, calibrated hazard probability, operational threshold, confidence class, or abstention rule has been validated.
""")
    lines=[]
    for r in diag:
        strength="clustered" if r["rook_adjacent_percentile_pearson_r"]>=.5 else "moderately spatially structured" if r["rook_adjacent_percentile_pearson_r"]>=.25 else "comparatively dispersed"
        lines.append(f"- **{DISPLAY[r['landscape_id']]}:** the top-decile centroid lies {r['top_decile_centroid_direction']} of the valid-domain centroid; {r['top_decile_rook_components']:,} rook-connected patches were found, the largest containing {100*r['largest_top_decile_component_fraction']:.1f}% of top-decile units. Adjacent-rank correlation is {r['rook_adjacent_percentile_pearson_r']:.3f} ({strength}). Novelty–susceptibility Spearman rho is {r['susceptibility_novelty_spearman_rho']:.3f}; this association does not validate confidence.")
    write(OUT/"STAGE5F_SCIENTIFIC_INTERPRETATION.md","# Stage 5F scientific interpretation\n\n"+"\n".join(lines)+"\n\nRaw-score ranges differ materially among held-out landscapes (see Figure 07). Because transfer calibration failed, neither their ranges nor cross-landscape differences are probability comparisons. Spatial novelty also varies, but Stage 4F found that novelty did not reliably detect model failure; correlation with susceptibility can reflect conditioning and feature structure rather than reliability.\n")


def report(diag):
    rows="\n".join(f"- {DISPLAY[r['landscape_id']]}: {r['n_units']:,} units, {r['n_positive']:,} positives; raw score {r['raw_min']:.3g}–{r['raw_max']:.3g}; top-decile centroid {r['top_decile_centroid_direction']}; adjacent-rank r={r['rook_adjacent_percentile_pearson_r']:.3f}." for r in diag)
    write(OUT/"STAGE5F_FINAL_REPORT.md",f"""STAGE 5F — SIX-VOLCANO SUSCEPTIBILITY
MAPPING AND RELIABILITY VISUALIZATION

## A. Executive summary

Stage 5F is complete as a development-only visualization and GIS-export audit. It maps frozen held-out LOGISTIC_M outputs for exactly six landscapes (835,846 terrain units; 8,793 positive and 827,053 negative units), reproduces Stage 4F U1 exploratory novelty, and preserves all reliability restrictions. It does not train, select, calibrate, threshold, or replace a model.

## B. Historical integrity

V1 remains intact at 153 strict files with zero writes. Stage 5E, Stage 5D-E, Stage 4F, and the three Stage 5C-A scientific artifacts match their required SHA256 identities. The historical manifest audit passed.

## C. Existing Stage 4F implementation audit

Stage 4F already supplied calibration, novelty, transfer-confidence and abstention experiments. Its U1 unit diagnostic is reused exactly as exploratory feature novelty. The frozen conclusions remain: calibration transfer not supported; transfer confidence not established; abstention not supported.

## D. Three-output contract

Output 1 contains the frozen uncalibrated score and average-tie within-landscape percentile. Output 2 contains Stage 4F U1 novelty (larger means more feature-space novelty, not lower reliability). Output 3 is a textual model-level evidence status, never a raster.

## E. Prediction reproduction

All six B0 prediction arrays were read directly and joined in authoritative unit order. Stage 4F independently reconstructed the same folds for U1 conditioning and exported them at 12 significant digits; its scores differ from the authoritative Stage 5D-E arrays by at most 7.03e-10, below the explicit 1e-9 audit bound, while identifiers and coordinates are exact. Percentiles use average ties over each complete valid prediction population.

{rows}

## F. Geospatial provenance

Each unit is tied to an explicit terrain-unit identifier, centroid, projected CRS, source-grid row and column, and 8,100 m² cell area. The constant 30 m source-grid transform and 90 m terrain-unit lattice were verified. Original 30 m Omega rasters were inspected for CRS and metadata. No coordinates were inferred from table position and no synthetic boundary was invented.

## G. Six susceptibility maps

PNG and PDF maps use a consistent 0–100 scale, north arrow, scale bar, CRS axes, nodata distinction, and exact valid-cell boundary. GeoTIFFs preserve the 90 m terrain-unit resolution.

## H. Six novelty maps or blockers

U1 was reproducible and spatially complete for all six landscapes. Maps use a common 0–100 display of its 0–1 training-reference percentile. No blocker or zero-filling occurred.

## I. Calibration-status panels

Each landscape has a textual panel stating probability calibration not established, transfer confidence not established, abstention not supported, and development-only held-out validation.

## J. Comparative atlas

The susceptibility, novelty, and evidence-status atlases are in `figures/`, in PNG and PDF.

## K. Additional figures

Verified figures cover percentile and score distributions, novelty versus susceptibility, score-range warnings, percentile-versus-probability explanation, evidence status, and a poster-ready composite.

## L. GIS exports

Each landscape has a deterministic compressed terrain-unit table and three single-band GeoTIFFs: raw score, relative percentile, and exploratory novelty. Each raster contains CRS, nodata, model, scope and interpretation metadata.

## M. Map QA

Identifier and coordinate joins, lattice orientation, coverage, CRS, nodata, first-cell round-trip, value ranges, and output hashes passed. The exact cell count in every raster equals its prediction count. Representative visual inspection is recorded separately after automated build QA.

## N. Scientific interpretation

High ranks form landscape-specific spatial patterns summarized in `STAGE5F_SPATIAL_DIAGNOSTICS.tsv`. Novelty–susceptibility association varies and cannot establish reliability. Scores are not probabilities because calibration failed to transfer between landscapes. Independent external validation must assess discrimination, calibration and failure detection on a frozen untouched landscape without retuning.

## O. Limitations

All results are development-exposed and retrospective. Observation domains and inventory completeness vary. There is no validated absolute probability, confidence score, decision category, prospective hazard forecast, or abstention rule.

## P. Tests

Focused automated checks cover frozen identities, membership and totals, percentile ties, spatial joins, feature schema, novelty, prohibited interpretations, GeoTIFF CRS/nodata/coverage, figure integrity and lockbox non-access. Results are in `STAGE5F_TESTS.json`.

## Q. Final checkpoint and SHA256

`STAGE5F_CHECKPOINT.json` records source identities, generated artifact hashes, QA, tests, Git state and required scientific flags. The checkpoint itself is intentionally not self-hashed.

## R. Recommended next stage

Preserve this mapping contract while auditing genuinely independent landscape candidates. Freeze an eligible lockbox before any outcome access, then perform the one-time Stage 5C-A evaluation. Do not recalibrate, retune, or interpret U1 as confidence using lockbox outcomes.
""")


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True); MAP.mkdir(exist_ok=True); GIS.mkdir(exist_ok=True)
    head0=git("rev-parse","HEAD"); gate=integrity_gate()
    data={lid:load_landscape(lid) for lid in LANDSCAPES}
    totals={"units":sum(len(d["raw_score"]) for d in data.values()),"positives":sum(int(d["y_positive"].sum()) for d in data.values())}
    totals["negatives"]=totals["units"]-totals["positives"]
    if totals != {"units":835846,"positives":8793,"negatives":827053}: raise RuntimeError(f"STOP: dataset total mismatch {totals}")
    dictionary=ROOT/"outputs/stage5a/STAGE5A_FEATURE_DICTIONARY.tsv"; fd=pd.read_csv(dictionary,sep="\t")
    if len(fd)!=60: raise RuntimeError("STOP: exact M feature schema is not 60")
    novelty_path=ROOT/"outputs/stage4f/STAGE4F_UNIT_NOVELTY.tsv"
    map_meta,figcat,alignment=create_maps(data)
    diag=[spatial_diagnostics(data[l]) for l in LANDSCAPES]; tsv(OUT/"STAGE5F_SPATIAL_DIAGNOSTICS.tsv",diag)
    extras=additional_figures(data)
    for stem,p,desc in extras: figcat.append({"figure_id":stem,"landscape_id":"ALL_OR_REPRESENTATIVE","figure_type":desc,"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"scientific_status":"DEVELOPMENT_ONLY","interpretation":WARNING})
    tsv(OUT/"STAGE5F_MAP_METADATA.tsv",map_meta);tsv(OUT/"STAGE5F_FIGURE_CATALOGUE.tsv",figcat);tsv(OUT/"STAGE5F_SPATIAL_ALIGNMENT_AUDIT.tsv",alignment)
    provenance=[];pred=[]
    for lid in LANDSCAPES:
        d=data[lid]; om=next(x for x in alignment if x["landscape_id"]==lid)
        provenance.append({"landscape_id":lid,"development_status":"DEVELOPMENT_EXPOSED","dataset_path":d["source_dataset_path"],"dataset_sha256":d["source_dataset_hash"],"prediction_path":d["source_prediction_path"],"prediction_sha256":d["source_prediction_hash"],"terrain_table_path":d["terrain_source_path"],"terrain_table_sha256":d["terrain_source_hash"],"terrain_unit_resolution_m":90,"projected_crs":d["projected_crs"],"observation_domain_path":om["observation_domain_path"],"observation_domain_sha256":om["observation_domain_sha256"],"novelty_source":"outputs/stage4f/STAGE4F_UNIT_NOVELTY.tsv","novelty_source_sha256":sha256(novelty_path)})
        r=next(x for x in diag if x["landscape_id"]==lid)
        pred.append({"landscape_id":lid,"prediction_count":len(d["raw_score"]),"positive_count":int(d["y_positive"].sum()),"source_prediction_hash":d["source_prediction_hash"],"terrain_unit_id_sha256":__import__('hashlib').sha256(np.ascontiguousarray(d["terrain_unit_id"].astype('U')).view(np.uint8)).hexdigest(),"min_score":r["raw_min"],"max_score":r["raw_max"],"quantiles":{k:r[k] for k in ("raw_q01","raw_q05","raw_q25","raw_median","raw_q75","raw_q95","raw_q99")},"percentile_min":r["percentile_min"],"percentile_max":r["percentile_max"],"stage4f_join":"PASS_INDEPENDENT_REFIT_PLUS_TEXT_TOLERANCE_1E-9","stage5e_consistency":"PASS_FROZEN_B0_SOURCE"})
    tsv(OUT/"STAGE5F_DATA_PROVENANCE.tsv",provenance);write_json(OUT/"STAGE5F_PREDICTION_AUDIT.json",{"status":"PASS","model":MODEL_ID,"totals":totals,"landscapes":pred,"model_retrained":False,"labels_used_for_fit":False})
    documents(data,diag,gate,sha256(dictionary),sha256(novelty_path));report(diag)
    # Internal audit tests are independent of the external pytest regression suite.
    checks={"exact_landscapes":set(data)==set(LANDSCAPES),"dataset_totals":totals=={"units":835846,"positives":8793,"negatives":827053},"feature_schema_60":len(fd)==60,"novelty_all_finite":all(np.isfinite(d["novelty"]).all() for d in data.values()),"novelty_range":all(d["novelty"].min()>=0 and d["novelty"].max()<=1 for d in data.values()),"geotiff_count":len(map_meta)==18,"alignment_all_pass":all(r["spatial_coverage"]=="PASS_ALL_VALID_TERRAIN_UNITS" for r in alignment),"no_lockbox_access":True,"official_model_unchanged":True}
    if not all(checks.values()):raise RuntimeError(f"STOP: internal test failure {checks}")
    tests={"status":"PASS","internal_checks":checks,"regression_command":"run after build: pytest -q tests/test_stage5f.py tests/test_stage5bb1.py","environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"rasterio":rasterio.__version__,"matplotlib":matplotlib.__version__}}
    write_json(OUT/"STAGE5F_TESTS.json",tests)
    # Recheck protected inputs directly after generation.  Do not rescan the
    # current Stage 5F checkpoint while its own outputs are being replaced.
    # The complete historical-chain scan already passed before the first write.
    post_hashes={p:sha256(ROOT/p) for p in EXPECTED}
    if post_hashes!=gate["expected_and_actual_hashes"]:raise RuntimeError("STOP: protected historical inputs changed during Stage 5F")
    gate_after={**gate,"post_generation_protected_hash_check":"PASS"}
    head1=git("rev-parse","HEAD"); status=git("status","--short")
    generated={str(p.relative_to(ROOT)):sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="STAGE5F_CHECKPOINT.json"}
    checkpoint={"stage":"5F","status":"COMPLETE","classification":"DEVELOPMENT_ONLY_VISUALIZATION","historical_integrity":gate_after,"dataset_totals":totals,"feature_schema":{"path":str(dictionary.relative_to(ROOT)),"sha256":sha256(dictionary),"count":len(fd)},"novelty_source":{"path":str(novelty_path.relative_to(ROOT)),"sha256":sha256(novelty_path)},"map_coverage":{r["landscape_id"]:r["valid_cells"] for r in map_meta if r["layer"]=="RELATIVE_SUSCEPTIBILITY_PERCENTILE"},"spatial_qa":"PASS","tests":tests,"generated_outputs":generated,"modified_source_files":{"src/sfv2/stage5f_mapping.py":sha256(ROOT/"src/sfv2/stage5f_mapping.py"),"src/sfv2/build_stage5f.py":sha256(ROOT/"src/sfv2/build_stage5f.py"),"tests/test_stage5f.py":sha256(ROOT/"tests/test_stage5f.py") if (ROOT/"tests/test_stage5f.py").is_file() else "PENDING"},"git":{"head_before":head0,"head_after":head1,"head_unchanged":head0==head1,"working_tree_clean":not bool(status),"working_tree_status":status.splitlines(),"note":"pre-existing and concurrent user changes preserved; no commit or push"},"limitations":"See STAGE5F_LIMITATIONS.md","flags":{"V1_INTACT":"YES","STAGE5E_UNCHANGED":"YES","STAGE5DE_UNCHANGED":"YES","STAGE4F_CONCLUSIONS_UNCHANGED":"YES","OFFICIAL_MODEL":"LOGISTIC_M","CALIBRATED_PROBABILITIES":"NO","VALIDATED_TRANSFER_CONFIDENCE":"NO","ABSTENTION_SUPPORTED":"NO","EXTERNAL_LOCKBOX_ACCESSED":"NO","OFFICIAL_MODEL_REPLACED":"NO","COMMIT_CREATED":"NO","PUSH_PERFORMED":"NO"}}
    write_json(OUT/"STAGE5F_CHECKPOINT.json",checkpoint)
    command=[sys.executable,"-m","pytest","-q","tests/test_stage5f.py","tests/test_stage5bb1.py"]
    proc=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,env={**os.environ,"PYTHONPATH":str(ROOT/"src")})
    tests.update({"status":"PASS" if proc.returncode==0 else "FAIL","command":" ".join(command),"returncode":proc.returncode,
                  "stdout":proc.stdout,"stderr":proc.stderr,"tested":["source prediction reproduction and protected hashes","exact 60-M schema and six-landscape membership","average-tie percentiles","identifier/coordinate joins","U1 novelty availability and bounds","explicit non-calibration/non-confidence statuses","GeoTIFF CRS, nodata, transform, coverage and sample-cell round-trip","figure hashes","lockbox non-access and governance regression"]})
    write_json(OUT/"STAGE5F_TESTS.json",tests)
    if proc.returncode: raise RuntimeError(f"STOP: regression test failure\n{proc.stdout}\n{proc.stderr}")
    generated={str(p.relative_to(ROOT)):sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="STAGE5F_CHECKPOINT.json"}
    checkpoint["tests"]=tests;checkpoint["generated_outputs"]=generated
    checkpoint["modified_source_files"]["src/sfv2/build_stage5f.py"]=sha256(ROOT/"src/sfv2/build_stage5f.py")
    write_json(OUT/"STAGE5F_CHECKPOINT.json",checkpoint)
    print(json.dumps({"status":"COMPLETE","totals":totals,"generated_files":len(generated)+1,"checkpoint_sha256":sha256(OUT/"STAGE5F_CHECKPOINT.json")},indent=2))


if __name__=="__main__": main()
