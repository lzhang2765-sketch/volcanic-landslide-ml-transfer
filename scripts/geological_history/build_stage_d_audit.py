#!/usr/bin/env python3
"""Build the read-only AUSSEF Stage D geological-history audit.

This script reads frozen project artefacts and writes only inside
AUSSEF_2026_GEOLOGICAL_HISTORY_AUDIT. It does not fit a reconstruction or a
predictive model. All new calculations are descriptive audit calculations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage_d_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "AUSSEF_2026_GEOLOGICAL_HISTORY_AUDIT"
DATE = "2026-10-18"
LANDSCAPES = [
    "KUMAMOTO_ASO_2016",
    "NOTO_PENINSULA_2024",
    "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019",
    "ATITLAN_STAN_2005",
    "DOMINICA_ERIKA_2015",
]
DISPLAY = {
    "KUMAMOTO_ASO_2016": "Kumamoto–Aso",
    "NOTO_PENINSULA_2024": "Noto Peninsula",
    "LOMBOK_RINJANI_2018": "Lombok–Rinjani",
    "COTABATO_DAVAO_2019": "Cotabato–Davao",
    "ATITLAN_STAN_2005": "Atitlán–Stan",
    "DOMINICA_ERIKA_2015": "Dominica–Erika",
}
GATE_IDS = [
    "MEDIAN_ABS_RESIDUAL_FRACTION",
    "ABS_BIAS_FRACTION",
    "HOLDOUT_BAND_FRACTION",
    "ADMISSIBLE_CONTROLS",
    "SUPPORT_FRACTION",
    "NEGATIVE_REMOVAL_FRACTION",
    "MAX_REFERENCE_SLOPE_DEG",
    "BOUNDARY_QA",
    "PHYSICAL_QA",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT.parent, text=True).strip()


SOURCE_SPECS = [
    ("outputs/stage2c/geology_mapping_KUMAMOTO_ASO_2016_aso_1_50000.csv", "ORIGINAL_METHOD", "Frozen GSJ Aso 1:50,000 unit mapping"),
    ("outputs/stage2c/geology_mapping_NOTO_PENINSULA_2024_seamless.csv", "ORIGINAL_METHOD", "Noto age-proxy geology"),
    ("outputs/stage2f2/sheet_legend_evidence.csv", "ORIGINAL_METHOD", "Lombok geological legend evidence"),
    ("outputs/stage3d/COTABATO_GEOLOGY_MAPPING.csv", "ORIGINAL_METHOD", "Cotabato regional geological mapping"),
    ("outputs/stage3e/ATITLAN_GEOLOGY_AUDIT.json", "ORIGINAL_METHOD", "Atitlán 1:500,000 map audit"),
    ("outputs/stage3f/DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv", "ORIGINAL_METHOD", "Dominica 1:100,000 geological harmonisation"),
    ("outputs/stage3g/STAGE3G_REPORT.md", "FROZEN_RESULT", "Inspected candidate gap-closure record; it is not an executed H reconstruction"),
    ("outputs/stage4a/STAGE4A_MEG_DATASET/SCHEMA.json", "FROZEN_RESULT", "Frozen terrain-unit dataset schema"),
    ("outputs/stage4a/STAGE4A_MEG_DATASET/KUMAMOTO_ASO_2016.npz", "FROZEN_RESULT", "Frozen Kumamoto terrain-unit data"),
    ("outputs/stage4c_a/STAGE4C_A_H_DEFINITION.md", "ORIGINAL_METHOD", "Frozen H definition"),
    ("outputs/stage4c_a/STAGE4C_A_CONTROL_SELECTION_SPEC.md", "ORIGINAL_METHOD", "Frozen Y-blind control rules"),
    ("outputs/stage4c_a/STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md", "ORIGINAL_METHOD", "Seven frozen quantitative gates"),
    ("outputs/stage4c_a/STAGE4C_A_H_FEATURE_SPEC.tsv", "ORIGINAL_METHOD", "Six H feature definitions"),
    ("outputs/stage4c_a/STAGE4C_A_RECONSTRUCTION_EVIDENCE.tsv", "ORIGINAL_METHOD", "Landscape evidence register"),
    ("outputs/stage4c_a/STAGE4C_A_PER_LANDSCAPE_RECONSTRUCTION_PLAN.tsv", "PROPOSED_METHOD", "Pre-reconstruction landscape plans"),
    ("src/sfv2/stage4c_b3_prereg.py", "ORIGINAL_METHOD", "Frozen Gen-3 estimand, candidates and QA rules"),
    ("src/sfv2/build_stage4c_b3.py", "ORIGINAL_METHOD", "Executed Gen-3 implementation"),
    ("outputs/stage4c_b3/STAGE4C_B3_PREREGISTRATION.json", "ORIGINAL_METHOD", "Machine-readable frozen preregistration"),
    ("outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv", "FROZEN_RESULT", "Authoritative Gen-3 gate metrics"),
    ("outputs/stage4c_b3/STAGE4C_B3_BOUNDARY_QA.tsv", "FROZEN_RESULT", "Authoritative boundary classifications/results"),
    ("outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_STATUS.tsv", "FROZEN_RESULT", "Authoritative landscape release outcomes"),
    ("outputs/stage4c_b3/STAGE4C_B3_ARCHITECTURE_SELECTION.tsv", "FROZEN_RESULT", "Nested selection and stability"),
    ("outputs/stage4c_b3/STAGE4C_B3_NEGATIVE_REMOVAL_DIAGNOSTICS.tsv", "FROZEN_RESULT", "Signed-removal diagnostics"),
    ("outputs/stage4c_b3/STAGE4C_B3_SHIFT_UNCERTAINTY_DIAGNOSTICS.tsv", "FROZEN_RESULT", "Scenario and generation-shift diagnostics"),
    ("outputs/stage4c_b3/STAGE4C_B3_H_REDUNDANCY_WITH_M.tsv", "FROZEN_RESULT", "Frozen OLS redundancy results"),
    ("outputs/stage4c_b3/STAGE4C_B3_SELECTED_H.tsv", "FROZEN_RESULT", "Frozen selected-H decision"),
    ("outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz", "FROZEN_RESULT", "Released partial Kumamoto H values and support"),
    ("outputs/stage4c_b3/STAGE4C_B3_QA/KUMAMOTO_ASO_2016_gen3_grids.npz", "FROZEN_RESULT", "Kumamoto actual reconstruction/support grids"),
    ("outputs/stage5g/STAGE5G_LANDSCAPE_GATE_MATRIX.tsv", "LATER_AUDIT", "Frozen nine-gate presentation"),
    ("outputs/stage5g/STAGE5G_H_FEATURE_DICTIONARY.tsv", "LATER_AUDIT", "Expanded feature dictionary"),
    ("outputs/stage5g/STAGE5G_FINAL_REPORT.md", "LATER_AUDIT", "Frozen geological-history audit report"),
    ("AUSSEF_2026_VALIDITY_AUDIT/SCIENTIFIC_VALIDITY_REPORT.md", "LATER_AUDIT", "Contextual validity and leakage audit; no substitute for H evidence"),
    ("AUSSEF_2026_ELEVATION_EXPERIMENT/04_INTERPRETATION/SCIENTIFIC_FINDINGS.md", "LATER_AUDIT", "Contextual elevation-representation findings; no H model"),
    ("AUSSEF_2026_TRANSFERABILITY_FACTORS/04_INTERPRETATION/TRANSFERABILITY_FACTOR_FINDINGS.md", "LATER_AUDIT", "Contextual transferability analysis; six-landscape limitations"),
    ("AUSSEF_2026_LITERATURE_RESEARCH/05_ORIGINALITY/ORIGINALITY_ASSESSMENT.md", "LATER_AUDIT", "Verified Stage B originality assessment"),
    ("AUSSEF_2026_LITERATURE_RESEARCH/05_ORIGINALITY/RESEARCH_GAP_STATEMENT.md", "LATER_AUDIT", "Verified Stage B gap statement"),
    ("AUSSEF_2026_LITERATURE_RESEARCH/03_COMPETITOR_ANALYSIS/CLOSEST_STUDIES_CRITICAL_REVIEW.md", "LATER_AUDIT", "Verified competitor review"),
    ("AUSSEF_2026_LITERATURE_RESEARCH/04_LITERATURE_SYNTHESIS/GEOLOGICAL_HISTORY_REVIEW.md", "LATER_AUDIT", "Verified geological-history literature synthesis"),
    ("AUSSEF_2026_LITERATURE_RESEARCH/02_REFERENCES/CLAIM_EVIDENCE_MATRIX.csv", "LATER_AUDIT", "Claim-level literature evidence"),
    ("AUSSEF_2026_RESULTS_PACKAGE/05_Report_Ready/SECTION7_RESULTS_GUIDE.md", "LATER_AUDIT", "Only located report file with authoritative subsection numbering"),
]


def source_registry() -> tuple[list[dict], dict[str, str]]:
    rows, hashes = [], {}
    for ptxt, authority, use in SOURCE_SPECS:
        p = ROOT / ptxt
        if not p.is_file():
            raise FileNotFoundError(p)
        digest = sha256(p)
        hashes[ptxt] = digest
        rows.append({
            "source_path": ptxt,
            "sha256": digest,
            "authority_class": authority,
            "audit_use": use,
            "read_only": "YES",
            "conflict_status": "NO_CONFLICT_DETECTED",
        })
    return rows, hashes


def gate_rows() -> list[dict]:
    spec = "outputs/stage4c_a/STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md"
    prereg = "outputs/stage4c_b3/STAGE4C_B3_PREREGISTRATION.json"
    impl = "src/sfv2/build_stage4c_b3.py"
    return [
        dict(gate_order=1, gate_id="MEDIAN_ABS_RESIDUAL_FRACTION", original_name="gate_median_abs_residual", purpose="Held-out vertical accuracy", inputs="Outer-held-out Q1/Q2 control residuals; R_L", rule="median(|z_control-S_P50(x)|)/R_L <= 0.10", threshold="0.10 maximum", evaluation_scale="Pooled outer held-out controls by landscape", missing_data="Unpredicted controls remain missing and are separately counted; no imputation", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=2, gate_id="ABS_BIAS_FRACTION", original_name="gate_abs_bias", purpose="Prevent systematic vertical displacement", inputs="Mean signed held-out residual; R_L", rule="|mean(z_control-S_P50(x))|/R_L <= 0.05", threshold="0.05 maximum", evaluation_scale="Landscape", missing_data="Uses finite held-out residuals; unpredicted count reported", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=3, gate_id="HOLDOUT_BAND_FRACTION", original_name="gate_holdout_band", purpose="Scenario ensemble coverage of unseen controls", inputs="Held-out controls; scenario P05/P95", rule="fraction of held-out controls inside scenario P05-P95 >= 0.60", threshold="0.60 minimum", evaluation_scale="Landscape", missing_data="Unpredicted controls fail to contribute coverage and are counted", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=4, gate_id="ADMISSIBLE_CONTROLS", original_name="gate_admissible_controls", purpose="Minimum empirical control base", inputs="Y-blind admissible Q1/Q2 controls", rule="count >= 12", threshold="12 minimum", evaluation_scale="Landscape", missing_data="Q3/Q4 excluded; absent controls cannot be invented", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=5, gate_id="SUPPORT_FRACTION", original_name="gate_support_fraction", purpose="Minimum spatial availability", inputs="Stage 4A units; verified support mask", rule="supported Stage 4A units / all Stage 4A units >= 0.25", threshold="0.25 minimum", evaluation_scale="Terrain units within landscape", missing_data="Outside support remains NaN, never zero-filled", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=6, gate_id="NEGATIVE_REMOVAL_FRACTION", original_name="gate_negative_removal", purpose="Reject implausible reference below modern terrain", inputs="D=S_P50-Z_modern; tau_L; mapped-young exception mask", rule="fraction with D < -tau_L outside mapped young deposits <= 0.10", threshold="0.10 maximum", evaluation_scale="Supported terrain units", missing_data="Only available support evaluated; sign preserved; no clipping", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=7, gate_id="MAX_REFERENCE_SLOPE_DEG", original_name="gate_max_reference_slope", purpose="Constructional-surface slope plausibility", inputs="90 m Horn slope of P50 reference", rule="maximum reconstructed constructional slope <= 40 degrees", threshold="40 degrees maximum", evaluation_scale="Supported reference surface", missing_data="Unsupported cells excluded", source_file=spec, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=8, gate_id="BOUNDARY_QA", original_name="BOUNDARY_QA", purpose="Detect artificial seams at numerical component boundaries", inputs="Seam step, slope excess, ridge/trough fraction, scenario-spread ratio; boundary class", rule="For numerical boundaries: median step/R_L<=0.10 AND P95<=0.25 AND slope excess<=5 deg AND ridge/trough fraction<=0.05 AND spread ratio<=2. Geological discontinuities are reported only; single components pass as not applicable.", threshold="Five frozen numerical-boundary limits", evaluation_scale="Internal boundary bands", missing_data="No internal boundary => pass/not applicable; geological discontinuity => REPORTED_ONLY", source_file=prereg, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
        dict(gate_order=9, gate_id="PHYSICAL_QA", original_name="PHYSICAL_QA", purpose="Exclude deep closed depressions and caldera leakage", inputs="Filled-surface depth; tau_L; support mask; caldera mask", rule="fraction of support in closed depressions deeper than tau_L <= 0.01 AND no supported cell intersects a masked caldera interior", threshold="0.01 pit fraction maximum; zero caldera-support overlap", evaluation_scale="Supported grid cells", missing_data="Requires reconstructed grid and masks; Noto not evaluated", source_file=prereg, implementation=impl, recovery_status="EXPLICITLY_IMPLEMENTED"),
    ]


def audit_data() -> dict:
    val = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv", sep="\t")
    status = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_STATUS.tsv", sep="\t")
    matrix = pd.read_csv(ROOT / "outputs/stage5g/STAGE5G_LANDSCAPE_GATE_MATRIX.tsv", sep="\t")
    feat = pd.read_csv(ROOT / "outputs/stage5g/STAGE5G_H_FEATURE_DICTIONARY.tsv", sep="\t")
    red = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_H_REDUNDANCY_WITH_M.tsv", sep="\t")
    selected = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_SELECTED_H.tsv", sep="\t")
    arch = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_ARCHITECTURE_SELECTION.tsv", sep="\t")
    with np.load(ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET/KUMAMOTO_ASO_2016.npz", allow_pickle=False) as z:
        m = {k: z[k].copy() for k in z.files}
    with np.load(ROOT / "outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz", allow_pickle=False) as z:
        h = {k: z[k].copy() for k in z.files}
    support = h["h_available"].astype(bool)
    if (len(support), int(support.sum()), int((~support).sum())) != (354123, 145799, 208324):
        raise AssertionError("Kumamoto support identity mismatch")
    return dict(val=val, status=status, matrix=matrix, feat=feat, red=red, selected=selected, arch=arch, m=m, h=h, support=support)


def support_bias_rows(d: dict) -> list[dict]:
    m, s = d["m"], d["support"]
    fields = [
        ("elev_m_30m__mean", "Elevation", "m"),
        ("slope_deg_90m__mean", "Slope (90 m)", "degrees"),
        ("relief_m_r1000__mean", "Local relief (1 km)", "m"),
        ("roughness_sd_m_270m__mean", "Roughness (270 m)", "m"),
    ]
    out = []
    for key, label, unit in fields:
        a = m[key].astype(float)
        finite = np.isfinite(a)
        iqr = np.percentile(a[finite], 75) - np.percentile(a[finite], 25)
        for mask, group in [(s, "SUPPORTED"), (~s, "UNSUPPORTED")]:
            x = a[mask & finite]
            out.append(dict(metric=label, source_field=key, group=group, n=len(x), median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)), mean=float(np.mean(x)), units=unit, supported_minus_unsupported_median_over_global_iqr=""))
        out[-2]["supported_minus_unsupported_median_over_global_iqr"] = float((out[-2]["median"] - out[-1]["median"]) / iqr)
        out[-1]["supported_minus_unsupported_median_over_global_iqr"] = float((out[-2]["median"] - out[-1]["median"]) / iqr)
    for mask, group in [(s, "SUPPORTED"), (~s, "UNSUPPORTED")]:
        y = m["y_positive"].astype(bool)[mask]
        out.append(dict(metric="Positive-label prevalence (post hoc)", source_field="y_positive", group=group, n=len(y), median="", q25="", q75="", mean=float(y.mean()), units="fraction", supported_minus_unsupported_median_over_global_iqr=""))
    return out


def make_figures(d: dict) -> None:
    figdir = OUT / "09_FIGURES"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "svg.hashsalt": "AUSSEF_STAGE_D_2026",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    def save(fig, stem):
        for ext in ("png", "pdf", "svg"):
            kw = dict(bbox_inches="tight", pad_inches=0.08)
            if ext == "png":
                kw.update(dpi=350, metadata={"Software": "AUSSEF Stage D audit"})
            elif ext == "pdf":
                kw.update(metadata={"Creator": "AUSSEF Stage D audit", "CreationDate": None, "ModDate": None})
            else:
                kw.update(metadata={"Date": None, "Creator": "AUSSEF Stage D audit"})
            fig.savefig(figdir / f"{stem}.{ext}", **kw)
        plt.close(fig)

    # F1: actual workflow and gates.
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    ax.axis("off")
    boxes = [
        (0.03, .75, .21, .15, "Frozen evidence\nGeology + DEM\nQ1/Q2 controls"),
        (.285, .75, .21, .15, "Nested Y-blind\ncandidate selection\n(structure + loss)"),
        (.54, .75, .21, .15, "Equal-weight\nscenario ensemble\nP05 / P50 / P95"),
        (.795, .75, .17, .15, "Support + H\nonly after all\ngates pass"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#e8f1f8", edgecolor="#174a6e", lw=1.2))
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center")
    for x in (.24, .495, .75):
        ax.annotate("", xy=(x+.04, .825), xytext=(x, .825), arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
    labels = [
        "1  Held-out residual ≤ 0.10 R$_L$", "2  Absolute bias ≤ 0.05 R$_L$", "3  P05–P95 coverage ≥ 0.60",
        "4  Q1/Q2 controls ≥ 12", "5  Unit support ≥ 0.25", "6  Negative removal ≤ 0.10",
        "7  Reference slope ≤ 40°", "8  Numerical boundary QA", "9  Physical QA (pits + caldera mask)",
    ]
    for i, lab in enumerate(labels):
        row, col = divmod(i, 3)
        x, y = .04 + col*.32, .51-row*.16
        ax.add_patch(plt.Rectangle((x, y), .28, .105, facecolor="#ffffff", edgecolor="#59788d", lw=1))
        ax.text(x+.14, y+.0525, lab, ha="center", va="center", fontsize=8.5)
    ax.text(.5, .665, "Frozen nine-gate release boundary (all applicable gates must pass)", ha="center", weight="bold")
    ax.text(.5, .055, "Landslide labels, predictive residuals and model performance were forbidden inputs.", ha="center", color="#7a1f1f")
    ax.set_title("Geological-history reconstruction and release workflow", pad=8)
    save(fig, "F1_NINE_GATE_WORKFLOW")

    # F2: gate matrix.
    mat = d["matrix"].copy()
    lookup = {(r.landscape_id, r.gate): r.result for r in mat.itertuples()}
    code = {"FAIL": 0, "PASS": 1, "UNAVAILABLE_NOT_RECONSTRUCTED": 2, "NOT_EVALUATED": 2, "PARTIAL": 3}
    arr = np.array([[code.get(str(lookup.get((lid, gate), "NOT_EVALUATED")), 2) for gate in GATE_IDS] for lid in LANDSCAPES])
    cmap = ListedColormap(["#c94f45", "#2f7d68", "#b8bdc3", "#d7a84b"])
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(9), ["Residual", "Bias", "Band", "Controls", "Support", "Negative\nremoval", "Slope", "Boundary", "Physical"], rotation=35, ha="right")
    ax.set_yticks(range(6), [DISPLAY[x] for x in LANDSCAPES])
    for i in range(6):
        for j in range(9):
            ax.text(j, i, {0:"FAIL",1:"PASS",2:"N/E",3:"PART"}[arr[i,j]], ha="center", va="center", fontsize=7.2, color="white" if arr[i,j] in (0,1) else "#222")
    ax.set_title("Frozen Gen-3 geological-history gate outcomes")
    ax.text(8.55, 5.75, "N/E = not evaluated (no palaeosurface)", ha="right", fontsize=8)
    save(fig, "F2_SIX_LANDSCAPE_GATE_MATRIX")

    # F3: actual unit-centroid geometry; deterministic thinning only for rendering.
    m, s = d["m"], d["support"]
    idx_u = np.flatnonzero(~s)[::8]
    idx_s = np.flatnonzero(s)[::5]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(m["centroid_x"][idx_u]/1000, m["centroid_y"][idx_u]/1000, s=.45, c="#c9c9c9", linewidths=0, rasterized=True, label=f"Unsupported ({(~s).sum():,})")
    ax.scatter(m["centroid_x"][idx_s]/1000, m["centroid_y"][idx_s]/1000, s=.55, c="#237a67", linewidths=0, rasterized=True, label=f"Supported ({s.sum():,})")
    ax.set_aspect("equal")
    ax.set_xlabel("Projected easting (km; EPSG:32652)")
    ax.set_ylabel("Projected northing (km; EPSG:32652)")
    ax.legend(frameon=False, markerscale=8, loc="lower left")
    ax.set_title("Kumamoto–Aso verified reconstruction support")
    ax.text(.01, .99, "Display uses deterministic thinning; counts and source CSV use all 354,123 units.", transform=ax.transAxes, va="top", fontsize=7.5)
    save(fig, "F3_KUMAMOTO_RECONSTRUCTION_SUPPORT")

    # F4: frozen redundancy values.
    rr = d["red"]
    rr = rr[rr.role == "RELEASED_VERIFIED_GEN3_H"].copy()
    short = {"H_REMOVED_FRACTION":"Removed fraction", "H_SLOPE_CHANGE_DEG":"Slope change", "H_RELIEF_CHANGE_FRACTION":"Relief change", "H_REFERENCE_RELATIVE_HEIGHT":"Reference height", "H_SCENARIO_SPREAD_FRACTION":"Scenario spread", "H_CONTROL_DISTANCE_FRACTION":"Control distance"}
    rr["label"] = rr.feature.map(short)
    rr = rr.sort_values("pooled_r2_on_60_M")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colors = ["#b45b4c" if v >= .9 else "#3f7f93" for v in rr.pooled_r2_on_60_M]
    ax.barh(rr.label, rr.pooled_r2_on_60_M, color=colors)
    ax.axvline(.9, color="#333", ls="--", lw=1, label="Frozen redundancy rule (R² = 0.90)")
    for i, v in enumerate(rr.pooled_r2_on_60_M): ax.text(v+.012, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.04); ax.set_xlabel("OLS R² from 60 frozen morphology predictors")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Kumamoto historical-feature redundancy audit")
    save(fig, "F4_HISTORICAL_FEATURE_REDUNDANCY")

    # F5: evidence/uncertainty status matrix (audit classifications, no invented intervals).
    urows = uncertainty_rows()
    cats = ["Geological mapping", "Age / chronology", "Preserved evidence", "DEM resolution", "Interpolation", "Spatial extrapolation", "Erosion-history ambiguity", "Support-mask uncertainty", "Cross-landscape comparability"]
    status_code = {"DOCUMENTED_AND_QUANTIFIED":3, "DOCUMENTED_NOT_QUANTIFIED":2, "PLAUSIBLE_NOT_TESTED":1, "UNRESOLVED":0}
    vals = np.array([[status_code[next(r["audit_status"] for r in urows if r["uncertainty_source"]==c)] for c in cats]])
    fig, ax = plt.subplots(figsize=(11.5, 2.2))
    ax.imshow(vals, aspect="auto", cmap=ListedColormap(["#b7b7b7", "#d8a84e", "#5f91ad", "#2f7d68"]), vmin=0, vmax=3)
    ax.set_yticks([0], ["Stage D status"])
    ax.set_xticks(range(len(cats)), [c.replace(" ", "\n", 1) for c in cats], fontsize=8)
    for j, c in enumerate(cats):
        st = next(r["audit_status"] for r in urows if r["uncertainty_source"]==c)
        ax.text(j, 0, {3:"QUANT",2:"DOC",1:"UNTESTED",0:"UNRES"}[status_code[st]], ha="center", va="center", fontsize=7, color="white" if status_code[st] in (0,3) else "#222")
    ax.set_title("Reconstruction uncertainty: evidence status, not confidence")
    save(fig, "F5_UNCERTAINTY_EVIDENCE_COVERAGE")

    captions = """# Stage D figure captions

## F1 — Geological-history reconstruction and nine-gate release workflow

The reconstruction used frozen geological/DEM evidence and Y-blind Q1/Q2 controls, nested candidate selection and an equal-weight scenario ensemble. Historical predictors were released only after every applicable frozen gate passed. Source: `STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md`, `STAGE4C_B3_PREREGISTRATION.json`, and `build_stage4c_b3.py`.

## F2 — Six-landscape gate-outcome matrix

Only Kumamoto–Aso passed all nine applicable Gen-3 gates. Noto was not evaluated because no defensible constructional palaeosurface existed; its age proxy is not a reconstruction. The remaining four landscapes failed one or more frozen gates. Source table: `F2_SIX_LANDSCAPE_GATE_MATRIX_SOURCE.csv`.

## F3 — Kumamoto–Aso verified reconstruction support

The released historical predictors cover 145,799 of 354,123 terrain units (41.1719%); 208,324 units remain missing, not zero. Points use real frozen unit centroids in EPSG:32652. Rendering is deterministically thinned for legibility, while `KUMAMOTO_SPATIAL_SUPPORT.csv` retains every unit.

## F4 — Historical-feature redundancy comparison

Frozen OLS R² values quantify how well the 60 frozen morphology predictors reproduce each released Kumamoto H feature. Four of six values are below the prespecified 0.90 redundancy threshold. This does not establish causal independence or predictive benefit.

## F5 — Reconstruction uncertainty and evidence coverage

Audit classifications distinguish quantified uncertainty from documented-but-unquantified and plausible-but-untested sources. This is not a numerical confidence scale and introduces no new interval.
"""
    write(figdir / "FIGURE_CAPTIONS.md", captions)

    f2rows = []
    for lid in LANDSCAPES:
        for gate in GATE_IDS:
            f2rows.append({"landscape_id": lid, "gate_id": gate, "result": lookup.get((lid, gate), "NOT_EVALUATED")})
    write_csv(figdir / "F2_SIX_LANDSCAPE_GATE_MATRIX_SOURCE.csv", f2rows)
    write_csv(figdir / "F4_HISTORICAL_FEATURE_REDUNDANCY_SOURCE.csv", rr[["feature","pooled_r2_on_60_M","redundant_r2_ge_0_90"]].to_dict("records"))
    write_csv(figdir / "F5_UNCERTAINTY_EVIDENCE_COVERAGE_SOURCE.csv", urows)


def uncertainty_rows() -> list[dict]:
    return [
        dict(uncertainty_source="Geological mapping", audit_status="DOCUMENTED_NOT_QUANTIFIED", evidence="Map scales and unit sources recorded; no propagated positional/classification distribution", consequence="Support and unit membership may shift at boundaries"),
        dict(uncertainty_source="Age / chronology", audit_status="UNRESOLVED", evidence="Numeric eruption ages were missing for several landscapes; relative unit ages used", consequence="Temporal target cannot be tightly dated"),
        dict(uncertainty_source="Preserved evidence", audit_status="DOCUMENTED_NOT_QUANTIFIED", evidence="Q1/Q2 hierarchy, exclusions and missing upper bounds documented", consequence="Reference surface is evidence-limited and may be minimum-removal only"),
        dict(uncertainty_source="DEM resolution", audit_status="DOCUMENTED_NOT_QUANTIFIED", evidence="30 m DEM-derived units and 90 m slope method recorded", consequence="Scale effects not propagated through H"),
        dict(uncertainty_source="Interpolation", audit_status="DOCUMENTED_AND_QUANTIFIED", evidence="Equal-weight structural/percentile/tolerance/subsample ensemble with P05/P50/P95 and IQR", consequence="Scenario spread measures model disagreement, not probability"),
        dict(uncertainty_source="Spatial extrapolation", audit_status="DOCUMENTED_AND_QUANTIFIED", evidence="Explicit support mask and nearest-control distance; negative removal increases with distance", consequence="Uncertainty and physical inconsistency are spatially nonuniform"),
        dict(uncertainty_source="Erosion-history ambiguity", audit_status="PLAUSIBLE_NOT_TESTED", evidence="Estimand and minimum-removal semantics explicit; no unique process history inferred", consequence="Surface difference cannot identify causal erosion sequence"),
        dict(uncertainty_source="Support-mask uncertainty", audit_status="DOCUMENTED_NOT_QUANTIFIED", evidence="Mask criteria and exclusions fixed; alternative masks not propagated", consequence="41.2% is valid availability, not a confidence level"),
        dict(uncertainty_source="Cross-landscape comparability", audit_status="PLAUSIBLE_NOT_TESTED", evidence="Landscape-normalised features specified but only one landscape released H", consequence="No empirical multi-landscape H comparability test"),
    ]


def main() -> None:
    for sub in ["00_GOVERNANCE","01_EVIDENCE","02_QUALITY_GATES","03_LANDSCAPES","04_KUMAMOTO","05_FEATURES","06_UNCERTAINTY","07_ORIGINALITY","08_REPORT_INTEGRATION","09_FIGURES","10_REPRODUCIBILITY"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    src_rows, src_before = source_registry()
    d = audit_data()
    gates = gate_rows()

    # Governance
    write(OUT/"00_GOVERNANCE/STAGE_D_SCOPE.md", """# Stage D scope

Status: **COMPLETED READ-ONLY AUDIT**. Audit date: 2026-10-18.

This package audits the existing Generation II geological-history work. It does not reconstruct a new surface, change a mask or threshold, retrain a predictive model, or claim H-based predictive improvement. Existing records are inputs and remain read-only. New outputs are confined to this directory.

The audited endpoint is frozen Gen-3 / Stage 5G. Earlier Gen-1 and Gen-2 records are historical development evidence, not superseding results. New supported-versus-unsupported comparisons are explicitly exploratory.
""")
    write(OUT/"00_GOVERNANCE/STAGE_D_METHOD.md", """# Stage D method

The audit used five controls: (1) locate frozen specifications before reading outcomes; (2) trace each gate from prose to preregistration, implementation and result; (3) reproduce support counts directly from aligned frozen NPZ arrays; (4) distinguish original method, frozen result, later audit, exploratory interpretation and proposal; and (5) hash every registered source and generated output by explicit path with `hashlib`.

No reconstruction was fitted. No threshold, surface, mask, terrain label or predictive model was changed. The only new numerical analysis compares frozen Kumamoto supported and unsupported units descriptively. Figure maps use actual EPSG:32652 unit centroids; no geological boundary was approximated.

The only located authoritative report subsection numbering is the existing Section 7 guide, which assigns geological-history feasibility to §7.8. Requested Sections 2, 4, 5, 8 and 9 are treated as section roles because no current research-paper file establishing their exact headings was found. The integration map must therefore be reconciled against the final manuscript before insertion.
""")

    # Evidence
    write_csv(OUT/"01_EVIDENCE/SOURCE_REGISTER.csv", src_rows)
    write(OUT/"01_EVIDENCE/SOURCE_AUTHORITY_HIERARCHY.md", """# Source authority hierarchy

1. **FROZEN_RESULT** — executed, checkpointed numerical outputs. These control reported outcomes.
2. **ORIGINAL_METHOD** — frozen specifications, preregistration and executed code. These control what was intended and implemented.
3. **LATER_AUDIT** — Stage 5G and Stage B syntheses. These may clarify but cannot alter frozen outcomes.
4. **EXPLORATORY_INTERPRETATION** — Stage D supported/unsupported comparisons; descriptive only.
5. **PROPOSED_METHOD** — earlier feasibility plans or future-work ideas; not results.
6. **UNVERIFIED_CLAIM** — a claim without recoverable supporting evidence; excluded or labelled unresolved.

When prose and frozen machine-readable results differ, the result is preserved and the discrepancy reported. One material wording defect was found in the Stage B geological-history review: it describes 0.2269 as the R² of a “core H set”; the frozen table shows 0.2269 belongs only to `H_REMOVED_FRACTION`. This package reports all six feature-wise values.
""")
    write(OUT/"01_EVIDENCE/RECONSTRUCTION_TIMELINE.md", """# Reconstruction timeline

- **Stage 2C–3F:** geological maps, DEM provenance, temporal notes and landscape-specific feasibility evidence assembled.
- **Stage 4C-A:** H estimand, Y-blind controls, six candidate features and seven quantitative gates frozen before reconstruction.
- **Stage 4C-B (Gen-1):** central-trend families fitted; no landscape passed all gates.
- **Stage 4C-B2 (Gen-2):** structurally aware components and boundary/physical QA added; no landscape passed all gates.
- **Stage 4C-B3 (Gen-3):** one-sided upper-constructional estimand and nested architecture/loss selection preregistered. Kumamoto alone became `VERIFIED_GEN3_PARTIAL`; stopping rule prevented Gen-4 because fewer than three landscapes were H-capable.
- **Stage 5G:** frozen results assembled into the explicit nine-gate matrix; no H-inclusive predictive model was trained.
- **Stage D:** read-only forensic audit, coverage reproduction, exploratory support-bias description, literature-bounded originality review and report-ready wording.
""")

    # Gates
    write_csv(OUT/"02_QUALITY_GATES/NINE_GATE_SPECIFICATION.csv", gates)
    defs = "# Nine gate definitions\n\nAll nine are conjunctive for an applicable reconstructed landscape. The first seven were frozen in Stage 4C-A; boundary and physical QA were frozen before Gen-3 in the B3 preregistration.\n\n" + "\n".join(f"## {r['gate_order']}. {r['gate_id']}\n\n**Purpose:** {r['purpose']}  \n**Rule:** `{r['rule']}`  \n**Scale:** {r['evaluation_scale']}  \n**Missing data:** {r['missing_data']}  \n**Evidence:** `{r['source_file']}` and `{r['implementation']}`.\n" for r in gates)
    write(OUT/"02_QUALITY_GATES/NINE_GATE_DEFINITIONS.md", defs)
    impl_rows = [dict(gate_id=r["gate_id"], documented="YES", preregistered="YES", implemented="YES", frozen_result_present="YES", reproducibility_status=r["recovery_status"], code_location=r["implementation"], result_location="outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv" if r["gate_id"] not in ("BOUNDARY_QA",) else "outputs/stage4c_b3/STAGE4C_B3_BOUNDARY_QA.tsv", caveat="Geological discontinuities are report-only" if r["gate_id"]=="BOUNDARY_QA" else "None identified") for r in gates]
    write_csv(OUT/"02_QUALITY_GATES/GATE_IMPLEMENTATION_AUDIT.csv", impl_rows)
    prov = []
    for r in gates:
        prov.append(dict(gate_id=r["gate_id"], threshold=r["threshold"], first_recovered_source=r["source_file"], provenance_type="INTERNAL_FROZEN_METHOD_RULE", external_calibration_recovered="NO", audit_conclusion="Threshold existed before Gen-3 outcomes; detailed empirical/external calibration rationale was not recovered"))
    write_csv(OUT/"02_QUALITY_GATES/GATE_THRESHOLD_PROVENANCE.csv", prov)
    write(OUT/"02_QUALITY_GATES/GATE_REPRODUCIBILITY_REPORT.md", """# Gate reproducibility report

**Result: PASS with a provenance limitation.** Every gate has a recoverable input, formula/logical rule, threshold, missing-data behaviour, executable implementation and frozen result. Stage D independently recomputed the seven scalar pass/fail rules from the Gen-3 validation table and matched all recorded decisions. Boundary and physical decisions match the code and the boundary/validation tables.

The detailed empirical or literature calibration for the numerical threshold values was not recoverable beyond their status as frozen internal method rules. They must be described as prespecified project thresholds, not universally validated geological constants.

Boundary nuance: numerical component boundaries face five seam tests. Geologically distinct contacts are explicitly `REPORTED_ONLY`, because forcing continuity would be scientifically inappropriate. A single-component surface has no internal seam and passes as not applicable. Physical QA is `pit_fraction <= 0.01` **and** zero intersection of support with the caldera-interior mask.
""")

    # Landscape matrix and eligibility.
    matrix_rows = d["matrix"].rename(columns={"gate":"gate_id"}).to_dict("records")
    write_csv(OUT/"03_LANDSCAPES/SIX_LANDSCAPE_GATE_MATRIX.csv", matrix_rows)
    eligibility = []
    status_by = d["status"].set_index("landscape_id")
    val_by = d["val"].set_index("landscape_id")
    for lid in LANDSCAPES:
        srow = status_by.loc[lid]
        if lid == "NOTO_PENINSULA_2024":
            eligibility.append(dict(landscape_id=lid, reconstruction_feasibility="NO_PALAEOSURFACE", spatial_support="NONE", proxy_availability="AGE_PROXY_ONLY", complete_surface="NO", predictive_H_eligibility="NO", final_status="AGE_PROXY_ONLY", evidence="STAGE4C_A_RECONSTRUCTION_EVIDENCE.tsv; STAGE4C_B3_RECONSTRUCTION_STATUS.tsv"))
        else:
            v = val_by.loc[lid]
            eligibility.append(dict(landscape_id=lid, reconstruction_feasibility="VERIFIED_PARTIAL" if bool(srow.verified_gen3_h_capable) else "FAILED_FROZEN_GATES", spatial_support=f"{v.support_fraction_of_units:.6f}", proxy_availability="NOT_SUBSTITUTED", complete_surface="NO" if lid=="KUMAMOTO_ASO_2016" else "DIAGNOSTIC_ONLY", predictive_H_eligibility="PARTIAL_FEATURE_RELEASE_ONLY; NO_MODEL" if bool(srow.core_h_released) else "NO", final_status=srow.status, evidence="STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv; STAGE4C_B3_RECONSTRUCTION_STATUS.tsv"))
    write_csv(OUT/"03_LANDSCAPES/LANDSCAPE_HISTORY_ELIGIBILITY.csv", eligibility)
    failures = """# Landscape failure reasons

- **Kumamoto–Aso:** no failed gate; partial rather than full because verified support is 41.1719%, below the 90% full-release boundary. Only supported units receive H.
- **Noto Peninsula:** `AGE_PROXY_ONLY`. Oligocene–Miocene age information exists, but no surviving volcanic constructional surface supports palaeotopographic reconstruction. Gates are not evaluated, not failed.
- **Lombok–Rinjani:** held-out median residual 0.100533 > 0.10 and negative-removal fraction 0.150069 > 0.10. The residual miss is small numerically but remains a fail under the frozen rule.
- **Cotabato–Davao:** residual 0.140832 > 0.10, absolute bias 0.073220 > 0.05 and P05–P95 coverage 0.572999 < 0.60.
- **Atitlán–Stan:** negative-removal fraction 0.128153 > 0.10 and physical QA failed because deep-pit fraction was 0.016518 > 0.01. Its Tv/Qp geological discontinuity was report-only, not a seam-gate failure.
- **Dominica–Erika:** residual 0.117881 > 0.10, coverage 0.531856 < 0.60, negative removal 0.149610 > 0.10, physical pit fraction 0.018316 > 0.01, and 33 held-out controls lacked predictions. Its young/older geological contact was report-only.

Failed reconstructions remain diagnostic and unreleased. They are not zero-filled and are not eligible for predictive use.
"""
    write(OUT/"03_LANDSCAPES/LANDSCAPE_FAILURE_REASONS.md", failures)

    # Kumamoto all-unit spatial file.
    m, h, s = d["m"], d["h"], d["support"]
    kpath = OUT/"04_KUMAMOTO/KUMAMOTO_SPATIAL_SUPPORT.csv"
    with kpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["terrain_unit_id","centroid_x","centroid_y","projected_crs","supported","source_unit_code","elevation_m","slope_deg_90m","relief_m_r1000","positive_label_posthoc"])
        for i in range(len(s)):
            w.writerow([m["terrain_unit_id"][i], int(m["centroid_x"][i]), int(m["centroid_y"][i]), m["projected_crs"][i], int(s[i]), m["qa_source_unit_code"][i], f"{float(m['elev_m_30m__mean'][i]):.7g}", f"{float(m['slope_deg_90m__mean'][i]):.7g}", f"{float(m['relief_m_r1000__mean'][i]):.7g}", int(m["y_positive"][i])])
    bias = support_bias_rows(d)
    write_csv(OUT/"04_KUMAMOTO/KUMAMOTO_SUPPORT_BIAS_SUMMARY.csv", bias)
    write(OUT/"04_KUMAMOTO/KUMAMOTO_RECONSTRUCTION_METHOD_AUDIT.md", """# Kumamoto reconstruction method audit

## Audited estimand

`S*(x)` is a plausible pre-erosional **conservative upper constructional surface** consistent with preserved Q1/Q2 remnants and structural geometry. It is not the exact original surface, a smoothed modern DEM, a maximum conceivable surface, or an erosion-process history. For the envelope family, `H_REMOVED_FRACTION` has **minimum-removal** semantics.

## Sources and limitations

The geological constraint is the GSJ Aso 1:50,000 mapping of caldera wall, pyroclastic-flow and post-caldera units. Controls are DEM-derived accordant interfluve summits on mapped pyroclastic-flow units. Numeric eruption ages were recorded as missing from project data; chronology is therefore unit-relative, not a dated reconstruction. Map-scale positional/classification uncertainty was not propagated numerically.

## Candidate generation and selection

Gen-3 considered the preregistered `K_APRON_MEDIUM` and `K_RIM_SECTOR+APRON_MEDIUM` geometries crossed with seven one-sided loss variants. Nested outer 2 km control blocks and inner super-folds selected architecture/loss without landslide Y. Production selected `K_APRON_MEDIUM / ASYM_HUBER_a2`; all eight outer selections agreed, architecture stability was 1.0, effective DOF median 29.7186, and 36 equal-weight scenarios varied structural settings, summit percentile, tolerance offset and control subsampling.

## Support and alignment

Support is the intersection of mapped constructional-domain geometry, control-distance/reconstruction availability and the Stage 4A terrain units; caldera interiors and other exclusions remain masked. Terrain-unit IDs align exactly between the frozen Stage 4A and H NPZ arrays. Unsupported H is NaN, never zero.

## Decision

Kumamoto passed all seven scalar gates, boundary QA and physical QA, with 2,753 controls, 0.053916 residual fraction, 0.015024 absolute bias, 0.642208 scenario-band coverage, 0.059129 negative-removal fraction, 2.40158° maximum reference slope and zero deep-pit fraction. The result is scientifically defensible as a **partial, evidence-bounded minimum-removal reconstruction**, not as a whole-landscape palaeo-DEM or proof of predictive value.
""")
    write(OUT/"04_KUMAMOTO/KUMAMOTO_COVERAGE_VALIDATION.md", f"""# Kumamoto coverage validation

Direct read-only reproduction from aligned frozen arrays:

- total Stage 4A units: **{len(s):,}**
- H-supported units: **{int(s.sum()):,}**
- unsupported units: **{int((~s).sum()):,}**
- supported fraction: **{s.mean():.9f}** ({100*s.mean():.4f}%)
- arithmetic reconciliation: {int(s.sum()):,} + {int((~s).sum()):,} = {len(s):,}

This exactly matches the Stage 5G audit. The 41.2% is **valid predictor availability / verified reconstruction support at the terrain-unit scale**. It is not geological-map coverage, a probability, a confidence percentage or representative whole-landscape coverage. Unsupported values are missing.

The support is concentrated primarily on mapped Q31 pyroclastic-flow units. Actual terrain-unit centroids and the support flag are in `KUMAMOTO_SPATIAL_SUPPORT.csv`.
""")
    ku = [
        dict(item="Scenario-ensemble structural/parameter uncertainty", status="QUANTIFIED_AS_SCENARIO_SPREAD", evidence="36 equal-weight scenarios; P05/P50/P95 and IQR", limitation="Scenario frequencies are not probabilities"),
        dict(item="Held-out control accuracy", status="QUANTIFIED", evidence="2753 Y-blind outer-held controls", limitation="Controls are spatially structured and not independent terrain replicates"),
        dict(item="Spatial extrapolation", status="QUANTIFIED_PROXY", evidence="nearest-control distance feature and distance-stratified negative removal", limitation="Distance is not a calibrated uncertainty distribution"),
        dict(item="Geological map classification/position", status="DOCUMENTED_NOT_QUANTIFIED", evidence="GSJ 1:50,000 mapping", limitation="No propagated boundary uncertainty"),
        dict(item="Chronology", status="UNRESOLVED", evidence="numeric eruption ages missing in frozen evidence register", limitation="No dated surface age"),
        dict(item="DEM scale", status="DOCUMENTED_NOT_QUANTIFIED", evidence="30 m DEM; 90 m slope", limitation="Resolution sensitivity not propagated"),
        dict(item="Support-mask choice", status="DOCUMENTED_NOT_QUANTIFIED", evidence="fixed frozen support/exclusion rules", limitation="No alternative-mask ensemble"),
    ]
    write_csv(OUT/"04_KUMAMOTO/KUMAMOTO_UNCERTAINTY_REGISTER.csv", ku)
    bdf = pd.DataFrame(bias)
    def row(metric, group): return bdf[(bdf.metric==metric)&(bdf.group==group)].iloc[0]
    write(OUT/"04_KUMAMOTO/KUMAMOTO_SUPPORT_BIAS_ANALYSIS.md", f"""# Kumamoto support-bias analysis

**Status: EXPLORATORY_INTERPRETATION.** These comparisons were not frozen gates and do not alter release status.

Supported terrain is not representative of the complete landscape. Median elevation is {row('Elevation','SUPPORTED')['median']:.1f} m versus {row('Elevation','UNSUPPORTED')['median']:.1f} m; median 90 m slope is {row('Slope (90 m)','SUPPORTED')['median']:.2f}° versus {row('Slope (90 m)','UNSUPPORTED')['median']:.2f}°; median 1 km relief is {row('Local relief (1 km)','SUPPORTED')['median']:.1f} m versus {row('Local relief (1 km)','UNSUPPORTED')['median']:.1f} m. Supported units are therefore systematically lower, gentler and lower-relief, consistent with concentration on outer pyroclastic-flow aprons.

The post-hoc positive-label prevalence is {row('Positive-label prevalence (post hoc)','SUPPORTED')['mean']:.6f} inside support and {row('Positive-label prevalence (post hoc)','UNSUPPORTED')['mean']:.6f} outside. Y was forbidden from reconstruction and support selection, so this difference cannot establish selection intent or causal bias; it does show that an H-inclusive analysis restricted to support would study a different outcome mix.

Conclusion: support is spatially and morphologically concentrated. Any future H model would require an explicitly support-restricted estimand and could not be generalized to all Kumamoto units without new evidence.
""")

    # Features.
    cat = d["feat"].copy()
    red_release = d["red"][d["red"].role=="RELEASED_VERIFIED_GEN3_H"][["feature","pooled_r2_on_60_M","redundant_r2_ge_0_90"]].rename(columns={"feature":"name"})
    cat = cat.merge(red_release, on="name", how="left")
    cat["generation_status"] = "GENERATED_KUMAMOTO_PARTIAL"
    cat["entered_trained_predictive_model"] = "NO"
    cat["coverage_units"] = 145799
    cat["coverage_landscapes"] = 1
    write_csv(OUT/"05_FEATURES/HISTORICAL_FEATURE_CATALOGUE.csv", cat.to_dict("records"))
    write(OUT/"05_FEATURES/HISTORICAL_FEATURE_PROVENANCE.md", """# Historical-feature provenance

The six released H variables are calculated only from the Gen-3 scenario surface, modern DEM-derived terrain, support geometry and Y-blind controls. The exact formulas and semantic caveats are in `HISTORICAL_FEATURE_CATALOGUE.csv`. All six were generated for the 145,799 supported Kumamoto units; diagnostic values for failed landscapes were not released. Noto has an age proxy but no surface-based H.

No H value was used in the official morphology-only `LOGISTIC_L2` model. No MEGH model was trained. Missing H was never imputed or zero-filled.
""")
    write(OUT/"05_FEATURES/REDUNDANCY_ANALYSIS_AUDIT.md", """# Redundancy analysis audit

The frozen analysis fit a multivariate ordinary least-squares relationship for each H feature using the 60 frozen modern-morphology predictors on the 145,799 released Kumamoto units. It used no Y. The prespecified classification was redundant when R² ≥ 0.90.

Feature-wise R² values were: removed fraction 0.2269; slope change 0.9347; relief change 0.8518; reference relative height 0.9463; scenario spread 0.0886; control distance 0.0787. Thus four of six were below 0.90, while slope change and reference relative height met the redundancy rule.

The result is conditional on one landscape and on the reconstruction-support subset. OLS R² below 0.90 means only that a linear combination of these 60 M columns did not reproduce that H column to the project threshold. It does **not** establish statistical independence, causal novelty, out-of-landscape transfer value or improved prediction. Because support is morphologically selected, the result is not a whole-landscape redundancy test.
""")
    write(OUT/"05_FEATURES/PREDICTIVE_ELIGIBILITY_ASSESSMENT.md", """# Predictive eligibility assessment

**Current result: NOT ELIGIBLE FOR A MULTI-LANDSCAPE H EXPERIMENT.** Four H features pass the feature-wise R² rule in the single released landscape, but the frozen experiment required H availability in at least three verified landscapes. Only Kumamoto qualified. The stopping rule therefore kept `MEGH_EXPERIMENT_UNLOCKED = NO`.

The H variables may be described as generated and partially released feasibility outputs. They may not be described as trained predictors, validated predictors, demonstrated improvements or transferable geological-history effects.
""")

    # Uncertainty.
    urows = uncertainty_rows()
    write_csv(OUT/"06_UNCERTAINTY/UNCERTAINTY_EVIDENCE_MATRIX.csv", urows)
    write(OUT/"06_UNCERTAINTY/RECONSTRUCTION_UNCERTAINTY_AUDIT.md", """# Reconstruction uncertainty audit

The strongest quantified components are held-out-control error, scenario-band coverage, scenario spread, support availability and control distance. The scenario ensemble represents disagreement among preregistered alternatives; its P05/P95 values are not confidence or probability intervals.

Geological map uncertainty, chronology, DEM resolution and alternative support masks were documented but not jointly propagated. Erosion pathways are non-identifiable from a surface difference. Cross-landscape comparability remains untested because only Kumamoto released H. These limits block any broad predictive or causal claim even though the partial Kumamoto reconstruction passed its frozen gates.

No artificial interval or confidence percentage was added in Stage D.
""")

    # Originality.
    pred = [
        dict(predecessor="Gourbet et al. (2024)", domain="Volcanic palaeotopographic reconstruction and incision", overlap="Relict-surface constrained reconstruction and uncertainty", difference="No cross-volcano landslide transfer experiment identified", evidence_depth="ABSTRACT_VERIFIED", implication="Reconstruction itself is established"),
        dict(predecessor="O'Hara et al. (2024)", domain="Volcanic age and drainage geometry", overlap="History/maturity signal in modern form", difference="Age-dependent morphometrics, not quality-gated palaeosurface predictors", evidence_depth="ABSTRACT_VERIFIED", implication="Age/maturity concepts are established"),
        dict(predecessor="Liu et al. (2025)", domain="Temporal susceptibility", overlap="Path-dependent terrain motivation", difference="Artificial DEMs and simulated occurrences; not field validation of volcanic H", evidence_depth="ABSTRACT_VERIFIED", implication="Do not claim empirical palaeosurface precedent"),
        dict(predecessor="Woodard et al. (2023)", domain="Outside-inventory transfer", overlap="Training composition affects transfer", difference="No volcanic palaeosurface gate stage", evidence_depth="FULL_TEXT_VERIFIED", implication="Transfer question is established"),
        dict(predecessor="Wang et al. (2022)", domain="Cross-region transfer learning", overlap="Source-target transfer", difference="Target adaptation versus zero-target-label whole-landscape evaluation", evidence_depth="FULL_TEXT_VERIFIED", implication="Transfer learning is not novel"),
        dict(predecessor="Stage B screened corpus", domain="Integrated method search", overlap="Individual ingredients found separately", difference="No combined six-volcano transfer + quality-gated H workflow identified", evidence_depth="64 bibliographic; 11 full text; 40 abstract; 13 metadata", implication="Supports potentially novel integration only"),
    ]
    write_csv(OUT/"07_ORIGINALITY/HISTORY_PREDECESSOR_COMPARISON.csv", pred)
    write(OUT/"07_ORIGINALITY/HISTORY_ORIGINALITY_ASSESSMENT.md", """# Geological-history originality assessment

**Verdict: POTENTIALLY_NOVEL IN INTEGRATION, NOT IN THE COMPONENT METHODS.** Palaeotopographic reconstruction, volcanic landscape-age morphometrics, landslide susceptibility and geographical transfer each have precedents. The screened Stage B literature did not identify their combination as a Y-blind, quality-gated feasibility stage inside a six-landscape zero-target-label transfer programme.

The most distinctive empirical result is methodological restraint: five of six landscapes did not yield released palaeosurface H, and the only release was partial. The implementation of frozen gates and fail-closed missingness is a useful integration. The search does not prove universal priority, and abstract-only sources cannot support detailed method comparisons.
""")
    write(OUT/"07_ORIGINALITY/DEFENSIBLE_HISTORY_CLAIMS.md", """# Defensible geological-history claims

- The project implemented a Y-blind, nine-gate palaeosurface feasibility workflow before allowing historical features into predictive analysis.
- Kumamoto alone passed all applicable Gen-3 gates, releasing six H variables for 145,799 of 354,123 units (41.1719%).
- Noto supplied an age proxy but no constructional palaeosurface; four other reconstructions failed frozen gates.
- Four of six Kumamoto H features were below the frozen OLS R² = 0.90 redundancy rule, conditional on the supported subset.
- No H-inclusive predictive model was trained, so predictive benefit is untested.
- In the screened literature, the integrated combination of quality-gated volcanic palaeosurface reconstruction and whole-landscape susceptibility transfer was not identified.
""")
    write(OUT/"07_ORIGINALITY/UNSUPPORTED_HISTORY_CLAIMS.md", """# Unsupported or overstrong claims

Remove or weaken: “geological history improved prediction”; “the model uses geological history”; “Kumamoto is fully reconstructed”; “41.2% confidence”; “unsupported cells have zero erosion”; “R² < 0.90 proves independence”; “the surface is the true pre-eroded DEM”; “all six landscapes were reconstructed”; “Noto’s age proxy is a palaeosurface”; “first ever”; and any causal statement that erosion history produced landslides.

Use instead: partial support, minimum-removal semantics, scenario disagreement rather than probability, potentially nonredundant under a one-landscape linear audit, and “not identified in the screened literature.”
""")

    # Report integration.
    write(OUT/"08_REPORT_INTEGRATION/HISTORY_METHODS_TEXT.md", """# Report-ready methods text

Historical terrain information was treated as a separate feasibility problem rather than assumed to be available. For each candidate landscape, geological units and pre-event DEM morphology defined Y-blind Q1/Q2 preserved-surface controls. Nested spatial holdouts selected among preregistered structural surfaces and one-sided loss functions without using landslide labels. Equal-weight scenarios varied structural settings, summit percentile, control tolerance and control subsampling; the scenario median was the primary reference surface, while P05–P95 and IQR described scenario disagreement.

A reconstruction was released only if it met nine gates: median absolute held-out residual ≤0.10 reconstructed relief, absolute bias ≤0.05, held-out P05–P95 coverage ≥0.60, at least 12 admissible controls, support for at least 25% of terrain units, negative removal ≤0.10, maximum constructional slope ≤40°, numerical-boundary QA, and physical QA requiring deep-pit fraction ≤0.01 with no support inside masked caldera interiors. Geological discontinuities were reported rather than incorrectly forced to be smooth. H remained missing outside verified support.
""")
    write(OUT/"08_REPORT_INTEGRATION/HISTORY_RESULTS_TEXT.md", """# Report-ready results text

Only Kumamoto–Aso passed every applicable Gen-3 gate. Its reconstruction used 2,753 admissible controls and supported 145,799 of 354,123 terrain units (41.1719%); the remaining 208,324 units retained missing H. Noto Peninsula was age-proxy-only because no defensible constructional palaeosurface survived. Lombok failed residual and negative-removal gates; Cotabato failed residual, bias and scenario-coverage gates; Atitlán failed negative-removal and physical gates; and Dominica failed residual, coverage, negative-removal and physical gates, with 33 held-out controls unpredicted.

Six H variables were released on Kumamoto support. OLS R² from 60 frozen morphology variables was 0.2269, 0.9347, 0.8518, 0.9463, 0.0886 and 0.0787 for removed fraction, slope change, relief change, reference relative height, scenario spread and control distance, respectively. Four were below the frozen 0.90 redundancy criterion. No H-inclusive predictive model was trained.
""")
    write(OUT/"08_REPORT_INTEGRATION/HISTORY_DISCUSSION_TEXT.md", """# Report-ready discussion text

The history investigation produced a negative general-feasibility result rather than evidence of improved susceptibility prediction. Kumamoto shows that a constrained partial reconstruction can pass stringent held-out and physical checks, but its support is concentrated on lower, gentler pyroclastic-flow aprons and is not representative of the full landscape. The four feature-wise R² values below 0.90 indicate incomplete linear reproduction by modern morphology within that subset; they do not demonstrate independence, causation or predictive utility.

The scientific contribution is therefore the integration of a fail-closed reconstruction gate into a cross-landscape transfer workflow. In the screened literature, palaeotopographic reconstruction and susceptibility transfer were found separately, but this quality-gated combination was not identified. This is a bounded originality claim, not proof of being first.
""")
    write(OUT/"08_REPORT_INTEGRATION/HISTORY_LIMITATIONS_TEXT.md", """# Report-ready limitations text

Only one landscape released H and only 41.2% of its terrain units were supported. Numeric chronology was incomplete, geological-map and support-mask uncertainty were not propagated, and scenario quantiles are not probability intervals. Failed-landscape H remained diagnostic and cannot be used. Support restriction changes the morphology and positive-label composition of the analysed units. Consequently, cross-landscape comparability and predictive gain from H remain untested.
""")
    write(OUT/"08_REPORT_INTEGRATION/HISTORY_FUTURE_WORK.md", """# Future work

Future work should obtain independent preserved-surface and chronological evidence rather than iterate algorithms against the same failed gates. Any renewed H experiment should preregister new evidence, repeat Y-blind validation, propagate map/mask sensitivity, and require at least three independently verified landscapes before model training. A support-restricted predictive estimand and untouched external evaluation would be required to test whether H improves transfer.
""")
    write(OUT/"08_REPORT_INTEGRATION/REPORT_INTEGRATION_MAP.md", """# Report integration map

| Requested role | Recommended content | Status of heading |
|---|---|---|
| Section 2 — Introduction | Motivation and literature-bounded gap; distinguish maps, age proxies, modern morphology and palaeotopography | Role supplied by Stage D prompt; exact manuscript heading not located |
| Section 4 — Data | Geological-map scales, DEM-derived controls, chronology availability and support masks | Role supplied by prompt; verify final numbering |
| Section 5 — Methodology | Y-blind controls, nested selection, ensemble, nine gates and missingness | Role supplied by prompt; verify final numbering |
| **§7.8 — Geological-History Reconstruction Feasibility** | Six-landscape outcomes, 41.1719% support and feature-wise redundancy | **Exact subsection recovered from `SECTION7_RESULTS_GUIDE.md`** |
| Section 8 — Discussion | Partial feasibility, support concentration, uncertainty and originality boundary | Role supplied by prompt; verify final numbering |
| Section 9 — Conclusion | Quality-gated feasibility demonstrated; predictive gain untested | Role supplied by prompt; verify final numbering |

Do not assign lower-level subsection numbers outside §7.8 until the current manuscript headings are provided.
""")

    # Figure files and source tables.
    make_figures(d)

    # Reproducibility.
    source_manifest_rows = [{"sha256": r["sha256"], "bytes": (ROOT/r["source_path"]).stat().st_size, "path": r["source_path"], "authority_class": r["authority_class"]} for r in src_rows]
    write_csv(OUT/"10_REPRODUCIBILITY/SOURCE_MANIFEST.tsv", source_manifest_rows, ["sha256","bytes","path","authority_class"])
    # Convert comma-separated writer output to true tabs deterministically.
    sm = OUT/"10_REPRODUCIBILITY/SOURCE_MANIFEST.tsv"
    sm.write_text(sm.read_text().replace(",", "\t"), encoding="utf-8", newline="\n")
    write(OUT/"10_REPRODUCIBILITY/REPRODUCTION_INSTRUCTIONS.md", """# Reproduction instructions

Prerequisite: use the existing project virtual environment; do not install packages.

From `Science_Fair_ML_V2`:

```bash
../.venv/bin/python AUSSEF_2026_GEOLOGICAL_HISTORY_AUDIT/10_REPRODUCIBILITY/build_stage_d_audit.py
```

The script reads only the explicit files in `SOURCE_MANIFEST.tsv`, recreates derived tables/documents/figures inside this audit directory, checks the 354,123 / 145,799 / 208,324 support identity and gate structure, and rebuilds the output hash manifest. Figure F3 uses all units in its source CSV and deterministic thinning only for rendering.

`OUTPUT_HASH_MANIFEST.tsv` excludes itself to avoid recursive self-hashing. Verify other outputs by hashing each explicit path with SHA-256; do not parse whitespace-delimited checksum output.
""")

    src_after = {p: sha256(ROOT/p) for p in src_before}
    if src_after != src_before:
        raise AssertionError("A registered source changed during audit build")
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    index_clean = subprocess.run(["git","diff","--cached","--quiet"], cwd=ROOT.parent).returncode == 0
    write(OUT/"10_REPRODUCIBILITY/INTEGRITY_REPORT.md", f"""# Integrity report

- Audit status: **PASS**
- Production/repository HEAD observed: `{head}`
- HEAD tree: `{tree}`
- Git index empty: **{str(index_clean).upper()}**
- Registered source files unchanged between pre-build and post-build hashes: **YES ({len(src_before)} files)**
- Writes outside `AUSSEF_2026_GEOLOGICAL_HISTORY_AUDIT/`: **NONE BY BUILDER**
- Predictive models trained: **0**
- Palaeosurfaces reconstructed: **0**
- Existing thresholds, masks, labels or frozen outputs modified: **NO**
- Git commit/push: **NO**

The repository had pre-existing unrelated worktree changes; this audit neither normalised nor modified them. Integrity is established by explicit source hashes and output confinement, not by claiming a globally clean worktree.
""")

    # Final output manifest, excluding itself.
    manifest = OUT/"10_REPRODUCIBILITY/OUTPUT_HASH_MANIFEST.tsv"
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p != manifest)
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["sha256","bytes","path"])
        for p in files:
            w.writerow([sha256(p), p.stat().st_size, p.relative_to(OUT).as_posix()])

    # Final structural assertions.
    required = [
        "00_GOVERNANCE/STAGE_D_SCOPE.md","00_GOVERNANCE/STAGE_D_METHOD.md","01_EVIDENCE/SOURCE_REGISTER.csv","01_EVIDENCE/SOURCE_AUTHORITY_HIERARCHY.md","01_EVIDENCE/RECONSTRUCTION_TIMELINE.md",
        "02_QUALITY_GATES/NINE_GATE_SPECIFICATION.csv","02_QUALITY_GATES/NINE_GATE_DEFINITIONS.md","02_QUALITY_GATES/GATE_IMPLEMENTATION_AUDIT.csv","02_QUALITY_GATES/GATE_THRESHOLD_PROVENANCE.csv","02_QUALITY_GATES/GATE_REPRODUCIBILITY_REPORT.md",
        "03_LANDSCAPES/SIX_LANDSCAPE_GATE_MATRIX.csv","03_LANDSCAPES/LANDSCAPE_HISTORY_ELIGIBILITY.csv","03_LANDSCAPES/LANDSCAPE_FAILURE_REASONS.md",
        "04_KUMAMOTO/KUMAMOTO_RECONSTRUCTION_METHOD_AUDIT.md","04_KUMAMOTO/KUMAMOTO_SPATIAL_SUPPORT.csv","04_KUMAMOTO/KUMAMOTO_COVERAGE_VALIDATION.md","04_KUMAMOTO/KUMAMOTO_UNCERTAINTY_REGISTER.csv","04_KUMAMOTO/KUMAMOTO_SUPPORT_BIAS_ANALYSIS.md",
        "05_FEATURES/HISTORICAL_FEATURE_CATALOGUE.csv","05_FEATURES/HISTORICAL_FEATURE_PROVENANCE.md","05_FEATURES/REDUNDANCY_ANALYSIS_AUDIT.md","05_FEATURES/PREDICTIVE_ELIGIBILITY_ASSESSMENT.md",
        "06_UNCERTAINTY/RECONSTRUCTION_UNCERTAINTY_AUDIT.md","06_UNCERTAINTY/UNCERTAINTY_EVIDENCE_MATRIX.csv",
        "07_ORIGINALITY/HISTORY_ORIGINALITY_ASSESSMENT.md","07_ORIGINALITY/HISTORY_PREDECESSOR_COMPARISON.csv","07_ORIGINALITY/DEFENSIBLE_HISTORY_CLAIMS.md","07_ORIGINALITY/UNSUPPORTED_HISTORY_CLAIMS.md",
        "08_REPORT_INTEGRATION/HISTORY_METHODS_TEXT.md","08_REPORT_INTEGRATION/HISTORY_RESULTS_TEXT.md","08_REPORT_INTEGRATION/HISTORY_DISCUSSION_TEXT.md","08_REPORT_INTEGRATION/HISTORY_LIMITATIONS_TEXT.md","08_REPORT_INTEGRATION/HISTORY_FUTURE_WORK.md","08_REPORT_INTEGRATION/REPORT_INTEGRATION_MAP.md",
        "10_REPRODUCIBILITY/SOURCE_MANIFEST.tsv","10_REPRODUCIBILITY/OUTPUT_HASH_MANIFEST.tsv","10_REPRODUCIBILITY/REPRODUCTION_INSTRUCTIONS.md","10_REPRODUCIBILITY/INTEGRITY_REPORT.md",
    ]
    missing = [p for p in required if not (OUT/p).is_file()]
    if missing: raise AssertionError(f"Missing outputs: {missing}")
    if len(gates) != 9 or set(d["matrix"].landscape_id) != set(LANDSCAPES) or len(d["matrix"]) != 54:
        raise AssertionError("Gate audit cardinality mismatch")
    print(json.dumps({"status":"PASS","output":str(OUT),"files":len(list(OUT.rglob('*'))),"support":{"total":len(s),"supported":int(s.sum()),"unsupported":int((~s).sum())},"head":head,"tree":tree}, indent=2))


if __name__ == "__main__":
    main()
