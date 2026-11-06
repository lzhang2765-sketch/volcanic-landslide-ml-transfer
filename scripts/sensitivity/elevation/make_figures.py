"""Figures for the elevation ablation experiment (reads 02_RESULTS only, plus the frozen Stage 4A NPZ for F4).

Static PNG/PDF for the report. Every figure has a table view in 02_RESULTS/*.csv.
Palette: validated reference categorical slots 1-3 (dataviz validate_palette.js, light mode: PASS; aqua <3:1 contrast,
so A2 also carries a hatch and every series is legend-labelled).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[2]
V2 = EXP.parent
RES, FIG = EXP / "02_RESULTS", EXP / "03_FIGURES"
L = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
SHORT = {k: k.split("_")[0].title().replace("Atitlan", "Atitlán") for k in L}
COL = {"B0": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a"}
HATCH = {"B0": "", "A1": "", "A2": "////"}
NAME = {"B0": "B0 original M (absolute elevation)", "A1": "A1 no absolute elevation", "A2": "A2 landscape-centred elevation"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(RES / "ELEVATION_ABLATION_FOLD_RESULTS.csv")
    d = pd.read_csv(RES / "PAIRED_PERFORMANCE_DIFFERENCES.csv")
    s = pd.read_csv(RES / "ELEVATION_ABLATION_SUMMARY.csv").set_index("config")
    r = pd.read_csv(RES / "ROBUSTNESS_COMPARISON.csv")

    # F1 landscape PR lift
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x, w = np.arange(len(L)), 0.26
    for i, c in enumerate(["B0", "A1", "A2"]):
        g = f[f.config == c].set_index("held_out").reindex(L)
        err = np.vstack([g.pr_lift - g.pr_lift_ci_low, g.pr_lift_ci_high - g.pr_lift])
        ax.bar(x + (i - 1) * w, g.pr_lift, w - 0.03, color=COL[c], hatch=HATCH[c], edgecolor=SURF, linewidth=0.8, label=NAME[c])
        ax.errorbar(x + (i - 1) * w, g.pr_lift, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.axhline(1, color=INK, lw=1, ls="--", label="chance (lift = 1)")
    ax.set_xticks(x, [SHORT[k] for k in L]); ax.set_ylim(0, None)
    ax.set_ylabel("PR lift on held-out landscape\n(AP ÷ prevalence)")
    ax.set_title("Held-out PR lift by landscape and elevation configuration\n(error bars: paired 3 km spatial-block bootstrap 95% interval, within-landscape sampling only)", fontsize=9, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "F1_LANDSCAPE_PR_LIFT")

    # F2 paired AP differences
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    lim = float(np.nanmax(np.abs(d[["delta_ap_ci_low", "delta_ap_ci_high"]].to_numpy()))) * 1.1
    for ax, c in zip(axes, ["A1", "A2"]):
        g = d[d.comparison == f"{c}_minus_B0"].set_index("held_out").reindex(L)
        y = np.arange(len(L))[::-1]
        ax.axvline(0, color=INK, lw=1)
        ax.hlines(y, g.delta_ap_ci_low, g.delta_ap_ci_high, color=COL[c], lw=2)
        ax.plot(g.delta_ap, y, "o", ms=7, color=COL[c], mec=SURF, mew=1.5)
        ax.set_yticks(y, [SHORT[k] for k in L]); ax.set_xlim(-lim, lim)
        ax.set_xlabel(f"Δ AP ({c} − B0)"); ax.set_title(NAME[c], fontsize=9, color=INK, loc="left")
    fig.suptitle("Paired difference in held-out average precision vs B0 (point and 95% block-bootstrap interval)", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F2_PAIRED_AP_DIFFERENCES")

    # F3 robustness
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, c in enumerate(["B0", "A1", "A2"]):
        ax.bar(i, s.loc[c, "robust_score"], 0.55, color=COL[c], hatch=HATCH[c], edgecolor=SURF, label=NAME[c])
        ax.errorbar(i, s.loc[c, "robust_score"], yerr=[[s.loc[c, "robust_score"] - s.loc[c, "robust_score_ci_low"]], [s.loc[c, "robust_score_ci_high"] - s.loc[c, "robust_score"]]],
                    fmt="none", ecolor=INK2, capsize=3, elinewidth=0.9)
        loo = r[(r.config == c) & (r.subset == "LEAVE_ONE_LANDSCAPE_OUT")].robust_score
        ax.plot(np.full(len(loo), i + 0.36), loo, "o", ms=4, color=INK2, alpha=0.8, label="R with one landscape removed" if i == 0 else None)
        ax.text(i, s.loc[c, "robust_score_ci_high"] + 0.02, f"R = {s.loc[c, 'robust_score']:.3f}", ha="center", color=INK, fontsize=8)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(3), ["B0", "A1", "A2"]); ax.set_ylabel("Robustness score R\n0.70·ln(median lift) + 0.30·ln(min lift)")
    ax.set_title("Robustness score by configuration (error bars: within-landscape bootstrap only)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "F3_ROBUSTNESS_SCORES")

    # F4 elevation distributions (raw vs landscape-centred), label-free
    raw = {}
    for k in L:
        with np.load(V2 / f"outputs/stage4a/STAGE4A_MEG_DATASET/{k}.npz", allow_pickle=False) as z:
            raw[k] = z["elev_m_30m__mean"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for ax, mode in zip(axes, ["raw", "centred"]):
        data = [raw[k] if mode == "raw" else raw[k] - raw[k].mean() for k in L]
        bp = ax.boxplot(data, vert=False, whis=(1, 99), showfliers=False, patch_artist=True, widths=0.55,
                        medianprops=dict(color=INK, lw=1.4), boxprops=dict(facecolor="#d7e6f7", edgecolor=INK2, lw=0.8),
                        whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
        ax.set_yticks(range(1, len(L) + 1), [SHORT[k] for k in L])
        ax.set_xlabel("Unit mean elevation (m)" if mode == "raw" else "Elevation minus landscape mean (m)")
        ax.set_title("B0 input: absolute elevation" if mode == "raw" else "A2 input: landscape-centred elevation", fontsize=9, color=INK, loc="left")
        if mode == "centred":
            ax.axvline(0, color=INK, lw=1)
    axes[0].invert_yaxis()  # shared y-axis: invert once so Kumamoto is on top
    fig.suptitle("Elevation distributions per landscape (box = IQR, whiskers = 1st–99th percentile, all terrain units, no labels used)", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "F4_ELEVATION_DISTRIBUTIONS")


if __name__ == "__main__":
    main()
