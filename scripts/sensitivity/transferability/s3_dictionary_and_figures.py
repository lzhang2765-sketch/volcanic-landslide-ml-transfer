"""Stage C Phase 3 dictionary + Phase 9 figures (reads 02_DATA only). POST HOC EXPLORATORY.

Each figure is written as PDF, SVG and PNG, alongside the CSV of exactly the plotted values.
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

OUT = Path(__file__).resolve().parents[2]
D, F = OUT / "02_DATA", OUT / "03_FIGURES"
L = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
SHORT = {"KUMAMOTO_ASO_2016": "Kumamoto (points)", "NOTO_PENINSULA_2024": "Noto", "LOMBOK_RINJANI_2018": "Lombok",
         "COTABATO_DAVAO_2019": "Cotabato", "ATITLAN_STAN_2005": "Atitlán", "DOMINICA_ERIKA_2015": "Dominica"}
INK, INK2, GRID, SURF, PT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
C3 = {"DESCRIPTIVELY_ASSOCIATED": ("#2a78d6", "o"), "INDETERMINATE": ("#eb6834", "s"), "NOT_SUPPORTED": ("#1baf7a", "^")}
TAG = "POST HOC EXPLORATORY · n = 6 landscapes · not confirmatory"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "svg.fonttype": "none"})


def save(fig, name, src: pd.DataFrame):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(F / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor=SURF)
    src.to_csv(F / f"{name}__source.csv", index=False)
    plt.close(fig)


def stamp(fig):
    fig.text(0.995, -0.03, TAG, ha="right", va="top", fontsize=7, color=INK2)


def rho_text(A, factor, resp="pr_lift_B0"):
    r = A[(A.factor == factor) & (A.response == resp)].iloc[0]
    return (f"Spearman ρ = {r.spearman_rho:+.3f} (n={r.n}); without Atitlán ρ = {r.loo_rho_without_atitlan:+.3f}\n"
            f"LOO range [{r.loo_rho_min:+.2f}, {r.loo_rho_max:+.2f}] · {r.stability_class}")


def scatter(ax, M, x, xlabel, A, logx=False, left=()):
    y, lo, hi = M.pr_lift_B0, M.pr_lift_B0_ci_low, M.pr_lift_B0_ci_high
    ax.errorbar(M[x], y, yerr=[y - lo, hi - y], fmt="o", ms=8, color=PT, mec=SURF, mew=1.5, ecolor=INK2, elinewidth=0.8, capsize=2)
    for l, xv, yv in zip(M.landscape_id, M[x], y):
        ax.annotate(SHORT[l], (xv, yv), xytext=(-6, 4) if l in left else (6, 4), textcoords="offset points", fontsize=8, color=INK,
                    ha="right" if l in left else "left")
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.text(0.99, 0.96, "chance (lift = 1)", transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK2, va="top", ha="right")
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0, None); ax.set_xlabel(xlabel); ax.set_ylabel("Baseline PR lift (AP ÷ prevalence)")
    ax.set_title(rho_text(A, x), fontsize=8, color=INK2, loc="left")


def dictionary(M: pd.DataFrame) -> None:
    std = {
        "landscape_id": ("Landscape identifier", "-", "STAGE5A_DATASET_REGISTER.tsv"),
        "event_year": ("Year of the primary triggering event", "year", "landscape_id suffix; STAGE5A_DATASET_REGISTER.tsv"),
        "n_units": ("Evaluated 90 m terrain units (ALL_VALID)", "count", "STAGE5A_MODEL_CONTRACT.json"),
        "n_positive": ("Positive terrain units", "count", "STAGE5A_MODEL_CONTRACT.json"),
        "prevalence": ("n_positive / n_units", "fraction", "STAGE5A_MODEL_CONTRACT.json"),
        "ap_B0": ("Held-out average precision, official baseline", "0-1", "STAGE5A_MODEL_CONTRACT.json"),
        "pr_lift_B0": ("PRIMARY RESPONSE: ap_B0 / prevalence", "ratio", "STAGE5A_MODEL_CONTRACT.json"),
        "pr_lift_B0_ci_low": ("2.5th pct of 3 km paired block-bootstrap PR lift (within-landscape only)", "ratio", "STAGE4EB_SPATIAL_BOOTSTRAP.tsv"),
        "pr_lift_B0_ci_high": ("97.5th pct of block-bootstrap PR lift", "ratio", "STAGE4EB_SPATIAL_BOOTSTRAP.tsv"),
        "roc_auc_B0": ("Held-out ROC-AUC, official baseline", "0-1", "STAGE5A_MODEL_CONTRACT.json"),
        "geometry": ("Inventory geometry", "category", "Stage A INVENTORY_PROVENANCE_AUDIT.csv"),
        "label_meaning": ("Source vs affected-terrain definition", "category", "Stage A INVENTORY_PROVENANCE_AUDIT.csv source_or_affected"),
        "observation_type": ("Target mapping method", "category", "STAGE3G_DIVERSITY_METADATA.tsv target_mapping_method"),
        "min_mapping_unit_m2": ("Documented minimum mapped landslide size; NA where undocumented", "m^2", "GSI disclaimers; Machado et al. 2022 (Stage A)"),
        "completeness_indicator": ("Documented quantitative completeness; NONE_DOCUMENTED otherwise", "text", "Stage A INVENTORY_PROVENANCE_AUDIT.csv"),
        "overlap_negative_fraction": ("Share of polygon-overlapping units labelled negative (cell-centre rule); NA for point inventory", "fraction", "Stage A E02"),
        "dem_product": ("DEM product used for M", "category", "Stage A DEM_CHRONOLOGY_AUDIT.csv"),
        "tectonic_setting": ("Tectonic setting", "category", "STAGE3G_DIVERSITY_METADATA.tsv"),
        "volcanic_morphology": ("Volcanic landform class", "category", "STAGE3G_DIVERSITY_METADATA.tsv morphology"),
        "erosion_state": ("Erosion state", "category", "STAGE3G_DIVERSITY_METADATA.tsv"),
        "climate": ("Climate class (Stage 3G; differs from Stage-1 record for Kumamoto, Atitlan)", "category", "STAGE3G_DIVERSITY_METADATA.tsv"),
        "trigger": ("Event trigger", "category", "STAGE3G_DIVERSITY_METADATA.tsv"),
        "pr_lift_A1": ("PR lift, absolute elevation removed", "ratio", "Stage A.1 ELEVATION_ABLATION_FOLD_RESULTS.csv"),
        "pr_lift_A2": ("PR lift, landscape-centred elevation", "ratio", "Stage A.1 ELEVATION_ABLATION_FOLD_RESULTS.csv"),
        "rel_dlift_A1": ("(lift_A1 - lift_B0) / lift_B0", "fraction", "Stage A.1 PAIRED_PERFORMANCE_DIFFERENCES.csv"),
        "rel_dlift_A2": ("(lift_A2 - lift_B0) / lift_B0", "fraction", "Stage A.1 PAIRED_PERFORMANCE_DIFFERENCES.csv"),
        "within_landscape_elev_auc": ("Descriptive Mann-Whitney AUC of elevation, positives vs negatives, within landscape", "0-1", "Stage A E03"),
        "std_coef_elev_B0": ("Standardised elevation coefficient, B0 fold model", "log-odds per training SD", "Stage A.1 fold results"),
        "std_coef_elev_centred_A2": ("Standardised centred-elevation coefficient, A2 fold model", "log-odds per training SD", "Stage A.1 fold results"),
        "stage4a_domain_shift_all": ("Existing Stage 4A mean |median shift|/pooled IQR, all MEG", "IQR units", "STAGE4A_DOMAIN_SHIFT_AUDIT.tsv"),
        "stage4a_domain_shift_M": ("Existing Stage 4A mean shift, M only", "IQR units", "STAGE4A_DOMAIN_SHIFT_AUDIT.tsv"),
        "M_oor_count_gt10pct": ("Number of 60 M features with >10% of target units outside training pool 1st-99th pct", "count", "computed (DISTRIBUTION_SHIFT_RESULTS.csv)"),
        "M_mean_abs_smd": ("Mean |SMD| over 60 M features", "training SD", "computed"),
        "M_features_oor_gt10pct": ("Names of those features", "text", "computed"),
        "pr_lift_atitlan_excluded_training": ("PR lift with Atitlan removed from training; NA for Atitlan", "ratio", "data/interim/_stage4b_results/ATITLAN_EXCLUDED.json"),
        "rel_dlift_atitlan_excluded_training": ("Relative change vs B0, same landscape and labels", "fraction", "computed from frozen results"),
        "pr_lift_kumamoto_excluded_training": ("PR lift with Kumamoto removed from training (POLYGON_ONLY); NA for Kumamoto", "ratio", "data/interim/_stage4b_results/POLYGON_ONLY.json"),
        "rel_dlift_kumamoto_excluded_training": ("Relative change vs B0, same landscape and labels", "fraction", "computed from frozen results"),
    }
    stat = {"smd": ("signed (target mean - pool mean)/pool SD", "training SD"), "abs_smd": ("PRIMARY shift: |SMD|", "training SD"),
            "w1_sd": ("1-Wasserstein distance / pool SD", "training SD"), "oor": ("fraction of target units outside pool weighted 1st-99th pct", "fraction"),
            "ks": ("weighted two-sample KS distance (descriptive)", "0-1")}
    feat = {"elev": "elev_m_30m__mean (m)", "slope": "slope_deg_30m__mean (deg)", "relief": "relief_m_r1000__mean (m)", "relelev": "rel_elev_local_r1000__mean (0-1)"}
    rows = []
    for c in M.columns:
        if c in std:
            d, u, s = std[c]
        else:
            t, k = c.split("_", 1)
            d, u = f"{feat[t]}: {stat[k][0]}; pool = other five landscapes, equal landscape weight; predictors only, no labels", stat[k][1]
            s = "computed from STAGE4A_MEG_DATASET/*.npz (predictor columns only)"
        rows.append(dict(column=c, definition=d, units=u, source=s,
                         missing_value_policy="NA where not documented / not applicable; never imputed", n_missing=int(M[c].isna().sum())))
    pd.DataFrame(rows).to_csv(D / "LANDSCAPE_FACTOR_DICTIONARY.csv", index=False)


def main():
    F.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(D / "LANDSCAPE_FACTOR_MASTER.csv")
    A = pd.read_csv(D / "FACTOR_ASSOCIATIONS.csv")
    T = pd.read_csv(D / "TRAINING_COMPOSITION_COMPARISONS.csv")
    dictionary(M)
    base = ["landscape_id", "pr_lift_B0", "pr_lift_B0_ci_low", "pr_lift_B0_ci_high"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    scatter(ax, M, "prevalence", "Positive-label prevalence (log scale; fraction of terrain units)", A, logx=True)
    fig.suptitle("F1 · Baseline PR lift vs positive-label prevalence", x=0.01, ha="left", fontsize=10, color=INK); stamp(fig)
    save(fig, "F1_PR_LIFT_VS_PREVALENCE", M[base + ["prevalence"]])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    scatter(ax, M, "elev_abs_smd", "Absolute-elevation shift |SMD| vs training pool (training-pool SDs)", A)
    fig.suptitle("F2 · Baseline PR lift vs absolute-elevation distribution shift", x=0.01, ha="left", fontsize=10, color=INK); stamp(fig)
    save(fig, "F2_PR_LIFT_VS_ELEVATION_SHIFT", M[base + ["elev_abs_smd", "elev_oor", "elev_w1_sd"]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    scatter(axes[0], M, "slope_abs_smd", "Slope shift |SMD| (training-pool SDs)", A, left=("KUMAMOTO_ASO_2016",))
    scatter(axes[1], M, "relief_abs_smd", "1 km relief shift |SMD| (training-pool SDs)", A, left=("DOMINICA_ERIKA_2015", "KUMAMOTO_ASO_2016"))
    axes[1].set_ylabel("")
    fig.suptitle("F3 · Baseline PR lift vs slope and relief distribution shift", x=0.01, ha="left", fontsize=10, color=INK); stamp(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "F3_PR_LIFT_VS_SLOPE_RELIEF_SHIFT", M[base + ["slope_abs_smd", "relief_abs_smd"]])

    # F4 training composition (matched landscapes and labels)
    t = T[T.comparison.isin(["ATITLAN_EXCLUDED_TRAINING", "KUMAMOTO_EXCLUDED_TRAINING"]) & ~T.landscape.str.startswith("MEDIAN")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, comp, col, title in zip(axes, ["ATITLAN_EXCLUDED_TRAINING", "KUMAMOTO_EXCLUDED_TRAINING"], ["#eb6834", "#2a78d6"],
                                    ["Atitlán removed from training", "Kumamoto removed from training"]):
        g = t[t.comparison == comp].set_index("landscape").reindex([l for l in L]).dropna(subset=["relative_change"])
        y = np.arange(len(g))[::-1]
        ax.axvline(0, color=INK, lw=1)
        ax.hlines(y, 0, g.relative_change * 100, color=col, lw=2)
        ax.plot(g.relative_change * 100, y, "o", ms=8, color=col, mec=SURF, mew=1.5)
        for yy, v in zip(y, g.relative_change * 100):
            ax.text(v + (2 if v >= 0 else -2), yy, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
        med = T[(T.comparison == comp) & T.landscape.str.startswith("MEDIAN")].iloc[0]
        ax.set_yticks(y, [SHORT[l] for l in g.index]); ax.set_xlim(-60, 60)
        ax.set_xlabel("Change in held-out PR lift vs baseline (%)\nsame landscape, same labels")
        ax.set_title(f"{title}\nmedian on same 5 landscapes: {med.pr_lift_reference_B0:.2f} → {med.pr_lift_alternative:.2f}", fontsize=9, color=INK, loc="left")
    fig.suptitle("F4 · Training-composition sensitivity (existing frozen Stage 4B analyses, LOGISTIC_L2 + M)", x=0.01, ha="left", fontsize=10, color=INK)
    stamp(fig); fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "F4_TRAINING_COMPOSITION_SENSITIVITY", t[["comparison", "landscape", "pr_lift_reference_B0", "pr_lift_alternative", "relative_change"]])

    # F5 elevation representation, landscapes ordered by elevation shift
    pdiff = pd.read_csv(OUT.parent / "AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/PAIRED_PERFORMANCE_DIFFERENCES.csv")
    order = M.sort_values("elev_abs_smd").landscape_id.tolist()
    src = []
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for k, (c, col, off, mk) in enumerate([("A1", "#eb6834", -0.15, "o"), ("A2", "#1baf7a", 0.15, "s")]):
        g = pdiff[pdiff.comparison == f"{c}_minus_B0"].set_index("held_out").reindex(order)
        b0 = M.set_index("landscape_id").reindex(order).pr_lift_B0
        rel, lo, hi = g.relative_lift_change * 100, g.delta_pr_lift_ci_low / b0 * 100, g.delta_pr_lift_ci_high / b0 * 100
        x = np.arange(len(order)) + off
        ax.errorbar(x, rel, yerr=[rel - lo, hi - rel], fmt=mk, ms=7, color=col, mec=SURF, mew=1.2, ecolor=col, elinewidth=1.2, capsize=2,
                    label={"A1": "A1 absolute elevation removed", "A2": "A2 landscape-centred elevation"}[c])
        src += [dict(config=c, landscape=l, elev_abs_smd=float(M.set_index("landscape_id").loc[l, "elev_abs_smd"]), rel_change_pct=r_, ci_low_pct=a, ci_high_pct=b)
                for l, r_, a, b in zip(order, rel, lo, hi)]
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(order)), [f"{SHORT[l]}\n|SMD| {M.set_index('landscape_id').loc[l, 'elev_abs_smd']:.2f}" for l in order], fontsize=8)
    ax.set_ylabel("Change in PR lift vs B0 (%)\n(95% block-bootstrap interval)")
    ax.set_xlabel("Held-out landscape, ordered by absolute-elevation shift |SMD| (left = smallest)")
    r1 = A[(A.factor == "elev_abs_smd") & (A.response == "rel_dlift_A1")].iloc[0]; r2 = A[(A.factor == "elev_abs_smd") & (A.response == "rel_dlift_A2")].iloc[0]
    ax.set_title(f"Spearman ρ(|SMD|, change): A1 {r1.spearman_rho:+.2f} ({r1.stability_class}); A2 {r2.spearman_rho:+.2f} ({r2.stability_class})", fontsize=8, color=INK2, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("F5 · Effect of elevation representation across the six landscapes (Stage A.1)", x=0.01, ha="left", fontsize=10, color=INK)
    stamp(fig); fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "F5_ELEVATION_REPRESENTATION_EFFECT", pd.DataFrame(src))

    # F6 association summary (primary + selected secondary)
    keep = A[(A.role == "PRIMARY") | ((A.role == "SECONDARY_SHIFT_OR_INVENTORY") & A.factor.isin(["elev_oor", "M_mean_abs_smd", "overlap_negative_fraction"])) |
             ((A.role == "SECONDARY_RESPONSE") & A.response.isin(["rel_dlift_A1", "rel_dlift_A2"]) & A.factor.isin(["prevalence", "n_positive", "elev_abs_smd"]))].copy()
    keep["label"] = keep.factor + "  →  " + keep.response
    keep = keep.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    y = np.arange(len(keep))
    ax.axvline(0, color=INK, lw=1)
    for ev, (col, mk) in C3.items():
        g = keep[keep.evidence_class == ev]
        yy = y[keep.evidence_class.values == ev]
        ax.hlines(yy, g.loo_rho_min, g.loo_rho_max, color=col, lw=2, alpha=0.6)
        ax.plot(g.spearman_rho, yy, mk, ms=8, color=col, mec=SURF, mew=1.2, label=ev.replace("_", " ").lower() + " (classification rule)")
    ax.set_yticks(y, keep.label, fontsize=7.5); ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman ρ, all six landscapes (marker) and leave-one-landscape-out range (bar)")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, -0.32), ncol=1)
    fig.suptitle("F6 · Exploratory factor-association summary (n = 6; ≥20 comparisons; no multiplicity correction)", x=0.01, ha="left", fontsize=10, color=INK)
    stamp(fig)
    save(fig, "F6_FACTOR_ASSOCIATION_SUMMARY", keep[["factor", "response", "role", "n", "spearman_rho", "loo_rho_min", "loo_rho_max", "stability_class", "evidence_class"]])


if __name__ == "__main__":
    main()
