"""Stage 4E-A assembly: predictive screen tables, pre-registered advancement decision, figures 5-9 (no new analysis choices)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import stage4b as S
from . import stage4d as D
from . import stage4ea_e_prereg as P
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_a"
FIG = OUT / "figures"
LS = S.LANDSCAPES
REP_OF = {"M+E0": "E0_ABSOLUTE", "M+E1": "E1_WITHIN_LANDSCAPE_PERCENTILE", "M+E2": "E2_LANDSCAPE_CENTERED_ROBUST", "M+E3": "E3_HYBRID_CONTEXT_RELATIVE"}
FAMILIES = ["LOGISTIC_L2", "RANDOM_FOREST"]
SH = {l: l.split("_")[0].title() for l in LS}


def predictions(T):
    pred4b, _ = D.predictions(T)  # verified Stage 4B reuse (M, ME)
    pred = {}
    for m in FAMILIES:
        for k in LS:
            pred[(m, "M", k)] = pred4b[(m, "M", k)]
            pred[(m, "M+E0", k)] = pred4b[(m, "ME", k)]
    sc = json.loads((ROOT / "data/interim/_stage4ea_screen/E_SCREEN_RESULTS.json").read_text())
    res = {(x["held_out"], x["model"], x["feature_set"]): x for x in sc["results"] if x["model"] != "NULL"}
    rows = []
    for key, meta in sc["fold_meta"].items():
        _, k, fs, m = key.split("|")
        p = np.load(ROOT / f"data/interim/_stage4ea_cache/{meta['outer_prediction_key']}__{k}.npy")
        ap = average_precision_score(T[k]["y"], p)
        assert abs(ap - res[(k, m, fs)]["pr_auc"]) < 1e-12
        pred[(m, fs, k)] = dict(raw=p, cal=S.apply_platt(p, meta["calibrator"]), thr=float(meta["threshold"]), source="STAGE4EA_SCREEN")
        rows.append(dict(model=m, feature_set=fs, held_out=k, pr_auc=ap, cal_slope=res[(k, m, fs)]["cal_slope"], cal_intercept=res[(k, m, fs)]["cal_intercept"]))
    s4b = pd.read_csv(ROOT / "outputs/stage4b/STAGE4B_LANDSCAPE_RESULTS.tsv", sep="\t")
    s4b = s4b[(s4b.analysis == "PRIMARY") & s4b.model.isin(FAMILIES) & s4b.feature_set.isin(["M", "ME"])]
    cal = {(r.model, "M" if r.feature_set == "M" else "M+E0", r.held_out): (r.cal_slope, r.cal_intercept) for r in s4b.itertuples()}
    cal.update({(r["model"], r["feature_set"], r["held_out"]): (r["cal_slope"], r["cal_intercept"]) for r in rows})
    return pred, cal


def screen_table(T, pred, cal, bs) -> pd.DataFrame:
    rows = []
    for m in FAMILIES:
        for fs in ["M", "M+E0", "M+E1", "M+E2", "M+E3"]:
            for k in LS:
                y, p = T[k]["y"], pred[(m, fs, k)]
                ap = float(average_precision_score(y, p["raw"]))
                apm = float(average_precision_score(y, pred[(m, "M", k)]["raw"]))
                d = bs[k][(m, fs)][:, 0] - bs[k][(m, "M")][:, 0]
                rows.append(dict(model=m, feature_set=fs, representation=REP_OF.get(fs, "NONE (M only)"), landscape=k, prevalence=float(y.mean()), pr_auc=ap,
                                 pr_lift=ap / y.mean(), roc_auc=float(roc_auc_score(y, p["raw"])), brier=float(np.mean((p["cal"] - y) ** 2)),
                                 cal_slope=cal[(m, fs, k)][0], cal_intercept=cal[(m, fs, k)][1], flagged_fraction=float((p["cal"] >= p["thr"]).mean()),
                                 pr_auc_ratio_vs_M=ap / apm, delta_pr_vs_M=ap - apm, delta_ci_low=float(np.nanpercentile(d, 2.5)), delta_ci_high=float(np.nanpercentile(d, 97.5))))
    return pd.DataFrame(rows)


def decision(scr: pd.DataFrame, idn: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    ba = idn[idn.scope == "E_ONLY"].set_index("representation").balanced_accuracy
    rows = []
    for fs in ["M+E0", "M+E1", "M+E2", "M+E3"]:
        rep = REP_OF[fs]
        row = dict(feature_set=fs, representation=rep, identity_balanced_accuracy=float(ba[rep]))
        ok_all = True
        for m in FAMILIES:
            g = scr[(scr.model == m) & (scr.feature_set == fs)].set_index("landscape")
            gm = scr[(scr.model == m) & (scr.feature_set == "M")].set_index("landscape")
            ratio = g.pr_auc_ratio_vs_M
            A = float(ratio.median()) >= 0.95
            Bc = bool((ratio >= 0.80).all())
            Cc = float(g.pr_lift.min()) >= 0.90 * float(gm.pr_lift.min())
            F = all(float(ratio.drop(k).median()) >= 0.95 for k in LS)
            row.update({f"{m}_median_ratio": float(ratio.median()), f"{m}_min_ratio": float(ratio.min()), f"{m}_min_ratio_landscape": ratio.idxmin(),
                        f"{m}_worst_lift": float(g.pr_lift.min()), f"{m}_worst_lift_M": float(gm.pr_lift.min()),
                        f"{m}_A": A, f"{m}_B": Bc, f"{m}_C": Cc, f"{m}_F": F})
            ok_all &= A and Bc and Cc and F
        Dd = float(ba[rep]) <= float(ba["E0_ABSOLUTE"]) - 0.20
        Ee = bool(ret[ret.representation == rep].passes_rule_E.all())
        row.update(D=Dd, E=Ee, passes_all=bool(ok_all and Dd and Ee and fs != "M+E0"),
                   worst_lift_ratio_mean=float(np.mean([row[f"{m}_worst_lift"] / row[f"{m}_worst_lift_M"] for m in FAMILIES])))
        rows.append(row)
    d = pd.DataFrame(rows)
    passing = d[d.passes_all].sort_values(["worst_lift_ratio_mean"], ascending=False)
    adv = list(passing.feature_set[:P.DECISION_RULE["max_advance"]])
    d["advance_to_stage4e_b"] = d.feature_set.isin(adv)
    d.loc[d.feature_set == "M+E0", "advance_to_stage4e_b"] = False
    d["note"] = np.where(d.feature_set == "M+E0", "historical control (never a candidate)", "")
    return d


def figures(scr, idn, ret, ar):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = {"Software": None}
    reps = ["E0_ABSOLUTE", "E1_WITHIN_LANDSCAPE_PERCENTILE", "E2_LANDSCAPE_CENTERED_ROBUST", "E3_HYBRID_CONTEXT_RELATIVE"]
    lab = ["E0 absolute", "E1 percentile", "E2 centred", "E3 hybrid"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sc in enumerate(["E_ONLY", "M_PLUS_E"]):
        v = idn[idn.scope == sc].set_index("representation").reindex(reps).balanced_accuracy
        ax.bar(np.arange(4) + (i - 0.5) * 0.38, v, 0.38, label=sc.replace("_", " "))
    ax.axhline(1 / 6, color="grey", ls="--", label="chance (1/6)"); ax.set_xticks(range(4), lab); ax.set_ylim(0, 1); ax.set_ylabel("landscape-ID balanced accuracy"); ax.legend()
    ax.set_title("Landscape identity encoded by each E representation (lower = less volcano-identifying)")
    fig.savefig(FIG / "fig5_identity.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    r = ret.groupby("representation").agg(bvf=("between_landscape_variance_fraction", "median"), rho=("min_within_landscape_spearman_with_absolute", "min"),
                                          elev=("median_within_landscape_spearman_with_elevation", lambda s: s.abs().median())).reindex(reps)
    axs[0].bar(lab, r.bvf); axs[0].set_title("Median between-landscape share of variance"); axs[0].set_ylim(0, 1)
    axs[1].bar(np.arange(4) - 0.2, r.rho, 0.4, label="min within-landscape rank corr. with absolute E")
    axs[1].bar(np.arange(4) + 0.2, r.elev, 0.4, label="median |rank corr.| with elevation")
    axs[1].set_xticks(range(4), lab); axs[1].set_ylim(0, 1.05); axs[1].legend(fontsize=8); axs[1].set_title("Information retained within landscapes")
    fig.suptitle("Environmental information retention")
    fig.savefig(FIG / "fig6_information_retention.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    fss = ["M", "M+E0", "M+E1", "M+E2", "M+E3"]
    fig, axs = plt.subplots(1, 2, figsize=(18, 5), sharey=False)
    for a, m in zip(axs, FAMILIES):
        for i, fs in enumerate(fss):
            g = scr[(scr.model == m) & (scr.feature_set == fs)].set_index("landscape").reindex(LS)
            a.bar(np.arange(6) + (i - 2) * 0.16, g.pr_lift, 0.16, label=fs)
        a.axhline(1, color="grey", ls="--"); a.set_xticks(range(6), [SH[l] for l in LS]); a.set_title(m); a.set_ylim(0, None); a.set_ylabel("PR lift over prevalence")
    axs[0].legend()
    fig.suptitle("LOLO transfer: M vs M+E0 vs pre-registered E representations")
    fig.savefig(FIG / "fig7_lolo_screen.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    for a, m in zip(axs, FAMILIES):
        for i, fs in enumerate(fss[1:]):
            g = scr[(scr.model == m) & (scr.feature_set == fs)].set_index("landscape").reindex(LS)
            x = np.arange(6) + (i - 1.5) * 0.2
            a.errorbar(x, g.delta_pr_vs_M, yerr=[g.delta_pr_vs_M - g.delta_ci_low, g.delta_ci_high - g.delta_pr_vs_M], fmt="o", capsize=3, label=fs)
        a.axhline(0, color="k", lw=1); a.set_xticks(range(6), [SH[l] for l in LS]); a.set_title(m); a.set_ylabel("Delta PR-AUC vs M (paired 95% block bootstrap)")
    axs[0].legend()
    fig.suptitle("Per-landscape change relative to morphology alone")
    fig.savefig(FIG / "fig8_delta_vs_M.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    at = ar[ar.column == "ANY_COLUMN"].set_index("representation").reindex(reps)
    axs[0].bar(lab, at.atitlan_fraction_outside_training_range); axs[0].set_ylim(0, 1.05); axs[0].set_title("Atitlan units outside training range (any E column)")
    k = "ATITLAN_STAN_2005"
    for a, (col, t) in zip(axs[1:], (("flagged_fraction", "LOGISTIC_L2: share of Atitlan units flagged at threshold"), ("pr_auc", "Atitlan PR-AUC"))):
        for i, m in enumerate(FAMILIES if col == "pr_auc" else ["LOGISTIC_L2"]):
            v = scr[(scr.model == m) & (scr.landscape == k)].set_index("feature_set").reindex(fss)[col]
            a.bar(np.arange(5) + (i - (0.5 if col == "pr_auc" else 0)) * 0.38, v, 0.38, label=m)
        a.set_xticks(range(5), fss); a.set_title(t); a.set_ylim(0, None); a.legend(fontsize=8)
    fig.suptitle("Atitlan extrapolation test: does a relative E representation avoid the E0 failure mode?")
    fig.savefig(FIG / "fig9_atitlan_extrapolation.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)


def main() -> dict:
    T = D.load_targets()
    pred, cal = predictions(T)
    cfg = [(m, f) for m in FAMILIES for f in ["M", "M+E0", "M+E1", "M+E2", "M+E3"]]
    old = D.P.SEED
    D.P.SEED = P.SEED  # Stage 4E-A seed for the bootstrap
    try:
        bs = D.bootstrap(T, pred, configs=cfg)
    finally:
        D.P.SEED = old
    scr = screen_table(T, pred, cal, bs)
    idn = pd.read_csv(OUT / "STAGE4EA_E_IDENTITY_AUDIT.tsv", sep="\t")
    ret = pd.read_csv(OUT / "STAGE4EA_E_INFORMATION_RETENTION.tsv", sep="\t")
    ar = pd.read_csv(OUT / "STAGE4EA_ATITLAN_RANGE_TEST.tsv", sep="\t")
    dec = decision(scr, idn, ret)
    w = lambda n, d: d.to_csv(OUT / n, sep="\t", index=False, lineterminator="\n", float_format="%.6g")
    w("STAGE4EA_E_TRANSFER_SCREEN.tsv", scr)
    w("STAGE4EA_E_ADVANCEMENT_DECISION.tsv", dec)
    st = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_STATUS.tsv", sep="\t")
    hs = pd.DataFrame([dict(landscape=r.landscape_id, gen3_status=r.status, H_AVAILABLE=int(str(r.status).startswith("VERIFIED")),
                            CORE_H_AVAILABLE=int(str(r.status).startswith("VERIFIED"))) for r in st.itertuples()])
    hs.loc[len(hs)] = dict(landscape="ALL", gen3_status="N_VERIFIED_H_CAPABLE=1; MEGH_EXPERIMENT_UNLOCKED=NO; no Gen-4", H_AVAILABLE=int(hs.H_AVAILABLE.sum()), CORE_H_AVAILABLE=int(hs.CORE_H_AVAILABLE.sum()))
    w("STAGE4EA_H_STATUS.tsv", hs)
    figures(scr, idn, ret, ar)
    return dict(scr=scr, dec=dec, idn=idn, ret=ret, ar=ar, hs=hs)


if __name__ == "__main__":
    main()
    print("DONE")
