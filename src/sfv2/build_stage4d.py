"""Build Stage 4D Y-aware outputs (requires STAGE4D_PREREGISTRATION.json and STAGE4D_YBLIND_FREEZE.json).  Development-only diagnosis."""
from __future__ import annotations

import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from . import stage4d as D
from . import stage4d_prereg as P
from . import stage4d_yblind as Y
from .raw_store import sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

OUT = D.OUT
FIG = OUT / "figures"
PREREG = dict(module="9788231b29aae7c697156a306098cfd23b25dfeb9e41c458988f5f143237d8c6", json="5ddc2e310b83e493f7da8535159f6f5f871b6ccaa6695e6658f74d0b0a788547",
              md="35faa141b01f13d9c6ddc4125c71d8a478169576f1a776812cd21f22e5b12e45")
PRIOR = {"stage3f": "outputs/stage3f/STAGE3F_CHECKPOINT.json", "stage3g": "outputs/stage3g/STAGE3G_CHECKPOINT.json", "stage3h": "outputs/stage3h/STAGE3H_CHECKPOINT.json",
         "stage3i": "outputs/stage3i/STAGE3I_CHECKPOINT.json", "stage3j": "outputs/stage3j/STAGE3J_CHECKPOINT.json", "stage4a": "outputs/stage4a/STAGE4A_CHECKPOINT.json",
         "stage4b": "outputs/stage4b/STAGE4B_CHECKPOINT.json", "stage4c_a": "outputs/stage4c_a/STAGE4C_A_CHECKPOINT.json", "stage4c_b": "outputs/stage4c_b/STAGE4C_B_CHECKPOINT.json",
         "stage4c_b2": "outputs/stage4c_b2/STAGE4C_B2_CHECKPOINT.json", "stage4c_b3": "outputs/stage4c_b3/STAGE4C_B3_CHECKPOINT.json"}
PRIOR_SHA = {"stage3f": "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a", "stage3g": "a4f8035123658c063f1572bdfd3276906053ba313735ae3371c442e45cbe4664",
             "stage3h": "f1ea33e848559bcc97f6c8fb7bc7a78c8fca9d004fe924d8c0136d1f488f2d1f", "stage3i": "90072d3562da197c46a0dee0bdc0e1fe752a242c520ee9f44a09bd1b09bc9ea1",
             "stage3j": "99a8c6f4cd90765d4d5bd7be790af22502b22194cd47c63443161e3ede4c82fd", "stage4a": "fd25accb50578bd4f4f28042e42504a89ef936c904d5ed262a0f65fc7fa2452c",
             "stage4b": "2685d1b6d565de7945e138d9b014d9aa4b53657511552299c19ebc1f4df946b8", "stage4c_a": "09463f7d76cc865b08ad602a5518532c8a4cc2a5131952b73618082bed09bfa7",
             "stage4c_b": "90835a71a53cb8a544a960ae7d1e36f5a3ea44e524e730cd6f62db22c6e4bd7c", "stage4c_b2": "f548ceb0c501560b603ee7835a6a18b8350899beef92b527dc4d9571fb6aa1f7",
             "stage4c_b3": "d1ad15c66bb1df312f6738c2c1aeb5d25dc868025641bd92661631408c8d7807"}
V1_FREEZE = ("../Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json", "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1")
SHORT = {"KUMAMOTO_ASO_2016": "Kumamoto", "NOTO_PENINSULA_2024": "Noto", "LOMBOK_RINJANI_2018": "Lombok", "COTABATO_DAVAO_2019": "Cotabato",
         "ATITLAN_STAN_2005": "Atitlan", "DOMINICA_ERIKA_2015": "Dominica"}


def _tsv(name, df):
    df.to_csv(OUT / name, sep="\t", index=False, lineterminator="\n", float_format="%.6g")


def integrity() -> dict:
    v1 = v1_verify()
    prior = {k: sha256(ROOT / v) for k, v in PRIOR.items()}
    hist = {}
    for k, v in PRIOR.items():
        cp = json.loads((ROOT / v).read_text())
        for key in ("output_hashes", "h_dataset_partitions", "meg_partitions"):
            for p, h in (cp.get(key) or {}).items():
                if isinstance(h, str) and len(h) == 64 and (ROOT / p).exists():
                    hist[p] = sha256(ROOT / p) == h
    return dict(v1={k: v1[k] for k in ("strict_files", "v1_intact", "writes_performed")}, v1_freeze_json_sha256=sha256(ROOT / V1_FREEZE[0]),
                v1_freeze_json_ok=sha256(ROOT / V1_FREEZE[0]) == V1_FREEZE[1], prior_checkpoints={PRIOR[k]: v for k, v in prior.items()},
                prior_unchanged=prior == PRIOR_SHA, historical_files_checked=len(hist), historical_files_unchanged=all(hist.values()),
                stage4b_cache_files=len(list((ROOT / "data/interim/_stage4b_cache").glob("*.npy"))))


def h_status() -> pd.DataFrame:
    st = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_STATUS.tsv", sep="\t")
    rows = []
    for r in st.itertuples():
        rows.append(dict(landscape=r.landscape_id, gen3_status=r.status, H_AVAILABLE=int(str(r.status).startswith("VERIFIED")),
                         note="AGE_PROXY_ONLY; no CORE_H for predictive use" if r.status == "AGE_PROXY_ONLY" else ""))
    df = pd.DataFrame(rows)
    df.loc[len(df)] = dict(landscape="ALL", gen3_status="N_VERIFIED_H_CAPABLE=1; MEGH_EXPERIMENT_UNLOCKED=NO", H_AVAILABLE=int(df.H_AVAILABLE.sum()),
                           note="Gen-3 selected-H diagnostic (Kumamoto only): H_REMOVED_FRACTION, H_RELIEF_CHANGE_FRACTION, H_SCENARIO_SPREAD_FRACTION, "
                                "H_CONTROL_DISTANCE_FRACTION; insufficient to unlock MEGH")
    return df


# ------------------------------------------------------------------------------------------------ figures
def figures(T, pred, boot, sig, ds, abl, ood_t, ood, fst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    L = P.LANDSCAPES
    names = [SHORT[l] for l in L]
    meta = {"Software": None}
    lead = boot[(boot.model == "RANDOM_FOREST") & (boot.feature_set == "MEG")]
    # 1 PR-AUC with CIs + prevalence
    fig, ax = plt.subplots(figsize=(10, 5))
    pr = lead[lead.metric == "pr_auc"].set_index("landscape").reindex(L)
    nu = boot[(boot.model == "NULL_PREVALENCE")].set_index("landscape").reindex(L)
    x = np.arange(6)
    ax.errorbar(x, pr.point_estimate, yerr=[pr.point_estimate - pr.ci_low, pr.ci_high - pr.point_estimate], fmt="o", capsize=5, label="RF + MEG (Stage 4B leader)")
    ax.errorbar(x + 0.15, nu.point_estimate, yerr=[nu.point_estimate - nu.ci_low, nu.ci_high - nu.point_estimate], fmt="s", color="grey", capsize=5, label="prevalence (null PR-AUC)")
    ax.set_xticks(x, names); ax.set_ylim(0, None); ax.set_ylabel("held-out PR-AUC (axis from 0)")
    ax.set_title("Leave-one-landscape-out PR-AUC with 95% spatial block-bootstrap intervals (3 km blocks)"); ax.legend()
    fig.savefig(FIG / "fig1_lolo_pr_auc_ci.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 2 lift
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, f) in enumerate([("RANDOM_FOREST", "MEG"), ("LOGISTIC_L2", "M"), ("HIST_GRADIENT_BOOSTING", "MEG")]):
        g = boot[(boot.model == m) & (boot.feature_set == f) & (boot.metric == "pr_lift")].set_index("landscape").reindex(L)
        ax.errorbar(x + (i - 1) * 0.15, g.point_estimate, yerr=[g.point_estimate - g.ci_low, g.ci_high - g.point_estimate], fmt="o", capsize=4, label=f"{m} + {f}")
    ax.axhline(1, color="grey", ls="--", label="no lift (= prevalence)")
    ax.set_xticks(x, names); ax.set_ylim(0, None); ax.set_ylabel("PR-AUC / prevalence"); ax.legend(fontsize=8)
    ax.set_title("PR lift over prevalence (95% spatial block-bootstrap intervals)")
    fig.savefig(FIG / "fig2_pr_lift.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 3 shift vs PR-AUC
    H = ds[ds.group == "MEG"].set_index("held_out").reindex(L)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, met, lab in ((axs[0], "pr_auc", "PR-AUC"), (axs[1], "pr_lift", "PR lift")):
        g = sig[(sig.model == "RANDOM_FOREST") & (sig.feature_set == "MEG")].set_index("landscape").reindex(L)
        ax.scatter(H.energy_H_primary, g[met])
        for l in L:
            ax.annotate(SHORT[l], (H.energy_H_primary[l], g[met][l]), fontsize=9, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("MEG domain shift: normalised energy statistic H (held-out vs training)"); ax.set_ylabel(f"RF + MEG held-out {lab}"); ax.set_ylim(0, None)
        ax.set_title(f"n = 6 landscapes - hypothesis-generating only\nSpearman rho = {D.spearmanr(H.energy_H_primary, g[met]).correlation:.2f}")
    fig.savefig(FIG / "fig3_shift_vs_performance.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 4 decomposition
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, gname in enumerate(["M", "E", "G", "MEG"]):
        v = ds[ds.group == gname].set_index("held_out").reindex(L)
        ax.bar(x + (i - 1.5) * 0.2, v.energy_H_primary, 0.2, yerr=2 * v.energy_H_subsample_sd, label=f"{gname} ({int(v.n_predictors.iloc[0])} predictors)")
    ax.set_xticks(x, names); ax.set_ylabel("normalised energy statistic H"); ax.legend(); ax.set_ylim(0, None)
    ax.set_title("Domain shift (held-out vs pooled training) by predictor group; error bars = 2 x subsample SD")
    fig.savefig(FIG / "fig4_shift_decomposition.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 5 ablation
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, m in zip(axs, D.MODELS):
        for i, f in enumerate(["M", "ME", "MG", "MEG"]):
            g = sig[(sig.model == m) & (sig.feature_set == f)].set_index("landscape").reindex(L)
            ax.bar(x + (i - 1.5) * 0.2, g.pr_lift, 0.2, label=f)
        ax.axhline(1, color="grey", ls="--"); ax.set_xticks(x, names, rotation=30); ax.set_title(m); ax.set_ylim(0, None)
    axs[0].set_ylabel("PR lift over prevalence"); axs[0].legend()
    fig.suptitle("Feature-group ablation: M vs ME vs MG vs MEG per held-out landscape")
    fig.savefig(FIG / "fig5_feature_group_ablation.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 6 calibration
    fig, ax = plt.subplots(figsize=(7, 6))
    for l in L:
        p, y = pred[("RANDOM_FOREST", "MEG", l)]["cal"], T[l]["y"]
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        r = pd.DataFrame(dict(q=q, p=p, y=y)).groupby("q").mean()
        ax.plot(r.p, r.y, "o-", label=SHORT[l])
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1); ax.set_xlabel("mean calibrated prediction (decile)"); ax.set_ylabel("observed positive rate"); ax.legend()
    ax.set_title("Reliability by held-out landscape (RF + MEG, Stage 4B Platt calibration)")
    fig.savefig(FIG / "fig6_calibration.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 7 Cotabato vs Noto
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for row, l in enumerate(["COTABATO_DAVAO_2019", "NOTO_PENINSULA_2024"]):
        p, y = pred[("RANDOM_FOREST", "MEG", l)], T[l]["y"]
        bins = np.linspace(0, np.percentile(p["raw"], 99.5), 50)
        axs[row, 0].hist(p["raw"][~y], bins, density=True, alpha=.6, label="negatives"); axs[row, 0].hist(p["raw"][y], bins, density=True, alpha=.6, label="positives")
        axs[row, 0].set_title(f"{SHORT[l]}: raw score distributions"); axs[row, 0].legend()
        flag = p["cal"] >= p["thr"]
        cx, cy = T[l]["cx"] / 1000, T[l]["cy"] / 1000
        axs[row, 1].scatter(cx, cy, s=0.2, c="0.85")
        axs[row, 1].scatter(cx[flag & ~y], cy[flag & ~y], s=0.5, c="tab:orange", label="false positive")
        axs[row, 1].scatter(cx[~flag & y], cy[~flag & y], s=2, c="tab:blue", label="false negative")
        axs[row, 1].scatter(cx[flag & y], cy[flag & y], s=2, c="tab:green", label="true positive")
        axs[row, 1].set_title(f"{SHORT[l]}: errors at the Stage 4B threshold"); axs[row, 1].legend(markerscale=6, fontsize=8); axs[row, 1].set_aspect("equal")
        sh = fst[fst.held_out == l].assign(a=lambda d: d.std_median_diff.abs()).sort_values("a", ascending=False).head(12)
        axs[row, 2].barh(sh.predictor, sh.std_median_diff, color=["tab:red" if s.startswith("e_") else "tab:green" if s.startswith("g_") else "tab:blue" for s in sh.predictor])
        axs[row, 2].set_title(f"{SHORT[l]}: most shifted predictors (std. median diff; red E, green G, blue M)"); axs[row, 2].tick_params(labelsize=7)
    fig.suptitle("Transfer forensics: Cotabato (chance-like) vs Noto (strongest)")
    fig.savefig(FIG / "fig7_cotabato_vs_noto.png", dpi=110, bbox_inches="tight", metadata=meta); plt.close(fig)
    # 8 OOD
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    q = ood_t[ood_t.ood_quintile > 0]
    for l in L:
        g = q[q.landscape == l]
        axs[0].plot(g.ood_quintile, g.fn_rate, "o-", label=SHORT[l]); axs[1].plot(g.ood_quintile, g.pr_lift, "o-", label=SHORT[l])
    axs[0].set_xlabel("OOD quintile within held-out landscape (5 = most OOD)"); axs[0].set_ylabel("false-negative rate among positives"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("OOD quintile"); axs[1].set_ylabel("PR lift within quintile"); axs[1].axhline(1, color="grey", ls="--")
    fig.suptitle("OOD prototype (PROTOTYPE / DEVELOPMENT DIAGNOSTIC / NONFINAL): RF + MEG errors by kNN OOD quintile")
    fig.savefig(FIG / "fig8_ood_prototype.png", dpi=130, bbox_inches="tight", metadata=meta); plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pre = dict(module=sha256(ROOT / "src/sfv2/stage4d_prereg.py"), json=sha256(OUT / "STAGE4D_PREREGISTRATION.json"), md=sha256(OUT / "STAGE4D_PREREGISTRATION.md"))
    if pre != PREREG:
        raise RuntimeError("STOP: Stage 4D preregistration changed")
    freeze = json.loads((OUT / "STAGE4D_YBLIND_FREEZE.json").read_text())
    if any(sha256(OUT / f) != h for f, h in freeze["files"].items()):
        raise RuntimeError("STOP: Y-blind outputs changed after freezing")
    integ_before = integrity()
    assert integ_before["prior_unchanged"] and integ_before["historical_files_unchanged"] and integ_before["v1_freeze_json_ok"] and integ_before["v1"]["v1_intact"]
    T = D.load_targets()
    pred, recon = D.predictions(T)
    _tsv("STAGE4D_FOLD_REUSE_PROVENANCE.tsv", recon)
    bs = D.bootstrap(T, pred)
    boot = D.lolo_bootstrap_table(T, pred, bs)
    sig = D.transfer_signal(T, pred, bs)
    abl = D.ablation(T, pred, bs)
    fsum = D.feature_set_summary(sig)
    bs6 = D.bootstrap(T, pred, blocks="b6", configs=[D.LEADER])
    s6 = D.transfer_signal(T, pred, bs6).assign(sensitivity="6 km blocks")
    ds = pd.read_csv(OUT / "STAGE4D_DOMAIN_SHIFT.tsv", sep="\t")
    fst = pd.read_csv(OUT / "STAGE4D_FEATURE_SHIFT.tsv", sep="\t")
    idn = pd.read_csv(OUT / "STAGE4D_LANDSCAPE_IDENTITY_AUDIT.tsv", sep="\t")
    assoc = D.association(ds, sig)
    st = Y.Store()
    ood = D.ood_scores(st)
    ood_t = D.ood_table(T, pred, ood)
    fx = D.forensics(T, pred, ood, fst)
    hs = h_status()
    _tsv("STAGE4D_LOLO_BOOTSTRAP.tsv", boot)
    _tsv("STAGE4D_TRANSFER_SIGNAL.tsv", pd.concat([sig.assign(sensitivity="primary 3 km blocks"), s6], ignore_index=True))
    _tsv("STAGE4D_FEATURE_GROUP_ABLATION.tsv", abl)
    _tsv("STAGE4D_FEATURE_SET_SUMMARY.tsv", fsum)
    _tsv("STAGE4D_SHIFT_PERFORMANCE_ASSOCIATION.tsv", assoc)
    _tsv("STAGE4D_OOD_PROTOTYPE.tsv", ood_t)
    _tsv("STAGE4D_FORENSICS.tsv", fx)
    _tsv("STAGE4D_H_STATUS.tsv", hs)
    # ---- summary
    res4b = pd.read_csv(ROOT / "outputs/stage4b/STAGE4B_LANDSCAPE_RESULTS.tsv", sep="\t")
    res4b = res4b[(res4b.analysis == "PRIMARY") & (res4b.model != "NULL_PREVALENCE")]
    rows = []
    for l in P.LANDSCAPES:
        best = res4b[res4b.held_out == l].sort_values("pr_auc").iloc[-1]
        sb = sig[(sig.landscape == l) & (sig.model == best.model) & (sig.feature_set == best.feature_set)].iloc[0]
        ld = sig[(sig.landscape == l) & (sig.model == "RANDOM_FOREST") & (sig.feature_set == "MEG")].iloc[0]
        bb = boot[(boot.landscape == l) & (boot.model == best.model) & (boot.feature_set == best.feature_set) & (boot.metric == "pr_auc")].iloc[0]
        dsl = ds[ds.held_out == l].set_index("group").energy_H_primary
        a = abl[(abl.landscape == l) & (abl.model == "RANDOM_FOREST")].set_index("contrast")
        cal = res4b[(res4b.held_out == l) & (res4b.model == "RANDOM_FOREST") & (res4b.feature_set == "MEG")].iloc[0]
        oo = ood_t[(ood_t.landscape == l) & (ood_t.ood_quintile > 0)]
        ol = ood_t[(ood_t.landscape == l) & (ood_t.ood_quintile == 0)].iloc[0]
        rows.append(dict(
            landscape=l, n_units=len(T[l]["y"]), n_positive=int(T[l]["y"].sum()), prevalence=float(T[l]["y"].mean()),
            best_existing_stage4b_model=best.model, best_existing_stage4b_feature_set=best.feature_set, pr_auc=best.pr_auc, pr_auc_ci_low=bb.ci_low, pr_auc_ci_high=bb.ci_high,
            pr_lift=sb.pr_lift, delta_pr_vs_null=sb.delta_pr, delta_pr_ci_low=sb.delta_ci_low, delta_pr_ci_high=sb.delta_ci_high,
            transfer_signal_label=ld.label, transfer_signal_label_best_cell=sb.label, leader_rf_meg_pr_auc=ld.pr_auc, leader_rf_meg_pr_lift=ld.pr_lift,
            domain_shift_M=dsl["M"], domain_shift_E=dsl["E"], domain_shift_G=dsl["G"], domain_shift_MEG=dsl["MEG"],
            dominant_shift_group=max(("M", "E", "G"), key=lambda g: dsl[g]),
            effect_E=a.loc["Delta_E", "delta"], effect_G=a.loc["Delta_G", "delta"], effect_E_given_G=a.loc["Delta_E_given_G", "delta"], effect_G_given_E=a.loc["Delta_G_given_E", "delta"],
            calibration_summary=f"RF+MEG cal slope {cal.cal_slope:.2f}, intercept {cal.cal_intercept:.2f}, Brier {cal.brier:.4f}",
            ood_summary=f"median OOD pct {ol.median_ood_percentile:.2f}; FN rate Q1 {oo.fn_rate.iloc[0]:.2f} -> Q5 {oo.fn_rate.iloc[-1]:.2f}; lift Q1 {oo.pr_lift.iloc[0]:.2f} -> Q5 {oo.pr_lift.iloc[-1]:.2f}",
            H_status=hs.set_index("landscape").loc[l, "gen3_status"], key_failure_or_strength="",
            note="best_existing_* = post-hoc maximum over Stage 4B PRIMARY cells (optimistic); effects are RANDOM_FOREST PR-AUC deltas"))
    summ = pd.DataFrame(rows)
    summ["key_failure_or_strength"] = summ.apply(lambda r: f"{r.transfer_signal_label} (RF+MEG lift {r.leader_rf_meg_pr_lift:.2f}); dominant shift {r.dominant_shift_group}", axis=1)
    _tsv("STAGE4D_LANDSCAPE_DIAGNOSTIC_SUMMARY.tsv", summ)
    figures(T, pred, boot, sig, ds, abl, ood_t, ood, fst)
    integ_after = integrity()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    return dict(T=T, pred=pred, boot=boot, sig=sig, abl=abl, fsum=fsum, assoc=assoc, ood_t=ood_t, fx=fx, ds=ds, fst=fst, idn=idn, summ=summ, hs=hs, s6=s6,
                integ_before=integ_before, integ_after=integ_after, pre=pre, freeze=freeze, runtime=round(time.time() - t0), head=git("rev-parse", "HEAD"),
                git_tracked_modified=len([s for s in git("status", "--porcelain").splitlines() if not s.startswith("??")]))


if __name__ == "__main__":
    import pickle
    r = main()
    (ROOT / "data/interim/_stage4d_mg").mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in r.items() if k not in ("T", "pred")}
    with open(ROOT / "data/interim/_stage4d_mg/stage4d_summary_objects.pkl", "wb") as f:
        pickle.dump(keep, f)
    print("DONE", r["runtime"])
