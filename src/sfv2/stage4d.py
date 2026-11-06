"""Stage 4D Y-aware diagnostics (run only after STAGE4D_YBLIND_FREEZE.json exists).

Reuses Stage 4B held-out predictions (read-only, verified against the published PR-AUC) and the Stage 4D MG ablation; computes the paired
3 km spatial block bootstrap, transfer-signal labels, feature-group ablation, shift-performance association, OOD prototype and forensics.
No model selection, no H, no lockbox.
"""
from __future__ import annotations

import json
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from . import stage4a_base as B
from . import stage4b as S
from . import stage4d_prereg as P
from . import stage4d_yblind as Y
from .build_stage4a import read_columns
from .schema import ROOT

OUT = ROOT / "outputs/stage4d"
B4 = ROOT / "outputs/stage4b"
CACHE4B = ROOT / "data/interim/_stage4b_cache"
CACHE4D = ROOT / "data/interim/_stage4d_cache"
MG_JSON = ROOT / "data/interim/_stage4d_mg/MG_RESULTS.json"
MODELS = ["LOGISTIC_L2", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING"]
FSETS = ["M", "ME", "MG", "MEG", "MEG_NO_BIO15"]
LEADER = ("RANDOM_FOREST", "MEG")


# ------------------------------------------------------------------------------------------------ data + predictions
def load_targets() -> dict:
    out = {}
    for lid in P.LANDSCAPES:
        B.firewall_check(lid)
        with np.load(Y.MEG_DIR / f"{lid}.npz", allow_pickle=False) as z:
            out[lid] = dict(y=z["y_positive"].astype(bool), b3=z["spatial_block_3000m"], b6=z["spatial_block_6000m"], cx=z["centroid_x"], cy=z["centroid_y"],
                            ids=z["terrain_unit_id"])
    return out


def predictions(T: dict) -> tuple[dict, pd.DataFrame]:
    """(model, fs, landscape) -> dict(raw, cal, thr); every reused Stage 4B fold is checked against the published PR-AUC."""
    # Amendment 01: full-precision Stage 4B results record (the TSV is its 6-significant-digit export)
    full = json.loads((ROOT / "data/interim/_stage4b_results/PRIMARY.json").read_text())["results"]
    tsv = pd.read_csv(B4 / "STAGE4B_LANDSCAPE_RESULTS.tsv", sep="\t", keep_default_na=False, na_values=[""])
    tsv = tsv[tsv.analysis == "PRIMARY"].set_index(["held_out", "model", "feature_set"])
    pred, rows = {}, []
    for r in [x for x in full if x["model"] != "NULL"]:
        k, m, f = r["held_out"], r["model"], r["feature_set"]
        train = [l for l in P.LANDSCAPES if l != k]
        key = S._key(sorted(train), f, m, json.loads(r["params"]), "ALL_VALID", "LANDSCAPE_BALANCED")
        p = np.load(CACHE4B / f"{key}__{k}.npy")
        ap = average_precision_score(T[k]["y"], p)
        t_ap = float(tsv.loc[(k, m, f), "pr_auc"])
        if abs(ap - r["pr_auc"]) > 1e-12 or abs(ap - t_ap) > 5e-6 * ap:
            raise RuntimeError(f"STOP: Stage 4B prediction for {k}/{m}/{f} does not reproduce ({ap} vs {r['pr_auc']} / TSV {t_ap})")
        cal = json.loads(r["calibrator"]) if r.get("calibrator") not in (None, "null") else None
        pred[(m, f, k)] = dict(raw=p, cal=S.apply_platt(p, cal), thr=float(r["threshold"]), source="STAGE4B_CACHE_VERIFIED")
        rows.append(dict(model=m, feature_set=f, held_out=k, pr_auc_stage4b=r["pr_auc"], pr_auc_stage4b_tsv=t_ap, pr_auc_reconstructed=ap, source="STAGE4B"))
    mg = json.loads(MG_JSON.read_text())
    mres = {(x["held_out"], x["model"]): x for x in mg["results"] if x["model"] != "NULL"}
    for key_, meta in mg["fold_meta"].items():
        _, k, fs, model = key_.split("|")
        p = np.load(CACHE4D / f"{meta['outer_prediction_key']}__{k}.npy")
        ap = average_precision_score(T[k]["y"], p)
        assert abs(ap - mres[(k, model)]["pr_auc"]) < 1e-12
        pred[(model, "MG", k)] = dict(raw=p, cal=S.apply_platt(p, meta["calibrator"]), thr=float(meta["threshold"]), source="STAGE4D_MG")
        rows.append(dict(model=model, feature_set="MG", held_out=k, pr_auc_stage4b=np.nan, pr_auc_reconstructed=ap, source="STAGE4D_MG"))
    return pred, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ fast weighted metrics
class Sorted:
    """Pre-sorted score structure for fast weighted AP / ROC-AUC with exact sklearn tie handling."""

    def __init__(self, p, y):
        o = np.argsort(-p, kind="stable")
        self.o, self.y = o, y[o]
        ps = p[o]
        self.starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]

    def ap_auc(self, w):
        ws = w[self.o]
        tp = np.add.reduceat(ws * self.y, self.starts)
        fp = np.add.reduceat(ws * ~self.y, self.starts)
        P_, N_ = tp.sum(), fp.sum()
        if P_ <= 0 or N_ <= 0:
            return np.nan, np.nan
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        prec = ctp / np.maximum(ctp + cfp, 1e-300)
        ap = float(np.sum(tp / P_ * prec))
        neg_below = N_ - cfp
        auc = float(np.sum(tp * (neg_below + 0.5 * fp)) / (P_ * N_))
        return ap, auc


def brier_w(pcal, y, w):
    return float(np.sum(w * (pcal - y) ** 2) / np.sum(w))


def block_codes(b):
    u, inv = np.unique(b, return_inverse=True)
    return inv, len(u)


def bootstrap(T, pred, blocks="b3", configs=None, reps=None):
    reps = reps or P.BOOTSTRAP["replicates"]
    out = {}
    for li, k in enumerate(P.LANDSCAPES):
        y = T[k]["y"]
        code, nb = block_codes(T[k][blocks])
        cfg = [c for c in (configs or [(m, f) for m in MODELS for f in FSETS]) if (c[0], c[1], k) in pred]
        srt = {c: Sorted(pred[(c[0], c[1], k)]["raw"], y) for c in cfg}
        rng = np.random.default_rng(P.SEED + li)
        R = {"prev": np.full(reps, np.nan), **{c: np.full((reps, 3), np.nan) for c in cfg}}
        dropped = 0
        for r in range(reps):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code].astype(np.float64)
            wp = w[y].sum()
            if wp == 0:
                dropped += 1
                continue
            R["prev"][r] = wp / w.sum()
            for c in cfg:
                ap, auc = srt[c].ap_auc(w)
                R[c][r] = (ap, auc, brier_w(pred[(c[0], c[1], k)]["cal"], y, w))
        R["dropped"] = dropped
        out[k] = R
    return out


def pct(a):
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


# ------------------------------------------------------------------------------------------------ tables
def point_metrics(T, pred, k, m, f):
    y, p = T[k]["y"], pred[(m, f, k)]
    return dict(pr_auc=float(average_precision_score(y, p["raw"])), roc_auc=float(roc_auc_score(y, p["raw"])), brier=float(np.mean((p["cal"] - y) ** 2)))


def lolo_bootstrap_table(T, pred, BS):
    rows = []
    for k in P.LANDSCAPES:
        y = T[k]["y"]
        prev = float(y.mean())
        base = dict(landscape=k, n_units=len(y), n_positive=int(y.sum()), prevalence=prev, bootstrap_method="paired spatial block bootstrap (3 km Stage 4A blocks)",
                    n_bootstrap=P.BOOTSTRAP["replicates"] - BS[k]["dropped"], seed=P.SEED + P.LANDSCAPES.index(k))
        lo, hi = pct(BS[k]["prev"])
        rows.append(dict(base, model="NULL_PREVALENCE", feature_set="NONE", metric="pr_auc", point_estimate=prev, ci_low=lo, ci_high=hi))
        for (m, f) in [c for c in BS[k] if isinstance(c, tuple)]:
            pm = point_metrics(T, pred, k, m, f)
            for j, met in enumerate(("pr_auc", "roc_auc", "brier")):
                lo, hi = pct(BS[k][(m, f)][:, j])
                rows.append(dict(base, model=m, feature_set=f, metric=met, point_estimate=pm[met], ci_low=lo, ci_high=hi))
            lift = BS[k][(m, f)][:, 0] / BS[k]["prev"]
            lo, hi = pct(lift)
            rows.append(dict(base, model=m, feature_set=f, metric="pr_lift", point_estimate=pm["pr_auc"] / prev, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def label(point, lo):
    if point <= 0:
        return "CHANCE_LIKE_OR_WORSE"
    return "ROBUST_POSITIVE_TRANSFER" if lo > 0 else "UNCERTAIN_POSITIVE_TRANSFER"


def transfer_signal(T, pred, BS):
    rows = []
    for k in P.LANDSCAPES:
        prev = float(T[k]["y"].mean())
        for (m, f) in [c for c in BS[k] if isinstance(c, tuple)]:
            pm = point_metrics(T, pred, k, m, f)
            d = BS[k][(m, f)][:, 0] - BS[k]["prev"]
            lo, hi = pct(d)
            dp = pm["pr_auc"] - prev
            rows.append(dict(landscape=k, model=m, feature_set=f, prevalence=prev, pr_auc=pm["pr_auc"], pr_lift=pm["pr_auc"] / prev, delta_pr=dp, delta_ci_low=lo,
                             delta_ci_high=hi, pct_replicates_delta_gt0=float(np.mean(d[np.isfinite(d)] > 0) * 100), label=label(dp, lo),
                             is_stage4b_leader=(m, f) == LEADER))
    return pd.DataFrame(rows)


def ablation(T, pred, BS):
    rows = []
    D = {"Delta_E": ("ME", "M"), "Delta_G": ("MG", "M"), "Delta_G_given_E": ("MEG", "ME"), "Delta_E_given_G": ("MEG", "MG")}
    for k in P.LANDSCAPES:
        for m in MODELS:
            for name, (a, b) in D.items():
                if (m, a) not in BS[k] or (m, b) not in BS[k]:
                    continue
                pa, pb = point_metrics(T, pred, k, m, a)["pr_auc"], point_metrics(T, pred, k, m, b)["pr_auc"]
                d = BS[k][(m, a)][:, 0] - BS[k][(m, b)][:, 0]
                lo, hi = pct(d)
                rows.append(dict(landscape=k, model=m, contrast=name, with_set=a, without_set=b, pr_auc_with=pa, pr_auc_without=pb, delta=pa - pb, ci_low=lo, ci_high=hi,
                                 direction="GAIN" if lo > 0 else "LOSS" if hi < 0 else "SPANS_ZERO"))
    return pd.DataFrame(rows)


def feature_set_summary(sig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, f), g in sig.groupby(["model", "feature_set"]):
        lift = g.set_index("landscape").pr_lift
        row = dict(model=m, feature_set=f, median_pr_auc=float(g.pr_auc.median()), median_pr_lift=float(lift.median()), worst_pr_lift=float(lift.min()),
                   worst_landscape=lift.idxmin(), stability_sd_log_lift=float(np.log(lift).std(ddof=1)))
        for k in lift.index:
            l2 = lift.drop(k)
            row[f"loo_median_lift_without_{k[:4]}"] = float(l2.median())
        rows.append(row)
    return pd.DataFrame(rows)


def association(ds: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    H = ds[ds.group == "MEG"].set_index("held_out").energy_H_primary
    for (m, f) in [(m, "MEG") for m in MODELS] + [("LOGISTIC_L2", "M")]:
        g = sig[(sig.model == m) & (sig.feature_set == f)].set_index("landscape")
        for metric in ("pr_auc", "pr_lift"):
            v = g[metric].reindex(H.index)
            rho = float(spearmanr(H, v).correlation)
            perm = np.array([spearmanr(H.values, np.array(pp)).correlation for pp in permutations(v.values)])
            loo = {k: float(spearmanr(H.drop(k), v.drop(k)).correlation) for k in H.index}
            rows.append(dict(model=m, feature_set=f, shift="energy_H_MEG", performance=metric, n_landscapes=6, spearman_rho=rho, pearson_r=float(pearsonr(H, v)[0]),
                             exact_permutation_p_two_sided=float(np.mean(np.abs(perm) >= abs(rho) - 1e-12)), loo_rho_min=min(loo.values()), loo_rho_max=max(loo.values()),
                             loo_rho=json.dumps({k[:4]: round(x, 3) for k, x in loo.items()}), primary=(m, f, metric) == ("RANDOM_FOREST", "MEG", "pr_auc"),
                             status="HYPOTHESIS-GENERATING ONLY (n = 6 independent landscapes)"))
    for grp in ("M", "E", "G"):
        Hg = ds[ds.group == grp].set_index("held_out").energy_H_primary
        g = sig[(sig.model == "RANDOM_FOREST") & (sig.feature_set == "MEG")].set_index("landscape").pr_auc.reindex(Hg.index)
        rows.append(dict(model="RANDOM_FOREST", feature_set="MEG", shift=f"energy_H_{grp}", performance="pr_auc", n_landscapes=6,
                         spearman_rho=float(spearmanr(Hg, g).correlation), pearson_r=float(pearsonr(Hg, g)[0]), primary=False,
                         status="HYPOTHESIS-GENERATING ONLY (n = 6 independent landscapes)"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ OOD prototype
def ood_scores(st) -> dict:
    out = {}
    for k in P.LANDSCAPES:
        _, ref, mu, sd = Y.fold_samples(st, k)
        dref = cdist(ref, ref)
        np.fill_diagonal(dref, np.inf)
        ref_knn = np.sort(np.sort(dref, axis=1)[:, :10].mean(1))
        X = (st.L[k]["X"] - mu) / sd
        s = np.empty(len(X))
        for i in range(0, len(X), 20000):
            d = cdist(X[i:i + 20000], ref)
            s[i:i + 20000] = np.partition(d, 10, axis=1)[:, :10].mean(1)
        out[k] = dict(knn=s, pct=np.searchsorted(ref_knn, s, side="right") / len(ref_knn))
    return out


def ood_table(T, pred, ood) -> pd.DataFrame:
    rows = []
    m, f = LEADER
    for k in P.LANDSCAPES:
        y, p = T[k]["y"], pred[(m, f, k)]
        flag = p["cal"] >= p["thr"]
        o = ood[k]["pct"]
        q = np.unique(np.quantile(o, [0, .2, .4, .6, .8, 1.0]))
        bins = np.clip(np.searchsorted(q, o, side="right") - 1, 0, len(q) - 2)
        for b in range(len(q) - 1):
            s = bins == b
            yy = y[s]
            ap = float(average_precision_score(yy, p["raw"][s])) if 0 < yy.sum() < len(yy) else np.nan
            rows.append(dict(landscape=k, model=m, feature_set=f, ood_quintile=b + 1, ood_percentile_low=q[b], ood_percentile_high=q[b + 1], units=int(s.sum()),
                             positives=int(yy.sum()), prevalence=float(yy.mean()), fn_rate=float((~flag[s] & yy).sum() / max(yy.sum(), 1)),
                             fp_rate=float((flag[s] & ~yy).sum() / max((~yy).sum(), 1)), pr_auc=ap, pr_lift=ap / yy.mean() if yy.mean() > 0 else np.nan,
                             status=P.OOD["status"]))
        rows.append(dict(landscape=k, model=m, feature_set=f, ood_quintile=0, units=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                         median_ood_percentile=float(np.median(o)), fraction_beyond_train_p99=float(np.mean(o > 0.99)),
                         pr_lift=float(average_precision_score(y, p["raw"]) / y.mean()), status=P.OOD["status"] + " (landscape-level row)"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ forensics
def forensics(T, pred, ood, fshift) -> pd.DataFrame:
    rows = []
    for k in P.FORENSICS["landscapes"]:
        y = T[k]["y"]
        for (m, f) in [LEADER, ("LOGISTIC_L2", "M"), ("LOGISTIC_L2", "ME")]:
            p = pred[(m, f, k)]
            flag = p["cal"] >= p["thr"]
            dec = pd.qcut(p["cal"], 10, labels=False, duplicates="drop")
            rel = pd.DataFrame(dict(d=dec, p=p["cal"], y=y)).groupby("d").agg(mean_pred=("p", "mean"), obs=("y", "mean"))
            blk = pd.DataFrame(dict(b=T[k]["b3"], fn=(~flag) & y, fp=flag & ~y))
            fnb, fpb = blk.groupby("b").fn.sum().sort_values(ascending=False), blk.groupby("b").fp.sum().sort_values(ascending=False)
            o = ood[k]["pct"]
            rows.append(dict(
                landscape=k, model=m, feature_set=f, prevalence=float(y.mean()), pr_auc=float(average_precision_score(y, p["raw"])),
                roc_auc=float(roc_auc_score(y, p["raw"])), raw_pred_median=float(np.median(p["raw"])), raw_pred_p99=float(np.percentile(p["raw"], 99)),
                cal_pred_mean=float(p["cal"].mean()), observed_rate=float(y.mean()), score_median_positives=float(np.median(p["raw"][y])),
                score_median_negatives=float(np.median(p["raw"][~y])), threshold=p["thr"], flagged_fraction=float(flag.mean()),
                recall=float((flag & y).sum() / y.sum()), precision=float((flag & y).sum() / max(flag.sum(), 1)),
                reliability_deciles=json.dumps({int(i): [round(a, 4), round(b, 4)] for i, (a, b) in rel.iterrows()}),
                fn_share_top10pct_blocks=float(fnb.head(max(1, len(fnb) // 10)).sum() / max(fnb.sum(), 1)),
                fp_share_top10pct_blocks=float(fpb.head(max(1, len(fpb) // 10)).sum() / max(fpb.sum(), 1)),
                median_ood_percentile_positives=float(np.median(o[y])), median_ood_percentile_negatives=float(np.median(o[~y])),
                median_ood_percentile_fn=float(np.median(o[(~flag) & y])) if ((~flag) & y).any() else np.nan,
                median_ood_percentile_fp=float(np.median(o[flag & ~y])) if (flag & ~y).any() else np.nan,
                top_shifted_predictors="; ".join(f"{r.predictor}:{r.std_median_diff:+.2f}" for r in
                                                 fshift[fshift.held_out == k].assign(a=lambda d: d.std_median_diff.abs()).sort_values("a", ascending=False).head(6).itertuples())))
    return pd.DataFrame(rows)
