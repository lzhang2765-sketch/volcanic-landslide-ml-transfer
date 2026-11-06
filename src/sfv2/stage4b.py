"""Stage 4B core: pre-registered development-only nested leave-one-landscape-out (LOLO) benchmark.

Everything that can influence a result is fixed in PREREG before any model is fitted.  The outer held-out landscape never
influences scaling, hyperparameters, calibration, threshold, weighting or sampling: all of these are computed from the outer
training landscapes only (inner leave-one-training-landscape-out).  Outer metrics always use EVERY row of the held-out landscape.
"""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from . import stage4a_base as B
from .build_stage4a import read_columns
from .schema import ROOT

SEED = 20261006
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
MEG_DIR = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET"
MANIFEST = ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv"
CACHE = ROOT / "data/interim/_stage4b_cache"
NEG_PER_LANDSCAPE = 20000
TIE_TOL = 0.005
MIN_CAL_POSITIVES = 50
EPS = 1e-7

GRIDS = {  # ordered simplest -> most complex (ties resolve to the earliest entry)
    "LOGISTIC_L2": [dict(C=0.01), dict(C=0.1), dict(C=1.0)],
    "RANDOM_FOREST": [dict(min_samples_leaf=100, max_features="sqrt"), dict(min_samples_leaf=20, max_features="sqrt"),
                      dict(min_samples_leaf=100, max_features=0.33), dict(min_samples_leaf=20, max_features=0.33)],
    "HIST_GRADIENT_BOOSTING": [dict(learning_rate=0.05, max_leaf_nodes=15), dict(learning_rate=0.1, max_leaf_nodes=15),
                               dict(learning_rate=0.05, max_leaf_nodes=31), dict(learning_rate=0.1, max_leaf_nodes=31)],
}
FIXED = {
    "LOGISTIC_L2": dict(penalty="l2", solver="lbfgs", max_iter=5000),
    "RANDOM_FOREST": dict(n_estimators=100, max_depth=None, bootstrap=True, max_samples=0.5, n_jobs=8, random_state=SEED),
    "HIST_GRADIENT_BOOSTING": dict(max_iter=250, min_samples_leaf=50, l2_regularization=1.0, early_stopping=False, random_state=SEED),
}
MODEL_SIMPLICITY = ["LOGISTIC_L2", "HIST_GRADIENT_BOOSTING", "RANDOM_FOREST"]
FS_SIMPLICITY = ["M", "ME", "MEG"]
ANALYSES = {  # name -> (universe, population, weighting, feature sets, models)
    "PRIMARY": (LANDSCAPES, "ALL_VALID", "LANDSCAPE_BALANCED", ["M", "ME", "MEG", "MEG_NO_BIO15"], list(GRIDS)),
    "ROW_WEIGHTING": (LANDSCAPES, "ALL_VALID", "ROW", ["MEG"], list(GRIDS)),
    "ATITLAN_EXCLUDED": ([l for l in LANDSCAPES if l != "ATITLAN_STAN_2005"], "ALL_VALID", "LANDSCAPE_BALANCED", ["M", "ME", "MEG"], list(GRIDS)),
    "STRICT_DEM_QUALITY": ([l for l in LANDSCAPES if l != "ATITLAN_STAN_2005"], "STRICT", "LANDSCAPE_BALANCED", ["M", "MEG"], list(GRIDS)),
    "POLYGON_ONLY": ([l for l in LANDSCAPES if l != "KUMAMOTO_ASO_2016"], "ALL_VALID", "LANDSCAPE_BALANCED", ["M", "MEG"], list(GRIDS)),
}


def feature_sets() -> dict[str, list[str]]:
    m = pd.read_csv(MANIFEST, sep="\t")
    by = {r: m[(m.role == r) & m.predictor].column.tolist() for r in ("M", "E", "G")}
    meg = by["M"] + by["E"] + by["G"]
    return {"M": by["M"], "ME": by["M"] + by["E"], "MEG": meg, "MEG_NO_BIO15": [c for c in meg if c != "e_chelsa_bio15_precip_seasonality"]}


def prereg() -> dict:
    return dict(
        stage="4B", status="PRE-REGISTERED BEFORE ANY MODEL FIT", seed=SEED, sklearn_version=sklearn.__version__,
        models={"NULL": "constant = landscape-balanced mean training prevalence", "LOGISTIC_L2": "sklearn.linear_model.LogisticRegression",
                "RANDOM_FOREST": "sklearn.ensemble.RandomForestClassifier", "HIST_GRADIENT_BOOSTING": "sklearn.ensemble.HistGradientBoostingClassifier"},
        grids=GRIDS, fixed_parameters=FIXED, feature_sets={k: len(v) for k, v in feature_sets().items()},
        outer_validation="Leave-one-landscape-out; held-out = all rows of one landscape", outer_folds=LANDSCAPES,
        inner_validation="Leave-one-training-landscape-out inside the outer training landscapes; inner held-out evaluated on all its rows",
        inner_selection="max mean inner-landscape PR-AUC (average precision); ties within 0.005 -> earliest (simplest) grid entry",
        training_sampling=f"per training landscape: all positives + {NEG_PER_LANDSCAPE} negatives chosen by ascending sha256(terrain_unit_id+'|stage4b') (fold-independent)",
        weighting={"LANDSCAPE_BALANCED (primary)": "each training landscape total weight 1/K; within a landscape the positive share of weight equals its true "
                                                   "population prevalence (sampling-corrected); no extra class up-weighting (avoids exploding positive weights)",
                   "ROW (sensitivity)": "each sampled row weighted by its population inverse sampling fraction (every population row counts once)",
                   "normalisation": "weights rescaled to mean 1 over the training sample"},
        preprocessing="Logistic: z-score of continuous predictors from the training sample only (binary g_* untouched); trees: none",
        calibration=f"Platt scaling (logistic on logit p) fitted on pooled inner out-of-fold predictions of the selected configuration, landscape-balanced weights; "
                    f"skipped (raw kept) if inner positives < {MIN_CAL_POSITIVES}",
        threshold="maximise landscape-balanced weighted F1 on pooled inner out-of-fold calibrated predictions; applied unchanged to the held-out landscape",
        metrics=["PR-AUC (primary)", "ROC-AUC", "Brier", "log loss", "precision/recall/F1 at training threshold", "calibration intercept and slope"],
        aggregation="median across held-out landscapes (primary); mean, min, max, IQR; pooled rows only as secondary",
        analyses={k: dict(universe=v[0], population=v[1], weighting=v[2], feature_sets=v[3], models=v[4]) for k, v in ANALYSES.items()},
        leader_rule=("Among PRIMARY {LOGISTIC_L2, RANDOM_FOREST, HIST_GRADIENT_BOOSTING} x {M, ME, MEG}: highest median PR-AUC; candidates within 0.01 of the "
                     "best are compared on worst-landscape PR-AUC, then median |calibration slope - 1|, then simplicity (LR < HGB < RF; M < ME < MEG). "
                     "MEG_NO_BIO15 is a sensitivity and not eligible. Label DEVELOPMENT_LEADER (NONFINAL)."),
        interpretation="permutation importance (PR-AUC drop) on a deterministic held-out subsample (all positives + 20,000 negatives), 3 repeats, "
                       "groups M/E/G and individual predictors; model-native importance / standardised coefficients where available",
        material_change_rule=("A sensitivity MATERIALLY changes the conclusion if it changes (a) the best model family by median MEG PR-AUC, (b) the sign "
                              "of the median Delta_MEG (MEG - M) for the leader family, or (c) the leader family's median MEG PR-AUC by more than 0.02 "
                              "relative to the matching comparator (STRICT vs ATITLAN_EXCLUDED ALL_VALID on the same 5 landscapes; others vs PRIMARY)"),
        spatial_consistency="Pearson correlation between each unit's raw probability and the mean of its 8 neighbouring units (no Y used)",
        lockbox_firewall=B.PROTECTED_LOCKBOX_CANDIDATES, h_used=False,
    )


# ------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self):
        fs = feature_sets()
        self.fs = fs
        self.cols = fs["MEG"]
        self.idx = {c: i for i, c in enumerate(self.cols)}
        self.L = {}
        for lid in LANDSCAPES:
            B.firewall_check(lid)
            df = read_columns(MEG_DIR / f"{lid}.npz")
            ids = df.terrain_unit_id.to_numpy()
            rank = np.argsort(np.array([hashlib.sha256(f"{i}|stage4b".encode()).hexdigest() for i in ids]), kind="stable")
            order = np.empty(len(ids), dtype=np.int64)
            order[rank] = np.arange(len(ids))
            self.L[lid] = dict(X=df[self.cols].to_numpy(np.float64), y=df.y_positive.to_numpy(bool), ids=ids, hash_order=order,
                               strict=(df.population_strict_dem_quality.to_numpy() == "IN"), cx=df.centroid_x.to_numpy(), cy=df.centroid_y.to_numpy())

    def rows(self, lid: str, pop: str) -> np.ndarray:
        n = len(self.L[lid]["y"])
        return np.arange(n) if pop == "ALL_VALID" else np.flatnonzero(self.L[lid]["strict"])

    def sample(self, lid: str, pop: str) -> np.ndarray:
        r = self.rows(lid, pop)
        y = self.L[lid]["y"][r]
        neg = r[~y]
        neg = neg[np.argsort(self.L[lid]["hash_order"][neg], kind="stable")[:NEG_PER_LANDSCAPE]]
        return np.sort(np.concatenate([r[y], neg]))


def weights(d: Data, train: list[str], pop: str, scheme: str) -> tuple[list[np.ndarray], dict]:
    ws, audit = [], {}
    K = len(train)
    for lid in train:
        r, s = d.rows(lid, pop), d.sample(lid, pop)
        y_full, y_s = d.L[lid]["y"][r], d.L[lid]["y"][s]
        Np, Nn, n_p, n_n = int(y_full.sum()), int((~y_full).sum()), int(y_s.sum()), int((~y_s).sum())
        if scheme == "LANDSCAPE_BALANCED":
            wp, wn = (1 / K) * (Np / len(r)) / n_p, (1 / K) * (Nn / len(r)) / n_n
        else:
            wp, wn = Np / n_p, Nn / n_n
        ws.append(np.where(y_s, wp, wn))
        audit[lid] = dict(pop_rows=len(r), pop_pos=Np, sampled_pos=n_p, sampled_neg=n_n, w_pos_raw=wp, w_neg_raw=wn)
    tot = np.concatenate(ws)
    scale = len(tot) / tot.sum()
    ws = [w * scale for w in ws]
    for lid, w in zip(train, ws):
        y_s = d.L[lid]["y"][d.sample(lid, pop)]
        audit[lid].update(total_weight=float(w.sum()), positive_weight=float(w[y_s].sum()), negative_weight=float(w[~y_s].sum()),
                          w_pos=float(w[y_s][0]), w_neg=float(w[~y_s][0]))
    allw = np.concatenate(ws)
    return ws, dict(per_landscape=audit, max_row_weight=float(allw.max()), min_row_weight=float(allw.min()), scheme=scheme)


def make_model(name: str, params: dict):
    if name == "LOGISTIC_L2":
        return LogisticRegression(**FIXED[name], **params)
    if name == "RANDOM_FOREST":
        return RandomForestClassifier(**FIXED[name], **params)
    return HistGradientBoostingClassifier(**FIXED[name], **params)


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def fit(d: Data, train: list[str], fs: str, model: str, params: dict, pop: str, scheme: str):
    cols = [d.idx[c] for c in d.fs[fs]]
    Xs = [d.L[l]["X"][d.sample(l, pop)][:, cols] for l in train]
    ys = [d.L[l]["y"][d.sample(l, pop)] for l in train]
    w, waudit = weights(d, train, pop, scheme)
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(w)
    prep = None
    if model == "LOGISTIC_L2":
        binary = np.array([d.fs[fs][i].startswith("g_") for i in range(len(cols))])
        mu, sd = X.mean(0), X.std(0)
        mu[binary], sd[binary] = 0.0, 1.0
        sd[sd == 0] = 1.0
        prep = dict(mean=mu.tolist(), sd=sd.tolist(), columns=d.fs[fs], fitted_on="training sample only")
        X = (X - mu) / sd
    est = make_model(model, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(X, y, sample_weight=w)
    return est, cols, prep, waudit, dict(train_rows=len(y), train_pos=int(y.sum()))


def predict(d: Data, est, cols, prep, lid: str, pop: str) -> np.ndarray:
    X = d.L[lid]["X"][d.rows(lid, pop)][:, cols]
    if prep is not None:
        X = (X - np.asarray(prep["mean"])) / np.asarray(prep["sd"])
    return est.predict_proba(X)[:, 1]


def cached_predictions(d: Data, train: list[str], fs: str, model: str, params: dict, pop: str, scheme: str, need: list[str]) -> dict:
    """Fit once per training set and cache predictions for every landscape outside it (inner-fold reuse is leakage-free)."""
    key = _key(sorted(train), fs, model, params, pop, scheme)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    if all((CACHE / f"{key}__{l}.npy").exists() for l in need):
        return {l: np.load(CACHE / f"{key}__{l}.npy") for l in need}
    est, cols, prep, _, _ = fit(d, train, fs, model, params, pop, scheme)
    for l in LANDSCAPES:
        if l in train or (pop == "STRICT" and not d.L[l]["strict"].any()):
            continue
        p = predict(d, est, cols, prep, l, pop).astype(np.float64)
        np.save(CACHE / f"{key}__{l}.npy", p)
        out[l] = p
    return {l: out[l] for l in need}


# ------------------------------------------------------------------------------------------------ calibration / metrics
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict | None:
    if y.sum() < MIN_CAL_POSITIVES:
        return None
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p)[:, None], y, sample_weight=w)
    return dict(a=float(lr.intercept_[0]), b=float(lr.coef_[0, 0]))


def apply_platt(p: np.ndarray, c: dict | None) -> np.ndarray:
    return p if c is None else 1 / (1 + np.exp(-(c["a"] + c["b"] * _logit(p))))


def best_threshold(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(-p, kind="stable")
    ps, ys, ws = p[o], y[o], w[o]
    tp, fp = np.cumsum(ws * ys), np.cumsum(ws * ~ys)
    P = (ws * ys).sum()
    f1 = 2 * tp / (tp + fp + P)
    last = np.r_[ps[1:] != ps[:-1], True]  # only evaluate at distinct score boundaries
    i = int(np.argmax(np.where(last, f1, -1)))
    return float(ps[i])


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = _logit(p)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(z[:, None], y)
    a = 0.0  # calibration-in-the-large: intercept with z as fixed offset (1-D Newton)
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + z)))
        g, h = (y - q).sum(), (q * (1 - q)).sum()
        a += g / max(h, 1e-12)
    return float(a), float(lr.coef_[0, 0])


def metrics(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, thr: float | None) -> dict:
    out = dict(n=len(y), positives=int(y.sum()), negatives=int((~y).sum()), prevalence=float(y.mean()),
               pr_auc=float(average_precision_score(y, p_raw)), roc_auc=float(roc_auc_score(y, p_raw)),
               brier_raw=float(brier_score_loss(y, p_raw)), log_loss_raw=float(log_loss(y, np.clip(p_raw, EPS, 1 - EPS), labels=[0, 1])),
               brier=float(brier_score_loss(y, p_cal)), log_loss=float(log_loss(y, np.clip(p_cal, EPS, 1 - EPS), labels=[0, 1])))
    out["pr_auc_lift"] = out["pr_auc"] / out["prevalence"]
    if np.ptp(p_cal) > 0:
        out["cal_intercept"], out["cal_slope"] = calibration_slope_intercept(p_cal, y)
    else:
        out["cal_intercept"], out["cal_slope"] = float("nan"), float("nan")
    if thr is not None:
        pred = p_cal >= thr
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.update(threshold=thr, precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return out


# ------------------------------------------------------------------------------------------------ nested LOLO
def run_analysis(d: Data, name: str, log=print) -> tuple[list[dict], list[dict], dict]:
    universe, pop, scheme, fsets, models = ANALYSES[name]
    results, inner_rows, fold_meta = [], [], {}
    for k in universe:
        train = [l for l in universe if l != k]
        yk = d.L[k]["y"][d.rows(k, pop)]
        prev_train = float(np.mean([d.L[l]["y"][d.rows(l, pop)].mean() for l in train]))
        null = np.full(len(yk), prev_train)
        results.append(dict(analysis=name, held_out=k, model="NULL", feature_set="NONE", params="{}", calibrated=False,
                            **metrics(yk, null, null, None)))
        for fs in fsets:
            for model in models:
                scores = []
                for gi, params in enumerate(GRIDS[model]):
                    per = {}
                    for j in train:
                        pj = cached_predictions(d, [l for l in train if l != j], fs, model, params, pop, scheme, [j])[j]
                        per[j] = float(average_precision_score(d.L[j]["y"][d.rows(j, pop)], pj))
                    scores.append((float(np.mean(list(per.values()))), gi, per))
                    inner_rows.append(dict(analysis=name, held_out=k, model=model, feature_set=fs, grid_index=gi, params=json.dumps(params),
                                           inner_mean_pr_auc=scores[-1][0], **{f"inner_{j}": v for j, v in per.items()}))
                best = max(s[0] for s in scores)
                gi = min(s[1] for s in scores if s[0] >= best - TIE_TOL)
                params = GRIDS[model][gi]
                # calibration + threshold from inner out-of-fold predictions of the selected configuration (training landscapes only)
                ps, ys, ws = [], [], []
                for j in train:
                    pj = cached_predictions(d, [l for l in train if l != j], fs, model, params, pop, scheme, [j])[j]
                    yj = d.L[j]["y"][d.rows(j, pop)]
                    ps.append(pj); ys.append(yj); ws.append(np.full(len(yj), 1.0 / (len(yj) * len(train))))
                p_in, y_in, w_in = np.concatenate(ps), np.concatenate(ys), np.concatenate(ws)
                cal = fit_platt(p_in, y_in, w_in)
                thr = best_threshold(apply_platt(p_in, cal), y_in, w_in)
                pk = cached_predictions(d, train, fs, model, params, pop, scheme, [k])[k]
                pk_cal = apply_platt(pk, cal)
                results.append(dict(analysis=name, held_out=k, model=model, feature_set=fs, params=json.dumps(params), calibrated=cal is not None,
                                    calibrator=json.dumps(cal), inner_best_mean_pr_auc=best, **metrics(yk, pk, pk_cal, thr)))
                fold_meta[f"{name}|{k}|{fs}|{model}"] = dict(params=params, calibrator=cal, threshold=thr, inner_scores=[s[0] for s in scores],
                                                             train=train, outer_prediction_key=_key(sorted(train), fs, model, params, pop, scheme))
                log(f"{name} {k} {fs} {model} done")
    return results, inner_rows, fold_meta
