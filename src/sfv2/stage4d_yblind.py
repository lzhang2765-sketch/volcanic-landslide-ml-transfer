"""Stage 4D Y-BLIND analyses: domain shift (primary normalised energy statistic), per-predictor shift, landscape-identity audit.

The loader reads only predictor / identifier / block / coordinate arrays from the Stage 4A archives; it never reads the target column.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance

from . import stage4a_base as B
from . import stage4d_prereg as P
from .schema import ROOT

MEG_DIR = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET"
MANIFEST = ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv"
FORBIDDEN = ("y_positive", "target_tier", "target_source")


def groups() -> dict[str, list[str]]:
    m = pd.read_csv(MANIFEST, sep="\t")
    by = {r: m[(m.role == r) & m.predictor].column.tolist() for r in ("M", "E", "G")}
    return {"M": by["M"], "E": by["E"], "G": by["G"], "ME": by["M"] + by["E"], "MG": by["M"] + by["G"], "EG": by["E"] + by["G"],
            "MEG": by["M"] + by["E"] + by["G"]}


def load_x(lid: str, cols: list[str]) -> dict:
    """Y-blind loader: only the requested keys are decompressed from the column archive."""
    B.firewall_check(lid)
    keys = ["terrain_unit_id", "spatial_block_3000m", "spatial_block_6000m", "centroid_x", "centroid_y"] + cols
    assert not set(keys) & set(FORBIDDEN)
    with np.load(MEG_DIR / f"{lid}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in keys}
    d["X"] = np.column_stack([d.pop(c).astype(np.float64) for c in cols])
    return d


def hash_rank(ids: np.ndarray, salt: str) -> np.ndarray:
    return np.argsort(np.array([hashlib.sha256(f"{i}|{salt}".encode()).hexdigest() for i in ids]), kind="stable")


def balanced_scaler(train_X: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean([x.mean(0) for x in train_X], axis=0)
    var = np.mean([((x - mu) ** 2).mean(0) for x in train_X], axis=0)
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    return mu, sd


def energy_H(a: np.ndarray, b: np.ndarray) -> float:
    xy = cdist(a, b).mean()
    xx = cdist(a, a).mean()
    yy = cdist(b, b).mean()
    return float((2 * xy - xx - yy) / (2 * xy)) if xy > 0 else 0.0


def sliced_w1(a: np.ndarray, b: np.ndarray, n_dir: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_dir, a.shape[1]))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(a @ d, b @ d) for d in v]))


class Store:
    def __init__(self):
        self.g = groups()
        self.cols = self.g["MEG"]
        self.L = {lid: load_x(lid, self.cols) for lid in P.LANDSCAPES}
        for lid, d in self.L.items():
            d["rank"] = hash_rank(d["terrain_unit_id"], "stage4d")


def fold_samples(st: Store, k: str, seed: int | None = None):
    train = [l for l in P.LANDSCAPES if l != k]
    mu, sd = balanced_scaler([st.L[l]["X"] for l in train])
    def pick(l, n):
        d = st.L[l]
        if seed is None:
            return d["rank"][:n]
        return np.random.default_rng(seed + P.LANDSCAPES.index(l)).choice(len(d["X"]), size=n, replace=False)
    a = (st.L[k]["X"][pick(k, 2000)] - mu) / sd
    b = np.vstack([(st.L[l]["X"][pick(l, 400)] - mu) / sd for l in train])
    return a, b, mu, sd


def domain_shift(st: Store) -> pd.DataFrame:
    rows = []
    idx = {c: i for i, c in enumerate(st.cols)}
    for k in P.LANDSCAPES:
        a, b, _, _ = fold_samples(st, k)
        reps = [fold_samples(st, k, P.SEED + 1000 * r) for r in range(20)]
        for gname, gcols in st.g.items():
            ci = [idx[c] for c in gcols]
            H = energy_H(a[:, ci], b[:, ci])
            Hr = [energy_H(x[:, ci], y[:, ci]) for x, y, _, _ in reps]
            rows.append(dict(held_out=k, group=gname, n_predictors=len(ci), energy_H_primary=H, energy_H_subsample_mean=float(np.mean(Hr)),
                             energy_H_subsample_sd=float(np.std(Hr)), sliced_w1_secondary=sliced_w1(a[:, ci], b[:, ci], 100, P.SEED),
                             n_heldout_sample=len(a), n_train_sample=len(b), scaling="landscape-balanced training z-score (held-out excluded)"))
    return pd.DataFrame(rows)


def feature_shift(st: Store) -> pd.DataFrame:
    rows = []
    for k in P.LANDSCAPES:
        a, b, mu, sd = fold_samples(st, k)
        for i, c in enumerate(st.cols):
            x, y = a[:, i], b[:, i]
            iqr = np.subtract(*np.percentile(y, [75, 25])) or np.std(y) or 1.0
            binary = c.startswith("g_")
            rows.append(dict(held_out=k, predictor=c, group="M" if not c.startswith(("e_", "g_")) else c[0].upper(),
                             std_median_diff=float((np.median(x) - np.median(y)) / iqr), wasserstein1_z=float(wasserstein_distance(x, y)),
                             ks_statistic=float(ks_2samp(x, y).statistic),
                             binary_prevalence_diff=float((x * sd[i] + mu[i]).mean() - (y * sd[i] + mu[i]).mean()) if binary else np.nan,
                             heldout_outside_train_range_fraction=float(((x < y.min()) | (x > y.max())).mean())))
    return pd.DataFrame(rows)


def identity_audit(st: Store) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    X, y, grp = [], [], []
    for i, l in enumerate(P.LANDSCAPES):
        d = st.L[l]
        r = hash_rank(d["terrain_unit_id"], "stage4d-id")[:3000]
        X.append(d["X"][r]); y.append(np.full(len(r), i)); grp.append(np.array([f"{l}|{b}" for b in d["spatial_block_6000m"][r]]))
    X, y, grp = np.vstack(X), np.concatenate(y), np.concatenate(grp)
    idx = {c: i for i, c in enumerate(st.cols)}
    folds = list(GroupKFold(n_splits=5).split(X, y, grp))

    def cv(ci):
        proba = np.zeros((len(y), 6))
        for tr, te in folds:
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=P.SEED).fit(X[tr][:, ci], y[tr])
            proba[te] = m.predict_proba(X[te][:, ci])
        return float(balanced_accuracy_score(y, proba.argmax(1))), float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    rows = []
    for gname in ("M", "E", "G", "MEG"):
        ba, auc = cv([idx[c] for c in st.g[gname]])
        rows.append(dict(scope="GROUP", name=gname, n_predictors=len(st.g[gname]), balanced_accuracy=ba, macro_ovr_auc=auc, chance_balanced_accuracy=1 / 6))
    for c in st.cols:
        ba, auc = cv([idx[c]])
        rows.append(dict(scope="PREDICTOR", name=c, n_predictors=1, balanced_accuracy=ba, macro_ovr_auc=auc, chance_balanced_accuracy=1 / 6))
    return pd.DataFrame(rows)
