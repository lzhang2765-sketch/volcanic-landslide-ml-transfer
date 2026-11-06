"""Stage 4E-A X-only audits of E representations: landscape identity (Stage 4D architecture), information retention, Atitlan range test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold

from . import stage4d_yblind as Y
from . import stage4ea_e_prereg as P
from . import stage4ea_screen as SC

LS = Y.P.LANDSCAPES
REPS = ["E0_ABSOLUTE", "E1_WITHIN_LANDSCAPE_PERCENTILE", "E2_LANDSCAPE_CENTERED_ROBUST", "E3_HYBRID_CONTEXT_RELATIVE"]


def build():
    st = Y.Store()
    reps, elev = {}, {}
    mi = st.cols.index("elev_m_30m__mean")
    for l in LS:
        reps[l] = SC.representations(SC.load_e(l))
        elev[l] = st.L[l]["X"][:, mi]
    return st, reps, elev


def identity(st, reps) -> pd.DataFrame:
    idx, Ms, Es, y, grp = [], [], {r: [] for r in REPS}, [], []
    mcols = [st.cols.index(c) for c in st.g["M"]]
    for i, l in enumerate(LS):
        r = Y.hash_rank(st.L[l]["terrain_unit_id"], "stage4d-id")[:3000]
        Ms.append(st.L[l]["X"][r][:, mcols])
        for rep in REPS:
            Es[rep].append(reps[l][rep][0][r])
        y.append(np.full(len(r), i))
        grp.append(np.array([f"{l}|{b}" for b in st.L[l]["spatial_block_6000m"][r]]))
    M, y, grp = np.vstack(Ms), np.concatenate(y), np.concatenate(grp)
    folds = list(GroupKFold(n_splits=5).split(M, y, grp))
    rows = []
    for rep in REPS:
        E = np.vstack(Es[rep])
        for scope, X in (("E_ONLY", E), ("M_PLUS_E", np.hstack([M, E]))):
            pred = np.zeros(len(y), int)
            for tr, te in folds:
                m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=Y.P.SEED).fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            cm = confusion_matrix(y, pred, labels=range(6))
            rec = recall_score(y, pred, labels=range(6), average=None)
            rows.append(dict(representation=rep, scope=scope, balanced_accuracy=float(balanced_accuracy_score(y, pred)), chance=1 / 6,
                             **{f"recall_{l[:4]}": float(v) for l, v in zip(LS, rec)}, confusion_matrix=";".join(",".join(map(str, r)) for r in cm)))
    ba = {(r["representation"], r["scope"]): r["balanced_accuracy"] for r in rows}
    rows.append(dict(representation="STRICT_TRAINING_REFERENCE_SENSITIVITY", scope="E_ONLY", balanced_accuracy=ba[("E0_ABSOLUTE", "E_ONLY")], chance=1 / 6,
                     confusion_matrix="identical to E0: a common affine/monotone transform of absolute values cannot change tree-based identification"))
    return pd.DataFrame(rows)


def retention(reps, elev) -> pd.DataFrame:
    rows = []
    for rep in REPS:
        cols = reps[LS[0]][rep][1]
        for j, c in enumerate(cols):
            per = []
            for l in LS:
                x = reps[l][rep][0][:, j]
                src = SC.load_e(l)
                srcj = P.E_SOURCE.index(next(s for s in P.E_SOURCE if SC.SHORT[s] in c or s == c))
                per.append(dict(l=l, iqr=np.subtract(*np.percentile(x, [75, 25])), nunique=len(np.unique(x)), tie=pd.Series(x).value_counts(normalize=True).iloc[0],
                                rho_src=spearmanr(x, src[:, srcj]).correlation, rho_elev=spearmanr(x, elev[l]).correlation, mean=x.mean(), var=x.var(), n=len(x)))
            df = pd.DataFrame(per)
            grand = np.average(df["mean"], weights=np.full(len(df), 1 / len(df)))
            between = float(np.mean((df["mean"] - grand) ** 2))
            within = float(np.mean(df["var"]))
            rows.append(dict(representation=rep, column=c, nonconstant_landscapes=int((df.iqr > 0).sum()), min_distinct_values=int(df["nunique"].min()),
                             max_largest_tie_share=float(df.tie.max()), min_within_landscape_spearman_with_absolute=float(df.rho_src.min()),
                             between_landscape_variance_fraction=between / (between + within) if between + within > 0 else np.nan,
                             median_within_landscape_spearman_with_elevation=float(df.rho_elev.median()),
                             max_abs_within_landscape_spearman_with_elevation=float(df.rho_elev.abs().max()),
                             passes_rule_E=bool((df.iqr > 0).all() and (df.tie <= 0.5).all())))
    return pd.DataFrame(rows)


def atitlan_range(reps) -> pd.DataFrame:
    rows = []
    k = "ATITLAN_STAN_2005"
    for rep in REPS:
        tr = np.vstack([reps[l][rep][0] for l in LS if l != k])
        x = reps[k][rep][0]
        lo, hi = tr.min(0), tr.max(0)
        out = (x < lo) | (x > hi)
        for j, c in enumerate(reps[k][rep][1]):
            rows.append(dict(representation=rep, column=c, atitlan_fraction_outside_training_range=float(out[:, j].mean())))
        rows.append(dict(representation=rep, column="ANY_COLUMN", atitlan_fraction_outside_training_range=float(out.any(1).mean())))
    return pd.DataFrame(rows)
