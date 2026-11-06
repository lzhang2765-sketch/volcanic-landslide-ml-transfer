"""Stage 4E-A environmental representations (X-only, deployment-available landscape normalisation) and the limited LOLO screen
through the UNCHANGED Stage 4B nested code (LOGISTIC_L2 and RANDOM_FOREST only)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import stage4a_base as B
from . import stage4b as S
from . import stage4ea_e_prereg as P
from .schema import ROOT

MEG = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET"
CACHE = ROOT / "data/interim/_stage4ea_cache"
OUT = ROOT / "data/interim/_stage4ea_screen"
SHORT = {c: c.replace("e_chelsa_", "").replace("e_hansen_", "") for c in P.E_SOURCE}
RELATIVE_IN_E3 = ["e_chelsa_bio1_mean_annual_temp", "e_chelsa_bio4_temp_seasonality", "e_chelsa_bio13_wettest_month_precip", "e_chelsa_bio15_precip_seasonality",
                  "e_hansen_treecover2000_pct__sd"]


def load_e(lid: str) -> np.ndarray:
    B.firewall_check(lid)
    with np.load(MEG / f"{lid}.npz", allow_pickle=False) as z:
        return np.column_stack([z[c].astype(np.float64) for c in P.E_SOURCE])


def representations(X: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    """X: all units of ONE landscape (X only).  Returns name -> (matrix, column names)."""
    n = len(X)
    pct = np.column_stack([(rankdata(X[:, j], method="average") - 0.5) / n for j in range(X.shape[1])])
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    assert (iqr > 0).all(), "validity check violated"
    cen = (X - med) / iqr
    e3 = np.column_stack([cen[:, P.E_SOURCE.index(c)] for c in RELATIVE_IN_E3]
                         + [np.log10(X[:, P.E_SOURCE.index("e_chelsa_bio12_annual_precip")]), X[:, P.E_SOURCE.index("e_hansen_treecover2000_pct__mean")]])
    return {"E0_ABSOLUTE": (X, [f"{c}" for c in P.E_SOURCE]),
            "E1_WITHIN_LANDSCAPE_PERCENTILE": (pct, [f"e1_pct_{SHORT[c]}" for c in P.E_SOURCE]),
            "E2_LANDSCAPE_CENTERED_ROBUST": (cen, [f"e2_ctr_{SHORT[c]}" for c in P.E_SOURCE]),
            "E3_HYBRID_CONTEXT_RELATIVE": (e3, [f"e3_ctr_{SHORT[c]}" for c in RELATIVE_IN_E3] + ["e3_abs_log10_bio12_annual_precip", "e3_abs_treecover2000_pct__mean"])}


FS = {"M+E1": "E1_WITHIN_LANDSCAPE_PERCENTILE", "M+E2": "E2_LANDSCAPE_CENTERED_ROBUST", "M+E3": "E3_HYBRID_CONTEXT_RELATIVE"}


def run() -> dict:
    S.CACHE = CACHE  # never write into the Stage 4B cache
    d = S.Data()
    extra = None
    for lid in S.LANDSCAPES:
        reps = representations(load_e(lid))
        cols = sum((reps[FS[f]][1] for f in FS), [])
        mat = np.hstack([reps[FS[f]][0] for f in FS])
        d.L[lid]["X"] = np.hstack([d.L[lid]["X"], mat])
        extra = cols
    d.cols = d.cols + extra
    d.idx = {c: i for i, c in enumerate(d.cols)}
    for f, rep in FS.items():
        d.fs[f] = d.fs["M"] + [c for c in extra if c.startswith(("e1_", "e2_", "e3_")[list(FS).index(f)])]
        assert len(d.fs[f]) == 67
    S.ANALYSES["E_SCREEN"] = (S.LANDSCAPES, "ALL_VALID", "LANDSCAPE_BALANCED", list(FS), ["LOGISTIC_L2", "RANDOM_FOREST"])
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda m: (OUT / "progress.log").open("a").write(m + "\n")
    res, inner, meta = S.run_analysis(d, "E_SCREEN", log)
    o = dict(results=res, fold_meta=meta, feature_sets={f: d.fs[f] for f in FS})
    (OUT / "E_SCREEN_RESULTS.json").write_text(json.dumps(o, indent=1, sort_keys=True, default=str) + "\n")
    return o


if __name__ == "__main__":
    run()
