"""Stage 4D missing ablation: MG (= M + G) through the UNCHANGED Stage 4B nested LOLO code (same folds, sampling, weights, grids, seed,
calibration, threshold).  Only the feature set is new; the cache is redirected so the authoritative Stage 4B cache is never written."""
from __future__ import annotations

import json

from . import stage4b as S
from .schema import ROOT

CACHE = ROOT / "data/interim/_stage4d_cache"
OUT = ROOT / "data/interim/_stage4d_mg"


def run() -> dict:
    S.CACHE = CACHE  # never write into data/interim/_stage4b_cache
    d = S.Data()
    d.fs["MG"] = [c for c in d.fs["M"]] + [c for c in d.cols if c.startswith("g_")]
    assert len(d.fs["MG"]) == len(d.fs["M"]) + 2
    S.ANALYSES["PRIMARY_MG"] = (S.LANDSCAPES, "ALL_VALID", "LANDSCAPE_BALANCED", ["MG"], list(S.GRIDS))
    log = lambda m: (OUT / "progress.log").open("a").write(m + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    res, inner, meta = S.run_analysis(d, "PRIMARY_MG", log)
    o = dict(results=res, inner=inner, fold_meta=meta, feature_set_MG=d.fs["MG"])
    (OUT / "MG_RESULTS.json").write_text(json.dumps(o, indent=1, sort_keys=True, default=str) + "\n")
    return o


if __name__ == "__main__":
    run()
