"""Stage 4E-B builder.  The preregistration phase is deliberately usable without XGBoost."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from . import stage4b as S4B
from . import stage4eb as E
from . import stage4eb_prereg as P
from .raw_store import sha256 as project_sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4e_b"
PREREG_FREEZE = OUT / "STAGE4EB_PREREGISTRATION_FREEZE.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _tsv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


def prereg_object(frozen_utc: str) -> dict:
    return dict(stage=P.STAGE, status="FROZEN BEFORE ANY STAGE 4E-B MODEL FIT", frozen_utc=frozen_utc,
                v1_freeze_sha256=P.V1_FREEZE_SHA256, input_checkpoints=P.INPUT_CHECKPOINTS,
                landscapes=P.LANDSCAPES, dataset=dict(rows=835846, positives=8793, negatives=827053),
                features=dict(primary=P.PRIMARY_FEATURE_SET, primary_columns=60, historical_controls=P.HISTORICAL_CONTROLS,
                              E_columns=7, G_columns=2, h_used=False, no_feature_engineering=P.NO_FEATURE_ENGINEERING),
                models=P.MODELS, complexity_order=P.COMPLEXITY_ORDER,
                historical_grids=S4B.GRIDS, historical_fixed=S4B.FIXED,
                xgboost=dict(version=P.XGBOOST_VERSION, fixed=P.XGBOOST_FIXED, configurations=P.XGBOOST_GRID,
                             isolation="Science_Fair_ML_V2/.ml_site only; shared .venv unchanged"),
                validation=P.VALIDATION, metrics=P.METRICS, bootstrap=P.BOOTSTRAP, selection=P.SELECTION,
                catastrophic_guard=P.CATASTROPHIC_GUARD, cotabato=P.COTABATO, controls=P.CONTROLS,
                calibration=P.CALIBRATION, importance=P.IMPORTANCE, stability=P.STABILITY,
                complexity=P.COMPLEXITY, ensemble=P.ENSEMBLE, ood=P.OOD, h_policy=P.H_POLICY,
                lockbox=P.LOCKBOX, random_seeds=dict(stage4eb=P.SEED, historical_stage4b=P.STAGE4B_SEED),
                explicit_prohibitions=["no lockbox", "no MEGH", "no Gen-4", "no Gen-3/H-gate change",
                                       "no new M/E/G", "no E1/E2/E3", "no post-result rule change", "no ensemble build"])


def prereg_markdown(o: dict) -> str:
    return f"""# Stage 4E-B preregistered cross-landscape model selection and transfer robustness

**Status:** {o['status']}  
**Frozen UTC:** {o['frozen_utc']}  
**Primary representation:** M (the frozen 60-column Stage 4A morphology set)  
**Independent transfer sample:** six development landscapes, not 835,846 independent landscapes

## Question

Does additional model complexity improve whole-volcano transfer when the predictor representation is held fixed?

## Models and features

Primary candidates are Logistic L2, Random Forest, HistGradientBoosting and XGBoost, all with M. The first three reuse authoritative Stage 4B predictions. XGBoost uses eight bounded configurations stored verbatim in the JSON and configuration table. ME and MEG are historical controls only; no E1/E2/E3 and no H.

## Validation and training

Outer validation leaves one whole landscape out. Inner validation leaves one of the five outer-training landscapes out. Every positive and at most 20,000 hash-selected negatives per training landscape are used with Stage 4B landscape-balanced, sampling-corrected weights. The outer landscape cannot affect tuning, sampling, weighting, thresholds or calibration.

## Metrics and uncertainty

Primary ranking is held-out PR-AUC interpreted as PR lift over each landscape's EQ1 prevalence. Summaries are landscape-level. A paired 3 km spatial-block bootstrap uses 2,000 replicates. Pooled terrain-unit performance is not the primary metric.

## Frozen selection rule

`ROBUST_SCORE = 0.70 × log(MEDIAN_LIFT) + 0.30 × log(WORST_LIFT)`.

Candidates must pass the numerical catastrophic-failure guard in the JSON. The raw winner has the greatest score. Logistic is selected instead when the complex winner's score advantage is within 5% of the Logistic score, or when its median PR-AUC gain is no more than 0.002 and the paired-bootstrap interval for the median gain includes zero. Exact ties use Logistic < Random Forest < HistGradientBoosting < XGBoost.

## Cotabato, calibration and controls

Cotabato EQ1 remains primary and included. Cumulative EQ3/EQ4 labels are sensitivity-only and may be scored only after the primary decision is frozen. Calibration is reported but cannot select the ranking architecture. XGBoost ME/MEG are secondary historical-control fits performed only after the primary selection freeze.

## Interpretation and prohibited actions

Permutation importance and coefficient summaries cannot alter features. Ensemble work is a correlation/complementarity diagnostic only, with no weights or averaging. No future lockbox, MEGH, Gen-4, H-gate change, new predictor, post-result feature engineering or rule alteration is permitted.
"""


def freeze_prereg() -> None:
    if PREREG_FREEZE.exists():
        fr = json.loads(PREREG_FREEZE.read_text())
        for rel, digest in fr["files"].items():
            if sha256(ROOT / rel) != digest:
                raise RuntimeError(f"STOP: frozen preregistration changed: {rel}")
        print(json.dumps(fr, indent=2))
        return
    # No result/cache may predate the freeze.
    if OUT.exists() and any(p.name not in {"STAGE4EB_PREREGISTRATION.json", "STAGE4EB_PREREGISTRATION.md", "STAGE4EB_MODEL_CONFIGURATIONS.tsv"} for p in OUT.iterdir()):
        raise RuntimeError("STOP: unexpected Stage 4E-B artifact exists before preregistration freeze")
    frozen = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    obj = prereg_object(frozen)
    pj, pm = OUT / "STAGE4EB_PREREGISTRATION.json", OUT / "STAGE4EB_PREREGISTRATION.md"
    _json(pj, obj)
    pm.write_text(prereg_markdown(obj))
    rows = []
    for model in ("LOGISTIC_L2", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING"):
        for i, cfg in enumerate(S4B.GRIDS[model]):
            rows.append(dict(model=model, feature_set="M", role="AUTHORITATIVE_STAGE4B_REUSE", configuration_id=i,
                             variable_parameters=json.dumps(cfg, sort_keys=True), fixed_parameters=json.dumps(S4B.FIXED[model], sort_keys=True)))
    for i, cfg in enumerate(P.XGBOOST_GRID):
        rows.append(dict(model="XGBOOST", feature_set="M", role="PRIMARY_NEW_FIT", configuration_id=i,
                         variable_parameters=json.dumps(cfg, sort_keys=True), fixed_parameters=json.dumps(P.XGBOOST_FIXED, sort_keys=True)))
    _tsv(OUT / "STAGE4EB_MODEL_CONFIGURATIONS.tsv", rows)
    module = ROOT / "src/sfv2/stage4eb_prereg.py"
    files = {str(p.relative_to(ROOT)): sha256(p) for p in (pj, pm, OUT / "STAGE4EB_MODEL_CONFIGURATIONS.tsv", module)}
    _json(PREREG_FREEZE, dict(stage=P.STAGE, frozen_utc=frozen, status="FROZEN_BEFORE_ANY_NEW_MODEL_FIT", files=files,
                              result_files_existing_before_freeze=[], xgboost_installed_before_freeze=False))
    print(json.dumps(json.loads(PREREG_FREEZE.read_text()), indent=2))


def verify_inputs() -> dict:
    E.verify_prereg()
    paths = {
        "stage3f": "outputs/stage3f/STAGE3F_CHECKPOINT.json", "stage3g": "outputs/stage3g/STAGE3G_CHECKPOINT.json",
        "stage3h": "outputs/stage3h/STAGE3H_CHECKPOINT.json", "stage4a": "outputs/stage4a/STAGE4A_CHECKPOINT.json",
        "stage4b": "outputs/stage4b/STAGE4B_CHECKPOINT.json", "stage4c_a": "outputs/stage4c_a/STAGE4C_A_CHECKPOINT.json",
        "stage4c_b": "outputs/stage4c_b/STAGE4C_B_CHECKPOINT.json", "stage4c_b2": "outputs/stage4c_b2/STAGE4C_B2_CHECKPOINT.json",
        "stage4c_b3": "outputs/stage4c_b3/STAGE4C_B3_CHECKPOINT.json", "stage4d": "outputs/stage4d/STAGE4D_CHECKPOINT.json",
        "stage4e_a": "outputs/stage4e_a/STAGE4EA_CHECKPOINT.json",
    }
    for key, rel in paths.items():
        got = project_sha256(ROOT / rel)
        if got != P.INPUT_CHECKPOINTS[key]:
            raise RuntimeError(f"STOP: {key} checkpoint changed: {got}")
    from .stage2f_transfer import v1_verify
    v = v1_verify()
    if not (v["strict_files"] == 153 and v["v1_intact"] and v["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 verification failed: {v}")
    return dict(checkpoints=paths, checkpoint_hashes=P.INPUT_CHECKPOINTS, v1=v)


def _stable_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.12g")


def freeze_primary() -> dict:
    """Fit XGBoost+M, combine verified historical M results, apply the rule, freeze decision."""
    inp = verify_inputs()
    sel_freeze = OUT / "STAGE4EB_PRIMARY_SELECTION_FREEZE.json"
    if sel_freeze.exists():
        fr = json.loads(sel_freeze.read_text())
        for rel, digest in fr["files"].items():
            if project_sha256(ROOT / rel) != digest:
                raise RuntimeError(f"STOP: frozen primary result changed: {rel}")
        print(json.dumps(fr, indent=2))
        return fr
    d = S4B.Data()
    xobj = E.run_xgb_feature_set(d, "M")
    lolo, pred, targets = E.primary_state(d, xobj)
    # Standardise output columns and verify exact held-out sizes / EQ1 targets.
    lolo["feature_set"] = "M"
    lolo["pr_lift"] = lolo.pr_auc / lolo.prevalence
    lolo["source"] = lolo.source.fillna("STAGE4EB_NEW_FIT")
    cols = ["analysis", "held_out", "model", "feature_set", "source", "params", "inner_best_mean_pr_auc",
            "n", "positives", "negatives", "prevalence", "pr_auc", "pr_lift", "roc_auc", "brier_raw", "log_loss_raw",
            "raw_cal_intercept", "raw_cal_slope", "brier", "log_loss", "cal_intercept", "cal_slope", "threshold", "precision", "recall", "f1"]
    for c in cols:
        if c not in lolo:
            lolo[c] = np.nan
    lolo = lolo[cols].sort_values(["held_out", "model"]).reset_index(drop=True)
    expected = {k: len(targets[k]["y"]) for k in P.LANDSCAPES}
    if not (lolo.n.astype(int) == lolo.held_out.map(expected)).all():
        raise RuntimeError("STOP: outer evaluation did not use complete held-out landscapes")
    bs = E.bootstrap(targets, pred)
    btab = E.bootstrap_summary(lolo, bs)
    comp = E.comparison(lolo)
    stab = E.rank_stability(lolo)
    robust, decision = E.selection(lolo, comp, bs)
    # Inner rows: exact historical Stage 4B rows plus new XGBoost rows.
    old = pd.read_csv(ROOT / "outputs/stage4b/STAGE4B_INNER_SELECTION.tsv", sep="\t")
    old = old[(old.analysis == "PRIMARY") & (old.feature_set == "M") & old.model.isin(P.MODELS[:3])].copy()
    old = old.rename(columns={"held_out": "outer_held_out"})
    old["source"] = "STAGE4B_VERIFIED_REUSE"
    xin = pd.DataFrame(xobj["inner"]); xin["source"] = "STAGE4EB_NEW_FIT"
    inner = pd.concat([old, xin], ignore_index=True, sort=False)
    _stable_tsv(OUT / "STAGE4EB_INNER_SELECTIONS.tsv", inner)
    _stable_tsv(OUT / "STAGE4EB_LOLO_RESULTS.tsv", lolo)
    _stable_tsv(OUT / "STAGE4EB_SPATIAL_BOOTSTRAP.tsv", btab)
    _stable_tsv(OUT / "STAGE4EB_MODEL_COMPARISON.tsv", comp)
    _stable_tsv(OUT / "STAGE4EB_ROBUST_SELECTION.tsv", robust)
    _stable_tsv(OUT / "STAGE4EB_RANK_STABILITY.tsv", stab)
    _json(OUT / "STAGE4EB_DEVELOPMENT_LEADER.json", dict(**decision, label="DEVELOPMENT_ONLY_NONFINAL",
                                                          feature_set="M", final_external_validation=False,
                                                          lockbox_touched=False, primary_outcome="COTABATO_EQ1"))
    frozen_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    fs = ["STAGE4EB_INNER_SELECTIONS.tsv", "STAGE4EB_LOLO_RESULTS.tsv", "STAGE4EB_SPATIAL_BOOTSTRAP.tsv",
          "STAGE4EB_MODEL_COMPARISON.tsv", "STAGE4EB_ROBUST_SELECTION.tsv", "STAGE4EB_RANK_STABILITY.tsv",
          "STAGE4EB_DEVELOPMENT_LEADER.json"]
    fr = dict(stage=P.STAGE, status="PRIMARY_M_SELECTION_FROZEN_BEFORE_CONTROLS_OR_CUMULATIVE_SENSITIVITY",
              frozen_utc=frozen_utc, preregistration_freeze_sha256=project_sha256(PREREG_FREEZE),
              files={f"outputs/stage4e_b/{f}": project_sha256(OUT / f) for f in fs}, decision=decision,
              xgboost_version=P.XGBOOST_VERSION, input_integrity=inp,
              COTABATO_EQ1_PRIMARY="YES", CUMULATIVE_COTABATO_ACCESSED_FOR_SELECTION="NO",
              LOCKBOX_TOUCHED="NO", MEGH_TRAINED="NO", H_USED="NO", GEN3_MODIFIED="NO")
    _json(sel_freeze, fr)
    print(json.dumps(fr, indent=2))
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "primary", "secondary"])
    args = ap.parse_args()
    if args.phase == "prereg":
        freeze_prereg()
    elif args.phase == "primary":
        freeze_primary()
    elif args.phase == "secondary":
        from .stage4eb_secondary import run_secondary
        print(json.dumps(run_secondary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
