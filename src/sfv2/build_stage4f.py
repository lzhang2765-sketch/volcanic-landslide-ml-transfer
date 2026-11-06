"""Stage 4F phase driver.

The preregistration phase is intentionally independent of all Stage 4F analysis
code so it can be frozen before any new outcome is calculated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import stage4b as S4B
from . import stage4f_prereg as P
from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage4f"
FREEZE = OUT / "STAGE4F_PREREGISTRATION_FREEZE.json"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _verify_inputs() -> dict:
    checks = {
        ROOT.parent / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json": P.V1_FREEZE_SHA256,
        ROOT / "outputs/stage4d/STAGE4D_CHECKPOINT.json": P.INPUT_CHECKPOINTS["stage4d"],
        ROOT / "outputs/stage4e_a/STAGE4EA_CHECKPOINT.json": P.INPUT_CHECKPOINTS["stage4e_a"],
        ROOT / "outputs/stage4e_b/STAGE4EB_CHECKPOINT.json": P.INPUT_CHECKPOINTS["stage4e_b"],
    }
    got = {str(path.relative_to(ROOT.parent)): sha256(path) for path in checks}
    bad = {str(path): (got[str(path.relative_to(ROOT.parent))], expected) for path, expected in checks.items()
           if got[str(path.relative_to(ROOT.parent))] != expected}
    if bad:
        raise RuntimeError(f"STOP: authoritative input hash mismatch: {bad}")
    from .stage2f_transfer import v1_verify
    v1 = v1_verify()
    if not (v1["strict_files"] == 153 and v1["v1_intact"] and v1["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 verification failed: {v1}")
    return {"hashes": got, "v1": v1}


def _input_audit() -> dict:
    manifest = pd.read_csv(ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv", sep="\t")
    mcols = manifest[(manifest.role == "M") & manifest.predictor].column.tolist()
    if len(mcols) != 60 or mcols != S4B.feature_sets()["M"]:
        raise RuntimeError("STOP: frozen M feature schema is not the exact ordered 60-column schema")
    import numpy as np
    rows = []
    for lid in P.LANDSCAPES:
        path = ROOT / f"outputs/stage4a/STAGE4A_MEG_DATASET/{lid}.npz"
        with np.load(path, allow_pickle=False) as z:
            required = {"terrain_unit_id", "spatial_block_3000m", "spatial_block_6000m", "centroid_x", "centroid_y",
                        "y_positive", "target_tier", "target_family", *mcols}
            missing = sorted(required - set(z.files))
            if missing:
                raise RuntimeError(f"STOP: {lid} lacks required arrays: {missing}")
            if set(np.unique(z["target_tier"])) != {"TIER_2"}:
                raise RuntimeError(f"STOP: {lid} target tier differs from Tier-2")
            if lid == "COTABATO_DAVAO_2019" and not all("EQ1" in str(x) for x in np.unique(z["target_source"])):
                raise RuntimeError("STOP: Cotabato primary target is not EQ1")
            rows.append({
                "landscape": lid, "rows": int(len(z["y_positive"])), "positives": int(z["y_positive"].sum()),
                "target_tier": "TIER_2", "target_family": "|".join(map(str, np.unique(z["target_family"]))),
                "has_3km_blocks": True, "has_6km_blocks": True, "m_columns": len(mcols),
                "archive_sha256": sha256(path),
            })
    return {
        "feature_manifest_sha256": sha256(ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv"),
        "m_columns_ordered": mcols,
        "m_columns_count": len(mcols),
        "landscapes": rows,
        "prediction_provenance": (
            "Stage 4F reconstructs nested LOGISTIC_M folds under Stage 4B/4E-B rules; Stage 4B cache may be read "
            "only after a full-precision PR-AUC identity check and is never written"
        ),
        "preprocessing_metadata": "reconstructed from training samples and recorded per fold",
        "cotabato_primary": "EQ1",
    }


def prereg_object(frozen_utc: str) -> dict:
    integrity = _verify_inputs()
    audit = _input_audit()
    return {
        "stage": P.STAGE,
        "status": "FROZEN BEFORE ANY STAGE 4F CALIBRATION OR NOVELTY RESULT",
        "frozen_utc": frozen_utc,
        "integrity": integrity,
        "development_landscapes": P.LANDSCAPES,
        "independent_sample_size": 6,
        "repeated_development_use": "Stage 4D/4E-A/4E-B already examined these landscapes; not independent confirmation",
        "input_audit": audit,
        "model": P.MODEL,
        "calibration": P.CALIBRATION,
        "ood": P.OOD,
        "failure_detection": P.FAILURE_DETECTION,
        "selective_prediction": P.SELECTIVE_PREDICTION,
        "product": P.PRODUCT,
        "h_status": P.H_STATUS,
        "seed": P.SEED,
        "prohibitions": P.PROHIBITIONS,
    }


def prereg_markdown(o: dict) -> str:
    return f"""# Stage 4F preregistration: calibration, OOD and transfer confidence

**Status:** {o['status']}  
**Frozen UTC:** {o['frozen_utc']}  
**Independent development sample:** six landscapes (repeated development use; not external validation)

## Frozen architecture and data

LOGISTIC_M is fixed: L2 logistic regression, the exact ordered 60 M predictors, Stage 4E-B nested LOLO selection, deterministic negative sampling, landscape-balanced sampling-corrected weights, training-only z-scales and fixed seeds. Every outer fold fits only five landscapes and evaluates every unit of the sixth. Cotabato EQ1 is primary. H, E and G are not predictive inputs.

## Calibration candidates

- C0: raw score.
- C1: training-only intercept recalibration.
- C2: training-only intercept-and-nonnegative-slope recalibration.
- C3: one monotonic flexible candidate, weighted isotonic regression.

Each calibrator is fitted only to honest inner-LOLO predictions from the five outer-training landscapes. Held-out labels and prevalence are unavailable to fitting. Selection minimises the frozen robust combination of landscape-normalised Brier and log-loss ratios in the JSON; candidates with a >25% joint Brier/log-loss failure on any landscape versus C0 are ineligible. Outcome A/B/C criteria are numerical and frozen in the JSON.

## Y-blind novelty

Landscape novelty is the Stage 4D normalised energy statistic H_M under identical training-only scaling and deterministic sampling. U0 is unconditional 10-NN M-space novelty. U1 repeats the calculation within five training-defined LOGISTIC_M susceptibility strata. Small strata expand to neighbours. Scores and confounding diagnostics are frozen and hashed before held-out labels are loaded for reliability evaluation.

## Reliability and selective prediction

OOD must add failure-detection information beyond the susceptibility-only reference, per landscape, under the numerical rule in the JSON. With only six development landscapes, landscape correlations are exploratory. Transfer-confidence categories are withheld unless that rule passes; otherwise status is `NOT_ESTABLISHED` and novelty remains descriptive. Coverage is frozen at 100%, 90%, 80% and 70%, with prevalence and excluded positives reported alongside PR-AUC/lift and calibration.

## Product communication

The product always reports relative susceptibility and target-landscape percentile rank. It calls a value a calibrated probability only under outcome A. Susceptibility and reconstruction/H status remain separate from transfer novelty and transfer confidence.

## Prohibitions

No lockbox, MEGH, Gen-4, H/Gen-3 change, new M/E/G variable, architecture change, historical write, post-result method invention, external-validation claim or universal percentage-risk claim.
"""


def freeze_prereg() -> dict:
    if FREEZE.exists():
        fr = json.loads(FREEZE.read_text())
        for rel, digest in fr["files"].items():
            if sha256(ROOT / rel) != digest:
                raise RuntimeError(f"STOP: frozen Stage 4F preregistration changed: {rel}")
        return fr
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("STOP: Stage 4F output exists before preregistration freeze")
    frozen = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    obj = prereg_object(frozen)
    json_path = OUT / "STAGE4F_PREREGISTRATION.json"
    md_path = OUT / "STAGE4F_PREREGISTRATION.md"
    _write_json(json_path, obj)
    md_path.write_text(prereg_markdown(obj))
    source = ROOT / "src/sfv2/stage4f_prereg.py"
    files = {str(p.relative_to(ROOT)): sha256(p) for p in (json_path, md_path, source)}
    fr = {
        "stage": P.STAGE,
        "status": "FROZEN_BEFORE_ANY_NEW_RESULT",
        "frozen_utc": frozen,
        "files": files,
        "result_files_existing_before_freeze": [],
        "freeze_payload_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }
    _write_json(FREEZE, fr)
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prereg", "ood", "evaluate", "finalize"])
    args = ap.parse_args()
    if args.phase == "prereg":
        print(json.dumps(freeze_prereg(), indent=2, sort_keys=True))
        return
    from . import stage4f
    if args.phase == "ood":
        print(json.dumps(stage4f.run_ood_phase(), indent=2, sort_keys=True))
    elif args.phase == "evaluate":
        print(json.dumps(stage4f.run_evaluation_phase(), indent=2, sort_keys=True))
    else:
        from . import finalize_stage4f
        print(json.dumps(finalize_stage4f.build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
