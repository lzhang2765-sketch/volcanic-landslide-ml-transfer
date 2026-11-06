"""Freeze the Stage 5D-E exploratory amendment before any Stage 5D-E fit.

This module deliberately performs integrity and data-contract checks only.  It
does not import or call a model-fitting routine.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .build_stage5bb1 import verify_baseline
from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5d_e"
FREEZE = OUT / "STAGE5DE_EXPLORATORY_AMENDMENT_FREEZE.json"
LANDSCAPES = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
]
EXPECTED = {"terrain_units": 835846, "positives": 8793, "negatives": 827053}


def write_text(path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def write_json(path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def experiment_config() -> dict:
    contracts = json.loads((ROOT / "outputs/stage5d_a/STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json").read_text())
    guards = json.loads((ROOT / "outputs/stage5d_a/STAGE5DA_SELECTION_AND_FAILURE_GUARDS.json").read_text())
    return {
        "stage": "5D-E", "version": "1.0.0", "classification": "DEVELOPMENT_ONLY_EXPLORATORY",
        "amendment_timing": "frozen before any Stage 5D-E model fit, but after Stage 4 results were known",
        "landscapes": LANDSCAPES, "target": contracts["common"]["target"],
        "features": {"family": "M", "count": 60, "order_sha256": contracts["common"]["feature_order_sha256"],
                     "excluded": ["E", "G", "H"], "landscape_id_predictor": False, "coordinate_predictors": False},
        "validation": {"outer": "six-fold leave-one-whole-landscape-out", "inner": "leave-one-training-landscape-out",
                       "preprocessing": "fit inside each training fold only", "outer_outcomes_for_selection": False},
        "candidates": {k: contracts["candidates"][k] for k in ("B0", "W1", "D1", "E1", "G1")},
        "primary_metric": guards["primary_metric"], "selection_and_failure_guards": guards,
        "uncertainty": {"method": "paired 3 km authoritative spatial-block bootstrap", "replicates": 2000,
                        "interval": "95% percentile", "seed": 20261008},
        "top_fraction_ties": "descending score; exact boundary ties resolved by terrain_unit_id ascending",
        "official_model_policy": "LOGISTIC_M remains official development-selected leader regardless of exploratory ranking",
        "lockbox_policy": "no external data or lockbox outcomes may be accessed",
    }


def dataset_audit() -> dict:
    manifest = pd.read_csv(ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv", sep="\t")
    m = manifest[(manifest.role == "M") & manifest.predictor].column.tolist()
    # Preserve the canonical Stage 5A serialization (one name per line, including
    # the final newline) rather than creating a second schema-hash convention.
    feature_hash = __import__("hashlib").sha256(("\n".join(m) + "\n").encode()).hexdigest()
    rows, totals = [], {"terrain_units": 0, "positives": 0, "negatives": 0}
    for lid in LANDSCAPES:
        path = ROOT / f"outputs/stage4a/STAGE4A_MEG_DATASET/{lid}.npz"
        with np.load(path, allow_pickle=False) as z:
            y = z["y_positive"].astype(bool)
            names = set(z.files)
            row = {"landscape_id": lid, "dataset_path": str(path.relative_to(ROOT)), "dataset_sha256": sha256(path),
                   "terrain_units": int(len(y)), "positives": int(y.sum()), "negatives": int((~y).sum()),
                   "prevalence": float(y.mean()), "has_3km_blocks": "spatial_block_3000m" in names,
                   "terrain_unit_id_unique": bool(len(np.unique(z["terrain_unit_id"])) == len(y))}
        rows.append(row)
        for k in totals:
            totals[k] += row[k]
    if totals != EXPECTED:
        raise RuntimeError(f"STOP: Stage 4A totals {totals} != {EXPECTED}")
    expected_feature_hash = json.loads((ROOT / "outputs/stage5d_a/STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json").read_text())["common"]["feature_order_sha256"]
    if len(m) != 60 or feature_hash != expected_feature_hash:
        raise RuntimeError(f"STOP: M schema mismatch: n={len(m)}, hash={feature_hash}, expected={expected_feature_hash}")
    if not all(r["has_3km_blocks"] and r["terrain_unit_id_unique"] for r in rows):
        raise RuntimeError("STOP: authoritative spatial blocks or unique terrain-unit identifiers missing")
    return {"status": "PASS", "landscapes": rows, "totals": totals, "expected_totals": EXPECTED,
            "m_feature_count": len(m), "m_feature_names": m, "m_feature_order_sha256": feature_hash,
            "manifest_path": "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv",
            "manifest_sha256": sha256(ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv")}


def main() -> None:
    if FREEZE.exists():
        freeze = json.loads(FREEZE.read_text())
        for rel, expected in freeze["frozen_files"].items():
            if sha256(ROOT / rel) != expected:
                raise RuntimeError(f"STOP: frozen Stage 5D-E preparatory artifact changed: {rel}")
        print(json.dumps({"status": "ALREADY_FROZEN", "freeze_sha256": sha256(FREEZE)}, indent=2))
        return
    integrity = verify_baseline()
    # Stage 5B-B1 itself post-dates the helper's upstream list, so bind it explicitly.
    bb1 = ROOT / "outputs/stage5b_b1/STAGE5BB1_CHECKPOINT.json"
    if sha256(bb1) != "341192d54838bed029fbec48a82dd14999f81305bc4069a6a6fd836febf75c54":
        raise RuntimeError("STOP: Stage 5B-B1 checkpoint mismatch")
    audit = dataset_audit()
    amendment = """# Stage 5D-E exploratory amendment

**Status: FROZEN BEFORE ANY STAGE 5D-E MODEL FIT.**

Stage 5D-A preferred execution only after additional qualified development landscapes became available. No additional qualified landscapes have arrived. Stage 5D-E therefore authorizes a bounded exploratory comparison on the same six already development-exposed landscapes: Kumamoto Aso 2016, Noto Peninsula 2024, Lombok Rinjani 2018, Cotabato Davao 2019, Atitlán Stan 2005 and Dominica Erika 2015.

This amendment was authored after the Stage 4 results were examined. It is not a pre-Stage-4 preregistration, and its results are subject to model-selection optimism and the small number of environments. It cannot supply independent external validation or alter the frozen external-validation protocol.

The frozen Stage 5D-A candidate definitions, grids, nested whole-landscape validation, gates and failure guards are used without expansion. Stage 5D-A and Stage 5C-A remain unchanged. G1 remains deferred. The official development-selected leader remains LOGISTIC_M regardless of any exploratory ranking. No E, G or H predictor, external dataset, candidate inbox, or lockbox outcome may be used.
"""
    write_text(OUT / "STAGE5DE_EXPLORATORY_AMENDMENT.md", amendment)
    write_json(OUT / "STAGE5DE_EXPERIMENT_CONFIG.json", experiment_config())
    write_json(OUT / "STAGE5DE_DATASET_AUDIT.json", audit)
    write_json(OUT / "STAGE5DE_PRE_FIT_INTEGRITY.json", {
        "status": "PASS", "verification": integrity,
        "stage5b_b1_checkpoint_sha256": sha256(bb1), "model_fits_before_freeze": 0,
        "lockbox_outcomes_accessed": False, "new_external_data_ingested": False,
    })
    frozen = {}
    for name in ("STAGE5DE_EXPLORATORY_AMENDMENT.md", "STAGE5DE_EXPERIMENT_CONFIG.json",
                 "STAGE5DE_DATASET_AUDIT.json", "STAGE5DE_PRE_FIT_INTEGRITY.json"):
        rel = f"outputs/stage5d_e/{name}"
        frozen[rel] = sha256(ROOT / rel)
    write_json(FREEZE, {
        "stage": "5D-E", "status": "FROZEN_BEFORE_ANY_STAGE5DE_MODEL_FIT",
        "frozen_at": datetime.now(ZoneInfo("Australia/Brisbane")).isoformat(),
        "frozen_files": frozen, "model_fits_before_freeze": 0,
        "statement": "This exploratory amendment is prospective only to Stage 5D-E fits, not to prior Stage 4 evidence.",
    })
    print(json.dumps({"status": "FROZEN", "freeze_sha256": sha256(FREEZE), "frozen_files": frozen}, indent=2))


if __name__ == "__main__":
    main()
