"""Build the read-only Stage 5A consolidation and readiness audit.

This module never rebuilds a historical stage.  It verifies frozen inputs, reads
their authoritative records, and writes only ``outputs/stage5a``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .raw_store import sha256
from .schema import ROOT

OUT = ROOT / "outputs/stage5a"
V1_FREEZE = ROOT.parent / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json"
V1_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
STAGE4EB_SHA256 = "d0d02641b1298262f6f68e80d48b73bb71016b6bbd6c8ee57eec0fd4b3ebb140"
STAGE4F_SHA256 = "a5f2caa78dd50c831eef9e691bec9c4b3f620329e280dc5e06d21e575de4c9d7"
LANDSCAPES = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tsv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty register: {path}")
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame[columns]
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _source(rel: str) -> tuple[str, str]:
    path = (ROOT.parent / rel) if rel.startswith("Science_Fair_Extension/") else (ROOT / rel)
    if not path.is_file():
        raise RuntimeError(f"STOP: authoritative source missing: {rel}")
    return rel, sha256(path)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def _checkpoint_paths() -> list[Path]:
    paths = sorted(ROOT.glob("outputs/*/STAGE*_CHECKPOINT.json"))
    return [p for p in paths if p.parent != OUT and "archive" not in p.parts]


def _recorded_hash_audit() -> dict:
    recorded: list[tuple[str, str, str]] = []
    for cp in _checkpoint_paths():
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, str) and HEX64.fullmatch(child) and (ROOT / key).is_file():
                        recorded.append((str(cp.relative_to(ROOT)), key, child))
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(json.loads(cp.read_text()))
    bad = [{"checkpoint": c, "path": p, "expected": h, "actual": sha256(ROOT / p)}
           for c, p, h in recorded if sha256(ROOT / p) != h]
    return {"recorded_hashes_checked": len(recorded), "mismatches": bad,
            "passed": len(recorded) >= 1157 and not bad}


def verify_inputs() -> dict:
    identities = {
        str(V1_FREEZE.relative_to(ROOT.parent)): (sha256(V1_FREEZE), V1_SHA256),
        "outputs/stage4e_b/STAGE4EB_CHECKPOINT.json": (
            sha256(ROOT / "outputs/stage4e_b/STAGE4EB_CHECKPOINT.json"), STAGE4EB_SHA256),
        "outputs/stage4f/STAGE4F_CHECKPOINT.json": (
            sha256(ROOT / "outputs/stage4f/STAGE4F_CHECKPOINT.json"), STAGE4F_SHA256),
    }
    bad = {p: {"actual": a, "expected": e} for p, (a, e) in identities.items() if a != e}
    if bad:
        raise RuntimeError(f"STOP: checkpoint identity mismatch: {bad}")
    from .stage2f_transfer import v1_verify
    v1 = v1_verify()
    if not (v1["strict_files"] == 153 and v1["v1_intact"] and v1["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 integrity failure: {v1}")
    historical = _recorded_hash_audit()
    if not historical["passed"]:
        raise RuntimeError(f"STOP: historical checkpoint hash failure: {historical}")
    return {
        "V1": v1,
        "checkpoint_identities": {p: {"actual": a, "expected": e, "meaning": "CHECKPOINT_FILE_SHA256"}
                                  for p, (a, e) in identities.items()},
        "historical_recorded_hashes": historical,
    }


def _cp_stage(path: Path) -> str:
    return json.loads(path.read_text()).get("stage", path.stem.replace("_CHECKPOINT", ""))


def build_provenance() -> tuple[list[dict], dict]:
    paths = _checkpoint_paths()
    order = [
        "outputs/screening/STAGE_1_SCREENING_CHECKPOINT.json", "outputs/stage2a/STAGE2A_CHECKPOINT.json",
        "outputs/stage2a2/STAGE2A2_CHECKPOINT.json", "outputs/stage2a_extension/STAGE2A_EXTENSION_CHECKPOINT.json",
        "outputs/stage2b/STAGE2B_CHECKPOINT.json", "outputs/stage2c/STAGE2C_CHECKPOINT.json",
        "outputs/stage2d/STAGE2D_CHECKPOINT.json", "outputs/stage2e/STAGE2E_CHECKPOINT.json",
        "outputs/stage2f/STAGE2F_CHECKPOINT.json", "outputs/stage2f1/STAGE2F1_CHECKPOINT.json",
        "outputs/stage2f2/STAGE2F2_CHECKPOINT.json", "outputs/stage3a/STAGE3A_CHECKPOINT.json",
        "outputs/stage3b/STAGE3B_CHECKPOINT.json", "outputs/stage3c/STAGE3C_CHECKPOINT.json",
        "outputs/stage3d/STAGE3D_CHECKPOINT.json", "outputs/stage3e/STAGE3E_CHECKPOINT.json",
        "outputs/stage3f/STAGE3F_CHECKPOINT.json", "outputs/stage3g/STAGE3G_CHECKPOINT.json",
        "outputs/stage3h/STAGE3H_CHECKPOINT.json", "outputs/stage3i/STAGE3I_CHECKPOINT.json",
        "outputs/stage3j/STAGE3J_CHECKPOINT.json", "outputs/stage4a/STAGE4A_CHECKPOINT.json",
        "outputs/stage4b/STAGE4B_CHECKPOINT.json", "outputs/stage4c_a/STAGE4C_A_CHECKPOINT.json",
        "outputs/stage4c_b/STAGE4C_B_CHECKPOINT.json", "outputs/stage4c_b2/STAGE4C_B2_CHECKPOINT.json",
        "outputs/stage4c_b3/STAGE4C_B3_CHECKPOINT.json", "outputs/stage4d/STAGE4D_CHECKPOINT.json",
        "outputs/stage4e_a/STAGE4EA_CHECKPOINT.json", "outputs/stage4e_b/STAGE4EB_CHECKPOINT.json",
        "outputs/stage4f/STAGE4F_CHECKPOINT.json",
    ]
    found = {str(p.relative_to(ROOT)): p for p in paths}
    if set(found) != set(order):
        raise RuntimeError(f"STOP: checkpoint inventory differs from frozen Stage 5A routing: missing={set(order)-set(found)}, extra={set(found)-set(order)}")
    objectives = {
        "1": "candidate screening and governance", "2A": "source acquisition and admissibility",
        "2A2": "observation-domain validation", "2A-EXTENSION": "candidate extension screening",
        "2B": "landscape inventory governance", "2C": "canonical domains, labels, DEM and geology",
        "2D": "deterministic morphology production", "2E": "history-reconstruction generation 1",
        "2F": "Kurikoma transfer and V1/V2 separation", "2F1": "Kumamoto production validation",
        "2F2": "Lombok production validation", "3A": "Noto acquisition", "3B": "Noto/production architecture",
        "3C": "six-landscape planning", "3D": "Cotabato EQ1 construction", "3E": "Atitlan/Dominica acquisition",
        "3F": "six-landscape dataset completion", "3G": "Kumamoto history development",
        "3H": "Lombok history development", "3I": "Cotabato history development", "3J": "Atitlan/Dominica history development",
        "4A": "canonical M/E/G dataset and schema freeze", "4B": "nested whole-landscape model evaluation",
        "4C-A": "Y-blind history specification and gates", "4C-B": "history reconstruction generation 1",
        "4C-B2": "history reconstruction generation 2", "4C-B3": "history reconstruction generation 3 and stop decision",
        "4D": "transfer diagnosis", "4E-A": "Cotabato failure forensics and feature-family alternatives",
        "4E-B": "robust development model selection", "4F": "calibration, OOD and selective-prediction diagnosis",
    }
    reports = {}
    for rel in order:
        directory = (ROOT / rel).parent
        candidates = sorted(directory.glob("*REPORT*.md")) + sorted(directory.glob("*report*.md"))
        reports[rel] = str(candidates[0].relative_to(ROOT)) if candidates else "NOT_PRESENT_SEPARATELY"
    major = {
        "4A": ("835846 units; 8793 positives; exact M/E/G schemas frozen", "M is morphology; E/G are controls", "Tier-2 labels and source heterogeneity"),
        "4B": ("whole-landscape evaluation showed heterogeneous transfer", "use nested LOLO and spatial bootstrap", "six reused development landscapes"),
        "4C-B3": ("one verified H-capable landscape; five unavailable/nonvalidated", "stop H at Gen3; MEGH remains locked", "insufficient H-capable landscapes"),
        "4D": ("Cotabato EQ1 transfer was chance-like/uncertain", "retain EQ1 as primary", "associations are diagnostic, not causal"),
        "4E-A": ("E alternatives did not robustly resolve failure", "retain M representation", "later inventories are sensitivity only"),
        "4E-B": ("LOGISTIC_M led RF_M by 0.0240498206 robust-score units", "freeze LOGISTIC_M development architecture", "development selection is not external validation"),
        "4F": ("calibration transfer, confidence and abstention unsupported", "communicate raw score and within-landscape percentile", "n=6 repeated development use"),
    }
    rows, nodes = [], []
    previous = "NONE"
    for rel in order:
        path, obj = ROOT / rel, json.loads((ROOT / rel).read_text())
        stage = "1" if path.parent.name == "screening" else path.parent.name.removeprefix("stage").upper().replace("_", "-")
        finding, decision, limitation = major.get(stage, ("historical checkpoint completed", "retain recorded historical decision", "see authoritative report/checkpoint"))
        manifest = obj.get("output_manifest", obj.get("outputs", obj.get("output_hashes", {})))
        manifest_desc = f"checkpoint-recorded entries={len(manifest)}" if hasattr(manifest, "__len__") else "checkpoint-recorded"
        row = {
            "STAGE_ID": stage, "OBJECTIVE": objectives.get(stage, objectives.get(stage.upper(), "historical governed development stage")),
            "AUTHORITATIVE_REPORT": reports[rel], "CHECKPOINT_PATH": rel, "VERIFIED_CHECKPOINT_HASH": sha256(path),
            "INPUTS": previous, "OUTPUT_MANIFEST": manifest_desc, "MAJOR_FINDINGS": finding,
            "BINDING_DECISIONS": decision, "LIMITATIONS": limitation, "INTEGRITY_STATUS": "PASS",
        }
        rows.append(row)
        nodes.append({"stage": stage, "checkpoint_path": rel, "checkpoint_sha256": sha256(path),
                      "authoritative_report": reports[rel], "predecessor_checkpoint": previous,
                      "integrity_status": "PASS"})
        previous = rel
    chain = {"chain_type": "CHRONOLOGICAL_PROVENANCE_NOT_A_SINGLE_HASH_LINKED_LEDGER",
             "checkpoint_reference_interpretation": {
                 STAGE4EB_SHA256: "SHA256 of outputs/stage4e_b/STAGE4EB_CHECKPOINT.json",
                 STAGE4F_SHA256: "SHA256 of outputs/stage4f/STAGE4F_CHECKPOINT.json",
             }, "nodes": nodes, "all_checkpoint_files_verified": True}
    return rows, chain


DATASET_META = {
    "KUMAMOTO_ASO_2016": ["Aso caldera/apron, Kumamoto, Japan", "2016-04-16 Kumamoto earthquake", "GSI_2016_MAPPED_LANDSLIDE_AREA_CENTRES", "POINT_EVENT_TIER2", "point centres; no buffer", "EPSG:32652", "COPERNICUS_GLO30_WITH_EDM_FLM", "GSJ Seamless 1:200k", "Mapped centres do not encode polygon extent"],
    "NOTO_PENINSULA_2024": ["Noto Peninsula, Ishikawa, Japan", "2024-01-01 Noto earthquake", "GSI_2024_AFFECTED_AREA_POLYGONS", "POLYGON_EVENT_TIER2", "affected-area polygons", "EPSG:32653", "COPERNICUS_GLO30_WITH_EDM_FLM", "GSJ Seamless 1:200k", "Affected-area interpretation and visibility limits"],
    "LOMBOK_RINJANI_2018": ["Rinjani/Lombok, Indonesia", "2018-07-29 earthquake sequence inventory", "FERRARIO_2019B_POST_5_AUG_2018_AFFECTED_AREA_POLYGONS", "POLYGON_EVENT_TIER2", "affected-area polygons", "EPSG:32750", "COPERNICUS_GLO30_WITH_EDM_FLM", "Badan Geologi 1:100k; legend 1:250k", "Sequence timing and affected-area interpretation"],
    "COTABATO_DAVAO_2019": ["Cotabato-Davao, Mindanao, Philippines", "2019-10-16 EQ1", "ZENODO_7520726_LANDSLIDES_OTT2019_EQ1_ONLY", "POLYGON_EVENT_TIER2", "EQ1 polygons inside four published study domains", "EPSG:32651", "COPERNICUS_GLO30_WITH_EDM_FLM", "Philippines MGB regional", "Sparse EQ1 positives; later EQ3/EQ4 are sensitivity only"],
    "ATITLAN_STAN_2005": ["Lake Atitlan volcanic highlands, Guatemala", "2005-10-04 Hurricane Stan", "V1_TRANSFER_LANDSLIDES_ATITLAN_6817", "POLYGON_EVENT_TIER2", "landslide polygons", "EPSG:32615", "NASADEM_2000_NO_QUALITY_COMPANION", "MAGA 1:500k", "No comparable DEM quality companion; source inherited from V1 transfer"],
    "DOMINICA_ERIKA_2015": ["Dominica volcanic island", "2015-08-27 Tropical Storm Erika", "NHESS_DOMINICA_ERIKA_1548", "POLYGON_EVENT_TIER2", "landslide polygons", "EPSG:32620", "COPERNICUS_GLO30_WITH_EDM_FLM", "official digitized 1:100k", "Inventory visibility/completeness and mixed-age volcanic terrain"],
}


def build_dataset_register() -> list[dict]:
    rows = []
    strict = {"KUMAMOTO_ASO_2016": 324925, "NOTO_PENINSULA_2024": 87547, "LOMBOK_RINJANI_2018": 169563,
              "COTABATO_DAVAO_2019": 64932, "ATITLAN_STAN_2005": "NOT_AVAILABLE", "DOMINICA_ERIKA_2015": 51754}
    for lid in LANDSCAPES:
        path = ROOT / f"outputs/stage4a/STAGE4A_MEG_DATASET/{lid}.npz"
        with np.load(path, allow_pickle=False) as z:
            n, pos = len(z["y_positive"]), int(z["y_positive"].sum())
        m = DATASET_META[lid]
        rows.append({
            "LANDSCAPE_ID": lid, "GEOGRAPHIC_IDENTITY": m[0], "INVENTORY_EVENT_DATE": m[1],
            "LANDSLIDE_INVENTORY_SOURCE": m[2], "LABEL_TIER": "TIER_2", "POLYGON_POINT_INTERPRETATION": m[4],
            "OBSERVATION_DOMAIN_PROVENANCE": "Stage 2A2/2C governed Omega; source-specific searched/affected domain",
            "TERRAIN_UNIT_RESOLUTION": "90 m (3x3 canonical 30 m cells)",
            "VALID_UNIT_RULE": ">=5 of 9 cell centres in Omega and valid canonical DEM",
            "POSITIVE_LABEL_RULE": "point: contains >=1 selected centre, no buffer; polygon: >=1 of 9 cell centres inside polygon",
            "N_UNITS": n, "POSITIVES": pos, "NEGATIVES": n-pos, "PREVALENCE": format(pos/n, ".12g"),
            "DEM_PROVENANCE": m[6], "CLIMATE_PROVENANCE": "CHELSA v2.1 1981-2010 normals",
            "VEGETATION_PROVENANCE": "Hansen treecover2000", "GEOLOGY_PROVENANCE": m[7],
            "QUALITY_RESTRICTIONS": f"strict DEM-quality subset={strict[lid]}; all-valid population is primary",
            "KNOWN_LIMITATIONS": m[8], "DEVELOPMENT_EXPOSURE_STATUS": "DEVELOPMENT_EXPOSED",
            "SOURCE_ARCHIVE_SHA256": sha256(path), "PROJECTED_CRS": m[5],
        })
    totals = tuple(sum(int(r[k]) for r in rows) for k in ("N_UNITS", "POSITIVES", "NEGATIVES"))
    if totals != (835846, 8793, 827053):
        raise RuntimeError(f"STOP: Stage 4A totals mismatch: {totals}")
    return rows


def build_features() -> tuple[dict, list[dict]]:
    definitions_path = ROOT / "outputs/stage2d/feature_definitions.csv"
    manifest_path = ROOT / "outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv"
    schema_path = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET/SCHEMA.json"
    defs = pd.read_csv(definitions_path).set_index("feature_name")
    manifest = pd.read_csv(manifest_path, sep="\t")
    mcols = manifest[(manifest.role == "M") & manifest.predictor].column.tolist()
    if len(mcols) != 60:
        raise RuntimeError(f"STOP: expected 60 M predictors, found {len(mcols)}")
    rows = []
    for i, col in enumerate(mcols, 1):
        base, aggregation = col.rsplit("__", 1)
        if base not in defs.index:
            raise RuntimeError(f"STOP: feature definition absent for {base}")
        d = defs.loc[base]
        rows.append({"ORDER": i, "FEATURE_NAME": col, "BASE_RASTER_FEATURE": base, "FEATURE_GROUP": d.feature_group,
                     "DEFINITION": d.formula_algorithm, "UNITS": d.units, "SOURCE_RASTER": d.source_raster,
                     "SOURCE_SPATIAL_SCALE": d.spatial_scale,
                     "TERRAIN_UNIT_AGGREGATION": "arithmetic mean" if aggregation == "mean" else "population standard deviation (ddof=0)",
                     "MISSING_DATA_POLICY": d.nodata_rule, "EDGE_RULE": d.edge_rule, "DTYPE": "float64"})
    order_hash = hashlib.sha256(("\n".join(mcols) + "\n").encode()).hexdigest()
    stage4eb_prereg_path = ROOT / "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json"
    stage4eb_prereg = json.loads(stage4eb_prereg_path.read_text())
    if stage4eb_prereg["features"]["primary"] != "M" or stage4eb_prereg["features"]["primary_columns"] != 60:
        raise RuntimeError("STOP: Stage 4E-B frozen feature contract disagrees with Stage 4A")
    contract = {
        "contract": "STAGE5A_FROZEN_M_FEATURE_CONTRACT", "predictive_representation": "M",
        "feature_count": 60, "ordered_features": mcols, "ordered_feature_list_sha256": order_hash,
        "source_schema": str(schema_path.relative_to(ROOT)), "source_schema_sha256": sha256(schema_path),
        "stage4a_manifest_sha256": sha256(manifest_path), "stage2d_definitions_sha256": sha256(definitions_path),
        "stage4eb_preregistration_sha256": sha256(stage4eb_prereg_path),
        "stage4a_stage4eb_schema_consistency": "PASS_EXACT_60_ORDERED_M_COLUMNS",
        "aggregation": "Each of 30 canonical raster features is aggregated to 90 m terrain units as mean and population SD",
        "missing_data": "Canonical raster NaN policy applies; production unit requires valid DEM and >=5/9 cells in Omega; no imputation creates predictors",
        "preprocessing": "LOGISTIC_M fits z-score mean and scale on each training fold only; transformations are applied unchanged to held-out landscape",
        "excluded_families": {
            "E": {"columns": 7, "meaning": "CHELSA climate and Hansen vegetation", "status": "HISTORICAL_CONTROL_NOT_SELECTED"},
            "G": {"columns": 2, "meaning": "harmonized geology", "status": "HISTORICAL_CONTROL_NOT_SELECTED"},
            "H": {"columns": 6, "meaning": "independently validated reconstructed history", "status": "SEPARATE_LOCKED_EXPERIMENT_NOT_AVAILABLE_FOR_MEGH"},
        },
        "selection_reason": "Across the governed six-landscape development comparison, M supported the selected robust LOGISTIC_M architecture; E and G did not justify replacement, and H capability existed for only one landscape.",
        "prohibitions": ["no new features", "no reordering", "no full-dataset scaling before folds", "no zero-fill for unavailable H"],
    }
    return contract, rows


def build_model_contract() -> dict:
    leader_path = ROOT / "outputs/stage4e_b/STAGE4EB_DEVELOPMENT_LEADER.json"
    robust_path = ROOT / "outputs/stage4e_b/STAGE4EB_ROBUST_SELECTION.tsv"
    prereg_path = ROOT / "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json"
    leader = json.loads(leader_path.read_text())
    robust = pd.read_csv(robust_path, sep="\t")
    if leader["development_leader"] != "LOGISTIC_L2" or leader["feature_set"] != "M" or leader["runner_up"] != "RANDOM_FOREST":
        raise RuntimeError("STOP: development leader identity mismatch")
    if abs(float(leader["selection_margin"]) - 0.0240498206) > 5e-11:
        raise RuntimeError("STOP: robust-score margin mismatch")
    landscape = pd.read_csv(ROOT / "outputs/stage4e_b/STAGE4EB_LOLO_RESULTS.tsv", sep="\t")
    selected = landscape[(landscape.model == "LOGISTIC_L2") & (landscape.feature_set == "M")].to_dict("records")
    return {
        "contract": "STAGE5A_FROZEN_DEVELOPMENT_MODEL_CONTRACT", "model_version": "LOGISTIC_M_STAGE4EB",
        "model_family": "scikit-learn L2-regularized logistic regression", "penalty": "L2", "solver": "lbfgs",
        "selected_hyperparameter": {"C": 0.01}, "candidate_grid": {"C": [0.01, 0.1, 1.0]},
        "outer_validation": "leave one entire landscape out (6 outer folds)",
        "inner_tuning": "nested leave-one-training-landscape-out; maximize mean PR-AUC; within 0.005 choose earliest grid value",
        "preprocessing": "training-fold-only z-score standardization; held-out landscape uses training parameters",
        "sampling": "all positives plus at most 20,000 deterministic hash-sampled negatives per training landscape",
        "weights": "landscape-balanced, class/sampling-corrected training weights",
        "seeds": {"stage4b": 20261006, "stage4e_b": 20261008},
        "primary_metrics": ["PR_AUC", "PREVALENCE_NORMALIZED_PR_LIFT", "SPATIAL_BOOTSTRAP_DELTA_PR"],
        "robust_score_formula": "0.70*log(median landscape PR lift)+0.30*log(worst landscape PR lift)",
        "catastrophic_failure_guard": "candidate fails if frozen landscape-level PR/lift and paired-bootstrap guard fails; see preregistration",
        "complexity_tie_break": "prefer LOGISTIC_L2 only within frozen score/uncertainty equivalence rule; not activated for raw winner",
        "numerical_tolerances": {"inner_near_best_mean_pr_auc": 0.005, "reported_margin_tolerance": 5e-11},
        "development_leader": "LOGISTIC_M", "runner_up": "RF_M", "robust_score_margin": leader["selection_margin"],
        "external_validation_status": "NOT_PERFORMED", "primary_outcome": "COTABATO_EQ1",
        "landscape_level_results": selected,
        "historical_model_comparison": robust[["model", "feature_set", "robust_score", "median_pr_auc", "median_pr_lift", "worst_pr_lift", "catastrophic_fail", "catastrophic_landscapes"]].to_dict("records"),
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in (leader_path, robust_path, prereg_path)},
        "frozen": True, "retuning_permitted": False,
    }


def build_h_register() -> list[dict]:
    validation = pd.read_csv(ROOT / "outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv", sep="\t").set_index("landscape_id")
    status = {
        "KUMAMOTO_ASO_2016": ("VALIDATED_GEN3_PARTIAL", "All seven frozen quantitative gates pass; historical file label VERIFIED_GEN3_PARTIAL"),
        "LOMBOK_RINJANI_2018": ("NOT_VALIDATED", "HIGH_RESIDUAL|EXCESS_NEGATIVE_REMOVAL"),
        "COTABATO_DAVAO_2019": ("NOT_VALIDATED", "HIGH_RESIDUAL|HIGH_BIAS|LOW_SCENARIO_COVERAGE"),
        "ATITLAN_STAN_2005": ("NOT_VALIDATED", "EXCESS_NEGATIVE_REMOVAL|PHYSICAL_QA_FAIL"),
        "DOMINICA_ERIKA_2015": ("NOT_VALIDATED", "HIGH_RESIDUAL|LOW_SCENARIO_COVERAGE|EXCESS_NEGATIVE_REMOVAL|PHYSICAL_QA_FAIL|UNPREDICTED_HELDOUT_33"),
        "NOTO_PENINSULA_2024": ("AGE_PROXY_ONLY", "No independently validated reconstruction; age proxy is not H"),
    }
    rows = []
    for lid in LANDSCAPES:
        v = validation.loc[lid].to_dict() if lid in validation.index else {}
        rows.append({"LANDSCAPE_ID": lid, "FINAL_H_STATUS": status[lid][0], "HISTORICAL_STATUS_ALIAS": "VERIFIED_GEN3_PARTIAL" if lid.startswith("KUMAMOTO") else "NONE",
                     "GEN1_STATUS": "HISTORICAL_ATTEMPT", "GEN2_STATUS": "HISTORICAL_ATTEMPT", "GEN3_STATUS": "FINAL_FROZEN",
                     "MEDIAN_ABS_RESIDUAL_FRACTION": v.get("median_abs_residual_fraction", "NOT_APPLICABLE"),
                     "ABS_BIAS_FRACTION": v.get("abs_bias_fraction", "NOT_APPLICABLE"),
                     "HOLDOUT_BAND_FRACTION": v.get("holdout_band_fraction", "NOT_APPLICABLE"),
                     "SUPPORT_FRACTION": v.get("support_fraction_of_units", "NOT_APPLICABLE"),
                     "NEGATIVE_REMOVAL_FRACTION": v.get("negative_removal_fraction", "NOT_APPLICABLE"),
                     "ALL_NUMERIC_GATES_PASS": v.get("all_gates_pass", False), "PHYSICAL_QA_PASS": v.get("physical_qa_pass", False),
                     "FAILURE_REASON": status[lid][1], "H_AVAILABLE_FOR_MEGH": "YES" if lid.startswith("KUMAMOTO") else "NO",
                     "SOURCE_FILE": "outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv"})
    return rows


def build_transfer() -> list[dict]:
    transfer = pd.read_csv(ROOT / "outputs/stage4d/STAGE4D_TRANSFER_SIGNAL.tsv", sep="\t")
    primary = transfer[(transfer.model == "LOGISTIC_L2") & (transfer.feature_set == "M") & (transfer.sensitivity == "primary 3 km blocks")]
    rows = []
    for r in primary.to_dict("records"):
        rows.append({"LANDSCAPE_ID": r["landscape"], "INVENTORY_ROLE": "PRIMARY_EQ1" if r["landscape"] == "COTABATO_DAVAO_2019" else "PRIMARY",
                     "MODEL": "LOGISTIC_M", "PREVALENCE": r["prevalence"], "PR_AUC": r["pr_auc"], "PR_LIFT": r["pr_lift"],
                     "SPATIAL_BOOTSTRAP_DELTA_CI": f"[{r['delta_ci_low']}, {r['delta_ci_high']}]", "TRANSFER_LABEL": r["label"],
                     "DISTRIBUTION_SHIFT_FINDING": "Cotabato was not the strongest morphology OOD case" if r["landscape"] == "COTABATO_DAVAO_2019" else "heterogeneous landscape shift; no causal attribution",
                     "SENSITIVITY_FINDING": "EQ3/EQ4 cumulative inventories improve some metrics but remain sensitivity only" if r["landscape"] == "COTABATO_DAVAO_2019" else "not applicable",
                     "ENVIRONMENTAL_IDENTITY_LEAKAGE": "E variables can encode landscape identity; E1/E2/E3 alternatives did not advance",
                     "BINDING_DECISION": "Retain morphology-only M; do not replace Cotabato EQ1 retrospectively",
                     "SOURCE_FILE": "outputs/stage4d/STAGE4D_TRANSFER_SIGNAL.tsv"})
    return rows


def evidence_rows(model: dict) -> list[dict]:
    def e(cid, stage, q, hyp, method, dataset, metric, result, unc, decision, level, limits, source):
        rel, digest = _source(source)
        return dict(CLAIM_ID=cid, STAGE=stage, SCIENTIFIC_QUESTION=q, HYPOTHESIS=hyp, METHOD=method, DATASET=dataset,
                    PRIMARY_METRIC=metric, RESULT=result, UNCERTAINTY=unc, DECISION=decision, EVIDENCE_LEVEL=level,
                    LIMITATIONS=limits, SOURCE_FILE=rel, SOURCE_HASH=digest)
    return [
        e("C001", "V1", "Is the V1 record frozen?", "Freeze is intact", "authoritative strict-file verifier", "V1 governance tree", "strict files", "153/153 intact; zero writes", "exact hashes", "retain permanent freeze", "A. FROZEN_HISTORICAL_RESULT", "not V2 external validation", "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json"),
        e("C002", "3F/4A", "Was a six-landscape development dataset constructed?", "governed domains yield auditable units", "canonical 90m unit assembly", "six development landscapes", "counts", "835846 units; 8793 positive; 827053 negative", "deterministic counts", "freeze historical datasets", "A. FROZEN_HISTORICAL_RESULT", "all Tier-2; heterogeneous inventories", "outputs/stage4a/STAGE4A_REPORT.md"),
        e("C003", "4C-B3", "Can H be independently validated across six landscapes?", "reconstruction passes frozen Y-blind gates", "held-out controls and physical gates", "six landscapes", "gate pass count", "1 H-capable; Kumamoto Gen3 partial only", "frozen thresholds", "stop; no Gen4; MEGH locked", "B. PREREGISTERED_DEVELOPMENT_RESULT", "five landscapes unavailable/nonvalidated", "outputs/stage4c_b3/STAGE4C_B3_REPORT.md"),
        e("C004", "4D", "Does LOGISTIC_M transfer above prevalence baseline?", "M transfers across whole landscapes", "LOLO plus 2000 spatial bootstraps", "six landscapes", "PR lift", "five robust-positive; Cotabato EQ1 uncertain", "3km spatial bootstrap CI", "diagnose heterogeneity", "B. PREREGISTERED_DEVELOPMENT_RESULT", "repeated development use", "outputs/stage4d/STAGE4D_TRANSFER_SIGNAL.tsv"),
        e("C005", "4E-A", "Can environmental alternatives resolve Cotabato failure?", "E alternatives improve robust transfer", "preregistered E0/E1/E2/E3 screen", "six landscapes; Cotabato primary EQ1", "advancement gate", "no alternative advanced", "landscape-level comparisons", "retain M", "B. PREREGISTERED_DEVELOPMENT_RESULT", "low landscape count and identity leakage", "outputs/stage4e_a/STAGE4EA_FINAL_REPORT.md"),
        e("C006", "4E-B", "Which fixed architecture leads development selection?", "one candidate has best guarded robust score", "nested whole-landscape comparison", "six landscapes", "robust score", "LOGISTIC_M 0.669162938062; RF_M 0.645113117463; margin 0.0240498206", "catastrophic guard and spatial bootstrap", "select LOGISTIC_M", "B. PREREGISTERED_DEVELOPMENT_RESULT", "not external validation; not universal superiority", "outputs/stage4e_b/STAGE4EB_ROBUST_SELECTION.tsv"),
        e("C007", "4F", "Do scores transfer as calibrated probabilities?", "training-only calibration transfers", "cross-fitted C0-C3 comparison", "six reused landscapes", "normalized Brier/log loss", "CALIBRATION_TRANSFER_NOT_SUPPORTED; C0 raw selected", "frozen multi-landscape rule", "prohibit probability language", "B. PREREGISTERED_DEVELOPMENT_RESULT", "development-only n=6", "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv"),
        e("C008", "4F", "Can novelty establish transfer confidence?", "novelty predicts failures beyond susceptibility", "Y-blind H_M/U0/U1 then frozen evaluation", "six reused landscapes", "incremental failure detection", "NOT_ESTABLISHED", "landscape and within-stratum rules", "withhold confidence categories", "B. PREREGISTERED_DEVELOPMENT_RESULT", "susceptibility confounding; n=6", "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv"),
        e("C009", "4F", "Does selective abstention improve reliability?", "novelty exclusions improve metrics without excluding positives", "70/80/90/100% coverage", "six reused landscapes", "joint lift/loss and exclusion guard", "NOT_SUPPORTED", "frozen coverage rule", "do not deploy abstention", "B. PREREGISTERED_DEVELOPMENT_RESULT", "limited landscapes; positive-exclusion guard failed", "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv"),
        e("C010", "5A", "Has prospective hazard forecasting been externally validated?", "untouched independent lockbox confirms model", "not yet performed", "none", "external validation", "NOT_TESTED", "none", "lockbox opening blocked", "E. NOT_TESTED", "all six landscapes development-exposed; Kurikoma exposed in V1", "outputs/stage4f/STAGE4F_FINAL_REPORT.md"),
        e("C011", "4E-B", "Is Logistic Regression universally superior?", "leader generalizes to all future domains", "not tested", "none", "universal performance", "NOT_TESTED", "none", "make only development-leader claim", "E. NOT_TESTED", "selection applies to frozen candidates and development data", "outputs/stage4e_b/STAGE4EB_FINAL_REPORT.md"),
        e("C012", "4C-B3", "Does H improve V2 transfer?", "MEGH outperforms M", "experiment remained locked", "insufficient H-capable landscapes", "MEGH", "NOT_TESTED", "none", "do not claim benefit", "E. NOT_TESTED", "N_VERIFIED_H_CAPABLE=1", "outputs/stage4c_b3/STAGE4C_B3_REPORT.md"),
    ]


def build_reliability() -> dict:
    path = ROOT / "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv"
    d = pd.read_csv(path, sep="\t").set_index("decision")["value"].to_dict()
    expected = {"CALIBRATION_STATUS": "CALIBRATION_TRANSFER_NOT_SUPPORTED", "TRANSFER_CONFIDENCE_STATUS": "NOT_ESTABLISHED", "ABSTENTION_STATUS": "NOT_SUPPORTED"}
    if any(d.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"STOP: Stage 4F decision mismatch: {d}")
    return {
        "contract": "STAGE5A_RELIABILITY_CONTRACT", "source_file": str(path.relative_to(ROOT)), "source_sha256": sha256(path),
        "calibration_candidates": {"C0": "raw score", "C1": "training-only intercept", "C2": "training-only intercept and nonnegative slope", "C3": "training-only weighted isotonic"},
        "selected_calibration": "C0_RAW", "calibration_status": expected["CALIBRATION_STATUS"],
        "sampling_weighting_implication": "sampling-corrected landscape-balanced fitting does not itself make scores transferable probabilities",
        "novelty_methods": ["H_M landscape energy distance", "U0 unconditional 10-NN", "U1 susceptibility-stratified 10-NN"],
        "susceptibility_confounding": "novelty can reflect susceptibility position; conditional U1 did not establish deployable confidence",
        "failure_detection": expected["TRANSFER_CONFIDENCE_STATUS"], "selective_prediction": expected["ABSTENTION_STATUS"],
        "allowed_product": ["RAW_MODEL_SCORE labelled uncalibrated", "target-landscape RELATIVE_SUSCEPTIBILITY_PERCENTILE"],
        "forbidden_product": ["calibrated probability", "transfer-confidence category", "selective abstention", "validated future hazard probability"],
        "limitations": "six previously examined development landscapes; no independent external validation",
    }


def deployment_schema() -> dict:
    fields = {
        "LANDSCAPE_ID": "string", "TERRAIN_UNIT_ID": "string", "RAW_MODEL_SCORE": "number",
        "RELATIVE_SUSCEPTIBILITY_PERCENTILE": "number [0,100]", "RELATIVE_SUSCEPTIBILITY_CATEGORY": "string",
        "CALIBRATION_STATUS": "constant CALIBRATION_TRANSFER_NOT_SUPPORTED", "TRANSFER_CONFIDENCE_STATUS": "constant NOT_ESTABLISHED",
        "H_RECONSTRUCTION_STATUS": "governed status string", "MODEL_VERSION": "constant LOGISTIC_M_STAGE4EB",
        "FEATURE_SCHEMA_VERSION": "frozen schema hash", "EVIDENCE_LEVEL": "controlled evidence-level string",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Stage 5A auditable inference output",
        "required": list(fields), "fields": fields, "additionalProperties": False,
        "raw_score_semantics": "uncalibrated LOGISTIC_M output; MUST NOT be called a probability",
        "percentile_domain": "all valid terrain units in the target landscape's frozen observation/inference domain",
        "percentile_definition": "100*(rank_average(score)-1)/(N-1), where rank_average is the average 1-based rank among tied scores; if N=1 use 50",
        "tie_handling": "equal scores receive their average rank and therefore the same percentile",
        "category_status": "PROVISIONAL_WITHHELD", "category_rule": "emit PROVISIONAL_WITHHELD; no categorical thresholds are frozen",
        "probability_language_permitted": False, "mapping_scope": "relative susceptibility only; no prospective probability or hazard claim",
    }


def leakage_rows() -> list[dict]:
    items = [
        ("whole_landscape_outer_splits", "PASS", "held-out landscape excluded from fitting/tuning", "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json"),
        ("nested_inner_tuning", "PASS", "LOLO inside five outer-training landscapes", "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json"),
        ("scaling", "PASS", "z-scaling fitted on training fold only", "outputs/stage4f/STAGE4F_CALIBRATION_FOLD_METADATA.json"),
        ("negative_sampling", "PASS", "deterministic training-only hash sampling", "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json"),
        ("class_landscape_weighting", "PASS_WITH_LIMITATION", "training-only correction documented; affects score calibration", "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json"),
        ("calibration", "PASS", "honest inner predictions; held-out Y unavailable to fitting", "outputs/stage4f/STAGE4F_CALIBRATION_FOLD_METADATA.json"),
        ("OOD_fitting", "PASS", "Y-blind freeze precedes held-out evaluation", "outputs/stage4f/STAGE4F_OOD_FREEZE.json"),
        ("Cotabato_sensitivity_inventories", "PASS_WITH_LIMITATION", "EQ1 remains primary; cumulative EQ3/EQ4 sensitivity only", "outputs/stage4e_b/STAGE4EB_COTABATO_SENSITIVITY.tsv"),
        ("H_independence_from_Y", "PASS_WITH_LIMITATION", "Y-blind controls/gates; only one H-capable landscape", "outputs/stage4c_a/STAGE4C_B_EXPERIMENT_CONTRACT.md"),
        ("preprocessing_heldout_folds", "PASS", "fold metadata records training-only preprocessing", "outputs/stage4f/STAGE4F_OOD_FOLD_METADATA.json"),
        ("label_domain_alignment", "PASS_WITH_LIMITATION", "explicit Omega/unit rules; Tier-2 heterogeneous inventories", "outputs/stage4a/STAGE4A_LEAKAGE_AUDIT.tsv"),
        ("test_landscape_exposure", "PASS_WITH_LIMITATION", "all six are development-exposed; repeated analyses are not independent confirmation", "outputs/stage4f/STAGE4F_PREREGISTRATION.json"),
        ("final_lockbox_status", "PASS", "no lockbox designated/opened; V1 Kurikoma excluded from untouched V2 status", "outputs/stage4a/STAGE4A_LOCKBOX_FIREWALL.json"),
    ]
    rows=[]
    for item, status, evidence, source in items:
        rel,digest=_source(source); rows.append({"AUDIT_ITEM":item,"STATUS":status,"EVIDENCE":evidence,"LIMITATION_OR_ACTION":"none" if status=="PASS" else evidence,"SOURCE_FILE":rel,"SOURCE_HASH":digest})
    return rows


def write_markdowns(integrity: dict, datasets: list[dict], model: dict, hrows: list[dict], leakage: list[dict]) -> None:
    (OUT / "STAGE5A_CLAIM_TRACEABILITY.md").write_text("""# Stage 5A claim traceability

Every material claim is assigned a `CLAIM_ID` in `STAGE5A_SCIENTIFIC_EVIDENCE_REGISTER.tsv`. The register records the exact source file and its SHA256. Evidence level D is absent because no untouched independent V2 external validation exists. The V1 Kurikoma work is frozen historical evidence, not an untouched V2 lockbox.

Narrative mapping: dataset construction → C002; H validation/stopping → C003/C012; transfer and Cotabato failure → C004/C005; development model choice → C006/C011; calibration/OOD/abstention → C007–C009; prospective validation → C010.
""")
    (OUT / "STAGE5A_LABEL_DOMAIN_AUDIT.md").write_text(f"""# Label and observation-domain audit

The six authoritative Stage 4A archives reconcile exactly to **{sum(r['N_UNITS'] for r in datasets):,} terrain units**, **{sum(r['POSITIVES'] for r in datasets):,} positives**, and **{sum(r['NEGATIVES'] for r in datasets):,} negatives**. All labels are Tier-2.

Terrain units are 90 m, derived from 3×3 canonical 30 m cells. A unit is valid when at least five of nine cell centres lie in the governed observation domain Ω and the canonical DEM is valid. Point inventories are positive when a unit contains at least one selected mapped centre, with no buffer. Polygon inventories are positive when at least one of the nine cell centres lies inside a Tier-2 polygon. These definitions are not interchangeable: point-centre data do not imply mapped polygon extent, and affected-area polygons may not mean complete scar inventories.

Cotabato EQ1 (16 October 2019) is the primary inventory. Later cumulative EQ3/EQ4 variants are sensitivity analyses only. No label was retrospectively replaced. All six landscapes are `DEVELOPMENT_EXPOSED`.
""")
    (OUT / "STAGE5A_MODEL_SELECTION_AUDIT.md").write_text(f"""# Model-selection audit

The authoritative Stage 4E-B record confirms **LOGISTIC_M** as the development leader and **RF_M** as runner-up. Their robust scores are {model['historical_model_comparison'][0]['robust_score']:.12f} and {model['historical_model_comparison'][1]['robust_score']:.12f}; the full-precision margin is {model['robust_score_margin']:.16f} (reporting value 0.0240498206).

Selection used nested leave-one-landscape-out validation, training-only scaling and sampling, prevalence-normalized PR lift, 2,000 spatial bootstrap replicates, a frozen catastrophic-failure guard, and a frozen complexity tie rule. HISTGB_M failed the guard at Kumamoto and Lombok; XGBOOST_M failed it at Lombok. LOGISTIC_M is frozen as a **development architecture**, not as a universally superior or externally validated model.
""")
    (OUT / "STAGE5A_H_GOVERNANCE.md").write_text("""# Reconstructed-history governance

The six H fields are `H_REMOVED_FRACTION`, `H_SLOPE_CHANGE_DEG`, `H_RELIEF_CHANGE_FRACTION`, `H_REFERENCE_RELATIVE_HEIGHT`, `H_SCENARIO_SPREAD_FRACTION`, and `H_CONTROL_DISTANCE_FRACTION`. Their specification and gates were frozen Y-blind before evaluation.

Gen1, Gen2 and Gen3 are historical, immutable attempts. Gen3’s quantitative gates were median absolute residual fraction ≤0.10, absolute bias ≤0.05, holdout-band fraction ≥0.60, at least 12 admissible controls, support fraction ≥0.25, negative-removal fraction ≤0.10, and maximum reconstructed reference slope ≤40°. Boundary and physical QA also apply.

Only Kumamoto passed and is recorded here as `VALIDATED_GEN3_PARTIAL`; the historical artifact uses the synonymous label `VERIFIED_GEN3_PARTIAL`, retained in the register. Noto is `AGE_PROXY_ONLY`; the other four are `NOT_VALIDATED`. Therefore `N_VERIFIED_H_CAPABLE = 1`, `MEGH_EXPERIMENT_UNLOCKED = NO`, no Gen4 is permitted, unavailable H is never zero-filled, and no H-transfer benefit was tested.
""")
    (OUT / "STAGE5A_ALLOWED_OUTPUT_CLAIMS.md").write_text("""# Allowed output claims

Allowed: “uncalibrated LOGISTIC_M score,” “relative susceptibility,” and “within-target-landscape percentile rank,” always within the valid frozen inference domain. The selected architecture is the six-landscape development leader.

Not allowed: probability of landsliding; validated future hazard; low/medium/high risk categories; established transfer confidence; reliable uncertainty detection; selective abstention; universal Logistic Regression superiority; or an H benefit. Calibration is not supported, transfer confidence is not established, and abstention is not supported.
""")
    (OUT / "STAGE5A_DEPLOYMENT_RESTRICTIONS.md").write_text("""# Deployment restrictions

`RAW_MODEL_SCORE` is an uncalibrated model output, not a probability. Percentiles are relative ranks computed only across valid units in the target landscape’s frozen observation/inference domain. Equal scores use average ranks. Category thresholds are not frozen, so `RELATIVE_SUSCEPTIBILITY_CATEGORY` must be `PROVISIONAL_WITHHELD`.

No output may claim real-world validated probability of future landsliding, transfer confidence, or automated abstention. H status must remain a separate provenance field and unavailable H must not be imputed with zero.
""")
    counts = pd.Series([r["STATUS"] for r in leakage]).value_counts().to_dict()
    (OUT / "STAGE5A_LEAKAGE_AUDIT.md").write_text(f"""# Retrospective leakage audit

Result: **no material leakage failure identified**. Counts: {counts}. This is evidence-specific, not a blanket proof that every pipeline operation is leakage-free.

Whole-landscape outer splits, nested tuning, training-only scaling/sampling/calibration/OOD fitting, and the lockbox firewall pass. Limitations remain for score calibration under weighting, heterogeneous Tier-2 label/domain semantics, repeated exposure of all six development landscapes, and the single-landscape H capability. Cotabato EQ1 remains primary; later cumulative inventories remain sensitivity-only.
""")
    hist = integrity["historical_recorded_hashes"]
    s4f = json.loads((ROOT / "outputs/stage4f/STAGE4F_CHECKPOINT.json").read_text())
    (OUT / "STAGE5A_REPRODUCIBILITY_AUDIT.md").write_text(f"""# Reproducibility audit

- V1: 153 strict files intact; zero writes.
- Historical checkpoint records: {hist['recorded_hashes_checked']} path/hash records checked in the current read-only audit; zero mismatches.
- Stage 4E-B and 4F checkpoint-file hashes exactly match the supplied identities.
- Stage 4F’s authoritative checkpoint records {s4f['historical_hash_audit']['recorded_hashes_checked']} historical hashes verified and a clean rerun of {s4f['clean_rerun']['evaluation_outputs_checked']} core evaluation artifacts byte-for-byte. Stage 5A verified that record and its checkpoint hash; it did not rerun the Stage 4F evaluation and therefore does not create a new byte-identical claim.
- Frozen preregistrations, fold metadata, deterministic seeds, source hashes, figures and tables are present and referenced by the checkpoint chain.
- Historical builders were not replayed. The focused Stage 5A tests and read-only Stage 4F/integrity regressions are recorded in the Stage 5A checkpoint.

The historical full-suite record (316 passed, 19 skipped, 4 failures) classified the four failures as legacy namespace-scanner false positives. Stage 5A also ran the two directly relevant legacy scanner files without editing them; their actual result and every scanner hit are recorded separately in `STAGE5A_TESTS.json`.
""")
    (OUT / "STAGE5A_LOCKBOX_GOVERNANCE.md").write_text("""# External-validation and lockbox governance

Statuses are `DEVELOPMENT_EXPOSED`, `NEW_DEVELOPMENT_CANDIDATE`, `LOCKBOX_CANDIDATE_UNOPENED`, `LOCKBOX_FROZEN_UNOPENED`, and `EXTERNAL_VALIDATION_OPENED`. A candidate becomes frozen only after independent source verification, defensible Ω, diversity/independence review, X/Y separation, exposure audit, and a dated cryptographic freeze made before any outcome access.

Allowed before opening: source/licence checks, X-only schema compatibility, CRS/DEM/domain processing, and deterministic inference code validation without Y or performance summaries. Prohibited: outcome labels, outcome-derived domains, prevalence, performance, threshold/calibration tuning, or using the candidate in model decisions.

The one-time protocol freezes the final training data, model binary/parameters, M schema, preprocessing, output claims, metrics and analysis code; then opens Y once, reports all results including failures, and prohibits post-lockbox retuning from being presented as confirmation. A failed lockbox remains a reported failed external validation.

All six current landscapes are development-exposed. V1 Kurikoma is also exposed and cannot be an untouched V2 lockbox. No frozen unopened lockbox currently exists.
""")
    (OUT / "STAGE5A_NEW_DATA_INTAKE_PROTOCOL.md").write_text("""# New dataset intake protocol

Candidates are not accepted datasets. For each candidate, record: (1) source identity, licence/permission; (2) inventory provenance; (3) event definition; (4) Tier-1/Tier-2 classification; (5) defensible searched/observed Ω; (6) spatial/temporal coverage; (7) CRS integrity; (8) DEM provenance and temporal suitability; (9) completeness/visibility; (10) independent source verification; (11) development/lockbox designation before Y access; (12) leakage/exposure audit; and (13) explicit inclusion/exclusion decision.

Intake order is governance → X/Y access plan → domain and event audit → geometry/CRS audit → independent verification → designation/freeze → permitted processing. Historical DEM or photogrammetry alone does not qualify a landscape for predictive ML. A separate reconstruction-validation intake may admit geological controls without admitting predictive labels.
""")
    (OUT / "STAGE5A_REUNION_INTAKE_PLAN.md").write_text("""# Réunion Salazie 1978/1984 intake plan

Status: `NEW_DEVELOPMENT_CANDIDATE`; promised photogrammetric products have not been audited or ingested.

First establish custodianship, licence, acquisition dates, camera/block geometry, DEM/orthomosaic processing lineage, CRS/vertical datum, uncertainty and independently checkable controls. Separately establish whether a landslide inventory exists, which event(s) it represents, visibility/completeness, and a defensible Ω. Do not infer predictive suitability from historical topography alone.

If the products support reconstruction controls but not event labels and Ω, route them only to a separately preregistered geological-reconstruction validation intake. Lockbox designation is forbidden until exposure and independence are evidenced and frozen before outcome access.
""")
    (OUT / "STAGE5A_BLOCKERS_AND_DEPENDENCIES.md").write_text("""# Blockers and dependencies

The development architecture can be frozen with limitations. Final training-data freeze is blocked until candidate datasets are audited and the development-versus-lockbox allocation is made. Final model fitting depends on that dataset freeze and an immutable training manifest. Lockbox opening depends on an independently verified, frozen, unopened dataset and a frozen one-time protocol.

No current artifact completes external validation. Calibration, transfer confidence and abstention remain unsupported. H cannot enter MEGH because only one landscape is verified H-capable.
""")
    (OUT / "STAGE5A_CHANGE_CONTROL_POLICY.md").write_text("""# Change-control policy

Allowed: audit new independent data; expand development landscapes before final freeze; preregister justified experiments before fits; fit the selected architecture after dataset freeze; perform one-time untouched lockbox evaluation; and map only under validated output rules.

Not automatically allowed: alter H gates, run Gen4, tune on lockbox results, change/reorder M, switch to XGBoost, optimize ensembles, invent calibration/risk categories, or add Group DRO.

Any future Group DRO proposal requires a scientific rationale, adequate landscape diversity, independent development data, preregistration before fitting, comparison against frozen LOGISTIC_M, whole-landscape validation, and documented absence of lockbox contamination. Any approved change creates a new version; it never rewrites historical outputs.
""")


def write_intake_and_status() -> None:
    candidates = ["SAINT_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020", "POPOCATEPETL_PUEBLA_2017", "REUNION_SALAZIE_1978_1984", "MOUNT_ELGON", "CHUUK_2002"]
    rows = [{"LANDSCAPE_ID": x, "STATUS": "DEVELOPMENT_EXPOSED", "BASIS": "used in Stages 3F-4F", "Y_ACCESS": "OPENED_FOR_DEVELOPMENT", "LOCKBOX_ELIGIBLE_NOW": "NO"} for x in LANDSCAPES]
    rows += [{"LANDSCAPE_ID": "KURIKOMA_V1", "STATUS": "DEVELOPMENT_EXPOSED", "BASIS": "historical V1 result and V2 transfer procedure", "Y_ACCESS": "PREVIOUSLY_OPENED", "LOCKBOX_ELIGIBLE_NOW": "NO"}]
    rows += [{"LANDSCAPE_ID": x, "STATUS": "NEW_DEVELOPMENT_CANDIDATE", "BASIS": "named candidate; not audited or accepted", "Y_ACCESS": "DO_NOT_ACCESS_IN_STAGE5A", "LOCKBOX_ELIGIBLE_NOW": "NOT_ESTABLISHED"} for x in candidates]
    _tsv(OUT / "STAGE5A_LANDSCAPE_STATUS_REGISTER.tsv", rows)
    checks = ["SOURCE_IDENTITY_PERMISSION", "INVENTORY_PROVENANCE", "INVENTORY_EVENT_DEFINITION", "TIER_CLASSIFICATION", "DEFENSIBLE_OMEGA", "SPATIOTEMPORAL_COVERAGE", "CRS_COORDINATE_INTEGRITY", "DEM_PROVENANCE_TEMPORAL_SUITABILITY", "COMPLETENESS_VISIBILITY", "INDEPENDENT_SOURCE_VERIFICATION", "DEVELOPMENT_LOCKBOX_DESIGNATION", "LEAKAGE_EXPOSURE_AUDIT", "INCLUSION_EXCLUSION_DECISION"]
    _tsv(OUT / "STAGE5A_NEW_DATA_CHECKLIST.tsv", [{"CHECK_ID": i+1, "REQUIREMENT": c, "REQUIRED_EVIDENCE": "documented source artifact and reviewer decision", "FAIL_ACTION": "EXCLUDE_OR_HOLD", "STATUS": "NOT_ASSESSED_PER_CANDIDATE"} for i,c in enumerate(checks)])


def research_narrative() -> None:
    (OUT / "STAGE5A_RESEARCH_NARRATIVE.md").write_text("""# Science Fair ML V2 research narrative

## Research question

Can a governed morphology-based model transfer landslide susceptibility ranking across diverse volcanic landscapes, and does independently reconstructed erosional history add a defensible predictive representation?

Volcanic-landslide transfer matters because mapped inventories are sparse, observation domains differ, and a model that performs within one landscape may learn local identity rather than portable geomorphic relations. Six event landscapes were constructed as 835,846 governed 90 m terrain units with explicit Tier-2 labels and Ω rules.

## What the development evidence says

**SUPPORTED:** whole-landscape development tests show above-prevalence ranking for LOGISTIC_M in five landscapes; Cotabato EQ1 is uncertain. Under the preregistered robust score and catastrophic guard, LOGISTIC_M led the tested fixed candidates, with RF_M second. This supports a development-architecture choice, not universal superiority.

**NOT SUPPORTED:** independently reconstructed H could not be validated broadly. Only Kumamoto passed Gen3 gates; the stopping rule locked H and prevented MEGH. Therefore the hypothesis that H improves V2 transfer was not tested. Environmental alternatives did not resolve the Cotabato failure, whose later cumulative inventories remain sensitivity analyses.

**NOT SUPPORTED:** calibration did not transfer under the frozen Stage 4F criteria. Novelty did not establish transfer confidence beyond susceptibility, and selective prediction did not meet the positive-exclusion guard. Raw scores are not probabilities.

**INCONCLUSIVE:** why Cotabato EQ1 fails. Label/inventory and trigger/concept shift are plausible; morphology OOD was not uniquely extreme. The current development sample is too small for causal attribution.

**NOT TESTED:** prospective forecasting, universal cross-volcano superiority, H benefit, and untouched external validation. All six landscapes are development-exposed, and V1 Kurikoma is not an untouched V2 lockbox.

## Remaining test

Independent candidate data must pass source, Ω, event, CRS, temporal, completeness, exposure and designation audits. After final training-data and model freezes, a genuinely unopened lockbox may be evaluated once, with failures reported and no post-lockbox retuning presented as confirmation.
""")
    (OUT / "STAGE5A_KEY_FINDINGS.md").write_text("""# Key findings

- SUPPORTED — the six-landscape governed dataset totals 835,846 units and 8,793 positives.
- SUPPORTED — LOGISTIC_M is the frozen development leader among the tested candidates; RF_M is runner-up.
- SUPPORTED — transfer is heterogeneous; Cotabato EQ1 remains the primary uncertain failure.
- NOT SUPPORTED — broad H validation, MEGH fitting, calibration transfer, confidence categories and selective abstention.
- INCONCLUSIVE — the causal mechanism of Cotabato failure and generality beyond the six exposed landscapes.
- NOT TESTED — prospective hazard probability, H improvement, universal model superiority and untouched external validation.
""")


def final_report(integrity: dict) -> None:
    report = """STAGE 5A DEVELOPMENT CONSOLIDATION &
PRE-VALIDATION READINESS AUDIT: COMPLETE

# A. Executive summary

Stage 5A consolidates the completed six-landscape development cycle without training, feature construction, calibration, threshold optimization, data ingestion or lockbox access. The evidence supports freezing LOGISTIC_M as the development architecture with limitations. It does not support calibrated probabilities, transfer-confidence classes, abstention, H improvement, prospective hazard forecasting, or completed external validation.

# B. Historical integrity

V1: 153 strict files intact with zero writes. All recorded historical V2 hashes inspected by Stage 5A match. Stage 4E-B and Stage 4F supplied identifiers are checkpoint-file SHA256 hashes, not internal identifiers.

# C. Provenance chain

`STAGE5A_PROVENANCE_REGISTER.tsv` and `STAGE5A_CHECKPOINT_CHAIN.json` chronologically trace screening through Stage 4F, including nonuniform historical names.

# D. Dataset register

The authoritative total is 835,846 units: 8,793 positive and 827,053 negative. All six landscapes are Tier-2 and development-exposed; label geometry and Ω limitations remain explicit.

# E. Scientific evidence register

Every central narrative claim has a source path and SHA256. No claim is assigned evidence level D. Kurikoma V1 is frozen historical evidence, not untouched V2 validation.

# F. Feature contract

M is the exact ordered 60-column morphology schema: 30 canonical raster quantities, each summarized by mean and population SD. E/G are historical controls. H is a separate locked experiment.

# G. Model-selection contract

LOGISTIC_M leads RF_M by 0.0240498206 robust-score units under nested whole-landscape validation. The architecture is unchanged and unretuned. The result is development-only.

# H. Geological history validation

Kumamoto is `VALIDATED_GEN3_PARTIAL` (historical alias `VERIFIED_GEN3_PARTIAL`); four landscapes are not validated and Noto is age-proxy-only. `N_VERIFIED_H_CAPABLE=1`; MEGH remains locked; Gen4 is prohibited.

# I. Transfer failure findings

Five LOGISTIC_M folds show robust-positive transfer; Cotabato EQ1 is uncertain. Later cumulative inventories are sensitivity-only. E alternatives did not justify replacing M.

# J. Calibration and OOD conclusions

`CALIBRATION_TRANSFER_NOT_SUPPORTED`; transfer confidence is `NOT_ESTABLISHED`; abstention is `NOT_SUPPORTED`. Raw scores are not probabilities.

# K. Deployment output contract

Permitted outputs are an explicitly uncalibrated raw score and within-target-domain percentile. Named susceptibility categories are `PROVISIONAL_WITHHELD`.

# L. Leakage audit

No material leakage failure was found. Several items pass with limitations, especially repeated development exposure, heterogeneous Tier-2 labels, weighting/calibration semantics and single-landscape H capability.

# M. Reproducibility audit

Historical hashes, preregistrations, seeds, fold provenance, source hashes, figures and tables were audited read-only. Stage 5A verifies the Stage 4F record of 1,157 hashes and eleven byte-identical core artifacts; it does not claim a new evaluation rerun.

# N. Lockbox governance

No frozen unopened V2 lockbox exists. All six landscapes and V1 Kurikoma are exposed. One-time evaluation requires X/Y separation, pre-opening freezes, complete failure reporting and no post-lockbox retuning.

# O. New dataset intake

Named future landscapes remain candidates. No downloads or ingestion occurred. Réunion photogrammetry requires independent source/geometry/uncertainty and separate label/Ω audits.

# P. Development freeze readiness

Architecture freeze: `READY_WITH_LIMITATIONS`. Final training-data freeze, final fitting and lockbox opening: `BLOCKED` pending audited independent datasets and formal freezes.

# Q. Scientific research narrative

Supported, unsupported, inconclusive and untested claims are separated in `STAGE5A_RESEARCH_NARRATIVE.md` and the evidence register.

# R. Remaining blockers

Audit and allocate new independent landscapes; freeze final training data; fit the frozen architecture once; designate and freeze a genuinely unopened lockbox; then execute the preregistered one-time evaluation.

# S. Tests and verification

Focused Stage 5A contract tests plus read-only historical-integrity and Stage 4F regressions are recorded in the checkpoint. Historical tests were not changed.

# T. Checkpoint and hashes

`STAGE5A_CHECKPOINT.json` records historical inputs, output hashes, integrity flags, Git state and test results. The final report is excluded from checkpoint hashing to avoid circularity.

# U. Recommended next stage

Run a governed candidate-intake and designation stage. Do not fit the final model or open a lockbox until the corresponding readiness gates pass.
"""
    (OUT / "STAGE5A_FINAL_REPORT.md").write_text(report)


def build() -> dict:
    integrity = verify_inputs()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    OUT.mkdir(parents=True, exist_ok=True)
    provenance, chain = build_provenance()
    _tsv(OUT / "STAGE5A_PROVENANCE_REGISTER.tsv", provenance)
    _json(OUT / "STAGE5A_CHECKPOINT_CHAIN.json", chain)
    datasets = build_dataset_register(); _tsv(OUT / "STAGE5A_DATASET_REGISTER.tsv", datasets)
    feature_contract, feature_rows = build_features()
    _json(OUT / "STAGE5A_FEATURE_CONTRACT.json", feature_contract); _tsv(OUT / "STAGE5A_FEATURE_DICTIONARY.tsv", feature_rows)
    model = build_model_contract(); _json(OUT / "STAGE5A_MODEL_CONTRACT.json", model)
    hrows = build_h_register(); _tsv(OUT / "STAGE5A_H_VALIDATION_REGISTER.tsv", hrows)
    transfer = build_transfer(); _tsv(OUT / "STAGE5A_TRANSFER_FAILURE_REGISTER.tsv", transfer)
    reliability = build_reliability(); _json(OUT / "STAGE5A_RELIABILITY_CONTRACT.json", reliability)
    _json(OUT / "STAGE5A_DEPLOYMENT_OUTPUT_SCHEMA.json", deployment_schema())
    leakage = leakage_rows(); _tsv(OUT / "STAGE5A_LEAKAGE_AUDIT.tsv", leakage)
    evidence = evidence_rows(model); _tsv(OUT / "STAGE5A_SCIENTIFIC_EVIDENCE_REGISTER.tsv", evidence)
    write_markdowns(integrity, datasets, model, hrows, leakage)
    write_intake_and_status(); research_narrative()
    _tsv(OUT / "STAGE5A_READINESS_MATRIX.tsv", [
        {"DECISION": "DEVELOPMENT_ARCHITECTURE_FREEZE", "STATUS": "READY_WITH_LIMITATIONS", "BASIS": "LOGISTIC_M selected under governed six-landscape development protocol", "DEPENDENCY": "retain M/model/H/reliability contracts"},
        {"DECISION": "FINAL_TRAINING_DATA_FREEZE", "STATUS": "BLOCKED", "BASIS": "candidate intake and development/lockbox allocation incomplete", "DEPENDENCY": "audited independent landscapes and immutable manifest"},
        {"DECISION": "FINAL_MODEL_FITTING", "STATUS": "BLOCKED", "BASIS": "must follow final training-data freeze", "DEPENDENCY": "frozen data, code, environment and selected architecture"},
        {"DECISION": "EXTERNAL_VALIDATION_LOCKBOX_OPENING", "STATUS": "BLOCKED", "BASIS": "no verified frozen unopened lockbox", "DEPENDENCY": "one-time preregistration and frozen unopened independent dataset"},
    ])
    final_report(integrity)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise RuntimeError(f"STOP: historical outputs changed during Stage 5A: {changed}")
    return {"stage": "5A", "status": "BUILT_PENDING_TEST_RECORD", "outputs": len(list(OUT.iterdir())), "integrity": integrity}


def finalize(test_result: dict) -> dict:
    integrity = verify_inputs()
    required = [
        "STAGE5A_PROVENANCE_REGISTER.tsv", "STAGE5A_CHECKPOINT_CHAIN.json", "STAGE5A_SCIENTIFIC_EVIDENCE_REGISTER.tsv",
        "STAGE5A_CLAIM_TRACEABILITY.md", "STAGE5A_DATASET_REGISTER.tsv", "STAGE5A_LABEL_DOMAIN_AUDIT.md",
        "STAGE5A_FEATURE_CONTRACT.json", "STAGE5A_FEATURE_DICTIONARY.tsv", "STAGE5A_MODEL_CONTRACT.json",
        "STAGE5A_MODEL_SELECTION_AUDIT.md", "STAGE5A_H_VALIDATION_REGISTER.tsv", "STAGE5A_H_GOVERNANCE.md",
        "STAGE5A_TRANSFER_FAILURE_REGISTER.tsv", "STAGE5A_RELIABILITY_CONTRACT.json", "STAGE5A_ALLOWED_OUTPUT_CLAIMS.md",
        "STAGE5A_DEPLOYMENT_OUTPUT_SCHEMA.json", "STAGE5A_DEPLOYMENT_RESTRICTIONS.md", "STAGE5A_LEAKAGE_AUDIT.tsv",
        "STAGE5A_LEAKAGE_AUDIT.md", "STAGE5A_REPRODUCIBILITY_AUDIT.md", "STAGE5A_LOCKBOX_GOVERNANCE.md",
        "STAGE5A_LANDSCAPE_STATUS_REGISTER.tsv", "STAGE5A_NEW_DATA_INTAKE_PROTOCOL.md", "STAGE5A_NEW_DATA_CHECKLIST.tsv",
        "STAGE5A_REUNION_INTAKE_PLAN.md", "STAGE5A_READINESS_MATRIX.tsv", "STAGE5A_BLOCKERS_AND_DEPENDENCIES.md",
        "STAGE5A_RESEARCH_NARRATIVE.md", "STAGE5A_KEY_FINDINGS.md", "STAGE5A_CHANGE_CONTROL_POLICY.md", "STAGE5A_FINAL_REPORT.md",
    ]
    missing = [x for x in required if not (OUT / x).is_file()]
    if missing:
        raise RuntimeError(f"STOP: missing Stage 5A outputs: {missing}")
    _json(OUT / "STAGE5A_TESTS.json", test_result)
    old_cp = OUT / "STAGE5A_CHECKPOINT.json"
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p != old_cp)
    output_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files if p.name != "STAGE5A_FINAL_REPORT.md"}
    status = _git("status", "--short")
    historical_inputs = {str(p.relative_to(ROOT)): sha256(p) for p in _checkpoint_paths()}
    cp = {
        "stage": "5A", "status": "COMPLETE", "audit_date": "2026-10-08",
        "historical_input_hashes": historical_inputs, "provenance_manifest": "outputs/stage5a/STAGE5A_PROVENANCE_REGISTER.tsv",
        "evidence_register_hash": sha256(OUT / "STAGE5A_SCIENTIFIC_EVIDENCE_REGISTER.tsv"),
        "dataset_register_hash": sha256(OUT / "STAGE5A_DATASET_REGISTER.tsv"),
        "feature_contract_hash": sha256(OUT / "STAGE5A_FEATURE_CONTRACT.json"), "model_contract_hash": sha256(OUT / "STAGE5A_MODEL_CONTRACT.json"),
        "H_governance_hash": sha256(OUT / "STAGE5A_H_GOVERNANCE.md"), "reliability_contract_hash": sha256(OUT / "STAGE5A_RELIABILITY_CONTRACT.json"),
        "leakage_audit_status": "PASS_WITH_LIMITATIONS_NO_MATERIAL_LEAKAGE_FOUND",
        "reproducibility_audit_status": "PASS_HISTORICAL_RECORDS_VERIFIED_NO_HISTORICAL_RERUN",
        "readiness_decisions": pd.read_csv(OUT / "STAGE5A_READINESS_MATRIX.tsv", sep="\t").set_index("DECISION")["STATUS"].to_dict(),
        "test_results": test_result, "historical_integrity": integrity,
        "software_versions": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
        "source_hashes": {"src/sfv2/build_stage5a.py": sha256(ROOT / "src/sfv2/build_stage5a.py"), "tests/test_stage5a.py": sha256(ROOT / "tests/test_stage5a.py")},
        "new_artifact_hashes": output_hashes, "report_hash_policy": "final report excluded to avoid circular checkpoint dependency",
        "git_head": _git("rev-parse", "HEAD"), "working_tree_dirty": bool(status), "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "working_tree_note": "Pre-existing user work preserved; no commit or push performed.",
        "V1_INTACT": "YES", "HISTORICAL_OUTPUTS_UNCHANGED": "YES", "LOGISTIC_M_UNCHANGED": "YES", "M_FEATURE_SCHEMA_UNCHANGED": "YES",
        "H_GATES_UNCHANGED": "YES", "GEN3_UNCHANGED": "YES", "MEGH_TRAINED": "NO", "LOCKBOX_TOUCHED": "NO",
        "NEW_MODEL_TRAINED": "NO", "NEW_DATA_INGESTED": "NO", "COMMIT_CREATED": "NO", "PUSH_PERFORMED": "NO",
    }
    _json(old_cp, cp)
    return {"checkpoint": str(old_cp.relative_to(ROOT)), "sha256": sha256(old_cp), "artifacts": len(output_hashes), "tests": test_result}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "finalize"])
    parser.add_argument("--test-result-json")
    args = parser.parse_args()
    if args.phase == "build":
        print(json.dumps(build(), indent=2, sort_keys=True))
    else:
        if not args.test_result_json:
            raise SystemExit("--test-result-json is required for finalize")
        print(json.dumps(finalize(json.loads(Path(args.test_result_json).read_text())), indent=2, sort_keys=True))
