"""Build and freeze the Stage 5D-A development-only robustness preregistration.

This module writes contracts and audits only.  It deliberately contains no model
fitting, prediction, performance evaluation, or outcome-access implementation.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5d_a"
AUTHORING_TIMESTAMP = "2026-10-08T00:00:00+10:00"
V1_FREEZE = ROOT.parent / "Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json"
V1_SHA256 = "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
UPSTREAM = {
    "stage5a": (ROOT / "outputs/stage5a/STAGE5A_CHECKPOINT.json", "f9f7799a6aa806e1c2ee3e9b7f9bc4e4a9a8c4e9387dc6c20a98288f9869490b"),
    "stage5b_a": (ROOT / "outputs/stage5b_a/STAGE5BA_CHECKPOINT.json", "dfd9251bff21f972e681d0050f04ec734cb8fd90771a177088251ef1310f6722"),
    "stage5c_a": (ROOT / "outputs/stage5c_a/STAGE5CA_CHECKPOINT.json", "ba557db4dd593ae49fd0bae6bf17d5e60615483d2d4291e59a90a0a6a29a76ec"),
}
STAGE5C_SCIENTIFIC = {
    "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
    "outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json": "bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
    "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1",
}
LANDSCAPES = [
    "KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
    "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
]
TOTALS = {"terrain_units": 835846, "positives": 8793, "negatives": 827053}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def _json(path: Path, value: object) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()


def _historical_hash_audit() -> dict:
    records: list[tuple[str, str, str]] = []
    for checkpoint in sorted(ROOT.glob("outputs/*/STAGE*_CHECKPOINT.json")):
        if checkpoint.parent == OUT:
            continue

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, str) and HEX64.fullmatch(child) and (ROOT / key).is_file():
                        records.append((str(checkpoint.relative_to(ROOT)), key, child))
                    else:
                        walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(json.loads(checkpoint.read_text()))
    mismatches = []
    for checkpoint, rel, expected in records:
        actual = sha256(ROOT / rel)
        if actual != expected:
            mismatches.append({"checkpoint": checkpoint, "path": rel, "expected": expected, "actual": actual})
    return {"records_checked": len(records), "mismatches": mismatches, "passed": len(records) >= 1326 and not mismatches}


def verify_baseline() -> dict:
    details = {}
    for stage, (path, expected) in UPSTREAM.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"STOP: {stage} checkpoint hash {actual} != {expected}")
        checkpoint = json.loads(path.read_text())
        manifest = checkpoint.get("new_artifact_hashes", checkpoint.get("output_manifest", checkpoint.get("output_hashes", {})))
        bad = [rel for rel, digest in manifest.items() if not (ROOT / rel).is_file() or sha256(ROOT / rel) != digest]
        if bad:
            raise RuntimeError(f"STOP: {stage} recorded artifacts differ: {bad}")
        details[stage] = {"checkpoint_sha256": actual, "artifacts_checked": len(manifest), "mismatches": []}
    stage5c_freeze = json.loads((ROOT / "outputs/stage5c_a/STAGE5CA_PREREGISTRATION_FREEZE.json").read_text())
    for rel, expected in STAGE5C_SCIENTIFIC.items():
        if stage5c_freeze["files"].get(rel) != expected:
            raise RuntimeError(f"STOP: Stage 5C-A freeze manifest does not bind expected scientific file: {rel}")
        actual = sha256(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"STOP: frozen Stage 5C-A scientific file changed: {rel}")
    from .stage2f_transfer import v1_verify
    v1 = v1_verify()
    if sha256(V1_FREEZE) != V1_SHA256 or not (v1["v1_intact"] and v1["strict_files"] == 153 and v1["writes_performed"] == 0):
        raise RuntimeError(f"STOP: V1 integrity failure: {v1}")
    chain = _historical_hash_audit()
    if not chain["passed"]:
        raise RuntimeError(f"STOP: historical hash-chain failure: {chain}")
    return {
        "status": "PASS", "V1": v1, "V1_freeze_sha256": sha256(V1_FREEZE),
        "checkpoints": details, "stage5c_scientific_hashes": STAGE5C_SCIENTIFIC,
        "historical_manifest": chain,
    }


def common_data_contract() -> dict:
    feature = json.loads((ROOT / "outputs/stage5a/STAGE5A_FEATURE_CONTRACT.json").read_text())
    return {
        "landscapes": LANDSCAPES,
        "membership_status": "CURRENT_SIX_LANDSCAPE_DEVELOPMENT_SET; FUTURE_EXECUTION_REQUIRES_SEPARATE_MEMBERSHIP_FREEZE",
        "totals": TOTALS,
        "target": "frozen Stage 4A Tier-2 affected-terrain label inside each governed observation domain Omega",
        "features": "exact ordered frozen 60 M morphology predictors",
        "feature_count": 60,
        "feature_order_sha256": feature["ordered_feature_list_sha256"],
        "excluded": ["E climate/vegetation", "G geology", "H reconstructed history"],
        "held_out_outcomes": "never used for preprocessing, weighting, tuning, gates, model selection, or stopping",
    }


def candidate_contracts() -> dict:
    common = common_data_contract()
    baseline = {
        "id": "B0", "name": "LOGISTIC_M", "role": "frozen development baseline",
        "historical_identity": {"robust_score": 0.669162938062, "runner_up": "RF_M", "runner_up_robust_score": 0.645113117463, "margin_calculation": "0.669162938062 - 0.645113117463 = 0.024049820599; reported 0.0240498206"},
        "family": "L2-regularized logistic regression", "solver": "lbfgs", "max_iter": 5000,
        "features": common["features"], "C_grid_ordered": [0.01, 0.1, 1.0],
        "preprocessing": "training-fold-only z-score mean and scale; apply unchanged to validation landscape",
        "sampling": "all positives plus at most 20,000 deterministic hash-sampled negatives per training landscape",
        "weights": "equal total weight per training landscape with inverse sampling-probability correction; natural within-landscape prevalence retained",
        "seed": 20261006, "inner_selection": "maximize unweighted mean inner-landscape Average Precision; within 0.005 choose earliest grid entry",
        "status": "MANDATORY_REFERENCE_UNCHANGED",
    }
    w1 = {
        "id": "W1", "name": "DIFFICULTY_REWEIGHTED_LOGISTIC_M", "role": "weighting-only candidate",
        "family": baseline["family"], "features": common["features"], "C_grid_ordered": baseline["C_grid_ordered"],
        "base_sampling_and_correction": baseline["sampling"] + "; " + baseline["weights"],
        "redundancy_audit": {
            "finding": "B0 already gives every training landscape equal total weight after sampling correction.",
            "excluded_as_not_distinct": ["nominal equal-landscape weighting", "inverse landscape sample-size weighting after B0 normalization"],
        },
        "difficulty_estimation": "For each fitting set, create honest B0 auxiliary leave-one-landscape-out predictions using only landscapes in that fitting set; compute sampling-corrected mean binary log loss L_g on each auxiliary holdout.",
        "multiplier": "m_g = clip(exp(L_g - mean_h(L_h)), 0.5, 2.0), then m_g <- K*m_g/sum_h(m_h)",
        "final_landscape_mass": "m_g / K; within-landscape sampling correction and prevalence preservation remain B0-exact",
        "nestedness": "Every inner training split recomputes auxiliary difficulty from its training landscapes only. Outer held-out outcomes are never used.",
        "new_tuning_parameters": [], "seed": 20261006, "status": "PREREGISTERED_NOT_EXECUTED",
    }
    d1 = {
        "id": "D1", "name": "GROUP_DRO_LOGISTIC_M", "role": "worst-group robustness candidate",
        "groups": "whole training landscapes", "features": common["features"],
        "group_loss": "sampling-corrected within-landscape mean binary log loss, so group size cannot dominate",
        "objective": "J(beta,b,q)=sum_g q_g*L_g(beta,b) + ||beta||_2^2/(2*C*n_eff)",
        "regularization": {"C": 0.01, "n_eff": "number of sampled training rows after all case weights are normalized to sum n_eff", "intercept_penalized": False},
        "configurations_ordered": [{"eta_q": 0.01}, {"eta_q": 0.05}, {"eta_q": 0.1}],
        "q_initialization": "q_g=1/K",
        "q_update": "q'_g=q_g*exp(eta_q*clip(L_g,0,10)); project deterministically onto simplex with 0.05<=q_g<=0.60 by iterative water-filling in ascending landscape ID",
        "coefficient_update": "deterministic full-batch gradient descent on J; initial beta=0,b=weighted logit prevalence; learning rate 0.05/(1+0.01*t)",
        "stability": {"logit_clip": [-30, 30], "gradient_l2_clip": 10.0, "max_epochs": 2000, "relative_objective_tolerance": 1e-8, "consecutive_checks": 20, "nonfinite": "INELIGIBLE", "nonconvergence": "INELIGIBLE"},
        "inner_selection": "same B0 inner-landscape mean-AP rule; within 0.005 choose earliest eta_q",
        "seed": 20261008, "claim_limit": "Group DRO targets worst training-landscape log loss; it does not optimize Average Precision or guarantee transfer.",
        "status": "PREREGISTERED_NOT_EXECUTED",
    }
    e1 = {
        "id": "E1", "name": "LOGISTIC_RF_RANK_ENSEMBLE", "role": "conditional complementarity candidate",
        "members": {
            "LOGISTIC_M": "B0 with its inner-selected C",
            "RF_M": "Stage 4E-B frozen random forest family: 100 trees, bootstrap=True, max_depth=None, max_samples=0.5, n_jobs=8, random_state=20261006; inner grid [(leaf=100,max_features=sqrt),(20,sqrt),(100,0.33),(20,0.33)]",
        },
        "scale_alignment": "Convert each member score to average-tie percentile rank within the scored validation landscape before mixing; this is ranking normalization, not label calibration.",
        "weights_ordered": [{"logistic": 0.75, "rf": 0.25}, {"logistic": 0.5, "rf": 0.5}, {"logistic": 0.25, "rf": 0.75}],
        "gate": "must pass every frozen rule in STAGE5DA_ENSEMBLE_COMPLEMENTARITY_GATE.md using honest inner out-of-fold predictions only",
        "selection": "if gate passes, choose by B0 inner rule; within 0.005 choose earliest listed weight; if gate fails, E1 is not fitted on the outer training set and is ineligible",
        "outer_outcome_use": False, "status": "CONDITIONAL_PREREGISTERED_NOT_EXECUTED",
    }
    g1 = {
        "id": "G1", "name": "RISK_EXTRAPOLATION_LOGISTIC_M", "role": "optional domain-generalization family",
        "decision": "DEFERRED", "reason": "With six landscapes, each outer split has five and each inner split only four environments; a risk-variance penalty and its stability cannot be identified credibly within the fixed budget.",
        "execution_status": "DO_NOT_IMPLEMENT_OR_EVALUATE_IN_CURRENT_SIX_LANDSCAPE_RUN",
        "reconsideration_gate": "at least eight qualified development landscapes plus a prospective amendment frozen before any new outcomes are examined",
        "open_ended_search": False,
    }
    return {"stage": "5D-A", "status": "PREREGISTERED_NOT_EXECUTED", "common": common, "candidates": {"B0": baseline, "W1": w1, "D1": d1, "E1": e1, "G1": g1}}


def selection_guards() -> dict:
    return {
        "primary_metric": {"name": "Average Precision", "implementation": "sklearn.metrics.average_precision_score", "trapezoidal_PR_AUC_substitution": False},
        "landscape_metrics": ["Average Precision", "prevalence", "PR lift = Average Precision/prevalence", "ROC-AUC secondary", "wall-clock seconds", "peak memory where measurable"],
        "aggregation": {"unit": "landscape, never pooled terrain units", "median_PR_lift": "unweighted median across complete outer landscapes", "worst_PR_lift": "minimum across complete outer landscapes", "dispersion": "range and interquartile range across complete outer landscapes", "catastrophic_failure_count": "number of landscapes that fail the frozen guard"},
        "robust_score": "0.70*ln(median landscape PR lift)+0.30*ln(worst landscape PR lift)",
        "eligibility": ["finite predictions for every outer landscape", "all candidate-specific gates pass", "no prohibited outcome use", "catastrophic-failure guard passes", "optimization stability passes"],
        "catastrophic_failure_guard": {
            "COTABATO_DAVAO_2019": "fail only if candidate PR lift <0.80 AND paired 95% upper CI of delta AP(candidate-B0) <0",
            "other_landscapes": "where B0 PR lift >=1.25, fail if paired 95% upper CI of delta AP <0 AND (candidate PR lift <=1 OR candidate PR lift <0.67*B0 PR lift)",
            "Cotabato_label": "EQ1 primary only; EQ3/EQ4 cumulative inventories remain sensitivity-only and cannot replace EQ1",
        },
        "raw_ranking": "highest robust score among eligible candidates; exact numerical tie tolerance 1e-12",
        "complexity_order": ["B0", "W1", "D1", "E1"],
        "simplicity_rule": "retain B0 when a more complex raw winner's robust-score advantage is <=5% of abs(B0 robust score)",
        "uncertainty_rule": "also retain B0 when the raw winner's median landscape AP gain is <=0.002 and the 95% paired spatial-bootstrap interval for the median six-landscape AP difference includes zero",
        "worst_landscape_deterioration": "No new arbitrary cutoff: the exact frozen catastrophic guard is authoritative; report every landscape delta and worst delta.",
        "prespecified_failure_diagnostics": {
            "Cotabato_sparse_event_specific_labels": "EQ1 stays primary; report positives, prevalence, block adequacy, AP/lift and paired delta interval; EQ3/EQ4 sensitivity cannot select a model.",
            "inventory_completeness": "report frozen provenance/quality tier and known visibility limitations; do not exclude or relabel after performance is known",
            "trigger_mechanism": "stratify descriptive result table by earthquake/rainfall trigger; no trigger subgroup tunes candidates",
            "class_prevalence": "report prevalence beside AP and lift; preserve natural within-landscape prevalence after sampling correction",
            "few_environments": "report all landscape results, leave-one-landscape summary sensitivity, and no volcano-population CI",
            "group_weight_collapse": "record q trajectory, minimum/maximum/final q and bound hits; projection or nonfinite failure makes D1 ineligible",
            "hard_group_overfit": "report training-group loss trajectory and inner held-out AP; no outer-label rescue",
            "ensemble_instability": "record every complementarity component and fold-specific gate result; gate failure makes E1 ineligible",
            "preprocessing_leakage": "hash fold membership, scaler training IDs, sampling IDs and auxiliary-prediction provenance",
            "hyperparameter_oversearch": "compare executed configurations to the frozen three-entry budget; any unregistered configuration invalidates advancement",
            "multiple_comparison_optimism": "report all attempted candidates and folds; use bounded selection language, not confirmatory discovery claims",
        },
        "fallback": "If no alternative survives all rules, retain B0. A negative result is valid.",
        "no_external_validation_claim": True,
    }


def preregistration() -> dict:
    source_paths = [
        "outputs/stage4e_b/STAGE4EB_PREREGISTRATION.json",
        "outputs/stage4e_b/STAGE4EB_DEVELOPMENT_LEADER.json",
        "outputs/stage4f/STAGE4F_RELIABILITY_DECISIONS.tsv",
        "outputs/stage5a/STAGE5A_FEATURE_CONTRACT.json",
        "outputs/stage5a/STAGE5A_MODEL_CONTRACT.json",
        "outputs/stage5a/STAGE5A_DATASET_REGISTER.tsv",
        *STAGE5C_SCIENTIFIC,
    ]
    return {
        "stage": "5D-A", "version": "1.0.0", "status": "FROZEN_BEFORE_ANY_STAGE5DA_MODEL_FIT_OR_PERFORMANCE_RESULT",
        "authoring_timestamp": AUTHORING_TIMESTAMP,
        "questions": [
            "Can outcome-blind difficulty weighting improve across-landscape robustness over LOGISTIC_M?",
            "Can whole-landscape Group DRO improve worst-landscape transfer without catastrophic failure?",
            "Are LOGISTIC_M and RF_M sufficiently complementary for a fixed low-capacity rank ensemble?",
            "Is any domain-generalization family identifiable under the current environment count and compute budget?",
        ],
        "hypotheses": {
            "H_W1": "Difficulty-aware landscape weights may improve the frozen robust score relative to B0.",
            "H_D1": "Group DRO may improve worst-landscape PR lift but can trade away median ranking performance.",
            "H_E1": "A rank ensemble can improve robustness only if honest inner predictions show nonredundant errors.",
            "H_G1": "NOT_TESTED; REx is deferred because the current environment count is inadequate.",
        },
        "data": common_data_contract(), "candidate_contract_file": "outputs/stage5d_a/STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json",
        "validation": "six-fold whole-landscape outer LOLO with candidate tuning/gating wholly inside each outer training set",
        "selection": selection_guards(),
        "uncertainty": {"method": "paired 3 km spatial-block bootstrap", "replicates": 2000, "seed": 20261008, "same_draw_for_candidates": True, "interval": "95% percentile", "population_volcano_CI": False},
        "restrictions": {
            "new_model_fit_in_5D_A": False, "performance_evaluation_in_5D_A": False,
            "lockbox_access": False, "new_real_data_ingestion": False, "new_features": False,
            "H_or_Gen3_change": False, "calibration": False, "confidence_categories": False,
            "abstention": False, "post_result_hyperparameter_expansion": False,
        },
        "reliability_contract": {"CALIBRATION_TRANSFER": "NOT_SUPPORTED", "TRANSFER_CONFIDENCE": "NOT_ESTABLISHED", "ABSTENTION": "NOT_SUPPORTED", "raw_scores_are_probabilities": False},
        "H_contract": {"N_VERIFIED_H_CAPABLE": 1, "MEGH_EXPERIMENT_UNLOCKED": "NO", "experiment_features": "M_ONLY", "Gen3_changed": False, "Gen4_permitted": False, "missing_H_zero_fill": False},
        "authoritative_sources": {rel: sha256(ROOT / rel) for rel in source_paths},
        "interpretation": "development-only comparative evidence; no candidate can become externally validated from this experiment",
    }


def write_documents() -> None:
    contracts = candidate_contracts()
    guards = selection_guards()
    _json(OUT / "STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json", contracts)
    _json(OUT / "STAGE5DA_SELECTION_AND_FAILURE_GUARDS.json", guards)
    _json(OUT / "STAGE5DA_EXPERIMENT_PREREGISTRATION.json", preregistration())
    _write(OUT / "STAGE5DA_RESEARCH_QUESTIONS.md", """# Stage 5D-A research questions

## Scope

Stage 5D-A asks whether a small, prospectively bounded set of training strategies can improve *development* transfer robustness over frozen `LOGISTIC_M`. It does not test prospective hazard forecasting or external validity.

## Questions and hypotheses

1. **Weighting:** Can cross-fitted, difficulty-aware landscape weighting improve the robust score without allowing a large landscape or an outer outcome to dominate? `H_W1` predicts possible improvement; the null is no eligible improvement over B0.
2. **Worst-group optimization:** Can Group DRO improve the worst-landscape PR lift while passing the frozen catastrophic-failure guard? `H_D1` allows a median-performance trade-off and does not assume success.
3. **Complementarity:** Do LOGISTIC_M and RF_M make sufficiently distinct honest inner-fold ranking errors to justify testing three fixed rank mixtures? `H_E1` requires the preregistered gate; otherwise the ensemble is rejected without an outer result.
4. **Domain generalization:** Is a distinct DG family scientifically identifiable with the available number of environments? `H_G1` is **NOT TESTED**: Risk Extrapolation is deferred.

Possible outcomes are support, non-support, or inconclusive. Retaining B0 is a valid result. None of these development comparisons constitutes independent external validation.
""")
    _write(OUT / "STAGE5DA_EXPERIMENT_PREREGISTRATION.md", """# Stage 5D-A advanced-ML robustness preregistration

**Status:** `FROZEN_BEFORE_ANY_STAGE5DA_MODEL_FIT_OR_PERFORMANCE_RESULT`  
**Design:** development-only, nested whole-landscape validation  
**Baseline:** unchanged LOGISTIC_M on the exact ordered 60 M features

This preregistration bounds the comparison to B0, W1, D1, conditional E1, and deferred G1. All preprocessing, sampling, auxiliary weighting, complementarity testing, tuning, and selection occur inside each outer training set. The outer landscape is touched once for candidate evaluation and never affects tuning or eligibility gates.

Primary ranking performance is `sklearn.metrics.average_precision_score`; PR lift is AP divided by the held-out landscape prevalence. Final development selection reuses the Stage 4E-B robust score, catastrophic-failure guard, simplicity preference, uncertainty rule, and numerical tolerance. Paired 3 km spatial-block bootstrap draws quantify within-landscape candidate differences. Variation across six landscapes is reported directly and is not a population confidence interval.

No model is fitted, no performance number is generated, no real data is ingested, and no lockbox is opened in Stage 5D-A. The JSON preregistration is the machine-readable authority.
""")
    _write(OUT / "STAGE5DA_WEIGHTING_PROTOCOL.md", """# W1 difficulty-weighting protocol

## Baseline audit

B0 already normalizes total training weight equally across landscapes and corrects deterministic negative subsampling. Therefore nominal “equal landscape” weighting and inverse-size weighting after that normalization are not distinct candidates and are excluded.

## Honest difficulty estimate

For every fitting set, run auxiliary B0 leave-one-landscape-out predictions using only landscapes in that fitting set. For landscape `g`, compute sampling-corrected mean binary log loss `L_g`. Set `m_g = clip(exp(L_g - mean(L)), 0.5, 2.0)` and renormalize so the `K` multipliers sum to `K`. W1 gives landscape `g` total mass `m_g/K`; it retains B0's within-landscape inverse-sampling correction and natural class prevalence.

The procedure is recomputed inside each inner training split. Thus an inner held-out landscape, the outer held-out landscape, and every lockbox outcome are absent from the weights used to fit their predictions. No weight exponent, clipping bound, or alternative loss is tuned. Only B0's ordered C grid is used.

W1 changes landscape emphasis only. It does not change features, labels, scaling, regularization family, negative sampling, or metric.
""")
    _write(OUT / "STAGE5DA_GROUP_DRO_PROTOCOL.md", """# D1 Group DRO protocol

Whole training landscapes are the groups. Each group loss is the sampling-corrected mean binary log loss, which prevents group sample count from becoming the objective weight. The objective and every numerical rule are frozen in `STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json`.

Initialize `q_g=1/K`. At every full-batch epoch update group weights by exponentiated clipped loss, then project to the simplex with bounds `[0.05, 0.60]` by deterministic water-filling in ascending landscape ID. Update the L2-logistic coefficients against the q-weighted loss. D1 fixes `C=0.01` and compares only `eta_q={0.01,0.05,0.10}` in that order. Inner selection uses mean landscape AP and the frozen 0.005 earliest-grid tolerance.

Nonfinite states, bound/projection failure, or failure to meet the frozen convergence criterion by 2,000 epochs makes that configuration ineligible; there is no rescue tuning. Group DRO optimizes training-environment log loss. It neither optimizes AP directly nor guarantees transfer to a new landscape.
""")
    _write(OUT / "STAGE5DA_ENSEMBLE_COMPLEMENTARITY_GATE.md", """# E1 ensemble complementarity gate

The gate is computed separately inside each outer training set from honest inner out-of-fold predictions. Raw member scores are converted to average-tie percentile ranks within the predicted landscape. Labels are used only for the following preregistered inner diagnostics.

E1 advances only if **all** conditions hold across the outer-training landscapes:

- median Spearman correlation between LOGISTIC_M and RF_M score ranks is `< 0.95`;
- median Spearman correlation of per-unit ranking-error magnitude is `< 0.95`, where error is `1-rank_percentile` for a positive and `rank_percentile` for a negative;
- median Jaccard overlap of false positives in each model's top 10% is `<= 0.80`;
- median Jaccard overlap of missed positives outside each model's top 20% is `<= 0.80`;
- bidirectional specialization: LOGISTIC_M has higher AP on at least two landscapes and RF_M has higher AP on at least two landscapes.

Exact-score boundary ties are broken by ascending `TERRAIN_UNIT_ID`. Empty-union Jaccard is 1.0 (no demonstrated complementarity). Thresholds are operational redundancy safeguards, not discovered biological constants.

If the gate passes, compare only Logistic/RF weights `75/25`, `50/50`, and `25/75`; choose by the standard inner rule. No stacking, learned weight, extra member, label calibration, or outer-outcome gate is allowed. If it fails, E1 is ineligible for that outer fold.
""")
    _write(OUT / "STAGE5DA_DOMAIN_GENERALIZATION_DECISION.md", """# Domain-generalization decision

One distinct family was considered: linear Risk Extrapolation (REx), which penalizes variation in training-landscape risks. It is **DEFERRED**, not silently omitted and not counted as a fitted candidate.

With six landscapes, outer LOLO leaves five environments and inner LOLO only four. That is too little independent environment variation to tune a risk-variance penalty and distinguish genuine domain robustness from one-landscape leverage within the fixed compute budget. An open-ended method search would also undermine the preregistration.

REx may be reconsidered only after at least eight qualified development landscapes exist and a separate prospective amendment freezes its objective, penalty grid, convergence rules, and comparisons before any new outcomes are inspected. Neural domain-generalization methods, representation learning, and post-result alternatives are outside scope.
""")
    _write(OUT / "STAGE5DA_NESTED_VALIDATION_PROTOCOL.md", """# Nested whole-landscape validation protocol

For each of the six outer folds, hold out one complete landscape. On the remaining five, perform leave-one-landscape-out inner selection. Scaling parameters, sampled negatives, sampling corrections, W1 auxiliary difficulty, D1 group weights, RF tuning, and the E1 gate are recomputed using only the relevant training subset.

Inner candidate selection maximizes the unweighted mean of landscape Average Precision. Configurations within 0.005 AP of the maximum choose the earliest preregistered configuration. After selection, refit the candidate once on all five outer-training landscapes and score the held-out landscape once. No outer label informs fitting, stopping, configuration selection, gate passage, percentile transformation, or retry.

Average Precision is the non-interpolated ranking summary returned by `sklearn.metrics.average_precision_score`, not trapezoidal integration. Report AP, prevalence, PR lift, ROC-AUC, full landscape results, and the robust score. Cotabato EQ1 is primary; EQ3/EQ4 are sensitivity-only. Candidate comparison uses paired 3 km block-bootstrap draws. Every incomplete, unstable, or missing fold is retained as a failure/ineligibility record.
""")
    _write(OUT / "STAGE5DA_UNCERTAINTY_PROTOCOL.md", """# Development-comparison uncertainty protocol

Within each held-out landscape, compare candidates to B0 using the same 2,000 deterministic resamples of frozen 3 km projected blocks (seed 20261008). A resampled block multiplicity becomes the unit weight for both models, so delta AP is paired. Drop and count single-class replicates and report the 95% percentile interval, effective block count, positive-containing blocks, and valid replicate fraction.

Reliable spatial uncertainty requires at least 10 blocks, five positive-containing blocks, and 80% valid replicates. Failure of this condition is reported as `NOT_RELIABLY_ESTIMABLE`; it is never replaced by a cell bootstrap. Across-landscape variability is summarized with all six estimates, median, worst value, range, and leave-one-landscape-out sensitivity. No confidence interval is described as sampling a population of volcanoes.

The historical catastrophic and uncertainty rules are applied exactly. Multiple candidate results are reported transparently; no multiplicity-adjusted discovery claim is made because selection is explicitly comparative and bounded, not a family of confirmatory causal tests.
""")
    _write(OUT / "STAGE5DA_DATASET_MEMBERSHIP_POLICY.md", """# Dataset membership policy

The currently available development set is the six frozen Stage 4A landscapes (835,846 units; 8,793 positives; 827,053 negatives). All are `DEVELOPMENT_EXPOSED`. V1 Kurikoma is not an untouched V2 lockbox.

Stage 5D-A performs no fit, so current membership is recorded rather than executed. A later Stage 5D-B run requires a hash-frozen membership table before fitting. If qualified new development landscapes arrive first, the entire nested comparison must use the prospectively frozen expanded set; no candidate may be selected on six and informally confirmed on additions. If a dataset is reserved as a lockbox, its outcomes cannot enter weighting, gates, tuning, or robustness selection.

The preferred execution condition is at least eight qualified development landscapes. This is a design threshold, not a power guarantee and not a guarantee of stable Group DRO. With only the current six, the default is to defer; any authorized fallback is explicitly exploratory and cannot automatically replace LOGISTIC_M.

Membership cannot change after execution starts. Source correction or exclusion requires a documented stop and new preregistration version, never silent replacement. Cotabato EQ1 remains the primary inventory.
""")
    _write(OUT / "STAGE5DA_COMPUTE_BUDGET.md", """# Compute and search budget

The search is intentionally small and fixed.

| Candidate | Configurations per inner comparison | Extra search |
|---|---:|---|
| B0 | 3 C values | none |
| W1 | 3 B0 C values | weighting formula fixed |
| D1 | 3 eta values; C fixed at 0.01 | none |
| E1 | gate, then 3 fixed mixtures | no learned stacking weight |
| G1 | 0 | deferred |

For six outer folds and five inner folds, the core upper bound is 270 candidate-configuration inner fits (6×5×(3+3+3+3)), before deterministic W1 auxiliary and member fits. Each selected eligible candidate permits one outer-training refit per fold. B0/RF component predictions may be cached only when hashes prove identical training membership, inputs, configuration, seed, and preprocessing.

There is no budget for feature engineering, new model families, wider grids, retries with altered seeds, ensembles beyond E1, post-result optimization, or lockbox evaluation. A runtime or convergence failure is reported under the frozen rules rather than repaired by search expansion.
""")
    _write(OUT / "STAGE5DA_ALLOWED_CLAIMS.md", """# Allowed and prohibited claims

## After Stage 5D-A preregistration only

Allowed: a bounded development-only robustness comparison has been prospectively specified; B0 remains the frozen leader; W1, D1 and conditional E1 are untested; G1 is deferred.

Prohibited: any candidate improved performance; Group DRO works; ensemble complementarity exists; LOGISTIC_M has been superseded; external transfer is validated; prospective landslide hazard is predicted.

## After a conforming future development execution

Allowed only from complete results: named candidate comparisons across the named development landscapes, the exact selection outcome, failures, and limitations. “Selected development architecture” is permissible; “externally validated” is not.

Always prohibited without later independent evidence: calibrated probability, established transfer confidence, supported abstention, universal model superiority, H benefit, prospective hazard forecasting, or use of a lockbox failure for retuning.
""")


def readiness_rows() -> list[dict]:
    return [
        {"ITEM": "A_SCIENTIFIC_HYPOTHESES", "STATUS": "READY", "BASIS": "questions and null-compatible outcomes frozen", "DEPENDENCY": "none"},
        {"ITEM": "B_CANDIDATE_MODEL_DEFINITIONS", "STATUS": "READY_WITH_LIMITATIONS", "BASIS": "B0/W1/D1/E1 exact; G1 deferred", "DEPENDENCY": "G1 requires >=8 development landscapes and amendment"},
        {"ITEM": "C_HYPERPARAMETER_BUDGETS", "STATUS": "READY", "BASIS": "three configurations per executable candidate", "DEPENDENCY": "none"},
        {"ITEM": "D_NESTED_VALIDATION", "STATUS": "READY", "BASIS": "whole-landscape outer and inner folds specified", "DEPENDENCY": "execution code validation later"},
        {"ITEM": "E_SELECTION_AND_FAILURE_RULES", "STATUS": "READY", "BASIS": "Stage 4E-B rules carried forward exactly", "DEPENDENCY": "none"},
        {"ITEM": "F_PREREGISTRATION_FREEZE", "STATUS": "READY", "BASIS": "scientific files hash-frozen before fit", "DEPENDENCY": "none"},
        {"ITEM": "G_NEW_DEVELOPMENT_DATA_AVAILABILITY", "STATUS": "BLOCKED", "BASIS": "no newly qualified independent development landscape", "DEPENDENCY": "Stage 5B-A intake"},
        {"ITEM": "H_DATASET_MEMBERSHIP_FREEZE", "STATUS": "BLOCKED", "BASIS": "decision whether to wait for new development data unresolved", "DEPENDENCY": "outcome-blind membership decision"},
        {"ITEM": "I_LOCKBOX_PROTECTION", "STATUS": "READY", "BASIS": "Stage 5C-A remains frozen; no lockbox touched", "DEPENDENCY": "continue fail-closed access"},
        {"ITEM": "J_STAGE5DB_EXECUTION", "STATUS": "BLOCKED", "BASIS": "membership not frozen and no execution implementation audited", "DEPENDENCY": "G/H plus separate implementation audit"},
        {"ITEM": "K_FINAL_MODEL_SELECTION", "STATUS": "DEFERRED", "BASIS": "no Stage 5D-A candidate has been run", "DEPENDENCY": "conforming Stage 5D-B results"},
    ]


def freeze_preregistration() -> dict:
    names = [
        "STAGE5DA_RESEARCH_QUESTIONS.md", "STAGE5DA_EXPERIMENT_PREREGISTRATION.md",
        "STAGE5DA_EXPERIMENT_PREREGISTRATION.json", "STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json",
        "STAGE5DA_WEIGHTING_PROTOCOL.md", "STAGE5DA_GROUP_DRO_PROTOCOL.md",
        "STAGE5DA_ENSEMBLE_COMPLEMENTARITY_GATE.md", "STAGE5DA_DOMAIN_GENERALIZATION_DECISION.md",
        "STAGE5DA_NESTED_VALIDATION_PROTOCOL.md", "STAGE5DA_SELECTION_AND_FAILURE_GUARDS.json",
        "STAGE5DA_UNCERTAINTY_PROTOCOL.md", "STAGE5DA_DATASET_MEMBERSHIP_POLICY.md",
        "STAGE5DA_COMPUTE_BUDGET.md", "STAGE5DA_ALLOWED_CLAIMS.md",
    ]
    files = {str((OUT / name).relative_to(ROOT)): sha256(OUT / name) for name in names}
    path = OUT / "STAGE5DA_PREREGISTRATION_FREEZE.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["files"] != files:
            raise RuntimeError("STOP: Stage 5D-A scientific preregistration differs from frozen hashes")
        return old
    obj = {
        "stage": "5D-A", "status": "FROZEN_BEFORE_ANY_MODEL_FIT_OR_PERFORMANCE_EVALUATION",
        "timestamp": AUTHORING_TIMESTAMP, "files": files,
        "preregistration_sha256": files["outputs/stage5d_a/STAGE5DA_EXPERIMENT_PREREGISTRATION.json"],
        "candidate_contracts_sha256": files["outputs/stage5d_a/STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json"],
        "selection_guards_sha256": files["outputs/stage5d_a/STAGE5DA_SELECTION_AND_FAILURE_GUARDS.json"],
        "intended_random_seeds": {"B0_and_W1_sampling": 20261006, "D1": 20261008, "RF_M": 20261006, "spatial_bootstrap": 20261008},
        "software_versions": {
            "authoring_python": sys.version.split()[0], "authoring_pandas": pd.__version__,
            "intended_execution_sklearn": json.loads((ROOT / "outputs/stage5c_a/STAGE5CA_PREREGISTRATION_FREEZE.json").read_text())["software_versions"]["sklearn"],
            "note": "Execution must freeze its full environment and verify compatibility before fitting.",
        },
        "model_fits_before_freeze": 0, "performance_results_before_freeze": 0,
    }
    _json(path, obj)
    return obj


def final_report() -> None:
    sections = {
        "A": ("Executive summary", "A bounded development-only comparison of B0, W1, D1 and conditional E1 is preregistered; G1 is deferred. No model was fitted and no performance result exists."),
        "B": ("Historical integrity", "V1 (153 strict files, zero writes), Stage 5A, 5B-A and 5C-A checkpoint identities and the frozen Stage 5C-A scientific hashes were verified before writing."),
        "C": ("Scientific motivation", "The design tests whether controlled weighting, worst-group optimization, or demonstrably complementary ranking can address observed cross-landscape fragility without broad model shopping."),
        "D": ("Existing development evidence", "LOGISTIC_M led RF_M by 0.669162938062−0.645113117463=0.024049820599 (reported 0.0240498206) in the frozen Stage 4E-B robust score. This is development evidence only."),
        "E": ("Candidate models", "B0 is unchanged LOGISTIC_M. W1 changes landscape influence, D1 changes the robust objective, E1 is gate-conditional, and G1/REx is deferred. Every executable candidate uses exact ordered 60 M."),
        "F": ("Landscape weighting", "B0 already equalizes landscape mass, so nominal equal/inverse-size alternatives are redundant. W1 alone tests cross-fitted difficulty multipliers with frozen clipping."),
        "G": ("Group DRO", "D1 uses whole landscapes as groups, normalized corrected log loss, bounded exponentiated group weights, fixed L2 strength, three eta values and explicit numerical failure rules."),
        "H": ("Ensemble complementarity", "E1 may proceed only when honest inner OOF predictions pass all score/ranking-error correlation, error-overlap and bidirectional-specialization criteria; only three fixed rank mixtures are permitted."),
        "I": ("Domain generalization decision", "REx was considered and deferred because four inner-training environments cannot credibly identify its penalty. It requires at least eight development landscapes and a prospective amendment."),
        "J": ("Nested validation", "Whole-landscape outer folds contain whole-landscape inner selection. Every transform, weight, gate and configuration is training-only; the outer landscape is scored once."),
        "K": ("Metrics", "Primary performance is non-interpolated Average Precision. Prevalence, PR lift, ROC-AUC, worst/median lift, dispersion, failures and compute cost accompany it; Cotabato EQ1 remains primary."),
        "L": ("Selection and failure guards", "The exact Stage 4E-B robust score, catastrophic guard, 1e-12 tie tolerance, 5% simplicity preference and 0.002/bootstrap uncertainty rule are retained. Missing, unstable or leaking candidates are ineligible."),
        "M": ("Uncertainty", "Paired 3 km block bootstraps quantify within-landscape deltas; all landscape variation is reported without inventing a volcano-population confidence interval."),
        "N": ("Dataset membership", "Execution prefers at least eight qualified development landscapes; this is not a power guarantee. With six, defer or explicitly exploratory analysis cannot automatically replace B0."),
        "O": ("Lockbox separation", "Stage 5C-A remains unchanged. No candidate lockbox, label, prevalence or performance information was accessed; a future winner requires a separate selection/fitting freeze before opening."),
        "P": ("Reliability restrictions", "Calibration transfer is not supported, transfer confidence is not established, and abstention is not supported. Candidate outputs remain uncalibrated relative-susceptibility rankings."),
        "Q": ("H restrictions", "H is locked: N_VERIFIED_H_CAPABLE=1, MEGH is not unlocked, Gen3/gates are unchanged, Gen4 and H zero-fill are prohibited, and this design is M-only."),
        "R": ("Preregistration hashes", "The preregistration, candidate contracts, nested protocol, selection/failure guards, and companion scientific design files are frozen in STAGE5DA_PREREGISTRATION_FREEZE.json."),
        "S": ("Testing", "Focused tests cover upstream identity, totals/features, candidate budgets, nested isolation, historical rules, freeze hashes, prohibited operations and report completeness."),
        "T": ("Readiness matrix", "Scientific design and freeze are ready; candidate definitions are ready with G1 deferred. New-data availability, membership freeze and Stage 5D-B execution remain blocked."),
        "U": ("Limitations", "Six landscapes offer limited environment-level evidence; trigger, completeness and prevalence differ; Group DRO can overfit hard groups; and multiple bounded candidates still create selection optimism."),
        "V": ("Recommended next stage", "Complete outcome-blind development-membership allocation. Then implement and audit Stage 5D-B against these hashes before any fit; do not open a lockbox."),
        "W": ("Final checkpoint", "The checkpoint records upstream verification, scientific freeze, artifact/code hashes, readiness, tests, Git identity, blockers, and all required negative integrity flags."),
    }
    text = "STAGE 5D-A ADVANCED ML ROBUSTNESS\nPREREGISTRATION: COMPLETE\n\n"
    for letter, (title, body) in sections.items():
        text += f"# {letter}. {title}\n\n{body}\n\n"
    _write(OUT / "STAGE5DA_FINAL_REPORT.md", text)


REQUIRED = [
    "STAGE5DA_RESEARCH_QUESTIONS.md", "STAGE5DA_EXPERIMENT_PREREGISTRATION.md",
    "STAGE5DA_EXPERIMENT_PREREGISTRATION.json", "STAGE5DA_CANDIDATE_MODEL_CONTRACTS.json",
    "STAGE5DA_WEIGHTING_PROTOCOL.md", "STAGE5DA_GROUP_DRO_PROTOCOL.md",
    "STAGE5DA_ENSEMBLE_COMPLEMENTARITY_GATE.md", "STAGE5DA_DOMAIN_GENERALIZATION_DECISION.md",
    "STAGE5DA_NESTED_VALIDATION_PROTOCOL.md", "STAGE5DA_SELECTION_AND_FAILURE_GUARDS.json",
    "STAGE5DA_UNCERTAINTY_PROTOCOL.md", "STAGE5DA_DATASET_MEMBERSHIP_POLICY.md",
    "STAGE5DA_COMPUTE_BUDGET.md", "STAGE5DA_EXECUTION_READINESS_MATRIX.tsv",
    "STAGE5DA_ALLOWED_CLAIMS.md", "STAGE5DA_PREREGISTRATION_FREEZE.json", "STAGE5DA_FINAL_REPORT.md",
]


def build() -> dict:
    baseline = verify_baseline()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    OUT.mkdir(parents=True, exist_ok=True)
    write_documents()
    pd.DataFrame(readiness_rows()).to_csv(OUT / "STAGE5DA_EXECUTION_READINESS_MATRIX.tsv", sep="\t", index=False, lineterminator="\n")
    freeze = freeze_preregistration()
    final_report()
    after = {str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise RuntimeError(f"STOP: historical outputs changed: {changed}")
    return {"stage": "5D-A", "status": "BUILT_PENDING_TESTS", "baseline": baseline, "freeze": freeze}


def finalize(test_results: dict) -> dict:
    baseline = verify_baseline()
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    if missing:
        raise RuntimeError(f"STOP: missing Stage 5D-A artifacts: {missing}")
    if not test_results.get("passed"):
        raise RuntimeError(f"STOP: tests did not pass: {test_results}")
    _json(OUT / "STAGE5DA_TESTS.json", test_results)
    checkpoint_path = OUT / "STAGE5DA_CHECKPOINT.json"
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p != checkpoint_path)
    output_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    freeze = json.loads((OUT / "STAGE5DA_PREREGISTRATION_FREEZE.json").read_text())
    readiness = pd.read_csv(OUT / "STAGE5DA_EXECUTION_READINESS_MATRIX.tsv", sep="\t").set_index("ITEM")["STATUS"].to_dict()
    git_status = _git("status", "--short")
    checkpoint = {
        "stage": "5D-A", "status": "COMPLETE", "audit_date": "2026-10-08",
        "historical_verification": baseline,
        "upstream_checkpoint_hashes": {k: v[1] for k, v in UPSTREAM.items()},
        "stage5c_scientific_hashes": STAGE5C_SCIENTIFIC,
        "preregistration_hash": freeze["preregistration_sha256"],
        "candidate_contracts_hash": freeze["candidate_contracts_sha256"],
        "selection_guards_hash": freeze["selection_guards_sha256"],
        "code_hashes": {"src/sfv2/build_stage5da.py": sha256(ROOT / "src/sfv2/build_stage5da.py"), "tests/test_stage5da.py": sha256(ROOT / "tests/test_stage5da.py")},
        "new_artifact_hashes": output_hashes, "readiness_decisions": readiness, "test_results": test_results,
        "software_versions": {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__},
        "git_head": _git("rev-parse", "HEAD"), "git_working_tree_dirty": bool(git_status),
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_note": "Pre-existing user work preserved; no commit or push.",
        "source_code_changes": ["src/sfv2/build_stage5da.py", "tests/test_stage5da.py"],
        "remaining_blockers": ["no two additional qualified development landscapes", "execution membership not frozen", "Stage 5D-B implementation not written or audited", "no final model-selection result"],
        "V1_INTACT": "YES", "HISTORICAL_OUTPUTS_UNCHANGED": "YES", "LOGISTIC_M_UNCHANGED": "YES",
        "M_SCHEMA_UNCHANGED": "YES", "STAGE5CA_PROTOCOL_UNCHANGED": "YES", "STAGE5CA_PREREGISTRATION_UNCHANGED": "YES",
        "STAGE5CA_SUCCESS_RULES_UNCHANGED": "YES", "H_GATES_UNCHANGED": "YES", "GEN3_UNCHANGED": "YES",
        "MEGH_TRAINED": "NO", "NEW_MODEL_TRAINED": "NO", "MODEL_PERFORMANCE_EVALUATED": "NO",
        "NEW_REAL_DATA_INGESTED": "NO", "LOCKBOX_OUTCOMES_ACCESSED": "NO", "COMMIT_CREATED": "NO", "PUSH_PERFORMED": "NO",
        "hash_policy": "all Stage 5D-A outputs except the checkpoint itself are hashed; the checkpoint cannot self-hash",
    }
    _json(checkpoint_path, checkpoint)
    return {"checkpoint": str(checkpoint_path.relative_to(ROOT)), "sha256": sha256(checkpoint_path), "hashed_outputs": len(output_hashes)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "finalize"])
    parser.add_argument("--tests-json")
    args = parser.parse_args()
    if args.phase == "build":
        print(json.dumps(build(), indent=2, sort_keys=True))
    else:
        if not args.tests_json:
            raise SystemExit("--tests-json required")
        print(json.dumps(finalize(json.loads(Path(args.tests_json).read_text())), indent=2, sort_keys=True))
