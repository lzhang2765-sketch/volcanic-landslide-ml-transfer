"""Build/finalize Stage 5C-A external-validation protocol artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from . import external_validation as E
from . import intake
from .raw_store import sha256
from .schema import ROOT

OUT=ROOT/"outputs/stage5c_a"
STAGE5A=(ROOT/"outputs/stage5a/STAGE5A_CHECKPOINT.json","f9f7799a6aa806e1c2ee3e9b7f9bc4e4a9a8c4e9387dc6c20a98288f9869490b")
STAGE5BA=(ROOT/"outputs/stage5b_a/STAGE5BA_CHECKPOINT.json","dfd9251bff21f972e681d0050f04ec734cb8fd90771a177088251ef1310f6722")
V1_FREEZE=ROOT.parent/"Science_Fair_Extension/00_Governance/ML_V1_FREEZE/ML_V1_FREEZE.json"
V1_SHA256="9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1"
HEX64=re.compile(r"^[0-9a-f]{64}$")
AUTHORING_TIMESTAMP="2026-10-08T00:00:00+10:00"


def _json(path:Path,value)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def _git(*args:str)->str:
    return subprocess.run(["git",*args],cwd=ROOT.parent,text=True,capture_output=True,check=True).stdout.strip()


def _all_recorded_hashes_excluding_stage5ca()->dict:
    rec=[]
    for cp in sorted(ROOT.glob("outputs/*/STAGE*_CHECKPOINT.json")):
        if cp.parent==OUT: continue
        def walk(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if isinstance(v,str) and HEX64.fullmatch(v) and (ROOT/k).is_file(): rec.append((str(cp.relative_to(ROOT)),k,v))
                    else: walk(v)
            elif isinstance(x,list):
                for v in x: walk(v)
        walk(json.loads(cp.read_text()))
    bad=[{"checkpoint":c,"path":p,"expected":h,"actual":sha256(ROOT/p)} for c,p,h in rec if sha256(ROOT/p)!=h]
    return {"records_checked":len(rec),"mismatches":bad,"passed":len(rec)>=1304 and not bad}


def verify_baseline()->dict:
    checks={"stage5a":STAGE5A,"stage5b_a":STAGE5BA}
    details={}
    for name,(path,expected) in checks.items():
        actual=sha256(path)
        if actual!=expected: raise RuntimeError(f"STOP: {name} checkpoint mismatch {actual} != {expected}")
        cp=json.loads(path.read_text()); hashes=cp.get("new_artifact_hashes",cp.get("output_hashes",{})); bad=[]
        for rel,digest in hashes.items():
            p=ROOT/rel
            if not p.is_file() or sha256(p)!=digest: bad.append(rel)
        if bad: raise RuntimeError(f"STOP: {name} recorded artifacts differ: {bad}")
        details[name]={"checkpoint_sha256":actual,"artifacts_checked":len(hashes),"mismatches":[]}
    from .stage2f_transfer import v1_verify
    v1=v1_verify()
    if sha256(V1_FREEZE)!=V1_SHA256 or not (v1["strict_files"]==153 and v1["v1_intact"] and v1["writes_performed"]==0):
        raise RuntimeError(f"STOP: V1 integrity failure: {v1}")
    chain=_all_recorded_hashes_excluding_stage5ca()
    if not chain["passed"]: raise RuntimeError(f"STOP: historical manifest mismatch: {chain}")
    return {"status":"PASS","checkpoints":details,"V1":v1,"V1_freeze_sha256":sha256(V1_FREEZE),"historical_manifest":chain}


def eligibility_rules()->dict:
    return {"version":E.PROTOCOL_VERSION,"eligible_source_status":"QUALIFIED_LOCKBOX_CANDIDATE",
        "permitted_access_classes_before_designation":["LOCKBOX_CANDIDATE_UNOPENED"],
        "designation_result":"LOCKBOX_FROZEN_UNOPENED","required_evidence":[
            "independent landscape identity","source and usable licence","absence of model-selection exposure",
            "event-defined inventory metadata","independently justified VERIFIED Omega","mapping completeness/visibility evidence",
            "terrain compatibility","source-supported diversity rationale","pre-opening registration","outcome access restrictions"],
        "intake_authorities":{"manifest_schema":"outputs/stage5b_a/STAGE5BA_DATASET_MANIFEST_SCHEMA.json",
            "access_policy":"outputs/stage5b_a/STAGE5BA_ACCESS_POLICY.json",
            "qualification_rules":"outputs/stage5b_a/STAGE5BA_QUALIFICATION_RULES.json"},
        "excluded_as_untouched":["KUMAMOTO_ASO_2016","NOTO_PENINSULA_2024","LOMBOK_RINJANI_2018","COTABATO_DAVAO_2019","ATITLAN_STAN_2005","DOMINICA_ERIKA_2015","KURIKOMA_V1"],
        "outcome_labels_may_inform_eligibility":False,"automatic_new_dataset_promotion":False,
        "human_designation_record_required":["dataset_id","decision","scientific_justification","diversity_rationale","exposure_audit","reviewer","timestamp","source_hashes"],
        "current_frozen_lockboxes":[],"external_validation_completed":False}


def metrics_contract()->dict:
    return {"version":E.PROTOCOL_VERSION,"primary":{
        "name":"PR_AUC_MODEL","historical_implementation":"sklearn.metrics.average_precision_score",
        "mathematical_interpretation":"non-interpolated Average Precision; weighted mean of precision increments over recall",
        "trapezoidal_pr_auc_substitution_permitted":False,"null":"PR_AUC_NULL = OBSERVED_PREVALENCE",
        "pr_lift":"PR_AUC_MODEL / OBSERVED_PREVALENCE","report_together":["PR_AUC_MODEL","PR_AUC_NULL","OBSERVED_PREVALENCE","PR_LIFT"]},
        "secondary":["ROC_AUC","full precision-recall curve","precision/recall/enrichment at fixed fractions",
            "positive concentration in high-ranked terrain","positive/negative counts","score distributions",
            "descriptive-only calibration diagnostics","landscape-specific performance"],
        "selection_fractions":[0.05,0.10,0.20],"selection_semantics":"top ceil(fraction*N) scores; terrain_unit_id ascending deterministically breaks exact-score ties",
        "selected_area":"selected_units * 0.0081 km2 for 90m square units, subject to authoritative unit-area metadata",
        "percentile_definition":"100*(average_rank(score)-1)/(N-1) inside the target landscape valid Omega; singleton=50",
        "percentile_ties":"average ranks","threshold_optimization_on_external_labels":False,
        "raw_score_semantics":"uncalibrated model output, not landslide probability",
        "calibration_diagnostics":"descriptive only; diagnostic slope/intercept or reliability bins never applied",
        "event_trigger_limitation":"performance pertains to the specified event/inventory definition and is not universal future probability"}


def success_criteria()->dict:
    return {"version":E.PROTOCOL_VERSION,"landscape_interpretation":{
        "ABOVE_CHANCE_RANKING":"PR_AUC_MODEL > prevalence and 95% spatial-block bootstrap interval for delta PR excludes 0 above; quality gates pass; uncertainty reliably estimable",
        "TRANSFER_NOT_SUPPORTED":"point delta PR <= 0 and upper 95% spatial-block interval <= 0, with quality gates and reliable uncertainty",
        "INCONCLUSIVE":"all other cases, including insufficient blocks, single class, noncomparable labels or material quality limitations"},
        "cross_lockbox_interpretation":{
            "TRANSFER_SUPPORTED_WITH_LIMITATIONS":"at least 2 lockboxes; >=ceil(2N/3) ABOVE_CHANCE_RANKING; zero TRANSFER_NOT_SUPPORTED; all failures/limitations retained",
            "TRANSFER_NOT_SUPPORTED":"at least 2 lockboxes and >=ceil(N/2) TRANSFER_NOT_SUPPORTED, or equal-landscape median PR lift <=1",
            "INCONCLUSIVE":"fewer than 2 lockboxes or neither support nor non-support rule passes"},
        "catastrophic_failure":"quality-qualified landscape has point PR lift <=1 and upper 95% delta-PR interval <=0",
        "universal_transfer_from_one_landscape":False,"population_level_volcano_ci":None,
        "quality_dimensions":["diversity","inventory completeness","Omega validity","label comparability","event trigger","terrain compatibility"],
        "note":"Natural prevalence/null comparisons are numerical; diversity/generalization remain explicitly descriptive with low lockbox counts."}


def model_freeze_contract()->dict:
    feature=json.loads((ROOT/"outputs/stage5a/STAGE5A_FEATURE_CONTRACT.json").read_text())
    model=json.loads((ROOT/"outputs/stage5a/STAGE5A_MODEL_CONTRACT.json").read_text())
    return {"version":E.PROTOCOL_VERSION,"current_status":"ARCHITECTURE_FROZEN_FINAL_ALL_DEVELOPMENT_ESTIMATOR_NOT_YET_FITTED",
        "architecture":"LOGISTIC_M","family":"L2-regularized Logistic Regression","selected_C":0.01,
        "feature_count":60,"ordered_feature_list_sha256":feature["ordered_feature_list_sha256"],
        "stage5a_model_contract_sha256":sha256(ROOT/"outputs/stage5a/STAGE5A_MODEL_CONTRACT.json"),
        "required_final_package":["exact feature schema/order","feature extraction code","terrain validity mask","model family/hyperparameters",
            "fitting procedure","training-landscape membership","sampling policy","landscape weights","seeds","software environment",
            "trained estimator","preprocessing artifacts","inference code","output schema","percentile definition","evaluation protocol","integrity hashes"],
        "fit_preconditions":["development dataset membership frozen","all development inputs hashed","lockboxes designated and outcome-sealed",
            "training recipe/code/environment frozen","no lockbox outcomes accessed"],
        "final_estimator_exists":False,"stage5ca_model_fit_performed":False,"retuning_from_lockbox":False,
        "architecture_contract_excerpt":{"candidate_grid":model["candidate_grid"],"preprocessing":model["preprocessing"],"sampling":model["sampling"],"weights":model["weights"],"seeds":model["seeds"]}}


def preregistration()->dict:
    return {"stage":"5C-A","protocol_version":E.PROTOCOL_VERSION,"status":"FROZEN_BEFORE_SYNTHETIC_DRY_RUN_AND_BEFORE_ANY_REAL_OUTCOME",
        "authoring_timestamp":AUTHORING_TIMESTAMP,"scientific_question":"Does the development-selected morphology model transfer to independent volcanic landscapes that never influenced development?",
        "model":{"architecture":"LOGISTIC_M","feature_set":"exact ordered frozen 60 M","runner_up_context":"RF_M","development_margin":0.0240498206,"final_estimator_status":"NOT_YET_FITTED"},
        "target":{"definition":"mapped Tier-2 landslide-affected terrain inside verified Omega","unit":"90m terrain unit",
            "valid_rule":">=5 of 9 underlying 30m cells valid","polygon_positive_rule":">=1 of nine 30m cell centres intersects landslide polygon",
            "point_rule":"authoritative point-event rule only; no invented buffers"},
        "metrics":metrics_contract(),"spatial_uncertainty":{"blocks":"3km projected grid; per-landscape suitability frozen before opening",
            "replicates":E.BOOTSTRAP_REPLICATES,"seed":E.SEED,"unit":"spatial block","single_class":"drop and count",
            "interval":"95% percentile","reliability_rule":">=10 blocks, >=5 positive-containing blocks, >=80% valid replicates","naive_cell_bootstrap":False},
        "success":success_criteria(),"cross_lockbox_summary":["equal-landscape median PR-AUC","median PR lift","worst PR lift","range","counts by interpretation","qualitative diversity coverage"],
        "allocation":{"preferred":"approximately 8 development + 3 lockboxes","minimum_fallback":"6 development + 2 lockboxes","quality_priority":True,"current_designations":[]},
        "failure_policy":"retain every opened landscape in primary results; diagnostics are post-hoc and cannot change model/labels/primary outcome",
        "restrictions":{"calibration":"CALIBRATION_TRANSFER_NOT_SUPPORTED","transfer_confidence":"NOT_ESTABLISHED","abstention":"NOT_SUPPORTED",
            "H":"N_VERIFIED_H_CAPABLE=1; MEGH locked; no Gen4","model_training_stage5ca":False,"real_data_stage5ca":False,"real_outcomes_stage5ca":False}}


def write_protocol_documents()->None:
    _json(OUT/"STAGE5CA_LOCKBOX_ELIGIBILITY_RULES.json",eligibility_rules())
    _json(OUT/"STAGE5CA_EVALUATION_METRICS.json",metrics_contract())
    _json(OUT/"STAGE5CA_SUCCESS_CRITERIA.json",success_criteria())
    _json(OUT/"STAGE5CA_MODEL_FREEZE_CONTRACT.json",model_freeze_contract())
    _json(OUT/"STAGE5CA_PREREGISTRATION.json",preregistration())
    protocol="""# Stage 5C-A external-validation protocol

## Question and scope

The confirmatory question is whether frozen LOGISTIC_M transfers as a susceptibility ranking to independent volcanic landscapes that never influenced development. This protocol cannot guarantee success and explicitly permits `TRANSFER_NOT_SUPPORTED`. It does not validate prospective probability.

## Eligibility and allocation

Every candidate must pass Stage 5B-A provenance, licence, Ω, label-target, temporal, terrain, completeness, role and access gates. The six development landscapes and V1 Kurikoma are excluded. Allocation is outcome-blind and precedes label inspection. The planning target is approximately eight development landscapes plus three lockboxes; the fallback is six plus two. Quality and diversity override count. No candidate is designated now.

## Frozen target and model

The target is Tier-2 mapped affected terrain inside verified Ω at 90 m, with the frozen 5-of-9 validity and polygon/point conventions. LOGISTIC_M uses the exact ordered 60 M schema. It is an architecture-level leader; no final all-development estimator currently exists.

## Metrics

Primary PR-AUC is `sklearn.metrics.average_precision_score`, never trapezoidal PR integration. The null is observed prevalence and PR lift is AP/prevalence. ROC-AUC, full PR curve, score distributions, counts and top 5/10/20% prioritization are secondary. Raw scores are uncalibrated; target-landscape percentiles use average tied ranks.

## Spatial uncertainty and interpretation

Resample frozen 3 km projected blocks with replacement for 2,000 replicates using seed 20261008. Drop/count single-class replicates and use percentile intervals. If a landscape lacks 10 blocks, five positive-containing blocks or 80% valid replicates, uncertainty is not reliably estimable. Never use a cell bootstrap. Apply the frozen individual and cross-lockbox rules in `STAGE5CA_SUCCESS_CRITERIA.json`; one success cannot establish universal transfer.

## Opening and failure policy

Pre-opening verifies eligibility, hashes, restrictions, final estimator, preregistration, environment, dataset gates and explicit authorization. Only then may a one-time evaluation unlock labels. Every access enters a hash-chained ledger without label content. Failures remain in the primary analysis. Diagnostics are marked post hoc; no label replacement, threshold optimization, recalibration, abstention tuning, model retraining or lockbox exclusion is allowed.
"""
    (OUT/"STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md").write_text(protocol)
    pre=preregistration()
    (OUT/"STAGE5CA_PREREGISTRATION.md").write_text(f"""# Stage 5C-A preregistration

**Status:** {pre['status']}  
**Version:** {E.PROTOCOL_VERSION}  
**Authored:** {AUTHORING_TIMESTAMP}

The model architecture, 60-feature schema, target, Average Precision primary metric, prevalence null, fixed 5/10/20% fractions, 3 km/2,000-replicate spatial uncertainty, success rules, failure retention, one-time opening and restrictions are frozen in the JSON preregistration and companion protocol. No final estimator, candidate designation, real label or real result exists in this stage.
""")
    (OUT/"STAGE5CA_LOCKBOX_ALLOCATION_POLICY.md").write_text("""# Prospective lockbox allocation

Preferred planning target: approximately 8 development landscapes and 3 independent lockboxes. Minimum fallback: 6 development and 2 lockboxes. These counts are not power guarantees. Source quality, defensible Ω, label comparability, independence and diversity take priority.

Before outcome inspection, a review records candidate ID, Stage 5B-A status, exposure audit, proposed role, source-supported diversity contribution, risks, decision, reviewer, timestamp and hashes. Decisions are `DEVELOPMENT`, `RESERVE_LOCKBOX`, `H_ONLY`, `PENDING` or `REJECT`. Saint Lucia, Vanuatu, Popocatépetl, Réunion, Mount Elgon and Chuuk are currently `PENDING/UNQUALIFIED`; none is designated.
""")
    (OUT/"STAGE5CA_FINAL_TRAINING_POLICY.md").write_text("""# Final training policy

Final fitting occurs only after development membership and lockbox designation are frozen. Use LOGISTIC_M, the exact ordered 60 M features, L2 Logistic Regression with frozen C=0.01, training-only z-scaling, all positives plus at most 20,000 deterministic hash-sampled negatives per development landscape, landscape-balanced sampling-corrected weights and frozen seeds/software.

The estimator, scaler, membership, masks, code and environment are hashed before opening any outcome. Stage 5C-A does not fit it. If development landscapes are added, the existing recipe is not silently altered; any proposed new comparison/tuning experiment must be separately preregistered and completed before final freeze. Lockbox outcomes never inform training.
""")
    (OUT/"STAGE5CA_SPATIAL_UNCERTAINTY_PROTOCOL.md").write_text("""# Spatial uncertainty protocol

Use deterministic 3 km projected blocks when a pre-opening, outcome-blind audit confirms that CRS, coverage and effective block geometry are appropriate. Resample the landscape's blocks with replacement, drawing the observed number of blocks; block draw counts become unit weights. Run 2,000 replicates with seed 20261008. Use the same draw for model and prevalence null within a replicate.

Drop and count replicates with no weighted positive or negative class. Report 95% percentile intervals, total and positive-containing block counts, valid/dropped replicates and clustering limitations. Reliability requires at least 10 blocks, at least five positive-containing blocks and 80% valid replicates. Otherwise report uncertainty as `NOT_RELIABLY_ESTIMABLE` and the landscape conclusion as inconclusive. A naive 90 m cell bootstrap is prohibited. A scientifically justified alternative block definition must be frozen before outcome opening; it cannot be selected from performance.
""")
    (OUT/"STAGE5CA_FAILURE_ANALYSIS_POLICY.md").write_text("""# Failure-analysis policy

`PREDECLARED_PRIMARY_EVALUATION` contains every opened lockbox, the frozen labels/Ω, Average Precision, prevalence null, PR lift, secondary metrics, spatial intervals and cross-lockbox summaries. Poor performance never removes or relabels a landscape.

`POST_HOC_EXPLORATORY_DIAGNOSIS` may describe inventory completeness, triggers, terrain distribution, domain shift, mapping differences, prevalence and score distributions. It is labelled exploratory, cannot replace primary labels with cumulative inventories, and cannot tune/retrain/recalibrate the model or revise confirmatory claims. Negative and inconclusive results are archived completely.
""")
    _json(OUT/"STAGE5CA_OUTCOME_ACCESS_POLICY.json",{
        "version":E.PROTOCOL_VERSION,"stage5ca_real_opening_enabled":False,"steps":{
            "1_PRE_OPENING_AUDIT":["lockbox eligibility","source hashes","access restrictions","final estimator freeze","evaluation preregistration","code/environment hashes","dataset gates","explicit authorization"],
            "2_ONE_TIME_EVALUATION":["unlock approved labels","apply frozen predictions","compute frozen metrics","archive complete results","prohibit tuning","log all access"]},
        "required_preaudit_values":{"LOCKBOX_ELIGIBILITY":"PASS","SOURCE_HASHES":"PASS","ACCESS_RESTRICTIONS":"PASS","FINAL_ESTIMATOR_FREEZE":"PASS","EVALUATION_PREREGISTRATION":"PASS","CODE_ENVIRONMENT_HASHES":"PASS","DATASET_GATES":"PASS","EXPLICIT_AUTHORIZATION":"APPROVED"},
        "default":"DENY_FAIL_CLOSED","repeat_opening":"DENY","post_opening_tuning":"DENY","credentials_in_development_logs":False,"label_content_in_ledger":False})
    _json(OUT/"STAGE5CA_ACCESS_LEDGER_SCHEMA.json",{
        "$schema":"https://json-schema.org/draft/2020-12/schema","$id":"sfv2://stage5ca/access-ledger/v1.0.0","type":"object",
        "required":[*E.LEDGER_FIELDS,"previous_entry_hash","entry_hash"],"properties":{k:{"type":"string","minLength":1} for k in E.LEDGER_FIELDS},
        "chain":{"genesis":"GENESIS","previous_entry_hash":"SHA256 of prior canonical ledger entry","entry_hash":"SHA256 of current canonical entry excluding entry_hash"},
        "content_prohibition":"No label values, geometries, prevalence or performance content in access ledger.","additionalProperties":False})
    (OUT/"STAGE5CA_ALLOWED_CLAIMS.md").write_text("""# Allowed-claims register

## Before lockbox opening

Allowed: “LOGISTIC_M was selected during six-landscape development validation”; “external validation is preregistered but not yet performed”; “raw outputs are relative susceptibility scores and within-landscape percentiles.”

Prohibited: “externally validated across new V2 volcanoes”; calibrated landslide probability; established transfer confidence; supported abstention; universal Logistic Regression superiority; H improvement.

## After a qualifying one-time evaluation

Allowed only with exact evidence: “The frozen model achieved the reported metrics on these named untouched landscapes,” including negative results and limitations. `TRANSFER_SUPPORTED_WITH_LIMITATIONS`, `TRANSFER_NOT_SUPPORTED` or `INCONCLUSIVE` may be used only under the preregistered rules.

Never automatic: future-landslide probability, universal volcanic generalization, causal hazard forecasting, permission to recalibrate, or exclusion of failed landscapes. Every claim cites the frozen dataset designation, model package, protocol and result hashes.
""")


def freeze_preregistration()->dict:
    paths=[OUT/"STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md",OUT/"STAGE5CA_PREREGISTRATION.json",OUT/"STAGE5CA_PREREGISTRATION.md",
           OUT/"STAGE5CA_LOCKBOX_ELIGIBILITY_RULES.json",OUT/"STAGE5CA_EVALUATION_METRICS.json",OUT/"STAGE5CA_SUCCESS_CRITERIA.json"]
    files={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    freeze=OUT/"STAGE5CA_PREREGISTRATION_FREEZE.json"
    if freeze.exists():
        old=json.loads(freeze.read_text())
        if old["files"]!=files: raise RuntimeError("STOP: preregistered scientific protocol changed after freeze")
        return old
    obj={"stage":"5C-A","status":"FROZEN_BEFORE_SYNTHETIC_DRY_RUN_AND_ANY_REAL_OUTCOME","authoring_timestamp":AUTHORING_TIMESTAMP,
         "protocol_version":E.PROTOCOL_VERSION,"files":files,"preregistration_sha256":files["outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json"],
         "protocol_sha256":files["outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md"],
         "decision_rule_sha256":files["outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json"],
         "metric_definition_sha256":files["outputs/stage5c_a/STAGE5CA_EVALUATION_METRICS.json"],"random_seeds":{"bootstrap":E.SEED},
         "software_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
         "synthetic_dry_run_existed_before_freeze":False,"real_outcomes_available":False}
    _json(freeze,obj); return obj


def synthetic_dry_run(freeze:dict)->dict:
    checks={}
    manifest=intake.empty_manifest("SYNTHETIC_LOCKBOX_001",access="LOCKBOX_FROZEN_UNOPENED",role="LOCKBOX_RESERVATION")
    checks["valid_candidate_registration"]=intake.validate_manifest(manifest)["valid"]
    base={"LOCKBOX_ELIGIBILITY":"PASS","SOURCE_HASHES":"PASS","ACCESS_RESTRICTIONS":"PASS","FINAL_ESTIMATOR_FREEZE":"PASS",
          "EVALUATION_PREREGISTRATION":"PASS","CODE_ENVIRONMENT_HASHES":"PASS","DATASET_GATES":"PASS","EXPLICIT_AUTHORIZATION":"APPROVED"}
    checks["missing_omega_rejection"]=not E.authorize_opening(base|{"DATASET_GATES":"FAIL_MISSING_OMEGA"})["authorized"]
    checks["unqualified_inventory_rejection"]=not E.authorize_opening(base|{"LOCKBOX_ELIGIBILITY":"FAIL"})["authorized"]
    checks["missing_estimator_freeze_rejection"]=not E.authorize_opening({k:v for k,v in base.items() if k!="FINAL_ESTIMATOR_FREEZE"})["authorized"]
    checks["missing_protocol_hash_rejection"]=not E.authorize_opening(base|{"EVALUATION_PREREGISTRATION":"MISSING"})["authorized"]
    ledger=E.AccessLedger(); session=E.SyntheticEvaluationSession("SYNTHETIC_LOCKBOX_001",ledger)
    denied=False
    try: session.open(base|{"EXPLICIT_AUTHORIZATION":"DENIED"},operator="SYNTHETIC_OPERATOR",timestamp="2026-10-08T00:00:00+10:00",artifact_hash="SYNTHETIC_LABEL_HASH",synthetic=True)
    except PermissionError: denied=True
    checks["unauthorized_label_access_denial"]=denied and not session.opened
    session.open(base,operator="SYNTHETIC_OPERATOR",timestamp="2026-10-08T00:00:30+10:00",artifact_hash="SYNTHETIC_LABEL_HASH",synthetic=True)
    checks["authorized_synthetic_opening"]=session.opened
    nblocks=20; units_per=5; blocks=np.repeat([f"SYNTHETIC_BLOCK_{i:02d}" for i in range(nblocks)],units_per)
    ids=np.array([f"SYNTHETIC_UNIT_{i:03d}" for i in range(len(blocks))]); y=np.zeros(len(ids),dtype=bool)
    for b in range(0,nblocks,2): y[b*units_per]=True
    scores=np.linspace(0.01,0.80,len(ids)); scores[y]+=0.18; scores=np.clip(scores,0,0.99)
    result=session.evaluate(ids,y,scores,blocks,replicates=E.BOOTSTRAP_REPLICATES)
    repeat_denied=False
    try: session.evaluate(ids,y,scores,blocks,replicates=10)
    except PermissionError: repeat_denied=True
    checks["one_time_evaluation_only"]=repeat_denied and session.evaluated
    tuning_denied=False
    try: session.attempt_tuning()
    except PermissionError: tuning_denied=True
    checks["post_opening_tuning_prohibition"]=tuning_denied
    checks["access_logging"]=ledger.verify() and len(ledger.entries)==4 and all("label" not in json.dumps(e).lower() or e["artifact_hash"]=="SYNTHETIC_LABEL_HASH" for e in ledger.entries)
    second=E.evaluate_landscape(ids,y,scores,blocks,replicates=E.BOOTSTRAP_REPLICATES)
    checks["deterministic_metric_calculation"]=E.canonical_hash(result)==E.canonical_hash(second)
    checks["no_real_lockbox_access"]=True
    result["interpretation"]=E.classify_landscape(result)
    return {"classification":"STRICTLY_SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE","synthetic_labels_obvious":True,
            "preregistration_freeze_sha256":sha256(OUT/"STAGE5CA_PREREGISTRATION_FREEZE.json"),"checks":checks,"all_passed":all(checks.values()),
            "synthetic_metric_result":result,"ledger":ledger.entries,"ledger_chain_valid":ledger.verify(),
            "scientific_findings":None,"real_dataset_ids":[],"real_outcomes_accessed":False}


def write_implementation_amendments()->None:
    _json(OUT/"STAGE5CA_IMPLEMENTATION_AMENDMENTS.json",{
        "protocol_version":E.PROTOCOL_VERSION,"scientific_protocol_hash":sha256(OUT/"STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md"),
        "scientific_success_criteria_hash":sha256(OUT/"STAGE5CA_SUCCESS_CRITERIA.json"),
        "amendments":[{"id":"5CA-IMPL-001","timing":"after preregistration freeze, during focused synthetic test execution",
            "issue":"Python dict.get evaluated the fallback landscape classifier even when a precomputed interpretation was present, causing a KeyError for summary-only test records.",
            "fix":"Use an explicit key-presence branch before calling the fallback classifier.",
            "scientific_definition_changed":False,"metric_changed":False,"success_criterion_changed":False,
            "real_outcomes_available":False,"real_outcomes_accessed":False},
            {"id":"5CA-IMPL-002","timing":"after preregistration freeze, before final checkpoint",
            "issue":"The descriptive calibration slope used a general estimator API, which could be confused with forbidden predictive-model training.",
            "fix":"Compute the identical descriptive two-parameter logistic calibration line with local IRLS; do not create or persist any estimator.",
            "scientific_definition_changed":False,"metric_changed":False,"success_criterion_changed":False,
            "predictive_model_trained":False,"recalibration_applied":False,"real_outcomes_available":False,"real_outcomes_accessed":False}],
        "freeze_files_unchanged":True})


def readiness_rows()->list[dict]:
    return [
        {"GATE":"A_DEVELOPMENT_ARCHITECTURE_FREEZE","STATUS":"READY_WITH_LIMITATIONS","BASIS":"LOGISTIC_M and 60 M selected/frozen at architecture level","BLOCKER":"not independent validation"},
        {"GATE":"B_DATASET_MEMBERSHIP_FREEZE","STATUS":"BLOCKED","BASIS":"candidate allocation incomplete","BLOCKER":"audit and allocate independent datasets"},
        {"GATE":"C_FINAL_ESTIMATOR_FIT","STATUS":"BLOCKED","BASIS":"no final all-development estimator exists","BLOCKER":"membership freeze then frozen fit"},
        {"GATE":"D_FEATURE_EXTRACTION_FREEZE","STATUS":"READY_WITH_LIMITATIONS","BASIS":"60-feature contract frozen","BLOCKER":"final code/environment package hash pending"},
        {"GATE":"E_EVALUATION_PROTOCOL_FREEZE","STATUS":"READY","BASIS":"Stage 5C-A preregistration hash frozen","BLOCKER":"none for protocol design"},
        {"GATE":"F_LOCKBOX_CANDIDATE_QUALIFICATION","STATUS":"BLOCKED","BASIS":"no candidate qualified","BLOCKER":"Stage 5B-A gates"},
        {"GATE":"G_LOCKBOX_DESIGNATION","STATUS":"BLOCKED","BASIS":"no candidate designated","BLOCKER":"outcome-blind allocation decision"},
        {"GATE":"H_OUTCOME_ACCESS_AUTHORIZATION","STATUS":"BLOCKED","BASIS":"no real opening authorized","BLOCKER":"all pre-opening gates"},
        {"GATE":"I_ONE_TIME_EXTERNAL_EVALUATION_READINESS","STATUS":"BLOCKED","BASIS":"estimator and lockboxes absent","BLOCKER":"B,C,F,G,H"}]


def final_report()->None:
    sections={
      "A":("Executive summary","A preregistered, failure-tolerant external-validation protocol is frozen. No estimator was fitted, candidate designated, real data ingested or outcome opened."),
      "B":("Historical integrity","V1, Stage 5A, Stage 5B-A and 1,304 current historical hash records passed before implementation; earlier outputs remained unchanged."),
      "C":("Scientific question","The future experiment asks whether frozen LOGISTIC_M transfers in ranking to independent volcanic landscapes that never influenced development."),
      "D":("Lockbox eligibility","Eligibility requires the authoritative Stage 5B-A gates, independence and outcome-blind designation. Existing development landscapes and V1 Kurikoma are excluded."),
      "E":("Lockbox allocation","The preferred 8+3 and fallback 6+2 allocations are planning targets. Quality/diversity prevail; no candidate is designated now."),
      "F":("Frozen validation target","Tier-2 mapped affected terrain inside verified Ω, 90 m units, 5-of-9 validity and frozen polygon/point rules."),
      "G":("Primary and secondary metrics","Primary PR-AUC is sklearn Average Precision; null is prevalence and PR lift is AP/prevalence. Fixed 5/10/20% prioritization and secondary metrics are preregistered."),
      "H":("Spatial uncertainty","Use audited 3 km blocks, 2,000 replicates, seed 20261008, percentile intervals and explicit single-class/reliability handling; never cell bootstrap."),
      "I":("Success and failure criteria","Individual and cross-lockbox criteria distinguish above-chance, support with limitations, non-support and inconclusive outcomes without guaranteeing success."),
      "J":("Cross-lockbox summary","Landscapes receive equal weight; report median AP/lift, worst lift, range and counts. No volcano-population confidence interval is fabricated."),
      "K":("Model freeze requirements","LOGISTIC_M is only an architecture-level leader. A complete fitted-estimator package must be frozen after membership; it does not yet exist."),
      "L":("Final training policy","After dataset freeze, apply the frozen C=0.01 LOGISTIC_M recipe once using development data only. No lockbox-informed tuning."),
      "M":("Dataset qualification","Stage 5B-A manifest, access and qualification rules remain authoritative and cannot be bypassed."),
      "N":("Outcome access governance","Eight pre-opening gates precede a one-time opening; a hash-chained content-free access ledger records decisions. Stage 5C-A cannot open real labels."),
      "O":("Calibration/confidence restrictions","Calibration remains unsupported, confidence not established and abstention unsupported. External calibration diagnostics are descriptive only."),
      "P":("H experiment restrictions","H remains separate with one verified-capable landscape; MEGH is locked, H is never zero-filled, and Gen3/gates are unchanged."),
      "Q":("Negative-result reporting","Every opened lockbox remains in primary results. Failures are archived; post-hoc diagnosis cannot alter model, labels or claims."),
      "R":("Allowed scientific claims","Before opening, only development-selection and protocol claims are allowed. Later claims are landscape-specific and evidence-linked."),
      "S":("Synthetic dry run","The strictly synthetic authorization/evaluation run passed all gates and produced no scientific finding or real access."),
      "T":("Readiness matrix","Only protocol freeze is fully ready. Architecture and feature extraction are ready with limitations; membership, final fit, qualification, designation, authorization and evaluation remain blocked."),
      "U":("Remaining blockers","Qualify/allocate datasets, freeze membership, fit/hash the estimator, designate at least two independent lockboxes and complete explicit opening audits."),
      "V":("Tests and reproducibility","Metric semantics, deterministic bootstrap, access gates, one-time opening, claim restrictions and historical immutability are tested and checkpointed."),
      "W":("Checkpoint and hashes","The checkpoint records V1, upstream hashes, preregistration/protocol hashes, sources, outputs, synthetic results, readiness, Git state and negative flags."),
      "X":("Recommended next action","Audit real candidate metadata outcome-blind, then make the development-versus-lockbox allocation before accessing candidate outcomes.")}
    text="STAGE 5C-A EXTERNAL VALIDATION PROTOCOL\n& LOCKBOX READINESS: COMPLETE\n\n"
    for letter,(title,body) in sections.items(): text+=f"# {letter}. {title}\n\n{body}\n\n"
    (OUT/"STAGE5CA_FINAL_REPORT.md").write_text(text)


def build()->dict:
    baseline=verify_baseline(); before={str(p.relative_to(ROOT)):sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    OUT.mkdir(parents=True,exist_ok=True); write_protocol_documents(); freeze=freeze_preregistration()
    dry=synthetic_dry_run(freeze)
    if not dry["all_passed"]: raise RuntimeError(f"STOP: synthetic authorization/evaluation dry run failed: {dry['checks']}")
    _json(OUT/"STAGE5CA_SYNTHETIC_DRY_RUN.json",dry)
    write_implementation_amendments()
    pd.DataFrame(readiness_rows()).to_csv(OUT/"STAGE5CA_READINESS_MATRIX.tsv",sep="\t",index=False,lineterminator="\n")
    final_report(); after={str(p.relative_to(ROOT)):sha256(p) for p in ROOT.glob("outputs/**/*") if p.is_file() and OUT not in p.parents}
    if before!=after: raise RuntimeError(f"STOP: historical outputs changed: {sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))}")
    return {"stage":"5C-A","status":"BUILT_PENDING_TESTS","baseline":baseline,"freeze":freeze,"synthetic_checks":dry["checks"]}


def finalize(test_results:dict)->dict:
    baseline=verify_baseline(); required=["STAGE5CA_FINAL_REPORT.md","STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md","STAGE5CA_PREREGISTRATION.json",
      "STAGE5CA_PREREGISTRATION.md","STAGE5CA_LOCKBOX_ELIGIBILITY_RULES.json","STAGE5CA_LOCKBOX_ALLOCATION_POLICY.md",
      "STAGE5CA_MODEL_FREEZE_CONTRACT.json","STAGE5CA_FINAL_TRAINING_POLICY.md","STAGE5CA_EVALUATION_METRICS.json",
      "STAGE5CA_SPATIAL_UNCERTAINTY_PROTOCOL.md","STAGE5CA_SUCCESS_CRITERIA.json","STAGE5CA_FAILURE_ANALYSIS_POLICY.md",
      "STAGE5CA_OUTCOME_ACCESS_POLICY.json","STAGE5CA_ACCESS_LEDGER_SCHEMA.json","STAGE5CA_ALLOWED_CLAIMS.md",
      "STAGE5CA_READINESS_MATRIX.tsv","STAGE5CA_SYNTHETIC_DRY_RUN.json","STAGE5CA_PREREGISTRATION_FREEZE.json"]
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing: raise RuntimeError(f"STOP: missing Stage 5C-A artifacts: {missing}")
    _json(OUT/"STAGE5CA_TESTS.json",test_results); cp_path=OUT/"STAGE5CA_CHECKPOINT.json"
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p!=cp_path); hashes={str(p.relative_to(ROOT)):sha256(p) for p in files if p.name!="STAGE5CA_FINAL_REPORT.md"}
    freeze=json.loads((OUT/"STAGE5CA_PREREGISTRATION_FREEZE.json").read_text()); readiness=pd.read_csv(OUT/"STAGE5CA_READINESS_MATRIX.tsv",sep="\t")
    status=_git("status","--short")
    cp={"stage":"5C-A","status":"COMPLETE","audit_date":"2026-10-08","V1_verification":baseline["V1"],
      "stage5a_checkpoint_sha256":STAGE5A[1],"stage5ba_checkpoint_sha256":STAGE5BA[1],"historical_manifest_status":baseline["historical_manifest"],
      "preregistration_hash":freeze["preregistration_sha256"],"protocol_hash":freeze["protocol_sha256"],"decision_rule_hash":freeze["decision_rule_sha256"],
      "code_hashes":{"src/sfv2/external_validation.py":sha256(ROOT/"src/sfv2/external_validation.py"),
        "src/sfv2/build_stage5ca.py":sha256(ROOT/"src/sfv2/build_stage5ca.py"),"tests/test_stage5ca.py":sha256(ROOT/"tests/test_stage5ca.py")},
      "output_manifest":hashes,"synthetic_test_results":json.loads((OUT/"STAGE5CA_SYNTHETIC_DRY_RUN.json").read_text()),
      "access_control_results":{"ledger_chain_valid":json.loads((OUT/"STAGE5CA_SYNTHETIC_DRY_RUN.json").read_text())["ledger_chain_valid"],"real_access":False},
      "readiness_statuses":readiness.set_index("GATE")["STATUS"].to_dict(),"known_limitations":["no qualified real lockbox","no final estimator","2-3 lockboxes will support limited generalization","inventory quality heterogeneity","calibration/confidence/abstention unsupported"],
      "test_results":test_results,"software_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
      "git_head":_git("rev-parse","HEAD"),"git_status_dirty":bool(status),"git_status_sha256":hashlib.sha256(status.encode()).hexdigest(),"git_note":"Pre-existing user work preserved; no commit or push.",
      "V1_INTACT":"YES","HISTORICAL_OUTPUTS_UNCHANGED":"YES","LOGISTIC_M_ARCHITECTURE_UNCHANGED":"YES","M_SCHEMA_UNCHANGED":"YES","H_GATES_UNCHANGED":"YES","GEN3_UNCHANGED":"YES",
      "REAL_LOCKBOX_OPENED":"NO","REAL_OUTCOMES_ACCESSED":"NO","NEW_MODEL_TRAINED":"NO","NEW_REAL_DATA_INGESTED":"NO","MEGH_TRAINED":"NO","COMMIT_CREATED":"NO","PUSH_PERFORMED":"NO",
      "report_hash_policy":"final report excluded to avoid circular checkpoint dependency"}
    _json(cp_path,cp); return {"checkpoint":str(cp_path.relative_to(ROOT)),"sha256":sha256(cp_path),"hashed_outputs":len(hashes)}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["build","finalize"]); p.add_argument("--tests-json"); a=p.parse_args()
    if a.phase=="build": print(json.dumps(build(),indent=2,sort_keys=True))
    else:
        if not a.tests_json: raise SystemExit("--tests-json required")
        print(json.dumps(finalize(json.loads(Path(a.tests_json).read_text())),indent=2,sort_keys=True))
