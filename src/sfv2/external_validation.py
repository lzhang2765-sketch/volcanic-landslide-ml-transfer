"""Frozen-design utilities for future external validation and synthetic dry runs.

No function in this module trains or changes a predictive model.  The only
label-aware operations are metric calculation after an explicit authorization
gate.  Stage 5C-A itself permits those operations for synthetic fixtures only.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score
from scipy.stats import rankdata

PROTOCOL_VERSION = "1.0.0"
SEED = 20261008
BOOTSTRAP_REPLICATES = 2000
SELECTION_FRACTIONS = (0.05, 0.10, 0.20)
ELIGIBLE_LOCKBOX_CLASSES = {"LOCKBOX_CANDIDATE_UNOPENED", "LOCKBOX_FROZEN_UNOPENED"}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Frozen Stage 5A percentile: average ties; singleton receives 50."""
    scores=np.asarray(scores,dtype=float)
    if len(scores)==0: return np.array([],dtype=float)
    if len(scores)==1: return np.array([50.0])
    return 100.0*(rankdata(scores,method="average")-1.0)/(len(scores)-1.0)


def primary_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """Average precision is the historical PR-AUC implementation."""
    y=np.asarray(y,dtype=bool); scores=np.asarray(scores,dtype=float)
    if len(y)!=len(scores) or not len(y): raise ValueError("nonempty equal-length y and score arrays required")
    positives=int(y.sum()); negatives=int((~y).sum()); prevalence=float(y.mean())
    if positives==0 or negatives==0:
        return {"n_units":len(y),"positives":positives,"negatives":negatives,"observed_prevalence":prevalence,
                "pr_auc_model":None,"pr_auc_null":prevalence,"pr_lift":None,"roc_auc":None,
                "status":"NOT_ESTIMABLE_SINGLE_CLASS","implementation":"sklearn.metrics.average_precision_score"}
    ap=float(average_precision_score(y,scores))
    return {"n_units":len(y),"positives":positives,"negatives":negatives,"observed_prevalence":prevalence,
            "pr_auc_model":ap,"pr_auc_null":prevalence,"pr_lift":ap/prevalence,
            "delta_pr":ap-prevalence,"roc_auc":float(roc_auc_score(y,scores)),"status":"ESTIMATED",
            "implementation":"sklearn.metrics.average_precision_score (non-interpolated Average Precision)",
            "trapezoidal_pr_auc_used":False}


def prioritization_metrics(ids: np.ndarray, y: np.ndarray, scores: np.ndarray,
                           fractions: tuple[float,...]=SELECTION_FRACTIONS, unit_area_km2: float=0.0081) -> list[dict]:
    ids=np.asarray(ids).astype(str); y=np.asarray(y,dtype=bool); scores=np.asarray(scores,dtype=float)
    order=np.lexsort((ids,-scores))
    prevalence=float(y.mean())
    rows=[]
    for fraction in fractions:
        n_selected=max(1,int(math.ceil(fraction*len(y)))); idx=order[:n_selected]
        captured=int(y[idx].sum()); precision=float(captured/n_selected); recall=float(captured/max(int(y.sum()),1))
        rows.append({"selection_fraction":fraction,"selection_rule":"top ceil(fraction*N) by score; terrain_unit_id ascending breaks exact-score ties",
                     "selected_units":n_selected,"selected_terrain_area_km2":n_selected*unit_area_km2,
                     "positives_captured":captured,"recall":recall,"precision":precision,
                     "enrichment_over_prevalence":precision/prevalence if prevalence>0 else None})
    return rows


def descriptive_calibration(y: np.ndarray, scores: np.ndarray) -> dict:
    """Diagnostics only; the fitted slope/intercept are never applied to scores."""
    y=np.asarray(y,dtype=bool); p=np.clip(np.asarray(scores,dtype=float),1e-7,1-1e-7)
    if y.sum()==0 or y.sum()==len(y) or np.ptp(p)==0:
        return {"status":"NOT_ESTIMABLE","used_for_recalibration":False}
    z=np.log(p/(1-p))[:,None]
    # Two-parameter IRLS for a descriptive calibration line.  This is a metric
    # diagnostic, not a fitted/persisted predictive estimator or recalibrator.
    x=np.column_stack([np.ones(len(z)),z[:,0]]); beta=np.zeros(2,dtype=float)
    for _ in range(100):
        q=expit(x@beta); w=np.maximum(q*(1-q),1e-9)
        step=np.linalg.solve(x.T@(w[:,None]*x)+np.eye(2)*1e-12,x.T@(y.astype(float)-q))
        beta+=step
        if np.max(np.abs(step))<1e-10: break
    bins=np.minimum((percentile_rank(p)//10).astype(int),9)
    reliability=[]
    for b in range(10):
        take=bins==b
        if take.any(): reliability.append({"score_decile":b+1,"n":int(take.sum()),"mean_raw_score":float(p[take].mean()),"observed_prevalence":float(y[take].mean())})
    return {"status":"DESCRIPTIVE_ONLY","brier_raw_score":float(brier_score_loss(y,p)),
            "log_loss_raw_score":float(log_loss(y,p,labels=[0,1])),"diagnostic_intercept":float(beta[0]),
            "diagnostic_slope":float(beta[1]),"reliability_deciles":reliability,
            "used_for_recalibration":False,"raw_scores_called_probabilities":False}


def spatial_block_bootstrap(y: np.ndarray, scores: np.ndarray, blocks: np.ndarray, *, seed: int=SEED,
                            replicates: int=BOOTSTRAP_REPLICATES) -> dict:
    y=np.asarray(y,dtype=bool); scores=np.asarray(scores,dtype=float); blocks=np.asarray(blocks).astype(str)
    if not (len(y)==len(scores)==len(blocks)): raise ValueError("y, scores and blocks must have equal length")
    unique,inverse=np.unique(blocks,return_inverse=True); positive_blocks=int(len(np.unique(blocks[y])))
    rng=np.random.default_rng(seed); aps=[]; deltas=[]; rocs=[]; dropped=0
    for _ in range(replicates):
        draw=rng.integers(0,len(unique),size=len(unique)); counts=np.bincount(draw,minlength=len(unique)); weights=counts[inverse]
        positive_weight=float(weights[y].sum()); negative_weight=float(weights[~y].sum())
        if positive_weight<=0 or negative_weight<=0:
            dropped+=1; continue
        prevalence=float(np.average(y,weights=weights)); ap=float(average_precision_score(y,scores,sample_weight=weights))
        aps.append(ap); deltas.append(ap-prevalence); rocs.append(float(roc_auc_score(y,scores,sample_weight=weights)))
    valid=len(aps); reliable=len(unique)>=10 and positive_blocks>=5 and valid>=math.ceil(0.8*replicates)
    def ci(values):
        return [float(x) for x in np.percentile(values,[2.5,97.5])] if values else [None,None]
    return {"method":"spatial block bootstrap with replacement; block draw counts become unit sample weights",
            "block_definition":"projected 3 km deterministic grid; suitability audited and frozen per landscape before opening",
            "bootstrap_unit":"spatial block","seed":seed,"requested_replicates":replicates,"valid_replicates":valid,
            "single_class_replicates_dropped":dropped,"interval_method":"95% percentile",
            "effective_block_count":len(unique),"positive_containing_block_count":positive_blocks,
            "pr_auc_ci":ci(aps),"delta_pr_vs_weighted_prevalence_ci":ci(deltas),"roc_auc_ci":ci(rocs),
            "uncertainty_status":"RELIABLY_ESTIMABLE" if reliable else "NOT_RELIABLY_ESTIMABLE",
            "reliability_rule":">=10 total blocks, >=5 positive-containing blocks, >=80% valid replicates",
            "naive_cell_bootstrap_used":False}


def evaluate_landscape(ids: np.ndarray,y: np.ndarray,scores: np.ndarray,blocks: np.ndarray,*,seed:int=SEED,replicates:int=BOOTSTRAP_REPLICATES) -> dict:
    primary=primary_metrics(y,scores)
    curve={"recall":[],"precision":[]}
    if primary["status"]=="ESTIMATED":
        precision,recall,_=precision_recall_curve(y,scores); curve={"recall":recall.tolist(),"precision":precision.tolist()}
    return {"primary":primary,"percentile_rank":percentile_rank(scores).tolist(),"precision_recall_curve":curve,
            "prioritization":prioritization_metrics(ids,y,scores),"score_distribution":{
                "all_quantiles":np.quantile(scores,[0,.05,.25,.5,.75,.95,1]).tolist(),
                "positive_quantiles":np.quantile(scores[np.asarray(y,dtype=bool)],[0,.25,.5,.75,1]).tolist() if np.asarray(y).sum() else [],
                "negative_quantiles":np.quantile(scores[~np.asarray(y,dtype=bool)],[0,.25,.5,.75,1]).tolist() if (~np.asarray(y,dtype=bool)).sum() else []},
            "calibration_diagnostics":descriptive_calibration(y,scores),
            "spatial_uncertainty":spatial_block_bootstrap(y,scores,blocks,seed=seed,replicates=replicates)}


def classify_landscape(result: dict, quality_gates_pass: bool=True) -> str:
    p=result["primary"]; u=result["spatial_uncertainty"]
    if not quality_gates_pass or p["status"]!="ESTIMATED" or u["uncertainty_status"]!="RELIABLY_ESTIMABLE": return "INCONCLUSIVE"
    lo,hi=u["delta_pr_vs_weighted_prevalence_ci"]
    if p["delta_pr"]>0 and lo is not None and lo>0: return "ABOVE_CHANCE_RANKING"
    if p["delta_pr"]<=0 and hi is not None and hi<=0: return "TRANSFER_NOT_SUPPORTED"
    return "INCONCLUSIVE"


def cross_lockbox_summary(results: list[dict]) -> dict:
    estimable=[r for r in results if r["primary"]["status"]=="ESTIMATED"]
    if not estimable: return {"status":"INCONCLUSIVE","reason":"no estimable lockbox landscapes"}
    aps=[r["primary"]["pr_auc_model"] for r in estimable]; lifts=[r["primary"]["pr_lift"] for r in estimable]
    labels=[r["interpretation"] if "interpretation" in r else classify_landscape(r) for r in results]
    n=len(results); above=labels.count("ABOVE_CHANCE_RANKING"); failed=labels.count("TRANSFER_NOT_SUPPORTED"); inconclusive=labels.count("INCONCLUSIVE")
    if n<2: status="INCONCLUSIVE"
    elif failed>=math.ceil(n/2) or float(np.median(lifts))<=1: status="TRANSFER_NOT_SUPPORTED"
    elif above>=math.ceil(2*n/3) and failed==0: status="TRANSFER_SUPPORTED_WITH_LIMITATIONS"
    else: status="INCONCLUSIVE"
    return {"status":status,"n_lockboxes":n,"equal_landscape_median_pr_auc":float(np.median(aps)),
            "median_pr_lift":float(np.median(lifts)),"worst_landscape_pr_lift":float(np.min(lifts)),
            "pr_auc_range":[float(np.min(aps)),float(np.max(aps))],"above_chance_count":above,
            "inconclusive_count":inconclusive,"failed_count":failed,"landscape_weighting":"EQUAL",
            "population_level_volcano_ci":None,"diversity_coverage":"QUALITATIVE_SOURCE_SUPPORTED_DESCRIPTION_REQUIRED",
            "universal_generalization_claim_permitted":False}


LEDGER_FIELDS=("dataset_id","authorization","operator","timestamp","dataset_role","action","artifact_hash","purpose","access_decision")


@dataclass
class AccessLedger:
    entries:list[dict]=field(default_factory=list)

    def append(self,entry:dict)->dict:
        missing=[k for k in LEDGER_FIELDS if not str(entry.get(k,""))]
        if missing: raise ValueError(f"ledger entry missing: {missing}")
        payload={k:entry[k] for k in LEDGER_FIELDS}; payload["previous_entry_hash"]=self.entries[-1]["entry_hash"] if self.entries else "GENESIS"
        payload["entry_hash"]=canonical_hash(payload); self.entries.append(payload); return payload

    def verify(self)->bool:
        previous="GENESIS"
        for entry in self.entries:
            if entry["previous_entry_hash"]!=previous: return False
            raw={k:v for k,v in entry.items() if k!="entry_hash"}
            if canonical_hash(raw)!=entry["entry_hash"]: return False
            previous=entry["entry_hash"]
        return True


def authorize_opening(preaudit:dict)->dict:
    required={"LOCKBOX_ELIGIBILITY":"PASS","SOURCE_HASHES":"PASS","ACCESS_RESTRICTIONS":"PASS",
              "FINAL_ESTIMATOR_FREEZE":"PASS","EVALUATION_PREREGISTRATION":"PASS","CODE_ENVIRONMENT_HASHES":"PASS",
              "DATASET_GATES":"PASS","EXPLICIT_AUTHORIZATION":"APPROVED"}
    failed={k:{"expected":v,"actual":preaudit.get(k,"MISSING")} for k,v in required.items() if preaudit.get(k)!=v}
    return {"authorized":not failed,"failed_gates":failed,"decision":"ALLOW_ONE_TIME_OPENING" if not failed else "DENY_FAIL_CLOSED"}


@dataclass
class SyntheticEvaluationSession:
    dataset_id:str
    ledger:AccessLedger
    opened:bool=False
    evaluated:bool=False
    tuning_prohibited:bool=True

    def open(self,preaudit:dict,*,operator:str,timestamp:str,artifact_hash:str,synthetic:bool)->None:
        auth=authorize_opening(preaudit)
        if not synthetic or not auth["authorized"] or self.opened:
            self.ledger.append({"dataset_id":self.dataset_id,"authorization":canonical_hash(preaudit),"operator":operator,
                "timestamp":timestamp,"dataset_role":"SYNTHETIC_LOCKBOX","action":"OPEN_OUTCOMES","artifact_hash":artifact_hash,
                "purpose":"synthetic protocol dry run","access_decision":"DENY"})
            raise PermissionError("outcome opening denied")
        self.ledger.append({"dataset_id":self.dataset_id,"authorization":canonical_hash(preaudit),"operator":operator,
            "timestamp":timestamp,"dataset_role":"SYNTHETIC_LOCKBOX","action":"OPEN_OUTCOMES","artifact_hash":artifact_hash,
            "purpose":"synthetic protocol dry run","access_decision":"ALLOW"})
        self.opened=True

    def evaluate(self,ids,y,scores,blocks,*,seed:int=SEED,replicates:int=BOOTSTRAP_REPLICATES)->dict:
        if not self.opened or self.evaluated: raise PermissionError("evaluation denied: unopened or already evaluated")
        result=evaluate_landscape(ids,y,scores,blocks,seed=seed,replicates=replicates); self.evaluated=True
        self.ledger.append({"dataset_id":self.dataset_id,"authorization":"PREVIOUS_OPENING","operator":"SYNTHETIC_OPERATOR",
            "timestamp":"2026-10-08T00:01:00+10:00","dataset_role":"SYNTHETIC_LOCKBOX","action":"EVALUATE_ONCE",
            "artifact_hash":canonical_hash({"primary":result["primary"]}),"purpose":"synthetic metric verification","access_decision":"ALLOW"})
        return result

    def attempt_tuning(self)->None:
        self.ledger.append({"dataset_id":self.dataset_id,"authorization":"NONE","operator":"SYNTHETIC_OPERATOR",
            "timestamp":"2026-10-08T00:02:00+10:00","dataset_role":"SYNTHETIC_LOCKBOX","action":"POST_OPENING_TUNING",
            "artifact_hash":"NOT_CREATED","purpose":"negative access-control test","access_decision":"DENY"})
        raise PermissionError("post-opening tuning prohibited")
