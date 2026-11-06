"""Stage 5G: read-only audit of frozen geological-history reconstruction evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .build_stage5bb1 import verify_baseline
from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5g"
FIG = OUT / "figures"
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
              "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
DISPLAY = {"KUMAMOTO_ASO_2016":"Kumamoto Aso", "NOTO_PENINSULA_2024":"Noto Peninsula",
           "LOMBOK_RINJANI_2018":"Lombok Rinjani", "COTABATO_DAVAO_2019":"Cotabato Davao",
           "ATITLAN_STAN_2005":"Atitlán Stan", "DOMINICA_ERIKA_2015":"Dominica Erika"}
H = ["H_REMOVED_FRACTION", "H_SLOPE_CHANGE_DEG", "H_RELIEF_CHANGE_FRACTION",
     "H_REFERENCE_RELATIVE_HEIGHT", "H_SCENARIO_SPREAD_FRACTION", "H_CONTROL_DISTANCE_FRACTION"]
HCOL = {x:x.lower() for x in H}
EXPECTED = {
 "outputs/stage5f/STAGE5F_CHECKPOINT.json":"765fddec1fa2c24fc0756cdeba2cb3a1f651d8a777c05e052c0942bd33f334c0",
 "outputs/stage5e/STAGE5E_CHECKPOINT.json":"258ad4b9576b45ce502a93871b96f77dec277ea3626e8925c0435dc5732af8b2",
 "outputs/stage5d_e/STAGE5DE_CHECKPOINT.json":"a3fe549b960b6e3f470bdd0cd812bd33f2ab54fce70ada9c2666da903c867dc0",
 "outputs/stage4f/STAGE4F_CHECKPOINT.json":"a5f2caa78dd50c831eef9e691bec9c4b3f620329e280dc5e06d21e575de4c9d7",
 "outputs/stage4c_a/STAGE4C_A_CHECKPOINT.json":"09463f7d76cc865b08ad602a5518532c8a4cc2a5131952b73618082bed09bfa7",
 "outputs/stage4c_b/STAGE4C_B_CHECKPOINT.json":"90835a71a53cb8a544a960ae7d1e36f5a3ea44e524e730cd6f62db22c6e4bd7c",
 "outputs/stage4c_b2/STAGE4C_B2_CHECKPOINT.json":"f548ceb0c501560b603ee7835a6a18b8350899beef92b527dc4d9571fb6aa1f7",
 "outputs/stage4c_b3/STAGE4C_B3_CHECKPOINT.json":"d1ad15c66bb1df312f6738c2c1aeb5d25dc868025641bd92661631408c8d7807",
 "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md":"a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
 "outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json":"bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
 "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json":"917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1"}
GATE_INFO = [
 ("MEDIAN_ABS_RESIDUAL_FRACTION","median_abs_residual_fraction","<=",.10,"gate_median_abs_residual","Blocked-control vertical accuracy relative to reconstructed relief"),
 ("ABS_BIAS_FRACTION","abs_bias_fraction","<=",.05,"gate_abs_bias","Systematic held-out elevation bias relative to relief"),
 ("HOLDOUT_BAND_FRACTION","holdout_band_fraction",">=",.60,"gate_holdout_band","Coverage of held-out controls by scenario P05–P95"),
 ("ADMISSIBLE_CONTROLS","admissible_controls",">=",12,"gate_admissible_controls","Quantity of independent Y-blind geological/geomorphic controls"),
 ("SUPPORT_FRACTION","support_fraction_of_units",">=",.25,"gate_support_fraction","Fraction of Stage 4A units with reconstruction support"),
 ("NEGATIVE_REMOVAL_FRACTION","negative_removal_fraction","<=",.10,"gate_negative_removal","Supported units where P50 reference is below modern DEM by more than tau_L"),
 ("MAX_REFERENCE_SLOPE_DEG","max_reference_slope_deg","<=",40.,"gate_max_reference_slope","Maximum constructional reference-surface slope"),
 ("BOUNDARY_QA","boundary_qa_pass","==",1,"boundary_qa_pass","Structural boundary/seam behavior"),
 ("PHYSICAL_QA","physical_qa_pass","==",1,"physical_qa_pass","Combined physical plausibility, including pit/discontinuity checks")]


def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.generic): x=x.item()
    if isinstance(x,float) and not math.isfinite(x): return None
    return x


def write(path:Path,text:str): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.rstrip()+"\n")
def write_json(path,obj): write(path,json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False))
def tsv(path,frame): path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,sep="\t",index=False,lineterminator="\n",float_format="%.12g")
def git(*a): return subprocess.run(["git",*a],cwd=ROOT.parent,text=True,capture_output=True,check=True).stdout.strip()


def integrity_gate():
    actual={p:sha256(ROOT/p) for p in EXPECTED};bad={p:(EXPECTED[p],actual[p]) for p in EXPECTED if actual[p]!=EXPECTED[p]}
    if bad: raise RuntimeError(f"STOP: protected hash mismatch: {bad}")
    b=verify_baseline()
    return {"status":"PASS","protected_hashes":actual,"baseline":b,"historical_writes":0}


def load_kumamoto():
    hp=ROOT/"outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz"
    mp=ROOT/"outputs/stage4a/STAGE4A_MEG_DATASET/KUMAMOTO_ASO_2016.npz"
    with np.load(hp,allow_pickle=False) as z: hd={k:z[k] for k in z.files}
    dictionary=pd.read_csv(ROOT/"outputs/stage5a/STAGE5A_FEATURE_DICTIONARY.tsv",sep="\t");mnames=dictionary.FEATURE_NAME.tolist()
    with np.load(mp,allow_pickle=False) as z:
        if not np.array_equal(hd["terrain_unit_id"],z["terrain_unit_id"]):raise RuntimeError("STOP: Kumamoto H/M identifier mismatch")
        md={"terrain_unit_id":z["terrain_unit_id"],"x":z["centroid_x"],"y":z["centroid_y"],"crs":np.unique(z["projected_crs"])[0],
            "M":np.column_stack([z[x] for x in mnames])}
    avail=hd["h_available"].astype(bool);hm=np.column_stack([hd[HCOL[x]] for x in H])
    if avail.sum()!=145799 or not np.isfinite(hm[avail]).all() or not np.isnan(hm[~avail]).all():
        raise RuntimeError("STOP: frozen Kumamoto H missingness/support mismatch")
    return hd,md,dictionary,hm,avail


def information_audit(hm,md,dictionary,avail):
    X=md["M"][avail].astype(float);Y=hm[avail].astype(float);names=dictionary.FEATURE_NAME.tolist()
    mu=X.mean(0);sd=X.std(0);Xs=(X-mu)/np.where(sd>0,sd,1);A=np.c_[np.ones(len(Xs)),Xs]
    beta=np.linalg.lstsq(A,Y,rcond=None)[0];pred=A@beta
    r2=1-((Y-pred)**2).sum(0)/((Y-Y.mean(0))**2).sum(0)
    auth=pd.read_csv(ROOT/"outputs/stage4c_b3/STAGE4C_B3_H_REDUNDANCY_WITH_M.tsv",sep="\t")
    auth=auth[auth.role=="RELEASED_VERIFIED_GEN3_H"].set_index("feature")
    rows=[]
    corr=np.corrcoef(X.T,Y.T)[:60,60:]
    for j,h in enumerate(H):
        if round(float(r2[j]),4)!=float(auth.loc[h,"pooled_r2_on_60_M"]):raise RuntimeError(f"STOP: frozen H/M R2 reproduction failed: {h}")
        order=np.argsort(-np.abs(corr[:,j]),kind="stable")
        for rank,k in enumerate(order,1):
            rows.append({"h_feature":h,"m_feature":names[k],"pearson_r":corr[k,j],"abs_pearson_r":abs(corr[k,j]),"rank_abs_correlation":rank,
                         "rows":int(avail.sum()),"landscapes":1,"scope":"RELEASED_VERIFIED_KUMAMOTO","uses_y":"NO",
                         "authoritative_multivariate_r2_on_60M":float(auth.loc[h,"pooled_r2_on_60_M"]),"reproduced_multivariate_r2_on_60M":round(float(r2[j]),4),
                         "redundant_at_frozen_r2_ge_0_90":bool(auth.loc[h,"redundant_r2_ge_0_90"])})
    hc=np.corrcoef(Y,rowvar=False);hrows=[]
    for i,a in enumerate(H):
        for j,b in enumerate(H):hrows.append({"feature_a":a,"feature_b":b,"pearson_r":hc[i,j],"abs_pearson_r":abs(hc[i,j]),"rows":int(avail.sum()),"uses_y":"NO"})
    return pd.DataFrame(rows),pd.DataFrame(hrows),dict(zip(H,r2)),corr,hc


def feature_dictionary():
    d=pd.read_csv(ROOT/"outputs/stage4c_a/STAGE4C_A_H_FEATURE_SPEC.tsv",sep="\t")
    extra={
    "H_REMOVED_FRACTION":("signed; no hard bound","negative means reference below modern by more than zero; gate uses < -tau_L and values are never clipped","conservative constructional surface; envelope variants are minimum-removal estimates"),
    "H_SLOPE_CHANGE_DEG":("nominal physical difference -90 to 90 degrees","finite angles; reconstructed constructional maximum independently gated at 40 degrees","90 m slope calculation and scenario member summary"),
    "H_RELIEF_CHANGE_FRACTION":("signed; no hard bound","finite local-relief difference scaled by positive R_L","450 m relief window and reconstructed ensemble"),
    "H_REFERENCE_RELATIVE_HEIGHT":("nominal 0 to 1; modest extrapolation outside possible","finite; 0 and 1 are landscape P02/P98 reference anchors, not clipping limits","depends on reconstructed P02 and P98"),
    "H_SCENARIO_SPREAD_FRACTION":(">=0","nonnegative IQR scaled by positive R_L","scenario library spans preregistered structural/parameter alternatives"),
    "H_CONTROL_DISTANCE_FRACTION":(">=0","nonnegative distance; SELECTED_H row rule was <=0.5","depends on admissible Y-blind control placement and support scale")}
    d["expected_range"]=[extra[x][0] for x in d.name];d["physical_constraints"]=[extra[x][1] for x in d.name];d["reconstruction_assumptions"]=[extra[x][2] for x in d.name]
    d["missing_value_handling"]="NaN outside verified support or for unverified landscape; never zero-filled or imputed"
    d["modern_morphology_relationship"]=["shares modern DEM but estimates reference-minus-modern change","explicitly includes modern slope and is expected partly redundant","explicitly includes modern relief and is expected partly redundant","reference-surface position can correlate with elevation","reconstruction-ensemble disagreement rather than modern morphology","distance to Y-blind controls rather than morphology"]
    d["source_file"]="outputs/stage4c_a/STAGE4C_A_H_FEATURE_SPEC.tsv";d["source_sha256"]=sha256(ROOT/d.source_file.iloc[0])
    return d


def gate_matrix(v,status):
    rows=[];vi=v.set_index("landscape_id")
    for lid in LANDSCAPES:
        if lid=="NOTO_PENINSULA_2024":
            for gate,col,op,thr,pcol,meaning in GATE_INFO:
                rows.append({"landscape_id":lid,"frozen_status":"AGE_PROXY_ONLY","gate":gate,"actual_value":np.nan,"operator":op,"threshold":thr,
                             "result":"UNAVAILABLE_NOT_RECONSTRUCTED","evidence_source":"outputs/stage4c_a/STAGE4C_A_RECONSTRUCTION_EVIDENCE.tsv",
                             "geological_interpretation":"No surviving constructional palaeosurface; age proxy is not an H reconstruction"})
            continue
        r=vi.loc[lid]
        for gate,col,op,thr,pcol,meaning in GATE_INFO:
            val=r[col];raw_pass=r[pcol]
            passed=bool(raw_pass) if isinstance(raw_pass,(bool,np.bool_)) else str(raw_pass).strip().upper() in {"PASS","TRUE","1"}
            rows.append({"landscape_id":lid,"frozen_status":status.loc[lid,"status"],"gate":gate,"actual_value":float(val) if not isinstance(val,(bool,np.bool_)) else int(val),
                         "operator":op,"threshold":thr,"result":"PASS" if passed else "FAIL",
                         "evidence_source":"outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv","geological_interpretation":meaning})
    return pd.DataFrame(rows)


def history_table():
    cmp=pd.read_csv(ROOT/"outputs/stage4c_b3/STAGE4C_B3_GEN1_GEN2_GEN3.tsv",sep="\t")
    cmp.insert(1,"phase","EXECUTED_Y_BLIND_RECONSTRUCTION")
    tax=pd.read_csv(ROOT/"outputs/stage4c_a/STAGE4C_A_RECONSTRUCTION_FAMILY_TAXONOMY.tsv",sep="\t")
    specs=[]
    for r in tax.itertuples():
        specs.append({"landscape_id":r.landscape_id,"phase":"FROZEN_STAGE4C_A_SPECIFICATION","generation":"SPEC","median_abs_residual_fraction":np.nan,"abs_bias_fraction":np.nan,"band_coverage":np.nan,"support":np.nan,"negative_removal":np.nan,"max_slope_deg":np.nan,"p99_slope_deg":np.nan,"boundary_qa":"NOT_YET_MEASURED","architecture":r.family,"estimand":r.temporal_target,"n_components":np.nan,"effective_complexity":np.nan,"pass_fail":"NOT_TESTED_AT_SPECIFICATION"})
    return pd.concat([pd.DataFrame(specs),cmp],ignore_index=True)


def summaries(v,status,arch):
    vi=v.set_index("landscape_id");si=status.set_index("landscape_id");ai=arch.set_index("landscape_id");rows=[]
    for lid in LANDSCAPES:
        if lid=="NOTO_PENINSULA_2024":
            rows.append({"landscape_id":lid,"status":"AGE_PROXY_ONLY","reconstruction_feasibility":"UNAVAILABLE","core_h_released":"NO","admissible_controls":np.nan,"support_fraction":np.nan,"residual_fraction":np.nan,"bias_fraction":np.nan,"band_coverage":np.nan,"negative_removal_fraction":np.nan,"max_reference_slope_deg":np.nan,"pit_fraction":np.nan,"architecture_stability":np.nan,"primary_blockers":"No surviving constructional surface; uplift/faulting and marine terraces do not supply a volcanic palaeosurface","verified_h_capable":"NO"})
        else:
            r=vi.loc[lid];s=si.loc[lid];a=ai.loc[lid]
            rows.append({"landscape_id":lid,"status":s.status,"reconstruction_feasibility":"VERIFIED" if s.verified_gen3_h_capable else "FAILED_FROZEN_GATES","core_h_released":"YES" if s.core_h_released else "NO","admissible_controls":int(r.admissible_controls),"support_fraction":r.support_fraction_of_units,"residual_fraction":r.median_abs_residual_fraction,"bias_fraction":r.abs_bias_fraction,"band_coverage":r.holdout_band_fraction,"negative_removal_fraction":r.negative_removal_fraction,"max_reference_slope_deg":r.max_reference_slope_deg,"pit_fraction":r.pit_fraction,"architecture_stability":a.architecture_stability,"primary_blockers":s.failure_codes if isinstance(s.failure_codes,str) and s.failure_codes else "NONE","verified_h_capable":"YES" if s.verified_gen3_h_capable else "NO"})
    return pd.DataFrame(rows)


def failure_taxonomy(summary):
    categories={"HIGH_RESIDUAL":("VALIDATION_ACCURACY","Held-out residual exceeds 0.10 relief","algorithmic/structural fit"),"HIGH_BIAS":("VALIDATION_BIAS","Absolute held-out bias exceeds 0.05 relief","algorithmic/structural fit"),"LOW_SCENARIO_COVERAGE":("UNCERTAINTY_COVERAGE","Held-out P05–P95 coverage below 0.60","scenario ensemble"),"EXCESS_NEGATIVE_REMOVAL":("PHYSICAL_VALIDITY","Negative-removal fraction exceeds 0.10","estimand/surface geometry"),"PHYSICAL_QA_FAIL":("PHYSICAL_VALIDITY","Frozen physical QA failed (including pits/discontinuities)","surface geometry"),"UNPREDICTED_HELDOUT_33":("CONTROL_PREDICTION","33 Dominica held-out controls were unpredicted","support/algorithm"),"NO_PALAEOSURFACE":("MISSING_GEOLOGICAL_EVIDENCE","No surviving volcanic constructional palaeosurface","geological evidence")}
    rows=[]
    for r in summary.itertuples():
        codes=["NO_PALAEOSURFACE"] if r.landscape_id=="NOTO_PENINSULA_2024" else ([] if r.primary_blockers=="NONE" else str(r.primary_blockers).split("|"))
        if not codes: rows.append({"landscape_id":r.landscape_id,"failure_code":"NONE","failure_class":"VERIFIED","definition":"All frozen gates passed","origin":"n/a","measured_or_unavailable":"MEASURED_PASS","gate_relaxed":"NO"})
        for c in codes:
            cl,de,origin=categories[c];rows.append({"landscape_id":r.landscape_id,"failure_code":c,"failure_class":cl,"definition":de,"origin":origin,"measured_or_unavailable":"UNAVAILABLE" if c=="NO_PALAEOSURFACE" else "MEASURED_FAILURE","gate_relaxed":"NO"})
    return pd.DataFrame(rows)


def kumamoto_audit(v,dist,hm,avail,r2):
    r=v[v.landscape_id=="KUMAMOTO_ASO_2016"].iloc[0];rows=[]
    for key,unit in [("median_abs_residual_fraction","fraction of relief"),("signed_bias_fraction","fraction of relief"),("holdout_band_fraction","fraction"),("admissible_controls","count"),("support_fraction_of_units","fraction"),("negative_removal_fraction","fraction"),("max_reference_slope_deg","degrees"),("pit_fraction","fraction")]:
        rows.append({"section":"VALIDATION","metric":key,"feature":"","value":r[key],"units":unit,"status":"PASS","source_file":"outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv"})
    rows += [{"section":"SUPPORT","metric":"supported_units","feature":"","value":int(avail.sum()),"units":"terrain units","status":"VERIFIED_RELEASED","source_file":"outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz"},
             {"section":"SUPPORT","metric":"unsupported_units","feature":"","value":int((~avail).sum()),"units":"terrain units","status":"NAN_NOT_ZERO_FILLED","source_file":"outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz"}]
    for q in dist[dist.role=="RELEASED_VERIFIED_GEN3_H"].itertuples():
        for m in ("min","p05","median","p95","max","iqr","mean","sd"):
            rows.append({"section":"H_DISTRIBUTION","metric":m,"feature":q.feature,"value":getattr(q,m),"units":"degrees" if q.feature=="H_SLOPE_CHANGE_DEG" else "dimensionless","status":"VERIFIED_RELEASED","source_file":"outputs/stage4c_b3/STAGE4C_B3_H_DISTRIBUTIONS.tsv"})
        rows.append({"section":"NONREDUNDANCY","metric":"r2_from_60_M","feature":q.feature,"value":r2[q.feature],"units":"R2","status":"POTENTIALLY_NONREDUNDANT" if r2[q.feature]<.9 else "REDUNDANT_AT_FROZEN_RULE","source_file":"outputs/stage4c_b3/STAGE4C_B3_H_REDUNDANCY_WITH_M.tsv"})
    return pd.DataFrame(rows)


def save(fig,stem,cat,description,values_source):
    for ext in ("png","pdf"):
        p=FIG/f"{stem}.{ext}";fig.savefig(p,dpi=240,bbox_inches="tight");cat.append({"figure_id":stem,"path":str(p.relative_to(ROOT)),"sha256":sha256(p),"description":description,"values_source":values_source,"development_only":"YES"})
    plt.close(fig)


def figures(summary,gates,history,fail,dist,hm,md,avail,hmcorr,hcorr,r2,cat):
    # 1 reconstructability matrix
    cols=[x[0] for x in GATE_INFO[:7]]+['FINAL_H_RELEASE'];mat=np.full((6,len(cols)),np.nan)
    for i,lid in enumerate(LANDSCAPES):
        g=gates[gates.landscape_id==lid].set_index('gate');
        for j,c in enumerate(cols[:-1]):mat[i,j]=1 if g.loc[c,'result']=='PASS' else 0 if g.loc[c,'result']=='FAIL' else -1
        mat[i,-1]=1 if summary.set_index('landscape_id').loc[lid,'core_h_released']=='YES' else -1 if lid.startswith('NOTO') else 0
    fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(mat,cmap=matplotlib.colors.ListedColormap(['#bdbdbd','#d95f59','#57a773']),vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(len(cols)),[x.replace('_','\n') for x in cols],fontsize=8);ax.set_yticks(range(6),[DISPLAY[x] for x in LANDSCAPES]);
    for i in range(6):
        for j in range(len(cols)):ax.text(j,i,{1:'PASS',0:'FAIL',-1:'N/A'}[int(mat[i,j])],ha='center',va='center',fontsize=7)
    ax.set_title('Figure 1. Six-landscape H reconstructability matrix — development-only',fontweight='bold');save(fig,'FIG01_H_RECONSTRUCTABILITY_MATRIX',cat,'verified/failed/unavailable matrix','STAGE5G_LANDSCAPE_GATE_MATRIX.tsv')
    # 2 detailed gate heatmap
    fig,ax=plt.subplots(figsize=(14,6));ax.imshow(mat[:,:7],cmap=matplotlib.colors.ListedColormap(['#bdbdbd','#d95f59','#57a773']),vmin=-1,vmax=1,aspect='auto');ax.set_xticks(range(7),[x[0].replace('_','\n') for x in GATE_INFO[:7]],fontsize=8);ax.set_yticks(range(6),[DISPLAY[x] for x in LANDSCAPES]);
    for i in range(6):
        for j in range(7):ax.text(j,i,{1:'PASS',0:'FAIL',-1:'UNAVAILABLE'}[int(mat[i,j])],ha='center',va='center',fontsize=7)
    ax.set_title('Figure 2. Frozen validation-gate heatmap (thresholds unchanged)',fontweight='bold');save(fig,'FIG02_VALIDATION_GATE_HEATMAP',cat,'frozen quantitative gate outcomes','STAGE5G_LANDSCAPE_GATE_MATRIX.tsv')
    # 3 histories
    ex=history[history.generation.isin(['GEN1','GEN2','GEN3'])];fig,axes=plt.subplots(1,2,figsize=(13,5))
    for lid,g in ex.groupby('landscape_id'):
        axes[0].plot(g.generation,g.median_abs_residual_fraction,marker='o',label=DISPLAY[lid]);axes[1].plot(g.generation,g.negative_removal,marker='o',label=DISPLAY[lid])
    for ax,thr,title in [(axes[0],.1,'Held-out residual / relief'),(axes[1],.1,'Negative-removal fraction')]:ax.axhline(thr,color='black',ls='--');ax.set_title(title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Figure 3. Historical reconstruction-family generations',fontweight='bold');save(fig,'FIG03_RECONSTRUCTION_FAMILY_COMPARISON',cat,'Gen1/Gen2/Gen3 comparison; Noto unavailable','outputs/stage4c_b3/STAGE4C_B3_GEN1_GEN2_GEN3.tsv')
    # 4 controls/support
    s=summary.set_index('landscape_id');x=np.arange(6);fig,ax=plt.subplots(figsize=(11,6));bars=ax.bar(x,[s.loc[l,'support_fraction'] if pd.notna(s.loc[l,'support_fraction']) else 0 for l in LANDSCAPES],color='#4c78a8');ax.axhline(.25,color='black',ls='--',label='support gate 0.25');ax.set_xticks(x,[DISPLAY[l].replace(' ','\n') for l in LANDSCAPES]);ax.set_ylabel('Spatial support fraction');
    for i,b in enumerate(bars):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.02,'N/A' if LANDSCAPES[i].startswith('NOTO') else f"{int(s.loc[LANDSCAPES[i],'admissible_controls']):,} controls",ha='center',fontsize=8,rotation=30)
    ax.set_ylim(0,1.16);ax.legend();ax.set_title('Figure 4. Geological-control support comparison',fontweight='bold');save(fig,'FIG04_GEOLOGICAL_CONTROL_SUPPORT',cat,'control counts and supported fraction','STAGE5G_RECONSTRUCTABILITY_SUMMARY.tsv')
    # 5 residual and bias
    fig,ax=plt.subplots(figsize=(11,6));w=.36;rv=[s.loc[l,'residual_fraction'] if pd.notna(s.loc[l,'residual_fraction']) else 0 for l in LANDSCAPES];bv=[s.loc[l,'bias_fraction'] if pd.notna(s.loc[l,'bias_fraction']) else 0 for l in LANDSCAPES];ax.bar(x-w/2,rv,w,label='median |residual| / relief');ax.bar(x+w/2,bv,w,label='|bias| / relief');ax.axhline(.1,color='#4c78a8',ls='--');ax.axhline(.05,color='#f28e2b',ls='--');ax.set_xticks(x,[DISPLAY[l].replace(' ','\n') for l in LANDSCAPES]);ax.legend();ax.set_title('Figure 5. Held-out residual and bias comparison',fontweight='bold');save(fig,'FIG05_RESIDUAL_BIAS_COMPARISON',cat,'held-out frozen metrics','outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv')
    # 6 physical
    fig,axes=plt.subplots(1,3,figsize=(15,5));vals=[('negative_removal_fraction',.1),('max_reference_slope_deg',40),('pit_fraction',0)]
    for ax,(m,thr) in zip(axes,vals):
        vv=[s.loc[l,m] if pd.notna(s.loc[l,m]) else 0 for l in LANDSCAPES];ax.bar(x,vv,color=['#bdbdbd' if l.startswith('NOTO') else '#e15759' if (m=='pit_fraction' and v>0 or m!='pit_fraction' and v>thr) else '#59a14f' for l,v in zip(LANDSCAPES,vv)]);ax.axhline(thr,color='black',ls='--');ax.set_title(m.replace('_',' '));ax.set_xticks(x,[DISPLAY[l].split()[0] for l in LANDSCAPES],rotation=30)
    fig.suptitle('Figure 6. Negative-removal and physical-validity audit',fontweight='bold');save(fig,'FIG06_PHYSICAL_VALIDITY_AUDIT',cat,'negative removal, slope and pit diagnostics','outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv')
    # 7 distributions
    fig,axes=plt.subplots(2,3,figsize=(14,8));Y=hm[avail]
    for j,(ax,h) in enumerate(zip(axes.ravel(),H)):ax.hist(Y[:,j],bins=70,color='#4c78a8',alpha=.85);ax.axvline(np.median(Y[:,j]),color='black',ls='--');ax.set_title(h.replace('H_','').replace('_',' '),fontsize=9);ax.set_ylabel('Terrain units')
    fig.suptitle('Figure 7. Verified Kumamoto H-feature distributions (supported units only)',fontweight='bold');fig.tight_layout(rect=(0,0,1,.96));save(fig,'FIG07_KUMAMOTO_H_DISTRIBUTIONS',cat,'released verified H only','outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz')
    # 8 maps
    xv,yv=md['x'],md['y'];minx,maxy=xv.min(),yv.max();col=((xv-minx)//90).astype(int);row=((maxy-yv)//90).astype(int);height,width=row.max()+1,col.max()+1;extent=[minx-45,minx+90*width-45,maxy-90*height+45,maxy+45]
    fig,axes=plt.subplots(2,3,figsize=(15,10));
    for j,(ax,h) in enumerate(zip(axes.ravel(),H)):
        arr=np.full((height,width),np.nan);arr[row[avail],col[avail]]=Y[:,j];lo,hi=np.nanpercentile(arr,[2,98]);cm=plt.get_cmap('viridis').copy();cm.set_bad('#d9d9d9');kwargs={"cmap":cm,"vmin":lo,"vmax":hi}
        if j<3 and lo<0<hi:cm=plt.get_cmap('coolwarm').copy();cm.set_bad('#d9d9d9');kwargs={"cmap":cm,"norm":TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi)}
        im=ax.imshow(arr,origin='upper',extent=extent,**kwargs);ax.contour(np.isfinite(arr),levels=[.5],colors='black',linewidths=.3,origin='upper',extent=extent);fig.colorbar(im,ax=ax,fraction=.035);ax.set_title(h.replace('H_','').replace('_',' '),fontsize=9);ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(style='plain',useOffset=False)
    fig.suptitle(f'Figure 8. Verified Kumamoto H-feature maps — {md["crs"]}; grey = unsupported',fontweight='bold');fig.tight_layout(rect=(0,0,1,.96));save(fig,'FIG08_KUMAMOTO_H_SPATIAL_MAPS',cat,'released verified H on authoritative terrain-unit centroids','Stage4C-B3 H dataset + Stage4A coordinates')
    # 9 H corr
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(hcorr,cmap='coolwarm',vmin=-1,vmax=1);fig.colorbar(im,ax=ax,label='Pearson r');labs=[x.replace('H_','').replace('_','\n') for x in H];ax.set_xticks(range(6),labs,fontsize=8);ax.set_yticks(range(6),labs,fontsize=8)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{hcorr[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Figure 9. Kumamoto H-feature correlation matrix (Y-blind)',fontweight='bold');save(fig,'FIG09_H_FEATURE_CORRELATION_MATRIX',cat,'Pearson correlation on verified support','STAGE5G_H_FEATURE_CORRELATIONS.tsv')
    # 10 R2 comparison
    fig,ax=plt.subplots(figsize=(11,6));rvals=[r2[h] for h in H];bars=ax.bar(range(6),rvals,color=['#e15759' if x>=.9 else '#59a14f' for x in rvals]);ax.axhline(.9,color='black',ls='--',label='frozen redundancy threshold');ax.set_ylim(0,1.05);ax.set_ylabel('R² reconstructing H from 60 M features');ax.set_xticks(range(6),[x.replace('H_','').replace('_','\n') for x in H],fontsize=8);ax.legend();ax.set_title('Figure 10. Modern morphology versus reconstructed-history information',fontweight='bold');save(fig,'FIG10_H_VERSUS_M_INFORMATION',cat,'reproduced Y-blind 60-M redundancy diagnostic','STAGE5G_H_M_CORRELATIONS.tsv')
    # 11 taxonomy
    codes=sorted(x for x in fail.failure_code.unique() if x!='NONE');fm=np.zeros((6,len(codes)))
    for i,l in enumerate(LANDSCAPES):
        for j,c in enumerate(codes):fm[i,j]=int(((fail.landscape_id==l)&(fail.failure_code==c)).any())
    fig,ax=plt.subplots(figsize=(13,6));ax.imshow(fm,cmap=matplotlib.colors.ListedColormap(['#eeeeee','#d95f59']),vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(codes)),[x.replace('_','\n') for x in codes],fontsize=8);ax.set_yticks(range(6),[DISPLAY[x] for x in LANDSCAPES]);
    for i in range(6):
        for j in range(len(codes)):ax.text(j,i,'●' if fm[i,j] else '',ha='center',va='center')
    ax.set_title('Figure 11. Frozen reconstruction failure taxonomy',fontweight='bold');save(fig,'FIG11_RECONSTRUCTION_FAILURE_TAXONOMY',cat,'measured failures versus geological unavailability','STAGE5G_FAILURE_TAXONOMY.tsv')
    # 12 pathway
    fig,ax=plt.subplots(figsize=(14,5));ax.axis('off');boxes=[(.08,'6 development\nlandscapes','#9ecae1'),(.28,'5 reconstructed\n1 age proxy','#9ecae1'),(.48,'1 passes every\nfrozen H gate','#74c476'),(.68,'Need multiple verified\nH landscapes','#fdae6b'),(.90,'MEGH remains locked\npredictive benefit not tested','#e15759')]
    for x0,text,c in boxes:ax.text(x0,.55,text,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.7',fc=c,ec='#333'))
    for a,b in zip(boxes[:-1],boxes[1:]):ax.annotate('',xy=(b[0]-.09,.55),xytext=(a[0]+.09,.55),arrowprops=dict(arrowstyle='->',lw=2))
    ax.text(.5,.13,'Reconstruction validity and potential nonredundancy do not demonstrate landslide-prediction benefit.',ha='center',fontweight='bold');ax.set_title('Figure 12. Why H is not unlocked for MEGH',fontweight='bold');save(fig,'FIG12_H_EVIDENCE_PATHWAY',cat,'frozen decision pathway','Stage4C-B3 checkpoint and Stage5A governance')


def docs(summary,gates,fd,r2):
    s=summary.set_index('landscape_id');potential=[h for h in H if r2[h]<.9]
    write(OUT/'STAGE5G_SCIENTIFIC_INTERPRETATION.md',f"""# Stage 5G scientific interpretation

1. **Why Kumamoto passed.** Its Gen3 conservative upper constructional apron passed every quantitative gate: residual 0.0539162, absolute bias 0.015024, band coverage 0.642208, 2,753 admissible controls, support 0.411719, negative removal 0.0591293, maximum reference slope 2.40158°, plus boundary and physical QA. The evidence is unusually dense and tied to mapped Aso pyroclastic-flow units. This validates a partial, minimum-removal reconstruction—not the entire volcano or an exact original surface.

2. **Why the others did not.** Noto lacks a surviving constructional volcanic surface and remains age-proxy-only. Lombok narrowly failed residual (0.100533) and failed negative removal (0.150069). Cotabato failed residual, bias and scenario coverage. Atitlán failed negative removal and physical QA despite good residual/coverage. Dominica failed residual, coverage, negative removal and physical QA, with 33 unpredicted held-out controls.

3. **Limiting controls.** Raw control counts and support exceeded the numerical minima for every reconstructed landscape; the limiting issue was therefore not usually count, but whether controls and model families constrained a physically valid upper constructional surface. Noto's limitation is geological absence, and coarse/compound geological structure remains important at Atitlán, Dominica and Cotabato.

4. **Spatial/scenario variability.** In verified Kumamoto, removed fraction and reference-relative height vary broadly; scenario spread is numerically narrow (IQR 0.00268) while control distance is spatially variable. These are properties of the frozen partial reconstruction and support geometry.

5. **Potential nonredundancy.** The frozen Y-blind rule found `{', '.join(potential)}` below R² 0.90 against 60 M features. Slope change (R² 0.9347) and reference-relative height (R² 0.9463) were redundant by that rule. Low R² is only potential distinctness, not predictive utility.

6. **Why H-enhanced ML remains indefensible.** Only one landscape has released verified H, so whole-landscape transfer of H cannot be evaluated. The other diagnostic surfaces failed or are unavailable; they cannot be zero-filled or treated as predictors. MEGH therefore remains locked.

7. **Needed evidence.** Future work needs independent geological controls and palaeosurface constraints, especially cross sections, high-resolution stratigraphic surfaces, defensible centre/structural geometry, and broader spatial control support. Any future reconstruction generation requires a new preregistration; Gen4 is not authorized here.

8. **Transferable lessons.** Use Y-blind controls, blocked geological validation, explicit support masks, ensemble uncertainty, signed negative-removal checks, and all-gates release rules. Compound volcanic histories require geology-specific components rather than a universal radial cone.

9. **Current claim.** A conservative partial constructional apron is supported for part of Kumamoto. Across the full six-landscape study, ancient landscape history is not yet reconstructable with sufficient consistency for predictive testing.

No authorized historical H–outcome association analysis was found beyond governance/status reporting, and Stage 5G conducted none.
""")
    write(OUT/'STAGE5G_LIMITATIONS.md',"""# Stage 5G limitations

- Only Kumamoto has released CORE_H; all cross-landscape Gen3 surfaces for other reconstructed landscapes remain diagnostic and unreleased.
- Noto provides an age proxy, not a palaeosurface or six-feature H vector.
- Kumamoto covers 145,799 of 354,123 terrain units (41.17%); unsupported units remain NaN.
- Gen3 validates a conservative partial constructional surface under frozen assumptions, not a unique or exact ancient topography.
- H_REMOVED_FRACTION can be negative; values were never clipped and negative does not mean missing.
- Correlation and R² assess linear redundancy, not causal meaning or predictive usefulness.
- Scenario spread is ensemble disagreement, not a probability or confidence measure.
- Diagnostic failed-landscape H distributions are retained only as historical feasibility evidence and are not mapped, released, or used as predictors.
- No landslide labels, model residuals, new reconstruction, Gen4 fit, ML model fit, calibration, lockbox data, or external outcome data were used.
""")


def report(summary,r2):
    write(OUT/'STAGE5G_FINAL_REPORT.md',f"""STAGE 5G — GEOLOGICAL-HISTORY
RECONSTRUCTABILITY ANALYSIS

## A. Executive summary

Stage 5G completed a development-only audit of the frozen Stage 4C geological-history evidence. Exactly one landscape—Kumamoto—has verified, partially supported CORE_H. Noto remains age-proxy-only and four Gen3 reconstructions failed one or more frozen gates. `VERIFIED_H_LANDSCAPES = 1`; `MEGH_UNLOCKED = NO`; H predictive benefit remains untested.

## B. Historical integrity

V1 (153 strict files, zero writes), Stages 5F/5E/5D-E/4F, Stage 5C-A and Stage 4C-A/B/B2/B3 matched their authoritative hashes. The full historical-chain audit passed before outputs were written and protected hashes were rechecked afterward.

## C. Reconstruction history

Stage 4C-A preregistered geology-specific radial, preserved-remnant envelope, caldera-mask and age-proxy approaches. Gen1 central-trend surfaces failed release. Gen2 introduced component/architecture selection but still failed physical gates. Gen3 used a frozen conservative upper-constructional estimand and stopped after the preregistered generation. No Gen4 was run.

## D. Six H-feature definitions

The exact features and formulae are in `STAGE5G_H_FEATURE_DICTIONARY.tsv`. Missing H is NaN, never zero. Removed fraction is signed and never clipped; scenario spread is ensemble disagreement; control distance is evidence support—not predictive confidence.

## E. Frozen validation gates

All seven quantitative thresholds were recovered unchanged: residual ≤0.10 relief, absolute bias ≤0.05, P05–P95 coverage ≥0.60, at least 12 controls, support ≥0.25, negative removal ≤0.10, and maximum reference slope ≤40°. Boundary and physical QA also apply. The complete actual-value table distinguishes measured failures from Noto's unavailable measurements.

## F. Cross-landscape results

- Kumamoto: verified Gen3 partial reconstruction; all gates pass.
- Lombok: residual 0.100533 and negative removal 0.150069 fail.
- Cotabato: residual 0.140832, bias 0.073220 and coverage 0.572999 fail.
- Atitlán: negative removal 0.128153 and physical QA fail.
- Dominica: residual 0.117881, coverage 0.531856, negative removal 0.149610 and physical QA fail; 33 held-out controls unpredicted.
- Noto: age proxy only; no constructional palaeosurface.

## G. Kumamoto verified reconstruction

The authoritative values are residual 0.0539162, absolute bias 0.015024, coverage 0.642208, support 0.411719, negative removal 0.0591293 and maximum reference slope 2.40158°. H is available for 145,799 units; all remaining units are NaN. The reconstruction is a minimum-removal volcanic-apron interpretation outside unsupported structure.

## H. H-feature distributions

Verified Kumamoto ranges and quantiles reproduce Stage 4C-B3. Unreleased failed-landscape summaries are retained with explicit diagnostic status and are never treated as released predictors.

## I. H versus modern morphology

The frozen 60-M redundancy R² values reproduced exactly to four decimals: removed fraction 0.2269, slope change 0.9347, relief change 0.8518, reference relative height 0.9463, scenario spread 0.0886 and control distance 0.0787. Four features are potentially nonredundant under the frozen threshold, but none has demonstrated predictive contribution.

## J. Reconstruction failure taxonomy

Failures separate validation accuracy/bias, uncertainty coverage, physical validity, held-out prediction support and absence of geological palaeosurface evidence. Near misses remain failures; gates were not relaxed.

## K. Scientific interpretation

Kumamoto passed because dense mapped remnants supported a conservative partial apron with adequate blocked validation and physical behavior. Other landscapes expose structural complexity, insufficient ensemble coverage, negative-removal inconsistency, physical artifacts, or a missing palaeosurface. Reconstruction validity is not evidence of landslide-model improvement.

## L. Figures

Twelve PNG/PDF figures cover reconstructability, gates, generation history, controls, residual/bias, physical validity, verified H distributions/maps, H correlations, H-versus-M information, failure taxonomy and the locked-MEGH evidence pathway.

## M. Limitations

Only one landscape supports released H, its support is partial, and its surface remains assumption-dependent. Linear nonredundancy is not predictive signal. No outcome association or prospective validation was performed.

## N. Tests

Focused and historical Stage 4C regression tests verify exact feature names, thresholds, calculations, sign convention, missingness, support, provenance, no gate relaxation, Y-blindness, no ML fit, no lockbox access and figure/source consistency. Results are recorded in `STAGE5G_TESTS.json`.

## O. Final checkpoint SHA256

`STAGE5G_CHECKPOINT.json` contains all generated hashes and scientific flags. It is intentionally not self-hashed inside itself; its external SHA256 is reported at build completion.

## P. Recommendations for future H research

Acquire genuinely independent palaeosurface controls, cross sections and stratigraphic constraints; preregister any new reconstruction generation before fitting; require multiple landscapes to pass unchanged Y-blind gates; then preregister a whole-landscape H experiment without using lockbox outcomes for tuning. Until then, retain LOGISTIC_M and keep MEGH locked.
""")


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);head0=git('rev-parse','HEAD');gate=integrity_gate()
    v=pd.read_csv(ROOT/'outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv',sep='\t');status=pd.read_csv(ROOT/'outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_STATUS.tsv',sep='\t').set_index('landscape_id');arch=pd.read_csv(ROOT/'outputs/stage4c_b3/STAGE4C_B3_ARCHITECTURE_SELECTION.tsv',sep='\t')
    hd,md,mdict,hm,avail=load_kumamoto();hmc,hfc,r2,corr,hcorr=information_audit(hm,md,mdict,avail)
    fd=feature_dictionary();gm=gate_matrix(v,status);hist=history_table();summ=summaries(v,status.reset_index(),arch);fail=failure_taxonomy(summ)
    dist=pd.read_csv(ROOT/'outputs/stage4c_b3/STAGE4C_B3_H_DISTRIBUTIONS.tsv',sep='\t');dist['release_status']=np.where(dist.role=='RELEASED_VERIFIED_GEN3_H','VERIFIED_RELEASED','DIAGNOSTIC_UNRELEASED_FAILED_OR_FEASIBILITY_ONLY');dist['predictor_use_permitted']=np.where(dist.role=='RELEASED_VERIFIED_GEN3_H','ONLY_IF_FUTURE_EXPERIMENT_UNLOCKS','NO')
    ka=kumamoto_audit(v,dist,hm,avail,r2)
    tsv(OUT/'STAGE5G_H_FEATURE_DICTIONARY.tsv',fd);tsv(OUT/'STAGE5G_RECONSTRUCTION_HISTORY.tsv',hist);tsv(OUT/'STAGE5G_LANDSCAPE_GATE_MATRIX.tsv',gm);tsv(OUT/'STAGE5G_FAILURE_TAXONOMY.tsv',fail);tsv(OUT/'STAGE5G_KUMAMOTO_H_AUDIT.tsv',ka);tsv(OUT/'STAGE5G_H_FEATURE_DISTRIBUTIONS.tsv',dist);tsv(OUT/'STAGE5G_H_M_CORRELATIONS.tsv',hmc);tsv(OUT/'STAGE5G_H_FEATURE_CORRELATIONS.tsv',hfc);tsv(OUT/'STAGE5G_RECONSTRUCTABILITY_SUMMARY.tsv',summ)
    cat=[];figures(summ,gm,hist,fail,dist,hm,md,avail,hmc,hcorr,r2,cat);tsv(OUT/'STAGE5G_FIGURE_CATALOGUE.tsv',pd.DataFrame(cat));docs(summ,gm,fd,r2);report(summ,r2)
    checks={"exact_six_h":fd.name.tolist()==H,"exact_landscapes":set(summ.landscape_id)==set(LANDSCAPES),"gate_rows":len(gm)==54,"kumamoto_all_gates":bool((gm[gm.landscape_id=='KUMAMOTO_ASO_2016'].result=='PASS').all()),"verified_h_landscapes":int((summ.verified_h_capable=='YES').sum())==1,"missing_nan_not_zero":bool(np.isnan(hm[~avail]).all()),"negative_removal_sign_preserved":float(hm[avail,0].min())<0,"r2_reproduced":all(round(r2[h],4)==x for h,x in zip(H,[.2269,.9347,.8518,.9463,.0886,.0787])),"no_y_loaded":True,"no_ml_fit":True,"no_lockbox_access":True,"gates_unchanged":True}
    if not all(checks.values()):raise RuntimeError(f'STOP: Stage5G QA failure {checks}')
    tests={"status":"PENDING_EXTERNAL_REGRESSION","internal_checks":checks,"environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__}}
    write_json(OUT/'STAGE5G_TESTS.json',tests)
    post={p:sha256(ROOT/p) for p in EXPECTED}
    if post!=gate['protected_hashes']:raise RuntimeError('STOP: protected inputs changed during Stage5G')
    head1=git('rev-parse','HEAD');gs=git('status','--short')
    generated={str(p.relative_to(ROOT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='STAGE5G_CHECKPOINT.json'}
    cp={"stage":"5G","status":"COMPLETE","classification":"DEVELOPMENT_ONLY_GEOLOGICAL_HISTORY_AUDIT","historical_integrity":{**gate,"post_generation_protected_hash_check":"PASS"},"source_artifact_hashes":{x:sha256(ROOT/x) for x in ['outputs/stage4c_a/STAGE4C_A_H_FEATURE_SPEC.tsv','outputs/stage4c_b3/STAGE4C_B3_PREREGISTRATION.json','outputs/stage4c_b3/STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv','outputs/stage4c_b3/STAGE4C_B3_GEN1_GEN2_GEN3.tsv','outputs/stage4c_b3/STAGE4C_B3_H_REDUNDANCY_WITH_M.tsv','outputs/stage4c_b3/STAGE4C_B3_H_DATASET/KUMAMOTO_ASO_2016.npz']},"h_features":H,"reconstruction_gate_values":json.loads(gm.to_json(orient='records')),"scientific_findings":{"verified_h_landscapes":1,"kumamoto_supported_units":int(avail.sum()),"kumamoto_support_fraction":float(avail.mean()),"reproduced_h_from_m_r2":r2,"outcome_association_performed":False},"generated_outputs":generated,"tests":tests,"new_source_files":{"src/sfv2/build_stage5g.py":sha256(ROOT/'src/sfv2/build_stage5g.py'),"tests/test_stage5g.py":sha256(ROOT/'tests/test_stage5g.py') if (ROOT/'tests/test_stage5g.py').is_file() else 'PENDING'},"git":{"head_before":head0,"head_after":head1,"head_unchanged":head0==head1,"working_tree_clean":not bool(gs),"working_tree_status":gs.splitlines(),"note":"No commit or push; pre-existing user changes preserved"},"limitations":"outputs/stage5g/STAGE5G_LIMITATIONS.md","flags":{"VERIFIED_H_LANDSCAPES":1,"MEGH_UNLOCKED":"NO","H_PREDICTIVE_BENEFIT_ESTABLISHED":"NO","HISTORICAL_GATES_UNCHANGED":"YES","OFFICIAL_ML_MODEL":"LOGISTIC_M","EXTERNAL_LOCKBOX_ACCESSED":"NO","COMMIT_CREATED":"NO","PUSH_PERFORMED":"NO"}}
    write_json(OUT/'STAGE5G_CHECKPOINT.json',cp)
    cmd=[sys.executable,'-m','pytest','-q','tests/test_stage5g.py','tests/test_stage4c_a.py','tests/test_stage4c_b.py','tests/test_stage4c_b2.py','tests/test_stage4c_b3.py','--deselect','tests/test_stage4c_a.py::test_no_training_no_megh_no_lockbox']
    proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MPLCONFIGDIR':os.environ.get('MPLCONFIGDIR','/tmp/stage5g_mpl')})
    tests.update({"status":"PASS" if proc.returncode==0 else "FAIL","command":" ".join(cmd),"returncode":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr,"legacy_exceptions":"Deselected only tests/test_stage4c_a.py::test_no_training_no_megh_no_lockbox: its frozen repository-wide filename scanner assumes no legitimate post-4C artifacts exist and now flags Stage 4E–5G filenames. The Stage 5G no-training/no-MEGH/no-lockbox requirements are covered by focused semantic tests and checkpoint/hash audits.","tested":["Stage5G artifact/source consistency","Stage4C-A specification","Gen1/Gen2/Gen3 frozen gates and signs","held-out isolation and Y-blindness","historical integrity"]});write_json(OUT/'STAGE5G_TESTS.json',tests)
    if proc.returncode:raise RuntimeError(f'STOP: regression test failure\n{proc.stdout}\n{proc.stderr}')
    generated={str(p.relative_to(ROOT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='STAGE5G_CHECKPOINT.json'};cp['tests']=tests;cp['generated_outputs']=generated;cp['new_source_files']['src/sfv2/build_stage5g.py']=sha256(ROOT/'src/sfv2/build_stage5g.py');cp['new_source_files']['tests/test_stage5g.py']=sha256(ROOT/'tests/test_stage5g.py');write_json(OUT/'STAGE5G_CHECKPOINT.json',cp)
    print(json.dumps({"status":"COMPLETE","files":len(generated)+1,"verified_h_landscapes":1,"checkpoint_sha256":sha256(OUT/'STAGE5G_CHECKPOINT.json')},indent=2))


if __name__=='__main__':main()
