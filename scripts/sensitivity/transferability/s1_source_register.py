"""Stage C Phase 1 — source evidence register and cross-source consistency checks (read-only, no new statistics).

Writes 01_PROTOCOL/SOURCE_EVIDENCE_REGISTER.csv and 01_PROTOCOL/SOURCE_CONSISTENCY_CHECKS.csv.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2]
V2 = OUT.parent
L = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]

SOURCES = [  # path, role, authority, relevant variables
    ("outputs/stage5a/STAGE5A_MODEL_CONTRACT.json", "Frozen official model contract", "AUTHORITATIVE_FROZEN", "per-landscape AP, prevalence, ROC-AUC; median AP/lift; R"),
    ("outputs/stage5a/STAGE5A_DATASET_REGISTER.tsv", "Frozen dataset register", "AUTHORITATIVE_FROZEN", "units, positives, prevalence, inventory source, label rule, DEM provenance"),
    ("data/interim/_stage4b_results/PRIMARY.json", "Frozen Stage 4B PRIMARY results", "AUTHORITATIVE_FROZEN", "LOGISTIC_L2 M per-landscape AP/ROC/prevalence"),
    ("data/interim/_stage4b_results/ATITLAN_EXCLUDED.json", "Frozen training-composition sensitivity", "AUTHORITATIVE_FROZEN", "LOGISTIC_L2 M AP with Atitlan excluded from training"),
    ("data/interim/_stage4b_results/POLYGON_ONLY.json", "Frozen training/evaluation-composition sensitivity", "AUTHORITATIVE_FROZEN", "LOGISTIC_L2 M AP with Kumamoto excluded"),
    ("data/interim/_stage4b_results/STRICT_DEM_QUALITY.json", "Frozen population sensitivity (not composition)", "AUTHORITATIVE_FROZEN", "LOGISTIC_L2 M AP on STRICT DEM-quality units"),
    ("outputs/stage4b/STAGE4B_FEATURE_ABLATIONS.tsv", "Frozen per-landscape sensitivity table", "AUTHORITATIVE_FROZEN", "cross-check of sensitivity APs"),
    ("outputs/stage4e_b/STAGE4EB_SPATIAL_BOOTSTRAP.tsv", "Frozen block-bootstrap intervals", "AUTHORITATIVE_FROZEN", "LOGISTIC_L2 M PR lift 95% intervals"),
    ("outputs/stage4e_b/STAGE4EB_COTABATO_SENSITIVITY.tsv", "Frozen label-definition sensitivity (Cotabato)", "AUTHORITATIVE_FROZEN", "PR lift under cumulative EQ3/EQ4 labels"),
    ("outputs/stage4a/STAGE4A_FEATURE_MANIFEST.tsv", "Frozen feature manifest", "AUTHORITATIVE_FROZEN", "M column list"),
    ("outputs/stage4a/STAGE4A_DOMAIN_SHIFT_AUDIT.tsv", "Existing covariate-shift diagnostic", "DERIVED_FROZEN", "per-landscape feature shift (median vs rest / pooled IQR)"),
    ("outputs/stage5e/STAGE5E_FEATURE_DISTRIBUTION_DIAGNOSTICS.tsv", "Existing standardized-shift diagnostic", "DERIVED_FROZEN", "standardized mean shift from training background (sampled)"),
    ("outputs/stage5e/STAGE5E_LANDSCAPE_IMPORTANCE.tsv", "Frozen SHAP / coefficients", "DERIVED_FROZEN", "elevation std coefficient, SHAP"),
    ("outputs/stage5e/STAGE5E_PERMUTATION_IMPORTANCE.tsv", "Frozen permutation importance", "DERIVED_FROZEN", "elevation permutation AP drop"),
    ("outputs/stage3g/STAGE3G_DIVERSITY_METADATA.tsv", "Production-stage landscape context metadata", "AUTHORITATIVE_METADATA", "tectonic setting, morphology, erosion, climate, trigger, mapping method"),
    ("metadata/landscape_candidates.csv", "Stage-1 screening metadata", "HISTORICAL", "superseded context fields; used only for discrepancy check"),
    ("AUSSEF_2026_VALIDITY_AUDIT/INVENTORY_PROVENANCE_AUDIT.csv", "Stage A inventory audit", "DERIVED_AUDIT", "geometry, label meaning, min mapping unit, completeness"),
    ("AUSSEF_2026_VALIDITY_AUDIT/DEM_CHRONOLOGY_AUDIT.csv", "Stage A DEM audit", "DERIVED_AUDIT", "DEM product, chronology"),
    ("AUSSEF_2026_VALIDITY_AUDIT/LEAKAGE_RISK_REGISTER.csv", "Stage A risk register", "DERIVED_AUDIT", "R08-R15"),
    ("AUSSEF_2026_VALIDITY_AUDIT/evidence/E02_LABEL_GEOMETRY_DIAGNOSTICS.json", "Stage A label diagnostics", "DERIVED_AUDIT", "overlap-negatives, Kumamoto July/undated centres"),
    ("AUSSEF_2026_VALIDITY_AUDIT/evidence/E03_ELEVATION_DISTRIBUTIONS.tsv", "Stage A elevation distributions", "DERIVED_AUDIT", "within-landscape elevation AUC"),
    ("AUSSEF_2026_VALIDITY_AUDIT/evidence/E04_ELEVATION_FOLD_DIAGNOSTICS.tsv", "Stage A fold elevation diagnostics", "DERIVED_AUDIT", "std coefficient, SHAP, permutation"),
    ("AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/ELEVATION_ABLATION_FOLD_RESULTS.csv", "Stage A.1 fold results", "DERIVED_EXPERIMENT", "B0/A1/A2 AP, lift, ROC, coefficients"),
    ("AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/PAIRED_PERFORMANCE_DIFFERENCES.csv", "Stage A.1 paired differences", "DERIVED_EXPERIMENT", "delta lift A1/A2 with block-bootstrap CI"),
    ("AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/ELEVATION_ABLATION_SUMMARY.csv", "Stage A.1 summary", "DERIVED_EXPERIMENT", "R, median/min lift"),
    ("AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/BASELINE_REPRODUCTION_AUDIT.csv", "Stage A.1 baseline gate", "DERIVED_EXPERIMENT", "B0 reproduction evidence"),
] + [(f"outputs/stage4a/STAGE4A_MEG_DATASET/{l}.npz", "Frozen feature matrix (label-free use only for shift)", "AUTHORITATIVE_FROZEN", "M predictors; y_positive counts") for l in L]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    (OUT / "01_PROTOCOL").mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, role, auth, var in SOURCES:
        p = V2 / rel
        rows.append(dict(path=rel, role=role, version=("mtime " + __import__("time").strftime("%Y-%m-%dT%H:%M:%S", __import__("time").localtime(p.stat().st_mtime))) if p.exists() else "MISSING",
                         sha256=sha(p) if p.exists() else "MISSING", authority_level=auth, relevant_variables=var, exists=p.exists()))
    with (OUT / "01_PROTOCOL/SOURCE_EVIDENCE_REGISTER.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)

    # consistency checks across frozen sources
    chk = []
    contract = {r["held_out"]: r for r in json.loads((V2 / "outputs/stage5a/STAGE5A_MODEL_CONTRACT.json").read_text())["landscape_level_results"]}
    prim = {r["held_out"]: r for r in json.loads((V2 / "data/interim/_stage4b_results/PRIMARY.json").read_text())["results"] if r["model"] == "LOGISTIC_L2" and r["feature_set"] == "M"}
    reg = {r["LANDSCAPE_ID"]: r for r in csv.DictReader((V2 / "outputs/stage5a/STAGE5A_DATASET_REGISTER.tsv").open(), delimiter="\t")}
    a1 = {r["held_out"]: r for r in csv.DictReader((V2 / "AUSSEF_2026_ELEVATION_EXPERIMENT/02_RESULTS/ELEVATION_ABLATION_FOLD_RESULTS.csv").open()) if r["config"] == "B0"}
    boot = {r["held_out"]: r for r in csv.DictReader((V2 / "outputs/stage4e_b/STAGE4EB_SPATIAL_BOOTSTRAP.tsv").open(), delimiter="\t") if r["model"] == "LOGISTIC_L2" and r["metric"] == "pr_lift" and r["comparison"] == "POINT_INTERVAL"}
    for l in L:
        with np.load(V2 / f"outputs/stage4a/STAGE4A_MEG_DATASET/{l}.npz", allow_pickle=False) as z:
            n, pos = len(z["y_positive"]), int(z["y_positive"].sum())
        vals = dict(npz_n=n, register_n=int(reg[l]["N_UNITS"]), contract_n=int(contract[l]["n"]), primary_n=int(prim[l]["n"]),
                    npz_pos=pos, register_pos=int(reg[l]["POSITIVES"]), contract_pos=int(contract[l]["positives"]), primary_pos=int(prim[l]["positives"]))
        chk.append(dict(landscape=l, check="units_consistent", values=json.dumps({k: v for k, v in vals.items() if k.endswith("_n")}), passed=len({v for k, v in vals.items() if k.endswith("_n")}) == 1))
        chk.append(dict(landscape=l, check="positives_consistent", values=json.dumps({k: v for k, v in vals.items() if k.endswith("_pos")}), passed=len({v for k, v in vals.items() if k.endswith("_pos")}) == 1))
        aps = dict(contract=float(contract[l]["pr_auc"]), primary=float(prim[l]["pr_auc"]), stageA1_B0=float(a1[l]["ap"]))
        chk.append(dict(landscape=l, check="ap_consistent_1e-11", values=json.dumps(aps), passed=max(aps.values()) - min(aps.values()) < 1e-11))
        lifts = dict(contract=float(contract[l]["pr_lift"]), stageA1_B0=float(a1[l]["pr_lift"]), stage4eb_bootstrap_point=float(boot[l]["point_estimate"]))
        chk.append(dict(landscape=l, check="pr_lift_consistent_1e-9", values=json.dumps(lifts), passed=max(lifts.values()) - min(lifts.values()) < 1e-9))
        roc = dict(contract=float(contract[l]["roc_auc"]), primary=float(prim[l]["roc_auc"]), stageA1_B0=float(a1[l]["roc_auc"]))
        chk.append(dict(landscape=l, check="roc_consistent_1e-11", values=json.dumps(roc), passed=max(roc.values()) - min(roc.values()) < 1e-11))
    # context metadata discrepancy check (production vs Stage-1 screening)
    g = {r["landscape_id"]: r for r in csv.DictReader((V2 / "outputs/stage3g/STAGE3G_DIVERSITY_METADATA.tsv").open(), delimiter="\t")}
    s1 = {r["landscape_id"]: r for r in csv.DictReader((V2 / "metadata/landscape_candidates.csv").open())}
    for l in L:
        if l in s1:
            for a, b in (("tectonic_setting", "tectonic_setting"), ("climate", "climate_class")):
                chk.append(dict(landscape=l, check=f"context_{a}_stage3g_vs_stage1", values=json.dumps({"stage3g": g[l][a], "stage1": s1[l][b]}),
                                passed=g[l][a] == s1[l][b]))
        else:
            chk.append(dict(landscape=l, check="context_stage1_record_present", values="absent from metadata/landscape_candidates.csv", passed=None))
    with (OUT / "01_PROTOCOL/SOURCE_CONSISTENCY_CHECKS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chk[0]), lineterminator="\n"); w.writeheader(); w.writerows(chk)
    print(sum(1 for r in rows if not r["exists"]), "missing sources;", sum(1 for c in chk if c["passed"] is False), "failed checks")
    for c in chk:
        if c["passed"] is not True:
            print(c)


if __name__ == "__main__":
    main()
