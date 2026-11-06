"""Build Stage 4C-A: Stage 4B results workbook (read-only view) + frozen H specification and feasibility.  No model is trained or evaluated."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import stage4c_a_spec as H
from .build_stage3a import scan_ml_h_artifacts
from .raw_store import sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

AS_OF = "2026-10-07"
OUT = ROOT / "outputs/stage4c_a"
B4 = ROOT / "outputs/stage4b"
STAGE4B_CHECKPOINT_SHA = "2685d1b6d565de7945e138d9b014d9aa4b53657511552299c19ebc1f4df946b8"
STAGE4A_CHECKPOINT_SHA = "fd25accb50578bd4f4f28042e42504a89ef936c904d5ed262a0f65fc7fa2452c"
WARN = "DEVELOPMENT ONLY - NONFINAL. Stage 4B results; no final V2 model, no lockbox, no significance testing."
HDR = PatternFill("solid", fgColor="1F3864")
WARNFILL = PatternFill("solid", fgColor="FCE4D6")


def _tsv(name: str) -> pd.DataFrame:
    return pd.read_csv(B4 / name, sep="\t", keep_default_na=False, na_values=[""])


def _write_tsv(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(p, sep="\t", index=False, lineterminator="\n")


# ------------------------------------------------------------------------------------------------ Part I: workbook
def _sheet(wb, title: str, df: pd.DataFrame, formats: dict | None = None, note: str = WARN):
    ws = wb.create_sheet(title)
    ws.append([note])
    ws["A1"].font = Font(bold=True, color="9C0006")
    ws["A1"].fill = WARNFILL
    ws.append(list(df.columns))
    for c in ws[2]:
        c.font, c.fill, c.alignment = Font(bold=True, color="FFFFFF"), HDR, Alignment(wrap_text=True, vertical="top")
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v) for v in row])
    ws.freeze_panes = "B3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(df.columns))}{len(df) + 2}"
    for i, col in enumerate(df.columns, start=1):
        width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(max(width, 10), 60)
        fmt = (formats or {}).get(col)
        if fmt:
            for cell in ws.iter_rows(min_row=3, min_col=i, max_col=i):
                cell[0].number_format = fmt
    return ws


def workbook() -> dict:
    res = _tsv("STAGE4B_LANDSCAPE_RESULTS.tsv")
    summ = _tsv("STAGE4B_MODEL_RESULTS.tsv")
    abl = _tsv("STAGE4B_FEATURE_ABLATIONS.tsv")
    interp = _tsv("STAGE4B_FEATURE_INTERPRETATION.tsv")
    dsp = _tsv("STAGE4B_DOMAIN_SHIFT_PERFORMANCE.tsv")
    leader = json.loads((B4 / "STAGE4B_DEVELOPMENT_LEADER.json").read_text())
    cp4b = json.loads((B4 / "STAGE4B_CHECKPOINT.json").read_text())
    prim = res[res.analysis == "PRIMARY"]
    ps = summ[summ.analysis == "PRIMARY"]
    null_med = float(ps[ps.model == "NULL_PREVALENCE"].median_pr_auc.iloc[0])
    best = ps[ps.model != "NULL_PREVALENCE"].sort_values("median_pr_auc").iloc[-1]
    lead = prim[(prim.model == leader["model"]) & (prim.feature_set == leader["feature_set"])].set_index("held_out")
    hardest, easiest = lead.pr_auc_lift.idxmin(), lead.pr_auc_lift.idxmax()
    wb = Workbook()
    ws = wb.active
    ws.title = "DASHBOARD"
    rows = [("WARNING", WARN), ("Development landscapes", 6), ("Terrain units", int(cp4b["data_check"]["rows"])),
            ("Positives", int(cp4b["data_check"]["positives"])), ("Negatives", int(cp4b["data_check"]["negatives"])),
            ("Null median PR-AUC (training prevalence)", null_med), ("Best median PR-AUC", float(best.median_pr_auc)),
            ("Best median model / features", f"{best.model} / {best.feature_set}"), ("DEVELOPMENT_LEADER (pre-registered rule)", f"{leader['model']} / {leader['feature_set']} (NONFINAL)"),
            ("Leader median / worst PR-AUC", f"{leader['median_pr_auc']:.4f} / {leader['min_pr_auc']:.4f}"),
            ("Hardest held-out landscape (leader, PR-AUC lift)", hardest), ("Easiest held-out landscape (leader, PR-AUC lift)", easiest),
            ("Headline", "Transfer to an unseen volcanic landscape is weak (median PR-AUC ~1.5-2.6x prevalence). Morphology carries most of the signal; "
                         "climate transfers inconsistently; geology adds little; calibration does not transfer.")]
    for k, v in rows:
        ws.append([k, v])
    for c in ws["A"]:
        c.font = Font(bold=True)
    ws["B1"].font = Font(bold=True, color="9C0006")
    ws["B1"].fill = WARNFILL
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 48, 110
    for c in ("B7", "B8"):
        ws[c].number_format = "0.0000"
    # chart data: leader PR-AUC vs null per landscape (absolute scale from 0; no truncation)
    ws.append([])
    start = ws.max_row + 1
    ws.append(["Held-out landscape", f"{leader['model']} {leader['feature_set']} PR-AUC", "Null PR-AUC (= prevalence)", "PR-AUC lift"])
    nul = prim[prim.model == "NULL_PREVALENCE"].set_index("held_out")
    for k in H.LANDSCAPES:
        ws.append([k, float(lead.loc[k, "pr_auc"]), float(nul.loc[k, "pr_auc"]), float(lead.loc[k, "pr_auc_lift"])])
    ch = BarChart()
    ch.title, ch.y_axis.title = "Held-out PR-AUC: leader vs prevalence (DEVELOPMENT ONLY)", "PR-AUC (axis starts at 0)"
    ch.y_axis.scaling.min = 0
    ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=start, max_row=start + 6), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=start + 1, max_row=start + 6))
    ch.width, ch.height = 24, 9
    ws.add_chart(ch, "D2")

    dec = "0.0000"
    lolo = res[["analysis", "held_out", "model", "feature_set", "pr_auc", "roc_auc", "brier", "log_loss", "precision", "recall", "f1", "prevalence", "positives", "negatives", "pr_auc_lift"]]
    lolo = lolo.assign(role=np.where(lolo.analysis == "PRIMARY", "PRIMARY", "SENSITIVITY"))
    _sheet(wb, "LOLO_RESULTS", lolo, {c: dec for c in ("pr_auc", "roc_auc", "brier", "log_loss", "precision", "recall", "f1")} | {"prevalence": "0.00%", "pr_auc_lift": "0.00"})
    mc = summ[summ.analysis == "PRIMARY"][["model", "feature_set", "median_pr_auc", "mean_pr_auc", "min_pr_auc", "max_pr_auc", "iqr_pr_auc", "median_pr_auc_lift", "median_roc_auc"]]
    w = _sheet(wb, "MODEL_COMPARISON", mc, {c: dec for c in mc.columns[2:8]} | {"median_pr_auc_lift": "0.00", "median_roc_auc": "0.000"})
    w.conditional_formatting.add(f"C3:C{len(mc) + 2}", ColorScaleRule(start_type="min", start_color="F8696B", end_type="max", end_color="63BE7B"))
    fa = abl[abl.analysis == "PRIMARY"][["model", "held_out", "M", "ME", "MEG", "MEG_NO_BIO15", "delta_E", "delta_G", "delta_MEG"]]
    w = _sheet(wb, "FEATURE_ABLATION", fa, {c: dec for c in fa.columns[2:]})
    for col in ("G", "H", "I"):
        w.conditional_formatting.add(f"{col}3:{col}{len(fa) + 2}", ColorScaleRule(start_type="num", start_value=-0.05, start_color="F8696B", mid_type="num", mid_value=0,
                                                                                   mid_color="FFFFFF", end_type="num", end_value=0.05, end_color="63BE7B"))
    dist = dict(zip(dsp.held_out, dsp.domain_distance))
    lrows = []
    for k in H.LANDSCAPES:
        r = prim[(prim.held_out == k) & (prim.model != "NULL_PREVALENCE")].sort_values("pr_auc").iloc[-1]
        lrows.append(dict(landscape_id=k, units=int(r.n), positives=int(r.positives), negatives=int(r.negatives), prevalence=float(r.prevalence), best_model=r.model,
                          best_feature_set=r.feature_set, best_pr_auc=float(r.pr_auc), best_roc_auc=float(r.roc_auc), domain_shift_4a=float(dist[k]),
                          notes="Kumamoto = point-event Tier 2; Atitlan M from NASADEM (no STRICT); best cell is the per-landscape maximum over all PRIMARY models/features (descriptive only)"))
    _sheet(wb, "LANDSCAPES", pd.DataFrame(lrows), {"prevalence": "0.00%", "best_pr_auc": dec, "best_roc_auc": dec, "domain_shift_4a": dec})
    cal = res[res.model != "NULL_PREVALENCE"][["analysis", "held_out", "model", "feature_set", "calibrated", "brier_raw", "brier", "log_loss_raw", "log_loss", "cal_slope", "cal_intercept",
                                               "threshold", "precision", "recall", "f1"]]
    _sheet(wb, "CALIBRATION", cal, {c: dec for c in cal.columns[5:]})
    _sheet(wb, "DOMAIN_SHIFT", dsp, {c: dec for c in dsp.columns[1:]}, note=WARN + " Spearman rows are descriptive (n=6), not tests.")
    fi = interp.copy()
    fi["feature_group"] = np.where(fi.unit.str.startswith("GROUP_"), fi.unit.str.replace("GROUP_", ""),
                                   np.where(fi.unit.str.startswith("e_"), "E", np.where(fi.unit.str.startswith("g_"), "G", "M")))
    fi["rank_in_fold"] = fi.groupby(["held_out", "kind"]).permutation_pr_auc_drop.rank(ascending=False, method="min")
    fi = fi.rename(columns={"unit": "feature", "held_out": "fold"})[["fold", "feature", "feature_group", "kind", "model", "feature_set", "permutation_pr_auc_drop",
                                                                      "drop_sd", "native", "rank_in_fold"]]
    fi["importance_measure"] = "held-out permutation PR-AUC drop (subsample); native = RF impurity importance"
    _sheet(wb, "FEATURE_IMPORTANCE", fi, {"permutation_pr_auc_drop": "0.00000", "drop_sd": "0.00000", "native": "0.00000"})
    sens = summ[["analysis", "model", "feature_set", "median_pr_auc", "mean_pr_auc", "min_pr_auc", "max_pr_auc", "iqr_pr_auc"]].copy()
    sens.insert(0, "role", np.where(sens.analysis == "PRIMARY", "PRIMARY", "SENSITIVITY"))
    verdicts = []
    for f in ("STAGE4B_STRICT_SENSITIVITY.tsv", "STAGE4B_POLYGON_ONLY_SENSITIVITY.tsv", "STAGE4B_ATITLAN_SENSITIVITY.tsv"):
        t = _tsv(f)
        v = t[t.model == "__VERDICT__"].iloc[0]
        verdicts.append(dict(role="SENSITIVITY_VERDICT", analysis=v.analysis, model=f"material change: {v.materially_changes_conclusion}",
                             feature_set=f"best {v.best_family} vs comparator {v.comparator_best_family}"))
    bio = _tsv("STAGE4B_BIO15_SENSITIVITY.tsv")
    verdicts += [dict(role="SENSITIVITY_VERDICT", analysis="MEG_NO_BIO15", model=m, feature_set=f"median delta {g.delta_no_bio15.median():+.4f}") for m, g in bio.groupby("model")]
    _sheet(wb, "SENSITIVITIES", pd.concat([sens, pd.DataFrame(verdicts)], ignore_index=True), {c: dec for c in ("median_pr_auc", "mean_pr_auc", "min_pr_auc", "max_pr_auc", "iqr_pr_auc")})
    hp = prim[prim.model != "NULL_PREVALENCE"][["held_out", "model", "feature_set", "params", "calibrator", "threshold", "inner_best_mean_pr_auc"]]
    pre = json.loads((B4 / "STAGE4B_PREREGISTRATION.json").read_text())
    hp = hp.assign(fixed_parameters=hp.model.map(lambda m: json.dumps(pre["fixed_parameters"].get(m, {}), sort_keys=True)))
    _sheet(wb, "HYPERPARAMETERS", hp, {"threshold": "0.00000", "inner_best_mean_pr_auc": dec})
    v1 = v1_verify()
    gov = pd.DataFrame([dict(item=k, value=v) for k, v in [
        ("Stage 4A checkpoint sha256", STAGE4A_CHECKPOINT_SHA), ("Stage 4B checkpoint sha256", sha256(B4 / "STAGE4B_CHECKPOINT.json")),
        ("Stage 4B pre-registration sha256", cp4b["preregistration_sha256"]), ("Seed", pre["seed"]), ("scikit-learn", cp4b["software"]["version"]),
        ("scikit-learn isolation", cp4b["software"].get("isolation", "")), ("numpy / scipy / pandas", f"{cp4b['software']['numpy']} / {cp4b['software']['scipy']} / {cp4b['software']['pandas']}"),
        ("V1 verification", f"{v1['strict_files']} strict files intact={v1['v1_intact']}, writes={v1['writes_performed']}"), ("Lockbox status", "NOT ACCESSED (firewall active)"),
        ("H status", "NOT GENERATED / NOT USED"), ("Model status", "NONFINAL DEVELOPMENT ONLY")]])
    _sheet(wb, "GOVERNANCE", gov)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "STAGE4B_ML_RESULTS.xlsx"
    wb.properties.creator, wb.properties.title = "SFV2 Stage 4C-A", "Stage 4B ML results (DEVELOPMENT ONLY)"
    wb.save(p)
    return dict(path=p, hardest=hardest, easiest=easiest, null=null_med, best=(best.model, best.feature_set, float(best.median_pr_auc)))


# ------------------------------------------------------------------------------------------------ Part II-XXIII: H specification
def indicative_support() -> dict:
    """Y-free upper bound on H support from mapped geology only (no target column is read)."""
    out = {}
    for lid in H.LANDSCAPES:
        z = np.load(ROOT / f"outputs/stage4a/STAGE4A_MEG_DATASET/{lid}.npz")
        cls, age, q = z["qa_harmonised_geology_class"], z["qa_source_unit_code"], z["g_age_quaternary"]
        src_age = None
        if lid == "KUMAMOTO_ASO_2016":
            t = pd.read_csv(ROOT / "data/processed/stage2d/KUMAMOTO_ASO_2016/terrain_units/terrain_units.csv.gz", usecols=["source_age", "harmonised_geology_class"])
            m = (t.harmonised_geology_class == "PYROCLASTIC") & t.source_age.str.contains("Late Pleistocene")
            frac, rule = float(m.mean()), "units on mapped Late-Pleistocene pyroclastic-flow units (outer plateau)"
        elif lid == "NOTO_PENINSULA_2024":
            frac, rule = 0.0, "no constructional surface"
        elif lid in ("LOMBOK_RINJANI_2018",):
            frac, rule = float(np.nanmean(q == 1.0)), "units on strictly Quaternary (young edifice) units"
        elif lid == "COTABATO_DAVAO_2019":
            t = pd.read_csv(ROOT / "data/processed/stage3d_production/COTABATO_DAVAO_2019/terrain_units/terrain_units.csv.gz", usecols=["source_age"])
            frac, rule = float(t.source_age.str.contains("Quaternary").mean()), "units on Pliocene-Quaternary units"
        else:
            frac, rule = 1.0, "all mapped volcanic units (radial young centres + envelope on older units)" if lid.startswith("DOMINICA") else "all mapped volcanic units (envelope)"
        out[lid] = dict(indicative_geology_support_fraction_upper_bound=round(frac, 4), rule=rule, units=len(cls))
    return out


def write_spec(support: dict) -> None:
    tax = pd.DataFrame(H.TAXONOMY)
    tax["candidate_family_registry"] = "stage4c_a_spec.FAMILIES (extends Stage 2E Gen-II registry)"
    _write_tsv(OUT / "STAGE4C_A_RECONSTRUCTION_FAMILY_TAXONOMY.tsv", tax)
    feas = []
    for lid in H.LANDSCAPES:
        st, why = H.FEASIBILITY[lid]
        feas.append(dict(landscape_id=lid, current_status="NOT_RECONSTRUCTED (specification stage)", expected_status=st, basis=why,
                         family=next(t["family"] for t in H.TAXONOMY if t["landscape_id"] == lid), confidence=next(t["confidence"] for t in H.TAXONOMY if t["landscape_id"] == lid),
                         **support[lid], zero_fill_forbidden=True))
    _write_tsv(OUT / "STAGE4C_A_H_FEASIBILITY.tsv", feas)
    _write_tsv(OUT / "STAGE4C_A_RECONSTRUCTION_EVIDENCE.tsv", [dict(landscape_id=a, evidence=b, kind=c, availability=d, provenance=e) for a, b, c, d, e in H.EVIDENCE])
    cols = ["name", "physical_interpretation", "expected_transferability", "volcano_size_dependence", "family_dependence", "redundancy_class_expected",
            "uncertainty_sensitivity", "leakage_risk", "availability", "decision"]
    cand = pd.DataFrame(H.CANDIDATES, columns=cols)
    _write_tsv(OUT / "STAGE4C_A_H_CANDIDATE_FEATURES.tsv", cand)
    _write_tsv(OUT / "STAGE4C_A_H_FEATURE_SPEC.tsv", pd.DataFrame(H.CORE_H))
    _write_tsv(OUT / "STAGE4C_A_H_OPTIONAL_FEATURES.tsv", cand[cand.decision.str.startswith("OPTIONAL")].assign(class_="OPTIONAL_H"))
    _write_tsv(OUT / "STAGE4C_A_H_REJECTED_FEATURES.tsv", cand[cand.decision.str.startswith("REJECTED") & ~cand.decision.str.contains("CORE")].assign(class_="REJECTED_H"))
    plans = []
    for t in H.TAXONOMY:
        lid = t["landscape_id"]
        ev = [e for e in H.EVIDENCE if e[0] == lid]
        radial = "RADIAL" in t["family"]
        plans.append(dict(
            landscape_id=lid, volcano_type=t["volcano_type"], candidate_family=t["family"], temporal_target=t["temporal_target"],
            available_control_evidence="; ".join(e[1] for e in ev if e[3].startswith(("AVAILABLE", "EXTRACTABLE"))),
            missing_control_evidence="; ".join(f"{e[1]} [{e[3]}]" for e in ev if not e[3].startswith(("AVAILABLE", "EXTRACTABLE", "PRE-REGISTERED"))) or "none required",
            centre_definition=("registered volcano-centre catalogue coordinates (acquired in 4C-B; never from Y)" if radial else "not used (envelope family)" if lid != "NOTO_PENINSULA_2024" else "none"),
            domain_definition="Stage 4A OMEGA_ANALYSIS units intersected with the family support mask (mapped constructional units, caldera interiors excluded)",
            core_caldera_treatment="caldera interior and intrusive/vent cores masked (collapse is not erosion)" if "CALDERA" in t["family"] or lid in ("ATITLAN_STAN_2005", "LOMBOK_RINJANI_2018", "KUMAMOTO_ASO_2016") else "dome/vent cores masked within 0.5 km of registered centres" if radial else "n/a",
            candidate_surface_model=("radial macro-surface Z = B + H(theta)(1 - r/R)^p per centre, upper-envelope or nearest-scaled-centre combination" if radial
                                     else "low-order (1st/2nd) envelope trend through accordant summits" if lid != "NOTO_PENINSULA_2024" else "none"),
            uncertainty_parameters=("centre summit +/- tolerance; R; p; asymmetry a1,b1; base B; crest tolerance tau; coalescence rule" if radial
                                    else "trend order {1,2}; summit percentile {90,95,99}; control tolerance" if lid != "NOTO_PENINSULA_2024" else "none"),
            validation_method="leave-one-sector-out (radial) / leave-one-2-km-block-out (envelope) control holdout + geometric checks (STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md)" if lid != "NOTO_PENINSULA_2024" else "not applicable",
            expected_h_status=H.FEASIBILITY[lid][0], major_risks=H.FEASIBILITY[lid][1],
            tweed_concepts_applicable="; ".join(f"{k}={v[lid]}" for k, v in H.TWEED_CONCEPTS.items())))
    _write_tsv(OUT / "STAGE4C_A_PER_LANDSCAPE_RECONSTRUCTION_PLAN.tsv", plans)
    subset = [dict(landscape_id=lid, expected_status=H.FEASIBILITY[lid][0], provisionally_h_capable=H.FEASIBILITY[lid][0] in ("FULL_RECONSTRUCTION", "PARTIAL_RECONSTRUCTION"),
                   verified_h_capable=False, condition="must pass every Y-independent reconstruction-validation gate in Stage 4C-B") for lid in H.LANDSCAPES]
    n_prov = sum(s["provisionally_h_capable"] for s in subset)
    subset.append(dict(landscape_id="N_H_CAPABLE_LANDSCAPES", expected_status=f"provisional={n_prov}; verified=0",
                       provisionally_h_capable=n_prov >= 3, verified_h_capable=False, condition="pooled H LOLO permitted only if >=3 landscapes are VERIFIED after 4C-B validation"))
    _write_tsv(OUT / "STAGE4C_A_H_EXPERIMENT_SUBSET.tsv", subset)
    for name, text in DOCS.items():
        (OUT / name).write_text(text)


DOCS = {
    "STAGE4C_A_H_DEFINITION.md": """# Formal definition of H (Stage 4C-A, frozen)

**H is information about how the present terrain relates to a scientifically plausible earlier volcanic landscape,** derived only from geological and
geomorphological evidence that is independent of landslide occurrence. H is not "old elevation" and not a palaeo-DEM pasted into the model.

For landscape L with family F_L and frozen temporal target T_L, an ensemble of J plausible reference surfaces Z_ref^(j)(x), j = 1..J, is built
inside an explicit support mask S_L. H at a terrain unit is defined only inside S_L; outside S_L every H value is missing (never zero).

* **A. Reconstructed state:** properties of Z_ref itself, e.g. normalised position on the reconstructed surface.
* **B. Landscape change:** differences between Z_ref and the modern surface Z_mod (removed fraction, slope change, dissection).
* **C. Position within the reconstructed volcano:** normalised height on the reconstructed surface; radial position only within radial families (optional).
* **D. Reconstruction uncertainty:** scenario spread and distance to direct evidence.

Landscape scales used for normalisation (computed from the reconstruction, never from Y):
R_L = P98 - P02 of the scenario-median surface inside S_L (reconstructed relief); Lambda_L = sqrt(area(S_L)/pi) (size scale).

Semantics flag (QA, never a predictor): ESTIMATED_REMOVAL (radial/caldera families) or MINIMUM_REMOVAL (envelope families).
AGE_PROXY_ONLY landscapes carry no surface-based H. Missing H is never zero-filled.
""",
    "STAGE4C_A_V1_H_FORENSIC_AUDIT.md": """# V1 H forensic audit (read-only; V1 untouched)

**Sources read:** `src/tweed_science_fair/sf4_controls.py`, `sf4_ensemble.py`, `sf4_grid.py`, `sf4f_production.py`, `sf4f_build.py`, `sf5_features.py`,
`sf6_member_h.py`, `src/model_e_fit/surface.py`, and `Science_Fair_Extension/00_Governance/SF4F, SF5, SF6, SF7` documents. Nothing was modified, and no V1 H raster or value is reused.

## How V1 built H
1. **Controls (landslide-blind, SF-4):**
   - GVP volcano centres and summit heights;
   - DEM-derived crest envelopes per (centre, 10 deg sector, 1 km radial bin) with TPI(500 m) >= 10 m;
   - caldera rims from DEM morphology (Lombok: Segara Anak);
   - plateau accordant summits;
   - a 1 km crater exclusion.
2. **Surface family:** the Tweed "Model E" radial macro-surface Z = B + H(1 + a1 cos th + b1 sin th)(1 - r/R)^p for cone landscapes, combined across centres by
   UPPER_ENVELOPE or NEAREST_SCALED_CENTRE. Kurikoma and Atitlan used plateau/edifice polynomial trends.
3. **Ensemble:** LHS/Sobol parameter sampling within frozen bounds; members accepted when RMS one-sided control misfit <= sampled tau.
   - Lombok: 893 members x 4 structural variants, including a RESTORED_SAMALAS_CONE variant.
   - Atitlan: 6 structural alternatives with no parameter spread.
   - Kurikoma: 558 members.
4. **Grid:** surfaces on a 90 m grid, bilinear to 30 m; H aggregated to terrain units with a 120 m median.
5. **H variables:**
   - primary: H_SURFACE_DIFFERENCE_M and H_LOCAL_SURFACE_DIFFERENCE_500M_M;
   - secondary: H_PALAEOELEVATION_M, H_PALAEO_SLOPE_DEG, H_PALAEO_RELATIVE_ELEVATION, H_FLANK_POSITION (not applicable at Atitlan).
6. **Uncertainty:** ensemble IQR, structural range, parameter spread, ordinal support and control distance were kept separate and labelled not-a-probability.
7. **Landscapes with H:** Lombok and Atitlan (development); Kurikoma (external test).

## Empirical result (must stay explicit)
- **Internal (SF-6, Lombok + Atitlan):** H helped. DELTA PR-AUC MGEH - MGE = +0.007 (logistic), +0.028 (RF), +0.030 (boosting).
- **External (SF-7, Kurikoma):** H hurt. DELTA PR-AUC = **-0.048** (95% block bootstrap -0.080 to -0.015); F1 fell from 0.26 to 0.03.
- **Already unstable within development:** the leave-one-landscape-out diagnostic was mixed (for example, boosting Lombok -> Atitlan -0.057).

## Why V1 H likely transferred poorly (diagnosis from the audit; not tested here)
1. **Raw metres encode volcano size and family.** Kurikoma H was far outside the training distribution: SMD -1.56 (surface difference), -2.15 (palaeoelevation);
   Lombok's median removed thickness was ~1,156 m against ~220 m at Kurikoma.
2. **Mixed reconstruction semantics.** Lombok used a restored cone (estimated removal), while Atitlan and Kurikoma were MINIMUM_REMOVAL plateaux with
   no upper bound. Equal numbers therefore meant different things.
3. **Redundancy with M.** H_LOCAL_SURFACE_DIFFERENCE_500M ~ -TPI of the modern DEM; palaeoelevation ~ modern elevation + smooth trend.
4. **Too few development landscapes.** Two landscapes from two different families cannot teach a transferable mapping.
5. **Caldera collapse.** The restored Samalas cone treated collapse volume as erosion.

## KEEP / MODIFY / REJECT
| V1 concept | V2 decision |
|---|---|
| Landslide-blind controls; Y-read guards | KEEP |
| Crest/remnant envelopes and caldera rims as controls | KEEP (with a control-quality hierarchy) |
| Model E radial macro-surface | MODIFY: radial families only, never global |
| Scenario ensemble with acceptance by control misfit | KEEP; label it "plausible scenarios", not probabilistic |
| Separate uncertainty components | KEEP (total spread is CORE; the decomposition is OPTIONAL) |
| H_SURFACE_DIFFERENCE_M (raw metres) | MODIFY to H_REMOVED_FRACTION (divided by reconstructed relief) |
| H_LOCAL_SURFACE_DIFFERENCE_500M_M | REJECT (redundant with M TPI) |
| H_PALAEOELEVATION_M | REJECT (size/identity-dependent, redundant) |
| H_PALAEO_SLOPE_DEG | MODIFY to slope change (OPTIONAL as raw) |
| H_PALAEO_RELATIVE_ELEVATION | MODIFY to H_REFERENCE_RELATIVE_HEIGHT (all families) |
| H_FLANK_POSITION | OPTIONAL (radial families only) |
| Restored pre-collapse cone (Samalas) | REJECT as primary; caldera interiors are masked |
| 120 m median aggregation | MODIFY to the frozen 90 m terrain units of Stage 4A |

**Conclusion:** V2 H must solve transferability through normalisation, a common family-agnostic CORE_H, per-landscape validation and identical-subset comparison.
It must not simply reproduce V1.
""",
    "STAGE4C_A_CONTROL_SELECTION_SPEC.md": """# Reconstruction control selection (frozen; landslide-blind)

## Admissibility
Every rule below is required:
1. **Volcanic relevance:** the control lies on a mapped Cenozoic volcanic constructional or depositional unit of the landscape's temporal target (frozen geology).
2. **Outside excluded zones:** not inside caldera interiors, vent/dome cores (0.5 km of registered centres), intrusive cores, or mapped surficial or landslide-deposit units.
3. **Elevated and preserved:** a ridge or interfluve crest with TPI(450 m) >= 10 m, or a plateau top with slope <= 8 deg over >= 25 cells.
4. **Coherence and continuity:** lies in a laterally continuous crest or plateau segment >= 3 cells and agrees with adjacent controls of the same unit within the tolerance tau.
5. **Plausible geometry:** for radial families, consistent with outward-decreasing flank height along its sector.
6. **No target information:** landslide occurrence, Y, Omega labels, Stage 4B outputs and residuals are never inputs. Code paths must fail closed on any Y path.

## Quality hierarchy (recorded per control; used for weighting only through evidence, never performance)
- **Q1:** summit/rim elevations of registered centres; plateau tops of mapped depositional units.
- **Q2:** accordant interfluve crests on mapped constructional units.
- **Q3:** geomorphic crests with weak or ambiguous unit support (diagnostic only; excluded from fitting).
- **Q4:** analogue or literature-only values (excluded from fitting).

Fitting uses Q1-Q2 only. The minimum is 12 admissible controls per landscape (Stage 2E gate).
""",
    "STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md": """# Reconstruction validation (Y-independent; must pass before any ML use)

**Holdout:** leave-one-sector-out (radial, 8 sectors) or leave-one-2-km-block-out (envelope) of admissible Q1-Q2 controls. Refit without the held-out
controls and predict them.

**Metrics:**
- median |vertical residual| / R_L;
- signed bias / R_L;
- fraction of held-out controls inside the scenario P05-P95 band;
- residuals by control quality and distance.

**Pre-registered gates (all must pass):**
- median |residual| / R_L <= 0.10;
- |bias| / R_L <= 0.05;
- holdout band fraction >= 0.60;
- >= 12 admissible controls;
- support covers >= 25% of the landscape's Stage 4A units;
- negative-removal fraction <= 0.10 outside mapped young deposits;
- constructional reference slopes <= 40 deg.

**Geometric checks:**
- no reference surface below preserved remnants beyond tolerance;
- no unsupported summit above the highest control + tolerance;
- caldera interiors masked;
- edifice dimensions within the range of the landscape's own controls;
- removed volume reported with scenario P05-P95 (no probability claim).

**Optional external checks:** comparison with published cross sections where they exist (not required to pass).

**Forbidden:** Y, landslide maps, Stage 4B residuals/predictions/importance, and tuning to model performance.
""",
    "STAGE4C_B_EXPERIMENT_CONTRACT.md": """# Stage 4C-B experiment contract (DESIGN ONLY; not executed)

## Steps
1. **Acquire** the registered volcano-centre catalogue for the radial landscapes (public; Y-free).
2. **Extract** admissible controls (STAGE4C_A_CONTROL_SELECTION_SPEC.md).
3. **Build** the reconstruction scenario ensembles per family.
4. **Validate** each landscape against the Y-independent gates (STAGE4C_A_RECONSTRUCTION_VALIDATION_SPEC.md).
5. **Freeze** the reconstructions and the CORE_H rasters/units, with hashes, before any model sees them.
6. **Append** CORE_H to the identical Stage 4A rows (h_* columns + h_available). Missing H is never zero-filled.
7. **Fix the subset:** the H-capable subset is the landscapes passing validation. **Pooled H LOLO only if >= 3 landscapes pass.**
8. **Compare** on that subset, on identical rows (units with h_available inside the subset) and identical outer/inner folds:
   - **M**
   - **MEG**
   - **MEGH** (MEG + 6 CORE_H)
   - **MEG + SELECTED_H** (pre-registered rule in STAGE4C_A_H_FEATURE_SPEC.tsv / report; Y-independent).

## Fairness
Training sampling, landscape-balanced weighting, model families, grids, tuning budget, calibration policy, thresholds and seeds are all identical
to Stage 4B. There are no H-specific hyperparameters.

## Primary effect
**Delta_H(L) = PR-AUC(MEGH, held-out L) - PR-AUC(MEG, held-out L)**, reported per held-out landscape, with the median, minimum and maximum.
Pooled metrics are secondary only.

## Interpretation
H is supported only if Delta_H is non-negative for most held-out landscapes and does not collapse the worst landscape. The V1 lesson is that internal gains
can reverse externally.
""",
    "STAGE4C_A_FINAL_PRODUCT_ARCHITECTURE.md": """# Final product architecture (for a genuinely new volcano)

**Inputs:** modern pre-event DEM; environment (CHELSA, tree cover); harmonised geology; reconstruction controls where they exist.

**Pipeline:**
1. **M** generation (frozen definitions).
2. **E** generation.
3. **G** harmonisation to the frozen ontology.
4. **H** reconstruction ensemble:
   - choose the family from evidence;
   - extract controls;
   - validate Y-free;
   - produce CORE_H and uncertainty only inside the support mask.
5. **OOD/domain assessment:** distance of the new landscape's M/E/G/H distributions to the development set; warn when outside.
6. **Prediction model:** the frozen final model chosen after 8 development landscapes + 3 lockboxes.

**Outputs:**
- relative landslide susceptibility (not event probability unless calibration is shown to transfer);
- prediction confidence;
- reconstruction uncertainty;
- OOD warning;
- interpretable map.

**Fallback:** if a defensible H cannot be built (AGE_PROXY_ONLY / H_UNAVAILABLE or validation fails), use the M or MEG model. H is never fabricated or zero-filled.
""",
}


def main() -> dict:
    v1_before = v1_verify()
    assert sha256(B4 / "STAGE4B_CHECKPOINT.json") == STAGE4B_CHECKPOINT_SHA, "STOP: Stage 4B checkpoint changed"
    assert sha256(ROOT / "outputs/stage4a/STAGE4A_CHECKPOINT.json") == STAGE4A_CHECKPOINT_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    wbinfo = workbook()
    support = indicative_support()
    write_spec(support)
    n_prov = sum(H.FEASIBILITY[l][0] in ("FULL_RECONSTRUCTION", "PARTIAL_RECONSTRUCTION") for l in H.LANDSCAPES)
    (OUT / "STAGE4C_A_REPORT.md").write_text(
        f"# Stage 4C-A report\n\nWorkbook: STAGE4B_ML_RESULTS.xlsx (read-only view of Stage 4B; null median {wbinfo['null']:.4f}; best median {wbinfo['best']}).\n"
        f"CORE_H ({len(H.CORE_H)}): {', '.join(c['name'] for c in H.CORE_H)}.\nExpected statuses: {json.dumps({k: v[0] for k, v in H.FEASIBILITY.items()})}.\n"
        f"Provisionally H-capable: {n_prov}; verified: 0 (validation happens in 4C-B). Indicative geology support (upper bound): "
        f"{json.dumps({k: v['indicative_geology_support_fraction_upper_bound'] for k, v in support.items()})}.\nSelected-H rule: {json.dumps(H.SELECTED_H_RULE)}\n"
        "No model trained, no MEGH evaluated, no H generated, no Y/residual used, no lockbox accessed.\n")
    v1_after = v1_verify()
    outs = sorted(p for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("STAGE4C_A_CHECKPOINT", "STAGE4C_A_DIAGNOSTICS")))
    cp = dict(schema="SFV2_STAGE4C_A_CHECKPOINT_V1", as_of=AS_OF, stage4b_checkpoint_sha256=STAGE4B_CHECKPOINT_SHA, stage4a_checkpoint_sha256=STAGE4A_CHECKPOINT_SHA,
              core_h=[c["name"] for c in H.CORE_H], expected_status={k: v[0] for k, v in H.FEASIBILITY.items()}, provisional_h_capable=n_prov, verified_h_capable=0,
              pooled_h_lolo_permitted_now=False, selected_h_rule=H.SELECTED_H_RULE, validation_gates=H.VALIDATION_GATES["gates"],
              output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs if p.suffix != ".xlsx"},
              workbook_sha256_volatile_excluded="xlsx zip timestamps vary; content verified by tests against Stage 4B TSVs",
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage4c_a.py", "stage4c_a_spec.py")},
              model_trained=False, megh_evaluated=False, h_generated=False, y_used_for_h=False, stage4b_residuals_used_for_h=False, lockbox_accessed=False,
              ml_or_h_artifacts_outside_stage4b=[p for p in scan_ml_h_artifacts() if not p.startswith("outputs/stage4b/")],
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")})
    (OUT / "STAGE4C_A_CHECKPOINT.json").write_text(json.dumps(cp, indent=2, sort_keys=True) + "\n")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    (OUT / "STAGE4C_A_DIAGNOSTICS.json").write_text(json.dumps(dict(authoritative=False, head=git("rev-parse", "HEAD")), indent=2) + "\n")
    return dict(cp=cp, wb=wbinfo, support=support)


if __name__ == "__main__":
    main()
