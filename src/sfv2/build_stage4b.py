"""Build Stage 4B: development-only nested LOLO benchmark (NONFINAL).  Phases: freeze pre-registration -> run -> assemble outputs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from . import stage4a_base as B
from . import stage4b as S
from .build_stage3a import scan_ml_h_artifacts
from .build_stage4a import write_columns
from .raw_store import sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

AS_OF = "2026-10-06"
OUT = ROOT / "outputs/stage4b"
RESULTS_CACHE = ROOT / "data/interim/_stage4b_results"
LOG = ROOT / "data/interim/_stage4b_progress.log"
STAGE4A_CHECKPOINT_SHA = "fd25accb50578bd4f4f28042e42504a89ef936c904d5ed262a0f65fc7fa2452c"
SOFTWARE = dict(package="scikit-learn", version=sklearn.__version__,
                installation_command=(".venv/bin/python -m pip install --target Science_Fair_ML_V2/.ml_site --no-deps scikit-learn==1.7.2 joblib==1.6.0 "
                                      "threadpoolctl==3.7.0 cloudpickle==3.1.2"),
                isolation=("V2-only site directory added to PYTHONPATH for V2 ML runs; the shared .venv is unchanged because frozen V1 governance tests "
                           "assert scikit-learn is not importable there (an initial shared-venv install was reverted)"),
                reason="Stage 4B benchmark models (LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier); no other ML library installed",
                dependencies_added="joblib 1.6.0, threadpoolctl 3.7.0, cloudpickle 3.1.2 (in the isolated site only)", numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, python=sys.version.split()[0])


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def _tsv(p: Path, df) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(p, sep="\t", index=False, lineterminator="\n", float_format="%.6g")


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


# ------------------------------------------------------------------------------------------------ phase 1
def freeze_prereg() -> str:
    """Write the pre-registration; refuse to change it once any result exists."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "STAGE4B_PREREGISTRATION.json"
    new = json.dumps(S.prereg(), indent=2, sort_keys=True) + "\n"
    if p.exists() and p.read_text() != new and any(RESULTS_CACHE.glob("*.json")):
        raise RuntimeError("STOP: pre-registration may not change after results exist")
    p.write_text(new)
    outer, inner = [], []
    for name, (universe, pop, scheme, fsets, models) in S.ANALYSES.items():
        for k in universe:
            train = [l for l in universe if l != k]
            outer.append(dict(analysis=name, held_out=k, training_landscapes="|".join(train), population=pop, weighting=scheme))
            for j in train:
                inner.append(dict(analysis=name, outer_held_out=k, inner_held_out=j, inner_training="|".join(l for l in train if l != j)))
    _tsv(OUT / "STAGE4B_OUTER_FOLDS.tsv", outer)
    _tsv(OUT / "STAGE4B_INNER_FOLDS.tsv", inner)
    _json(OUT / "STAGE4B_HYPERPARAMETER_GRID.json", dict(grids=S.GRIDS, fixed=S.FIXED, tie_tolerance=S.TIE_TOL, selection="mean inner-landscape PR-AUC"))
    return sha256(p)


# ------------------------------------------------------------------------------------------------ phase 2
def run(d: S.Data | None = None) -> dict:
    d = d or S.Data()
    RESULTS_CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in S.ANALYSES:
        f = RESULTS_CACHE / f"{name}.json"
        if not f.exists():
            res, inner, meta = S.run_analysis(d, name, log=_log)
            f.write_text(json.dumps(dict(results=res, inner=inner, meta=meta), sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
        out[name] = json.loads(f.read_text())
    return out


# ------------------------------------------------------------------------------------------------ phase 3 helpers
def summarise(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby(["analysis", "model", "feature_set"])
    s = g.pr_auc.agg(median_pr_auc="median", mean_pr_auc="mean", min_pr_auc="min", max_pr_auc="max",
                     q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
    s["iqr_pr_auc"] = s.q75 - s.q25
    extra = g.agg(median_pr_auc_lift=("pr_auc_lift", "median"), median_roc_auc=("roc_auc", "median"), median_brier=("brier", "median"),
                  median_log_loss=("log_loss", "median"), median_cal_slope=("cal_slope", "median"),
                  median_abs_slope_dev=("cal_slope", lambda x: float(np.nanmedian(np.abs(x - 1)))), median_f1=("f1", "median"),
                  worst_landscape=("pr_auc", lambda x: res.loc[x.idxmin(), "held_out"])).reset_index()
    return s.drop(columns=["q25", "q75"]).merge(extra, on=["analysis", "model", "feature_set"])


def select_leader(summ: pd.DataFrame) -> dict:
    c = summ[(summ.analysis == "PRIMARY") & summ.model.isin(S.MODEL_SIMPLICITY) & summ.feature_set.isin(S.FS_SIMPLICITY)].copy()
    best = c.median_pr_auc.max()
    c = c[c.median_pr_auc >= best - 0.01]
    c["simp"] = c.model.map(S.MODEL_SIMPLICITY.index) * 10 + c.feature_set.map(S.FS_SIMPLICITY.index)
    c = c.sort_values(["min_pr_auc", "median_abs_slope_dev", "simp"], ascending=[False, True, True])
    top = c.iloc[0]
    return dict(model=top.model, feature_set=top.feature_set, median_pr_auc=top.median_pr_auc, min_pr_auc=top.min_pr_auc,
                candidates_within_0_01=c[["model", "feature_set", "median_pr_auc", "min_pr_auc", "median_abs_slope_dev"]].to_dict("records"))


def ablations(res: pd.DataFrame, analysis: str) -> pd.DataFrame:
    r = res[(res.analysis == analysis) & (res.model != "NULL_PREVALENCE")].pivot_table(index=["model", "held_out"], columns="feature_set", values="pr_auc").reset_index()
    if {"M", "ME"} <= set(r.columns):
        r["delta_E"] = r.ME - r.M
    if {"ME", "MEG"} <= set(r.columns):
        r["delta_G"] = r.MEG - r.ME
    if {"M", "MEG"} <= set(r.columns):
        r["delta_MEG"] = r.MEG - r.M
    if {"MEG", "MEG_NO_BIO15"} <= set(r.columns):
        r["delta_no_bio15"] = r.MEG_NO_BIO15 - r.MEG
    r.insert(0, "analysis", analysis)
    return r


def domain_distance() -> dict:
    a = pd.read_csv(ROOT / "outputs/stage4a/STAGE4A_DOMAIN_SHIFT_AUDIT.tsv", sep="\t")
    a = a[a.predictor == "__LANDSCAPE_DISTANCE__"]
    return dict(zip(a.landscape_id, a.mean_standardised_median_shift_all.astype(float)))


def spatial_consistency(d: S.Data, lid: str, p: np.ndarray) -> float:
    cx, cy = d.L[lid]["cx"], d.L[lid]["cy"]
    ix, iy = np.round(cx / 90).astype(np.int64), np.round(cy / 90).astype(np.int64)
    pos = {(a, b): i for i, (a, b) in enumerate(zip(ix, iy))}
    nb = np.full(len(p), np.nan)
    for i, (a, b) in enumerate(zip(ix, iy)):
        v = [p[pos[(a + dx, b + dy)]] for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (a + dx, b + dy) in pos]
        if v:
            nb[i] = np.mean(v)
    ok = ~np.isnan(nb)
    return float(np.corrcoef(p[ok], nb[ok])[0, 1])


def interpretation(d: S.Data, leader: dict, meta: dict) -> tuple[pd.DataFrame, list]:
    fs, model = leader["feature_set"], leader["model"]
    groups = {"M": [c for c in d.fs[fs] if not c.startswith(("e_", "g_"))], "E": [c for c in d.fs[fs] if c.startswith("e_")],
              "G": [c for c in d.fs[fs] if c.startswith("g_")]}
    groups = {g: v for g, v in groups.items() if v}
    rows, models = [], []
    for fi, k in enumerate(S.LANDSCAPES):
        m = meta[f"PRIMARY|{k}|{fs}|{model}"]
        est, cols, prep, waudit, _ = S.fit(d, m["train"], fs, model, m["params"], "ALL_VALID", "LANDSCAPE_BALANCED")
        models.append((k, est))
        s = d.sample(k, "ALL_VALID")  # deterministic interpretation subsample of the held-out landscape (not used for primary metrics)
        X, y = d.L[k]["X"][s][:, cols], d.L[k]["y"][s]
        tf = (lambda Z: (Z - np.asarray(prep["mean"])) / np.asarray(prep["sd"])) if prep else (lambda Z: Z)
        base = average_precision_score(y, est.predict_proba(tf(X))[:, 1])
        names = d.fs[fs]
        units = [(f"GROUP_{g}", [names.index(c) for c in v]) for g, v in groups.items()] + [(c, [i]) for i, c in enumerate(names)]
        for label, idx in units:
            drops = []
            for rep in range(3):
                rng = np.random.default_rng(S.SEED + 1000 * fi + rep)
                Xp = X.copy()
                perm = rng.permutation(len(X))
                Xp[:, idx] = X[perm][:, idx]
                drops.append(base - average_precision_score(y, est.predict_proba(tf(Xp))[:, 1]))
            native = None
            if not label.startswith("GROUP_"):
                if hasattr(est, "feature_importances_"):
                    native = float(est.feature_importances_[idx[0]])
                elif hasattr(est, "coef_"):
                    native = float(est.coef_[0, idx[0]])
            rows.append(dict(held_out=k, model=model, feature_set=fs, unit=label, kind="group" if label.startswith("GROUP_") else "feature",
                             baseline_pr_auc_subsample=base, permutation_pr_auc_drop=float(np.mean(drops)), drop_sd=float(np.std(drops)), native=native))
    return pd.DataFrame(rows), models


# ------------------------------------------------------------------------------------------------ phase 3
def main() -> dict:
    v1_before = v1_verify()
    assert sha256(ROOT / "outputs/stage4a/STAGE4A_CHECKPOINT.json") == STAGE4A_CHECKPOINT_SHA, "STOP: Stage 4A checkpoint changed"
    prereg_sha = freeze_prereg()
    d = S.Data()
    n = {l: len(d.L[l]["y"]) for l in S.LANDSCAPES}
    pos = {l: int(d.L[l]["y"].sum()) for l in S.LANDSCAPES}
    data_check = dict(landscapes=len(d.L), rows=sum(n.values()), positives=sum(pos.values()), negatives=sum(n.values()) - sum(pos.values()),
                      M=len(d.fs["M"]), E=len(d.fs["ME"]) - len(d.fs["M"]), G=len(d.fs["MEG"]) - len(d.fs["ME"]),
                      finite=bool(all(np.isfinite(d.L[l]["X"]).all() for l in S.LANDSCAPES)))
    assert data_check["rows"] == 835846 and data_check["positives"] == 8793 and (data_check["M"], data_check["E"], data_check["G"]) == (60, 7, 2)
    raw = run(d)
    res = pd.DataFrame([r for a in raw.values() for r in a["results"]])
    res["model"] = res.model.replace({"NULL": "NULL_PREVALENCE"})  # "NULL" is parsed as missing by TSV readers
    inner = pd.DataFrame([r for a in raw.values() for r in a["inner"]])
    meta = {k: v for a in raw.values() for k, v in a["meta"].items()}
    summ = summarise(res)
    leader = select_leader(summ)
    dist = domain_distance()

    _tsv(OUT / "STAGE4B_MODEL_RESULTS.tsv", summ)
    _tsv(OUT / "STAGE4B_LANDSCAPE_RESULTS.tsv", res)
    _tsv(OUT / "STAGE4B_INNER_SELECTION.tsv", inner)
    abl = ablations(res, "PRIMARY")
    _tsv(OUT / "STAGE4B_FEATURE_ABLATIONS.tsv", pd.concat([abl] + [ablations(res, a) for a in ("ATITLAN_EXCLUDED", "STRICT_DEM_QUALITY", "POLYGON_ONLY")], ignore_index=True))
    bio = abl[["model", "held_out", "MEG", "MEG_NO_BIO15", "delta_no_bio15"]].copy()
    bio["domain_distance"] = bio.held_out.map(dist)
    bio_summary = {m: dict(median_delta=float(g.delta_no_bio15.median()), improved_landscapes=int((g.delta_no_bio15 > 0).sum()),
                           spearman_delta_vs_distance=float(spearmanr(g.domain_distance, g.delta_no_bio15).statistic)) for m, g in bio.groupby("model")}
    _tsv(OUT / "STAGE4B_BIO15_SENSITIVITY.tsv", bio)

    lm, lf = leader["model"], leader["feature_set"]
    lab = abl[abl.model == lm].set_index("held_out")
    lead_res = res[(res.analysis == "PRIMARY") & (res.model == lm) & (res.feature_set == lf)].set_index("held_out")
    ds = pd.DataFrame([dict(held_out=k, domain_distance=dist[k], leader_pr_auc=lead_res.loc[k, "pr_auc"], leader_pr_auc_lift=lead_res.loc[k, "pr_auc_lift"],
                            delta_E=lab.loc[k, "delta_E"], delta_G=lab.loc[k, "delta_G"], delta_MEG=lab.loc[k, "delta_MEG"]) for k in S.LANDSCAPES])
    corr = {c: float(spearmanr(ds.domain_distance, ds[c]).statistic) for c in ("leader_pr_auc", "leader_pr_auc_lift", "delta_E", "delta_G", "delta_MEG")}
    _tsv(OUT / "STAGE4B_DOMAIN_SHIFT_PERFORMANCE.tsv", pd.concat([ds, pd.DataFrame([dict(held_out="SPEARMAN_RHO_n6_DESCRIPTIVE", **{k: v for k, v in corr.items()})])],
                                                                 ignore_index=True))

    def sens(analysis, comparator):
        a = summ[(summ.analysis == analysis) & (summ.model != "NULL_PREVALENCE")]
        cmp = summ[(summ.analysis == comparator) & (summ.model != "NULL_PREVALENCE")]
        best_a = a[a.feature_set == "MEG"].sort_values("median_pr_auc").iloc[-1].model
        best_c = cmp[cmp.feature_set == "MEG"].sort_values("median_pr_auc").iloc[-1].model
        ab_a, ab_c = ablations(res, analysis), ablations(res, comparator)
        if comparator == "PRIMARY":
            ab_c = ab_c[ab_c.held_out.isin(S.ANALYSES[analysis][0])]
        dm_a = float(ab_a[ab_a.model == lm].delta_MEG.median()) if "delta_MEG" in ab_a else float("nan")
        dm_c = float(ab_c[ab_c.model == lm].delta_MEG.median())
        la = float(a[(a.model == lm) & (a.feature_set == "MEG")].median_pr_auc.iloc[0])
        lc = float(cmp[(cmp.model == lm) & (cmp.feature_set == "MEG")].median_pr_auc.iloc[0]) if comparator != "PRIMARY" else \
            float(res[(res.analysis == "PRIMARY") & (res.model == lm) & (res.feature_set == "MEG") & res.held_out.isin(S.ANALYSES[analysis][0])].pr_auc.median())
        material = (best_a != best_c) or (np.sign(dm_a) != np.sign(dm_c)) or abs(la - lc) > 0.02
        rows = a.copy()
        rows["comparator"] = comparator
        verdict = dict(analysis=analysis, comparator=comparator, best_family=best_a, comparator_best_family=best_c, leader_family_median_delta_MEG=dm_a,
                       comparator_median_delta_MEG=dm_c, leader_family_median_MEG_pr_auc=la, comparator_median_MEG_pr_auc=lc, materially_changes_conclusion=bool(material))
        return pd.concat([rows, pd.DataFrame([dict(model="__VERDICT__", **{k: str(v) for k, v in verdict.items()})])], ignore_index=True), verdict

    verdicts = {}
    for a, comp, fname in (("STRICT_DEM_QUALITY", "ATITLAN_EXCLUDED", "STAGE4B_STRICT_SENSITIVITY.tsv"), ("POLYGON_ONLY", "PRIMARY", "STAGE4B_POLYGON_ONLY_SENSITIVITY.tsv"),
                           ("ATITLAN_EXCLUDED", "PRIMARY", "STAGE4B_ATITLAN_SENSITIVITY.tsv")):
        t, verdicts[a] = sens(a, comp)
        _tsv(OUT / fname, t)
    rw = summ[summ.analysis == "ROW_WEIGHTING"]
    verdicts["ROW_WEIGHTING"] = dict(rows=rw[["model", "feature_set", "median_pr_auc", "min_pr_auc"]].to_dict("records"))
    _tsv(OUT / "STAGE4B_ROW_WEIGHTING_SENSITIVITY.tsv", rw)

    interp, models = interpretation(d, leader, meta)
    grp = interp[interp.kind == "group"].groupby("unit").permutation_pr_auc_drop.agg(["median", "min", "max"]).reset_index()
    feat = interp[interp.kind == "feature"]
    top = feat.groupby("unit").permutation_pr_auc_drop.median().sort_values(ascending=False).head(12)
    ranks = feat.pivot_table(index="unit", columns="held_out", values="permutation_pr_auc_drop").rank(ascending=False)
    stability = float(ranks.corr(method="spearman").where(~np.eye(len(ranks.columns), dtype=bool)).stack().mean())
    _tsv(OUT / "STAGE4B_FEATURE_INTERPRETATION.tsv", interp)

    # predictions for mapping (leader family x primary feature sets); coordinates are mapping-only NON_PREDICTOR columns
    pred_dir = OUT / "predictions_NONFINAL"
    pred_files, spatial = {}, {}
    for fs in S.ANALYSES["PRIMARY"][3]:
        frames = []
        for k in S.LANDSCAPES:
            m = meta[f"PRIMARY|{k}|{fs}|{lm}"]
            p = np.load(S.CACHE / f"{m['outer_prediction_key']}__{k}.npy")
            pc = S.apply_platt(p, m["calibrator"])
            frames.append(pd.DataFrame(dict(terrain_unit_id=d.L[k]["ids"], landscape_id=k, fold=f"HOLD_OUT_{k}", model=lm, feature_set=fs,
                                            raw_probability=p, calibrated_probability=pc, target=d.L[k]["y"],
                                            centroid_x_mapping_only=d.L[k]["cx"], centroid_y_mapping_only=d.L[k]["cy"])))
            if fs == lf:
                spatial[k] = spatial_consistency(d, k, p)
        f = pred_dir / f"PREDICTIONS_{lm}_{fs}_NONFINAL.npz"
        write_columns(f, pd.concat(frames, ignore_index=True))
        pred_files[str(f.relative_to(ROOT))] = sha256(f)

    # NONFINAL leader artifacts: per-fold settings (models are cheap to refit deterministically and are not persisted as pickles)
    leader_doc = dict(label="DEVELOPMENT_LEADER (NONFINAL - not the final V2 model)", **{k: v for k, v in leader.items()},
                      per_fold={k: dict(params=meta[f"PRIMARY|{k}|{lf}|{lm}"]["params"], calibrator=meta[f"PRIMARY|{k}|{lf}|{lm}"]["calibrator"],
                                        threshold=meta[f"PRIMARY|{k}|{lf}|{lm}"]["threshold"]) for k in S.LANDSCAPES},
                      fixed_parameters=S.FIXED[lm], selection_rule=S.prereg()["leader_rule"], spatial_consistency_neighbour_r=spatial,
                      final=False, frozen=False, lockbox_evaluated=False)
    _json(OUT / "STAGE4B_DEVELOPMENT_LEADER.json", leader_doc)
    w_audit = {f"{name}|{k}": S.weights(d, [l for l in S.ANALYSES[name][0] if l != k], S.ANALYSES[name][1], S.ANALYSES[name][2])[1]
               for name in ("PRIMARY", "ROW_WEIGHTING") for k in S.LANDSCAPES}
    _json(OUT / "STAGE4B_WEIGHT_AUDIT.json", w_audit)
    prep_meta = {}
    for k in S.LANDSCAPES:
        for fs in ("M", "ME", "MEG"):
            m = meta[f"PRIMARY|{k}|{fs}|LOGISTIC_L2"]
            _, _, prep, _, info = S.fit(d, m["train"], fs, "LOGISTIC_L2", m["params"], "ALL_VALID", "LANDSCAPE_BALANCED")
            prep_meta[f"{k}|{fs}"] = dict(prep, **info)
    _json(OUT / "STAGE4B_FOLD_PREPROCESSING.json", prep_meta)
    (OUT / "STAGE4B_H_EXPERIMENT_REQUIREMENTS.md").write_text(h_requirements(leader))

    summary = dict(data=data_check, leader=leader, verdicts=verdicts, bio15=bio_summary, corr=corr, groups=grp.to_dict("records"), top=top.to_dict(),
                   stability=stability, spatial=spatial, software=SOFTWARE, summ=summ, res=res, abl=abl)
    (OUT / "STAGE4B_REPORT.md").write_text(report(summary))
    ml = [p for p in scan_ml_h_artifacts() if not p.startswith(("outputs/stage4b/",))]
    v1_after = v1_verify()
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith(("STAGE4B_CHECKPOINT", "STAGE4B_DIAGNOSTICS")))
    cp = dict(schema="SFV2_STAGE4B_CHECKPOINT_V1", as_of=AS_OF, status="DEVELOPMENT_ONLY_NONFINAL", preregistration_sha256=prereg_sha,
              stage4a_checkpoint_sha256=STAGE4A_CHECKPOINT_SHA, data_check=data_check, software=SOFTWARE, leader=dict(model=lm, feature_set=lf),
              output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs},
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage4b.py", "stage4b.py")},
              final_model_frozen=False, lockbox_accessed=False, h_generated=False, h_used=False, unexpected_ml_or_h_outside_stage4b=ml,
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              volatile_diagnostics={"path": "outputs/stage4b/STAGE4B_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE4B_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE4B_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD")))
    return summary


def h_requirements(leader: dict) -> str:
    return f"""# Stage 4B -> controlled MEG vs MEGH experiment: requirements (H not built here)

1. Comparison arms: MEG vs MEGH, and MEG + pre-registered selected-H, on IDENTICAL landscapes, terrain units, outer LOLO folds, inner folds,
   sampling rule (all positives + {S.NEG_PER_LANDSCAPE} hash-ranked negatives), weighting (landscape-balanced), model family and tuning grid.
2. Model family for the primary H test: the DEVELOPMENT_LEADER family ({leader['model']}); the other two families are secondary.
3. H rows: `h_available` flag per row. Rows without H are excluded from BOTH arms of the MEGH comparison (never zero-filled, never imputed).
   Report how many rows and positives each landscape loses, and repeat MEG on the H-available subset as the matched comparator.
4. H reconstruction must be specified and frozen from geological evidence BEFORE any H-model result; Stage 4B performance must not shape how H is built.
5. Primary metric: per-landscape held-out PR-AUC, median across landscapes; report Delta_H = PR-AUC(MEGH) - PR-AUC(MEG) per landscape.
6. V1 lesson: H improved internal validation but harmed Kurikoma transfer, so H is a hypothesis that must earn its place in held-out landscapes.
7. Lockboxes stay closed; the final MEG/MEGH decision waits for development landscapes #7-#8 and the 8+3 design.
"""


def report(s: dict) -> str:
    summ = s["summ"]
    p = summ[summ.analysis == "PRIMARY"].sort_values(["model", "feature_set"])
    rows = "\n".join(f"| {r.model} | {r.feature_set} | {r.median_pr_auc:.4f} | {r.mean_pr_auc:.4f} | {r.min_pr_auc:.4f} | {r.max_pr_auc:.4f} | {r.iqr_pr_auc:.4f} | "
                     f"{r.median_pr_auc_lift:.2f} | {r.median_roc_auc:.3f} | {r.median_cal_slope:.2f} |" for r in p.itertuples())
    return f"""# Stage 4B report (DEVELOPMENT ONLY, NONFINAL)

Software: scikit-learn {s['software']['version']}.  Data: {s['data']}.
Leader: {s['leader']['model']} + {s['leader']['feature_set']} (median PR-AUC {s['leader']['median_pr_auc']:.4f}, worst {s['leader']['min_pr_auc']:.4f}).

| Model | Features | median PR-AUC | mean | min | max | IQR | median lift | median ROC-AUC | median cal slope |
|---|---|---|---|---|---|---|---|---|---|
{rows}

Sensitivity verdicts: {json.dumps({k: v.get('materially_changes_conclusion') for k, v in s['verdicts'].items() if isinstance(v, dict) and 'materially_changes_conclusion' in v})}
bio15: {json.dumps(s['bio15'])}
Domain shift Spearman (n=6, descriptive): {json.dumps(s['corr'])}
Feature groups (median permutation PR-AUC drop): {json.dumps(s['groups'])}
Top predictors: {json.dumps(s['top'])}; fold rank stability (mean Spearman): {s['stability']:.3f}
Spatial consistency (neighbour r): {json.dumps(s['spatial'])}
"""


if __name__ == "__main__":
    main()
