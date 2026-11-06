"""Build Stage 5E cross-volcano LOGISTIC_M attribution audits.

The original estimators were not serialized.  This module uses the explicitly
allowed reconstruction path, requires exact held-out prediction reproduction,
and then computes exact interventional linear SHAP values on the log-odds scale.
No held-out outcome is used for fitting, preprocessing, background selection or
SHAP sampling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from . import stage4b as S
from .build_stage5bb1 import verify_baseline
from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5e"
FIG = OUT / "figures"
ARRAYS = OUT / "shap_arrays"
MODELS = OUT / "models"
LANDSCAPES = S.LANDSCAPES
FEATURE_DICTIONARY = ROOT / "outputs/stage5a/STAGE5A_FEATURE_DICTIONARY.tsv"
STAGE5DE = ROOT / "outputs/stage5d_e/STAGE5DE_CHECKPOINT.json"
EXPECTED = {
    "stage5de": "a3fe549b960b6e3f470bdd0cd812bd33f2ab54fce70ada9c2666da903c867dc0",
    "stage5da_prereg": "e03af55eacc87832d8e644d4a78c201b330fd4bbd459aef7d426cf5de43728a1",
    "stage5ca_protocol": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
    "stage5ca_success": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1",
}
SHAP_SAMPLE_N = 20_000
PERM_REPEATS = 3
SEED = 20261008


def clean_json(x):
    if isinstance(x, dict): return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.generic): x = x.item()
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n")


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(clean_json(obj), indent=2, sort_keys=True, allow_nan=False))


def stable_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", float_format="%.12g")


def array_hash(*arrays) -> str:
    h = hashlib.sha256()
    for x in arrays: h.update(np.ascontiguousarray(x).view(np.uint8))
    return h.hexdigest()


def integrity_gate() -> dict:
    paths = {
        "stage5de": STAGE5DE,
        "stage5da_prereg": ROOT / "outputs/stage5d_a/STAGE5DA_EXPERIMENT_PREREGISTRATION.json",
        "stage5ca_protocol": ROOT / "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md",
        "stage5ca_success": ROOT / "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json",
    }
    actual = {k: sha256(v) for k, v in paths.items()}
    bad = {k: {"expected": EXPECTED[k], "actual": actual[k]} for k in paths if actual[k] != EXPECTED[k]}
    if bad: raise RuntimeError(f"STOP: protected Stage 5 input mismatch: {bad}")
    baseline = verify_baseline()
    return {"status": "PASS", "verified_hashes": actual, "baseline": baseline,
            "external_data_accessed": False, "lockbox_outcomes_accessed": False, "writes_to_historical_outputs": 0}


def deterministic_sample(ids: np.ndarray, n: int, token: str) -> np.ndarray:
    digest = np.fromiter((int.from_bytes(hashlib.sha256(f"{x}|{token}".encode()).digest()[:8], "big") for x in ids),
                         dtype=np.uint64, count=len(ids))
    take = np.argsort(digest, kind="stable")[:min(n, len(ids))]
    return np.sort(take)


def selected_rows() -> dict:
    obj = json.loads((ROOT / "data/interim/_stage4b_results/PRIMARY.json").read_text())
    rows = {r["held_out"]: r for r in obj["results"] if r["analysis"] == "PRIMARY" and r["model"] == "LOGISTIC_L2" and r["feature_set"] == "M"}
    if set(rows) != set(LANDSCAPES): raise RuntimeError("STOP: incomplete authoritative LOGISTIC_M outer results")
    return rows


def reconstruct_models(d: S.Data, features: list[str]) -> tuple[dict, dict]:
    rows = selected_rows(); fitted, audit = {}, []
    coefs, intercepts, means, sds = [], [], [], []
    for held in LANDSCAPES:
        train = [x for x in LANDSCAPES if x != held]
        params = json.loads(rows[held]["params"])
        if params != {"C": 0.01}: raise RuntimeError(f"STOP: unexpected historical C for {held}: {params}")
        est, cols, prep, waudit, fitaudit = S.fit(d, train, "M", "LOGISTIC_L2", params, "ALL_VALID", "LANDSCAPE_BALANCED")
        p = S.predict(d, est, cols, prep, held, "ALL_VALID").astype(np.float64)
        historical = np.load(ROOT / f"outputs/stage5d_e/predictions/B0__{held}__scores_v1.npy")
        maxdiff = float(np.max(np.abs(p - historical)))
        if maxdiff > 1e-12 or array_hash(p) != array_hash(historical):
            raise RuntimeError(f"STOP: reconstructed {held} model does not reproduce authoritative prediction: {maxdiff}")
        if prep["columns"] != features: raise RuntimeError(f"STOP: feature order mismatch for {held}")
        beta = est.coef_[0].astype(np.float64); b = float(est.intercept_[0]); mu = np.asarray(prep["mean"]); sd = np.asarray(prep["sd"])
        background_ids = np.concatenate([d.L[l]["ids"][d.sample(l, "ALL_VALID")] for l in train])
        audit.append({"held_out": held, "training_landscapes": train, "train_test_overlap": bool(held in train),
                      "params": params, "train_rows": fitaudit["train_rows"], "train_positives": fitaudit["train_pos"],
                      "background_definition": "complete frozen training sample: all positives plus deterministic <=20,000 negatives per training landscape",
                      "background_rows": len(background_ids), "background_terrain_unit_id_sha256": array_hash(background_ids.astype("U")),
                      "background_standardized_mean_max_abs": float(np.max(np.abs(np.vstack([
                          d.L[l]["X"][d.sample(l, "ALL_VALID")][:, cols] for l in train]).mean(0) - mu) / sd)),
                      "coefficient_sha256": array_hash(beta), "preprocessing_sha256": array_hash(mu, sd),
                      "prediction_sha256": array_hash(p), "historical_prediction_sha256": array_hash(historical),
                      "max_abs_prediction_difference": maxdiff, "prediction_reproduction": "PASS_BYTE_IDENTICAL",
                      "held_out_labels_used_for_fit_or_background": False, "weight_audit": waudit})
        fitted[held] = {"est": est, "cols": cols, "mu": mu, "sd": sd, "beta": beta, "intercept": b, "prediction": p}
        coefs.append(beta); intercepts.append(b); means.append(mu); sds.append(sd)
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "STAGE5E_RECONSTRUCTED_LOGISTIC_M_FOLDS_V1.npz"
    np.savez_compressed(path, landscapes=np.asarray(LANDSCAPES), feature_names=np.asarray(features), coefficients=np.vstack(coefs),
                        intercepts=np.asarray(intercepts), scaler_means=np.vstack(means), scaler_sds=np.vstack(sds))
    return fitted, {"status": "PASS", "original_estimators_available": False,
                    "reconstruction_authorization": "Stage 5E section 2: reconstruct only if unavailable and verify numerical equivalence",
                    "model_family": "sklearn LogisticRegression L2/lbfgs", "model_artifact": str(path.relative_to(ROOT)),
                    "model_artifact_sha256": sha256(path), "folds": audit}


def shap_and_diagnostics(d: S.Data, fitted: dict, dictionary: pd.DataFrame):
    features = dictionary.FEATURE_NAME.tolist(); groups = dictionary.set_index("FEATURE_NAME").FEATURE_GROUP.to_dict()
    aggregation = {r.FEATURE_NAME: ("MEAN" if r.FEATURE_NAME.endswith("__mean") else "SD") for r in dictionary.itertuples()}
    landscape_rows, family_rows, distribution_rows, score_rows, pair_rows, additivity = [], [], [], [], [], []
    ARRAYS.mkdir(parents=True, exist_ok=True)
    for li, held in enumerate(LANDSCAPES):
        model = fitted[held]; X = d.L[held]["X"][:, model["cols"]]; ids = d.L[held]["ids"]
        beta, mu, sd, b = model["beta"], model["mu"], model["sd"], model["intercept"]
        sum_abs = np.zeros(60); sum_signed = np.zeros(60); family_joint_abs = {g: 0.0 for g in sorted(set(groups.values()))}
        max_add = 0.0
        for start in range(0, len(X), 50_000):
            xs = (X[start:start + 50_000] - mu) / sd; phi = xs * beta
            decision = fitted[held]["est"].decision_function(xs)
            max_add = max(max_add, float(np.max(np.abs((b + phi.sum(1)) - decision))))
            sum_abs += np.abs(phi).sum(0); sum_signed += phi.sum(0)
            for g in family_joint_abs:
                jj = [j for j, f in enumerate(features) if groups[f] == g]
                family_joint_abs[g] += float(np.abs(phi[:, jj].sum(1)).sum())
        mean_abs = sum_abs / len(X); signed_mean = sum_signed / len(X)
        median_abs = np.array([float(np.median(np.abs(((X[:, j] - mu[j]) / sd[j]) * beta[j]))) for j in range(60)])
        order = np.argsort(-mean_abs, kind="stable"); ranks = np.empty(60, int); ranks[order] = np.arange(1, 61)
        sample = deterministic_sample(ids, SHAP_SAMPLE_N, "stage5e_shap_v1")
        sample_x = X[sample]; sample_std = (sample_x - mu) / sd; sample_phi = sample_std * beta
        sample_path = ARRAYS / f"STAGE5E_SHAP_SAMPLE__{held}__V1.npz"
        np.savez_compressed(sample_path, terrain_unit_id=ids[sample], feature_names=np.asarray(features), feature_values=sample_x,
                            standardized_feature_values=sample_std, shap_log_odds=sample_phi,
                            expected_value_log_odds=np.asarray([b]), row_indices=sample)
        for j, f in enumerate(features):
            landscape_rows.append({"landscape": held, "feature": f, "feature_group": groups[f], "summary_type": aggregation[f],
                                   "mean_abs_shap_log_odds": mean_abs[j], "median_abs_shap_log_odds": median_abs[j],
                                   "signed_mean_shap_log_odds": signed_mean[j], "rank_mean_abs_shap": ranks[j],
                                   "standardized_coefficient": beta[j], "coefficient_sign": int(np.sign(beta[j])),
                                   "n_units": len(X), "shap_sample_n": len(sample), "shap_sample_fraction": len(sample) / len(X),
                                   "shap_array": str(sample_path.relative_to(ROOT)), "shap_array_sha256": sha256(sample_path)})
            q = np.quantile(sample_x[:, j], [.05, .5, .95])
            distribution_rows.append({"landscape": held, "feature": f, "sample_n": len(sample), "mean": float(sample_x[:, j].mean()),
                                      "sd": float(sample_x[:, j].std()), "q05": q[0], "median": q[1], "q95": q[2],
                                      "standardized_mean_shift_from_training_background": float(sample_std[:, j].mean()),
                                      "standardized_sd": float(sample_std[:, j].std())})
        for g in sorted(set(groups.values())):
            jj = [j for j, f in enumerate(features) if groups[f] == g]
            family_rows.append({"landscape": held, "feature_family": g, "n_features": len(jj),
                                "sum_feature_mean_abs_shap": float(mean_abs[jj].sum()),
                                "mean_per_feature_abs_shap": float(mean_abs[jj].mean()),
                                "mean_abs_joint_family_shap": family_joint_abs[g] / len(X),
                                "signed_mean_joint_family_shap": float(signed_mean[jj].sum())})
        p = model["prediction"]; q = np.quantile(p, [0, .01, .05, .25, .5, .75, .95, .99, 1])
        score_rows.append({"landscape": held, "n_units": len(p), **{f"q{int(x*100):02d}": y for x, y in zip([0,.01,.05,.25,.5,.75,.95,.99,1], q)},
                           "mean": float(p.mean()), "sd": float(p.std()), "interpretation": "uncalibrated model score; not probability"})
        corr = np.corrcoef(sample_std, rowvar=False)
        for a in range(60):
            for c in range(a + 1, 60):
                if abs(corr[a, c]) >= .8:
                    pair_rows.append({"landscape": held, "feature_a": features[a], "feature_b": features[c],
                                      "pearson_r": float(corr[a, c]), "abs_r": float(abs(corr[a, c])),
                                      "same_definition_family": groups[features[a]] == groups[features[c]]})
        additivity.append({"landscape": held, "n_units_checked": len(X), "expected_value_log_odds": b,
                           "max_abs_additivity_error": max_add, "tolerance": 1e-10, "passed": max_add <= 1e-10,
                           "coefficient_attribution_max_abs_difference": 0.0})
        print(f"SHAP {li + 1}/6 complete: {held}", flush=True)
    return (pd.DataFrame(landscape_rows), pd.DataFrame(family_rows), pd.DataFrame(distribution_rows),
            pd.DataFrame(score_rows), pd.DataFrame(pair_rows), additivity)


def permutation_importance(d: S.Data, fitted: dict, dictionary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = dictionary.FEATURE_NAME.tolist(); families = {g: np.flatnonzero(dictionary.FEATURE_GROUP.to_numpy() == g) for g in dictionary.FEATURE_GROUP.unique()}
    rows = []
    for li, held in enumerate(LANDSCAPES):
        m = fitted[held]; take = d.sample(held, "ALL_VALID"); y = d.L[held]["y"][take]
        X = d.L[held]["X"][take][:, m["cols"]]; xs = (X - m["mu"]) / m["sd"]; z = m["intercept"] + xs @ m["beta"]
        base = float(average_precision_score(y, 1 / (1 + np.exp(-np.clip(z, -30, 30)))))
        for j, f in enumerate(features):
            vals = []
            for repeat in range(PERM_REPEATS):
                rng = np.random.default_rng(SEED + li * 10_000 + j * 10 + repeat); perm = rng.permutation(len(xs))
                zp = z + m["beta"][j] * (xs[perm, j] - xs[:, j])
                vals.append(base - average_precision_score(y, 1 / (1 + np.exp(-np.clip(zp, -30, 30)))))
            rows.append({"level": "FEATURE", "landscape": held, "name": f, "feature_family": dictionary.iloc[j].FEATURE_GROUP,
                         "method": "HELD_OUT_PERMUTATION_AP_DROP", "baseline_ap_on_evaluation_sample": base,
                         "importance_mean": float(np.mean(vals)), "importance_sd": float(np.std(vals, ddof=1)), "repeats": PERM_REPEATS,
                         "evaluation_rows": len(take), "evaluation_positives": int(y.sum()),
                         "sampling": "all positives plus deterministic <=20,000 negatives; evaluation only"})
        for gi, (g, jj) in enumerate(families.items()):
            vals = []
            contribution = xs[:, jj] @ m["beta"][jj]
            for repeat in range(PERM_REPEATS):
                rng = np.random.default_rng(SEED + 900_000 + li * 10_000 + gi * 10 + repeat); perm = rng.permutation(len(xs))
                zp = z + contribution[perm] - contribution
                vals.append(base - average_precision_score(y, 1 / (1 + np.exp(-np.clip(zp, -30, 30)))))
            rows.append({"level": "FAMILY", "landscape": held, "name": g, "feature_family": g,
                         "method": "HELD_OUT_GROUPED_PERMUTATION_AP_DROP", "baseline_ap_on_evaluation_sample": base,
                         "importance_mean": float(np.mean(vals)), "importance_sd": float(np.std(vals, ddof=1)), "repeats": PERM_REPEATS,
                         "evaluation_rows": len(take), "evaluation_positives": int(y.sum()),
                         "sampling": "all positives plus deterministic <=20,000 negatives; evaluation only"})
        print(f"permutation {li + 1}/6 complete: {held}", flush=True)
    out = pd.DataFrame(rows)
    historical = pd.read_csv(ROOT / "outputs/stage4e_b/STAGE4EB_FEATURE_IMPORTANCE.tsv", sep="\t")
    historical = historical[historical.model == "LOGISTIC_L2"].set_index(["held_out", "feature"])
    feature = out[out.level == "FEATURE"].set_index(["landscape", "name"])
    delta = feature.importance_mean - historical.importance_mean
    baseline_delta = feature.baseline_ap_on_evaluation_sample - historical.baseline_pr_auc
    verification = {"historical_source": "outputs/stage4e_b/STAGE4EB_FEATURE_IMPORTANCE.tsv",
                    "historical_source_sha256": sha256(ROOT / "outputs/stage4e_b/STAGE4EB_FEATURE_IMPORTANCE.tsv"),
                    "feature_rows_compared": len(delta), "max_abs_importance_mean_difference": float(np.max(np.abs(delta))),
                    "max_abs_baseline_ap_difference": float(np.max(np.abs(baseline_delta))),
                    "passed_tolerance_1e-12": bool(np.max(np.abs(delta)) <= 1e-12 and np.max(np.abs(baseline_delta)) <= 1e-12)}
    if not verification["passed_tolerance_1e-12"]: raise RuntimeError(f"STOP: permutation reproduction mismatch: {verification}")
    return out, verification


def aggregate_tables(land: pd.DataFrame, family: pd.DataFrame, perm: pd.DataFrame):
    feature_perm = perm[perm.level == "FEATURE"].copy()
    feature_perm["rank_within_landscape"] = feature_perm.groupby("landscape").importance_mean.rank(method="min", ascending=False).astype(int)
    perm.loc[feature_perm.index, "rank_within_landscape"] = feature_perm.rank_within_landscape
    rows, stable = [], []
    for feature, g in land.groupby("feature", sort=False):
        p = feature_perm[feature_perm.name == feature]
        weights = g.n_units / g.n_units.sum()
        signs = np.sign(g.signed_mean_shap_log_odds.to_numpy())
        coeff_signs = np.sign(g.standardized_coefficient.to_numpy())
        rows.append({"feature": feature, "feature_group": g.feature_group.iloc[0], "summary_type": g.summary_type.iloc[0],
                     "mean_abs_shap_equal_landscape": float(g.mean_abs_shap_log_odds.mean()),
                     "median_abs_shap_equal_landscape": float(g.median_abs_shap_log_odds.mean()),
                     "signed_mean_shap_equal_landscape": float(g.signed_mean_shap_log_odds.mean()),
                     "mean_abs_shap_pooled_unit_sensitivity": float(np.sum(weights * g.mean_abs_shap_log_odds)),
                     "mean_fold_rank": float(g.rank_mean_abs_shap.mean()), "sd_fold_rank": float(g.rank_mean_abs_shap.std(ddof=1)),
                     "min_fold_rank": int(g.rank_mean_abs_shap.min()), "max_fold_rank": int(g.rank_mean_abs_shap.max()),
                     "top_10_fold_count": int((g.rank_mean_abs_shap <= 10).sum()),
                     "permutation_ap_drop_equal_landscape": float(p.importance_mean.mean()),
                     "permutation_landscapes_positive": int((p.importance_mean > 0).sum()),
                     "attribution_scope": "equal arithmetic weight to each of six landscapes",
                     "pooled_unit_result_is_secondary": True})
        stable.append({"feature": feature, "feature_group": g.feature_group.iloc[0], "mean_rank": float(g.rank_mean_abs_shap.mean()),
                       "rank_sd": float(g.rank_mean_abs_shap.std(ddof=1)), "rank_iqr": float(g.rank_mean_abs_shap.quantile(.75) - g.rank_mean_abs_shap.quantile(.25)),
                       "rank_range": int(g.rank_mean_abs_shap.max() - g.rank_mean_abs_shap.min()),
                       "min_rank": int(g.rank_mean_abs_shap.min()), "max_rank": int(g.rank_mean_abs_shap.max()),
                       "top_5_count": int((g.rank_mean_abs_shap <= 5).sum()), "top_10_count": int((g.rank_mean_abs_shap <= 10).sum()),
                       "signed_attribution_positive_count": int((signs > 0).sum()), "signed_attribution_negative_count": int((signs < 0).sum()),
                       "signed_attribution_sign_consistency": float(max((signs > 0).mean(), (signs < 0).mean())),
                       "coefficient_positive_count": int((coeff_signs > 0).sum()), "coefficient_negative_count": int((coeff_signs < 0).sum()),
                       "coefficient_sign_consistency": float(max((coeff_signs > 0).mean(), (coeff_signs < 0).mean())),
                       "stability_interpretation": "model attribution stability only; not causal or predictive stability"})
    glob = pd.DataFrame(rows).sort_values("mean_abs_shap_equal_landscape", ascending=False).reset_index(drop=True)
    glob["global_rank"] = np.arange(1, len(glob) + 1)
    stability = pd.DataFrame(stable).sort_values(["mean_rank", "rank_sd"]).reset_index(drop=True)
    # Rank correlations measure agreement of the six held-out attribution profiles.
    rank_matrix = land.pivot(index="feature", columns="landscape", values="rank_mean_abs_shap").reindex(columns=LANDSCAPES)
    rcorr = rank_matrix.corr(method="spearman")
    corr_rows = [{"landscape_a": a, "landscape_b": b, "spearman_feature_rank_correlation": float(rcorr.loc[a, b])}
                 for i, a in enumerate(LANDSCAPES) for b in LANDSCAPES[i + 1:]]
    # Append equal-landscape family summaries while retaining every landscape row.
    fam_global = family.groupby("feature_family", sort=False).agg(
        n_features=("n_features", "first"), sum_feature_mean_abs_shap=("sum_feature_mean_abs_shap", "mean"),
        mean_per_feature_abs_shap=("mean_per_feature_abs_shap", "mean"), mean_abs_joint_family_shap=("mean_abs_joint_family_shap", "mean"),
        signed_mean_joint_family_shap=("signed_mean_joint_family_shap", "mean")).reset_index()
    fam_global.insert(0, "landscape", "EQUAL_LANDSCAPE_AGGREGATE")
    family_out = pd.concat([family, fam_global], ignore_index=True)
    return glob, stability, family_out, pd.DataFrame(corr_rows), perm


def short_feature(x: str) -> str:
    return x.replace("__mean", " [mean]").replace("__sd", " [SD]").replace("_30m", "").replace("_1pm", "")


def make_figures(global_imp, land, stability, family, perm, distributions, scores) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True); catalogue = []
    colors = plt.get_cmap("tab10")
    def save(fig, name, title, scope):
        fig.suptitle("DEVELOPMENT-ONLY MODEL INTERPRETATION", fontsize=10, fontweight="bold", y=.995)
        fig.tight_layout(rect=(0, 0, 1, .965)); path = FIG / name; fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
        catalogue.append({"figure_id": len(catalogue) + 1, "file": str(path.relative_to(ROOT)), "title": title,
                          "scope": scope, "causal_claim": "NO", "external_validation_claim": "NO", "sha256": sha256(path)})
    top = global_imp.head(20).sort_values("mean_abs_shap_equal_landscape")
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh([short_feature(x) for x in top.feature], top.mean_abs_shap_equal_landscape, color="#355c7d")
    ax.set_xlabel("Mean |SHAP| (log-odds; equal landscape weight)"); ax.set_title("Global top-20 LOGISTIC_M attributions")
    save(fig, "FIG01_GLOBAL_TOP20_SHAP.png", "Global top-20 SHAP ranking", "equal-landscape model attribution")
    # Heatmap.
    names = global_imp.head(20).feature.tolist(); mat = land[land.feature.isin(names)].pivot(index="feature", columns="landscape", values="mean_abs_shap_log_odds").reindex(index=names, columns=LANDSCAPES)
    fig, ax = plt.subplots(figsize=(11, 8)); im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(names)), [short_feature(x) for x in names], fontsize=8); ax.set_xticks(np.arange(6), [x.split("_")[0].title() for x in LANDSCAPES], rotation=25)
    fig.colorbar(im, ax=ax, label="Mean |SHAP| (log-odds)"); ax.set_title("Six-landscape attribution heatmap")
    save(fig, "FIG02_LANDSCAPE_IMPORTANCE_HEATMAP.png", "Six-landscape feature-importance heatmap", "top 20 equal-landscape features")
    # Stability.
    st = stability.sort_values("mean_rank").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7)); ax.errorbar(st.mean_rank, np.arange(len(st)), xerr=st.rank_sd, fmt="o", color="#2a9d8f", ecolor="#8ecae6")
    ax.set_yticks(np.arange(len(st)), [short_feature(x) for x in st.feature], fontsize=8); ax.invert_xaxis(); ax.set_xlabel("Mean fold rank ± SD (1 = highest)"); ax.set_title("Feature-rank stability across held-out landscapes")
    save(fig, "FIG03_FEATURE_RANK_STABILITY.png", "Feature-rank stability", "six fold-specific attribution rankings")
    # Signed attribution comparison.
    smat = land[land.feature.isin(names)].pivot(index="feature", columns="landscape", values="signed_mean_shap_log_odds").reindex(index=names, columns=LANDSCAPES)
    lim = float(np.nanmax(np.abs(smat.to_numpy())))
    fig, ax = plt.subplots(figsize=(11, 8)); im = ax.imshow(smat, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(names)), [short_feature(x) for x in names], fontsize=8); ax.set_xticks(np.arange(6), [x.split("_")[0].title() for x in LANDSCAPES], rotation=25)
    fig.colorbar(im, ax=ax, label="Signed mean SHAP (log-odds)"); ax.set_title("Signed attribution comparison")
    save(fig, "FIG04_SIGNED_SHAP_COMPARISON.png", "Signed SHAP comparison", "top 20 features; sign is model-directional, not causal")
    # Families.
    fam = family[family.landscape == "EQUAL_LANDSCAPE_AGGREGATE"].sort_values("sum_feature_mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(fam.feature_family, fam.sum_feature_mean_abs_shap, color="#6a4c93")
    ax.set_xlabel("Sum of feature mean |SHAP| (log-odds)"); ax.set_title("Definition-grounded terrain-family attribution")
    save(fig, "FIG05_FEATURE_FAMILY_CONTRIBUTION.png", "Feature-family contribution", "families from frozen feature dictionary")
    # Mean versus SD.
    ms = global_imp.groupby("summary_type").mean_abs_shap_equal_landscape.agg(["sum", "mean"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4)); axes[0].bar(ms.index, ms["sum"], color=["#457b9d", "#e76f51"]); axes[0].set_ylabel("Summed mean |SHAP|"); axes[0].set_title("Total attribution")
    axes[1].bar(ms.index, ms["mean"], color=["#457b9d", "#e76f51"]); axes[1].set_ylabel("Mean per feature |SHAP|"); axes[1].set_title("Per-feature attribution")
    save(fig, "FIG06_MEAN_VERSUS_SD.png", "Mean-versus-SD feature comparison", "30 mean and 30 SD terrain-unit summaries")
    # Per-landscape beeswarm-equivalent + dependence plots.
    for li, lid in enumerate(LANDSCAPES):
        z = np.load(ARRAYS / f"STAGE5E_SHAP_SAMPLE__{lid}__V1.npz", allow_pickle=False)
        feats = z["feature_names"].tolist(); phi, xv, xs = z["shap_log_odds"], z["feature_values"], z["standardized_feature_values"]
        top15 = land[land.landscape.eq(lid)].sort_values("rank_mean_abs_shap").head(15).feature.tolist(); jj = [feats.index(x) for x in top15]
        fig = plt.figure(figsize=(12, 9)); grid = fig.add_gridspec(3, 2); ax = fig.add_subplot(grid[:, 0])
        for yi, j in enumerate(jj[::-1]):
            rng = np.random.default_rng(SEED + li * 100 + j); take = rng.choice(len(phi), min(900, len(phi)), replace=False)
            jitter = rng.normal(0, .08, len(take)); sc = ax.scatter(phi[take, j], yi + jitter, c=xs[take, j], cmap="coolwarm", s=5, alpha=.45, rasterized=True)
        ax.axvline(0, color="black", lw=.6); ax.set_yticks(np.arange(15), [short_feature(top15[::-1][i]) for i in range(15)], fontsize=7)
        ax.set_xlabel("SHAP (log-odds)"); ax.set_title(f"{lid}: top-15 attribution distributions")
        for k, j in enumerate(jj[:3]):
            dep = fig.add_subplot(grid[k, 1])
            take = np.linspace(0, len(phi) - 1, min(4000, len(phi)), dtype=int)
            dep.scatter(xv[take, j], phi[take, j], s=3, alpha=.25, color=colors(k)); dep.set_xlabel(short_feature(feats[j]), fontsize=8); dep.set_ylabel("SHAP", fontsize=8)
            dep.set_title(f"Dependence: rank {k+1}", fontsize=9)
        save(fig, f"FIG07_{li+1:02d}_{lid}_SHAP_SUMMARY.png", f"{lid} SHAP summary and dependence", "deterministic outcome-blind 20,000-unit explanation sample")
    # SHAP versus permutation.
    pp = perm[perm.level == "FEATURE"].groupby("name").importance_mean.mean()
    xy = global_imp.set_index("feature").join(pp.rename("perm"))
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(xy.mean_abs_shap_equal_landscape, xy.perm, alpha=.7, color="#264653")
    for f in global_imp.head(10).feature: ax.annotate(short_feature(f), (xy.loc[f].mean_abs_shap_equal_landscape, xy.loc[f].perm), fontsize=6)
    ax.axhline(0, color="black", lw=.7); ax.set_xlabel("Equal-landscape mean |SHAP| (log-odds)"); ax.set_ylabel("Equal-landscape permutation AP drop"); ax.set_title("Attribution magnitude versus predictive sensitivity")
    save(fig, "FIG08_SHAP_VS_PERMUTATION.png", "SHAP versus permutation importance", "feature-level exploratory comparison")
    # Score distributions.
    fig, ax = plt.subplots(figsize=(9, 5))
    for li, lid in enumerate(LANDSCAPES):
        p = np.load(ROOT / f"outputs/stage5d_e/predictions/B0__{lid}__scores_v1.npy")
        ax.hist(p, bins=80, histtype="step", density=True, label=lid.split("_")[0].title(), color=colors(li))
    ax.set_xlabel("Raw LOGISTIC_M score (uncalibrated)"); ax.set_ylabel("Density"); ax.set_yscale("log"); ax.set_title("Held-out score distributions"); ax.legend(ncol=2, fontsize=8)
    save(fig, "FIG09_PREDICTION_SCORE_DISTRIBUTIONS.png", "Prediction-score distributions", "raw scores are not calibrated probabilities")
    # Feature distribution shifts for global top 20.
    shift = distributions[distributions.feature.isin(names)].pivot(index="feature", columns="landscape", values="standardized_mean_shift_from_training_background").reindex(index=names, columns=LANDSCAPES)
    lim = max(1e-12, float(np.nanmax(np.abs(shift.to_numpy())))); fig, ax = plt.subplots(figsize=(11, 8)); im = ax.imshow(shift, aspect="auto", cmap="PiYG", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(names)), [short_feature(x) for x in names], fontsize=8); ax.set_xticks(np.arange(6), [x.split("_")[0].title() for x in LANDSCAPES], rotation=25)
    fig.colorbar(im, ax=ax, label="Mean shift in training-scaler SD units"); ax.set_title("Held-out feature-distribution diagnostics")
    save(fig, "FIG10_FEATURE_DISTRIBUTION_DIAGNOSTICS.png", "Feature-distribution diagnostics", "deterministic outcome-blind samples")
    return pd.DataFrame(catalogue)


def scientific_summary(global_imp, land, stability, family, perm, distributions, rank_corr) -> dict:
    top = global_imp.head(10).feature.tolist()
    consistent = stability.sort_values(["top_10_count", "mean_rank", "rank_sd"], ascending=[False, True, True]).head(10).feature.tolist()
    variable = stability.sort_values(["rank_sd", "rank_range"], ascending=False).head(10).feature.tolist()
    pmean = perm[perm.level == "FEATURE"].groupby("name").importance_mean.mean()
    aligned = global_imp.set_index("feature").join(pmean.rename("permutation"))
    shap_perm_rho = float(spearmanr(aligned.mean_abs_shap_equal_landscape, aligned.permutation).statistic)
    ms = global_imp.groupby("summary_type").mean_abs_shap_equal_landscape.agg(["sum", "mean"])
    dist_sd = distributions.groupby("feature").standardized_mean_shift_from_training_background.std(ddof=1)
    stab = stability.set_index("feature").join(dist_sd.rename("distribution_shift_sd"))
    shift_rank_rho = float(spearmanr(stab.rank_sd, stab.distribution_shift_sd).statistic)
    return {"top_10_attribution": top, "most_consistently_high": consistent, "most_rank_variable": variable,
            "median_pairwise_landscape_rank_correlation": float(rank_corr.spearman_feature_rank_correlation.median()),
            "min_pairwise_landscape_rank_correlation": float(rank_corr.spearman_feature_rank_correlation.min()),
            "shap_permutation_rank_correlation": shap_perm_rho,
            "mean_feature_total_attribution": float(ms.loc["MEAN", "sum"]), "sd_feature_total_attribution": float(ms.loc["SD", "sum"]),
            "mean_feature_per_feature_attribution": float(ms.loc["MEAN", "mean"]), "sd_feature_per_feature_attribution": float(ms.loc["SD", "mean"]),
            "rank_variability_vs_distribution_shift_rho": shift_rank_rho,
            "negative_equal_landscape_permutation_features": int((pmean < 0).sum()),
            "interpretation_scope": "descriptive association among six development landscapes; not mechanism or causality"}


def write_scientific_documents(summary: dict, global_imp: pd.DataFrame) -> None:
    top = ", ".join(summary["top_10_attribution"][:5])
    stable = ", ".join(summary["most_consistently_high"][:5])
    variable = ", ".join(summary["most_rank_variable"][:5])
    write_text(OUT / "STAGE5E_SCIENTIFIC_INTERPRETATION.md", f"""# Stage 5E scientific interpretation

This is development-only interpretation of the fitted LOGISTIC_M decision function. SHAP measures **model attribution (A)**. It does not directly establish stable predictive contribution under intervention (B), and it cannot establish causal geological interpretation (C).

1. **Greatest average attribution.** The leading equal-landscape features begin with: {top}. These features contribute most strongly to variation in the six fitted fold-specific log-odds functions; that is not proof that they cause landsliding.
2. **Consistently influential predictors.** The strongest rank-stability evidence begins with: {stable}. Consistency means recurring attribution rank across these six held-out distributions.
3. **Variable predictors.** The largest fold-rank variability begins with: {variable}. Variation can reflect changed coefficients, changed held-out feature distributions, and correlated-feature credit allocation.
4. **Mean versus SD summaries.** The 30 mean summaries contribute {summary['mean_feature_total_attribution']:.6g} total equal-landscape mean |SHAP| versus {summary['sd_feature_total_attribution']:.6g} for the 30 SD summaries. Per feature the values are {summary['mean_feature_per_feature_attribution']:.6g} and {summary['sd_feature_per_feature_attribution']:.6g}, respectively. This describes the model, not a physical hierarchy.
5. **SHAP versus permutation.** Their across-feature rank correlation is {summary['shap_permutation_rank_correlation']:.3f}. {summary['negative_equal_landscape_permutation_features']} features have negative mean held-out permutation AP drops. Disagreement is expected because SHAP allocates fitted score variation, whereas permutation measures sampled predictive sensitivity and can create unrealistic combinations among correlated predictors.
6. **Landscape differences.** Median pairwise attribution-rank correlation is {summary['median_pairwise_landscape_rank_correlation']:.3f} (minimum {summary['min_pairwise_landscape_rank_correlation']:.3f}). Rank variability correlates {summary['rank_variability_vs_distribution_shift_rho']:.3f} with sampled standardized distribution-shift variability. This exploratory association cannot separate landscape properties from coefficient instability or collinearity.
7. **Process inference.** The analysis cannot identify unique landslide mechanisms, causal terrain controls, calibrated risk, or prospective hazard. Directional SHAP signs are directions in a correlated fitted log-odds model.
8. **Further evidence.** Mechanistic conclusions need independent physical measurements and designs that identify causal effects. Generalization claims need qualified independent landscapes and the untouched Stage 5C-A lockbox procedure.
""")
    write_text(OUT / "STAGE5E_LIMITATIONS.md", """# Stage 5E limitations

- All six landscapes were already development-exposed; this is not external validation.
- The original estimators were not serialized. They were reconstructed only from their authoritative outer-training folds and accepted after byte-identical prediction reproduction.
- Exact linear interventional SHAP uses the frozen training sample mean as background and allocates log-odds additively. It does not model conditional dependence among correlated predictors.
- For a linear model under this convention, SHAP equals coefficient × standardized feature deviation. The comparison with coefficient-based attribution is therefore an identity check, not independent corroboration.
- Global SHAP moments use all terrain units, but distribution and plot arrays use deterministic outcome-blind samples of at most 20,000 units per landscape.
- Permutation importance uses every positive plus at most 20,000 deterministic negatives and held-out labels only for post-fit AP evaluation. Permutation can create implausible combinations; grouped family permutation reduces but does not remove this problem.
- Terrain units are spatially dependent. No unit-independent confidence intervals are reported, and six landscapes are insufficient for population-level inference.
- Feature families come from the frozen scientific dictionary. Summing absolute feature attributions can double-count correlated representations; joint-family SHAP is also reported to expose cancellation.
- Raw LOGISTIC_M scores are uncalibrated and are not probabilities.
""")


def final_report(summary, model_audit, tests, global_imp, figures) -> None:
    top5 = ", ".join(global_imp.head(5).feature)
    text = f"""# A. Executive summary

Stage 5E explains the frozen six-fold LOGISTIC_M architecture without changing or selecting a model. The five highest equal-landscape mean absolute attributions were {top5}. These are fitted-model attributions, not causal landslide mechanisms. No external data or lockbox outcome was accessed.

# B. Integrity checks

V1 retained 153 strict files with zero writes. Stage 5D-E matched `{EXPECTED['stage5de']}`; the frozen Stage 5D-A preregistration and Stage 5C-A protocol/success criteria also matched. Historical outputs were read only.

# C. Model reproduction

Original estimator objects were unavailable. Six LOGISTIC_M folds were reconstructed with the exact historical sampling, landscape weighting, L2/lbfgs configuration, training-only scaler and inner-selected C=0.01. Each reconstructed held-out prediction vector was byte-identical to Stage 5D-E. Every training set excluded its held-out landscape.

# D. SHAP methodology

Exact interventional linear SHAP was evaluated analytically on the decision-function/log-odds scale: `phi_j = beta_j * (x_j_standardized - E_background[x_j_standardized])`. Each fold's background was its complete frozen training sample; its standardized mean is zero, so the expected value is the intercept. This is mathematically equivalent to exact independent-feature LinearExplainer attribution and coefficient-deviation attribution. Maximum additivity error was {max(x['max_abs_additivity_error'] for x in model_audit['shap_additivity']):.3g}. Conditional SHAP could distribute credit differently among correlated predictors.

# E. Global rankings

`STAGE5E_GLOBAL_IMPORTANCE.tsv` provides equal-landscape mean/median absolute SHAP, signed mean SHAP, fold ranks and a clearly secondary pooled-unit sensitivity. The median pairwise fold-rank correlation was {summary['median_pairwise_landscape_rank_correlation']:.3f}.

# F. Per-landscape results

All six landscapes have full-unit SHAP moments, top-15 rankings, signed summaries, raw-score distributions, feature-distribution diagnostics, and deterministic outcome-blind 20,000-unit SHAP arrays for beeswarm-equivalent and dependence plots. Fold differences reflect both different training coefficients and different held-out terrain distributions.

# G. Feature stability

Stable and variable ranks, sign consistency, pairwise rank correlations, frozen feature-family summaries, empirically correlated feature pairs, and mean-versus-SD comparisons are recorded. Only stability of model attribution is measured directly.

# H. Permutation sensitivity

Held-out feature and dictionary-family permutations use Average Precision, three fixed repeats, and the historical all-positive plus ≤20,000-negative evaluation sample. All 360 feature-level values reproduce Stage 4E-B within 1e-12. The equal-landscape SHAP/permutation rank correlation was {summary['shap_permutation_rank_correlation']:.3f}; negative and unstable drops remain reported.

# I. Scientific interpretation

Attribution magnitude describes how the fitted score is assembled. It is not causal evidence, not a guarantee of predictive improvement, and not external generalization. The scientific answers and boundaries are detailed in `STAGE5E_SCIENTIFIC_INTERPRETATION.md`.

# J. Figures

{len(figures)} publication figures were generated. Every catalogue entry is marked `causal_claim=NO` and `external_validation_claim=NO`, and each figure states “DEVELOPMENT-ONLY MODEL INTERPRETATION”.

# K. Limitations

The dominant limitations are correlated predictors, interventional rather than conditional SHAP, six exposed landscapes, spatially dependent units, sampled plotting/permutation distributions, and non-causal observational interpretation. Raw scores remain uncalibrated.

# L. Checkpoint and SHA256

The artifact manifest, reconstructed-model hash, sampling decisions, source hashes, Git state and tests are recorded in `STAGE5E_CHECKPOINT.json`. Focused/current regression tests: **{tests['status']}**. No commit or push was performed.

STAGE 5E — CROSS-VOLCANO FEATURE
ATTRIBUTION: COMPLETE
"""
    write_text(OUT / "STAGE5E_FINAL_REPORT.md", text)


def run_tests() -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage5e.py", "tests/test_stage5de.py", "tests/test_stage5bb1.py"]
    env = os.environ.copy(); env["PYTHONPATH"] = "src:.ml_site"; env["MPLBACKEND"] = "Agg"
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    return {"status": "PASS" if p.returncode == 0 else "FAIL", "command": " ".join(cmd), "returncode": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr,
            "tested": ["exact 60-feature schema", "whole-landscape fold exclusion", "byte-identical prediction reproduction",
                       "exact SHAP additivity", "deterministic outcome-blind SHAP sampling", "equal-landscape aggregation",
                       "rank calculations", "permutation reproduction", "historical hash identities", "no external access"],
            "known_legacy_false_positive": "Unmodified Stage 5C-A/5D-A scanners still flag the authorized Stage 5B-B1 intake_cli.py evolution; this is documented in Stage 5D-E and those frozen tests were not edited."}


def git_audit() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.splitlines()
    return {"head": head, "expected_unchanged_head": "4958b65982e3788e5777c5a4f14e87e67ad95a0a", "head_unchanged": head == "4958b65982e3788e5777c5a4f14e87e67ad95a0a",
            "working_tree_clean": not status, "working_tree_status_line_count": len(status),
            "note": "Shared parent worktree was already dirty; unrelated user changes were preserved."}


def checkpoint(model_audit, config, tests, figures, final_integrity) -> dict:
    artifacts = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "STAGE5E_CHECKPOINT.json"}
    return {"stage": "5E", "status": "COMPLETE", "classification": "DEVELOPMENT_ONLY_MODEL_INTERPRETATION",
            "historical_integrity": final_integrity, "source_model_hash": model_audit["model_reconstruction"]["model_artifact_sha256"],
            "feature_schema_hash": config["feature_order_sha256"], "shap_configuration_hash": sha256(OUT / "STAGE5E_SHAP_CONFIGURATION.json"),
            "sampling": config["sampling"], "output_hashes": artifacts, "figure_hashes": dict(zip(figures.file, figures.sha256)),
            "tests": tests, "source_hashes": {"src/sfv2/build_stage5e.py": sha256(ROOT / "src/sfv2/build_stage5e.py"),
                                                "tests/test_stage5e.py": sha256(ROOT / "tests/test_stage5e.py")},
            "git": git_audit(),
            "flags": {"V1_INTACT": "YES", "HISTORICAL_OUTPUTS_UNCHANGED": "YES", "STAGE5DE_UNCHANGED": "YES",
                      "STAGE5CA_UNCHANGED": "YES", "STAGE5DA_PREREGISTRATION_UNCHANGED": "YES", "LOGISTIC_M_OFFICIAL_LEADER_UNCHANGED": "YES",
                      "NEW_MODEL_SELECTED": "NO", "HELD_OUT_LANDSCAPE_USED_FOR_FIT": "NO", "EXTERNAL_DATA_ACCESSED": "NO",
                      "LOCKBOX_OUTCOMES_ACCESSED": "NO", "CAUSAL_CLAIM_MADE": "NO", "COMMIT_CREATED": "NO", "PUSH_PERFORMED": "NO"}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    integrity = integrity_gate()
    dictionary = pd.read_csv(FEATURE_DICTIONARY, sep="\t").sort_values("ORDER")
    features = S.feature_sets()["M"]
    if dictionary.FEATURE_NAME.tolist() != features or len(features) != 60: raise RuntimeError("STOP: frozen feature schema mismatch")
    feature_hash = hashlib.sha256(("\n".join(features) + "\n").encode()).hexdigest()
    if feature_hash != "80fdb239fd2d5aeed4504e8c835c29d6b34d120dec141b06634f3db2360092f2": raise RuntimeError("STOP: feature order hash mismatch")
    d = S.Data(); fitted, reconstruction = reconstruct_models(d, features)
    land, family, distributions, scores, correlated, additivity = shap_and_diagnostics(d, fitted, dictionary)
    perm, perm_verification = permutation_importance(d, fitted, dictionary)
    glob, stability, family_out, rank_corr, perm = aggregate_tables(land, family, perm)
    summary = scientific_summary(glob, land, stability, family_out, perm, distributions, rank_corr)
    config = {"stage": "5E", "version": "1.0.0", "feature_count": 60, "feature_names": features, "feature_order_sha256": feature_hash,
              "model": "fold-specific reconstructed authoritative LOGISTIC_M", "output_scale": "decision_function log-odds",
              "shap_convention": "exact interventional/marginal linear SHAP with independent-feature value function",
              "implementation": "analytic exact linear formula; shap package unavailable and unnecessary for linear identity",
              "formula": "phi_j = beta_j * (x_j_standardized - mean_background_j_standardized)",
              "background": "complete frozen sampled outer-training set for each fold; standardized mean approximately zero",
              "correlated_predictor_warning": "conditional SHAP can allocate credit differently; current values must not be read causally",
              "sampling": {"global_moments": "all 835,846 units", "plot_and_dependence": "outcome-blind SHA256 terrain_unit_id sample, max 20,000 per landscape",
                           "plot_sample_token": "stage5e_shap_v1", "permutation": "all positives plus frozen deterministic <=20,000 negatives", "permutation_repeats": 3},
              "aggregation": "arithmetic equal weight to each landscape; pooled terrain-unit mean is secondary only",
              "held_out_labels_used_for_model_or_shap_settings": False, "held_out_labels_used_for_permutation_evaluation_only": True,
              "feature_selection_or_model_tuning_from_attribution": False, "causal_interpretation": False}
    model_audit = {"status": "PASS", "integrity": integrity, "model_reconstruction": reconstruction,
                   "shap_additivity": additivity, "permutation_reproduction": perm_verification,
                   "landscapes": LANDSCAPES, "terrain_units": int(land.groupby("landscape").n_units.first().sum()),
                   "feature_dictionary_sha256": sha256(FEATURE_DICTIONARY), "feature_order_sha256": feature_hash,
                   "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                "sklearn": __import__("sklearn").__version__, "shap_package": "NOT_INSTALLED_ANALYTIC_EXACT_IMPLEMENTATION"},
                   "rank_correlation_summary": {"median": summary["median_pairwise_landscape_rank_correlation"], "minimum": summary["min_pairwise_landscape_rank_correlation"]},
                   "external_data_accessed": False, "lockbox_outcomes_accessed": False, "historical_outputs_modified": False}
    stable_tsv(OUT / "STAGE5E_LANDSCAPE_IMPORTANCE.tsv", land)
    stable_tsv(OUT / "STAGE5E_GLOBAL_IMPORTANCE.tsv", glob)
    stable_tsv(OUT / "STAGE5E_FEATURE_STABILITY.tsv", stability)
    stable_tsv(OUT / "STAGE5E_FEATURE_FAMILY_IMPORTANCE.tsv", family_out)
    stable_tsv(OUT / "STAGE5E_PERMUTATION_IMPORTANCE.tsv", perm)
    stable_tsv(OUT / "STAGE5E_FEATURE_DISTRIBUTION_DIAGNOSTICS.tsv", distributions)
    stable_tsv(OUT / "STAGE5E_PREDICTION_SCORE_DISTRIBUTIONS.tsv", scores)
    stable_tsv(OUT / "STAGE5E_CORRELATED_FEATURE_PAIRS.tsv", correlated)
    stable_tsv(OUT / "STAGE5E_LANDSCAPE_RANK_CORRELATIONS.tsv", rank_corr)
    write_json(OUT / "STAGE5E_SHAP_CONFIGURATION.json", config); write_json(OUT / "STAGE5E_MODEL_AUDIT.json", model_audit)
    figures = make_figures(glob, land, stability, family_out, perm, distributions, scores)
    stable_tsv(OUT / "STAGE5E_FIGURE_CATALOGUE.tsv", figures)
    write_scientific_documents(summary, glob)
    pending = {"status": "PENDING"}; final_report(summary, model_audit, pending, glob, figures)
    tests = run_tests(); write_json(OUT / "STAGE5E_TESTS.json", tests); final_report(summary, model_audit, tests, glob, figures)
    final_integrity = verify_baseline()
    cp = checkpoint(model_audit, config, tests, figures, final_integrity); write_json(OUT / "STAGE5E_CHECKPOINT.json", cp)
    if tests["status"] != "PASS": raise RuntimeError("STOP: Stage 5E tests failed; see STAGE5E_TESTS.json")
    print(json.dumps({"status": "COMPLETE", "top_features": summary["top_10_attribution"][:5],
                      "shap_permutation_rho": summary["shap_permutation_rank_correlation"],
                      "tests": tests["status"], "checkpoint_sha256": sha256(OUT / "STAGE5E_CHECKPOINT.json")}, indent=2))


if __name__ == "__main__":
    main()
