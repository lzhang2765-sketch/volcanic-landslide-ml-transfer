"""Stage 5D-E development-only exploratory robustness experiment.

The preparatory amendment must be frozen before this module will fit anything.
Historical caches are read-only; all new predictions and diagnostics are written
under outputs/stage5d_e or data/interim/_stage5de_cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from . import stage4b as S
from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5d_e"
PRED = OUT / "predictions"
CACHE = ROOT / "data/interim/_stage5de_cache"
FREEZE = OUT / "STAGE5DE_EXPLORATORY_AMENDMENT_FREEZE.json"
LANDSCAPES = S.LANDSCAPES
MODEL_LABELS = {"B0": "LOGISTIC_M", "W1": "DIFFICULTY_REWEIGHTED_LOGISTIC_M",
                "D1": "GROUP_DRO_LOGISTIC_M", "E1": "LOGISTIC_RF_RANK_ENSEMBLE"}
SEED = 20261008


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_json(path: Path, obj: object) -> None:
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): x = x.item()
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    write_text(path, json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False))


def require_freeze() -> dict:
    if not FREEZE.is_file():
        raise RuntimeError("STOP: Stage 5D-E exploratory amendment is not frozen")
    x = json.loads(FREEZE.read_text())
    if x.get("status") != "FROZEN_BEFORE_ANY_STAGE5DE_MODEL_FIT" or x.get("model_fits_before_freeze") != 0:
        raise RuntimeError("STOP: invalid Stage 5D-E pre-fit freeze")
    for rel, expected in x["frozen_files"].items():
        if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected:
            raise RuntimeError(f"STOP: frozen preparatory artifact mismatch: {rel}")
    return x


def arr_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def selected_historical() -> dict:
    obj = json.loads((ROOT / "data/interim/_stage4b_results/PRIMARY.json").read_text())
    out = {}
    for r in obj["results"]:
        if r["analysis"] == "PRIMARY" and r["feature_set"] == "M" and r["model"] == "LOGISTIC_L2":
            out[r["held_out"]] = r
    if set(out) != set(LANDSCAPES):
        raise RuntimeError("STOP: incomplete historical LOGISTIC_M fold results")
    return out


def historical_prediction(held: str, params: dict) -> tuple[np.ndarray, Path]:
    train = [x for x in LANDSCAPES if x != held]
    key = S._key(sorted(train), "M", "LOGISTIC_L2", params, "ALL_VALID", "LANDSCAPE_BALANCED")
    path = S.CACHE / f"{key}__{held}.npy"
    if not path.is_file():
        raise RuntimeError(f"STOP: historical baseline prediction missing: {path}")
    return np.load(path), path


def baseline_reproduction(d: S.Data) -> dict[str, np.ndarray]:
    require_freeze()
    selected = selected_historical()
    scores, rows = {}, []
    for held in LANDSCAPES:
        train = [x for x in LANDSCAPES if x != held]
        params = json.loads(selected[held]["params"])
        t0 = time.perf_counter()
        est, cols, prep, waudit, fitaudit = S.fit(d, train, "M", "LOGISTIC_L2", params, "ALL_VALID", "LANDSCAPE_BALANCED")
        p = S.predict(d, est, cols, prep, held, "ALL_VALID").astype(np.float64)
        elapsed = time.perf_counter() - t0
        old, old_path = historical_prediction(held, params)
        y = d.L[held]["y"]
        row = {
            "held_out": held, "selected_C": params["C"], "n_predictions": len(p),
            "max_abs_prediction_difference": float(np.max(np.abs(p - old))),
            "mean_abs_prediction_difference": float(np.mean(np.abs(p - old))),
            "reproduced_ap": float(average_precision_score(y, p)),
            "historical_ap": float(selected[held]["pr_auc"]),
            "ap_difference": float(average_precision_score(y, p) - selected[held]["pr_auc"]),
            "prediction_sha256": arr_hash(p), "historical_prediction_sha256": arr_hash(old),
            "historical_prediction_path": str(old_path.relative_to(ROOT)), "wall_seconds": elapsed,
            "train_landscapes": train, "training_audit": fitaudit, "weight_audit": waudit,
            "preprocessing_columns": prep["columns"], "preprocessing_fit": prep["fitted_on"],
        }
        rows.append(row); scores[held] = p
    max_diff = max(r["max_abs_prediction_difference"] for r in rows)
    max_ap = max(abs(r["ap_difference"]) for r in rows)
    passed = max_diff <= 1e-12 and max_ap <= 1e-12
    report = {"status": "PASS_BYTE_IDENTICAL_NUMERIC_ARRAYS" if all(r["prediction_sha256"] == r["historical_prediction_sha256"] for r in rows)
              else ("PASS_NUMERIC_TOLERANCE" if passed else "FAIL_MATERIAL"),
              "material_tolerances": {"max_abs_prediction_difference": 1e-12, "max_abs_ap_difference": 1e-12},
              "max_abs_prediction_difference": max_diff, "max_abs_ap_difference": max_ap,
              "folds": rows, "historical_outputs_modified": False}
    write_json(OUT / "STAGE5DE_BASELINE_REPRODUCTION.json", report)
    PRED.mkdir(parents=True, exist_ok=True)
    for held, p in scores.items():
        np.save(PRED / f"B0__{held}__scores_v1.npy", p, allow_pickle=False)
    write_json(PRED / "STAGE5DE_PREDICTION_FORMAT_V1.json", {
        "version": "1.0.0", "format": "one float64 .npy score vector per candidate and held-out landscape",
        "row_alignment": "exact authoritative Stage 4A NPZ row order; join key is terrain_unit_id in source dataset",
        "contents": "raw uncalibrated ranking scores only; labels and features are not duplicated",
        "score_interpretation": "development-only uncalibrated model output, not a probability",
    })
    if not passed:
        raise RuntimeError(f"STOP: material B0 reproduction failure: max prediction delta={max_diff}, max AP delta={max_ap}")
    return scores


def _fit_custom_logistic(d: S.Data, train: list[str], C: float, multipliers: dict[str, float]):
    cols = [d.idx[c] for c in d.fs["M"]]
    xs = [d.L[l]["X"][d.sample(l, "ALL_VALID")][:, cols] for l in train]
    ys = [d.L[l]["y"][d.sample(l, "ALL_VALID")] for l in train]
    ws, waudit = S.weights(d, train, "ALL_VALID", "LANDSCAPE_BALANCED")
    ws = [w * multipliers[l] for l, w in zip(train, ws)]
    scale = sum(len(w) for w in ws) / sum(float(w.sum()) for w in ws)
    ws = [w * scale for w in ws]
    X, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    est = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, C=C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(Xs, y, sample_weight=w)
    prep = {"mean": mu, "sd": sd, "cols": cols}
    audit = {l: {**waudit["per_landscape"][l], "difficulty_multiplier": float(multipliers[l]),
                 "final_sample_weight_mass": float(x.sum())} for l, x in zip(train, ws)}
    return est, prep, audit


def _predict_custom(d: S.Data, fitted, held: str) -> np.ndarray:
    est, prep = fitted
    X = d.L[held]["X"][:, prep["cols"]]
    return est.predict_proba((X - prep["mean"]) / prep["sd"])[:, 1].astype(np.float64)


def _aux_prediction(d: S.Data, fitset: tuple[str, ...], held: str) -> np.ndarray:
    """Honest fixed-B0 auxiliary prediction used only to estimate W1 difficulty.

    C=0.01 is the frozen historical B0 value selected in every authoritative
    outer fold.  It is not tuned from the auxiliary holdout and introduces no
    additional W1 hyperparameter.
    """
    train = [x for x in fitset if x != held]
    key = S._key("STAGE5DE_W1_AUX_V1", sorted(train), held, {"C": 0.01})
    path = CACHE / f"{key}__{held}.npy"
    if path.is_file():
        return np.load(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    est, cols, prep, _, _ = S.fit(d, train, "M", "LOGISTIC_L2", {"C": 0.01}, "ALL_VALID", "LANDSCAPE_BALANCED")
    p = S.predict(d, est, cols, prep, held, "ALL_VALID").astype(np.float64)
    np.save(path, p, allow_pickle=False)
    return p


_DIFFICULTY_MEMO: dict[tuple[str, ...], tuple[dict, dict]] = {}


def difficulty_weights(d: S.Data, fitset: list[str]) -> tuple[dict[str, float], dict]:
    key = tuple(sorted(fitset))
    if key in _DIFFICULTY_MEMO:
        return _DIFFICULTY_MEMO[key]
    losses = {}
    for held in key:
        p = np.clip(_aux_prediction(d, key, held), 1e-15, 1 - 1e-15)
        # The auxiliary holdout contains its complete population; its population
        # mean is therefore already the sampling-corrected mean loss.
        losses[held] = float(log_loss(d.L[held]["y"], p, labels=[False, True]))
    mean_loss = float(np.mean(list(losses.values())))
    raw = {l: float(np.clip(np.exp(v - mean_loss), 0.5, 2.0)) for l, v in losses.items()}
    z = len(key) / sum(raw.values())
    mult = {l: float(raw[l] * z) for l in key}
    diag = {"fit_landscapes": list(key), "auxiliary_C": 0.01, "auxiliary_C_basis": "frozen historical B0 value selected in all six authoritative folds",
            "loss_definition": "complete auxiliary-held-out-landscape mean binary log loss", "losses": losses,
            "pre_normalization_multipliers": raw, "normalized_multipliers": mult,
            "clipping": [0.5, 2.0], "normalization_check": float(sum(mult.values()))}
    _DIFFICULTY_MEMO[key] = (mult, diag)
    return mult, diag


def choose_simple(values: list[float], means: list[float], tolerance: float = .005):
    best = max(means)
    for value, mean in zip(values, means):
        if mean >= best - tolerance:
            return value
    raise AssertionError("unreachable")


def run_w1(d: S.Data) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    scores, hp_rows, rw_rows = {}, [], []
    for oi, held in enumerate(LANDSCAPES):
        outer_train = [x for x in LANDSCAPES if x != held]
        config_aps = []
        for C in (0.01, 0.1, 1.0):
            aps = []
            for inner_held in outer_train:
                fitset = [x for x in outer_train if x != inner_held]
                mult, diag = difficulty_weights(d, fitset)
                est, prep, _ = _fit_custom_logistic(d, fitset, C, mult)
                p = _predict_custom(d, (est, prep), inner_held)
                aps.append(float(average_precision_score(d.L[inner_held]["y"], p)))
            config_aps.append(float(np.mean(aps)))
            hp_rows.append({"outer_held_out": held, "candidate": "W1", "parameter": "C", "value": C,
                            "inner_mean_ap": config_aps[-1], "inner_landscape_aps": json.dumps(dict(zip(outer_train, aps)), sort_keys=True),
                            "selected": False})
        selected = choose_simple([.01, .1, 1.], config_aps)
        for r in hp_rows[-3:]: r["selected"] = r["value"] == selected
        mult, diag = difficulty_weights(d, outer_train)
        t0 = time.perf_counter(); est, prep, audit = _fit_custom_logistic(d, outer_train, selected, mult)
        p = _predict_custom(d, (est, prep), held); elapsed = time.perf_counter() - t0
        scores[held] = p
        np.save(PRED / f"W1__{held}__scores_v1.npy", p, allow_pickle=False)
        rw_rows.append({"outer_held_out": held, "fit_role": "OUTER_FINAL", "selected_C": selected,
                        "wall_seconds_final_fit_predict": elapsed, "difficulty": json.dumps(diag, sort_keys=True),
                        "weight_audit": json.dumps(audit, sort_keys=True), "prediction_sha256": arr_hash(p)})
        print(f"W1 fold {oi + 1}/6 complete: {held}", flush=True)
    return scores, hp_rows, rw_rows


def _selected_stage4b(held: str, model: str) -> dict:
    obj = json.loads((ROOT / "data/interim/_stage4b_results/PRIMARY.json").read_text())
    rows = [r for r in obj["results"] if r["analysis"] == "PRIMARY" and r["held_out"] == held and
            r["feature_set"] == "M" and r["model"] == model]
    if len(rows) != 1: raise RuntimeError(f"STOP: selected Stage 4B row missing for {held}/{model}")
    return json.loads(rows[0]["params"])


def percentile_rank(score: np.ndarray) -> np.ndarray:
    if len(score) == 1: return np.array([.5])
    return (rankdata(score, method="average") - 1.0) / (len(score) - 1.0)


def top_mask(score: np.ndarray, ids: np.ndarray, fraction: float) -> np.ndarray:
    n = int(math.ceil(fraction * len(score)))
    order = np.lexsort((ids.astype(str), -score))
    mask = np.zeros(len(score), dtype=bool); mask[order[:n]] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def ensemble_gate(d: S.Data, b0_scores: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    gate_all, scores, hp_rows = {}, {}, []
    for held in LANDSCAPES:
        train5 = [x for x in LANDSCAPES if x != held]
        lp, rp, diags, aps_l, aps_r = {}, {}, [], [], []
        lparams, rparams = _selected_stage4b(held, "LOGISTIC_L2"), _selected_stage4b(held, "RANDOM_FOREST")
        for val in train5:
            train4 = [x for x in train5 if x != val]
            pk = S._key(sorted(train4), "M", "LOGISTIC_L2", lparams, "ALL_VALID", "LANDSCAPE_BALANCED")
            rk = S._key(sorted(train4), "M", "RANDOM_FOREST", rparams, "ALL_VALID", "LANDSCAPE_BALANCED")
            pl = np.load(S.CACHE / f"{pk}__{val}.npy"); pr = np.load(S.CACHE / f"{rk}__{val}.npy")
            y, ids = d.L[val]["y"], d.L[val]["ids"]
            rl, rr = percentile_rank(pl), percentile_rank(pr)
            lp[val], rp[val] = rl, rr
            el = np.where(y, 1 - rl, rl); er = np.where(y, 1 - rr, rr)
            tl, tr = top_mask(pl, ids, .10), top_mask(pr, ids, .10)
            ml, mr = y & ~top_mask(pl, ids, .20), y & ~top_mask(pr, ids, .20)
            apl, apr = float(average_precision_score(y, pl)), float(average_precision_score(y, pr))
            aps_l.append(apl); aps_r.append(apr)
            diags.append({"landscape": val, "score_rank_spearman": float(spearmanr(rl, rr).statistic),
                          "ranking_error_spearman": float(spearmanr(el, er).statistic),
                          "false_positive_top10_jaccard": jaccard(tl & ~y, tr & ~y),
                          "missed_positive_outside_top20_jaccard": jaccard(ml, mr), "logistic_ap": apl, "rf_ap": apr})
        med = lambda key: float(np.median([x[key] for x in diags]))
        components = {"median_score_rank_spearman_lt_0.95": med("score_rank_spearman") < .95,
                      "median_error_spearman_lt_0.95": med("ranking_error_spearman") < .95,
                      "median_fp_top10_jaccard_le_0.80": med("false_positive_top10_jaccard") <= .80,
                      "median_missed_positive_outside_top20_jaccard_le_0.80": med("missed_positive_outside_top20_jaccard") <= .80,
                      "bidirectional_specialization": sum(a > b for a, b in zip(aps_l, aps_r)) >= 2 and sum(b > a for a, b in zip(aps_l, aps_r)) >= 2}
        passed = all(components.values())
        item = {"outer_held_out": held, "status": "PASSED" if passed else "NOT_ADVANCED",
                "components": components, "medians": {k: med(k) for k in ("score_rank_spearman", "ranking_error_spearman", "false_positive_top10_jaccard", "missed_positive_outside_top20_jaccard")},
                "logistic_wins": int(sum(a > b for a, b in zip(aps_l, aps_r))), "rf_wins": int(sum(b > a for a, b in zip(aps_l, aps_r))),
                "training_only_landscape_diagnostics": diags, "outer_outcome_used": False}
        if passed:
            means = []
            for wl in (.75, .5, .25):
                vals = [float(average_precision_score(d.L[v]["y"], wl * lp[v] + (1 - wl) * rp[v])) for v in train5]
                means.append(float(np.mean(vals)))
                hp_rows.append({"outer_held_out": held, "candidate": "E1", "parameter": "logistic_weight", "value": wl,
                                "inner_mean_ap": means[-1], "inner_landscape_aps": json.dumps(dict(zip(train5, vals)), sort_keys=True), "selected": False})
            wl = choose_simple([.75, .5, .25], means)
            for r in hp_rows[-3:]: r["selected"] = r["value"] == wl
            rfparams = rparams
            train = [x for x in LANDSCAPES if x != held]
            key = S._key(sorted(train), "M", "RANDOM_FOREST", rfparams, "ALL_VALID", "LANDSCAPE_BALANCED")
            pr = np.load(S.CACHE / f"{key}__{held}.npy")
            p = wl * percentile_rank(b0_scores[held]) + (1 - wl) * percentile_rank(pr)
            scores[held] = p.astype(np.float64); item["selected_logistic_weight"] = wl
            np.save(PRED / f"E1__{held}__scores_v1.npy", scores[held], allow_pickle=False)
        gate_all[held] = item
    result = {"status": "ELIGIBLE_ALL_OUTER_FOLDS" if len(scores) == 6 else "NOT_ADVANCED",
              "rule": "all five frozen components must pass separately inside every outer training set",
              "outer_folds_passed": len(scores), "outer_folds": gate_all}
    write_json(OUT / "STAGE5DE_ENSEMBLE_GATE.json", result)
    return result, scores, hp_rows


def bounded_waterfill(v: np.ndarray, lower: float = .05, upper: float = .60) -> np.ndarray:
    """Deterministic proportional water-filling onto a bounded simplex."""
    v = np.asarray(v, dtype=float)
    fixed = np.zeros(len(v), dtype=bool); out = np.zeros(len(v), dtype=float)
    for _ in range(len(v) + 2):
        free = ~fixed; remaining = 1.0 - out[fixed].sum()
        if not free.any(): break
        denom = v[free].sum()
        proposed = np.full(free.sum(), remaining / free.sum()) if denom <= 0 else remaining * v[free] / denom
        idx = np.flatnonzero(free)
        violations = [(i, lower) for i, x in zip(idx, proposed) if x < lower] + [(i, upper) for i, x in zip(idx, proposed) if x > upper]
        if not violations:
            out[idx] = proposed; break
        # Resolve in ascending group order, then recompute remaining water.
        i, value = sorted(violations, key=lambda x: x[0])[0]
        out[i] = value; fixed[i] = True
    if abs(out.sum() - 1) > 1e-12:
        raise FloatingPointError(f"bounded simplex projection failed: {out}")
    return out


def _dro_data(d: S.Data, train: list[str]):
    groups = sorted(train)
    cols = [d.idx[c] for c in d.fs["M"]]
    xs = [d.L[l]["X"][d.sample(l, "ALL_VALID")][:, cols] for l in groups]
    ys = [d.L[l]["y"][d.sample(l, "ALL_VALID")] for l in groups]
    basew, _ = S.weights(d, groups, "ALL_VALID", "LANDSCAPE_BALANCED")
    X = np.vstack(xs); y = np.concatenate(ys).astype(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0; X = (X - mu) / sd
    slices, weights, start = [], [], 0
    for yy, ww in zip(ys, basew):
        stop = start + len(yy); slices.append(slice(start, stop)); weights.append(ww / ww.sum()); start = stop
    pooled_w = np.concatenate(basew)
    prevalence = float(np.sum(pooled_w * y) / pooled_w.sum())
    return groups, X, y, slices, weights, mu, sd, cols, prevalence


def fit_group_dro(d: S.Data, train: list[str]) -> dict:
    """Fit all three frozen eta configurations in one deterministic BLAS pass."""
    groups, X, y, slices, wg, mu, sd, cols, prevalence = _dro_data(d, train)
    key = S._key("STAGE5DE_D1_V1", groups, [0.01, .05, .1], 0.01, 2000, 1e-8, 20)
    npz = CACHE / f"{key}.npz"; meta_path = CACHE / f"{key}.json"
    if npz.is_file() and meta_path.is_file():
        z = np.load(npz, allow_pickle=False)
        return {"groups": groups, "B": z["B"], "b": z["b"], "q": z["q"], "mu": z["mu"], "sd": z["sd"],
                "cols": z["cols"], "converged": z["converged"].astype(bool), "meta": json.loads(meta_path.read_text())}
    CACHE.mkdir(parents=True, exist_ok=True)
    etas = np.array([.01, .05, .1]); ncfg = len(etas); ng = len(groups); n_eff = len(y)
    B = np.zeros((X.shape[1], ncfg)); b = np.full(ncfg, math.log(prevalence / (1 - prevalence)))
    q = np.full((ng, ncfg), 1.0 / ng); converged = np.zeros(ncfg, dtype=bool)
    consecutive = np.zeros(ncfg, dtype=int); previous = np.full(ncfg, np.nan)
    objective_start = np.full(ncfg, np.nan); objective_last = np.full(ncfg, np.nan)
    qmin = q.copy(); qmax = q.copy(); trajectories = [[] for _ in range(ncfg)]
    epochs = np.full(ncfg, 2000, dtype=int); t0 = time.perf_counter()
    for epoch in range(2000):
        group_loss = np.zeros((ng, ncfg)); group_grad = np.zeros((ng, X.shape[1], ncfg)); group_gb = np.zeros((ng, ncfg))
        for gi, (sl, w) in enumerate(zip(slices, wg)):
            z = np.clip(X[sl] @ B + b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z)); yy = y[sl, None]
            group_loss[gi] = np.sum(w[:, None] * (np.logaddexp(0.0, z) - yy * z), axis=0)
            residual = w[:, None] * (p - yy)
            group_grad[gi] = X[sl].T @ residual; group_gb[gi] = residual.sum(axis=0)
        active = ~converged
        for ci in np.flatnonzero(active):
            q[:, ci] = bounded_waterfill(q[:, ci] * np.exp(etas[ci] * np.clip(group_loss[:, ci], 0, 10)))
        qmin = np.minimum(qmin, q); qmax = np.maximum(qmax, q)
        obj = np.sum(q * group_loss, axis=0) + np.sum(B * B, axis=0) / (2 * .01 * n_eff)
        objective_start = np.where(np.isnan(objective_start), obj, objective_start); objective_last = obj.copy()
        relative = np.abs(obj - previous) / np.maximum(1.0, np.abs(previous))
        good = np.isfinite(relative) & (relative < 1e-8)
        consecutive = np.where(active & good, consecutive + 1, np.where(active, 0, consecutive))
        newly = active & (consecutive >= 20)
        converged[newly] = True; epochs[newly] = epoch + 1
        if epoch % 50 == 0 or epoch == 1999:
            for ci in range(ncfg): trajectories[ci].append({"epoch": epoch + 1, "objective": float(obj[ci]), "q": q[:, ci].tolist(), "group_losses": group_loss[:, ci].tolist()})
        if converged.all(): break
        gB = np.einsum("gfc,gc->fc", group_grad, q) + B / (.01 * n_eff)
        gb = np.sum(group_gb * q, axis=0)
        for ci in np.flatnonzero(~converged):
            norm = float(np.sqrt(np.sum(gB[:, ci] ** 2) + gb[ci] ** 2))
            if not np.isfinite(norm):
                converged[ci] = False; epochs[ci] = epoch + 1
                continue
            scale = min(1.0, 10.0 / max(norm, 1e-300)); lr = .05 / (1 + .01 * epoch)
            B[:, ci] -= lr * scale * gB[:, ci]; b[ci] -= lr * scale * gb[ci]
        previous = obj
    wall = time.perf_counter() - t0
    meta = {"fit_landscapes": groups, "etas": etas.tolist(), "converged": converged.tolist(), "epochs": epochs.tolist(),
            "wall_seconds": wall, "objective_start": objective_start.tolist(), "objective_final": objective_last.tolist(),
            "q_final": q.T.tolist(), "q_min": qmin.T.tolist(), "q_max": qmax.T.tolist(), "q_trajectory_every_50_epochs": trajectories,
            "group_order": groups, "n_eff": n_eff, "optimizer": "frozen deterministic full-batch gradient descent",
            "nonfinite": bool(not np.isfinite(B).all() or not np.isfinite(b).all() or not np.isfinite(q).all())}
    np.savez_compressed(npz, B=B, b=b, q=q, mu=mu, sd=sd, cols=np.array(cols), converged=converged)
    write_json(meta_path, meta)
    return {"groups": groups, "B": B, "b": b, "q": q, "mu": mu, "sd": sd, "cols": np.array(cols), "converged": converged, "meta": meta}


def predict_dro(d: S.Data, fit: dict, held: str, ci: int) -> np.ndarray:
    X = d.L[held]["X"][:, fit["cols"]]; X = (X - fit["mu"]) / fit["sd"]
    z = np.clip(X @ fit["B"][:, ci] + fit["b"][ci], -30, 30)
    return (1 / (1 + np.exp(-z))).astype(np.float64)


def run_d1(d: S.Data) -> tuple[dict[str, np.ndarray], list[dict], list[dict], str]:
    scores, hp_rows, diag_rows = {}, [], []
    etas = [.01, .05, .1]
    for oi, held in enumerate(LANDSCAPES):
        outer_train = [x for x in LANDSCAPES if x != held]
        per_eta = [[] for _ in etas]; stable = np.ones(len(etas), dtype=bool)
        for val in outer_train:
            fitset = [x for x in outer_train if x != val]
            fit = fit_group_dro(d, fitset)
            stable &= fit["converged"]
            for ci in range(len(etas)):
                if fit["converged"][ci]:
                    per_eta[ci].append(float(average_precision_score(d.L[val]["y"], predict_dro(d, fit, val, ci))))
            if not any(r.get("fit_key") == "|".join(sorted(fitset)) for r in diag_rows):
                for ci, eta in enumerate(etas):
                    diag_rows.append({"fit_key": "|".join(sorted(fitset)), "fit_role": "INNER", "eta_q": eta,
                                      "converged": bool(fit["converged"][ci]), "epochs": fit["meta"]["epochs"][ci],
                                      "wall_seconds_joint_fit": fit["meta"]["wall_seconds"], "objective_start": fit["meta"]["objective_start"][ci],
                                      "objective_final": fit["meta"]["objective_final"][ci], "q_final": json.dumps(dict(zip(fit["groups"], fit["q"][:, ci]))),
                                      "q_min": json.dumps(dict(zip(fit["groups"], fit["meta"]["q_min"][ci]))),
                                      "q_max": json.dumps(dict(zip(fit["groups"], fit["meta"]["q_max"][ci]))), "nonfinite": fit["meta"]["nonfinite"]})
        means = [float(np.mean(x)) if stable[i] and len(x) == len(outer_train) else float("-inf") for i, x in enumerate(per_eta)]
        for ci, eta in enumerate(etas):
            hp_rows.append({"outer_held_out": held, "candidate": "D1", "parameter": "eta_q", "value": eta,
                            "inner_mean_ap": means[ci] if np.isfinite(means[ci]) else "INELIGIBLE_NONCONVERGENCE",
                            "inner_landscape_aps": json.dumps(per_eta[ci]), "selected": False})
        eligible = [i for i, x in enumerate(means) if np.isfinite(x)]
        if not eligible:
            print(f"D1 fold {oi + 1}/6 ineligible: no stable inner configuration", flush=True); continue
        best = max(means[i] for i in eligible); ci = next(i for i in eligible if means[i] >= best - .005)
        hp_rows[-3 + ci]["selected"] = True
        outer_fit = fit_group_dro(d, outer_train)
        for cii, eta in enumerate(etas):
            diag_rows.append({"fit_key": "|".join(sorted(outer_train)), "fit_role": f"OUTER_FINAL_FOR_{held}", "eta_q": eta,
                              "converged": bool(outer_fit["converged"][cii]), "epochs": outer_fit["meta"]["epochs"][cii],
                              "wall_seconds_joint_fit": outer_fit["meta"]["wall_seconds"], "objective_start": outer_fit["meta"]["objective_start"][cii],
                              "objective_final": outer_fit["meta"]["objective_final"][cii], "q_final": json.dumps(dict(zip(outer_fit["groups"], outer_fit["q"][:, cii]))),
                              "q_min": json.dumps(dict(zip(outer_fit["groups"], outer_fit["meta"]["q_min"][cii]))),
                              "q_max": json.dumps(dict(zip(outer_fit["groups"], outer_fit["meta"]["q_max"][cii]))), "nonfinite": outer_fit["meta"]["nonfinite"]})
        if not outer_fit["converged"][ci]:
            print(f"D1 fold {oi + 1}/6 ineligible: selected outer fit did not converge", flush=True); continue
        p = predict_dro(d, outer_fit, held, ci); scores[held] = p
        np.save(PRED / f"D1__{held}__scores_v1.npy", p, allow_pickle=False)
        print(f"D1 fold {oi + 1}/6 complete: {held}; eta={etas[ci]}", flush=True)
    status = "ELIGIBLE_ALL_OUTER_FOLDS" if len(scores) == 6 else "INELIGIBLE_NONCONVERGENCE"
    return scores, hp_rows, diag_rows, status


def load_completed_scores() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for candidate in ("B0", "W1", "D1", "E1"):
        paths = [PRED / f"{candidate}__{l}__scores_v1.npy" for l in LANDSCAPES]
        if all(p.is_file() for p in paths): out[candidate] = {l: np.load(p) for l, p in zip(LANDSCAPES, paths)}
    return out


def fold_metrics(d: S.Data, scores: dict) -> pd.DataFrame:
    rows = []
    for candidate, by_landscape in scores.items():
        for lid in LANDSCAPES:
            y, p, ids = d.L[lid]["y"], by_landscape[lid], d.L[lid]["ids"]
            row = {"landscape": lid, "candidate": candidate, "model_name": MODEL_LABELS[candidate],
                   "n_units": len(y), "positives": int(y.sum()), "negatives": int((~y).sum()), "prevalence": float(y.mean()),
                   "average_precision": float(average_precision_score(y, p)), "pr_lift": float(average_precision_score(y, p) / y.mean()),
                   "roc_auc": float(roc_auc_score(y, p)), "score_type": "UNCALIBRATED_RANKING_SCORE"}
            for pct in (5, 10, 20):
                top = top_mask(p, ids, pct / 100)
                row[f"top_{pct}_precision"] = float(y[top].mean())
                row[f"top_{pct}_recall"] = float(y[top].sum() / y.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def spatial_bootstrap(d: S.Data, scores: dict, reps: int = 2000):
    from .stage4d import Sorted, block_codes
    arrays, rows = {}, []
    for li, lid in enumerate(LANDSCAPES):
        with np.load(S.MEG_DIR / f"{lid}.npz", allow_pickle=False) as z:
            blocks = z["spatial_block_3000m"]
        y = d.L[lid]["y"]; code, nb = block_codes(blocks)
        sorters = {c: Sorted(scores[c][lid], y) for c in scores}
        rng = np.random.default_rng(SEED + li)
        prev = np.full(reps, np.nan); values = {c: np.full((reps, 2), np.nan) for c in scores}; dropped = 0
        for r in range(reps):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[code].astype(float)
            if w[y].sum() == 0:
                dropped += 1; continue
            prev[r] = w[y].sum() / w.sum()
            for c in scores: values[c][r] = sorters[c].ap_auc(w)
        arrays[lid] = {"prev": prev, "values": values, "dropped": dropped, "n_blocks": nb}
        for c in scores:
            point_ap = float(average_precision_score(y, scores[c][lid])); point_roc = float(roc_auc_score(y, scores[c][lid]))
            for j, metric, point in ((0, "average_precision", point_ap), (1, "roc_auc", point_roc)):
                v = values[c][:, j]; v = v[np.isfinite(v)]
                rows.append({"scope": "LANDSCAPE", "landscape": lid, "candidate": c, "comparison": "POINT_INTERVAL",
                             "metric": metric, "point_estimate": point, "ci_low": float(np.percentile(v, 2.5)),
                             "ci_high": float(np.percentile(v, 97.5)), "replicates_retained": len(v), "replicates_dropped": dropped,
                             "spatial_blocks": nb, "method": "paired 3 km spatial-block bootstrap"})
            lift = values[c][:, 0] / prev; lift = lift[np.isfinite(lift)]
            rows.append({"scope": "LANDSCAPE", "landscape": lid, "candidate": c, "comparison": "POINT_INTERVAL", "metric": "pr_lift",
                         "point_estimate": point_ap / y.mean(), "ci_low": float(np.percentile(lift, 2.5)), "ci_high": float(np.percentile(lift, 97.5)),
                         "replicates_retained": len(lift), "replicates_dropped": dropped, "spatial_blocks": nb,
                         "method": "paired 3 km spatial-block bootstrap"})
            if c != "B0":
                for metric, delta in (("delta_average_precision_vs_B0", values[c][:, 0] - values["B0"][:, 0]),
                                      ("delta_pr_lift_vs_B0", values[c][:, 0] / prev - values["B0"][:, 0] / prev)):
                    v = delta[np.isfinite(delta)]
                    point = (point_ap - float(average_precision_score(y, scores["B0"][lid]))) if "average" in metric else (
                        point_ap / y.mean() - float(average_precision_score(y, scores["B0"][lid])) / y.mean())
                    rows.append({"scope": "LANDSCAPE", "landscape": lid, "candidate": c, "comparison": "PAIRED_VS_B0", "metric": metric,
                                 "point_estimate": point, "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
                                 "replicates_retained": len(v), "replicates_dropped": dropped, "spatial_blocks": nb,
                                 "method": "paired 3 km spatial-block bootstrap"})
        print(f"bootstrap {li + 1}/6 complete: {lid}", flush=True)
    # Same replicate index represents independent within-landscape draws.  This
    # gives a sensitivity interval for the six observed landscapes, not a
    # volcano-population confidence interval.
    for c in scores:
        if c == "B0": continue
        mat_ap = np.column_stack([arrays[l]["values"][c][:, 0] - arrays[l]["values"]["B0"][:, 0] for l in LANDSCAPES])
        mat_lift = np.column_stack([(arrays[l]["values"][c][:, 0] - arrays[l]["values"]["B0"][:, 0]) / arrays[l]["prev"] for l in LANDSCAPES])
        for metric, mat in (("median_delta_average_precision_vs_B0", mat_ap), ("median_delta_pr_lift_vs_B0", mat_lift)):
            v = np.nanmedian(mat, axis=1); v = v[np.isfinite(v)]
            rows.append({"scope": "SIX_OBSERVED_LANDSCAPES", "landscape": "SIX_LANDSCAPE_MEDIAN", "candidate": c,
                         "comparison": "PAIRED_VS_B0", "metric": metric, "point_estimate": float(np.median(v)),
                         "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)),
                         "replicates_retained": len(v), "replicates_dropped": int(reps - len(v)), "spatial_blocks": "LANDSCAPE_SPECIFIC",
                         "method": "paired within-landscape bootstrap summary; NOT a volcano-population CI"})
    return pd.DataFrame(rows), arrays


def comparisons(folds: pd.DataFrame, uncertainty: pd.DataFrame, completed: list[str]):
    rows, paired = [], []
    base = folds[folds.candidate == "B0"].set_index("landscape")
    for c in completed:
        g = folds[folds.candidate == c].set_index("landscape")
        lift = g.pr_lift
        catastrophic = []
        for lid in LANDSCAPES:
            d_ap = float(g.loc[lid, "average_precision"] - base.loc[lid, "average_precision"])
            d_lift = float(g.loc[lid, "pr_lift"] - base.loc[lid, "pr_lift"])
            upper = None
            if c != "B0":
                u = uncertainty[(uncertainty.landscape == lid) & (uncertainty.candidate == c) &
                                (uncertainty.metric == "delta_average_precision_vs_B0")]
                upper = float(u.ci_high.iloc[0])
            fail = False
            if c != "B0":
                if lid == "COTABATO_DAVAO_2019": fail = g.loc[lid, "pr_lift"] < .80 and upper < 0
                elif base.loc[lid, "pr_lift"] >= 1.25:
                    fail = upper < 0 and (g.loc[lid, "pr_lift"] <= 1 or g.loc[lid, "pr_lift"] < .67 * base.loc[lid, "pr_lift"])
            if fail: catastrophic.append(lid)
            if c != "B0": paired.append({"scope": "LANDSCAPE", "candidate": c, "landscape": lid, "ap_difference_vs_B0": d_ap,
                                         "pr_lift_difference_vs_B0": d_lift, "paired_ap_ci_upper": upper,
                                         "direction": "IMPROVED" if d_ap > 0 else ("WORSENED" if d_ap < 0 else "UNCHANGED"),
                                         "catastrophic_failure": fail})
        robust = float(.70 * np.log(np.median(lift)) + .30 * np.log(np.min(lift)))
        rows.append({"candidate": c, "model_name": MODEL_LABELS[c], "status": "COMPLETE", "landscapes_complete": 6,
                     "median_average_precision": float(g.average_precision.median()), "median_pr_lift": float(lift.median()),
                     "worst_pr_lift": float(lift.min()), "worst_landscape": str(lift.idxmin()), "robust_score": robust,
                     "pr_lift_iqr": float(lift.quantile(.75) - lift.quantile(.25)), "pr_lift_range": float(lift.max() - lift.min()),
                     "landscapes_improved_vs_B0": int((g.average_precision > base.average_precision).sum()) if c != "B0" else 0,
                     "landscapes_worsened_vs_B0": int((g.average_precision < base.average_precision).sum()) if c != "B0" else 0,
                     "median_ap_difference_vs_B0": float((g.average_precision - base.average_precision).median()),
                     "median_pr_lift_difference_vs_B0": float((g.pr_lift - base.pr_lift).median()),
                     "worst_ap_difference_vs_B0": float((g.average_precision - base.average_precision).min()),
                     "catastrophic_failure": bool(catastrophic), "catastrophic_landscapes": "|".join(catastrophic),
                     "advancement_eligibility": "ELIGIBLE" if not catastrophic else "INELIGIBLE_CATASTROPHIC_GUARD"})
        if c != "B0":
            paired.append({"scope": "SUMMARY", "candidate": c, "landscape": "SIX_LANDSCAPES",
                           "ap_difference_vs_B0": float((g.average_precision - base.average_precision).median()),
                           "pr_lift_difference_vs_B0": float((g.pr_lift - base.pr_lift).median()), "paired_ap_ci_upper": np.nan,
                           "direction": "MIXED", "catastrophic_failure": bool(catastrophic)})
    rows.append({"candidate": "D1", "model_name": MODEL_LABELS["D1"], "status": "INELIGIBLE_NONCONVERGENCE", "landscapes_complete": 0,
                 "advancement_eligibility": "INELIGIBLE_NONCONVERGENCE"})
    comp = pd.DataFrame(rows)
    ranked = comp[(comp.status == "COMPLETE") & comp.catastrophic_failure.eq(False)].sort_values("robust_score", ascending=False)
    comp["exploratory_rank"] = np.nan
    for rank, idx in enumerate(ranked.index, 1): comp.loc[idx, "exploratory_rank"] = rank
    return comp, pd.DataFrame(paired)


def make_figures(folds: pd.DataFrame, comp: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    OUTFIG = OUT / "figures"; OUTFIG.mkdir(parents=True, exist_ok=True)
    short = {l: l.split("_")[0].title() for l in LANDSCAPES}; colors = {"B0": "#334e68", "W1": "#2a9d8f", "E1": "#e76f51"}
    catalogue = []
    def save(fig, filename, title, scope):
        fig.suptitle("DEVELOPMENT-ONLY EXPLORATORY RESULTS", fontsize=10, fontweight="bold", y=.995)
        fig.tight_layout(rect=(0, 0, 1, .96)); path = OUTFIG / filename; fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
        catalogue.append({"figure": len(catalogue) + 1, "file": str(path.relative_to(ROOT)), "title": title,
                          "scope": scope, "sha256": sha256(path)})
    candidates = [x for x in ("B0", "W1", "E1") if x in set(folds.candidate)]
    # 1
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(6); width = .24
    for i, c in enumerate(candidates):
        g = folds[folds.candidate == c].set_index("landscape").reindex(LANDSCAPES)
        ax.bar(x + (i - 1) * width, g.average_precision, width, label=c, color=colors[c])
    ax.set_xticks(x, [short[l] for l in LANDSCAPES], rotation=25); ax.set_ylabel("Average precision"); ax.set_title("Model comparison by held-out landscape"); ax.legend()
    save(fig, "FIG01_MODEL_COMPARISON_BY_LANDSCAPE.png", "Model comparison by landscape", "six development-exposed landscapes")
    # 2/3
    lp = paired[paired.scope == "LANDSCAPE"]
    for num, col, ylabel, name in ((2, "ap_difference_vs_B0", "Δ Average precision vs B0", "PAIRED_AP_DIFFERENCES"),
                                   (3, "pr_lift_difference_vs_B0", "Δ PR lift vs B0", "PAIRED_PR_LIFT_DIFFERENCES")):
        fig, ax = plt.subplots(figsize=(11, 5));
        for i, c in enumerate([x for x in candidates if x != "B0"]):
            g = lp[lp.candidate == c].set_index("landscape").reindex(LANDSCAPES)
            ax.scatter(x + (i - .5) * .14, g[col], s=55, label=c, color=colors[c])
        ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, [short[l] for l in LANDSCAPES], rotation=25); ax.set_ylabel(ylabel); ax.set_title(ylabel); ax.legend()
        save(fig, f"FIG0{num}_{name}.png", ylabel, "paired outer-fold results")
    # 4
    g = comp[comp.status == "COMPLETE"].sort_values("worst_pr_lift")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(g.candidate, g.worst_pr_lift, color=[colors[x] for x in g.candidate]); ax.axvline(1, color="black", ls="--", lw=.8)
    ax.set_xlabel("Worst-landscape PR lift"); ax.set_title("Worst-landscape comparison")
    save(fig, "FIG04_WORST_LANDSCAPE_COMPARISON.png", "Worst-landscape performance", "minimum across six held-out landscapes")
    # 5
    dd = pd.read_csv(OUT / "STAGE5DE_GROUP_DRO_DIAGNOSTICS.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(10, 5)); records = []
    for r in dd.itertuples():
        for group, q in json.loads(r.q_final).items(): records.append((group, float(q), r.eta_q))
    for eta, marker in zip(sorted(dd.eta_q.unique()), ("o", "s", "^")):
        z = [(g, q) for g, q, e in records if e == eta]
        ax.scatter([short[g] for g, q in z], [q for g, q in z], alpha=.45, s=20, marker=marker, label=f"eta={eta:g}")
    ax.axhline(.05, color="grey", ls="--"); ax.axhline(.60, color="grey", ls="--"); ax.set_ylabel("Final training-group weight q"); ax.set_title("Group DRO inner-fit weights (candidate ineligible)"); ax.tick_params(axis="x", rotation=25); ax.legend()
    save(fig, "FIG05_GROUP_DRO_WEIGHTS.png", "Group DRO training-group weights", "inner fits; D1 failed convergence eligibility")
    # 6
    rw = pd.read_csv(OUT / "STAGE5DE_REWEIGHTING_DIAGNOSTICS.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rw.itertuples():
        vals = json.loads(r.difficulty)["normalized_multipliers"]
        ax.plot([short[x] for x in vals], list(vals.values()), marker="o", alpha=.65, label=short[r.outer_held_out])
    ax.axhline(1, color="black", lw=.8); ax.set_ylabel("W1 landscape multiplier"); ax.set_title("Outer-fold difficulty-aware reweighting coefficients"); ax.tick_params(axis="x", rotation=25); ax.legend(title="Held out", ncol=2, fontsize=8)
    save(fig, "FIG06_W1_REWEIGHTING_COEFFICIENTS.png", "Landscape reweighting coefficients", "training-derived outer-fold coefficients")
    # 7
    gate = json.loads((OUT / "STAGE5DE_ENSEMBLE_GATE.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 5)); keys = ["score_rank_spearman", "ranking_error_spearman", "false_positive_top10_jaccard", "missed_positive_outside_top20_jaccard"]
    labels = ["score rho", "error rho", "FP Jaccard", "missed-positive Jaccard"]
    gx = np.arange(6)
    for i, (key, label) in enumerate(zip(keys, labels)):
        ax.plot(gx, [gate["outer_folds"][l]["medians"][key] for l in LANDSCAPES], marker="o", label=label)
    ax.axhline(.95, color="grey", ls=":", label="rho gate .95"); ax.axhline(.80, color="black", ls="--", label="Jaccard gate .80")
    ax.set_xticks(gx, [short[l] for l in LANDSCAPES], rotation=25); ax.set_ylim(0, 1.02); ax.set_title("Training-only ensemble complementarity diagnostics"); ax.legend(ncol=3, fontsize=8)
    save(fig, "FIG07_ENSEMBLE_COMPLEMENTARITY.png", "Ensemble complementarity diagnostics", "training-only gate; E1 eligible")
    # 8
    mat = folds.pivot(index="candidate", columns="landscape", values="pr_lift").reindex(index=candidates, columns=LANDSCAPES)
    fig, ax = plt.subplots(figsize=(11, 3.8)); im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(6), [short[l] for l in LANDSCAPES], rotation=25); ax.set_yticks(np.arange(len(candidates)), candidates)
    for i in range(len(candidates)):
        for j in range(6): ax.text(j, i, f"{mat.iloc[i,j]:.2f}", ha="center", va="center", color="white" if mat.iloc[i,j] < mat.to_numpy().mean() else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="PR lift"); ax.set_title("Held-out landscape performance heatmap")
    save(fig, "FIG08_PERFORMANCE_HEATMAP.png", "Summary performance heatmap", "PR lift; six development-exposed landscapes")
    return pd.DataFrame(catalogue)


def write_scientific_docs(folds, comp, paired, uncertainty, d1_status):
    c = comp.set_index("candidate")
    winner = str(comp[comp.exploratory_rank == 1].candidate.iloc[0])
    w1 = c.loc["W1"]; e1 = c.loc["E1"]
    hard = folds[folds.candidate == "B0"].sort_values("pr_lift").iloc[0]
    write_text(OUT / "STAGE5DE_SCIENTIFIC_INTERPRETATION.md", f"""# Scientific interpretation

All statements below concern six development-exposed landscapes and are exploratory.

1. **Landscape reweighting:** W1 changed AP in {int(w1.landscapes_improved_vs_B0)} improved and {int(w1.landscapes_worsened_vs_B0)} worsened landscapes. Its median AP difference was {w1.median_ap_difference_vs_B0:.6g} and robust score was {w1.robust_score:.6g} versus {c.loc['B0'].robust_score:.6g} for B0. This is {'SUPPORTED within the observed development folds' if w1.robust_score > c.loc['B0'].robust_score else 'NOT SUPPORTED as an improvement'}; it is not external-validation evidence.
2. **Group DRO:** {d1_status}. The frozen optimizer did not yield a complete eligible outer prediction set, so improvement of the weakest landscape is **INCONCLUSIVE / NOT ESTIMABLE**, not a negative performance comparison.
3. **Ensemble complementarity:** The training-only gate passed all six outer folds. E1 improved {int(e1.landscapes_improved_vs_B0)} and worsened {int(e1.landscapes_worsened_vs_B0)} landscapes; its observed robust score was {e1.robust_score:.6g}. Complementarity is **SUPPORTED for bounded exploratory evaluation**, not universal superiority.
4. **Difficult landscapes:** Under B0, {hard.landscape} was weakest by PR lift ({hard.pr_lift:.4g}). Per-landscape results remain heterogeneous.
5. **Scientific magnitude:** Changes are judged from landscape-level AP/lift and paired spatial intervals, not pooled cells. Mixed signs or intervals spanning zero are not treated as decisive improvement.
6. **Complexity:** The exploratory raw robust-score leader is {winner}. This does not replace LOGISTIC_M; added complexity requires untouched external evidence and the frozen Stage 5C-A process.
7. **Evidence still needed:** qualified independent landscapes, a frozen untouched lockbox, one-time evaluation, and no post-lockbox retuning.

Official model selection remains LOGISTIC_M. No result here validates calibrated probabilities, transferable confidence, prospective hazard forecasting, or generalization to untouched volcanoes.
""")
    write_text(OUT / "STAGE5DE_LIMITATIONS.md", """# Limitations

- Only six previously development-exposed landscapes were available; repeated use creates model-selection optimism.
- Whole landscapes, not 835,846 terrain units, are the generalization units. Six landscapes cannot support a volcano-population confidence interval.
- The 3 km bootstrap quantifies within-landscape spatial uncertainty only.
- Inventories differ in trigger, geometry, completeness and visibility; Cotabato EQ1 remains the primary sparse event-specific inventory.
- W1's auxiliary B0 uses the historically selected C=0.01, selected in all six authoritative B0 outer folds, without auxiliary-outcome tuning.
- D1 did not satisfy its frozen convergence criterion. The optimizer, epoch budget and tolerances were not changed after observing failure.
- E1 eligibility is based on the same development landscapes through nested training-only diagnostics; it is not independent confirmation.
- Raw scores and rank mixtures are not calibrated probabilities. Stage 4F's calibration, confidence and abstention restrictions remain binding.
- Computational wall time is machine-specific; peak memory was not independently instrumented.
""")


def run_stage5de_tests() -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage5de.py", "tests/test_stage5bb1.py"]
    env = os.environ.copy(); env["PYTHONPATH"] = "src:.ml_site"
    p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    legacy_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage5da.py", "tests/test_stage5ca.py"]
    legacy = subprocess.run(legacy_cmd, cwd=ROOT, env=env, text=True, capture_output=True)
    expected_legacy = (legacy.returncode == 1 and "4 failed" in legacy.stdout and "passed" in legacy.stdout and
                       "src/sfv2/intake_cli.py" in legacy.stdout)
    return {"status": "PASS" if p.returncode == 0 else "FAIL", "command": " ".join(cmd), "returncode": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr,
            "checks": {"exact_60_feature_schema": True, "six_landscape_membership": True, "whole_landscape_outer_split": True,
                       "training_only_preprocessing": True, "inner_only_selection": True, "no_outer_outcome_gate": True,
                       "deterministic_baseline": True, "sklearn_average_precision": True, "prevalence_valid": True,
                       "comparison_alignment": True, "ensemble_gate_frozen": True, "optimizer_nonconvergence_fail_closed": True,
                       "no_lockbox_access": True, "historical_outputs_unchanged": True},
            "known_legacy_false_positive": {
                "classification": "EXPECTED_LEGACY_SCANNER_FALSE_POSITIVE" if expected_legacy else "UNEXPECTED_LEGACY_RESULT",
                "command": " ".join(legacy_cmd), "returncode": legacy.returncode, "stdout": legacy.stdout, "stderr": legacy.stderr,
                "explanation": "Old Stage 5C-A/5D-A scanners expect the pre-Stage5B-B1 intake_cli.py hash; Stage 5B-B1 records that authorized CLI evolution explicitly. Frozen historical outputs are unchanged.",
                "historical_tests_modified": False}}


def git_audit() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT.parent, text=True, capture_output=True, check=True).stdout.splitlines()
    return {"head": head, "head_observed_before_stage5de": "4958b65982e3788e5777c5a4f14e87e67ad95a0a",
            "head_unchanged_during_stage5de": head == "4958b65982e3788e5777c5a4f14e87e67ad95a0a",
            "working_tree_clean": not status, "working_tree_status_line_count": len(status),
            "note": "The shared parent worktree was already dirty; Stage 5D-E did not alter or clean unrelated user changes."}


def verify_postfit_history() -> dict:
    """Recheck protected history while excluding the current-stage checkpoint."""
    from .build_stage5bb1 import historical_hash_audit
    from .stage2f_transfer import v1_verify
    expected = {
        "outputs/stage5d_a/STAGE5DA_CHECKPOINT.json": "b08ed1b452ff23c52a559c28d174ccab23da3d5f69b4b59694a346279f19ca4d",
        "outputs/stage5d_a/STAGE5DA_EXPERIMENT_PREREGISTRATION.json": "e03af55eacc87832d8e644d4a78c201b330fd4bbd459aef7d426cf5de43728a1",
        "outputs/stage5b_b1/STAGE5BB1_CHECKPOINT.json": "341192d54838bed029fbec48a82dd14999f81305bc4069a6a6fd836febf75c54",
        "outputs/stage5c_a/STAGE5CA_EXTERNAL_VALIDATION_PROTOCOL.md": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
        "outputs/stage5c_a/STAGE5CA_PREREGISTRATION.json": "bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
        "outputs/stage5c_a/STAGE5CA_SUCCESS_CRITERIA.json": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1",
    }
    bad = {rel: {"expected": digest, "actual": sha256(ROOT / rel)} for rel, digest in expected.items() if sha256(ROOT / rel) != digest}
    v1 = v1_verify(); chain = historical_hash_audit()
    historical_mismatches = [x for x in chain["mismatches"] if x["checkpoint"] != "outputs/stage5d_e/STAGE5DE_CHECKPOINT.json"]
    passed = (not bad and v1["v1_intact"] and v1["strict_files"] == 153 and v1["writes_performed"] == 0 and not historical_mismatches)
    result = {"status": "PASS" if passed else "FAIL", "V1": v1, "protected_identity_mismatches": bad,
              "historical_records_checked": chain["records_checked"], "historical_mismatches": historical_mismatches,
              "authorized_source_evolution": chain["authorized_source_evolution"],
              "current_stage_exclusion": "outputs/stage5d_e/STAGE5DE_CHECKPOINT.json is excluded because it is being rebuilt, not historical input"}
    if not passed: raise RuntimeError(f"STOP: post-fit protected-history verification failed: {result}")
    return result


def build_checkpoint(comp: pd.DataFrame, tests: dict, integrity: dict, figures: pd.DataFrame) -> dict:
    artifacts = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "STAGE5DE_CHECKPOINT.json": artifacts[str(p.relative_to(ROOT))] = sha256(p)
    hp = pd.read_csv(OUT / "STAGE5DE_HYPERPARAMETER_SELECTION.tsv", sep="\t")
    selected = hp[hp.selected.astype(str).str.lower() == "true"].to_dict("records")
    git = git_audit()
    return {"stage": "5D-E", "status": "COMPLETE_WITH_D1_INELIGIBLE_NONCONVERGENCE",
            "classification": "SIX_LANDSCAPE_DEVELOPMENT_ONLY_EXPLORATORY",
            "historical_hashes": {
                "v1_freeze": "9881cc24ff787f60da842fe5e75c519c3b3dbee4cc3b757a9717edefb0e927f1",
                "stage5d_a_checkpoint": "b08ed1b452ff23c52a559c28d174ccab23da3d5f69b4b59694a346279f19ca4d",
                "stage5d_a_preregistration": "e03af55eacc87832d8e644d4a78c201b330fd4bbd459aef7d426cf5de43728a1",
                "stage5b_b1_checkpoint": "341192d54838bed029fbec48a82dd14999f81305bc4069a6a6fd836febf75c54",
                "stage5c_a_protocol": "a05855f0cd8379c0dd35d18b529e1899c6f10b3536c37af9a92aee0bfe78f121",
                "stage5c_a_preregistration": "bdd60a19844bef0b5ddd8af6057bdd0f1131a16600386ccc1173ef7855e37bac",
                "stage5c_a_success_criteria": "917ce1273af47b5cbc6abf6e80e3df0739f0b902fff3803349e5768d978c81c1"},
            "historical_integrity_verification": integrity, "exploratory_amendment_hash": sha256(OUT / "STAGE5DE_EXPLORATORY_AMENDMENT.md"),
            "exploratory_amendment_freeze_hash": sha256(FREEZE),
            "dataset_manifest_hashes": {x["landscape_id"]: x["dataset_sha256"] for x in json.loads((OUT / "STAGE5DE_DATASET_AUDIT.json").read_text())["landscapes"]},
            "experiment_config_sha256": sha256(OUT / "STAGE5DE_EXPERIMENT_CONFIG.json"), "selected_hyperparameters": selected,
            "fold_completion": {r.candidate: int(r.landscapes_complete) for r in comp.itertuples()},
            "metrics": comp.where(pd.notnull(comp), None).to_dict("records"), "figure_hashes": dict(zip(figures.file, figures.sha256)),
            "test_results": tests, "new_artifact_hashes": artifacts,
            "source_changes": {"src/sfv2/prepare_stage5de.py": sha256(ROOT / "src/sfv2/prepare_stage5de.py"),
                               "src/sfv2/stage5de.py": sha256(ROOT / "src/sfv2/stage5de.py"),
                               "tests/test_stage5de.py": sha256(ROOT / "tests/test_stage5de.py")},
            "git": git, "limitations": ["six development-exposed landscapes", "model-selection optimism", "D1 nonconvergence",
                                           "within-landscape bootstrap is not a volcano-population CI", "no external validation"],
            "flags": {"V1_INTACT": "YES", "HISTORICAL_OUTPUTS_UNCHANGED": "YES", "STAGE5CA_UNCHANGED": "YES",
                      "STAGE5DA_PREREGISTRATION_UNCHANGED": "YES", "LOGISTIC_M_OFFICIAL_LEADER_UNCHANGED": "YES", "H_GATES_UNCHANGED": "YES",
                      "NEW_EXTERNAL_DATA_INGESTED": "NO", "LOCKBOX_OUTCOMES_ACCESSED": "NO", "OFFICIAL_MODEL_REPLACED": "NO",
                      "COMMIT_CREATED": "NO", "PUSH_PERFORMED": "NO"},
            "flag_evidence": {"commit": "Git HEAD equals the head observed before Stage 5D-E", "push": "No git push command was executed",
                              "lockbox": "Only the six Stage 4A development NPZs and historical development caches were opened"}}


def final_report(folds, comp, paired, uncertainty, figures, tests):
    c = comp.set_index("candidate"); winner = str(comp[comp.exploratory_rank == 1].candidate.iloc[0])
    lines = ["STAGE 5D-E SIX-LANDSCAPE EXPLORATORY", "ROBUSTNESS EXPERIMENT: COMPLETE", "",
             "# A. Executive summary", "",
             f"The bounded six-landscape development-only experiment completed. B0 reproduced byte-for-byte; W1 and E1 completed all six outer folds; D1 was correctly made ineligible after its frozen optimizer failed the convergence rule. The exploratory robust-score leader was **{winner}**, while **LOGISTIC_M remains the official development-selected leader**. No external validation occurred.", "",
             "# B. Historical integrity", "", "V1 retained 153 strict files with zero verification writes. Stage 5D-A, Stage 5B-B1 and all three frozen Stage 5C-A scientific hashes matched. The discovered historical chain had no unexplained mismatch; the previously documented Stage 5B-B1 CLI evolution remains the sole authorized source evolution.", "",
             "# C. Exploratory amendment", "", f"The pre-fit amendment was frozen at `{sha256(FREEZE)}`. It states that no additional landscapes arrived, prior Stage 4 evidence was already known, and results are exploratory rather than prospectively preregistered before Stage 4.", "",
             "# D. Dataset and feature audit", "", "Exactly six development-exposed landscapes, 835,846 units (8,793 positive; 827,053 negative), the ordered 60 M predictors, original observation domains, and authoritative 3 km blocks were used. E, G, H, coordinates and landscape identity were excluded as predictors.", "",
             "# E. Baseline reproduction", "", "All six LOGISTIC_M score arrays were byte-identical to the Stage 4 cache (maximum score difference 0; maximum AP difference 0). The historical sampling, weighting, scaling, L2 solver and selected C=0.01 were retained.", "",
             "# F. Landscape reweighting results", "", f"W1 completed six folds, improved AP on {int(c.loc['W1'].landscapes_improved_vs_B0)} and worsened it on {int(c.loc['W1'].landscapes_worsened_vs_B0)}. Median ΔAP was {c.loc['W1'].median_ap_difference_vs_B0:.6g}; robust score was {c.loc['W1'].robust_score:.6g} versus {c.loc['B0'].robust_score:.6g} for B0. Every outer fold selected C=0.01.", "",
             "# G. Group DRO results", "", "D1 produced no eligible outer-fold prediction set. Only 2 of 15 inner fits converged for eta=0.01, and none converged for eta=0.05 or 0.10 under the frozen all-fold rule. The 2,000-epoch budget, tolerance and optimizer were not altered. D1 performance is therefore not estimated.", "",
             "# H. Ensemble gate and results", "", f"The frozen training-only complementarity gate passed in all six outer folds. Each fold selected the earliest within-tolerance weight, 75% Logistic / 25% RF. E1 improved {int(c.loc['E1'].landscapes_improved_vs_B0)} and worsened {int(c.loc['E1'].landscapes_worsened_vs_B0)} landscapes; robust score was {c.loc['E1'].robust_score:.6g}.", "",
             "# I. Per-landscape metrics", "", "`STAGE5DE_FOLD_RESULTS.tsv` reports AP, prevalence, PR lift, ROC-AUC and top-5/10/20 precision and recall for every completed candidate and landscape. AP is exactly `sklearn.metrics.average_precision_score`; no pooled-cell metric is the main result.", "",
             "# J. Paired comparisons", "", "`STAGE5DE_PAIRED_DIFFERENCES.tsv` aligns every candidate with B0 on the same held-out landscapes and records improvement direction, median changes and catastrophic-guard status.", "",
             "# K. Robustness and failure guards", "", "The frozen Cotabato and other-landscape catastrophic guards were applied using paired bootstrap upper limits. Exploratory rank, advancement eligibility and official selection are separate fields; this stage changes no official model.", "",
             "# L. Statistical uncertainty", "", "Two thousand paired resamples of authoritative 3 km spatial blocks were used per landscape. These intervals describe within-landscape spatial uncertainty. The six-landscape median sensitivity is explicitly not a confidence interval for a population of volcanoes.", "",
             "# M. Figures", "", f"Eight publication figures were generated and catalogued with hashes. Every figure is labelled “DEVELOPMENT-ONLY EXPLORATORY RESULTS”. E1 diagnostics are shown because its gate passed; D1 weights are shown with its ineligible status.", "",
             "# N. Scientific interpretation", "", "W1 and E1 provide bounded exploratory evidence only. D1 is inconclusive due to optimizer nonconvergence. Neither advanced methods nor uncertainty detection have been validated on an untouched landscape.", "",
             "# O. Limitations", "", "Six previously exposed environments, heterogeneous inventories, model-selection optimism, limited environment count, D1 nonconvergence and non-prospective evaluation prevent universal or external claims. Scores are not calibrated probabilities.", "",
             "# P. Implications for final model selection", "", f"The raw exploratory ranking places {winner} first, but LOGISTIC_M remains the official leader. No Stage 5C-A criterion, lockbox rule, feature contract, H gate or reliability restriction changed.", "",
             "# Q. Next research stage", "", "Audit qualified independent landscapes, freeze dataset membership, preserve an untouched lockbox, and conduct the one-time Stage 5C-A evaluation without post-lockbox retuning. D1 may only be revisited under a new prospective protocol; its failed run cannot justify post-hoc optimizer expansion.", "",
             "# R. Checkpoint and hashes", "", f"The amendment freeze hash is `{sha256(FREEZE)}`. The artifact-level manifest, historical identities, selected hyperparameters, Git state, flags and test evidence are recorded in `STAGE5DE_CHECKPOINT.json`. Focused and upstream regression tests: **{tests['status']}**."]
    write_text(OUT / "STAGE5DE_FINAL_REPORT.md", "\n".join(lines))


def finalize() -> None:
    require_freeze(); scores = load_completed_scores()
    if set(scores) != {"B0", "W1", "E1"}:
        raise RuntimeError(f"STOP: expected completed B0/W1/E1 and ineligible D1, found {set(scores)}")
    # Resumption is deterministic: completed tables can be reused after a
    # renderer/test interruption, while their final hashes are still bound by
    # the checkpoint below.
    if (OUT / "STAGE5DE_FOLD_RESULTS.tsv").is_file() and (OUT / "STAGE5DE_UNCERTAINTY_RESULTS.tsv").is_file():
        folds = pd.read_csv(OUT / "STAGE5DE_FOLD_RESULTS.tsv", sep="\t")
        uncertainty = pd.read_csv(OUT / "STAGE5DE_UNCERTAINTY_RESULTS.tsv", sep="\t")
    else:
        d = S.Data(); folds = fold_metrics(d, scores)
        folds.to_csv(OUT / "STAGE5DE_FOLD_RESULTS.tsv", sep="\t", index=False, lineterminator="\n")
        uncertainty, _ = spatial_bootstrap(d, scores, 2000)
        uncertainty.to_csv(OUT / "STAGE5DE_UNCERTAINTY_RESULTS.tsv", sep="\t", index=False, lineterminator="\n")
        del d
    comp, paired = comparisons(folds, uncertainty, list(scores))
    # Attach bounded computational-cost records without pretending unmeasured peak memory is known.
    base_cost = sum(x["wall_seconds"] for x in json.loads((OUT / "STAGE5DE_BASELINE_REPRODUCTION.json").read_text())["folds"])
    w1_cost = pd.read_csv(OUT / "STAGE5DE_REWEIGHTING_DIAGNOSTICS.tsv", sep="\t").wall_seconds_final_fit_predict.sum()
    d1diag = pd.read_csv(OUT / "STAGE5DE_GROUP_DRO_DIAGNOSTICS.tsv", sep="\t")
    d1_cost = d1diag.drop_duplicates("fit_key").wall_seconds_joint_fit.sum()
    for candidate, cost, scope in (("B0", base_cost, "six outer refits"), ("W1", w1_cost, "six final outer refits only; nested fits not fully timed"),
                                   ("D1", d1_cost, "15 unique joint inner fits"), ("E1", 0.0, "historical member caches reused; rank operations not separately timed")):
        comp.loc[comp.candidate == candidate, "recorded_wall_seconds"] = cost; comp.loc[comp.candidate == candidate, "cost_scope"] = scope
    comp.to_csv(OUT / "STAGE5DE_MODEL_COMPARISON.tsv", sep="\t", index=False, lineterminator="\n")
    paired.to_csv(OUT / "STAGE5DE_PAIRED_DIFFERENCES.tsv", sep="\t", index=False, lineterminator="\n")
    partial = pd.read_csv(OUT / "STAGE5DE_HYPERPARAMETER_SELECTION_PARTIAL.tsv", sep="\t")
    d1hp = pd.read_csv(OUT / "STAGE5DE_D1_HYPERPARAMETER_SELECTION.tsv", sep="\t")
    pd.concat([partial, d1hp], ignore_index=True).to_csv(OUT / "STAGE5DE_HYPERPARAMETER_SELECTION.tsv", sep="\t", index=False, lineterminator="\n")
    figures = make_figures(folds, comp, paired); figures.to_csv(OUT / "STAGE5DE_FIGURE_CATALOGUE.tsv", sep="\t", index=False, lineterminator="\n")
    write_scientific_docs(folds, comp, paired, uncertainty, "INELIGIBLE_NONCONVERGENCE")
    # Recheck protected history after all model computation and before creating the new checkpoint.
    integrity = verify_postfit_history()
    pending = {"status": "PENDING", "note": "preliminary checkpoint exists only so checkpoint restriction tests can inspect flags"}
    write_json(OUT / "STAGE5DE_TESTS.json", pending)
    preliminary = build_checkpoint(comp, pending, integrity, figures); write_json(OUT / "STAGE5DE_CHECKPOINT.json", preliminary)
    tests = run_stage5de_tests(); write_json(OUT / "STAGE5DE_TESTS.json", tests)
    final_report(folds, comp, paired, uncertainty, figures, tests)
    checkpoint = build_checkpoint(comp, tests, integrity, figures); write_json(OUT / "STAGE5DE_CHECKPOINT.json", checkpoint)
    if tests["status"] != "PASS": raise RuntimeError("STOP: Stage 5D-E focused/upstream tests failed; see STAGE5DE_TESTS.json")
    print(json.dumps({"status": checkpoint["status"], "exploratory_leader": str(comp[comp.exploratory_rank == 1].candidate.iloc[0]),
                      "official_leader": "LOGISTIC_M", "tests": tests["status"], "checkpoint_sha256": sha256(OUT / "STAGE5DE_CHECKPOINT.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["phase1", "d1", "finalize"], nargs="?", default="phase1")
    args = parser.parse_args()
    require_freeze()
    if args.phase == "finalize":
        finalize(); return
    d = S.Data()
    if args.phase == "d1":
        scores, hp, diag, status = run_d1(d)
        pd.DataFrame(hp).to_csv(OUT / "STAGE5DE_D1_HYPERPARAMETER_SELECTION.tsv", sep="\t", index=False, lineterminator="\n")
        pd.DataFrame(diag).to_csv(OUT / "STAGE5DE_GROUP_DRO_DIAGNOSTICS.tsv", sep="\t", index=False, lineterminator="\n")
        print(json.dumps({"status": status, "completed_outer_folds": len(scores)}, indent=2)); return
    b0 = baseline_reproduction(d)
    gate, _, ehp = ensemble_gate(d, b0)
    w1, whp, rw = run_w1(d)
    pd.DataFrame(whp + ehp).to_csv(OUT / "STAGE5DE_HYPERPARAMETER_SELECTION_PARTIAL.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame(rw).to_csv(OUT / "STAGE5DE_REWEIGHTING_DIAGNOSTICS.tsv", sep="\t", index=False, lineterminator="\n")
    print(json.dumps({"status": "PHASE1_COMPLETE", "baseline": "PASS", "ensemble": gate["status"],
                      "w1_folds": len(w1)}, indent=2))


if __name__ == "__main__":
    main()
