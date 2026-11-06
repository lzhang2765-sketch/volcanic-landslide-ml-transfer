"""Build Stage 4C-B2: Gen-2 structurally-aware piecewise ensemble reconstruction with nested Y-blind validation.

New outputs only (outputs/stage4c_b2).  Gen-1 (outputs/stage4c_b) and every earlier stage are read-only and hash-verified before and
after.  No model is trained, no MEGH / Delta_H, no H predictive performance, no Y, no Stage 4B residual, no lockbox.
"""
from __future__ import annotations

import heapq
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from . import stage4a_base as B
from . import stage4c_a_spec as H
from . import stage4c_b2_prereg as P2
from . import stage4c_b2_recon as G
from . import stage4c_b_recon as R
from .build_stage3a import scan_ml_h_artifacts
from .build_stage4a import identity_stats, write_columns
from .build_stage4c_b import PRIOR as PRIOR_GEN1, PRIOR_SHA as PRIOR_SHA_GEN1, redundancy, distributions
from .raw_store import sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

OUT = ROOT / "outputs/stage4c_b2"
QA = OUT / "STAGE4C_B2_QA"
HDS = OUT / "STAGE4C_B2_H_DATASET"
PRIOR = dict(PRIOR_GEN1, stage4c_b="outputs/stage4c_b/STAGE4C_B_CHECKPOINT.json")
PRIOR_SHA = dict(PRIOR_SHA_GEN1, stage4c_b="90835a71a53cb8a544a960ae7d1e36f5a3ea44e524e730cd6f62db22c6e4bd7c")
CORE = [c["name"] for c in H.CORE_H]
HCOL = {n: n.lower() for n in CORE}
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def _tsv(p: Path, df) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(p, sep="\t", index=False, lineterminator="\n", float_format="%.6g")


def gen1_state() -> dict:
    cp = json.loads((ROOT / PRIOR["stage4c_b"]).read_text())
    files = dict(cp["output_hashes"], **cp["h_dataset_partitions"])
    return {p: sha256(ROOT / p) for p in files}, files


# ------------------------------------------------------------------------------------------------ nested outer validation
def ensemble_predict(ctx, comps, train, test_idx):
    cache, preds, rej = {}, [], {}
    for (v, pct, rep) in G.scenarios(comps):
        fits = G.fit_arch(ctx, comps, train, v, pct, rep, cache)
        bad = [f.reason for f in fits if not f.ok and f.reason != "UNDERDETERMINED"]
        if bad:
            rej[bad[0]] = rej.get(bad[0], 0) + 1
            continue
        for off in P2.ENSEMBLE["tolerance_offset_sigma"]:
            preds.append(G.predict_pts(ctx, comps, fits, test_idx, off))
    return (np.stack(preds, axis=1) if preds else np.full((len(test_idx), 0), np.nan)), rej


def nested(ctx, arch_comps):
    c = ctx.ctrl
    sf = c.super_fold.to_numpy()
    sel_rows, res_rows = [], []
    for F in np.unique(sf):
        chosen, rows = G.select(ctx, arch_comps, sf != F)
        for r in rows:
            sel_rows.append(dict(landscape_id=ctx.lid, scope=f"OUTER_SUPERFOLD_{F}", **r))
        comps = arch_comps[chosen]
        for g in np.unique(c.outer_group[sf == F]):
            te = np.nonzero((c.outer_group == g).to_numpy())[0]
            train = (c.outer_group != g).to_numpy()
            pred, rej = ensemble_predict(ctx, comps, train, te)
            for i, idx in enumerate(te):
                p = pred[i][np.isfinite(pred[i])]
                res_rows.append(dict(landscape_id=ctx.lid, control_id=c.control_id.iloc[idx], quality=c.quality_class.iloc[idx], outer_group=g, super_fold=int(F),
                                     selected_architecture=chosen, z=c.z95.iloc[idx], pred_median=float(np.median(p)) if len(p) else np.nan,
                                     p05=float(np.percentile(p, 5)) if len(p) else np.nan, p95=float(np.percentile(p, 95)) if len(p) else np.nan,
                                     n_scenarios=len(p), rejected=json.dumps(rej)))
    v = pd.DataFrame(res_rows)
    v["residual"] = v.z - v.pred_median
    v["inside_band"] = (v.z >= v.p05) & (v.z <= v.p95)
    return pd.DataFrame(sel_rows), v


# ------------------------------------------------------------------------------------------------ production
def fill_depth(z: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Priority-flood depression depth (cells outside `ok` and the array edge drain)."""
    h, w = z.shape
    filled = np.where(ok, np.inf, z)
    done = ~ok
    heap = []
    edge = ok & (ndimage.binary_dilation(~ok, np.ones((3, 3))) | np.pad(np.zeros((h - 2, w - 2), bool), 1, constant_values=True))
    for r, c in zip(*np.nonzero(edge)):
        filled[r, c] = z[r, c]
        done[r, c] = True
        heapq.heappush(heap, (z[r, c], r, c))
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        e, r, c = heapq.heappop(heap)
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not done[rr, cc]:
                done[rr, cc] = True
                filled[rr, cc] = max(z[rr, cc], e)
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return np.where(ok, filled - z, 0.0)


def production(ctx, arch_comps):
    c = ctx.ctrl
    chosen, sel_rows = G.select(ctx, arch_comps, np.ones(len(c), bool))
    comps = arch_comps[chosen]
    allm = np.ones(len(c), bool)
    cache, scen, fits_all = {}, [], []
    man = []
    for j, (v, pct, rep) in enumerate(G.scenarios(comps)):
        fits = G.fit_arch(ctx, comps, allm, v, pct, rep, cache)
        bad = [f.reason for f in fits if not f.ok and f.reason != "UNDERDETERMINED"]
        for off in P2.ENSEMBLE["tolerance_offset_sigma"]:
            sid = f"{ctx.lid[:4]}_{chosen.split('_')[0]}_{len(man):04d}"
            man.append(dict(landscape_id=ctx.lid, architecture=chosen, scenario_id=sid, structural_variant=v, summit_percentile=pct, subsample_replicate=rep,
                            tolerance_offset_sigma=off, accepted=not bad, physical_rejection_reason=bad[0] if bad else "",
                            component_sigmas_m="|".join(f"{f.sigma:.1f}" for f in fits), component_edf="|".join(f"{f.edf:.1f}" for f in fits),
                            component_params="|".join(json.dumps({k: (v_ if not isinstance(v_, (np.ndarray, list, tuple)) else "...") for k, v_ in f.par.items()
                                                                   if k in ("p", "K", "ue", "B")}) for f in fits)))
            if not bad:
                scen.append((fits, off))
    man = pd.DataFrame(man)
    if len(man):
        man["weight"] = np.where(man.accepted, 1.0 / max(int(man.accepted.sum()), 1), 0.0)

    h, w = ctx.X.shape
    dctrl = cKDTree(c[["x", "y"]].to_numpy()).query(np.c_[ctx.X.ravel(), ctx.Y.ravel()])[0].reshape(h, w)
    sup = ctx.app & ctx.valid90 & ~ctx.excl & (dctrl <= P2.SUPPORT["max_control_distance_m"])
    W = np.stack([k["w"] for k in comps])
    dominant = np.argmax(W, axis=0)
    for i, k in enumerate(comps):
        if k["family"].split(":")[0].startswith(("RADIAL", "CALDERA")):
            cm = k["cmask"]
            cents = [k["centre"]] if k["centre"] else [(cc.volcano_name, cc.x, cc.y) for cc in ctx.centres.itertuples()]
            rr = np.min([np.hypot(ctx.X - ce[1], ctx.Y - ce[2]) for ce in cents], axis=0)
            rc = np.min([np.hypot(c.x[cm] - ce[1], c.y[cm] - ce[2]) for ce in cents], axis=0) if cm.sum() else np.array([0.0])
            sup &= ~((dominant == i) & (rr > np.percentile(rc, 95)))
    if not scen or not sup.any():
        return dict(chosen=chosen, sel_rows=sel_rows, comps=comps, man=man, ok=False, sup=sup)
    rows_, cols_ = np.nonzero(sup)
    r0, r1, c0, c1 = max(rows_.min() - 8, 0), min(rows_.max() + 9, h), max(cols_.min() - 8, 0), min(cols_.max() + 9, w)
    sub = sup[r0:r1, c0:c1]
    Xs, Ys = ctx.X[r0:r1, c0:c1], ctx.Y[r0:r1, c0:c1]
    Ws = W[:, r0:r1, c0:c1]
    u = ctx.L.units
    ur, uc = u.ur.to_numpy(), u.uc.to_numpy()
    usup = sup[ur, uc]
    uix = np.nonzero(usup)[0]
    ul = (ur[uix] - r0, uc[uix] - c0)
    srt = np.sort(Ws, axis=0)
    band = sub & (srt[-2] >= 0.10) if len(comps) > 1 else np.zeros_like(sub)
    J = len(scen)
    Zs = np.empty((int(sub.sum()), J), np.float32)
    Zu, Su, Ru = (np.empty((len(uix), J), np.float32) for _ in range(3))
    Sk_band = np.empty((len(comps), int(band.sum()), J), np.float32)
    lap_med = None
    comp_cache = {}
    for j, (fits, off) in enumerate(scen):
        num = np.zeros(sub.shape)
        den = np.zeros(sub.shape)
        for i, (k, f) in enumerate(zip(comps, fits)):
            if not f.ok:
                Sk_band[i, :, j] = np.nan
                continue
            key = (i, id(f))
            if key not in comp_cache:
                comp_cache.clear() if len(comp_cache) > 64 else None
                comp_cache[key] = f.predict(Xs, Ys)
            Sk = comp_cache[key] + off * f.sigma
            num += Ws[i] * Sk
            den += Ws[i]
            Sk_band[i, :, j] = Sk[band]
        Zg = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
        Zf = R.nearest_fill(Zg, np.isfinite(Zg))
        Zs[:, j] = Zg[sub]
        Zu[:, j] = Zg[ul]
        Su[:, j] = R.horn(Zf, URES)[ul]
        Ru[:, j] = R.relief(Zf, 450 / URES)[ul]
    zq = np.percentile(Zs, [25, 50, 75], axis=1)
    zmed = np.full((h, w), np.nan)
    ziqr = np.full((h, w), np.nan)
    zmed[r0:r1, c0:c1][sub] = zq[1]
    ziqr[r0:r1, c0:c1][sub] = zq[2] - zq[0]
    fin = sup & np.isfinite(zmed)
    p02, p98 = np.percentile(zmed[fin], [2, 98])
    R_L = float(p98 - p02)
    area = float(fin.sum() * URES * URES)
    Lam = float(np.sqrt(area / np.pi))
    tau_L = float(np.median([f.sigma for fits, _ in scen for f in fits if f.ok]))
    qu = np.percentile(Zu, [5, 25, 50, 75, 95], axis=1)
    zmod_u = ctx.z90[ur, uc]
    slope_mod = np.where(np.isfinite(ctx.z90), R.horn(R.nearest_fill(ctx.z90, np.isfinite(ctx.z90)), URES), np.nan)
    relief_mod = np.where(np.isfinite(ctx.z90), R.relief(R.nearest_fill(ctx.z90, np.isfinite(ctx.z90)), 450 / URES), np.nan)
    n_u = len(u)
    full = lambda a: (lambda o: (o.__setitem__(uix, a), o)[1])(np.full(n_u, np.nan))
    zr_med, zr_p05, zr_p25, zr_p75, zr_p95 = (full(q) for q in qu)
    sl_med, rl_med = full(np.median(Su, axis=1)), full(np.median(Ru, axis=1))
    d_u = cKDTree(c[["x", "y"]].to_numpy()).query(u[["centroid_x", "centroid_y"]].to_numpy())[0]
    ok = usup & np.isfinite(zr_med)
    hv = {}
    with np.errstate(invalid="ignore"):
        hv["H_REMOVED_FRACTION"] = np.where(ok, (zr_med - zmod_u) / R_L, np.nan)
        hv["H_SLOPE_CHANGE_DEG"] = np.where(ok, slope_mod[ur, uc] - sl_med, np.nan)
        hv["H_RELIEF_CHANGE_FRACTION"] = np.where(ok, (relief_mod[ur, uc] - rl_med) / R_L, np.nan)
        hv["H_REFERENCE_RELATIVE_HEIGHT"] = np.where(ok, (zr_med - p02) / R_L, np.nan)
        hv["H_SCENARIO_SPREAD_FRACTION"] = np.where(ok, (zr_p75 - zr_p25) / R_L, np.nan)
        hv["H_CONTROL_DISTANCE_FRACTION"] = np.where(ok, d_u / Lam, np.nan)
        qa = dict(qa_h_removed_fraction_p05=np.where(ok, (zr_p05 - zmod_u) / R_L, np.nan), qa_h_removed_fraction_p95=np.where(ok, (zr_p95 - zmod_u) / R_L, np.nan),
                  qa_h_reference_relative_height_p05=np.where(ok, (zr_p05 - p02) / R_L, np.nan), qa_h_reference_relative_height_p95=np.where(ok, (zr_p95 - p02) / R_L, np.nan))
    avail = ok & np.all(np.stack([np.isfinite(v) for v in hv.values()]), axis=0)
    young = np.isin(u.source_unit_code.to_numpy(), [])  # no mapped young deposits inside support (as Gen-1)
    neg_frac = float(((zr_med - zmod_u)[avail & ~young] < -tau_L).mean())
    sl = sl_med[avail]

    # ---- boundary + physical QA on the scenario-median surface
    zm_sub = zmed[r0:r1, c0:c1]
    zf = R.nearest_fill(zm_sub, np.isfinite(zm_sub))
    slope_med = R.horn(zf, URES)
    lap = np.abs(ndimage.laplace(zf)) / URES ** 2
    interior = sub & ~band
    bq = dict(n_components=len(comps), band_cells=int(band.sum()))
    if band.any():
        Sb = np.nanmedian(Sk_band, axis=2)
        top2 = np.argsort(Ws[:, band], axis=0)[-2:]
        a, b = Sb[top2[1], np.arange(band.sum())], Sb[top2[0], np.arange(band.sum())]
        step = np.abs(a - b) / R_L
        step = step[np.isfinite(step)]
        bq.update(seam_step_median=float(np.median(step)), seam_step_p95=float(np.percentile(step, 95)),
                  slope_excess_deg=float(np.percentile(slope_med[band], 99) - np.percentile(slope_med[interior], 99)),
                  ridge_trough_fraction=float((lap[band] > np.percentile(lap[interior], 99)).mean()),
                  spread_ratio=float(np.median(ziqr[r0:r1, c0:c1][band]) / max(np.median(ziqr[r0:r1, c0:c1][interior]), 1e-9)))
        T = P2.BOUNDARY_QA
        bq["pass_seam_step"] = bq["seam_step_median"] <= 0.10 and bq["seam_step_p95"] <= 0.25
        bq["pass_slope_excess"] = bq["slope_excess_deg"] <= 5.0
        bq["pass_ridge_trough"] = bq["ridge_trough_fraction"] <= 0.05
        bq["pass_spread_ratio"] = bq["spread_ratio"] <= 2.0
        bq["boundary_qa_pass"] = all(bq[k] for k in ("pass_seam_step", "pass_slope_excess", "pass_ridge_trough", "pass_spread_ratio"))
    else:
        bq.update(seam_step_median=np.nan, seam_step_p95=np.nan, slope_excess_deg=np.nan, ridge_trough_fraction=np.nan, spread_ratio=np.nan,
                  boundary_qa_pass=True, note="single component: no internal boundary")
    depth = fill_depth(np.where(sub, zf, np.nan), sub)
    pit_frac = float((depth[sub] > tau_L).mean())
    bq["pit_fraction"] = pit_frac
    bq["physical_qa_pass"] = pit_frac <= 0.01 and not (sup & ctx.cal).any()
    grids = dict(z_modern=ctx.z90.astype(np.float32), zref_median=zmed.astype(np.float32), zref_iqr=ziqr.astype(np.float32), support=sup,
                 removed_fraction=((zmed - ctx.z90) / R_L).astype(np.float32), spread_fraction=(ziqr / R_L).astype(np.float32),
                 control_distance_fraction=(dctrl / Lam).astype(np.float32), dominant_component=np.where(ctx.app, dominant, -1).astype(np.int16),
                 boundary_band=np.pad(band, ((r0, h - r1), (c0, w - c1))), caldera_interior=ctx.cal, x0=np.float64(ctx.L.transform.c),
                 y0=np.float64(ctx.L.transform.f), res=np.float64(URES))
    sem = {i: ("ESTIMATED_REMOVAL" if k["family"].split(":")[0].startswith(("RADIAL", "CALDERA")) else "MINIMUM_REMOVAL") for i, k in enumerate(comps)}
    dom_u = dominant[ur, uc]
    sem_u = np.where(usup, np.array([sem[i] for i in dom_u], dtype=object), "")
    fam_u = np.where(usup, np.array([comps[i]["ctype"] + ":" + comps[i]["family"] for i in dom_u], dtype=object), "")
    return dict(chosen=chosen, sel_rows=sel_rows, comps=comps, man=man, ok=True, sup=sup, R_L=R_L, P02=float(p02), Lambda=Lam, tau_L=tau_L,
                area_km2=area / 1e6, hv=hv, qa=qa, avail=avail, usup=usup, sup_frac=float(avail.mean()), neg_frac=neg_frac,
                max_slope=float(np.nanmax(sl)) if sl.size else np.nan, p99_slope=float(np.nanpercentile(sl, 99)) if sl.size else np.nan,
                bq=bq, grids=grids, sem_u=sem_u, fam_u=fam_u, J=J, edf=float(np.median([G.arch_edf(f) for f, _ in scen])))


# ------------------------------------------------------------------------------------------------ figures
def qa_figure(ctx, prod, val):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = prod["grids"]
    h, w = g["z_modern"].shape
    x0, y0 = float(g["x0"]) / 1000, float(g["y0"]) / 1000
    ext = [x0, x0 + w * 0.09, y0 - h * 0.09, y0]
    sup = g["support"]
    m = lambda a: np.where(sup, a, np.nan)
    c = ctx.ctrl
    vv = val.set_index("control_id")
    fig, ax = plt.subplots(3, 4, figsize=(26, 18))
    ax = ax.ravel()
    ax[0].imshow(g["z_modern"], extent=ext, cmap="terrain"); ax[0].set_title("1. Modern DEM (m)")
    comp = np.where(g["dominant_component"] >= 0, g["dominant_component"], np.nan).astype(float)
    ax[1].imshow(comp, extent=ext, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax[1].set_title("2. Geology-constrained components:\n" + "; ".join(f"{i}={k['ctype']}:{k['family']}" for i, k in enumerate(prod["comps"]))[:200], fontsize=8)
    ax[2].imshow(g["z_modern"], extent=ext, cmap="gray")
    for q, col in (("Q1", "tab:red"), ("Q2", "tab:blue")):
        s = c[c.quality_class == q]
        ax[2].scatter(s.x / 1000, s.y / 1000, s=3, c=col, label=f"{q} ({len(s)})")
    if len(ctx.centres):
        ax[2].scatter(ctx.centres.x / 1000, ctx.centres.y / 1000, marker="^", s=90, c="yellow", edgecolors="k", label="registered centre")
    ax[2].legend(fontsize=7); ax[2].set_title("3. Q1/Q2 controls")
    ax[3].imshow(g["z_modern"], extent=ext, cmap="gray")
    sc = ax[3].scatter(c.x / 1000, c.y / 1000, c=c.super_fold, cmap="tab10", s=3)
    ax[3].set_title("4. Outer held-out groups (colour = super-fold; every control held out once)")
    for i, (t, a, cm, lim) in enumerate([("5. Scenario-median reference (m)", m(g["zref_median"]), "terrain", None),
                                         ("6. Reference - modern (m)", m(g["zref_median"] - g["z_modern"]), "RdBu", (-500, 500)),
                                         ("7. Scenario spread IQR/R_L", m(g["spread_fraction"]), "viridis", (0, np.nanpercentile(g["spread_fraction"][sup], 99))),
                                         ("8. Support mask", sup.astype(float) + 2 * g["caldera_interior"], "Set2", (0, 3))], start=4):
        im = ax[i].imshow(a, extent=ext, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, interpolation="nearest")
        fig.colorbar(im, ax=ax[i], fraction=0.04)
        ax[i].set_title(t)
    ax[8].imshow(m(g["zref_median"]), extent=ext, cmap="terrain")
    if g["boundary_band"].any():
        ax[8].contour(np.flipud(g["boundary_band"].astype(float)), levels=[0.5], colors="red", extent=ext, origin="lower")
    ax[8].set_title("9. Component boundary band (red)")
    ax[9].imshow(g["z_modern"], extent=ext, cmap="gray")
    rr = vv.reindex(c.control_id).residual.to_numpy() / prod["R_L"]
    sc = ax[9].scatter(c.x / 1000, c.y / 1000, c=rr, cmap="RdBu", vmin=-0.3, vmax=0.3, s=4)
    fig.colorbar(sc, ax=ax[9], fraction=0.04)
    ax[9].set_title("10. Outer held-out residual / R_L")
    ax[10].hist(np.clip(rr[np.isfinite(rr)], -0.6, 0.6), bins=60); ax[10].set_title("Held-out residual / R_L distribution")
    ax[11].axis("off")
    ax[11].text(0, 1, prod["summary_text"], va="top", family="monospace", fontsize=10)
    fig.suptitle(f"Stage 4C-B2 Gen-2 reconstruction QA - {ctx.lid} (Y-blind; no landslide data)", fontsize=14)
    QA.mkdir(parents=True, exist_ok=True)
    p = QA / f"{ctx.lid}_gen2_qa.png"
    fig.savefig(p, dpi=70, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    p2 = QA / f"{ctx.lid}_gen2_grids.npz"
    np.savez_compressed(p2, **g)
    return [p, p2]


# ------------------------------------------------------------------------------------------------ main
def main() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    prior = {k: sha256(ROOT / v) for k, v in PRIOR.items()}
    if prior != PRIOR_SHA:
        raise RuntimeError(f"STOP: authoritative checkpoint changed: {[k for k in prior if prior[k] != PRIOR_SHA[k]]}")
    g1_before, g1_files = gen1_state()
    if g1_before != g1_files:
        raise RuntimeError("STOP: Gen-1 outputs differ from the Stage 4C-B checkpoint")
    s4a = json.loads((ROOT / PRIOR["stage4a"]).read_text())
    meg_before = {p: sha256(ROOT / p) for p in s4a["meg_partitions"]}
    assert meg_before == s4a["meg_partitions"]
    assert P2.GATES == H.VALIDATION_GATES["gates"], "STOP: gates differ from Stage 4C-A"
    prereg = dict(module="src/sfv2/stage4c_b2_prereg.py", sha256=sha256(ROOT / "src/sfv2/stage4c_b2_prereg.py"),
                  recorded_before_any_gen2_fit="0ba8c187cfdd4c06c099a382eaf0431486e611e9e81de49c182cf3ad395ff330 (2026-10-07T07:58:12Z)",
                  content={k: getattr(P2, k) for k in dir(P2) if k.isupper()})
    _json(OUT / "STAGE4C_B2_PREREGISTRATION.json", prereg)

    res = {}
    for lid in P2.RECONSTRUCTED:
        t = time.time()
        ctx = G.load_ctx(lid)
        arch = {a: G.components(ctx, a) for a in P2.CANDIDATES[lid]}
        sel, val = nested(ctx, arch)
        prod = production(ctx, arch)
        res[lid] = dict(ctx=ctx, arch=arch, sel=sel, val=val, prod=prod)
        print(f"[4C-B2] {lid}: selected {prod['chosen']} ({time.time() - t:.0f} s)", flush=True)

    comp_rows, cc_rows, lib_rows, nv_rows, sa_rows, bq_rows, vrows, st_rows, g12 = [], [], [], [], [], [], [], [], []
    man = []
    status = {"NOTO_PENINSULA_2024": "AGE_PROXY_ONLY"}
    gen1 = pd.read_csv(ROOT / "outputs/stage4c_b/STAGE4C_B_RECONSTRUCTION_VALIDATION.tsv", sep="\t")
    gen1 = gen1[gen1.scope == "LANDSCAPE"].set_index("landscape_id")
    gen1_comp = pd.read_csv(ROOT / "outputs/stage4c_b/STAGE4C_B_CONTROL_AUDIT.tsv", sep="\t").groupby("landscape_id").size()
    gen1_man = pd.read_csv(ROOT / "outputs/stage4c_b/STAGE4C_B_SCENARIO_MANIFEST.tsv", sep="\t", usecols=["landscape_id", "family", "trend_order"])
    for lid, r in res.items():
        ctx, prod, val, sel = r["ctx"], r["prod"], r["val"], r["sel"]
        c = ctx.ctrl
        for a, comps in r["arch"].items():
            for i, k in enumerate(comps):
                lib_rows.append(dict(landscape_id=lid, architecture=a, component_index=i, component_type=k["ctype"], family=k["family"], domain_rule=k["rule"],
                                     centre=k["centre"][0] if k["centre"] else "", fit_controls=int(k["cmask"].sum()), domain_cells=int(k["gmask"].sum()),
                                     complexity_note=P2.FAMILIES[k["family"].split(":")[0]].get("parameters", P2.FAMILIES[k["family"].split(":")[0]].get("complexity", "")),
                                     selected_in_production=a == prod["chosen"]))
        for i, k in enumerate(prod["comps"]):
            cm = k["cmask"]
            ck = c[cm]
            comp_rows.append(dict(component_id=f"{lid[:4]}_{prod['chosen'].split('_')[0]}_{i}", landscape_id=lid, architecture=prod["chosen"], component_type=k["ctype"],
                                  family=k["family"], geological_units=";".join(sorted(ck.geological_unit.unique())),
                                  centre=k["centre"][0] if k["centre"] else "", centre_x=k["centre"][1] if k["centre"] else np.nan,
                                  centre_y=k["centre"][2] if k["centre"] else np.nan, spatial_domain=k["rule"], domain_area_km2=round(k["gmask"].sum() * 0.0081, 2),
                                  temporal_interpretation={"RADIAL_EDIFICE": "younger constructional edifice", "VOLCANIC_APRON": "distal constructional apron/plateau",
                                                           "CALDERA_RIM_SECTOR": "pre-collapse outer flank truncated by the caldera rim",
                                                           "OLDER_VOLCANIC_TERRAIN": "older dissected volcanic terrain (minimum removal)",
                                                           "PLATEAU": "pumice-fill plateau top", "PRESERVED_REMNANT": "preserved-remnant envelope (minimum removal)",
                                                           "YOUNGER_VOLCANIC_TERRAIN": "younger constructional terrain"}.get(k["ctype"], ""),
                                  reconstruction_family=k["family"], control_count=int(cm.sum()),
                                  controls_per_sector=";".join(f"{a}={b}" for a, b in ck.groupby("sector").size().items()) if (ck.sector >= 0).any() else "",
                                  controls_per_unit=";".join(f"{a}={b}" for a, b in ck.geological_unit.value_counts().items()),
                                  domain_within_3km_of_component_control=round(float((cKDTree(ck[["x", "y"]].to_numpy()).query(np.c_[ctx.X[k["gmask"]], ctx.Y[k["gmask"]]])[0] <= 3000).mean()), 4) if len(ck) and k["gmask"].any() else 0.0,
                                  support_rule="applicable geology, outside caldera/vent, <= 3 km from a control" + ("; r <= P95 control radius" if k["family"].startswith(("RADIAL", "CALDERA")) else ""),
                                  combination_rule=P2.COMBINATION["rule"]))
            for idx in np.nonzero(cm)[0]:
                cc_rows.append(dict(landscape_id=lid, control_id=c.control_id.iloc[idx], component_index=i, component_type=k["ctype"], quality=c.quality_class.iloc[idx],
                                    geological_unit=c.geological_unit.iloc[idx], sector=int(c.sector.iloc[idx]), outer_group=c.outer_group.iloc[idx],
                                    super_fold=int(c.super_fold.iloc[idx]), x=c.x.iloc[idx], y=c.y.iloc[idx], z95=c.z95.iloc[idx]))
        nv_rows.append(sel)
        nv_rows.append(pd.DataFrame(prod["sel_rows"]).assign(landscape_id=lid, scope="PRODUCTION_ALL_CONTROLS"))
        fs = sel[sel.selected].groupby("architecture").size()
        sa_rows.append(dict(landscape_id=lid, production_architecture=prod["chosen"], outer_superfold_selections=";".join(f"{a}={n}" for a, n in fs.items()),
                            selection_stability=float(fs.get(prod["chosen"], 0) / fs.sum()), n_components=len(prod["comps"]),
                            effective_dof_median=prod.get("edf", np.nan), scenarios=prod.get("J", 0),
                            pusuk_nangi=P2.PUSUK_NANGI if lid.startswith("LOMBOK") else ""))
        man.append(prod["man"])
        G_ = P2.GATES
        if prod["ok"]:
            med = float(np.median(np.abs(val.residual.dropna())) / prod["R_L"])
            bias = float(val.residual.dropna().mean() / prod["R_L"])
            band = float(val.inside_band.mean())
        else:
            med = bias = band = np.nan
        gates = dict(median_abs_residual=(med, med <= G_["median_abs_residual_fraction_max"]), abs_bias=(abs(bias), abs(bias) <= G_["abs_bias_fraction_max"]),
                     holdout_band=(band, band >= G_["holdout_band_fraction_min"]), admissible_controls=(len(c), len(c) >= G_["min_admissible_controls"]),
                     support_fraction=(prod.get("sup_frac", 0.0), prod.get("sup_frac", 0.0) >= G_["min_support_fraction_of_units"]),
                     negative_removal=(prod.get("neg_frac", np.nan), prod.get("neg_frac", 1.0) <= G_["negative_removal_fraction_max_outside_mapped_young_deposits"]),
                     max_reference_slope=(prod.get("max_slope", np.nan), prod.get("max_slope", 99.0) <= G_["max_reconstructed_slope_deg_constructional"]))
        codes = dict(median_abs_residual="HIGH_RESIDUAL", abs_bias="HIGH_BIAS", holdout_band="LOW_SCENARIO_COVERAGE", admissible_controls="INSUFFICIENT_CONTROLS",
                     support_fraction="LOW_SUPPORT", negative_removal="EXCESS_NEGATIVE_REMOVAL", max_reference_slope="IMPLAUSIBLE_SLOPE")
        fails = [codes[k] for k, (_, ok) in gates.items() if not ok]
        bq = prod.get("bq", {})
        if not bq.get("boundary_qa_pass", False):
            fails.append("BOUNDARY_QA_FAIL")
        if not bq.get("physical_qa_pass", False):
            fails.append("PHYSICAL_QA_FAIL")
        missing_pred = int(val.pred_median.isna().sum())
        if missing_pred:
            fails.append(f"UNPREDICTED_HELDOUT_CONTROLS_{missing_pred}")
        st = ("VERIFIED_GEN2_FULL" if prod["sup_frac"] >= 0.90 else "VERIFIED_GEN2_PARTIAL") if not fails else "FAILED_GEN2"
        status[lid] = st
        prod["status"], prod["fails"], prod["gates"] = st, fails, gates
        prod["summary_text"] = (f"{lid}\narchitecture: {prod['chosen']}\nstatus: {st}\nfailures: {', '.join(fails) or 'none'}\nR_L {prod['R_L']:.0f} m  Lambda {prod['Lambda'] / 1000:.1f} km\n"
                                + "\n".join(f"{k}: {v[0]:.4g} {'PASS' if v[1] else 'FAIL'}" for k, v in gates.items())
                                + f"\nboundary QA: {bq.get('boundary_qa_pass')}  pits: {bq.get('pit_fraction', np.nan):.4f}\nP99 slope {prod.get('p99_slope', np.nan):.1f} deg")
        vrows.append(dict(landscape_id=lid, architecture=prod["chosen"], heldout_controls=len(val), heldout_groups=val.outer_group.nunique(), unpredicted=missing_pred,
                          R_L_m=prod.get("R_L"), tau_L_m=prod.get("tau_L"), median_abs_residual_fraction=med, signed_bias_fraction=bias, abs_bias_fraction=abs(bias),
                          holdout_band_fraction=band, admissible_controls=len(c), support_fraction_of_units=prod.get("sup_frac"), negative_removal_fraction=prod.get("neg_frac"),
                          max_reference_slope_deg=prod.get("max_slope"), p99_reference_slope_deg=prod.get("p99_slope"),
                          **{f"gate_{k}": "PASS" if ok else "FAIL" for k, (_, ok) in gates.items()}, all_gates_pass=all(ok for _, ok in gates.values()),
                          boundary_qa_pass=bq.get("boundary_qa_pass"), physical_qa_pass=bq.get("physical_qa_pass"),
                          median_abs_residual_fraction_Q1=float(np.median(np.abs(val[val.quality == "Q1"].residual.dropna())) / prod["R_L"]),
                          median_abs_residual_fraction_Q2=float(np.median(np.abs(val[val.quality == "Q2"].residual.dropna())) / prod["R_L"])))
        bq_rows.append(dict(landscape_id=lid, architecture=prod["chosen"], **bq))
        st_rows.append(dict(landscape_id=lid, architecture=prod["chosen"], semantics="+".join(sorted(set(prod["sem_u"][prod["usup"]]))), status=st,
                            failure_codes="|".join(fails), verified_gen2_h_capable=st.startswith("VERIFIED"), core_h_released=st.startswith("VERIFIED")))
        g1 = gen1.loc[lid]
        g12.append(dict(landscape_id=lid, metric_source="GEN1", architecture=g1.family, median_abs_residual_fraction=g1.median_abs_residual_fraction,
                        signed_bias_fraction=g1.signed_bias_fraction, holdout_band_fraction=g1.holdout_band_fraction, support_fraction=g1.support_fraction_of_units,
                        negative_removal_fraction=g1.negative_removal_fraction, max_slope_deg=g1.max_reference_slope_deg, p99_slope_deg=g1.p99_reference_slope_deg,
                        boundary_qa="n/a (Gen-1 hybrid not blended)", n_components=int(gen1_comp.get(lid, 1)), complexity="Gen-1 frozen families (<= 6 linear parameters per component)",
                        gates_passed=int(sum(g1[f"gate_{k}"] == "PASS" for k in codes)), status="FAILED_H"))
        g12.append(dict(landscape_id=lid, metric_source="GEN2", architecture=prod["chosen"], median_abs_residual_fraction=med, signed_bias_fraction=bias,
                        holdout_band_fraction=band, support_fraction=prod["sup_frac"], negative_removal_fraction=prod["neg_frac"], max_slope_deg=prod["max_slope"],
                        p99_slope_deg=prod["p99_slope"], boundary_qa=bq.get("boundary_qa_pass"), n_components=len(prod["comps"]),
                        complexity=f"effective dof ~ {prod['edf']:.1f}", gates_passed=int(sum(ok for _, ok in gates.values())), status=st))
    st_rows.append(dict(landscape_id="NOTO_PENINSULA_2024", architecture="AGE_HISTORY_PROXY", semantics="NONE", status="AGE_PROXY_ONLY", failure_codes="",
                        verified_gen2_h_capable=False, core_h_released=False))
    verified = [l for l in LANDSCAPES if status[l].startswith("VERIFIED")]
    n_ver = len(verified)
    unlocked = n_ver >= 3

    _tsv(OUT / "STAGE4C_B2_COMPONENTS.tsv", comp_rows)
    _tsv(OUT / "STAGE4C_B2_COMPONENT_CONTROLS.tsv", cc_rows)
    _tsv(OUT / "STAGE4C_B2_FAMILY_LIBRARY.tsv", lib_rows)
    _tsv(OUT / "STAGE4C_B2_NESTED_VALIDATION.tsv", pd.concat(nv_rows, ignore_index=True))
    _tsv(OUT / "STAGE4C_B2_SELECTED_ARCHITECTURES.tsv", sa_rows)
    _tsv(OUT / "STAGE4C_B2_SCENARIO_MANIFEST.tsv", pd.concat(man, ignore_index=True))
    _tsv(OUT / "STAGE4C_B2_BOUNDARY_QA.tsv", bq_rows)
    _tsv(OUT / "STAGE4C_B2_RECONSTRUCTION_VALIDATION.tsv", vrows)
    _tsv(OUT / "STAGE4C_B2_HOLDOUT_RESIDUALS.tsv", pd.concat([r["val"] for r in res.values()], ignore_index=True))
    _tsv(OUT / "STAGE4C_B2_GEN1_VS_GEN2.tsv", g12)
    _tsv(OUT / "STAGE4C_B2_RECONSTRUCTION_STATUS.tsv", st_rows)

    # ---- H dataset (released only for verified Gen-2 landscapes)
    frames, diag, files = {}, {}, {}
    for lid in LANDSCAPES:
        u = B.load(lid, "terrain_units/terrain_units", usecols=["terrain_unit_id"])
        df = pd.DataFrame(dict(landscape_id=lid, terrain_unit_id=u.terrain_unit_id.to_numpy()))
        r = res.get(lid)
        rel = lid in verified
        for cn in CORE:
            df[HCOL[cn]] = r["prod"]["hv"][cn] if (r and rel) else np.nan
        df["h_available"] = r["prod"]["avail"] if (r and rel) else False
        for q in ("qa_h_removed_fraction_p05", "qa_h_removed_fraction_p95", "qa_h_reference_relative_height_p05", "qa_h_reference_relative_height_p95"):
            df[q] = r["prod"]["qa"][q] if (r and rel) else np.nan
        df["qa_h_support"] = r["prod"]["usup"] if r else False
        df["qa_h_component_family"] = r["prod"]["fam_u"] if r else "AGE_HISTORY_PROXY"
        df["qa_h_semantics"] = r["prod"]["sem_u"] if r else "NONE"
        df["qa_h_landscape_status"] = status[lid]
        frames[lid] = df
        if r:
            d = df.copy()
            for cn in CORE:
                d[HCOL[cn]] = r["prod"]["hv"][cn]
            d["h_available"] = r["prod"]["avail"]
            diag[lid] = d
        p = HDS / f"{lid}.npz"
        write_columns(p, df)
        files[str(p.relative_to(ROOT))] = sha256(p)
        assert np.isnan(df[[HCOL[c] for c in CORE]].to_numpy()[~df.h_available.to_numpy()]).all()
    assert sum(len(f) for f in frames.values()) == B.EXPECTED["rows"]
    _json(HDS / "SCHEMA.json", dict(format="numpy .npz of .npy columns (sfv2.build_stage4a.read_columns)", key="terrain_unit_id (Stage 4A order)",
                                    core_h_columns={c: HCOL[c] for c in CORE}, availability="h_available",
                                    rules=["no Y column", "released only for VERIFIED_GEN2 landscapes", "NaN elsewhere; never zero-filled or imputed", "qa_* never predictors"]))
    relf = {l: frames[l] for l in verified}
    dist = pd.concat([distributions(relf, "RELEASED_VERIFIED_GEN2_H"), distributions(diag, "DIAGNOSTIC_UNRELEASED_GEN2")], ignore_index=True)
    _tsv(OUT / "STAGE4C_B2_H_DISTRIBUTIONS.tsv", dist)
    red = pd.concat([redundancy(relf, "RELEASED_VERIFIED_GEN2_H"), redundancy(diag, "DIAGNOSTIC_UNRELEASED_GEN2")], ignore_index=True)
    _tsv(OUT / "STAGE4C_B2_H_REDUNDANCY_WITH_M.tsv", red)
    rr = red[red.role == "RELEASED_VERIFIED_GEN2_H"] if len(red) else pd.DataFrame()
    sel_h = []
    for cn in CORE:
        r2 = float(rr[rr.feature == cn].pooled_r2_on_60_M.iloc[0]) if len(rr) and (rr.feature == cn).any() else np.nan
        a = bool(np.isfinite(r2) and r2 < 0.90)
        b = bool(n_ver > 0 and all(frames[l][HCOL[cn]].notna().any() for l in verified))
        sel_h.append(dict(feature=cn, pooled_r2_on_60_M=r2, criterion_a_r2_lt_0_90=a, criterion_b_available_in_every_verified_landscape=b,
                          criterion_c_reconstruction_validation_passes=n_ver > 0, selected_h=a and b and n_ver > 0, verified_landscapes="|".join(verified) or "none",
                          note="" if n_ver else "no verified Gen-2 landscape: SELECTED_H empty by the frozen rule"))
    seltab = pd.DataFrame(sel_h)
    _tsv(OUT / "STAGE4C_B2_SELECTED_H.tsv", seltab)

    qa_files = []
    for lid, r in res.items():
        if r["prod"]["ok"]:
            qa_files += qa_figure(r["ctx"], r["prod"], r["val"])

    v1_after = v1_verify()
    prior_after = {k: sha256(ROOT / v) for k, v in PRIOR.items()}
    g1_after, _ = gen1_state()
    meg_after = {p: sha256(ROOT / p) for p in s4a["meg_partitions"]}
    ml = [p for p in scan_ml_h_artifacts() if not p.startswith("outputs/stage4b/")]
    s = dict(res=res, status=status, verified=verified, n_ver=n_ver, unlocked=unlocked, vt=pd.DataFrame(vrows), sa=pd.DataFrame(sa_rows), comp=pd.DataFrame(comp_rows),
             bq=pd.DataFrame(bq_rows), g12=pd.DataFrame(g12), red=red, seltab=seltab, v1_before=v1_before, v1_after=v1_after, prior_ok=prior_after == PRIOR_SHA,
             g1_ok=g1_after == g1_files, meg_ok=meg_after == meg_before, ml=ml, st=pd.DataFrame(st_rows), lib=pd.DataFrame(lib_rows))
    (OUT / "STAGE4C_B2_REPORT.md").write_text(report(s))
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith(("STAGE4C_B2_CHECKPOINT", "STAGE4C_B2_DIAGNOSTICS")))
    import matplotlib
    import scipy
    cp = dict(schema="SFV2_STAGE4C_B2_CHECKPOINT_V1", as_of=P2.AS_OF, prior_checkpoints={PRIOR[k]: v for k, v in prior_after.items()},
              prior_checkpoints_unchanged=prior_after == PRIOR_SHA, gen1_outputs_unchanged=g1_after == g1_files, stage4a_meg_partitions_unchanged=meg_after == meg_before,
              preregistration=dict(path="outputs/stage4c_b2/STAGE4C_B2_PREREGISTRATION.json", module_sha256=prereg["sha256"]),
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage4c_b2.py", "stage4c_b2_recon.py", "stage4c_b2_prereg.py", "stage4c_b_recon.py", "build_stage4c_b.py")},
              software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
              random_seed=P2.SEED, candidate_architectures={l: list(v) for l, v in P2.CANDIDATES.items()}, complexity_rule=P2.COMPLEXITY_RULE, nested=P2.NESTED,
              ensemble=P2.ENSEMBLE, boundary_qa=P2.BOUNDARY_QA, physical_qa=P2.PHYSICAL_QA, validation_gates=P2.GATES,
              selected_architectures={l: r["prod"]["chosen"] for l, r in res.items()}, validation=vrows, statuses=status, verified_gen2_h_capable=verified,
              n_verified_gen2_h_capable=n_ver, megh_experiment_unlocked="YES" if unlocked else "NO", h_dataset_partitions=files,
              thresholds_changed=False, negative_removal_clipped=False, h_zero_filled=False, h_imputed=False, model_trained=False, megh_trained=False,
              delta_h_computed=False, h_predictive_performance_observed=False, y_loaded=False, y_used=False, stage4b_residuals_inspected=False,
              lockbox_accessed=False, stage4a_dataset_modified=False, gen1_modified=False, ml_artifacts_outside_stage4b=ml,
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")}, runtime_s=round(time.time() - t0),
              output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs})
    _json(OUT / "STAGE4C_B2_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE4C_B2_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD")))
    return dict(cp=cp, s=s)


URES = G.URES


def _md(df: pd.DataFrame, cols: list) -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r.get(c, "")) for c in cols) + " |" for _, r in df.iterrows())


def report(s: dict) -> str:
    vt, sa = s["vt"], s["sa"]
    gcols = [c for c in vt.columns if c.startswith("gate_")]
    return f"""# Stage 4C-B2 report - Gen-2 structurally-aware palaeolandscape reconstruction

**DEVELOPMENT ONLY.** No model trained, no MEGH, no Delta_H, no H predictive performance, no Y, no Stage 4B residuals, no lockbox. Gen-1 untouched.

## A. Executive result
Statuses: {json.dumps(s['status'])}. **N_VERIFIED_GEN2_H_CAPABLE = {s['n_ver']}**; **MEGH_EXPERIMENT_UNLOCKED = {'YES' if s['unlocked'] else 'NO'}**.

## C. Pre-registration
stage4c_b2_prereg.py (hash recorded before any Gen-2 fit). Complexity rule: {P2.COMPLEXITY_RULE['score']}; {P2.COMPLEXITY_RULE['rule']}.

## D/H. Selected architectures (nested; production selection over all super-folds)
{_md(sa, ['landscape_id', 'production_architecture', 'outer_superfold_selections', 'selection_stability', 'n_components', 'effective_dof_median', 'scenarios'])}

## Components of the selected architectures
{_md(s['comp'], ['component_id', 'component_type', 'family', 'geological_units', 'centre', 'spatial_domain', 'control_count', 'domain_within_3km_of_component_control'])}

## G/I. Nested outer Y-blind validation (frozen 4C-A gates)
{_md(vt, ['landscape_id', 'architecture', 'heldout_controls', 'heldout_groups', 'median_abs_residual_fraction', 'abs_bias_fraction', 'holdout_band_fraction', 'support_fraction_of_units', 'negative_removal_fraction', 'max_reference_slope_deg', 'p99_reference_slope_deg'])}

{_md(vt, ['landscape_id'] + gcols + ['all_gates_pass', 'boundary_qa_pass', 'physical_qa_pass'])}

## J. Boundary / physical QA
{_md(s['bq'], ['landscape_id', 'architecture', 'n_components', 'band_cells', 'seam_step_median', 'seam_step_p95', 'slope_excess_deg', 'ridge_trough_fraction', 'spread_ratio', 'boundary_qa_pass', 'pit_fraction', 'physical_qa_pass'])}

## K. Gen-1 vs Gen-2 (reconstruction quality only)
{_md(s['g12'], ['landscape_id', 'metric_source', 'architecture', 'median_abs_residual_fraction', 'signed_bias_fraction', 'holdout_band_fraction', 'support_fraction', 'negative_removal_fraction', 'max_slope_deg', 'p99_slope_deg', 'n_components', 'gates_passed', 'status'])}

## L-N. CORE_H release, redundancy, SELECTED_H
Released for: {', '.join(s['verified']) or 'none'}.
{_md(s['red'], ['role', 'feature', 'rows', 'pooled_r2_on_60_M', 'redundant_r2_ge_0_90']) if len(s['red']) else ''}

{_md(s['seltab'], ['feature', 'criterion_a_r2_lt_0_90', 'criterion_b_available_in_every_verified_landscape', 'selected_h', 'note'])}

## O-P. Status
{_md(s['st'], ['landscape_id', 'architecture', 'semantics', 'status', 'failure_codes'])}

## Q. Integrity
V1 {s['v1_before']['strict_files']}/{s['v1_before']['writes_performed']} -> {s['v1_after']['strict_files']}/{s['v1_after']['writes_performed']}; prior 3F-4C-B unchanged {s['prior_ok']};
Gen-1 outputs unchanged {s['g1_ok']}; Stage 4A partitions unchanged {s['meg_ok']}; ML artifacts outside 4B: {s['ml'] or 'none'}.
Pusuk/Nangi: {P2.PUSUK_NANGI}.
"""


if __name__ == "__main__":
    main()
