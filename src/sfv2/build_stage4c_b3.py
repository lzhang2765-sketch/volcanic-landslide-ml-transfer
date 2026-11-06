"""Build Stage 4C-B3: Gen-3 one-sided structurally-aware palaeosurface reconstruction (new outputs only).

Gen-1, Gen-2 and all earlier outputs are read-only and hash-verified.  The pre-registration JSON was frozen before any Gen-3 fit and is
verified (never rewritten) here.  No Y, no Stage 4B residual, no lockbox, no model training, no Delta_H, no modern-DEM lower bound.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from . import stage4a_base as B
from . import stage4c_a_spec as H
from . import stage4c_b2_recon as G2
from . import stage4c_b3_prereg as P3
from . import stage4c_b3_recon as G3
from . import stage4c_b_recon as R
from .build_stage3a import scan_ml_h_artifacts
from .build_stage4a import write_columns
from .build_stage4c_b import distributions, redundancy
from .build_stage4c_b2 import PRIOR as PRIOR_B2, PRIOR_SHA as PRIOR_SHA_B2, fill_depth, gen1_state
from .raw_store import sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

OUT = ROOT / "outputs/stage4c_b3"
QA = OUT / "STAGE4C_B3_QA"
HDS = OUT / "STAGE4C_B3_H_DATASET"
PREREG_JSON = OUT / "STAGE4C_B3_PREREGISTRATION.json"
PREREG_JSON_SHA = "77c1f29ff20eba2cf071442e23866fdc93955a4180c848864642b648c61c009e"
PREREG_MODULE_SHA = "631f07196fa2586d976573664375bf7444eb07d90ac8e5c49b94b070449343c3"
PRIOR = dict(PRIOR_B2, stage4c_b2="outputs/stage4c_b2/STAGE4C_B2_CHECKPOINT.json")
PRIOR_SHA = dict(PRIOR_SHA_B2, stage4c_b2="f548ceb0c501560b603ee7835a6a18b8350899beef92b527dc4d9571fb6aa1f7")
CORE = [c["name"] for c in H.CORE_H]
HCOL = {n: n.lower() for n in CORE}
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
URES = 90.0


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def _tsv(p: Path, df) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(p, sep="\t", index=False, lineterminator="\n", float_format="%.6g")


def gen2_state():
    cp = json.loads((ROOT / PRIOR["stage4c_b2"]).read_text())
    files = dict(cp["output_hashes"], **cp["h_dataset_partitions"])
    return {p: sha256(ROOT / p) for p in files}, files


def integrity() -> dict:
    v1 = v1_verify()
    prior = {k: sha256(ROOT / v) for k, v in PRIOR.items()}
    g1n, g1r = gen1_state()
    g2n, g2r = gen2_state()
    s4a = json.loads((ROOT / PRIOR["stage4a"]).read_text())
    meg = {p: sha256(ROOT / p) for p in s4a["meg_partitions"]}
    return dict(v1={k: v1[k] for k in ("strict_files", "v1_intact", "writes_performed")}, prior_checkpoints={PRIOR[k]: v for k, v in prior.items()},
                prior_unchanged=prior == PRIOR_SHA, gen1_files=len(g1r), gen1_identical=g1n == g1r, gen2_files=len(g2r), gen2_identical=g2n == g2r,
                stage4a_partitions_unchanged=meg == s4a["meg_partitions"],
                prereg_json_sha256=sha256(PREREG_JSON), prereg_json_unchanged=sha256(PREREG_JSON) == PREREG_JSON_SHA,
                prereg_module_sha256=sha256(ROOT / "src/sfv2/stage4c_b3_prereg.py"),
                prereg_module_unchanged=sha256(ROOT / "src/sfv2/stage4c_b3_prereg.py") == PREREG_MODULE_SHA)


# ------------------------------------------------------------------------------------------------ nested
def ensemble_predict(ctx, comps, loss, train, te):
    cache, preds, rej = {}, [], {}
    for v, pct, rep in G3.scenarios(comps):
        fits = G3.fit_arch(ctx, comps, train, v, pct, rep, loss, cache)
        bad = [f.reason for f in fits if not f.ok and f.reason != "UNDERDETERMINED"]
        if bad:
            rej[bad[0]] = rej.get(bad[0], 0) + 1
            continue
        for off in P3.ENSEMBLE["tolerance_offset_sigma"]:
            preds.append(G3.predict_pts(ctx, comps, fits, te, off))
    return (np.stack(preds, axis=1) if preds else np.full((len(te), 0), np.nan)), rej


def nested(ctx, geoms):
    c = ctx.ctrl
    sf = c.super_fold.to_numpy()
    sel_rows, res_rows = [], []
    for F in np.unique(sf):
        (gname, loss), rows = G3.select(ctx, geoms, sf != F)
        for r in rows:
            sel_rows.append(dict(landscape_id=ctx.lid, scope=f"OUTER_SUPERFOLD_{F}", **r))
        comps = geoms[gname]
        for g in np.unique(c.outer_group[sf == F]):
            te = np.nonzero((c.outer_group == g).to_numpy())[0]
            pred, rej = ensemble_predict(ctx, comps, loss, (c.outer_group != g).to_numpy(), te)
            for i, idx in enumerate(te):
                p = pred[i][np.isfinite(pred[i])]
                res_rows.append(dict(landscape_id=ctx.lid, control_id=c.control_id.iloc[idx], quality=c.quality_class.iloc[idx], outer_group=g, super_fold=int(F),
                                     selected_geometry=gname, selected_loss=loss, z=c.z95.iloc[idx], pred_median=float(np.median(p)) if len(p) else np.nan,
                                     p05=float(np.percentile(p, 5)) if len(p) else np.nan, p95=float(np.percentile(p, 95)) if len(p) else np.nan,
                                     n_scenarios=len(p), rejected=json.dumps(rej)))
    v = pd.DataFrame(res_rows)
    v["residual"] = v.z - v.pred_median
    v["inside_band"] = (v.z >= v.p05) & (v.z <= v.p95)
    return pd.DataFrame(sel_rows), v


# ------------------------------------------------------------------------------------------------ production
def production(ctx, geoms):
    c = ctx.ctrl
    (gname, loss), sel_rows = G3.select(ctx, geoms, np.ones(len(c), bool))
    comps = geoms[gname]
    allm = np.ones(len(c), bool)
    cache, scen, man = {}, [], []
    for v, pct, rep in G3.scenarios(comps):
        fits = G3.fit_arch(ctx, comps, allm, v, pct, rep, loss, cache)
        bad = [f.reason for f in fits if not f.ok and f.reason != "UNDERDETERMINED"]
        for off in P3.ENSEMBLE["tolerance_offset_sigma"]:
            man.append(dict(landscape_id=ctx.lid, geometry=gname, loss=loss, scenario_id=f"{ctx.lid[:4]}_G3_{len(man):04d}", structural_variant=v, summit_percentile=pct,
                            subsample_replicate=rep, tolerance_offset_sigma=off, accepted=not bad, physical_rejection_reason=bad[0] if bad else "",
                            component_sigmas_m="|".join(f"{f.sigma:.1f}" for f in fits), component_edf="|".join(f"{f.edf:.1f}" for f in fits), seed=P3.SEED + 7919 * rep))
            if not bad:
                scen.append((fits, off))
    man = pd.DataFrame(man)
    man["weight"] = np.where(man.accepted, 1.0 / max(int(man.accepted.sum()), 1), 0.0)
    h, w = ctx.X.shape
    dctrl = cKDTree(c[["x", "y"]].to_numpy()).query(np.c_[ctx.X.ravel(), ctx.Y.ravel()])[0].reshape(h, w)
    sup = ctx.app & ctx.valid90 & ~ctx.excl & (dctrl <= 3000.0)
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
        return dict(geometry=gname, loss=loss, sel_rows=sel_rows, comps=comps, man=man, ok=False)
    rows_, cols_ = np.nonzero(sup)
    r0, r1, c0, c1 = max(rows_.min() - 8, 0), min(rows_.max() + 9, h), max(cols_.min() - 8, 0), min(cols_.max() + 9, w)
    sl = np.s_[r0:r1, c0:c1]
    sub = sup[sl]
    Xs, Ys = ctx.X[sl], ctx.Y[sl]
    u = ctx.L.units
    ur, uc = u.ur.to_numpy(), u.uc.to_numpy()
    usup = sup[ur, uc]
    uix = np.nonzero(usup)[0]
    ul = (ur[uix] - r0, uc[uix] - c0)
    groups = comps[0]["groups"]
    Ws = W[:, r0:r1, c0:c1]
    srt = np.sort(Ws, axis=0)
    band = sub & (srt[-2] >= 0.10) if len(comps) > 1 else np.zeros_like(sub)   # numerical (same-group) blending band
    member = comps[0]["member"][sl]
    geo_edge = np.zeros_like(sub)
    if len(groups) > 1:
        geo_edge = sub & (ndimage.maximum_filter(member, 3) != ndimage.minimum_filter(member, 3))
    J = len(scen)
    Zs = np.empty((int(sub.sum()), J), np.float32)
    Zu, Su, Ru = (np.empty((len(uix), J), np.float32) for _ in range(3))
    Sk_band = np.empty((len(comps), int(band.sum()), J), np.float32)
    Gedge = {g: np.empty((int(geo_edge.sum()), J), np.float32) for g in groups}
    for j, (fits, off) in enumerate(scen):
        gs, comp, _ = G3.group_surface(ctx, comps, fits, Xs, Ys, sl, off)
        slope = np.full(sub.shape, np.nan)
        relief = np.full(sub.shape, np.nan)
        for i, g in enumerate(groups):
            zg = gs[g]
            if not np.isfinite(zg).any():
                continue
            zf = R.nearest_fill(zg, np.isfinite(zg))
            m = member == i
            slope[m] = R.horn(zf, URES)[m]
            relief[m] = R.relief(zf, 450 / URES)[m]
            if geo_edge.any():
                Gedge[g][:, j] = zf[geo_edge]
        Zs[:, j] = comp[sub]
        Zu[:, j] = comp[ul]
        Su[:, j] = slope[ul]
        Ru[:, j] = relief[ul]
        if band.any():
            for i, (k, f) in enumerate(zip(comps, fits)):
                Sk_band[i, :, j] = (f.predict(Xs[band], Ys[band]) + off * f.sigma) if f.ok else np.nan
    zq = np.percentile(Zs, [5, 25, 50, 75, 95], axis=1)
    grid = lambda vals: (lambda a: (a.__setitem__((slice(r0, r1), slice(c0, c1)), np.where(sub, 0, np.nan)), a[sl].__setitem__(sub, vals), a)[2])(np.full((h, w), np.nan))
    zp05, zmed, zp95, ziqr = grid(zq[0]), grid(zq[2]), grid(zq[4]), grid(zq[3] - zq[1])
    fin = sup & np.isfinite(zmed)
    p02, p98 = np.percentile(zmed[fin], [2, 98])
    R_L = float(p98 - p02)
    Lam = float(np.sqrt(fin.sum() * URES * URES / np.pi))
    tau_L = float(np.median([f.sigma for fits, _ in scen for f in fits if f.ok]))
    qu = np.percentile(Zu, [5, 25, 50, 75, 95], axis=1)
    n_u = len(u)
    full = lambda a: (lambda o: (o.__setitem__(uix, a), o)[1])(np.full(n_u, np.nan))
    zr = {k: full(q) for k, q in zip(("p05", "p25", "p50", "p75", "p95"), qu)}
    sl_med, rl_med = full(np.median(Su, axis=1)), full(np.median(Ru, axis=1))
    zmod_u = ctx.z90[ur, uc]
    zf_mod = R.nearest_fill(ctx.z90, np.isfinite(ctx.z90))
    slope_mod = np.where(np.isfinite(ctx.z90), R.horn(zf_mod, URES), np.nan)
    relief_mod = np.where(np.isfinite(ctx.z90), R.relief(zf_mod, 450 / URES), np.nan)
    d_u = cKDTree(c[["x", "y"]].to_numpy()).query(u[["centroid_x", "centroid_y"]].to_numpy())[0]
    ok = usup & np.isfinite(zr["p50"])
    with np.errstate(invalid="ignore"):
        hv = {"H_REMOVED_FRACTION": np.where(ok, (zr["p50"] - zmod_u) / R_L, np.nan), "H_SLOPE_CHANGE_DEG": np.where(ok, slope_mod[ur, uc] - sl_med, np.nan),
              "H_RELIEF_CHANGE_FRACTION": np.where(ok, (relief_mod[ur, uc] - rl_med) / R_L, np.nan),
              "H_REFERENCE_RELATIVE_HEIGHT": np.where(ok, (zr["p50"] - p02) / R_L, np.nan),
              "H_SCENARIO_SPREAD_FRACTION": np.where(ok, (zr["p75"] - zr["p25"]) / R_L, np.nan), "H_CONTROL_DISTANCE_FRACTION": np.where(ok, d_u / Lam, np.nan)}
    avail = ok & np.all(np.stack([np.isfinite(v) for v in hv.values()]), axis=0)
    young = np.isin(u.source_unit_code.to_numpy(), [])  # unchanged: no young-deposit exceptions (Gen-1/Gen-2)
    D = zr["p50"] - zmod_u  # NOT clipped
    negm = avail & ~young & (D < -tau_L)
    neg_frac = float(negm[avail & ~young].mean())
    slv = sl_med[avail]
    # ---- negative-removal diagnostics
    nd = [dict(scope="GLOBAL", value="ALL", units=int(avail.sum()), negative_fraction=neg_frac)]
    for code, g in pd.Series(negm[avail]).groupby(u.source_unit_code.to_numpy()[avail]):
        nd.append(dict(scope="GEOLOGICAL_UNIT", value=code, units=len(g), negative_fraction=float(g.mean())))
    dom_u = dominant[ur, uc]
    for i, k in enumerate(comps):
        m = avail & (dom_u == i)
        if m.any():
            nd.append(dict(scope="COMPONENT", value=f"{i}:{k['ctype']}", units=int(m.sum()), negative_fraction=float(negm[m].mean())))
    for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 3000.1)):
        m = avail & (d_u >= lo) & (d_u < hi)
        if m.any():
            nd.append(dict(scope="CONTROL_DISTANCE_M", value=f"{lo}-{int(hi)}", units=int(m.sum()), negative_fraction=float(negm[m].mean())))
    ngrid = np.zeros((h, w), bool)
    ngrid[ur[negm], uc[negm]] = True
    lab, n = ndimage.label(ngrid, np.ones((3, 3)))
    sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1] if n else np.array([0])
    nd.append(dict(scope="SPATIAL_CLUSTERS", value=f"n={n}", units=int(negm.sum()), negative_fraction=neg_frac,
                   detail=f"largest cluster units {int(sizes[0])}; top-10 share {float(sizes[:10].sum() / max(sizes.sum(), 1)):.3f}"))
    if negm.any():
        q = np.percentile(D[negm], [5, 25, 50, 75, 95])
        nd.append(dict(scope="NEGATIVE_MAGNITUDE_M", value="P05|P25|P50|P75|P95", units=int(negm.sum()), negative_fraction=neg_frac,
                       detail="|".join(f"{v:.1f}" for v in q), magnitude_over_R_L_median=float(np.median(D[negm]) / R_L)))
    # ---- boundary + physical QA
    zf_med = R.nearest_fill(zmed[sl], np.isfinite(zmed[sl]))
    slope_med = R.horn(zf_med, URES)
    lap = np.abs(ndimage.laplace(zf_med)) / URES ** 2
    interior = sub & ~band & ~geo_edge
    bq = []
    if band.any():
        Sb = np.nanmedian(Sk_band, axis=2)
        top2 = np.argsort(Ws[:, band], axis=0)[-2:]
        a, b_ = Sb[top2[1], np.arange(band.sum())], Sb[top2[0], np.arange(band.sum())]
        step = np.abs(a - b_) / R_L
        step = step[np.isfinite(step)]
        r_ = dict(boundary="NUMERICAL (same history group)", classification="NUMERICAL", continuity_expected=True, cells=int(band.sum()),
                  seam_step_median=float(np.median(step)), seam_step_p95=float(np.percentile(step, 95)),
                  slope_excess_deg=float(np.percentile(slope_med[band], 99) - np.percentile(slope_med[interior], 99)),
                  ridge_trough_fraction=float((lap[band] > np.percentile(lap[interior], 99)).mean()),
                  spread_ratio=float(np.median(ziqr[sl][band]) / max(np.median(ziqr[sl][interior]), 1e-9)))
        r_["qa_pass"] = r_["seam_step_median"] <= 0.10 and r_["seam_step_p95"] <= 0.25 and r_["slope_excess_deg"] <= 5 and r_["ridge_trough_fraction"] <= 0.05 and r_["spread_ratio"] <= 2
        bq.append(r_)
    if geo_edge.any():
        ga, gb = groups[0], groups[1]
        dd = np.abs(np.median(Gedge[ga], axis=1) - np.median(Gedge[gb], axis=1)) / R_L
        bq.append(dict(boundary=f"{ga}|{gb}", classification="GEOLOGICAL_DISCONTINUITY", continuity_expected=False, cells=int(geo_edge.sum()),
                       geological_relationship=P3.HISTORY_GROUP_EVIDENCE.get(ctx.lid, ""), seam_step_median=float(np.median(dd)), seam_step_p95=float(np.percentile(dd, 95)),
                       slope_excess_deg=float(np.nanpercentile(np.median(Su, axis=1), 99)) if False else np.nan,
                       spread_ratio=float(np.median(ziqr[sl][geo_edge]) / max(np.median(ziqr[sl][interior]), 1e-9)), qa_pass="REPORTED_ONLY"))
    if not bq:
        bq.append(dict(boundary="none", classification="SINGLE_COMPONENT", continuity_expected=True, cells=0, qa_pass=True))
    bqa_pass = all(r_["qa_pass"] is True for r_ in bq if r_["classification"] != "GEOLOGICAL_DISCONTINUITY")
    depth = fill_depth(np.where(sub, zf_med, np.nan), sub)
    pit = float((depth[sub] > tau_L).mean())
    phys_pass = pit <= 0.01 and not (sup & ctx.cal).any()
    # ---- positive-offset / shift diagnostics (diagnostic only)
    g2p = ROOT / f"outputs/stage4c_b2/STAGE4C_B2_QA/{ctx.lid}_gen2_grids.npz"
    with np.load(g2p) as g2:
        z2 = g2["zref_median"]
    both = fin & np.isfinite(z2)
    sh = (zmed - z2)[both]
    Dg = (zmed - ctx.z90)[fin]
    iqr_g, dist_g = ziqr[fin], dctrl[fin]
    rho = float(stats.spearmanr(iqr_g, dist_g).correlation) if fin.sum() > 10 else np.nan
    diag = dict(D_over_R_L_p99=float(np.percentile(Dg / R_L, 99)), fraction_D_gt_0_5_R_L=float((Dg > 0.5 * R_L).mean()),
                shift_vs_gen2_mean_m=float(sh.mean()), shift_vs_gen2_sd_m=float(sh.std()), shift_uniformity_index_sd_over_abs_mean=float(sh.std() / max(abs(sh.mean()), 1e-9)),
                shift_vs_gen2_p05_p95_m=[float(np.percentile(sh, 5)), float(np.percentile(sh, 95))], spearman_spread_vs_control_distance=rho,
                removed_thickness_m_p05_p50_p95=[float(v) for v in np.percentile(Dg, [5, 50, 95])])
    grids = dict(z_modern=ctx.z90.astype(np.float32), zref_p50=zmed.astype(np.float32), zref_p05=zp05.astype(np.float32), zref_p95=zp95.astype(np.float32),
                 zref_iqr=ziqr.astype(np.float32), zref_gen2_p50=z2.astype(np.float32), support=sup, negative_units=ngrid, dominant_component=np.where(ctx.app, dominant, -1).astype(np.int16),
                 history_member=comps[0]["member"].astype(np.int8), boundary_band=np.pad(band, ((r0, h - r1), (c0, w - c1))),
                 geological_edge=np.pad(geo_edge, ((r0, h - r1), (c0, w - c1))), caldera_interior=ctx.cal, control_distance_m=dctrl.astype(np.float32),
                 x0=np.float64(ctx.L.transform.c), y0=np.float64(ctx.L.transform.f), res=np.float64(URES))
    sem = {i: ("ESTIMATED_REMOVAL" if k["family"].split(":")[0].startswith(("RADIAL", "CALDERA")) else "MINIMUM_REMOVAL") for i, k in enumerate(comps)}
    sem_u = np.where(usup, np.array([sem[i] for i in dom_u], dtype=object), "")
    return dict(geometry=gname, loss=loss, sel_rows=sel_rows, comps=comps, man=man, ok=True, R_L=R_L, P02=float(p02), Lambda=Lam, tau_L=tau_L, hv=hv, avail=avail,
                usup=usup, sup_frac=float(avail.mean()), neg_frac=neg_frac, max_slope=float(np.nanmax(slv)), p99_slope=float(np.nanpercentile(slv, 99)),
                nd=pd.DataFrame(nd).assign(landscape_id=ctx.lid), bq=pd.DataFrame(bq).assign(landscape_id=ctx.lid), bqa_pass=bqa_pass, pit=pit, phys_pass=phys_pass,
                diag=diag, grids=grids, sem_u=sem_u, J=J, edf=float(np.median([sum(f.edf for f in fits) for fits, _ in scen])),
                qa=dict(qa_h_removed_fraction_p05=np.where(ok, (zr["p05"] - zmod_u) / R_L, np.nan), qa_h_removed_fraction_p95=np.where(ok, (zr["p95"] - zmod_u) / R_L, np.nan)))


# ------------------------------------------------------------------------------------------------ figures (no Y)
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
    D = g["zref_p50"] - g["z_modern"]
    rr = val.set_index("control_id").reindex(c.control_id).residual.to_numpy() / prod["R_L"]
    fig, ax = plt.subplots(4, 4, figsize=(26, 24))
    ax = ax.ravel()

    def show(i, a, t, cm="terrain", lim=None):
        im = ax[i].imshow(a, extent=ext, cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, interpolation="nearest")
        fig.colorbar(im, ax=ax[i], fraction=0.04)
        ax[i].set_title(t, fontsize=10)
    show(0, g["z_modern"], "1. Modern DEM (m)")
    show(1, np.where(g["dominant_component"] >= 0, g["dominant_component"] + 10 * g["history_member"], np.nan).astype(float),
         "2. Components (colour) / history groups:\n" + "; ".join(f"{i}={k['ctype']}[{k['history']}]" for i, k in enumerate(prod["comps"]))[:150], "tab20")
    ax[2].imshow(g["z_modern"], extent=ext, cmap="gray")
    for q, col in (("Q1", "tab:red"), ("Q2", "tab:blue")):
        s = c[c.quality_class == q]
        ax[2].scatter(s.x / 1000, s.y / 1000, s=3, c=col, label=f"{q} ({len(s)})")
    ax[2].legend(fontsize=7); ax[2].set_title("3. Fitting controls (Q1/Q2)")
    ax[3].imshow(g["z_modern"], extent=ext, cmap="gray")
    ax[3].scatter(c.x / 1000, c.y / 1000, c=c.super_fold, cmap="tab10", s=3); ax[3].set_title("4. Outer held-out super-folds")
    show(4, m(g["zref_p50"]), "5. Gen-3 P50 palaeosurface (m)")
    show(5, m(g["zref_p95"] - g["zref_p05"]), "6. P95 - P05 (m)", "viridis")
    show(6, m(D), "7. P50 - modern (m)", "RdBu", (-500, 500))
    show(7, m(g["negative_units"].astype(float)), "8. Negative-removal unit cells", "Reds", (0, 1))
    show(8, m(np.where(D < -prod["tau_L"], D, np.nan)), "9. Negative-removal magnitude (m)", "magma", None)
    show(9, sup.astype(float) + 2 * g["caldera_interior"], "10. Support mask", "Set2", (0, 3))
    show(10, m(g["zref_iqr"] / prod["R_L"]), "11. Scenario spread IQR/R_L", "viridis")
    ax[11].imshow(g["z_modern"], extent=ext, cmap="gray")
    sc = ax[11].scatter(c.x / 1000, c.y / 1000, c=rr, cmap="RdBu", vmin=-0.3, vmax=0.3, s=4)
    fig.colorbar(sc, ax=ax[11], fraction=0.04); ax[11].set_title("12. Outer held-out residual / R_L (r = z - S)")
    show(12, m(g["zref_p50"]), "13. Component boundaries (red numerical, black geological)")
    if g["boundary_band"].any():
        ax[12].contour(np.flipud(g["boundary_band"].astype(float)), levels=[0.5], colors="red", extent=ext, origin="lower")
    if g["geological_edge"].any():
        ax[12].contour(np.flipud(g["geological_edge"].astype(float)), levels=[0.5], colors="black", extent=ext, origin="lower")
    show(13, m(g["zref_p50"] - g["zref_gen2_p50"]), "14. Gen-3 P50 - Gen-2 P50 (m)", "PuOr", (-300, 300))
    ax[14].hist(np.clip(rr[np.isfinite(rr)], -0.6, 0.6), bins=60); ax[14].set_title("Held-out residual / R_L")
    ax[15].axis("off")
    ax[15].text(0, 1, prod["summary_text"], va="top", family="monospace", fontsize=10)
    fig.suptitle(f"Stage 4C-B3 Gen-3 one-sided reconstruction QA - {ctx.lid} (Y-blind; no landslide data)", fontsize=14)
    QA.mkdir(parents=True, exist_ok=True)
    p = QA / f"{ctx.lid}_gen3_qa.png"
    fig.savefig(p, dpi=65, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    p2 = QA / f"{ctx.lid}_gen3_grids.npz"
    np.savez_compressed(p2, **g)
    return [p, p2]


# ------------------------------------------------------------------------------------------------ main
ESTIMAND_MD = """# Stage 4C-B3 scientific estimand (frozen before any Gen-3 fit)

**S\\*(x)** = plausible pre-erosional constructional reference elevation at x: a *conservative upper constructional surface* consistent with the
preserved Q1/Q2 remnant controls and the structural geometry of the landscape.

It is **not** the expected modern summit elevation, a smoothed modern DEM, the exact original elevation, the maximum conceivable elevation,
or a surface lifted until negative removal passes.

Uncertainty: an equal-weight ensemble S_j(x) (structural variants x summit percentile x control subsample x control tolerance); P05/P50/P95.
The scenario median P50 is the primary reconstruction for every frozen Stage 4C-A gate.

Residual convention: r = z_control - S(x).  r > 0 (reference below a preserved control) is penalised more than r < 0 by the pre-registered
one-sided losses (upper quantile q in {0.75, 0.85, 0.90}; asymmetric Huber a in {2, 4}; soft upper envelope lambda in {4, 16}).  No hard
constraint S >= control and no constraint S >= modern DEM are used; the modern DEM never enters any objective.

Negative removal: D(x) = S_P50(x) - Z_modern(x); negative where D < -tau_L (tau_L = median scenario robust control scale; unchanged
definition); never clipped; young-deposit exceptions unchanged (none).
"""


def main() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    integ_before = integrity()
    if not (integ_before["prior_unchanged"] and integ_before["gen1_identical"] and integ_before["gen2_identical"] and integ_before["stage4a_partitions_unchanged"]
            and integ_before["v1"]["v1_intact"] and integ_before["v1"]["writes_performed"] == 0):
        raise RuntimeError(f"STOP: integrity failure {integ_before}")
    if not (integ_before["prereg_json_unchanged"] and integ_before["prereg_module_unchanged"]):
        raise RuntimeError("STOP: Gen-3 pre-registration changed after freezing")
    assert P3.GATES == H.VALIDATION_GATES["gates"]
    (OUT / "STAGE4C_B3_ESTIMAND.md").write_text(ESTIMAND_MD)

    res = {}
    for lid in P3.RECONSTRUCTED:
        t = time.time()
        ctx = G2.load_ctx(lid)
        geoms = {gname: G3.components(ctx, gname) for gname in P3.GEOMETRIES[lid]}
        sel, val = nested(ctx, geoms)
        prod = production(ctx, geoms)
        res[lid] = dict(ctx=ctx, geoms=geoms, sel=sel, val=val, prod=prod)
        print(f"[4C-B3] {lid}: selected {prod['geometry']} / {prod['loss']} ({time.time() - t:.0f} s)", flush=True)

    gen1 = pd.read_csv(ROOT / "outputs/stage4c_b/STAGE4C_B_RECONSTRUCTION_VALIDATION.tsv", sep="\t")
    gen1 = gen1[gen1.scope == "LANDSCAPE"].set_index("landscape_id")
    gen2 = pd.read_csv(ROOT / "outputs/stage4c_b2/STAGE4C_B2_RECONSTRUCTION_VALIDATION.tsv", sep="\t").set_index("landscape_id")
    gen2bq = pd.read_csv(ROOT / "outputs/stage4c_b2/STAGE4C_B2_BOUNDARY_QA.tsv", sep="\t").set_index("landscape_id")
    gen2sa = pd.read_csv(ROOT / "outputs/stage4c_b2/STAGE4C_B2_SELECTED_ARCHITECTURES.tsv", sep="\t").set_index("landscape_id")
    lib, comp_rows, ctrl_rows, nv, asel, man, vrows, ndiag, bqs, st_rows, cmp_rows, diag_rows = ([] for _ in range(12))
    status = {"NOTO_PENINSULA_2024": "AGE_PROXY_ONLY"}
    codes = dict(median_abs_residual="HIGH_RESIDUAL", abs_bias="HIGH_BIAS", holdout_band="LOW_SCENARIO_COVERAGE", admissible_controls="INSUFFICIENT_CONTROLS",
                 support_fraction="LOW_SUPPORT", negative_removal="EXCESS_NEGATIVE_REMOVAL", max_reference_slope="IMPLAUSIBLE_SLOPE")
    for lid, r in res.items():
        ctx, prod, val, sel = r["ctx"], r["prod"], r["val"], r["sel"]
        c = ctx.ctrl
        for gname, comps in r["geoms"].items():
            for loss in P3.LOSSES:
                for i, k in enumerate(comps):
                    lib.append(dict(landscape_id=lid, geometry=gname, loss=loss, loss_family=P3.LOSSES[loss]["family"], loss_level=json.dumps(P3.LOSSES[loss]),
                                    component_index=i, component_type=k["ctype"], family=k["family"], domain_rule=k["rule"], history_group=k["history"],
                                    centre=k["centre"][0] if k["centre"] else "", fit_controls=int(k["cmask"].sum()),
                                    selected_in_production=(gname == prod["geometry"] and loss == prod["loss"])))
        comps = prod["comps"]
        for i, k in enumerate(comps):
            ck = c[k["cmask"]]
            comp_rows.append(dict(component_id=f"{lid[:4]}_G3_{i}", landscape_id=lid, geometry=prod["geometry"], loss=prod["loss"], component_type=k["ctype"],
                                  family=k["family"], history_group=k["history"], geological_units=";".join(sorted(ck.geological_unit.unique())),
                                  centre=k["centre"][0] if k["centre"] else "", spatial_domain=k["rule"], domain_area_km2=round(k["gmask"].sum() * 0.0081, 2),
                                  control_count=int(k["cmask"].sum()), controls_per_sector=";".join(f"{a}={b}" for a, b in ck.groupby("sector").size().items()) if (ck.sector >= 0).any() else ""))
        dom_of = np.full(len(c), -1)
        for i, k in enumerate(comps):
            dom_of[k["cmask"] & (dom_of < 0)] = i
        vv = val.set_index("control_id")
        for idx in range(len(c)):
            ctrl_rows.append(dict(landscape_id=lid, control_id=c.control_id.iloc[idx], component_id=f"{lid[:4]}_G3_{dom_of[idx]}", quality_class=c.quality_class.iloc[idx],
                                  x=c.x.iloc[idx], y=c.y.iloc[idx], modern_elevation_p95=c.z95.iloc[idx], geology=c.geological_unit.iloc[idx],
                                  outer_fold=c.outer_group.iloc[idx], outer_super_fold=int(c.super_fold.iloc[idx]),
                                  inner_folds="leave-one-of-remaining-super-folds (all except " + str(int(c.super_fold.iloc[idx])) + ")",
                                  heldout_prediction=vv.pred_median.get(c.control_id.iloc[idx], np.nan), heldout_residual=vv.residual.get(c.control_id.iloc[idx], np.nan)))
        nv.append(sel)
        nv.append(pd.DataFrame(prod["sel_rows"]).assign(landscape_id=lid, scope="PRODUCTION_ALL_CONTROLS"))
        fs = sel[sel.selected].assign(arch=lambda d: d.geometry + "/" + d.loss).groupby("arch").size()
        prod_arch = prod["geometry"] + "/" + prod["loss"]
        geo_stab = sel[sel.selected].geometry.value_counts()
        asel.append(dict(landscape_id=lid, production_geometry=prod["geometry"], production_loss=prod["loss"], outer_selections=";".join(f"{a}={n}" for a, n in fs.items()),
                         architecture_stability=float(fs.get(prod_arch, 0) / fs.sum()), geometry_stability=float(geo_stab.get(prod["geometry"], 0) / geo_stab.sum()),
                         n_components=len(comps), effective_dof_median=prod.get("edf"), scenarios=prod.get("J"),
                         note=P3.EXCLUDED_FROM_GEN3["pusuk_nangi"] if lid.startswith("LOMBOK") else ""))
        man.append(prod["man"])
        Gt = P3.GATES
        med = float(np.median(np.abs(val.residual.dropna())) / prod["R_L"])
        bias = float(val.residual.dropna().mean() / prod["R_L"])
        band = float(val.inside_band.mean())
        gates = dict(median_abs_residual=(med, med <= Gt["median_abs_residual_fraction_max"]), abs_bias=(abs(bias), abs(bias) <= Gt["abs_bias_fraction_max"]),
                     holdout_band=(band, band >= Gt["holdout_band_fraction_min"]), admissible_controls=(len(c), len(c) >= Gt["min_admissible_controls"]),
                     support_fraction=(prod["sup_frac"], prod["sup_frac"] >= Gt["min_support_fraction_of_units"]),
                     negative_removal=(prod["neg_frac"], prod["neg_frac"] <= Gt["negative_removal_fraction_max_outside_mapped_young_deposits"]),
                     max_reference_slope=(prod["max_slope"], prod["max_slope"] <= Gt["max_reconstructed_slope_deg_constructional"]))
        fails = [codes[k] for k, (_, ok) in gates.items() if not ok]
        if not prod["bqa_pass"]:
            fails.append("BOUNDARY_QA_FAIL")
        if not prod["phys_pass"]:
            fails.append("PHYSICAL_QA_FAIL")
        unpred = int(val.pred_median.isna().sum())
        if unpred:
            fails.append(f"UNPREDICTED_HELDOUT_{unpred}")
        st = ("VERIFIED_GEN3_FULL" if prod["sup_frac"] >= 0.90 else "VERIFIED_GEN3_PARTIAL") if not fails else "FAILED_GEN3"
        status[lid] = st
        prod.update(status=st, fails=fails, gates=gates, med=med, bias=bias, band=band)
        prod["summary_text"] = (f"{lid}\n{prod['geometry']} / {prod['loss']}\nstatus: {st}\nfailures: {', '.join(fails) or 'none'}\nR_L {prod['R_L']:.0f} m  tau_L {prod['tau_L']:.0f} m\n"
                                + "\n".join(f"{k}: {v[0]:.4g} {'PASS' if v[1] else 'FAIL'}" for k, v in gates.items())
                                + f"\nboundary QA {prod['bqa_pass']}  pits {prod['pit']:.4f}\nP99 slope {prod['p99_slope']:.1f}\n"
                                + f"shift vs Gen-2: mean {prod['diag']['shift_vs_gen2_mean_m']:.0f} m, SD {prod['diag']['shift_vs_gen2_sd_m']:.0f} m")
        vrows.append(dict(landscape_id=lid, geometry=prod["geometry"], loss=prod["loss"], heldout_controls=len(val), heldout_groups=val.outer_group.nunique(), unpredicted=unpred,
                          R_L_m=prod["R_L"], tau_L_m=prod["tau_L"], median_abs_residual_fraction=med, signed_bias_fraction=bias, abs_bias_fraction=abs(bias),
                          median_signed_residual_fraction=float(val.residual.median() / prod["R_L"]), upper_tail_overprediction_p05_fraction=float(val.residual.quantile(0.05) / prod["R_L"]),
                          heldout_controls_above_reference_by_tau=float((val.residual > prod["tau_L"]).mean()),
                          holdout_band_fraction=band, admissible_controls=len(c), support_fraction_of_units=prod["sup_frac"], negative_removal_fraction=prod["neg_frac"],
                          max_reference_slope_deg=prod["max_slope"], p99_reference_slope_deg=prod["p99_slope"],
                          **{f"gate_{k}": "PASS" if ok else "FAIL" for k, (_, ok) in gates.items()}, all_gates_pass=all(ok for _, ok in gates.values()),
                          boundary_qa_pass=prod["bqa_pass"], physical_qa_pass=prod["phys_pass"], pit_fraction=prod["pit"]))
        ndiag.append(prod["nd"])
        bqs.append(prod["bq"])
        diag_rows.append(dict(landscape_id=lid, **prod["diag"]))
        st_rows.append(dict(landscape_id=lid, geometry=prod["geometry"], loss=prod["loss"], semantics="+".join(sorted(set(prod["sem_u"][prod["usup"]]))), status=st,
                            failure_codes="|".join(fails), verified_gen3_h_capable=st.startswith("VERIFIED"), core_h_released=st.startswith("VERIFIED")))
        g1, g2 = gen1.loc[lid], gen2.loc[lid]
        for gen, d in (("GEN1", dict(median=g1.median_abs_residual_fraction, bias=g1.abs_bias_fraction, band=g1.holdout_band_fraction, support=g1.support_fraction_of_units,
                                     neg=g1.negative_removal_fraction, mx=g1.max_reference_slope_deg, p99=g1.p99_reference_slope_deg, bqa="n/a", arch=g1.family,
                                     est="central (Huber) trend", ncomp=np.nan, cx="<= 6 linear params / component", pf="FAIL")),
                       ("GEN2", dict(median=g2.median_abs_residual_fraction, bias=g2.abs_bias_fraction, band=g2.holdout_band_fraction, support=g2.support_fraction_of_units,
                                     neg=g2.negative_removal_fraction, mx=g2.max_reference_slope_deg, p99=g2.p99_reference_slope_deg, bqa=g2.boundary_qa_pass,
                                     arch=g2.architecture, est="central (Huber) piecewise trend", ncomp=gen2sa.loc[lid, "n_components"],
                                     cx=f"edf ~ {gen2sa.loc[lid, 'effective_dof_median']:.1f}", pf="FAIL")),
                       ("GEN3", dict(median=med, bias=abs(bias), band=band, support=prod["sup_frac"], neg=prod["neg_frac"], mx=prod["max_slope"], p99=prod["p99_slope"],
                                     bqa=prod["bqa_pass"], arch=prod["geometry"] + "/" + prod["loss"], est="one-sided upper constructional surface",
                                     ncomp=len(comps), cx=f"edf ~ {prod['edf']:.1f}", pf="PASS" if st.startswith("VERIFIED") else "FAIL"))):
            cmp_rows.append(dict(landscape_id=lid, generation=gen, median_abs_residual_fraction=d["median"], abs_bias_fraction=d["bias"], band_coverage=d["band"],
                                 support=d["support"], negative_removal=d["neg"], max_slope_deg=d["mx"], p99_slope_deg=d["p99"], boundary_qa=d["bqa"], architecture=d["arch"],
                                 estimand=d["est"], n_components=d["ncomp"], effective_complexity=d["cx"], pass_fail=d["pf"]))
    st_rows.append(dict(landscape_id="NOTO_PENINSULA_2024", geometry="AGE_HISTORY_PROXY", loss="", semantics="NONE", status="AGE_PROXY_ONLY", failure_codes="",
                        verified_gen3_h_capable=False, core_h_released=False))
    verified = [l for l in LANDSCAPES if status[l].startswith("VERIFIED")]
    n_ver = len(verified)
    unlocked = n_ver >= 3
    stop = n_ver < 3

    _tsv(OUT / "STAGE4C_B3_FAMILY_LIBRARY.tsv", lib)
    _tsv(OUT / "STAGE4C_B3_COMPONENTS.tsv", comp_rows)
    _tsv(OUT / "STAGE4C_B3_CONTROLS.tsv", ctrl_rows)
    _tsv(OUT / "STAGE4C_B3_NESTED_VALIDATION.tsv", pd.concat(nv, ignore_index=True))
    _tsv(OUT / "STAGE4C_B3_ARCHITECTURE_SELECTION.tsv", asel)
    _tsv(OUT / "STAGE4C_B3_SCENARIO_MANIFEST.tsv", pd.concat(man, ignore_index=True))
    _tsv(OUT / "STAGE4C_B3_RECONSTRUCTION_VALIDATION.tsv", vrows)
    _tsv(OUT / "STAGE4C_B3_HOLDOUT_RESIDUALS.tsv", pd.concat([r["val"] for r in res.values()], ignore_index=True))
    _tsv(OUT / "STAGE4C_B3_NEGATIVE_REMOVAL_DIAGNOSTICS.tsv", pd.concat(ndiag, ignore_index=True))
    _tsv(OUT / "STAGE4C_B3_BOUNDARY_QA.tsv", pd.concat(bqs, ignore_index=True))
    _tsv(OUT / "STAGE4C_B3_SHIFT_UNCERTAINTY_DIAGNOSTICS.tsv", diag_rows)
    _tsv(OUT / "STAGE4C_B3_GEN1_GEN2_GEN3.tsv", cmp_rows)
    _tsv(OUT / "STAGE4C_B3_RECONSTRUCTION_STATUS.tsv", st_rows)

    # ---- H: dataset only if a landscape is verified; diagnostic redundancy otherwise
    frames, diag = {}, {}
    for lid, r in res.items():
        u = B.load(lid, "terrain_units/terrain_units", usecols=["terrain_unit_id"])
        d = pd.DataFrame(dict(landscape_id=lid, terrain_unit_id=u.terrain_unit_id.to_numpy()))
        for cn in CORE:
            d[HCOL[cn]] = r["prod"]["hv"][cn]
        d["h_available"] = r["prod"]["avail"]
        diag[lid] = d
    files = {}
    if verified:
        for lid in LANDSCAPES:
            u = B.load(lid, "terrain_units/terrain_units", usecols=["terrain_unit_id"])
            df = pd.DataFrame(dict(landscape_id=lid, terrain_unit_id=u.terrain_unit_id.to_numpy()))
            rel = lid in verified
            for cn in CORE:
                df[HCOL[cn]] = res[lid]["prod"]["hv"][cn] if rel else np.nan
            df["h_available"] = res[lid]["prod"]["avail"] if rel else False
            df["qa_h_landscape_status"] = status[lid]
            frames[lid] = df
            p = HDS / f"{lid}.npz"
            write_columns(p, df)
            files[str(p.relative_to(ROOT))] = sha256(p)
    relf = {l: frames[l] for l in verified}
    red = pd.concat([redundancy(relf, "RELEASED_VERIFIED_GEN3_H"), redundancy(diag, "DIAGNOSTIC_UNRELEASED_GEN3")], ignore_index=True)
    _tsv(OUT / "STAGE4C_B3_H_REDUNDANCY_WITH_M.tsv", red)
    _tsv(OUT / "STAGE4C_B3_H_DISTRIBUTIONS.tsv", pd.concat([distributions(relf, "RELEASED_VERIFIED_GEN3_H"), distributions(diag, "DIAGNOSTIC_UNRELEASED_GEN3")], ignore_index=True))
    rr = red[red.role == "RELEASED_VERIFIED_GEN3_H"] if len(red) else pd.DataFrame()
    sel_h = []
    for cn in CORE:
        r2 = float(rr[rr.feature == cn].pooled_r2_on_60_M.iloc[0]) if len(rr) and (rr.feature == cn).any() else np.nan
        a = bool(np.isfinite(r2) and r2 < 0.90)
        b = bool(n_ver > 0 and all(frames[l][HCOL[cn]].notna().any() for l in verified))
        sel_h.append(dict(feature=cn, pooled_r2_on_60_M=r2, criterion_a_r2_lt_0_90=a, criterion_b_available_in_every_verified_landscape=b,
                          criterion_c_reconstruction_validation_passes=n_ver > 0, selected_h=a and b and n_ver > 0,
                          note="" if n_ver else "no verified Gen-3 landscape: SELECTED_H empty by the frozen rule"))
    seltab = pd.DataFrame(sel_h)
    _tsv(OUT / "STAGE4C_B3_SELECTED_H.tsv", seltab)
    for lid, r in res.items():
        qa_figure(r["ctx"], r["prod"], r["val"])

    integ_after = integrity()
    ml = [p for p in scan_ml_h_artifacts() if not p.startswith("outputs/stage4b/")]
    _json(OUT / "STAGE4C_B3_INTEGRITY.json", dict(before=integ_before, after=integ_after, ml_artifacts_outside_stage4b=ml))
    s = dict(res=res, status=status, verified=verified, n_ver=n_ver, unlocked=unlocked, stop=stop, vt=pd.DataFrame(vrows), asel=pd.DataFrame(asel),
             cmp=pd.DataFrame(cmp_rows), nd=pd.concat(ndiag, ignore_index=True), bq=pd.concat(bqs, ignore_index=True), diag=pd.DataFrame(diag_rows), red=red,
             seltab=seltab, st=pd.DataFrame(st_rows), comp=pd.DataFrame(comp_rows), ib=integ_before, ia=integ_after, ml=ml)
    (OUT / "STAGE4C_B3_REPORT.md").write_text(report(s))
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith(("STAGE4C_B3_CHECKPOINT", "STAGE4C_B3_DIAGNOSTICS")))
    import matplotlib
    import scipy
    cp = dict(schema="SFV2_STAGE4C_B3_CHECKPOINT_V1", as_of=P3.AS_OF, integrity_before=integ_before, integrity_after=integ_after,
              preregistration=dict(json=str(PREREG_JSON.relative_to(ROOT)), json_sha256=PREREG_JSON_SHA, module_sha256=PREREG_MODULE_SHA,
                                  amendments=sorted(p.name for p in OUT.glob("STAGE4C_B3_AMENDMENT_*.json"))),
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage4c_b3.py", "stage4c_b3_recon.py", "stage4c_b3_prereg.py", "stage4c_b2_recon.py", "stage4c_b_recon.py")},
              software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
              random_seed=P3.SEED, selected={l: [r["prod"]["geometry"], r["prod"]["loss"]] for l, r in res.items()}, validation=vrows, statuses=status,
              verified_gen3_h_capable=verified, n_verified_gen3_h_capable=n_ver, megh_experiment_unlocked="YES" if unlocked else "NO",
              stopping_rule_activated=stop, stopping_rule=P3.STOPPING_RULE, h_dataset_created=bool(verified), h_dataset_partitions=files,
              thresholds_changed=False, negative_removal_clipped=False, modern_dem_lower_bound_used=False, h_zero_filled=False, model_trained=False, megh_trained=False,
              delta_h_computed=False, h_predictive_performance_observed=False, y_loaded=False, stage4b_residuals_inspected=False, lockbox_accessed=False,
              gen1_modified=False, gen2_modified=False, ml_artifacts_outside_stage4b=ml, runtime_s=round(time.time() - t0),
              output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs})
    _json(OUT / "STAGE4C_B3_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE4C_B3_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD")))
    return dict(cp=cp, s=s)


def _md(df, cols):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r.get(c, "")) for c in cols) + " |" for _, r in df.iterrows())


def report(s):
    vt = s["vt"]
    gcols = [c for c in vt.columns if c.startswith("gate_")]
    nd = s["nd"]
    return f"""# Stage 4C-B3 report - Gen-3 one-sided structurally-aware palaeosurface reconstruction

**DEVELOPMENT ONLY.** No Y, no Stage 4B residuals, no lockbox, no model training, no MEGH, no Delta_H, no modern-DEM lower bound, no clipping.
Pre-registration frozen before any Gen-3 fit (STAGE4C_B3_PREREGISTRATION.json, sha256 {PREREG_JSON_SHA}); no amendments.

## Result
Statuses: {json.dumps(s['status'])}. **N_VERIFIED_GEN3_H_CAPABLE = {s['n_ver']}** -> **MEGH_EXPERIMENT_UNLOCKED = {'YES' if s['unlocked'] else 'NO'}**.
Stopping rule activated: **{s['stop']}**. {P3.STOPPING_RULE if s['stop'] else ''}

## Architecture selection (nested)
{_md(s['asel'], ['landscape_id', 'production_geometry', 'production_loss', 'outer_selections', 'architecture_stability', 'geometry_stability', 'n_components', 'effective_dof_median'])}

## Validation (outer, frozen 4C-A gates)
{_md(vt, ['landscape_id', 'median_abs_residual_fraction', 'signed_bias_fraction', 'holdout_band_fraction', 'heldout_controls_above_reference_by_tau', 'support_fraction_of_units', 'negative_removal_fraction', 'max_reference_slope_deg', 'p99_reference_slope_deg'])}

{_md(vt, ['landscape_id'] + gcols + ['boundary_qa_pass', 'physical_qa_pass'])}

## Negative-removal diagnostics
{_md(nd, ['landscape_id', 'scope', 'value', 'units', 'negative_fraction', 'detail'])}

## Boundary QA
{_md(s['bq'], ['landscape_id', 'boundary', 'classification', 'continuity_expected', 'cells', 'seam_step_median', 'seam_step_p95', 'spread_ratio', 'qa_pass'])}

## Shift / uncertainty diagnostics (diagnostic only)
{_md(s['diag'], ['landscape_id', 'shift_vs_gen2_mean_m', 'shift_vs_gen2_sd_m', 'shift_uniformity_index_sd_over_abs_mean', 'D_over_R_L_p99', 'fraction_D_gt_0_5_R_L', 'spearman_spread_vs_control_distance'])}

## Gen-1 vs Gen-2 vs Gen-3
{_md(s['cmp'], ['landscape_id', 'generation', 'median_abs_residual_fraction', 'abs_bias_fraction', 'band_coverage', 'support', 'negative_removal', 'max_slope_deg', 'p99_slope_deg', 'boundary_qa', 'architecture', 'pass_fail'])}

## H
Released: {', '.join(s['verified']) or 'none'} (no H dataset is created when nothing is verified).
{_md(s['red'], ['role', 'feature', 'pooled_r2_on_60_M', 'redundant_r2_ge_0_90'])}

## Integrity
Before: {json.dumps({k: v for k, v in s['ib'].items() if k != 'prior_checkpoints'})}
After: {json.dumps({k: v for k, v in s['ia'].items() if k != 'prior_checkpoints'})}
"""


if __name__ == "__main__":
    main()
