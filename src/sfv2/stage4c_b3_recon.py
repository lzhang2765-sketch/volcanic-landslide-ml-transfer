"""Stage 4C-B3 Gen-3 engine: one-sided (upper-surface) structurally-aware reconstruction.

Re-uses the Gen-2 geometric primitives (read-only import) but replaces the symmetric Huber objective with pre-registered one-sided losses,
uses hard membership across pre-registered geological discontinuities, and selects with the frozen E/U/B/C score.  No Y, no modern-DEM
lower bound: the modern DEM never enters any objective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, sparse

from . import stage4c_b2_prereg as P2
from . import stage4c_b2_recon as G2
from . import stage4c_b3_prereg as P3
from . import stage4c_b_recon as R

URES = 90.0
CompFit = G2.CompFit


# ------------------------------------------------------------------------------------------------ one-sided losses
def loss_weights(r: np.ndarray, s: float, loss: str) -> tuple[np.ndarray, np.ndarray]:
    """IRLS weights and working targets offset for residual r = z - S (r > 0: reference BELOW control)."""
    spec = P3.LOSSES[loss]
    k = P3.LOSS_DEFINITIONS["huber_k"]
    shift = np.zeros_like(r)
    if spec["family"] == "UPPER_QUANTILE":
        m = np.where(r > 0, spec["q"], 1 - spec["q"])
        w = s * m / np.maximum(np.abs(r), 0.1 * s)
    elif spec["family"] == "ASYMMETRIC_ROBUST":
        a = np.abs(r) / (k * s)
        w = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12)) * np.where(r > 0, spec["a"], 1.0)
    elif spec["family"] == "SOFT_UPPER_ENVELOPE":
        a = np.abs(r) / (k * s)
        wh = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12))
        hit = r > s
        w = wh + spec["lam"] * hit
        shift = np.where(hit, spec["lam"] * s / w, 0.0)  # working target z - shift: penalty pulls S towards z - s
    else:
        raise ValueError(loss)
    return w, shift


def check_loss(r: np.ndarray, q: float) -> np.ndarray:
    return r * (q - (r < 0))


def fit_loss(X, z: np.ndarray, pen, loss: str):
    issp = sparse.issparse(X)
    w = np.ones(len(z))
    shift = np.zeros(len(z))
    beta = None
    s = 1.0
    for it in range(P3.LOSS_DEFINITIONS["irls_iterations"]):
        zt = z - shift
        if issp:
            Xw = X.multiply(w[:, None]).tocsr()
            A = (X.T @ Xw).toarray()
        else:
            Xw = X * w[:, None]
            A = X.T @ Xw
        b = Xw.T @ zt
        Gm = A + pen if pen is not None else A
        try:
            beta = np.linalg.solve(Gm, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Gm, b, rcond=None)[0]
        r = z - X @ beta
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1.0)
        if it < 3:  # symmetric Huber start (central), then the one-sided loss
            a = np.abs(r) / (1.345 * s)
            w, shift = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12)), np.zeros(len(z))
        else:
            w, shift = loss_weights(r, s, loss)
    edf = float(np.trace(np.linalg.solve(Gm, A))) if pen is not None else float(X.shape[1])
    return beta, s, edf


# ------------------------------------------------------------------------------------------------ components with history groups
def components(ctx, geom: str) -> list[dict]:
    spec = P3.GEOMETRIES[ctx.lid][geom]
    c = ctx.ctrl
    out, taken_g, taken_c = [], np.zeros(ctx.app.shape, bool), np.zeros(len(c), bool)

    def add(ctype, fam, rule, hg, gmask, cmask, centre=None):
        out.append(dict(component_id=f"{geom}:{len(out)}", ctype=ctype, family=fam, rule=rule, history=hg, gmask=gmask & ctx.app, cmask=cmask, centre=centre))

    for ctype, fam, rule, hg in spec:
        if rule == "ALL":
            add(ctype, fam, rule, hg, ctx.app.copy(), np.ones(len(c), bool))
        elif rule.startswith("CODES:"):
            codes = rule.split(":")[1].split(",")
            add(ctype, fam, rule, hg, np.isin(ctx.code, codes), c.geological_unit.isin(codes).to_numpy())
        elif rule.startswith("DOMAIN:"):
            d = rule.split(":")[1]
            add(ctype, fam, rule, hg, ctx.dom == d, (c.domain == d).to_numpy())
        elif rule in ("RIM_ZONE", "DISTAL_ZONE"):
            z = P2.STRUCTURAL_ZONES["rim_zone_m"]
            g = (ctx.dist_cal <= z) if rule == "RIM_ZONE" else (ctx.dist_cal > z)
            cm = (c.dist_cal <= z).to_numpy() if rule == "RIM_ZONE" else (c.dist_cal > z).to_numpy()
            add(ctype, fam, rule, hg, g, cm, centre=("ASO_CALDERA", *ctx.cal_centroid) if rule == "RIM_ZONE" else None)
        elif rule == "YOUNG_CENTRE":
            young_g, young_c = ctx.dom == "RADIAL", (c.domain == "RADIAL").to_numpy()
            near_g = np.argmin(np.stack([np.hypot(ctx.X - cc.x, ctx.Y - cc.y) for cc in ctx.centres.itertuples()]), axis=0)
            near_c = np.argmin(np.hypot(c.x.to_numpy()[:, None] - ctx.centres.x.to_numpy()[None], c.y.to_numpy()[:, None] - ctx.centres.y.to_numpy()[None]), axis=1)
            for k, cc in enumerate(ctx.centres.itertuples()):
                cm = young_c & (near_c == k)
                if cm.sum() >= P2.FIT["min_centre_controls"]:
                    gm = young_g & (near_g == k)
                    add(ctype, fam, f"YOUNG_CENTRE:{cc.volcano_name}", hg, gm, cm, centre=(cc.volcano_name, cc.x, cc.y))
                    taken_g |= gm & ctx.app
                    taken_c |= cm
            # young cells of under-constrained centres stay YOUNG (history unchanged): they join the nearest qualified young centre
            if any(k["rule"].startswith("YOUNG_CENTRE") for k in out):
                yc = [k for k in out if k["rule"].startswith("YOUNG_CENTRE")]
                rest_y = young_g & ctx.app & ~taken_g
                d = np.stack([np.hypot(ctx.X - k["centre"][1], ctx.Y - k["centre"][2]) for k in yc])
                nn = np.argmin(d, axis=0)
                rest_c = young_c & ~taken_c
                dc = np.stack([np.hypot(c.x - k["centre"][1], c.y - k["centre"][2]) for k in yc])
                nc = np.argmin(dc, axis=0)
                for i, k in enumerate(yc):
                    k["gmask"] |= rest_y & (nn == i)
                    k["cmask"] = k["cmask"] | (rest_c & (nc == i))
                taken_g |= rest_y
                taken_c |= rest_c
        elif rule == "REST":
            add(ctype, fam, rule, hg, ~taken_g, ~taken_c)
        else:
            raise ValueError(rule)
        if fam.startswith("RADIAL") and out[-1]["centre"] is None and len(ctx.centres) and rule != "DOMAIN:RADIAL":
            cc = ctx.centres.iloc[0]
            out[-1]["centre"] = (cc.volcano_name, cc.x, cc.y)
    sig = P2.COMBINATION["blend_sigma_m"] / URES
    smooth = [ndimage.gaussian_filter(k["gmask"].astype(np.float32), sig) for k in out]
    groups = sorted({k["history"] for k in out})
    gsm = {g: np.sum([s for s, k in zip(smooth, out) if k["history"] == g], axis=0) for g in groups}
    # hard group membership of each cell: the group whose component domain contains it, else the strongest smoothed group
    member = np.full(ctx.app.shape, -1)
    for i, g in enumerate(groups):
        for k in out:
            if k["history"] == g:
                member[k["gmask"]] = i
    free = member < 0
    member[free] = np.argmax(np.stack([gsm[g] for g in groups]), axis=0)[free]
    for k, s in zip(out, smooth):
        gi = groups.index(k["history"])
        within = np.divide(s, gsm[k["history"]], out=np.zeros_like(s), where=gsm[k["history"]] > 1e-6)
        k["w_group"] = within  # within-group smooth weight (whole window): continuous group surface
        k["w"] = within * (member == gi)  # hard across geological discontinuities
        rr, cc_ = np.nonzero(k["gmask"])
        k["bbox"] = (ctx.X[0, cc_.min()] - 2000, ctx.X[0, cc_.max()] + 2000, ctx.Y[rr.max(), 0] - 2000, ctx.Y[rr.min(), 0] + 2000) if len(rr) else None
    for k in out:
        k["groups"], k["member"] = groups, member
    return out


# ------------------------------------------------------------------------------------------------ component fit (Gen-2 geometry, one-sided loss)
def fit_component(comp, c: pd.DataFrame, variant: int, pct: int, rep: int, seed: int, centres, loss: str) -> CompFit:
    fam = comp["family"]
    base = fam.split(":")[0]
    kappa = {"STIFF": 10.0, "MEDIUM": 1.0}.get(fam.split(":")[1] if ":" in fam else "", 10.0)
    n0 = len(c)
    if rep >= 0 and n0 >= 6:
        rng = np.random.default_rng(seed)
        c = c.iloc[np.sort(rng.choice(n0, size=max(5, int(round(P3.ENSEMBLE["subsample_fraction"] * n0))), replace=False))]
    if len(c) < 5:
        return CompFit(fam, ok=False, reason="UNDERDETERMINED")
    z = c[f"z{pct}"].to_numpy(float)
    x, y = c.x.to_numpy(float), c.y.to_numpy(float)
    s_ = P2.FAMILIES["PSPLINE_ENVELOPE"]["knot_spacing_m"]
    if base == "PSPLINE_ENVELOPE":
        off = P2.PSPLINE_STRUCTURAL_VARIANTS["knot_offset_fraction"][max(variant, 0) % 2]
        b = comp["bbox"]
        tx, ty = G2.knots(b[0], b[1], s_, off), G2.knots(b[2], b[3], s_, off)
        nx, ny = len(tx) - 4, len(ty) - 4
        beta, s, edf = fit_loss(G2.bs2(x, y, tx, ty), z, kappa * len(z) / (nx * ny) * G2.pspline_penalty(nx, ny), loss)
        return CompFit(fam, s, edf, par=dict(spline=(tx, ty, beta.reshape(nx, ny))))
    pv, rv = P2.RADIAL_STRUCTURAL_VARIANTS["p"], P2.RADIAL_STRUCTURAL_VARIANTS["radius_factor"]
    p_, rf = (pv[variant // 3 % 4], rv[variant % 3]) if variant >= 0 else (2.0, 1.3)
    K = 3 if base in ("RADIAL_SECTOR_K3", "RADIAL_K3_PLUS_PSPLINE", "CALDERA_RIM_SECTOR") else 1
    cents = [(cc.volcano_name, cc.x, cc.y) for cc in centres.itertuples()] if "ALL_YOUNG_CENTRES_UPPER_ENVELOPE" in fam else [comp["centre"]]
    grids = [(None, None)]
    if base == "RADIAL_ELLIPTICAL_K1":
        grids = [(ph, q) for ph in P2.FAMILIES["RADIAL_ELLIPTICAL_K1"]["phi_deg"] for q in P2.FAMILIES["RADIAL_ELLIPTICAL_K1"]["q"]]
    best = None
    for phi, q in grids:
        d = np.stack([G2.radial_cols(x, y, ce, 1, 1e12, 0, phi, q)[1] for ce in cents])
        near = np.argmin(d, axis=0)
        keep = [k for k in range(len(cents)) if (near == k).sum() >= 3]
        if not keep:
            continue
        Rk = {k: rf * float(np.percentile(d[k, near == k], 95)) for k in keep}
        assign = near.copy()
        tx = ty = None
        for _ in range(5 if len(keep) > 1 else 1):
            blocks = [np.ones((len(z), 1))] if base != "RADIAL_K3_PLUS_PSPLINE" else []
            for k in keep:
                cols, _ = G2.radial_cols(x, y, cents[k], p_, Rk[k], K, phi, q)
                blocks.append(cols * (assign == k)[:, None] if len(keep) > 1 else cols)
            Xr = np.hstack(blocks)
            Xf, pen = Xr, None
            if base == "RADIAL_K3_PLUS_PSPLINE":
                b = comp["bbox"]
                off = P2.PSPLINE_STRUCTURAL_VARIANTS["knot_offset_fraction"][max(variant, 0) % 2]
                tx, ty = G2.knots(b[0], b[1], s_, off), G2.knots(b[2], b[3], s_, off)
                nx, ny = len(tx) - 4, len(ty) - 4
                Xf = sparse.hstack([sparse.csr_matrix(Xr), G2.bs2(x, y, tx, ty)]).tocsr()
                pen = np.zeros((Xf.shape[1], Xf.shape[1]))
                pen[Xr.shape[1]:, Xr.shape[1]:] = 10.0 * len(z) / (nx * ny) * G2.pspline_penalty(nx, ny)
            beta, s, edf = fit_loss(Xf, z, pen, loss)
            if len(keep) == 1:
                break
            cones, i = [], 1
            for k in keep:
                cols, _ = G2.radial_cols(x, y, cents[k], p_, Rk[k], K, phi, q)
                cones.append(cols @ beta[i:i + cols.shape[1]])
                i += cols.shape[1]
            new = np.array(keep)[np.argmax(np.stack(cones), axis=0)]
            if (new == assign).all():
                break
            assign = new
        if best is None or s < best[0]:
            best = (s, phi, q, keep, Rk, beta, edf, tx, ty, Xr.shape[1])
    if best is None:
        return CompFit(fam, ok=False, reason="NO_CENTRE_WITH_3_CONTROLS")
    s, phi, q, keep, Rk, beta, edf, tx, ty, nr = best
    par = dict(p=p_, K=K, ue=len(keep) > 1, radial=[], B=0.0 if base == "RADIAL_K3_PLUS_PSPLINE" else float(beta[0]))
    i = 0 if base == "RADIAL_K3_PLUS_PSPLINE" else 1
    ncol = 1 + 2 * K
    fit = CompFit(fam, s, edf + (2 if phi is not None else 0), par=par)
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for k in keep:
        bk = beta[i:i + ncol]
        i += ncol
        par["radial"].append((cents[k], Rk[k], 0.0, bk, phi, q))
        Hth = bk[0] + sum(bk[2 * j - 1] * np.cos(j * th) + bk[2 * j] * np.sin(j * th) for j in range(1, K + 1))
        if bk[0] <= 0:
            fit.ok, fit.reason = False, "NON_POSITIVE_EDIFICE_HEIGHT"
        elif Hth.min() / bk[0] < 0.5:
            fit.ok, fit.reason = False, "ASYMMETRY_BOUND (min H(theta)/c0 < 0.5)"
    if base == "RADIAL_K3_PLUS_PSPLINE":
        par["spline"] = (tx, ty, beta[nr:].reshape(len(tx) - 4, len(ty) - 4))
    return fit


def fit_arch(ctx, comps, train, v, pct, rep, loss, cache=None):
    fits = []
    for i, k in enumerate(comps):
        cv = G2.comp_variant(k, v)
        key = (i, cv, pct, rep)
        if cache is not None and key in cache:
            fits.append(cache[key])
            continue
        f = fit_component(k, ctx.ctrl[train & k["cmask"]], cv, pct, rep, P3.SEED + 7919 * max(rep, 0) + 31 * i, ctx.centres, loss)
        if cache is not None:
            cache[key] = f
        fits.append(f)
    return fits


predict_pts = G2.predict_pts


def group_surface(ctx, comps, fits, Xs, Ys, sl, offset=0.0):
    """Per-history-group continuous surfaces on a sub-grid, plus the hard-membership composite."""
    out = {}
    for g in comps[0]["groups"]:
        num = np.zeros(Xs.shape)
        den = np.zeros(Xs.shape)
        for k, f in zip(comps, fits):
            if k["history"] != g or not f.ok:
                continue
            w = k["w_group"][sl]
            num += w * (f.predict(Xs, Ys) + offset * f.sigma)
            den += w
        out[g] = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    member = comps[0]["member"][sl]
    comp = np.full(Xs.shape, np.nan)
    for i, g in enumerate(comps[0]["groups"]):
        comp = np.where(member == i, out[g], comp)
    return out, comp, member


# ------------------------------------------------------------------------------------------------ selection
def physically_admissible(ctx, comps, train, loss) -> tuple[bool, float]:
    fits = fit_arch(ctx, comps, train, -1, 95, -1, loss)
    from scipy.spatial import cKDTree
    tr = ctx.ctrl[train]
    d = cKDTree(tr[["x", "y"]].to_numpy()).query(np.c_[ctx.X.ravel(), ctx.Y.ravel()])[0].reshape(ctx.X.shape)
    dom = ctx.app & (d <= 3000) & ~ctx.excl
    if not dom.any():
        return True, np.nan
    sl = np.s_[:, :]
    groups, comp, member = group_surface(ctx, comps, fits, ctx.X, ctx.Y, sl)
    mx = 0.0
    for i, g in enumerate(comps[0]["groups"]):
        z = groups[g]
        if not np.isfinite(z).any():
            continue
        sl_ = R.horn(R.nearest_fill(z, np.isfinite(z)), URES)
        m = dom & (member == i)
        if m.any():
            mx = max(mx, float(sl_[m].max()))
    return mx <= P3.GATES["max_reconstructed_slope_deg_constructional"], mx


def inner_score(ctx, comps, train_mask, loss) -> dict:
    sf = ctx.ctrl.super_fold.to_numpy()
    res, edfs, sig = [], [], []
    for g in np.unique(sf[train_mask]):
        tr = train_mask & (sf != g)
        te = np.nonzero(train_mask & (sf == g))[0]
        fits = fit_arch(ctx, comps, tr, -1, 95, -1, loss)
        res.append(ctx.ctrl.z95.to_numpy()[te] - predict_pts(ctx, comps, fits, te))
        edfs.append(sum(f.edf for f in fits))
        sig += [f.sigma for f in fits if f.ok]
    r = np.concatenate(res)
    r = r[np.isfinite(r)]
    tau_in = float(np.median(sig)) if sig else np.inf
    E = float(np.median(np.abs(r)) / ctx.R_ctrl)
    U = float((r > tau_in).mean())
    Bv = float(abs(r.mean()) / ctx.R_ctrl)
    C = float(np.mean(edfs))
    S = P3.SELECTION
    return dict(E=E, U=U, B=Bv, effective_dof=C, tau_in=tau_in, score=E + S["alpha_U"] * U + S["gamma_B"] * Bv + S["beta_C"] * C)


def select(ctx, geoms: dict, train_mask: np.ndarray) -> tuple[tuple, list]:
    rows = []
    for gname, comps in geoms.items():
        for loss in P3.LOSSES:
            sc = inner_score(ctx, comps, train_mask, loss)
            adm, mx = physically_admissible(ctx, comps, train_mask, loss)
            rows.append(dict(geometry=gname, loss=loss, admissible=adm, max_slope_central_deg=mx, **sc))
    pool = [r for r in rows if r["admissible"]] or rows
    best = min(r["score"] for r in pool)
    ok = [r for r in pool if r["score"] <= best + 0.01]
    ch = min(ok, key=lambda r: (round(r["effective_dof"] * 2) / 2, r["score"]))
    for r in rows:
        r["selected"] = r is ch
    return (ch["geometry"], ch["loss"]), rows


def scenarios(comps):
    E = P3.ENSEMBLE
    return [(v, pct, rep) for v in range(G2.n_variants(comps)) for pct in E["summit_percentile"] for rep in range(E["control_subsample_replicates"])]
