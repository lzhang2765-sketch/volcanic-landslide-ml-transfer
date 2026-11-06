"""Stage 4C-B2 Gen-2 engine: structurally-aware piecewise ensemble reconstruction with nested Y-blind validation.

Inputs: Gen-1 Y-blind controls and centres (read-only TSVs), the frozen M DEM / geology / caldera sources through the Gen-1 guarded
loaders.  No function receives or derives landslide Y, Omega labels, Stage 4B outputs or predictive quantities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.interpolate import BSpline

from . import stage4c_b2_prereg as P2
from . import stage4c_b_recon as R
from .schema import ROOT

GEN1 = ROOT / "outputs/stage4c_b"
URES = 90.0


# ------------------------------------------------------------------------------------------------ context
@dataclass
class Ctx:
    lid: str
    L: object
    X: np.ndarray
    Y: np.ndarray
    app: np.ndarray          # applicable geology (90 m)
    code: np.ndarray         # majority source code (90 m), "" if none
    dom: np.ndarray          # Gen-1 domain name (90 m), "" if none
    excl: np.ndarray         # caldera interior or vent core (90 m)
    cal: np.ndarray          # caldera interior (90 m)
    dist_cal: np.ndarray
    z90: np.ndarray
    valid90: np.ndarray
    ctrl: pd.DataFrame       # Q1/Q2 fit controls with attributes
    centres: pd.DataFrame    # registered radial centres (projected)
    cal_centroid: tuple | None
    R_ctrl: float
    extra: dict = field(default_factory=dict)

    def cell(self, x, y):
        c = np.clip(((np.asarray(x) - self.L.transform.c) / URES).astype(int), 0, self.X.shape[1] - 1)
        r = np.clip(((self.L.transform.f - np.asarray(y)) / URES).astype(int), 0, self.X.shape[0] - 1)
        return r, c


def load_ctx(lid: str) -> Ctx:
    L = R.load_landscape(lid)
    dnames, look = R.domain_lookup(L)
    dom30 = look[L.code]
    cen = pd.read_csv(R.guard(GEN1 / "STAGE4C_B_VOLCANO_CENTRES.tsv"), sep="\t", keep_default_na=False, na_values=[""])
    cen = cen[(cen.landscape_id == lid)]
    cat = cen[cen.catalogue_layer.isin(["HOLOCENE", "PLEISTOCENE"])].rename(columns={"catalogue_elevation_m": "elevation_m"})
    proj = R.project_centres(L, cat[["landscape_id", "volcano_name", "latitude", "longitude", "elevation_m"]])
    cin = proj[proj.inside_window].reset_index(drop=True)
    cal30, cal_info = R.caldera_mask(L, cin)
    vent30 = R.vent_mask(L, cin)
    radial = cin[cin.volcano_name.isin(cen[cen.role == "RADIAL_RECONSTRUCTION_CENTRE"].volcano_name)].reset_index(drop=True)
    h, w = L.shape90
    cnt = R.block90(L.valid.astype(float), "sum")
    z90 = np.where(cnt >= 1, R.block90(L.z, "mean"), np.nan)
    d90 = R.domain90(L, dom30, len(dnames))
    dom = np.array([""] + [dnames[k] for k in sorted(dnames)], dtype=object)[d90]
    has = (L.code > 0) & L.valid
    idx = np.unique(L.code[has])
    counts = np.stack([R.block90((has & (L.code == i)).astype(float), "sum") for i in idx])
    code = np.array(L.codes, dtype=object)[idx[np.argmax(counts, axis=0)]]
    code[counts.sum(axis=0) == 0] = ""
    cal = R.block90(cal30.astype(float), "mean") >= 0.5
    excl = cal | (R.block90(vent30.astype(float), "mean") >= 0.5)
    dist_cal = ndimage.distance_transform_edt(~cal) * URES if cal.any() else np.full((h, w), np.inf)
    X, Y = np.meshgrid(L.x90, L.y90)
    c = pd.read_csv(R.guard(GEN1 / "STAGE4C_B_CONTROLS.tsv"), sep="\t", keep_default_na=False, na_values=[""])
    c = c[(c.landscape_id == lid) & (c.included_in_fit.astype(str) == "True")].reset_index(drop=True)
    c["domain"] = c.family_relevance.str.split(":").str[0]
    c["z90"], c["z95"], c["z99"] = c.elevation_p90, c.elevation, c.elevation_p99
    ctx = Ctx(lid, L, X, Y, d90 > 0, code, dom, excl, cal, dist_cal, z90, cnt >= 5, c, radial,
              (cal_info.get("centroid_x"), cal_info.get("centroid_y")) if cal.any() and "centroid_x" in cal_info else None,
              float(np.percentile(c.z95, 98) - np.percentile(c.z95, 2)), dict(cal_info=cal_info, dnames=dnames))
    r, cc = ctx.cell(c.x, c.y)
    c["dist_cal"] = dist_cal[r, cc]
    c["cell_r"], c["cell_c"] = r, cc
    assign_groups(ctx)
    return ctx


def assign_groups(ctx: Ctx) -> None:
    """Outer validation groups + super-folds (frozen in P2.NESTED)."""
    c = ctx.ctrl
    radial_landscape = ctx.lid in ("LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019")
    if len(ctx.centres):
        d = np.hypot(c.x.to_numpy()[:, None] - ctx.centres.x.to_numpy()[None], c.y.to_numpy()[:, None] - ctx.centres.y.to_numpy()[None])
        k = np.argmin(d, axis=1)
        c["near_centre"] = ctx.centres.volcano_name.to_numpy()[k]
        c["r_near"] = d[np.arange(len(c)), k]
        az = np.degrees(np.arctan2(c.x.to_numpy() - ctx.centres.x.to_numpy()[k], c.y.to_numpy() - ctx.centres.y.to_numpy()[k])) % 360
        c["sector"] = (az // 45).astype(int)
    else:
        c["near_centre"], c["r_near"], c["sector"] = "", np.nan, -1
    use_sector = np.full(len(c), radial_landscape) | ((ctx.lid == "DOMINICA_ERIKA_2015") & (c.domain == "RADIAL")).to_numpy()
    bx, by = np.floor(c.x / 2000).astype(int), np.floor(c.y / 2000).astype(int)
    tx, ty = np.floor(c.x / 16000).astype(int), np.floor(c.y / 16000).astype(int)
    c["outer_group"] = np.where(use_sector, c.near_centre.astype(str) + "_S" + c.sector.astype(str), "B" + bx.astype(str) + "_" + by.astype(str))
    c["super_fold"] = np.where(use_sector, c.sector, (tx % 4) + 4 * (ty % 2)).astype(int)


# ------------------------------------------------------------------------------------------------ components
def components(ctx: Ctx, arch: str) -> list[dict]:
    spec = P2.CANDIDATES[ctx.lid][arch]
    c = ctx.ctrl
    gcode, gdom, gapp = ctx.code, ctx.dom, ctx.app
    out, taken_g, taken_c = [], np.zeros(gapp.shape, bool), np.zeros(len(c), bool)

    def add(ctype, fam, rule, gmask, cmask, centre=None):
        out.append(dict(component_id=f"{arch}:{len(out)}", ctype=ctype, family=fam, rule=rule, gmask=gmask & gapp, cmask=cmask, centre=centre))

    for ctype, fam, rule in spec:
        if rule == "ALL":
            add(ctype, fam, rule, gapp.copy(), np.ones(len(c), bool))
        elif rule.startswith("CODES:"):
            codes = rule.split(":")[1].split(",")
            add(ctype, fam, rule, np.isin(gcode, codes), c.geological_unit.isin(codes).to_numpy())
        elif rule.startswith("DOMAIN:"):
            d = rule.split(":")[1]
            add(ctype, fam, rule, gdom == d, (c.domain == d).to_numpy())
        elif rule in ("RIM_ZONE", "DISTAL_ZONE"):
            z = P2.STRUCTURAL_ZONES["rim_zone_m"]
            g = (ctx.dist_cal <= z) if rule == "RIM_ZONE" else (ctx.dist_cal > z)
            cm = (c.dist_cal <= z).to_numpy() if rule == "RIM_ZONE" else (c.dist_cal > z).to_numpy()
            add(ctype, fam, rule, g, cm, centre=("ASO_CALDERA", *ctx.cal_centroid) if rule == "RIM_ZONE" else None)
        elif rule == "YOUNG_CENTRE":
            young_g = gdom == "RADIAL"
            young_c = (c.domain == "RADIAL").to_numpy()
            d = np.stack([np.hypot(ctx.X - cc.x, ctx.Y - cc.y) for cc in ctx.centres.itertuples()])
            near_g = np.argmin(d, axis=0)
            dc = np.hypot(c.x.to_numpy()[:, None] - ctx.centres.x.to_numpy()[None], c.y.to_numpy()[:, None] - ctx.centres.y.to_numpy()[None])
            near_c = np.argmin(dc, axis=1)
            for k, cc in enumerate(ctx.centres.itertuples()):
                cm = young_c & (near_c == k)
                if cm.sum() >= P2.FIT["min_centre_controls"]:  # qualification uses control COUNTS only (locations), never elevations/residuals
                    gm = young_g & (near_g == k)
                    add(ctype, fam, f"YOUNG_CENTRE:{cc.volcano_name}", gm, cm, centre=(cc.volcano_name, cc.x, cc.y))
                    taken_g |= gm & gapp
                    taken_c |= cm
        elif rule == "REST":
            add(ctype, fam, rule, ~taken_g, ~taken_c)
        else:
            raise ValueError(rule)
        if fam.startswith("RADIAL") and out[-1]["centre"] is None and len(ctx.centres) and rule != "DOMAIN:RADIAL":
            cc = ctx.centres.iloc[0]
            out[-1]["centre"] = (cc.volcano_name, cc.x, cc.y)
    sig = P2.COMBINATION["blend_sigma_m"] / URES
    ws = [ndimage.gaussian_filter(k["gmask"].astype(np.float32), sig) for k in out]
    tot = np.sum(ws, axis=0)
    for k, wk in zip(out, ws):
        k["w"] = np.divide(wk, tot, out=np.zeros_like(wk), where=tot > 1e-6)
        rr, cc_ = np.nonzero(k["gmask"])
        k["bbox"] = (ctx.X[0, cc_.min()] - 2000, ctx.X[0, cc_.max()] + 2000, ctx.Y[rr.max(), 0] - 2000, ctx.Y[rr.min(), 0] + 2000) if len(rr) else None
    return out


# ------------------------------------------------------------------------------------------------ fitting primitives
def huber(X, z, pen=None, iters=None):
    """Penalised Huber IRLS; X dense or sparse; pen = (lambda * P) dense or None.  Returns beta, sigma, edf."""
    iters = iters or P2.FIT["irls_iterations"]
    w = np.ones(len(z))
    issp = sparse.issparse(X)
    for _ in range(iters):
        if issp:
            Xw = X.multiply(w[:, None]).tocsr()
            A = (X.T @ Xw).toarray()
            b = Xw.T @ z
        else:
            Xw = X * w[:, None]
            A = X.T @ Xw
            b = Xw.T @ z
        G = A + pen if pen is not None else A
        try:
            beta = np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(G, b, rcond=None)[0]
        r = z - X @ beta
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), P2.FIT["sigma_floor_m"])
        a = np.abs(r) / (P2.FIT["huber_k"] * s)
        w = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12))
    edf = float(np.trace(np.linalg.solve(G, A))) if pen is not None else float(X.shape[1])
    return beta, s, edf


def knots(lo, hi, s, off):
    start = (np.floor(lo / s) - 4 + off) * s
    t = np.arange(start, hi + 6 * s, s)
    return t


def bs1(x, t):
    x = np.clip(x, t[3] + 1e-6, t[-4] - 1e-6)
    return BSpline.design_matrix(x, t, 3).tocsr()


def bs2(x, y, tx, ty):
    Bx, By = bs1(x, tx), bs1(y, ty)
    n, ny = len(x), By.shape[1]
    ix = Bx.indices.reshape(n, 4)
    iy = By.indices.reshape(n, 4)
    dx = Bx.data.reshape(n, 4)
    dy = By.data.reshape(n, 4)
    cols = (ix[:, :, None] * ny + iy[:, None, :]).reshape(n, 16)
    vals = (dx[:, :, None] * dy[:, None, :]).reshape(n, 16)
    return sparse.csr_matrix((vals.ravel(), cols.ravel(), np.arange(0, 16 * n + 1, 16)), shape=(n, Bx.shape[1] * ny))


def pspline_penalty(nx, ny):
    def d2(n):
        return np.diff(np.eye(n), 2, axis=0)
    Dx, Dy = d2(nx), d2(ny)
    return np.kron(Dx.T @ Dx, np.eye(ny)) + np.kron(np.eye(nx), Dy.T @ Dy)


def radial_cols(x, y, centre, p, R, K, phi=None, q=None):
    dx, dy = np.asarray(x) - centre[1], np.asarray(y) - centre[2]
    if phi is not None:
        a = np.radians(phi)
        u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
        r = np.sqrt(q * u * u + v * v / q)
    else:
        r = np.hypot(dx, dy)
    th = np.arctan2(dx, dy)
    f = np.clip(1 - r / R, 0, None) ** p
    t = r / (r + 1000.0)
    cols = [f]
    for k in range(1, K + 1):
        cols += [f * t * np.cos(k * th), f * t * np.sin(k * th)]
    return np.stack(cols, axis=-1), r


@dataclass
class CompFit:
    family: str
    sigma: float = 1.0
    edf: float = 0.0
    ok: bool = True
    reason: str = ""
    par: dict = field(default_factory=dict)

    def predict(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        p = self.par
        if self.family == "QUADRATIC_ENVELOPE":
            u, v = (x - p["xm"]) / p["ls"], (y - p["ym"]) / p["ls"]
            return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1) @ p["beta"]
        out = np.zeros(np.broadcast(x, y).shape)
        if "radial" in p:
            for (centre, R_, B, beta, phi, q) in p["radial"]:
                cols, _ = radial_cols(x, y, centre, p["p"], R_, p["K"], phi, q)
                cone = cols @ beta
                out = np.maximum(out, cone) if p.get("ue") else out + cone
            out = out + p.get("B", 0.0)
        if "spline" in p:
            tx, ty, C = p["spline"]
            if x.ndim == 2:  # rectilinear grid
                gx = bs1(x[0], tx)
                gy = bs1(y[:, 0], ty)
                out = out + (gy @ (gx @ C).T)
            else:
                out = out + bs2(x.ravel(), y.ravel(), tx, ty) @ C.ravel()
        return out


def fit_component(comp, c: pd.DataFrame, variant: int, pct: int, rep: int, seed: int, all_centres=None) -> CompFit:
    fam = comp["family"]
    base = fam.split(":")[0]
    kappa = {"STIFF": 10.0, "MEDIUM": 1.0}.get(fam.split(":")[1] if ":" in fam else "", 10.0)
    n0 = len(c)
    if rep >= 0 and n0 >= 6:
        rng = np.random.default_rng(seed)
        c = c.iloc[np.sort(rng.choice(n0, size=max(5, int(round(P2.ENSEMBLE["subsample_fraction"] * n0))), replace=False))]
    z = c[f"z{pct}"].to_numpy(float)
    x, y = c.x.to_numpy(float), c.y.to_numpy(float)
    if len(c) < 5:
        return CompFit(fam, ok=False, reason="UNDERDETERMINED")
    if base == "QUADRATIC_ENVELOPE":
        xm, ym = x.mean(), y.mean()
        ls = max(np.hypot(x - xm, y - ym).max(), 1000.0)
        u, v = (x - xm) / ls, (y - ym) / ls
        beta, s, edf = huber(np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1), z)
        return CompFit(fam, s, edf, par=dict(xm=xm, ym=ym, ls=ls, beta=beta))
    if base == "PSPLINE_ENVELOPE":
        off = P2.PSPLINE_STRUCTURAL_VARIANTS["knot_offset_fraction"][variant % 2]
        s_ = P2.FAMILIES["PSPLINE_ENVELOPE"]["knot_spacing_m"]
        b = comp["bbox"]
        tx, ty = knots(b[0], b[1], s_, off), knots(b[2], b[3], s_, off)
        X = bs2(x, y, tx, ty)
        nx, ny = len(tx) - 4, len(ty) - 4
        pen = kappa * len(z) / (nx * ny) * pspline_penalty(nx, ny)
        beta, s, edf = huber(X, z, pen)
        return CompFit(fam, s, edf, par=dict(spline=(tx, ty, beta.reshape(nx, ny))))
    # radial families
    pv, rv = P2.RADIAL_STRUCTURAL_VARIANTS["p"], P2.RADIAL_STRUCTURAL_VARIANTS["radius_factor"]
    p_, rf = (pv[variant // 3 % 4], rv[variant % 3]) if variant >= 0 else (2.0, 1.3)
    K = 3 if base in ("RADIAL_SECTOR_K3", "RADIAL_K3_PLUS_PSPLINE", "CALDERA_RIM_SECTOR") else 1
    if "ALL_YOUNG_CENTRES_UPPER_ENVELOPE" in fam:
        cents = [(cc.volcano_name, cc.x, cc.y) for cc in all_centres.itertuples()]
    else:
        cents = [comp["centre"]]
    grids = [(None, None)]
    if base == "RADIAL_ELLIPTICAL_K1":
        grids = [(ph, q) for ph in P2.FAMILIES["RADIAL_ELLIPTICAL_K1"]["phi_deg"] for q in P2.FAMILIES["RADIAL_ELLIPTICAL_K1"]["q"]]
    best = None
    for phi, q in grids:
        d = np.stack([radial_cols(x, y, ce, 1, 1e12, 0, phi, q)[1] for ce in cents])
        near = np.argmin(d, axis=0)
        keep = [k for k in range(len(cents)) if (near == k).sum() >= 3]
        if not keep:
            continue
        Rk = {k: rf * float(np.percentile(d[k, near == k], 95)) for k in keep}
        assign = near.copy()
        for _ in range(5 if len(keep) > 1 else 1):
            blocks = [np.ones((len(z), 1))] if base != "RADIAL_K3_PLUS_PSPLINE" else []
            for k in keep:
                cols, _ = radial_cols(x, y, cents[k], p_, Rk[k], K, phi, q)
                blocks.append(cols * (assign == k)[:, None] if len(keep) > 1 else cols)
            Xr = np.hstack(blocks)
            pen = None
            Xf = Xr
            if base == "RADIAL_K3_PLUS_PSPLINE":
                s_ = P2.FAMILIES["PSPLINE_ENVELOPE"]["knot_spacing_m"]
                b = comp["bbox"]
                off = P2.PSPLINE_STRUCTURAL_VARIANTS["knot_offset_fraction"][max(variant, 0) % 2]
                tx, ty = knots(b[0], b[1], s_, off), knots(b[2], b[3], s_, off)
                Xs = bs2(x, y, tx, ty)
                nx, ny = len(tx) - 4, len(ty) - 4
                Xf = sparse.hstack([sparse.csr_matrix(Xr), Xs]).tocsr()
                pen = np.zeros((Xf.shape[1], Xf.shape[1]))
                pen[Xr.shape[1]:, Xr.shape[1]:] = 10.0 * len(z) / (nx * ny) * pspline_penalty(nx, ny)
            beta, s, edf = huber(Xf, z, pen)
            if len(keep) == 1:
                break
            cones = []
            i = 1
            for k in keep:
                cols, _ = radial_cols(x, y, cents[k], p_, Rk[k], K, phi, q)
                cones.append(cols @ beta[i:i + cols.shape[1]])
                i += cols.shape[1]
            new = np.array(keep)[np.argmax(np.stack(cones), axis=0)]
            if (new == assign).all():
                break
            assign = new
        if best is None or s < best[0]:
            best = (s, phi, q, keep, Rk, beta, edf, locals().get("tx"), locals().get("ty"), Xr.shape[1])
    if best is None:
        return CompFit(fam, ok=False, reason="NO_CENTRE_WITH_3_CONTROLS")
    s, phi, q, keep, Rk, beta, edf, tx, ty, nr = best
    par = dict(p=p_, K=K, ue=len(keep) > 1, radial=[])
    i = 0 if base == "RADIAL_K3_PLUS_PSPLINE" else 1
    par["B"] = 0.0 if base == "RADIAL_K3_PLUS_PSPLINE" else float(beta[0])
    ncol = 1 + 2 * K
    fit = CompFit(fam, s, edf + (2 if phi is not None else 0), par=par)
    for k in keep:
        bk = beta[i:i + ncol]
        i += ncol
        par["radial"].append((cents[k], Rk[k], 0.0, bk, phi, q))
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        Hth = bk[0] + sum(bk[2 * j - 1] * np.cos(j * th) + bk[2 * j] * np.sin(j * th) for j in range(1, K + 1))
        if bk[0] <= 0:
            fit.ok, fit.reason = False, "NON_POSITIVE_EDIFICE_HEIGHT"
        elif Hth.min() / bk[0] < 0.5:
            fit.ok, fit.reason = False, "ASYMMETRY_BOUND (min H(theta)/c0 < 0.5)"
    if base == "RADIAL_K3_PLUS_PSPLINE":
        nx, ny = len(tx) - 4, len(ty) - 4
        par["spline"] = (tx, ty, beta[nr:].reshape(nx, ny))
    return fit


# ------------------------------------------------------------------------------------------------ architecture fit / predict
def n_variants(comps) -> int:
    v = 1
    for k in comps:
        b = k["family"].split(":")[0]
        v = max(v, 12 if b.startswith(("RADIAL", "CALDERA")) else 2 if b == "PSPLINE_ENVELOPE" else 1)
    return v


def comp_variant(comp, v):
    b = comp["family"].split(":")[0]
    if v < 0:
        return -1 if b.startswith(("RADIAL", "CALDERA")) else 0
    return v % (12 if b.startswith(("RADIAL", "CALDERA")) else 2 if b == "PSPLINE_ENVELOPE" else 1)


def fit_arch(ctx: Ctx, comps, train: np.ndarray, v: int, pct: int, rep: int, cache: dict | None = None) -> list[CompFit]:
    fits = []
    for i, k in enumerate(comps):
        m = train & k["cmask"]
        cv = comp_variant(k, v)
        key = (i, cv, pct, rep)
        if cache is not None and key in cache:
            fits.append(cache[key])
            continue
        f = fit_component(k, ctx.ctrl[m], cv, pct, rep, P2.SEED + 7919 * max(rep, 0) + 31 * i, ctx.centres)
        if cache is not None:
            cache[key] = f
        fits.append(f)
    return fits


def predict_pts(ctx: Ctx, comps, fits, idx: np.ndarray, offset: float = 0.0) -> np.ndarray:
    c = ctx.ctrl.iloc[idx]
    r, cc = c.cell_r.to_numpy(), c.cell_c.to_numpy()
    num = np.zeros(len(c))
    den = np.zeros(len(c))
    for k, f in zip(comps, fits):
        if not f.ok:
            continue
        w = k["w"][r, cc]
        if not (w > 0).any():
            continue
        num += w * (f.predict(c.x.to_numpy(), c.y.to_numpy()) + offset * f.sigma)
        den += w
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)


def arch_edf(fits) -> float:
    return float(sum(f.edf for f in fits))


# ------------------------------------------------------------------------------------------------ nested selection
def inner_score(ctx, comps, train_mask: np.ndarray) -> tuple[float, float]:
    """Leave-one-super-fold-out inside train_mask; central variant, P95, no subsample, no offset."""
    res, edfs = [], []
    sf = ctx.ctrl.super_fold.to_numpy()
    for g in np.unique(sf[train_mask]):
        tr = train_mask & (sf != g)
        te = np.nonzero(train_mask & (sf == g))[0]
        fits = fit_arch(ctx, comps, tr, -1, 95, -1)
        pred = predict_pts(ctx, comps, fits, te)
        res.append(ctx.ctrl.z95.to_numpy()[te] - pred)
        edfs.append(arch_edf(fits))
    r = np.concatenate(res)
    r = r[np.isfinite(r)]
    ne = float(np.median(np.abs(r)) / ctx.R_ctrl) if len(r) else np.inf
    return ne, float(np.mean(edfs))


def select(ctx, arch_comps: dict, train_mask: np.ndarray) -> tuple[str, list]:
    rows = []
    for a, comps in arch_comps.items():
        ne, edf = inner_score(ctx, comps, train_mask)
        rows.append(dict(architecture=a, inner_norm_error=ne, effective_dof=edf, score=ne + P2.COMPLEXITY_RULE["lambda_per_dof"] * edf))
    best = min(r["score"] for r in rows)
    ok = [r for r in rows if r["score"] <= best + P2.COMPLEXITY_RULE["simpler_within_tolerance"]]
    chosen = min(ok, key=lambda r: (r["effective_dof"], r["score"]))["architecture"]
    for r in rows:
        r["selected"] = r["architecture"] == chosen
    return chosen, rows


def scenarios(comps) -> list[tuple]:
    E = P2.ENSEMBLE
    return [(v, pct, rep) for v in range(n_variants(comps)) for pct in E["summit_percentile"] for rep in range(E["control_subsample_replicates"])]
