"""Stage 4C-B reconstruction engine: Y-blind controls, scenario ensembles, blocked validation and CORE_H operators.

Inputs are restricted to the modern M DEM, frozen geology rasters/mapping tables, the GSJ Aso caldera map, the Copernicus water-body
mask and public GVP centres.  Every file access goes through `guard`, which refuses target/omega/Stage 4B/lockbox paths.  No function
here receives, reads or derives landslide occurrence (Y), Omega labels, Stage 4B outputs, or any predictive result.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from . import stage4a_base as B
from . import stage4c_b_prereg as P
from .schema import ROOT
from .stage2d_production import _disk, _focal_mean  # frozen M focal operators (TPI exactly as M)

PROC = ROOT / "data/processed"
RES = 30.0
URES = 90.0
UNIT_COLUMNS = ["terrain_unit_id", "centroid_x", "centroid_y", "projected_crs", "source_grid_row", "source_grid_col", "source_unit_code"]


# ------------------------------------------------------------------------------------------------ Y firewall
def guard(path) -> Path:
    """Fail closed on any path that could carry target, Omega, Stage 4B or lockbox information."""
    p = Path(path)
    for comp in p.parts:
        c = comp.lower()
        if c.startswith(P.FORBIDDEN_PATH_PREFIX) or any(t in c for t in P.FORBIDDEN_PATH_TOKENS):
            raise PermissionError(f"Y-FIREWALL: refused {p}")
        if any(t in c for t in ("vanuatu", "popocatepetl", "chuuk", "elgon")):
            raise PermissionError(f"LOCKBOX FIREWALL: refused {p}")
    return p


# ------------------------------------------------------------------------------------------------ grids
def horn(z: np.ndarray, res: float) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / (8 * res)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64) / (8 * res)
    p = ndimage.convolve(z, kx, mode="nearest")
    q = ndimage.convolve(z, ky, mode="nearest")
    return np.degrees(np.arctan(np.hypot(p, q)))


def relief(z: np.ndarray, radius_cells: float) -> np.ndarray:
    fp = _disk(radius_cells)
    return ndimage.maximum_filter(z, footprint=fp, mode="nearest") - ndimage.minimum_filter(z, footprint=fp, mode="nearest")


def nearest_fill(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return z
    ix = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[ix[0], ix[1]]


@dataclass
class Landscape:
    lid: str
    crs: object
    transform: object          # 30 m transform of the cropped window
    z: np.ndarray              # 30 m DEM (NaN invalid)
    valid: np.ndarray
    code: np.ndarray           # 30 m source unit code index into `codes` (0 = no geology)
    codes: np.ndarray          # index -> source_unit_code ("" at 0)
    units: pd.DataFrame        # Stage 4A units with window-relative unit-grid row/col
    r0: int = 0
    c0: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def x30(self):
        return self.transform.c + (np.arange(self.z.shape[1]) + 0.5) * RES

    @property
    def y30(self):
        return self.transform.f - (np.arange(self.z.shape[0]) + 0.5) * RES

    @property
    def shape90(self):
        return self.z.shape[0] // 3, self.z.shape[1] // 3

    @property
    def x90(self):
        return self.transform.c + (np.arange(self.shape90[1]) + 0.5) * URES

    @property
    def y90(self):
        return self.transform.f - (np.arange(self.shape90[0]) + 0.5) * URES


def load_landscape(lid: str) -> Landscape:
    B.firewall_check(lid)
    s = P.SOURCES[lid]
    d = PROC / s["dir"] / lid
    units = B.load(lid, "terrain_units/terrain_units", usecols=UNIT_COLUMNS)
    with rasterio.open(guard(d / s["dem"])) as r:
        z, T, crs, nod = r.read(1).astype(np.float64), r.transform, r.crs, r.nodata
    with rasterio.open(guard(d / s["geo"])) as g:
        gi = g.read(1).astype(np.int32)
        assert g.transform == T and g.shape == z.shape
    mapping = pd.read_csv(guard(ROOT / s["mapping"]), usecols=P.MAPPING_COLUMNS, keep_default_na=False)
    codes = np.array([""] + mapping.source_unit_code.astype(str).tolist(), dtype=object)
    # identity check against the frozen unit table (STOP on disagreement)
    gidx = B.load(lid, "terrain_units/terrain_units", usecols=["geology_index", "source_unit_code"])
    ok = gidx.geology_index.to_numpy() > 0
    if not (codes[gidx.geology_index.to_numpy()[ok]] == gidx.source_unit_code.to_numpy()[ok]).all():
        raise RuntimeError(f"STOP: geology index mapping disagrees with frozen Stage 4A units for {lid}")
    valid = np.isfinite(z) & (z != nod)
    z[~valid] = np.nan
    # window = bbox(unit centroids) + buffer, aligned to the 90 m unit grid
    buf = P.WINDOW_BUFFER_M
    cmin = max(0, int((units.centroid_x.min() - buf - T.c) // RES) // 3 * 3)
    cmax = min(z.shape[1] // 3 * 3, -(-int((units.centroid_x.max() + buf - T.c) // RES) // 3) * 3 + 3)
    rmin = max(0, int((T.f - units.centroid_y.max() - buf) // RES) // 3 * 3)
    rmax = min(z.shape[0] // 3 * 3, -(-int((T.f - units.centroid_y.min() + buf) // RES) // 3) * 3 + 3)
    Tw = rasterio.Affine(T.a, T.b, T.c + cmin * RES, T.d, T.e, T.f - rmin * RES)
    u = units.copy()
    u["ur"] = (u.source_grid_row - rmin) // 3
    u["uc"] = (u.source_grid_col - cmin) // 3
    assert ((u.source_grid_row - rmin) % 3 == 0).all() and (u.ur >= 0).all() and (u.uc >= 0).all()
    L = Landscape(lid, crs, Tw, z[rmin:rmax, cmin:cmax], valid[rmin:rmax, cmin:cmax], gi[rmin:rmax, cmin:cmax], codes, u, rmin, cmin)
    L.extra["src_dir"] = d
    L.extra["mapping"] = mapping
    return L


def domain_lookup(L: Landscape) -> tuple[dict, np.ndarray]:
    """Map every geology index to a reconstruction domain id (0 = not applicable)."""
    rules = P.APPLICABLE[L.lid]
    names = list(rules)
    look = np.zeros(len(L.codes), np.int8)
    m = L.extra["mapping"]
    for k, (dname, rule) in enumerate(rules.items(), start=1):
        for i in range(1, len(L.codes)):
            row = m.iloc[i - 1]
            if rule["kind"] == "codes":
                hit = row.source_unit_code in rule["codes"]
            else:
                hit = (rule["age"] in str(row.source_age)) and (row.harmonised_class in rule["classes"])
            if hit:
                look[i] = k
    return {k: n for k, n in enumerate(names, start=1)}, look


def block90(a: np.ndarray, how: str) -> np.ndarray:
    h, w = a.shape[0] // 3 * 3, a.shape[1] // 3 * 3
    x = a[:h, :w].reshape(h // 3, 3, w // 3, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "mean":
            return np.nanmean(x, axis=(1, 3))
        if how == "sum":
            return np.nansum(x, axis=(1, 3))
        if how == "any":
            return x.any(axis=(1, 3))
    raise ValueError(how)


def domain90(L: Landscape, dom30: np.ndarray, ndom: int) -> np.ndarray:
    """Majority domain of the geology-bearing 30 m cells of each 90 m cell (0 if the non-applicable majority wins)."""
    has = (L.code > 0) & L.valid
    counts = [block90((has & (dom30 == 0)).astype(float), "sum")] + [block90((has & (dom30 == k)).astype(float), "sum") for k in range(1, ndom + 1)]
    c = np.stack(counts)
    out = np.argmax(c, axis=0).astype(np.int8)
    out[c.sum(axis=0) == 0] = 0
    return out


# ------------------------------------------------------------------------------------------------ centres / exclusions
def project_centres(L: Landscape, centres: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, L.crs, always_xy=True)
    c = centres[centres.landscape_id == L.lid].copy()
    if len(c):
        c["x"], c["y"] = tr.transform(c.longitude.to_numpy(), c.latitude.to_numpy())
        x0, x1 = L.transform.c, L.transform.c + L.z.shape[1] * RES
        y1, y0 = L.transform.f, L.transform.f - L.z.shape[0] * RES
        c["inside_window"] = (c.x >= x0) & (c.x <= x1) & (c.y >= y0) & (c.y <= y1)
    return c


def caldera_mask(L: Landscape, centres: pd.DataFrame) -> tuple[np.ndarray, dict]:
    cfg = P.CALDERA.get(L.lid)
    mask = np.zeros(L.z.shape, bool)
    if cfg is None:
        return mask, dict(method="NONE")
    if cfg["method"] == "GSJ_ASO_POST_CALDERA_UNITS":
        import geopandas as gpd
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        g = gpd.read_file(guard(ROOT / P.ASO_CALDERA_SOURCE)).to_crs(L.crs)
        sel = g[(g.LEGEND04E == cfg["legend04e"]) & ~g.LEGEND05E.isin(cfg["exclude_symbols_legend05e"])]
        u = unary_union(sel.geometry.buffer(0)).buffer(cfg["closing_m"]).buffer(-cfg["closing_m"])
        geoms = list(u.geoms) if hasattr(u, "geoms") else [u]
        big = max(geoms, key=lambda q: q.area)
        filled = Polygon(big.exterior).buffer(cfg["buffer_m"])
        mask = rasterize([(filled, 1)], out_shape=L.z.shape, transform=L.transform, fill=0, dtype="uint8").astype(bool)
        return mask, dict(method=cfg["method"], interior_area_km2=round(filled.area / 1e6, 2), units_used=int(len(sel)),
                          centroid_x=round(filled.centroid.x, 1), centroid_y=round(filled.centroid.y, 1))
    if cfg["method"] == "WBM_LAKE_RAY_RIM":
        with rasterio.open(guard(L.extra["src_dir"] / cfg["wbm"])) as w:
            wbm = w.read(1)[L.r0:L.r0 + L.z.shape[0], L.c0:L.c0 + L.z.shape[1]]
        c = centres[centres.volcano_name == "Rinjani"].iloc[0]
        lab, n = ndimage.label(wbm == cfg["lake_class"])
        best, info = None, {}
        X, Y = np.meshgrid(L.x30, L.y30)
        for i in range(1, n + 1):
            rr, cc = np.nonzero(lab == i)
            cx, cy = L.x30[cc].mean(), L.y30[rr].mean()
            if np.hypot(cx - c.x, cy - c.y) <= cfg["search_km"] * 1000 and (best is None or len(rr) > best[0]):
                best = (len(rr), cx, cy)
        if best is None:
            return mask, dict(method=cfg["method"], status="NO_LAKE_FOUND")
        _, cx, cy = best
        rmax = cfg["rim_max_km"] * 1000
        ang = np.linspace(0, 2 * np.pi, cfg["n_rays"], endpoint=False)
        rr_ = np.arange(0, rmax + RES, RES)
        zf = nearest_fill(L.z, L.valid)
        rim = np.zeros(len(ang))
        for k, a in enumerate(ang):
            px, py = cx + rr_ * np.cos(a), cy + rr_ * np.sin(a)
            col = np.clip(((px - L.transform.c) / RES).astype(int), 0, L.z.shape[1] - 1)
            row = np.clip(((L.transform.f - py) / RES).astype(int), 0, L.z.shape[0] - 1)
            rim[k] = rr_[np.argmax(zf[row, col])]
        th = np.mod(np.arctan2(Y - cy, X - cx), 2 * np.pi)
        rim_at = np.interp(th, np.r_[ang, 2 * np.pi], np.r_[rim, rim[0]])
        mask = np.hypot(X - cx, Y - cy) < rim_at
        return mask, dict(method=cfg["method"], lake_cells=int(best[0]), centre_x=round(cx, 1), centre_y=round(cy, 1),
                          rim_radius_km_median=round(float(np.median(rim)) / 1000, 2), rim_radius_km_min=round(float(rim.min()) / 1000, 2),
                          rim_radius_km_max=round(float(rim.max()) / 1000, 2), interior_area_km2=round(mask.sum() * RES * RES / 1e6, 2))
    raise ValueError(cfg["method"])


def vent_mask(L: Landscape, centres: pd.DataFrame) -> np.ndarray:
    X, Y = np.meshgrid(L.x30, L.y30)
    m = np.zeros(L.z.shape, bool)
    for c in centres.itertuples():
        m |= np.hypot(X - c.x, Y - c.y) <= P.CONTROL["vent_core_radius_m"]
    return m


# ------------------------------------------------------------------------------------------------ Part III: controls
def extract_controls(L: Landscape, dom30: np.ndarray, dnames: dict, excluded: np.ndarray, rim_band: np.ndarray) -> pd.DataFrame:
    C = P.CONTROL
    zf = nearest_fill(L.z, L.valid)
    tpi450 = zf - _focal_mean(zf, L.valid, C["tpi_radius_m"] / RES, annulus=True)
    tpi1500 = zf - _focal_mean(zf, L.valid, 1500 / RES, annulus=True)
    slope30 = horn(zf, RES)
    base = L.valid & (dom30 > 0) & ~excluded
    st = np.ones((3, 3), bool) if C["connectivity"] == 8 else None
    crest = base & (tpi450 >= C["crest_tpi_min_m"])
    lab, n = ndimage.label(crest, structure=st)
    size = np.bincount(lab.ravel())
    crest &= size[lab] >= C["crest_min_cells"]
    plat = base & (slope30 <= C["plateau_slope_max_deg"]) & (tpi1500 >= C["plateau_tpi1500_min_m"])
    lab, n = ndimage.label(plat, structure=st)
    size = np.bincount(lab.ravel())
    plat &= size[lab] >= C["plateau_min_cells"]
    rr, cc = np.nonzero(crest | plat)
    x, y = L.x30[cc], L.y30[rr]
    df = pd.DataFrame(dict(r=rr, c=cc, x=x, y=y, z=L.z[rr, cc], dom=dom30[rr, cc], code=L.code[rr, cc], plateau=plat[rr, cc],
                           rim=rim_band[rr, cc], bx=np.floor(x / C["thinning_block_m"]).astype(int), by=np.floor(y / C["thinning_block_m"]).astype(int)))
    out = []
    for (bx, by), g in df.groupby(["bx", "by"], sort=True):
        dom = g.dom.mode().iloc[0]
        g = g[g.dom == dom]
        share = g.code.value_counts(normalize=True).iloc[0]
        ctype = "PLATEAU" if g.plateau.any() else "CREST"
        cells = g[g.plateau] if ctype == "PLATEAU" else g
        if len(cells) < C["block_min_cells"]:
            continue
        zp = {p: float(np.percentile(cells.z, p)) for p in C["summit_percentiles"]}
        k = int(np.argmin(np.abs(cells.z.to_numpy() - zp[C["central_percentile"]])))
        pick = cells.iloc[k]
        if pick.rim:
            ctype = "RIM"
        out.append(dict(x=float(pick.x), y=float(pick.y), z90=zp[90], z95=zp[95], z99=zp[99], domain=dnames[int(dom)],
                        geological_unit=str(L.codes[int(pick.code)]), dominant_unit_share=round(float(share), 3), n_cells=int(len(cells)),
                        control_type=ctype, block=f"K{bx}_{by}"))
    ctrl = pd.DataFrame(out)
    if len(ctrl):
        ctrl["quality"] = np.where(ctrl.control_type.isin(["PLATEAU", "RIM"]), "Q1", "Q2")
        ctrl["selection_reason"] = np.where(ctrl.control_type == "PLATEAU", "plateau top (slope<=8 deg, >=25 cells, TPI1500>=0) on applicable unit",
                                   np.where(ctrl.control_type == "RIM", "crest/plateau within 2 km outside the masked caldera interior",
                                            "interfluve crest (TPI450>=10 m, >=3 connected cells) on applicable unit"))
        amb = ctrl.dominant_unit_share < C["dominant_unit_share_min"]
        ctrl.loc[amb, "quality"] = "Q3"
        ctrl.loc[amb, "selection_reason"] = "AMBIGUOUS_UNIT_SUPPORT (dominant unit share < 0.60)"
    return ctrl


def tau_coherence(ctrl: pd.DataFrame) -> dict:
    C = P.CONTROL
    out = {}
    for d, g in ctrl.groupby("domain"):
        rc = float(np.percentile(g.z95, 98) - np.percentile(g.z95, 2)) if len(g) > 1 else 0.0
        out[d] = max(C["tau_min_m"], C["tau_fraction"] * rc)
    return out


def coherence(ctrl: pd.DataFrame, tau: dict) -> pd.DataFrame:
    """Accordance: each Q1/Q2 control must agree within tau with a local plane through its same-domain neighbours."""
    C = P.CONTROL
    ctrl = ctrl.copy()
    ctrl["coherence_residual_m"] = np.nan
    for d, g in ctrl.groupby("domain"):
        cand = g[(g.quality != "Q3") & (g.control_type != "SUMMIT")]
        if not len(cand):
            continue
        xy = cand[["x", "y"]].to_numpy()
        tree = cKDTree(xy)
        zz = cand.z95.to_numpy()
        for i, idx in enumerate(cand.index):
            nb = [j for j in tree.query_ball_point(xy[i], C["coherence_radius_m"]) if j != i]
            if len(nb) < C["coherence_min_neighbours"]:
                ctrl.loc[idx, ["quality", "selection_reason"]] = ["Q3", "ISOLATED: accordance unverifiable (< 5 neighbours within 3 km)"]
                continue
            A = np.c_[np.ones(len(nb)), xy[nb, 0] - xy[i, 0], xy[nb, 1] - xy[i, 1]]
            beta = np.linalg.lstsq(A, zz[nb], rcond=None)[0]
            res = zz[i] - beta[0]
            ctrl.loc[idx, "coherence_residual_m"] = res
            if abs(res) > tau[d]:
                ctrl.loc[idx, ["quality", "selection_reason"]] = ["Q3", f"NON_ACCORDANT: {res:+.0f} m from neighbour plane (tau {tau[d]:.0f} m)"]
    return ctrl


def assign_centres(ctrl: pd.DataFrame, cen: pd.DataFrame) -> pd.DataFrame:
    ctrl = ctrl.copy()
    d = np.hypot(ctrl.x.to_numpy()[:, None] - cen.x.to_numpy()[None], ctrl.y.to_numpy()[:, None] - cen.y.to_numpy()[None])
    k = np.argmin(d, axis=1)
    ctrl["centre"] = cen.volcano_name.to_numpy()[k]
    ctrl["r_centre_m"] = d[np.arange(len(ctrl)), k]
    az = np.degrees(np.arctan2(ctrl.x.to_numpy() - cen.x.to_numpy()[k], ctrl.y.to_numpy() - cen.y.to_numpy()[k])) % 360  # azimuth from north
    ctrl["azimuth_deg"] = az
    ctrl["sector"] = (az // P.CONTROL["radial_sector_deg"]).astype(int)
    return ctrl


def outward_decrease(ctrl: pd.DataFrame, tau: dict, domain: str) -> pd.DataFrame:
    """Radial plausibility: along each centre sector a flank control may not exceed every closer control by more than tau."""
    ctrl = ctrl.copy()
    sel = (ctrl.domain == domain) & ctrl.quality.isin(["Q1", "Q2"])
    for _, g in ctrl[sel].groupby(["centre", "sector"]):
        g = g.sort_values("r_centre_m")
        run = -np.inf
        for idx, row in g.iterrows():
            if np.isfinite(run) and row.control_type != "SUMMIT" and row.z95 > run + tau[domain]:
                ctrl.loc[idx, ["quality", "selection_reason"]] = ["Q3", f"NOT_OUTWARD_DECREASING: {row.z95 - run:+.0f} m above closer controls in sector"]
            else:
                run = max(run, row.z95)
    return ctrl


def summit_controls(L: Landscape, cen: pd.DataFrame, dom30: np.ndarray, dnames: dict, domain: str) -> pd.DataFrame:
    X, Y = np.meshgrid(L.x30, L.y30)
    rows = []
    for c in cen.itertuples():
        m = (np.hypot(X - c.x, Y - c.y) <= P.CONTROL["summit_search_m"]) & L.valid
        lit = dict(x=c.x, y=c.y, z90=c.elevation_m, z95=c.elevation_m, z99=c.elevation_m, domain=domain, geological_unit="(catalogue)",
                   dominant_unit_share=np.nan, n_cells=0, control_type="CATALOGUE_SUMMIT", block="GVP", quality="Q4",
                   selection_reason="GVP catalogue summit elevation (literature value; never fitted)", centre_hint=c.volcano_name)
        rows.append(lit)
        if not m.any():
            continue
        zz = np.where(m, L.z, -np.inf)
        r, k = np.unravel_index(np.argmax(zz), zz.shape)
        rows.append(dict(x=float(L.x30[k]), y=float(L.y30[r]), z90=float(L.z[r, k]), z95=float(L.z[r, k]), z99=float(L.z[r, k]), domain=domain,
                         geological_unit=str(L.codes[L.code[r, k]]), dominant_unit_share=1.0, n_cells=1, control_type="SUMMIT", block="SUMMIT",
                         quality="Q1", selection_reason="DEM summit within 1 km of the registered GVP centre", centre_hint=c.volcano_name))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ Part V-VI: scenario models
def huber_fit(X: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float]:
    w = np.ones(len(z))
    beta = np.zeros(X.shape[1])
    s = 1.0
    for _ in range(P.FIT["irls_iterations"]):
        Xw = X * w[:, None]
        try:
            beta = np.linalg.solve(Xw.T @ X, Xw.T @ z)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)[0]
        r = z - X @ beta
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1.0)
        a = np.abs(r) / (P.FIT["huber_k"] * s)
        w = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12))
    return beta, s


def poly(x, y, order, xm, ym, ls):
    u, v = (np.asarray(x) - xm) / ls, (np.asarray(y) - ym) / ls
    cols = [np.ones_like(u), u, v] + ([u * u, u * v, v * v] if order == 2 else [])
    return np.stack(cols, axis=-1)


@dataclass
class Fitted:
    kind: str
    params: dict
    sigma: float
    offset: float
    coef: np.ndarray | None = None
    extra: dict = field(default_factory=dict)
    accepted: bool = True
    reject_reason: str = ""

    def predict(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.kind == "ENVELOPE":
            e = self.extra
            return poly(x, y, self.params["trend_order"], e["xm"], e["ym"], e["ls"]) @ self.coef + self.offset * self.sigma
        cones = radial_cones(x, y, self.extra)
        val = cones.max(axis=0) if self.params["coalescence"] == "UPPER_ENVELOPE" else cones.sum(axis=0)
        return self.extra["B"] + val + self.offset * self.sigma


def radial_cones(x, y, e) -> np.ndarray:
    out = []
    for k in e["centres"]:
        cx, cy, R, p, c0, c1, c2 = k
        dx, dy = x - cx, y - cy
        r = np.hypot(dx, dy)
        th = np.arctan2(dx, dy)
        f = np.clip(1 - r / R, 0, None) ** p
        out.append(f * (c0 + c1 * np.cos(th) + c2 * np.sin(th)))
    return np.stack(out) if out else np.zeros((1,) + np.shape(x))


def envelope_scenarios() -> list[dict]:
    S = P.ENVELOPE_SCENARIOS
    return [dict(trend_order=o, summit_percentile=p, tolerance_offset_sigma=d, replicate=r)
            for o, p, d, r in itertools.product(S["trend_order"], S["summit_percentile"], S["tolerance_offset_sigma"], range(S["control_subsample_replicates"]))]


def radial_scenarios(n_centres: int) -> list[dict]:
    S = P.RADIAL_SCENARIOS
    coal = S["coalescence"] if n_centres > 1 else ("SINGLE_CENTRE",)
    return [dict(p=p, radius_factor=rf, summit_percentile=sp, tolerance_offset_sigma=d, coalescence=c, replicate=r)
            for p, rf, sp, d, c, r in itertools.product(S["p"], S["radius_factor"], S["summit_percentile"], S["tolerance_offset_sigma"], coal,
                                                       range(S["control_subsample_replicates"]))]


def _subsample(n: int, frac: float, seed: int) -> np.ndarray:
    if n < 4:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max(3, int(round(frac * n))), replace=False))


def fit_envelope(ctrl: pd.DataFrame, sc: dict, scale: tuple, seed: int) -> Fitted:
    S = P.ENVELOPE_SCENARIOS
    idx = _subsample(len(ctrl), S["subsample_fraction"], seed + sc["replicate"])
    c = ctrl.iloc[idx]
    z = c[f"z{sc['summit_percentile']}"].to_numpy()
    X = poly(c.x.to_numpy(), c.y.to_numpy(), sc["trend_order"], *scale)
    if len(c) <= X.shape[1]:
        return Fitted("ENVELOPE", sc, 1.0, 0.0, accepted=False, reject_reason="TOO_FEW_CONTROLS")
    beta, s = huber_fit(X, z)
    return Fitted("ENVELOPE", sc, s, sc["tolerance_offset_sigma"], beta, dict(xm=scale[0], ym=scale[1], ls=scale[2]))


def fit_radial(ctrl: pd.DataFrame, cen: pd.DataFrame, sc: dict, seed: int) -> Fitted:
    S = P.RADIAL_SCENARIOS
    idx = _subsample(len(ctrl), S["subsample_fraction"], seed + sc["replicate"])
    c = ctrl.iloc[idx]
    z = c[f"z{sc['summit_percentile']}"].to_numpy()
    x, y = c.x.to_numpy(), c.y.to_numpy()
    d = np.hypot(x[:, None] - cen.x.to_numpy()[None], y[:, None] - cen.y.to_numpy()[None])
    near = np.argmin(d, axis=1)
    keep = [k for k in range(len(cen)) if (near == k).sum() >= 3]
    if not keep:
        return Fitted("RADIAL", sc, 1.0, 0.0, accepted=False, reject_reason="NO_CENTRE_WITH_3_CONTROLS")
    R = {k: sc["radius_factor"] * float(np.percentile(d[near == k, k], 95)) for k in keep}
    asym = {k: (near == k).sum() >= 6 for k in keep}

    def design(assign):
        cols = [np.ones(len(z))]
        for k in keep:
            dx, dy = x - cen.x.iloc[k], y - cen.y.iloc[k]
            r, th = np.hypot(dx, dy), np.arctan2(dx, dy)
            f = np.clip(1 - r / R[k], 0, None) ** sc["p"]
            if assign is not None:
                f = f * (assign == k)
            cols.append(f)
            if asym[k]:
                cols += [f * np.cos(th), f * np.sin(th)]
        return np.stack(cols, axis=1)

    def unpack(beta):
        out, i = [], 1
        for k in keep:
            c0 = beta[i]; i += 1
            c1 = c2 = 0.0
            if asym[k]:
                c1, c2 = beta[i], beta[i + 1]; i += 2
            out.append((cen.x.iloc[k], cen.y.iloc[k], R[k], sc["p"], c0, c1, c2))
        return out

    if sc["coalescence"] in ("SINGLE_CENTRE", "ADDITIVE_SUPERPOSITION"):
        beta, s = huber_fit(design(None), z)
    else:  # UPPER_ENVELOPE: iterate assignment to the highest fitted cone
        assign = near.copy()
        for _ in range(P.FIT["upper_envelope_iterations"]):
            beta, s = huber_fit(design(assign), z)
            cones = radial_cones(x, y, dict(centres=unpack(beta)))
            new = np.array(keep)[np.argmax(cones, axis=0)]
            if (new == assign).all():
                break
            assign = new
    cs = unpack(beta)
    f = Fitted("RADIAL", sc, s, sc["tolerance_offset_sigma"], beta, dict(B=float(beta[0]), centres=cs, kept=[cen.volcano_name.iloc[k] for k in keep]))
    for (_, _, _, _, c0, c1, c2) in cs:
        if c0 <= 0:
            f.accepted, f.reject_reason = False, "NON_POSITIVE_EDIFICE_HEIGHT"
        elif np.hypot(c1, c2) / c0 > 0.5:
            f.accepted, f.reject_reason = False, "ASYMMETRY_GT_0.5"
    return f


# ------------------------------------------------------------------------------------------------ Part VIII: validation
def holdout_groups(ctrl: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "RADIAL":
        return (ctrl.centre + "_S" + ctrl.sector.astype(str)).to_numpy()
    b = P.CONTROL_GEOMETRY["envelope"]["block_m"]
    return (np.floor(ctrl.x / b).astype(int).astype(str) + "_" + np.floor(ctrl.y / b).astype(int).astype(str)).to_numpy()


def ensemble_fit(ctrl, kind, scen, cen, scale, seed):
    if kind == "ENVELOPE":
        return [fit_envelope(ctrl, s, scale, seed) for s in scen]
    return [fit_radial(ctrl, cen, s, seed) for s in scen]


def validate(ctrl: pd.DataFrame, kind: str, scen: list, cen, scale, seed: int) -> pd.DataFrame:
    """Leave-one-group-out over spatial groups; returns one row per held-out control with the scenario band."""
    groups = holdout_groups(ctrl, kind)
    rows = []
    for g in np.unique(groups):
        test = groups == g
        train = ctrl[~test].reset_index(drop=True)
        if len(train) < 6:
            continue
        fits = [f for f in ensemble_fit(train, kind, scen, cen, scale, seed) if f.accepted]
        if not fits:
            continue
        te = ctrl[test]
        pred = np.stack([f.predict(te.x.to_numpy(), te.y.to_numpy()) for f in fits], axis=1)
        for i, (idx, row) in enumerate(te.iterrows()):
            p = pred[i]
            rows.append(dict(control_index=idx, group=g, quality=row.quality, z=row.z95, pred_median=float(np.median(p)),
                             p05=float(np.percentile(p, 5)), p95=float(np.percentile(p, 95)), n_scenarios=len(fits)))
    v = pd.DataFrame(rows)
    if len(v):
        v["residual"] = v.z - v.pred_median
        v["inside_band"] = (v.z >= v.p05) & (v.z <= v.p95)
    return v


# ------------------------------------------------------------------------------------------------ Part IV: control geometry
def control_geometry(ctrl: pd.DataFrame, kind: str, excluded_hits: int) -> dict:
    fit = ctrl[ctrl.quality.isin(["Q1", "Q2"])]
    out = dict(kind=kind, n_fit=len(fit), controls_in_excluded_zones=int(excluded_hits))
    reasons = []
    if kind == "RADIAL":
        G = P.CONTROL_GEOMETRY["radial"]
        sc = fit.groupby(["centre", "sector"]).size()
        occ = int((sc >= G["min_per_occupied_sector"]).sum())
        share = float(sc.max() / sc.sum()) if len(sc) else 1.0
        out.update(occupied_sectors=occ, max_sector_share=round(share, 3), sector_counts=";".join(f"{a}:{b}={n}" for (a, b), n in sc.items()))
        if occ < G["min_occupied_sectors"]:
            reasons.append(f"only {occ} occupied sectors")
        if share > G["max_sector_share"]:
            reasons.append(f"one sector holds {share:.0%} of controls")
    else:
        G = P.CONTROL_GEOMETRY["envelope"]
        b = holdout_groups(fit, "ENVELOPE")
        vc = pd.Series(b).value_counts()
        share = float(vc.iloc[0] / vc.sum()) if len(vc) else 1.0
        out.update(occupied_blocks_2km=int(len(vc)), max_block_share=round(share, 3))
        if len(vc) < G["min_blocks"]:
            reasons.append(f"only {len(vc)} occupied 2 km blocks")
        if share > G["max_block_share"]:
            reasons.append(f"one 2 km block holds {share:.0%} of controls")
    if excluded_hits:
        reasons.append(f"{excluded_hits} controls inside excluded zones")
    out["verdict"] = "FAIL" if reasons else "PASS"
    out["reasons"] = "; ".join(reasons) or "none"
    return out


def centre_geometry_gate(units_xy: np.ndarray, ctrl: pd.DataFrame, cen: pd.DataFrame) -> dict:
    G = P.CENTRE_GEOMETRY_GATE
    res = {}
    for c in cen.itertuples():
        d_units = np.hypot(units_xy[:, 0] - c.x, units_xy[:, 1] - c.y) / 1000
        frac = float((d_units <= G["flank_radius_km"]).mean()) if len(d_units) else 0.0
        fit = ctrl[ctrl.quality.isin(["Q1", "Q2"])]
        dc = np.hypot(fit.x - c.x, fit.y - c.y)
        rho = float(stats.spearmanr(fit.z95, dc).correlation) if len(fit) > 3 else np.nan
        az = (np.degrees(np.arctan2(fit.x - c.x, fit.y - c.y)) % 360 // 45).astype(int)
        nsec = int((az.value_counts() >= 3).sum())
        ok = frac >= G["min_unit_fraction_within"] and rho <= G["max_spearman_elev_vs_distance"] and nsec >= G["min_sectors_with_3_controls"]
        res[c.volcano_name] = dict(unit_fraction_within_30km=round(frac, 4), spearman_elev_vs_distance=round(rho, 4), sectors_with_3_controls=nsec, passes=bool(ok))
    res["PASS"] = any(v["passes"] for v in res.values() if isinstance(v, dict))
    return res
