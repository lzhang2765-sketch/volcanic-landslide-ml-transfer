"""Build Stage 4C-B: production H reconstruction + Y-blind validation for the six development landscapes.

No model is trained, no MEGH/Delta_H is computed, no H predictive performance is observed, no Stage 4B residual is read and no
lockbox is accessed.  Landslide Y is never loaded: every input passes stage4c_b_recon.guard.  Operational choices are frozen in
stage4c_b_prereg.py (hashed into STAGE4C_B_PREREGISTRATION.json before the reconstruction runs).
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from . import stage4a_base as B
from . import stage4c_a_spec as H
from . import stage4c_b_prereg as P
from . import stage4c_b_recon as R
from .build_stage3a import scan_ml_h_artifacts
from .build_stage4a import identity_stats, read_columns, write_columns
from .raw_store import sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify

OUT = ROOT / "outputs/stage4c_b"
QA = OUT / "STAGE4C_B_RECONSTRUCTION_QA"
HDS = OUT / "STAGE4C_B_H_DATASET"
RAW = ROOT / "data/raw/stage4c_b"
REGISTRY = ROOT / "metadata/stage4c_b_raw_registry.json"
PRIOR = {"stage3f": "outputs/stage3f/STAGE3F_CHECKPOINT.json", "stage3g": "outputs/stage3g/STAGE3G_CHECKPOINT.json",
         "stage3h": "outputs/stage3h/STAGE3H_CHECKPOINT.json", "stage3i": "outputs/stage3i/STAGE3I_CHECKPOINT.json",
         "stage3j": "outputs/stage3j/STAGE3J_CHECKPOINT.json", "stage4a": "outputs/stage4a/STAGE4A_CHECKPOINT.json",
         "stage4b": "outputs/stage4b/STAGE4B_CHECKPOINT.json", "stage4c_a": "outputs/stage4c_a/STAGE4C_A_CHECKPOINT.json"}
PRIOR_SHA = {"stage3f": "43ecb1f75fadc1bf30710b52a3124c7f19880526088a32a7cece6f983f4ff05a", "stage3g": "a4f8035123658c063f1572bdfd3276906053ba313735ae3371c442e45cbe4664",
             "stage3h": "f1ea33e848559bcc97f6c8fb7bc7a78c8fca9d004fe924d8c0136d1f488f2d1f", "stage3i": "90072d3562da197c46a0dee0bdc0e1fe752a242c520ee9f44a09bd1b09bc9ea1",
             "stage3j": "99a8c6f4cd90765d4d5bd7be790af22502b22194cd47c63443161e3ede4c82fd", "stage4a": "fd25accb50578bd4f4f28042e42504a89ef936c904d5ed262a0f65fc7fa2452c",
             "stage4b": "2685d1b6d565de7945e138d9b014d9aa4b53657511552299c19ebc1f4df946b8", "stage4c_a": "09463f7d76cc865b08ad602a5518532c8a4cc2a5131952b73618082bed09bfa7"}
CORE = [c["name"] for c in H.CORE_H]
HCOL = {n: n.lower() for n in CORE}  # Stage 4A H contract: reserved h_ prefix


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def _tsv(p: Path, df) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(p, sep="\t", index=False, lineterminator="\n", float_format="%.6g")


# ------------------------------------------------------------------------------------------------ Part I: public centres
def dem_bbox_ll(lid: str) -> tuple:
    from pyproj import Transformer
    s = P.SOURCES[lid]
    with rasterio.open(R.guard(R.PROC / s["dir"] / lid / s["dem"])) as r:
        b = r.bounds
        tr = Transformer.from_crs(r.crs, 4326, always_xy=True)
    xs, ys = tr.transform([b.left, b.right, b.left, b.right], [b.bottom, b.bottom, b.top, b.top])
    return round(min(xs) - 0.1, 4), round(min(ys) - 0.1, 4), round(max(xs) + 0.1, 4), round(max(ys) + 0.1, 4)


def acquire_centres() -> tuple[pd.DataFrame, list]:
    """Bounding-box-limited WFS queries per development landscape: no catalogue row outside a development box is ever downloaded."""
    reg = json.loads(REGISTRY.read_text())["entries"] if REGISTRY.exists() else []
    rows = []
    for lid in P.LANDSCAPES:
        if lid == "NOTO_PENINSULA_2024":
            continue  # AGE_PROXY_ONLY: no centre required
        bbox = dem_bbox_ll(lid)
        for layer_key, layer in P.GVP["layers"].items():
            aid = f"ACQ_4CB_GVP_{layer_key}_{lid}"
            dest = R.guard(RAW / lid / f"gvp_{layer_key.lower()}_bbox.json")
            url = P.GVP["url"].format(layer=layer) + "&bbox=" + ",".join(map(str, bbox))
            prior = next((e for e in reg if e["acquisition_id"] == aid), None)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "--max-time", "120", "-A", "Mozilla/5.0", "-o", str(tmp), url], check=True)
                json.loads(tmp.read_text())  # must be valid GeoJSON
                tmp.rename(dest)
                dest.chmod(0o444)
            digest = sha256(dest)
            if prior and prior["sha256"] != digest:
                raise RuntimeError(f"STOP: registered GVP response {aid} changed")
            if not prior:
                prior = dict(acquisition_id=aid, landscape_id=lid, source_id="SMITHSONIAN_GVP_VOTW", dataset_name=P.GVP["dataset"], layer=layer,
                             download_url=url, publisher=P.GVP["publisher"], access_date=P.AS_OF, licence=P.GVP["licence"], bbox_lonlat=list(bbox),
                             raw_path=str(dest.relative_to(ROOT)), sha256=digest, actual_bytes=dest.stat().st_size,
                             acquisition_timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
                reg.append(prior)
            fc = json.loads(dest.read_text())
            for f in fc["features"]:
                p = f["properties"]
                rows.append(dict(landscape_id=lid, volcano_name=p["Volcano_Name"], gvp_volcano_number=p["Volcano_Number"], latitude=float(p["Latitude"]),
                                 longitude=float(p["Longitude"]), elevation_m=p.get("Elevation"), primary_volcano_type=p.get("Primary_Volcano_Type"),
                                 catalogue_layer=layer_key, last_eruption_year=p.get("Last_Eruption_Year"), source=layer, source_url=url,
                                 source_authority=P.GVP["publisher"], access_date=prior["access_date"], raw_sha256=digest,
                                 coordinate_provenance=f"GVP VOTW {layer_key.title()} list, Volcano_Number {p['Volcano_Number']} (WFS GeoJSON, EPSG:4326)",
                                 centre_semantics=P.GVP["coordinate_semantics"], licence_reuse=P.GVP["licence"]))
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps({"schema": "SFV2_STAGE4C_B_RAW_REGISTRY_V1", "entries": sorted(reg, key=lambda e: e["acquisition_id"])}, indent=2, sort_keys=True) + "\n")
    df = pd.DataFrame(rows).drop_duplicates(["landscape_id", "gvp_volcano_number"]).reset_index(drop=True)
    return df, reg


# ------------------------------------------------------------------------------------------------ per-landscape reconstruction
def _excluded90(ex30: np.ndarray) -> np.ndarray:
    return R.block90(ex30.astype(float), "mean") >= 0.5


def reconstruct(lid: str, cen_all: pd.DataFrame) -> dict:
    L = R.load_landscape(lid)
    dnames, look = R.domain_lookup(L)
    dom30 = look[L.code]
    cen = R.project_centres(L, cen_all)
    cin = cen[cen.inside_window].reset_index(drop=True)
    cal30, cal_info = R.caldera_mask(L, cin)
    vent30 = R.vent_mask(L, cin)
    excluded30 = cal30 | vent30
    from scipy import ndimage
    dist_out = ndimage.distance_transform_edt(~cal30) * R.RES if cal30.any() else np.full(L.z.shape, np.inf)
    rim30 = (~cal30) & (dist_out <= P.CONTROL["rim_band_m"]) if cal30.any() else np.zeros(L.z.shape, bool)
    ctrl = R.extract_controls(L, dom30, dnames, excluded30, rim30)
    tau = R.tau_coherence(ctrl)
    ctrl = R.coherence(ctrl, tau)

    # ---- family decision (Y-free)
    names = P.RADIAL_CENTRES.get(lid, {}).get("names", [])
    rad_cen = cin[cin.volcano_name.isin(names)].reset_index(drop=True)
    gate = None
    comps = []  # (domain, kind)
    if lid == "COTABATO_DAVAO_2019":
        codes = P.APPLICABLE[lid]["RADIAL_OR_ENVELOPE"]["codes"]
        uxy = L.units[L.units.source_unit_code.isin(codes)][["centroid_x", "centroid_y"]].to_numpy()
        gate = R.centre_geometry_gate(uxy, ctrl, rad_cen)
        comps = [("RADIAL_OR_ENVELOPE", "RADIAL" if gate["PASS"] else "ENVELOPE")]
    elif lid == "DOMINICA_ERIKA_2015":
        comps = [("RADIAL", "RADIAL"), ("ENVELOPE", "ENVELOPE")]
    elif lid == "LOMBOK_RINJANI_2018":
        comps = [("RADIAL", "RADIAL")]
    else:
        comps = [("ENVELOPE", "ENVELOPE")]
    for dom, kind in comps:
        if kind == "RADIAL":
            sm = R.summit_controls(L, rad_cen, dom30, dnames, dom)
            ctrl = pd.concat([ctrl, sm.drop(columns="centre_hint")], ignore_index=True)
    radial_doms = [d for d, k in comps if k == "RADIAL"]
    if radial_doms:
        ctrl = R.assign_centres(ctrl, rad_cen)
        for d in radial_doms:
            ctrl = R.outward_decrease(ctrl, tau | {d: tau.get(d, P.CONTROL["tau_min_m"])}, d)
    else:
        ctrl["centre"], ctrl["r_centre_m"], ctrl["azimuth_deg"], ctrl["sector"] = "", np.nan, np.nan, -1
    # controls inside excluded zones (summits are the registered centre summits: exempt by definition)
    col = np.clip(((ctrl.x - L.transform.c) / R.RES).astype(int), 0, L.z.shape[1] - 1)
    row = np.clip(((L.transform.f - ctrl.y) / R.RES).astype(int), 0, L.z.shape[0] - 1)
    ctrl["in_excluded_zone"] = excluded30[row, col] & ~ctrl.control_type.isin(["SUMMIT", "CATALOGUE_SUMMIT"])
    ctrl["included_in_fit"] = ctrl.quality.isin(["Q1", "Q2"]) & ~ctrl.in_excluded_zone
    ctrl["landscape_id"] = lid
    ctrl["family_relevance"] = [f"{d}:{dict(comps).get(d, 'NONE')}" for d in ctrl.domain]

    # ---- 90 m grids
    h90, w90 = L.shape90
    cnt = R.block90(L.valid.astype(float), "sum")
    z90 = np.where(cnt >= 1, R.block90(L.z, "mean"), np.nan)
    v90 = cnt >= 5
    dom90 = R.domain90(L, dom30, len(dnames))
    ex90 = _excluded90(excluded30)
    X90, Y90 = np.meshgrid(L.x90, L.y90)
    zf90 = R.nearest_fill(z90, np.isfinite(z90))
    slope_mod = np.where(np.isfinite(z90), R.horn(zf90, R.URES), np.nan)
    relief_mod = np.where(np.isfinite(z90), R.relief(zf90, 450 / R.URES), np.nan)
    u = L.units
    uin = (u.ur < h90) & (u.uc < w90)
    assert uin.all()
    ur, uc = u.ur.to_numpy(), u.uc.to_numpy()

    from scipy.spatial import cKDTree
    seed = P.SEED
    comp_res = {}
    support90 = np.zeros((h90, w90), bool)
    for dom, kind in comps:
        did = [k for k, v in dnames.items() if v == dom][0]
        fitc = ctrl[ctrl.included_in_fit & (ctrl.domain == dom)].reset_index(drop=True)
        geom = R.control_geometry(fitc.assign(quality="Q1") if len(fitc) else fitc, kind, int((ctrl.in_excluded_zone & ctrl.quality.isin(["Q1", "Q2"]) & (ctrl.domain == dom)).sum()))
        scen = R.envelope_scenarios() if kind == "ENVELOPE" else R.radial_scenarios(len(rad_cen))
        if len(fitc):
            xm, ym = float(fitc.x.mean()), float(fitc.y.mean())
            ls = float(max(np.hypot(fitc.x - xm, fitc.y - ym).max(), 1000.0))
        else:
            xm = ym = 0.0
            ls = 1.0
        scale = (xm, ym, ls)
        fits = R.ensemble_fit(fitc, kind, scen, rad_cen, scale, seed) if len(fitc) >= 6 else []
        acc = [f for f in fits if f.accepted]
        # support
        sup = (dom90 == did) & v90 & ~ex90
        if len(fitc):
            d_ctrl, _ = cKDTree(fitc[["x", "y"]].to_numpy()).query(np.c_[X90.ravel(), Y90.ravel()])
            d_ctrl = d_ctrl.reshape(h90, w90)
            sup &= d_ctrl <= P.SUPPORT["max_control_distance_m"]
        else:
            d_ctrl = np.full((h90, w90), np.inf)
            sup &= False
        if kind == "RADIAL" and len(fitc):
            dd = np.stack([np.hypot(X90 - c.x, Y90 - c.y) for c in rad_cen.itertuples()])
            near = np.argmin(dd, axis=0)
            rlim = np.zeros(len(rad_cen))
            for k, c in enumerate(rad_cen.itertuples()):
                rk = fitc.r_centre_m[fitc.centre == c.volcano_name]
                rlim[k] = float(np.percentile(rk, 95)) if len(rk) >= 3 else 0.0
            sup &= dd.min(axis=0) <= rlim[near]
        sup &= ~support90  # components are disjoint by domain; guard anyway
        comp_res[dom] = dict(kind=kind, fitc=fitc, scen=scen, fits=fits, acc=acc, sup=sup, d_ctrl=d_ctrl, geom=geom, scale=scale)
        support90 |= sup

    # ---- ensemble evaluation on support (grid) and on supported units
    zmed = np.full((h90, w90), np.nan)
    ziqr = np.full((h90, w90), np.nan)
    unit_sup = support90[ur, uc]
    n_u = len(u)
    U = {k: np.full(n_u, np.nan) for k in ("zref_med", "zref_p05", "zref_p25", "zref_p75", "zref_p95", "slope_ref_med", "relief_ref_med")}
    comp_of_unit = np.full(n_u, "", dtype=object)
    sigmas = []
    for dom, cr in comp_res.items():
        if not cr["acc"] or not cr["sup"].any():
            continue
        rows_, cols_ = np.nonzero(cr["sup"])
        r0, r1 = max(rows_.min() - 8, 0), min(rows_.max() + 9, h90)
        c0, c1 = max(cols_.min() - 8, 0), min(cols_.max() + 9, w90)
        sub = cr["sup"][r0:r1, c0:c1]
        um = unit_sup & cr["sup"][ur, uc]
        uix = np.nonzero(um)[0]
        ul = (ur[uix] - r0, uc[uix] - c0)
        Zs = np.empty((int(sub.sum()), len(cr["acc"])), np.float32)
        Zu = np.empty((len(uix), len(cr["acc"])), np.float32)
        Su = np.empty_like(Zu)
        Ru = np.empty_like(Zu)
        Xs, Ys = X90[r0:r1, c0:c1], Y90[r0:r1, c0:c1]
        for j, f in enumerate(cr["acc"]):
            Zg = f.predict(Xs, Ys)
            Zs[:, j] = Zg[sub]
            Zu[:, j] = Zg[ul]
            Su[:, j] = R.horn(Zg, R.URES)[ul]
            Ru[:, j] = R.relief(Zg, 450 / R.URES)[ul]
            sigmas.append(f.sigma)
        q = np.percentile(Zs, [25, 50, 75], axis=1)
        zmed[r0:r1, c0:c1][sub] = q[1]
        ziqr[r0:r1, c0:c1][sub] = q[2] - q[0]
        qu = np.percentile(Zu, [5, 25, 50, 75, 95], axis=1)
        U["zref_p05"][uix], U["zref_p25"][uix], U["zref_med"][uix], U["zref_p75"][uix], U["zref_p95"][uix] = qu
        U["slope_ref_med"][uix] = np.median(Su, axis=1)
        U["relief_ref_med"][uix] = np.median(Ru, axis=1)
        comp_of_unit[uix] = dom
        cr["n_support_cells"] = int(sub.sum())
    fin = support90 & np.isfinite(zmed)
    support90 = fin
    if fin.any():
        p02, p98 = np.percentile(zmed[fin], [2, 98])
        R_L = float(p98 - p02)
        area = float(fin.sum() * R.URES * R.URES)
        Lam = float(np.sqrt(area / np.pi))
    else:
        p02 = R_L = Lam = np.nan
        area = 0.0
    tau_L = float(np.median(sigmas)) if sigmas else np.nan

    # ---- unit-level H (computed for QA; released only for verified landscapes)
    zmod_u = z90[ur, uc]
    fitall = ctrl[ctrl.included_in_fit]
    d_u = cKDTree(fitall[["x", "y"]].to_numpy()).query(u[["centroid_x", "centroid_y"]].to_numpy())[0] if len(fitall) else np.full(n_u, np.inf)
    ok = unit_sup & np.isfinite(U["zref_med"])
    hv = {}
    with np.errstate(invalid="ignore"):
        hv["H_REMOVED_FRACTION"] = np.where(ok, (U["zref_med"] - zmod_u) / R_L, np.nan)
        hv["H_SLOPE_CHANGE_DEG"] = np.where(ok, slope_mod[ur, uc] - U["slope_ref_med"], np.nan)
        hv["H_RELIEF_CHANGE_FRACTION"] = np.where(ok, (relief_mod[ur, uc] - U["relief_ref_med"]) / R_L, np.nan)
        hv["H_REFERENCE_RELATIVE_HEIGHT"] = np.where(ok, (U["zref_med"] - p02) / R_L, np.nan)
        hv["H_SCENARIO_SPREAD_FRACTION"] = np.where(ok, (U["zref_p75"] - U["zref_p25"]) / R_L, np.nan)
        hv["H_CONTROL_DISTANCE_FRACTION"] = np.where(ok, d_u / Lam, np.nan)
        qa_rem_p05 = np.where(ok, (U["zref_p05"] - zmod_u) / R_L, np.nan)
        qa_rem_p95 = np.where(ok, (U["zref_p95"] - zmod_u) / R_L, np.nan)
        qa_rrh_p05 = np.where(ok, (U["zref_p05"] - p02) / R_L, np.nan)
        qa_rrh_p95 = np.where(ok, (U["zref_p95"] - p02) / R_L, np.nan)
    avail = ok & np.all(np.stack([np.isfinite(v) for v in hv.values()]), axis=0)

    # ---- validation (held-out controls; no Y)
    vrows = []
    for dom, cr in comp_res.items():
        if len(cr["fitc"]) >= 6 and cr["acc"]:
            v = R.validate(cr["fitc"], cr["kind"], cr["scen"], rad_cen, cr["scale"], seed)
            v["component"] = dom
            v["kind"] = cr["kind"]
            vrows.append(v)
    val = pd.concat(vrows, ignore_index=True) if vrows else pd.DataFrame(columns=["residual", "inside_band", "component", "quality"])
    n_fit = int(ctrl.included_in_fit.sum())
    sup_frac = float(avail.mean())
    young = np.isin(u.source_unit_code.to_numpy(), P.YOUNG_DEPOSIT_CODES[lid])
    neg_base = avail & ~young
    neg_frac = float(((U["zref_med"] - zmod_u)[neg_base] < -tau_L).mean()) if neg_base.any() else np.nan
    slope_ref_units = U["slope_ref_med"][avail]
    max_slope = float(np.nanmax(slope_ref_units)) if slope_ref_units.size else np.nan
    p99_slope = float(np.nanpercentile(slope_ref_units, 99)) if slope_ref_units.size else np.nan
    if len(val):
        med_abs = float(np.median(np.abs(val.residual)) / R_L)
        bias = float(val.residual.mean() / R_L)
        band = float(val.inside_band.mean())
    else:
        med_abs = bias = band = np.nan
    G = P.GATES
    gates = dict(median_abs_residual=(med_abs, med_abs <= G["median_abs_residual_fraction_max"]),
                 abs_bias=(abs(bias), abs(bias) <= G["abs_bias_fraction_max"]),
                 holdout_band=(band, band >= G["holdout_band_fraction_min"]),
                 admissible_controls=(n_fit, n_fit >= G["min_admissible_controls"]),
                 support_fraction=(sup_frac, sup_frac >= G["min_support_fraction_of_units"]),
                 negative_removal=(neg_frac, (neg_frac <= G["negative_removal_fraction_max_outside_mapped_young_deposits"]) if np.isfinite(neg_frac) else False),
                 max_reference_slope=(max_slope, (max_slope <= G["max_reconstructed_slope_deg_constructional"]) if np.isfinite(max_slope) else False))
    code_of = dict(median_abs_residual="HIGH_RESIDUAL", abs_bias="HIGH_BIAS", holdout_band="LOW_SCENARIO_COVERAGE", admissible_controls="INSUFFICIENT_CONTROLS",
                   support_fraction="LOW_SUPPORT", negative_removal="EXCESS_NEGATIVE_REMOVAL", max_reference_slope="IMPLAUSIBLE_SLOPE")
    failures = [code_of[k] for k, (_, ok_) in gates.items() if not ok_]
    if any(cr["geom"]["verdict"] == "FAIL" for cr in comp_res.values()):
        failures.insert(0, "POOR_CONTROL_GEOMETRY")
    if any(not cr["acc"] for cr in comp_res.values()):
        failures.insert(0, "FAMILY_MISMATCH")
    all_pass = not failures
    status = ("VERIFIED_FULL" if sup_frac >= 0.90 else "VERIFIED_PARTIAL") if all_pass else "FAILED_H"

    # geometric checks (reported; 4C-A lists them alongside the gates)
    zmax_ctrl = float(ctrl[ctrl.included_in_fit].z99.max()) if n_fit else np.nan
    geo_checks = dict(
        unsupported_summit=bool(np.nanmax(zmed) <= zmax_ctrl + 2 * tau_L) if fin.any() else None,
        caldera_interiors_masked=bool(not (support90 & _excluded90(cal30)).any()),
        reference_below_preserved_controls_beyond_tolerance_fraction=float((val.residual > 2 * tau_L).mean()) if len(val) else None,
        removed_volume_km3_p05_p50_p95=None,
    )
    if avail.any():
        cell = R.URES ** 2 / 1e9
        geo_checks["removed_volume_km3_p05_p50_p95"] = [round(float(np.nansum((U[k] - zmod_u)[avail]) * cell), 3) for k in ("zref_p05", "zref_med", "zref_p95")]

    comp_sem = {}
    for dom, cr in comp_res.items():
        comp_sem[dom] = "ESTIMATED_REMOVAL" if cr["kind"] == "RADIAL" else "MINIMUM_REMOVAL"
    sem_u = np.array([comp_sem.get(c, "") for c in comp_of_unit], dtype=object)
    fam_u = np.array([(comp_res[c]["kind"] if c else "") for c in comp_of_unit], dtype=object)
    conf = np.where(~avail, "", np.where((hv["H_CONTROL_DISTANCE_FRACTION"] <= 0.25) & (hv["H_SCENARIO_SPREAD_FRACTION"] <= 0.05), "HIGH",
                                         np.where((hv["H_CONTROL_DISTANCE_FRACTION"] > 0.5) | (hv["H_SCENARIO_SPREAD_FRACTION"] > 0.15), "LOW", "MEDIUM")))
    grids = dict(z_modern=z90.astype(np.float32), zref_median=zmed.astype(np.float32), zref_iqr=ziqr.astype(np.float32), support=support90,
                 removed_fraction=((zmed - z90) / R_L).astype(np.float32), spread_fraction=(ziqr / R_L).astype(np.float32),
                 control_distance_fraction=(np.fmin.reduce([cr["d_ctrl"] for cr in comp_res.values()]) / Lam).astype(np.float32) if comp_res else None,
                 caldera_interior=_excluded90(cal30), vent_core=_excluded90(vent30), domain=dom90,
                 x0=np.float64(L.transform.c), y0=np.float64(L.transform.f), res=np.float64(R.URES))
    return dict(lid=lid, L=L, ctrl=ctrl, tau=tau, cen=cen, rad_cen=rad_cen, cal_info=cal_info, gate=gate, comps=comps, comp_res=comp_res,
                R_L=R_L, P02=float(p02), Lambda=Lam, support_area_km2=area / 1e6, tau_L=tau_L, val=val, gates=gates, failures=failures,
                status=status, hv=hv, avail=avail, unit_sup=unit_sup, sup_frac=sup_frac, neg_frac=neg_frac, max_slope=max_slope, p99_slope=p99_slope,
                n_fit=n_fit, geo_checks=geo_checks, sem_u=sem_u, fam_u=fam_u, conf=conf, grids=grids,
                qa=dict(qa_h_removed_fraction_p05=qa_rem_p05, qa_h_removed_fraction_p95=qa_rem_p95,
                        qa_h_reference_relative_height_p05=qa_rrh_p05, qa_h_reference_relative_height_p95=qa_rrh_p95))


# ------------------------------------------------------------------------------------------------ tables
def centre_table(cen_all: pd.DataFrame, results: dict) -> pd.DataFrame:
    rows = []
    for lid, r in results.items():
        c = r["cen"]
        radial = set(r["rad_cen"].volcano_name) if r["comps"] and any(k == "RADIAL" for _, k in r["comps"]) else set()
        for x in c.itertuples():
            role = ("RADIAL_RECONSTRUCTION_CENTRE" if x.volcano_name in radial else
                    "VENT_CORE_EXCLUSION_ONLY" if x.inside_window else "CONTEXT_OUTSIDE_RECONSTRUCTION_WINDOW")
            amb = []
            if x.catalogue_layer == "PLEISTOCENE":
                amb.append("Pleistocene list: no confirmed Holocene eruption; location is the GVP representative summit")
            if x.volcano_name == "Asosan":
                amb.append("GVP point marks the post-caldera central cones (Nakadake), not the caldera centre; caldera interior taken from GSJ 1:50,000 map")
            if x.volcano_name == "Rinjani":
                amb.append("single GVP entry for the Rinjani-Samalas complex; GVP point lies ~1.5 km from the DEM summit")
            if pd.isna(x.elevation_m):
                amb.append("no catalogue elevation")
            rows.append(dict(landscape_id=lid, volcano_name=x.volcano_name, gvp_volcano_number=x.gvp_volcano_number, latitude=x.latitude, longitude=x.longitude,
                             projected_x=round(x.x, 1), projected_y=round(x.y, 1), catalogue_elevation_m=x.elevation_m, primary_volcano_type=x.primary_volcano_type,
                             catalogue_layer=x.catalogue_layer, source=x.source, source_url=x.source_url, source_authority=x.source_authority,
                             access_date=x.access_date, raw_sha256=x.raw_sha256, coordinate_provenance=x.coordinate_provenance, centre_semantics=x.centre_semantics,
                             licence_reuse=x.licence_reuse, inside_reconstruction_window=bool(x.inside_window), role=role,
                             uncertainty_ambiguity="; ".join(amb) or "GVP catalogue precision (~0.001-0.01 deg)"))
        ci = r["cal_info"]
        if ci.get("method") in ("WBM_LAKE_RAY_RIM", "GSJ_ASO_POST_CALDERA_UNITS"):
            from pyproj import Transformer
            cx, cy = (ci.get("centre_x"), ci.get("centre_y")) if "centre_x" in ci else (ci.get("centroid_x"), ci.get("centroid_y"))
            lon, lat = Transformer.from_crs(r["L"].crs, 4326, always_xy=True).transform(cx, cy)
            name = "Segara Anak / Samalas caldera (derived)" if lid.startswith("LOMBOK") else "Aso caldera interior (derived)"
            src = ("Copernicus GLO-30 Water Body Mask (lake class) + DEM ray rim" if lid.startswith("LOMBOK")
                   else "GSJ Aso Volcano 1:50,000 geological map: post-caldera volcanoes and deposits")
            rows.append(dict(landscape_id=lid, volcano_name=name, gvp_volcano_number="", latitude=round(lat, 5), longitude=round(lon, 5), projected_x=round(cx, 1),
                             projected_y=round(cy, 1), catalogue_elevation_m=np.nan, primary_volcano_type="Caldera", catalogue_layer="DERIVED", source=src,
                             source_url="", source_authority="derived in Stage 4C-B from frozen V2 inputs", access_date=P.AS_OF, raw_sha256="",
                             coordinate_provenance=json.dumps(ci, default=str), centre_semantics="caldera-mask reference (collapse is masked, never scored as erosion)",
                             licence_reuse="derived from inputs already registered in V2", inside_reconstruction_window=True, role="CALDERA_MASK_REFERENCE",
                             uncertainty_ambiguity="mask geometry, not a vent; see coordinate_provenance"))
    for lid, nm in (("LOMBOK_RINJANI_2018", "Pusuk"), ("LOMBOK_RINJANI_2018", "Nangi")):
        rows.append(dict(landscape_id=lid, volcano_name=nm, gvp_volcano_number="", latitude=np.nan, longitude=np.nan, projected_x=np.nan, projected_y=np.nan,
                         catalogue_elevation_m=np.nan, primary_volcano_type="", catalogue_layer="NOT_ACQUIRED", source="none", source_url="", source_authority="",
                         access_date=P.AS_OF, raw_sha256="", coordinate_provenance="", centre_semantics="", licence_reuse="",
                         inside_reconstruction_window=False, role="NOT_ACQUIRED",
                         uncertainty_ambiguity="named in the Sheet-1807 legend (Qhv p/n) but not a separate GVP catalogue entry; coordinates NOT invented"))
    rows.append(dict(landscape_id="NOTO_PENINSULA_2024", volcano_name="(none required)", gvp_volcano_number="", latitude=np.nan, longitude=np.nan, projected_x=np.nan,
                     projected_y=np.nan, catalogue_elevation_m=np.nan, primary_volcano_type="", catalogue_layer="NOT_REQUIRED", source="", source_url="", source_authority="",
                     access_date=P.AS_OF, raw_sha256="", coordinate_provenance="", centre_semantics="", licence_reuse="", inside_reconstruction_window=False,
                     role="AGE_PROXY_ONLY_NO_CENTRE", uncertainty_ambiguity="no constructional surface; no centre used"))
    return pd.DataFrame(rows)


def control_table(results: dict) -> pd.DataFrame:
    out = []
    for lid, r in results.items():
        c = r["ctrl"].reset_index(drop=True)
        df = pd.DataFrame(dict(
            control_id=[f"{lid[:4]}_C{i:05d}" for i in range(len(c))], landscape_id=lid, x=c.x.round(1), y=c.y.round(1), elevation=c.z95.round(2),
            elevation_p90=c.z90.round(2), elevation_p99=c.z99.round(2), geological_unit=c.geological_unit, control_type=c.control_type, quality_class=c.quality,
            selection_reason=c.selection_reason, family_relevance=c.family_relevance, spatial_cluster=c.block,
            sector=np.where(c.centre.astype(str) != "", c.centre.astype(str) + ":S" + c.sector.astype(str), ""),
            centre=c.centre, r_centre_m=c.r_centre_m.round(1), coherence_residual_m=c.coherence_residual_m.round(2),
            dominant_unit_share=c.dominant_unit_share, n_cells=c.n_cells, in_excluded_zone=c.in_excluded_zone, included_in_fit=c.included_in_fit))
        out.append(df)
    return pd.concat(out, ignore_index=True)


def control_audit(results: dict) -> pd.DataFrame:
    rows = []
    for lid, r in results.items():
        c = r["ctrl"]
        for dom, cr in r["comp_res"].items():
            cc = c[c.domain == dom]
            fit = cc[cc.included_in_fit]
            g = cr["geom"]
            sup_dom = (r["grids"]["domain"] == [k for k, v in R.domain_lookup(r["L"])[0].items() if v == dom][0])
            cov = float((cr["d_ctrl"][sup_dom] <= P.SUPPORT["max_control_distance_m"]).mean()) if sup_dom.any() else np.nan
            az = (fit.azimuth_deg // 45).value_counts().sort_index() if cr["kind"] == "RADIAL" else pd.Series(dtype=int)
            rows.append(dict(
                landscape_id=lid, component=dom, family=cr["kind"], candidates=len(cc), q1=int((cc.quality == "Q1").sum()), q2=int((cc.quality == "Q2").sum()),
                q3=int((cc.quality == "Q3").sum()), q4=int((cc.quality == "Q4").sum()), fit_controls=len(fit),
                types=";".join(f"{k}={v}" for k, v in fit.control_type.value_counts().items()),
                units=";".join(f"{k}={v}" for k, v in fit.geological_unit.value_counts().items()),
                q3_reasons=";".join(f"{k}={v}" for k, v in cc[cc.quality == "Q3"].selection_reason.str.split(":").str[0].str.split(" \\(").str[0].value_counts().items()),
                elevation_p05=fit.z95.quantile(0.05), elevation_p50=fit.z95.median(), elevation_p95=fit.z95.quantile(0.95),
                centre_distance_km_p05=(fit.r_centre_m.quantile(0.05) / 1000) if cr["kind"] == "RADIAL" else np.nan,
                centre_distance_km_p50=(fit.r_centre_m.median() / 1000) if cr["kind"] == "RADIAL" else np.nan,
                centre_distance_km_p95=(fit.r_centre_m.quantile(0.95) / 1000) if cr["kind"] == "RADIAL" else np.nan,
                azimuth_45deg_counts=";".join(f"{int(k) * 45}-{int(k) * 45 + 45}={v}" for k, v in az.items()),
                occupied_sectors=g.get("occupied_sectors", np.nan), max_sector_share=g.get("max_sector_share", np.nan),
                occupied_blocks_2km=g.get("occupied_blocks_2km", np.nan), max_block_share=g.get("max_block_share", np.nan),
                controls_in_excluded_zones=g["controls_in_excluded_zones"], applicable_domain_within_3km_of_control=round(cov, 4),
                accordance_tau_m=round(r["tau"].get(dom, np.nan), 1), verdict=g["verdict"], reasons=g["reasons"]))
    return pd.DataFrame(rows)


def scenario_manifest(results: dict) -> pd.DataFrame:
    rows = []
    for lid, r in results.items():
        for dom, cr in r["comp_res"].items():
            n = len(cr["fits"])
            for j, f in enumerate(cr["fits"]):
                row = dict(landscape_id=lid, component=dom, family=cr["kind"], scenario_id=f"{lid[:4]}_{dom[:3]}_{j:04d}", weight=round(1 / max(len(cr["acc"]), 1), 6) if f.accepted else 0.0,
                           accepted=f.accepted, reject_reason=f.reject_reason, sigma_m=round(f.sigma, 2), seed=P.SEED + f.params["replicate"], n_controls=len(cr["fitc"]))
                row.update({k: v for k, v in f.params.items()})
                if f.kind == "RADIAL" and f.coef is not None:
                    row["base_B_m"] = round(f.extra["B"], 2)
                    row["centres_R_m_c0_c1_c2"] = "|".join(f"{nm}:{k[2]:.0f},{k[4]:.1f},{k[5]:.1f},{k[6]:.1f}" for nm, k in zip(f.extra["kept"], f.extra["centres"]))
                elif f.coef is not None:
                    row["poly_coef"] = ",".join(f"{v:.4g}" for v in f.coef)
                rows.append(row)
            if not n:
                rows.append(dict(landscape_id=lid, component=dom, family=cr["kind"], scenario_id="NONE", accepted=False, reject_reason="FEWER_THAN_6_FIT_CONTROLS"))
    return pd.DataFrame(rows)


def validation_table(results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, res = [], []
    G = P.GATES
    for lid, r in results.items():
        v = r["val"]
        base = dict(landscape_id=lid, scope="LANDSCAPE", family="+".join(f"{d}:{k}" for d, k in r["comps"]), R_L_m=r["R_L"], Lambda_L_m=r["Lambda"], tau_L_m=r["tau_L"],
                    heldout_controls=len(v), heldout_groups=v.group.nunique() if len(v) else 0,
                    median_abs_residual_fraction=r["gates"]["median_abs_residual"][0], signed_bias_fraction=(v.residual.mean() / r["R_L"]) if len(v) else np.nan,
                    abs_bias_fraction=r["gates"]["abs_bias"][0], holdout_band_fraction=r["gates"]["holdout_band"][0], admissible_controls=r["n_fit"],
                    support_fraction_of_units=r["sup_frac"], negative_removal_fraction=r["neg_frac"], max_reference_slope_deg=r["max_slope"],
                    p99_reference_slope_deg=r["p99_slope"])
        for k, (val, ok) in r["gates"].items():
            base[f"gate_{k}"] = "PASS" if ok else "FAIL"
        base["all_gates_pass"] = all(ok for _, ok in r["gates"].values())
        base["geometric_checks"] = json.dumps(r["geo_checks"], default=str)
        for q in ("Q1", "Q2"):
            vq = v[v.quality == q] if len(v) else v
            base[f"median_abs_residual_fraction_{q}"] = float(np.median(np.abs(vq.residual)) / r["R_L"]) if len(vq) else np.nan
        rows.append(base)
        for dom in r["comp_res"]:
            vd = v[v.component == dom] if len(v) else v
            if len(vd):
                rows.append(dict(landscape_id=lid, scope=f"COMPONENT:{dom}", family=r["comp_res"][dom]["kind"], R_L_m=r["R_L"], heldout_controls=len(vd),
                                 heldout_groups=vd.group.nunique(), median_abs_residual_fraction=float(np.median(np.abs(vd.residual)) / r["R_L"]),
                                 signed_bias_fraction=float(vd.residual.mean() / r["R_L"]), abs_bias_fraction=abs(float(vd.residual.mean() / r["R_L"])),
                                 holdout_band_fraction=float(vd.inside_band.mean())))
        if len(v):
            res.append(v.assign(landscape_id=lid, residual_fraction=v.residual / r["R_L"]))
    return pd.DataFrame(rows), (pd.concat(res, ignore_index=True) if res else pd.DataFrame())


# ------------------------------------------------------------------------------------------------ QA maps (no Y)
def qa_maps(r: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, lid = r["grids"], r["lid"]
    h, w = g["z_modern"].shape
    x0, y0 = float(g["x0"]) / 1000, float(g["y0"]) / 1000
    ext = [x0, x0 + w * 0.09, y0 - h * 0.09, y0]
    sup = g["support"]
    m = lambda a: np.where(sup, a, np.nan)
    c = r["ctrl"]
    fit = c[c.included_in_fit]
    panels = [("Modern elevation (m)", g["z_modern"], "terrain", None),
              ("Scenario-median reference (m), support only", m(g["zref_median"]), "terrain", None),
              ("Median removed fraction (Zref-Zmod)/R_L", m(g["removed_fraction"]), "RdBu", (-0.6, 0.6)),
              ("Scenario spread IQR/R_L", m(g["spread_fraction"]), "viridis", (0, np.nanpercentile(g["spread_fraction"][sup], 99) if sup.any() else 1)),
              ("Control locations (Q1/Q2 fit; Q3 grey)", g["z_modern"], "gray", None),
              ("Control distance / Lambda_L", m(g["control_distance_fraction"]) if g["control_distance_fraction"] is not None else np.full((h, w), np.nan), "magma_r", (0, 0.5)),
              ("Support mask (caldera interior, vent cores hatched)", sup.astype(float) + 2 * g["caldera_interior"] + 3 * g["vent_core"], "Set2", (0, 4))]
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for ax, (title, arr, cmap, lim) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, extent=ext, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04)
        if title.startswith("Control"):
            q3 = c[c.quality == "Q3"]
            ax.scatter(q3.x / 1000, q3.y / 1000, s=2, c="0.6", label=f"Q3 ({len(q3)})")
            for q, col in (("Q1", "tab:red"), ("Q2", "tab:blue")):
                s = fit[fit.quality == q]
                ax.scatter(s.x / 1000, s.y / 1000, s=3, c=col, label=f"{q} ({len(s)})")
            rc = r["rad_cen"]
            if len(rc):
                ax.scatter(rc.x / 1000, rc.y / 1000, marker="^", s=80, c="yellow", edgecolors="k", label="registered centre")
            ax.legend(fontsize=7, loc="lower right")
    ax = axes.ravel()[-1]
    ax.axis("off")
    txt = (f"{lid}\nfamily: {'+'.join(f'{d}:{k}' for d, k in r['comps'])}\nstatus: {r['status']}\nfailures: {', '.join(r['failures']) or 'none'}\n"
           f"R_L = {r['R_L']:.0f} m   Lambda_L = {r['Lambda'] / 1000:.1f} km\nsupport (units) = {r['sup_frac']:.3f}\n"
           + "\n".join(f"{k}: {v[0]:.4g} {'PASS' if v[1] else 'FAIL'}" for k, v in r["gates"].items())
           + "\n\nReconstruction QA only: no landslide data shown or used.")
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=10)
    for a in axes.ravel()[:-1]:
        a.set_xlabel("x (km)", fontsize=8)
        a.set_ylabel("y (km)", fontsize=8)
    fig.suptitle(f"Stage 4C-B reconstruction QA - {lid} (development; Y-blind)", fontsize=13)
    p = QA / f"{lid}_reconstruction_qa_maps.png"
    QA.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=80, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    # control-geometry audit figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(g["z_modern"], extent=ext, cmap="gray")
    for q, col in (("Q1", "tab:red"), ("Q2", "tab:blue"), ("Q3", "0.7")):
        s = c[c.quality == q]
        axes[0].scatter(s.x / 1000, s.y / 1000, s=3, c=col, label=f"{q} ({len(s)})")
    if g["caldera_interior"].any():
        axes[0].contour(np.flipud(g["caldera_interior"].astype(float)), levels=[0.5], colors="orange",
                        extent=[ext[0], ext[1], ext[2], ext[3]], origin="lower")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Control map (orange = masked caldera interior)")
    axes[1].hist([fit.z95, c[c.quality == "Q3"].z95], bins=40, label=["fit (Q1/Q2)", "Q3"], stacked=True)
    axes[1].set_title("Control elevation distribution (m)")
    axes[1].legend()
    if "azimuth_deg" in fit and fit.azimuth_deg.notna().any():
        axes[2].remove()
        axp = fig.add_subplot(1, 3, 3, projection="polar")
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        axp.hist(np.radians(fit.azimuth_deg.dropna()), bins=8)
        axp.set_title("Azimuth of fit controls about assigned centre")
    else:
        vc = pd.Series(R.holdout_groups(fit, "ENVELOPE")).value_counts()
        axes[2].hist(vc.values, bins=30)
        axes[2].set_title("Fit controls per 2 km validation block")
    fig.suptitle(f"Stage 4C-B control audit - {lid} (Y-blind)")
    p2 = QA / f"{lid}_control_audit.png"
    fig.savefig(p2, dpi=80, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    p3 = QA / f"{lid}_reconstruction_grids.npz"
    np.savez_compressed(p3, **{k: v for k, v in g.items() if v is not None})
    return [p, p2, p3]


# ------------------------------------------------------------------------------------------------ H dataset / audits
def h_frame(lid: str, r: dict | None, released: bool, status: str) -> pd.DataFrame:
    if r is None:
        u = B.load(lid, "terrain_units/terrain_units", usecols=["terrain_unit_id"])
        n = len(u)
        df = pd.DataFrame(dict(landscape_id=lid, terrain_unit_id=u.terrain_unit_id.to_numpy()))
        for c in CORE:
            df[HCOL[c]] = np.full(n, np.nan)
        df["h_available"] = False
        for q in ("qa_h_removed_fraction_p05", "qa_h_removed_fraction_p95", "qa_h_reference_relative_height_p05", "qa_h_reference_relative_height_p95"):
            df[q] = np.nan
        df["qa_h_support"] = False
        df["qa_h_family"] = "AGE_HISTORY_PROXY"
        df["qa_h_semantics"] = "NONE"
        df["qa_h_confidence"] = ""
        df["qa_h_landscape_status"] = status
        df["qa_h_validation_status"] = "NOT_APPLICABLE"
        return df
    u = r["L"].units
    df = pd.DataFrame(dict(landscape_id=lid, terrain_unit_id=u.terrain_unit_id.to_numpy()))
    for c in CORE:
        df[HCOL[c]] = r["hv"][c] if released else np.full(len(u), np.nan)
    df["h_available"] = r["avail"] if released else np.zeros(len(u), bool)
    for q, v in r["qa"].items():
        df[q] = v if released else np.full(len(u), np.nan)
    df["qa_h_support"] = r["unit_sup"]
    df["qa_h_family"] = r["fam_u"]
    df["qa_h_semantics"] = r["sem_u"]
    df["qa_h_confidence"] = r["conf"] if released else ""
    df["qa_h_landscape_status"] = status
    df["qa_h_validation_status"] = "PASS_ALL_GATES" if released else "FAILED:" + "|".join(r["failures"])
    return df


def distributions(frames: dict, role: str) -> pd.DataFrame:
    rows = []
    for lid, df in frames.items():
        a = df[df.h_available]
        if not len(a):
            continue
        for c in CORE:
            s = a[HCOL[c]]
            q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append(dict(role=role, landscape_id=lid, feature=c, n=len(s), median=q[0.5], iqr=q[0.75] - q[0.25], p05=q[0.05], p95=q[0.95], min=s.min(), max=s.max(),
                             mean=s.mean(), sd=s.std(), fraction_unique_values=s.nunique() / len(s), near_constant=bool((q[0.95] - q[0.05]) < 1e-3)))
    return pd.DataFrame(rows)


def identity_audit(frames: dict, role: str) -> pd.DataFrame:
    a = pd.concat([df[df.h_available] for df in frames.values()], ignore_index=True)
    if a.landscape_id.nunique() < 2:
        return pd.DataFrame()
    rows = []
    lids = a.landscape_id.to_numpy()
    for c in CORE:
        st = identity_stats(a[HCOL[c]], lids, categorical=False)
        pooled = a[HCOL[c]]
        piqr = float(pooled.quantile(0.75) - pooled.quantile(0.25)) or 1.0
        meds = a.groupby("landscape_id")[HCOL[c]].median()
        rows.append(dict(role=role, landscape_id="ALL", feature=c, **st, max_standardised_median_shift=float((meds.max() - meds.min()) / piqr),
                         identity_flag="EXCLUDED_IDENTITY_LEVEL" if st["identity_purity90_coverage"] >= 0.9 else "IDENTITY_RISK" if st["identity_purity90_coverage"] >= 0.5 else "NONE"))
    return pd.DataFrame(rows)


def redundancy(frames: dict, role: str) -> pd.DataFrame:
    parts = []
    for lid, df in frames.items():
        a = df[df.h_available]
        if not len(a):
            continue
        m = B.load(lid, "features_m/features_m")  # frozen M only (no target column exists in this table)
        assert "y_positive" not in m.columns
        mm = m.set_index("terrain_unit_id").loc[a.terrain_unit_id]
        parts.append(pd.concat([a.set_index("terrain_unit_id")[[HCOL[c] for c in CORE] + ["landscape_id"]], mm], axis=1))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts)
    mcols = [c for c in d.columns if c not in [HCOL[x] for x in CORE] + ["landscape_id"]]
    assert len(mcols) == 60
    d = d.dropna(subset=mcols)
    Xm = d[mcols].to_numpy(float)
    Xm = (Xm - Xm.mean(0)) / np.where(Xm.std(0) > 0, Xm.std(0), 1)
    X = np.c_[np.ones(len(d)), Xm]
    rows = []
    for c in CORE:
        yv = d[HCOL[c]].to_numpy(float)
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        r2 = 1 - np.sum((yv - X @ beta) ** 2) / np.sum((yv - yv.mean()) ** 2)
        per = {}
        for lid, g in d.groupby("landscape_id"):
            Xg = np.c_[np.ones(len(g)), (g[mcols].to_numpy(float) - g[mcols].to_numpy(float).mean(0)) / np.where(g[mcols].to_numpy(float).std(0) > 0, g[mcols].to_numpy(float).std(0), 1)]
            yg = g[HCOL[c]].to_numpy(float)
            bg = np.linalg.lstsq(Xg, yg, rcond=None)[0]
            per[lid] = round(float(1 - np.sum((yg - Xg @ bg) ** 2) / np.sum((yg - yg.mean()) ** 2)), 4)
        corr = pd.Series({mc: np.corrcoef(d[mc], d[HCOL[c]])[0, 1] for mc in mcols}).abs().sort_values(ascending=False)
        rows.append(dict(role=role, feature=c, rows=len(d), landscapes=d.landscape_id.nunique(), pooled_r2_on_60_M=round(float(r2), 4),
                         redundant_r2_ge_0_90=bool(r2 >= P.REDUNDANCY["threshold"]), per_landscape_r2=json.dumps(per),
                         top3_abs_corr_M="; ".join(f"{k}:{v:.3f}" for k, v in corr.head(3).items())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ main
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    prior = {k: sha256(ROOT / v) for k, v in PRIOR.items()}
    if prior != PRIOR_SHA:
        raise RuntimeError(f"STOP: authoritative prior checkpoint changed: {[k for k in prior if prior[k] != PRIOR_SHA[k]]}")
    s4a = json.loads((ROOT / PRIOR["stage4a"]).read_text())
    meg_before = {p: sha256(ROOT / p) for p in s4a["meg_partitions"]}
    assert meg_before == s4a["meg_partitions"], "STOP: Stage 4A production dataset differs from its checkpoint"
    prereg = dict(module="src/sfv2/stage4c_b_prereg.py", sha256=sha256(ROOT / "src/sfv2/stage4c_b_prereg.py"), frozen_before_reconstruction=True,
                  amendments_before_any_fit=["v0 26037b94... (2026-10-07T06:57Z)",
                                             "amendment 1 c799488e... (06:59Z, before any fit): Y-guard path prefix; Lombok caldera centre from Copernicus WBM lake; "
                                             "ADDITIVE_SUPERPOSITION replaces NEAREST_CENTRE (continuous); slope statistic wording; control-geometry FAIL -> FAILED_H"],
                  content={k: getattr(P, k) for k in ("SEED", "SOURCES", "GVP", "RADIAL_CENTRES", "FAMILY", "APPLICABLE", "YOUNG_DEPOSIT_CODES", "WINDOW_BUFFER_M",
                                                      "CONTROL", "QUALITY", "CALDERA", "CONTROL_GEOMETRY", "CENTRE_GEOMETRY_GATE", "ENVELOPE_SCENARIOS",
                                                      "RADIAL_SCENARIOS", "FIT", "SUPPORT", "VALIDATION", "GATES", "FAILURE_CODES", "STATUS_RULE", "H_OPERATORS",
                                                      "CONFIDENCE_QA", "REDUNDANCY")})
    assert P.GATES == H.VALIDATION_GATES["gates"], "STOP: validation gates differ from Stage 4C-A"
    _json(OUT / "STAGE4C_B_PREREGISTRATION.json", prereg)

    cen_all, reg = acquire_centres()
    results = {}
    for lid in P.LANDSCAPES:
        if P.FAMILY[lid]["primary"] == "AGE_HISTORY_PROXY":
            continue
        print(f"[4C-B] reconstructing {lid}", flush=True)
        results[lid] = reconstruct(lid, cen_all)

    centres = centre_table(cen_all, results)
    ctab = control_table(results)
    audit = control_audit(results)
    manifest = scenario_manifest(results)
    vtab, vres = validation_table(results)
    _tsv(OUT / "STAGE4C_B_VOLCANO_CENTRES.tsv", centres)
    _tsv(OUT / "STAGE4C_B_CONTROLS.tsv", ctab)
    _tsv(OUT / "STAGE4C_B_CONTROL_AUDIT.tsv", audit)
    _tsv(OUT / "STAGE4C_B_SCENARIO_MANIFEST.tsv", manifest)
    _tsv(OUT / "STAGE4C_B_RECONSTRUCTION_VALIDATION.tsv", vtab)
    _tsv(OUT / "STAGE4C_B_VALIDATION_HOLDOUT_RESIDUALS.tsv", vres)

    # ---- statuses
    status, st_rows = {}, []
    for lid in P.LANDSCAPES:
        if lid not in results:
            status[lid] = "AGE_PROXY_ONLY"
            st_rows.append(dict(landscape_id=lid, family_used="AGE_HISTORY_PROXY", semantics="NONE", status="AGE_PROXY_ONLY", failure_codes="",
                                verified_h_capable=False, core_h_released=False, cotabato_centre_gate="", notes="No constructional surface (Stage 4C-A frozen); no new Y-free evidence changes this"))
            continue
        r = results[lid]
        status[lid] = r["status"]
        fam = "+".join(f"{d}:{k}" for d, k in r["comps"])
        sem = "+".join(("ESTIMATED_REMOVAL" if k == "RADIAL" else "MINIMUM_REMOVAL") for _, k in r["comps"])
        st_rows.append(dict(landscape_id=lid, family_used=fam, semantics=sem, status=r["status"], failure_codes="|".join(r["failures"]),
                            verified_h_capable=r["status"].startswith("VERIFIED"), core_h_released=r["status"].startswith("VERIFIED"),
                            cotabato_centre_gate=json.dumps(r["gate"]) if r["gate"] else "",
                            notes=("caldera: " + json.dumps(r["cal_info"], default=str)) if r["cal_info"].get("method") != "NONE" else ""))
    stab = pd.DataFrame(st_rows)
    _tsv(OUT / "STAGE4C_B_RECONSTRUCTION_STATUS.tsv", stab)
    verified = [l for l in P.LANDSCAPES if status[l].startswith("VERIFIED")]
    n_ver = len(verified)
    unlocked = n_ver >= 3

    # ---- H dataset (all Stage 4A units; H only for verified reconstructions; never zero-filled)
    frames, diag_frames, files = {}, {}, {}
    for lid in P.LANDSCAPES:
        r = results.get(lid)
        df = h_frame(lid, r, released=lid in verified, status=status[lid])
        frames[lid] = df
        if r is not None:
            diag_frames[lid] = h_frame(lid, r, released=True, status=status[lid])  # diagnostic (unreleased) values for numerical audit only
        p = HDS / f"{lid}.npz"
        write_columns(p, df)
        files[str(p.relative_to(ROOT))] = sha256(p)
    total_rows = sum(len(f) for f in frames.values())
    assert total_rows == B.EXPECTED["rows"], "STOP: H dataset does not cover the Stage 4A units exactly"
    for lid, df in frames.items():
        t = B.load(lid, "terrain_units/terrain_units", usecols=["terrain_unit_id"])
        assert (df.terrain_unit_id.to_numpy() == t.terrain_unit_id.to_numpy()).all()
        hv = df[[HCOL[c] for c in CORE]].to_numpy()
        assert np.isnan(hv[~df.h_available.to_numpy()]).all(), "STOP: H outside support must be missing"
    _json(HDS / "SCHEMA.json", dict(format="numpy .npz (zip of .npy columns; deterministic; sfv2.build_stage4a.read_columns)", partition="landscape_id",
                                    rows=total_rows, key="terrain_unit_id (identical order to Stage 4A terrain units)",
                                    core_h_columns={c: HCOL[c] for c in CORE}, availability="h_available",
                                    qa_columns=["qa_h_removed_fraction_p05", "qa_h_removed_fraction_p95", "qa_h_reference_relative_height_p05",
                                                "qa_h_reference_relative_height_p95", "qa_h_support", "qa_h_family", "qa_h_semantics", "qa_h_confidence",
                                                "qa_h_landscape_status", "qa_h_validation_status"],
                                    rules=["no Y column", "H released only for VERIFIED landscapes", "H outside support or unverified = NaN (never zero-filled, never imputed)",
                                           "qa_* columns are never predictors"]))

    # ---- distributions / identity / redundancy / selected-H
    rel = {l: frames[l] for l in verified}
    dist = pd.concat([distributions(rel, "RELEASED_VERIFIED_H"), distributions(diag_frames, "DIAGNOSTIC_UNRELEASED_ALL_RECONSTRUCTIONS")], ignore_index=True)
    ident = pd.concat([identity_audit(rel, "RELEASED_VERIFIED_H") if len(rel) >= 2 else pd.DataFrame(),
                       identity_audit(diag_frames, "DIAGNOSTIC_UNRELEASED_ALL_RECONSTRUCTIONS")], ignore_index=True)
    _tsv(OUT / "STAGE4C_B_H_DISTRIBUTIONS.tsv", pd.concat([dist, ident.assign(n=np.nan)], ignore_index=True))
    red = pd.concat([redundancy(rel, "RELEASED_VERIFIED_H"), redundancy(diag_frames, "DIAGNOSTIC_UNRELEASED_ALL_RECONSTRUCTIONS")], ignore_index=True)
    _tsv(OUT / "STAGE4C_B_H_REDUNDANCY_WITH_M.tsv", red)
    sel = []
    rr = red[red.role == "RELEASED_VERIFIED_H"] if len(red) else pd.DataFrame()
    for c in CORE:
        r2 = float(rr[rr.feature == c].pooled_r2_on_60_M.iloc[0]) if len(rr) and (rr.feature == c).any() else np.nan
        avail_all = bool(n_ver > 0 and all(frames[l][HCOL[c]].notna().any() for l in verified))
        a = r2 < 0.90 if np.isfinite(r2) else False
        sel.append(dict(feature=c, criterion_a_r2_lt_0_90=a, pooled_r2_on_60_M=r2, criterion_b_available_in_every_verified_landscape=avail_all,
                        criterion_c_reconstruction_validation_passes=n_ver > 0, verified_landscapes="|".join(verified) or "none",
                        selected_h=bool(a and avail_all and n_ver > 0),
                        row_rule="h_control_distance_fraction <= 0.5",
                        selected_rows={l: int((frames[l].h_available & (frames[l].h_control_distance_fraction <= 0.5)).sum()) for l in verified},
                        note="" if n_ver else "No landscape passed every frozen gate: SELECTED_H is empty by the frozen rule"))
    seltab = pd.DataFrame(sel)
    _tsv(OUT / "STAGE4C_B_SELECTED_H.tsv", seltab)
    sup_rows = []
    for lid in P.LANDSCAPES:
        r = results.get(lid)
        n = len(frames[lid])
        if r is None:
            sup_rows.append(dict(landscape_id=lid, status=status[lid], stage4a_units=n, supported_units=0, supported_unit_fraction=0.0, support_area_km2=0.0,
                                 support_fraction_of_landscape=0.0, h_available_units_released=0, R_L_m=np.nan, Lambda_L_km=np.nan))
            continue
        sup_rows.append(dict(landscape_id=lid, status=status[lid], stage4a_units=n, supported_units=int(r["avail"].sum()), supported_unit_fraction=r["sup_frac"],
                             support_area_km2=round(r["support_area_km2"], 2), support_fraction_of_landscape=r["sup_frac"],
                             window_support_cells=int(r["grids"]["support"].sum()), h_available_units_released=int(frames[lid].h_available.sum()),
                             selected_rows_ctrl_dist_le_0_5=int((r["avail"] & (r["hv"]["H_CONTROL_DISTANCE_FRACTION"] <= 0.5)).sum()),
                             R_L_m=round(r["R_L"], 2), Lambda_L_km=round(r["Lambda"] / 1000, 3), tau_L_m=round(r["tau_L"], 2),
                             caldera_or_vent_units_excluded=int((~r["unit_sup"]).sum())))
    _tsv(OUT / "STAGE4C_B_SUPPORT_SUMMARY.tsv", pd.DataFrame(sup_rows))

    qa_files = []
    for lid, r in results.items():
        qa_files += qa_maps(r)

    # ---- report + checkpoint
    v1_after = v1_verify()
    prior_after = {k: sha256(ROOT / v) for k, v in PRIOR.items()}
    meg_after = {p: sha256(ROOT / p) for p in s4a["meg_partitions"]}
    ml = [p for p in scan_ml_h_artifacts() if not p.startswith("outputs/stage4b/")]
    import scipy
    import matplotlib
    summary = dict(results=results, status=status, verified=verified, n_ver=n_ver, unlocked=unlocked, centres=centres, audit=audit, vtab=vtab,
                   manifest=manifest, red=red, dist=dist, ident=ident, seltab=seltab, sup=pd.DataFrame(sup_rows), stab=stab, v1_before=v1_before,
                   v1_after=v1_after, prior_ok=prior_after == PRIOR_SHA, meg_ok=meg_after == meg_before, ml=ml, reg=reg, ctab=ctab)
    (OUT / "STAGE4C_B_REPORT.md").write_text(report(summary))
    outs = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith(("STAGE4C_B_CHECKPOINT", "STAGE4C_B_DIAGNOSTICS")))
    cp = dict(
        schema="SFV2_STAGE4C_B_CHECKPOINT_V1", as_of=P.AS_OF, stage4c_a_checkpoint_sha256=PRIOR_SHA["stage4c_a"], prior_checkpoints={PRIOR[k]: v for k, v in prior_after.items()},  # keyed by path, as in every V2 checkpoint
        prior_checkpoints_unchanged=prior_after == PRIOR_SHA, stage4a_meg_partitions_unchanged=meg_after == meg_before,
        preregistration=dict(path="outputs/stage4c_b/STAGE4C_B_PREREGISTRATION.json", module_sha256=prereg["sha256"]),
        code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage4c_b.py", "stage4c_b_recon.py", "stage4c_b_prereg.py", "stage4a_base.py", "stage2d_production.py")},
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, rasterio=rasterio.__version__,
                      matplotlib=matplotlib.__version__),
        random_seed=P.SEED, raw_acquisitions=[{k: e[k] for k in ("acquisition_id", "download_url", "sha256", "access_date", "raw_path", "actual_bytes")} for e in reg],
        scenario_grids=dict(envelope=P.ENVELOPE_SCENARIOS, radial=P.RADIAL_SCENARIOS), control_parameters=P.CONTROL, support_rules=P.SUPPORT,
        validation_gates=P.GATES,
        validation=vtab[vtab.scope == "LANDSCAPE"].drop(columns=["geometric_checks"]).to_dict("records"),
        statuses=status, verified_h_capable=verified, n_verified_h_capable=n_ver, megh_experiment_unlocked="YES" if unlocked else "NO",
        core_h=CORE, h_dataset_partitions=files, h_zero_filled=False, h_imputed=False,
        model_trained=False, megh_trained=False, delta_h_computed=False, h_predictive_performance_observed=False, y_used_for_h=False,
        y_loaded=False, stage4b_residuals_inspected=False, lockbox_accessed=False, stage4a_dataset_modified=False,
        ml_artifacts_outside_stage4b=ml,
        v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
        v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
        output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs},
        volatile_diagnostics={"path": "outputs/stage4c_b/STAGE4C_B_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE4C_B_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE4C_B_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD")))
    return dict(cp=cp, summary=summary)


def _md(df: pd.DataFrame, cols: list, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c in fmt and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                v = format(v, fmt[c])
            cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(body)


def report(s: dict) -> str:
    res, vt = s["results"], s["vtab"]
    lv = vt[vt.scope == "LANDSCAPE"].copy()
    f4 = {c: ".4f" for c in ("median_abs_residual_fraction", "abs_bias_fraction", "signed_bias_fraction", "holdout_band_fraction", "support_fraction_of_units",
                              "negative_removal_fraction")} | {"max_reference_slope_deg": ".2f", "p99_reference_slope_deg": ".2f", "R_L_m": ".1f", "tau_L_m": ".1f"}
    gate_cols = [c for c in lv.columns if c.startswith("gate_")]
    cen = s["centres"]
    cen_rows = cen[cen.role.isin(["RADIAL_RECONSTRUCTION_CENTRE", "CALDERA_MASK_REFERENCE", "NOT_ACQUIRED", "VENT_CORE_EXCLUSION_ONLY"])]
    au = s["audit"]
    man = s["manifest"]
    mc = man.groupby(["landscape_id", "component"]).agg(scenarios=("accepted", "size"), accepted=("accepted", "sum")).reset_index()
    red = s["red"]
    dist = s["dist"]
    sel = s["seltab"]
    sup = s["sup"]
    gate_txt = []
    for lid, r in res.items():
        if r["gate"]:
            gate_txt.append(f"- {lid}: {json.dumps(r['gate'])} -> family {'RADIAL' if r['gate']['PASS'] else 'ENVELOPE (fallback)'}")
    unl = "YES" if s["unlocked"] else "NO"
    return f"""# Stage 4C-B report - production H reconstruction + Y-blind validation

**DEVELOPMENT ONLY.** No model trained, no MEGH, no Delta_H, no H predictive performance, no Y, no Stage 4B residuals, no lockbox.

## A. Executive result
- Reconstructed landscapes: {len(res)} (Noto = AGE_PROXY_ONLY by the frozen 4C-A taxonomy).
- Statuses: {json.dumps(s['status'])}.
- **N_VERIFIED_H_CAPABLE = {s['n_ver']}** -> **MEGH_EXPERIMENT_UNLOCKED = {unl}**.
- CORE_H is released (non-missing) only for verified landscapes: {', '.join(s['verified']) or 'none'}.

## B. Public centres (Smithsonian GVP VOTW, bbox-limited WFS queries; no row outside a development box was downloaded)
{_md(cen_rows, ['landscape_id', 'volcano_name', 'gvp_volcano_number', 'latitude', 'longitude', 'catalogue_layer', 'role', 'uncertainty_ambiguity'])}

## C-D. Controls and control geometry (one control per 1 km block; Q1/Q2 fitted only)
{_md(au, ['landscape_id', 'component', 'family', 'candidates', 'q1', 'q2', 'q3', 'q4', 'fit_controls', 'occupied_sectors', 'max_sector_share', 'occupied_blocks_2km', 'max_block_share', 'controls_in_excluded_zones', 'applicable_domain_within_3km_of_control', 'verdict'])}

Q3 reasons: {'; '.join(f"{a.landscape_id}/{a.component}: {a.q3_reasons}" for a in au.itertuples())}

## E. Families used
{_md(s['stab'], ['landscape_id', 'family_used', 'semantics', 'status', 'failure_codes'])}

Cotabato centre-geometry gate (pre-registered in stage4c_b_prereg.py before any fit):
{chr(10).join(gate_txt) or '- n/a'}

## F. Scenario ensembles (equal weights; parameters bounded by controls; never by Y or ML)
{_md(mc, ['landscape_id', 'component', 'scenarios', 'accepted'])}

## G. Support masks
{_md(sup, ['landscape_id', 'status', 'stage4a_units', 'supported_units', 'supported_unit_fraction', 'support_area_km2', 'R_L_m', 'Lambda_L_km', 'h_available_units_released'], {'supported_unit_fraction': '.4f'})}

## H-I. Y-blind validation (held-out controls; gates verbatim from Stage 4C-A)
{_md(lv, ['landscape_id', 'heldout_controls', 'heldout_groups', 'median_abs_residual_fraction', 'abs_bias_fraction', 'holdout_band_fraction', 'admissible_controls', 'support_fraction_of_units', 'negative_removal_fraction', 'max_reference_slope_deg', 'p99_reference_slope_deg'], f4)}

{_md(lv, ['landscape_id'] + gate_cols + ['all_gates_pass'])}

Component-level rows, per-control held-out residuals and geometric checks are in STAGE4C_B_RECONSTRUCTION_VALIDATION.tsv and STAGE4C_B_VALIDATION_HOLDOUT_RESIDUALS.tsv.
Independent levels are controls, sectors/blocks and landscapes; terrain cells are never treated as replicates.

## J. CORE_H generation
Six frozen CORE_H columns (h_ prefix) in STAGE4C_B_H_DATASET/ for all {sum(len(r['L'].units) for r in res.values()) + int(sup[sup.landscape_id == 'NOTO_PENINSULA_2024'].stage4a_units.sum()):,} Stage 4A units; released values only where status is VERIFIED; NaN elsewhere (no zero-fill, no imputation).

## K. H distributions (released and diagnostic-unreleased; no Y)
{_md(dist[dist.feature.notna() & dist["median"].notna()], ['role', 'landscape_id', 'feature', 'n', 'median', 'iqr', 'p05', 'p95', 'min', 'max'], {c: '.4g' for c in ('median', 'iqr', 'p05', 'p95', 'min', 'max')})}

Identity / scale-shift audit:
{_md(s['ident'], ['role', 'feature', 'identity_purity90_coverage', 'between_landscape_variance_fraction', 'max_standardised_median_shift', 'identity_flag']) if len(s['ident']) else 'n/a'}

## L. Redundancy with M (pooled OLS R^2 on 60 frozen M predictors; no Y)
{_md(red, ['role', 'feature', 'rows', 'landscapes', 'pooled_r2_on_60_M', 'redundant_r2_ge_0_90', 'top3_abs_corr_M']) if len(red) else 'n/a'}

## M. SELECTED_H (frozen Y-independent rule)
{_md(sel, ['feature', 'criterion_a_r2_lt_0_90', 'criterion_b_available_in_every_verified_landscape', 'criterion_c_reconstruction_validation_passes', 'selected_h', 'note'])}

## N-O. Verified subset and unlock
N_VERIFIED_H_CAPABLE = {s['n_ver']} ({', '.join(s['verified']) or 'none'}). MEGH_EXPERIMENT_UNLOCKED = {unl}. MEGH is NOT trained in this stage.

## P. QA maps
STAGE4C_B_RECONSTRUCTION_QA/<landscape>_reconstruction_qa_maps.png (modern elevation, scenario-median reference, removed fraction, spread, controls,
control distance, support mask), <landscape>_control_audit.png and <landscape>_reconstruction_grids.npz. No landslide data shown.

## Q. Integrity
V1 before {s['v1_before']['strict_files']} strict / {s['v1_before']['writes_performed']} writes; after {s['v1_after']['strict_files']} / {s['v1_after']['writes_performed']}.
Prior checkpoints 3F-4C-A unchanged: {s['prior_ok']}. Stage 4A MEG partitions unchanged: {s['meg_ok']}. ML artifacts outside Stage 4B: {s['ml'] or 'none'}.
"""


if __name__ == "__main__":
    main()
