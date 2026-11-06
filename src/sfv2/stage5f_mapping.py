"""Reusable Stage 5F development-only susceptibility mapping interface.

Only the six governed development landscapes and frozen LOGISTIC_M / Stage 4F
novelty artifacts are readable.  No model fitting or calibration is provided.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.stats import rankdata

from . import stage4a_base as B
from .raw_store import sha256
from .schema import ROOT


OUT = ROOT / "outputs/stage5f"
MODEL_ID = "LOGISTIC_M_STAGE4EB_FROZEN"
LANDSCAPES = ["KUMAMOTO_ASO_2016", "NOTO_PENINSULA_2024", "LOMBOK_RINJANI_2018",
              "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
MEG_DIR = ROOT / "outputs/stage4a/STAGE4A_MEG_DATASET"
WARNING = ("Relative within-landscape susceptibility only. Raw score is uncalibrated and percentile is not landslide probability. "
           "Exploratory feature novelty measures feature-space difference, not predictive reliability.")
DISPLAY = {
    "KUMAMOTO_ASO_2016": "Kumamoto Aso 2016", "NOTO_PENINSULA_2024": "Noto Peninsula 2024",
    "LOMBOK_RINJANI_2018": "Lombok Rinjani 2018", "COTABATO_DAVAO_2019": "Cotabato Davao 2019",
    "ATITLAN_STAN_2005": "Atitlán Stan 2005", "DOMINICA_ERIKA_2015": "Dominica Erika 2015",
}


def _array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def within_landscape_percentile(score: np.ndarray) -> np.ndarray:
    """Average-tie percentile: average 1-based rank divided by N, times 100."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise ValueError("scores must be a non-empty finite vector")
    return rankdata(score, method="average") / len(score) * 100.0


@lru_cache(maxsize=1)
def _novelty_table() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/stage4f/STAGE4F_UNIT_NOVELTY.tsv", sep="\t")


def _terrain(lid: str) -> pd.DataFrame:
    B.firewall_check(lid)
    t = B.load(lid, "terrain_units/terrain_units")
    t = t[t.analysis_population_all_valid.astype(bool)].reset_index(drop=True)
    return t


def load_landscape(lid: str) -> dict:
    """Load and verify the frozen score, spatial-unit and U1 novelty join."""
    if lid not in LANDSCAPES: raise ValueError(f"not an authorized Stage 5F development landscape: {lid}")
    path = MEG_DIR / f"{lid}.npz"
    with np.load(path, allow_pickle=False) as z:
        ids = z["terrain_unit_id"].astype(str); x = z["centroid_x"].astype(np.int64); y = z["centroid_y"].astype(np.int64)
        crs_values = np.unique(z["projected_crs"].astype(str)); target = z["y_positive"].astype(bool)
    if len(crs_values) != 1: raise RuntimeError(f"STOP: non-unique CRS for {lid}")
    score_path = ROOT / f"outputs/stage5d_e/predictions/B0__{lid}__scores_v1.npy"
    score = np.load(score_path).astype(np.float64)
    if not (len(score) == len(ids) == len(x) == len(y)): raise RuntimeError(f"STOP: prediction length mismatch for {lid}")
    novelty = _novelty_table(); novelty = novelty[novelty.landscape_id == lid].reset_index(drop=True)
    if len(novelty) != len(ids) or not np.array_equal(novelty.terrain_unit_id.astype(str).to_numpy(), ids):
        raise RuntimeError(f"STOP: Stage 4F novelty identifier join mismatch for {lid}")
    if not (np.array_equal(novelty.centroid_x.to_numpy(), x) and np.array_equal(novelty.centroid_y.to_numpy(), y)):
        raise RuntimeError(f"STOP: Stage 4F novelty coordinate mismatch for {lid}")
    # Stage 4F independently reconstructed each frozen fold before its
    # 12-significant-digit TSV export. Its score is used only to condition U1;
    # Stage 5D-E remains the authoritative mapped score. Bound the combined
    # reconstruction/text difference rather than claiming binary identity.
    if np.max(np.abs(novelty.susceptibility_score.to_numpy() - score)) > 1e-9:
        raise RuntimeError(f"STOP: Stage 4F/5D-E score mismatch for {lid}")
    t = _terrain(lid)
    if not np.array_equal(t.terrain_unit_id.astype(str).to_numpy(), ids): raise RuntimeError(f"STOP: production terrain-unit order mismatch for {lid}")
    for a, b, name in ((t.centroid_x.to_numpy(), x, "x"), (t.centroid_y.to_numpy(), y, "y")):
        if not np.array_equal(a, b): raise RuntimeError(f"STOP: production centroid {name} mismatch for {lid}")
    if t.projected_crs.nunique() != 1 or t.projected_crs.iloc[0] != crs_values[0]: raise RuntimeError(f"STOP: production CRS mismatch for {lid}")
    if set(t.cell_area_m2.unique()) != {8100}: raise RuntimeError(f"STOP: non-90m terrain unit for {lid}")
    encoded = t.terrain_unit_id.str.extract(r"_(\d{5})_(\d{5})$").astype(int)
    if not (np.array_equal(encoded[0].to_numpy(), t.source_grid_row.to_numpy()) and np.array_equal(encoded[1].to_numpy(), t.source_grid_col.to_numpy())):
        raise RuntimeError(f"STOP: terrain identifier/source-grid mismatch for {lid}")
    x_offset = np.unique(t.centroid_x.to_numpy() - 30 * t.source_grid_col.to_numpy())
    y_offset = np.unique(t.centroid_y.to_numpy() + 30 * t.source_grid_row.to_numpy())
    if len(x_offset) != 1 or len(y_offset) != 1: raise RuntimeError(f"STOP: source-grid coordinate transform is not constant for {lid}")
    pct = within_landscape_percentile(score)
    return {"landscape_id": lid, "display_name": DISPLAY[lid], "terrain_unit_id": ids, "centroid_x": x, "centroid_y": y,
            "projected_crs": crs_values[0], "raw_score": score, "percentile": pct,
            "novelty": novelty.u1_percentile.to_numpy(np.float64), "novelty_distance": novelty.u1_distance.to_numpy(np.float64),
            "novelty_availability": "AVAILABLE_EXPLORATORY_ONLY", "y_positive": target, "terrain": t,
            "source_prediction_path": str(score_path.relative_to(ROOT)), "source_prediction_hash": sha256(score_path),
            "source_dataset_path": str(path.relative_to(ROOT)), "source_dataset_hash": sha256(path),
            "terrain_source_path": str(B.path(lid, "terrain_units/terrain_units.csv.gz").relative_to(ROOT)),
            "terrain_source_hash": sha256(B.path(lid, "terrain_units/terrain_units.csv.gz")),
            "spatial_transform_audit": {"source_x_offset": int(x_offset[0]), "source_y_offset": int(y_offset[0]),
                                        "source_resolution_m": 30, "terrain_unit_resolution_m": 90}}


def grid_layout(d: dict) -> dict:
    x, y = d["centroid_x"], d["centroid_y"]; res = 90
    minx, maxx, miny, maxy = int(x.min()), int(x.max()), int(y.min()), int(y.max())
    col = ((x - minx) / res).astype(int); row = ((maxy - y) / res).astype(int)
    if not (np.allclose(minx + col * res, x) and np.allclose(maxy - row * res, y)):
        raise RuntimeError(f"STOP: centroids do not occupy the authoritative 90m lattice for {d['landscape_id']}")
    width, height = int(col.max() + 1), int(row.max() + 1)
    linear = row.astype(np.int64) * width + col
    if len(np.unique(linear)) != len(linear): raise RuntimeError(f"STOP: duplicate raster cell for {d['landscape_id']}")
    transform = from_origin(minx - res / 2, maxy + res / 2, res, res)
    return {"row": row, "col": col, "width": width, "height": height, "resolution": res, "transform": transform,
            "bounds": (minx - 45, miny - 45, maxx + 45, maxy + 45)}


def rasterize_values(d: dict, values: np.ndarray, nodata: float = -9999.0) -> tuple[np.ndarray, dict]:
    layout = grid_layout(d); arr = np.full((layout["height"], layout["width"]), nodata, dtype=np.float32)
    arr[layout["row"], layout["col"]] = np.asarray(values, dtype=np.float32)
    if int(np.count_nonzero(arr != nodata)) != len(values): raise RuntimeError("STOP: raster coverage mismatch")
    return arr, layout


def structured_table(d: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "landscape_id": d["landscape_id"], "terrain_unit_id": d["terrain_unit_id"], "centroid_x": d["centroid_x"], "centroid_y": d["centroid_y"],
        "projected_crs": d["projected_crs"], "model_id": MODEL_ID, "source_prediction_hash": d["source_prediction_hash"],
        "raw_score": d["raw_score"], "within_landscape_percentile": d["percentile"], "exploratory_novelty": d["novelty"],
        "novelty_availability": d["novelty_availability"], "probability_calibration_status": "NOT_ESTABLISHED",
        "calibration_transfer_status": "CALIBRATION_TRANSFER_NOT_SUPPORTED", "transfer_confidence_status": "NOT_ESTABLISHED",
        "abstention_status": "NOT_SUPPORTED", "validation_status": "DEVELOPMENT_ONLY_HELD_OUT_LANDSCAPE",
        "interpretation_warning": WARNING,
    })


def export_table(d: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    structured_table(d).to_csv(path, index=False, lineterminator="\n", float_format="%.12g",
                               compression={"method": "gzip", "mtime": 0})
    return path


def export_geotiff(d: dict, values: np.ndarray, path: Path, layer: str, units: str, nodata: float = -9999.0) -> Path:
    arr, layout = rasterize_values(d, values, nodata); path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=layout["height"], width=layout["width"], count=1, dtype="float32",
                       crs=d["projected_crs"], transform=layout["transform"], nodata=nodata, compress="DEFLATE", predictor=3) as dst:
        dst.write(arr, 1)
        dst.update_tags(LANDSCAPE_ID=d["landscape_id"], MODEL_ID=MODEL_ID, LAYER=layer, UNITS=units,
                        DEVELOPMENT_ONLY="YES", CALIBRATED_PROBABILITY="NO", TRANSFER_CONFIDENCE="NOT_ESTABLISHED",
                        INTERPRETATION_WARNING=WARNING, SOURCE_PREDICTION_SHA256=d["source_prediction_hash"])
    return path


def _map_array(ax, d: dict, values: np.ndarray, cmap: str, vmin: float, vmax: float, label: str, title: str):
    arr, layout = rasterize_values(d, values); masked = np.ma.masked_equal(arr, -9999.0)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#dddddd")
    extent = [layout["bounds"][0], layout["bounds"][2], layout["bounds"][1], layout["bounds"][3]]
    im = ax.imshow(masked, extent=extent, origin="upper", cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    valid = (arr != -9999).astype(float)
    ax.contour(valid, levels=[.5], colors="black", linewidths=.35, origin="upper", extent=extent)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.set_xlabel(f"Easting (m) — {d['projected_crs']}"); ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False); ax.tick_params(labelsize=7)
    cb = ax.figure.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)
    # UTM maps are north-up under the affine transform.
    ax.annotate("N", xy=(.94, .91), xytext=(.94, .78), xycoords="axes fraction", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), fontsize=9, fontweight="bold")
    width_m = layout["bounds"][2] - layout["bounds"][0]
    candidates = np.array([1000, 2000, 5000, 10000, 20000, 50000]); bar = int(candidates[np.argmin(np.abs(candidates - width_m * .2))])
    x0 = layout["bounds"][0] + width_m * .07; y0 = layout["bounds"][1] + (layout["bounds"][3] - layout["bounds"][1]) * .06
    ax.plot([x0, x0 + bar], [y0, y0], color="black", lw=3); ax.text(x0 + bar / 2, y0, f" {bar/1000:g} km", ha="center", va="bottom", fontsize=7)
    return im


def status_panel(ax, d: dict):
    ax.axis("off"); ax.set_title("Calibration and evidence status", fontsize=11, fontweight="bold")
    lines = [("MODEL", "LOGISTIC_M"), ("OUTPUT", "Relative susceptibility ranking"),
             ("PROBABILITY CALIBRATION", "Not established"), ("CALIBRATION TRANSFER", "Not supported"),
             ("TRANSFER CONFIDENCE", "Not established"), ("ABSTENTION", "Not supported"),
             ("VALIDATION", "Development-only held-out landscape")]
    y = .88
    for key, value in lines:
        ax.text(.03, y, key, transform=ax.transAxes, fontsize=8, fontweight="bold", color="#355c7d")
        ax.text(.03, y - .055, value, transform=ax.transAxes, fontsize=9, wrap=True); y -= .125
    ax.text(.03, .02, "No numerical confidence or probability is supplied.", transform=ax.transAxes, fontsize=8,
            color="#9c2f2f", fontweight="bold", wrap=True)


def render_map(d: dict, map_type: str, path: Path, inventory_overlay: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    if map_type == "susceptibility":
        _map_array(ax, d, d["percentile"], "magma", 0, 100, "Within-landscape susceptibility percentile", d["display_name"])
    elif map_type == "novelty":
        _map_array(ax, d, 100 * d["novelty"], "viridis", 0, 100, "Exploratory feature novelty percentile",
                   d["display_name"] + " — larger = more feature-space novelty")
        ax.text(.5, -.12, "Feature-space difference, not predictive reliability", transform=ax.transAxes, ha="center", fontsize=8, fontweight="bold")
    elif map_type == "inventory_diagnostic":
        _map_array(ax, d, d["percentile"], "magma", 0, 100, "Within-landscape susceptibility percentile", d["display_name"])
        pos = d["y_positive"]; ax.scatter(d["centroid_x"][pos], d["centroid_y"][pos], s=4, facecolors="none", edgecolors="#00ffff", linewidths=.35,
                                          label="Mapped affected terrain unit")
        ax.legend(loc="upper left", fontsize=7); ax.text(.5, -.12, "DEVELOPMENT-ONLY DIAGNOSTIC — unlabelled does not prove absence", transform=ax.transAxes, ha="center", fontsize=8, fontweight="bold")
    else: raise ValueError(map_type)
    fig.suptitle("DEVELOPMENT-ONLY RESULTS", fontsize=11, fontweight="bold")
    fig.text(.5, .012, "Percentile is a relative rank, not landslide probability.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .96)); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
    return path


def render_three_panel(d: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1, .75]})
    _map_array(axes[0], d, d["percentile"], "magma", 0, 100, "Susceptibility percentile", "Relative susceptibility")
    _map_array(axes[1], d, 100 * d["novelty"], "viridis", 0, 100, "Novelty percentile", "Exploratory feature novelty")
    axes[1].text(.5, -.14, "Feature-space difference, not reliability", transform=axes[1].transAxes, ha="center", fontsize=8, fontweight="bold")
    status_panel(axes[2], d); fig.suptitle(f"{d['display_name']} — DEVELOPMENT-ONLY RESULTS", fontsize=13, fontweight="bold")
    fig.text(.5, .015, "Relative ranking only; no calibrated probability, validated confidence, or abstention rule.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95)); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)
    return path


def render_status(d: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6)); status_panel(ax, d); fig.suptitle(f"{d['display_name']} — DEVELOPMENT-ONLY RESULTS", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94)); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig); return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5F development-only mapping interface")
    ap.add_argument("action", choices=["export", "render"]); ap.add_argument("--landscape", required=True, choices=LANDSCAPES)
    ap.add_argument("--output-dir", required=True); ap.add_argument("--map-type", choices=["susceptibility", "novelty", "three_panel", "status", "inventory_diagnostic"], default="susceptibility")
    ap.add_argument("--format", choices=["png", "pdf", "geotiff", "csv.gz"], default="png"); args = ap.parse_args()
    d = load_landscape(args.landscape); out = Path(args.output_dir)
    if args.action == "export":
        if args.format == "csv.gz": path = export_table(d, out / f"{args.landscape}_STAGE5F_OUTPUT.csv.gz")
        elif args.format == "geotiff":
            if args.map_type == "susceptibility": path = export_geotiff(d, d["percentile"], out / f"{args.landscape}_SUSCEPTIBILITY_PERCENTILE.tif", "RELATIVE_SUSCEPTIBILITY_PERCENTILE", "percentile_0_100")
            elif args.map_type == "novelty": path = export_geotiff(d, d["novelty"], out / f"{args.landscape}_EXPLORATORY_NOVELTY.tif", "EXPLORATORY_FEATURE_NOVELTY", "training_reference_percentile_0_1")
            else: raise ValueError("GeoTIFF export supports susceptibility or novelty")
        else: raise ValueError("export supports geotiff or csv.gz")
    else:
        if args.format not in ("png", "pdf"): raise ValueError("render supports png or pdf")
        path = out / f"{args.landscape}_{args.map_type}.{args.format}"
        if args.map_type == "three_panel": render_three_panel(d, path)
        elif args.map_type == "status": render_status(d, path)
        else: render_map(d, args.map_type, path)
    print(json.dumps({"path": str(path), "sha256": sha256(path), "landscape_id": args.landscape, "map_type": args.map_type,
                      "interpretation_warning": WARNING}, indent=2))


if __name__ == "__main__": main()
