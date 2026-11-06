"""Generic pre-event M audit for multi-tile Copernicus GLO-30 coverage (generalises the Lombok Stage 2F.1 m_audit)."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import box, mapping

from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage3b_acquire import TILES, acq_id

# Copernicus DEM Product Handbook v5.0: EDM 1 = no edit (original); 8/9/10 = water (ocean, lake, river).  FLM 1 = no fill (Lombok
# convention kept: codes 0-5 and 9 are pre-event-compatible fill sources; others (e.g. 6/7/8 newer DEMs) are flagged).
WATER_EDM = {8, 9, 10}
PRE_EVENT_FLM = {0, 1, 2, 3, 4, 5, 9}


def _tag(xml: str, tag: str) -> str:
    m = re.search(rf"<{tag}>([^<]+)</{tag}>", xml)
    return m.group(1)[:10] if m else ""


def m_audit(lid: str, aoi_4326, event: str, earliest_trigger: str | None = None) -> dict:
    reg = {e["acquisition_id"]: e for e in load_registry()}
    cutoff = min(event, earliest_trigger or event)
    tiles, counts, nodata, cells = [], {k: Counter() for k in ("EDM", "FLM", "WBM")}, 0, 0
    covered = None
    for tile in TILES[lid]:
        p = {k: ROOT / reg[acq_id(lid, tile, k)]["raw_path"] for k in ("DEM", "XML", "SRC", "EDM", "FLM", "WBM")}
        xml = p["XML"].read_text(errors="ignore")
        src = re.sub(r"<[^>]+>", " ", p["SRC"].read_text(errors="ignore"))
        dates = sorted(set(re.findall(r"Acquisition Date:\s*(\d{4}-\d{2}-\d{2})", src)))
        with rasterio.open(p["DEM"]) as r:
            tb = box(*r.bounds)
            covered = tb if covered is None else covered.union(tb)
            inter = aoi_4326.intersection(tb)
            rec = dict(tile=tile, dem_sha256=reg[acq_id(lid, tile, "DEM")]["sha256"], dem_bytes=reg[acq_id(lid, tile, "DEM")]["actual_bytes"],
                       tandemx_start=_tag(xml, "tsxx:tsxx_startTime"), tandemx_stop=_tag(xml, "tsxx:tsxx_stopTime"),
                       editing_date=_tag(xml, "tsxx:DateofAnalysis"), src_scene_first=dates[0] if dates else "", src_scene_last=dates[-1] if dates else "",
                       intersects_aoi=not inter.is_empty)
            if not inter.is_empty:
                d, _ = rasterio.mask.mask(r, [mapping(inter)], crop=True, filled=False)
                a = d[0]
                vals = a.compressed()
                nodata += int((~np.isfinite(vals)).sum()) if r.nodata is None else int((vals == r.nodata).sum())
                cells += int(a.count())
            for k in ("EDM", "FLM", "WBM"):
                if not inter.is_empty:
                    with rasterio.open(p[k]) as q:
                        dk, _ = rasterio.mask.mask(q, [mapping(inter)], crop=True, filled=False)
                        v, n = np.unique(dk[0].compressed(), return_counts=True)
                        counts[k].update({int(a_): int(b_) for a_, b_ in zip(v, n)})
            rec["scenes_pre_event"] = bool(dates) and dates[-1] < cutoff and rec["tandemx_stop"] < cutoff
            rec["editing_after_event"] = rec["editing_date"] >= event if rec["editing_date"] else None
        tiles.append(rec)
    frac = {k: {c: round(n / max(sum(v.values()), 1), 5) for c, n in sorted(v.items())} for k, v in counts.items()}
    used = [t for t in tiles if t["intersects_aoi"]]
    return dict(landscape_id=lid, event_date=event, temporal_cutoff=cutoff, tiles=tiles, aoi_fully_covered=aoi_4326.difference(covered).area < 1e-9,  # tolerance for float seams between tiles
                aoi_cells=cells, dem_nodata_cells_in_aoi=nodata, edm_fraction_in_aoi=frac["EDM"], flm_fraction_in_aoi=frac["FLM"],
                wbm_fraction_in_aoi=frac["WBM"], water_edm_fraction_in_aoi=round(sum(frac["EDM"].get(c, 0) for c in WATER_EDM), 5),
                non_original_fraction_in_aoi=round(1 - frac["EDM"].get(1, 0), 5),
                fill_sources_all_pre_event_compatible=all(c in PRE_EVENT_FLM for c in frac["FLM"]),
                all_used_tiles_pre_event=all(t["scenes_pre_event"] for t in used),
                any_editing_after_event=any(t["editing_after_event"] for t in used),
                m_temporal_pass=bool(used) and all(t["scenes_pre_event"] for t in used) and all(c in PRE_EVENT_FLM for c in frac["FLM"]))
