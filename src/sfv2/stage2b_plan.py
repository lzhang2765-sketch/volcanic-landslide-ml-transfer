"""Stage 2B bounded acquisition plan: KUMAMOTO_ASO_2016 -> HOKKAIDO_IBURI_2018 -> NOTO_PENINSULA_2024.

Sources come from the Stage-2A-extension / Stage-2A.2 acquisition manifests.  The manifests' preferred GSI historical
5/10 m DEM is BLOCKED (free user registration required; account creation is not authorised), so M uses the manifest's
global source SRC_GLOBAL_COP30, whose per-tile metadata documents TanDEM-X acquisition 2011-02 .. 2014-08 (pre-event for
all three landscapes).  E is not downloaded (allowlist proposal only).
"""
from __future__ import annotations

DATE = "2026-10-05"
GSI_TERMS = "GSI content terms: free use including redistribution with source attribution"
COP_LICENCE = "Copernicus DEM GLO-30 licence (free use with attribution; see INFO/eula_F.pdf in each tile)"
COP_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
EVENT_DATES = {"KUMAMOTO_ASO_2016": "2016-04-14/2016-04-16", "HOKKAIDO_IBURI_2018": "2018-09-06", "NOTO_PENINSULA_2024": "2024-01-01"}
ORDER = ["KUMAMOTO_ASO_2016", "HOKKAIDO_IBURI_2018", "NOTO_PENINSULA_2024"]

# COP30 1x1 degree tiles intersecting each landscape's official interpretation range (bounds verified from the acquired Omega)
COP_TILES = {
    "KUMAMOTO_ASO_2016": ["N32_00_E130_00", "N32_00_E131_00", "N33_00_E130_00", "N33_00_E131_00"],
    "HOKKAIDO_IBURI_2018": ["N42_00_E141_00", "N42_00_E142_00"],  # range bounds 141.683-142.125E, 42.542-42.953N
    "NOTO_PENINSULA_2024": ["N37_00_E136_00", "N37_00_E137_00"],
}
COP_PARTS = [  # (suffix, role, relative path template, note)
    ("DEM", "DEM", "Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif", "Elevation (EGM2008 heights, WGS84)"),
    ("XML", "DEM_METADATA", "Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_10_{t}.xml", "Tile metadata incl. TanDEM-X start/stop times"),
    ("SRC", "DEM_METADATA", "Copernicus_DSM_COG_10_{t}_DEM/PREVIEW/Copernicus_DSM_10_{t}_SRC.kml", "Per-scene acquisition dates"),
    ("FLM", "DEM_QUALITY", "Copernicus_DSM_COG_10_{t}_DEM/AUXFILES/Copernicus_DSM_COG_10_{t}_FLM.tif", "Filling mask (non-TanDEM-X infill sources)"),
    ("EDM", "DEM_QUALITY", "Copernicus_DSM_COG_10_{t}_DEM/AUXFILES/Copernicus_DSM_COG_10_{t}_EDM.tif", "Editing mask"),
    ("WBM", "DEM_QUALITY", "Copernicus_DSM_COG_10_{t}_DEM/AUXFILES/Copernicus_DSM_COG_10_{t}_WBM.tif", "Water-body mask"),
]


def _e(acq, lid, sid, role, name, fn, src, url, pub, lic, efmt, esize, temporal, spatial, res, notes):
    return dict(acquisition_id=acq, landscape_id=lid, source_id=sid, data_role=role, dataset_name=name, original_filename=fn,
                source_url=src, download_url=url, publisher=pub, access_date=DATE, licence=lic, expected_format=efmt, expected_size=esize,
                temporal_coverage=temporal, spatial_coverage=spatial, resolution=res, verification_status="DOWNLOADED_PENDING_QC", notes=notes)


GSI = "Geospatial Information Authority of Japan (GSI)"
NOTO_PAGE = "https://www.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html"
NOTO_BASE = "https://www1.gsi.go.jp/geowww/saigai/20240102noto/earthquake/"

FIXED = [
    # ---------------- KUMAMOTO_ASO_2016
    _e("ACQ_KUMAMOTO_Y_OMEGA", "KUMAMOTO_ASO_2016", "S2_KUMAMOTO_GSI", "Y|OMEGA", "2016 Kumamoto aerial-photo landslide distribution", "000143456.zip",
       "https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html", "https://www.gsi.go.jp/common/000143456.zip", GSI,
       "GSI content terms; attribution required", "ZIP/KML", "48.0 KB", "2016-04-16 to 2016-07-24 post-event imagery",
       "Published interpretation ranges, Aso and Kumamoto districts", "Mapped occurrence centres", "Stage-2A-extension acquisition manifest row"),
    _e("ACQ_KUMAMOTO_DOC_GSI_PAGE", "KUMAMOTO_ASO_2016", "S2_KUMAMOTO_GSI", "SOURCE_DOC", "GSI Kumamoto earthquake index page", "gsi_kumamoto_2016_index.html",
       "https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html", "https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html", GSI,
       GSI_TERMS, "HTML", "UNKNOWN", "Page as accessed", "N/A", "N/A", "Landing-page provenance snapshot"),
    _e("ACQ_KUMAMOTO_G_H", "KUMAMOTO_ASO_2016", "S2_ASO_GEOLOGY", "GEOLOGY|AGE|H_CONTROL", "Geological Map of Aso Volcano 1:50,000",
       "GSJ_MAP_VOLC_04_1985_V01.zip", "https://gbank.gsj.jp/volcano/Act_Vol/aso/index-e.html",
       "https://www.gsj.jp/data/VOLC/VCT/GSJ_MAP_VOLC_04_1985_V01.zip", "Geological Survey of Japan (AIST)", "GSJ/AIST content terms; confirm in archive",
       "ZIP/Shapefile/KML", "14.2 MB", "1985 publication", "Aso Volcano", "1:50,000", "Stage-2A-extension acquisition manifest row"),
    _e("ACQ_KUMAMOTO_DOC_GSJ_PAGE", "KUMAMOTO_ASO_2016", "S2_ASO_GEOLOGY", "SOURCE_DOC", "GSJ Aso volcano geological map page", "gsj_aso_index_e.html",
       "https://gbank.gsj.jp/volcano/Act_Vol/aso/index-e.html", "https://gbank.gsj.jp/volcano/Act_Vol/aso/index-e.html",
       "Geological Survey of Japan (AIST)", "GSJ/AIST content terms", "HTML", "UNKNOWN", "Page as accessed", "N/A", "N/A", "Landing-page provenance snapshot"),
    # ---------------- HOKKAIDO_IBURI_2018
    _e("ACQ_HOKKAIDO_Y_OMEGA", "HOKKAIDO_IBURI_2018", "S2_HOKKAIDO_GSI", "Y|OMEGA", "Hokkaido Iburi slope-failure/deposition distribution D1-No.991",
       "000204728.zip", "https://www.gsi.go.jp/BOUSAI/H30-hokkaidoiburi-east-earthquake-index.html", "https://www.gsi.go.jp/common/000204728.zip", GSI,
       "GSI content terms; attribution required", "ZIP/GeoJSON", "1.1 MB", "2018-09-06 to 2018-09-11 post-event aerial photography",
       "Published D1-No.991 interpretation extent around Atsuma", "Mapped combined failure/deposition polygons", "Stage-2A-extension acquisition manifest row"),
    _e("ACQ_HOKKAIDO_DOC_GSI_PAGE", "HOKKAIDO_IBURI_2018", "S2_HOKKAIDO_GSI", "SOURCE_DOC", "GSI 2018 Hokkaido Eastern Iburi earthquake index page",
       "gsi_hokkaido_2018_index.html", "https://www.gsi.go.jp/BOUSAI/H30-hokkaidoiburi-east-earthquake-index.html",
       "https://www.gsi.go.jp/BOUSAI/H30-hokkaidoiburi-east-earthquake-index.html", GSI, GSI_TERMS, "HTML", "UNKNOWN", "Page as accessed", "N/A", "N/A",
       "Landing-page provenance snapshot"),
    # ---------------- NOTO_PENINSULA_2024
    *[_e(f"ACQ_NOTO_Y_OMEGA_{k}", "NOTO_PENINSULA_2024", "S2A2_NOTO_GSI", "Y|OMEGA", "GSI 2024 Noto slope-failure/deposition GeoJSON with interpretation range",
         fn, NOTO_PAGE, NOTO_BASE + fn, GSI, "GSI content terms; attribution required", "ZIP/GeoJSON", size, "Orthophotos 2-17 Jan 2024",
         "Published district interpretation range", "Features >= ~100 m2", "Stage-2A.2 acquisition manifest row; screening hash in transient_inspection_manifest.json")
      for k, fn, size in [("SUZU_WAJIMA_EAST", "noto_suzu-wajimaEAST_20240106.zip", "212 KB"), ("WAJIMA_CENTRAL", "noto_wajimaCENTRAL_20240115.zip", "129 KB"),
                          ("WAJIMA_WEST", "noto_wajimaWEST_20240122.zip", "152 KB"), ("ANAMIZU", "noto_anamizu_20240118.zip", "47 KB"),
                          ("NANAO", "noto_nanao_20240118.zip", "32 KB")]],
    _e("ACQ_NOTO_DOC_GSI_PAGE", "NOTO_PENINSULA_2024", "S2A2_NOTO_GSI", "SOURCE_DOC", "GSI 2024 Noto earthquake page", "gsi_noto_2024_index.html",
       NOTO_PAGE, NOTO_PAGE, GSI, GSI_TERMS, "HTML", "UNKNOWN", "Page as accessed", "N/A", "N/A", "Landing-page provenance snapshot"),
]


def cop_entries(lid: str) -> list[dict]:
    out = []
    for t in COP_TILES[lid]:
        for suffix, role, tmpl, note in COP_PARTS:
            path = tmpl.format(t=t)
            out.append(_e(f"ACQ_{lid.split('_')[0]}_COP30_{t.replace('_00', '')}_{suffix}", lid, "SRC_GLOBAL_COP30", role,
                          f"Copernicus DEM GLO-30 tile {t} ({suffix})", path.split("/")[-1], f"{COP_BASE}/{path.split('/')[0]}/", f"{COP_BASE}/{path}",
                          "ESA / Airbus (Copernicus programme); AWS Open Data registry", COP_LICENCE, "GeoTIFF" if path.endswith(".tif") else "XML/KML",
                          "UNKNOWN", "TanDEM-X acquisitions 2011-2015 (verified per tile from metadata)", f"1x1 degree tile {t}", "1 arcsec (~30 m)", note))
    out.append(_e(f"ACQ_{lid.split('_')[0]}_COP30_EULA", lid, "SRC_GLOBAL_COP30", "SOURCE_DOC", "Copernicus DEM licence (EULA)", "eula_F.pdf",
                  f"{COP_BASE}/", f"{COP_BASE}/Copernicus_DSM_COG_10_{COP_TILES[lid][0]}_DEM/INFO/eula_F.pdf", "ESA / Airbus (Copernicus programme)",
                  COP_LICENCE, "PDF", "117522 bytes", "N/A", "N/A", "N/A", "Licence document shipped with every tile")) if COP_TILES[lid] else None
    return out


def plan(lid: str) -> list[dict]:
    return [e for e in FIXED if e["landscape_id"] == lid] + cop_entries(lid)
