"""Stage 3B public acquisitions: Copernicus GLO-30 tiles (DEM + epoch/quality layers) for Talakmau and Puerto Rico."""
from __future__ import annotations

import subprocess
import time

from .raw_store import acquire, load_registry, save_registry

AS_OF = "2026-10-06"
COP_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
TILES = {
    "TALAKMAU_PASAMAN_2022": ["N00_00_E099_00", "N00_00_E100_00", "S01_00_E099_00", "S01_00_E100_00"],
    "PUERTO_RICO_MARIA_2017": ["N17_00_W068_00", "N17_00_W067_00", "N17_00_W066_00", "N18_00_W068_00", "N18_00_W067_00", "N18_00_W066_00"],
}
SHORT = {"TALAKMAU_PASAMAN_2022": "TALAKMAU", "PUERTO_RICO_MARIA_2017": "PR"}
LAYERS = [("DEM", "{t}_DEM.tif", "M"), ("XML", "{x}.xml", "DEM_METADATA"), ("SRC", "PREVIEW/{x}_SRC.kml", "DEM_METADATA"),
          ("EDM", "AUXFILES/{t}_EDM.tif", "DEM_QUALITY"), ("FLM", "AUXFILES/{t}_FLM.tif", "DEM_QUALITY"), ("WBM", "AUXFILES/{t}_WBM.tif", "DEM_QUALITY")]


def acq_id(lid: str, tile: str, layer: str) -> str:
    return f"ACQ_{SHORT[lid]}_COP30_{tile[:3]}_{tile[7:11]}_{layer}"


def acquire_cop30() -> list[dict]:
    registry, out = load_registry(), []
    for lid, tiles in TILES.items():
        for tile in tiles:
            t, x = f"Copernicus_DSM_COG_10_{tile}", f"Copernicus_DSM_10_{tile}"
            folder = f"{COP_BASE}/{t}_DEM"
            for layer, pattern, role in LAYERS:
                rel = pattern.format(t=t, x=x)
                entry = dict(
                    acquisition_id=acq_id(lid, tile, layer), landscape_id=lid, source_id="SRC_GLOBAL_COP30", data_role=role,
                    dataset_name=f"Copernicus DEM GLO-30 tile {tile} ({layer})", original_filename=rel.split("/")[-1],
                    source_url=f"{folder}/", download_url=f"{folder}/{rel}", doi="", publisher="ESA / Airbus (Copernicus); AWS Open Data",
                    access_date=AS_OF, licence="Copernicus DEM GLO-30 licence (free, attribution)", expected_format="GeoTIFF" if rel.endswith(".tif") else "XML/KML",
                    expected_size="UNKNOWN", temporal_coverage="TanDEM-X acquisitions (verify per tile from XML/SRC)", spatial_coverage=f"Tile {tile}",
                    resolution="1 arcsec", verification_status="DOWNLOADED_PENDING_QC", gate_affected="M|M_TEMPORAL|DEM_COVERAGE",
                    notes="Pre-event M candidate; quality layers per Copernicus DEM Product Handbook v5.0 Tables 6-7")
                out.append(_acquire_retry(entry, registry))
                save_registry(registry)  # persist after every file so an interrupted run never loses provenance
    return out


def verify_cop30() -> list[dict]:
    """Offline re-verification of every registered tile layer (hash of the immutable raw file == registry)."""
    from .raw_store import sha256
    from .schema import ROOT
    reg = {e["acquisition_id"]: e for e in load_registry()}
    out = []
    for lid, tiles in TILES.items():
        for tile in tiles:
            for layer, _, _ in LAYERS:
                e = reg[acq_id(lid, tile, layer)]
                if sha256(ROOT / e["raw_path"]) != e["sha256"]:
                    raise RuntimeError(f"STOP: raw hash mismatch for {e['acquisition_id']}")
                out.append(e)
    return out


def _acquire_retry(entry: dict, registry: list[dict], attempts: int = 4) -> dict:
    for i in range(attempts):
        try:
            return acquire(entry, registry)
        except subprocess.CalledProcessError:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


# Atitlan geology route (Stage 2F/3A): both MAGA PDFs exist (HTTP 200 to a browser; 52,361,973 and 94,660,888 bytes on 2026-10-06) but the
# server returns HTTP 403 to automated clients.  Browser user-agent spoofing is NOT used, so these remain a browser-only human action.
ATITLAN_DOCS = [
    dict(url="https://www.maga.gob.gt/wp-content/uploads/pdf/suelos/solola1.pdf", expected_bytes=52361973, filename="solola1.pdf"),
    dict(url="https://www.maga.gob.gt/download/suelos-solola1.pdf", expected_bytes=94660888, filename="suelos-solola1.pdf"),
]
