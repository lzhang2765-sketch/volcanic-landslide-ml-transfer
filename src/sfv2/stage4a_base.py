"""Stage 4A base: authoritative six-landscape production tables (read-only) and the development/lockbox firewall."""
from __future__ import annotations

import pandas as pd

from .schema import ROOT

PROC = ROOT / "data/processed"
DEVELOPMENT = {  # landscape -> production directory (frozen; read only)
    "KUMAMOTO_ASO_2016": "stage2d", "NOTO_PENINSULA_2024": "stage2d", "LOMBOK_RINJANI_2018": "stage2f2_production",
    "COTABATO_DAVAO_2019": "stage3d_production", "ATITLAN_STAN_2005": "stage3f_production", "DOMINICA_ERIKA_2015": "stage3f_production",
}
PROTECTED_LOCKBOX_CANDIDATES = ["VANUATU_HAROLD_2020", "POPOCATEPETL_PUEBLA_2017", "CHUUK_CHATAAN_2002", "MOUNT_ELGON_COMPLETE_EVENT_IF_DESIGNATED"]
EXPECTED = dict(rows=835846, positives=8793, negatives=827053)
EVENT_DATE = {"KUMAMOTO_ASO_2016": "2016-04-16", "NOTO_PENINSULA_2024": "2024-01-01", "LOMBOK_RINJANI_2018": "2018-07-29",
              "COTABATO_DAVAO_2019": "2019-10-16", "ATITLAN_STAN_2005": "2005-10-04", "DOMINICA_ERIKA_2015": "2015-08-27"}
DEM_SOURCE = {"ATITLAN_STAN_2005": "NASADEM_2000_NO_QUALITY_COMPANION"}  # all others: Copernicus GLO-30 with EDM/FLM


def firewall_check(lid: str) -> None:
    if lid not in DEVELOPMENT or lid in PROTECTED_LOCKBOX_CANDIDATES:
        raise PermissionError(f"FIREWALL: {lid} is not a Stage 4A development landscape")


def load(lid: str, table: str, **kw) -> pd.DataFrame:
    firewall_check(lid)
    return pd.read_csv(PROC / DEVELOPMENT[lid] / lid / f"{table}.csv.gz", **kw)


def path(lid: str, rel: str):
    firewall_check(lid)
    return PROC / DEVELOPMENT[lid] / lid / rel
