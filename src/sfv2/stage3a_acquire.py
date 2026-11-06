"""Stage 3A public acquisitions: only small, attributable, reusable artifacts that decide a named gate."""
from __future__ import annotations

from .raw_store import acquire, load_registry, save_registry

AS_OF = "2026-10-06"
ACQUISITIONS = [
    dict(acquisition_id="ACQ_PUERTO_RICO_GEOLOGY_USGS_OFR98_38", landscape_id="PUERTO_RICO_MARIA_2017", source_id="SRC_USGS_OFR_98_38_PR_GEOLOGY",
         data_role="G", dataset_name="Bawiec (1998) USGS OFR 98-38 digital geology of Puerto Rico (units + faults)", original_filename="ofr-98-38.zip",
         source_url="https://doi.org/10.3133/ofr9838", download_url="https://mrdata.usgs.gov/geology/pr/ofr-98-38.zip", doi="10.3133/ofr9838",
         publisher="U.S. Geological Survey", access_date=AS_OF, licence="US Government public domain", expected_format="ZIP", expected_size="10780956",
         temporal_coverage="1998 compilation", spatial_coverage="Puerto Rico", resolution="1:100,000 compilation", verification_status="DOWNLOADED_PUBLIC",
         gate_affected="GEOLOGY|VOLCANIC_DOMAIN",
         notes="Needed to quantify the Cenozoic volcanic subset under the frozen DOMAIN_RULE (pre-Cenozoic volcanic rocks are basement)"),
    dict(acquisition_id="ACQ_KALEHE_UNOSAT_FL20230508COD_SHP", landscape_id="KIVU_KALEHE_2023", source_id="SRC_UNOSAT_FL20230508COD",
         data_role="Y|OMEGA_CANDIDATE", dataset_name="UNOSAT FL20230508COD landslide impact analysis, Kalehe Territory, Sentinel-2 7 May 2023",
         original_filename="FL20230508COD_SHP.zip", source_url="https://data.humdata.org/dataset/landslide-impact-analysis-in-kalehe-territory-sud-kivu-province-democratic-republic-of-the",
         download_url="https://unosat.org/static/unosat_filesystem/3582/FL20230508COD_SHP.zip", doi="",
         publisher="UN Operational Satellite Applications Programme (UNOSAT)", access_date=AS_OF, licence="CC BY-SA (HDX record)", expected_format="ZIP",
         expected_size="925841", temporal_coverage="Sentinel-2 2023-05-07 (event 2023-05-04)", spatial_coverage="Kalehe Territory, South Kivu, DR Congo",
         resolution="Sentinel-2 10 m", verification_status="DOWNLOADED_PUBLIC", gate_affected="Y|OMEGA",
         notes="Preliminary, not field-validated; HDX states an analysed area of about 9,000 ha with 90 ha of landslide scars"),
    dict(acquisition_id="ACQ_TALAKMAU_UNOSAT_LS20220308IDN_SHP", landscape_id="TALAKMAU_PASAMAN_2022", source_id="SRC_UNOSAT_LS20220308IDN",
         data_role="Y|OMEGA_CANDIDATE", dataset_name="UNOSAT LS20220308IDN landslide analysis, Mount Talakmau, Sentinel-2 4 Mar 2022",
         original_filename="LS20220308IDN_SHP.zip",
         source_url="https://data.humdata.org/dataset/landslide-analysis-in-mount-talakmau-in-pasaman-pasaman-barat-districts-indonesia-as-of-04",
         download_url="https://unosat-maps.web.cern.ch/ID/LS20220308IDN/LS20220308IDN_SHP.zip", doi="",
         publisher="UN Operational Satellite Applications Programme (UNOSAT)", access_date=AS_OF, licence="CC BY-SA (HDX record)", expected_format="ZIP",
         expected_size="150958", temporal_coverage="Sentinel-2 2022-03-04 (Mw 6.1 earthquake 2022-02-25)", spatial_coverage="Mount Talakmau, West Sumatra",
         resolution="Sentinel-2 10 m", verification_status="DOWNLOADED_PUBLIC", gate_affected="Y|OMEGA",
         notes="Preliminary, not field-validated; HDX states about 6 km2 of landslides within the analysed area"),
]


def acquire_all() -> list[dict]:
    registry = load_registry()
    out = [acquire(dict(a), registry) for a in ACQUISITIONS]
    save_registry(registry)
    return out
