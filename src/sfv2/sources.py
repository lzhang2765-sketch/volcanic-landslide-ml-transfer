"""Stage 1 source manifest: candidate-derived Y rows plus explicitly recorded extra and global sources.

Discovery only - nothing listed here was downloaded in Stage 1.  'verified' = VERIFIED / LIKELY / UNKNOWN
(VERIFIED means the record/landing page was seen in Stage 1 or in frozen V1 records).
"""

from __future__ import annotations

from .discovery_records import DATE, RECORDS

U = "UNKNOWN"


def _row(source_id, landscape_id, data_role, title, org, stype, url, **kw):
    r = dict(source_id=source_id, landscape_id=landscape_id, data_role=data_role, source_title=title, source_organisation=org, source_type=stype,
             citation=U, doi=U, url=url, access_date=DATE, downloadable=U, file_format=U, licence=U, spatial_coverage=U, temporal_coverage=U,
             resolution_or_scale=U, tier_relevance=U, omega_obs_evidence=U, h_relevance=U, verified=U, notes="Discovery only; not downloaded in Stage 1")
    r.update(kw)
    return r


def _is_link(v: str) -> bool:
    return v.startswith("http") or v.startswith("10.")


def candidate_y_sources() -> list[dict]:
    out = []
    audited_notes = {
        "MATZACO_PUEBLA_2017": "Omega evidence links to the frozen V1 downloaded and hashed USGS mapping_extent shapefile",
        "PUERTO_RICO_MARIA_2017": "Frozen V1 registry documents approximately 97% main-island imagery coverage and mapping boundaries",
        "COLUMBIA_RIVER_GORGE_BASALT": "RI-45 documents a published 900 square-mile study area; digital boundary remains a Stage-2 check",
    }
    for rec in RECORDS:
        url = rec["inventory_url_or_identifier"]
        if not _is_link(url):
            continue
        stype = "PAPER" if rec["y_evidence_stage"] == "PAPER_EXISTS" else "DATASET"
        out.append(_row(f"SRC_Y_{rec['landscape_id']}", rec["landscape_id"], "Y", rec["inventory_name"], rec["inventory_source"], stype, url,
                        citation=rec["primary_reference"], doi=rec["doi"], downloadable=rec["inventory_downloadable"], licence=rec["licence"],
                        spatial_coverage=rec["region"], temporal_coverage=rec["inventory_year"], tier_relevance=rec["inventory_tier"],
                        omega_obs_evidence=rec["observation_domain_source"], verified=rec["y_verification"],
                        notes=audited_notes.get(rec["landscape_id"], "Discovery only; not downloaded in Stage 1")))
    return out


def candidate_omega_sources() -> list[dict]:
    """Make the existing observation-domain evidence explicit by role.

    These records do not add evidence. They point to the same existing
    publication/dataset as the candidate record and identify only its Omega
    contribution.
    """
    out = []
    for rec in RECORDS:
        if rec["omega_gate"] != "PASS":
            continue
        url = rec["inventory_url_or_identifier"]
        if not _is_link(url):
            continue
        out.append(_row(
            f"SRC_OMEGA_{rec['landscape_id']}", rec["landscape_id"], "OMEGA",
            f"Observation-domain evidence for {rec['inventory_name']}", rec["inventory_source"],
            "EXISTING_PUBLICATION_OR_DATASET", url, citation=rec["primary_reference"], doi=rec["doi"],
            downloadable=rec["inventory_downloadable"], licence=rec["licence"], spatial_coverage=rec["region"],
            temporal_coverage=rec["inventory_year"], omega_obs_evidence=rec["observation_domain_source"],
            verified=rec["omega_verification"],
            notes=f"Role-specific link to existing evidence; {rec['observation_domain_type']}; {rec['observation_domain_notes']}",
        ))
    return out


EXTRA = [
    _row("SRC_Y_NICARAGUA_MITCH_FILES", "NICARAGUA_MITCH_1998", "Y", "OFR 01-412-A digital files (shapefiles by department)", "USGS", "DATASET",
         "http://pubs.usgs.gov/of/2001/ofr-01-0412-a/digital_files/", downloadable="TRUE", file_format="Shapefile", resolution_or_scale="1:50,000 (~200 m accuracy)",
         tier_relevance="BOTH_AVAILABLE", verified="LIKELY"),
    _row("SRC_OMEGA_DOMINICA_CLOUD", "DOMINICA_MARIA_2017", "OMEGA", "Cloud.shp unmappable-area mask (within Zenodo 15224820)", "University of Twente", "DATASET",
         "https://doi.org/10.5281/zenodo.15224820", doi="10.5281/zenodo.15224820", downloadable="FALSE", file_format="Shapefile", licence="CC BY 4.0",
         omega_obs_evidence="Explicit cloud mask of unmappable terrain", verified="VERIFIED", notes="Restricted files; listed in record metadata"),
    _row("SRC_Y_DOMINICA_RASTER", "DOMINICA_MARIA_2017", "Y", "Dominica landslide raster (open record)", "University of Twente", "DATASET",
         "https://zenodo.org/records/15182981", downloadable="TRUE", file_format="Raster", verified="LIKELY", notes="Semantics (tiers, domain) undocumented"),
    _row("SRC_GEOLOGY_GSJ_SEAMLESS", "KURIKOMA_2008", "GEOLOGY", "Seamless Digital Geological Map of Japan V2", "Geological Survey of Japan (AIST)", "PORTAL",
         "https://gbank.gsj.jp/seamless/", downloadable="TRUE", resolution_or_scale="1:200,000", h_relevance="Unit ages for time_since_emplacement", verified="VERIFIED",
         notes="Also applies to HOKKAIDO_IBURI_2018, KUMAMOTO_ASO_2016, ASO_2012_RAIN, IZU_OSHIMA_2013"),
    _row("SRC_GEOLOGY_NSW_SEAMLESS", "WILSONS_RIVER_NSW_2022", "GEOLOGY", "NSW Seamless Geology dataset", "Geological Survey of NSW", "PORTAL",
         "https://www.regional.nsw.gov.au/meg/geoscience/products-and-data/gis/seamless-geology", verified="LIKELY",
         h_relevance="Lamington Volcanics unit boundaries (volcanic masking)", notes="URL path to be confirmed at acquisition"),
    _row("SRC_DEM_NSW_ELVIS", "WILSONS_RIVER_NSW_2022", "DEM", "ELVIS elevation data portal (NSW lidar)", "Geoscience Australia / ICSM", "PORTAL",
         "https://elevation.fsdf.org.au/", resolution_or_scale="1-5 m", verified="LIKELY", notes="Pre-2022 epoch must be confirmed"),
    _row("SRC_GEOLOGY_GNS_QMAP", "BANKS_PENINSULA_2021", "GEOLOGY", "QMAP 1:250,000 Geological Map of New Zealand", "GNS Science", "PORTAL",
         "https://data.gns.cri.nz/", resolution_or_scale="1:250,000", verified="LIKELY", notes="Also applies to COROMANDEL_GABRIELLE_2023"),
    _row("SRC_DEM_OREGON_LIDAR", "OREGON_CASCADES_SLIDO", "DEM", "Oregon Lidar Consortium data", "DOGAMI", "PORTAL",
         "https://www.oregongeology.org/lidar/", resolution_or_scale="~1 m", verified="LIKELY"),
    _row("SRC_DEM_PUERTO_RICO_PRE_EVENT", "PUERTO_RICO_MARIA_2017", "DEM", "USGS 3DEP lidar for Puerto Rico", "USGS", "DATASET",
         "https://apps.nationalmap.gov/downloader/", downloadable="TRUE", resolution_or_scale="1 m", temporal_coverage="2015-2016 acquisition; pre-Hurricane Maria",
         verified="VERIFIED", notes="Existing frozen V1 SF-1 M registry evidence; no download performed in V2 Stage 1"),
    _row("SRC_GEOLOGY_ISPRA_CARG", "IFFI_ITALIAN_VOLCANIC_DISTRICTS", "GEOLOGY", "CARG geological map of Italy 1:50,000", "ISPRA", "PORTAL",
         "https://www.isprambiente.gov.it/it/progetti/cartella-progetti-in-corso/suolo-e-territorio-1/progetto-carg-cartografia-geologica-e-geotematica",
         resolution_or_scale="1:50,000 (partial coverage)", verified="LIKELY", notes="Also applies to CAMPANIA_SARNO_1998, ISCHIA"),
]

GLOBAL = [
    _row("SRC_GLOBAL_COP30", "GLOBAL", "DEM", "Copernicus DEM GLO-30", "ESA / Airbus", "DATASET", "https://doi.org/10.5270/ESA-c5d3d65",
         downloadable="TRUE", resolution_or_scale="30 m", temporal_coverage="2011-2015 acquisition", verified="VERIFIED", notes="Used in V1; epoch must be checked per event"),
    _row("SRC_GLOBAL_NASADEM", "GLOBAL", "DEM", "NASADEM", "NASA JPL", "DATASET", "https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001",
         downloadable="TRUE", resolution_or_scale="30 m", temporal_coverage="Feb 2000", verified="VERIFIED", notes="Used in V1; pre-event for post-2000 events"),
    _row("SRC_GLOBAL_CHELSA", "GLOBAL", "ENVIRONMENT", "CHELSA climatologies", "WSL", "DATASET", "https://chelsa-climate.org/", downloadable="TRUE",
         resolution_or_scale="~1 km", verified="LIKELY"),
    _row("SRC_GLOBAL_WORLDCLIM", "GLOBAL", "ENVIRONMENT", "WorldClim 2.1", "WorldClim", "DATASET", "https://www.worldclim.org/", downloadable="TRUE",
         resolution_or_scale="~1 km", verified="LIKELY"),
    _row("SRC_GLOBAL_IMERG", "GLOBAL", "ENVIRONMENT", "GPM IMERG precipitation", "NASA GES DISC", "DATASET", "https://gpm.nasa.gov/data/imerg",
         downloadable="TRUE", temporal_coverage="2000-present", resolution_or_scale="0.1 deg", verified="LIKELY", notes="Login (Earthdata) required"),
    _row("SRC_GLOBAL_SHAKEMAP", "GLOBAL", "ENVIRONMENT", "USGS ShakeMap", "USGS", "DATASET", "https://earthquake.usgs.gov/data/shakemap/", downloadable="TRUE",
         verified="LIKELY", notes="Earthquake trigger intensity"),
    _row("SRC_GLOBAL_GLIM", "GLOBAL", "GEOLOGY", "GLiM Global Lithological Map", "Hartmann & Moosdorf (2012)", "DATASET", "https://doi.org/10.1029/2012GC004370",
         doi="10.1029/2012GC004370", resolution_or_scale="~1:3,750,000", verified="LIKELY", notes="Coarse fallback only"),
    _row("SRC_GLOBAL_GVP", "GLOBAL", "AGE", "Global Volcanism Program Volcanoes of the World", "Smithsonian Institution", "DATASET", "https://volcano.si.edu/",
         downloadable="TRUE", h_relevance="Vent locations / eruption ages for H controls", verified="VERIFIED", notes="Used in V1 SF-3"),
    _row("SRC_CAT_UGLC", "GLOBAL", "DISCOVERY_CATALOGUE", "Unified Global Landslide Catalogue (UGLC)", "Multiple (compilation)", "DATASET",
         "https://doi.org/10.5281/zenodo.18643456", doi="10.5281/zenodo.18643456", licence="CC BY-NC 4.0", verified="VERIFIED",
         notes="984,126 polygons compiled; constituent inventories must be traced to originals"),
    _row("SRC_CAT_USGS_EQ_REPO", "GLOBAL", "DISCOVERY_CATALOGUE", "USGS Earthquake-triggered ground-failure inventories repository", "USGS", "PORTAL",
         "https://www.sciencebase.gov/catalog/item/586d824ce4b0f5ce109fc9a6", downloadable="TRUE", verified="VERIFIED", notes="49 event child items"),
    _row("SRC_CAT_GLOBAL_EQ_DB", "GLOBAL", "DISCOVERY_CATALOGUE", "Global earthquake-induced landslide database 2016-2025", "Fang, Fan, Tanyas, van Westen", "DATASET",
         "https://zenodo.org/records/20157141", downloadable="FALSE", licence="CC BY 4.0", verified="VERIFIED", notes="Files restricted (login); 24 mapped + 42 no-landslide events"),
    _row("SRC_CAT_NASA_COOLR", "GLOBAL", "DISCOVERY_CATALOGUE", "NASA Cooperative Open Online Landslide Repository (COOLR)", "NASA GSFC", "DATASET",
         "https://gpm.nasa.gov/landslides/data.html", verified="LIKELY", notes="Mostly reported-event points: discovery only, never Omega"),
    _row("SRC_CAT_US_LS_V3", "GLOBAL", "DISCOVERY_CATALOGUE", "U.S. Landslide Inventory (v3)", "USGS", "DATASET", "https://www.usgs.gov/programs/landslide-hazards/science/us-landslide-inventory",
         verified="LIKELY", notes="Compilation; trace to state inventories (SLIDO, WASLID)"),
]


def all_sources() -> list[dict]:
    return candidate_y_sources() + candidate_omega_sources() + EXTRA + GLOBAL
