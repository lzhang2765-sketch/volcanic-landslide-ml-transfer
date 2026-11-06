"""Stage 2C design/governance records: final E allowlist, terrain-unit decision, feature-engine spec, H feasibility,
Kumamoto trigger decision, V1 non-transfer note, and (unsent) external request packages."""
from __future__ import annotations

# ------------------------------------------------------------------------------------------------- E allowlist (FINAL, not downloaded)
# (variable, physical meaning, unit, dataset/version, resolution, period, licence, relevance, leakage, missing-data policy, status)
E_FINAL = [
    ("E_MAT", "Mean annual air temperature", "deg C", "CHELSA V2.1 bio1", "30 arcsec (~1 km)", "1981-2010 climatology", "VERIFY_AT_FREEZE",
     "Weathering rate, freeze-thaw regime", "LOW (climatology)", "Nearest valid 1 km cell within 2 km; else landscape excluded from E block", "CORE"),
    ("E_TSEAS", "Temperature seasonality (SD x100)", "deg C x100", "CHELSA V2.1 bio4", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE",
     "Thermal regime contrast across climates", "LOW", "As E_MAT", "CORE"),
    ("E_MAP", "Mean annual precipitation", "mm/yr", "CHELSA V2.1 bio12", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE",
     "Long-term wetness, regolith development", "LOW", "As E_MAT", "CORE"),
    ("E_PWET", "Precipitation of wettest month", "mm/month", "CHELSA V2.1 bio13", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE",
     "Wet-season loading", "LOW", "As E_MAT", "CORE"),
    ("E_PSEAS", "Precipitation seasonality (CV)", "%", "CHELSA V2.1 bio15", "30 arcsec", "1981-2010", "VERIFY_AT_FREEZE",
     "Monsoonal vs. uniform regimes", "LOW", "As E_MAT", "CORE"),
    ("E_TREE2000", "Tree canopy cover in 2000", "% canopy", "Hansen GFC v1.11 treecover2000", "1 arcsec (~30 m)", "Year 2000", "CC BY 4.0",
     "Root reinforcement / land-use proxy", "LOW for leakage (pre-event); STALENESS caveat 16-24 yr (Noto 24 yr)", "Missing -> NaN + indicator; no imputation from Y",
     "CORE_WITH_STALENESS_FLAG"),
    ("E_LC_PRE", "Land-cover class in the year before each event", "categorical (LCCS)", "ESA CCI / C3S Land Cover v2.0.7 / v2.1.1", "300 m",
     "Event year minus 1", "VERIFY_AT_FREEZE", "Land use", "MODERATE unless strictly pre-event; ESA WorldCover excluded (post-event for 2016/2018)",
     "Categorical nearest; no interpolation", "OPTIONAL_SENSITIVITY"),
    ("E_SOIL_TEXT", "Clay and sand fraction 0-30 cm", "g/kg", "SoilGrids 2.0", "250 m", "Static model", "CC BY 4.0",
     "Regolith hydraulic behaviour", "MODERATE: modelled from terrain covariates (collinearity with M)", "NaN + indicator", "OPTIONAL_SENSITIVITY"),
]
E_EXCLUDED = [("ERA5-Land soil moisture", "Event-window and trigger-specific; not comparable across EQ/rain landscapes"),
              ("ESA WorldCover 2020/2021", "Post-event for Kumamoto 2016 and Hokkaido 2018 (landslide scars would be classified)"),
              ("Additional CHELSA bioclim (bio2-3, 5-11, 14, 16-19)", "Highly collinear with the retained five; no added physical meaning"),
              ("Trigger intensity (ShakeMap PGA/MMI, IMERG event rain)", "Kept as a separate TRIGGER block, never mixed into the core E block")]

# ------------------------------------------------------------------------------------------------- terrain-unit architecture
TERRAIN_UNITS = [
    ("A_REGULAR_GRID_90M", "SELECTED_PRIMARY",
     "Deterministic; aligns exactly with the 30 m canonical DEM, geology and future H rasters (3x3 cells); works for point (contains-centre) and polygon "
     "(cell-centre-in-polygon) labels; supports spatial block CV and LOLO; interpretable maps; no extra software",
     "Units cut across hillslopes; 90 m is coarser than many Noto failures (median ~0.4 ha) -> retain 30 m cell-level sensitivity"),
    ("B_SLOPE_UNITS", "DEFERRED_SENSITIVITY",
     "Geomorphologically meaningful; standard in landslide susceptibility",
     "Requires r.slopeunits (GRASS GIS) which is not installed (installation needs authorisation); strongly parameter-sensitive; unit sizes vary by "
     "landscape, complicating cross-landscape comparison and H raster integration"),
    ("C_HYBRID_GRID_IN_SLOPE_UNITS", "DEFERRED",
     "Grid rows carrying slope-unit context features", "Inherits the slope-unit dependency; revisit after B is available"),
    ("D_REGULAR_GRID_30M", "SENSITIVITY",
     "Native resolution; maximal detail", "Strong autocorrelation; Kumamoto point labels too sparse at 30 m (one centre per >0.1 ha failure)"),
]
LABEL_RULES = {
    "POLYGON_AFFECTED_AREA": "Unit positive if >=1 of its nine 30 m cell centres lies inside a Tier-2 affected-area polygon; unit excluded unless >=5/9 cells are in OMEGA_ANALYSIS",
    "POINT_OCCURRENCE_CENTRE": "Unit positive if it contains >=1 mapped occurrence centre of the selected target set (point-event formulation; no buffering); "
                               "this is a distinct target family from polygon affected-area labels and is never pooled with it",
}

# ------------------------------------------------------------------------------------------------- feature-engine specification (M)
# (feature, definition/formula, unit, scales)
FEATURES = [
    ("ELEV", "Canonical DEM elevation z", "m", "30 m"),
    ("SLOPE", "atan(sqrt(dz/dx^2 + dz/dy^2)), Horn (1981) 3x3", "deg", "30 m; also 90 m smoothed"),
    ("ASPECT_SIN / ASPECT_COS", "sin/cos of Horn aspect; undefined (flat, slope<1 deg) -> 0,0 + flat flag", "-", "30 m"),
    ("CURV_PROFILE / CURV_PLAN / CURV_TOTAL", "Zevenbergen-Thorne (1987) second derivatives along/across slope; total = Laplacian", "1/m", "30 m, 90 m"),
    ("ROUGHNESS", "SD of elevation in window", "m", "90, 270 m"),
    ("TRI", "Riley et al. (1999): mean |z_i - z_0| over 8 neighbours", "m", "30 m"),
    ("TPI", "z_0 - mean(z) in annulus", "m", "150, 450, 1500 m radii"),
    ("RELIEF", "max(z) - min(z) in circular window", "m", "250, 1000 m"),
    ("FLOW_ACC / CONTRIB_AREA", "D-infinity (Tarboton 1997) specific contributing area on depression-breached DEM", "m2/m", "30 m"),
    ("TWI", "ln(a / tan(beta)), a = specific contributing area, beta = slope (min 0.1 deg)", "-", "30 m"),
    ("SPI", "a * tan(beta)", "m", "30 m"),
    ("DIST_DRAINAGE", "Euclidean distance to channel cells (contributing area > A_c; A_c frozen per stage, default 0.1 km2)", "m", "30 m"),
    ("DIST_RIDGE", "Distance to divide cells (inverted-DEM channel network)", "m", "30 m"),
    ("VALLEY_DEPTH", "Height of cell above interpolated channel-network surface", "m", "30 m"),
    ("REL_SLOPE_POSITION", "valley_depth / (valley_depth + height_below_ridge)", "0-1", "30 m"),
    ("REL_ELEV_LOCAL", "(z - min) / (max - min) in window", "0-1", "1000 m"),
]
FEATURE_RULES = ["Features computed once on the canonical 30 m grid, then aggregated to 90 m units as mean and SD (no re-resampling of DEM)",
                 "Edge cells within the 1.5 km buffer used for neighbourhoods; rows only inside OMEGA_ANALYSIS",
                 "DEM-quality flags (Copernicus EDM/FLM) carried as row covariates/filters, never silently dropped",
                 "No feature computed from Y; no feature uses post-event data",
                 "Feature list frozen before any model fitting; no additions to raise feature count"]

# ------------------------------------------------------------------------------------------------- H feasibility (preparation only)
H_FEASIBILITY = [
    ("KUMAMOTO_ASO_2016", "CALDERA_RIM + RADIAL_FLANK (post-caldera cones)", "GSJ Aso 1:50,000 unit polygons (59 units), caldera wall morphology, "
     "Aso-4 pyroclastic-flow plateau", "MEDIUM", "Dated Aso-4 surface remnants outside caldera; pre-caldera edifice controls", "YES (ensemble over rim/plateau reconstructions)"),
    ("HOKKAIDO_IBURI_2018", "AGE_HISTORY_PROXY_ONLY (tephra mantle)", "Tephra stratigraphy (Spfa-1, En-a, Ta-d) in literature; no edifice",
     "HIGH", "Tephra isopach maps (digital) for Atsuma; not acquired", "NO (no constructional surface; H_STATUS FAIL retained)"),
    ("NOTO_PENINSULA_2024", "AGE_HISTORY_PROXY_ONLY (Oligo-Miocene volcanic pile)", "Seamless unit ages", "HIGH",
     "No preserved constructional surfaces", "NO (H not claimed)"),
    ("KURIKOMA_2008", "RADIAL_FLANK + PLATEAU (V1-proven family, to be rebuilt independently in V2)", "GSJ Seamless (transferred), DEM",
     "MEDIUM", "V2 Omega digitisation from GSI PDF first; V1 H products are NOT reused", "YES (rebuild from raw; V1 H excluded)"),
]

# ------------------------------------------------------------------------------------------------- Kumamoto trigger decision
KUMAMOTO_TRIGGER_DECISION = """# Kumamoto 2016 — trigger-mixture decision

**Decision.** The primary Kumamoto target is earthquake-only. It uses the occurrence centres GSI interpreted from the 16, 19 and
20 April 2016 photography (`PRIMARY_APRIL_EQ`), evaluated only inside the **April** interpretation range.

July-interpreted centres (photography 5-24 July 2016) are held out of the primary target. They follow the 20-21 June 2016 heavy
rain, so they may include rainfall-triggered failures. Centres with no photo date are also excluded from the primary target,
because their epoch cannot be established. Both are retained as sensitivity sets.

## Why not combine?

A combined inventory mixes earthquake and rainfall triggers with no attribute that separates them. The July interpretation range
also covers a different area and epoch than the April range. Pooling the two would let the label meaning change silently with
location.

## Sensitivity plan (for later stages; nothing is run now)

1. **S1 (primary):** APRIL_EQ centres inside the April range.
2. **S2:** S1 plus JULY centres, with the domain set to the April ∪ July ranges. This is the combined "event-period" target.
3. **S3:** S2 plus undated centres.
4. **Report:** label prevalence for each set. Once features exist, also report feature-distribution shifts between S1 and the July-only
   additions. A materially different July population supports rainfall contamination and keeps S1 as primary.
5. **Exact duplicates:** the 26 duplicate locations are dropped, keeping the first occurrence. Duplicates are flagged, never deleted
   from RAW.
"""

V1_OTHER_LANDSCAPES = """# Lombok, Atitlán and Puerto Rico — V1 raw reuse status

There is no V1-to-V2 provenance plan or authorisation for these landscapes, so they were left untouched in Stage 2C.
A future transfer would need, for each landscape:

1. A prepared transfer manifest like Kurikoma's: raw inputs only, V1 hash verification, and an exclusion list for every V1-derived,
   H, feature, ML-dataset and prediction product.
2. Explicit user authorisation.
3. V2-native Omega evidence where V1 used processing domains.
4. Eligibility flags. Lombok and Atitlán were V1 training landscapes, so they are `INELIGIBLE_V1_HISTORICAL` for any unseen V2 test.
"""

# ------------------------------------------------------------------------------------------------- request packages (NOT SENT)
# (id, landscape, dataset/study, custodian, missing artifact, why, preferred format, spatial metadata, licence question, draft text, diversity)
REQUEST_PACKAGES = [
    ("RP01", "PNG_PORGERA_2018", "Tanyas et al. (2022) PNG 2018 landslide inventory (USGS repository item 61f04680)", "Hakan Tanyas (corresponding author) / USGS GHSC data steward",
     "Documentation of how `mapping_extent.shp` was derived; polygons of cloud-covered or image-free areas inside it; imagery footprints and acquisition dates",
     "Omega must be the area actually searched. The paper describes the extent as the area encompassing all landslides, which cannot define non-landslide background",
     "Shapefile/GeoPackage polygons", "CRS EPSG:32754 or WGS84; per-polygon image source and date", "Reuse for a non-commercial student research project with attribution",
     "Dear Dr Tanyas, I am a student using your 2018 Papua New Guinea landslide inventory (USGS ground-failure repository). Could you clarify how the distributed "
     "mapping_extent polygon was drawn: is it the area systematically searched on PlanetScope/RapidEye imagery, or an envelope around mapped landslides? If "
     "cloud-obscured or image-free areas inside it are available as polygons, they would let me define the observed area rigorously. Thank you.",
     "Oceania; non-arc collisional volcanism; tropical high-altitude"),
    ("RP02", "BANKS_PENINSULA_2021", "Yates et al. (2025) Banks Peninsula 2021 rainfall-landslide inventory", "Corresponding author (University of Canterbury)",
     "Source and runout polygons plus the 115 km2 mapping boundary as vectors", "All gates pass on paper; only access blocks an intraplate-shield landscape with strong H potential",
     "GeoPackage/Shapefile", "NZTM2000; mapping date and imagery/LiDAR source", "Research reuse with attribution",
     "Dear Dr Yates, I am a student building a multi-landscape volcanic landslide study. Could you share the 2021 Banks Peninsula landslide source/runout polygons "
     "and the mapped-area boundary as GIS files for non-commercial research with full citation? Thank you.", "Oceania; intraplate hotspot shield; H-capable"),
    ("RP03", "KIVU_RIFT_NTK_2000_2019", "Depicker et al. (2021) North Tanganyika-Kivu shallow-landslide inventory (Zenodo 5027004)", "Arthur Depicker / Olivier Dewitte (RMCA)",
     "Source-area polygons; a spatial imagery-availability or search mask by year; confirmation of the Year attribute coding (negative values)",
     "Points with a statistical bias correction cannot define where absence was observed; the rift basalts would add Africa and continental-rift diversity",
     "GeoPackage polygons + raster mask", "WGS84; Google Earth image dates per mask cell", "CC BY 4.0 confirmed for derived masks?",
     "Dear Dr Depicker and Dr Dewitte, I am a student using your Kivu Rift landslide inventory (Zenodo 5027004). Would you be able to share the source-area "
     "polygons and any spatial mask of where Google Earth imagery was inspected, and explain the negative Year values? This would let me use the inventory "
     "with a rigorous observation domain. Thank you.", "Africa; continental rift; basalt plateau"),
    ("RP04", "MOUNT_ELGON_UGANDA", "Broeckx et al. (2019) Mount Elgon landslide inventory", "Jente Broeckx / Liesbet Jacobs (KU Leuven / VUB)",
     "Landslide polygons with dates and the systematically mapped area", "Africa and intraplate-shield diversity with high H potential",
     "GeoPackage/Shapefile", "WGS84; imagery dates", "Research reuse with attribution",
     "Dear Dr Broeckx, I am a student studying landslides on eroded volcanoes. Could you share the Mount Elgon landslide inventory and its mapped-area "
     "boundary for non-commercial research with citation? Thank you.", "Africa; intraplate shield; H-capable"),
    ("RP05", "GUATEMALA_1976", "Harp et al. (1981) USGS PP 1204-A", "USGS Landslide Hazards Program",
     "A digital mapped-area boundary for PP 1204-A (Plate 1), if held", "Omega exists only on the printed plate; digitising it independently of positives is the fallback",
     "Shapefile", "UTM 15N", "Public domain", "Hello, is a digital version of the mapped-area boundary for USGS PP 1204-A available alongside the "
     "Harp et al. (1981) inventory in the ground-failure repository? Thank you.", "Central America arc (adds a second non-Asian landscape)"),
    ("RP06", "KIVU_KALEHE_RIFT_FLANK", "Maki Mateso et al. (2023) NHESS 23:643", "Corresponding author",
     "Inventory vectors, study boundary, visibility/fieldwork mask, epoch per landslide", "Rift-flank basalt; joint request with RP03",
     "GeoPackage", "WGS84", "Research reuse", "Dear authors, could you share the landslide inventory, study boundary and any visibility mask from your 2023 NHESS "
     "paper for non-commercial research with citation? Thank you.", "Africa; continental rift"),
    ("RP07", "MIRAVALLES_OTTO_2016", "Quesada-Roman et al. (2019) Landslides 16:1539", "Corresponding author",
     "Polygons with trigger attribution (Bijagua earthquake vs. Hurricane Otto), mapped area, imagery dates", "Compound trigger must be separable",
     "GeoPackage", "CRTM05 or WGS84", "Research reuse", "Dear Dr Quesada-Roman, could you share the Miravalles 2016 landslide polygons, the mapped area and the "
     "imagery dates for non-commercial research with citation? Thank you.", "Central America arc; H-capable stratovolcanoes"),
]
