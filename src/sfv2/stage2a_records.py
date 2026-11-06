"""Curated Stage-2A evidence decisions.

These records refine, but never overwrite, the immutable Stage-1 catalogue.
Every PASS or readiness decision is tied to a source-manifest identifier.
"""

from __future__ import annotations

DATE = "2026-10-05"

WAVE1 = [
    "BANKS_PENINSULA_2021", "CAMPANIA_SARNO_1998", "CAPELLADES_TURRIALBA_2016",
    "CHARIM_LESSER_ANTILLES", "CHUUK_CHATAAN_2002", "CIANJUR_2022",
    "COLUMBIA_RIVER_GORGE_BASALT", "COROMANDEL_GABRIELLE_2023", "DOMINICA_MARIA_2017",
    "EL_SALVADOR_2001_JAN", "HOKKAIDO_IBURI_2018", "IFFI_ITALIAN_VOLCANIC_DISTRICTS",
    "IZU_OSHIMA_2013", "KUMAMOTO_ASO_2016", "OREGON_CASCADES_SLIDO", "PATAGONIA_PALI",
    "POHNPEI_CUMULATIVE", "WASHINGTON_CASCADES_WASLID", "WILSONS_RIVER_NSW_2022",
    "MOUNT_CAMEROON_LIMBE", "PICO_DE_ORIZABA", "SETE_CIDADES_AZORES",
    "ETHIOPIA_DESSIE_RIFT_MARGIN", "HAWAII_OAHU_KOOLAU",
]

WAVE2 = [
    "LOMBOK_RINJANI_2018", "ATITLAN_STAN_2005", "KURIKOMA_2008", "CINCHONA_POAS_2009",
    "PUERTO_RICO_MARIA_2017", "NICARAGUA_MITCH_1998", "MATZACO_PUEBLA_2017",
]

def _r(y, omega, m, h, readiness, y_note, omega_note, m_note, h_note,
       unresolved, sources, burden="UNKNOWN", tier_note="Retain audited inventory tier"):
    return dict(y_gate=y, omega_gate=omega, m_gate=m, h_status=h,
                readiness=readiness, y_summary=y_note, omega_summary=omega_note,
                m_summary=m_note, e_summary="Global CHELSA/WorldClim climatology is available; event-trigger products remain event-specific and are not yet acquired.",
                g_summary="Geology source identified where noted; ontology harmonisation awaits acquisition.",
                h_summary=h_note, unresolved=unresolved, source_ids=sources,
                estimated_burden=burden, target_tier_note=tier_note)

R = {}
def add(identifier, *args, **kwargs): R[identifier] = _r(*args, **kwargs)

add("BANKS_PENINSULA_2021", "PASS","PASS","PASS","PASS","BLOCKED",
    "1,353 event source polygons with separate runout representation; core spatial product is request-only.",
    "Published 115 km2 systematic mapping area from post-event imagery and field checks.",
    "2018-2019 1 m LiDAR predates the 2021 event.",
    "Restricted Akaroa constructional-flank/planèze reconstruction is credible.",
    "Obtain the source inventory and mapping boundary from the custodians; access is the blocker.",
    ["S2_BANKS_PAPER","SRC_GEOLOGY_GNS_QMAP"], "Y small/request-only; LiDAR and geology potentially multi-GB")
add("CAMPANIA_SARNO_1998","UNCERTAIN","UNCERTAIN","UNCERTAIN","FAIL","PARTIAL",
    "A publication describes 133 source areas, but an authoritative open event GIS was not verified.",
    "The published 60 km2 study context is not yet bound to a digital exhaustive-search boundary.",
    "A demonstrably pre-May-1998 terrain surface was not verified.",
    "The terrain is volcanic cover over carbonate relief; no coherent volcanic palaeosurface H target is established.",
    "Verify event GIS, exact AOI and a pre-event DEM.",["S2_SARNO_ARCHIVE","SRC_GEOLOGY_ISPRA_CARG"])
add("CAPELLADES_TURRIALBA_2016","PASS","UNCERTAIN","PASS","UNCERTAIN","PARTIAL",
    "USGS earthquake-triggered inventory is public and spatial.",
    "Inventory bounds do not by themselves prove an exhaustive mapping/search domain.",
    "NASADEM February 2000 predates the 2016 event at 30 m.",
    "A restricted volcanic-flank H may be possible; controls and target surface are unresolved.",
    "Bind an authoritative search AOI and inventory geometry semantics.",["SRC_Y_CAPELLADES_TURRIALBA_2016","SRC_GLOBAL_NASADEM"])
add("CHARIM_LESSER_ANTILLES","UNCERTAIN","UNCERTAIN","UNCERTAIN","UNCERTAIN","PARTIAL",
    "The regional compilation exists, but island/event geometry and target tier need source-level verification.",
    "Regional catalogue coverage is heterogeneous and cannot yet serve as Omega.",
    "A temporally matched DEM for each inventory component is unresolved.",
    "Some island edifices may support H, but the pooled catalogue has no single coherent target.",
    "Resolve constituent inventories rather than treat CHARIM as one labelled landscape.",["SRC_Y_CHARIM_LESSER_ANTILLES"])
add("CHUUK_CHATAAN_2002","UNCERTAIN","UNCERTAIN","PASS","PASS","PARTIAL",
    "USGS report maps 269 source/deposit features; reusable vector geometry is not verified.",
    "Named surveyed islands are documented, but exact observable/search coverage is not digital.",
    "NASADEM February 2000 predates Typhoon Chata'an.",
    "Deeply dissected shield remnants support a bounded flank-envelope H concept.",
    "Verify digital Y and the actual surveyed-domain boundary.",["S2_CHUUK_USGS","SRC_GLOBAL_NASADEM"])
add("CIANJUR_2022","UNCERTAIN","UNCERTAIN","PASS","UNCERTAIN","BLOCKED",
    "The global event database record is verified but its files are restricted.",
    "Official emergency AOIs are not proof of exhaustive landslide mapping support.",
    "NASADEM predates the 2022 earthquake.",
    "A volcanic-flank history target needs geological scoping.",
    "Core inventory files and mapping support are inaccessible from the verified record.",["S2_GLOBAL_EQ_DB_META","SRC_GLOBAL_NASADEM"])
add("COLUMBIA_RIVER_GORGE_BASALT","PASS","PASS","UNCERTAIN","UNCERTAIN","PARTIAL",
    "DOGAMI RI-45/SLIDO distinguishes scarps, headscarps and deposits.",
    "RI-45 documents a bounded 900 square-mile inventory study area.",
    "LiDAR-derived cumulative mapping creates predictor/target lineage risk; an independent terrain epoch is not selected.",
    "Plateau/valley incision history is plausible but not yet a volcanic constructional-surface H.",
    "Select an inventory epoch and non-leaking DEM before acquisition.",["S2_SLIDO","S2_RI45","SRC_DEM_OREGON_LIDAR"])
add("COROMANDEL_GABRIELLE_2023","PASS","UNCERTAIN","PASS","UNCERTAIN","BLOCKED",
    "NZ landslide mapping includes source-centred oriented polylines, but full polygon access is registered/restricted.",
    "Priority mapping cells are documented; complete volcano-wide observation support is not established.",
    "NASADEM predates Cyclone Gabrielle.",
    "A volcanic-unit restricted H is plausible but target coverage and geology must be intersected independently.",
    "Acquire the registered data and exact mapping-cell support.",["S2_NZLD_GABRIELLE","SRC_GEOLOGY_GNS_QMAP","SRC_GLOBAL_NASADEM"])
add("DOMINICA_MARIA_2017","UNCERTAIN","FAIL","PASS","UNCERTAIN","REJECT",
    "Open raster semantics are unclear and the richer polygon/cloud product is restricted.",
    "The verified cloud mask alone does not establish all searched observable terrain; prior support audit failed.",
    "NASADEM predates Hurricane Maria, but does not repair the failed target/support architecture.",
    "A multiphase island H remains conditional.",
    "Preserve the prior rejection unless new independent Y/Omega evidence appears.",["SRC_Y_DOMINICA_RASTER","SRC_OMEGA_DOMINICA_CLOUD","SRC_GLOBAL_NASADEM"])
add("EL_SALVADOR_2001_JAN","PASS","UNCERTAIN","PASS","UNCERTAIN","PARTIAL",
    "USGS hosts the MARN January 2001 inventory as an open spatial dataset.",
    "A catalogue bounding box/affected-area statement is not a verified systematic mapping boundary.",
    "NASADEM was acquired in February 2000, before the event.",
    "Volcanic-landscape H may be defined only after limiting the geographically broad inventory.",
    "Verify geometry tier and mapping/search AOI.",["S2_EL_SALVADOR_USGS","SRC_GLOBAL_NASADEM"])
add("HOKKAIDO_IBURI_2018","PASS","PASS","PASS","FAIL","READY_CORE",
    "GSI publishes deterministic open GeoJSON of failure/deposition distributions interpreted from aerial photographs.",
    "GSI publishes the corresponding aerial-photo interpretation range.",
    "Historical GSI 5/10 m DEM editions predate the 2018 event; exact tiles/epochs must be bound at acquisition.",
    "No defensible volcanic palaeosurface H target is required or currently established; use core M/E/G only.",
    "Acquisition must bind the specific historical DEM tile dates and preserve combined target semantics.",["S2_HOKKAIDO_GSI","S2_GSI_HISTORICAL_DEM","SRC_GEOLOGY_GSJ_SEAMLESS"], "Y ~1.1 MB; DEM tiles likely tens to hundreds of MB")
add("IFFI_ITALIAN_VOLCANIC_DISTRICTS","PASS","UNCERTAIN","UNCERTAIN","UNCERTAIN","PARTIAL",
    "IFFI is an authoritative cumulative national inventory with mixed points/lines/polygons.",
    "Administrative coverage and source-map indices are heterogeneous, not a single verified Omega.",
    "Modern terrain may encode cumulative mapped morphology; epoch matching is unresolved.",
    "H must be evaluated district-by-district rather than for pooled IFFI.",
    "Choose one volcanic district and bind its survey support, target tier and terrain epoch.",["S2_IFFI","SRC_GEOLOGY_ISPRA_CARG"])
add("IZU_OSHIMA_2013","PASS","FAIL","PASS","FAIL","REJECT",
    "Official mapping exists but the audited product combines failure/runout semantics.",
    "Prior verification did not establish an independent searchable support domain.",
    "Historical Japanese terrain can predate 2013.",
    "Prior audit found no independently defensible deep-time H for this event design.",
    "Preserve the closed support/H rejection.",["SRC_Y_IZU_OSHIMA_2013","S2_GSI_HISTORICAL_DEM"])
add("KUMAMOTO_ASO_2016","PASS","PASS","PASS","PASS","READY_CORE_H",
    "GSI publishes open event-specific landslide-centre vectors interpreted from aerial photographs; these are Tier-2 occurrence labels, not source polygons.",
    "GSI publishes the aerial-photograph interpretation/map range, independently of positive geometry.",
    "Historical GSI 5/10 m DEM editions predate the April 2016 earthquakes.",
    "Aso caldera/flank geological architecture supports bounded caldera-rim or stratigraphic H ensembles.",
    "Acquisition must bind exact DEM tile epochs and keep post-event products out of M.",["S2_KUMAMOTO_GSI","S2_GSI_HISTORICAL_DEM","S2_ASO_GEOLOGY"], "Y small; geology ~14.2 MB; DEM tiles likely tens to hundreds of MB")
add("OREGON_CASCADES_SLIDO","PASS","PASS","UNCERTAIN","UNCERTAIN","PARTIAL",
    "SLIDO is open and distinguishes deposits, scarps/headscarps, historic points and source maps.",
    "Study extents/source-map index provide candidate support only after a coherent subset is selected.",
    "Most detailed targets were mapped from the same LiDAR morphology proposed for M.",
    "Cascades subsets may support age/stratigraphic H but the statewide pool does not.",
    "Select a bounded volcanic subset and establish a non-leaking M/Y chronology.",["S2_SLIDO","SRC_DEM_OREGON_LIDAR"])
add("PATAGONIA_PALI","PASS","PASS","UNCERTAIN","FAIL","PARTIAL",
    "PALI has 10,026 manually labelled polygons, separate from model predictions; only manual labels may be Y.",
    "A study-area geometry is supplied, subject to confirming manual-search coverage.",
    "Cumulative morphology and DEM epoch matching are unresolved.",
    "No coherent volcanic palaeosurface target is defined for the regional compilation.",
    "Separate manual labels from predictions and isolate a genuinely volcanic study subset.",["S2_PALI"])
add("POHNPEI_CUMULATIVE","PASS","FAIL","PASS","PASS","REJECT",
    "Prior audits verified a cumulative target but not usable public observation support.",
    "POHNPEI_PUBLIC_SUPPORT_PATH_FAILED remains authoritative; positives cannot define Omega.",
    "NASADEM predates much of the modern cumulative mapping but does not repair Omega.",
    "A shield-remnant H is promising but cannot override the support failure.",
    "Primary path remains retired.",["SRC_Y_POHNPEI_CUMULATIVE","SRC_GLOBAL_NASADEM"])
add("WASHINGTON_CASCADES_WASLID","PASS","UNCERTAIN","UNCERTAIN","UNCERTAIN","PARTIAL",
    "WASLID/US landslide compilation contains source-related geometry, but volcanic subset provenance varies.",
    "Source-map extents may support subsets; a single exhaustive Cascades domain is not verified.",
    "Cumulative target morphology may be encoded in modern DEMs.",
    "Local stratigraphic H may be possible after site selection.",
    "Select a coherent mapped project with explicit support and chronology.",["SRC_Y_WASHINGTON_CASCADES_WASLID"])
add("WILSONS_RIVER_NSW_2022","PASS","PASS","PASS","PASS","PARTIAL",
    "Zenodo provides 3,689 manually validated polygons derived from Sentinel-2 NDVI differencing.",
    "The catchment/study domain is bounded, but exact searchable exclusions need inspection.",
    "Pre-2022 NSW elevation is available through ELVIS; exact tile epoch remains to bind.",
    "Lamington Volcanics remnants support a bounded envelope/age H concept.",
    "Terrain-based filtering was used during target derivation; audit the filter to prevent M/Y leakage before readiness.",["S2_WILSON_ZENODO","SRC_DEM_NSW_ELVIS","SRC_GEOLOGY_NSW_SEAMLESS"], "Y ~1.65 MB; DEM and geology potentially GB-scale")
add("MOUNT_CAMEROON_LIMBE","UNCERTAIN","UNCERTAIN","UNCERTAIN","UNCERTAIN","BLOCKED",
    "A paper reports 63 field-mapped scars; a reusable GIS was not verified.",
    "Survey visibility and complete search extent are unresolved.",
    "DEM chronology relative to the cumulative target is unresolved.",
    "A young active edifice offers only conditional local H.",
    "Obtain target GIS, support and codebook.",["SRC_Y_MOUNT_CAMEROON_LIMBE"])
add("PICO_DE_ORIZABA","UNCERTAIN","UNCERTAIN","UNCERTAIN","PASS","BLOCKED",
    "Published landslide mapping exists, but reusable source geometry was not verified.",
    "No digital systematic observation boundary was verified.",
    "Cumulative target chronology is not matched to a terrain epoch.",
    "Preserved stratovolcano flanks support radial/stratigraphic H in principle.",
    "Verify public GIS and mapping support.",["SRC_Y_PICO_DE_ORIZABA"])
add("SETE_CIDADES_AZORES","FAIL","UNCERTAIN","UNCERTAIN","PASS","REJECT",
    "No independently usable spatial landslide target was verified.",
    "No target mapping support can be established without Y.",
    "Terrain exists but is not sufficient for labelled-dataset construction.",
    "Caldera/flank H is plausible but cannot compensate for absent Y.",
    "Reject under the current experiment; do not search opportunistically for positives.",["SRC_Y_SETE_CIDADES_AZORES"])
add("ETHIOPIA_DESSIE_RIFT_MARGIN","UNCERTAIN","UNCERTAIN","UNCERTAIN","UNCERTAIN","BLOCKED",
    "Published mapped failures exist but authoritative reusable geometry was not verified.",
    "Survey extent/visibility is not available as a digital support product.",
    "Target chronology and suitable pre-inventory DEM are unresolved.",
    "Rift-margin volcanics may permit a local H, but the constructional target is undefined.",
    "Core Y/Omega access must be resolved before geology work.",["SRC_Y_ETHIOPIA_DESSIE_RIFT_MARGIN"])
add("HAWAII_OAHU_KOOLAU","UNCERTAIN","UNCERTAIN","UNCERTAIN","PASS","BLOCKED",
    "Research inventories exist, but an open, source-domain spatial product was not verified.",
    "Whole-island absence cannot be treated as systematic support.",
    "Cumulative inventory timing is not matched to an uncontaminated terrain surface.",
    "Ko'olau's eroded shield and dike-controlled structure make H scientifically strong.",
    "Obtain a reproducible source inventory and observation mask.",["SRC_Y_HAWAII_OAHU_KOOLAU"])

# Wave 2 was triggered because Wave 1 produced only two ready landscapes.
add("LOMBOK_RINJANI_2018","PASS","PASS","PASS","PASS","READY_CORE_H",
    "Frozen V1 verifies the event inventory and local files.","Frozen V1 verifies a non-positive-derived mapping domain.",
    "Frozen V1 verifies the selected pre-event DEM.","Rinjani/Samalas structural and stage controls support H ensembles.",
    "No acquisition blocker; experimental role remains unassigned.",["SRC_Y_LOMBOK_RINJANI_2018"],"Already held in frozen V1; no V2 copy yet")
add("ATITLAN_STAN_2005","PASS","PASS","PASS","PASS","READY_CORE_H",
    "Frozen V1 verifies the event target.","Frozen V1 verifies the mapped search domain.","Frozen V1 verifies pre-event terrain suitability.",
    "Caldera/rim and volcanic-stage controls support bounded H ensembles.","No data gate blocker; role assignment remains future work.",["SRC_Y_ATITLAN_STAN_2005"],"Already held in frozen V1; no V2 copy yet")
add("KURIKOMA_2008","PASS","UNCERTAIN","PASS","PASS","PARTIAL",
    "Frozen V1 verifies event geometry.","The catchment rule is not yet verified as the original exhaustive mapping/search support.",
    "Frozen V1 verifies pre-event terrain suitability.","Stratovolcano/caldera terrain supports H.",
    "Bind the original inventory's systematic support rather than reuse a processing catchment.",["SRC_Y_KURIKOMA_2008","SRC_GEOLOGY_GSJ_SEAMLESS"])
add("CINCHONA_POAS_2009","PASS","UNCERTAIN","PASS","UNCERTAIN","PARTIAL",
    "Frozen V1 verifies event geometry.","Original exhaustive mapping/search support remains unresolved.",
    "Frozen V1 verifies pre-event terrain suitability.","Multiedifice Poas/Barva/Platanar history requires a restricted target surface.",
    "Resolve Omega and the H domain.",["SRC_Y_CINCHONA_POAS_2009"])
add("PUERTO_RICO_MARIA_2017","PASS","PASS","PASS","FAIL","READY_CORE",
    "Frozen V1 verifies headscarp-centre/source labels.","Frozen V1 records approximately 97% imagery coverage and the mapping boundary.",
    "USGS 3DEP 2015-2016 1 m LiDAR predates Hurricane Maria.","This is a non-volcanic/reference core landscape; no volcanic H is claimed.",
    "Acquire only if retained as a reference landscape; role remains unassigned.",["SRC_Y_PUERTO_RICO_MARIA_2017","SRC_OMEGA_PUERTO_RICO_MARIA_2017","SRC_DEM_PUERTO_RICO_PRE_EVENT"],"Large 1 m LiDAR; existing frozen V1 assets must not be copied wholesale")
add("NICARAGUA_MITCH_1998","PASS","UNCERTAIN","UNCERTAIN","UNCERTAIN","PARTIAL",
    "USGS OFR 01-412-A provides separate source, landslide, debris-flow/runout and deposit geometries.",
    "Mapped departments/study sheets do not yet prove uniform exhaustive support.",
    "NASADEM is post-Mitch; major Casita morphology makes it unsuitable without an independent pre-event source.",
    "Casita and other volcanic sites may permit local H, but targets cannot define it.",
    "Find a defensible pre-1998 DEM and exact search support.",["SRC_Y_NICARAGUA_MITCH_FILES"])
add("MATZACO_PUEBLA_2017","PASS","PASS","PASS","UNCERTAIN","PARTIAL",
    "Frozen V1 verifies USGS spatial target geometry.","Frozen V1 includes a downloaded, hashed mapping-extent shapefile.",
    "NASADEM predates the 2017 event.","The candidate's coherent volcanic-landscape/H scope is not established.",
    "Confirm volcanic-unit scope and geology before treating this as a volcanic landscape.",["SRC_Y_MATZACO_PUEBLA_2017","SRC_OMEGA_MATZACO_PUEBLA_2017","SRC_GLOBAL_NASADEM"])

NEW_SOURCES = [
    ("SRC_Y_KURIKOMA_2008","KURIKOMA_2008","Y","Yagi et al. (2009) Kurikoma earthquake landslide inventory","Yagi et al.","UNKNOWN","UNKNOWN","PRIOR_FROZEN_V1_RECORD","Verified from frozen V1 registry; original access route remains to be rebound"),
    ("SRC_Y_MOUNT_CAMEROON_LIMBE","MOUNT_CAMEROON_LIMBE","Y","Mount Cameroon/Limbe field-mapped landslide study","UNKNOWN","UNKNOWN","UNKNOWN","PAPER","Stage-1 paper evidence; reusable GIS and survey support not verified"),
    ("SRC_Y_PICO_DE_ORIZABA","PICO_DE_ORIZABA","Y","Legorreta Paulin et al. (2014) Pico de Orizaba landslide study","Legorreta Paulin et al.","UNKNOWN","UNKNOWN","PAPER","Paper evidence; reusable GIS and support not verified"),
    ("SRC_Y_HAWAII_OAHU_KOOLAU","HAWAII_OAHU_KOOLAU","Y","Oahu shallow-landslide susceptibility study","Geomorphology study authors","UNKNOWN","UNKNOWN","PAPER","Paper evidence; modelled susceptibility is not accepted as Y"),
    ("SRC_Y_POHNPEI_CUMULATIVE","POHNPEI_CUMULATIVE","Y","USGS Pohnpei cumulative landslide inventory","USGS","UNKNOWN","UNKNOWN","PRIOR_STAGE5CP_RECORD","Prior project audit; public support path failed"),
    ("SRC_Y_SETE_CIDADES_AZORES","SETE_CIDADES_AZORES","Y","Sete Cidades candidate target search","UNKNOWN","UNKNOWN","UNKNOWN","NEGATIVE_EVIDENCE_RECORD","No usable target verified; retained to trace rejection without inventing a dataset"),
    ("S2_BANKS_PAPER","BANKS_PENINSULA_2021","Y|OMEGA","2021 Banks Peninsula rainfall landslide inventory","University of Canterbury","https://doi.org/10.1007/s10346-023-02095-5","10.1007/s10346-023-02095-5","PAPER","Request-only spatial data; paper verifies 1,353 source polygons, runout representation and 115 km2 mapping area"),
    ("S2_SARNO_ARCHIVE","CAMPANIA_SARNO_1998","Y|OMEGA","Sarno 1998 source-area study","Peer-reviewed study","https://doi.org/10.1007/s10346-010-0218-5","10.1007/s10346-010-0218-5","PAPER","Paper evidence only; open authoritative GIS not verified"),
    ("S2_CHUUK_USGS","CHUUK_CHATAAN_2002","Y|OMEGA","Landslides triggered by Typhoon Chata'an in the Chuuk Islands","USGS","https://pubs.usgs.gov/of/2004/1348/","UNKNOWN","REPORT","Report plates document 269 mapped features; reusable vectors unresolved"),
    ("S2_GLOBAL_EQ_DB_META","CIANJUR_2022","Y","Global earthquake-induced landslide database 2016-2025","University of Twente / Zenodo","https://zenodo.org/records/20157141","10.5281/zenodo.20157141","METADATA","Files restricted; downloaded metadata hash 5e8405e8ab05761248daca5a7d96885b059015475a15721c13effd5d3f687acf"),
    ("S2_SLIDO","OREGON_CASCADES_SLIDO","Y|OMEGA","Statewide Landslide Information Database for Oregon","DOGAMI","https://www.oregongeology.org/slido/","UNKNOWN","PORTAL","Open geodatabase; deposits, scarps/headscarps, historic points and source-map index"),
    ("S2_RI45","COLUMBIA_RIVER_GORGE_BASALT","Y|OMEGA","Landslide inventory of the Columbia River Gorge, RI-45","DOGAMI","https://pubs.oregon.gov/dogami/ri/p-RI-45.htm","UNKNOWN","REPORT_DATASET","Bounded 900 square-mile study area; LiDAR morphology mapping"),
    ("S2_NZLD_GABRIELLE","COROMANDEL_GABRIELLE_2023","Y|OMEGA","Cyclone Gabrielle landslide mapping / New Zealand Landslide Database","GNS Science","https://doi.org/10.21420/X3DR-6M46","10.21420/X3DR-6M46","DATASET","Source-centred oriented polylines; polygon access/complete support require registration and verification"),
    ("S2_EL_SALVADOR_USGS","EL_SALVADOR_2001_JAN","Y","Landslides from the January 13, 2001 El Salvador earthquake","USGS / MARN","https://doi.org/10.5066/F7PR7THQ","10.5066/F7PR7THQ","DATASET","Open spatial inventory; catalogue bounds are not Omega"),
    ("S2_HOKKAIDO_GSI","HOKKAIDO_IBURI_2018","Y|OMEGA","2018 Hokkaido Eastern Iburi earthquake slope-failure/deposition distribution","Geospatial Information Authority of Japan","https://www.gsi.go.jp/BOUSAI/H30-hokkaidoiburi-east-earthquake-index.html","UNKNOWN","DATASET","Official GeoJSON and aerial-photo interpretation range"),
    ("S2_GSI_HISTORICAL_DEM","GLOBAL","DEM","Fundamental Geospatial Data / historical 5 m and 10 m DEM editions","Geospatial Information Authority of Japan","https://fgd.gsi.go.jp/download/menu.php","UNKNOWN","PORTAL","Historical editions include pre-2016 and pre-2018 terrain; exact tile epochs must be bound"),
    ("S2_KUMAMOTO_GSI","KUMAMOTO_ASO_2016","Y|OMEGA","2016 Kumamoto earthquake landslide distribution","Geospatial Information Authority of Japan","https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html","UNKNOWN","DATASET","Official event vectors and aerial-photograph interpretation range"),
    ("S2_ASO_GEOLOGY","KUMAMOTO_ASO_2016","GEOLOGY|AGE|H_CONTROL","Geological Map of Aso Volcano","Geological Survey of Japan, AIST","https://gbank.gsj.jp/volcano/Act_Vol/aso/index-e.html","UNKNOWN","DATASET","Official 1:50,000 raster/vector map and explanation; vector download approximately 14.2 MB"),
    ("S2_IFFI","IFFI_ITALIAN_VOLCANIC_DISTRICTS","Y|OMEGA","IFFI Italian Landslide Inventory","ISPRA","https://idrogeo.isprambiente.it/app/iffi","UNKNOWN","PORTAL","Authoritative cumulative inventory; heterogeneous source projects/support"),
    ("S2_PALI","PATAGONIA_PALI","Y|OMEGA","Patagonian Landslide Inventory (PALI)","Zenodo","https://doi.org/10.5281/zenodo.7057656","10.5281/zenodo.7057656","DATASET","CC BY 4.0; 10,026 manual labels plus separate model predictions and study area"),
    ("S2_WILSON_ZENODO","WILSONS_RIVER_NSW_2022","Y|OMEGA","2022 Wilsons River catchment landslide inventory","Zenodo","https://doi.org/10.5281/zenodo.22105158","10.5281/zenodo.22105158","METADATA","CC BY 4.0; 3,689 polygons; terrain-based filtering plus manual validation; metadata hash 79abe916c7c1f8762c7abf6b6cd83b95ffab5ac018bbfb3bfda48e06c8a35f59"),
]
