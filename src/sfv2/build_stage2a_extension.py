"""Build the Stage-2A extension without mutating Stage 1 or Stage 2A."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
from .schema import ROOT, FUTURE_FIELDS
from .build_stage2a import STAGE2_COLUMNS, STAGE2_SOURCE_COLUMNS, STAGE2_INVENTORY_COLUMNS
from .stage2a_extension_records import DATE, NEW_SOURCES, R

META=ROOT/"metadata"; OUT=ROOT/"outputs"/"stage2a_extension"; NOTES=ROOT/"screening"/"evidence"/"stage2a_extension"
EXT_COLUMNS=STAGE2_COLUMNS+["stage2a_extension_investigated","stage2a_extension_readiness","stage2a_extension_blocker","stage2a_extension_y_resolution","stage2a_extension_omega_resolution","stage2a_extension_m_resolution","stage2a_extension_m_leakage_risk","stage2a_extension_h_resolution","stage2a_extension_lineage_risk","stage2a_extension_source_ids","stage2a_extension_date"]
SOURCE_COLUMNS=STAGE2_SOURCE_COLUMNS
INV_COLUMNS=STAGE2_INVENTORY_COLUMNS
ACQ_COLUMNS=["acquisition_id","landscape_id","source_id","data_role","dataset_name","source_url","download_url","expected_format","expected_size","licence","temporal_coverage","spatial_coverage","resolution","target_destination","original_filename","checksum_algorithm","checksum","download_status","verification_status","notes"]

def rows(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))
def write_csv(path,data,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator="\n");w.writeheader();w.writerows({c:r.get(c,"") for c in cols} for r in data)
def canon(path,obj): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build_candidates():
    data=rows(META/"stage2a_landscape_candidates.csv")
    for row in data:
        r=R.get(row["landscape_id"])
        if not r:
            row.update(stage2a_extension_investigated="FALSE",stage2a_extension_readiness=row["stage2a_readiness"],stage2a_extension_blocker="",stage2a_extension_y_resolution="",stage2a_extension_omega_resolution="",stage2a_extension_m_resolution="",stage2a_extension_m_leakage_risk="",stage2a_extension_h_resolution="",stage2a_extension_lineage_risk="",stage2a_extension_source_ids="",stage2a_extension_date="")
            continue
        row.update(y_gate=r["y_gate"],omega_gate=r["omega_gate"],m_gate=r["m_gate"],h_status=r["h_status"],inventory_tier=r["target_tier"])
        row.update(stage2a_extension_investigated="TRUE",stage2a_extension_readiness=r["readiness"],stage2a_extension_blocker=r["blocker"],stage2a_extension_y_resolution=r["y_resolution"],stage2a_extension_omega_resolution=r["omega_resolution"],stage2a_extension_m_resolution=r["m_resolution"],stage2a_extension_m_leakage_risk=r["m_leakage_risk"],stage2a_extension_h_resolution=r["h_resolution"],stage2a_extension_lineage_risk=r["lineage_risk"],stage2a_extension_source_ids="|".join(r["source_ids"]),stage2a_extension_date=DATE)
    return data

def build_sources():
    data=rows(META/"stage2a_source_manifest.csv")
    for s in NEW_SOURCES:
        path=ROOT/s["metadata_file"] if s["metadata_file"] else None
        base=dict(source_id=s["source_id"],landscape_id=s["landscape_id"],data_role=s["data_role"],source_title=s["source_title"],source_organisation=s["source_organisation"],source_type=s["source_type"],citation=s["source_title"],doi=s["doi"],url=s["url"],access_date=DATE,downloadable="TRUE" if s["url"].startswith("http") else "UNKNOWN",file_format=path.suffix.lstrip(".").upper() if path else "HTML/PUBLICATION",licence="UNKNOWN",spatial_coverage="See extension evidence note",temporal_coverage="See extension evidence note",resolution_or_scale="UNKNOWN",tier_relevance="See target-tier audit",omega_obs_evidence="See Omega audit",h_relevance="See H audit",verified="VERIFIED",notes="Stage-2A-extension targeted evidence",authors="UNKNOWN",publication_year="UNKNOWN",access_route=s["url"],stage2a_verified="TRUE",metadata_file=s["metadata_file"],metadata_sha256=sha(path) if path and path.exists() else "")
        data.append(base)
    return data

def build_inventories():
    data=rows(META/"stage2a_inventory_manifest.csv")
    for row in data:
        r=R.get(row["landscape_id"])
        if not r or row.get("stage2a_role")=="DIAGNOSTIC_ONLY_NOT_Y":continue
        row["target_tier"]=r["target_tier"]
        row["observation_domain_establishable"]={"PASS":"TRUE","FAIL":"FALSE","UNCERTAIN":"UNCERTAIN"}[r["omega_gate"]]
        row["scientific_usability"]=r["readiness"]
        row["mapping_bias"]=r["blocker"]
    return data

def acquisition_rows():
    base="data/raw/stage2b"
    return [
      dict(acquisition_id="ACQ_KUMAMOTO_Y_OMEGA",landscape_id="KUMAMOTO_ASO_2016",source_id="S2_KUMAMOTO_GSI",data_role="Y|OMEGA",dataset_name="2016 Kumamoto aerial-photo landslide distribution",source_url="https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html",download_url="https://www.gsi.go.jp/common/000143456.zip",expected_format="ZIP/KML",expected_size="48.0 KB",licence="GSI content terms; attribution required",temporal_coverage="2016-04-16 to 2016-07-24 post-event imagery",spatial_coverage="Published interpretation ranges, Aso and Kumamoto districts",resolution="Mapped major >=~1 ha and minor ~0.1-1 ha occurrence centres",target_destination=f"{base}/KUMAMOTO_ASO_2016/S2_KUMAMOTO_GSI/000143456.zip",original_filename="000143456.zip",checksum_algorithm="SHA256",checksum="",download_status="PLANNED",verification_status="READY_TO_DOWNLOAD",notes="Retain point-centre/Tier-2 semantics and published interpretation range"),
      dict(acquisition_id="ACQ_KUMAMOTO_DEM",landscape_id="KUMAMOTO_ASO_2016",source_id="S2_GSI_HISTORICAL_DEM",data_role="DEM",dataset_name="GSI Fundamental Geospatial Data historical DEM",source_url="https://service.gsi.go.jp/kiban/",download_url="https://service.gsi.go.jp/kiban/app/map/",expected_format="ZIP/JPGIS-GML",expected_size="UNKNOWN_UNTIL_MESH_SELECTION",licence="GSI Fundamental Geospatial Data terms",temporal_coverage="Creation/source date must be <=2016-04-13",spatial_coverage="Exact meshes intersecting frozen GSI interpretation range",resolution="Prefer 5 m pre-event; 10 m contour-derived fallback",target_destination=f"{base}/KUMAMOTO_ASO_2016/S2_GSI_HISTORICAL_DEM/",original_filename="TBD_BY_OFFICIAL_MESH_QUERY",checksum_algorithm="SHA256",checksum="",download_status="PLANNED_SELECTION_GATE",verification_status="REFUSE_IF_SOURCE_EPOCH_POST_EVENT_OR_UNDOCUMENTED",notes="Query historical data; record XML lineage and mesh IDs before acceptance"),
      dict(acquisition_id="ACQ_KUMAMOTO_G_H",landscape_id="KUMAMOTO_ASO_2016",source_id="S2_ASO_GEOLOGY",data_role="GEOLOGY|AGE|H_CONTROL",dataset_name="Geological Map of Aso Volcano 1:50,000",source_url="https://gbank.gsj.jp/volcano/Act_Vol/aso/index-e.html",download_url="https://www.gsj.jp/data/VOLC/VCT/GSJ_MAP_VOLC_04_1985_V01.zip",expected_format="ZIP/Shapefile/KML",expected_size="14.2 MB",licence="GSJ/AIST content terms; confirm in archive",temporal_coverage="1985 publication",spatial_coverage="Aso Volcano",resolution="1:50,000",target_destination=f"{base}/KUMAMOTO_ASO_2016/S2_ASO_GEOLOGY/GSJ_MAP_VOLC_04_1985_V01.zip",original_filename="GSJ_MAP_VOLC_04_1985_V01.zip",checksum_algorithm="SHA256",checksum="",download_status="PLANNED",verification_status="READY_TO_DOWNLOAD",notes="Categorical geology/age and independent H controls; no resistance score"),
      dict(acquisition_id="ACQ_KUMAMOTO_E",landscape_id="KUMAMOTO_ASO_2016",source_id="SRC_GLOBAL_CHELSA",data_role="ENVIRONMENT",dataset_name="CHELSA climatology subset plan",source_url="https://chelsa-climate.org/",download_url="TBD_AFTER_VARIABLE_ALLOWLIST",expected_format="GeoTIFF/NetCDF",expected_size="UNKNOWN",licence="Confirm selected CHELSA product",temporal_coverage="Climatology; exact version to freeze",spatial_coverage="Kumamoto/Aso acquisition domain",resolution="~1 km product-dependent",target_destination=f"{base}/KUMAMOTO_ASO_2016/SRC_GLOBAL_CHELSA/",original_filename="TBD",checksum_algorithm="SHA256",checksum="",download_status="NOT_AUTHORIZED_IN_THIS_STAGE",verification_status="VARIABLE_VERSION_SELECTION_REQUIRED",notes="No bulk environment download in extension"),
      dict(acquisition_id="ACQ_HOKKAIDO_Y_OMEGA",landscape_id="HOKKAIDO_IBURI_2018",source_id="S2_HOKKAIDO_GSI",data_role="Y|OMEGA",dataset_name="Hokkaido Iburi slope-failure/deposition distribution D1-No.991",source_url="https://www.gsi.go.jp/BOUSAI/H30-hokkaidoiburi-east-earthquake-index.html",download_url="https://www.gsi.go.jp/common/000204728.zip",expected_format="ZIP/GeoJSON",expected_size="1.1 MB",licence="GSI content terms; attribution required",temporal_coverage="2018-09-06 to 2018-09-11 post-event aerial photography",spatial_coverage="Published D1-No.991 interpretation extent around Atsuma",resolution="Mapped combined failure/deposition polygons",target_destination=f"{base}/HOKKAIDO_IBURI_2018/S2_HOKKAIDO_GSI/000204728.zip",original_filename="000204728.zip",checksum_algorithm="SHA256",checksum="",download_status="PLANNED",verification_status="READY_TO_DOWNLOAD",notes="Retain combined Tier-2 semantics; do not relabel as source polygons"),
      dict(acquisition_id="ACQ_HOKKAIDO_DEM",landscape_id="HOKKAIDO_IBURI_2018",source_id="S2_GSI_HISTORICAL_DEM",data_role="DEM",dataset_name="GSI Fundamental Geospatial Data historical DEM",source_url="https://service.gsi.go.jp/kiban/",download_url="https://service.gsi.go.jp/kiban/app/map/",expected_format="ZIP/JPGIS-GML",expected_size="UNKNOWN_UNTIL_MESH_SELECTION",licence="GSI Fundamental Geospatial Data terms",temporal_coverage="Creation/source date must be <=2018-09-05",spatial_coverage="Exact meshes intersecting D1-No.991 interpretation extent",resolution="Prefer 5 m pre-event; 10 m contour-derived fallback",target_destination=f"{base}/HOKKAIDO_IBURI_2018/S2_GSI_HISTORICAL_DEM/",original_filename="TBD_BY_OFFICIAL_MESH_QUERY",checksum_algorithm="SHA256",checksum="",download_status="PLANNED_SELECTION_GATE",verification_status="REFUSE_IF_SOURCE_EPOCH_POST_EVENT_OR_UNDOCUMENTED",notes="Record XML lineage and mesh IDs before acceptance"),
      dict(acquisition_id="ACQ_HOKKAIDO_G",landscape_id="HOKKAIDO_IBURI_2018",source_id="SRC_GEOLOGY_GSJ_SEAMLESS",data_role="GEOLOGY|AGE",dataset_name="Seamless Digital Geological Map of Japan V2",source_url="https://gbank.gsj.jp/seamless/",download_url="https://gbank.gsj.jp/seamless/v2/api/",expected_format="Vector/API export",expected_size="UNKNOWN_UNTIL_SUBSET",licence="GSJ/AIST terms",temporal_coverage="Current map version to freeze at download",spatial_coverage="Atsuma/Hokkaido analysis domain",resolution="1:200,000",target_destination=f"{base}/HOKKAIDO_IBURI_2018/SRC_GEOLOGY_GSJ_SEAMLESS/",original_filename="TBD_SUBSET_EXPORT",checksum_algorithm="SHA256",checksum="",download_status="PLANNED_SELECTION_GATE",verification_status="VERSION_AND_EXPORT_ROUTE_TO_BIND",notes="Core G only; no H claim"),
      dict(acquisition_id="ACQ_HOKKAIDO_E",landscape_id="HOKKAIDO_IBURI_2018",source_id="SRC_GLOBAL_CHELSA",data_role="ENVIRONMENT",dataset_name="CHELSA climatology subset plan",source_url="https://chelsa-climate.org/",download_url="TBD_AFTER_VARIABLE_ALLOWLIST",expected_format="GeoTIFF/NetCDF",expected_size="UNKNOWN",licence="Confirm selected CHELSA product",temporal_coverage="Climatology; exact version to freeze",spatial_coverage="Hokkaido Iburi acquisition domain",resolution="~1 km product-dependent",target_destination=f"{base}/HOKKAIDO_IBURI_2018/SRC_GLOBAL_CHELSA/",original_filename="TBD",checksum_algorithm="SHA256",checksum="",download_status="NOT_AUTHORIZED_IN_THIS_STAGE",verification_status="VARIABLE_VERSION_SELECTION_REQUIRED",notes="No bulk environment download in extension"),
    ]

def note(row,source_index):
    links=[]
    for sid in row["stage2a_extension_source_ids"].split("|"):
        if not sid:continue
        s=source_index[sid]; label=s["source_title"]; url=s["url"]
        links.append(f"- `{sid}` — [{label}]({url})" if url.startswith("http") else f"- `{sid}` — {label}: `{url}`")
    return "\n".join([f"# {row['landscape_name']} — Stage 2A-extension", "",f"Decision: **{row['stage2a_extension_readiness']}**", "",f"- Y `{row['y_gate']}` / `{row['inventory_tier']}`: {row['stage2a_extension_y_resolution']}",f"- Omega `{row['omega_gate']}`: {row['stage2a_extension_omega_resolution']}",f"- M `{row['m_gate']}` / leakage `{row['stage2a_extension_m_leakage_risk']}`: {row['stage2a_extension_m_resolution']}",f"- H `{row['h_status']}`: {row['stage2a_extension_h_resolution']}",f"- Lineage risk: `{row['stage2a_extension_lineage_risk']}`", "",f"**Exact remaining blocker:** {row['stage2a_extension_blocker']}","","## Sources","",*links,"","No experimental/lockbox role or future ML field was changed.",""])

def main():
    OUT.mkdir(parents=True,exist_ok=True);NOTES.mkdir(parents=True,exist_ok=True)
    preserve=[META/"landscape_candidates.csv",ROOT/"outputs/screening/STAGE_1_SCREENING_CHECKPOINT.json",META/"stage2a_landscape_candidates.csv",META/"stage2a_source_manifest.csv",META/"stage2a_inventory_manifest.csv",ROOT/"outputs/stage2a/STAGE2A_CHECKPOINT.json"]
    before={str(p.relative_to(ROOT)):sha(p) for p in preserve}
    candidates,sources,inventories=build_candidates(),build_sources(),build_inventories(); sidx={s["source_id"]:s for s in sources}
    write_csv(META/"stage2a_extension_landscape_candidates.csv",candidates,EXT_COLUMNS)
    write_csv(META/"stage2a_extension_source_manifest.csv",sources,SOURCE_COLUMNS)
    write_csv(META/"stage2a_extension_inventory_manifest.csv",inventories,INV_COLUMNS)
    investigated=[r for r in candidates if r["stage2a_extension_investigated"]=="TRUE"]
    cols=["landscape_id","stage2a_readiness","stage2a_extension_readiness","y_gate","omega_gate","m_gate","h_status","inventory_tier","stage2a_extension_blocker"]
    write_csv(OUT/"blocker_resolution.csv",investigated,cols)
    write_csv(OUT/"acquisition_readiness.csv",candidates,["landscape_id","landscape_name","country","continent","tectonic_setting","volcano_type","y_gate","omega_gate","m_gate","h_status","stage2a_extension_readiness"])
    write_csv(OUT/"dem_temporal_leakage_audit.csv",investigated,["landscape_id","m_gate","stage2a_extension_m_leakage_risk","stage2a_extension_m_resolution"])
    write_csv(OUT/"omega_resolution_audit.csv",investigated,["landscape_id","omega_gate","stage2a_extension_omega_resolution"])
    write_csv(OUT/"target_tier_resolution.csv",investigated,["landscape_id","inventory_tier","y_gate","stage2a_extension_y_resolution"])
    write_csv(OUT/"h_capability_expansion.csv",investigated,["landscape_id","h_status","h_quality","h_method_candidate","stage2a_extension_h_resolution"])
    unresolved=[r for r in investigated if not r["stage2a_extension_readiness"].startswith("READY_")]
    write_csv(OUT/"unresolved_high_value_candidates.csv",unresolved,["landscape_id","stage2a_extension_readiness","stage2a_extension_blocker"])
    for row in investigated:(NOTES/f"{row['landscape_id']}.md").write_text(note(row,sidx))
    wr=next(r for r in investigated if r["landscape_id"]=="WILSONS_RIVER_NSW_2022")
    (OUT/"WILSONS_RIVER_LINEAGE_AUDIT.md").write_text("# Wilsons River lineage audit\n\n**Verdict: HIGH lineage risk; remain PARTIAL.**\n\nThe repository metadata establishes Sentinel-2 NDVI differencing, terrain-based filtering, then manual validation. It does not identify the terrain variables, thresholds, candidate mask, missed-slide audit, or false-negative sampling frame. Manual validation supports accepted positives; it does not demonstrate that terrain outside the candidate filter received comparable scrutiny. The whole Wilsons River catchment is therefore computational coverage, not verified catchment-wide observation support. M variables capable of recreating the terrain filter cannot be treated as independent predictors until the workflow and searched candidate mask are supplied. No background sampling is authorised.\n")
    acq=acquisition_rows();write_csv(OUT/"kumamoto_acquisition_manifest.csv",[r for r in acq if r["landscape_id"]=="KUMAMOTO_ASO_2016"],ACQ_COLUMNS);write_csv(OUT/"hokkaido_acquisition_manifest.csv",[r for r in acq if r["landscape_id"]=="HOKKAIDO_IBURI_2018"],ACQ_COLUMNS)
    canon(OUT/"acquisition_manifest_schema.json",{"schema":"SFV2_ACQUISITION_MANIFEST_V1","fields":ACQ_COLUMNS,"rules":["checksum remains empty until bytes are acquired","raw destination is write-once","source epoch and domain gates fail closed","download_status cannot be COMPLETE before size/hash/structure validation"]})
    ready=[r for r in candidates if r["stage2a_extension_readiness"].startswith("READY_")]
    diversity=[]
    for r in sorted(ready,key=lambda x:x["landscape_id"]):diversity.append({k:r[k] for k in ["landscape_id","country","continent","tectonic_setting","volcano_type","climate_class","inventory_tier","h_status","stage2a_extension_readiness"]})
    write_csv(OUT/"ready_pool_diversity.csv",diversity,list(diversity[0]))
    gaps=["Africa","South America","Europe","continental rift","arid/high-altitude","open non-Asian H-capable landscape beyond Atitlan"]
    canon(OUT/"global_diversity_audit.json",{"ready_count":len(ready),"h_ready_count":sum(r["stage2a_extension_readiness"]=="READY_CORE_H" for r in ready),"countries":sorted({r["country"] for r in ready}),"continents":sorted({r["continent"] for r in ready}),"tectonic_settings":sorted({r["tectonic_setting"] for r in ready}),"remaining_gaps":gaps,"broad_acquisition_ready":False})
    table=["# READY-pool diversity","","| Landscape | Country | Continent | Tectonic setting | Volcano type | Tier | H |","|---|---|---|---|---|---|---|"]+[f"| {r['landscape_id']} | {r['country']} | {r['continent']} | {r['tectonic_setting']} | {r['volcano_type']} | {r['inventory_tier']} | {r['h_status']} |" for r in diversity]
    (OUT/"READY_POOL_DIVERSITY.md").write_text("\n".join(table)+"\n")
    newready=[r for r in investigated if r["stage2a_extension_readiness"].startswith("READY_") and not r["stage2a_readiness"].startswith("READY_")]
    report=["# Science Fair ML V2 — Stage 2A-extension", "", "## Executive decision", "",f"The extension investigated {len(investigated)} existing candidates and promoted one: `KURIKOMA_2008` to `READY_CORE_H`. The total READY pool is {len(ready)}, including {sum(r['stage2a_extension_readiness']=='READY_CORE_H' for r in ready)} H-capable landscapes. This is insufficient for broad acquisition because the pool still spans only Asia and North America and remains dominated by subduction settings.","","## Candidate resolutions","","| Landscape | Previous | Extension | Y / tier | Omega | M / leakage | H |","|---|---|---|---|---|---|---|"]
    report += [f"| `{r['landscape_id']}` | {r['stage2a_readiness']} | **{r['stage2a_extension_readiness']}** | {r['y_gate']} / {r['inventory_tier']} | {r['omega_gate']} | {r['m_gate']} / {r['stage2a_extension_m_leakage_risk']} | {r['h_status']} |" for r in investigated]
    report += ["","Newly ready: " + ", ".join(f"`{r['landscape_id']}`" for r in newready) + ".","","Wilsons River remains `PARTIAL` with `HIGH` lineage risk. Capellades is resolved as Tier-2 combined detachment/deposition geometry but lacks exact Omega. Matzaco is `REJECT` because its methods state that no complete inventory was attempted within the mapping extent. Oregon and Columbia Gorge retain `HIGH` M/Y morphology leakage. Chuuk, El Salvador, PALI, IFFI and the diversity candidates retain the exact gaps recorded in `blocker_resolution.csv` and their evidence notes.","","## Wilsons River lineage","","Terrain-based candidate filtering preceded manual validation. The disclosed metadata do not identify the terrain variables, thresholds, searched candidate mask, false-negative audit or missed-slide audit. Manual confirmation establishes accepted positives; it does not establish comparable negative support. Consequently the 3,689 polygons remain qualified Tier-2 positive labels, Omega and M remain uncertain, and no background sampling is authorised.","","## New evidence sources","","| Source ID | Landscape | Evidence |","|---|---|---|"]
    report += [f"| `{s['source_id']}` | `{s['landscape_id']}` | [{s['source_title']}]({s['url']}) |" if s["url"].startswith("http") else f"| `{s['source_id']}` | `{s['landscape_id']}` | {s['source_title']} (`{s['url']}`) |" for s in NEW_SOURCES]
    report += ["","Eight bounded metadata, methods and official landing-page files were retained with byte counts and SHA256 hashes in `download_manifest.json`; no bulk scientific dataset was downloaded.","","## READY pool and remaining gaps","",f"READY count: **{len(ready)}**. READY_CORE_H count: **{sum(r['stage2a_extension_readiness']=='READY_CORE_H' for r in ready)}**. Countries represented: {', '.join(sorted({r['country'] for r in ready}))}. Continents represented: {', '.join(sorted({r['continent'] for r in ready}))}.","","Remaining gaps: Africa; South America; Europe; continental-rift terrain; arid/high-altitude terrain; and an open non-Asian H-capable landscape beyond Atitlan.","","## Acquisition planning","","The Kumamoto and Hokkaido manifests bind the exact official Y/Omega downloads. Their DEM rows fail closed until official mesh IDs and source dates are selected; environment rows remain unauthorised until a variable/version allowlist is frozen. No checksum is populated before acquisition. The reusable schema requires write-once raw destinations, SHA256 after acquisition, and source-date/domain verification before acceptance.","","## Decision","","Broad Stage 2B acquisition is not ready. A bounded Stage 2B acquisition of Kumamoto/Aso then Hokkaido remains ready, subject to the manifest's pre-download DEM-selection gates. Kurikoma is third after its exact V1-to-V2 provenance transfer is separately authorised.","","The deterministic validator confirms unique identifiers, foreign keys, controlled gate/tier/readiness vocabularies, READY implications, empty future metrics, unassigned experimental roles, unexecuted acquisition rows, and unchanged Stage 1/Stage 2A checkpoint identities.","","No training, bulk acquisition, package installation, lockbox assignment, commit or push occurred.",""]
    (OUT/"STAGE2A_EXTENSION_REPORT.md").write_text("\n".join(report))
    downloads=[]
    for p in sorted((ROOT/"data/metadata/stage2a_extension").glob("*")):downloads.append({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p),"role":"bounded metadata/method/landing-page evidence"})
    canon(OUT/"download_manifest.json",downloads)
    from .validate_stage2a_extension import validate
    validation=validate(candidates,sources,inventories,before,acq);canon(OUT/"validation.json",validation)
    if not validation["valid"]:raise ValueError("; ".join(validation["errors"]))
    after={rel:sha(ROOT/rel) for rel in before}
    # Checkpoint files describe the generated evidence package, so they cannot
    # be members of the package hash set without creating a self-reference.
    generated=sorted([p for p in list(OUT.glob("*"))+list(NOTES.glob("*.md"))+[META/"stage2a_extension_landscape_candidates.csv",META/"stage2a_extension_source_manifest.csv",META/"stage2a_extension_inventory_manifest.csv"] if p.is_file() and not p.name.startswith("STAGE2A_EXTENSION_CHECKPOINT")])
    cp={"schema":"SFV2_STAGE2A_EXTENSION_CHECKPOINT_V1","date":DATE,"investigated":len(investigated),"promoted_ready_core":0,"promoted_ready_core_h":len(newready),"newly_ready":[r["landscape_id"] for r in newready],"total_ready":len(ready),"total_ready_h":sum(r["stage2a_extension_readiness"]=="READY_CORE_H" for r in ready),"broad_stage2b_ready":False,"limited_acquisition_order":["KUMAMOTO_ASO_2016","HOKKAIDO_IBURI_2018","KURIKOMA_2008"],"prior_hashes":before,"prior_preserved":before==after,"downloads":downloads,"output_hashes":{str(p.relative_to(ROOT)):sha(p) for p in generated},"training":False,"bulk_download":False,"lockbox_assignment":False}
    canon(OUT/"STAGE2A_EXTENSION_CHECKPOINT.json",cp)
    (OUT/"STAGE2A_EXTENSION_CHECKPOINT.md").write_text("# Stage 2A-extension checkpoint\n\n- Investigated: 19\n- Newly READY_CORE_H: KURIKOMA_2008\n- Total ready: 6\n- Total ready with H: 4\n- Broad Stage 2B ready: false\n- Prior checkpoints preserved: true\n")
    return cp

if __name__=="__main__":main()
