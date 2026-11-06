"""Build Stage 3B: automation-first 3->5 sprint (Talakmau + Puerto Rico), human-gate minimisation, auto-resume manifest.

No ML, no H, no production landscape unless every gate (including user-reserved decisions) has passed.  Prior checkpoints are read only.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import geomap_ingest as G
from . import stage3b_pr as PR
from . import stage3b_talakmau as T
from .build_stage3a import scan_ml_h_artifacts
from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage2f_transfer import INCOMING, v1_verify
from .stage3b_acquire import ATITLAN_DOCS, verify_cop30
from .stage3b_m import m_audit
from .stage3b_point_production import EQUIVALENCE_JSON

AS_OF = "2026-10-06"
OUT = ROOT / "outputs/stage3b"
STAGE3A_CHECKPOINT_SHA = "c2d2af66034765171a687bb778255a66b83ccda921071e8b0ef2a65b8e591351"
STAGE2F2_CHECKPOINT_SHA = "5357dd80f3f5430cb7a34388c658d5d7d84794bd341c97587c652c7b1b2c7b91"
PRIOR_CHECKPOINTS = [("screening", "STAGE_1_SCREENING_CHECKPOINT.json"), ("stage2a", "STAGE2A_CHECKPOINT.json"),
                     ("stage2a2", "STAGE2A2_CHECKPOINT.json"), ("stage2a_extension", "STAGE2A_EXTENSION_CHECKPOINT.json"),
                     ("stage2b", "STAGE2B_CHECKPOINT.json"), ("stage2c", "STAGE2C_CHECKPOINT.json"), ("stage2d", "STAGE2D_CHECKPOINT.json"),
                     ("stage2e", "STAGE2E_CHECKPOINT.json"), ("stage2f", "STAGE2F_CHECKPOINT.json"), ("stage2f1", "STAGE2F1_CHECKPOINT.json"),
                     ("stage2f2", "STAGE2F2_CHECKPOINT.json"), ("stage3a", "STAGE3A_CHECKPOINT.json")]
DECISIONS_FILE = INCOMING / "STAGE3B_DECISIONS.json"
HUMAN_REASONS = {"A": "authenticated account access", "B": "user acceptance of terms/licence/survey", "C": "external communication sent by the user",
                 "D": "browser-only artifact the machine cannot retrieve", "E": "scientific/governance decision reserved for the user"}


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _csv(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False, lineterminator="\n")


# ------------------------------------------------------------------------------------------------ 1. verify state
def verify_state() -> dict:
    # Explicit prior-stage list (never a glob): later stages must not change Stage 3B's outputs on a rebuild.
    cps = [ROOT / f"outputs/{d}/{n}" for d, n in PRIOR_CHECKPOINTS]
    mismatches, shas = [], {}
    for cp in cps:
        d = json.loads(cp.read_text())
        for key in ("output_hashes", "derived_hashes", "production_artifact_hashes", "prior_checkpoints", "authoritative_file_hashes"):
            for path, digest in (d.get(key) or {}).items():
                if isinstance(digest, str) and len(digest) == 64 and (not (ROOT / path).exists() or sha256(ROOT / path) != digest):
                    mismatches.append(dict(checkpoint=str(cp.relative_to(ROOT)), path=path))
        shas[str(cp.relative_to(ROOT))] = sha256(cp)
    summary = json.loads((ROOT / "outputs/stage2f2/landscape_state_summary.json").read_text())
    return dict(checkpoints=shas, hash_mismatches=mismatches,
                stage3a_unchanged=shas["outputs/stage3a/STAGE3A_CHECKPOINT.json"] == STAGE3A_CHECKPOINT_SHA,
                stage2f2_unchanged=shas["outputs/stage2f2/STAGE2F2_CHECKPOINT.json"] == STAGE2F2_CHECKPOINT_SHA,
                production_ready=summary["production_ready"], production_ready_count=summary["count"], total_units=summary["total_units"])


# ------------------------------------------------------------------------------------------------ Track A
def talakmau() -> dict:
    lay = T.layers()
    y = T.y_semantics_audit(lay)
    om = T.omega(lay)
    lineage = T.write_omega(om)
    aoi = gpd.GeoSeries([om["extent"]], crs=T.UTM).to_crs(4326).iloc[0]
    m = m_audit(T.LID, aoi, T.EVENT)
    districts = T.districts_required(om["geometry"])
    present = {k: [p.name for p in G.find_incoming(v["pattern"])] for k, v in {**T.GEOMAP, **T.SHEETS}.items()}
    geology_present = all(present[k] for k in T.GEOMAP) and all(present[k] for k, v in T.SHEETS.items() if v["required"])
    gates = T.gates(y, om, m, districts, geology_present)
    return dict(y=y, omega=om["stats"], m=m, districts=districts, incoming_present=present, geology_present=geology_present,
                gates=gates, lineage=lineage, production_allowed=all(g[1] == "PASS" for g in gates))


GEOLOGY_INGESTION_SPEC = dict(
    landscape_id=T.LID, machinery="sfv2.geomap_ingest (Lombok-validated: reproduces the frozen Stage 2F.2 seam QC exactly)",
    steps=["register untouched originals (register_district / register_sheet; hash, immutable copy, provenance)",
           "archive_audit: CRC, geologi_ar*.shp with .shp/.shx/.dbf/.prj", "load_districts: preserve source_district + source CRS; reproject to UTM 47N",
           "seam_qc between PASAMAN and PASAMAN_BARAT", "coverage of Omega_obs (land after COP30 water) >= 99.9% or gap explained",
           "unit_inventory inside Omega (classification PENDING_SHEET_LEGEND)",
           "legend evidence table from the Lubuksikaping sheet: unit -> official lithology -> age -> evidence -> confidence",
           "frozen ontology mapping with UNKNOWN preserved; cross-island name contamination flagged (Lombok precedent)",
           "volcanic domain = volcanic class AND Cenozoic age (frozen DOMAIN_RULE); OMEGA_ANALYSIS removals quantified"],
    expected_attributes=list(G.UNIT_FIELDS), expected_crs="EPSG:4326 per .prj (as Lombok)", never=["classify units before legend evidence",
                                                                                                  "infer lithology from formation names alone", "force UNKNOWN into volcanic"])


# ------------------------------------------------------------------------------------------------ Track B
def puerto_rico() -> dict:
    mapping = PR.mapping()
    g = PR.geology_frame()
    island = PR.main_island(g)
    y = PR.y_points()
    yv = PR.y_verification(y, island)
    q = PR.geology_quantification(g, y, island)
    m = m_audit(PR.LID, gpd.GeoSeries([island], crs=PR.UTM).to_crs(4326).iloc[0], PR.EVENT, PR.PRIOR_STORM)
    share = q["cenozoic_volcanic_eligible_headscarps"] / q["headscarps_total"]
    domain_pass = False  # see verdict: fragmented 156 km2 epiclastic fringe is not a coherent volcanic landscape under the frozen rule
    gates = [
        ("Y", "PASS", f"{yv['features']} headscarp points (TIER_1); {yv['points_in_duplicate_groups']} points in {yv['exact_duplicate_coordinate_groups']} "
                      "exact-duplicate groups handled by the frozen once-per-unit rule"),
        ("OMEGA", "DECISION_REQUIRED_MOOT", "D-PR-OMEGA not needed while the volcanic-domain gate fails"),
        ("M_TEMPORAL", "PASS" if m["m_temporal_pass"] else "FAIL", "TanDEM-X scenes to 2014-01-29 < Irma 2017-09-06; editing 2018-07 (post-event, as Lombok)"),
        ("DEM_COVERAGE", "PASS" if m["aoi_fully_covered"] and m["dem_nodata_cells_in_aoi"] == 0 else "FAIL", "6 tiles cover the main island; 0 nodata cells"),
        ("GEOLOGY", "PASS", f"OFR 98-38 harmonised unit-by-unit ({len(mapping)} units; {int((mapping.harmonised_class == 'UNKNOWN').sum())} UNKNOWN)"),
        ("VOLCANIC_DOMAIN", "FAIL_FROZEN_RULE", f"Eligible Cenozoic volcanic {q['cenozoic_volcanic_eligible_km2']} km2 in {q['eligible_domain_parts']} fragments "
                                               f"with {q['cenozoic_volcanic_eligible_headscarps']} headscarps ({share:.2%}); pre-Cenozoic volcanic "
                                               f"{q['pre_cenozoic_volcanic_km2']} km2 and K-T straddling {q['k_t_straddling_volcanic_km2']} km2 excluded"),
        ("TARGET_SEMANTICS", "PASS", "POINT_HEADSCARP_TIER1, separate family; never pooled with Kumamoto POINT_EVENT_TIER2"),
        ("TERRAIN_GENERATION", "NOT_RUN", "Point mirror proven on Kumamoto; not run because VOLCANIC_DOMAIN fails"),
    ]
    verdict = ("BLOCKED_VOLCANIC_DOMAIN: under the frozen DOMAIN_RULE Puerto Rico is not a coherent volcanic landscape. Its mapped volcanic "
               "rocks are Cretaceous (or straddle the K-T boundary) island-arc basement; the only eligible units are scattered Paleocene-Eocene "
               "volcaniclastic sandstone/siltstone/tuff fringes. Emailing USGS for Omega would not change this outcome.")
    return dict(mapping=mapping, y=yv, geology=q, m=m, gates=gates, verdict=verdict, domain_pass=domain_pass, eligible_share=share)


def pr_decision_packet(pr: dict) -> str:
    q = pr["geology"]
    return f"""# D-PR-OMEGA decision packet — Puerto Rico (Hurricane María 2017)

## Read this first: no Omega decision is needed now

The Stage 3B unit-by-unit harmonisation of USGS OFR 98-38 with the **frozen** V2 geology ontology shows that Puerto Rico **fails the
volcanic-domain gate independently of Omega**:

| Quantity (main island, official geology polygons) | Area km² | Headscarps |
|---|---|---|
| Cenozoic volcanic (frozen rule: eligible) | {q['cenozoic_volcanic_eligible_km2']} (in {q['eligible_domain_parts']} fragments) | {q['cenozoic_volcanic_eligible_headscarps']} |
| Pre-Cenozoic volcanic (excluded by the frozen rule) | {q['pre_cenozoic_volcanic_km2']} | {q['pre_cenozoic_volcanic_headscarps']} |
| Volcanic units straddling the Cretaceous–Tertiary boundary (not Cenozoic) | {q['k_t_straddling_volcanic_km2']} | {q['k_t_straddling_volcanic_headscarps']} |
| All headscarps | — | {q['headscarps_total']} |

The eligible units ({', '.join(q['eligible_units'])}) are mostly Paleocene–Eocene volcanic sandstone/siltstone/tuff fringes, not a volcanic
landform. Stage 3A's ≤940 km² was an upper bound. It counted ages such as "Eocene–middle Campanian" as Cenozoic, but Campanian is Cretaceous.

**Consequence:** Puerto Rico is `BLOCKED_VOLCANIC_DOMAIN`. D-PR-OMEGA is moot, and emailing USGS is **not** worth doing now.

## The Omega options (kept for completeness; only relevant if the domain rule were ever changed)

| | Option A — documented main-island search | Option B — block (no footprint geometry) | Option C — hold for USGS documentation |
|---|---|---|---|
| Source evidence | Hughes et al. 2019 metadata: FEMA ~97% + DigitalGlobe ~99% of the main island; overlapping images used against cloud/shadow; small islands not examined | Same metadata; FEMA S3 footprint prefix now empty; NOAA covers ~10% | Same, plus a reply from the USGS data contact |
| Assumption | Unobserved remnants (≤ ~1–3%, location unknown) are negligible and unbiased | None | Depends on reply |
| False-negative / background risk | Low–moderate: an unknown ≤3% of "background" may be unobserved | None (no data) | Lowest if footprints exist |
| Reproducibility | High (main-island land from official geology polygons minus Copernicus water; no Y used) | — | High if a file is supplied |
| Consistency with V2 Omega rules | Weaker than Noto/Lombok (published polygons): text-documented extent, not a polygon | Fully consistent | Fully consistent |
| Frozen rule change needed | No (but requires an explicit recorded decision) | No | No |

## The only real Puerto Rico choice (optional governance, not required)

- **KEEP the frozen DOMAIN_RULE (default; no action):** Puerto Rico stays `BLOCKED_VOLCANIC_DOMAIN`.
- **AMEND the rule to admit pre-Cenozoic arc volcanics:** this would redefine "volcanic landscape" for the whole V2 database. It would require a new rule version and re-review of every landscape before ML. It must not be done to gain sample size.
"""


# ------------------------------------------------------------------------------------------------ 5/6. broader prep + requests
EXPANSION = [
    dict(landscape_id="BANKS_PENINSULA_2021", local_artifacts="none", missing="Source + runout polygons and the ~115 km2 mapped boundary (Yates et al. 2025)",
         request="RQ1", unlocks=["BANKS_PENINSULA_2021"], ingestion="register reply files; Omega = supplied mapped boundary; QMAP geology (check loess units vs domain); COP30 pre-event"),
    dict(landscape_id="COROMANDEL_GABRIELLE_2023|AUCKLAND_WAITAKERE_2023", local_artifacts="Massey et al. 2025 open PDF (scratch); OSF table header only",
         missing="V1.0 polygons + fully mapped priority grid cells for Coromandel Peninsula and Waitakere Ranges", request="RQ2",
         unlocks=["COROMANDEL_GABRIELLE_2023", "AUCKLAND_WAITAKERE_2023"], ingestion="Omega = fully mapped priority cells; separate source vs deposit; QMAP volcanic units"),
    dict(landscape_id="MIRAVALLES_OTTO_2016", local_artifacts="none", missing="Polygons with trigger attribution (earthquake vs Hurricane Otto) + mapped area + imagery dates",
         request="RQ3", unlocks=["MIRAVALLES_OTTO_2016"], ingestion="split triggers into separate target phases; never pool silently"),
    dict(landscape_id="KIVU_RIFT_NTK_2000_2019|KIVU_KALEHE_RIFT_FLANK", local_artifacts="UNOSAT FL20230508COD (registered, Stage 3A)",
         missing="Year-specific search mask + source polygons (NTK); Kalehe NHESS inventory + study boundary + visibility mask", request="RQ4",
         unlocks=["KIVU_RIFT_NTK_2000_2019", "KIVU_KALEHE_RIFT_FLANK"], ingestion="single-epoch support only; rift-basalt geology route"),
    dict(landscape_id="ATITLAN_STAN_2005", local_artifacts="V1-transferable raw Y/Omega (V1 core folder; development use only)",
         missing="MAGA Solola soil-study PDFs (browser-only) for the IGN 1:50,000 geology quadrangles", request="none (browser download)",
         unlocks=["ATITLAN_STAN_2005"], ingestion="machine georeferencing-feasibility audit of the quadrangles (ticks, residuals, legend completeness)"),
]

REQUESTS = """# Stage 3B — ready-to-send requests (NOT sent)

Four contacts, deduplicated. Each one unlocks every landscape listed beside it. Puerto Rico's USGS email from Stage 3A is **withdrawn**: the landscape fails the frozen volcanic-domain gate, so the reply could not change the outcome.
Save every reply's attachments, plus the email text (licence/citation terms), untouched in `Science_Fair_ML_V2/data/incoming_user/`.

## RQ1 — Banks Peninsula 2021 → corresponding author listed on the source record (address removed from public source copy)
**Subject:** Request: Eastern Bays (Banks Peninsula) December 2021 landslide inventory and mapping boundary

> Dear Dr Yates,
> I am a student doing a non-commercial science-fair project that tests whether landslide susceptibility learned on some volcanic landscapes transfers to others. Your 2025 New Zealand Journal of Geology and Geophysics paper says the data are available on reasonable request. Could I please have (1) the landslide source and runout polygons and (2) the boundary of the ~115 km² area that was systematically mapped? I need the mapped boundary so that "no landslide" areas can be distinguished from areas that were not examined. GeoPackage or shapefile with CRS and the imagery/LiDAR dates would be ideal. Please let me know how you would like the data cited and any licence conditions; I will not redistribute it.
> With thanks, [your name, school]

## RQ2 — Coromandel + Waitākere (Cyclone Gabrielle) → GNS Science NZ Landslide Database institutional custodian (address removed from public source copy)
**Subject:** Request: Cyclone Gabrielle V1.0 landslide polygons and priority mapping cells for Coromandel and Waitākere

> Dear NZLD team,
> I am a student using landslide inventories in volcanic terrain for a non-commercial science-fair project. The public OSF extract for Massey et al. (2025) covers training cells on sedimentary terrain only. For the Coromandel Peninsula and the Waitākere Ranges only, could I request (1) the V1.0 landslide polygons (source areas separate from runout/deposit if available) and (2) the 5 × 2.5 km priority mapping grid cells, with a flag for cells that were completely mapped and the imagery dates? The completely mapped cells tell me where an absence of landslides was actually observed. GeoPackage or shapefile in NZTM2000 would be ideal. Please tell me the citation and licence terms you require.
> Kind regards, [your name, school]

## RQ3 — Miravalles 2016 → corresponding author of Quesada-Román et al. (2019), *Landslides*, doi:10.1007/s10346-019-01209-4 (use the address on the article page)
**Subject:** Request: Miravalles 2016 landslide polygons with trigger attribution

> Dear Dr Quesada-Román,
> I am a student doing a non-commercial science-fair study of landslide susceptibility on volcanoes. Could I please have the 2016 Miravalles landslide inventory polygons? An attribute saying whether each landslide followed the July 2016 Bijagua earthquake or Hurricane Otto (or is unknown) would be especially helpful, together with the boundary of the area you mapped and the imagery dates. I need the mapped boundary to know where landslides were absent rather than unexamined. Shapefile or GeoPackage would be ideal. Please tell me how to cite the data and any licence conditions; I will not redistribute it.
> Thank you, [your name, school]

## RQ4 — Kivu NTK + Kalehe → Dr Arthur Depicker / Dr Olivier Dewitte, Royal Museum for Central Africa (address on Zenodo record 5027004 and the NHESS 23:643 paper)
**Subject:** Request: NTK landslide inventory search mask (Zenodo 5027004) and the Kalehe inventory

> Dear Dr Depicker and Dr Dewitte,
> I am a student using your open North Tanganyika–Kivu inventory for a non-commercial science-fair project on landslide susceptibility transfer between volcanic landscapes. To use it correctly I need to know where landslides could have been observed. Could you share (1) the source-area polygons, (2) any year-specific imagery-availability or search mask, and (3) the meaning of negative values in the Year field? If possible, the Kalehe inventory from NHESS 23:643 (vectors, study boundary and any fieldwork/visibility mask) would also help. GeoPackage or shapefile is ideal. Please tell me the citation and licence terms you require.
> With thanks, [your name, school]
"""


# ------------------------------------------------------------------------------------------------ 8. human-gate classifier
CANDIDATE_ACTIONS = [
    dict(id="HA1", title="TALAKMAU GEOMAP PACKAGE", reasons=["A", "B"], machine_possible=False, impact=1,
         action="Sign in to Badan Geologi GeoMap and download the 3 official products below (plus 1 optional) in one session; accept the GeoMap map-use policy yourself",
         files=[f"{v['district']} ({v['code']}): {v['product']} — {v['page']}" for v in T.GEOMAP.values()] +
               [f"{v['title']}: {v['product']} — https://geologi.esdm.go.id/geomap/pages/preview/{v['slug']}" + ("" if v["required"] else " [OPTIONAL]")
                for v in T.SHEETS.values()],
         destination="Science_Fair_ML_V2/data/incoming_user/ (untouched originals)"),
    dict(id="HA2", title="D-TAL-Y DECISION", reasons=["E"], machine_possible=False, impact=2,
         action="Decide whether the preliminary UNOSAT 10 m Sentinel-2 landslide extent may serve as Talakmau's Tier-2 affected-terrain target",
         files=["ACCEPT_TIER2_RAPID_MAPPING (labels carry a 'preliminary, detection-limited' quality flag) or HOLD_FOR_VALIDATED_INVENTORY"],
         destination="Write {\"D-TAL-Y\": \"<choice>\"} in Science_Fair_ML_V2/data/incoming_user/STAGE3B_DECISIONS.json (or tell the agent)"),
    dict(id="HA3", title="SEND 4 DATA REQUESTS", reasons=["C"], machine_possible=False, impact=3,
         action="Send the four ready drafts in outputs/stage3b/STAGE3B_REQUESTS.md (Banks; GNS for Coromandel+Waitakere; Miravalles; RMCA for Kivu NTK+Kalehe)",
         files=["RQ1 corresponding-author route", "RQ2 institutional-custodian route", "RQ3 corresponding author, doi:10.1007/s10346-019-01209-4",
                "RQ4 RMCA (Depicker/Dewitte; address on Zenodo 5027004)"],
         destination="Save replies + attachments in Science_Fair_ML_V2/data/incoming_user/"),
    dict(id="HA4", title="ATITLAN BROWSER DOWNLOAD", reasons=["D"], machine_possible=False, impact=4,
         action="Open the two MAGA links in a normal browser and save both PDFs untouched (the server refuses automated clients)",
         files=[f"{d['url']} (~{d['expected_bytes'] / 1e6:.0f} MB)" for d in ATITLAN_DOCS], destination="Science_Fair_ML_V2/data/incoming_user/"),
    # Rejected as machine work or not worth doing (kept to prove the classifier filters them out)
    dict(id="X1", title="Inspect GeoMap units / transcribe sheet legend", reasons=[], machine_possible=True, impact=9, action="machine", files=[], destination=""),
    dict(id="X2", title="Download Copernicus DEM tiles", reasons=[], machine_possible=True, impact=9, action="machine (done)", files=[], destination=""),
    dict(id="X3", title="Digitise / georeference Atitlan quadrangles", reasons=[], machine_possible=True, impact=9, action="machine after HA4", files=[], destination=""),
    dict(id="X4", title="Email USGS about Puerto Rico Omega", reasons=["C"], machine_possible=False, impact=9, worth_doing=False,
         action="withdrawn: Puerto Rico fails VOLCANIC_DOMAIN independently of Omega", files=[], destination=""),
]


def is_legitimate_human_action(a: dict) -> bool:
    return bool(a["reasons"]) and set(a["reasons"]) <= set(HUMAN_REASONS) and not a["machine_possible"] and a.get("worth_doing", True)


# ------------------------------------------------------------------------------------------------ 9. auto-resume manifest
def auto_resume_manifest() -> dict:
    items = []
    for k, v in T.GEOMAP.items():
        items.append(dict(action_id="HA1", landscape_id=T.LID, expected_artifact=v["product"], filename_pattern=v["pattern"], scientific_role="GEOLOGY",
                          acquisition_id=v["acq"], downstream_task="geomap_ingest.register_district -> archive_audit -> load_districts -> seam_qc -> unit_inventory",
                          gate_unlocked="GEOLOGY", auto_resume=True, required=True))
    for k, v in T.SHEETS.items():
        items.append(dict(action_id="HA1", landscape_id=T.LID, expected_artifact=v["title"], filename_pattern=v["pattern"], scientific_role="GEOLOGY_LEGEND_MAP",
                          acquisition_id=v["acq"], downstream_task="geomap_ingest.register_sheet -> legend evidence table -> frozen ontology mapping",
                          gate_unlocked="GEOLOGY|VOLCANIC_DOMAIN", auto_resume=True, required=v["required"]))
    items.append(dict(action_id="HA2", landscape_id=T.LID, expected_artifact="decision D-TAL-Y", filename_pattern=r"STAGE3B_DECISIONS\.json",
                      scientific_role="DECISION", acquisition_id="", downstream_task="record decision; if ACCEPT and geology gates PASS -> build_polygon_landscape",
                      gate_unlocked="Y|TARGET_SEMANTICS", auto_resume=True, required=True,
                      accepted_values=["ACCEPT_TIER2_RAPID_MAPPING", "HOLD_FOR_VALIDATED_INVENTORY"]))
    for d in ATITLAN_DOCS:
        items.append(dict(action_id="HA4", landscape_id="ATITLAN_STAN_2005", expected_artifact=d["url"], filename_pattern=d["filename"].replace(".", r"\.") + r"( \(\d+\))?",
                          scientific_role="GEOLOGY_SOURCE_DOC", acquisition_id="", downstream_task="register; size check vs expected bytes; quadrangle georeferencing audit",
                          gate_unlocked="GEOLOGY", auto_resume=True, required=True, expected_bytes=d["expected_bytes"]))
    for e in EXPANSION[:4]:
        items.append(dict(action_id="HA3", landscape_id=e["landscape_id"], expected_artifact=e["missing"], filename_pattern="(any reply attachment; tell the agent the names)",
                          scientific_role="Y|OMEGA", acquisition_id="", downstream_task=e["ingestion"], gate_unlocked="Y|OMEGA",
                          auto_resume=False, required=False, note="Reply file names are unknown in advance; registration needs the reply's licence text"))
    return dict(manifest_id="STAGE3B_AUTO_RESUME", as_of=AS_OF, incoming_dir="Science_Fair_ML_V2/data/incoming_user/", decisions_file=str(DECISIONS_FILE.relative_to(ROOT)),
                items=items, resume_entrypoint="python -m sfv2.build_stage3b  (Track A re-checks incoming patterns and gates on every run)")


# ------------------------------------------------------------------------------------------------ main
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    state = verify_state()
    cop = verify_cop30()  # tiles were acquired once (stage3b_acquire.acquire_cop30); re-verified offline here
    ta = talakmau()
    pr = puerto_rico()
    equivalence = json.loads(EQUIVALENCE_JSON.read_text()) if EQUIVALENCE_JSON.exists() else {"status": "NOT_RUN"}

    _json(OUT / "STATE_VERIFICATION.json", state)
    _json(OUT / "TALAKMAU_TARGET_SEMANTICS_AUDIT.json", ta["y"])
    _json(OUT / "TALAKMAU_OMEGA.json", ta["omega"])
    _json(OUT / "TALAKMAU_M_AUDIT.json", ta["m"])
    _json(OUT / "TALAKMAU_GEOLOGY_REQUIREMENTS.json", dict(districts_intersecting_omega=ta["districts"], geomap_products=T.GEOMAP, sheets=T.SHEETS,
                                                          incoming_present=ta["incoming_present"], unauthenticated_route="NONE: 'Unduh SHP' and 'Unduh Raster' "
                                                          "both redirect to /geomap/pages/signin and require accepting the map-use policy; public page shows only a low-resolution preview",
                                                          ingestion_spec=GEOLOGY_INGESTION_SPEC))
    _csv(OUT / "TALAKMAU_GATES.csv", [dict(gate=a, status=b, evidence=c) for a, b, c in ta["gates"]])
    pr["mapping"].to_csv(OUT / "PR_GEOLOGY_MAPPING_OFR98_38.csv", index=False, lineterminator="\n")
    _json(OUT / "PR_GEOLOGY_QUANTIFICATION.json", pr["geology"])
    _json(OUT / "PR_Y_VERIFICATION.json", pr["y"])
    _json(OUT / "PR_M_AUDIT.json", pr["m"])
    _csv(OUT / "PR_GATES.csv", [dict(gate=a, status=b, evidence=c) for a, b, c in pr["gates"]])
    (OUT / "D_PR_OMEGA_DECISION_PACKET.md").write_text(pr_decision_packet(pr))
    _csv(OUT / "EXPANSION_PREPARATION.csv", [dict(e, unlocks="|".join(e["unlocks"])) for e in EXPANSION])
    (OUT / "STAGE3B_REQUESTS.md").write_text(REQUESTS)
    actions = [dict(a, legitimate=is_legitimate_human_action(a)) for a in CANDIDATE_ACTIONS]
    _json(OUT / "HUMAN_ACTION_CLASSIFICATION.json", dict(reasons=HUMAN_REASONS, actions=actions))
    _json(OUT / "STAGE3B_AUTO_RESUME.json", auto_resume_manifest())
    _json(OUT / "STAGE3B_DECISIONS_TEMPLATE.json", {"D-TAL-Y": "ACCEPT_TIER2_RAPID_MAPPING | HOLD_FOR_VALIDATED_INVENTORY"})
    _json(OUT / "LINEAGE.json", dict(derived=[dict(output=str(p.relative_to(ROOT)), sha256=sha256(p), operation=op,
                                                   inputs={str(i.relative_to(ROOT)): sha256(i) for i in ins}) for p, ins, op in ta["lineage"]]))
    ml_h = scan_ml_h_artifacts()
    v1_after = v1_verify()
    prod_count = state["production_ready_count"]  # no landscape is generated in Stage 3B: every candidate stops at a reserved gate
    outs = sorted(p for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("STAGE3B_CHECKPOINT", "STAGE3B_DIAGNOSTICS")))
    cp = dict(schema="SFV2_STAGE3B_CHECKPOINT_V1", as_of=AS_OF, production_ready_count=prod_count, production_terrain_units_total=state["total_units"],
              state_verified=dict(stage3a_unchanged=state["stage3a_unchanged"], stage2f2_unchanged=state["stage2f2_unchanged"],
                                  prior_hash_mismatches=len(state["hash_mismatches"]), production_count=state["production_ready_count"]),
              talakmau=dict(gates={a: b for a, b, _ in ta["gates"]}, omega_obs_km2=ta["omega"]["omega_obs_km2"], target_tier=ta["y"]["target_tier"],
                            production_allowed=ta["production_allowed"]),
              puerto_rico=dict(gates={a: b for a, b, _ in pr["gates"]}, verdict=pr["verdict"].split(":")[0],
                               eligible_km2=pr["geology"]["cenozoic_volcanic_eligible_km2"], eligible_headscarps=pr["geology"]["cenozoic_volcanic_eligible_headscarps"]),
              point_mirror_kumamoto_equivalence=equivalence,
              new_raw_acquisitions={r["acquisition_id"]: r["sha256"] for r in cop},
              legitimate_human_actions=[a["id"] for a in actions if a["legitimate"]],
              output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs},
              derived_hashes={str(p.relative_to(ROOT)): sha256(p) for p, _, _ in ta["lineage"]},
              code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("build_stage3b.py", "stage3b_talakmau.py", "stage3b_pr.py", "stage3b_m.py",
                                                                      "stage3b_point_production.py", "stage3b_acquire.py", "geomap_ingest.py")},
              prior_checkpoints=state["checkpoints"], ml_or_h_artifacts_found=ml_h, training=False, h_generated=False,
              production_landscapes_generated=0, external_messages_sent=0, user_approval_inferred=False,
              v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
              volatile_diagnostics={"path": "outputs/stage3b/STAGE3B_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE3B_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    st = git("status", "--porcelain").splitlines()
    _json(OUT / "STAGE3B_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD"),
                                                 modified_tracked=sum(not s.startswith("??") for s in st), untracked=sum(s.startswith("??") for s in st)))
    return dict(cp=cp, state=state, ta=ta, pr=pr, actions=actions)


if __name__ == "__main__":
    main()
