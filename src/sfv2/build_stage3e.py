"""Build the Stage 3E screening outputs without running production."""
from __future__ import annotations

import argparse
import json

from . import stage3e as S


def build(v2_tests: str, science_fair_tests: str, v1_verifier: str) -> dict:
    S.OUT.mkdir(parents=True, exist_ok=True)
    state = S.verify_state()
    assert state["all_inputs_verified"]
    assert (state["production_landscapes"], state["terrain_units"], state["positive_units"]) == (4, 737722, 6204)
    assert state["dominica_maria_verdict"] == "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION"
    assert all(S.machine_score(row) == row["machine_accessibility"] for row in S.CANDIDATES)
    S.write_json(S.OUT / "STATE_VERIFICATION.json", state)

    acquisitions = [{**item, "retrieval_timestamp": S.RETRIEVED,
                     "filename": item["relative_path"].rsplit("/", 1)[-1]} for item in S.ACQUISITIONS]
    # Put required human-facing fields first.
    acquisitions = [{k: row[k] for k in ("source", "doi_or_url", "licence", "retrieval_timestamp", "filename",
                                          "bytes", "sha256", "role", "relative_path")} for row in acquisitions]
    S.write_tsv(S.OUT / "STAGE3E_RAW_ACQUISITIONS.tsv", acquisitions)
    S.write_tsv(S.OUT / "STAGE3E_CANDIDATE_AUDIT.tsv", S.CANDIDATES)
    S.write_tsv(S.OUT / "STAGE3E_NEXT_BATCH_PRODUCTION.tsv", S.NEXT_BATCH)

    saint = {
        "candidate_id": "ST_LUCIA_TOMAS_2010", "verdict": "SAINT_LUCIA_ONE_BLOCKER",
        "Y": {"gate": "PASS", "archive_polygons": 1233, "tomas_epoch": "2011-01-03",
              "active_2011": 1025, "inactive_2011": 3,
              "evidence": "Report says 2011 epoch captures all Hurricane Tomas events; World Bank product is CC BY 4.0."},
        "Omega": {"gate": "BLOCKED", "is_y_derived": False,
                  "known_visibility": "2011 image about 2% thick cloud, 5% cloud fringe, 10% shadow; report says shadow did not significantly obstruct interpretation",
                  "missing": ["machine-readable 2011 image footprint", "cloud/shadow/fringe exclusion polygons",
                              "exact systematically searched/visible polygon"],
                  "rejected_substitutes": ["whole island without visibility exclusions", "Y convex hull", "Y buffer", "figure tracing"]},
        "M": {"gate": "PASS_ROUTE", "route": "NASADEM/SRTM February 2000 predates Tomas 2010"},
        "G": {"gate": "PASS_ROUTE", "route": "BGS/CHARIM national geology route; full spatial audit remains"},
        "target": {"tier": "TIER_2", "gate": "PASS"},
        "only_acquisition_blocker": "target-independent Tomas observation support geometry",
        "archived_service": "CHARIM GeoNode listed geonode:landslides_2010 but now redirects to a homepage; no recoverable WFS feature response or archived feature payload was found",
        "production_ready": False,
    }
    S.write_json(S.OUT / "SAINT_LUCIA_RESCUE_AUDIT.json", saint)

    vanuatu = {
        "candidate_id": "VANUATU_HAROLD_2020", "verdict": "REQUEST_REQUIRED_PUBLIC_ITEM_REMOVED",
        "documented": {"Y_polygons": 1481, "mapping_image": "PlanetScope 2020-04-05", "mapped_boundary": True,
                       "target": "whole landslide bodies; pre-existing landslides not differentiated"},
        "public_recovery": "FAIL", "former_arcgis_item": "b6598e9b92bf4979a76e8fccee741ed2",
        "arcgis_result": "Item does not exist or is inaccessible", "public_mirror_found": False,
        "M_gate": "PASS_ROUTE_PRE_EVENT", "G_gate": "PASS_ROUTE_NOT_FULLY_AUDITED",
        "missing": ["original 1,481 polygons", "mapped/search boundary", "provenance", "reuse licence"],
        "production_ready": False,
    }
    S.write_json(S.OUT / "VANUATU_RECOVERY_AUDIT.json", vanuatu)

    popo = {
        "candidate_id": "POPOCATEPETL_PUEBLA_2017", "verdict": "REQUEST_REQUIRED_PROMISED_FILES_ABSENT",
        "documented": {"study_area": "C", "Y_polygons": 754, "study_area_km2": 365,
                       "cloud_shadow_snow_mask_method": "manual masking documented"},
        "article_data_availability_claim": "GitHub repository delivers new inventories for study areas B, C, E and F",
        "repository_audit": {"url": "https://github.com/nprksh/landslide-mapping-with-cnn",
                             "branches_tags_history_checked": True, "study_area_c_files_found": False},
        "alternate_public_repository_found": False, "M_gate": "PASS_ROUTE_PRE_EVENT", "G_gate": "PASS_ROUTE_NOT_FULLY_AUDITED",
        "missing": ["754 Study Area C polygons", "exact 365 km2 search boundary", "cloud/snow mask",
                    "imagery dates/provenance", "CRS", "reuse licence"], "production_ready": False,
    }
    S.write_json(S.OUT / "POPOCATEPETL_RECOVERY_AUDIT.json", popo)

    atitlan = {
        "candidate_id": "ATITLAN_STAN_2005", "verdict": "RAW_INPUTS_COMPLETE_FOR_STAGE3F",
        "Y": {"features": 6817, "gate": "PASS"},
        "Omega": {"source": "published Boundary_Atitlan_UTM15N", "area_km2": 221.010068,
                  "is_y_derived": False, "gate": "PASS"},
        "M": {"source": "NASADEM HGT n14w092", "acquisition_epoch": "2000-02", "event_year": 2005,
              "gate": "PASS", "warning": "Do not use the post-event Copernicus DEM as primary M"},
        "G": {"source": "MAGA SUELOS:geologia_nacional", "scale": "1:500,000", "gate": "PASS_RAPID_SPATIAL_AUDIT",
              "source_units": [{"code": "Qp", "omega_km2": 86.041224, "Y_centroids": 2649,
                                "meaning": "Quaternary pumice-ash cover/fill"},
                               {"code": "Tv", "omega_km2": 134.968845, "Y_centroids": 4167,
                                "meaning": "Tertiary undivided volcanic rocks"}],
              "omega_covered_km2": 221.010069, "volcanic_domain_rule_weakened": False},
        "Y_area_km2": 2.442639, "target_tier": "TIER_2", "v1_history": "V1_HISTORICAL",
    }
    S.write_json(S.OUT / "ATITLAN_GEOLOGY_AUDIT.json", atitlan)

    erika = {
        "candidate_id": "DOMINICA_ERIKA_2015", "verdict": "RAW_INPUTS_COMPLETE_FOR_STAGE3F",
        "Y": {"source_layer": "Landslides_20150929_Dominica", "features": 1548,
              "event_code": "LS20150909DMA", "area_km2": 1.832437, "gate": "PASS"},
        "Omega": {"construction": "union of supplied dated Analysis_Extent polygons minus same-date CloudObstruction polygons",
                  "analysis_extent_features": 3, "cloud_features": 1983, "analysis_union_km2": 890.404151,
                  "cloud_within_analysis_km2": 112.180187, "visible_km2": 778.223964,
                  "Y_centroids_inside_visible": 1548, "is_y_derived": False, "gate": "PASS"},
        "M": {"source": "Copernicus GLO-30", "source_years": "2011-2014", "event_year": 2015, "gate": "PASS"},
        "G": {"source": "official Dominica Government 1:100,000 volcanic geology map",
              "gate": "PASS_ROUTE", "remaining_stage3f_work": "georeference, digitise, and audit volcanic-domain coverage"},
        "target_tier": "TIER_2", "v1_history": "V1_HISTORICAL",
        "maria_verdict_unchanged": "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION",
    }
    S.write_json(S.OUT / "DOMINICA_ERIKA_AUDIT.json", erika)

    chuuk = {
        "candidate_id": "CHUUK_CHATAAN_2002", "verdict": "OMEGA_REQUEST_REQUIRED",
        "Y": {"source": "USGS P14VTB5D original mapping", "features_in_nhess_copy": 273, "gate": "PASS"},
        "Omega": {"gate": "BLOCKED", "nhess_mapped_within_rejected": True,
                  "reason": "NHESS mapped-within geometry is a Y-derived convex hull; USGS metadata says the inventory is not complete or exhaustive",
                  "is_y_derived_omega_accepted": False},
        "M": {"route": "SRTM/NASADEM February 2000 predates July 2002", "gate": "PASS_ROUTE"},
        "G": {"route": "USGS report documents eastern volcanic islands", "gate": "PASS_ROUTE"},
        "diversity": ["OCEANIA", "CYCLONE_STORM", "OCEANIC_HOTSPOT", "NON_SUBDUCTION"],
        "production_ready": False, "lockbox_potential": "ONLY_IF_INDEPENDENT_COMPLETE_OMEGA_IS_RECOVERED",
    }
    S.write_json(S.OUT / "CHUUK_AUDIT.json", chuuk)

    search_rows = [
        {"round": 1, "axis": "SAINT_LUCIA", "route": "World Bank/CDEMA/BGS/CHARIM live and archived services", "outcome": "Y and methodology confirmed; exact Tomas visibility/search geometry absent"},
        {"round": 1, "axis": "VANUATU", "route": "NASA ArcGIS item, dependencies, NTRS, journal and archive mirrors", "outcome": "paper recovered; former dataset item absent; no public mirror"},
        {"round": 1, "axis": "POPOCATEPETL", "route": "article supplements, full Git history/tags, Zenodo, Figshare, OSF, ETH", "outcome": "article promises Study Area C files; no files found"},
        {"round": 1, "axis": "AFRICA_RIFT", "route": "Kivu/Virunga", "outcome": "available points mainly outside volcanic units and lack Omega"},
        {"round": 1, "axis": "AFRICA_RIFT", "route": "Elgon/Mbale and Hanang", "outcome": "incomplete/preliminary or single-system products; no landscape inventory plus Omega"},
        {"round": 1, "axis": "AFRICA_RIFT", "route": "NHESS Zimbabwe Idai and Malawi Mulanje", "outcome": "non-volcanic granitoid/metamorphic or intrusive terrain"},
        {"round": 1, "axis": "SHIELD_HOTSPOT", "route": "NHESS Micronesia plus USGS Chuuk", "outcome": "official Y acquired; independent complete Omega absent"},
        {"round": 1, "axis": "OTHER_HIGH_VALUE", "route": "MAGA Guatemala Geoportal", "outcome": "official Atitlan geology vector acquired; candidate raw-ready"},
        {"round": 1, "axis": "OTHER_HIGH_VALUE", "route": "NHESS Dominica Erika archive", "outcome": "event-specific Y plus independent dated extent/cloud layers; candidate raw-ready"},
    ]
    S.write_tsv(S.OUT / "STAGE3E_SEARCH_LOG.tsv", search_rows)

    diversity_rows = [
        {"scenario": "CURRENT_PRODUCTION", "count": 4, "added": "NONE", "continents": 1, "outside_asia": 0,
         "rainfall_cyclone_groups": 0, "tectonic_summary": "SUBDUCTION_DOMINATED; <3 SETTINGS",
         "morphology_erosion_climate": "C3/C4/C5 PASS", "potential_lockboxes": 0, "status": "FROZEN_CURRENT"},
        {"scenario": "CONSERVATIVE", "count": 6, "added": "ATITLAN_STAN_2005;DOMINICA_ERIKA_2015", "continents": 2,
         "outside_asia": 2, "rainfall_cyclone_groups": 2, "tectonic_summary": "SUBDUCTION_DOMINATED; <3 SETTINGS",
         "morphology_erosion_climate": "PASS", "potential_lockboxes": 3, "status": "IF_STAGE3F_GATES_PASS"},
        {"scenario": "REALISTIC_4_TO_8", "count": 8,
         "added": "ATITLAN;DOMINICA_ERIKA;ST_LUCIA;VANUATU", "continents": 3, "outside_asia": 4,
         "rainfall_cyclone_groups": 4, "tectonic_summary": "STILL_SUBDUCTION_DOMINATED; <3 SETTINGS",
         "morphology_erosion_climate": "PASS", "potential_lockboxes": 3,
         "status": "REQUIRES_TWO_STAGE3F_PASSES_PLUS_TWO_CUSTODIAN_RECOVERIES"},
        {"scenario": "BEST_DEFENSIBLE_IDENTIFIED", "count": 10,
         "added": "ATITLAN;DOMINICA_ERIKA;ST_LUCIA;VANUATU;POPOCATEPETL;CHUUK", "continents": 3,
         "outside_asia": 6, "rainfall_cyclone_groups": 5,
         "tectonic_summary": "SUBDUCTION_PLUS_HOTSPOT; STILL <3 SETTINGS AND NO RIFT",
         "morphology_erosion_climate": "PASS_PLUS_OCEANIC_HOTSPOT", "potential_lockboxes": 3,
         "status": "CONDITIONAL_ON_ALL_MISSING_ARTIFACTS; NOT A CURRENT CLAIM"},
    ]
    S.write_tsv(S.OUT / "STAGE3E_DIVERSITY_DASHBOARD.tsv", diversity_rows)

    trajectory = {
        "current": {"production_groups": 4, "terrain_units": 737722, "positive_units": 6204,
                    "frozen_rule_satisfied": False},
        "conservative": {"production_groups_if_all_stage3f_pass": 6,
                         "added": ["ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]},
        "realistic_to_8": {"sequence": ["ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015",
                                         "ST_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020"],
                           "requires_requests": True},
        "from_8_to_11": {"identified_next": ["POPOCATEPETL_PUEBLA_2017", "CHUUK_CHATAAN_2002"],
                         "unfilled_slot": "one new defensible African continental-rift volcanic Tier-2 landscape",
                         "claim": "NO FULLY FILE-BACKED ROUTE YET"},
        "stopping_rule_after_conservative": {"C1_11_GROUPS": False, "C2_3_LOCKBOXES": False,
                                             "C3_MORPHOLOGY": True, "C4_EROSION": True, "C5_CLIMATE": True,
                                             "C6_3_TECTONIC_WITH_NON_SUBDUCTION": False,
                                             "C7_TRIGGER": True, "C8_GEOGRAPHY": False,
                                             "C9_TIER2_LOLO": True, "C10_MAJOR_GAPS": False},
        "discovery_stop_condition": "A", "reason": "Two raw-ready candidates form a batch and both add rainfall/cyclone plus outside-Asia coverage",
    }
    S.write_json(S.OUT / "STAGE3E_TRAJECTORIES.json", trajectory)

    requests = """# Stage 3E request package — drafts only; not sent

## Saint Lucia–Tomas 2010

Could you provide the geospatial observation-support artifact used for the 3 January 2011 Hurricane Tomas landslide inventory: the exact systematically searched/visible polygon or original image footprint(s), together with machine-readable cloud, shadow, and cloud-fringe exclusions? Please also include metadata that links those geometries specifically to the 2011 Tomas epoch and states the reuse licence. The public landslide polygons themselves are already held.

## Vanuatu–Harold 2020

Could you provide the original GIS package formerly associated with NASA ArcGIS item `b6598e9b92bf4979a76e8fccee741ed2`: the 1,481 manually mapped landslide polygons, the mapped/search boundary, dataset provenance, CRS, and reuse licence? The public paper is already held; this request is only for the unavailable source dataset and observation domain.

## Popocatépetl/Puebla 2017 — Study Area C

Could you provide the Study Area C GIS artifacts described in DOI 10.1038/s41598-021-89015-8: all 754 landslide polygons, the exact approximately 365 km² study/search boundary, the manual cloud/shadow/snow exclusion mask, imagery dates and provenance, CRS, and reuse licence? The currently public Git repository and its reachable history contain the model/example materials but not these Study Area C files.

## Chuuk–Chata'an 2002 (conditional)

Could you clarify whether an independently defined, complete mapped/search domain exists for the original 2002 landslide inventory? If so, please provide that polygon and any visibility/coverage exclusions. The public USGS polygons are already held; a convex hull of the landslides is not suitable.
"""
    (S.OUT / "STAGE3E_REQUESTS.md").write_text(requests)

    diagnostics = {
        "stage": "3E", "production_count_unchanged": 4, "newly_discovered_candidates": 1,
        "new_public_machine_readable_data_collections": 3, "supporting_documents_downloaded": 3,
        "raw_files_registered": len(acquisitions), "next_batch_count": len(S.NEXT_BATCH),
        "saint_lucia_verdict": saint["verdict"], "vanuatu_verdict": vanuatu["verdict"], "popocatepetl_verdict": popo["verdict"],
        "african_volcanic_ready": False, "continental_rift_ready": False, "oceanic_hotspot_ready": False,
        "no_y_derived_omega_accepted": True, "dominica_maria_unchanged": True,
        "ml_trained": False, "lolo_run": False, "h_generated": False, "terrain_units_generated": False,
        "emails_sent": False, "committed": False, "pushed": False,
    }
    S.write_json(S.OUT / "STAGE3E_DIAGNOSTICS.json", diagnostics)

    report = _report(v2_tests, science_fair_tests, v1_verifier)
    (S.OUT / "STAGE3E_REPORT.md").write_text(report)
    artifact_names = [
        "STATE_VERIFICATION.json", "STAGE3E_RAW_ACQUISITIONS.tsv", "STAGE3E_CANDIDATE_AUDIT.tsv",
        "SAINT_LUCIA_RESCUE_AUDIT.json", "VANUATU_RECOVERY_AUDIT.json", "POPOCATEPETL_RECOVERY_AUDIT.json",
        "ATITLAN_GEOLOGY_AUDIT.json", "DOMINICA_ERIKA_AUDIT.json", "CHUUK_AUDIT.json",
        "STAGE3E_NEXT_BATCH_PRODUCTION.tsv", "STAGE3E_DIVERSITY_DASHBOARD.tsv", "STAGE3E_TRAJECTORIES.json",
        "STAGE3E_SEARCH_LOG.tsv", "STAGE3E_REQUESTS.md", "STAGE3E_DIAGNOSTICS.json", "STAGE3E_REPORT.md",
    ]
    checkpoint = {
        "stage": "3E", "as_of": S.AS_OF, "input_checkpoint_hashes": state["prior_checkpoint_hashes"],
        "raw_artifact_hashes": {row["relative_path"]: row["sha256"] for row in acquisitions},
        "production_state": {"landscapes": 4, "terrain_units": 737722, "positive_units": 6204},
        "verdicts": {"ST_LUCIA_TOMAS_2010": saint["verdict"], "VANUATU_HAROLD_2020": vanuatu["verdict"],
                     "POPOCATEPETL_PUEBLA_2017": popo["verdict"],
                     "DOMINICA_MARIA_2017": "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION"},
        "next_batch": [row["candidate_id"] for row in S.NEXT_BATCH], "projected_count_if_batch_passes": 6,
        "potential_lockboxes": ["ST_LUCIA_TOMAS_2010", "VANUATU_HAROLD_2020", "POPOCATEPETL_PUEBLA_2017"],
        "tests": {"relevant_v2": v2_tests, "science_fair_regression": science_fair_tests, "v1_verifier": v1_verifier},
        "ml_trained": False, "h_generated": False, "terrain_units_generated": False,
        "git": "DIRTY_WORKTREE_INSPECTED; NO_COMMIT; NO_PUSH",
        "artifact_hashes": {f"outputs/stage3e/{name}": S.sha256(S.OUT / name) for name in artifact_names},
    }
    S.write_json(S.OUT / "STAGE3E_CHECKPOINT.json", checkpoint)
    return checkpoint


def _report(v2_tests: str, science_fair_tests: str, v1_verifier: str) -> str:
    return f"""STAGE 3E SAINT LUCIA RESCUE + GLOBAL 4→8 SPRINT: COMPLETE

## A. EXECUTIVE RESULT

- Current production remains 4 landscapes, 737,722 terrain units and 6,204 positives.
- Saint Lucia: `SAINT_LUCIA_ONE_BLOCKER`. Vanuatu: `REQUEST_REQUIRED_PUBLIC_ITEM_REMOVED`. Popocatépetl: `REQUEST_REQUIRED_PROMISED_FILES_ABSENT`.
- One newly discovered candidate (Chuuk), three newly acquired public machine-readable data collections, three supporting documents, and nine registered raw files.
- Two candidates are raw-ready for the next production batch: Atitlán–Stan and Dominica–Erika. Neither has been admitted yet.

## B. SAINT LUCIA RESCUE

- The CC BY 4.0 World Bank inventory and BGS/CDEMA report agree on the Tomas epoch: the 3 January 2011 image contains 1,025 active and 3 inactive mapped landslides and is described as capturing all Tomas events.
- The report quantifies about 2% thick cloud, 5% cloud fringe and 10% shadow, but no machine-readable 2011 footprint, visibility exclusions, or exact searched polygon was recovered. The retired CHARIM layer name survives in service metadata, but the WFS now redirects and no archived feature payload was found.
- Omega is therefore not defensible. A whole-island assumption, Y hull/buffer, or traced figure was not used. The exact missing artifact is a Tomas-linked searched/visible polygon or imagery footprints plus cloud/shadow/fringe polygons.

## C. VANUATU

- The NASA paper confirms 1,481 whole-body polygons, 5 April 2020 PlanetScope mapping and a mapped boundary. The accepted manuscript was preserved.
- The cited ArcGIS item no longer exists or is inaccessible, and no legitimate public mirror was recovered. Y, Omega, provenance and reuse licence still require the precise custodian request in `STAGE3E_REQUESTS.md`.

## D. POPOCATÉPETL

- The open article documents Study Area C (754 polygons; about 365 km²) and manual cloud/shadow/snow masking, and says the inventory is delivered in its GitHub repository.
- All reachable branches, tags and commits were audited; Study Area C GIS files were never present. No alternate DOI repository was found. The exact inventory, boundary, mask, provenance/CRS and reuse licence must be requested.

## E. NEW AFRICA / RIFT RESULTS

- No defensible African volcanic or continental-rift candidate passed. Kivu labels remain mainly outside Virunga volcanic units and lack Omega; Elgon/Mbale and Hanang products are incomplete, preliminary or single-system; Zimbabwe–Idai is non-volcanic basement and Mulanje is intrusive.

## F. NEW SHIELD / HOTSPOT RESULTS

- Chuuk–Chata'an is scientifically valuable (Oceania, cyclone, oceanic hotspot, non-subduction), and official USGS Y was acquired. It is not production-ready: the metadata says the inventory is not complete/exhaustive, while the NHESS “mapped within” polygon is a Y-derived convex hull. Omega remains blocked.

## G. OTHER HIGH-VALUE DISCOVERIES

- Atitlán was rescued by an official MAGA national-geology vector: its independent 221.010068 km² boundary intersects only accepted Qp/Tv Cenozoic volcanic units in the rapid audit. Existing 6,817 Y polygons and February-2000 NASADEM complete its raw route.
- Dominica–Erika is separately recoverable from the NHESS archive: 1,548 event-coded polygons, three dated analysis extents and 1,983 cloud polygons yield a target-independent visible Omega of 778.223964 km². Maria remains blocked and was not reinterpreted.

## H. PUBLIC DATA ACQUIRED

- Machine-readable collections: NHESS supplementary archive, MAGA Guatemala geology, and USGS Chuuk event inventory/metadata.
- Supporting documents: Saint Lucia BGS/CDEMA report, Vanuatu NASA accepted manuscript, and Popocatépetl Scientific Reports article.
- URL/DOI, exact licence wording, timestamp, bytes and SHA256 for all nine raw files are in `STAGE3E_RAW_ACQUISITIONS.tsv`.

## I. NEXT BATCH PRODUCTION MANIFEST

1. `ATITLAN_STAN_2005` — use NASADEM 2000, not the post-event Copernicus DEM.
2. `DOMINICA_ERIKA_2015` — use event-specific NHESS Y/extent/cloud layers; georeference and audit the official geology map; keep Maria blocked.

## J. 4→8 TRAJECTORY

- Conservative: Stage 3F passes Atitlán and Erika → 6.
- Realistic: then recover Saint Lucia Omega and Vanuatu Y/Omega → 8, with three continents, four outside-Asia groups and four rainfall/cyclone groups.
- Best currently identified, conditional route reaches 10 if Popocatépetl and Chuuk are also rescued. There is no honest file-backed 8→11 route yet; the eleventh slot must be a new defensible African/rift volcanic landscape.

## K. CURRENT VS PROJECTED DIVERSITY

- Current production stays frozen at four Asian, earthquake-dominated groups.
- The two-candidate batch adds two North American rainfall/cyclone groups. Morphology, erosion and climate remain passed; trigger coverage becomes sufficient if both pass, but geography, group count, lockboxes, and tectonic breadth do not.
- All projected rows are explicitly conditional in `STAGE3E_DIVERSITY_DASHBOARD.tsv`.

## L. LOCKBOX CANDIDATES

- Three remain potential—not frozen—V1-unseen lockboxes: Saint Lucia, Vanuatu and Popocatépetl. Atitlán and Dominica are V1-historical. Chuuk is not promoted unless independent complete Omega is recovered.

## M. REMAINING STOPPING-RULE GAPS

- After the conservative batch: five more independent groups to reach 11; three diverse production-feasible V1-unseen lockboxes; at least three tectonic settings including non-subduction; at least three continents/four groups outside Asia; shield/hotspot and continental-rift production groups.
- Tier 1 remains excluded. It does not block a Tier-2-only experiment, whose LOLO group-count rule already passes.

## N. REQUESTS STILL REQUIRED

- Unsent exact-artifact drafts exist for Saint Lucia, Vanuatu, Popocatépetl and conditionally Chuuk. User action is to send those drafts to the named data custodians; no accounts, licence acceptance or emails were performed here.

## O. TESTS / CHECKPOINT / V1 / GIT

- Relevant V2 tests: {v2_tests}.
- Science Fair regression: {science_fair_tests}.
- V1 verifier: {v1_verifier}.
- Prior checkpoints and all nine downloaded raw files are checksum-verified. No terrain units, ML, LOLO, H, email, commit or push.

1. **Was Saint Lucia rescued?** Partially: `SAINT_LUCIA_ONE_BLOCKER`; its provenance and exact blocker were rescued, but Omega was not.
2. **Is Saint Lucia ready for production?** No. The exact Tomas observation-support geometry is missing.
3. **Was Vanuatu publicly recovered?** No; the paper was recovered, but the public Y/Omega item was not.
4. **Was Popocatépetl publicly recovered?** No; the article was recovered, but the promised GIS files are absent.
5. **Did we find a defensible African volcanic landscape?** No.
6. **Did we find a defensible continental-rift landscape?** No.
7. **Did we find a defensible oceanic shield/hotspot landscape?** Not production-ready. Chuuk fits scientifically, but independent complete Omega is absent.
8. **How many candidates now have Y + Omega + M + G sufficiently ready?** Two: Atitlán–Stan and Dominica–Erika.
9. **Which candidates should enter Stage 3F production, in order?** Atitlán–Stan first; Dominica–Erika second.
10. **If every Stage 3F candidate passes, what will the production landscape count become?** Six.
11. **How many potential lockboxes would we then have?** Three, still conditional and not formally frozen.
12. **What is the fastest realistic route from 4 to 8?** Stage 3F Atitlán and Erika (4→6), then recover and produce Saint Lucia and Vanuatu (6→8).
13. **What is the fastest realistic route from 8 to 11?** Recover Popocatépetl and an independent Chuuk Omega (8→10), then acquire one new African continental-rift volcanic Tier-2 landscape; this final step is not yet file-backed.
14. **Which frozen diversity criteria would still remain unsatisfied?** At six: 11 groups, three lockboxes, three tectonic settings with non-subduction, three continents/four outside Asia, and production shield/hotspot plus continental-rift coverage.
15. **Are there any user actions required right now?** Yes—send the unsent exact-artifact requests for Saint Lucia, Vanuatu and Popocatépetl; optionally ask USGS about complete Chuuk search support.
16. **Are we ready to train ML?** **NO.** The portfolio has 4 admitted groups (6 only if Stage 3F passes), not 11; it lacks three frozen lockboxes, tectonic/non-subduction breadth, the required geography, and production-ready shield/hotspot and rift groups.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-tests", default="PENDING")
    parser.add_argument("--science-fair-tests", default="PENDING")
    parser.add_argument("--v1-verifier", default="PENDING")
    args = parser.parse_args()
    build(args.v2_tests, args.science_fair_tests, args.v1_verifier)


if __name__ == "__main__":
    main()

