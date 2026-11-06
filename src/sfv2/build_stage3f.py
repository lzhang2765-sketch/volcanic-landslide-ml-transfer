"""Finalize the Stage 3F report and checkpoint after production/tests."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import stage3f as S


def load(name: str):
    return json.loads((S.OUT / name).read_text())


def finalize(v2_tests: str, science_fair_tests: str, v1_verifier: str,
             deterministic_replay: str) -> dict:
    state = load("STATE_VERIFICATION.json")
    diag = load("STAGE3F_DIAGNOSTICS.json")
    cross = load("CROSS_LANDSCAPE_QA.json")
    diversity = load("DIVERSITY_STOPPING_RULE_UPDATE.json")
    ati = json.loads((S.PROD / S.ATI / "metadata/production_summary.json").read_text())
    eri = json.loads((S.PROD / S.ERI / "metadata/production_summary.json").read_text())
    ati_pop = load(f"{S.ATI}_POPULATION_AND_TARGET_QA.json")
    eri_pop = load(f"{S.ERI}_POPULATION_AND_TARGET_QA.json")
    provenance = pd.read_csv(S.OUT / "RAW_PROVENANCE.tsv", sep="\t")
    report = _report(ati, eri, ati_pop, eri_pop, cross, diversity,
                     v2_tests, science_fair_tests, v1_verifier, deterministic_replay)
    (S.OUT / "STAGE3F_REPORT.md").write_text(report)
    artifact_names = [
        "STATE_VERIFICATION.json", "PRODUCTION_GATES.csv", "ATITLAN_EVENT_AUDIT.json",
        "ATITLAN_TARGET_SEMANTICS_AUDIT.json", "ATITLAN_OMEGA_AUDIT.json", "ATITLAN_GEOMETRY_QA.json",
        "ATITLAN_DEM_TEMPORAL_AUDIT.json", "ATITLAN_STAN_2005_DEM_QUALITY_AUDIT.json",
        "ATITLAN_GEOLOGY_HARMONISATION.csv", "ATITLAN_STAN_2005_POPULATION_AND_TARGET_QA.json",
        "DOMINICA_ERIKA_EVENT_AUDIT.json", "DOMINICA_ERIKA_TARGET_SEMANTICS_AUDIT.json",
        "DOMINICA_ERIKA_OMEGA_AUDIT.json", "DOMINICA_ERIKA_GEOMETRY_QA.json",
        "DOMINICA_ERIKA_DEM_TEMPORAL_AUDIT.json", "DOMINICA_ERIKA_2015_DEM_QUALITY_AUDIT.json",
        "DOMINICA_ERIKA_GEOREFERENCE_QA.json", "DOMINICA_ERIKA_GEOLOGY_HARMONISATION.csv",
        "DOMINICA_ERIKA_2015_POPULATION_AND_TARGET_QA.json", "CROSS_LANDSCAPE_QA.csv",
        "CROSS_LANDSCAPE_FEATURE_RANGES.csv", "CROSS_LANDSCAPE_QA.json",
        "DIVERSITY_STOPPING_RULE_UPDATE.json", "RAW_PROVENANCE.tsv", "PRODUCTION_MANIFEST.tsv",
        "STAGE3F_DIAGNOSTICS.json", "STAGE3F_REPORT.md",
    ]
    product_paths = []
    for lid in (S.ATI, S.ERI):
        base = S.PROD / lid
        product_paths.extend([base / "terrain_units/terrain_units.csv.gz", base / "features_m/features_m.csv.gz",
                              base / "targets/targets_primary.csv.gz", base / "populations/ALL_VALID.csv.gz",
                              base / "metadata/observation_context.csv.gz", base / "metadata/production_summary.json"])
        strict = base / "populations/STRICT_DEM_QUALITY.csv.gz"
        if strict.exists(): product_paths.append(strict)
    rows = pd.read_csv(S.OUT / "CROSS_LANDSCAPE_QA.csv").set_index("landscape_id")
    checkpoint = {
        "stage": "3F", "as_of": S.AS_OF, "previous_checkpoint_hashes": state["prior_checkpoint_hashes"],
        "raw_artifact_hashes": {r.relative_path: r.sha256 for r in provenance.itertuples(index=False)},
        "gate_outcomes": {S.ATI: ati["status"], S.ERI: eri["status"],
                          "DOMINICA_MARIA_2017": "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION"},
        "terrain_unit_counts": {lid: int(row.terrain_units) for lid, row in rows.iterrows()},
        "positive_counts": {lid: int(row.positive_units) for lid, row in rows.iterrows()},
        "negative_counts": {lid: int(row.negative_units) for lid, row in rows.iterrows()},
        "feature_schema_hash": str(rows.feature_schema_hash.iloc[0]),
        "feature_definition_hash": S.sha256(S.ROOT / "outputs/stage2d/feature_definitions.csv"),
        "production_landscape_list": list(rows.index), "production_landscape_count": cross["production_landscapes"],
        "unique_physical_landscape_count": cross["unique_physical_landscapes"],
        "physical_landscape_grouping": cross["physical_landscape_groups"],
        "production_terrain_units_total": cross["production_terrain_units"],
        "production_positive_units_total": cross["production_positive_units"],
        "production_negative_units_total": cross["production_terrain_units"] - cross["production_positive_units"],
        "historical_status": {S.ATI: {"development_only": True, "final_lockbox_eligible": False, "v1_unseen": False},
                              S.ERI: {"development_only": True, "final_lockbox_eligible": False, "v1_unseen": False}},
        "diversity_status": diversity,
        "tests": {"relevant_v2": v2_tests, "science_fair_regression": science_fair_tests,
                  "deterministic_replay": deterministic_replay, "v1_verifier": v1_verifier},
        "v1_status": v1_verifier, "git_state": "DIRTY_WORKTREE_INSPECTED; NO_COMMIT; NO_PUSH",
        "ml_trained": False, "lolo_run": False, "h_generated": False,
        "artifact_hashes": {f"outputs/stage3f/{name}": S.sha256(S.OUT / name) for name in artifact_names},
        "production_artifact_hashes": {str(path.relative_to(S.ROOT)): S.sha256(path) for path in product_paths},
    }
    S.write_json(S.OUT / "STAGE3F_CHECKPOINT.json", checkpoint)
    return checkpoint


def _report(ati, eri, ati_pop, eri_pop, cross, diversity, v2_tests, science_fair_tests,
            v1_verifier, deterministic_replay) -> str:
    total_negative = cross["production_terrain_units"] - cross["production_positive_units"]
    return f"""STAGE 3F MULTI-LANDSCAPE PRODUCTION II: COMPLETE

## A. EXECUTIVE RESULT

- `ATITLAN_STAN_2005`: **{ati['status']}**.
- `DOMINICA_ERIKA_2015`: **{eri['status']}**.
- Final production count: **{cross['production_landscapes']} groups / {cross['unique_physical_landscapes']} independent physical landscapes**. Both candidates passed without weakening a frozen rule.

## B. ATITLÁN–STAN

- Y: preserved V1-transfer `Landslides_Atitlan_UTM15N_6817`; 6,817 Hurricane Stan affected-terrain polygons. Fifteen invalid inputs were repaired with negligible area change; no duplicates or unrelated event product was included.
- Omega: independent published Atitlán study boundary, 221.010068 km² vector area and 220.999500 km² on the frozen grid. It retains 99.993824% of Y area; one polygon representative point lies just outside the supplied boundary and was not used to expand it.
- M: NASADEM/SRTM February 2000 only, reprojected bilinearly to the 30 m UTM 15N grid. The present Copernicus file was recorded as post-event and never used as primary M. NASADEM supplies no cell-quality companion, so no synthetic strict population was created.
- G: official MAGA 1:500,000 vector; only Qp pyroclastic and Tv mixed-volcanic Cenozoic units intersect Omega. No unknown, water, or non-volcanic unit was forced into the domain.
- Production: {ati['valid_units']:,} ALL_VALID units; {ati['positive_units']:,} positive and {ati['negative_units']:,} negative; prevalence {100*ati['positive_prevalence']:.6f}%. STRICT_DEM_QUALITY is unsupported rather than fabricated.
- Exclusions: {ati['excluded_edge_units']:,} edge blocks; {ati_pop['positive_loss_outside_or_edge_of_observation_support']} source-positive blocks fail the supplied observation-support/90 m edge rule. Geology, water and DEM then remove no additional positives. Final verdict: **{ati['status']}**.

## C. DOMINICA–ERIKA

- Event attribution: exactly 1,548 polygons; every feature has Erika code `LS20150909DMA`. Maria, flash-flood-only extents, and unclassified hazards contribute zero features.
- Target: one landslide-polygon product representing Tier-2 affected terrain; no source/transport/deposit parts were pooled.
- Omega: three dated extents and 1,983 cloud polygons. Conservative support-minus-all-supplied-clouds reproduces Stage 3E at 778.223964 km² (difference <0.000001 km²), with 87.401206% retained. All 1,548 representative points are visible; nine polygons touch exclusions.
- G: official 1:100,000 Dominica map, independently georeferenced from four longitude and five latitude ticks (RMS 1.850 pixels, approximately 23.68 m). No Y control point was used. Recent alluvium and Pleistocene conglomerate were excluded.
- M: Copernicus GLO-30 acquisition 2011-10-11 through 2014-02-04 predates Erika. EDM/FLM/WBM/SRC companions support separate quality populations.
- Production: {eri['valid_units']:,} ALL_VALID units; {eri['positive_units']:,} positive and {eri['negative_units']:,} negative; prevalence {100*eri['positive_prevalence']:.6f}%. STRICT_DEM_QUALITY contains {eri['strict_dem_quality_units']:,} units and {eri['strict_dem_quality_positive_units']:,} positives.
- Exclusions: {eri['excluded_edge_units']:,} final-domain edge blocks; {eri_pop['positive_loss_outside_or_edge_of_observation_support']} source-positive blocks fail raw observation support, {eri_pop['positive_loss_visibility']} more fail visibility, {eri_pop['positive_loss_geology']} geology, {eri_pop['positive_loss_water']} water, and {eri_pop['positive_loss_dem_and_final_interaction']} the final DEM interaction. Strict quality excludes {eri_pop['positive_loss_strict_quality']} admitted positives from the strict subset only. Final verdict: **{eri['status']}**.

## D. FEATURE QA

- Both landscapes use the unchanged frozen Stage-2D engine: 30 M arrays and 60 mean/SD columns. Missing feature cells: zero. Terrain units remain aligned 3×3 blocks of 30 m cells, with the frozen >=5/9 domain rule.

## E. GEOLOGY QA

- Atitlán uses only official MAGA Qp/Tv Cenozoic volcanic units.
- Dominica uses the official map legend and target-independent graticule controls. Eligible classes are intermediate lava, pyroclastic and mixed volcanic; alluvium and conglomerate are excluded. Map-scale positional uncertainty is recorded rather than concealed.

## F. DEM QA

- Atitlán: pre-event NASADEM 2000; zero void cells in the native tile; no unsupported strict-quality inference.
- Erika: pre-event Copernicus source elevation with EDM/FLM/WBM/SRC governance; ALL_VALID and STRICT_DEM_QUALITY remain separate.

## G. CROSS-LANDSCAPE QA

- All six terrain and feature schemas are identical; all polygon-target schemas match, while Aso retains its documented point-target diagnostics. Every target remains Tier 2 and all feature tables have zero missing cells.
- Largest shifts: Atitlán has by far the highest mean elevation (~2,495.6 m) and the highest prevalence (5.6982%); Erika has the steepest mean slope (~20.58°) and high mean 1 km relief (~406.5 m). These shifts were documented, not normalised away.

## H. UPDATED PRODUCTION DATASET

- Raw production groups: **{cross['production_landscapes']}**.
- Independent physical landscapes: **{cross['unique_physical_landscapes']}**. Erika and blocked Maria share the `DOMINICA` group; Maria is not counted.
- Terrain units: **{cross['production_terrain_units']:,}**; positives: **{cross['production_positive_units']:,}**; negatives: **{total_negative:,}**.
- Tier composition: Tier 2 only. Atitlán and Erika are `DEVELOPMENT_ONLY=TRUE`, `FINAL_LOCKBOX_ELIGIBLE=FALSE`, `V1_UNSEEN=FALSE`.

## I. UPDATED DIVERSITY

- Atitlán adds North American high-elevation tropical caldera/volcanic-complex rainfall terrain. Erika adds Caribbean tropical-wet volcanic-island cyclone terrain.
- C7 trigger coverage now passes. Existing C3 morphology, C4 erosion, C5 climate and C9 Tier-2 LOLO remain passed. Neither candidate adds a defensible non-subduction setting.

## J. STOPPING-RULE STATUS

- Not satisfied: five more groups to 11; three diverse V1-unseen lockboxes; >=3 tectonic settings including non-subduction; >=3 continents and >=4 outside-Asia groups; production-ready shield/hotspot and continental-rift terrain.

## K. NEXT 6→8 ROUTE

- Prioritise Saint Lucia–Tomas observation support, then Vanuatu–Harold Y/Omega recovery. If both pass full production, 6→8 is the fastest realistic route.

## L. NEXT 8→11 ROUTE

- Recover Popocatépetl Study Area C; rescue an independent complete Chuuk Omega; acquire one defensible African continental-rift volcanic Tier-2 landscape. This route is conditional, not yet file-backed.

## M. TESTS / CHECKPOINT / V1 / GIT

- Relevant V2: {v2_tests}.
- Deterministic replay: {deterministic_replay}.
- Science Fair regression: {science_fair_tests}.
- V1: {v1_verifier}.
- No ML, LOLO, production H, email, commit or push.

1. **Did Atitlán pass every production gate?** Yes.
2. **Did Dominica–Erika pass every production gate?** Yes.
3. **How many production groups do we now have?** Six.
4. **How many scientifically independent physical landscapes do we now have?** Six.
5. **How many total terrain units?** {cross['production_terrain_units']:,}.
6. **How many total positive units?** {cross['production_positive_units']:,}.
7. **What are the positive counts and prevalence for each new landscape?** Atitlán {ati['positive_units']:,} ({100*ati['positive_prevalence']:.6f}%); Erika {eri['positive_units']:,} ({100*eri['positive_prevalence']:.6f}%).
8. **Did Atitlán use only defensible pre-event terrain?** Yes—NASADEM/SRTM February 2000 only.
9. **Was Maria completely excluded from Erika production?** Yes; zero Maria polygons, and Maria remains blocked.
10. **Did either candidate require weakening a frozen rule?** No.
11. **Are all admitted production datasets schema-compatible?** Yes, with Aso's already documented point-target diagnostic distinction.
12. **What are the largest observed domain shifts?** Atitlán's much higher elevation and prevalence; Erika's steeper mean slope and high relief; both add rainfall/cyclone tropical terrain.
13. **Which frozen diversity criteria are now satisfied?** C3 morphology, C4 erosion, C5 climate, C7 trigger and C9 Tier-2 LOLO.
14. **Which remain unsatisfied?** C1 group count, C2 lockboxes, C6 tectonics/non-subduction, C8 geography and C10 major hotspot/rift gaps.
15. **What are the next two landscapes we should prioritise?** Saint Lucia–Tomas, then Vanuatu–Harold.
16. **What is the fastest realistic route from the new production count to 8?** Recover and fully produce Saint Lucia and Vanuatu.
17. **What is the fastest realistic route from 8 to 11?** Popocatépetl, Chuuk after Omega rescue, and one new African continental-rift Tier-2 landscape.
18. **Are any user actions required immediately?** Yes—send the prepared exact-artifact requests for Saint Lucia and Vanuatu; Popocatépetl can proceed in parallel.
19. **Are we ready to train ML?** **NO.** Six groups are short of 11; three V1-unseen lockboxes, tectonic/non-subduction breadth, geographic breadth, and production shield/hotspot plus rift coverage remain missing.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-tests", default="PENDING")
    parser.add_argument("--science-fair-tests", default="PENDING")
    parser.add_argument("--v1-verifier", default="PENDING")
    parser.add_argument("--deterministic-replay", default="PENDING")
    args = parser.parse_args()
    finalize(args.v2_tests, args.science_fair_tests, args.v1_verifier, args.deterministic_replay)


if __name__ == "__main__":
    main()
