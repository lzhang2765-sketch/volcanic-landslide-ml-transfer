"""Finalize the Stage 3D report and immutable, machine-readable checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import stage3d as S
from .schema import ROOT


OUT = ROOT / "outputs/stage3d"


def _read(name: str):
    return json.loads((OUT / name).read_text())


def _artifact_hashes() -> dict[str, str]:
    paths = [
        OUT / "STATE_VERIFICATION.json", OUT / "PRODUCTION_GATES.csv",
        OUT / "COTABATO_EVENT_FREEZE.json", OUT / "COTABATO_GEOMETRY_QA.json",
        OUT / "COTABATO_OMEGA_QA.json", OUT / "COTABATO_GEOLOGY_MAPPING.csv",
        OUT / "COTABATO_DEM_QA.json", OUT / "COTABATO_TARGET_QA.json",
        OUT / "DOMINICA_GEOMETRY_QA.json", OUT / "DOMINICA_OMEGA_QA.json",
        OUT / "DOMINICA_TARGET_SEMANTICS_AUDIT.json", OUT / "DOMINICA_GEOLOGY_MAPPING.csv",
        OUT / "DOMINICA_DEM_QA.json", OUT / "CROSS_LANDSCAPE_QA.csv",
        OUT / "CROSS_LANDSCAPE_FEATURE_RANGES.csv", OUT / "CROSS_LANDSCAPE_QA.json",
        OUT / "DIVERSITY_STOPPING_RULE_UPDATE.json", OUT / "RAW_PROVENANCE.tsv",
        OUT / "STAGE3D_DIAGNOSTICS.json",
        S.PROD / S.COT / "terrain_units/terrain_units.csv.gz",
        S.PROD / S.COT / "features_m/features_m.csv.gz",
        S.PROD / S.COT / "targets/targets_primary.csv.gz",
        S.PROD / S.COT / "populations/ALL_VALID.csv.gz",
        S.PROD / S.COT / "populations/STRICT_DEM_QUALITY.csv.gz",
        S.PROD / S.COT / "metadata/production_summary.json",
    ]
    return {str(p.relative_to(ROOT)): S.sha256(p) for p in paths}


def finalize(v2_tests: str, science_fair_tests: str, v1_verifier: str,
             broader_regression: str) -> dict:
    diagnostics = _read("STAGE3D_DIAGNOSTICS.json")
    cot = json.loads((S.PROD / S.COT / "metadata/production_summary.json").read_text())
    dom_target = _read("DOMINICA_TARGET_SEMANTICS_AUDIT.json")
    dom_omega = _read("DOMINICA_OMEGA_QA.json")
    diversity = _read("DIVERSITY_STOPPING_RULE_UPDATE.json")
    state = _read("STATE_VERIFICATION.json")
    qa = pd.read_csv(OUT / "CROSS_LANDSCAPE_QA.csv").set_index("landscape_id")
    tests = {
        "relevant_v2": v2_tests,
        "science_fair_regression": science_fair_tests,
        "broader_legacy_regression": broader_regression,
    }
    checkpoint = {
        "stage": "3D", "as_of": S.AS_OF,
        "input_checkpoint_hashes": state["prior_checkpoint_hashes"],
        "raw_artifact_hashes": {
            r.relative_path: r.sha256 for r in pd.read_csv(OUT / "RAW_PROVENANCE.tsv", sep="\t").itertuples(index=False)
        },
        "production_gate_outcomes": {
            S.COT: "PRODUCTION_READY", S.DOM: "BLOCKED_EVENT_ATTRIBUTION_AND_HAZARD_SEPARATION"
        },
        "terrain_unit_counts": {lid: int(row.terrain_units) for lid, row in qa.iterrows()},
        "positive_counts": {lid: int(row.positive_units) for lid, row in qa.iterrows()},
        "negative_counts": {lid: int(row.negative_units) for lid, row in qa.iterrows()},
        "feature_schema_hash": str(qa.iloc[0].feature_schema_hash),
        "feature_definition_hash": S.sha256(ROOT / "outputs/stage2d/feature_definitions.csv"),
        "production_landscape_list": list(S.PRODUCTION_LANDSCAPES),
        "production_landscape_count": 4,
        "production_terrain_units_total": int(qa.terrain_units.sum()),
        "production_positive_units_total": int(qa.positive_units.sum()),
        "diversity_status": diversity,
        "test_results": tests,
        "v1_verification": v1_verifier,
        "git_state": "DIRTY_WORKTREE_INSPECTED; NO_COMMIT; NO_PUSH",
        "ml_trained": False, "predictions_generated": False, "h_generated": False,
        "artifact_hashes": _artifact_hashes(),
    }
    S.write_json(OUT / "STAGE3D_CHECKPOINT.json", checkpoint)

    positives = {lid: int(row.positive_units) for lid, row in qa.iterrows()}
    report = f"""STAGE 3D MULTI-LANDSCAPE PRODUCTION: COMPLETE

## A. EXECUTIVE RESULT

- `{S.COT}`: **PRODUCTION_READY** — all ten mandatory gates pass.
- `{S.DOM}`: **BLOCKED** — the public product combines Erika 2015, Maria 2017 and flash-flood extents without an event field or supplied TYPE-code dictionary.
- Final production count: **4 independent landscapes**. Scientific gates prevented the planned count of five from being forced.

## B. COTABATO

- Source inventory: Zenodo 7520726 `landslides_ott2019`, kept separate from November and December cumulative inventories.
- Event date: 16 October 2019 (EQ1); mapped from 24 and 28 October imagery before EQ2.
- Y: 190 valid polygons, {cot['target_exclusion_audit']['source_y_area_km2']:.6f} km²; Tier-2 affected terrain.
- Omega: four published October study-area polygons, independently defined; 653.179743 km² raw and 634.611600 km² after geology/water/DEM rules.
- Geology: official Philippine MGB polygons; 97.1658% of raw Omega and 92.8218% of Y area match accepted explicit Cenozoic volcanic units.
- DEM: four Copernicus GLO-30 tiles acquired 2011–2014, before EQ1; EDM/FLM/WBM retained.
- Terrain units: {int(qa.loc[S.COT, 'terrain_units']):,} frozen 90 m units; 298 positive and 78,046 negative; prevalence {100*qa.loc[S.COT, 'prevalence']:.6f}%.
- ALL_VALID: 78,344 units. STRICT_DEM_QUALITY: 64,932 units (264 positive, 64,668 negative).
- Exclusions: 1,194 edge blocks failed the >=5/9 Omega rule; 13,412 included units are DEM-quality affected; 34 positive units are excluded only from the strict population. These remain documented, not corrected away.
- Gate verdict: **PASS / PRODUCTION_READY**.

## C. DOMINICA

- Source inventory: 10,551 whole boundaries and 22,333 source/transport/deposit parts from the open EO4Multihazards archive.
- Event date: nominal Maria date 18 September 2017, but the public product covers both Erika 2015 and Maria 2017.
- Y: **not admissibly event-separable**. Whole boundaries have no event/trigger field; image dates identify post-event imagery, not the causing event; TYPE codes are not defined in supplied metadata.
- Whole versus parts: whole boundaries would be the Tier-2 product if attribution were recoverable; semantic parts remain auxiliary and were not pooled.
- Omega: independent, date-aware union of each image footprint minus same-date clouds; {dom_omega['omega_obs_raw_km2']:.6f} km² raw, {dom_omega['omega_visible_km2']:.6f} km² visible, {dom_omega['retained_pct']:.4f}% retained. Cloud handling itself passes.
- Geology: official Dominica Government legend maps to the frozen ontology, but spatial volcanic-domain production was intentionally stopped after the target gate failed.
- DEM: Copernicus GLO-30 source dates 2011–2014, before Maria; DEM temporal gate passes.
- Terrain units / positives / negatives / populations: **not produced**.
- Gate verdict: **BLOCKED** by event attribution, hazard separation and therefore downstream target/domain/production gates.

## D. FEATURE QA

- Cotabato uses the unchanged frozen Stage-2D engine: 30 M arrays aggregated to 60 mean/SD columns.
- All 78,344 rows have all 60 features; missing cells: 0.
- Feature-definition SHA256: `{checkpoint['feature_definition_hash']}`.

## E. GEOLOGY QA

- Cotabato accepts only the frozen MGB tag/lithology rules; sedimentary and alluvial units are excluded rather than forced volcanic.
- Included Cotabato units: 45,578 MIXED_VOLCANIC and 32,766 VOLCANICLASTIC.
- Dominica’s official legend mapping is preserved, but no target-dependent spatial geology product was fabricated.

## F. DEM QUALITY QA

- Both DEM bundles predate their events and include source/quality companions.
- ALL_VALID and STRICT_DEM_QUALITY remain separate. Cotabato strict retention is {100*qa.loc[S.COT, 'strict_retention_fraction']:.2f}%.

## G. CROSS-LANDSCAPE QA

- All four actual production landscapes have identical 29-column terrain and 61-column feature-table schemas (ID + 60 features).
- Polygon-target schemas are identical for Noto, Lombok and Cotabato. Aso retains its scientifically appropriate point-target diagnostics while sharing the required relational key and target fields.
- Every production label remains Tier 2. No distribution shift was silently normalised or corrected.
- Material shifts: Cotabato adds young, high-altitude tropical-wet stratovolcanic-arc terrain; prevalence and DEM-quality retention vary substantially; geology-class mixtures and feature ranges differ genuinely across landscapes.

## H. UPDATED PRODUCTION DATASET

- Landscapes: **4** — Aso/Kumamoto, Noto, Lombok and Cotabato–Davao.
- Terrain units: **{int(qa.terrain_units.sum()):,}**.
- Positive units: **{int(qa.positive_units.sum()):,}**.
- Target families: Tier 2 only (point-event Tier 2 for Aso; polygon-affected Tier 2 for Noto, Lombok and Cotabato).
- Contributions: Aso {positives['KUMAMOTO_ASO_2016']:,}; Noto {positives['NOTO_PENINSULA_2024']:,}; Lombok {positives['LOMBOK_RINJANI_2018']:,}; Cotabato {positives[S.COT]:,}.

## I. UPDATED DIVERSITY COVERAGE

- Cotabato adds STRATOVOLCANIC_ARC, YOUNG_INTACT and TROPICAL_WET coverage.
- Frozen criteria newly satisfied: C3 morphology, C4 erosion and C5 climate. C9 target-family LOLO remains satisfied because Tier 2 is the sole admitted family with four development-capable groups (Lombok remains V1-historical/ineligible as a lockbox, not a hidden test group).
- The full stopping rule remains **not satisfied**.

## J. REMAINING STOPPING-RULE GAPS

- Seven additional independent groups to reach 11, including three genuinely V1-unseen, diverse production-ready lockboxes.
- At least two rainfall/cyclone groups; at least three tectonic settings including a defensible non-subduction setting.
- At least three continents and four groups outside Asia.
- Oceanic shield/hotspot and continental-rift terrain remain absent.
- Tier 1 remains excluded unless at least three valid development groups are produced.
- Next production rescue: Saint Lucia–Tomas 2010, specifically its missing visibility/search mask; pursue Dominica only if the custodian supplies an authoritative event/type crosswalk.

## K. TESTS / CHECKPOINT / V1 / GIT

- Relevant V2 tests: {v2_tests}.
- Science Fair regression: {science_fair_tests}.
- Broader legacy regression: {broader_regression}.
- V1 verifier: {v1_verifier}.
- Checkpoint: `outputs/stage3d/STAGE3D_CHECKPOINT.json`; all registered Stage-3C inputs and prior checkpoints verified unchanged.
- No ML, predictions, production H, email, commit or push.

1. **Did Cotabato pass every production gate?** Yes—all ten mandatory gates pass.
2. **Did Dominica pass every production gate?** No. Independent Omega and pre-event DEM pass, but event/hazard attribution does not; downstream production remains blocked.
3. **How many independent production landscapes do we now have?** Four.
4. **How many total production terrain units do we now have?** {int(qa.terrain_units.sum()):,}.
5. **How many positive units does each landscape contribute?** Aso 656; Noto 2,148; Lombok 3,102; Cotabato 298. Dominica contributes none because it was not admitted.
6. **Did either landscape require changing a frozen rule?** No.
7. **Are all five landscapes schema-compatible?** There are not five production landscapes. All four admitted landscapes are relationally compatible; their terrain and feature schemas match exactly, with the documented point-versus-polygon target diagnostics distinction.
8. **What major domain shifts are visible?** Cotabato adds young tropical-wet high-altitude arc terrain, a lower polygon-label prevalence than Noto/Lombok, different geology mixtures, and a relatively large DEM-quality-affected fraction.
9. **What diversity gaps remain?** Seven groups, three lockboxes, rainfall/cyclone replication, non-subduction and broader tectonics, two more continents/four outside-Asia groups, shield/hotspot and rift terrain, and three Tier-1 groups before Tier 1 can enter pooling.
10. **Which landscape should be processed next after Stage 3D?** Saint Lucia–Tomas 2010, after rescuing its visibility/search mask; Dominica requires an authoritative custodian crosswalk first.
11. **Are we ready to train ML?** **NO.** The frozen dataset-diversity stopping rule is not satisfied.
"""
    (OUT / "STAGE3D_REPORT.md").write_text(report)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-tests", required=True)
    parser.add_argument("--science-fair-tests", required=True)
    parser.add_argument("--v1-verifier", required=True)
    parser.add_argument("--broader-regression", required=True)
    args = parser.parse_args()
    finalize(args.v2_tests, args.science_fair_tests, args.v1_verifier, args.broader_regression)


if __name__ == "__main__":
    main()
