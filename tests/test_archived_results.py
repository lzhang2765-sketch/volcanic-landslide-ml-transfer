from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_tsv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_dataset_identity() -> None:
    rows = read_tsv("results/tables/STAGE5A_DATASET_REGISTER.tsv")
    assert len(rows) == 6
    assert sum(int(row["N_UNITS"]) for row in rows) == 835_846
    assert sum(int(row["POSITIVES"]) for row in rows) == 8_793
    assert {row["DEVELOPMENT_EXPOSURE_STATUS"] for row in rows} == {"DEVELOPMENT_EXPOSED"}


def test_selected_model_and_headline_values() -> None:
    rows = read_tsv("results/tables/baseline/STAGE4EB_MODEL_COMPARISON.tsv")
    selected = next(row for row in rows if row["model"] == "LOGISTIC_L2")
    assert selected["feature_set"] == "M"
    assert math.isclose(float(selected["median_pr_auc"]), 0.0285931177553, abs_tol=1e-13)
    assert math.isclose(float(selected["median_pr_lift"]), 2.56665260507, abs_tol=1e-11)
    assert math.isclose(float(selected["worst_pr_lift"]), 1.03162698154, abs_tol=1e-11)
    assert selected["worst_landscape"] == "COTABATO_DAVAO_2019"


def test_selected_model_fold_range() -> None:
    rows = [
        row
        for row in read_tsv("results/tables/baseline/STAGE4EB_LOLO_RESULTS.tsv")
        if row["analysis"] == "PRIMARY" and row["model"] == "LOGISTIC_L2" and row["feature_set"] == "M"
    ]
    assert len(rows) == 6
    lifts = {row["held_out"]: float(row["pr_lift"]) for row in rows}
    assert min(lifts, key=lifts.get) == "COTABATO_DAVAO_2019"
    assert max(lifts, key=lifts.get) == "ATITLAN_STAN_2005"
    assert math.isclose(min(lifts.values()), 1.03162698154, abs_tol=1e-11)
    assert math.isclose(max(lifts.values()), 4.05232380304, abs_tol=1e-11)


def test_selection_and_integrity_guards() -> None:
    freeze = json.loads((ROOT / "configs/STAGE4EB_PRIMARY_SELECTION_FREEZE.json").read_text())
    assert freeze["decision"]["development_leader"] == "LOGISTIC_L2"
    assert freeze["decision"]["runner_up"] == "RANDOM_FOREST"
    for key in ("LOCKBOX_TOUCHED", "MEGH_TRAINED", "GEN3_MODIFIED"):
        assert freeze[key] == "NO"
    assert freeze["COTABATO_EQ1_PRIMARY"] == "YES"
    assert freeze["CUMULATIVE_COTABATO_ACCESSED_FOR_SELECTION"] == "NO"


def test_geological_history_remained_feasibility_only() -> None:
    rows = read_tsv("results/tables/geological_history/STAGE5G_RECONSTRUCTABILITY_SUMMARY.tsv")
    assert len(rows) == 6
    verified = [row for row in rows if row["verified_h_capable"] == "YES"]
    assert [row["landscape_id"] for row in verified] == ["KUMAMOTO_ASO_2016"]
    assert verified[0]["status"] == "VERIFIED_GEN3_PARTIAL"
    assert all(row["core_h_released"] == "NO" for row in rows if row not in verified)


def test_elevation_analysis_is_post_selection_and_does_not_replace_baseline() -> None:
    with (ROOT / "results/tables/elevation/ELEVATION_ABLATION_SUMMARY.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["config"]: row for row in csv.DictReader(handle)}
    assert rows["B0"]["feature_set"] == "M"
    assert rows["A1"]["delta_R_band"] == "LOWER_R"
    assert rows["A2"]["delta_R_band"] == "LOWER_R"
