"""Build Stage 1 screening tables from discovery records (deterministic; no network, no downloads).

Run:  .venv/bin/python -m sfv2.build_stage1   (with PYTHONPATH=Science_Fair_ML_V2/src)
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from .discovery_records import RECORDS
from .inventory_manifest import build_inventory_rows
from .schema import COLUMNS, CONFIG, FIELDS, FUTURE_FIELDS, INVENTORY_COLUMNS, RATING_FIELDS, ROOT, SOURCE_COLUMNS, VOCAB
from .sources import all_sources

META = ROOT / "metadata"
SCREEN = ROOT / "outputs" / "screening"
CRITICAL = ("y_gate", "omega_gate", "m_gate")
OPEN = {"OPEN_DOWNLOAD", "VERIFIED_LOCAL_V1"}
NO_ACCESS = {"PAPER_ONLY", "NOT_FOUND", "UNKNOWN"}
SCHEMA_VERSION = "1.2.0"  # 1.2.0 (Stage 2A.3): volatile git state moved out of the scientific checkpoint
BUILDER_VERSION = "SFV2_STAGE1_BUILDER_1.2.0"
GIT_DIAGNOSTICS = "STAGE_1_GIT_DIAGNOSTICS.json"  # volatile, non-authoritative, never hashed
CHECKPOINT_TIMESTAMP = "2026-10-04T14:00:00Z"  # 2026-10-05 00:00 Australia/Brisbane; deterministic as-of time.


def score(ratings: dict[str, int]) -> float:
    comps = CONFIG["score_components"]
    return round(sum(comps[k]["weight"] * ratings[k] for k in comps) / 3, 1)


def priority(r: dict) -> tuple[str, str]:
    """Frozen priority rule (config/screening_config.json). Returns (priority_class, reason)."""
    if r["candidate_class"] == "REFERENCE_ONLY":
        return "REFERENCE_ONLY", "Methodological reference only"
    if r["candidate_class"] == "KNOWN_V1":
        return "KNOWN_V1_DECISION_PENDING", "V1 landscape; V2 role decided later"
    if r["volcano_type"] == "NON_VOLCANIC_REFERENCE":
        return "REJECT", "Outside V2 volcanic scope"
    if any(r[g] == "FAIL" for g in CRITICAL):
        return "REJECT", "Critical gate FAIL: " + ", ".join(g.upper() for g in CRITICAL if r[g] == "FAIL")
    if r["manual_priority_override"] in ("REJECT", "HOLD"):
        return r["manual_priority_override"], "Manual downgrade override"
    s, acc = r["screening_score"], r["data_accessibility"]
    if all(r[g] == "PASS" for g in CRITICAL) and acc in OPEN and s >= 55:
        return "PRIORITY_A", "All critical gates PASS, open access, score >= 55"
    if (r["y_gate"] in ("PASS", "UNCERTAIN") and r["omega_gate"] in ("PASS", "UNCERTAIN") and r["m_gate"] == "PASS"
            and acc not in NO_ACCESS and s >= 40):
        return "PRIORITY_B", "Gates PASS/UNCERTAIN, accessible route, score >= 40"
    if (r["diversity_value"] == "HIGH" and r["y_gate"] == "PASS" and r["omega_gate"] != "FAIL" and r["m_gate"] == "PASS"
            and acc in OPEN | {"OPEN_PORTAL_UNVERIFIED"}):
        return "PRIORITY_B", "Diversity promotion"
    return "HOLD", "Unresolved gate/access/score"


def build_rows() -> list[dict]:
    rows = []
    for rec in RECORDS:
        rec = dict(rec)
        rec["experimental_role"] = "UNASSIGNED"
        rec["lockbox_status"] = "UNASSIGNED"
        rec["lockbox_assignment_stage"] = "NOT_ASSIGNED"
        rec["lockbox_notes"] = "No experimental or final-lockbox assignment is permitted during Stage 1"
        ratings = dict(zip(CONFIG["score_components"], rec.pop("ratings")))
        for comp, col in RATING_FIELDS.items():
            rec[col] = ratings[comp]
        rec["screening_score"] = score(ratings)
        pc, why = priority(rec)
        rec["priority_class"] = pc
        rec["screen_status"] = CONFIG["status_from_priority"][pc]
        if pc in ("REJECT", "HOLD") and rec["exclusion_reason"] == "NONE":
            rec["exclusion_reason"] = why
        for f in FUTURE_FIELDS:
            rec[f] = ""
        missing = set(COLUMNS) - set(rec)
        extra = set(rec) - set(COLUMNS)
        if missing or extra:
            raise ValueError(f"{rec['landscape_id']}: missing {missing} extra {extra}")
        rows.append({c: rec[c] for c in COLUMNS})
    return rows


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        # Intentional projection: subset outputs must never rely on silent
        # extrasaction handling, which could conceal upstream schema mistakes.
        w.writerows({field: record.get(field, "") for field in cols} for record in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_context() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT.parent, check=True, capture_output=True, text=True).stdout.strip()

    status = run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "staged_entries": sum(1 for line in status if line[:1] not in (" ", "?")),
        "tracked_modified_entries": sum(1 for line in status if line[:2] != "??"),
        "untracked_entries": sum(1 for line in status if line.startswith("??")),
        "context_note": "Working tree contains pre-existing unrelated work; checkpoint does not imply a clean tree or commit.",
    }


def _classification_changes(rows: list[dict]) -> list[dict]:
    baseline_path = ROOT / "config" / "pre_patch_classification_baseline.json"
    baseline = json.loads(baseline_path.read_text())["candidates"]
    old = {row["landscape_id"]: row for row in baseline}
    changes = []
    for row in sorted(rows, key=lambda item: item["landscape_id"]):
        prior = old[row["landscape_id"]]
        changed = [field for field in ("m_gate", "omega_gate", "priority_class", "screen_status") if str(prior[field]) != str(row[field])]
        if changed:
            changes.append({
                "landscape_id": row["landscape_id"],
                "changed_fields": "|".join(changed),
                "old_m_gate": prior["m_gate"], "new_m_gate": row["m_gate"],
                "old_omega_gate": prior["omega_gate"], "new_omega_gate": row["omega_gate"],
                "old_priority_class": prior["priority_class"], "new_priority_class": row["priority_class"],
                "old_screen_status": prior["screen_status"], "new_screen_status": row["screen_status"],
                "reason": "Default DEM epoch validity is now UNCERTAIN unless explicitly supported; Omega evidence was audited without inventing sources",
            })
    return changes


def _counts(rows, key, mapper=None):
    c = Counter((mapper or (lambda v: v))(r[key]) for r in rows)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def _table(pairs, head=("Category", "Count")) -> str:
    return "\n".join([f"| {head[0]} | {head[1]} |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in pairs])


DIVERSITY_DIMS = [("continent", None), ("tectonic_setting", None), ("tectonic_broad", None), ("volcano_type", None), ("erosional_state", None),
                  ("climate_class", None), ("inventory_trigger", None), ("inventory_tier", None), ("h_quality", None), ("observation_domain_establishable", None)]


def diversity_audit(rows):
    out = []
    broad = VOCAB["tectonic_broad_class"]
    for dim, _ in DIVERSITY_DIMS:
        key = "tectonic_setting" if dim == "tectonic_broad" else dim
        mapper = (lambda v: broad[v]) if dim == "tectonic_broad" else None
        allc = dict(_counts(rows, key, mapper))
        actionable = [r for r in rows if r["priority_class"] in ("PRIORITY_A", "PRIORITY_B", "KNOWN_V1_DECISION_PENDING")]
        actc = dict(_counts(actionable, key, mapper))
        for cat in sorted(allc):
            out.append(dict(dimension=dim, category=cat, n_candidates=allc[cat], n_actionable=actc.get(cat, 0),
                            flag="UNDERREPRESENTED" if allc[cat] <= 3 or actc.get(cat, 0) == 0 else ""))
    return out


def main() -> None:
    rows = build_rows()
    sources = all_sources()
    inventories = build_inventory_rows(rows, sources)
    _write_csv(META / "landscape_candidates.csv", rows, COLUMNS)
    _write_csv(META / "landscape_data_dictionary.csv",
               [dict(field=f, group=g, vocabulary_or_type=v, required=str(req).upper(),
                     allowed_values="|".join(VOCAB[v]) if v in VOCAB and isinstance(VOCAB[v], list) else "", description=d)
                for f, g, v, req, d in FIELDS],
               ["field", "group", "vocabulary_or_type", "required", "allowed_values", "description"])
    _write_csv(META / "source_manifest.csv", sources, SOURCE_COLUMNS)
    _write_csv(META / "inventory_manifest.csv", inventories, INVENTORY_COLUMNS)

    short_cols = ["landscape_id", "continent", "tectonic_setting", "volcano_type", "inventory_trigger", "inventory_tier", "y_gate", "omega_gate", "m_gate",
                  "h_status", "data_accessibility", "screening_score", "priority_class", "screen_status", "major_limitations"]
    ordered = sorted(rows, key=lambda r: (["PRIORITY_A", "PRIORITY_B", "KNOWN_V1_DECISION_PENDING", "REFERENCE_ONLY", "HOLD", "REJECT"].index(r["priority_class"]),
                                          -r["screening_score"], r["landscape_id"]))
    _write_csv(SCREEN / "shortlist.csv", [r for r in ordered if r["priority_class"] in ("PRIORITY_A", "PRIORITY_B", "KNOWN_V1_DECISION_PENDING")], short_cols)
    _write_csv(SCREEN / "reject_hold.csv", [r for r in ordered if r["priority_class"] in ("HOLD", "REJECT", "REFERENCE_ONLY")], short_cols + ["exclusion_reason"])
    source_landscapes = {source["landscape_id"] for source in sources}
    unresolved = []
    for r in ordered:
        issues = []
        if r["y_gate"] != "PASS":
            issues.append("Y: inventory geometry/semantics/access unverified")
        if r["omega_gate"] != "PASS":
            issues.append("OMEGA: observation domain not established")
        if r["m_gate"] != "PASS":
            issues.append("M: DEM temporal suitability is unverified or unsuitable")
        if r["inventory_tier"] == "UNCLEAR":
            issues.append("TIER: source vs affected-terrain semantics unclear")
        if r["data_accessibility"] not in OPEN:
            issues.append(f"ACCESS: {r['data_accessibility']}")
        if r["landscape_id"] not in source_landscapes:
            issues.append("SOURCE: no source-manifest row; resolve during Stage 2")
        if r["omega_gate"] == "PASS" and not any(
            source["landscape_id"] == r["landscape_id"] and source["data_role"] == "OMEGA" and source["omega_obs_evidence"] != "UNKNOWN" for source in sources
        ):
            issues.append("OMEGA_SOURCE: no traceable source-manifest evidence link")
        if issues:
            unresolved.append(dict(landscape_id=r["landscape_id"], priority_class=r["priority_class"], data_accessibility=r["data_accessibility"],
                                   unresolved_issues="; ".join(issues), next_check=r["major_limitations"]))
    _write_csv(SCREEN / "unresolved_data.csv", unresolved, ["landscape_id", "priority_class", "data_accessibility", "unresolved_issues", "next_check"])
    audit = diversity_audit(rows)
    _write_csv(SCREEN / "diversity_audit.csv", audit, ["dimension", "category", "n_candidates", "n_actionable", "flag"])
    changes = _classification_changes(rows)
    change_columns = ["landscape_id", "changed_fields", "old_m_gate", "new_m_gate", "old_omega_gate", "new_omega_gate",
                      "old_priority_class", "new_priority_class", "old_screen_status", "new_screen_status", "reason"]
    _write_csv(SCREEN / "classification_changes.csv", changes, change_columns)

    # ----- markdown summaries
    lines = ["# Stage 1 global screening summary", "",
             f"Candidates screened: **{len(rows)}**. Generated by `src/sfv2/build_stage1.py` from `src/sfv2/discovery_records.py`. "
             "Discovery only - no inventory, DEM or geology was downloaded. Scores are workflow prioritisation only.", ""]
    for title, key, mapper in [("Priority class", "priority_class", None), ("Screen status", "screen_status", None), ("Continent", "continent", None),
                               ("Tectonic setting", "tectonic_setting", None), ("Tectonic broad class", "tectonic_setting", lambda v: VOCAB["tectonic_broad_class"][v]),
                               ("Volcano type", "volcano_type", None), ("Trigger", "inventory_trigger", None), ("Inventory tier", "inventory_tier", None),
                               ("Omega_obs establishable", "observation_domain_establishable", None), ("H quality", "h_quality", None),
                               ("Data accessibility", "data_accessibility", None), ("Y evidence stage", "y_evidence_stage", None)]:
        lines += [f"## {title}", "", _table(_counts(rows, key, mapper)), ""]
    lines += [
        "## Stage 1 completion-patch decisions", "",
        "- The default M gate is `UNCERTAIN` whenever DEM temporal suitability remains unverified. This removes all new candidates from the acquisition shortlist until Stage 2 binds suitable DEM epochs; it does not reject them.",
        "- `MATZACO_PUEBLA_2017` retains Omega PASS because frozen V1 contains the downloaded, hashed USGS mapping-extent shapefile.",
        "- `PUERTO_RICO_MARIA_2017` retains Omega PASS because frozen V1 documents the approximately 97% imagery footprint and mapping boundary; its exact digital boundary remains a Stage 2 binding task.",
        "- `COLUMBIA_RIVER_GORGE_BASALT` retains Omega PASS because RI-45 documents a bounded 900 square-mile study area; its exact digital boundary remains a Stage 2 check.",
        "- Every Omega PASS has an explicit role-specific source-manifest link. No observation domain is inferred from positive geometry.",
        "- All experimental and lockbox roles remain `UNASSIGNED`. `metadata/inventory_manifest.csv` is the inventory-level child table; landscape inventory fields remain screening summaries.", "",
    ]
    lines += ["## All candidates", "", "| landscape_id | continent | priority | status | score | Y | Omega | M | H | access |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['landscape_id']} | {r['continent']} | {r['priority_class']} | {r['screen_status']} | {r['screening_score']} | {r['y_gate']} | {r['omega_gate']} | "
              f"{r['m_gate']} | {r['h_status']} | {r['data_accessibility']} |" for r in ordered]
    (SCREEN / "global_screening_summary.md").write_text("\n".join(lines) + "\n")

    flagged = [a for a in audit if a["flag"]]
    dl = ["# Stage 1 diversity audit", "",
          "A category is flagged UNDERREPRESENTED when it has <= 3 candidates or no actionable (PRIORITY_A/B or KNOWN_V1) candidate.", "",
          "| dimension | category | candidates | actionable |", "| --- | --- | --- | --- |"]
    dl += [f"| {a['dimension']} | {a['category']} | {a['n_candidates']} | {a['n_actionable']} |" for a in flagged]
    dl += ["", "## Interpretation", "",
           "- Candidate discovery is geographically broad (all six inhabited continents). After the DEM-epoch correction, no new candidate is acquisition-actionable; "
           "the four shortlist records are known V1 landscapes retained for later role decisions.",
           "- Africa, continental rifts, South America and arid/high-altitude climates are well represented at DISCOVERY level but almost entirely PAPER_ONLY / "
           "NOT_FOUND: their inclusion depends on author data requests, not on open downloads.",
           "- NORTH_AMERICA (including Central America and the Caribbean) and SUBDUCTION_ARC dominate the discovery pool. Any later acquired V2 pool must reassess "
           "this imbalance before LOLO, OOD or lockbox assignment.",
           "- The global balance search (rifts, intraplate continental, arid, cold, Africa, South America) was performed before finishing; remaining gaps are real "
           "data-availability gaps, not unsearched categories."]
    (SCREEN / "diversity_audit.md").write_text("\n".join(dl) + "\n")

    # Validation runs after every generated table exists, and fails the build
    # rather than allowing a partially valid Stage-1 checkpoint.
    from .validate_stage1 import validate_stage1

    validation = validate_stage1(rows, sources, inventories)
    validation_path = SCREEN / "stage1_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if not validation["valid"]:
        raise ValueError("Stage-1 validation failed: " + "; ".join(validation["errors"]))

    authoritative_paths = [
        ROOT / "config" / "controlled_vocabularies.json",
        ROOT / "config" / "screening_config.json",
        ROOT / "config" / "pre_patch_classification_baseline.json",
        ROOT / "src" / "sfv2" / "build_stage1.py",
        ROOT / "src" / "sfv2" / "discovery_records.py",
        ROOT / "src" / "sfv2" / "inventory_manifest.py",
        ROOT / "src" / "sfv2" / "schema.py",
        ROOT / "src" / "sfv2" / "sources.py",
        ROOT / "src" / "sfv2" / "validate_stage1.py",
        META / "landscape_candidates.csv",
        META / "landscape_data_dictionary.csv",
        META / "source_manifest.csv",
        META / "inventory_manifest.csv",
        SCREEN / "shortlist.csv",
        SCREEN / "reject_hold.csv",
        SCREEN / "unresolved_data.csv",
        SCREEN / "diversity_audit.csv",
        SCREEN / "classification_changes.csv",
        SCREEN / "global_screening_summary.md",
        SCREEN / "diversity_audit.md",
        validation_path,
    ]
    hashes = {
        str(path.relative_to(ROOT)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(authoritative_paths)
    }
    priority_counts = dict(_counts(rows, "priority_class"))
    checkpoint = {
        "schema": "STAGE_1_SCREENING_CHECKPOINT",
        "checkpoint_timestamp_utc": CHECKPOINT_TIMESTAMP,
        "schema_version": SCHEMA_VERSION,
        "builder": {"version": BUILDER_VERSION, "path": "Science_Fair_ML_V2/src/sfv2/build_stage1.py"},
        "counts": {
            "candidates": len(rows), "sources": len(sources), "inventories": len(inventories),
            "acquisition_shortlist": sum(priority_counts.get(key, 0) for key in ("PRIORITY_A", "PRIORITY_B")),
            "shortlist_including_known_v1": sum(priority_counts.get(key, 0) for key in ("PRIORITY_A", "PRIORITY_B", "KNOWN_V1_DECISION_PENDING")),
        },
        "validation": {"valid": True, "error_count": 0, "warning_count": validation["warning_count"], "report": "outputs/screening/stage1_validation.json"},
        "authoritative_file_hashes": hashes,
        "volatile_diagnostics": {
            "path": f"outputs/screening/{GIT_DIAGNOSTICS}", "authoritative": False, "excluded_from_checkpoint_identity": True,
            "note": "Repository state (HEAD, branch, untracked counts) changes independently of the science; recorded separately since schema 1.2.0.",
        },
        "governance": {
            "checkpoint_not_freeze": True,
            "candidate_metadata_subject_to_stage2_refinement": True,
            "all_experimental_roles_unassigned": True,
            "ignored_files": ["metadata/*.csv (repository-wide *.csv rule)", "outputs/ (repository-wide outputs rule)"],
            "ignored_file_note": "Checkpoint hashes provide identity without force-adding ignored data. Future governance may narrow ignore rules or use an external manifest after owner review.",
        },
    }
    (SCREEN / GIT_DIAGNOSTICS).write_text(json.dumps({"schema": "STAGE_1_GIT_DIAGNOSTICS", "authoritative": False,
                                                    "git_context": _git_context()}, indent=2, sort_keys=True) + "\n")
    checkpoint_json = SCREEN / "STAGE_1_SCREENING_CHECKPOINT.json"
    checkpoint_json.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    checkpoint_md = [
        "# Stage 1 screening checkpoint", "",
        f"- Timestamp: `{CHECKPOINT_TIMESTAMP}`", f"- Schema version: `{SCHEMA_VERSION}`",
        f"- Builder: `{BUILDER_VERSION}` (`src/sfv2/build_stage1.py`)",
        f"- Candidates: **{len(rows)}**", f"- Sources: **{len(sources)}**", f"- Inventory child records: **{len(inventories)}**",
        f"- Validation: **PASS** ({validation['warning_count']} unresolved warnings)", "",
        "This is a reproducibility checkpoint, not a V2 dataset freeze. Candidate metadata remains subject to Stage-2 evidence refinement.", "",
        "Authoritative Stage-1 CSV files are ignored by the repository-wide `*.csv` rule and generated screening outputs by the repository-wide `outputs/` rule. "
        "Their byte sizes and SHA256 hashes are bound in the JSON checkpoint; no ignored file was force-added.", "",
        "All candidates remain `UNASSIGNED`; no development or final-lockbox role has been selected.",
    ]
    checkpoint_md_path = SCREEN / "STAGE_1_SCREENING_CHECKPOINT.md"
    checkpoint_md_path.write_text("\n".join(checkpoint_md) + "\n")


if __name__ == "__main__":
    main()
