"""Stage 2C builder: harmonisation outputs, governance records, readiness decisions and checkpoint (no ML, no H)."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from . import build_stage2b as b2
from . import stage2c_harmonise as H
from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage2c_geology import CLASSES, COMPOSITIONS, DOMAIN_RULE, VOLCANIC, aso_mapping
from .stage2c_governance import (E_EXCLUDED, E_FINAL, FEATURE_RULES, FEATURES, H_FEASIBILITY, KUMAMOTO_TRIGGER_DECISION, LABEL_RULES,
                                 REQUEST_PACKAGES, TERRAIN_UNITS, V1_OTHER_LANDSCAPES)

OUT = H.OUT
PRIOR = ["outputs/screening/STAGE_1_SCREENING_CHECKPOINT.json", "outputs/stage2a/STAGE2A_CHECKPOINT.json",
         "outputs/stage2a_extension/STAGE2A_EXTENSION_CHECKPOINT.json", "outputs/stage2a2/STAGE2A2_CHECKPOINT.json", "outputs/stage2b/STAGE2B_CHECKPOINT.json"]
ACTIVE = list(H.LANDSCAPES)


def _csv(path: Path, rows: list[dict], cols=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def verify_stage2c_raw(registry: list[dict]) -> list[str]:
    errors = []
    for e in registry:
        if e["source_id"] not in ("S2C_GSJ_SEAMLESS_V2_API", "V1_FREEZE_TRANSFER", "S2EXT_KURIKOMA_GSI") and e["acquisition_id"] != "ACQ_COP30_PRODUCT_HANDBOOK":
            continue
        p = ROOT / e["raw_path"]
        if not p.exists() or sha256(p) != e["sha256"]:
            errors.append(f"{e['acquisition_id']}: checksum mismatch or missing")
        if p.exists() and p.stat().st_mode & 0o222:
            errors.append(f"{e['acquisition_id']}: raw file writable")
        if e["source_id"] == "S2C_GSJ_SEAMLESS_V2_API" and e["data_role"] == "GEOLOGY":
            for t in json.loads(p.read_text())["tiles"]:
                tp = ROOT / t["file"]
                if sha256(tp) != t["sha256"] or tp.stat().st_mode & 0o222:
                    errors.append(f"tile {t['file']} checksum/immutability failure")
        if e["source_id"] == "V1_FREEZE_TRANSFER" and e["sha256"] != e.get("v1_sha256"):
            errors.append(f"{e['acquisition_id']}: transferred hash differs from V1")
    return errors


def aso_crosscheck(registry, kres) -> tuple[list[dict], dict]:
    e = next(e for e in registry if e["acquisition_id"] == "ACQ_KUMAMOTO_G_H")
    work = ROOT / "data" / "interim" / "_stage2c_aso"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name, path in b2._members(ROOT / e["raw_path"], "KUMAMOTO_ASO_2016", "stage2c_aso"):
        shutil.copy(path, work / Path(name).name)
    aso = gpd.read_file(work / "geo_A.shp")
    units = aso.drop(columns="geometry").drop_duplicates("SYMBOL").to_dict("records")
    amap = aso_mapping(units)
    elig = {m["source_unit_code"]: m["domain_eligible"] == "TRUE" for m in amap}
    grid = kres["grid_obj"]
    a = aso.to_crs(grid["crs"])
    footprint = rasterize([(g, 1) for g in a.geometry], out_shape=(grid["height"], grid["width"]), transform=grid["transform"], fill=0, dtype="uint8").astype(bool)
    aso_volc = rasterize([(g, 1) for g, s in zip(a.geometry, a["SYMBOL"]) if elig.get(s)], out_shape=footprint.shape, transform=grid["transform"],
                         fill=0, dtype="uint8").astype(bool) if any(elig.values()) else np.zeros(footprint.shape, bool)
    seam_volc = np.isin(kres["geo_cls"], list(VOLCANIC))
    n = int(footprint.sum())
    agree = int(((aso_volc == seam_volc) & footprint).sum())
    return amap, dict(aso_footprint_km2=round(n * 900 / 1e6, 3), volcanic_agreement_fraction=round(agree / n, 4) if n else None,
                      aso_volcanic_seamless_not=round(int((aso_volc & ~seam_volc & footprint).sum()) * 900 / 1e6, 3),
                      seamless_volcanic_aso_not=round(int((~aso_volc & seam_volc & footprint).sum()) * 900 / 1e6, 3),
                      role="Cross-check and future H control only; Seamless V2 is the common domain framework")


def readiness(lid: str, r: dict) -> dict:
    y = r["y"]
    if lid == "KUMAMOTO_ASO_2016":
        ok = y["primary_in_analysis"] > 0
        return dict(status="READY_FOR_TERRAIN_UNITS" if ok else "BLOCKED", target_family="POINT_OCCURRENCE_CENTRE (Tier 2)",
                    conditions=["Primary target = APRIL_EQ centres in April range; July/undated sensitivity only",
                                "Copernicus non-original DEM cells carried as mandatory flag; sensitivity run excluding them (9.9% of primary centres vs 1.9% of area)",
                                "Domain uncertainty: 1:200,000 Seamless vs 1:50,000 Aso map disagree on volcanic status over part of the Aso footprint "
                                "(see geology_summary.json); a 1:50,000-restricted domain is a required sensitivity"],
                    blockers=[] if ok else ["No primary Y in analysis domain"])
    if y["intersecting_primary_analysis"] == 0:
        return dict(status="BLOCKED", target_family="POLYGON_AFFECTED_AREA (Tier 2)", conditions=[],
                    blockers=["Authoritative 1:200,000 geology places all Y on non-volcanic (Neogene sedimentary / Quaternary surficial) units; the Cenozoic "
                              "volcanic analysis domain contains zero Y. Volcanic character is a tephra mantle not represented in bedrock geology. Requires "
                              "authoritative digital tephra-cover/isopach evidence to define a tephra-mantled domain, or an explicit user decision to admit "
                              "the landscape as a tephra-mantled sedimentary reference (not a volcanic-bedrock landscape)."])
    return dict(status="READY_FOR_TERRAIN_UNITS", target_family="POLYGON_AFFECTED_AREA (Tier 2)",
                conditions=["Copernicus non-original DEM cells carried as mandatory flag with exclusion sensitivity",
                            f"{y['total'] - y['intersecting_primary_analysis']} Y polygons fall wholly on non-volcanic geology and are outside the analysis domain"],
                blockers=[])


def main() -> dict:
    H.LINEAGE.clear()
    registry = load_registry()
    errors = verify_stage2c_raw(registry)
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    OUT.mkdir(parents=True, exist_ok=True)
    results, summary = {}, {}
    for lid in ACTIVE:
        results[lid] = H.harmonise(lid, registry)
    amap, aso_cc = aso_crosscheck(registry, results["KUMAMOTO_ASO_2016"])
    _json(OUT / "geology_ontology.json", dict(classes=CLASSES, volcanic_classes=sorted(VOLCANIC), compositions=COMPOSITIONS, domain_rule=DOMAIN_RULE,
                                              ambiguity_policy="Ambiguous units are not forced: lava+pyroclastic undivided -> MIXED_VOLCANIC; unmatched -> UNKNOWN"))
    for lid, r in results.items():
        _csv(OUT / f"geology_mapping_{lid}_seamless.csv", r["geology_mapping"])
    _csv(OUT / "geology_mapping_KUMAMOTO_ASO_2016_aso_1_50000.csv", amap)
    _json(OUT / "grid_specs.json", {lid: r["grid"] for lid, r in results.items()})
    deriv = [dict(landscape_id=lid, variant=v, step=s["step"], cells=s["cells"], km2=s["km2"]) for lid, r in results.items()
             for v, d in r["omega_analysis"].items() for s in d["steps"]]
    _csv(OUT / "omega_analysis_derivation.csv", deriv)
    _json(OUT / "y_harmonisation_summary.json", {lid: r["y"] for lid, r in results.items()})
    _json(OUT / "dem_quality_flags.json", dict(edm_legend=H.EDM_LEGEND, flm_legend=H.FLM_LEGEND, fill_source_epochs=H.FILL_SOURCE_EPOCH,
                                               legend_source="Copernicus DEM Product Handbook issue 5.0, Tables 6-7 (archived raw: ACQ_COP30_PRODUCT_HANDBOOK)",
                                               landscapes={lid: r["dem_quality"] for lid, r in results.items()}))
    _json(OUT / "pilot_terrain_units.json", {lid: r["pilot"] for lid, r in results.items()})
    _json(OUT / "geology_summary.json", {**{lid: r["geology"] for lid, r in results.items()}, "KUMAMOTO_ASO_1_50000_CROSSCHECK": aso_cc})
    (OUT / "KUMAMOTO_TRIGGER_DECISION.md").write_text(KUMAMOTO_TRIGGER_DECISION)
    (OUT / "V1_OTHER_LANDSCAPES_TRANSFER_STATUS.md").write_text(V1_OTHER_LANDSCAPES)
    _csv(OUT / "e_allowlist_final.csv", [dict(zip(["variable", "physical_meaning", "unit", "dataset_version", "resolution", "period", "licence", "relevance",
                                                    "temporal_leakage_risk", "missing_data_policy", "status"], x)) for x in E_FINAL])
    _csv(OUT / "e_excluded.csv", [dict(item=a, reason=b) for a, b in E_EXCLUDED])
    _csv(OUT / "terrain_unit_architecture.csv", [dict(option=a, decision=b, strengths=c, weaknesses=d) for a, b, c, d in TERRAIN_UNITS])
    _json(OUT / "terrain_unit_label_rules.json", LABEL_RULES)
    _csv(OUT / "feature_engine_spec.csv", [dict(feature=a, definition=b, unit=c, scales=d) for a, b, c, d in FEATURES])
    _json(OUT / "feature_engine_rules.json", FEATURE_RULES)
    _csv(OUT / "h_feasibility.csv", [dict(zip(["landscape_id", "likely_family", "available_controls", "expected_uncertainty", "major_missing_controls",
                                              "ensemble_feasible"], x)) for x in H_FEASIBILITY])
    rp = OUT / "request_packages"
    rp.mkdir(exist_ok=True)
    for pid, lid, study, who, missing, why, fmt, meta, lic, text, div in REQUEST_PACKAGES:
        (rp / f"{pid}_{lid}.md").write_text(f"# {pid} — {lid} (NOT SENT)\n\n- **Dataset/study:** {study}\n- **Custodian:** {who}\n"
                                            f"- **Exact missing artifact:** {missing}\n- **Why required:** {why}\n- **Preferred format:** {fmt}\n"
                                            f"- **Spatial metadata:** {meta}\n- **Licence/use question:** {lic}\n- **Diversity contribution:** {div}\n\n"
                                            f"## Draft request\n\n> {text}\n")
    (OUT / "COROMANDEL_GABRIELLE_ACCESS_RESULT.md").write_text(
        "# Coromandel / Cyclone Gabrielle access attempt — STOPPED\n\nGNS states that non-registered NZ Landslide Database users can view "
        "landslide points only, while registered users (free registration) can view all feature classes, including polygons and tables. "
        "Registration needs a personal account (interactive credentials/email). The authorisation forbids inventing credentials, so no account was created. "
        "The public sources found do not describe any vector download for registered users (viewing and PDF map reports only).\n\n"
        "**Exact user actions:** (1) register personally for the NZ Landslide Database webmap at "
        "https://www.gns.cri.nz/data-and-resources/new-zealand-landslide-database/ and check whether the Coromandel Gabrielle polygons can be "
        "exported as vectors together with the mapping-cell support; (2) if not, the only route is a data request to the named institutional custodian, "
        "which is external messaging and was not sent.\n")
    kt = OUT / "kurikoma_transfer_log.csv"
    with kt.open(newline="") as f:
        klog = list(csv.DictReader(f))
    status = {lid: readiness(lid, r) for lid, r in results.items()}
    status["KURIKOMA_2008"] = dict(status="BLOCKED", target_family="POLYGON_AFFECTED_AREA (Tier 2, Yagi et al. 2009)",
                                   conditions=["22 raw files transferred byte-identically; v2_unseen_test_eligibility = INELIGIBLE_V1_HISTORICAL"],
                                   blockers=["V2 Omega exists only as the official GSI PDF map (archived raw); needs controlled georeferencing/digitisation "
                                             "independent of Y", "Geology is the transferred national Seamless archive; it must be subset and mapped with the V2 ontology"])
    _json(OUT / "landscape_readiness.json", status)
    _json(OUT / "lineage.json", sorted(H.LINEAGE, key=lambda x: x["output"]))
    report = _report(results, status, aso_cc, klog)
    (OUT / "STAGE2C_REPORT.md").write_text(report)
    prior = {rel: sha256(ROOT / rel) for rel in PRIOR}
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith(("STAGE2C_CHECKPOINT", "STAGE2C_DIAGNOSTICS")))
    derived = sorted({x["output"]: x["output_sha256"] for x in H.LINEAGE}.items())
    cp = {"schema": "SFV2_STAGE2C_CHECKPOINT_V1", "as_of_date": H.AS_OF, "active_landscapes": ACTIVE,
          "readiness": {k: v["status"] for k, v in status.items()},
          "omega_analysis_km2": {lid: {v: d["analysis_km2"] for v, d in r["omega_analysis"].items()} for lid, r in results.items()},
          "derived_products": dict(derived), "output_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
          "stage2c_raw": {e["raw_path"]: e["sha256"] for e in sorted(registry, key=lambda e: e["raw_path"])
                          if e["source_id"] in ("S2C_GSJ_SEAMLESS_V2_API", "V1_FREEZE_TRANSFER", "S2EXT_KURIKOMA_GSI")
                          or e["acquisition_id"] == "ACQ_COP30_PRODUCT_HANDBOOK"},
          "prior_checkpoint_hashes": prior, "kurikoma_transfer": {"files": len(klog), "all_equal": all(r["equality"] == "PASS" for r in klog)},
          "volatile_diagnostics": {"path": "outputs/stage2c/STAGE2C_DIAGNOSTICS.json", "authoritative": False, "excluded_from_checkpoint_identity": True},
          "training": False, "h_reconstruction": False, "lockbox_assignment": False, "e_downloaded": False, "final_ml_table": False}
    _json(OUT / "STAGE2C_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    _json(OUT / "STAGE2C_DIAGNOSTICS.json", {"authoritative": False, "head": git("rev-parse", "HEAD"),
                                             "untracked_entries": len([l for l in git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if l.startswith("??")])})
    return cp


def _report(results, status, aso_cc, klog) -> str:
    L = ["# Stage 2C — core harmonisation", ""]
    for lid, r in results.items():
        prim = next(iter(r["omega_analysis"].values()))
        L += [f"## {lid}: {status[lid]['status']}", "", f"- OMEGA_ANALYSIS: {prim['analysis_km2']} km2 ({prim['retained_fraction']:.1%} of official Omega)",
              f"- Geology point-validation: {r['geology']['tile_qc']['point_validation']}", f"- Y: {json.dumps(r['y'], default=str)}",
              f"- DEM non-original fraction in domain: {r['dem_quality']['non_original_fraction']}",
              *[f"- Blocker: {b}" for b in status[lid]["blockers"]], *[f"- Condition: {c}" for c in status[lid]["conditions"]], ""]
    L += [f"Aso 1:50,000 cross-check: {aso_cc}", "", f"Kurikoma transfer: {len(klog)} files, all equal = {all(r['equality'] == 'PASS' for r in klog)}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
