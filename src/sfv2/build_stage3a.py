"""Build Stage 3A: global landscape expansion sprint (triage, public acquisition, human/machine action queues, stopping rule).

No ML, no H, no production landscape is generated here.  Outputs go to outputs/stage3a/; prior checkpoints are only read.
"""
from __future__ import annotations

import json
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from . import stage3a as S
from .fast_path import DECISION_GATES, REQUIRED_GATES, STEPS
from .raw_store import load_registry, sha256
from .schema import ROOT
from .stage2f_transfer import v1_verify
from .stage3a_acquire import ACQUISITIONS, acquire_all

OUT = ROOT / "outputs/stage3a"
STAGE2F2_CHECKPOINT_SHA = "5357dd80f3f5430cb7a34388c658d5d7d84794bd341c97587c652c7b1b2c7b91"
EXPECTED_PRODUCTION = {"count": 3, "total_units": 659378}
ML_SUFFIXES = (".pkl", ".joblib", ".onnx", ".pt", ".pth", ".h5", ".cbm")
ML_H_NAME_PATTERNS = ("shap", "predict", "surface", "paleo", "model")


def _json(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _tsv(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False, lineterminator="\n")


def _md_table(rows: list[dict], cols: list[str]) -> str:
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(esc(r.get(c, "")) for c in cols) + " |" for r in rows])


# ---------------------------------------------------------------- 1. state verification
def verify_state() -> dict:
    cps = sorted(p for p in ROOT.glob("outputs/*/STAGE*_CHECKPOINT.json") if p.parent.name != "stage3a")  # prior stages only
    cps.append(ROOT / "outputs/screening/STAGE_1_SCREENING_CHECKPOINT.json")
    checkpoints, mismatches = {}, []
    for cp in cps:
        d = json.loads(cp.read_text())
        checked = 0
        for key in ("output_hashes", "derived_hashes", "production_artifact_hashes", "prior_checkpoints", "authoritative_file_hashes"):
            for path, digest in (d.get(key) or {}).items():
                if not isinstance(digest, str) or len(digest) != 64:
                    continue
                checked += 1
                p = ROOT / path
                if not p.exists() or sha256(p) != digest:
                    mismatches.append(dict(checkpoint=str(cp.relative_to(ROOT)), path=path, expected=digest,
                                           actual=sha256(p) if p.exists() else "MISSING"))
        checkpoints[str(cp.relative_to(ROOT))] = dict(sha256=sha256(cp), hashes_checked=checked)
    summary = json.loads((ROOT / "outputs/stage2f2/landscape_state_summary.json").read_text())
    f2 = json.loads((ROOT / "outputs/stage2f2/STAGE2F2_CHECKPOINT.json").read_text())
    return dict(
        stage2f2_checkpoint_sha256=checkpoints["outputs/stage2f2/STAGE2F2_CHECKPOINT.json"]["sha256"],
        stage2f2_checkpoint_unchanged=checkpoints["outputs/stage2f2/STAGE2F2_CHECKPOINT.json"]["sha256"] == STAGE2F2_CHECKPOINT_SHA,
        checkpoints=checkpoints, hash_mismatches=mismatches,
        production_ready=summary["production_ready"], production_ready_count=summary["count"], total_units=summary["total_units"],
        production_matches_expected=summary["count"] == EXPECTED_PRODUCTION["count"] and summary["total_units"] == EXPECTED_PRODUCTION["total_units"],
        lombok_production_artifacts_verified=all(sha256(ROOT / p) == h for p, h in f2["production_artifact_hashes"].items()),
        lombok_gates=f2["gates"],
    )


def scan_ml_h_artifacts() -> list[str]:
    hits = []
    for base in (ROOT / "outputs", ROOT / "data/processed"):
        for p in base.rglob("*"):
            n = p.name.lower()
            if p.is_file() and (n.endswith(ML_SUFFIXES) or any(k in n for k in ML_H_NAME_PATTERNS)):
                hits.append(str(p.relative_to(ROOT)))
    return sorted(hits)


# ---------------------------------------------------------------- evidence audits on acquired public artifacts
def pr_geology_screen() -> dict:
    import geopandas as gpd
    reg = {e["acquisition_id"]: e for e in load_registry()}
    zp = ROOT / reg["ACQ_PUERTO_RICO_GEOLOGY_USGS_OFR98_38"]["raw_path"]
    lines = zipfile.ZipFile(zp).read("prunit.txt").decode("latin-1").splitlines()
    rows = [[c.strip() for c in ln.split("|")] for ln in lines[2:] if ln.count("|") >= 4]
    units = pd.DataFrame([r[:5] for r in rows], columns=["symbol", "name", "age", "descript", "ref"]).drop_duplicates("symbol")
    cen = r"eocene|paleocene|oligocene|miocene|pliocene|pleistocene|holocene|tertiary|quaternary|paleogene|neogene"
    vol = r"lava|tuff|volcanic breccia|volcaniclastic|pyroclastic|basalt|andesite|dacite|volcanic"
    age = units.age.str.lower()
    units["age_class"] = "CRETACEOUS_OR_OLDER"
    units.loc[age.str.contains(cen), "age_class"] = "CENOZOIC"
    units.loc[age.str.contains(cen) & age.str.contains("cretaceous"), "age_class"] = "K_T_STRADDLING"
    units["volcanic_text"] = (units.name + " " + units.descript).str.lower().str.contains(vol)
    g = gpd.read_file(f"zip://{zp}!PR_Geology.shp").to_crs(32619).merge(units, left_on="FMATN", right_on="symbol", how="left")
    g["age_class"] = g.age_class.fillna("UNMATCHED")
    g["volcanic_text"] = g.volcanic_text.fillna(False).astype(bool)
    g["km2"] = g.area / 1e6
    y = gpd.read_file(f"zip://{ROOT / 'data/raw/stage2e/PUERTO_RICO_MARIA_2017/V1_TRANSFER/Y/landslides.zip'}").to_crs(32619)
    j = gpd.sjoin(y, g[["age_class", "volcanic_text", "geometry"]], how="left", predicate="within")
    j["age_class"] = j.age_class.fillna("OUTSIDE_MAPPED_UNITS")
    j["volcanic_text"] = j.volcanic_text.fillna(False).astype(bool)
    area = {f"{a}|{'VOLCANIC_TEXT' if v else 'OTHER'}": round(float(x), 1) for (a, v), x in g.groupby(["age_class", "volcanic_text"]).km2.sum().items()}
    pts = {f"{a}|{'VOLCANIC_TEXT' if v else 'OTHER'}": int(x) for (a, v), x in j.groupby(["age_class", "volcanic_text"]).size().items()}
    return dict(status="SCREENING_ONLY_NOT_ONTOLOGY_MAPPING", source="ACQ_PUERTO_RICO_GEOLOGY_USGS_OFR98_38 (prunit.txt + PR_Geology.shp)",
                rule="Frozen DOMAIN_RULE: volcanic class AND Cenozoic age; keyword screen on unit name/description is an UPPER BOUND",
                mapped_area_km2=round(float(g.km2.sum()), 1), area_km2_by_age_and_text=area, headscarp_points_by_age_and_text=pts,
                headscarps_total=len(y), cenozoic_volcanic_text_upper_bound_km2=area.get("CENOZOIC|VOLCANIC_TEXT", 0.0),
                implication="Most failures sit on Cretaceous arc volcanics that the frozen rule treats as basement; the eligible PR domain is a "
                            "Paleocene-Eocene volcaniclastic subset that needs unit-by-unit harmonisation (e.g. conglomerates with volcanic "
                            "clasts and intrusive dacite must not be counted as volcanic surface units)")


def unosat_audit() -> dict:
    import geopandas as gpd
    reg = {e["acquisition_id"]: e for e in load_registry()}
    spec = {"ACQ_KALEHE_UNOSAT_FL20230508COD_SHP": ("FL20230508COD_SHP/S2_20230507_", "_KaleheTerritory_COD", "CloudObstruction", 32735),
            "ACQ_TALAKMAU_UNOSAT_LS20220308IDN_SHP": ("LS20220308IDN_SHP/S2_20220304_", "_MountTalakmau", "CloudObsruction", 32647)}
    out = {}
    for acq, (pre, suf, cloud, epsg) in spec.items():
        zp = ROOT / reg[acq]["raw_path"]
        rd = lambda n: gpd.read_file(f"zip://{zp}!{pre}{n}{suf}.shp").to_crs(epsg)
        ext, cld, ls = rd("AnalysisExtent"), rd(cloud), rd("LandslideExtent")
        e, c = ext.union_all(), cld.union_all()
        out[acq] = dict(analysis_extent_km2=round(e.area / 1e6, 3), cloud_obstruction_km2=round(e.intersection(c).area / 1e6, 3),
                        visible_analysis_km2=round(e.difference(c).area / 1e6, 3), landslide_extent_km2=round(ls.area.sum() / 1e6, 3),
                        landslide_features=len(ls), landslide_outside_extent_km2=round(ls.difference(e).area.sum() / 1e6, 4),
                        confidence=str(ls.iloc[0].get("Confidence")), field_validation=str(ls.iloc[0].get("FieldValid")),
                        sensor=str(ls.iloc[0].get("SensorID")), sensor_date=str(ls.iloc[0].get("SensorDate"))[:10],
                        omega_candidate="AnalysisExtent minus CloudObstruction (independent of Y)",
                        y_semantics="Dissolved rapid-mapping landslide extent (POLYGON_AFFECTED, preliminary; requires Y-quality decision)")
    out["ACQ_TALAKMAU_UNOSAT_LS20220308IDN_SHP"]["sensor_date_resolution"] = (
        "SensorDate attribute 2022-04-03 is a day/month swap: layer names and HDX state 2022-03-04 and the shapefile metadata was "
        "created 2022-03-08; image date taken as 2022-03-04 (7 days after the 2022-02-25 Mw 6.1 earthquake)")
    out["ACQ_TALAKMAU_UNOSAT_LS20220308IDN_SHP"]["cop30_tiles_required"] = ["N00_E099", "N00_E100", "S01_E099", "S01_E100"]
    return out


# ---------------------------------------------------------------- 3. gaps
GAP_DEFS = [
    ("SHIELD_OR_HOTSPOT_SYSTEMS", lambda r: r["morphology"] == "SHIELD" or r["tectonic_setting"] in ("HOTSPOT_OCEAN_ISLAND", "CONTINENTAL_HOTSPOT"), 2, True,
     "No basaltic shield or hotspot landscape; the parent Tweed application is an eroded continental-hotspot shield, so transfer to it is untestable"),
    ("OLD_DEEPLY_ERODED_TERRAIN", lambda r: r["erosion_state"] in ("STRONGLY_ERODED", "ANCIENT_REMNANT"), 3, True,
     "Only Noto represents strongly eroded volcanic terrain; erosion state is the core axis for Tweed-like transfer"),
    ("YOUNG_INTACT_EDIFICES", lambda r: r["erosion_state"] == "YOUNG_INTACT", 1, True, "No intact constructional edifice: opposite end of the erosion axis"),
    ("STRATOVOLCANO", lambda r: r["morphology"] == "STRATOVOLCANO", 2, True, "The most common volcano type worldwide is absent"),
    ("CALDERA_DOMINATED", lambda r: r["morphology"] == "CALDERA", 2, False, "Single caldera landscape (Aso)"),
    ("CONTINENTAL_RIFT", lambda r: r["tectonic_setting"] == "CONTINENTAL_RIFT", 1, True, "No rift volcanism"),
    ("TROPICAL_WET", lambda r: r["climate"] == "TROPICAL_WET", 2, True, "Weathering regime with deep saprolite absent"),
    ("DRY_SEMI_ARID", lambda r: r["climate"] == "DRY_SEMI_ARID", 1, False, "No dry volcanic terrain (few event inventories exist)"),
    ("HIGH_ALTITUDE", lambda r: r["climate"] == "HIGH_ALTITUDE_ALPINE", 1, False, "No high-altitude volcanic terrain"),
    ("AFRICA", lambda r: r["continent"] == "AFRICA", 1, True, "No African landscape"),
    ("AMERICAS", lambda r: r["continent"] in ("NORTH_AMERICA", "SOUTH_AMERICA"), 2, True, "No landscape in the Americas"),
    ("EUROPE", lambda r: r["continent"] == "EUROPE", 1, False, "No European landscape"),
    ("OCEANIA", lambda r: r["continent"] == "OCEANIA", 1, True, "No Oceania landscape"),
    ("OCEANIC_ISLANDS", lambda r: r["tectonic_setting"] == "HOTSPOT_OCEAN_ISLAND", 1, True, "No oceanic hotspot island"),
    ("RAINFALL_TRIGGERED", lambda r: r["trigger"] in ("EXTREME_RAINFALL", "PROLONGED_RAINFALL"), 2, True, "All production triggers are earthquakes"),
    ("CYCLONE_TRIGGERED", lambda r: r["trigger"] == "CYCLONE_STORM", 1, True, "No cyclone/storm trigger"),
    ("BASALTIC_LITHOLOGY", lambda r: r["material"] == "BASALTIC", 2, True, "No basalt-dominated material system"),
    ("ANDESITIC_LITHOLOGY", lambda r: r["material"] == "ANDESITIC", 1, False, "Andesitic arc material not explicitly represented"),
    ("TIER_1_TARGETS", lambda r: r["target_tier"] in ("TIER_1", "BOTH_AVAILABLE"), 1, False, "No Tier-1 (source/headscarp) target family"),
]


def gaps(matrix: list[dict], ranking: list[dict]) -> list[dict]:
    tiers = {r["landscape_id"]: r["priority_tier"] for r in ranking}
    out = []
    for cat, pred, desired, major, why in GAP_DEFS:
        prod = sorted({r["landscape_group"] for r in matrix if r["production_ready"] and pred(r)})
        fillers = [f"{r['landscape_id']}({tiers[r['landscape_id']]})" for r in matrix
                   if not r["production_ready"] and tiers[r["landscape_id"]] in ("A", "B") and pred(r)]
        out.append(dict(category=cat, major_gap=major, current_production_groups=len(prod), current_landscapes="|".join(prod) or "NONE",
                        desired_minimum_groups=desired, desired_additional=max(desired - len(prod), 0), scientific_reason=why,
                        candidate_fillers_tier_A_B="|".join(sorted(fillers)) or "NONE_IN_TIER_A_OR_B"))
    return out


# ---------------------------------------------------------------- 16. stopping rule
STOPPING_RULE = dict(
    rule_id="SFV2_DATASET_DIVERSITY_STOPPING_RULE_V1", frozen_on=S.AS_OF, frozen_before_ml_optimisation=True,
    unit_of_count="independent landscape group (stage3a.GROUPS); extra events or inventories inside a group never add N",
    criteria={
        "C1_INDEPENDENT_N": ">= 11 production-ready independent groups (>= 8 development + 3 reserved lockboxes)",
        "C2_LOCKBOXES": ">= 3 production-ready POTENTIAL_LOCKBOX groups, V1-unseen, each differing from every other lockbox and from the "
                        "development set in >= 2 coverage axes",
        "C3_MORPHOLOGY": ">= 4 morphology classes", "C4_EROSION": ">= 3 erosion states incl. YOUNG_INTACT and STRONGLY_ERODED/ANCIENT_REMNANT",
        "C5_CLIMATE": ">= 3 climate classes incl. TROPICAL_WET and a TEMPERATE class",
        "C6_TECTONIC": ">= 3 tectonic settings incl. >= 1 non-subduction setting",
        "C7_TRIGGER": ">= 2 earthquake groups and >= 2 rainfall/cyclone groups", "C8_GEOGRAPHY": ">= 3 continents and >= 4 groups outside Asia",
        "C9_TARGET_FAMILY_LOLO": "every target family used in pooled LOLO has >= 3 development groups",
        "C10_MAJOR_GAPS": "every MAJOR gap is filled or documented infeasible after its Tier A/B routes are exhausted"},
    early_acceptance=">= 15 groups: C3-C8 shortfalls may be accepted only if each is documented as infeasible",
    explicitly_not_criteria=["terrain-unit count", "positive count or prevalence", "any model performance"],
)


def evaluate_stopping_rule(rows: list[dict], gap_rows: list[dict]) -> dict:
    groups = {}
    for r in rows:
        groups.setdefault(r["landscape_group"], r)
    g = list(groups.values())
    lock = [r for r in g if r["lockbox_status"] == "POTENTIAL_LOCKBOX"]
    nonsub = {r["tectonic_setting"] for r in g} - {"SUBDUCTION_ARC", "OTHER_UNCERTAIN"}
    fam = Counter(r["target_tier"] for r in g)
    rain = sum(r["trigger"] in ("EXTREME_RAINFALL", "PROLONGED_RAINFALL", "CYCLONE_STORM") for r in g)
    checks = {
        "C1_INDEPENDENT_N": len(g) >= 11, "C2_LOCKBOXES": len(lock) >= 3,
        "C3_MORPHOLOGY": len({r["morphology"] for r in g} - S.UNKNOWNS) >= 4,
        "C4_EROSION": "YOUNG_INTACT" in {r["erosion_state"] for r in g} and bool({"STRONGLY_ERODED", "ANCIENT_REMNANT"} & {r["erosion_state"] for r in g})
                      and len({r["erosion_state"] for r in g} - S.UNKNOWNS) >= 3,
        "C5_CLIMATE": "TROPICAL_WET" in {r["climate"] for r in g} and bool({"TEMPERATE_WET", "TEMPERATE_SEASONAL"} & {r["climate"] for r in g})
                      and len({r["climate"] for r in g} - S.UNKNOWNS) >= 3,
        "C6_TECTONIC": len({r["tectonic_setting"] for r in g} - S.UNKNOWNS) >= 3 and bool(nonsub),
        "C7_TRIGGER": sum(r["trigger"] == "EARTHQUAKE" for r in g) >= 2 and rain >= 2,
        "C8_GEOGRAPHY": len({r["continent"] for r in g}) >= 3 and sum(r["continent"] != "ASIA" for r in g) >= 4,
        "C9_TARGET_FAMILY_LOLO": all(n >= 3 for n in fam.values()) if fam else False,
        "C10_MAJOR_GAPS": all(x["current_production_groups"] >= x["desired_minimum_groups"] for x in gap_rows if x["major_gap"]),
    }
    return dict(independent_groups=len(g), groups=sorted(groups), potential_lockboxes=len(lock), checks=checks, satisfied=all(checks.values()))


# ---------------------------------------------------------------- 8. rescue tracks
RESCUE = [
    ("8A", "PUERTO_RICO_MARIA_2017", "OMEGA NOT RECOVERED AS GEOMETRY; DECISION ROUTE IDENTIFIED",
     "Metadata documents an island-wide systematic search (FEMA ~97% + DigitalGlobe ~99% of main island, overlaps used for cloud/shadow, "
     "small islands not examined). No footprint geometry is public: FEMA S3 prefix Others/Maria/ now lists empty; NOAA tiles cover ~10%; "
     "DigitalGlobe Open Data footprints unavailable. Bounding box/hull/buffer/plain island boundary NOT used. New public G acquired "
     "(USGS OFR 98-38): frozen Cenozoic rule removes Cretaceous arc volcanics (3,040 km2, 24,716 headscarps); eligible Paleocene-Eocene "
     "volcanic-text units <= 940 km2 (~14,860 headscarps, upper bound). M: COP30 pre-event.",
     "D-PR-OMEGA decision (accept the documented main-island search scope minus COP30 water as Omega_obs, with residual-visibility caveat) "
     "or obtain coverage documentation from USGS; then unit-by-unit geology harmonisation"),
    ("8B", "BANKS_PENINSULA_2021", "STILL REQUEST-ONLY",
     "Publisher record states data available on reasonable request; no repository, thesis or supplement deposit found (re-checked "
     "2026-10-06). Macrostrat confirms Akaroa Volcanics (Tortonian basalt/trachyte) -> Cenozoic domain OK; loess mantle is a domain risk.",
     "Send one request to the corresponding author (Katherine Yates) for source/runout polygons + 115 km2 mapping boundary"),
    ("8C", "COROMANDEL_GABRIELLE_2023", "PUBLIC EVENT ROUTE FOUND BUT NON-VOLCANIC",
     "Massey et al. 2025 + OSF 8sbrj (CC BY 4.0, 2.0 GB 16 m point table) give an event-specific systematic Omega (337 priority cells) "
     "but its GeolCodes are Quaternary soils, Neogene/Cretaceous-Paleogene sedimentary, intrusive and Torlesse only -> no volcanic terrain; "
     "the 2.0 GB file was NOT downloaded (only a 3 kB header range). Full V1.0 inventory (>160,000 polygons, 12,219 priority cells) not public.",
     "Request V1.0 polygons + priority-cell grid for Coromandel Peninsula and Waitakere Ranges from GNS (batched)"),
    ("8D", "PNG_PORGERA_2018", "HOLD",
     "Tanyas et al. 2022 inventory public (Tier 2 polygons); its mapping_extent is described as encompassing all landslides (Y-derived) -> "
     "FAIL as Omega; volcanic share of fold-thrust-belt epicentral area uncertain; COP30 pre-event.",
     "Optional request for an independent searched/visibility mask; not worth active effort before Tier A/B items"),
    ("8E", "KIVU_RIFT_NTK_2000_2019", "REQUEST REQUIRED",
     "Public Zenodo 5027004 points (2000-2019, Google Earth, CC BY 4.0) have no year-specific search mask; multi-temporal labels make "
     "pre-event M undefined; NTK region is largely Precambrian with Neogene-Quaternary basalt subsets. High rift/Africa diversity.",
     "Request year-specific imagery/search mask + source polygons from RMCA (batched with Kalehe)"),
    ("8F", "MOUNT_ELGON_UGANDA", "STOP (Omega impossible by construction)",
     "Open PDF of Broeckx et al. 2019: inventory from field surveys + Google Earth; authors 'do not consider it to be complete for the "
     "entire study area' (areas never or once surveyed); multi-decade, partly undated -> no observation domain, no pre-event M.",
     "Stop; an email cannot recreate a systematic search. MBALE_ELGON_2022 remains rejected (same landscape group)"),
    ("8G", "GUATEMALA_1976", "STOP (temporal)",
     "Inventory public (Harp et al. 1981) and held in V1 SF1_verification (V1-historical reference). Earliest DEMs (SRTM 2000, COP30 "
     "2011-2014) post-date the event by 24-38 years; modern morphology includes 1976 scars and 50 years of change -> violates V2 "
     "pre-event M rule; not waived.", "Stop"),
    ("8H", "KIVU_KALEHE_RIFT_FLANK", "NEW PUBLIC ROUTE, WEAK",
     "UNOSAT FL20230508COD acquired (CC BY-SA): analysis extent 144.8 km2, cloud 54.0 km2 (37%), one dissolved 0.9 km2 landslide polygon "
     "from 10 m Sentinel-2, not field-validated; Kalehe flank largely Precambrian (volcanic share unverified). NHESS 2023 inventory "
     "still request-only.", "Hold; include Kalehe in the RMCA batch request only"),
    ("8I", "MIRAVALLES_OTTO_2016", "REQUEST REQUIRED (trigger separation)",
     "942 landslides from a compound Bijagua Mw 5.4 (Jul 2016) + Hurricane Otto (Nov 2016) sequence; no public polygons with trigger "
     "attribution or mapped area. Would add Costa Rica, stratovolcano, cyclone trigger; V1-unseen -> potential lockbox.",
     "Request trigger-attributed polygons + mapped area; never pool the two triggers silently"),
    ("8J", "ATITLAN_STAN_2005", "STILL BLOCKED (geology)",
     "MAGA Solola soil-study PDFs (cite four IGN 1:50,000 geology quadrangles) return HTTP 403 to scripts; Macrostrat returns no unit "
     "at 14.69N 91.19W. Exact missing artifact: authoritative geology polygons or full-resolution, coordinate-supported IGN 1:50,000 "
     "quadrangles with complete legend. V1-historical: development only.",
     "User opens the two MAGA PDF links in a normal browser and saves them if they load"),
    ("8K", "KURIKOMA_2008", "STOP (no coordinate-supported Omega)",
     "New check: the only other public inventory (Yagi et al. 2009 in the USGS repository) answers 'Is the boundary of the mapped area "
     "indicated? No'. GSI Material 4 remains un-georeferenced. Not unseen (V1-historical).", "Optional single GSI request; otherwise stop"),
    ("NEW", "TALAKMAU_PASAMAN_2022", "UPGRADED TO TIER A",
     "UNOSAT LS20220308IDN acquired (CC BY-SA): analysis extent 881.7 km2, cloud 30.0 km2, landslide extent 6.24 km2 (Sentinel-2, "
     "7 days after the Mw 6.1 earthquake); COP30 pre-event; GeoMap district SHPs (Kab. Pasaman 13.08, Kab. Pasaman Barat 13.12) and "
     "1:250k Lubuksikaping sheet listed on the official portal (sign-in, as for Lombok).",
     "User GeoMap download; then Y-quality decision D-TAL-Y and fast path"),
]

SEARCHED_NOT_ADDED = [
    ("Kauai April 2018 rainstorm", "No event inventory or mapped area found (bounded search)"),
    ("Hurricane Fiona 2022 Puerto Rico", "USGS 2022-23 records are report-based points without a search domain; same landscape group as PR"),
    ("Sao Tome Lemba Jan 2022 (UNOSAT FL20220119STP)", "Structure-impact product over a small town analysis area; not a landslide inventory"),
    ("Hanang 2023 / Solomon Is. / Java UNOSAT products", "Single-event debris flows or non-volcanic/redundant settings; Hanang already rejected"),
    ("USGS ground-failure repository Central America items", "Ruiz 2019/2020, MARN 2001, Rymer 1987, Harp 1981/1984, Marc 2016: no mapping-extent files"),
    ("Madeira 2010", "Defined 178 km2 study area (Lira et al. 2013) but vectors not public; kept as Tier B request"),
]


# ---------------------------------------------------------------- 13. human actions
HUMAN_ACTIONS = """# Stage 3A — human action queue

Everything below needs your identity, a browser, or your sending authority. Nothing has been sent, registered or accepted for you.
Put every file you obtain, untouched (original archive + any licence/readme/email text), in:

`Science_Fair_ML_V2/data/incoming_user/`

Then tell the coding agent the filenames. Ingestion preserves the originals, hashes them, and registers immutable RAW copies.

## ACTION NOW

### N1 — Talakmau (West Sumatra) geology: GeoMap sign-in (≈10 min, same steps as Lombok)
- **Why:** this is the only missing artifact on the fastest new landscape (Tier A). Y and Omega (UNOSAT) and M (Copernicus) are already public.
- **Where:** official Badan Geologi GeoMap portal (sign-in required; free account):
  - Kab. Pasaman: https://geologi.esdm.go.id/geomap/pages/district/13.08 → **Unduh SHP** ("SHP: Kab./Kota Pasaman", ~1 MB)
  - Kab. Pasaman Barat: https://geologi.esdm.go.id/geomap/pages/district/13.12 → **Unduh SHP** ("SHP: Kab./Kota Pasaman Barat", ~1 MB)
  - Also from either page: raster **Peta Geologi Lembar Lubuksikaping, Sumatera 1:250.000** (legend/explanation needed to resolve units, as the Lombok sheet was).
- **Registration/login:** yes (your own account). **Email:** no.
- **Expected files:** two district SHP ZIPs + one sheet JPEG/PDF.

### N2 — Puerto Rico: two items
1. **Scientific decision D-PR-OMEGA (yours, not automatic).** Choose one:
   - (a) Accept Omega_obs = the main island of Puerto Rico that the USGS metadata says was systematically searched (FEMA ~97% + DigitalGlobe ~99%, overlapping imagery used against cloud/shadow; small islands excluded), minus Copernicus water, with an explicit residual-visibility caveat; or
   - (b) Wait for coverage documentation from USGS (email below) before any Omega is built.
   - Option (a) is **not** a bounding box, hull or buffer of points; but it has no exact cloud/shadow mask, so it must be recorded as a decision, not as an automatic PASS.
2. **Email (send yourself) — to the USGS metadata contact listed on the source record (address removed from public source copy)**

> Subject: Hurricane María slope-failure inventory (doi:10.5066/P9BVMD74) — mapped-area coverage question
>
> Dear Dr Schulz,
> I am a student using the USGS "Map of slope-failure locations in Puerto Rico after Hurricane María" (Hughes et al., 2019) in a non-commercial science-fair study of landslide susceptibility in volcanic terrain. To define the area that was actually searched, could you tell me whether a footprint of the imagery that was examined, or a polygon of areas that could not be mapped (cloud, shadow, missing imagery), exists for the main island? If no such file exists, would it be accurate to describe the inventory as a systematic search of the entire main island? Any file or short statement would be very helpful, and I will cite the data release.
> Thank you for your time,
> [your name, school]

### N3 — Batched data requests (send yourself; drafts below)
Send these now. They are independent, so do not wait for one before sending another.

**N3a — Banks Peninsula 2021 → corresponding author listed on the source record (address removed from public source copy)**

> Subject: Request: Eastern Bays (Banks Peninsula) December 2021 landslide inventory and mapping boundary
>
> Dear Dr Yates,
> I am a student working on a non-commercial science-fair project that tests whether landslide susceptibility learned on some volcanic landscapes transfers to others. Your 2025 NZJGG paper notes the data are available on reasonable request. Could I please have (1) the source and runout polygons and (2) the boundary of the ~115 km² area that was systematically mapped, as GeoPackage or shapefile with CRS and the imagery/LiDAR dates? I would attribute the data as you specify and not redistribute it.
> With thanks, [your name, school]

**N3b — Coromandel + Auckland (Waitākere), Cyclone Gabrielle → GNS Science, NZ Landslide Database institutional custodian (address removed from public source copy)**

> Subject: Request: Cyclone Gabrielle V1.0 landslide polygons and priority mapping cells for Coromandel and Waitākere
>
> Dear NZLD team,
> I am a student using landslide inventories in volcanic terrain for a non-commercial science-fair project. The public OSF extract for Massey et al. (2025) covers training cells on sedimentary terrain only. Could I request, for the Coromandel Peninsula and the Waitākere Ranges only, (1) the V1.0 landslide polygons (source areas separated from runout/deposit if available) and (2) the 5 × 2.5 km priority mapping grid cells that were fully mapped, with a flag for cells that were completely mapped and the imagery dates? GeoPackage or shapefile in NZTM2000 would be ideal. I will cite the dataset and follow your licence terms.
> Kind regards, [your name, school]

**N3c — Miravalles 2016 → corresponding author of Quesada-Román et al. (2019), Landslides, doi:10.1007/s10346-019-01209-4 (use the address on the article page)**

> Subject: Request: Miravalles 2016 landslide polygons with trigger attribution
>
> Dear Dr Quesada-Román,
> For a non-commercial science-fair study of landslide susceptibility on volcanoes, I would be grateful for the 2016 Miravalles landslide inventory polygons, with an attribute indicating whether each landslide followed the July 2016 Bijagua earthquake or Hurricane Otto (or is unknown), plus the boundary of the area you mapped and the imagery dates. Shapefile or GeoPackage is ideal. I will cite your work and not redistribute the data.
> Thank you, [your name, school]

**N3d — Kivu NTK + Kalehe → Arthur Depicker / Olivier Dewitte, Royal Museum for Central Africa (address on Zenodo record 5027004 / the NHESS 2023 Kalehe paper)**

> Subject: Request: North Tanganyika–Kivu landslide inventory search mask (Zenodo 5027004) and Kalehe inventory
>
> Dear Dr Depicker and Dr Dewitte,
> I am a student using your open NTK inventory for a non-commercial science-fair project on landslide susceptibility transfer between volcanic landscapes. To use it correctly I need to know where landslides could have been observed. Could you share (1) the source-area polygons, (2) any year-specific imagery-availability or search mask, and (3) the meaning of negative values in the Year field? If possible, the Kalehe inventory from NHESS 23:643 (vectors, study boundary and any fieldwork/visibility mask) would also be very helpful. I will cite both datasets.
> With thanks, [your name, school]

### N4 — Atitlán geology (≈5 min, browser only)
- Open these in a normal browser (scripts receive HTTP 403):
  - https://www.maga.gob.gt/wp-content/uploads/pdf/suelos/solola1.pdf
  - https://www.maga.gob.gt/download/suelos-solola1.pdf
- If either loads, save the PDF untouched. It is needed to identify, and possibly digitise with uncertainty, the four IGN 1:50,000 geology quadrangles it cites. Atitlán stays development-only (V1-historical).
- **Registration/login:** no. **Email:** no.

## ACTION AFTER RESPONSE
- **R1 — Wilsons River NSW 2022:** send the Stage 2F second-wave request (search/candidate-generation mask, rejected-area mask, imagery and DEM epochs) only after Banks/GNS replies show which hotspot/shield landscape is closest. **Decide first** whether Wilsons River (Lamington Volcanics, the Tweed volcanic province) may be used at all, given the parent Tweed project.
- **R2 — Any reply with data:** save the attachments, the email text (licence terms) and the citation in `data/incoming_user/`.
- **R3 — D-TAL-Y decision** (after N1 arrives): accept or reject UNOSAT's preliminary 10 m Sentinel-2 landslide extent as a Tier-2 affected-area target (with a "rapid-mapping, not field-validated" caveat), or seek a published Talakmau inventory first.

## OPTIONAL / LOW PRIORITY
- **O1 PNG Porgera:** ask H. Tanyas for an independent searched/visibility mask (Stage 2F RP01 draft). The public extent is Y-derived.
- **O2 Dominica (V1-historical):** request access to restricted Zenodo 15224820 (login), and ask for the imagery/search footprint.
- **O3 Madeira 2010:** ask the NHESS 13:709 corresponding author for the inventory + Funchal/Ribeira Brava study area. The event predates Copernicus DEM acquisitions, so an M-source decision would be needed.
- **O4 Kurikoma:** one GSI request for the digital interpretation boundary of Material 4. Otherwise stop.
- **O5 São Miguel (Azores):** custodian request for occurrence polygons and the searched domain (not hazard zones).

## Stop spending time on
Mount Elgon (incomplete by the authors' own statement), Guatemala 1976 (no pre-event DEM), Mbale-Elgon 2022, cumulative undated inventories (Oregon SLIDO, WASLID, Columbia River Gorge, IFFI), Patagonia (predictions are not Y), Hokkaido Iburi (fails the frozen domain rule), and generic NZLD downloads.
"""


MACHINE_ACTIONS = [
    ("MA01", "TALAKMAU_PASAMAN_2022", "Register GeoMap district SHPs + Lubuksikaping sheet; audit units against the sheet legend", "N1 files in incoming_user", False, "metadata/stage2b_raw_registry.json entries; legend evidence table", "GEOLOGY|VOLCANIC_DOMAIN", "P1"),
    ("MA02", "TALAKMAU_PASAMAN_2022", "Acquire COP30 tiles N00/S01 x E099/E100 (extent straddles the equator) with XML/EDM/FLM/WBM; verify acquisition < 2022-02-25", "none", True, "COP30 GeoTIFF + quality layers", "M_TEMPORAL|DEM_COVERAGE", "P1"),
    ("MA03", "TALAKMAU_PASAMAN_2022", "Derive Omega_obs = AnalysisExtent - CloudObstruction; quantify; never use LandslideExtent", "none", True, "omega_obs vector + derivation table", "OMEGA", "P1"),
    ("MA04", "TALAKMAU_PASAMAN_2022", "Record decision D-TAL-Y, then run fast path (stage2f2_production.build_polygon_landscape)", "MA01-MA03 + R3", False, "terrain_units/features_m/targets + checkpoint", "Y|TARGET_SEMANTICS|TERRAIN_GENERATION", "P1"),
    ("MA05", "PUERTO_RICO_MARIA_2017", "Unit-by-unit harmonisation of OFR 98-38 into the frozen ontology (exclude volcanic-clast conglomerates, intrusive dacite); quantify the Cenozoic volcanic domain", "none", True, "geology mapping CSV + domain table", "GEOLOGY|VOLCANIC_DOMAIN", "P1"),
    ("MA06", "PUERTO_RICO_MARIA_2017", "Acquire COP30 tiles N17/N18 x W066/W067 + quality layers; verify acquisition < 2017-09-20", "none", True, "COP30 GeoTIFF + quality layers", "M_TEMPORAL|DEM_COVERAGE", "P1"),
    ("MA07", "PUERTO_RICO_MARIA_2017", "Build a parameterised mirror of the frozen Stage 2D POINT branch and prove byte-identity on Kumamoto (as done for Noto)", "none", True, "stage3 point mirror + equivalence test", "TERRAIN_GENERATION", "P1"),
    ("MA08", "PUERTO_RICO_MARIA_2017", "Apply D-PR-OMEGA; build Omega_obs and OMEGA_ANALYSIS; production via fast path", "N2 decision (+ USGS reply if option b)", False, "Omega derivation + production", "OMEGA|TERRAIN_GENERATION", "P1"),
    ("MA09", "BANKS_PENINSULA_2021", "Ingest inventory + 115 km2 boundary; QMAP geology (check loess mapping vs volcanic domain); COP30", "N3a reply", False, "registered RAW + gate audit", "Y|OMEGA|GEOLOGY", "P2"),
    ("MA10", "COROMANDEL_GABRIELLE_2023|AUCKLAND_WAITAKERE_2023", "Ingest V1.0 polygons + priority cells; Omega = fully mapped cells; QMAP volcanic units", "N3b reply", False, "registered RAW + gate audit", "Y|OMEGA|GEOLOGY", "P2"),
    ("MA11", "MIRAVALLES_OTTO_2016", "Ingest; split earthquake vs hurricane labels into separate target phases; Costa Rica geology route", "N3c reply", False, "phase-separated Y + gate audit", "Y|TARGET_SEMANTICS", "P2"),
    ("MA12", "KIVU_RIFT_NTK_2000_2019|KIVU_KALEHE_RIFT_FLANK", "Ingest search mask; restrict to single-epoch support; rift-basalt geology route", "N3d reply", False, "gate audit", "OMEGA|M_TEMPORAL|GEOLOGY", "P2"),
    ("MA13", "ATITLAN_STAN_2005", "Georeferencing-feasibility audit of the IGN quadrangles in the MAGA PDF (ticks, residuals, legend completeness)", "N4 PDF", False, "geology feasibility report", "GEOLOGY", "P3"),
    ("MA14", "ALL", "Re-evaluate DATASET_DIVERSITY_STOPPING_RULE and the coverage matrix after each new production landscape", "any new production landscape", True, "updated evaluation", "NONE (governance)", "P1"),
    ("MA15", "ALL_PRODUCTION", "Bind E allowlist sources (CHELSA bounded windows, Hansen tiles) with licence + temporal-staleness metadata", "none", True, "E rasters per landscape", "E (not a MEG production gate)", "P3"),
]


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_before = v1_verify()
    state = verify_state()
    acquisitions = acquire_all()
    pr_screen = pr_geology_screen()
    unosat = unosat_audit()
    matrix = S.build_matrix()
    ranking = S.rank(matrix)
    gap_rows = gaps(matrix, ranking)
    prod_rows = [r for r in matrix if r["production_ready"]]
    current = evaluate_stopping_rule(prod_rows, gap_rows)
    tier_a_ids = {r["landscape_id"] for r in ranking if r["priority_tier"] == "A"}
    scenario = evaluate_stopping_rule(prod_rows + [r for r in matrix if r["landscape_id"] in tier_a_ids], gap_rows)
    tiers = {r["landscape_id"]: r for r in ranking}

    _json(OUT / "STATE_VERIFICATION.json", state)
    _tsv(OUT / "LANDSCAPE_SPACE_MATRIX.tsv", matrix)
    _tsv(OUT / "LANDSCAPE_SPACE_GAPS.tsv", gap_rows)
    _tsv(OUT / "CANDIDATE_PRIORITY_MATRIX.tsv", ranking)
    _json(OUT / "PRIORITY_SCORING_RUBRIC.json", dict(
        weights=S.WEIGHTS, max_total=36, tier_a_min_score=S.TIER_A_MIN_SCORE, status_points=S.STATUS_POINTS, m_points=S.M_POINTS,
        access_points=S.ACCESS_POINTS,
        A_diversity="novel non-UNKNOWN values vs production set over axes " + ",".join(S.AXES) + ": 0->0, 1-2->1, 3-4->2, >=5->3",
        B_target="STATUS_POINTS[y]; capped at 1 when target tier is UNCLEAR and Y not PASS", C_omega="STATUS_POINTS[omega]", D_m="M_POINTS[m]",
        E_geology="STATUS_POINTS[g]", F_access="ACCESS_POINTS[access]",
        G_time="0 if hard blocker else open gates among Y/Omega/G/M: 0-1->3, 2->2, 3->1, 4->0",
        H_validation="POTENTIAL_LOCKBOX 3, UNDECIDED/DEVELOPMENT_PREFERRED 2, INELIGIBLE_V1_HISTORICAL 1",
        I_redundancy="-3 same landscape group as production; -2 if >=5 known axes equal a production landscape; -1 if >=4",
        tier_rules=dict(C="hard blocker, or no data route, or score < 18 with diversity < 2",
                        A="no hard blocker, score >= 18, access PUBLIC/USER_SIGNIN, pre-event M feasible (D=3), Y and Omega >= 2 points, "
                          "geology route PASS/PUBLIC/USER_SIGNIN",
                        B="otherwise (score >= 18 or diversity >= 2) with an existing data route"),
        excluded_from_scoring=["expected ML performance", "Y prevalence/class balance", "terrain-unit count"]))
    _tsv(OUT / "RESCUE_TRACKS.tsv", [dict(track=t, landscape_id=l, result=res, evidence=ev, next_step=nx) for t, l, res, ev, nx in RESCUE])
    _tsv(OUT / "SEARCHED_NOT_ADDED.tsv", [dict(search=a, reason=b) for a, b in SEARCHED_NOT_ADDED])
    reg = {e["acquisition_id"]: e for e in load_registry()}
    acq_rows = [dict(acquisition_id=a["acquisition_id"], landscape_id=a["landscape_id"], role=a["data_role"], gate_affected=a["gate_affected"],
                     url=a["download_url"], source_page=a["source_url"], publisher=a["publisher"], doi=a["doi"] or "NONE", licence=a["licence"],
                     access_date=a["access_date"], sha256=reg[a["acquisition_id"]]["sha256"], bytes=reg[a["acquisition_id"]]["actual_bytes"],
                     raw_path=reg[a["acquisition_id"]]["raw_path"], temporal_relevance=a["temporal_coverage"]) for a in ACQUISITIONS]
    _tsv(OUT / "PUBLIC_ACQUISITIONS.tsv", acq_rows)
    _json(OUT / "PUERTO_RICO_GEOLOGY_SCREEN.json", pr_screen)
    _json(OUT / "UNOSAT_Y_OMEGA_AUDIT.json", unosat)
    _tsv(OUT / "LOCKBOX_PLANNING.tsv", [dict(landscape_id=r["landscape_id"], lockbox_status=r["lockbox_status"], v1_historical=r["v1_historical"],
                                             priority_tier=tiers[r["landscape_id"]]["priority_tier"], note="PLANNING ONLY - no labels hidden or frozen")
                                        for r in matrix])
    _tsv(OUT / "H_FEASIBILITY.tsv", [dict(landscape_id=r["landscape_id"], h_feasibility=r["h_feasibility"], likely_evidence=r["h_evidence"] or "NOT_ASSESSED",
                                          h_generated=False, missing_h_encoding="CATEGORICAL_ONLY_NEVER_ZERO") for r in matrix
                                     if tiers[r["landscape_id"]]["priority_tier"] in ("PRODUCTION", "A", "B")])
    (OUT / "LANDSCAPE_INDEPENDENCE.md").write_text(
        "# Landscape independence (Stage 3A)\n\nA landscape is one coherent geographic/geological prediction domain: one volcanic edifice, "
        "complex or province, mapped with one geology source and one observation domain. Independent N counts landscape groups, not "
        "inventories, events or terrain units.\n\n- Additional events in the same group are temporal replicates or separate event "
        "labels (e.g. ASO_2012_RAIN vs KUMAMOTO_ASO_2016; MBALE_ELGON_2022 vs MOUNT_ELGON_UGANDA).\n- Successive phases of one inventory "
        "(e.g. Ferrario 2019b vs 2019a) are sensitivity targets.\n- Overlapping study regions are one group (Kivu NTK / Kalehe / Rwanda "
        "Virunga; Banks Peninsula / Port Hills; El Salvador 1986/2001; Sao Miguel / Sete Cidades; Reunion BDMvT / Salazie).\n- LOLO "
        "must hold out whole groups.\n\nGroups: see `landscape_group` in LANDSCAPE_SPACE_MATRIX.tsv.\n")
    (OUT / "STAGE3A_HUMAN_ACTIONS.md").write_text(HUMAN_ACTIONS)
    _tsv(OUT / "STAGE3A_MACHINE_ACTIONS.tsv", [dict(task_id=a, landscape_id=b, task=c, prerequisite=d, can_run_now=e, expected_artifact=f,
                                                    gate_affected=g, priority=h) for a, b, c, d, e, f, g, h in MACHINE_ACTIONS])
    _json(OUT / "FAST_PATH_SPEC.json", dict(steps=[dict(step=a, implementation=b, gate=c) for a, b, c in STEPS], required_gates=list(REQUIRED_GATES),
                                            decision_gates=sorted(DECISION_GATES), module="sfv2.fast_path.GateManifest",
                                            rule="Scientific gates cannot PASS without a decision record; production only when every gate is PASS"))
    _json(OUT / "DATASET_DIVERSITY_STOPPING_RULE.json", dict(STOPPING_RULE, current_evaluation=current,
                                                             scenario_if_all_tier_A_pass=scenario))
    dash_cols = ["landscape_id", "status", "target_tier", "trigger", "morphology", "erosion_state", "climate", "tectonic_setting", "continent",
                 "y", "omega", "m", "g", "e", "h_feasibility", "main_blocker", "next_action", "priority_tier", "lockbox_status"]
    dash = [dict({c: r.get(c, "") for c in dash_cols}, status=("PRODUCTION_READY" if r["production_ready"] else r["prior_state"]),
                 priority_tier=tiers[r["landscape_id"]]["priority_tier"]) for r in matrix]
    order = {"PRODUCTION": 0, "A": 1, "B": 2, "C": 3}
    dash.sort(key=lambda d: (order[d["priority_tier"]], -tiers[d["landscape_id"]]["total"], d["landscape_id"]))
    _tsv(OUT / "DATASET_STATUS_DASHBOARD.tsv", dash)
    short = lambda s, n=70: s if len(s) <= n else s[:n - 1] + "…"
    shown = [dict(d, main_blocker=short(d["main_blocker"]), next_action=short(d["next_action"])) for d in dash if d["priority_tier"] in ("PRODUCTION", "A", "B")]
    cov_rows = [dict(category=x["category"], major="MAJOR" if x["major_gap"] else "minor", production=x["current_production_groups"],
                     need=x["desired_minimum_groups"], status="COVERED" if x["current_production_groups"] >= x["desired_minimum_groups"] else "MISSING",
                     closest_fillers=short(x["candidate_fillers_tier_A_B"], 90)) for x in gap_rows]
    (OUT / "DATASET_STATUS_DASHBOARD.md").write_text(
        f"# V2 dataset status dashboard (Stage 3A, {S.AS_OF})\n\nProduction-ready: **{len(prod_rows)}** landscapes / "
        f"**{state['total_units']:,}** terrain units. Stopping rule satisfied: **{'YES' if current['satisfied'] else 'NO'}** "
        f"({current['independent_groups']} of >= 11 independent groups).\n\n## Coverage (what we have vs what is missing)\n\n"
        + _md_table(cov_rows, list(cov_rows[0])) + "\n\n## Production + Tier A + Tier B\n\n" + _md_table(shown, dash_cols)
        + "\n\nUser actions: see STAGE3A_HUMAN_ACTIONS.md (ACTION NOW: N1 Talakmau GeoMap, N2 Puerto Rico decision + email, "
          "N3 four batched requests, N4 Atitlan PDFs). Full table incl. Tier C: DATASET_STATUS_DASHBOARD.tsv\n")
    report_rows = [dict(rank=i + 1, **{k: r[k] for k in ("landscape_id", "priority_tier", "total", "A_diversity", "B_target", "C_omega", "D_m",
                                                          "E_geology", "F_access", "G_time", "H_validation", "I_redundancy", "tier_reason")})
                   for i, r in enumerate(x for x in ranking if x["priority_tier"] != "PRODUCTION")]
    (OUT / "CANDIDATE_PRIORITY_REPORT.md").write_text(
        "# Candidate priority report (Stage 3A)\n\nTotal = 3A + 2B + 2C + D + E + F + G + H + 2I (max 36; see PRIORITY_SCORING_RUBRIC.json). "
        "Expected ML performance and prevalence are not inputs.\n\n" + _md_table(report_rows, list(report_rows[0])) + "\n")
    ml_h_hits = scan_ml_h_artifacts()
    v1_after = v1_verify()
    outs = sorted(p for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("STAGE3A_CHECKPOINT", "STAGE3A_DIAGNOSTICS")))
    cp = dict(
        schema="SFV2_STAGE3A_CHECKPOINT_V1", as_of=S.AS_OF, production_ready_count=state["production_ready_count"],
        production_terrain_units_total=state["total_units"], state_verified=dict(
            stage2f2_checkpoint_unchanged=state["stage2f2_checkpoint_unchanged"], prior_hash_mismatches=len(state["hash_mismatches"]),
            production_matches_expected=state["production_matches_expected"],
            lombok_production_artifacts_verified=state["lombok_production_artifacts_verified"]),
        prior_checkpoints={k: v["sha256"] for k, v in state["checkpoints"].items()},
        new_raw_acquisitions={r["acquisition_id"]: r["sha256"] for r in acquisitions},
        tiers={t: sorted(r["landscape_id"] for r in ranking if r["priority_tier"] == t) for t in ("A", "B")},
        tier_counts=dict(Counter(r["priority_tier"] for r in ranking)),
        stopping_rule_satisfied=current["satisfied"], independent_groups=current["independent_groups"],
        output_hashes={str(p.relative_to(ROOT)): sha256(p) for p in outs},
        code_hashes={n: sha256(ROOT / "src/sfv2" / n) for n in ("stage3a.py", "build_stage3a.py", "stage3a_acquire.py", "fast_path.py")},
        ml_or_h_artifacts_found=ml_h_hits, training=False, h_generated=False, production_landscapes_generated=0, lockboxes_frozen=False,
        external_messages_sent=0,
        v1_integrity_before={k: v1_before[k] for k in ("strict_files", "v1_intact", "writes_performed")},
        v1_integrity_after={k: v1_after[k] for k in ("strict_files", "v1_intact", "writes_performed")},
        volatile_diagnostics={"path": "outputs/stage3a/STAGE3A_DIAGNOSTICS.json", "authoritative": False})
    _json(OUT / "STAGE3A_CHECKPOINT.json", cp)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    status = git("status", "--porcelain")
    _json(OUT / "STAGE3A_DIAGNOSTICS.json", dict(authoritative=False, head=git("rev-parse", "HEAD"),
                                                 modified_tracked=sum(1 for ln in status.splitlines() if not ln.startswith("??")),
                                                 untracked=sum(1 for ln in status.splitlines() if ln.startswith("??"))))
    return dict(cp=cp, state=state, ranking=ranking, gaps=gap_rows, current=current, scenario=scenario, pr=pr_screen, unosat=unosat, matrix=matrix)


if __name__ == "__main__":
    main()
