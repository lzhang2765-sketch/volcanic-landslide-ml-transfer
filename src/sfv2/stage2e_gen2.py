"""General Gen-2 palaeolandscape contracts and the Aso evidence audit.

This module is deliberately Y-blind.  It inventories geological controls and
defines reconstruction families, uncertainty, and common H semantics.  Stage
2E does not construct a palaeosurface because Aso has no admissible numeric
palaeoelevation controls after the evidence audit.
"""
from __future__ import annotations

import platform
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

from .schema import ROOT

AS_OF = "2026-10-06"
SEED = 20261006
CONTROL_FIELDS = (
    "control_id", "geometry", "source", "source_sha256", "measurement_interpretation_type",
    "age", "uncertainty", "reliability", "reconstruction_role", "decision", "rationale",
    "numeric_value", "numeric_unit", "temporal_target_compatibility",
)
H_STATES = ("FULL_RECONSTRUCTION", "PARTIAL_RECONSTRUCTION", "AGE_PROXY_ONLY", "H_UNAVAILABLE")

FAMILY_REGISTRY = [
    {"family_id": "CALDERA_RIM_CONSTRAINED", "required_controls": "mapped rim/wall geometry plus elevations and target age",
     "applicable_morphologies": "caldera complexes", "common_output": "ensemble palaeoelevation with support and uncertainty"},
    {"family_id": "RADIAL_FLANK", "required_controls": "centre distribution, preserved flank profiles, constructional unit ages",
     "applicable_morphologies": "single or multi-edifice cones/shields", "common_output": "ensemble palaeoelevation with support and uncertainty"},
    {"family_id": "PRESERVED_REMNANT_ENVELOPE", "required_controls": "age-correlated remnant surfaces and unconformity/erosion masks",
     "applicable_morphologies": "eroded shields and volcanic plateaux", "common_output": "ensemble palaeoelevation with support and uncertainty"},
    {"family_id": "STRATIGRAPHIC_SURFACE", "required_controls": "correlated contacts, numerical ages, thickness/attitude or borehole picks",
     "applicable_morphologies": "layered volcanic piles and buried surfaces", "common_output": "ensemble palaeoelevation with support and uncertainty"},
    {"family_id": "ANALOGUE_REFERENCE_MORPHOLOGY", "required_controls": "external analogue population and transferability criteria",
     "applicable_morphologies": "poorly preserved edifices with defensible analogues", "common_output": "ensemble palaeoelevation with family-disagreement flag"},
    {"family_id": "AGE_HISTORY_PROXY", "required_controls": "mapped units and defensible age intervals",
     "applicable_morphologies": "landscapes where a full surface is unsupported", "common_output": "age/history only; palaeoelevation remains missing"},
]


def software() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "geopandas": gpd.__version__, "rasterio": rasterio.__version__}


def _geom_text(g) -> str:
    return "GEOMETRY_UNAVAILABLE" if g is None or g.is_empty else g.wkt


def aso_controls(sha256) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Inventory actual GSJ vector evidence without interpreting it as height."""
    src = ROOT / "data/interim/_stage2c_aso"
    polys = gpd.read_file(src / "geo_A.shp")
    lines = gpd.read_file(src / "geo_L.shp")
    other = gpd.read_file(src / "ol1.shp")
    points = gpd.read_file(src / "pnt.shp")
    dem_path = ROOT / "data/processed/stage2c/KUMAMOTO_ASO_2016/dem_cop30_30m.tif"
    rows: list[dict] = []

    def add(cid, geom, source, kind, age, uncertainty, reliability, role, decision, rationale,
            value="", unit="", compatibility="UNRESOLVED"):
        rows.append({
            "control_id": cid, "geometry": _geom_text(geom), "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source), "measurement_interpretation_type": kind, "age": age,
            "uncertainty": uncertainty, "reliability": reliability, "reconstruction_role": role,
            "decision": decision, "rationale": rationale, "numeric_value": value, "numeric_unit": unit,
            "temporal_target_compatibility": compatibility, "_geometry": geom,
        })

    # One record per mapped unit code: this preserves rather than collapses the
    # map's unit semantics.  The polygons constrain masks/order, not elevation.
    for symbol, group in polys.groupby("SYMBOL", dropna=False, sort=True):
        r = group.iloc[0]
        sym = str(symbol) if pd.notna(symbol) else "UNNAMED"
        stage = str(r.get("LEGEND04E") or r.get("LEGEND03E") or "UNKNOWN")
        age = str(r.get("LEGEND03E") or "UNKNOWN")
        role = "STRATIGRAPHIC_MASK"
        if "Pre-Aso" in stage or "pre-Aso" in stage or "Pliocene" in age:
            role = "PRESERVED_OLDER_ROCK_MASK"
        elif "pyroclastic-flow" in stage.lower():
            role = "CALDERA_FORMING_DEPOSIT_MASK"
        add(f"ASO_UNIT_{sym}", unary_union(group.geometry), src / "geo_A.shp", "MAPPED_GEOLOGICAL_UNIT",
            age, "Map-scale and unit-age interval; no surface-height uncertainty supplied", "HIGH_FOR_UNIT_ID_MEDIUM_FOR_H",
            role, "INCLUDE_QUALITATIVE_ONLY",
            "Authoritative 1:50,000 unit geometry can constrain temporal/structural masks but is not a palaeoelevation measurement.")

    for i, r in lines.iterrows():
        label = str(r["LEGEND01E"])
        decision = "INCLUDE_QUALITATIVE_ONLY" if label == "Geologic boundary" else "INCLUDE_STRUCTURAL_UNCERTAINTY_ONLY"
        role = "CONTACT_CONSTRAINT" if label == "Geologic boundary" else "STRUCTURAL_BREAK_OR_UNCERTAINTY"
        if label == "Map frame":
            decision, role = "EXCLUDE", "CARTOGRAPHIC_ONLY"
        add(f"ASO_LINE_{i:02d}", r.geometry, src / "geo_L.shp", label.upper().replace(" ", "_"), "UNKNOWN",
            "Positional uncertainty limited by 1:50,000 source; displacement/age mostly unspecified", "HIGH_FOR_LOCATION",
            role, decision, "Useful as a mask or discontinuity; not a measured palaeosurface height.")

    for (major, minor, label), group in other.groupby(["MAJOR_CODE", "MINOR_CODE", "LEGEND01E"], dropna=False, sort=True):
        label = str(label)
        decision = "INCLUDE_QUALITATIVE_ONLY" if label == "Concealed lava front" else "EXCLUDE_FROM_HEIGHT_FIT"
        add(f"ASO_OTHER_{int(major):02d}_{int(minor):02d}", unary_union(group.geometry), src / "ol1.shp",
            label.upper().replace(" ", "_"), "UNKNOWN", "1:50,000 positional uncertainty; feature age not numeric",
            "HIGH_FOR_LOCATION", "STRATIGRAPHIC_EXTENT" if label == "Concealed lava front" else "QC_CONTEXT",
            decision, "Lava fronts can bound units; dikes and craters do not supply the target palaeosurface height.")

    for _, r in points[points["LEGEND01E"].str.contains("basement", case=False, na=False)].sort_values("Serial_No_").iterrows():
        raw = r.get("Attribute2")
        value = "" if pd.isna(raw) or float(raw) == 0 else f"{float(raw):g}"
        add(f"ASO_BOREHOLE_{int(r['Serial_No_']):02d}", r.geometry, src / "pnt.shp", "BASEMENT_REACHING_BOREHOLE",
            "UNKNOWN", "Attribute2 meaning/datum/unit is not defined by the distributed legend workbook", "HIGH_FOR_LOCATION_LOW_FOR_VALUE",
            "POTENTIAL_BASEMENT_CONTACT", "HOLD_SEMANTICS_UNRESOLVED",
            "The site is authoritative, but treating Attribute2 as depth or elevation would be an unsupported reinterpretation.",
            value, "UNDEFINED_SOURCE_ATTRIBUTE")

    with rasterio.open(dem_path) as ds:
        b = ds.bounds
        footprint = gpd.GeoSeries([box(b.left, b.bottom, b.right, b.top)], crs=ds.crs).to_crs(polys.crs).iloc[0]
    add("ASO_MODERN_DEM_FOOTPRINT", footprint, dem_path, "MODERN_TOPOGRAPHY",
        "2010-2015 Copernicus acquisition epoch", "DEM vertical/quality uncertainty retained in Stage 2D EDM/FLM flags", "HIGH",
        "CANDIDATE_PRESERVED_SECTION_SAMPLING", "HOLD_UNTIL_REMNANTS_CLASSIFIED",
        "Modern topography may be sampled only after independent geological identification of preserved constructional surfaces; it is not itself palaeo evidence.")

    # Published evidence is recorded with the map footprint because no reusable
    # vector geometry is distributed in the cited article.
    footprint_geo = unary_union(polys.geometry).envelope
    publication_source = src / "geo_A.shp"
    add("ASO_PUBLISHED_PRECALDERA_NW_WALL", footprint_geo, publication_source, "PUBLISHED_STRATIGRAPHIC_INTERPRETATION",
        "Pre-caldera stage; no common numeric surface age", "Article constrains NW-wall stratigraphy, not a continuous palaeosurface",
        "MEDIUM_FOR_RECONSTRUCTION", "PRESERVED_SECTION_CONCEPT", "INCLUDE_QUALITATIVE_ONLY",
        "Furukawa et al. (2009, DOI 10.5575/geosoc.115.658) supports pre-Aso wall remnants but supplies no project-ready elevation controls.")
    add("ASO_PUBLISHED_CALDERA_HISTORY", footprint_geo, publication_source, "PUBLISHED_TEMPORAL_INTERPRETATION",
        "Four caldera-forming stages; Aso-4 about 90 ka", "Regional stage constraint; not a pointwise surface observation", "HIGH_FOR_HISTORY",
        "TEMPORAL_TARGET_DEFINITION", "INCLUDE_QUALITATIVE_ONLY",
        "GSJ Ono and Watanabe (1985) establishes pre-, caldera-forming and post-caldera units; a single composite palaeosurface would mix ages.")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_geometry"} for r in rows], columns=CONTROL_FIELDS)
    gdf = gpd.GeoDataFrame(
        [{"control_id": r["control_id"], "decision": r["decision"], "reconstruction_role": r["reconstruction_role"]}
         for r in rows], geometry=[r["_geometry"] for r in rows], crs=polys.crs,
    )
    return df, gdf


def method_spec(controls: pd.DataFrame) -> dict:
    numeric = controls[
        controls["decision"].eq("INCLUDE_NUMERICAL") & controls["numeric_value"].astype(str).ne("")
    ]
    return {
        "framework_version": "GEN2_STAGE2E_1.0", "landscape_id": "KUMAMOTO_ASO_2016",
        "temporal_target": "UNRESOLVED: choose either pre-caldera, immediately post-Aso-4, or unit-age-conditioned surfaces before fitting",
        "selected_family": "HYBRID_CALDERA_RIM_PRESERVED_FLANK_STRATIGRAPHIC",
        "selection_status": "CONCEPTUALLY_SELECTED_NUMERIC_GATE_FAILED",
        "why_selected": "Aso combines a caldera wall, pre-Aso remnants, caldera-forming deposits, and post-caldera edifices; no single radial cone or unconstrained interpolator represents that history.",
        "alternatives": {
            "CALDERA_RIM_CONSTRAINED": "Necessary component, but the current vector package has no explicit rim-elevation control layer.",
            "PRESERVED_REMNANT_ENVELOPE": "Necessary component, but preserved constructional surfaces are not yet independently classified/datetimed.",
            "RADIAL_FLANK": "Rejected alone: multiple post-caldera centres and the caldera discontinuity violate a single-centre radial assumption.",
            "STRATIGRAPHIC_SURFACE": "Necessary component, but mapped contacts have broad ages and no thickness/attitude model.",
            "ANALOGUE_REFERENCE_MORPHOLOGY": "Deferred: an external analogue population and transfer criteria have not been assembled.",
        },
        "control_gate": {"required_minimum_numeric_height_controls": 12, "available_admissible_numeric_height_controls": int(len(numeric)),
                         "required_explicit_rim_or_preserved_surface_geometry": True,
                         "explicit_rim_or_preserved_surface_geometry_available": False, "pass": False},
        "surface_generation": "NOT_RUN", "scientific_qc": "FAIL_CONTROL_AND_TEMPORAL_TARGET_GATE",
        "validation": {
            "design": "spatially blocked holdout of independently mapped elevation controls; never random cells and never occurrence labels",
            "metrics": ["median_absolute_vertical_residual_m", "bias_m", "IQR_coverage", "residual_by_control_type", "distance-stratified residual"],
            "geological_checks": ["no surface below contemporaneous basement/contact", "no unsupported summit", "family-consistent gradients",
                                    "cross-section continuity except mapped structures", "holdout coverage by ensemble"],
            "status": "NOT_RUN_NO_ADMISSIBLE_NUMERIC_CONTROL_SET",
        },
        "production_h_authorised_next_stage": False,
    }


def ensemble_members(n: int = 48) -> pd.DataFrame:
    """Deterministic candidate parameter design; surface execution is gated off."""
    rng = np.random.default_rng(SEED)
    families = np.array(["RBF_THIN_PLATE", "TENSION_SPLINE", "CONSTRAINED_ENVELOPE"])
    rows = []
    for i in range(n):
        rows.append({
            "member_id": f"ASO_GEN2_{i + 1:03d}", "seed": SEED,
            "surface_family": families[i % len(families)],
            "rim_weight": round(float(rng.uniform(0.7, 1.3)), 6),
            "flank_weight": round(float(rng.uniform(0.5, 1.5)), 6),
            "contact_weight": round(float(rng.uniform(0.2, 0.8)), 6),
            "gradient_prior_deg": round(float(rng.uniform(2.0, 12.0)), 6),
            "curvature_penalty": round(float(rng.uniform(0.1, 2.0)), 6),
            "control_jitter_m": round(float(rng.uniform(10.0, 75.0)), 6),
            "execution_status": "NOT_RUN_CONTROL_GATE", "accepted_surface": False,
            "rejection_reason": "No admissible numeric palaeoelevation control set and temporal target unresolved",
        })
    return pd.DataFrame(rows)


def h_feature_ontology() -> list[dict]:
    def r(name, definition, units, interpretation, availability, redundancy, leakage, propagation, status):
        return dict(feature=name, definition=definition, units=units, scientific_interpretation=interpretation,
                    availability=availability, potential_redundancy=redundancy, potential_leakage=leakage,
                    uncertainty_propagation=propagation, approval_status=status)
    return [
        r("Z_palaeo", "ensemble palaeoelevation", "m", "reconstructed surface height", "full/partial surface families", "base for many derivatives", "none if controls are Y-blind", "median, IQR and SD", "CORE"),
        r("D", "Z_palaeo - Z_modern", "m", "apparent removed/fill depth conditional on target age", "full/partial", "correlated with Z_palaeo", "modern DEM must predate event", "member-wise difference", "CORE"),
        r("palaeoslope", "gradient magnitude of each Z_palaeo member", "degrees", "palaeosurface steepness", "full/partial", "with gradient(D) and DeltaS", "derivative amplifies artifacts", "derive per member then summarise", "CORE"),
        r("palaeoaspect_sin_cos", "sin/cos of ensemble-member aspect", "unitless", "orientation without circular discontinuity", "full/partial", "two columns", "none", "circular ensemble summaries", "CANDIDATE"),
        r("palaeocurvature", "second derivatives of each Z_palaeo member", "1/m", "palaeosurface form", "full/partial", "highly scale-dependent", "artifact-sensitive", "derive per member at frozen scales", "SENSITIVITY_ONLY"),
        r("palaeorelief", "max-min Z_palaeo in frozen window", "m", "constructional relief", "full/partial", "with palaeo-TPI", "none", "member-wise windows", "CANDIDATE"),
        r("palaeo_TPI", "Z_palaeo minus neighbourhood mean", "m", "relative palaeoposition", "full/partial", "with relief/relative elevation", "none", "member-wise windows", "CANDIDATE"),
        r("palaeo_relative_elevation", "(Z-min)/(max-min) in frozen window", "0-1", "relative palaeoposition", "full/partial", "with TPI", "none", "member-wise ratio", "CANDIDATE"),
        r("palaeo_flank_position", "distance-normalised position from independently defined vent/rim to toe", "0-1", "edifice position", "radial/caldera families", "with relative elevation", "centre must not use Y", "ensemble over centre/toe geometry", "FAMILY_SPECIFIC"),
        r("D_local", "D minus neighbourhood median D", "m", "local erosion contrast", "full/partial", "with local erosion contrast", "none", "member-wise robust filter", "CANDIDATE"),
        r("gradient_D", "gradient magnitude of D", "unitless", "spatial change in reconstructed loss", "full/partial", "with DeltaS", "artifact-sensitive", "member-wise derivative", "SENSITIVITY_ONLY"),
        r("relative_erosion_depth", "D divided by local palaeorelief", "unitless", "erosion relative to available relief", "full/partial", "with Phi", "denominator instability", "member-wise ratio plus denominator flag", "CANDIDATE"),
        r("distance_from_reconstructed_flank", "distance to independently constrained palaeoflank boundary", "m", "proximity to edifice margin", "radial/remnant families", "none", "boundary must be independent of Y", "ensemble distance summaries", "FAMILY_SPECIFIC"),
        r("DeltaS", "palaeoslope - modern slope", "degrees", "slope change", "full/partial", "with gradient_D", "modern DEM chronology", "member-wise difference", "CANDIDATE"),
        r("time_since_emplacement", "event date minus mapped unit age distribution", "years", "exposure/weathering time", "dated units and age-proxy family", "with mapped unit age", "broad ages can create false precision", "sample age interval/distribution", "CORE_WHEN_DATED"),
        r("mapped_volcanic_unit_age", "source age interval/category", "years or category", "geological history", "mapped dated units", "with time since emplacement", "none", "retain bounds/category", "CORE_WHEN_DATED"),
        r("reconstruction_support", "weighted effective control support", "0-1", "degree of evidential constraint", "all H states except unavailable", "with control distance", "none", "summarise weights/family availability", "CORE"),
        r("ensemble_IQR", "Q75-Q25 of Z_palaeo", "m", "within-design uncertainty", "full/partial", "with ensemble SD", "none", "direct ensemble statistic", "CORE"),
        r("ensemble_SD", "SD of Z_palaeo", "m", "within-design uncertainty", "full/partial", "with IQR", "none", "direct ensemble statistic", "CORE"),
        r("control_distance", "distance to nearest admissible control", "m", "extrapolation distance", "all reconstructed families", "with support", "none", "member-independent plus type-specific variants", "CORE"),
        r("structural_disagreement", "between-structure-scenario spread", "m", "uncertainty from faults/rim/contact choices", "structurally constrained families", "part of total spread", "none", "variance component", "CORE_WHEN_APPLICABLE"),
        r("family_disagreement", "spread of family-level medians", "m", "model-form uncertainty", "multi-family ensembles", "part of total spread", "none", "between-family variance", "CORE_WHEN_MULTIFAMILY"),
        r("Psi", "D * S_modern", "m*degree", "depth-steepness interaction", "full/partial", "interaction of existing terms", "no direct leakage", "member-wise product", "EXPLORATORY"),
        r("Phi", "D / (Z_palaeo + epsilon)", "unitless", "fractional relief loss", "full/partial", "with relative erosion depth", "unstable near datum/sea level", "member-wise ratio with denominator guard", "REJECT_AS_GLOBAL_CORE"),
        r("DeltaD_anomaly", "D minus externally expected D for matched volcanic morphology", "m", "departure from external empirical expectation", "future only", "with D_local", "self-reference if same landscape defines expectation", "external-reference predictive distribution", "DEFER_EXTERNAL_REFERENCE_REQUIRED"),
    ]
