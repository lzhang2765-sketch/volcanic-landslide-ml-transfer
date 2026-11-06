"""Stage 4C-B2 (Gen-2) pre-registration: structurally-aware piecewise ensemble reconstruction, frozen BEFORE any Gen-2 fit.

Pure data (no imports, no file access).  Motivated only by the Y-blind Gen-1 reconstruction-validation failures recorded in
outputs/stage4c_b (and a Y-free Gen-1 grid diagnostic of Atitlan).  No landslide Y, Omega label, Stage 4B result or predictive
quantity informed anything here.  The Stage 4C-A validation gates are reused verbatim (asserted equal at build time).
"""
from __future__ import annotations

AS_OF = "2026-10-07"
SEED = 4402
RECONSTRUCTED = ["KUMAMOTO_ASO_2016", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]
GEN1_DIAGNOSTICS = {  # authoritative Gen-1 Y-blind failures (outputs/stage4c_b/STAGE4C_B_RECONSTRUCTION_VALIDATION.tsv)
    "KUMAMOTO_ASO_2016": "median|res|/R_L 0.263; negative removal 0.164: one quadratic cannot represent steep rim terrain and the distal apron",
    "LOMBOK_RINJANI_2018": "median|res|/R_L 0.192; negative removal 0.234; max slope 40.5 deg: one Rinjani cone; apex asymmetry singularity; Pusuk/Nangi unresolved",
    "COTABATO_DAVAO_2019": "median|res|/R_L 0.129; negative removal 0.152: single radial Apo surface insufficient",
    "ATITLAN_STAN_2005": "negative removal 0.129 only: Y-free grid diagnostic shows it on the high Tv highland (Tv 16.7% vs Qp 4.9%; negative cells median 2949 m)",
    "DOMINICA_ERIKA_2015": "median|res|/R_L 0.110; negative removal 0.207: young multi-centre + older terrain poorly represented",
}

# ------------------------------------------------------------------------------------------------ controls
CONTROLS = dict(source="outputs/stage4c_b/STAGE4C_B_CONTROLS.tsv (Gen-1 Y-blind controls, read-only, reused verbatim: same locations, P90/P95/P99 elevations, Q classes, 1 km thinning)",
                fit_classes=["Q1", "Q2"], component_assignment="a control constrains only the component whose domain contains it (geology code + structural zone)",
                note="Gen-1 outward-decrease rule rejected 1 control in total, so re-deriving controls per component would change nothing material")

# ------------------------------------------------------------------------------------------------ surface-family library
FAMILIES = {
    "QUADRATIC_ENVELOPE": dict(parameters=6, description="Gen-1 global 2nd-order trend (baseline)"),
    "PSPLINE_ENVELOPE": dict(description="tensor-product cubic B-spline on an absolute knot grid, 2nd-order difference penalty (P-spline); piecewise-local but smooth; "
                                         "null space = bilinear plane", knot_spacing_m=8000.0, min_knot_spacing_m=5000.0,
                             lambda_rule="lambda = kappa * n_train / n_basis; kappa STIFF = 10, MEDIUM = 1",
                             complexity="effective dof = trace of the penalised hat matrix"),
    "RADIAL_CIRCULAR_K1": dict(description="Z = B + f(r)[c0 + t(r)(c1 cos + c2 sin)]; f = (1 - r/R)_+^p; apex taper t(r) = r/(r + 1000 m) removes the Gen-1 apex singularity",
                               parameters=4),
    "RADIAL_ELLIPTICAL_K1": dict(description="as K1 with elliptical distance r_e = sqrt(q u^2 + v^2 / q) in axes rotated by phi; phi, q chosen by in-sample Huber loss on TRAINING controls only",
                                 phi_deg=(0, 30, 60, 90, 120, 150), q=(0.8, 0.6), parameters="4 linear + 2 grid"),
    "RADIAL_SECTOR_K3": dict(description="sector-wise radial: H(theta) = 3rd-order Fourier series (smooth across sectors, no wedges), apex taper on all non-constant terms",
                             parameters=8),
    "RADIAL_K3_PLUS_PSPLINE": dict(description="RADIAL_SECTOR_K3 + STIFF P-spline correction (penalised; radial columns unpenalised)"),
    "CALDERA_RIM_SECTOR": dict(description="RADIAL_SECTOR_K3 about the derived caldera-mask centroid, restricted to the rim-proximal annulus; caldera interior masked",
                               parameters=8),
}
RADIAL_STRUCTURAL_VARIANTS = dict(p=(1.0, 1.5, 2.0, 2.5), radius_factor=(1.1, 1.3, 1.6), central=dict(p=2.0, radius_factor=1.3),
                                  radius_definition="R = radius_factor x P95 distance of the component's training controls from its centre")
PSPLINE_STRUCTURAL_VARIANTS = dict(knot_offset_fraction=(0.0, 0.5), central=dict(knot_offset_fraction=0.0))
FIT = dict(huber_k=1.345, irls_iterations=12, sigma_floor_m=1.0, min_component_controls=8, min_centre_controls=8,
           radial_rejection=["c0 <= 0", "min over theta of H(theta)/c0 < 0.5 (evidence-bounded asymmetry)"])

# ------------------------------------------------------------------------------------------------ candidate architectures (structural alternatives)
# components: list of (component_type, family, domain rule).  Domain rules use geology codes of the landscape's applicable units and
# structural zones (distance to the caldera mask, nearest registered young centre).  Nothing uses Y.
CANDIDATES = {
    "KUMAMOTO_ASO_2016": {
        "K0_GEN1_QUADRATIC": [("VOLCANIC_APRON", "QUADRATIC_ENVELOPE", "ALL")],
        "K1_PSPLINE_STIFF": [("VOLCANIC_APRON", "PSPLINE_ENVELOPE:STIFF", "ALL")],
        "K2_PSPLINE_MEDIUM": [("VOLCANIC_APRON", "PSPLINE_ENVELOPE:MEDIUM", "ALL")],
        "K3_RIM_SECTOR+APRON_STIFF": [("CALDERA_RIM_SECTOR", "CALDERA_RIM_SECTOR", "RIM_ZONE"), ("VOLCANIC_APRON", "PSPLINE_ENVELOPE:STIFF", "DISTAL_ZONE")],
        "K4_RIM_SECTOR+APRON_MEDIUM": [("CALDERA_RIM_SECTOR", "CALDERA_RIM_SECTOR", "RIM_ZONE"), ("VOLCANIC_APRON", "PSPLINE_ENVELOPE:MEDIUM", "DISTAL_ZONE")],
    },
    "LOMBOK_RINJANI_2018": {
        "L0_CIRCULAR_K1": [("RADIAL_EDIFICE", "RADIAL_CIRCULAR_K1", "ALL")],
        "L1_ELLIPTICAL_K1": [("RADIAL_EDIFICE", "RADIAL_ELLIPTICAL_K1", "ALL")],
        "L2_SECTOR_K3": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "ALL")],
        "L3_SECTOR_K3+PSPLINE": [("RADIAL_EDIFICE", "RADIAL_K3_PLUS_PSPLINE", "ALL")],
        "L4_EDIFICE_K3+APRON_STIFF": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "CODES:Qhv,Qhv1"), ("VOLCANIC_APRON", "PSPLINE_ENVELOPE:STIFF", "CODES:Qvl")],
    },
    "COTABATO_DAVAO_2019": {
        "C0_CIRCULAR_K1": [("RADIAL_EDIFICE", "RADIAL_CIRCULAR_K1", "ALL")],
        "C1_ELLIPTICAL_K1": [("RADIAL_EDIFICE", "RADIAL_ELLIPTICAL_K1", "ALL")],
        "C2_SECTOR_K3": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "ALL")],
        "C3_SECTOR_K3+PSPLINE": [("RADIAL_EDIFICE", "RADIAL_K3_PLUS_PSPLINE", "ALL")],
        "C4_EDIFICE_K3+APRON_STIFF": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "CODES:I-QV"), ("VOLCANIC_APRON", "PSPLINE_ENVELOPE:STIFF", "CODES:I-QVP")],
    },
    "ATITLAN_STAN_2005": {
        "A0_GEN1_QUADRATIC": [("PRESERVED_REMNANT", "QUADRATIC_ENVELOPE", "ALL")],
        "A1_PSPLINE_STIFF": [("PRESERVED_REMNANT", "PSPLINE_ENVELOPE:STIFF", "ALL")],
        "A2_PSPLINE_MEDIUM": [("PRESERVED_REMNANT", "PSPLINE_ENVELOPE:MEDIUM", "ALL")],
        "A3_TV_REMNANT+QP_PLATEAU_STIFF": [("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:STIFF", "CODES:Tv"), ("PLATEAU", "PSPLINE_ENVELOPE:STIFF", "CODES:Qp")],
        "A4_TV_REMNANT+QP_PLATEAU_MEDIUM": [("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:MEDIUM", "CODES:Tv"), ("PLATEAU", "PSPLINE_ENVELOPE:MEDIUM", "CODES:Qp")],
    },
    "DOMINICA_ERIKA_2015": {
        "D0_GEN1_LIKE": [("YOUNGER_VOLCANIC_TERRAIN", "RADIAL_CIRCULAR_K1:ALL_YOUNG_CENTRES_UPPER_ENVELOPE", "DOMAIN:RADIAL"), ("OLDER_VOLCANIC_TERRAIN", "QUADRATIC_ENVELOPE", "DOMAIN:ENVELOPE")],
        "D1_CENTRES_K1+OLDER_STIFF": [("RADIAL_EDIFICE", "RADIAL_CIRCULAR_K1", "YOUNG_CENTRE"), ("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:STIFF", "REST")],
        "D2_CENTRES_K1+OLDER_MEDIUM": [("RADIAL_EDIFICE", "RADIAL_CIRCULAR_K1", "YOUNG_CENTRE"), ("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:MEDIUM", "REST")],
        "D3_ISLAND_PSPLINE_STIFF": [("PRESERVED_REMNANT", "PSPLINE_ENVELOPE:STIFF", "ALL")],
        "D4_ISLAND_PSPLINE_MEDIUM": [("PRESERVED_REMNANT", "PSPLINE_ENVELOPE:MEDIUM", "ALL")],
    },
}
STRUCTURAL_ZONES = dict(
    RIM_ZONE="applicable cells within 5,000 m outside the GSJ-derived Aso caldera-interior mask", DISTAL_ZONE="applicable cells beyond 5,000 m",
    YOUNG_CENTRE="young (Pleistocene-Recent) units assigned to the nearest registered Holocene GVP centre; a centre needs >= 8 Q1/Q2 controls in its domain, "
                 "otherwise its cells join REST", REST="all other applicable cells (older units and young cells of under-constrained centres)",
    rim_zone_m=5000.0,
)
PUSUK_NANGI = "UNRESOLVED: no authoritative coordinates (not separate GVP entries) and Sheet-1807 codes (Qhv, Qhv1) do not separate p/n/r products; no centred or non-centred component is created"
COMBINATION = dict(rule="geology/zone-constrained membership with smooth blending: w_k = Gaussian-smoothed (sigma 1,500 m) indicator of component k, normalised; "
                        "Z = sum_k w_k S_k (a convex combination: no additive stacking, no overshoot ridges)", blend_sigma_m=1500.0,
                   D0_exception="D0 (Gen-1-like baseline) keeps the Gen-1 upper envelope over young centres inside its young component")

# ------------------------------------------------------------------------------------------------ nested validation + complexity rule
NESTED = dict(
    outer_groups=dict(RADIAL_LANDSCAPES="leave-one-sector-out: 8 x 45 deg sectors about the nearest registered centre (Lombok: Rinjani; Cotabato: Apo)",
                      ENVELOPE_LANDSCAPES="leave-one-2-km-block-out (Kumamoto, Atitlan)",
                      DOMINICA="young-domain controls: sectors about the nearest young centre; older-domain controls: 2 km blocks (as Gen-1)"),
    super_folds="outer groups are partitioned into 8 super-folds (sector index; or 16 km tiles: (tx mod 4) + 4 (ty mod 2)); architecture selection for every outer group "
                "in super-fold F uses only controls outside F, so no held-out control ever influences selection",
    inner="leave-one-super-fold-out over the 7 remaining super-folds, central structural variant, P95 controls, no offset",
    outer_evaluation="the architecture selected for F is refitted (every structural/percentile/subsample scenario) without the held-out outer group and predicts it",
    production="final architecture = inner selection over all 8 super-folds; production ensemble fitted on all Q1/Q2 controls",
    normaliser="R_ctrl = P98 - P02 of the landscape's Q1/Q2 control P95 elevations (fixed per landscape, Y-free)",
)
COMPLEXITY_RULE = dict(score="score = median|inner residual| / R_ctrl + 0.0005 x effective_dof",
                       lambda_per_dof=0.0005, simpler_within_tolerance=0.01,
                       rule="pick the candidate with the lowest effective dof among those whose score <= min(score) + 0.01",
                       effective_dof="sum over components of trace(hat) (P-spline) or linear parameter count, + 2 per elliptical grid search")

# ------------------------------------------------------------------------------------------------ ensembles
ENSEMBLE = dict(summit_percentile=(90, 95, 99), tolerance_offset_sigma=(-1.0, 0.0, 1.0), control_subsample_replicates=2, subsample_fraction=0.8,
                structural="radial: p x radius_factor (12); P-spline: knot offset (2); quadratic: 1; multi-component scenarios use variant index v for every "
                           "component (cycling shorter lists)", weighting="equal")

# ------------------------------------------------------------------------------------------------ gates, QA, release
GATES = dict(median_abs_residual_fraction_max=0.10, abs_bias_fraction_max=0.05, holdout_band_fraction_min=0.60, min_admissible_controls=12,
             min_support_fraction_of_units=0.25, negative_removal_fraction_max_outside_mapped_young_deposits=0.10, max_reconstructed_slope_deg_constructional=40.0)
BOUNDARY_QA = dict(
    band="cells where the second-largest blending weight >= 0.10",
    seam_step="median |S_k - S_j| / R_L in the band <= 0.10 and P95 <= 0.25",
    slope_excess="P99 slope in band - P99 slope in component interiors <= 5 deg",
    ridge_trough="fraction of band cells with |Laplacian| > P99 interior |Laplacian| <= 0.05",
    spread_ratio="median scenario IQR in band / median IQR in interiors <= 2.0",
    single_component="no internal boundary: PASS by construction (support edges are not component boundaries)",
)
PHYSICAL_QA = dict(pits="fraction of support cells inside closed depressions of the scenario-median reference deeper than tau_L <= 0.01",
                   collapse="caldera interiors masked (never scored as erosion)", extrapolation="support <= 3 km from a fitted control; radial components r <= P95 control radius",
                   not_enforced="reference >= modern is NOT enforced; negative removal is never clipped")
SUPPORT = dict(max_control_distance_m=3000.0, note="identical to Gen-1")
STATUS_RULE = dict(VERIFIED_GEN2_FULL="all frozen gates + boundary QA + physical QA pass; supported unit fraction >= 0.90",
                   VERIFIED_GEN2_PARTIAL="as FULL with 0.25 <= supported fraction < 0.90", FAILED_GEN2="anything else", AGE_PROXY_ONLY="Noto (not reconstructed)")
