"""Stage 4C-B3 (Gen-3) pre-registration: one-sided structurally-aware palaeosurface reconstruction, frozen BEFORE any Gen-3 fit.

Pure data.  Motivated only by the Y-blind Gen-2 diagnosis (central-trend estimand -> negative removal 0.29-0.52) recorded in
outputs/stage4c_b2.  Nothing here was chosen from landslide Y, Omega labels, Stage 4B outputs or any predictive quantity.
Validation gates are the Stage 4C-A gates verbatim (asserted at build time).
"""
from __future__ import annotations

AS_OF = "2026-10-07"
SEED = 4502
RECONSTRUCTED = ["KUMAMOTO_ASO_2016", "LOMBOK_RINJANI_2018", "COTABATO_DAVAO_2019", "ATITLAN_STAN_2005", "DOMINICA_ERIKA_2015"]

ESTIMAND = dict(
    symbol="S*(x)",
    definition="plausible pre-erosional constructional reference elevation at x: a conservative UPPER constructional surface consistent with the "
               "preserved (Q1/Q2) remnant controls and the structural geometry of the volcanic landscape",
    is_not=["expected modern summit elevation", "smoothed modern DEM", "exact original elevation", "maximum conceivable elevation",
            "a surface lifted until negative removal passes"],
    summary="ensemble S_j(x), j = 1..J; P05 / P50 / P95; the scenario median (P50) is the primary reconstruction for every frozen gate (as Stage 4C-A)",
    residual_convention="r = z_control - S(x): r > 0 means the reference lies BELOW a preserved control (under-prediction), which the one-sided losses penalise more",
    modern_dem_role="used only for frozen control derivation, segmentation and support geometry; NEVER as a cell-wise lower bound on S",
)

# ------------------------------------------------------------------------------------------------ one-sided loss families (small, fixed)
LOSSES = {
    "QUANTILE_q0.75": dict(family="UPPER_QUANTILE", q=0.75), "QUANTILE_q0.85": dict(family="UPPER_QUANTILE", q=0.85),
    "QUANTILE_q0.90": dict(family="UPPER_QUANTILE", q=0.90),
    "ASYM_HUBER_a2": dict(family="ASYMMETRIC_ROBUST", a=2.0), "ASYM_HUBER_a4": dict(family="ASYMMETRIC_ROBUST", a=4.0),
    "SOFT_UPPER_l4": dict(family="SOFT_UPPER_ENVELOPE", lam=4.0), "SOFT_UPPER_l16": dict(family="SOFT_UPPER_ENVELOPE", lam=16.0),
}
LOSS_DEFINITIONS = dict(
    UPPER_QUANTILE="rho_q(r) = r (q - I[r < 0]); minimised by IRLS with weights s * m(r) / max(|r|, 0.1 s), m = q if r > 0 else 1 - q (s = robust residual scale)",
    ASYMMETRIC_ROBUST="Huber(k = 1.345 s) with weight multiplied by a when r > 0 (under-prediction) and 1 otherwise",
    SOFT_UPPER_ENVELOPE="symmetric Huber + lam * sum (r - s)_+^2 : soft penalty when a control lies more than one robust scale above the reference",
    hard_constraint="NONE (no S >= control constraint; no S >= modern DEM constraint)",
    irls_iterations=30, huber_k=1.345, scale="s = 1.4826 MAD(r - median r), floor 1 m",
)

# ------------------------------------------------------------------------------------------------ geometry candidates (small; one-sided versions)
# Each architecture = geometry x loss.  Component tuple: (component_type, family, domain_rule, history_group).
GEOMETRIES = {
    "KUMAMOTO_ASO_2016": {
        "K_APRON_MEDIUM": [("VOLCANIC_APRON", "PSPLINE_ENVELOPE:MEDIUM", "ALL", "LP_PFL")],
        "K_RIM_SECTOR+APRON_MEDIUM": [("CALDERA_RIM_SECTOR", "CALDERA_RIM_SECTOR", "RIM_ZONE", "LP_PFL"),
                                      ("VOLCANIC_APRON", "PSPLINE_ENVELOPE:MEDIUM", "DISTAL_ZONE", "LP_PFL")],
    },
    "LOMBOK_RINJANI_2018": {
        "L_ELLIPTICAL_K1": [("RADIAL_EDIFICE", "RADIAL_ELLIPTICAL_K1", "ALL", "QUATERNARY_EDIFICE")],
        "L_SECTOR_K3": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "ALL", "QUATERNARY_EDIFICE")],
        "L_SECTOR_K3+PSPLINE": [("RADIAL_EDIFICE", "RADIAL_K3_PLUS_PSPLINE", "ALL", "QUATERNARY_EDIFICE")],
    },
    "COTABATO_DAVAO_2019": {
        "C_ELLIPTICAL_K1": [("RADIAL_EDIFICE", "RADIAL_ELLIPTICAL_K1", "ALL", "PQ_APO")],
        "C_SECTOR_K3": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "ALL", "PQ_APO")],
        "C_EDIFICE_K3+APRON_STIFF": [("RADIAL_EDIFICE", "RADIAL_SECTOR_K3", "CODES:I-QV", "PQ_APO"), ("VOLCANIC_APRON", "PSPLINE_ENVELOPE:STIFF", "CODES:I-QVP", "PQ_APO")],
    },
    "ATITLAN_STAN_2005": {
        "A_SINGLE_MEDIUM": [("PRESERVED_REMNANT", "PSPLINE_ENVELOPE:MEDIUM", "ALL", "ALL_UNITS")],
        "A_SEPARATE_TV_QP_STIFF": [("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:STIFF", "CODES:Tv", "TV"), ("PLATEAU", "PSPLINE_ENVELOPE:STIFF", "CODES:Qp", "QP")],
        "A_SEPARATE_TV_QP_MEDIUM": [("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:MEDIUM", "CODES:Tv", "TV"), ("PLATEAU", "PSPLINE_ENVELOPE:MEDIUM", "CODES:Qp", "QP")],
    },
    "DOMINICA_ERIKA_2015": {
        "D_CENTRES_K1+OLDER_STIFF": [("RADIAL_EDIFICE", "RADIAL_CIRCULAR_K1", "YOUNG_CENTRE", "YOUNG"), ("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:STIFF", "REST", "OLDER")],
        "D_CENTRES_K1+OLDER_MEDIUM": [("RADIAL_EDIFICE", "RADIAL_CIRCULAR_K1", "YOUNG_CENTRE", "YOUNG"), ("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:MEDIUM", "REST", "OLDER")],
        "D_YOUNG_UE_K1+OLDER_MEDIUM": [("YOUNGER_VOLCANIC_TERRAIN", "RADIAL_CIRCULAR_K1:ALL_YOUNG_CENTRES_UPPER_ENVELOPE", "DOMAIN:RADIAL", "YOUNG"),
                                       ("OLDER_VOLCANIC_TERRAIN", "PSPLINE_ENVELOPE:MEDIUM", "DOMAIN:ENVELOPE", "OLDER")],
    },
}
EXCLUDED_FROM_GEN3 = dict(
    island_wide_or_global_single_smooth_for_Dominica="excluded: geology says multi-centre young + older terrain; Gen-2 D4 won only by central fit",
    gen1_quadratic="excluded everywhere (Gen-1/Gen-2 showed it cannot represent any landscape)",
    stiff_single_kumamoto_and_circular_k1_lombok_cotabato="excluded (dominated in Gen-2 and not needed to test the estimand change)",
    pusuk_nangi="UNRESOLVED: no authoritative coordinates; never used as centres",
)
REUSED_FROM_GEN2 = "component rules, P-spline/radial families, apex taper, ellipse search, 8 km knots, kappa STIFF 10 / MEDIUM 1, rim zone 5 km, centre qualification >= 8 controls"

# ------------------------------------------------------------------------------------------------ boundary classification (decided BEFORE fitting, Y-free)
BOUNDARY_CLASS = {
    "same history group": "NUMERICAL (continuity expected): smooth Gaussian blending (sigma 1.5 km); Gen-2 seam thresholds apply as pass/fail",
    "different history groups": "GEOLOGICAL_DISCONTINUITY (continuity not expected): hard membership, no blending; separate reconstruction/support domains; "
                                "slope/relief computed on each group's own continuous surface; seam metrics reported, not pass/fail",
}
HISTORY_GROUP_EVIDENCE = {
    "ATITLAN_STAN_2005": "Tv = Tertiary (Mio-Pliocene) tuffs, lavas, laharic material (dissected highland); Qp = Quaternary 'thick pumice-ash fills and covers' "
                         "(MAGA 1:500,000 legend, frozen mapping). Fills were emplaced in/over the already-eroded Tv relief -> different constructional surfaces "
                         "(Gen-1 Y-free grid diagnostic: Qp median 2,375 m vs Tv 2,560 m). Boundary = GEOLOGICAL_DISCONTINUITY.",
    "DOMINICA_ERIKA_2015": "Younger Pleistocene-Recent block-and-ash aprons, ignimbrites and domes (< 1.8 Ma) built against older Miocene-Older-Pleistocene "
                           "edifices (official 1:100,000 age legend) -> GEOLOGICAL_DISCONTINUITY between YOUNG and OLDER; young centres blend smoothly with each other.",
    "KUMAMOTO_ASO_2016": "rim and apron are the same Late-Pleistocene pyroclastic-flow sheets -> NUMERICAL (continuity expected).",
    "LOMBOK_RINJANI_2018": "single component (all Quaternary edifice units) -> no internal boundary.",
    "COTABATO_DAVAO_2019": "I-QV edifice and I-QVP piedmont deposits are the same Pliocene-Quaternary system (MGB) -> NUMERICAL (continuity expected).",
}

# ------------------------------------------------------------------------------------------------ controls, nesting, selection
CONTROLS = "Gen-1 Y-blind Q1/Q2 controls reused verbatim (same as Gen-2); each control constrains only its own component; held-out elevations never enter any fit"
NESTED = dict(outer="identical to Gen-2: sectors (Lombok, Cotabato), 2 km blocks (Kumamoto, Atitlan), hybrid (Dominica); 8 super-folds; "
                    "selection for every outer group uses only controls outside its super-fold",
              inner="leave-one-super-fold-out over the remaining super-folds; central structural variant, P95 controls, no subsample, no offset")
SELECTION = dict(
    E="median |r| / R_ctrl over pooled inner held-out controls",
    U="fraction of inner held-out controls with r > tau_in (control lies materially ABOVE the predicted reference); tau_in = median robust scale of the inner fits",
    B="|mean r| / R_ctrl (absolute inner bias: penalises a surface lifted (or dropped) as a whole)",
    C="effective dof (trace of hat / parameter count; + 2 per ellipse search)",
    score="score = E + 0.25 U + 0.5 B + 0.0005 C", alpha_U=0.25, gamma_B=0.5, beta_C=0.0005,
    physical_admissibility="a candidate is inadmissible for a fold if its central fit on that fold's training controls has a 90 m Horn slope > 40 deg anywhere in its "
                           "applicable domain within 3 km of a training control (if no candidate is admissible, all are considered)",
    rule="among admissible candidates, take those with score <= min score + 0.01; choose the lowest effective dof (rounded to 0.5), ties -> lowest score",
    justification="E keeps held-out accuracy primary; alpha_U = 0.25 makes a 0.2 reduction of above-reference controls worth 0.05 E (half the residual gate); "
                  "gamma_B = 0.5 charges whole-surface lifting at half the rate of E so that 'raise everything' cannot win; beta_C as Gen-2",
)
ENSEMBLE = dict(summit_percentile=(90, 95, 99), tolerance_offset_sigma=(-1.0, 0.0, 1.0), control_subsample_replicates=2, subsample_fraction=0.8,
                structural="as Gen-2 (radial 12 p x radius variants; P-spline 2 knot offsets)", loss="the selected loss is fixed (not a scenario dimension)", weighting="equal")

# ------------------------------------------------------------------------------------------------ gates, diagnostics, release, stopping
GATES = dict(median_abs_residual_fraction_max=0.10, abs_bias_fraction_max=0.05, holdout_band_fraction_min=0.60, min_admissible_controls=12,
             min_support_fraction_of_units=0.25, negative_removal_fraction_max_outside_mapped_young_deposits=0.10, max_reconstructed_slope_deg_constructional=40.0)
NEGATIVE_REMOVAL = dict(D="D(x) = S_P50(x) - Z_modern(x) at Stage 4A units", negative="D < -tau_L, tau_L = median scenario robust control scale (Gen-1/Gen-2 definition)",
                        young_deposit_exceptions="unchanged: none (identical to Gen-1 / Gen-2)",
                        diagnostics=["global", "by geological unit", "by component", "by distance-to-control class (0-0.5, 0.5-1, 1-2, 2-3 km)",
                                     "spatial clusters (8-connected negative unit cells)", "magnitude quantiles of negative D"])
BOUNDARY_QA = dict(numerical="as Gen-2: median seam step/R_L <= 0.10, P95 <= 0.25; slope excess <= 5 deg; ridge/trough fraction <= 0.05; spread ratio <= 2",
                   geological="reported only (continuity not expected)")
PHYSICAL_QA = dict(pits="support fraction in closed depressions deeper than tau_L <= 0.01 (as Gen-2)", collapse="caldera interiors masked",
                   positive_offset_diagnostic="DIAGNOSTIC ONLY (no prior threshold): P99 of D/R_L, fraction of support with D > 0.5 R_L, "
                                              "and Gen-3 minus Gen-2 P50 shift: mean, SD, SD/|mean| (uniform-shift index)")
RELEASE = dict(rule="H released only if every frozen gate, numerical boundary QA and physical QA pass with explicit support and a scenario ensemble",
               VERIFIED_GEN3_FULL="supported unit fraction >= 0.90", VERIFIED_GEN3_PARTIAL="0.25 <= supported fraction < 0.90",
               h_dataset="created only if at least one landscape is VERIFIED_GEN3")
STOPPING_RULE = ("If N_VERIFIED_GEN3_H_CAPABLE < 3: do NOT create Gen-4; stop the reconstruction-development sequence and report a failure analysis on whether "
                 "globally standardised palaeosurface H is feasible with the available evidence. A further generation requires genuinely new external "
                 "geological evidence or a distinct scientific hypothesis, not another model/parameter change.")
