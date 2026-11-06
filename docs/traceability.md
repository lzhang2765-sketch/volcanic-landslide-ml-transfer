# Manuscript-to-code traceability

The authoritative manuscript for this audit is the G5.1 corrected DOCX, SHA-256 `160579d991d616f65e8d45aa7c3da004cfb37955eb5c43ac09e8ddf36571018b`, also copied byte-identically into the 25 October mentor package. The manuscript itself is not published in this code package.

| Major claim | Source script/code | Frozen configuration | Input dataset/artifact | Archived output | Experiment ID | Figure/table |
|---|---|---|---|---|---|---|
| Six landscapes; 835,846 units; 8,793 positives | `src/sfv2/build_stage5a.py` and production modules in `src/sfv2/` | Stage 5A contracts | Six governed terrain/feature/target streams; Stage 4A partitions | `results/tables/STAGE5A_DATASET_REGISTER.tsv` | STAGE5A | Figure 1; dataset table |
| 60 morphology predictors | `src/sfv2/build_stage4a.py`, `stage4a_base.py`, `stage4b.py` | Stage 4A manifest and Stage 4E-B preregistration | Stage 4A feature partitions | `results/tables/STAGE5A_FEATURE_DICTIONARY.tsv` | STAGE4A / STAGE5A | Methods; Figure 3/6 context |
| Nested six-fold LOLO comparison of four algorithms | `src/sfv2/stage4b.py`, `stage4eb.py`, `build_stage4eb.py` | `configs/STAGE4EB_*` | Frozen Stage 4A MEG NPZ partitions | `results/tables/baseline/STAGE4EB_LOLO_RESULTS.tsv` | STAGE4EB | Figure 2; model-comparison table |
| Logistic L2 + M selected | `src/sfv2/stage4eb.py` | selection rule and primary freeze | LOLO results plus paired bootstrap | `STAGE4EB_MODEL_COMPARISON.tsv`, `STAGE4EB_ROBUST_SELECTION.tsv` | STAGE4EB | Model-comparison table |
| Median AP 0.0285931177553; median lift 2.56665260507 | same as above | same | same | `STAGE4EB_MODEL_COMPARISON.tsv` | STAGE4EB | Abstract/results; Figure 2 |
| Lift range 1.03162698154–4.05232380304 | same as above | same | per-landscape held-out scores and labels | `STAGE4EB_LOLO_RESULTS.tsv` | STAGE4EB | Figure 2 |
| Within-landscape spatial intervals | `src/sfv2/stage4eb.py`, `stage4d.py` | 2,000 paired 3 km block resamples | frozen outer scores and `spatial_block_3000m` | `STAGE4EB_SPATIAL_BOOTSTRAP.tsv` | STAGE4EB | Figure 4 |
| Elevation representation sensitivity | `scripts/sensitivity/elevation/run_elevation_experiment.py` | `ELEVATION_EXPERIMENT_PROTOCOL.md` | frozen Stage 4A partitions and baseline scores | `results/tables/elevation/*` | ELEV_ABLATION_PROTOCOL_v1.0.0 | Figure 5; elevation supplement |
| Training-landscape composition/factor summaries | `scripts/sensitivity/transferability/s1_source_register.py`, `s2_factor_analysis.py`, `s3_dictionary_and_figures.py` | `TRANSFERABILITY_FACTOR_ANALYSIS_PLAN.md` | frozen primary/alternative Stage 4B results and metadata | `results/tables/transferability/*` | TRANSFER_FACTOR_PLAN_v1.0.0 | Figure 5; factor supplement |
| Cotabato inventory-definition sensitivity | `src/sfv2/stage4ea_audit.py`, `stage4eb_secondary.py` | Cotabato audit freeze; EQ1-primary guard | EQ1 target plus cumulative EQ3/EQ4 inventories | `STAGE4EB_COTABATO_SENSITIVITY.tsv` | STAGE4EA / STAGE4EB-secondary | Results sensitivity text |
| Geological-history work was feasibility-only | `src/sfv2/build_stage5g.py`; `scripts/geological_history/build_stage_d_audit.py` | unchanged Stage 4C/5G gates | frozen reconstruction validation and support artifacts | `results/tables/geological_history/*` | STAGE5G | Figure 8 |
| Only partial Kumamoto H passed; no MEGH model completed | `src/sfv2/build_stage5g.py` | `MEGH_UNLOCKED=NO` guard | Stage 4C Gen3 evidence | reconstructability summary and gate matrix | STAGE5G | Figure 8; limitations |

## Archived-output identity check

All 30 files under `results/` (24 tables and six figures) were SHA-256 compared with the authoritative V2 workspace. Every file had at least one byte-identical authoritative source. The baseline tables match `outputs/stage4e_b/`; dataset and feature registers match `outputs/stage5a/`; elevation tables match `AUSSEF_2026_ELEVATION_EXPERIMENT/`; transferability tables match `AUSSEF_2026_TRANSFERABILITY_FACTORS/`; geological-history tables match `outputs/stage5g/`; and the figures match their G5 manuscript sources. The release tests independently recompute the headline totals, selection and metrics from those included tables.

## Traceability gaps

- The raw and processed scientific datasets are not included, so end-to-end execution cannot be independently verified from this package alone.
- The selected summary figures are archived outputs; not every manuscript figure-generation script is included in a single uniform pipeline.
- The susceptibility atlas is excluded pending derived-data licensing review.
- No completed seventh-landscape external-validation artifact exists.
