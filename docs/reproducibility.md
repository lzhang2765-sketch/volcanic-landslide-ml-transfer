# Reproducibility status

## Status definitions

- **A — Fully reproduced:** the computation was rerun from source inputs in this release audit.
- **B — Verified against archived outputs:** hashes, identities or independent calculations agree with frozen outputs.
- **C — Code available, execution not verified:** implementation is present but was not rerun here.
- **D — Missing dependencies or restricted data:** execution requires unavailable/non-redistributable inputs.
- **E — Unresolved provenance:** a required input or claim cannot yet be traced adequately.

## Evidence by experiment

| Experiment | Status | Evidence | Remaining limitation |
|---|---|---|---|
| Dataset totals and six-landscape identity | B | Stage 5A register and mentor integrity report agree on 835,846 units and 8,793 positives; release tests recompute totals | Source terrain-unit data excluded |
| Four-family Stage 4E-B comparison | B | Frozen comparison, LOLO and selection tables are internally consistent; selection freeze identifies logistic `M` | No retraining in this preparation pass |
| Selected-model per-landscape results | B | Exact AP/lift values agree across archived Stage 4E-B, Stage 5A and mentor-package records | Predictions and fitted models excluded |
| Spatial-block bootstrap | B | Frozen table and manuscript outputs are traceable; formula/result identities tested | Bootstrap was not rerun |
| Elevation representation sensitivity | B | Protocol, code manifest, baseline reproduction audit and archived outputs retained | Requires frozen NPZ inputs for rerun |
| Training-composition/factor analysis | B/C | Archived tables and scripts retained; source manifest records frozen inputs | Post hoc; not rerun here |
| Cotabato inventory-definition sensitivity | B/C | Frozen primary-vs-sensitivity guard and output table retained | Alternate inventories excluded |
| Geological-history feasibility | B/C | Gate matrix and reconstructability summary agree with Stage 5G report; code retained | Source reconstruction arrays excluded; no predictive H model exists |
| Raw-data processing and feature extraction | C/D | Canonical V2 source snapshot retained | Raw data, large rasters and some provider permissions absent |
| Untouched external validation | D | No such completed experiment exists | A seventh landscape is required |

No major manuscript result is assigned status A by this release-preparation pass because computationally expensive fitting was intentionally not rerun.

## Lightweight verification

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The tests recompute identities from the included archived tables and confirm selection, exposure and geological-history guardrails. They do not access the authoritative workspace.

### Clean-environment verification

On 6 November 2026, a new Python 3.10.9 virtual environment was created in a temporary directory and populated only from `requirements.txt`. `pip check` reported no broken Python requirements, and all eight release tests passed. Core Stage 4B/4E-B/5G modules imported; an in-memory scikit-learn logistic fit, GeoPandas reprojection and Rasterio read/write test passed; and the Whitebox Python wrapper imported without downloading its executable or accessing scientific data.

The macOS XGBoost 2.1.4 wheel requires `libomp.dylib`. Its import fails on a machine without that operating-system runtime even though `pip` installation succeeds. With an existing isolated OpenMP runtime exposed through the dynamic-library path, XGBoost imported and completed a toy in-memory fit. macOS users must install `libomp` (normally `brew install libomp`) before using XGBoost. No authoritative input or output was opened for writing during these checks.

## Full execution

Full execution requires separately obtained inputs whose hashes match the original provenance records. The most relevant commands in the preserved stage architecture are:

```bash
PYTHONPATH=src python -m sfv2.build_stage4a
PYTHONPATH=src python -m sfv2.build_stage4b
PYTHONPATH=src python -m sfv2.build_stage4eb --help
```

Do not run these against substituted data and do not interpret a successful import as scientific reproduction. Before any full rerun, restore the exact input hashes, Python/package versions, random seeds and frozen protocols. Several historical orchestration modules depend on source artifacts intentionally omitted from this public package; those paths are status D until the data-release boundary is resolved.
