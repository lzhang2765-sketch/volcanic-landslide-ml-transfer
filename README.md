# Geographical Transferability of Machine-Learning Landslide Susceptibility Models Across Volcanic Landscapes

This repository is a publication package for a six-landscape study asking: **how well do landslide-susceptibility models trained on some volcanic landscapes rank affected terrain in a geographically different volcanic landscape?**

Geographical transfer matters because performance measured by mixing nearby terrain units can be optimistic. Here, each outer evaluation fold withholds one complete landscape and trains on the other five. The design evaluates transfer within a development set; it is **not** an untouched external validation.

## Study landscapes and data

| Landscape | Country | Terrain units | Labelled affected units |
|---|---|---:|---:|
| Kumamoto–Aso (2016) | Japan | 354,123 | 656 |
| Noto Peninsula (2024) | Japan | 96,812 | 2,148 |
| Lombok–Rinjani (2018) | Indonesia | 208,443 | 3,102 |
| Cotabato–Davao (2019) | Philippines | 78,344 | 298 |
| Atitlán–Stan (2005) | Guatemala | 27,272 | 1,554 |
| Dominica–Erika (2015) | Dominica | 70,852 | 1,035 |
| **Total** |  | **835,846** | **8,793** |

The analysis used 90 m terrain units and 60 terrain-morphology predictors. A positive label means that a unit met the operational mapping rule for its landscape. An operational negative means that no mapped positive was recorded under that rule inside the governed observation domain; it is **not confirmed stable terrain**. Inventory geometry, completeness, timing and mapping thresholds differ among landscapes.

## Methods

Four model families were compared with nested leave-one-landscape-out (LOLO) evaluation:

- L2-regularised logistic regression;
- random forest;
- histogram gradient boosting; and
- XGBoost.

Inner landscape holdouts selected hyperparameters. Outer folds evaluated every valid unit in the withheld landscape. Average precision (AP) measured ranking performance, while precision–recall lift divided AP by the positive-label prevalence. Lift therefore answers a different question from AP and is not a calibrated probability. Spatial-block bootstrap intervals describe within-landscape spatial sensitivity conditional on fitted models and inventories; they do not quantify variation across future landscapes.

## Main archived results

The selected development model was **L2 logistic regression with the 60 morphology predictors**. Archived outputs report:

- median AP: **0.0285931177553**;
- median precision–recall lift: **2.56665260507**;
- lowest fold lift: **1.03162698154** (Cotabato–Davao); and
- highest fold lift: **4.05232380304** (Atitlán–Stan).

These are development estimates, not evidence of operational deployment readiness. The scores rank susceptibility; they do not estimate landslide timing (hazard) or consequences to people and assets (risk).

Additional analyses examined training-landscape composition, absolute-elevation representation and Cotabato inventory definition. They are exploratory or post-selection analyses and do not replace the frozen primary comparison. Geological-history work was a feasibility screen: only a partial Kumamoto reconstruction passed the frozen gates, so no complete history-inclusive V2 predictive model was trained.

## Reproducing the package checks

Python 3.10 is the recorded environment. Create an environment, install the pinned dependencies, and run the lightweight release tests:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

On macOS, the XGBoost wheel also requires the system OpenMP runtime. Install it with `brew install libomp` before importing or running XGBoost. This is an operating-system prerequisite rather than a Python package; the selected logistic model and the lightweight release tests do not require an XGBoost fit.

These tests verify archived results and guardrails without fitting models. Full model execution requires the governed source datasets and frozen intermediate arrays, which are not distributed here. See [docs/reproducibility.md](docs/reproducibility.md) for exact status labels and commands.

## Repository structure

```text
config/                     Historical runtime configuration used by sfv2
configs/                    Frozen publication-facing experiment protocols
src/sfv2/                   Canonical V2 implementation snapshot
scripts/                    Standalone audit and sensitivity scripts
results/tables/             Small archived numerical outputs
results/figures/            Selected summary figures (no susceptibility atlas)
docs/                       Methods, data, limitations and traceability
tests/                      Lightweight archived-output verification
```

## Data availability and licensing

Raw DEMs, inventories, processed terrain-unit tables, prediction arrays, model caches and GIS rasters are intentionally excluded. Obtain source data from their original providers and comply with each provider's terms. The item-level review confirms attribution routes for GSI, Copernicus GLO-30, Cotabato and Dominica, while Lombok, Atitlán, CHELSA data-file terms and several national geology files remain unresolved for redistribution. The complete derived dataset is therefore not cleared. See [docs/datasets.md](docs/datasets.md) and [docs/licensing.md](docs/licensing.md).

Original project code is released under the [MIT License](LICENSE), limited by the explicit [licence scope](LICENSE_SCOPE.md). The MIT grant does not apply to dependencies, datasets, archived scientific results/figures, provider material or other third-party content; those remain governed separately. The repository continues to exclude all raw/processed scientific data and unit-level spatial products.

## Limitations

All six landscapes influenced model development or selection. Labels are heterogeneous and incomplete; AP depends on prevalence; calibration did not transfer reliably; the number of independent landscapes is six; sensitivity analyses were not all prespecified; and no seventh-landscape validation was completed. See [docs/limitations.md](docs/limitations.md).

## Citation

Citation metadata is provided in `CITATION.cff`. A paper DOI and public repository URL should be added after approval and publication.

## AI assistance

AI tools substantially assisted programming, debugging, automated analysis and auditing, figures, reference checking, and manuscript drafting/revision; this was not limited to proofreading. The student made the research decisions, reviewed and modified generated code, ran the analyses, processed some data manually, and checked reported values against stored outputs and literature. ISEF-specific student-authorship and disclosure boundaries are recorded in [docs/ai_assistance.md](docs/ai_assistance.md).

No personal contact information is included. Use the future repository issue tracker for public questions after publication.
