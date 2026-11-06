# Repository preparation audit

**Prepared:** 6 November 2026  
**Package:** `volcanic-landslide-ml-transfer`  
**Publication action:** none performed

## Overall finding

The local package is technically coherent, installs in a fresh Python 3.10 environment and traces every headline numerical claim to frozen V2 outputs. The code/results-only release boundary excludes all source and unit-level spatial data. The owner approved MIT for original code and approved creation/push of a private GitHub repository, but did **not** approve public visibility. ISEF/school/AUSSEF compliance for the student's AI-assisted submission materials still requires human confirmation. Unresolved source-data rights prevent a later data/map release but do not place restricted data inside this package.

No Git commit, push, remote creation, deletion of original files or modification of a frozen scientific record was performed.

## Source directories inspected

| Location | Role | Finding |
|---|---|---|
| `Science_Fair_ML_V2/` | Canonical V2 research workspace | Authoritative code, frozen Stage 4E-B comparison, Stage 5A contracts, sensitivity outputs, Stage 5G feasibility evidence and manuscript lineage located |
| `Science_Fair_Extension/ML_V2/` | Historical pointer | Confirms `Science_Fair_ML_V2/` is canonical and V1 must remain separate |
| `Science_Fair_Extension/00_Governance/ML_V1_FREEZE/` | Frozen V1 pilot | Kept separate; not copied or modified |
| Authoritative Google Drive project root | Tweed scientific project | Searched read-only for competing ML/AUSSEF sources; no competing V2 workspace found |
| Parent `.git/` | Existing repository metadata | Remote is `https://github.com/lzhang2765-sketch/Tweed-Volcano-Reconstruction.git`, which is unrelated to this publication target |

An internet search found no repository matching `volcanic-landslide-ml-transfer`, the paper title, or `Science_Fair_ML_V2` under the locally recorded GitHub owner. This is evidence of no readily discoverable existing project repository, not a guarantee that no private/unindexed remote exists.

## Authoritative files identified

- **Implementation:** `Science_Fair_ML_V2/src/sfv2/*.py`.
- **Primary experiment:** `outputs/stage4e_b/STAGE4EB_*`.
- **Selected-model and dataset contracts:** `outputs/stage5a/STAGE5A_*`.
- **Elevation sensitivity:** `AUSSEF_2026_ELEVATION_EXPERIMENT/`.
- **Training-composition/factor analysis:** `AUSSEF_2026_TRANSFERABILITY_FACTORS/`.
- **Cotabato sensitivity:** Stage 4E-A audit and Stage 4E-B secondary output.
- **Geological-history feasibility:** `outputs/stage5g/` and `AUSSEF_2026_GEOLOGICAL_HISTORY_AUDIT/`.
- **Final paper source:** G5.1 corrected DOCX, SHA-256 `160579d991d616f65e8d45aa7c3da004cfb37955eb5c43ac09e8ddf36571018b`; its mentor-package copy is byte-identical.

### Manuscript preservation discrepancy

The correction log's hash `28461738b10149976bc924e450ea5c7e0d3e5a12b2982eaa28af1f0a1eb083f0` is still exactly preserved by `00_ORIGINAL_BACKUP/AUSSEF FINAL REVIEW (original backup).docx`. The root predecessor was later substantively edited or resaved on 25 October 2026: it hashes to `1aafab429957e6cc6bd740d91d8d0f6971592b2ceb611ec54a525d5e8c2d7e7d`, identifies revision 4 and has 429 non-empty paragraphs versus 382 in the backup, with 79 non-equal paragraph blocks. This explains the hash mismatch, but Git does not record who made the later revision or by what process. The corrected final file still matches its recorded hash and byte-identical mentor copy, so the later predecessor edit does not change the selected final manuscript authority. No manuscript was modified during this investigation.

## Files included

- 117 canonical `sfv2` Python modules, with six contact-text-only privacy redactions documented in `docs/release_redactions.md`.
- Frozen baseline model configurations, preregistrations and selection freeze.
- Elevation and transferability-factor protocols.
- Eight standalone sensitivity/audit scripts.
- 24 small archived result tables covering dataset identity, baseline comparison, elevation, transferability and geological-history feasibility.
- Six summary PNG figures: dataset composition, AP/prevalence, spatial bootstrap, transfer sensitivity, global SHAP summary and history-gate matrix.
- Documentation, citation metadata, dependency/data-rights review, pinned requirements and eight package-level tests.

## Files excluded

- All raw and processed scientific data.
- Stage 4A NPZ matrices, prediction arrays, fitted models, caches and checkpoints.
- GeoTIFFs, shapefiles, GeoPackages and large compressed archives.
- Susceptibility maps/atlas and per-unit susceptibility score exports pending derived-data licence review.
- Manuscript DOCX/PDF, logbooks, mentor-only documents and internal development logs.
- `.ml_site`, virtual environments, `__pycache__`, `.pytest_cache`, `.DS_Store` and other generated files.
- Frozen V1 files and unrelated Tweed reconstruction work.

## Security and privacy audit

| Check | Result | Evidence/action |
|---|---|---|
| API keys, tokens, private keys, passwords | PASS | Pattern scan found no candidate secret; package test repeats common-pattern checks |
| Local absolute/cloud paths | PASS | No machine-specific home, cloud-storage or notebook-drive path remains in public text |
| Email addresses | PASS | No address remains; six historical source modules had direct public data-custodian/corresponding-author addresses replaced with route descriptions |
| Student contact details | PASS | No contact information included |
| Raw/restricted data | PASS for package boundary | No raw/processed scientific or GIS files included |
| Large files | PASS | No file exceeds 10 MB; the 189-file package is about 4.6 MB |
| Caches/binaries | PASS | No pyc, DS_Store, DOCX, XLSX, PKL, NPZ, NPY, TIFF or gzip file included |
| Figure metadata | PASS | Selected PNGs identify Matplotlib only; no personal path or author metadata found |

## Licensing audit

- A provenance/header scan found no vendored third-party source. All direct Python dependencies are installed separately and use permissive BSD/MIT/PSF/Apache-2.0-family licences. The initially omitted direct imports were added to `requirements.txt`.
- The owner approved MIT for original code to which Leo Zhang holds the necessary rights. `LICENSE` and `LICENSE_SCOPE.md` apply the grant only to original code and expressly exclude dependencies, provider data, archived scientific results/figures, manuscripts and third-party materials.
- GSI Kumamoto/Noto reuse and attribution routes were verified under GSI's Public Data License 1.0/site terms. Source citation and edited-use identification are required; third-party and Survey Act-controlled products still require item-specific care.
- Copernicus DEM GLO-30 rights to reproduce, distribute, communicate, modify and combine were verified. Exact source/modified-product and no-liability notices are required.
- Cotabato Zenodo `7520726` and Dominica Zenodo `13834495` were verified as CC BY 4.0. The prior Dominica DOI in the draft documentation was wrong and has been corrected.
- Atitlán OSF `a8kp3` has no dataset licence metadata, and the Lombok ScienceBase item has no rights statement. CHELSA data-file terms and several national geology files also remain unresolved for redistribution.
- Because all raw, processed, unit-level and spatial products are excluded, no restricted provider data is present. Redistribution of the complete derived dataset, maps or predictions remains **not cleared**. Full details and required notices are in `docs/licensing.md`.

## Reproducibility verification

### Checks run in this preparation pass

- Package archived-output tests: **8 passed** in 0.10 s in the final local gate.
- Frozen-workspace focused tests: **8 passed, 9 deselected** in 13.43 s. These checked preregistration ordering, selection, output manifests, firewall flags and geological-history gates without fitting models.
- Static AST parse: all 127 included Python files passed.
- Configuration and metadata checks: all 11 JSON configurations parsed, `CITATION.cff` parsed as YAML/CFF 1.2.0, and all local Markdown link targets resolved.
- Fresh Python 3.10.9 environment: all pinned Python packages installed; `pip check` reported no broken requirements; package tests, core imports, a logistic fit, GeoPandas reprojection, in-memory Rasterio I/O and Whitebox wrapper import passed.
- XGBoost 2.1.4 imported and completed a toy fit when an OpenMP runtime was available. Its normal macOS `libomp` system prerequisite is now documented; no system package was installed during this task.
- Source comparison: the public source snapshot differs from the canonical source only in six documented email/contact-text redactions; caches were not copied.

### Scientific identities verified

- 835,846 terrain units and 8,793 positives.
- Selected development model: `LOGISTIC_L2` + `M` (60 predictors).
- Median AP: 0.0285931177553.
- Median PR lift: 2.56665260507.
- Fold lift range: 1.03162698154–4.05232380304.
- All six landscapes are development-exposed.
- Cotabato EQ1 remained primary and cumulative inventories could not select the model.
- Exactly one partial geological-history reconstruction was verified; `MEGH_TRAINED=NO`.

### Status classification

See `docs/reproducibility.md`. Major result tables are status **B (verified against archived outputs)**. Raw processing and full fitting are **C/D (code available; execution not verified because data are not distributed)**. Untouched external validation is **D (not performed)**. Nothing is claimed as fully reproduced in this pass.

## Manuscript-to-code traceability

`docs/traceability.md` maps the dataset totals, 60-feature set, four-family LOLO comparison, model selection, headline metrics, spatial bootstrap, three sensitivity families and geological-history feasibility claim to scripts, configurations, inputs, outputs, experiment IDs and figures/tables. All 30 included result artifacts (24 tables and six figures) were SHA-256 compared with the authoritative V2 workspace; every file had at least one byte-identical authoritative source.

No headline result was found to be untraceable. The principal traceability limitation is that the governed source datasets are not distributable in this package, so a third party cannot yet rerun the entire pipeline from raw inputs.

## Unresolved issues

1. Student and relevant fair officials confirm that the AI-assisted work and independently prepared submission materials comply with school, AUSSEF and ISEF rules. The disclosure is accurate, but ISEF prohibits AI-written research plans, abstracts, posters and citations.
2. Exact provenance of the substantive 25 October edit to the root predecessor manuscript remains unknown, although the hash mismatch itself is explained and the logged original survives unchanged in the backup.
3. Decide whether any per-unit derived dataset, susceptibility map or prediction product will ever be released. Atitlán, Lombok, CHELSA data-file terms and several national geology licences must be resolved before such a release; none is included now.
4. Add any paper DOI only after it exists and publication is separately approved.

## Release recommendation

**PRIVATE REPOSITORY CREATION AND INITIAL PUSH ARE APPROVED; PUBLICATION IS NOT APPROVED.** Code/dependency review, provider-rights review, clean-environment testing, traceability, licence scope and placeholder cleanup are complete for the conservative code/results-only boundary. Public visibility must wait for AI/fair-policy confirmation and a separate explicit owner approval. At the time of this audit update, no visibility change was authorized.
