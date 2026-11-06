# GitHub publication plan

## Proposed repository metadata

- **Name:** `volcanic-landslide-ml-transfer`
- **Description:** Code and reproducibility materials for a six-landscape study of geographical transfer in machine-learning landslide susceptibility modelling.
- **Visibility:** keep private until every checklist blocker is closed; then change to public manually.
- **Default branch:** `main`
- **Topics:** `landslides`, `susceptibility-mapping`, `machine-learning`, `geospatial`, `volcanology`, `geographical-transfer`, `reproducible-research`, `science-fair`, `aussef`, `isef`

## Recommended settings

- Enable issues after publication; disable Discussions initially.
- Require pull requests and at least one review for `main` after the first public release.
- Enable secret scanning and push protection if available.
- Disable GitHub Actions until workflow permissions and dependency scripts have been reviewed.
- Do not enable Git LFS as a substitute for unresolved data licensing.
- Add a release tag only after the paper/repository version and citation details are fixed.
- Do not attach raw data, model caches, prediction rasters or the manuscript to a release without separate approval.

## Licence recommendation

The completed source/dependency review found no vendored third-party code and only permissively licensed direct dependencies installed separately by `pip`. The owner approved MIT for original project code. `LICENSE_SCOPE.md` keeps dependencies, provider data, archived scientific results/figures, manuscripts and third-party materials outside that grant; see `docs/licensing.md`.
