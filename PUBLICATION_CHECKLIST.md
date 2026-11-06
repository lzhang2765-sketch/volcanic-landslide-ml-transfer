# Publication checklist

## Scientific identity

- [x] Confirm canonical V2 workspace and separation from frozen V1.
- [x] Verify six-landscape totals: 835,846 units and 8,793 positives.
- [x] Verify selected model and exact headline values against frozen outputs.
- [x] Preserve AP versus PR-lift definitions.
- [x] State that scores rank susceptibility, not hazard or risk.
- [x] State that operational negatives are not confirmed stable terrain.
- [x] State that all six landscapes are development-exposed.
- [x] State that no seventh-landscape validation was completed.
- [x] State that geological-history work was feasibility-only and no completed MEGH predictive model exists.
- [x] Investigate and document the changed predecessor-manuscript hash: logged original preserved in backup; root file substantively revised on 25 October.
- [ ] If available, identify and preserve the exact authoring/process provenance of the untracked 25 October predecessor revision; this does not change final-manuscript authority.

## Code and reproducibility

- [x] Copy canonical Python implementation without caches.
- [x] Include frozen primary configurations and protocols.
- [x] Include small archived outputs supporting headline claims.
- [x] Add manuscript-to-code traceability.
- [x] Run package archived-output tests (8 passed).
- [x] Run focused frozen-workspace tests without fitting (8 passed, 9 deselected).
- [x] Parse all included Python files and import core modules in the existing recorded environment.
- [x] Create a fresh Python 3.10.9 environment and install `requirements.txt` from scratch.
- [x] Run `pip check`, package tests, core imports and in-memory numerical/geospatial smoke tests in the fresh environment.
- [x] Document the macOS `libomp` system prerequisite and verify an XGBoost toy fit with that runtime available.
- [ ] Decide whether to add CI; if added, review workflow permissions and use only public/test data.
- [ ] If full reproduction is promised later, provide legally redistributable inputs or a verified acquisition/staging workflow and rerun end to end.

## Security and privacy

- [x] Scan for credentials, tokens, keys and passwords.
- [x] Remove local absolute/cloud paths.
- [x] Remove direct email addresses from historical request templates.
- [x] Exclude student contact information.
- [x] Exclude caches, logs, private documents and large binaries.
- [x] Exclude raw/processed scientific data, predictions and GIS rasters.
- [ ] Repeat secret and large-file scans immediately before the first commit.
- [ ] Review the staged Git diff before any commit.

## Licensing and data

- [x] Keep raw and derived spatial data out of the package by default.
- [x] Document provider access routes and known terms.
- [x] Keep code licence pending owner approval until approval is explicitly received.
- [x] Complete third-party-code provenance and dependency-licence review; no vendored dependency code found and direct dependencies are permissively licensed.
- [x] Verify GSI Kumamoto/Noto reuse and attribution requirements.
- [x] Verify Copernicus GLO-30 rights and required source/modified-product/no-liability notices.
- [x] Verify Cotabato Zenodo `7520726` and Dominica Zenodo `13834495` as CC BY 4.0; correct the prior Dominica DOI.
- [x] Confirm that Atitlán and Lombok item metadata do not currently establish redistribution rights; retain exclusion.
- [x] Confirm no raw, processed, unit-level, prediction or GIS provider data is included.
- [x] Owner approves MIT for original code only.
- [x] Add standard MIT `LICENSE` for 2026 Leo Zhang and a scope notice excluding dependencies, data, results/figures, manuscripts and third-party material.
- [ ] Resolve Atitlán, Lombok, CHELSA data-file and non-Japanese national-geology rights before any future data/map release.
- [ ] Decide separately whether derived terrain-unit data, scores or maps may ever be released and under what notice.

## Documentation and attribution

- [x] Add README, methodology, datasets, limitations, reproducibility and AI-assistance documentation.
- [x] Add a code/dependency/data licensing review and exact attribution requirements.
- [x] Add `CITATION.cff` without personal contact information.
- [x] Draft a Code Availability paragraph.
- [x] Confirm and use `Leo Zhang` as the public author name.
- [x] Remove fake repository-owner URL placeholders without assuming a GitHub username.
- [x] Validate `CITATION.cff`, JSON configs and local documentation links.
- [ ] Add paper DOI/citation when final.
- [x] Preserve a transparent statement that AI substantially assisted code, analyses, audits, figures and writing.
- [x] Review current ISEF AI rules and document the student-authorship/citation restrictions.
- [ ] Student and relevant fair officials confirm school/AUSSEF/ISEF compliance and independently authored submission materials.
- [ ] Confirm whether any public contact method beyond GitHub issues is desired.

## GitHub release

- [x] Confirm no standalone existing local remote for this project; the only parent remote is the unrelated Tweed reconstruction repository.
- [x] Search for a readily discoverable matching public GitHub repository; none found.
- [x] Obtain explicit owner approval to create a private remote, commit and push the audited package.
- [ ] Create `volcanic-landslide-ml-transfer` with private visibility first.
- [ ] Configure branch protection, secret scanning and push protection.
- [ ] Make the first local commit only after reviewing staged contents.
- [ ] Push only after explicit approval.
- [ ] Review the private GitHub rendering, links, citation file and file inventory.
- [ ] Obtain explicit approval to change visibility to public.
- [ ] Create a tagged release/DOI only after the public package is approved.

## Stop condition

Do not commit, push, create a remote, change visibility or publish a release until the owner explicitly approves the relevant action.
