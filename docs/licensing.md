# Code, dependency and data licensing review

This review applies to the local release candidate prepared on 6 November 2026. It is a technical provenance and compatibility review, not legal advice.

## Original code and third-party code

The public source snapshot contains project modules and scripts, not copied dependency source trees. A source-header and provenance scan found no vendored third-party code or inherited third-party copyright notice. The six differences from the canonical `sfv2` tree are documented contact-text redactions; they do not change scientific logic.

The dependency packages are installed separately by `pip` and are not redistributed in this repository. Their recorded licences are compatible with an MIT licence for independently written project code. Dependency names and trademarks must not be used to imply endorsement, and their own licence files govern their distributions.

| Direct dependency | Recorded version | Licence in installed distribution | Role |
|---|---:|---|---|
| NumPy | 2.2.6 | BSD-3-Clause family | Numerical arrays |
| pandas | 2.3.3 | BSD-3-Clause | Tables |
| SciPy | 1.15.3 | BSD-3-Clause | Scientific routines |
| scikit-learn | 1.7.2 | BSD-3-Clause | Models and metrics |
| Matplotlib | 3.10.9 | PSF-based permissive licence | Figures |
| joblib | 1.6.0 | BSD-3-Clause | Parallelism/serialization |
| threadpoolctl | 3.7.0 | BSD-3-Clause | Thread control |
| XGBoost | 2.1.4 | Apache-2.0 | Boosted-tree comparison |
| GeoPandas | 1.1.4 | BSD-3-Clause | Vector data handling |
| Rasterio | 1.4.4 | BSD-3-Clause | Raster data handling |
| Pillow | 12.3.0 | MIT-CMU | Image support |
| jsonschema | 4.26.0 | MIT | Configuration validation |
| openpyxl | 3.1.5 | MIT | Spreadsheet input |
| pyogrio | 0.13.0 | MIT | Vector I/O |
| pyproj | 3.7.1 | MIT | Coordinate transforms |
| Shapely | 2.1.2 | BSD-3-Clause | Geometry operations |
| whitebox | 2.3.6 | MIT | WhiteboxTools interface |
| pytest | 9.1.1 | MIT | Release tests (development only) |

The pinned direct-import packages missing from the first draft of `requirements.txt` have now been added. Transitive compiled libraries remain governed by their own licences in the wheels installed by `pip`; no wheel or shared library is included in this repository.

### Applied licence

The owner approved MIT for original code to which Leo Zhang holds the necessary rights. `LICENSE` contains the standard MIT text and `LICENSE_SCOPE.md` defines the covered Software as original project code in `src/`, `scripts/`, `tests/` and original code-like configuration. The scope expressly excludes third-party dependencies, provider data, archived scientific results/figures, manuscripts and other third-party materials. It does not relicense source datasets, scientific outputs or external software.

## Item-level data-rights findings

No raw or processed inventory, DEM, geology, climate, vegetation, terrain-unit, prediction or GIS file is present in this release candidate. The included tables are small archived numerical results and the included PNGs are project-generated summary figures. Provider data must still be acknowledged in the paper and repository, and any future release of spatial or unit-level derivatives requires a separate decision.

| Item used in the study | Rights finding | Required action / release decision |
|---|---|---|
| GSI 2016 Kumamoto mapped landslide centres | GSI website content is generally under Public Data License 1.0 unless otherwise stated. Source citation is required; edited content must be identified as edited and must not imply GSI authorship. Third-party components and Survey Act-controlled survey results require separate checking. | Cite the GSI 2016 event page and identify any edited derivative. Raw points and spatial derivatives remain excluded from this repository. |
| GSI 2024 Noto slope-collapse/deposit polygons | The event page explicitly permits reuse, including reproduction, with source attribution and points to GSI's detailed terms. The same edited-use and third-party-rights conditions apply. | Cite the exact Noto page and identify edited products. Raw GeoJSON and spatial derivatives remain excluded. |
| Copernicus DEM GLO-30 | The GLO-30 licence grants worldwide, time-unlimited reproduction, distribution, public communication, adaptation, modification and combination free of charge. Public communication/distribution requires the prescribed source notice; modified products require the prescribed “produced using” notice and the licence also requires a no-liability sentence. | Use the exact notices below in any release that communicates or distributes GLO-30 or a modified GLO-30 product. No DEM or terrain-unit derivative is included here. |
| Lombok–Rinjani Ferrario (2018) inventory, ScienceBase item `5ef2514782ced62aaae3bcab` | The item metadata provides a citation but no rights statement. USGS hosting does not establish that this non-USGS-authored inventory is public domain. | **Unresolved.** Do not redistribute the inventory or spatial derivatives without a rights statement or permission. |
| Cotabato–Davao Zenodo `7520726` | The live record is open access under CC BY 4.0 and lists the exact event shapefile components. | Attribution is required. Raw and spatial derived files remain excluded by the conservative release boundary. |
| Atitlán–Stan OSF `a8kp3` | The project is public but its current API metadata has no licence. The associated article's licence does not automatically license the OSF files. | **Unresolved.** Do not redistribute the inventory or spatial derivatives without dataset-level permission or an explicit OSF licence. |
| Dominica–Erika EO4Multihazards | The frozen provenance register identifies Zenodo `13834495`, `version1`; the live record is open under CC BY 4.0. Earlier draft documentation incorrectly pointed to `15224819`, which now redirects to an unrelated exposure-data record. | Cite Zenodo `13834495` and its creators; retain the Erika-subset limitation. Do not use the superseded/wrong DOI. |
| NASADEM HGT v001 | NASA's dataset terms permit use for any purpose unless an exception is stated; the product should still be cited by DOI and original source. | Cite NASADEM HGT v001. The Atitlán DEM and its derivatives remain excluded. |
| Hansen Global Forest Change | Provider documentation states CC BY 4.0. | Attribute the dataset/version used. Rasters and unit-level extracts remain excluded. |
| CHELSA v2.1 climate data | Current CHELSA pages clearly identify the model/code licence, but the exact data-file licence for the archived v2.1 climatology used here was not independently established in this pass. The project's older record says CC BY 4.0, which is insufficient by itself for redistribution. | **Unresolved for redistribution.** Cite the dataset and paper; do not redistribute source rasters or unit-level extracts without the exact data-file terms. |
| GSJ Seamless Geological Map of Japan V2 | Government of Japan Standard Terms of Use 2.0; reuse and modification are permitted with source attribution and no application is required. | Use the GSJ/AIST attribution appropriate to the exact legend/version. Raw and unit-level geology remain excluded. |
| Indonesian, Philippine, Guatemalan and Dominican geology | The project records provider identities, but a complete current item/version licence chain was not established for every exact file. | **Unresolved for redistribution.** Keep source and unit-level geology excluded. |

### Required Copernicus notices

For unmodified GLO-30 communication or distribution:

> © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

For an adapted or modified product:

> Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

The applicable legal notice must also include the licence's required statement that the organisations responsible for the Copernicus programme accept no liability for use of Copernicus WorldDEM-30.

## Release conclusion

The current code/results package does not contain restricted provider data and is safe under the conservative exclusion boundary. Data rights are **not** cleared for publishing the full terrain-unit dataset, susceptibility atlas, prediction rasters, or source/processed GIS data. Those materials must remain outside Git unless the unresolved item-level permissions are closed and the exact attribution notices accompany the release.
