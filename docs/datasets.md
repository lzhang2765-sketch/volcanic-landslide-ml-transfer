# Datasets and redistribution status

## Public-release boundary

This repository does **not** redistribute raw inventories, DEMs, geology, climate/vegetation rasters, processed terrain-unit matrices, prediction vectors, model caches or GIS outputs. Small numerical result tables and summary figures are retained because they are necessary to audit reported claims. This conservative boundary remains in place until every input and derived-product licence has been reviewed for the exact intended distribution.

## Landscape sources

| Landscape | Inventory source used by the project | Access route | Recorded/checked terms | Publication decision |
|---|---|---|---|---|
| Kumamoto–Aso | GSI 2016 mapped affected-area centres | [GSI Kumamoto earthquake information](https://www.gsi.go.jp/BOUSAI/H27-kumamoto-earthquake-index.html) | GSI Public Data License 1.0 applies unless otherwise stated: cite the source, label edited use and check third-party/Survey Act exceptions | Raw and derived spatial data excluded; attribution route verified |
| Noto Peninsula | GSI 2024 slope-collapse/deposit polygons | [GSI Noto earthquake information](https://web1.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html) | The event page expressly permits reuse including reproduction with attribution; general GSI edited-use and third-party conditions also apply | Raw and derived spatial data excluded; attribution route verified |
| Lombok–Rinjani | Ferrario (2019) post-sequence inventory | [USGS ScienceBase item](https://www.sciencebase.gov/catalog/item/5ef2514782ced62aaae3bcab) | Item metadata provides a citation but no rights statement; USGS hosting does not make non-USGS data public domain | Raw and derived spatial data excluded; rights unresolved |
| Cotabato–Davao | Ferrario et al. multi-temporal inventory, EQ1 primary | [Zenodo record 7520726](https://zenodo.org/records/7520726) | Live record verified open under CC BY 4.0 | Not redistributed here; attribution required |
| Atitlán–Stan | Machado et al. (2022) Hurricane Stan inventory | [OSF DOI 10.17605/OSF.IO/A8KP3](https://doi.org/10.17605/OSF.IO/A8KP3) | Public OSF project currently has no dataset licence metadata; the article licence does not license the files | Raw and derived spatial data excluded; rights unresolved |
| Dominica–Erika | EO4Multihazards Case Study 4, Erika subset | [Zenodo DOI 10.5281/zenodo.13834495](https://doi.org/10.5281/zenodo.13834495) | Frozen provenance and live metadata agree on `version1`, CC BY 4.0 | Not redistributed here; attribution and subset limitation required |

## Common covariates

- **Copernicus DEM GLO-30:** its free licence permits reproduction, distribution, communication, adaptation and combination but requires exact source/modified-product and no-liability notices. DOI: [10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65).
- **NASADEM:** used for Atitlán because of event chronology. NASA permits use unless an exception is stated; cite NASADEM HGT v001 and its DOI.
- **CHELSA v2.1 climate normals:** cite the exact dataset and paper. The current data-file redistribution terms were not independently established, so rasters and unit-level extracts remain excluded.
- **Hansen Global Forest Change:** provider documentation states CC BY 4.0; retain version/citation metadata and give attribution.
- **Geology:** sources differ by country. Japanese GSJ Seamless geology permits reuse with attribution under the Government of Japan terms; other national geology sources require item-specific checks.

The full findings and required attribution language are in [licensing.md](licensing.md).

## Derived-data warning

An open licence for one source does not automatically clear a multi-source derived dataset. The full terrain-unit table joins inventory labels, terrain derivatives and other national/global sources. Redistribution therefore remains **not cleared** even where individual components are open.

## Expected local layout for full execution

The preserved scripts resolve their project root from the clone location and expect governed inputs under the relative paths recorded in the protocols (for example, `outputs/stage4a/STAGE4A_MEG_DATASET/`). Do not place restricted inputs under Git. A local symlink or an ignored staging directory may be used only after checking provider terms. The original macOS and Colab paths are deliberately absent from this public package.
