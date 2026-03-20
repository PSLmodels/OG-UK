# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0).

## [0.3.1] - 2026-03-20

### Added

* Extended calibration to an 8-sector multi-industry model: Energy, Manufacturing, Construction, Trade & Transport, Information & Finance, Real Estate, Business Services, and Public & Other.
* Added `oguk/industry_params.py` with `get_industry_params()` returning sector-specific capital shares, TFP, and production parameters calibrated from ONS data.
* Computed sector-level TFP as Solow residuals using ONS capital stocks by industry (CAPSTK) and workforce jobs by SIC (JOBS02), normalised so GVA-weighted mean equals 1.0.
* Added `run_tpi_and_export.py` to extract and export per-industry TPI results (`Y_m`, `K_m`, `L_m`, `p_m`) to `tpi_results.xlsx`.
* Added `tpi_charts_plotly.py` with sector-level transition path charts anchored to ONS historical outturn (GVA, capital stocks, workforce jobs from 2000) and projected forward using OBR nominal GDP growth.

## [0.3.0] - 2026-02-24

### Changed

* Calibrated fiscal parameters to OBR November 2025 Economic and Fiscal Outlook: steady-state debt ratio 95% of GDP, fiscal adjustment beginning in period 4 (matching 2029-30 fiscal rules).
* Recalibrated tax parameters so model revenue/GDP matches OBR forecasts (~40% of GDP). Added wealth tax parameters (council tax, stamp duty, CGT proxy), raised effective indirect tax rate, and aligned capital depreciation allowances with economic depreciation.
* Set non-pension welfare transfers (alpha_T) to 6% of GDP, separate from state pension which is modelled via OG-Core's pension system.
* Replaced five legacy CI workflows (conda, Python 3.9, black, 3-OS matrix) with a single uv-based workflow on Python 3.13 using ruff for linting and formatting.
* Modernised `oguk/sources.py` data-fetching to return calibrated tax rates consistent with OBR revenue shares.

## [0.2.0] - 2023-01-16

### Updated

* Updated all OpenFisca references to PolicyEngine. This represents a big change in the underlying microsimulation dependency structure
* Updated `get_micro_data.py` and `test_get_micro_data.py` with `PolicyEngine` references instead of `OpenFisca`.
* Updated `demographics.py` to use the UN data portal. Also updated corresponding structures in `calibrate.py`
* Added four `.csv` files to `oguk/data/demographic/`. This files allow for `demographics.py` to have the option to not download the data from the UN data portal.
* Updated `environment.yml` and `setup.py`.
* Updated `run_oguk.py` with more consistent specification, and updated references from `openfisca` to `policyengine`.
* Small updates to `.gitignore`, `README.md`, `demographics.rst`, `get_micro_data.rst`, and `tax_functions.md`.
* Deleted `pyproject.toml` which was just a reference to the black package.
* Updates the `CHANGELOG.md` and updates the version number in `setup.py`.
* Updates the GitHub Actions `deploy_docs.yml`, `docs_check.yml`, and `build_and_test.yml`.

## [0.1.2] - 2022-07-29

### Fixed

* UK corporate tax rate schedule added.

## [0.1.1] - 2022-07-19

### Changed

* Initial guesses at the steady state interest rate, `initial_guess_r_SS`, and transfer amount, `initial_guess_TR_SS` were changed so that the example script, `./examples/run_oguk.py` solves with the latest OG-Core.

## [0.1.0] - 2022-07-01

### Changed

* Firm parameters updated to be vectors and 2D arrays to be compatible with multiple industry model of `ogcore 0.9.0`.  OG-UK now required `ogcore >= 0.9.0`.
* `environment.yml` file set to install `ogcore` from PyPI

## [0.0.3] - 2022-04-26

### Changed

* Budget window for microsimulation changed to 2022-27 (from 2019-22).

## [0.0.2] - 2022-01-23

### Added

* Better handling and logging to the user of the status of microdata.

## [0.0.1] - 2022-01-20

### Fixed

* OpenFisca-UK dataset handling.

## [0.0.2] - 2022-01-24

### Updated

* Consolidate GitHub Actions.
