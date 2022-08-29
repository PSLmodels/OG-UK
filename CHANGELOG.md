# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0).

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
