[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/PSLmodels/OG-UK/actions/workflows/ci.yml/badge.svg)](https://github.com/PSLmodels/OG-UK/actions/workflows/ci.yml)

# OG-UK

OG-UK is an overlapping-generations model of the UK economy that allows for dynamic general equilibrium analysis of fiscal policy. It builds on [OG-Core](https://github.com/PSLmodels/OG-Core), using a calibration specific to the UK sourced from ONS, OBR, Bank of England, and GOV.UK data. The model outputs changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and tax revenues over time.

## Quickstart

```bash
uv sync
uv run python examples/run_oguk.py
```

This solves the baseline and reform (1pp basic rate rise) steady states and prints real-world £bn impacts.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -x -m "not full_run" --tb=short
```

## CI secrets

The test suite downloads private microdata from HuggingFace. To run tests (locally or in CI), set the `HUGGING_FACE_TOKEN` environment variable to a token with read access to the `policyengine/policyengine-uk-data` repo. In GitHub Actions this must be added as a repository secret.

## Citing OG-UK

Please do not cite — the model is under active development.
