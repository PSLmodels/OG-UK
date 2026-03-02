[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/PSLmodels/OG-UK/actions/workflows/ci.yml/badge.svg)](https://github.com/PSLmodels/OG-UK/actions/workflows/ci.yml)
[![Docs](https://github.com/PSLmodels/OG-UK/actions/workflows/deploy_docs.yml/badge.svg)](https://pslmodels.github.io/OG-UK)

# OG-UK

OG-UK is an overlapping-generations model of the UK economy that allows for dynamic general equilibrium analysis of fiscal policy. It builds on [OG-Core](https://github.com/PSLmodels/OG-Core), using a calibration specific to the UK sourced from ONS, OBR, Bank of England, and GOV.UK data. The model outputs changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and tax revenues over time.

Regularly updated documentation is available at **https://pslmodels.github.io/OG-UK**.

## Quickstart (uv — recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. If you already have Python 3.11+ and uv installed:

```bash
git clone https://github.com/PSLmodels/OG-UK.git
cd OG-UK
uv sync
uv run python examples/run_oguk.py
```

This solves the baseline and reform (1pp basic rate rise) steady states and prints real-world £bn impacts.

## Full setup from scratch

If you are starting without Python installed:

1. **Install Python 3.11+** from https://www.python.org/downloads/ (tick "Add Python to PATH" on Windows).

2. **Install uv** (Python package manager):
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

3. **Clone and install OG-UK**:
   ```bash
   git clone https://github.com/PSLmodels/OG-UK.git
   cd OG-UK
   uv sync
   ```

4. **Set the HuggingFace token** (required to download PolicyEngine-UK microdata):
   ```bash
   export HUGGING_FACE_TOKEN=hf_your_token_here   # macOS/Linux
   set HUGGING_FACE_TOKEN=hf_your_token_here       # Windows
   ```
   Obtain a token with read access to `policyengine/policyengine-uk-data` from https://huggingface.co/settings/tokens.

5. **Run the example**:
   ```bash
   uv run python examples/run_oguk.py
   ```

### Alternative: conda + pip install

If you prefer Anaconda:

```bash
conda env create -f environment.yml
conda activate oguk-dev
pip install -e .
python examples/run_oguk.py
```

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
