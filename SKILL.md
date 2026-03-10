# OG-UK — codebase notes

## Purpose
OG-UK is an overlapping-generations (OLG) macroeconomic model for the UK, built on top of PSL's OG-Core, that scores tax/benefit policies for their long-run GDP, investment, consumption, tax revenue, and debt impacts.

## Key files
- `oguk/api.py` — all public functions and data classes; the only file you need for scoring
- `oguk/__init__.py` — re-exports everything public from `api.py`
- `oguk/oguk_default_parameters.json` — OG-Core calibration (S=80 ages, T=60 periods, cit_rate=0.27)
- `oguk/macro_params.py` — ONS/OBR data fetching helpers used internally
- `examples/run_oguk.py` — complete working example (SS + TPI) to crib from
- `.venv/` — use `.venv/bin/python3` to run; all deps installed here

## Core API

### Scoring a policy (steady state — fast, ~5-15 min)

```python
from datetime import datetime
from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest
from oguk import solve_steady_state, map_to_real_world

# 1. Build a reform Policy from PolicyEngine parameters
param = uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate")
reform = Policy(
    name="Basic rate 21%",
    parameter_values=[
        ParameterValue(parameter=param, value=0.21, start_date=datetime(2026, 1, 1))
    ],
)

# 2. Solve baseline and reform steady states
baseline = solve_steady_state(start_year=2026)           # no policy = baseline
reform_ss = solve_steady_state(start_year=2026, policy=reform)

# 3. Map to real-world £bn
impact = map_to_real_world(baseline, reform_ss)

# 4. Read results
print(f"GDP change: {impact.gdp_change:+.1f}bn ({impact.gdp_pct:+.3f}%)")
print(f"Tax revenue change: {impact.tax_revenue_change:+.1f}bn")
print(f"Investment change: {impact.investment_change:+.1f}bn")
print(f"Interest rate: {impact.r_baseline:.2%} → {impact.r_reform:.2%}")
```

### MacroImpact fields (all £bn except r and _pct)
- `.gdp`, `.gdp_change`, `.gdp_pct`
- `.consumption`, `.consumption_change`, `.consumption_pct`
- `.investment`, `.investment_change`, `.investment_pct`
- `.government`, `.government_change`, `.government_pct`
- `.tax_revenue`, `.tax_revenue_change`, `.tax_revenue_pct`
- `.debt`, `.debt_change`, `.debt_pct`
- `.r_baseline`, `.r_reform` — steady-state interest rates (model units)

### Full transition path (slow, ~60-90 min)

```python
from dask.distributed import Client
from oguk import run_transition_path, map_transition_to_real_world

client = Client(n_workers=4, threads_per_worker=1)
base_tp, reform_tp = run_transition_path(start_year=2026, policy=reform, client=client)
client.close()

impact = map_transition_to_real_world(base_tp, reform_tp)
# impact.years — array of fiscal year labels e.g. "2026-27"
# impact.gdp_change[t], impact.tax_revenue_change[t], etc. — £bn time series
```

## Finding PolicyEngine parameter paths

```python
from policyengine.tax_benefit_models.uk import uk_latest

# Look up a parameter by path
p = uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate")
print(p.label)  # "Basic rate"

# Common paths:
# gov.hmrc.income_tax.rates.uk[0].rate          — basic rate (20%)
# gov.hmrc.income_tax.rates.uk[1].rate          — higher rate (40%)
# gov.hmrc.income_tax.rates.uk[2].rate          — additional rate (45%)
# gov.hmrc.income_tax.allowances.personal_allowance.amount  — PA (£12,570)
# gov.hmrc.national_insurance.class_1.rates.employee.main   — NI main rate
# gov.hmrc.national_insurance.class_1.rates.employee.higher — NI upper rate
```

**Note:** Corporation tax (cit_rate) is NOT a PolicyEngine parameter — it is set
directly as an OG-Core parameter (`cit_rate` in `oguk_default_parameters.json`).
Use `param_overrides={"cit_rate": [[0.26]]}` to shock CT rate.

### Structural parameter shocks (param_overrides)

Both `solve_steady_state()` and `run_transition_path()` accept `param_overrides`,
a dict of OG-Core parameter names → values applied after all other configuration.
Use this for shocks to structural parameters that aren't PolicyEngine tax/benefit
parameters (e.g. TFP, corporation tax rate, government spending share).

```python
# TFP level shock (+0.4% cumulative productivity gain)
base_tp, reform_tp = run_transition_path(
    start_year=2026,
    client=client,
    param_overrides={"Z": [[1.004]]},
)
impact = map_transition_to_real_world(base_tp, reform_tp)

# Corporation tax cut
ss = solve_steady_state(param_overrides={"cit_rate": [[0.25]]})
```

**Important:** `g_y_annual` is a normalisation parameter that defines the balanced
growth path the model detrends by. It is NOT a productivity lever. To model
productivity shocks, use `Z` (TFP level in the CES production function).
Hat variables (Ŷ, K̂, etc.) are already stationary — dividing by `(1+g_y)^t` is
wrong and manufactures fake differences.

## Multiple simultaneous reforms

```python
reform = Policy(
    name="Income tax package",
    parameter_values=[
        ParameterValue(parameter=uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate"),
                       value=0.21, start_date=datetime(2026, 1, 1)),
        ParameterValue(parameter=uk_latest.get_parameter("gov.hmrc.income_tax.allowances.personal_allowance.amount"),
                       value=13_000, start_date=datetime(2026, 1, 1)),
    ],
)
```

## age_specific parameter
Controls how tax functions are estimated from microdata:
- `"pooled"` (default) — one function for all ages; fastest; use for first pass
- `"brackets"` — separate function per age group (4 groups); more accurate
- `"each"` — one per individual age (80 functions); slowest

## Running scripts
```bash
cd ~/og/og-uk
.venv/bin/python3 examples/run_oguk.py ss        # steady state
.venv/bin/python3 examples/run_oguk.py tpi       # full transition path
.venv/bin/python3 examples/run_oguk.py ss brackets  # bracket age functions
```

## Patterns and conventions
- Always use `.venv/bin/python3`; system Python doesn't have deps
- `solve_steady_state()` caches nothing — each call re-solves from scratch
- The model anchors to ONS/OBR data via live API calls (with hardcoded fallbacks if offline)
- `T=60` periods for TPI (reduced from 160 to cut runtime); `S=80` age cohorts
- `cit_rate=0.27` is the OG-UK effective CT rate (UK statutory 25% + adjustments)
- `adjustment_factor_for_cit_receipts=1.1785` calibrates CT/GDP to ~3.3% to match OBR
- Fiscal year labels are formatted as `"2026-27"` strings in TPI output
- Suppress output: all logging suppressed by default in `api.py`; print statements in scripts are fine

## Common tasks

### Score a single policy reform
Use `solve_steady_state()` + `map_to_real_world()` — see example above.

### Compare multiple reforms
Call `solve_steady_state(policy=...)` once per reform against a shared baseline.

### Get transition dynamics (not just long-run)
Use `run_transition_path()` + `map_transition_to_real_world()`.

### Diagnose a non-converging solve
- Increase `max_iter` in `solve_steady_state()` (default 250)
- Try `age_specific="pooled"` (most stable)
- Check if the reform is very large (OLG models can diverge on extreme reforms)

### Run in CI / non-interactively
Set `HUGGING_FACE_TOKEN` env var — required for PolicyEngine microdata download.
