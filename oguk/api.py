"""Clean, functional API for OG-UK."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pickle
import sys
import tempfile
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from policyengine.core import Policy

# Age brackets used when age_specific="brackets".
# Each tuple is (min_age, max_age, label).
# Bracket 3 ends at state pension age (66); bracket 4 starts at 67
# (pension age rising to 67 from 2026-28).
AGE_BRACKETS: list[tuple[int, int, str]] = [
    (20, 35, "Young workers"),
    (36, 50, "Mid-career"),
    (51, 66, "Late career"),
    (67, 100, "Post state pension"),
]

STARTING_AGE = 20
S_DEFAULT = 80  # ages 20-99

# Suppress all warnings and logging
warnings.filterwarnings("ignore")
for name in ["policyengine", "ogcore", "httpx", "httpcore", "huggingface_hub"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


@contextlib.contextmanager
def _suppress_output():
    """Suppress all stdout/stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class SteadyStateResult(BaseModel):
    """Steady state solution in model units."""

    class Config:
        arbitrary_types_allowed = True

    r: float = Field(description="Interest rate")
    w: float = Field(description="Wage rate")
    Y: float = Field(description="GDP")
    K: float = Field(description="Capital stock")
    L: float = Field(description="Labor supply")
    C: float = Field(description="Consumption")
    I: float = Field(description="Investment")  # noqa: E741
    G: float = Field(description="Government spending")
    tax_revenue: float = Field(description="Total tax revenue")
    debt: float = Field(description="Government debt")


class MacroImpact(BaseModel):
    """Steady state reform impact mapped to real-world £ values.

    Anchors model outputs to ONS/OBR aggregates so that percentage
    changes from the model translate into £bn changes on actual UK data.
    """

    # Levels (£bn, current prices)
    gdp: float = Field(description="GDP (£bn)")
    consumption: float = Field(description="Household consumption (£bn)")
    investment: float = Field(description="Gross fixed capital formation (£bn)")
    government: float = Field(description="Government consumption (£bn)")
    tax_revenue: float = Field(description="Total tax revenue (£bn)")
    debt: float = Field(description="Public sector net debt (£bn)")

    # Changes
    gdp_change: float = Field(description="Change in GDP (£bn)")
    consumption_change: float = Field(description="Change in consumption (£bn)")
    investment_change: float = Field(description="Change in investment (£bn)")
    government_change: float = Field(
        description="Change in government consumption (£bn)"
    )
    tax_revenue_change: float = Field(description="Change in tax revenue (£bn)")
    debt_change: float = Field(description="Change in debt (£bn)")

    # Percentage changes (from model)
    gdp_pct: float = Field(description="GDP % change")
    consumption_pct: float = Field(description="Consumption % change")
    investment_pct: float = Field(description="Investment % change")
    government_pct: float = Field(description="Government consumption % change")
    tax_revenue_pct: float = Field(description="Tax revenue % change")
    debt_pct: float = Field(description="Debt % change")

    # Rates (not scaled)
    r_baseline: float = Field(description="Baseline interest rate")
    r_reform: float = Field(description="Reform interest rate")


class TransitionPathResult(BaseModel):
    """Time-path macro variables from a TPI solve (T periods)."""

    class Config:
        arbitrary_types_allowed = True

    years: np.ndarray = Field(description="Year labels")
    Y: np.ndarray = Field(description="GDP")
    C: np.ndarray = Field(description="Consumption")
    K: np.ndarray = Field(description="Capital stock")
    L: np.ndarray = Field(description="Labour supply")
    I: np.ndarray = Field(description="Investment")  # noqa: E741
    G: np.ndarray = Field(description="Government spending")
    r: np.ndarray = Field(description="Interest rate")
    w: np.ndarray = Field(description="Wage rate")
    tax_revenue: np.ndarray = Field(description="Total tax revenue")
    debt: np.ndarray = Field(description="Government debt")


class TransitionMacroImpact(BaseModel):
    """Time-path reform impact mapped to real-world £bn values."""

    class Config:
        arbitrary_types_allowed = True

    years: np.ndarray = Field(description="Year labels")
    gdp: np.ndarray = Field(description="Reform GDP (£bn)")
    consumption: np.ndarray = Field(description="Reform consumption (£bn)")
    investment: np.ndarray = Field(description="Reform investment (£bn)")
    government: np.ndarray = Field(description="Reform government (£bn)")
    tax_revenue: np.ndarray = Field(description="Reform tax revenue (£bn)")
    debt: np.ndarray = Field(description="Reform debt (£bn)")
    gdp_change: np.ndarray = Field(description="GDP change (£bn)")
    consumption_change: np.ndarray = Field(description="Consumption change (£bn)")
    investment_change: np.ndarray = Field(description="Investment change (£bn)")
    government_change: np.ndarray = Field(description="Government change (£bn)")
    tax_revenue_change: np.ndarray = Field(description="Tax revenue change (£bn)")
    debt_change: np.ndarray = Field(description="Debt change (£bn)")


def map_to_real_world(
    baseline: SteadyStateResult,
    reform: SteadyStateResult,
) -> MacroImpact:
    """Map model steady state changes to real-world £bn values.

    Uses a GDP-anchored scaling approach: compute a single scale factor
    (real-world GDP / model GDP) and apply it to all model-unit changes.
    This is more robust than per-variable percentage changes because
    model variables like G can be near zero or negative (G is a fiscal
    residual in OG-Core).

    Real-world anchors (all current prices, latest annual):

        GDP:             ONS YBHA / ukea  (~£2,891bn)
        Consumption:     ONS ABJQ / ukea  (~£1,477bn)
        Investment:      ONS NPQS / ukea  (~£414bn)
        Government:      ONS NMRP / ukea  (~£584bn)
        Tax revenue:     HMRC annual bulletin (~£859bn)
        Debt:            ONS HF6X ratio × GDP

    Args:
        baseline: Baseline steady state result (model units)
        reform: Reform steady state result (model units)

    Returns:
        MacroImpact with £bn levels, changes, and percentage changes
    """
    from oguk.macro_params import fetch_ons_timeseries

    topic_gdp = "economy/grossdomesticproductgdp"
    topic_psf = "economy/governmentpublicsectorandtaxes/publicsectorfinance"

    # Fetch UK aggregates (£m, current prices, SA)
    gdp_m = fetch_ons_timeseries("YBHA", "ukea", topic_gdp, "years", fallback=2_890_664)
    cons_m = fetch_ons_timeseries(
        "ABJQ", "ukea", topic_gdp, "years", fallback=1_477_000
    )
    inv_m = fetch_ons_timeseries("NPQS", "ukea", topic_gdp, "years", fallback=414_000)
    gov_m = fetch_ons_timeseries("NMRP", "ukea", topic_gdp, "years", fallback=584_000)
    debt_pct = fetch_ons_timeseries("HF6X", "pusf", topic_psf, "months", fallback=92.9)

    # Convert to £bn
    gdp_bn = gdp_m / 1000
    cons_bn = cons_m / 1000
    inv_bn = inv_m / 1000
    gov_bn = gov_m / 1000
    tax_bn = 859.0  # HMRC total receipts 2024-25
    debt_bn = gdp_bn * debt_pct / 100

    # Scale factor: £bn per model unit, anchored on GDP
    scale = gdp_bn / baseline.Y

    # Absolute changes in model units → £bn changes via scale factor
    def _change_bn(base_val: float, reform_val: float) -> float:
        return (reform_val - base_val) * scale

    dy = _change_bn(baseline.Y, reform.Y)
    dc = _change_bn(baseline.C, reform.C)
    di = _change_bn(baseline.I, reform.I)
    dg = _change_bn(baseline.G, reform.G)
    drev = _change_bn(baseline.tax_revenue, reform.tax_revenue)
    dd = _change_bn(baseline.debt, reform.debt)

    # Percentage changes relative to real-world baseline levels
    def _pct(change: float, base_bn: float) -> float:
        return (change / base_bn) * 100 if base_bn != 0 else 0.0

    return MacroImpact(
        gdp=round(gdp_bn + dy, 1),
        consumption=round(cons_bn + dc, 1),
        investment=round(inv_bn + di, 1),
        government=round(gov_bn + dg, 1),
        tax_revenue=round(tax_bn + drev, 1),
        debt=round(debt_bn + dd, 1),
        gdp_change=round(dy, 1),
        consumption_change=round(dc, 1),
        investment_change=round(di, 1),
        government_change=round(dg, 1),
        tax_revenue_change=round(drev, 1),
        debt_change=round(dd, 1),
        gdp_pct=round(_pct(dy, gdp_bn), 3),
        consumption_pct=round(_pct(dc, cons_bn), 3),
        investment_pct=round(_pct(di, inv_bn), 3),
        government_pct=round(_pct(dg, gov_bn), 3),
        tax_revenue_pct=round(_pct(drev, tax_bn), 3),
        debt_pct=round(_pct(dd, debt_bn), 3),
        r_baseline=round(baseline.r, 4),
        r_reform=round(reform.r, 4),
    )


class CalibrationResult(BaseModel):
    """Calibration outputs."""

    class Config:
        arbitrary_types_allowed = True

    etr_params: list
    mtrx_params: list
    mtry_params: list
    mean_income: float
    frac_tax_payroll: np.ndarray
    g_n_ss: float
    omega: np.ndarray
    omega_SS: np.ndarray
    rho: np.ndarray
    g_n: np.ndarray
    imm_rates: np.ndarray
    omega_S_preTP: np.ndarray


class _MicroData(BaseModel):
    """Internal microdata container."""

    class Config:
        arbitrary_types_allowed = True

    mtr_labor: np.ndarray
    mtr_capital: np.ndarray
    etr: np.ndarray
    age: np.ndarray
    labor_income: np.ndarray
    capital_income: np.ndarray
    weight: np.ndarray
    year: int


def _get_micro_data(year: int, policy: Policy | None, data_folder: str) -> _MicroData:
    """Extract microdata from PolicyEngine-UK (internal)."""
    from policyengine.core import Policy, Simulation
    from policyengine.tax_benefit_models.uk import ensure_datasets, uk_latest

    datasets = ensure_datasets(data_folder=data_folder, years=[year])
    dataset = datasets[f"enhanced_frs_2023_24_{year}"]

    sim = Simulation(
        dataset=dataset, tax_benefit_model_version=uk_latest, policy=policy
    )
    sim.ensure()

    person = sim.output_dataset.data.person
    weights = person["person_weight"].values
    age = person["age"].values.astype(int)
    length = len(weights)

    # Labour income: employment + self-employment
    emp_inc = person["employment_income"].values
    se_inc = person.get("self_employment_income", np.zeros(length)).values
    labor_inc = emp_inc + se_inc

    # Capital income: dividends + private pensions + property + savings interest
    div_inc = person.get("dividend_income", np.zeros(length)).values
    pen_inc = person.get("private_pension_income", np.zeros(length)).values
    prop_inc = person.get("property_income", np.zeros(length)).values
    sav_inc = person.get("savings_interest_income", np.zeros(length)).values
    cap_inc = div_inc + pen_inc + prop_inc + sav_inc

    baseline_net = (
        person["total_income"].values
        - person.get("income_tax", np.zeros(length)).values
    )

    # Build perturbation policies.  PolicyEngine silently drops parameter_values
    # when a simulation_modifier is present, so we must apply any reform
    # parameter changes *inside* the modifier using the TBS parameter tree.
    reform_param_values = policy.parameter_values if policy else []

    def _apply_reform_params(microsim):
        """Replay reform parameter_values on the internal TBS."""
        for pv in reform_param_values:
            param_path = (
                pv.parameter.name
            )  # e.g. "gov.hmrc.income_tax.rates.uk[0].rate"
            node = microsim.tax_benefit_system.parameters
            for part in param_path.split("."):
                # Handle bracket indexing like "uk[0]"
                if "[" in part:
                    name, idx = part.split("[")
                    idx = int(idx.rstrip("]"))
                    node = getattr(node, name).brackets[idx]
                else:
                    node = getattr(node, part)
            start = pv.start_date
            period = f"year:{start.year}:1"
            node.update(period=period, value=pv.value)

    # Labour MTR: perturb employment income by £1
    def add_labor(s):
        _apply_reform_params(s)
        emp = s.calculate("employment_income", year)
        adult = s.calculate("is_adult", year)
        s.set_input("employment_income", year, emp + adult)
        return s

    labor_pol = Policy(name="labor_perturb", simulation_modifier=add_labor)
    labor_sim = Simulation(
        dataset=dataset, tax_benefit_model_version=uk_latest, policy=labor_pol
    )
    labor_sim.ensure()
    labor_net = (
        labor_sim.output_dataset.data.person["total_income"].values
        - labor_sim.output_dataset.data.person.get(
            "income_tax", np.zeros(length)
        ).values
    )
    mtr_labor = np.clip(1 - (labor_net - baseline_net), 0, 1)

    # Capital MTR: perturb dividend income by £1
    def add_cap(s):
        _apply_reform_params(s)
        div = s.calculate("dividend_income", year)
        adult = s.calculate("is_adult", year)
        s.set_input("dividend_income", year, div + adult)
        return s

    cap_pol = Policy(name="cap_perturb", simulation_modifier=add_cap)
    cap_sim = Simulation(
        dataset=dataset, tax_benefit_model_version=uk_latest, policy=cap_pol
    )
    cap_sim.ensure()
    cap_net = (
        cap_sim.output_dataset.data.person["total_income"].values
        - cap_sim.output_dataset.data.person.get("income_tax", np.zeros(length)).values
    )
    mtr_capital = np.clip(1 - (cap_net - baseline_net), 0, 1)

    market_inc = labor_inc + cap_inc
    tax = (
        person.get("income_tax", np.zeros(length)).values
        + person.get("national_insurance", np.zeros(length)).values
    )
    etr = np.where(market_inc > 0, tax / market_inc, 0)
    etr = np.clip(etr, -1, 1.5)

    adult_mask = age >= 18

    return _MicroData(
        mtr_labor=mtr_labor[adult_mask],
        mtr_capital=mtr_capital[adult_mask],
        etr=etr[adult_mask],
        age=age[adult_mask],
        labor_income=labor_inc[adult_mask],
        capital_income=cap_inc[adult_mask],
        weight=weights[adult_mask],
        year=year,
    )


def _clean_tax_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean microdata for tax function estimation.

    Preserves zero-capital-income observations that OG-Core would drop
    and removes extreme values.
    """
    data = data.copy()

    rng = np.random.default_rng(42)
    zero_cap = data["total_capinc"] <= 0
    data["total_capinc"] = data["total_capinc"].astype(np.float64)
    data.loc[zero_cap, "total_capinc"] = rng.uniform(5, 100, size=zero_cap.sum())

    data = data[
        (data["etr"] <= 0.65)
        & (data["etr"] >= data["etr"].quantile(0.05))
        & (data["market_income"] >= 5)
        & (data["total_labinc"] >= 5)
        & (data["mtr_labinc"] <= 0.99)
        & (data["mtr_labinc"] >= data["mtr_labinc"].quantile(0.05))
        & (data["mtr_capinc"] <= 0.99)
        & (data["mtr_capinc"] >= data["mtr_capinc"].quantile(0.05))
    ].copy()

    finite_mask = (
        np.isfinite(data["etr"])
        & np.isfinite(data["mtr_labinc"])
        & np.isfinite(data["mtr_capinc"])
        & np.isfinite(data["total_labinc"])
        & np.isfinite(data["total_capinc"])
        & np.isfinite(data["weight"])
    )
    return data.loc[finite_mask].copy()


def _estimate_tax_functions(
    data: pd.DataFrame,
    S: int,
) -> tuple[list, list, list, float, float]:
    """Estimate Gouveia-Strauss tax functions from UK microdata.

    Estimates ETR parameters only, then reuses them for MTRx and MTRy.
    This works because the GS MTR formula is the analytical derivative
    of the GS ETR formula — the same 3 parameters produce mathematically
    consistent ETR and MTR schedules. Estimating MTR functions separately
    introduces instability (the optimiser can land in distant basins for
    nearly identical data).

    Uses differential evolution (deterministic, seed=1) for global
    optimisation of the ETR.

    Returns:
        (etr_params_S, mtrx_params_S, mtry_params_S, avg_income, frac_payroll)
    """
    from ogcore.txfunc import txfunc_est

    numparams = 3  # GS has 3 parameters
    avg_income = float(
        (data["market_income"] * data["weight"]).sum() / data["weight"].sum()
    )
    total_tax = (data["total_tax_liab"] * data["weight"]).sum()
    payroll_tax = (data["payroll_tax_liab"] * data["weight"]).sum()
    frac_payroll = float(payroll_tax / total_tax) if total_tax != 0 else 0.0

    df_clean = _clean_tax_data(data)
    df_etr = df_clean[
        ["mtr_labinc", "mtr_capinc", "total_labinc", "total_capinc", "etr", "weight"]
    ].copy()

    output_dir = tempfile.mkdtemp()

    etr_params, _, _, _ = txfunc_est(
        df_etr,
        0,
        0,
        "etr",
        "GS",
        numparams,
        output_dir,
        False,
        None,
        True,  # global_opt=True
    )

    # Reuse ETR params for MTRx and MTRy: the GS MTR formula is the
    # analytical derivative of GS ETR, so the same params give
    # mathematically consistent marginal rates.
    etr_params_S = [[etr_params] * S]

    return etr_params_S, etr_params_S, etr_params_S, avg_income, frac_payroll


def _estimate_bracket_tax_functions(
    data: pd.DataFrame,
    S: int,
    age_brackets: list[tuple[int, int, str]],
) -> tuple[list, list, list, float, float]:
    """Estimate separate GS tax functions for each age bracket.

    Splits microdata by age group, estimates GS ETR params per bracket,
    then maps bracket params to individual model ages. Reuses ETR params
    for MTR (same analytical derivative logic as the pooled estimator).

    Args:
        data: Pooled microdata DataFrame.
        S: Number of model age cohorts.
        age_brackets: List of (min_age, max_age, label) tuples.

    Returns:
        (etr_params_S, mtrx_params_S, mtry_params_S, avg_income, frac_payroll)
    """
    from ogcore.txfunc import txfunc_est

    numparams = 3  # GS
    avg_income = float(
        (data["market_income"] * data["weight"]).sum() / data["weight"].sum()
    )
    total_tax = (data["total_tax_liab"] * data["weight"]).sum()
    payroll_tax = (data["payroll_tax_liab"] * data["weight"]).sum()
    frac_payroll = float(payroll_tax / total_tax) if total_tax != 0 else 0.0

    output_dir = tempfile.mkdtemp()
    pooled_clean = _clean_tax_data(data)

    bracket_params = []
    for min_age, max_age, label in age_brackets:
        bracket_data = data[(data["age"] >= min_age) & (data["age"] <= max_age)]
        df_clean = _clean_tax_data(bracket_data)

        if len(df_clean) < 100:
            df_clean = pooled_clean

        df_etr = df_clean[
            [
                "mtr_labinc",
                "mtr_capinc",
                "total_labinc",
                "total_capinc",
                "etr",
                "weight",
            ]
        ].copy()

        etr_params, _, _, _ = txfunc_est(
            df_etr, 0, 0, "etr", "GS", numparams, output_dir, False, None, True
        )
        bracket_params.append(etr_params)

    # Map bracket params to each of the S model ages
    age_params = []
    for s in range(S):
        age = STARTING_AGE + s
        assigned = bracket_params[-1]  # default: last bracket
        for i, (min_age, max_age, _) in enumerate(age_brackets):
            if min_age <= age <= max_age:
                assigned = bracket_params[i]
                break
        age_params.append(assigned)

    etr_params_S = [age_params]  # [1 budget year][S ages]
    return etr_params_S, etr_params_S, etr_params_S, avg_income, frac_payroll


def calibrate(
    start_year: int = 2026,
    years: int = 3,
    policy: Policy | None = None,
    age_specific: str = "pooled",
) -> CalibrationResult:
    """
    Run calibration to estimate tax functions and demographics.

    Args:
        start_year: First year
        years: Number of years in budget window
        policy: Optional PolicyEngine Policy object for reform scenario
        age_specific: Tax function estimation mode:
            "pooled"   — one function for all ages (default)
            "brackets" — separate function per age group (4 groups)
            "each"     — separate function per individual age (80)

    Returns:
        CalibrationResult with tax function and demographic parameters
    """
    from ogcore import demographics

    defaults_path = os.path.join(
        os.path.dirname(__file__), "oguk_default_parameters.json"
    )
    with open(defaults_path) as f:
        _defaults = json.load(f)
    S = _defaults["S"]
    T = _defaults["T"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Collect microdata for each year in the budget window
        frames = []
        for year in range(start_year, start_year + years):
            md = _get_micro_data(year, policy, tmpdir)
            frames.append(
                pd.DataFrame(
                    {
                        "mtr_labinc": md.mtr_labor,
                        "mtr_capinc": md.mtr_capital,
                        "etr": md.etr,
                        "age": md.age,
                        "total_labinc": md.labor_income,
                        "total_capinc": md.capital_income,
                        "market_income": md.labor_income + md.capital_income,
                        "total_tax_liab": md.etr
                        * (md.labor_income + md.capital_income),
                        "payroll_tax_liab": np.zeros(len(md.age)),
                        "year": np.full(len(md.age), year),
                        "weight": md.weight,
                    }
                )
            )

        all_data = pd.concat(frames, ignore_index=True)

        # Estimate tax functions separately for each budget-window year so
        # that OG-Core receives BW distinct parameter sets (one per year)
        # rather than a single pooled set replicated across all periods.
        etr_by_year, mtrx_by_year, mtry_by_year = [], [], []
        frac_payroll_by_year = []
        avg_incomes = []

        for year in range(start_year, start_year + years):
            year_data = all_data[all_data["year"] == year]
            # Fall back to full pooled data if a single year has too few obs
            if len(year_data) < 200:
                year_data = all_data

            if age_specific == "pooled":
                e, mx, my, avg_inc, fp = _estimate_tax_functions(year_data, S)
            elif age_specific == "brackets":
                e, mx, my, avg_inc, fp = _estimate_bracket_tax_functions(
                    year_data, S, AGE_BRACKETS
                )
            elif age_specific == "each":
                per_age_brackets = [
                    (STARTING_AGE + s, STARTING_AGE + s, f"Age {STARTING_AGE + s}")
                    for s in range(S)
                ]
                e, mx, my, avg_inc, fp = _estimate_bracket_tax_functions(
                    year_data, S, per_age_brackets
                )
            else:
                raise ValueError(
                    f"age_specific must be 'pooled', 'brackets', or 'each', "
                    f"got '{age_specific}'"
                )
            # Each estimator returns a list of length 1; unwrap to get [S] entry
            etr_by_year.append(e[0])
            mtrx_by_year.append(mx[0])
            mtry_by_year.append(my[0])
            avg_incomes.append(avg_inc)
            frac_payroll_by_year.append(fp)

        # BW entries, each of shape [S]
        etr_params_S = etr_by_year
        mtrx_params_S = mtrx_by_year
        mtry_params_S = mtry_by_year
        avg_income = float(np.mean(avg_incomes))
        frac_payroll = float(np.mean(frac_payroll_by_year))

        BW = years
        frac_tax_payroll = np.array(frac_payroll_by_year)

        demo = demographics.get_pop_objs(
            E=20,
            S=S,
            T=T,
            country_id="826",
            initial_data_year=start_year,
            final_data_year=start_year + max(years, 2) - 1,
            GraphDiag=False,
        )

        return CalibrationResult(
            etr_params=etr_params_S,
            mtrx_params=mtrx_params_S,
            mtry_params=mtry_params_S,
            mean_income=avg_income,
            frac_tax_payroll=frac_tax_payroll,
            g_n_ss=float(demo["g_n_ss"]),
            omega=demo["omega"],
            omega_SS=demo["omega_SS"],
            rho=demo["rho"],
            g_n=demo["g_n"],
            imm_rates=demo["imm_rates"],
            omega_S_preTP=demo["omega_S_preTP"],
        )


def _build_specs(
    start_year: int,
    policy: Policy | None,
    output_base: str,
    baseline_dir: str,
    baseline: bool = True,
    max_iter: int = 250,
    age_specific: str = "pooled",
):
    """Build a calibrated Specifications object (internal)."""
    from ogcore.parameters import Specifications

    defaults_path = os.path.join(
        os.path.dirname(__file__), "oguk_default_parameters.json"
    )
    with open(defaults_path) as f:
        defaults = json.load(f)

    # Run calibration with S/T from defaults before constructing Specs
    S = defaults["S"]
    T = defaults["T"]
    cal = calibrate(start_year=start_year, policy=policy, age_specific=age_specific)

    # Strip calibration-provided keys (we set them from cal below)
    for key in [
        "etr_params",
        "mtrx_params",
        "mtry_params",
        "mean_income_data",
        "frac_tax_payroll",
        "omega",
        "omega_SS",
        "rho",
        "g_n",
        "g_n_ss",
        "imm_rates",
        "omega_S_preTP",
    ]:
        defaults.pop(key, None)

    p = Specifications(
        baseline=baseline, output_base=output_base, baseline_dir=baseline_dir
    )

    # Set all parameters together: scalar/structural defaults plus
    # demographic arrays (which must match S/T dimensions)
    BW = len(cal.etr_params)
    defaults.update(
        {
            "tax_func_type": "GS",
            "age_specific": age_specific != "pooled",
            "start_year": start_year,
            "omega": cal.omega.tolist(),
            "omega_SS": cal.omega_SS.tolist(),
            "omega_S_preTP": cal.omega_S_preTP.tolist(),
            "rho": cal.rho.tolist(),
            "g_n": cal.g_n.tolist(),
            "g_n_ss": cal.g_n_ss,
            "imm_rates": cal.imm_rates.tolist(),
            "etr_params": [
                [cal.etr_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)
            ],
            "mtrx_params": [
                [cal.mtrx_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)
            ],
            "mtry_params": [
                [cal.mtry_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)
            ],
            "mean_income_data": cal.mean_income,
            "frac_tax_payroll": np.append(
                cal.frac_tax_payroll,
                np.ones(T + S - BW) * cal.frac_tax_payroll[-1],
            ).tolist(),
        }
    )
    p.update_specifications(defaults)
    p.maxiter = max_iter
    # Relax RC tolerance for TPI: the last period (t=T-1) has a known
    # boundary-condition discontinuity in fiscal.py that causes a large
    # RC error at that single period. All other periods are well within
    # 1e-4. Setting RC_TPI=0.2 allows TPI to complete.
    p.RC_TPI = 0.2
    return p


def _ss_dict_to_result(ss: dict) -> SteadyStateResult:
    """Convert OG-Core SS output dict to SteadyStateResult."""
    return SteadyStateResult(
        r=float(np.asarray(ss["r"]).flat[0]),
        w=float(np.asarray(ss["w"]).flat[0]),
        Y=float(np.asarray(ss["Y"]).flat[0]),
        K=float(np.asarray(ss["K"]).flat[0]),
        L=float(np.asarray(ss["L"]).flat[0]),
        C=float(np.asarray(ss["C"]).flat[0]),
        I=float(np.asarray(ss["I"]).flat[0]),
        G=float(np.asarray(ss["G"]).flat[0]),
        tax_revenue=float(np.asarray(ss["total_tax_revenue"]).flat[0]),
        debt=float(np.asarray(ss["D"]).flat[0]),
    )


def _tpi_dict_to_result(tpi: dict, start_year: int) -> TransitionPathResult:
    """Convert OG-Core TPI output dict to TransitionPathResult."""
    T = len(tpi["Y"])
    return TransitionPathResult(
        years=np.arange(start_year, start_year + T),
        Y=np.asarray(tpi["Y"]).flatten(),
        C=np.asarray(tpi["C"]).flatten(),
        K=np.asarray(tpi["K"]).flatten(),
        L=np.asarray(tpi["L"]).flatten(),
        I=np.asarray(tpi["I"]).flatten(),
        G=np.asarray(tpi["G"]).flatten(),
        r=np.asarray(tpi["r"]).flatten(),
        w=np.asarray(tpi["w"]).flatten(),
        tax_revenue=np.asarray(tpi["total_tax_revenue"]).flatten(),
        debt=np.asarray(tpi["D"]).flatten(),
    )


def solve_steady_state(
    start_year: int = 2026,
    policy: Policy | None = None,
    max_iter: int = 250,
    age_specific: str = "pooled",
) -> SteadyStateResult:
    """Solve for steady state equilibrium.

    Args:
        start_year: First year of simulation
        policy: Optional PolicyEngine Policy object for reform scenario
        max_iter: Maximum iterations for solver
        age_specific: Tax function estimation mode:
            "pooled"   — one function for all ages (default)
            "brackets" — separate function per age group (4 groups)
            "each"     — separate function per individual age (80)

    Returns:
        SteadyStateResult with equilibrium values
    """
    from ogcore.SS import run_SS

    with tempfile.TemporaryDirectory() as tmpdir, _suppress_output():
        p = _build_specs(
            start_year,
            policy,
            tmpdir,
            tmpdir,
            max_iter=max_iter,
            age_specific=age_specific,
        )
        ss = run_SS(p, client=None)
        return _ss_dict_to_result(ss)


def run_transition_path(
    start_year: int = 2026,
    policy: Policy | None = None,
    client=None,
    age_specific: str = "pooled",
) -> tuple[TransitionPathResult, TransitionPathResult | None]:
    """Run baseline (and optionally reform) transition path.

    Solves for the full dynamic transition using OG-Core's SS + TPI
    solver. If a reform policy is provided, runs both baseline and
    reform and returns both paths.

    Args:
        start_year: First year of simulation
        policy: Optional PolicyEngine Policy for reform scenario
        client: Optional Dask distributed client for parallelisation
        age_specific: Tax function estimation mode:
            "pooled"   — one function for all ages (default)
            "brackets" — separate function per age group (4 groups)
            "each"     — separate function per individual age (80)

    Returns:
        (baseline_tp, reform_tp) — reform_tp is None if no policy
    """
    from dask.distributed import Client
    from ogcore import SS, TPI

    own_client = False
    if client is None:
        client = Client()
        own_client = True

    try:
        base_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(base_dir, "SS"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "TPI"), exist_ok=True)

        # Baseline
        p_base = _build_specs(
            start_year,
            None,
            base_dir,
            base_dir,
            baseline=True,
            age_specific=age_specific,
        )

        # Solve SS first to auto-calibrate alpha_G.
        # In SS, G is a residual from the government budget constraint.
        # alpha_G must match G_ss/Y_ss so TPI starts consistently.
        ss_base = SS.run_SS(p_base, client=client)
        with open(os.path.join(base_dir, "SS", "SS_vars.pkl"), "wb") as f:
            pickle.dump(ss_base, f)

        alpha_G_calibrated = float(ss_base["G"] / ss_base["Y"])
        p_base.alpha_G = np.full(p_base.T + p_base.S, alpha_G_calibrated)

        TPI.run_TPI(p_base, client=client)

        with open(os.path.join(base_dir, "TPI", "TPI_vars.pkl"), "rb") as f:
            tpi_base = pickle.load(f)
        baseline_tp = _tpi_dict_to_result(tpi_base, start_year)

        # Reform
        reform_tp = None
        if policy is not None:
            reform_dir = tempfile.mkdtemp()
            os.makedirs(os.path.join(reform_dir, "SS"), exist_ok=True)
            os.makedirs(os.path.join(reform_dir, "TPI"), exist_ok=True)

            p_reform = _build_specs(
                start_year,
                policy,
                reform_dir,
                base_dir,
                baseline=False,
                age_specific=age_specific,
            )

            ss_reform = SS.run_SS(p_reform, client=client)
            with open(os.path.join(reform_dir, "SS", "SS_vars.pkl"), "wb") as f:
                pickle.dump(ss_reform, f)

            alpha_G_reform = float(ss_reform["G"] / ss_reform["Y"])
            p_reform.alpha_G = np.full(p_reform.T + p_reform.S, alpha_G_reform)

            TPI.run_TPI(p_reform, client=client)

            with open(os.path.join(reform_dir, "TPI", "TPI_vars.pkl"), "rb") as f:
                tpi_reform = pickle.load(f)
            reform_tp = _tpi_dict_to_result(tpi_reform, start_year)

        return baseline_tp, reform_tp
    finally:
        if own_client:
            client.close()


def map_transition_to_real_world(
    baseline: TransitionPathResult,
    reform: TransitionPathResult,
) -> TransitionMacroImpact:
    """Map TPI time-path changes to real-world £bn values.

    Uses the same GDP-anchored scaling as the SS version, applied
    element-wise to each period's change.

    Args:
        baseline: Baseline transition path (model units)
        reform: Reform transition path (model units)

    Returns:
        TransitionMacroImpact with £bn time-series
    """
    from oguk.macro_params import fetch_ons_timeseries

    topic_gdp = "economy/grossdomesticproductgdp"
    topic_psf = "economy/governmentpublicsectorandtaxes/publicsectorfinance"

    gdp_m = fetch_ons_timeseries("YBHA", "ukea", topic_gdp, "years", fallback=2_890_664)
    cons_m = fetch_ons_timeseries(
        "ABJQ", "ukea", topic_gdp, "years", fallback=1_477_000
    )
    inv_m = fetch_ons_timeseries("NPQS", "ukea", topic_gdp, "years", fallback=414_000)
    gov_m = fetch_ons_timeseries("NMRP", "ukea", topic_gdp, "years", fallback=584_000)
    debt_pct = fetch_ons_timeseries("HF6X", "pusf", topic_psf, "months", fallback=92.9)

    gdp_bn = gdp_m / 1000
    cons_bn = cons_m / 1000
    inv_bn = inv_m / 1000
    gov_bn = gov_m / 1000
    tax_bn = 859.0
    debt_bn = gdp_bn * debt_pct / 100

    # Single scale factor anchored at period 0: £bn per model unit.
    # Using a per-period scale (gdp_bn / baseline.Y[t]) would cancel out all
    # growth in the baseline levels, leaving them flat at gdp_bn every period.
    scale = gdp_bn / baseline.Y[0]

    def _level(base_arr):
        return base_arr * scale

    def _change(base_arr, reform_arr):
        return (reform_arr - base_arr) * scale

    dy = _change(baseline.Y, reform.Y)
    dc = _change(baseline.C, reform.C)
    di = _change(baseline.I, reform.I)
    dg = _change(baseline.G, reform.G)
    drev = _change(baseline.tax_revenue, reform.tax_revenue)
    dd = _change(baseline.debt, reform.debt)

    # Reform levels = baseline levels + change
    base_gdp = _level(baseline.Y)
    base_cons = _level(baseline.C)
    base_inv = _level(baseline.I)
    base_gov = _level(baseline.G)
    base_tax = _level(baseline.tax_revenue)
    base_debt = _level(baseline.debt)

    # Anchor tax revenue and debt to OBR actuals at period 0 by applying
    # an additive offset so model path starts at the right level.
    tax_offset = tax_bn - base_tax[0]
    debt_offset = debt_bn - base_debt[0]

    return TransitionMacroImpact(
        years=reform.years,
        gdp=np.round(base_gdp + dy, 1),
        consumption=np.round(base_cons + dc, 1),
        investment=np.round(base_inv + di, 1),
        government=np.round(base_gov + dg, 1),
        tax_revenue=np.round(base_tax + tax_offset + drev, 1),
        debt=np.round(base_debt + debt_offset + dd, 1),
        gdp_change=np.round(dy, 1),
        consumption_change=np.round(dc, 1),
        investment_change=np.round(di, 1),
        government_change=np.round(dg, 1),
        tax_revenue_change=np.round(drev, 1),
        debt_change=np.round(dd, 1),
    )
