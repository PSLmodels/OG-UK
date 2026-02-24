"""Clean, functional API for OG-UK."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from policyengine.core import Policy

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
    """Steady state solution."""

    class Config:
        arbitrary_types_allowed = True

    r: float = Field(description="Interest rate")
    w: float = Field(description="Wage rate")
    Y: float = Field(description="GDP")
    K: float = Field(description="Capital stock")
    L: float = Field(description="Labor supply")
    C: float = Field(description="Consumption")
    I: float = Field(description="Investment")
    G: float = Field(description="Government spending")
    tax_revenue: float = Field(description="Total tax revenue")
    debt: float = Field(description="Government debt")


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

    sim = Simulation(dataset=dataset, tax_benefit_model_version=uk_latest, policy=policy)
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

    baseline_net = person["total_income"].values - person.get("income_tax", np.zeros(length)).values

    # Build perturbation policies.  PolicyEngine silently drops parameter_values
    # when a simulation_modifier is present, so we must apply any reform
    # parameter changes *inside* the modifier using the TBS parameter tree.
    reform_param_values = policy.parameter_values if policy else []

    def _apply_reform_params(microsim):
        """Replay reform parameter_values on the internal TBS."""
        for pv in reform_param_values:
            param_path = pv.parameter.name  # e.g. "gov.hmrc.income_tax.rates.uk[0].rate"
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
    labor_sim = Simulation(dataset=dataset, tax_benefit_model_version=uk_latest, policy=labor_pol)
    labor_sim.ensure()
    labor_net = labor_sim.output_dataset.data.person["total_income"].values - labor_sim.output_dataset.data.person.get("income_tax", np.zeros(length)).values
    mtr_labor = np.clip(1 - (labor_net - baseline_net), 0, 1)

    # Capital MTR: perturb dividend income by £1
    def add_cap(s):
        _apply_reform_params(s)
        div = s.calculate("dividend_income", year)
        adult = s.calculate("is_adult", year)
        s.set_input("dividend_income", year, div + adult)
        return s

    cap_pol = Policy(name="cap_perturb", simulation_modifier=add_cap)
    cap_sim = Simulation(dataset=dataset, tax_benefit_model_version=uk_latest, policy=cap_pol)
    cap_sim.ensure()
    cap_net = cap_sim.output_dataset.data.person["total_income"].values - cap_sim.output_dataset.data.person.get("income_tax", np.zeros(length)).values
    mtr_capital = np.clip(1 - (cap_net - baseline_net), 0, 1)

    market_inc = labor_inc + cap_inc
    tax = person.get("income_tax", np.zeros(length)).values + person.get("national_insurance", np.zeros(length)).values
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
    df_etr = df_clean[["mtr_labinc", "mtr_capinc", "total_labinc", "total_capinc", "etr", "weight"]].copy()

    output_dir = tempfile.mkdtemp()

    etr_params, _, _, _ = txfunc_est(
        df_etr, 0, 0, "etr", "GS", numparams, output_dir, False,
        None, True,  # global_opt=True
    )

    # Reuse ETR params for MTRx and MTRy: the GS MTR formula is the
    # analytical derivative of GS ETR, so the same params give
    # mathematically consistent marginal rates.
    etr_params_S = [[etr_params] * S]

    return etr_params_S, etr_params_S, etr_params_S, avg_income, frac_payroll


def calibrate(
    start_year: int = 2026,
    years: int = 3,
    policy: Policy | None = None,
) -> CalibrationResult:
    """
    Run calibration to estimate tax functions and demographics.

    Args:
        start_year: First year
        years: Number of years in budget window
        policy: Optional PolicyEngine Policy object for reform scenario

    Returns:
        CalibrationResult with tax function and demographic parameters
    """
    from oguk import demographics

    S = 80

    with tempfile.TemporaryDirectory() as tmpdir:
        # Collect microdata for each year in the budget window
        frames = []
        for year in range(start_year, start_year + years):
            md = _get_micro_data(year, policy, tmpdir)
            frames.append(pd.DataFrame({
                "mtr_labinc": md.mtr_labor,
                "mtr_capinc": md.mtr_capital,
                "etr": md.etr,
                "age": md.age,
                "total_labinc": md.labor_income,
                "total_capinc": md.capital_income,
                "market_income": md.labor_income + md.capital_income,
                "total_tax_liab": md.etr * (md.labor_income + md.capital_income),
                "payroll_tax_liab": np.zeros(len(md.age)),
                "year": np.full(len(md.age), year),
                "weight": md.weight,
            }))

        all_data = pd.concat(frames, ignore_index=True)

        etr_params_S, mtrx_params_S, mtry_params_S, avg_income, frac_payroll = (
            _estimate_tax_functions(all_data, S)
        )
        BW = years
        frac_tax_payroll = np.full(BW, frac_payroll)

        demo = demographics.get_pop_objs(20, S, 320, start_year)

        return CalibrationResult(
            etr_params=etr_params_S,
            mtrx_params=mtrx_params_S,
            mtry_params=mtry_params_S,
            mean_income=avg_income,
            frac_tax_payroll=frac_tax_payroll,
            g_n_ss=float(demo["g_n_ss"]),
            omega=demo["omega"],
            omega_SS=demo["omega_SS"],
            rho=np.tile(demo["rho"].reshape(1, -1), (320 + S, 1)),
            g_n=demo["g_n"],
            imm_rates=demo["imm_rates"],
            omega_S_preTP=demo["omega_S_preTP"],
        )


def solve_steady_state(
    start_year: int = 2026,
    policy: Policy | None = None,
    max_iter: int = 250,
) -> SteadyStateResult:
    """
    Solve for steady state equilibrium.

    Args:
        start_year: First year of simulation
        policy: Optional PolicyEngine Policy object for reform scenario
        max_iter: Maximum iterations for solver

    Returns:
        SteadyStateResult with equilibrium values
    """
    from ogcore.parameters import Specifications
    from ogcore.SS import run_SS

    with tempfile.TemporaryDirectory() as tmpdir, _suppress_output():
        p = Specifications(baseline=True, output_base=tmpdir, baseline_dir=tmpdir)

        # Load UK default parameters (fiscal, tax rates, economic params)
        defaults_path = os.path.join(os.path.dirname(__file__), "oguk_default_parameters.json")
        with open(defaults_path) as f:
            defaults = json.load(f)
        # Strip keys that calibration will override
        for key in [
            "etr_params", "mtrx_params", "mtry_params", "mean_income_data",
            "frac_tax_payroll", "omega", "omega_SS", "rho", "g_n", "g_n_ss",
            "imm_rates", "omega_S_preTP",
        ]:
            defaults.pop(key, None)
        p.update_specifications(defaults)

        p.update_specifications({
            "tax_func_type": "GS",
            "age_specific": False,
            "start_year": start_year,
        })

        cal = calibrate(start_year=start_year, policy=policy)

        T, S = p.T, p.S
        BW = len(cal.etr_params)
        etr_params = [[cal.etr_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)]
        mtrx_params = [[cal.mtrx_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)]
        mtry_params = [[cal.mtry_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)]

        frac_tax = np.append(cal.frac_tax_payroll, np.ones(T + S - BW) * cal.frac_tax_payroll[-1])
        p.update_specifications({
            "etr_params": etr_params,
            "mtrx_params": mtrx_params,
            "mtry_params": mtry_params,
            "mean_income_data": cal.mean_income,
            "frac_tax_payroll": frac_tax.tolist(),
        })

        p.g_n_ss = cal.g_n_ss
        p.omega = cal.omega
        p.omega_SS = cal.omega_SS
        p.rho = cal.rho
        p.g_n = cal.g_n
        p.imm_rates = cal.imm_rates
        p.omega_S_preTP = cal.omega_S_preTP
        p.maxiter = max_iter

        ss = run_SS(p, client=None)

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
