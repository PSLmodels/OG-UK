"""Clean, functional API for OG-UK."""

from __future__ import annotations

import contextlib
import io
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
    emp_inc = person["employment_income"].values
    cap_inc = person["savings_interest_income"].values
    length = len(weights)

    baseline_net = person["total_income"].values - person.get("income_tax", np.zeros(length)).values

    def add_labor(s):
        emp = s.calculate("employment_income", year)
        adult = s.calculate("is_adult", year)
        s.set_input("employment_income", year, emp + adult)
        return s

    labor_pol = Policy(name="labor", simulation_modifier=add_labor)
    if policy:
        labor_pol = policy + labor_pol
    labor_sim = Simulation(dataset=dataset, tax_benefit_model_version=uk_latest, policy=labor_pol)
    labor_sim.ensure()
    labor_net = labor_sim.output_dataset.data.person["total_income"].values - labor_sim.output_dataset.data.person.get("income_tax", np.zeros(length)).values
    mtr_labor = np.clip(1 - (labor_net - baseline_net), 0, 1)

    def add_cap(s):
        sav = s.calculate("savings_interest_income", year)
        adult = s.calculate("is_adult", year)
        s.set_input("savings_interest_income", year, sav + adult)
        return s

    cap_pol = Policy(name="cap", simulation_modifier=add_cap)
    if policy:
        cap_pol = policy + cap_pol
    cap_sim = Simulation(dataset=dataset, tax_benefit_model_version=uk_latest, policy=cap_pol)
    cap_sim.ensure()
    cap_net = cap_sim.output_dataset.data.person["total_income"].values - cap_sim.output_dataset.data.person.get("income_tax", np.zeros(length)).values
    mtr_capital = np.clip(1 - (cap_net - baseline_net), 0, 1)

    market_inc = emp_inc + cap_inc
    tax = person.get("income_tax", np.zeros(length)).values + person.get("national_insurance", np.zeros(length)).values
    etr = np.where(market_inc > 0, tax / market_inc, 0)
    etr = np.clip(etr, -1, 1.5)

    adult_mask = age >= 18

    return _MicroData(
        mtr_labor=mtr_labor[adult_mask],
        mtr_capital=mtr_capital[adult_mask],
        etr=etr[adult_mask],
        age=age[adult_mask],
        labor_income=emp_inc[adult_mask],
        capital_income=cap_inc[adult_mask],
        weight=weights[adult_mask],
        year=year,
    )


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
    from ogcore import txfunc
    from oguk import demographics

    with tempfile.TemporaryDirectory() as tmpdir:
        micro_data = {}
        for year in range(start_year, start_year + years):
            md = _get_micro_data(year, policy, tmpdir)
            micro_data[str(year)] = pd.DataFrame({
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
            })

        S = 80
        tax_func_path = os.path.join(tmpdir, "tax_funcs.pkl")

        dict_params = txfunc.tax_func_estimate(
            micro_data, years, S, 21, 100,
            start_year=start_year,
            analytical_mtrs=False,
            tax_func_type="DEP",
            age_specific=False,
            tax_func_path=tax_func_path,
        )

        demo = demographics.get_pop_objs(20, S, 320, start_year)

        return CalibrationResult(
            etr_params=dict_params["tfunc_etr_params_S"],
            mtrx_params=dict_params["tfunc_mtrx_params_S"],
            mtry_params=dict_params["tfunc_mtry_params_S"],
            mean_income=float(dict_params["tfunc_avginc"][0]),
            frac_tax_payroll=dict_params["tfunc_frac_tax_payroll"],
            g_n_ss=float(demo["g_n_ss"]),
            omega=demo["omega"],
            omega_SS=demo["omega_SS"],
            rho=demo["rho"],
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
        p.update_specifications({
            "tax_func_type": "DEP",
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
        p.rho = np.tile(cal.rho.reshape(1, -1), (T + S, 1))
        p.g_n = cal.g_n
        p.imm_rates = cal.imm_rates
        p.omega_S_preTP = cal.omega_S_preTP
        p.maxiter = max_iter

        ss = run_SS(p, client=None)

        return SteadyStateResult(
            r=float(ss["r"]),
            w=float(ss["w"]),
            Y=float(ss["Y"]),
            K=float(ss["K"]),
            L=float(ss["L"]),
            C=float(ss["C"]),
            I=float(ss["I"]),
            G=float(ss["G"]),
            tax_revenue=float(ss["total_tax_revenue"]),
            debt=float(ss["D"]),
        )
