"""Example: Run OG-UK baseline and reform steady state simulations."""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime

import numpy as np
from ogcore.parameters import Specifications
from ogcore.SS import run_SS
from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest

from oguk import calibrate


def _load_defaults() -> dict:
    """Load UK default parameters, stripping keys overridden by calibration."""
    path = os.path.join(os.path.dirname(__file__), "..", "oguk", "oguk_default_parameters.json")
    with open(path) as f:
        defaults = json.load(f)
    for key in [
        "etr_params", "mtrx_params", "mtry_params", "mean_income_data",
        "frac_tax_payroll", "omega", "omega_SS", "rho", "g_n", "g_n_ss",
        "imm_rates", "omega_S_preTP",
    ]:
        defaults.pop(key, None)
    return defaults


def _apply_calibration(p: Specifications, cal) -> None:
    """Apply calibration results to a Specifications object."""
    T, S = p.T, p.S
    BW = len(cal.etr_params)
    p.update_specifications({
        "etr_params": [[cal.etr_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mtrx_params": [[cal.mtrx_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mtry_params": [[cal.mtry_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mean_income_data": cal.mean_income,
        "frac_tax_payroll": (list(cal.frac_tax_payroll)
                             + [cal.frac_tax_payroll[-1]] * (T + S - BW)),
    })
    p.g_n_ss = cal.g_n_ss
    p.omega = cal.omega
    p.omega_SS = cal.omega_SS
    p.rho = cal.rho
    p.g_n = cal.g_n
    p.imm_rates = cal.imm_rates
    p.omega_S_preTP = cal.omega_S_preTP


def main():
    """Run baseline and reform OG-UK steady state simulations."""
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-UK-Example", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-UK-Example", "OUTPUT_REFORM")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(reform_dir, exist_ok=True)

    defaults = _load_defaults()

    # ---- Baseline ----
    print("Setting up baseline...")
    p = Specifications(baseline=True, baseline_dir=base_dir, output_base=base_dir)
    p.update_specifications(defaults)
    p.update_specifications({
        "tax_func_type": "GS",
        "age_specific": False,
        "start_year": 2026,
    })

    print("Calibrating baseline tax functions and demographics...")
    cal = calibrate(start_year=2026)
    _apply_calibration(p, cal)

    print("Solving baseline steady state...")
    t0 = time.time()
    ss_base = run_SS(p, client=None)
    print(f"Baseline SS solved in {time.time() - t0:.1f}s")

    # ---- Reform: increase basic rate from 20% to 21% ----
    print("\nSetting up reform (basic rate 20% -> 21%)...")
    basic_rate_param = uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate")
    reform = Policy(
        name="Basic rate 21%",
        parameter_values=[
            ParameterValue(
                parameter=basic_rate_param,
                value=0.21,
                start_date=datetime(2026, 1, 1),
            )
        ],
    )

    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir

    print("Calibrating reform tax functions...")
    cal2 = calibrate(start_year=2026, policy=reform)
    _apply_calibration(p2, cal2)

    print("Solving reform steady state...")
    t0 = time.time()
    ss_reform = run_SS(p2, client=None)
    print(f"Reform SS solved in {time.time() - t0:.1f}s")

    # ---- Results ----
    print("\n" + "=" * 60)
    print("Steady state comparison: baseline vs reform")
    print("=" * 60)
    print(f"{'Variable':<15} {'Baseline':>12} {'Reform':>12} {'% Change':>12}")
    print("-" * 51)
    for var in ["Y", "C", "K", "L", "r", "w", "G", "D", "total_tax_revenue"]:
        base_val = float(np.asarray(ss_base[var]).flat[0])
        ref_val = float(np.asarray(ss_reform[var]).flat[0])
        pct = ((ref_val - base_val) / abs(base_val)) * 100 if base_val != 0 else 0
        label = "tax_revenue" if var == "total_tax_revenue" else var
        print(f"{label:<15} {base_val:>12.4f} {ref_val:>12.4f} {pct:>11.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
