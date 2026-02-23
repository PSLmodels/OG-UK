"""Example: Run OG-UK baseline and reform simulations."""

import copy
import json
import os
import time
from datetime import datetime

from ogcore import output_plots as op
from ogcore import output_tables as ot
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle
from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest

from oguk import calibrate


def main():
    """Run baseline and reform OG-UK simulations."""
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-UK-Example", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-UK-Example", "OUTPUT_REFORM")

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    p = Specifications(baseline=True, baseline_dir=base_dir, output_base=base_dir)
    p.update_specifications(
        json.load(open(os.path.join(CUR_DIR, "..", "oguk", "oguk_default_parameters.json")))
    )
    p.update_specifications({
        "tax_func_type": "DEP",
        "age_specific": False,
        "start_year": 2026,
    })

    # Calibrate tax functions and demographics
    cal = calibrate(start_year=2026)

    # Update tax function parameters
    T, S = p.T, p.S
    BW = len(cal.etr_params)
    p.update_specifications({
        "etr_params": [[cal.etr_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mtrx_params": [[cal.mtrx_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mtry_params": [[cal.mtry_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mean_income_data": cal.mean_income,
        "frac_tax_payroll": list(cal.frac_tax_payroll) + [cal.frac_tax_payroll[-1]] * (T + S - BW),
    })

    # Update demographic parameters
    p.g_n_ss = cal.g_n_ss
    p.omega = cal.omega
    p.omega_SS = cal.omega_SS
    p.rho = cal.rho
    p.g_n = cal.g_n
    p.imm_rates = cal.imm_rates
    p.omega_S_preTP = cal.omega_S_preTP

    start_time = time.time()
    runner(p, time_path=True)
    print(f"Baseline run time: {time.time() - start_time:.1f}s")

    # -------------------------------------------------------------------------
    # Reform: Increase basic rate from 20% to 30%
    # -------------------------------------------------------------------------
    basic_rate_param = uk_latest.get_parameter("gov.hmrc.income_tax.rates.uk[0].rate")
    reform = Policy(
        name="Basic rate 30%",
        parameter_values=[
            ParameterValue(
                parameter=basic_rate_param,
                value=0.30,
                start_date=datetime(2026, 1, 1),
            )
        ],
    )

    p2 = copy.deepcopy(p)
    p2.baseline = False
    p2.output_base = reform_dir

    # Calibrate with reform
    cal2 = calibrate(start_year=2026, policy=reform)
    p2.update_specifications({
        "etr_params": [[cal2.etr_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mtrx_params": [[cal2.mtrx_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mtry_params": [[cal2.mtry_params[min(t, BW - 1)][s] for s in range(S)] for t in range(T)],
        "mean_income_data": cal2.mean_income,
        "frac_tax_payroll": list(cal2.frac_tax_payroll) + [cal2.frac_tax_payroll[-1]] * (T + S - BW),
    })

    start_time = time.time()
    runner(p2, time_path=True)
    print(f"Reform run time: {time.time() - start_time:.1f}s")

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(os.path.join(reform_dir, "TPI", "TPI_vars.pkl"))
    reform_params = safe_read_pickle(os.path.join(reform_dir, "model_params.pkl"))

    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )
    print("Percentage changes in aggregates:")
    print(ans)

    op.plot_all(base_dir, reform_dir, os.path.join(CUR_DIR, "OG-UK_example_plots"))
    ans.to_csv("oguk_example_output.csv")


if __name__ == "__main__":
    main()
