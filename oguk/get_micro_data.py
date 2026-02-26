"""
Extract tax rate and income data from PolicyEngine-UK.

This module provides the interface between OG-UK and the PolicyEngine-UK
microsimulation model. It uses the modern policyengine Python package
(not the legacy policyengine_uk package) via the internal helpers in
oguk.api.
"""

import numpy as np
import pandas as pd


def get_data(
    baseline=False,
    start_year=2026,
    reform=None,
    data="frs",
    path=None,
    client=None,
    num_workers=1,
):
    """Extract microdata from PolicyEngine-UK for tax function estimation.

    Uses the new policyengine package API (see oguk.api._get_micro_data).
    Returns data in the format expected by ogcore.txfunc.tax_func_estimate.

    Args:
        baseline (bool): True if running baseline (no reform) policy
        start_year (int): first year of budget window
        reform (PolicyEngine Policy or None): reform policy, None for baseline
        data (str): dataset identifier (unused; kept for API compatibility)
        path (str or None): output directory (unused; kept for API compatibility)
        client: Dask client (unused; kept for API compatibility)
        num_workers (int): number of workers (unused; kept for API compatibility)

    Returns:
        micro_data_dict (dict): dict of Pandas DataFrames keyed by year string,
            one for each year from start_year to start_year + 2. Each DataFrame
            has columns: mtr_labinc, mtr_capinc, etr, age, total_labinc,
            total_capinc, market_income, total_tax_liab, payroll_tax_liab,
            year, weight.
        policyengine_version (str or None): PolicyEngine version used
    """
    import tempfile

    from oguk.api import _get_micro_data

    years = range(start_year, start_year + 3)
    micro_data_dict = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for year in years:
            policy = None if baseline else reform
            md = _get_micro_data(year, policy, tmpdir)
            df = pd.DataFrame(
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
            micro_data_dict[str(year)] = df

    try:
        import importlib.metadata

        policyengine_version = importlib.metadata.version("policyengine")
    except Exception:
        policyengine_version = None

    return micro_data_dict, policyengine_version
