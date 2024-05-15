"""
------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation
model (PolicyEngine-UK).
------------------------------------------------------------------------
"""

from dask import delayed, compute
import dask.multiprocessing
import numpy as np
import os
import pickle
from policyengine_uk import Microsimulation
import pandas as pd
import warnings
from policyengine_uk.model_api import *
from policyengine_uk.data import EnhancedFRS, SynthFRS
import logging

logging.basicConfig(level=logging.INFO)

if 2022 in EnhancedFRS.years:
    dataset = EnhancedFRS
else:
    logging.warn(
        """
    Could not locate FRS microdata. If you have access to the data, try running:

    policyengine-uk-data enhanced_frs download 2022
    """
    )
    dataset = SynthFRS  # Change to EnhancedFRS if running locally
    if 2022 not in dataset.years:
        dataset.download(2022)

warnings.filterwarnings("ignore")

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_LAST_YEAR = 2023  # this is the last year data are extrapolated for


def get_household_mtrs(
    reform,
    variable: str,
    period: int = None,
    baseline: Microsimulation = None,
    **kwargs: dict,
) -> pd.Series:
    """Calculates household MTRs with respect to a given variable.

    Args:
        reform (Reform): The reform to apply to the simulation.
        variable (str): The variable to increase.
        period (int): The period (year) to calculate the MTRs for.
        kwargs (dict): Additional arguments to pass to the simulation.

    Returns:
        pd.Series: The household MTRs.
    """
    baseline = baseline or Microsimulation(reform=reform, **kwargs)
    baseline_var = baseline.calc(variable, period)
    bonus = baseline.calc("is_adult", period) * 1  # Increase only adult values
    reformed = Microsimulation(reform=reform, **kwargs)
    reformed.set_input(variable, period, baseline_var + bonus)

    household_bonus = reformed.calc(
        variable, map_to="household", period=period
    ) - baseline.calc(variable, map_to="household", period=period)
    household_net_change = reformed.calc(
        "household_net_income", period=period
    ) - baseline.calc("household_net_income", period=period)
    mtr = (household_bonus - household_net_change) / household_bonus
    mtr = mtr.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    return mtr


def get_calculator_output(baseline, year, reform=None, data=None):
    """
    This function creates a PolicyEngine Microsimulation object with the
    policy specified in reform and the data specified with the data
    kwarg.

    Args:
        baseline (boolean): True if baseline tax policy
        year (int): year of data to simulate
        reform (PolicyEngine Reform object): IIT policy reform parameters,
            None if baseline
        data (DataFrame or str): DataFrame or path to datafile for
            the PopulationSim object

    Returns:
        tax_dict (dict): a dictionary of microdata with marginal tax
            rates and other information computed from PolicyEngine-UK

    """
    # create a simulation
    sim_kwargs = dict(dataset=dataset, dataset_year=2022)
    if reform is None:
        sim = Microsimulation(**sim_kwargs)
        reform = ()
    else:
        sim = Microsimulation(reform=reform, **sim_kwargs)
    if baseline:
        print("Running current law policy baseline")
    else:
        print("Baseline policy is: ", reform)

    sim.year = 2022

    # Check that start_year is appropriate
    if year > DATA_LAST_YEAR:
        raise RuntimeError("Start year is beyond data extrapolation.")

    # define market income - taking expanded_income and excluding gov't
    # transfer benefits
    market_income = sim.calc("household_market_income", period=year).values

    # Compute marginal tax rates (can only do on earned income now)

    # Put MTRs, income, tax liability, and other variables in dict
    length = sim.calc("household_weight", period=year).size
    household = sim.populations["household"]
    person = sim.populations["person"]
    max_age_in_hh = household.max(person("age", "2022"))
    tax_dict = {
        "mtr_labinc": get_household_mtrs(
            reform,
            "employment_income",
            period=year,
            baseline=sim,
            **sim_kwargs,
        ).values,
        "mtr_capinc": get_household_mtrs(
            reform,
            "savings_interest_income",
            period=year,
            baseline=sim,
            **sim_kwargs,
        ).values,
        "age": max_age_in_hh,
        "total_labinc": sim.calc(
            "earned_income", map_to="household", period=year
        ).values,
        "total_capinc": sim.calc(
            "capital_income", map_to="household", period=year
        ).values,
        "market_income": market_income,
        "total_tax_liab": sim.calc("household_tax", period=year).values,
        "payroll_tax_liab": sim.calc(
            "national_insurance", map_to="household", period=year
        ).values,
        "etr": (
            1
            - (
                sim.calc(
                    "household_net_income", map_to="household", period=year
                ).values
            )
            / market_income
        ).clip(-1, 1.5),
        "year": year * np.ones(length),
        "weight": sim.calc("household_weight", period=year).values,
    }

    return tax_dict


def get_data(
    baseline=False,
    start_year=2021,
    reform=None,
    data="frs",
    path=CUR_PATH,
    client=None,
    num_workers=1,
):
    """
    This function creates dataframes of micro data with marginal tax
    rates and information to compute effective tax rates from the
    PopulationSim object.  The resulting dictionary of dataframes is
    returned and saved to disk in a pickle file.

    Args:
        baseline (boolean): True if baseline tax policy
        start_year (int): first year of budget window
        reform (PolicyEngine Reform object): IIT policy reform parameters,
            None if baseline
        data (DataFrame or str): DataFrame or path to datafile for
            the PopulationSim object
        path (str): path to save microdata files to
        client (Dask Client object): client for Dask multiprocessing
        num_workers (int): number of workers to use for Dask
            multiprocessing

    Returns:
        micro_data_dict (dict): dict of Pandas Dataframe, one for each
            year from start_year to the maximum year PolicyEngine-UK can
            analyze
        PolicyEngineUK_version (str): version of PolicyEngine-UK used

    """
    # Compute MTRs and taxes or each year, but not beyond DATA_LAST_YEAR
    lazy_values = []
    for year in range(start_year, DATA_LAST_YEAR + 1):
        lazy_values.append(
            delayed(get_calculator_output)(baseline, year, reform, data)
        )
    if client:  # pragma: no cover
        futures = client.compute(lazy_values, num_workers=num_workers)
        results = client.gather(futures)
    else:
        results = compute(
            *lazy_values,
            scheduler=dask.multiprocessing.get,
            num_workers=num_workers,
        )

    # dictionary of data frames to return
    micro_data_dict = {}
    for i, result in enumerate(results):
        year = start_year + i
        df = pd.DataFrame.from_dict(result)
        micro_data_dict[str(year)] = df

    if baseline:
        pkl_path = os.path.join(path, "micro_data_baseline.pkl")
    else:
        pkl_path = os.path.join(path, "micro_data_policy.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(micro_data_dict, f)

    # Do some garbage collection
    del results

    # Pull PolicyEngine-UK version for reference
    PolicyEngineUK_version = (
        None  # pkg_resources.get_distribution("taxcalc").version
    )

    return micro_data_dict, PolicyEngineUK_version
