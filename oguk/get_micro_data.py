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
import logging

logging.basicConfig(level=logging.INFO)

# New API: Microsimulation handles dataset internally
dataset = None

warnings.filterwarnings("ignore")

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_LAST_YEAR = 2026  # this is the last year data are extrapolated for

def generate_age_brackets(n_brackets, min_age=21, max_age=100):
    """
    Generate age brackets by dividing the age range into N equal parts.

    Args:
        n_brackets (int): Number of brackets to create (e.g., 4)
        min_age (int): Minimum age (default 21)
        max_age (int): Maximum age (default 100)

    Returns:
        list: List of (min_age, max_age, representative_age) tuples

    Example:
        >>> generate_age_brackets(4)
        [(21, 40, 30), (41, 60, 50), (61, 80, 70), (81, 100, 90)]
    """
    if n_brackets < 1:
        raise ValueError("n_brackets must be at least 1")

    age_range = max_age - min_age + 1
    bracket_size = age_range // n_brackets
    brackets = []

    for i in range(n_brackets):
        bracket_min = min_age + i * bracket_size
        if i == n_brackets - 1:
            # Last bracket goes to max_age
            bracket_max = max_age
        else:
            bracket_max = bracket_min + bracket_size - 1
        rep_age = (bracket_min + bracket_max) // 2
        brackets.append((bracket_min, bracket_max, rep_age))

    return brackets


def create_custom_brackets(age_ranges):
    """
    Create age brackets from custom age ranges.

    Args:
        age_ranges (list): List of (min_age, max_age) tuples
            Example: [(21, 35), (36, 50), (51, 65), (66, 100)]

    Returns:
        list: List of (min_age, max_age, representative_age) tuples
    """
    brackets = []
    for min_age, max_age in age_ranges:
        rep_age = (min_age + max_age) // 2
        brackets.append((min_age, max_age, rep_age))
    return brackets


# Default age brackets for UK tax function estimation (4 brackets)
# Each tuple is (min_age, max_age, representative_age)
DEFAULT_AGE_BRACKETS = [
    (21, 35, 28),   # Young workers
    (36, 50, 43),   # Mid-career
    (51, 65, 58),   # Late career
    (66, 100, 83),  # Retirement
]


def filter_micro_data_by_age_bracket(micro_data, age_min, age_max):
    """
    Filter micro data dictionary to only include ages in the specified range.

    Args:
        micro_data (dict): Dictionary of DataFrames keyed by year strings
        age_min (int): Minimum age to include
        age_max (int): Maximum age to include

    Returns:
        dict: Filtered dictionary of DataFrames
    """
    filtered = {}
    for year, df in micro_data.items():
        mask = (df['age'] >= age_min) & (df['age'] <= age_max)
        filtered_df = df[mask].copy()
        if len(filtered_df) > 0:
            filtered[year] = filtered_df
    return filtered


def map_age_to_bracket(age, age_brackets=None):
    """
    Map an age to its bracket representative age.

    Args:
        age (int): The actual age
        age_brackets (list): List of (min_age, max_age, representative_age) tuples

    Returns:
        int: The representative age for this bracket
    """
    if age_brackets is None:
        age_brackets = DEFAULT_AGE_BRACKETS

    for min_age, max_age, rep_age in age_brackets:
        if min_age <= age <= max_age:
            return rep_age

    # If age is below first bracket, use first bracket
    if age < age_brackets[0][0]:
        return age_brackets[0][2]
    # If age is above last bracket, use last bracket
    return age_brackets[-1][2]


def apply_age_brackets(df, age_brackets=None):
    """
    Apply age bracket mapping to a DataFrame.

    Args:
        df (DataFrame): DataFrame with 'age' column
        age_brackets (list): List of (min_age, max_age, representative_age) tuples

    Returns:
        DataFrame: DataFrame with ages mapped to bracket representatives
    """
    if age_brackets is None:
        return df

    df = df.copy()
    df["age"] = df["age"].apply(lambda x: map_age_to_bracket(x, age_brackets))
    return df


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
    if baseline is None:
        if reform is not None:
            baseline = Microsimulation(reform=reform, **kwargs)
        else:
            baseline = Microsimulation(**kwargs)
    baseline_var = baseline.calculate(variable, period)
    bonus = baseline.calculate("is_adult", period) * 1  # Increase only adult values
    if reform is not None:
        reformed = Microsimulation(reform=reform, **kwargs)
    else:
        reformed = Microsimulation(**kwargs)
    reformed.set_input(variable, period, baseline_var + bonus)

    household_bonus = reformed.calculate(
        variable, period, map_to="household"
    ) - baseline.calculate(variable, period, map_to="household")
    household_net_change = reformed.calculate(
        "household_net_income", period
    ) - baseline.calculate("household_net_income", period)
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
    sim_kwargs = dict()
    if reform is None:
        sim = Microsimulation(**sim_kwargs)
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
    market_income = sim.calculate("household_market_income", year).values

    # Compute marginal tax rates (can only do on earned income now)

    # Put MTRs, income, tax liability, and other variables in dict
    length = sim.calculate("household_weight", year).size
    household = sim.household
    person = sim.person
    max_age_in_hh = household.max(person("age", year))
    # Fill NaN ages with a default value (e.g., 40 - working age)
    max_age_in_hh = np.where(np.isnan(max_age_in_hh), 40, max_age_in_hh)
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
        "total_labinc": sim.calculate(
            "earned_income", year, map_to="household"
        ).values,
        "total_capinc": sim.calculate(
            "capital_income", year, map_to="household"
        ).values,
        "market_income": market_income,
        "total_tax_liab": sim.calculate("household_tax", year).values,
        "payroll_tax_liab": sim.calculate(
            "national_insurance", year, map_to="household"
        ).values,
        "etr": (
            1
            - (
                sim.calculate(
                    "household_net_income", year, map_to="household"
                ).values
            )
            / market_income
        ).clip(-1, 1.5),
        "year": year * np.ones(length),
        "weight": sim.calculate("household_weight", year).values,
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
    age_brackets=None,
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
        age_brackets (list): List of (min_age, max_age, representative_age)
            tuples for grouping ages. If None, uses individual ages.
            Use DEFAULT_AGE_BRACKETS for 5-group UK brackets.

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
        futures = client.compute(lazy_values)
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
        # Drop rows with NaN ages and fill any remaining NaN values
        # (required by ogcore tax function estimation)
        df = df.dropna(subset=["age"])
        # Also fill any remaining NaN in numeric columns
        df = df.fillna(0)
        # Ensure age is integer
        df["age"] = df["age"].astype(int)
        # Apply age brackets if specified
        if age_brackets is not None:
            df = apply_age_brackets(df, age_brackets)
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
