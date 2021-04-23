"""
------------------------------------------------------------------------
This program extracts tax rate and income data from the microsimulation
model (OpenFisca-UK).
------------------------------------------------------------------------
"""
from dask import delayed, compute
import dask.multiprocessing
import numpy as np
import os
import pickle
from openfisca_uk import PopulationSim
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

CUR_PATH = os.path.split(os.path.abspath(__file__))[0]
DATA_LAST_YEAR = 2030  # this is the last year data are extrapolated for


def get_calculator_output(baseline, year, reform=None, data=None):
    """
    This function creates an OpenFisca PopulationSim object with the
    policy specified in reform and the data specified with the data
    kwarg.

    Args:
        baseline (boolean): True if baseline tax policy
        year (int): year of data to simulate
        reform (OpenFisca Reform object): IIT policy reform parameters,
            None if baseline
        data (DataFrame or str): DataFrame or path to datafile for
            the PopulationSim object

    Returns:
        tax_dict (dict): a dictionary of microdata with marginal tax
            rates and other information computed from OpenFisca-UK

    """
    # create a simulation
    if data is None or "frs":
        sim = PopulationSim(reform)
    else:
        # pass PopulationSim a data argument
        pass
    if baseline:
        print("Running current law policy baseline")
    else:
        print("Baseline policy is: ", reform)

    # Check that start_year is appropriate
    if year > DATA_LAST_YEAR:
        raise RuntimeError("Start year is beyond data extrapolation.")

    # define market income - taking expanded_income and excluding gov't
    # transfer benefits found in the Tax-Calculator expanded income
    market_income = sim.df(["gross_income"]).values.squeeze()

    # Compute marginal tax rates (can only do on earned income now)
    mtr = sim.calc_mtr()

    # Put MTRs, income, tax liability, and other variables in dict
    length = len(sim.df(["benunit_weight"]))
    tax_dict = {
        "mtr_labinc": mtr,
        "mtr_capinc": mtr,
        "age": sim.df(["age"]).values.squeeze(),
        "total_labinc": sim.df(["earned_income"]).values.squeeze(),
        "total_capinc": market_income
        - sim.df(["earned_income"]).values.squeeze(),
        "market_income": market_income,
        "total_tax_liab": sim.df(["income_tax"]).values.squeeze(),
        "payroll_tax_liab": np.zeros(length),  # is this in OpenFisca-UK?
        "etr": sim.df(["income_tax"]).values.squeeze() / market_income,
        "year": year * np.ones(length),
        "weight": sim.df(["benunit_weight"]).values.squeeze(),
    }

    # garbage collection
    del sim

    return tax_dict


def get_data(
    baseline=False,
    start_year=2021,
    reform=None,
    data=None,
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
        reform (OpenFisca Reform object): IIT policy reform parameters,
            None if baseline
        data (DataFrame or str): DataFrame or path to datafile for
            the PopulationSim object
        path (str): path to save microdata files to
        client (Dask Client object): client for Dask multiprocessing
        num_workers (int): number of workers to use for Dask
            multiprocessing

    Returns:
        micro_data_dict (dict): dict of Pandas Dataframe, one for each
            year from start_year to the maximum year OpenFisca-UK can
            analyze
        OpenFiscaUK_version (str): version of OpenFisca-UK used

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
        results = results = compute(
            *lazy_values,
            scheduler=dask.multiprocessing.get,
            num_workers=num_workers,
        )

    # dictionary of data frames to return
    micro_data_dict = {}
    for i, result in enumerate(results):
        year = start_year + i
        micro_data_dict[str(year)] = pd.DataFrame.from_dict(result)

    if baseline:
        pkl_path = os.path.join(path, "micro_data_baseline.pkl")
    else:
        pkl_path = os.path.join(path, "micro_data_policy.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(micro_data_dict, f)

    # Do some garbage collection
    del results

    # Pull OpenFisca-UK version for reference
    OpenFiscaUK_version = (
        None  # pkg_resources.get_distribution("taxcalc").version
    )

    return micro_data_dict, OpenFiscaUK_version
