import multiprocessing
from distributed import Client
import pickle
import cloudpickle
import os
from openfisca_core.model_api import Reform
from og_uk_calibrate.calibrate import Calibration
import ogusa
from ogusa import output_tables as ot
from ogusa import output_plots as op
from ogusa import SS, TPI, utils
from ogusa.utils import safe_read_pickle
import time
from argparse import ArgumentParser

start_time = time.time()
# Set start year and last year -- note that OpenFisca-UK can only do one year
# It does not have data like TaxData produced nor logic for future policy
# like Tax-Calculator does
START_YEAR = 2018
ogusa.parameters.TC_LAST_YEAR = START_YEAR
from ogusa.parameters import Specifications


def main(reform):
    # Define parameters to use for multiprocessing
    client = Client()
    num_workers = min(multiprocessing.cpu_count(), 7)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(CUR_DIR, "OG-UK-Example", "OUTPUT_BASELINE")
    reform_dir = os.path.join(CUR_DIR, "OG-UK-Example", "OUTPUT_REFORM")

    # For now, just call SS and TPI functions, bypassing ogusa.execute.py
    # Note that updates in progress to OG-Core will make the code below
    # more streamlined
    # and will use a modified execute.py with just one call to the
    # execute.runner
    """
    ------------------------------------------------------------------------
    Run baseline policy
    ------------------------------------------------------------------------
    """
    # Set up baseline parameterization
    p = Specifications(
        baseline=True,
        client=client,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )
    # specify tax function form and start year
    p.update_specifications(
        {
            "tax_func_type": "DEP",
            "age_specific": False,
            "start_year": START_YEAR,
            "alpha_T": [5e-3],
            "alpha_G": [5e-3],
        }
    )
    # Estimate baseline tax functions from OpenFisca-UK
    c = Calibration(p, estimate_tax_functions=True)
    # update tax function parameters in Specifications Object
    # Note that updates in progress to OG-Core will make the code below
    # more streamlined
    p.etr_params = c.tax_function_params["etr_params"]
    p.mtrx_params = c.tax_function_params["mtrx_params"]
    p.mtry_params = c.tax_function_params["mtry_params"]
    p.mean_income_data = c.tax_function_params["mean_income_data"]
    p.frac_tax_payroll = c.tax_function_params["frac_tax_payroll"]
    # Solve SS
    p.initial_guess_r_SS = 0.02

    # create new Specifications object for reform simulation
    p2 = Specifications(
        baseline=False,
        client=client,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=reform_dir,
    )
    # specify tax function form and start year
    p2.update_specifications(
        {
            "tax_func_type": "DEP",
            "age_specific": False,
            "start_year": START_YEAR,
            "alpha_T": [5e-3],
            "alpha_G": [5e-3],
        }
    )
    # Estimate reform tax functions from OpenFisca-UK, passing Reform
    # class object
    c2 = Calibration(p2, iit_reform=reform, estimate_tax_functions=True)
    # update tax function parameters in Specifications Object
    # Note that updates in progress to OG-Core will make the code below
    # more streamlined
    p2.etr_params = c2.tax_function_params["etr_params"]
    p2.mtrx_params = c2.tax_function_params["mtrx_params"]
    p2.mtry_params = c2.tax_function_params["mtry_params"]
    p2.mean_income_data = c2.tax_function_params["mean_income_data"]
    p2.frac_tax_payroll = c2.tax_function_params["frac_tax_payroll"]

    ss_outputs = SS.run_SS(p, client=client)
    # Save SS results
    utils.mkdirs(os.path.join(base_dir, "SS"))
    ss_file = os.path.join(base_dir, "SS", "SS_vars.pkl")
    with open(ss_file, "wb") as f:
        pickle.dump(ss_outputs, f)
    # Save parameters
    param_file = os.path.join(base_dir, "model_params.pkl")
    with open(param_file, "wb") as f:
        cloudpickle.dump((p), f)
    # Run TPI
    tpi_output = TPI.run_TPI(p, client=client)
    # Save TPI results
    tpi_dir = os.path.join(base_dir, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_file = os.path.join(tpi_dir, "TPI_vars.pkl")
    with open(tpi_file, "wb") as f:
        pickle.dump(tpi_output, f)
    """
    ------------------------------------------------------------------------
    Run reform policy
    ------------------------------------------------------------------------
    """

    # Solve SS
    ss_outputs2 = SS.run_SS(p2, client=client)
    # Save SS results
    utils.mkdirs(os.path.join(reform_dir, "SS"))
    ss_file = os.path.join(reform_dir, "SS", "SS_vars.pkl")
    with open(ss_file, "wb") as f:
        pickle.dump(ss_outputs2, f)
    # Save parameters
    param_file = os.path.join(reform_dir, "model_params.pkl")
    with open(param_file, "wb") as f:
        cloudpickle.dump((p2), f)
    # Run TPI
    tpi_output2 = TPI.run_TPI(p2, client=client)
    # Save TPI results
    tpi_dir = os.path.join(reform_dir, "TPI")
    utils.mkdirs(tpi_dir)
    tpi_file = os.path.join(tpi_dir, "TPI_vars.pkl")
    with open(tpi_file, "wb") as f:
        pickle.dump(tpi_output2, f)
    """
    ------------------------------------------------------------------------
    Save some results of simulations
    ------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
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

    # create plots of output
    op.plot_all(
        base_dir, reform_dir, os.path.join(CUR_DIR, "OG-UK_example_plots")
    )

    print("Percentage changes in aggregates:", ans)
    # save percentage change output to csv file
    ans.to_csv("ogusa_example_output.csv")
    print(f"Completed in {time.time() - start_time}s")
    client.close()


if __name__ == "__main__":
    # execute only if run as a script

    parser = ArgumentParser(
        description="A script to run the main OG-UK routine on a reform."
    )
    parser.add_argument(
        "reform",
        help="The Python reform object to use as a reform (if `reform` is defined in `reform_file.py`, then use `reform_file.reform`)",
    )
    args = parser.parse_args()

    reform_path = args.reform.split(".")
    python_module, object_name = ".".join(reform_path[:-1]), reform_path[-1]
    reform = getattr(__import__(python_module), object_name)
    main(reform)
