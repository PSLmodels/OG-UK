from ogcore import txfunc
from oguk import get_micro_data
from oguk import demographics
import os
import numpy as np
import hashlib
import json
import pickle
from ogcore.utils import safe_read_pickle, mkdirs
import pkg_resources


CUR_PATH = os.path.split(os.path.abspath(__file__))[0]

# Import age bracket utilities
from oguk.get_micro_data import (
    DEFAULT_AGE_BRACKETS,
    map_age_to_bracket,
    generate_age_brackets,
    create_custom_brackets,
    filter_micro_data_by_age_bracket,
)


def _compute_taxfunc_cache_key(
    baseline, start_year, reform, tax_func_type, age_brackets, S, analytical_mtrs
):
    """
    Compute a hash-based cache key for tax function estimates.

    Args:
        baseline (bool): Whether this is baseline policy
        start_year (int): Start year of simulation
        reform (dict): Reform parameters (None for baseline)
        tax_func_type (str): Type of tax function (e.g., 'DEP')
        age_brackets (list): Age bracket configuration
        S (int): Number of model periods
        analytical_mtrs (bool): Whether to use analytical MTRs

    Returns:
        str: A hex hash string that uniquely identifies this configuration
    """
    cache_dict = {
        'baseline': bool(baseline),
        'start_year': int(start_year),
        'tax_func_type': str(tax_func_type),
        'S': int(S),
        'analytical_mtrs': bool(analytical_mtrs),
    }

    # Add reform parameters
    if reform is not None:
        try:
            cache_dict['reform'] = json.dumps(reform, sort_keys=True)
        except (TypeError, ValueError):
            cache_dict['reform'] = str(reform)
    else:
        cache_dict['reform'] = None

    # Add age brackets
    if age_brackets is not None:
        cache_dict['age_brackets'] = str(age_brackets)
    else:
        cache_dict['age_brackets'] = None

    cache_str = json.dumps(cache_dict, sort_keys=True)
    cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()[:16]

    return cache_hash


def _get_taxfunc_cache_path(output_base, baseline, cache_key, use_brackets=False):
    """Get the cache file path for tax function estimates."""
    policy_type = "baseline" if baseline else "reform"
    bracket_suffix = "_brackets" if use_brackets else ""
    filename = f"TxFuncEst_cache_{policy_type}{bracket_suffix}_{cache_key}.pkl"
    return os.path.join(output_base, filename)


def _load_cached_taxfunc(cache_path, p):
    """
    Load cached tax function parameters if valid.

    Returns:
        tuple: (params_dict, is_valid) - params if cache hit, None otherwise
    """
    if not os.path.exists(cache_path):
        return None, False

    try:
        cached = safe_read_pickle(cache_path)

        # Validate cache
        if cached.get('start_year') != p.start_year:
            print(f"  [CACHE INVALID] start_year mismatch")
            return None, False
        if cached.get('tax_func_type') != p.tax_func_type:
            print(f"  [CACHE INVALID] tax_func_type mismatch")
            return None, False

        print(f"  [CACHE HIT] Loading cached tax functions from {os.path.basename(cache_path)}")
        return cached, True

    except Exception as e:
        print(f"  [CACHE ERROR] Could not load cache: {e}")
        return None, False


def _save_taxfunc_cache(cache_path, params_dict):
    """Save tax function parameters to cache."""
    try:
        mkdirs(os.path.dirname(cache_path))
        with open(cache_path, 'wb') as f:
            pickle.dump(params_dict, f)
        print(f"  [CACHE SAVED] Tax functions cached to {os.path.basename(cache_path)}")
    except Exception as e:
        print(f"  [CACHE WARNING] Could not save cache: {e}")


class Calibration:
    """OG-UK calibration class"""

    def __init__(
        self,
        p,
        estimate_tax_functions=False,
        estimate_beta=False,
        estimate_chi_n=False,
        tax_func_path=None,
        iit_reform=None,
        guid="",
        data="frs",
        client=None,
        num_workers=1,
        age_brackets=None,
        use_age_brackets=False,
        n_brackets=None,
    ):
        """
        Initialize the Calibration class.

        Args:
            p: OG-Core Specifications object
            estimate_tax_functions (bool): Whether to estimate tax functions
            estimate_beta (bool): Whether to estimate beta
            estimate_chi_n (bool): Whether to estimate chi_n
            tax_func_path (str): Path to cached tax function parameters
            iit_reform (dict): PolicyEngine reform dictionary
            guid (str): Unique identifier for output files
            data (str): Data source (default "frs")
            client: Dask client for parallel processing
            num_workers (int): Number of workers for parallel processing
            age_brackets (list): Custom age brackets as list of (min, max, rep) tuples
                or list of (min, max) tuples
            use_age_brackets (bool): Whether to use age bracket estimation
            n_brackets (int): Number of brackets to auto-generate (e.g., 4)
                Only used if use_age_brackets=True and age_brackets is None
        """
        self.estimate_tax_functions = estimate_tax_functions
        self.estimate_beta = estimate_beta
        self.estimate_chi_n = estimate_chi_n
        self.use_age_brackets = use_age_brackets
        self.n_brackets = n_brackets

        # Resolve age brackets
        if use_age_brackets:
            if age_brackets is not None:
                # Check if tuples have 2 or 3 elements
                if len(age_brackets[0]) == 2:
                    self.age_brackets = create_custom_brackets(age_brackets)
                else:
                    self.age_brackets = age_brackets
            elif n_brackets is not None:
                self.age_brackets = generate_age_brackets(n_brackets)
            else:
                self.age_brackets = DEFAULT_AGE_BRACKETS
            print(
                f"Using {len(self.age_brackets)} age brackets: {self.age_brackets}"
            )
        else:
            self.age_brackets = age_brackets

        if estimate_tax_functions:
            if tax_func_path is not None:
                run_micro = False
            else:
                run_micro = True

            if use_age_brackets:
                # Use bracket-based estimation
                self.tax_function_params = (
                    self.estimate_tax_functions_by_bracket(
                        p,
                        iit_reform,
                        guid,
                        data,
                        client,
                        num_workers,
                        run_micro=run_micro,
                        tax_func_path=tax_func_path,
                    )
                )
            else:
                # Original behavior
                self.tax_function_params = self.get_tax_function_parameters(
                    p,
                    iit_reform,
                    guid,
                    data,
                    client,
                    num_workers,
                    run_micro=run_micro,
                    tax_func_path=tax_func_path,
                    age_brackets=self.age_brackets,
                )
        # if estimate_beta:
        #     self.beta_j = estimate_beta_j.beta_estimate(self)
        # # if estimate_chi_n:
        # #     chi_n = self.get_chi_n()

        # # Macro estimation
        # self.macro_params = macro_params.get_macro_params()

        # # eta estimation
        # self.eta = transfer_distribution.get_transfer_matrix()

        # # zeta estimation
        # self.zeta = bequest_transmission.get_bequest_matrix()

        # # earnings profiles
        # self.e = income.get_e_interp(
        #     p.S, p.omega_SS, p.omega_SS_80, p.lambdas, plot=False
        # )

        # demographics
        self.demographic_params = demographics.get_pop_objs(
            p.E, p.S, p.T, p.start_year
        )

    # Tax Functions
    def get_tax_function_parameters(
        self,
        p,
        pit_reform={},
        guid="",
        data="",
        client=None,
        num_workers=1,
        run_micro=False,
        tax_func_path=None,
        age_brackets=None,
    ):
        """
        Reads pickle file of tax function parameters or estimates the
        parameters from microsimulation model output.

        Args:
            client (Dask client object): client
            run_micro (bool): whether to estimate parameters from
                microsimulation model
            tax_func_path (string): path where find or save tax
                function parameter estimates

        Returns:
            None

        """
        # set paths if none given
        if tax_func_path is None:
            if p.baseline:
                pckl = "TxFuncEst_baseline{}.pkl".format(guid)
                tax_func_path = os.path.join(p.output_base, pckl)
                print("Using baseline tax parameters from ", tax_func_path)
            else:
                pckl = "TxFuncEst_policy{}.pkl".format(guid)
                tax_func_path = os.path.join(p.output_base, pckl)
                print(
                    "Using reform policy tax parameters from ", tax_func_path
                )
        # create directory for tax function pickles to be saved to
        mkdirs(os.path.split(tax_func_path)[0])
        # If run_micro is false, check to see if parameters file exists
        # and if it is consistent with Specifications instance
        if not run_micro:
            dict_params, run_micro = self.read_tax_func_estimate(
                p, tax_func_path
            )
            taxcalc_version = "Cached tax parameters, no taxcalc version"
        if run_micro:
            micro_data, taxcalc_version = get_micro_data.get_data(
                baseline=p.baseline,
                start_year=p.start_year,
                reform=pit_reform,
                data=data,
                path=p.output_base,
                client=client,
                num_workers=num_workers,
                age_brackets=age_brackets,
            )
            if age_brackets is not None:
                print(
                    f"Using {len(age_brackets)} age brackets for tax function estimation"
                )
            p.BW = len(micro_data)
            dict_params = txfunc.tax_func_estimate(  # pragma: no cover
                micro_data,
                p.BW,
                p.S,
                p.starting_age,
                p.ending_age,
                start_year=p.start_year,
                analytical_mtrs=p.analytical_mtrs,
                tax_func_type=p.tax_func_type,
                age_specific=p.age_specific,
                client=client,
                num_workers=num_workers,
                tax_func_path=tax_func_path,
            )
        mean_income_data = dict_params["tfunc_avginc"][0]
        frac_tax_payroll = np.append(
            dict_params["tfunc_frac_tax_payroll"],
            np.ones(p.T + p.S - p.BW)
            * dict_params["tfunc_frac_tax_payroll"][-1],
        )
        # Conduct checks to be sure tax function params are consistent
        # with the model run
        params_list = ["etr", "mtrx", "mtry"]
        BW_in_tax_params = dict_params["BW"]
        start_year_in_tax_params = dict_params["start_year"]
        S_in_tax_params = len(dict_params["tfunc_etr_params_S"][0])
        # Check that start years are consistent in model and cached tax functions
        if p.start_year != start_year_in_tax_params:
            print(
                "Input Error: There is a discrepancy between the start"
                + " year of the model and that of the tax functions!!"
            )
            assert False
        # Check that S is consistent in model and cached tax functions
        # Note: even if p.age_specific = False, the arrays coming from
        # ogcore.txfunc_est should be of length S
        if p.S != S_in_tax_params:
            print(
                "Input Error: There is a discrepancy between the ages"
                + " used in the model and those in the tax functions!!"
            )
            assert False

        # Extrapolate tax function parameters for years after budget window
        # list of list: BW x S - either an array of function at that element...
        etr_params = [[None] * p.S] * p.T
        mtrx_params = [[None] * p.S] * p.T
        mtry_params = [[None] * p.S] * p.T
        for s in range(p.S):
            for t in range(p.T):
                if t < p.BW:
                    etr_params[t][s] = dict_params["tfunc_etr_params_S"][t][s]
                    mtrx_params[t][s] = dict_params["tfunc_mtrx_params_S"][t][
                        s
                    ]
                    mtry_params[t][s] = dict_params["tfunc_mtry_params_S"][t][
                        s
                    ]
                else:
                    etr_params[t][s] = dict_params["tfunc_etr_params_S"][-1][s]
                    mtrx_params[t][s] = dict_params["tfunc_mtrx_params_S"][-1][
                        s
                    ]
                    mtry_params[t][s] = dict_params["tfunc_mtry_params_S"][-1][
                        s
                    ]

        if p.constant_rates:
            print("Using constant rates!")
            # Make all tax rates equal the average
            p.tax_func_type = "linear"
            etr_params = [[None] * p.S] * p.T
            mtrx_params = [[None] * p.S] * p.T
            mtry_params = [[None] * p.S] * p.T
            for s in range(p.S):
                for t in range(p.T):
                    if t < p.BW:
                        etr_params[t][s] = dict_params["tfunc_avg_etr"][t]
                        mtrx_params[t][s] = dict_params["tfunc_avg_mtrx"][t]
                        mtry_params[t][s] = dict_params["tfunc_avg_mtry"][t]
                    else:
                        etr_params[t][s] = dict_params["tfunc_avg_etr"][-1]
                        mtrx_params[t][s] = dict_params["tfunc_avg_mtrx"][-1]
                        mtry_params[t][s] = dict_params["tfunc_avg_mtry"][-1]
        if p.zero_taxes:
            print("Zero taxes!")
            etr_params = [[0] * p.S] * p.T
            mtrx_params = [[0] * p.S] * p.T
            mtry_params = [[0] * p.S] * p.T
        tax_param_dict = {
            "etr_params": etr_params,
            "mtrx_params": mtrx_params,
            "mtry_params": mtry_params,
            "taxcalc_version": taxcalc_version,
            "mean_income_data": mean_income_data,
            "frac_tax_payroll": frac_tax_payroll,
        }

        return tax_param_dict

    def estimate_tax_functions_by_bracket(
        self,
        p,
        pit_reform={},
        guid="",
        data="",
        client=None,
        num_workers=1,
        run_micro=True,
        tax_func_path=None,
    ):
        """
        Estimate tax functions for each age bracket separately, then replicate
        to all ages in that bracket. This is faster than estimating for each
        individual age (80 estimations) while still capturing age-related
        variation in tax functions.

        Args:
            p: OG-Core Specifications object
            pit_reform (dict): PolicyEngine reform dictionary
            guid (str): Unique identifier for output files
            data (str): Data source
            client: Dask client for parallel processing
            num_workers (int): Number of workers
            run_micro (bool): Whether to run micro data estimation
            tax_func_path (str): Path to cached tax function parameters

        Returns:
            dict: Tax function parameters dictionary
        """
        print(f"\n{'='*60}")
        print(f"BRACKET-BASED TAX FUNCTION ESTIMATION")
        print(f"{'='*60}")
        print(f"Estimating {len(self.age_brackets)} bracket tax functions")
        for i, (age_min, age_max, rep_age) in enumerate(self.age_brackets):
            print(
                f"  Bracket {i+1}: ages {age_min}-{age_max} (rep: {rep_age})"
            )
        print(f"{'='*60}\n")

        # Compute hash-based cache key
        cache_key = _compute_taxfunc_cache_key(
            baseline=p.baseline,
            start_year=p.start_year,
            reform=pit_reform if pit_reform else None,
            tax_func_type=p.tax_func_type,
            age_brackets=self.age_brackets,
            S=p.S,
            analytical_mtrs=p.analytical_mtrs,
        )
        cache_path = _get_taxfunc_cache_path(
            p.output_base, p.baseline, cache_key, use_brackets=True
        )

        # Set up legacy paths for backwards compatibility
        if tax_func_path is None:
            if p.baseline:
                pckl = "TxFuncEst_baseline_brackets{}.pkl".format(guid)
                tax_func_path = os.path.join(p.output_base, pckl)
            else:
                pckl = "TxFuncEst_policy_brackets{}.pkl".format(guid)
                tax_func_path = os.path.join(p.output_base, pckl)
        mkdirs(os.path.split(tax_func_path)[0])

        # Check for hash-based cached results first
        cached_params, cache_valid = _load_cached_taxfunc(cache_path, p)
        if cache_valid and cached_params is not None:
            # Also save to legacy path
            with open(tax_func_path, 'wb') as f:
                pickle.dump(cached_params, f)
            return self._process_bracket_tax_params(cached_params, p)

        # Check legacy cache as fallback
        if not run_micro and os.path.exists(tax_func_path):
            print(f"Checking legacy cache at {tax_func_path}")
            cached_params = safe_read_pickle(tax_func_path)
            if cached_params.get("start_year") == p.start_year:
                print(f"  [CACHE HIT] Using legacy cached tax functions")
                # Save to new hash-based cache for future use
                _save_taxfunc_cache(cache_path, cached_params)
                return self._process_bracket_tax_params(cached_params, p)
            print("  [CACHE INVALID] Legacy cache incompatible, re-estimating...")

        # Get micro data ONCE (without age bracket mapping) - uses its own cache
        print("Getting micro data from PolicyEngine-UK...")
        micro_data, taxcalc_version = get_micro_data.get_data(
            baseline=p.baseline,
            start_year=p.start_year,
            reform=pit_reform,
            data=data,
            path=p.output_base,
            client=client,
            num_workers=num_workers,
            age_brackets=None,  # Don't apply brackets yet
            use_cache=True,  # Enable micro data caching
        )
        p.BW = len(micro_data)

        # Estimate tax functions for each bracket
        bracket_params = []
        mean_income_data = None
        frac_tax_payroll = None

        for i, (age_min, age_max, rep_age) in enumerate(self.age_brackets):
            print(
                f"\n--- Estimating bracket {i+1}/{len(self.age_brackets)}: ages {age_min}-{age_max} ---"
            )

            # Filter micro data to this bracket's ages
            bracket_data = filter_micro_data_by_age_bracket(
                micro_data, age_min, age_max
            )

            if not bracket_data:
                print(
                    f"  WARNING: No data for bracket {age_min}-{age_max}, using previous bracket"
                )
                if bracket_params:
                    bracket_params.append(bracket_params[-1])
                continue

            # Count observations
            total_obs = sum(len(df) for df in bracket_data.values())
            print(f"  Observations in bracket: {total_obs}")

            # Estimate with age_specific=False (pool all ages in bracket)
            dict_params = txfunc.tax_func_estimate(
                bracket_data,
                p.BW,
                1,  # S=1 since we're estimating one function per bracket
                p.starting_age,
                p.ending_age,
                start_year=p.start_year,
                analytical_mtrs=p.analytical_mtrs,
                tax_func_type=p.tax_func_type,
                age_specific=False,  # Pool all ages in this bracket
                client=client,
                num_workers=num_workers,
                tax_func_path=None,  # Don't save individual bracket results
            )

            bracket_params.append(
                {
                    "age_min": age_min,
                    "age_max": age_max,
                    "rep_age": rep_age,
                    "etr": dict_params["tfunc_etr_params_S"],
                    "mtrx": dict_params["tfunc_mtrx_params_S"],
                    "mtry": dict_params["tfunc_mtry_params_S"],
                }
            )

            # Use first bracket's average income and payroll fraction
            if mean_income_data is None:
                mean_income_data = dict_params["tfunc_avginc"][0]
                frac_tax_payroll = np.append(
                    dict_params["tfunc_frac_tax_payroll"],
                    np.ones(p.T + p.S - p.BW)
                    * dict_params["tfunc_frac_tax_payroll"][-1],
                )

        print(f"\n--- Replicating bracket params to {p.S} model ages ---")

        # Replicate bracket params to all ages in each bracket
        # Final shape: T x S (list of lists)
        etr_params = [[None for _ in range(p.S)] for _ in range(p.T)]
        mtrx_params = [[None for _ in range(p.S)] for _ in range(p.T)]
        mtry_params = [[None for _ in range(p.S)] for _ in range(p.T)]

        for bracket in bracket_params:
            age_min = bracket["age_min"]
            age_max = bracket["age_max"]

            # Map ages to model indices (s = age - starting_age - 1)
            for age in range(age_min, age_max + 1):
                s = age - p.starting_age - 1  # Model age index (0 to S-1)
                if 0 <= s < p.S:
                    for t in range(p.T):
                        if t < p.BW:
                            # Use bracket's estimated params (index 0 since S=1)
                            etr_params[t][s] = bracket["etr"][t][0]
                            mtrx_params[t][s] = bracket["mtrx"][t][0]
                            mtry_params[t][s] = bracket["mtry"][t][0]
                        else:
                            # Extrapolate using last year
                            etr_params[t][s] = bracket["etr"][-1][0]
                            mtrx_params[t][s] = bracket["mtrx"][-1][0]
                            mtry_params[t][s] = bracket["mtry"][-1][0]

        # Handle any remaining None values (ages outside all brackets)
        for t in range(p.T):
            for s in range(p.S):
                if etr_params[t][s] is None:
                    # Find nearest bracket
                    age = s + p.starting_age + 1
                    if age < self.age_brackets[0][0]:
                        # Use first bracket
                        etr_params[t][s] = bracket_params[0]["etr"][
                            min(t, p.BW - 1)
                        ][0]
                        mtrx_params[t][s] = bracket_params[0]["mtrx"][
                            min(t, p.BW - 1)
                        ][0]
                        mtry_params[t][s] = bracket_params[0]["mtry"][
                            min(t, p.BW - 1)
                        ][0]
                    else:
                        # Use last bracket
                        etr_params[t][s] = bracket_params[-1]["etr"][
                            min(t, p.BW - 1)
                        ][0]
                        mtrx_params[t][s] = bracket_params[-1]["mtrx"][
                            min(t, p.BW - 1)
                        ][0]
                        mtry_params[t][s] = bracket_params[-1]["mtry"][
                            min(t, p.BW - 1)
                        ][0]

        tax_param_dict = {
            "etr_params": etr_params,
            "mtrx_params": mtrx_params,
            "mtry_params": mtry_params,
            "taxcalc_version": taxcalc_version,
            "mean_income_data": mean_income_data,
            "frac_tax_payroll": frac_tax_payroll,
            "start_year": p.start_year,
            "tax_func_type": p.tax_func_type,
            "BW": p.BW,
            "age_brackets": self.age_brackets,
        }

        # Save to hash-based cache for future runs
        _save_taxfunc_cache(cache_path, tax_param_dict)

        # Also save to legacy path for backwards compatibility
        with open(tax_func_path, "wb") as f:
            pickle.dump(tax_param_dict, f)
        print(f"Saved bracket tax functions to {tax_func_path}")

        print(f"\n{'='*60}")
        print(f"BRACKET ESTIMATION COMPLETE")
        print(f"{'='*60}\n")

        return tax_param_dict

    def _process_bracket_tax_params(self, cached_params, p):
        """
        Process cached bracket tax parameters to ensure they're ready for use.

        This method validates and returns cached tax function parameters,
        updating p.BW if needed.

        Args:
            cached_params (dict): Cached tax function parameters
            p: OG-Core Specifications object

        Returns:
            dict: Processed tax function parameters ready for model use
        """
        # Update BW from cached params if available
        if "BW" in cached_params:
            p.BW = cached_params["BW"]

        return cached_params

    def read_tax_func_estimate(self, p, tax_func_path):
        """
        This function reads in tax function parameters from pickle
        files.

        Args:
            tax_func_path (str): path to pickle with tax function
                parameter estimates

        Returns:
            dict_params (dict): dictionary containing arrays of tax
                function parameters
            run_micro (bool): whether to estimate tax function parameters

        """
        flag = 0
        if os.path.exists(tax_func_path):
            print("Tax Function Path Exists")
            dict_params = safe_read_pickle(tax_func_path)
            # check to see if tax_functions compatible
            try:
                if p.start_year != dict_params["start_year"]:
                    print(
                        "Model start year not consistent with tax "
                        + "function parameter estimates"
                    )
                    flag = 1
            except KeyError:
                pass
            try:
                p.BW = dict_params["BW"]  # QUICK FIX
                if p.BW != dict_params["BW"]:
                    print(
                        "Model budget window length is "
                        + str(p.BW)
                        + " but the tax function parameter "
                        + "estimates have a budget window length of "
                        + str(dict_params["BW"])
                    )
                    flag = 1
            except KeyError:
                pass
            try:
                if p.tax_func_type != dict_params["tax_func_type"]:
                    print(
                        "Model tax function type is not "
                        + "consistent with tax function parameter "
                        + "estimates"
                    )
                    flag = 1
            except KeyError:
                pass
            if flag >= 1:
                raise RuntimeError(
                    "Tax function parameter estimates at given path"
                    + " are not consistent with model parameters"
                    + " specified."
                )
        else:
            flag = 1
            print(
                "Tax function parameter estimates do not exist at"
                + " given path. Running new estimation."
            )
        if flag >= 1:
            dict_params = None
            run_micro = True
        else:
            run_micro = False

        return dict_params, run_micro

    # method to return all newly calibrated parameters in a dictionary
    def get_dict(self):
        dict = {}
        if self.estimate_tax_functions:
            dict.update(self.tax_function_params)
        # if self.estimate_beta:
        #     dict["beta_annual"] = self.beta
        # if self.estimate_chi_n:
        #     dict["chi_n"] = self.chi_n
        # dict["eta"] = self.eta
        # dict["zeta"] = self.zeta
        # dict.update(self.macro_params)
        # dict["e"] = self.e
        dict.update(self.demographic_params)

        return dict
