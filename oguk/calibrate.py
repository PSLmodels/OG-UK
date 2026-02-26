"""
OG-UK Calibration class.

Provides a Calibration object with a consistent API across OG-Core
country calibrations (see OG-ZAF, OG-PHL, OG-IDN, OG-ETH for reference).

The Calibration class wraps the functional calibrate() API in oguk.api
and stores results as attributes for downstream use in ogcore.
"""

from oguk import api, demographics


class Calibration:
    """OG-UK calibration class.

    Estimates tax functions from PolicyEngine-UK microdata and fetches
    demographic parameters for the UK, providing a dictionary of
    calibrated parameters compatible with ogcore.parameters.Specifications.

    Args:
        p (ogcore.parameters.Specifications): model parameters object
        estimate_tax_functions (bool): whether to estimate tax functions
            from PolicyEngine-UK microdata
        estimate_beta (bool): whether to estimate the discount factor
            (not yet implemented)
        estimate_chi_n (bool): whether to estimate labour supply
            disutility parameters (not yet implemented)
        tax_func_path (str or None): path to cached tax function pickle
        iit_reform (PolicyEngine Policy or None): reform policy to apply
        guid (str): unique ID appended to output file names
        data (str): dataset identifier (unused, for API compatibility)
        client: Dask client for parallelisation
        num_workers (int): number of Dask workers
    """

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
    ):
        self.estimate_tax_functions = estimate_tax_functions
        self.estimate_beta = estimate_beta
        self.estimate_chi_n = estimate_chi_n

        # tax_func_path, guid, data, client, num_workers are accepted for
        # API compatibility with other OG-Core country calibrations but are
        # not yet used in OG-UK's implementation.
        _ = (tax_func_path, guid, data, client, num_workers)

        if estimate_tax_functions:
            cal = api.calibrate(
                start_year=p.start_year,
                policy=iit_reform,
            )
            self.tax_function_params = {
                "etr_params": cal.etr_params,
                "mtrx_params": cal.mtrx_params,
                "mtry_params": cal.mtry_params,
                "mean_income_data": cal.mean_income,
                "frac_tax_payroll": cal.frac_tax_payroll,
                "taxcalc_version": "PolicyEngine-UK",
            }

        # Demographics
        self.demographic_params = demographics.get_pop_objs(
            p.E, p.S, p.T, p.start_year
        )

    def get_dict(self):
        """Return all calibrated parameters as a dictionary.

        Returns:
            dict: calibrated parameters ready to pass to
                Specifications.update_specifications()
        """
        result = {}
        if self.estimate_tax_functions:
            result.update(self.tax_function_params)
        result.update(self.demographic_params)
        return result
