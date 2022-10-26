import multiprocessing
from distributed import Client, LocalCluster
import pytest
import os
from oguk import get_micro_data
from oguk.get_micro_data import DATA_LAST_YEAR
from policyenginge_core.model_api import Reform


NUM_WORKERS = min(multiprocessing.cpu_count(), 7)
# get path to puf if puf.csv in ogusa/ directory
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, "..", "puf.csv")


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


def test_frs():
    """
    Check that setting `data` to 'frs' uses frs data
    """
    baseline = False
    start_year = 2022

    # create a parametric reform
    def lower_pa(parameters):
        parameters.tax.income_tax.allowances.personal_allowance.amount.update(
            period="year:2022:10", value=10000
        )
        return parameters

    class lower_personal_tax_allowance(Reform):
        def apply(self):
            self.modify_parameters(lower_pa)

    reform = lower_personal_tax_allowance

    calc_out = get_micro_data.get_calculator_output(
        baseline, start_year, reform=reform, data="frs"
    )
    # check some trivial variable
    assert calc_out["age"].sum() > 0


def test_get_calculator_exception():
    with pytest.raises(Exception):
        assert get_micro_data.get_calculator_output(
            baseline=False, year=DATA_LAST_YEAR + 1, reform=None, data=None
        )


def test_household_mtr_calculation():
    """Test that the household MTR function works as expected"""
    mtr_x = get_micro_data.get_household_mtrs(
        (),
        "employment_income",
        2022,
        dataset=get_micro_data.dataset,
        year=2022,
    )
    assert mtr_x.isna().sum() == 0
    assert mtr_x.min() >= 0
    assert mtr_x.max() <= 1

    mtr_y = get_micro_data.get_household_mtrs(
        (),
        "savings_interest_income",
        2022,
        dataset=get_micro_data.dataset,
        year=2022,
    )
    assert mtr_y.isna().sum() == 0
    assert mtr_y.min() >= 0
    assert mtr_y.max() <= 1
