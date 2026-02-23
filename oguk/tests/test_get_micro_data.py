"""Tests for OG-UK calibration API."""

import pytest
from datetime import datetime
from policyengine.core import Policy, ParameterValue
from policyengine.tax_benefit_models.uk import uk_latest

from oguk import calibrate, CalibrationResult


def test_baseline_calibration():
    """Test baseline calibration produces valid results."""
    result = calibrate(start_year=2026, years=1)

    assert isinstance(result, CalibrationResult)
    assert result.mean_income > 0
    assert len(result.etr_params) == 1
    assert len(result.omega_SS) > 0


def test_reform_calibration():
    """Test calibration with a policy reform."""
    pa_param = uk_latest.get_parameter(
        "gov.hmrc.income_tax.allowances.personal_allowance.amount"
    )
    reform = Policy(
        name="Lower PA",
        parameter_values=[
            ParameterValue(
                parameter=pa_param,
                value=10000,
                start_date=datetime(2026, 1, 1),
            )
        ],
    )

    result = calibrate(start_year=2026, years=1, policy=reform)

    assert isinstance(result, CalibrationResult)
    assert result.mean_income > 0


def test_demographic_outputs():
    """Test demographic parameters are valid."""
    result = calibrate(start_year=2026, years=1)

    # Population growth should be reasonable
    assert -0.05 < result.g_n_ss < 0.05

    # Mortality rates should be probabilities
    assert result.rho.min() >= 0
    assert result.rho.max() <= 1

    # Population shares should sum to 1
    assert abs(result.omega_SS.sum() - 1.0) < 0.01
