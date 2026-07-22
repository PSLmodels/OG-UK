"""Tests for OG-UK calibration API."""

from datetime import datetime

from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest

from oguk import CalibrationResult, calibrate


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


# --- dataset-key resolution across policyengine-uk vintages (issue #68) ---

def test_resolve_year_dataset_populace_keys():
    from oguk.api import _resolve_year_dataset

    ds = {"populace_uk_2023_2026": "a", "populace_uk_2023_2027": "b"}
    assert _resolve_year_dataset(ds, 2026) == "a"
    assert _resolve_year_dataset(ds, 2027) == "b"


def test_resolve_year_dataset_legacy_enhanced_frs_keys():
    from oguk.api import _resolve_year_dataset

    ds = {"enhanced_frs_2023_24_2026": "x"}
    assert _resolve_year_dataset(ds, 2026) == "x"


def test_resolve_year_dataset_missing_year_lists_available():
    import pytest

    from oguk.api import _resolve_year_dataset

    with pytest.raises(KeyError, match="available keys"):
        _resolve_year_dataset({"populace_uk_2023_2027": "b"}, 2026)


def test_resolve_year_dataset_ambiguous_stems_refused():
    import pytest

    from oguk.api import _resolve_year_dataset

    with pytest.raises(KeyError, match="ambiguous"):
        _resolve_year_dataset(
            {"populace_uk_2023_2026": "a", "enhanced_frs_2023_24_2026": "x"},
            2026,
        )



def test_labor_mtr_bites_for_midband_earners():
    """Regression for the silent flat-MTR failure on policyengine-uk >= 2.89
    (employment income moved to employment_income_before_lsr, so perturbing
    employment_income directly stopped reaching the tax pipeline: ~80% of
    earners showed a zero labour MTR). A £25-35k earner's marginal rate is
    ~28% (basic-rate income tax + employee NI); the median across that band
    must land near it."""
    import numpy as np

    from oguk.api import _get_micro_data

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        md = _get_micro_data(2026, None, tmp)
    mask = (md.labor_income > 25_000) & (md.labor_income < 35_000)
    median_mtr = float(np.median(md.mtr_labor[mask]))
    assert 0.20 < median_mtr < 0.45, (
        f"mid-band labour MTR median {median_mtr:.3f} — the perturbation "
        "is not reaching the tax pipeline"
    )
