"""Tests for OG-UK calibration API."""

import os
from datetime import datetime

import pytest
from policyengine.core import ParameterValue, Policy
from policyengine.tax_benefit_models.uk import uk_latest

from oguk import CalibrationResult, calibrate


# The UK microdata lives in a private Hugging Face repo. Same-repo CI has the
# token secret; FORK pull requests do not (GitHub withholds secrets), and
# contributors may not have access either — so data-dependent tests skip
# cleanly without a token instead of failing on a 401.
def _hf_token_present() -> bool:
    """True when any Hugging Face credential is available — env vars or the
    hub's stored login (huggingface-cli login writes a token file that the
    hub uses regardless of the environment)."""
    if os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


requires_uk_microdata = pytest.mark.skipif(
    not _hf_token_present(),
    reason="needs a Hugging Face token with access to the private UK microdata",
)


@requires_uk_microdata
def test_baseline_calibration():
    """Test baseline calibration produces valid results."""
    result = calibrate(start_year=2026, years=1)

    assert isinstance(result, CalibrationResult)
    assert result.mean_income > 0
    assert len(result.etr_params) == 1
    assert len(result.omega_SS) > 0


@requires_uk_microdata
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


@requires_uk_microdata
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


def test_resolve_year_dataset_prefers_calibrated_stems():
    """2.88's default ensure_datasets returned BOTH frs_* and
    enhanced_frs_* for each year — the calibrated stem must win, and
    populace_uk_* outranks both."""
    from oguk.api import _resolve_year_dataset

    legacy_pair = {"frs_2023_24_2026": "raw", "enhanced_frs_2023_24_2026": "x"}
    assert _resolve_year_dataset(legacy_pair, 2026) == "x"
    mixed = {"populace_uk_2023_2026": "a", "enhanced_frs_2023_24_2026": "x"}
    assert _resolve_year_dataset(mixed, 2026) == "a"


def test_resolve_year_dataset_ambiguous_within_stem_refused():
    from oguk.api import _resolve_year_dataset

    with pytest.raises(KeyError, match="ambiguous"):
        _resolve_year_dataset(
            {"populace_uk_2023_2026": "a", "populace_uk_2024_2026": "b"}, 2026
        )


@requires_uk_microdata
def test_person_level_mtrs_discriminate_household_structure():
    """The two historical failure modes, pinned separately:

    (1) no household normalisation -> multi-adult households clip to a
        0.000 MTR (80% of mid-band earners measured);
    (2) household-AVERAGING -> every adult gets the household mean (a
        GBP 30k / GBP 0 couple both read ~0.14).

    Person-level rates must show mid-band earners near the statutory
    ~28% in BOTH single- and multi-adult households, while zero-earning
    partners in those same households stay near zero.
    """
    import tempfile

    import numpy as np

    from oguk.api import _get_micro_data

    with tempfile.TemporaryDirectory() as tmp:
        md = _get_micro_data(2026, None, tmp)
    # _MicroData has no household ids, so rebuild bands from income alone:
    mid = (md.labor_income > 25_000) & (md.labor_income < 35_000)
    med_mid = float(np.median(md.mtr_labor[mid]))
    assert 0.20 < med_mid < 0.45, f"mid-band median {med_mid:.3f}"
    # THE household-structure discriminator is the mid-band LOWER TAIL:
    # under household-averaging, a GBP 30k earner married to a non-earner
    # reads ~0.14, dragging q10 to ~0.14 (measured); person-level rates
    # put every mid-band earner near the statutory ~0.28 (q10 measured
    # 0.277). Under the original no-normalisation clipping, the median
    # itself was 0.000.
    q10_mid = float(np.percentile(md.mtr_labor[mid], 10))
    assert q10_mid > 0.20, (
        f"mid-band q10 {q10_mid:.3f} — household averaging is back "
        "(earners in multi-adult households diluted toward the mean)"
    )
    # Low-EARNINGS adults are not low-INCOME (pensioners' marginal rate on
    # GBP 1 of earnings is legitimately ~20%), so no near-zero assertion
    # there — but averaging would also DILUTE their rates toward household
    # means; sanity-bound the population mean instead.
    assert 0.15 < float(md.mtr_labor.mean()) < 0.40
    # Capital: the GBP 1 dividend perturbation sits inside the GBP 500
    # dividend allowance for anyone without existing dividends, so a zero
    # MEDIAN among broad capital-income holders (mostly pension income) is
    # correct; require instead that a real mass faces positive dividend
    # marginal rates (8.75%/33.75% bands).
    divs = md.capital_income > 2_000
    if divs.sum() > 1_000:
        positive_share = float(np.mean(md.mtr_capital[divs] > 0.05))
        assert positive_share > 0.10, (
            f"only {positive_share:.0%} of capital-income holders face a "
            "positive marginal dividend rate — perturbation likely flat"
        )
        assert 0.02 < float(md.mtr_capital.mean()) < 0.30


@requires_uk_microdata
def test_labor_mtr_bites_for_midband_earners():
    """Regression for the silent flat-MTR failure on policyengine-uk >= 2.89
    (employment income moved to employment_income_before_lsr, so perturbing
    employment_income directly stopped reaching the tax pipeline: ~80% of
    earners showed a zero labour MTR). A £25-35k earner's marginal rate is
    ~28% (basic-rate income tax + employee NI); the median across that band
    must land near it."""
    import tempfile

    import numpy as np

    from oguk.api import _get_micro_data

    with tempfile.TemporaryDirectory() as tmp:
        md = _get_micro_data(2026, None, tmp)
    mask = (md.labor_income > 25_000) & (md.labor_income < 35_000)
    median_mtr = float(np.median(md.mtr_labor[mask]))
    assert 0.20 < median_mtr < 0.45, (
        f"mid-band labour MTR median {median_mtr:.3f} — the perturbation "
        "is not reaching the tax pipeline"
    )
