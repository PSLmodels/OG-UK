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


@pytest.fixture()
def _headless_un_token():
    """ogcore.demographics.get_un_data prompts on stdin for a UN API token
    unless un_api_token.txt exists in the CWD; under pytest the prompt
    raises (stdin is captured, and ogcore catches only EOFError). Pre-seed
    an empty token: the UN API then returns 401 and ogcore falls back to
    its public GitHub population-data mirror, which needs no auth. The
    file is removed only if this fixture created it."""
    import pathlib

    path = pathlib.Path("un_api_token.txt")
    created = not path.exists()
    if created:
        path.write_text("")
    yield
    if created:
        path.unlink(missing_ok=True)


@requires_uk_microdata
def test_baseline_calibration(_headless_un_token):
    """Test baseline calibration produces valid results."""
    result = calibrate(start_year=2026, years=1)

    assert isinstance(result, CalibrationResult)
    assert result.mean_income > 0
    assert len(result.etr_params) == 1
    assert len(result.omega_SS) > 0


@requires_uk_microdata
def test_reform_calibration(_headless_un_token):
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

    result = calibrate(start_year=2026, years=2, policy=reform)

    assert isinstance(result, CalibrationResult)
    assert result.mean_income > 0
    # multi-year: one fitted ETR set per year, both under the reform
    assert len(result.etr_params) == 2


@requires_uk_microdata
def test_demographic_outputs(_headless_un_token):
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


def _extract_or_skip(year, policy=None):
    """Run the microdata extraction, skipping (not failing) when the private
    UK dataset is unreachable from this environment. Attempt-based, because
    availability is more than env vars: a huggingface-cli login or a warm
    hub cache also works."""
    import tempfile

    from oguk.api import _get_micro_data

    try:
        with tempfile.TemporaryDirectory() as tmp:
            return _get_micro_data(year, policy, tmp)
    except Exception as e:  # noqa: BLE001 — availability gate
        marker = f"{type(e).__name__}: {e}"
        if any(
            s in marker
            for s in (
                "401",
                "Unauthorized",
                "RepositoryNotFound",
                "GatedRepo",
                "LocalEntryNotFound",
            )
        ):
            pytest.skip(f"UK microdata unavailable here: {marker[:200]}")
        raise


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
    import numpy as np

    md = _extract_or_skip(2026)
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
    assert divs.sum() > 1_000, (
        f"only {int(divs.sum())} capital-income holders in the sample — "
        "the capital assertions below would be vacuous"
    )
    positive_share = float(np.mean(md.mtr_capital[divs] > 0.05))
    assert positive_share > 0.10, (
        f"only {positive_share:.0%} of capital-income holders face a "
        "positive marginal dividend rate — perturbation likely flat"
    )
    assert 0.02 < float(md.mtr_capital.mean()) < 0.30


def test_labor_mtr_bites_for_midband_earners():
    """Regression for silently flat labour MTRs. The originally observed
    failure (~80% of £25-35k earners at a 0.000 MTR) was clipping caused
    by comparing household-level net-income deltas against per-person tax
    deltas without household normalisation — NOT an input-layout change;
    the perturbation reaches the tax pipeline on both 2.88 and 2.89
    layouts via _perturb_first_populated. A mid-band earner's marginal
    rate is ~28% (basic-rate income tax + employee NI); the median across
    the band must land near it."""
    import numpy as np

    md = _extract_or_skip(2026)
    mask = (md.labor_income > 25_000) & (md.labor_income < 35_000)
    median_mtr = float(np.median(md.mtr_labor[mask]))
    assert 0.20 < median_mtr < 0.45, (
        f"mid-band labour MTR median {median_mtr:.3f} — the perturbation "
        "is not reaching the tax pipeline"
    )


def test_reform_modifier_composes_before_perturbation():
    """Parametric reforms are applied INSIDE the perturbation modifier,
    before the income perturbation — pinned with fakes so the ordering
    cannot silently invert (perturbing the baseline world instead of the
    reformed one)."""
    import numpy as np

    from oguk.api import _build_perturbation_modifier

    calls = []

    class _FakeHolder:
        def __init__(self):
            self.arrays = {2026: np.array([1.0, 2.0])}

        def get_known_periods(self):
            return list(self.arrays)

        def get_array(self, period):
            return self.arrays[period]

        def delete_arrays(self, period):
            del self.arrays[period]

    class _FakeSim:
        def __init__(self):
            self.holder = _FakeHolder()
            self.values = None

        def get_holder(self, name):
            return self.holder

        def calculate(self, name, year):
            calls.append(f"calc:{name}")
            return np.array([1.0, 0.0])  # adult, child

        def set_input(self, name, period, values):
            calls.append(f"perturb:{name}")
            self.values = values

    def fake_reform(s):
        calls.append("reform")
        return s

    modifier = _build_perturbation_modifier(fake_reform, 2026, ("employment_income",))
    sim = _FakeSim()
    modifier(sim)
    assert calls[0] == "reform", calls
    assert "perturb:employment_income" in calls, calls
    # GBP 1 added only for the adult member
    assert list(sim.values) == [2.0, 2.0]


def test_perturbation_refuses_silently_flat():
    """No populated candidate input -> hard error, never flat MTRs."""
    import numpy as np

    from oguk.api import _build_perturbation_modifier

    class _EmptyHolder:
        def get_known_periods(self):
            return []

    class _FakeSim:
        def get_holder(self, name):
            return _EmptyHolder()

        def calculate(self, name, year):
            return np.array([1.0])

    modifier = _build_perturbation_modifier(None, 2026, ("employment_income",))
    with pytest.raises(RuntimeError, match="refusing"):
        modifier(_FakeSim())
