"""Tests for the tax-function estimation flags (issue #80).

These tests deliberately avoid ``calibrate()`` and ``_build_specs()`` — both
run a live PolicyEngine calibration and take minutes. They exercise the
pieces: flag plumbing/defaults, the payroll split, the liability columns, and
the GS estimation helpers on a small synthetic sample.
"""

import inspect
import tempfile

import numpy as np
import pandas as pd
import pytest

from oguk import api


def _synthetic_micro(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """A synthetic estimation frame with DIFFERENT labour and capital MTRs.

    Labour income is progressive with a high top marginal rate (NICs + higher
    rate); capital income faces a distinctly lower marginal schedule (dividend
    rates, savings allowances). The ETR sits between the two.
    """
    rng = np.random.default_rng(seed)
    labinc = rng.uniform(6_000, 120_000, size=n)
    capinc = rng.uniform(0, 20_000, size=n)
    income = labinc + capinc
    # Marginal schedules: labour clearly steeper than capital.
    mtr_lab = np.clip(0.32 + 0.25 * (income / 120_000), 0.05, 0.6)
    mtr_cap = np.clip(0.09 + 0.08 * (income / 120_000), 0.02, 0.4)
    etr = np.clip(0.15 + 0.15 * (income / 120_000), 0.01, 0.5)
    income_tax = 0.7 * etr * income
    nics = 0.3 * etr * income
    return pd.DataFrame(
        {
            "mtr_labinc": mtr_lab,
            "mtr_capinc": mtr_cap,
            "etr": etr,
            "age": rng.integers(20, 80, size=n),
            "total_labinc": labinc,
            "total_capinc": capinc,
            "market_income": income,
            "total_tax_liab": income_tax + nics,
            "payroll_tax_liab": nics,
            "year": np.full(n, 2026),
            "weight": rng.uniform(50, 500, size=n),
        }
    )


class _FakeMicroData:
    """Duck-typed stand-in for api._MicroData (no engine run required)."""

    def __init__(self):
        self.labor_income = np.array([30_000.0, 60_000.0, 0.0])
        self.capital_income = np.array([0.0, 10_000.0, 5_000.0])
        self.etr = np.array([0.2, 0.3, 0.1])
        self.income_tax = np.array([3_500.0, 15_000.0, 200.0])
        self.national_insurance = np.array([2_000.0, 4_000.0, 0.0])
        self.age = np.array([30, 45, 70])


# --- flag plumbing / defaults ------------------------------------------


@pytest.mark.parametrize(
    "func",
    [api.calibrate, api._build_specs, api.solve_steady_state, api.run_transition_path],
)
def test_flags_present_and_default_to_current_behaviour(func):
    params = inspect.signature(func).parameters
    for flag in ("estimate_mtrs", "separate_payroll"):
        assert flag in params, f"{func.__name__} is missing {flag}"
        assert params[flag].default is False, (
            f"{func.__name__}.{flag} must default False"
        )


def test_estimators_accept_the_flag_and_default_off():
    for func in (api._estimate_tax_functions, api._estimate_bracket_tax_functions):
        params = inspect.signature(func).parameters
        assert params["estimate_mtrs"].default is False


# --- liability columns / payroll split ---------------------------------


def test_liability_columns_default_reproduces_zero_payroll():
    md = _FakeMicroData()
    total, payroll = api._liability_columns(md, separate_payroll=False)
    assert np.all(payroll == 0.0)
    np.testing.assert_allclose(total, md.etr * (md.labor_income + md.capital_income))


def test_liability_columns_separate_payroll_uses_engine_columns():
    md = _FakeMicroData()
    total, payroll = api._liability_columns(md, separate_payroll=True)
    np.testing.assert_allclose(payroll, md.national_insurance)
    np.testing.assert_allclose(total, md.income_tax + md.national_insurance)
    assert payroll.sum() > 0


def test_payroll_split_is_computed_not_hardcoded():
    data = _synthetic_micro()
    frac = api._payroll_split(data)
    expected = (data["payroll_tax_liab"] * data["weight"]).sum() / (
        data["total_tax_liab"] * data["weight"]
    ).sum()
    assert frac == pytest.approx(expected)
    assert 0.0 < frac < 1.0


def test_payroll_split_is_zero_under_the_default_columns():
    """With the default (zeros) payroll column the split must stay 0."""
    md = _FakeMicroData()
    total, payroll = api._liability_columns(md, separate_payroll=False)
    data = pd.DataFrame(
        {
            "total_tax_liab": total,
            "payroll_tax_liab": payroll,
            "weight": np.ones(len(total)),
        }
    )
    assert api._payroll_split(data) == 0.0


def test_payroll_split_handles_zero_total_tax():
    data = pd.DataFrame(
        {
            "total_tax_liab": [0.0, 0.0],
            "payroll_tax_liab": [0.0, 0.0],
            "weight": [1.0, 1.0],
        }
    )
    assert api._payroll_split(data) == 0.0


# --- estimation ---------------------------------------------------------


def test_fit_gs_triple_reuses_etr_params_by_default():
    df_clean = api._clean_tax_data(_synthetic_micro())
    with tempfile.TemporaryDirectory() as out:
        etr, mtrx, mtry = api._fit_gs_triple(df_clean, out, estimate_mtrs=False)
    np.testing.assert_array_equal(etr, mtrx)
    np.testing.assert_array_equal(etr, mtry)


def test_fit_gs_triple_estimates_distinct_mtr_params():
    """The point of change (1): labour and capital diverge."""
    df_clean = api._clean_tax_data(_synthetic_micro())
    with tempfile.TemporaryDirectory() as out:
        etr, mtrx, mtry = api._fit_gs_triple(df_clean, out, estimate_mtrs=True)
    assert not np.allclose(mtrx, mtry), (mtrx, mtry)
    assert not np.allclose(etr, mtrx)


def test_estimate_tax_functions_pooled_flag_off_and_on():
    data = _synthetic_micro()
    S = 3
    e0, mx0, my0, avg0, fp0 = api._estimate_tax_functions(data, S, estimate_mtrs=False)
    assert len(e0[0]) == len(mx0[0]) == len(my0[0]) == S
    np.testing.assert_array_equal(mx0[0][0], my0[0][0])
    assert avg0 == pytest.approx(
        (data["market_income"] * data["weight"]).sum() / data["weight"].sum()
    )
    assert fp0 == pytest.approx(api._payroll_split(data))

    e1, mx1, my1, _, _ = api._estimate_tax_functions(data, S, estimate_mtrs=True)
    np.testing.assert_array_equal(e0[0][0], e1[0][0])  # ETR fit unchanged
    assert not np.allclose(mx1[0][0], my1[0][0])


def test_estimate_bracket_tax_functions_flag_on_diverges():
    data = _synthetic_micro()
    brackets = [(20, 49, "young"), (50, 100, "old")]
    S = 4
    _, mx, my, _, _ = api._estimate_bracket_tax_functions(
        data, S, brackets, estimate_mtrs=True
    )
    assert len(mx[0]) == len(my[0]) == S
    assert not np.allclose(mx[0][0], my[0][0])

    _, mx0, my0, _, _ = api._estimate_bracket_tax_functions(
        data, S, brackets, estimate_mtrs=False
    )
    np.testing.assert_array_equal(mx0[0][0], my0[0][0])
