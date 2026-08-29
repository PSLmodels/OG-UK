"""Tests for the interior resource-constraint check.

``_build_specs`` sets ``RC_TPI = 0.2`` so TPI can complete despite a known
boundary discontinuity at the terminal period. OG-Core applies that tolerance
to every period at once, so these tests pin down that an interior violation
of the same magnitude is still caught.
"""

import numpy as np
import pytest

from oguk.api import (
    INTERIOR_RC_TOL,
    _check_interior_resource_constraint,
)


def _rc(values):
    return {"resource_constraint_error": np.array(values, dtype=float)}


def test_terminal_only_violation_passes():
    """A large error confined to the last period is a truncation artifact."""
    rc = np.full(60, 1e-8)
    rc[-1] = 0.109  # the magnitude that motivated RC_TPI = 0.2
    _check_interior_resource_constraint(_rc(rc))


def test_interior_violation_raises():
    """An interior error of the same magnitude must not pass silently."""
    rc = np.full(60, 1e-8)
    rc[10] = 0.109
    rc[-1] = 0.109
    with pytest.raises(RuntimeError, match="period 10"):
        _check_interior_resource_constraint(_rc(rc))


def test_sign_is_ignored():
    """The check is on absolute error."""
    rc = np.full(60, 1e-8)
    rc[3] = -0.5
    with pytest.raises(RuntimeError, match="period 3"):
        _check_interior_resource_constraint(_rc(rc))


def test_interior_within_tolerance_passes():
    rc = np.full(60, INTERIOR_RC_TOL / 10)
    rc[-1] = 0.2
    _check_interior_resource_constraint(_rc(rc))


def test_trailing_axes_are_collapsed():
    """resource_constraint_error may carry trailing axes; period is axis 0."""
    rc = np.full((60, 3), 1e-8)
    rc[7, 2] = 0.4
    with pytest.raises(RuntimeError, match="period 7"):
        _check_interior_resource_constraint(_rc(rc))


def test_label_appears_in_message():
    rc = np.full(10, 1e-8)
    rc[1] = 1.0
    with pytest.raises(RuntimeError, match="reform transition path"):
        _check_interior_resource_constraint(_rc(rc), label="reform")


@pytest.mark.parametrize("missing", [{}, {"resource_constraint_error": None}])
def test_missing_key_is_a_no_op(missing):
    _check_interior_resource_constraint(missing)


def test_single_period_is_a_no_op():
    """With only a terminal period there is no interior to check."""
    _check_interior_resource_constraint(_rc([0.5]))
