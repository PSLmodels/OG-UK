"""Tests for the 8-sector industry calibration in oguk.industry_params."""

import numpy as np
import pytest

from oguk.industry_params import (
    _CAPITAL_STOCK,
    _EPSILON,
    _GAMMA,
    _WORKFORCE_JOBS,
    M,
    _sector_gva,
    _sector_tfp,
    get_industry_params,
)

# The calibration as actually shipped by get_industry_params().
_GAMMA_SHRUNK = [0.35 + 0.6 * (g - 0.35) for g in _GAMMA]


def _z(k_factor=1.0, l_factor=1.0):
    return np.array(
        _sector_tfp(
            epsilon=list(_EPSILON),
            gamma=_GAMMA_SHRUNK,
            capital=[k * k_factor for k in _CAPITAL_STOCK],
            labour=[n * l_factor for n in _WORKFORCE_JOBS],
        )
    )


@pytest.mark.parametrize(
    "k_factor,l_factor",
    [
        (1e6, 1.0),  # K in £ rather than £m
        (1e-3, 1.0),  # K in £bn
        (1.0, 1e3),  # L in jobs rather than thousands of jobs
        (1.0, 137.0),  # arbitrary
        (1e6, 1e3),  # both, together
        (0.017, 4200.0),  # both, arbitrary
    ],
)
def test_sector_tfp_is_unit_invariant(k_factor, l_factor):
    """Z must not depend on the units K and L are measured in.

    Under CES with epsilon != 1 the capital and labour terms are *summed*
    inside the aggregator, so if K and L entered in their raw ONS units
    (£m and thousands of jobs) their measurement scale would set their
    relative weight and the Z dispersion would be an artefact. Rescaling
    either input by a positive constant must leave Z exactly unchanged.
    """
    np.testing.assert_allclose(
        _z(k_factor=k_factor, l_factor=l_factor), _z(), rtol=1e-12, atol=0
    )


def test_sector_tfp_gva_weighted_mean_is_one():
    gva = _sector_gva()
    gva_shares = gva / gva.sum()
    assert np.isclose(np.dot(gva_shares, _z()), 1.0)


def test_get_industry_params_z_matches_sector_tfp():
    z = np.array(get_industry_params()["Z"][0])
    assert z.shape == (M,)
    assert np.all(z > 0)
    np.testing.assert_allclose(z, _z(), rtol=1e-12, atol=0)
