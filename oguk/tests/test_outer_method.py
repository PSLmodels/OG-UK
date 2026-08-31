"""The TPI outer-loop method selected for a run.

Anderson acceleration halves the outer iteration count at OG-UK's
production shape; multi-sector runs stay on OG-Core's default damped
Picard, which is the only regime with evidence behind it.
"""

from ogcore.parameters import Specifications

from oguk.api import tpi_outer_method


def test_single_sector_uses_anderson():
    assert tpi_outer_method(False) == "anderson"


def test_multi_sector_stays_on_picard():
    assert tpi_outer_method(True) == "picard"


def test_both_values_are_accepted_by_ogcore():
    """Guard against the parameter being renamed or restricted upstream."""
    p = Specifications()
    for multi_sector in (False, True):
        method = tpi_outer_method(multi_sector)
        p.update_specifications({"TPI_outer_method": method})
        assert p.TPI_outer_method == method


def test_trust_region_guard_is_enabled_by_default():
    """Anderson is only safe with the trust region on; we do not disable it."""
    p = Specifications()
    assert p.TPI_trust_radius > 0
