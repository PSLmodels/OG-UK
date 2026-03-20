"""OG-UK: United Kingdom calibration for OG-Core."""

from oguk.api import (
    AGE_BRACKETS,
    CalibrationResult,
    MacroImpact,
    SteadyStateResult,
    TransitionMacroImpact,
    TransitionPathResult,
    calibrate,
    map_to_real_world,
    map_transition_to_real_world,
    run_transition_path,
    solve_steady_state,
)
from oguk.industry_params import (
    ENERGY,
    ENERGY_COST_SHARES,
    SECTOR_NAMES,
    SECTOR_SIC_SECTIONS,
    M,
    get_industry_params,
)

__version__ = "0.3.1"
__all__ = [
    "AGE_BRACKETS",
    "CalibrationResult",
    "ENERGY",
    "ENERGY_COST_SHARES",
    "M",
    "MacroImpact",
    "SECTOR_NAMES",
    "SECTOR_SIC_SECTIONS",
    "SteadyStateResult",
    "TransitionMacroImpact",
    "TransitionPathResult",
    "calibrate",
    "get_industry_params",
    "map_to_real_world",
    "map_transition_to_real_world",
    "run_transition_path",
    "solve_steady_state",
]
