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

__version__ = "0.3.0"
__all__ = [
    "AGE_BRACKETS",
    "CalibrationResult",
    "MacroImpact",
    "SteadyStateResult",
    "TransitionMacroImpact",
    "TransitionPathResult",
    "calibrate",
    "map_to_real_world",
    "map_transition_to_real_world",
    "run_transition_path",
    "solve_steady_state",
]
