from pypsa_battery_degradation.degradation import (
    DegradationResult,
    compute_calendar_aging,
    compute_cycle_aging,
    compute_total_degradation,
    nonlinear_life_loss,
)

from pypsa_battery_degradation.rainflow_analysis import extract_cycles_from_soc

__all__ = [
    "DegradationResult",
    "compute_calendar_aging",
    "compute_cycle_aging",
    "compute_total_degradation",
    "nonlinear_life_loss",
]