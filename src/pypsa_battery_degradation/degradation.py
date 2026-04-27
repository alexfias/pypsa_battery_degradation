from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pypsa_battery_degradation.stress_factors import (
    dod_stress,
    soc_stress,
    temperature_stress,
    time_stress,
)


ALPHA_SEI = 5.75e-2
BETA_SEI = 121.0


@dataclass
class DegradationResult:
    calendar_aging: float
    cycle_aging: float
    total_stress: float
    life_loss: float


def nonlinear_life_loss(total_stress: float) -> float:
    """
    Xu et al. nonlinear degradation mapping including SEI formation.
    """
    return float(
        1
        - ALPHA_SEI * np.exp(-BETA_SEI * total_stress)
        - (1 - ALPHA_SEI) * np.exp(-total_stress)
    )


def compute_cycle_aging(cycles: pd.DataFrame) -> float:
    """
    Compute cycle aging from rainflow-counted cycles.

    Expected columns:
    - dod: depth of discharge, normalized 0-1
    - mean_soc: average SOC during cycle, normalized 0-1
    - mean_temperature_kelvin: average temperature during cycle in K
    - count: rainflow count, usually 0.5 or 1.0
    """
    if cycles.empty:
        return 0.0

    required = {"dod", "mean_soc", "mean_temperature_kelvin", "count"}
    missing = required - set(cycles.columns)
    if missing:
        raise ValueError(f"Missing required cycle columns: {missing}")

    cycle_stress = (
        dod_stress(cycles["dod"])
        * soc_stress(cycles["mean_soc"])
        * temperature_stress(cycles["mean_temperature_kelvin"])
    )

    return float(np.sum(cycles["count"] * cycle_stress))


def compute_calendar_aging(
    soc: pd.Series,
    temperature_kelvin: pd.Series,
    timestep_seconds: float = 3600.0,
) -> float:
    """
    Compute calendar aging from SOC and temperature time series.
    """
    if len(soc) != len(temperature_kelvin):
        raise ValueError("SOC and temperature series must have the same length.")

    total_seconds = len(soc) * timestep_seconds
    mean_soc = float(soc.mean())
    mean_temperature = float(temperature_kelvin.mean())

    return float(
        time_stress(total_seconds)
        * soc_stress(mean_soc)
        * temperature_stress(mean_temperature)
    )


def compute_total_degradation(
    soc: pd.Series,
    temperature_kelvin: pd.Series,
    cycles: pd.DataFrame,
    timestep_seconds: float = 3600.0,
) -> DegradationResult:
    calendar = compute_calendar_aging(
        soc=soc,
        temperature_kelvin=temperature_kelvin,
        timestep_seconds=timestep_seconds,
    )

    cycle = compute_cycle_aging(cycles)

    total_stress = calendar + cycle
    life_loss = nonlinear_life_loss(total_stress)

    return DegradationResult(
        calendar_aging=calendar,
        cycle_aging=cycle,
        total_stress=total_stress,
        life_loss=life_loss,
    )