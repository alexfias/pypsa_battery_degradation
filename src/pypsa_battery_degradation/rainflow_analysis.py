from __future__ import annotations

import numpy as np
import pandas as pd
import rainflow


def extract_cycles_from_soc(
    soc: pd.Series,
    temperature_kelvin: pd.Series,
) -> pd.DataFrame:
    """
    Convert SOC time series into rainflow cycles.

    Returns DataFrame with:
    - dod
    - mean_soc
    - mean_temperature_kelvin
    - count
    """
    if len(soc) != len(temperature_kelvin):
        raise ValueError("SOC and temperature must have same length")

    cycles = []

    # rainflow expects numeric array
    soc_values = soc.values

    for cycle in rainflow.extract_cycles(soc_values):
        # cycle = (range, mean, count, start_idx, end_idx)
        rng, mean, count, start, end = cycle

        dod = abs(rng)  # already normalized if SOC is 0-1
        mean_soc = mean

        # approximate temperature over cycle window
        temp_slice = temperature_kelvin.iloc[int(start): int(end) + 1]
        mean_temp = temp_slice.mean() if len(temp_slice) > 0 else temperature_kelvin.mean()

        cycles.append(
            {
                "dod": float(dod),
                "mean_soc": float(mean_soc),
                "mean_temperature_kelvin": float(mean_temp),
                "count": float(count),
            }
        )

    return pd.DataFrame(cycles)