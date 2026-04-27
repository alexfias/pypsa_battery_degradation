from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pypsa_battery_degradation.degradation import compute_total_degradation
from pypsa_battery_degradation.rainflow_analysis import extract_cycles_from_soc


def _get_storage_soc(network) -> pd.DataFrame:
    """
    Return storage unit state of charge time series.
    """
    if not hasattr(network, "storage_units_t"):
        raise ValueError("Network has no storage_units_t attribute.")

    soc = getattr(network.storage_units_t, "state_of_charge", None)

    if soc is None or soc.empty:
        raise ValueError("No storage unit state_of_charge time series found.")

    return soc


def _get_storage_capacities(network) -> pd.Series:
    """
    Return usable energy capacity for each StorageUnit.

    PyPSA StorageUnit energy capacity is approximated as:
        E_nom = p_nom * max_hours

    If p_nom_opt exists and is nonzero, it is preferred.
    """
    storage_units = network.storage_units

    if "p_nom_opt" in storage_units.columns:
        p_nom = storage_units["p_nom_opt"].where(
            storage_units["p_nom_opt"] > 0,
            storage_units.get("p_nom", 0.0),
        )
    else:
        p_nom = storage_units.get("p_nom", pd.Series(0.0, index=storage_units.index))

    max_hours = storage_units.get(
        "max_hours",
        pd.Series(1.0, index=storage_units.index),
    )

    return p_nom * max_hours


def _normalize_soc(soc: pd.Series, energy_capacity: float) -> pd.Series:
    """
    Normalize SOC to 0-1.

    If capacity is missing or zero, fall back to max observed SOC.
    """
    if energy_capacity and energy_capacity > 0:
        normalized = soc / energy_capacity
    else:
        max_soc = soc.max()
        if max_soc <= 0:
            return pd.Series(0.0, index=soc.index)
        normalized = soc / max_soc

    return normalized.clip(lower=0.0, upper=1.0)


def _get_temperature_series(
    temperature: pd.DataFrame | pd.Series | None,
    bus: str,
    snapshots: pd.Index,
    default_temperature_kelvin: float,
) -> pd.Series:
    """
    Return node-specific temperature series in Kelvin.

    Accepted inputs:
    - None: constant default temperature
    - Series: one common temperature time series
    - DataFrame: columns should be bus/node names
    """
    if temperature is None:
        return pd.Series(default_temperature_kelvin, index=snapshots)

    if isinstance(temperature, pd.Series):
        return temperature.reindex(snapshots).ffill().bfill()

    if isinstance(temperature, pd.DataFrame):
        if bus in temperature.columns:
            return temperature[bus].reindex(snapshots).ffill().bfill()

        warnings.warn(
            f"No temperature column found for bus '{bus}'. "
            f"Using default {default_temperature_kelvin} K.",
            stacklevel=2,
        )
        return pd.Series(default_temperature_kelvin, index=snapshots)

    raise TypeError("temperature must be None, pandas Series, or pandas DataFrame.")


def compute_degradation_from_network(
    network,
    temperature: pd.DataFrame | pd.Series | None = None,
    storage_carrier: str | None = "battery",
    default_temperature_kelvin: float = 298.0,
    timestep_seconds: float = 3600.0,
) -> pd.DataFrame:
    """
    Compute battery degradation for all matching PyPSA StorageUnits.

    Parameters
    ----------
    network:
        Solved PyPSA network.
    temperature:
        Optional temperature time series in Kelvin.
        If DataFrame, columns should correspond to bus/node names.
    storage_carrier:
        If given, only StorageUnits with this carrier are evaluated.
        Set to None to evaluate all StorageUnits.
    default_temperature_kelvin:
        Used when no temperature data is supplied.
    timestep_seconds:
        Usually 3600 for hourly PyPSA-Eur networks.

    Returns
    -------
    pandas.DataFrame
        One row per storage unit with degradation indicators.
    """
    soc_df = _get_storage_soc(network)
    storage_units = network.storage_units.copy()
    energy_capacities = _get_storage_capacities(network)

    if storage_carrier is not None and "carrier" in storage_units.columns:
        storage_units = storage_units[
            storage_units["carrier"].astype(str).str.lower()
            == storage_carrier.lower()
        ]

    results = []

    for storage_name, storage in storage_units.iterrows():
        if storage_name not in soc_df.columns:
            continue

        bus = storage.get("bus", None)
        carrier = storage.get("carrier", None)

        raw_soc = soc_df[storage_name]
        energy_capacity = float(energy_capacities.get(storage_name, np.nan))
        normalized_soc = _normalize_soc(raw_soc, energy_capacity)

        temp_series = _get_temperature_series(
            temperature=temperature,
            bus=bus,
            snapshots=raw_soc.index,
            default_temperature_kelvin=default_temperature_kelvin,
        )

        cycles = extract_cycles_from_soc(
            soc=normalized_soc,
            temperature_kelvin=temp_series,
        )

        degradation = compute_total_degradation(
            soc=normalized_soc,
            temperature_kelvin=temp_series,
            cycles=cycles,
            timestep_seconds=timestep_seconds,
        )

        equivalent_full_cycles = float(
            (cycles["dod"] * cycles["count"]).sum()
            if not cycles.empty
            else 0.0
        )

        results.append(
            {
                "storage_unit": storage_name,
                "bus": bus,
                "carrier": carrier,
                "energy_capacity_mwh": energy_capacity,
                "mean_soc": float(normalized_soc.mean()),
                "mean_temperature_kelvin": float(temp_series.mean()),
                "number_of_cycles": int(len(cycles)),
                "equivalent_full_cycles": equivalent_full_cycles,
                "calendar_aging": degradation.calendar_aging,
                "cycle_aging": degradation.cycle_aging,
                "total_stress": degradation.total_stress,
                "life_loss": degradation.life_loss,
            }
        )

    return pd.DataFrame(results)