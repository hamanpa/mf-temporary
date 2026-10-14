"""
This module defines the data structure for storing results from a single neuron simulation.

"""

import numpy as np
import warnings

from .base import BaseSingleNeuronResults
from ..network_params.translators import get_unit_multiplier
from pydantic import BaseModel


def drive_values(drive_rate_grid: np.ndarray) -> np.ndarray:
    """Drive rates [Hz] along the drive axis of a 3D (exc, inh, drive) grid (or the single value of a 2D slice)."""
    if drive_rate_grid.ndim == 2:
        return drive_rate_grid.reshape(-1)[:1]
    values = drive_rate_grid[0, 0, :]
    if not np.allclose(drive_rate_grid, values[np.newaxis, np.newaxis, :]):
        raise ValueError("The drive rate must be constant along the exc and inh axes of the grid.")
    return values


def drive_index(drive_rate_grid: np.ndarray, drive_rate: float | None = None) -> int:
    """
    Index along the drive axis of `drive_rate` [Hz]; None selects the first drive value.
    Raises ValueError listing the available values if `drive_rate` is not in the grid.
    """
    values = drive_values(drive_rate_grid)
    if drive_rate is None:
        return 0
    matches = np.flatnonzero(np.isclose(values, drive_rate))
    if matches.size == 0:
        raise ValueError(f"drive_rate {drive_rate} Hz is not in the grid. Available drive rates [Hz]: {values.tolist()}")
    return int(matches[0])


class SingleNeuronResults(BaseSingleNeuronResults):
    """
    Data structure for storing results from single neuron simulations.

    Internal data is strictly maintained in the default units. Physical quantities
    are accessed via methods (e.g., `results.out_rate_mean(unit="kHz")`).

    All grid arrays are indexed (exc_rate, inh_rate, drive_rate). Arrays given without a
    `drive_rate_grid` (e.g. older data, reference simulators) are treated as drive = 0 and get a
    drive axis of length 1. `at_drive(drive_rate)` returns a 2D (exc_rate, inh_rate) slice.

    STP state at the input spikes (`stp_at_spikes`, unitless): for each projection onto the simulated neuron
    (target-source code with the source e | i | d, e.g. "ed" = onto E from the drive), the mean and std over the
    input spikes in the averaging window of x⁻ (resources before a spike), u⁺ (utilisation used) and u⁺·x⁻
    (efficacy), keys `{projection}_{x|u|ux}_{mean|std}`. Recorded by the PyNN simulator; empty for results
    without it (other simulators, older caches).
    """

    STP_VARIABLES = ("x", "u", "ux")
    STP_STATS = ("mean", "std")

    # Array fields, all with the same (exc_rate, inh_rate, drive_rate) shape
    GRID_FIELDS = (
        "exc_rate_grid", "inh_rate_grid", "drive_rate_grid",
        "out_rate_mean", "out_rate_std",
        "adaptation_mean", "adaptation_std",
        "voltage_mean", "voltage_std", "voltage_tau",
        "exc_conductance_mean", "exc_conductance_std",
        "inh_conductance_mean", "inh_conductance_std",
    )

    DEFAULT_UNITS = {
        "exc_rate_grid": "Hz",
        "inh_rate_grid": "Hz",
        "drive_rate_grid": "Hz",
        "out_rate_mean": "Hz",
        "out_rate_std": "Hz",
        "adaptation_mean": "nA",
        "adaptation_std": "nA",
        "voltage_mean": "mV",
        "voltage_std": "mV",
        "voltage_tau": "ms",
        "exc_conductance_mean": "nS",
        "exc_conductance_std": "nS",
        "inh_conductance_mean": "nS",
        "inh_conductance_std": "nS",
    }

    def __init__(self,
                 simulator_name: str = None,
                 neuron_name: str = None,
                 neuron_params: BaseModel = None,
                 neuron_sim_params: BaseModel = None,
                 spikes: np.ndarray = None,
                 exc_rate_grid: np.ndarray = None,
                 inh_rate_grid: np.ndarray = None,
                 drive_rate_grid: np.ndarray = None,
                 out_rate_mean: np.ndarray = None,
                 out_rate_std: np.ndarray = None,
                 adaptation_mean: np.ndarray = None,
                 adaptation_std: np.ndarray = None,
                 voltage_mean: np.ndarray = None,
                 voltage_std: np.ndarray = None,
                 voltage_tau: np.ndarray = None,
                 exc_conductance_mean: np.ndarray = None,
                 exc_conductance_std: np.ndarray = None,
                 inh_conductance_mean: np.ndarray = None,
                 inh_conductance_std: np.ndarray = None,
                 stp_at_spikes: dict = None,
                 input_units: dict = None):
        
        # --- Public Metadata (No units required) ---
        self.simulator_name = simulator_name
        self.neuron_name = neuron_name
        self.neuron_params = neuron_params
        self.neuron_sim_params = neuron_sim_params
        self.spikes = spikes

        # --- Unit Ingestion Logic ---
        input_units = input_units or {}

        arrays = {
            "exc_rate_grid": exc_rate_grid, "inh_rate_grid": inh_rate_grid, "drive_rate_grid": drive_rate_grid,
            "out_rate_mean": out_rate_mean, "out_rate_std": out_rate_std,
            "adaptation_mean": adaptation_mean, "adaptation_std": adaptation_std,
            "voltage_mean": voltage_mean, "voltage_std": voltage_std, "voltage_tau": voltage_tau,
            "exc_conductance_mean": exc_conductance_mean, "exc_conductance_std": exc_conductance_std,
            "inh_conductance_mean": inh_conductance_mean, "inh_conductance_std": inh_conductance_std,
        }
        stp_at_spikes = dict(stp_at_spikes or {})
        for key in stp_at_spikes:
            if self.parse_stp_key(key) is None:
                raise ValueError(f"Invalid STP key '{key}'. Expected '<projection>_<{'|'.join(self.STP_VARIABLES)}>_<mean|std>', e.g. 'ee_ux_mean'.")
        if drive_rate_grid is None:
            arrays = self._with_zero_drive_axis(arrays)
            stp_at_spikes = self._with_zero_drive_axis(stp_at_spikes)

        # --- Protected Physical Data (Stored in Default Units) ---
        for name in self.GRID_FIELDS:
            setattr(self, f"_{name}", self._ingest(arrays[name], name, input_units))
        self._stp_at_spikes = stp_at_spikes

        # Freeze the object to prevent accidental attribute creation or modification
        self._finalized = True

    @staticmethod
    def _with_zero_drive_axis(arrays: dict) -> dict:
        """Adds a drive axis of length 1 (drive = 0) to 2D (exc, inh) arrays."""
        arrays = {
            name: (value[:, :, np.newaxis] if isinstance(value, np.ndarray) and value.ndim == 2 else value)
            for name, value in arrays.items()
        }
        if arrays.get("exc_rate_grid") is not None:
            arrays["drive_rate_grid"] = np.zeros_like(arrays["exc_rate_grid"], dtype=float)
        return arrays

    @classmethod
    def parse_stp_key(cls, key: str) -> tuple[str, str, str] | None:
        """(projection, variable, stat) of an STP key like 'ed_ux_mean', otherwise None."""
        parts = key.split("_")
        if len(parts) != 3:
            return None
        projection, variable, stat = parts
        if len(projection) != 2 or projection[0] not in "ei" or projection[1] not in "eid":
            return None
        if variable not in cls.STP_VARIABLES or stat not in cls.STP_STATS:
            return None
        return projection, variable, stat

    def default_unit(self, variable: str) -> str:
        """As `BaseResults.default_unit`; STP keys (see `parse_stp_key`) are unitless."""
        if self.parse_stp_key(variable) is not None:
            return ""
        return super().default_unit(variable)

    def stp_keys(self) -> tuple[str, ...]:
        """The recorded STP keys, e.g. ('ee_x_mean', 'ee_x_std', ...); empty if not recorded."""
        return tuple(self._stp_at_spikes)

    def stp_at_spikes(self, projection: str, variable: str, stat: str = "mean") -> np.ndarray | None:
        """
        STP state at the input spikes over the grid [unitless], or None if not recorded.

        Parameters
        ----------
        projection : str
            Target-source code onto the simulated neuron, e.g. 'ee', 'ei', 'ed'.
        variable : str
            'x' (x⁻), 'u' (u⁺) or 'ux' (efficacy u⁺·x⁻).
        stat : str
            'mean' or 'std' over the input spikes in the averaging window.
        """
        key = f"{projection}_{variable}_{stat}"
        if self.parse_stp_key(key) is None:
            raise ValueError(f"Invalid STP key '{key}'. Variables {self.STP_VARIABLES}, stats {self.STP_STATS}.")
        return self._stp_at_spikes.get(key)

    def __setstate__(self, state: dict):
        # Results pickled before the drive axis existed (e.g. try_load caches) get a drive = 0 axis.
        if "_drive_rate_grid" not in state:
            prefixed = {name: state.get(f"_{name}") for name in self.GRID_FIELDS}
            state = {**state, **{f"_{name}": value for name, value in self._with_zero_drive_axis(prefixed).items()}}
        # Results pickled before the STP state was recorded have none.
        state.setdefault("_stp_at_spikes", {})
        self.__dict__.update(state)

    def at_drive(self, drive_rate: float | None = None) -> "SingleNeuronResults":
        """
        2D (exc_rate, inh_rate) slice of the results at one drive rate [Hz] (None: the first drive value).
        Raises ValueError listing the available drive rates if `drive_rate` is not in the grid.
        """
        index = drive_index(self._drive_rate_grid, drive_rate)
        if self._drive_rate_grid.ndim == 2:
            return self  # already a slice
        sliced = {
            name: (None if getattr(self, f"_{name}") is None else getattr(self, f"_{name}")[:, :, index])
            for name in self.GRID_FIELDS
        }
        return SingleNeuronResults(
            simulator_name=self.simulator_name,
            neuron_name=self.neuron_name,
            neuron_params=self.neuron_params,
            neuron_sim_params=self.neuron_sim_params,
            spikes=self.spikes,
            stp_at_spikes={key: values[:, :, index] for key, values in self._stp_at_spikes.items()},
            **sliced,
        )

    def drive_rate_grid(self, unit=None):
        default_unit = self.DEFAULT_UNITS["drive_rate_grid"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._drive_rate_grid, default_unit, target_unit)

    def exc_rate_grid(self, unit=None): 
        default_unit = self.DEFAULT_UNITS["exc_rate_grid"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_rate_grid, default_unit, target_unit)
    
    def inh_rate_grid(self, unit=None): 
        default_unit = self.DEFAULT_UNITS["inh_rate_grid"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_rate_grid, default_unit, target_unit)
    
    def out_rate_mean(self, unit=None): 
        default_unit = self.DEFAULT_UNITS["out_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._out_rate_mean, default_unit, target_unit)
    
    def out_rate_std(self, unit=None): 
        default_unit = self.DEFAULT_UNITS["out_rate_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._out_rate_std, default_unit, target_unit)

    def adaptation_mean(self, unit=None):
        default_unit=self.DEFAULT_UNITS["adaptation_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._adaptation_mean, default_unit, target_unit)
    
    def adaptation_std(self, unit=None): 
        default_unit=self.DEFAULT_UNITS["adaptation_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._adaptation_std, default_unit, target_unit)
    
    def voltage_mean(self, unit=None):
        default_unit=self.DEFAULT_UNITS["voltage_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._voltage_mean, default_unit, target_unit)
    
    def voltage_std(self, unit=None): 
        default_unit=self.DEFAULT_UNITS["voltage_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._voltage_std, default_unit, target_unit)
    
    def voltage_tau(self, unit=None): 
        default_unit=self.DEFAULT_UNITS["voltage_tau"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._voltage_tau, default_unit, target_unit)

    def exc_conductance_mean(self, unit=None):
        default_unit=self.DEFAULT_UNITS["exc_conductance_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_conductance_mean, default_unit, target_unit)
    
    def exc_conductance_std(self, unit=None): 
        default_unit=self.DEFAULT_UNITS["exc_conductance_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_conductance_std, default_unit, target_unit)
    
    def inh_conductance_mean(self, unit=None):
        default_unit=self.DEFAULT_UNITS["inh_conductance_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_conductance_mean, default_unit, target_unit)
    
    def inh_conductance_std(self, unit=None):
        default_unit=self.DEFAULT_UNITS["inh_conductance_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_conductance_std, default_unit, target_unit)