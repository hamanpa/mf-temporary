"""
This module defines the data structure for storing results from a single neuron simulation.

"""

import numpy as np
import warnings

from .base import BaseSingleNeuronResults
from ..network_params.translators import get_unit_multiplier
from pydantic import BaseModel


class SingleNeuronResults(BaseSingleNeuronResults):
    """
    Data structure for storing results from single neuron simulations.
    
    Internal data is strictly maintained in the default units. Physical quantities
    are accessed via methods (e.g., `results.out_rate_mean(unit="kHz")`).
    """

    DEFAULT_UNITS = {
        "exc_rate_grid": "Hz",
        "inh_rate_grid": "Hz",
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
                 input_units: dict = None):
        
        # --- Public Metadata (No units required) ---
        self.simulator_name = simulator_name
        self.neuron_name = neuron_name
        self.neuron_params = neuron_params
        self.neuron_sim_params = neuron_sim_params
        self.spikes = spikes

        # --- Unit Ingestion Logic ---
        input_units = input_units or {}

        # --- Protected Physical Data (Stored in Default Units) ---
        self._exc_rate_grid = self._ingest(exc_rate_grid, "exc_rate_grid", input_units)
        self._inh_rate_grid = self._ingest(inh_rate_grid, "inh_rate_grid", input_units)
        self._out_rate_mean = self._ingest(out_rate_mean, "out_rate_mean", input_units)
        self._out_rate_std = self._ingest(out_rate_std, "out_rate_std", input_units)
        self._adaptation_mean = self._ingest(adaptation_mean, "adaptation_mean", input_units)
        self._adaptation_std = self._ingest(adaptation_std, "adaptation_std", input_units)
        self._voltage_mean = self._ingest(voltage_mean, "voltage_mean", input_units)
        self._voltage_std = self._ingest(voltage_std, "voltage_std", input_units)
        self._voltage_tau = self._ingest(voltage_tau, "voltage_tau", input_units)
        self._exc_conductance_mean = self._ingest(exc_conductance_mean, "exc_conductance_mean", input_units)
        self._exc_conductance_std = self._ingest(exc_conductance_std, "exc_conductance_std", input_units)
        self._inh_conductance_mean = self._ingest(inh_conductance_mean, "inh_conductance_mean", input_units)
        self._inh_conductance_std = self._ingest(inh_conductance_std, "inh_conductance_std", input_units)

        # Freeze the object to prevent accidental attribute creation or modification
        self._finalized = True

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