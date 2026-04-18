"""
This module defines the data structure for storing results from a single neuron simulation.

"""

import numpy as np
import warnings

from .base import Results
from ..network_params.translators import get_unit_multiplier
from pydantic import BaseModel


class SingleNeuronResults(Results):
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

        def _ingest(value, name):
            """Rescales the input value to the DEFAULT_UNITS if needed."""
            if value is None:
                return None
            
            default_unit = self.DEFAULT_UNITS.get(name)
            provided_unit = input_units.get(name, default_unit)
            
            if provided_unit != default_unit:
                factor = get_unit_multiplier(provided_unit, default_unit)
                return value * factor
            return value

        # --- Protected Physical Data (Stored in Default Units) ---
        self._exc_rate_grid = _ingest(exc_rate_grid, "exc_rate_grid")
        self._inh_rate_grid = _ingest(inh_rate_grid, "inh_rate_grid")
        self._out_rate_mean = _ingest(out_rate_mean, "out_rate_mean")
        self._out_rate_std = _ingest(out_rate_std, "out_rate_std")
        self._adaptation_mean = _ingest(adaptation_mean, "adaptation_mean")
        self._adaptation_std = _ingest(adaptation_std, "adaptation_std")
        self._voltage_mean = _ingest(voltage_mean, "voltage_mean")
        self._voltage_std = _ingest(voltage_std, "voltage_std")
        self._voltage_tau = _ingest(voltage_tau, "voltage_tau")
        self._exc_conductance_mean = _ingest(exc_conductance_mean, "exc_conductance_mean")
        self._exc_conductance_std = _ingest(exc_conductance_std, "exc_conductance_std")
        self._inh_conductance_mean = _ingest(inh_conductance_mean, "inh_conductance_mean")
        self._inh_conductance_std = _ingest(inh_conductance_std, "inh_conductance_std")

        # Freeze the object to prevent accidental attribute creation or modification
        self._finalized = True

    def __setattr__(self, name, value):
        if getattr(self, '_finalized', False) and name != '_finalized':
            raise AttributeError(f"Instance of {self.__class__.__name__} is frozen. Data should not be modified post-simulation.")
        super().__setattr__(name, value)

    # --- Data Retrieval Methods ---
    
    def _get_scaled(self, data, source_unit, target_unit):
        """Internal helper to serve data in requested units."""
        if data is None or target_unit == source_unit:
            return data
        return data * get_unit_multiplier(source_unit, target_unit)

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
    
    # ==========================================
    # DEPRECATED PROPERTIES (For backward compatibility)
    # ==========================================
    
    def _warn_deprecated(self, old_name, new_name):
        warnings.warn(
            f"The attribute '{old_name}' is deprecated and will be removed in a future version. "
            f"Please use the method '{new_name}()' instead.",
            category=FutureWarning,
            stacklevel=3
        )

    @property
    def exc_drive_mean(self):
        self._warn_deprecated('exc_drive_mean', 'exc_rate_grid')
        return self.exc_rate_grid()

    @property
    def nu_e(self):
        self._warn_deprecated('nu_e', 'exc_rate_grid')
        return self.exc_rate_grid()

    @property
    def nu_i(self):
        self._warn_deprecated('nu_i', 'inh_rate_grid')
        return self.inh_rate_grid()

    @property
    def inh_drive_mean(self):
        self._warn_deprecated('inh_drive_mean', 'inh_rate_grid')
        return self.inh_rate_grid()

    @property
    def nu_out_mean(self):
        self._warn_deprecated('nu_out_mean', 'out_rate_mean')
        return self.out_rate_mean()
    
    @property
    def nu_out_std(self):
        self._warn_deprecated('nu_out_std', 'out_rate_std')
        return self.out_rate_std()

    @property
    def w_mean(self):
        self._warn_deprecated('w_mean', 'adaptation_mean')
        return self.adaptation_mean()

    @property
    def w_std(self):
        self._warn_deprecated('w_std', 'adaptation_std')
        return self.adaptation_std()
 
    @property
    def v_mean(self):
        self._warn_deprecated('v_mean', 'voltage_mean')
        return self.voltage_mean()

    @property
    def v_std(self):
        self._warn_deprecated('v_std', 'voltage_std')
        return self.voltage_std()

    @property
    def v_tau(self):
        self._warn_deprecated('v_tau', 'voltage_tau')
        return self.voltage_tau()

    @property
    def gsyn_e_mean(self):
        self._warn_deprecated('gsyn_e_mean', 'exc_conductance_mean')
        return self.exc_conductance_mean()

    @property
    def gsyn_e_std(self):
        self._warn_deprecated('gsyn_e_std', 'exc_conductance_std')
        return self.exc_conductance_std()

    @property
    def gsyn_i_mean(self):
        self._warn_deprecated('gsyn_i_mean', 'inh_conductance_mean')
        return self.inh_conductance_mean()

    @property
    def gsyn_i_std(self):
        self._warn_deprecated('gsyn_i_std', 'inh_conductance_std')
        return self.inh_conductance_std()  
