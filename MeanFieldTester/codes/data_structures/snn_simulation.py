from .base import BaseSNNResults
from ..utils import snn_helpers
from pydantic import BaseModel
import numpy as np


class SNNResults(BaseSNNResults):
    DEFAULT_UNITS = {
        "exc_spikes_all" : "ms",
        "inh_spikes_all" : "ms",
        "times" : "ms",
        "drive_rate_mean" : "Hz",
        "stim_rate_mean" : "Hz",
        "exc_rate_mean" : "Hz",
        "exc_rate_std" : "Hz",
        "inh_rate_mean" : "Hz",
        "inh_rate_std" : "Hz",
        "exc_voltage_all" : "mV",
        "inh_voltage_all" : "mV",
        "exc_adaptation_all" : "nA",
        "inh_adaptation_all" : "nA",
        "ee_conductance_all" : "nS",
        "ei_conductance_all" : "nS",
        "ie_conductance_all" : "nS",
        "ii_conductance_all" : "nS",
    }


    smoothing_options = {
        "histogram": snn_helpers.activity_from_spikes_histogram,
        "sliding_window": snn_helpers.activity_from_spikes_sliding_window,
        "alpha_window": snn_helpers.activity_from_spikes_alpha_window
    }
    
    def __init__(self,
                 label_name: str = None,
                 stim_name: str = None,

                 snn_sim_params: BaseModel = None,
                 network_params: BaseModel = None,
                 stim_params: BaseModel = None,

                 exc_spikes_all: list[np.ndarray] = None,
                 inh_spikes_all: list[np.ndarray] = None,
                 times: np.ndarray = None,
                 drive_rate_mean: np.ndarray = None,
                 stim_rate_mean: np.ndarray = None,
                 exc_voltage_all: np.ndarray = None,
                 inh_voltage_all: np.ndarray = None,
                 exc_adaptation_all: np.ndarray = None,
                 inh_adaptation_all: np.ndarray = None,
                 ee_conductance_all: np.ndarray = None,
                 ei_conductance_all: np.ndarray = None,
                 ie_conductance_all: np.ndarray = None,
                 ii_conductance_all: np.ndarray = None,
                 input_units: dict = None,
                 ):

        self.label_name = label_name
        self.stim_name = stim_name
        self.snn_sim_params = snn_sim_params
        self.network_params = network_params
        self.stim_params = stim_params

        self.set_smoothing_function(
            snn_sim_params.smoothing.function, 
            snn_sim_params.smoothing.time_constant, 
            **(snn_sim_params.smoothing.kwargs or {}),
        )

        input_units = input_units or {}

        # --- Protected Physical Data (Stored in Default Units) ---
        self._exc_spikes_all = self._ingest(exc_spikes_all, "exc_spikes_all", input_units)
        self._inh_spikes_all = self._ingest(inh_spikes_all, "inh_spikes_all", input_units)
        self._times = self._ingest(times, "times", input_units)

        self._drive_rate_mean = self._ingest(drive_rate_mean, "drive_rate_mean", input_units)
        self._stim_rate_mean = self._ingest(stim_rate_mean, "stim_rate_mean", input_units)
        
        # shape (time, neuron)
        self._exc_voltage_all = self._ingest(exc_voltage_all, "exc_voltage_all", input_units)
        self._inh_voltage_all = self._ingest(inh_voltage_all, "inh_voltage_all", input_units)
        self._exc_adaptation_all = self._ingest(exc_adaptation_all, "exc_adaptation_all", input_units)
        self._inh_adaptation_all = self._ingest(inh_adaptation_all, "inh_adaptation_all", input_units)
        self._ee_conductance_all = self._ingest(ee_conductance_all, "ee_conductance_all", input_units)
        self._ei_conductance_all = self._ingest(ei_conductance_all, "ei_conductance_all", input_units)
        self._ie_conductance_all = self._ingest(ie_conductance_all, "ie_conductance_all", input_units)
        self._ii_conductance_all = self._ingest(ii_conductance_all, "ii_conductance_all", input_units)

        self._finalized = True



    def set_smoothing_function(self, smoothing_function:str, smoothing_constant:float, **kwargs):
        if smoothing_function not in self.smoothing_options:
            raise ValueError(f"Unknown smoothing function: {smoothing_function}. "
                             f"Available options: {list(self.smoothing_options.keys())}")
        self.smoothing_setup = {
            'smoothing_function': smoothing_function,
            'smoothing_constant': smoothing_constant,
            'smoothing_kwargs': kwargs
        }
        function = self.smoothing_options[smoothing_function]
        match smoothing_function:
            case "histogram":
                self._smoothing_function = lambda x: function(x, self.times(), bin_size=smoothing_constant, **kwargs)
            case "sliding_window":
                self._smoothing_function = lambda x: function(x, self.times(), window_size=smoothing_constant, **kwargs)
            case "alpha_window":
                self._smoothing_function = lambda x: function(x, self.times(), alpha_tau=smoothing_constant, **kwargs)

        # every time we set a new smoothing function, we reset the cached rates
        self._exc_rate_all = None
        self._inh_rate_all = None

    def times(self, unit=None):
        default_unit = self.DEFAULT_UNITS["times"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._times, default_unit, target_unit)
    
    def exc_spikes_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_spikes_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_spikes_all, default_unit, target_unit)

    def inh_spikes_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_spikes_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_spikes_all, default_unit, target_unit)

    def exc_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_rate_mean"]
        target_unit = default_unit if unit is None else unit
        if self._exc_rate_all is None:
            self._exc_rate_all = self._smoothing_function(self.exc_spikes_all())
        return self._get_scaled(self._exc_rate_all.mean(axis=1), default_unit, target_unit)

    def exc_rate_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_rate_std"]
        target_unit = default_unit if unit is None else unit
        if self._exc_rate_all is None:
            self._exc_rate_all = self._smoothing_function(self.exc_spikes_all())
        return self._get_scaled(self._exc_rate_all.std(axis=1), default_unit, target_unit)

    def inh_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_rate_mean"]
        target_unit = default_unit if unit is None else unit
        if self._inh_rate_all is None:
            self._inh_rate_all = self._smoothing_function(self.inh_spikes_all())
        return self._get_scaled(self._inh_rate_all.mean(axis=1), default_unit, target_unit)

    def inh_rate_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_rate_std"]
        target_unit = default_unit if unit is None else unit
        if self._inh_rate_all is None:
            self._inh_rate_all = self._smoothing_function(self.inh_spikes_all())
        return self._get_scaled(self._inh_rate_all.std(axis=1), default_unit, target_unit)

    def stim_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["stim_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._stim_rate_mean, default_unit, target_unit)

    def drive_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["drive_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._drive_rate_mean, default_unit, target_unit)

    def exc_adaptation_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_adaptation_all.mean(axis=1), default_unit, target_unit)

    def exc_adaptation_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_adaptation_all.std(axis=1), default_unit, target_unit)

    def inh_adaptation_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_adaptation_all.mean(axis=1), default_unit, target_unit)

    def inh_adaptation_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_adaptation_all.std(axis=1), default_unit, target_unit)

    def exc_voltage_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_voltage_all.mean(axis=1), default_unit, target_unit)

    def exc_voltage_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_voltage_all.std(axis=1), default_unit, target_unit)

    def inh_voltage_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_voltage_all.mean(axis=1), default_unit, target_unit)

    def inh_voltage_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_voltage_all.std(axis=1), default_unit, target_unit)

    def ee_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ee_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ee_conductance_all.mean(axis=1), default_unit, target_unit)

    def ee_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ee_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ee_conductance_all.std(axis=1), default_unit, target_unit)

    def ei_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ei_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ei_conductance_all.mean(axis=1), default_unit, target_unit)

    def ei_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ei_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ei_conductance_all.std(axis=1), default_unit, target_unit)

    def ie_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ie_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ie_conductance_all.mean(axis=1), default_unit, target_unit)

    def ie_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ie_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ie_conductance_all.std(axis=1), default_unit, target_unit)

    def ii_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ii_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ii_conductance_all.mean(axis=1), default_unit, target_unit)

    def ii_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ii_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ii_conductance_all.std(axis=1), default_unit, target_unit)
