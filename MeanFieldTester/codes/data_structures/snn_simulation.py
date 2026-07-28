from .base import BaseSNNResults
from ..utils import snn_helpers
from pydantic import BaseModel
import numpy as np
from functools import partial

class SNNResults(BaseSNNResults):
    DEFAULT_UNITS = {
        "exc_spikes_all" : "ms",
        "inh_spikes_all" : "ms",
        "times" : "ms",
        "drive_rate_mean" : "Hz",
        "stim_rate_mean" : "Hz",
        "exc_rate_all" : "Hz",
        "exc_rate_mean" : "Hz",
        "exc_rate_std" : "Hz",
        "inh_rate_all" : "Hz",
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
        "exc_rate" : "Hz",
        "inh_rate" : "Hz",
        "exc_voltage" : "mV",
        "inh_voltage" : "mV",
        "exc_adaptation" : "nA",
        "inh_adaptation" : "nA",
        "ee_conductance" : "nS",
        "ei_conductance" : "nS",
        "ie_conductance" : "nS",
        "ii_conductance" : "nS",
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

        self.set_smoothing_function(
            snn_sim_params.smoothing.function, 
            snn_sim_params.smoothing.time_constant, 
            **(snn_sim_params.smoothing.kwargs or {}),
        )

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
                self._smoothing_function = partial(function, times=self.times(), bin_size=smoothing_constant, **kwargs)
            case "sliding_window":
                self._smoothing_function = partial(function, times=self.times(), window_size=smoothing_constant, **kwargs)
            case "alpha_window":
                self._smoothing_function = partial(function, times=self.times(), alpha_tau=smoothing_constant, **kwargs)

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

    def exc_rate_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_rate_all"]
        target_unit = default_unit if unit is None else unit
        if self._exc_rate_all is None:
            self._exc_rate_all = self._smoothing_function(self.exc_spikes_all())
        return self._get_scaled(self._exc_rate_all, default_unit, target_unit)

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

    def inh_rate_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_rate_all"]
        target_unit = default_unit if unit is None else unit
        if self._inh_rate_all is None:
            self._inh_rate_all = self._smoothing_function(self.inh_spikes_all())
        return self._get_scaled(self._inh_rate_all, default_unit, target_unit)

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

    def exc_adaptation_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_adaptation_all, default_unit, target_unit)

    def exc_adaptation_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_adaptation_all.mean(axis=1), default_unit, target_unit)

    def exc_adaptation_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_adaptation_all.std(axis=1), default_unit, target_unit)

    def inh_adaptation_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_adaptation_all, default_unit, target_unit)

    def inh_adaptation_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_adaptation_all.mean(axis=1), default_unit, target_unit)

    def inh_adaptation_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_adaptation_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_adaptation_all.std(axis=1), default_unit, target_unit)

    def exc_voltage_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_voltage_all, default_unit, target_unit)

    def exc_voltage_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_voltage_all.mean(axis=1), default_unit, target_unit)

    def exc_voltage_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_voltage_all.std(axis=1), default_unit, target_unit)

    def inh_voltage_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_voltage_all, default_unit, target_unit)

    def inh_voltage_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_voltage_all.mean(axis=1), default_unit, target_unit)

    def inh_voltage_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_voltage_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_voltage_all.std(axis=1), default_unit, target_unit)

    def ee_conductance_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ee_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ee_conductance_all, default_unit, target_unit)

    def ee_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ee_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ee_conductance_all.mean(axis=1), default_unit, target_unit)

    def ee_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ee_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ee_conductance_all.std(axis=1), default_unit, target_unit)

    def ei_conductance_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ei_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ei_conductance_all, default_unit, target_unit)

    def ei_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ei_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ei_conductance_all.mean(axis=1), default_unit, target_unit)

    def ei_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ei_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ei_conductance_all.std(axis=1), default_unit, target_unit)

    def ie_conductance_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ie_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ie_conductance_all, default_unit, target_unit)

    def ie_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ie_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ie_conductance_all.mean(axis=1), default_unit, target_unit)

    def ie_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ie_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ie_conductance_all.std(axis=1), default_unit, target_unit)

    def ii_conductance_all(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ii_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ii_conductance_all, default_unit, target_unit)

    def ii_conductance_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ii_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ii_conductance_all.mean(axis=1), default_unit, target_unit)

    def ii_conductance_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["ii_conductance_all"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._ii_conductance_all.std(axis=1), default_unit, target_unit)

    def _compute_stp_variables(self, neuron_name):
        """Internal method to lazily evaluate and cache STP variables."""
        syn_params = self.network_params.synapses[neuron_name].syn_params
        # NOTE: in case of static synapses, U, tau_rec, and tau_fac will 
        # default to 1.0, 0.0, and 0.0 respectively
        U = getattr(syn_params, "U", 1.0)
        tau_rec = getattr(syn_params, "tau_rec", 0.0)
        tau_fac = getattr(syn_params, "tau_fac", 0.0)

        if neuron_name == "exc_neuron":
            u,x = snn_helpers.reconstruct_stp_dynamics(
                self._exc_spikes_all, 
                U, 
                tau_rec, 
                tau_fac, 
                self.times()
            )
            self._exc_u_all = u
            self._exc_x_all = x
        elif neuron_name == "inh_neuron":
            u,x = snn_helpers.reconstruct_stp_dynamics(
                self._inh_spikes_all, 
                U, 
                tau_rec, 
                tau_fac, 
                self.times()
            )
            self._inh_u_all = u
            self._inh_x_all = x

    def exc_u_all(self, unit=None):
        if not hasattr(self, '_exc_u_all'):
            self._compute_stp_variables("exc_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_u_all, default_unit, target_unit)

    def exc_u_mean(self, unit=None):
        if not hasattr(self, '_exc_u_all'):
            self._compute_stp_variables("exc_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_u_all.mean(axis=1), default_unit, target_unit)

    def exc_u_std(self, unit=None):
        if not hasattr(self, '_exc_u_all'):
            self._compute_stp_variables("exc_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_u_all.std(axis=1), default_unit, target_unit)

    def exc_x_all(self, unit=None):
        if not hasattr(self, '_exc_x_all'):
            self._compute_stp_variables("exc_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_x_all, default_unit, target_unit)

    def exc_x_mean(self, unit=None):
        if not hasattr(self, '_exc_x_all'):
            self._compute_stp_variables("exc_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_x_all.mean(axis=1), default_unit, target_unit)

    def exc_x_std(self, unit=None):
        if not hasattr(self, '_exc_x_all'):
            self._compute_stp_variables("exc_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_x_all.std(axis=1), default_unit, target_unit)

    def inh_u_all(self, unit=None):
        if not hasattr(self, '_inh_u_all'):
            self._compute_stp_variables("inh_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_u_all, default_unit, target_unit)

    def inh_u_mean(self, unit=None):
        if not hasattr(self, '_inh_u_all'):
            self._compute_stp_variables("inh_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_u_all.mean(axis=1), default_unit, target_unit)

    def inh_u_std(self, unit=None):
        if not hasattr(self, '_inh_u_all'):
            self._compute_stp_variables("inh_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_u_all.std(axis=1), default_unit, target_unit)

    def inh_x_all(self, unit=None):
        if not hasattr(self, '_inh_x_all'):
            self._compute_stp_variables("inh_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_x_all, default_unit, target_unit)

    def inh_x_mean(self, unit=None):
        if not hasattr(self, '_inh_x_all'):
            self._compute_stp_variables("inh_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_x_all.mean(axis=1), default_unit, target_unit)

    def inh_x_std(self, unit=None):
        if not hasattr(self, '_inh_x_all'):
            self._compute_stp_variables("inh_neuron")
        default_unit = ""
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_x_all.std(axis=1), default_unit, target_unit)

    # --- Generic Metric Methods for Output Extraction & Savings ---
    def _get_n_neurons(self, variable: str) -> int:
        """Helper to resolve neuron count N for array shaping when data is missing."""
        if "inh" in variable and self._inh_spikes_all is not None:
            return len(self._inh_spikes_all)
        elif self._exc_spikes_all is not None:
            return len(self._exc_spikes_all)
        return 1

    def _get_raw_all(self, variable: str) -> np.ndarray:
        """Retrieves raw (T, N) spatio-temporal data array for any variable name."""
        attr_name = f"_{variable}_all"
        if hasattr(self, attr_name):
            raw = getattr(self, attr_name)
            if raw is None and hasattr(self, f"{variable}_all"):
                raw = getattr(self, f"{variable}_all")()
            return raw
        
        # Try method lookups
        for candidate in [f"{variable}_all", f"{variable}_mean", variable]:
            if hasattr(self, candidate) and callable(getattr(self, candidate)):
                try:
                    return getattr(self, candidate)()
                except Exception:
                    pass
        return None

    def get_all(self, variable: str, unit=None) -> np.ndarray:
        """Returns full (T, N) spatio-temporal matrix for variable."""
        raw = self._get_raw_all(variable)
        if raw is None:
            return None
        default_unit = self.DEFAULT_UNITS.get(f"{variable}_all", self.DEFAULT_UNITS.get(variable, ""))
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(raw, default_unit, target_unit)

    def get_pop_mean(self, variable: str, unit=None) -> np.ndarray:
        """Computes population mean time-series of shape (T,). Returns np.nan array if unrecorded."""
        if hasattr(self, f"{variable}_mean") and callable(getattr(self, f"{variable}_mean")):
            try:
                res = getattr(self, f"{variable}_mean")(unit=unit)
                if res is not None:
                    return res
            except Exception:
                pass

        raw = self._get_raw_all(variable)
        if raw is None:
            T = len(self.times()) if self.times() is not None else 0
            return np.full(T, np.nan)

        pop_avg = np.mean(raw, axis=1) if raw.ndim == 2 else raw
        default_unit = self.DEFAULT_UNITS.get(f"{variable}_mean", self.DEFAULT_UNITS.get(variable, ""))
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(pop_avg, default_unit, target_unit)

    def get_pop_std(self, variable: str, unit=None) -> np.ndarray:
        """Computes population standard deviation time-series of shape (T,). Returns np.nan array if unrecorded."""
        if hasattr(self, f"{variable}_std") and callable(getattr(self, f"{variable}_std")):
            try:
                res = getattr(self, f"{variable}_std")(unit=unit)
                if res is not None:
                    return res
            except Exception:
                pass

        raw = self._get_raw_all(variable)
        if raw is None:
            T = len(self.times()) if self.times() is not None else 0
            return np.full(T, np.nan)

        pop_std = np.std(raw, axis=1) if raw.ndim == 2 else np.zeros_like(raw)
        default_unit = self.DEFAULT_UNITS.get(f"{variable}_std", self.DEFAULT_UNITS.get(variable, ""))
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(pop_std, default_unit, target_unit)

    def get_time_mean(self, variable: str, start_time: float = 0.0, end_time: float = None, unit=None) -> np.ndarray:
        """Computes per-neuron time-averaged steady-state of shape (N,) over [start_time, end_time]. Returns np.nan array if unrecorded."""
        raw = self._get_raw_all(variable)
        if raw is None or raw.ndim != 2:
            N = self._get_n_neurons(variable)
            return np.full(N, np.nan)

        t_end = np.inf if end_time is None else end_time
        time_mask = (self.times() >= start_time) & (self.times() <= t_end)
        time_avg = np.mean(raw[time_mask, :], axis=0)
        default_unit = self.DEFAULT_UNITS.get(f"{variable}_all", self.DEFAULT_UNITS.get(variable, ""))
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(time_avg, default_unit, target_unit)

    def get_time_std(self, variable: str, start_time: float = 0.0, end_time: float = None, unit=None) -> np.ndarray:
        """Computes per-neuron time standard deviation of shape (N,) over [start_time, end_time]. Returns np.nan array if unrecorded."""
        raw = self._get_raw_all(variable)
        if raw is None or raw.ndim != 2:
            N = self._get_n_neurons(variable)
            return np.full(N, np.nan)

        t_end = np.inf if end_time is None else end_time
        time_mask = (self.times() >= start_time) & (self.times() <= t_end)
        time_std = np.std(raw[time_mask, :], axis=0)
        default_unit = self.DEFAULT_UNITS.get(f"{variable}_all", self.DEFAULT_UNITS.get(variable, ""))
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(time_std, default_unit, target_unit)

    def get_full_mean(self, variable: str, start_time: float = 0.0, end_time: float = None, unit=None) -> float:
        """Computes global steady-state average scalar over [start_time, end_time]. Returns np.nan if unrecorded."""
        time_avg = self.get_time_mean(variable, start_time, end_time, unit)
        return float(np.nanmean(time_avg)) if time_avg is not None else np.nan


