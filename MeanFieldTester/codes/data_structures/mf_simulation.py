from .base import BaseMFResults
from ..transfer_function.neuropsi_tf import MembranePotentialFluctuations
from pydantic import BaseModel
import numpy as np



class MFResults(BaseMFResults):
    DEFAULT_UNITS = {
        "times" : "ms",
        "exc_rate_mean" : "Hz",
        "exc_rate_std" : "Hz",
        "inh_rate_mean" : "Hz",
        "inh_rate_std" : "Hz",
        "stim_rate_mean" : "Hz",
        "drive_rate_mean" : "Hz",
        "exc_adaptation_mean" : "nA",
        "inh_adaptation_mean" : "nA",
        "rate_cov" : "Hz^2",
        "exc_x_mean" : "",
        "exc_y_mean" : "",
        "exc_u_mean" : "",
        "inh_x_mean" : "",
        "inh_y_mean" : "",
        "inh_u_mean" : "",
        "exc_voltage_mean" : "mV",
        "inh_voltage_mean" : "mV",
        "ee_conductance_mean" : "nS",
        "ei_conductance_mean" : "nS",
        "ie_conductance_mean" : "nS",
        "ii_conductance_mean" : "nS",
    }

    def __init__(self,
                 label_name: str = None,
                 mf_sim_params: BaseModel = None,
                 network_params: BaseModel = None,
                 stim_name: str = None,
                 stim_params: BaseModel = None,
                 times: np.ndarray = None,
                 exc_rate_mean: np.ndarray = None,
                 exc_rate_std: np.ndarray = None,
                 inh_rate_mean: np.ndarray = None,
                 inh_rate_std: np.ndarray = None,
                 stim_rate_mean: np.ndarray = None,
                 drive_rate_mean: np.ndarray = None,
                 exc_adaptation_mean: np.ndarray = None,
                 inh_adaptation_mean: np.ndarray = None,
                 rate_cov: np.ndarray = None,
                 exc_x_mean: np.ndarray = None,
                 exc_y_mean: np.ndarray = None,
                 exc_u_mean: np.ndarray = None,
                 inh_x_mean: np.ndarray = None,
                 inh_y_mean: np.ndarray = None,
                 inh_u_mean: np.ndarray = None,
                 input_units: dict = None,
                 ):

        # --- Public Metadata (No units required) ---
        self.label_name = label_name
        self.stim_name = stim_name
        self.mf_sim_params = mf_sim_params
        self.network_params = network_params
        self.stim_params = stim_params

        input_units = input_units or {}

        # --- Protected Physical Data (Stored in Default Units) ---
        self._times = self._ingest(times, "times", input_units)
        self._exc_rate_mean = self._ingest(exc_rate_mean, "exc_rate_mean", input_units)
        self._exc_rate_std = self._ingest(exc_rate_std, "exc_rate_std", input_units)
        self._inh_rate_mean = self._ingest(inh_rate_mean, "inh_rate_mean", input_units)
        self._inh_rate_std = self._ingest(inh_rate_std, "inh_rate_std", input_units)
        self._stim_rate_mean = self._ingest(stim_rate_mean, "stim_rate_mean", input_units)
        self._drive_rate_mean = self._ingest(drive_rate_mean, "drive_rate_mean", input_units)
        self._exc_adaptation_mean = self._ingest(exc_adaptation_mean, "exc_adaptation_mean", input_units)
        self._inh_adaptation_mean = self._ingest(inh_adaptation_mean, "inh_adaptation_mean", input_units)
        self._rate_cov = self._ingest(rate_cov, "rate_cov", input_units)
        self._exc_x_mean = self._ingest(exc_x_mean, "exc_x_mean", input_units)
        self._exc_y_mean = self._ingest(exc_y_mean, "exc_y_mean", input_units)
        self._exc_u_mean = self._ingest(exc_u_mean, "exc_u_mean", input_units)
        self._inh_x_mean = self._ingest(inh_x_mean, "inh_x_mean", input_units)
        self._inh_y_mean = self._ingest(inh_y_mean, "inh_y_mean", input_units)
        self._inh_u_mean = self._ingest(inh_u_mean, "inh_u_mean", input_units)

        self._exc_neuron_mpf = MembranePotentialFluctuations(
            neuron_name = network_params.exc_neuron_name,
            network_params = network_params,
        )

        self._inh_neuron_mpf = MembranePotentialFluctuations(
            neuron_name = network_params.inh_neuron_name,
            network_params = network_params,
        )

        # Freeze the object to prevent accidental attribute creation or modification
        self._finalized = True

    def times(self, unit=None):
        default_unit = self.DEFAULT_UNITS["times"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._times, default_unit, target_unit)

    def exc_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_rate_mean, default_unit, target_unit)

    def exc_rate_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_rate_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_rate_std, default_unit, target_unit)

    def inh_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_rate_mean, default_unit, target_unit)

    def inh_rate_std(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_rate_std"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_rate_std, default_unit, target_unit)

    def stim_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["stim_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._stim_rate_mean, default_unit, target_unit)

    def drive_rate_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["drive_rate_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._drive_rate_mean, default_unit, target_unit)

    def exc_adaptation_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_adaptation_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_adaptation_mean, default_unit, target_unit)

    def inh_adaptation_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_adaptation_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_adaptation_mean, default_unit, target_unit)

    def rate_cov(self, unit=None):
        default_unit = self.DEFAULT_UNITS["rate_cov"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._rate_cov, default_unit, target_unit)

    def exc_x_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_x_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_x_mean, default_unit, target_unit)

    def exc_y_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_y_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_y_mean, default_unit, target_unit)

    def exc_u_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_u_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._exc_u_mean, default_unit, target_unit)

    def inh_x_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_x_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_x_mean, default_unit, target_unit)

    def inh_y_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_y_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_y_mean, default_unit, target_unit)

    def inh_u_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_u_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._inh_u_mean, default_unit, target_unit)

    def exc_voltage_mean(self, unit=None):
        rates={
            "exc_neuron": self.exc_rate_mean("Hz"),
            "inh_neuron": self.inh_rate_mean("Hz"),
            "stim_neuron": self.stim_rate_mean("Hz"),
            "drive_neuron": self.drive_rate_mean("Hz"),
        }

        effective_weights={}
        if self._exc_x_mean is not None:
            effective_weights["exc_neuron"] = self.exc_x_mean()*self.exc_u_mean()*self.network_params.synapses["exc_neuron"].syn_params.weight
        if self._inh_x_mean is not None:
            effective_weights["inh_neuron"] = self.inh_x_mean()*self.inh_u_mean()*self.network_params.synapses["inh_neuron"].syn_params.weight
        for source_neuron_name in rates:
            if source_neuron_name not in effective_weights:
                effective_weights[source_neuron_name] = self._exc_neuron_mpf._weight_effective(rates[source_neuron_name], **self._exc_neuron_mpf.synapse_params[source_neuron_name])

        return self._exc_neuron_mpf.voltage_mean(
            rates=rates,
            effective_weights=effective_weights,
            adaptation=self.exc_adaptation_mean("nA"),
        )


    def inh_voltage_mean(self, unit=None):
        rates={
            "exc_neuron": self.exc_rate_mean("Hz"),
            "inh_neuron": self.inh_rate_mean("Hz"),
            "stim_neuron": self.stim_rate_mean("Hz"),
            "drive_neuron": self.drive_rate_mean("Hz"),
        }

        effective_weights={}
        if self._exc_x_mean is not None:
            effective_weights["exc_neuron"] = self.exc_x_mean()*self.exc_u_mean()*self.network_params.synapses["exc_neuron"].syn_params.weight
        if self._inh_x_mean is not None:
            effective_weights["inh_neuron"] = self.inh_x_mean()*self.inh_u_mean()*self.network_params.synapses["inh_neuron"].syn_params.weight
        for source_neuron_name in rates:
            if source_neuron_name not in effective_weights:
                effective_weights[source_neuron_name] = self._inh_neuron_mpf._weight_effective(rates[source_neuron_name], **self._inh_neuron_mpf.synapse_params[source_neuron_name])

        return self._inh_neuron_mpf.voltage_mean(
            rates=rates,
            effective_weights=effective_weights,
            adaptation=self.inh_adaptation_mean("nA"),
        )

    def _conductance_mean(self, source_neuron_name, target_neuron_name, unit=None):
        print(f"WARNING: _conductance_mean is a draft implementation and unit conversion may not work properly.")

        if target_neuron_name == self.network_params.exc_neuron_name:
            mpf = self._exc_neuron_mpf
        elif target_neuron_name == self.network_params.inh_neuron_name:
            mpf = self._inh_neuron_mpf
        else:
            raise ValueError(f"Unknown target neuron name: {target_neuron_name}")

        synapse_params = mpf.synapse_params(source_neuron_name)

        if source_neuron_name == self.network_params.exc_neuron_name:
            rate = self.exc_rate_mean("Hz")
            if self._exc_x_mean is not None:
                effective_weight = self.exc_x_mean()*self.exc_u_mean()*self.network_params.synapses[source_neuron_name].syn_params.weight
            else:
                effective_weight = mpf._weight_effective(rate, **synapse_params)
        elif source_neuron_name == self.network_params.inh_neuron_name:
            rate = self.inh_rate_mean("Hz")
            if self._inh_x_mean is not None:
                effective_weight = self.inh_x_mean()*self.inh_u_mean()*self.network_params.synapses[source_neuron_name].syn_params.weight
            else:
                effective_weight = mpf._weight_effective(rate, **synapse_params)
        elif source_neuron_name == "stim_neuron":
            rate = self.stim_rate_mean("Hz")
            effective_weight = mpf._weight_effective(rate, **synapse_params)
        elif source_neuron_name == "drive_neuron":
            rate = self.drive_rate_mean("Hz")
            effective_weight = mpf._weight_effective(rate, **synapse_params)
        else:
            raise ValueError(f"Unknown source neuron name: {source_neuron_name}")

        conductance =  mpf._conductance_mean(
            rate=rate,
            effective_weight=effective_weight,
            **synapse_params
        )

        # WARNING: This is a temporary solution to handle the unit conversion. 
        # The proper way would be to implement unit handling in the MembranePotentialFluctuations class.
        # and also to implement default units for the conductance in the DEFAULT_UNITS dictionary.
        default_unit = self.DEFAULT_UNITS["ee_conductance_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(conductance, default_unit, target_unit)


    def ee_conductance_mean(self, unit=None):

        # following is draft and ideas 
        # target neuron is exc_neuron (the first letter)
        # what is source?
        #   - either exc neuron
        #   - or all excitatory sources (exc, stim, drive)?

        # Also this should be possible to make more generic, so that we can have a method for any source-target pair, e.g. ee, ei, ie, ii.

        # so far exc_neuron -> exc_neuron

        return self._conductance_mean(
            source_neuron_name=self.network_params.exc_neuron_name,
            target_neuron_name=self.network_params.exc_neuron_name,
            unit=unit
        )

    
    def ei_conductance_mean(self, unit=None):
        #  from inh to exc 
        
        return self._conductance_mean(
            source_neuron_name=self.network_params.inh_neuron_name,
            target_neuron_name=self.network_params.exc_neuron_name,
            unit=unit
        )

    def ie_conductance_mean(self, unit=None):
        # from exc to inh
        return self._conductance_mean(
            source_neuron_name=self.network_params.exc_neuron_name,
            target_neuron_name=self.network_params.inh_neuron_name,
            unit=unit
        )

    def ii_conductance_mean(self, unit=None):
        # from inh to inh
        return self._conductance_mean(
            source_neuron_name=self.network_params.inh_neuron_name,
            target_neuron_name=self.network_params.inh_neuron_name,
            unit=unit
        )