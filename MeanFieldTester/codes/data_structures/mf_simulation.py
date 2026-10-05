from .base import BaseMFResults, STP_VARIABLES, parse_stp_variable
from ..network_params.models import INTERNAL_PROJECTIONS
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
        # STP variables per projection (target-source code), e.g. "ei_x_mean"; unitless
        **{f"{projection}_{variable}_mean": "" for projection in INTERNAL_PROJECTIONS for variable in STP_VARIABLES},
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
                 stp_means: dict[str, np.ndarray] = None,
                 input_units: dict = None,
                 ):
        """
        Parameters
        ----------
        stp_means : dict, optional
            STP time courses per projection, keyed "<projection>_<variable>" (e.g. "ee_x", "ei_u"),
            as provided by the MF simulator for its model (dynamic state, steady state, or static).
            The efficacy factor of a projection is u * x (see `data_structures.base.STP_VARIABLES`).
        """

        # --- Public Metadata (No units required) ---
        self.label_name = label_name
        self.stim_name = stim_name
        self.mf_sim_params = mf_sim_params
        self.network_params = network_params
        self.stim_params = stim_params

        self.ignore_stp = mf_sim_params.transfer_function.tf_model.static_synapses

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

        self._stp_means = {}
        for name, values in (stp_means or {}).items():
            if parse_stp_variable(name) is None:
                raise ValueError(f"Invalid STP variable name '{name}'. Expected '<projection>_<variable>', e.g. 'ee_x'.")
            self._stp_means[name] = self._ingest(values, f"{name}_mean", input_units)

        self._exc_neuron_mpf = MembranePotentialFluctuations(
            neuron_name = network_params.exc_neuron_name,
            network_params = network_params,
            ignore_stp = self.ignore_stp,
        )

        self._inh_neuron_mpf = MembranePotentialFluctuations(
            neuron_name = network_params.inh_neuron_name,
            network_params = network_params,
            ignore_stp = self.ignore_stp,
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

    # --- Short-term plasticity ---
    @property
    def stp_variables(self) -> tuple[str, ...]:
        """Names of the available STP time courses, e.g. ('ee_x', 'ee_u', ...)."""
        return tuple(self._stp_means)

    def stp_mean(self, projection: str, variable: str, unit=None) -> np.ndarray | None:
        """
        STP variable of a projection, shape (T,), or None if the model does not provide it.

        Parameters
        ----------
        projection : str
            Target-source code, e.g. 'ei' = synapses onto E from I.
        variable : str
            'x', 'u' or 'y' (see `data_structures.base.STP_VARIABLES`).
        """
        if variable not in STP_VARIABLES:
            raise ValueError(f"Unknown STP variable '{variable}'. Expected one of {STP_VARIABLES}.")
        name = f"{projection}_{variable}"
        default_unit = self.DEFAULT_UNITS[f"{name}_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(self._stp_means.get(name), default_unit, target_unit)

    # --- Derived quantities (membrane potential fluctuation formulas applied to the MF rates) ---
    def _mpf(self, target_name: str) -> MembranePotentialFluctuations:
        if target_name == self.network_params.exc_neuron_name:
            return self._exc_neuron_mpf
        if target_name == self.network_params.inh_neuron_name:
            return self._inh_neuron_mpf
        raise ValueError(f"Unknown target neuron name: {target_name}")

    def _source_rates(self, target_name: str) -> dict[str, np.ndarray]:
        """Rates [Hz] of all sources projecting onto `target_name`."""
        # NOTE: external populations are still identified by name here (see todo.md, hard-coded population names)
        all_rates = {
            self.network_params.exc_neuron_name: self.exc_rate_mean("Hz"),
            self.network_params.inh_neuron_name: self.inh_rate_mean("Hz"),
            "stim_neuron": self.stim_rate_mean("Hz"),
            "drive_neuron": self.drive_rate_mean("Hz"),
        }
        return {source: all_rates[source] for source in self._mpf(target_name).synapse_params}

    def _effective_weight(self, target_name: str, source_name: str, rate: np.ndarray) -> np.ndarray:
        """
        Effective synaptic weight [nS] of source -> target: weight * u * x.
        Uses the model's STP time courses for internal projections when available,
        otherwise the steady-state STP at the source rate (static synapses: weight).
        """
        internal = (self.network_params.exc_neuron_name, self.network_params.inh_neuron_name)
        if source_name in internal:
            projection = self.network_params.projection_code(target_name, source_name)
            x = self.stp_mean(projection, "x")
            u = self.stp_mean(projection, "u")
            if x is not None and u is not None:
                return x * u * self.network_params.network.connectivity[target_name][source_name].syn_params.weight

        mpf = self._mpf(target_name)
        return mpf._weight_effective(rate, **mpf.synapse_params[source_name])

    def _voltage_mean(self, target_name: str, adaptation: np.ndarray) -> np.ndarray:
        rates = self._source_rates(target_name)
        effective_weights = {source: self._effective_weight(target_name, source, rate) for source, rate in rates.items()}
        return self._mpf(target_name).voltage_mean(
            rates=rates,
            effective_weights=effective_weights,
            adaptation=adaptation,
        )

    def exc_voltage_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["exc_voltage_mean"]
        target_unit = default_unit if unit is None else unit
        voltage = self._voltage_mean(self.network_params.exc_neuron_name, self.exc_adaptation_mean("nA"))
        return self._get_scaled(voltage, default_unit, target_unit)

    def inh_voltage_mean(self, unit=None):
        default_unit = self.DEFAULT_UNITS["inh_voltage_mean"]
        target_unit = default_unit if unit is None else unit
        voltage = self._voltage_mean(self.network_params.inh_neuron_name, self.inh_adaptation_mean("nA"))
        return self._get_scaled(voltage, default_unit, target_unit)

    def _conductance_mean(self, source_neuron_name, target_neuron_name, unit=None):
        print(f"WARNING: _conductance_mean is a draft implementation and unit conversion may not work properly.")

        mpf = self._mpf(target_neuron_name)
        rate = self._source_rates(target_neuron_name)[source_neuron_name]
        effective_weight = self._effective_weight(target_neuron_name, source_neuron_name, rate)

        conductance =  mpf._conductance_mean(
            rate=rate,
            effective_weight=effective_weight,
            **mpf.synapse_params[source_neuron_name]
        )

        # WARNING: This is a temporary solution to handle the unit conversion.
        # The proper way would be to implement unit handling in the MembranePotentialFluctuations class.
        # and also to implement default units for the conductance in the DEFAULT_UNITS dictionary.
        default_unit = self.DEFAULT_UNITS["ee_conductance_mean"]
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(conductance, default_unit, target_unit)


    def ee_conductance_mean(self, unit=None):
        # onto exc from exc
        return self._conductance_mean(
            source_neuron_name=self.network_params.exc_neuron_name,
            target_neuron_name=self.network_params.exc_neuron_name,
            unit=unit
        )

    def ei_conductance_mean(self, unit=None):
        # onto exc from inh
        return self._conductance_mean(
            source_neuron_name=self.network_params.inh_neuron_name,
            target_neuron_name=self.network_params.exc_neuron_name,
            unit=unit
        )

    def ie_conductance_mean(self, unit=None):
        # onto inh from exc
        return self._conductance_mean(
            source_neuron_name=self.network_params.exc_neuron_name,
            target_neuron_name=self.network_params.inh_neuron_name,
            unit=unit
        )

    def ii_conductance_mean(self, unit=None):
        # onto inh from inh
        return self._conductance_mean(
            source_neuron_name=self.network_params.inh_neuron_name,
            target_neuron_name=self.network_params.inh_neuron_name,
            unit=unit
        )
