"""
Module for defining the base data structure for storing simulation results.

This module provides abstract base classes for results storage.
It is intended to be extended by specific simulation result classes.

This module also define constants
"""

import pickle
from ..network_params.translators import get_unit_multiplier
from ..network_params.models import INTERNAL_PROJECTIONS


# Short-term plasticity variables (Tsodyks-Markram), stored per projection as "<projection>_<variable>",
# e.g. "ei_x" = available resources x of the synapses onto E from I (target-source code).
#   x: available resources, u: utilisation a spike would use, y: active (released) resources.
# The efficacy factor of a projection is u * x (multiplied by the synaptic weight).
STP_VARIABLES = ("x", "u", "y")


def parse_stp_variable(name: str) -> tuple[str, str] | None:
    """Returns (projection, variable) for an STP variable name like 'ee_x', otherwise None."""
    projection, _, variable = name.partition("_")
    if projection in INTERNAL_PROJECTIONS and variable in STP_VARIABLES:
        return projection, variable
    return None


class BaseResults:
    DEFAULT_UNITS = {

    }
    """
    Data structure for storing results from the SNN and MF simulations.

    Units (see units.md):
    - time: [ms]
    - rate, frequency: [Hz]
    - adaptation, current: [nA]
    - conductance: [nS]
    - voltage: [mV]
    """

    def __setattr__(self, name, value):
        if getattr(self, '_finalized', False) and name in self.DEFAULT_UNITS:
            raise AttributeError(f"Instance of {self.__class__.__name__} is frozen. Data should not be modified post-simulation.")
        super().__setattr__(name, value)

    def default_unit(self, variable: str) -> str:
        """
        Default unit of a variable or array key, e.g. 'exc_rate', 'exc_rate_mean' or 'exc_rate_pop_mean'.
        Returns "" for unitless quantities (e.g. STP variables) and raises KeyError for unknown names.
        """
        candidates = [variable, f"{variable}_all", f"{variable}_mean"]
        for metric in ("pop_mean", "pop_std", "time_mean", "time_std", "full_mean", "all"):
            if variable.endswith(f"_{metric}"):
                base = variable[: -len(metric) - 1]
                candidates += [base, f"{base}_all", f"{base}_mean"]
        for name in candidates:
            if name in self.DEFAULT_UNITS:
                return self.DEFAULT_UNITS[name]
        raise KeyError(f"{self.__class__.__name__}: no default unit known for '{variable}'.")

    def _ingest(self, var_value, var_name:str, input_units:dict):
        """Rescales the input value to the DEFAULT_UNITS if needed."""
        if var_value is None:
            return None
        
        default_unit = self.DEFAULT_UNITS.get(var_name)
        provided_unit = input_units.get(var_name, default_unit)
        
        if provided_unit != default_unit:
            return self._get_scaled(var_value, provided_unit, default_unit)
        return var_value

    def _get_scaled(self, data, source_unit, target_unit):
        """Internal helper to serve data in requested units."""
        if data is None or target_unit == source_unit:
            return data
        factor = get_unit_multiplier(source_unit, target_unit)
        if isinstance(data, list):
            return [x * factor for x in data]
        return data *factor

    def save(self, filepath):
        """
        Save the results to a file.
        """
        print(f"Saving results to {filepath}")
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        file_size = filepath.stat().st_size / 1024 / 1024  # size in MB
        print(f"WARNING: File size: {int(file_size)} MB")


class BaseSingleNeuronResults(BaseResults):
    """
    Intermediate base class for single neuron simulations.
    """
    pass


class BaseMFResults(BaseResults):
    """
    Intermediate base class for all Mean-Field results.
    Use this for `isinstance(obj, BaseMFResults)` checks.
    """
    pass

class BaseSNNResults(BaseResults):
    """
    Intermediate base class for all Spiking Neural Network results.
    Use this for `isinstance(obj, BaseSNNResults)` checks.
    """
    pass


class BaseInspectionResults(BaseResults):
    """
    Intermediate base class for all inspection results.
    Use this for `isinstance(obj, BaseInspectionResults)` checks.
    """
    pass