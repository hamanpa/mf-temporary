"""
Module for defining the base data structure for storing simulation results.

This module provides abstract base classes for results storage.
It is intended to be extended by specific simulation result classes.

This module also define constants
"""

import pickle
from ..network_params.translators import get_unit_multiplier


class BaseResults:
    DEFAULT_UNITS = {
        
    }
    """
    Data structure for storing results from the SNN and MF simulations.

    Units:
    - time: [ms]
    - rate, frequency: [Hz]
    - adaptation, current: [pA]
    - conductance: [nS]
    - voltage: [mV] 
    """

    def __setattr__(self, name, value):
        if getattr(self, '_finalized', False) and name in self.DEFAULT_UNITS:
            raise AttributeError(f"Instance of {self.__class__.__name__} is frozen. Data should not be modified post-simulation.")
        super().__setattr__(name, value)

    def _ingest(self,var_value, var_name:str, input_units:dict):
        """Rescales the input value to the DEFAULT_UNITS if needed."""
        if var_value is None:
            return None
        
        default_unit = self.DEFAULT_UNITS.get(var_name)
        provided_unit = input_units.get(var_name, default_unit)
        
        if provided_unit != default_unit:
            factor = get_unit_multiplier(provided_unit, default_unit)
            return var_value * factor
        return var_value

    def _get_scaled(self, data, source_unit, target_unit):
        """Internal helper to serve data in requested units."""
        if data is None or target_unit == source_unit:
            return data
        return data * get_unit_multiplier(source_unit, target_unit)

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

