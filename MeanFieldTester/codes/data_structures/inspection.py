"""
Each InspectionResults class stores the results of inspecting a specific parameter
and for single network and stimulus configuration.

- Having multiple networks/stimuli inspected at once would complicate working with/filtering the data

- Different types of inspections (e.g., spontaneous activity, response to stimuli)
may require different data to be stored, hence different classes.

"""

"""
Data structures for storing the results of parameter inspections.
"""

from .base import BaseInspectionResults
import numpy as np
from pydantic import BaseModel


class CoreInspectionResults(BaseInspectionResults):
    """
    Abstract base class for all Inspection Results.
    Handles incremental data collection, array freezing (transposition), 
    and dynamic unit-scaling getters.
    """
    
    DEFAULT_UNITS = {}
    ALLOWED_MEASURED_VARS = []


    def __init__(self, 
                 inspected_param: str, 
                 inspected_values: list | np.ndarray, 
                 network_names: list[str], 
                 network_params: BaseModel,
                 stimulus_params: BaseModel,
                 measured_variables: list[str] = None,
                 ):
        
        self.inspected_param = inspected_param
        self.param_values = np.array(inspected_values)
        self.network_names = network_names

        self.network_params = network_params
        self.stimulus_params = stimulus_params

        if measured_variables is not None:
            var_difference = set(measured_variables) - set(self.ALLOWED_MEASURED_VARS)
            if var_difference:
                raise ValueError(f"Measured variables {var_difference} are not allowed. Allowed variables are: {self.ALLOWED_MEASURED_VARS}")
            self.measured_variables = measured_variables
        else:
            self.measured_variables = self.ALLOWED_MEASURED_VARS

        self._finalized = False

        # Init data containers
        for var in self.measured_variables:
            setattr(self, f"_{var}", [])


    def __getattr__(self, name):
        """
        Dynamically catches requests for measured variables (e.g., results.exc_rate_time_mean).
        Returns a callable that applies unit scaling, matching the SNNResults API.
        """
        if '_finalized' not in self.__dict__:
            raise AttributeError(f"Attribute '{name}' not found. Object not initialized.")
        
        if name in self.measured_variables:
            
            def getter_method(unit=None):
                if not self._finalized:
                    raise RuntimeError("Cannot access data before freezing. Call freeze() first.")
                
                internal_data = getattr(self, f"_{name}")
                default_unit = self.DEFAULT_UNITS.get(name)
                target_unit = default_unit if unit is None else unit
                return self._get_scaled(internal_data, default_unit, target_unit)

            return getter_method

        elif name in self.ALLOWED_MEASURED_VARS:
            raise AttributeError(f"'{name}' is a valid measured variable but has not been collected yet. Ensure it is included in `measured_variables` during initialization.")            

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


    def add_inspection_data(self, extracted_metrics: list[dict[str, float]]):
        """
        Adds one step of the parameter sweep.
        
        Parameters
        ----------
        extracted_metrics : list[dict[str, float]]
            A list of extracted metrics. Must match the exact length and order 
            of `self.network_names` (e.g., [SNN_dict, DiVolo_dict, Zerlaut_dict]).
        """
        if self._finalized:
            raise RuntimeError("Cannot add data to a finalized InspectionResult.")
            
        if len(extracted_metrics) != len(self.network_names):
            raise ValueError("Length of extracted_metrics must match length of network_names.")

        # Reorganize the data so each variable gets a list of values for this sweep
        for var in self.measured_variables:
            extracted_values = [net_metrics[var] for net_metrics in extracted_metrics]
            getattr(self, f"_{var}").append(extracted_values)

    def freeze(self):
        """
        Converts internal lists into NumPy arrays and locks the data structure.
        The resulting arrays have shape: (number_of_networks, number_of_parameters)
        This shape makes plotting easy: plt.plot(param_values, data[network_idx])
        """
        for var in self.measured_variables:
            # list of lists (shape: num_params x num_networks)
            raw_list = getattr(self, f"_{var}")
            
            # Convert to numpy and Transpose to (inspected_network_index, inspected_param_index)
            frozen_array = np.array(raw_list).T
            
            # Set the public attribute and delete the private list
            setattr(self, f"_{var}", frozen_array)
            
        self._finalized = True



class SpontInspectionResults(CoreInspectionResults):
    """
    Data structure for spontaneous activity inspections.
    Incrementally collects data and freezes it into NumPy arrays.
    """
    DEFAULT_UNITS = {
        "exc_rate_time_mean" : "Hz",
        "exc_rate_time_std" : "Hz",
        "inh_rate_time_mean" : "Hz",
        "inh_rate_time_std" : "Hz",
        "exc_voltage_time_mean" : "mV",
        "exc_voltage_time_std" : "mV",
        "inh_voltage_time_mean" : "mV",
        "inh_voltage_time_std" : "mV",
        "exc_adaptation_time_mean" : "nA",
        "exc_adaptation_time_std" : "nA",
        "inh_adaptation_time_mean" : "nA",
        "inh_adaptation_time_std" : "nA",
    }

    ALLOWED_MEASURED_VARS = list(DEFAULT_UNITS.keys())


class DynamicStimulusInspectionResults(CoreInspectionResults):
    """Data structure for dynamic stimulus comparisons (SNN vs MF)."""

    DEFAULT_UNITS = {
        "exc_rate_rmse": "Hz",
        "exc_rate_error_mean": "Hz",
        "exc_rate_error_std": "Hz",
        "exc_rate_pearson": "Hz",

        "inh_rate_rmse": "Hz",
        "inh_rate_error_mean": "Hz",
        "inh_rate_error_std": "Hz",
        "inh_rate_pearson": "Hz",

        "exc_adaptation_rmse": "Hz",
        "exc_adaptation_error_mean": "Hz",
        "exc_adaptation_error_std": "Hz",
        "exc_adaptation_pearson": "Hz",
        # TODO:
        # add voltage
        # add adaptation
    }

    ALLOWED_MEASURED_VARS = list(DEFAULT_UNITS.keys())
