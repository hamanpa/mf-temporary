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


class VariableResultGroup:
    """Helper class to support dot-notation metric access (e.g., results.exc_rate.time_mean())."""
    def __init__(self, results_obj, variable: str):
        self._results_obj = results_obj
        self._variable = variable

    def __getattr__(self, metric: str):
        if metric in self._results_obj.metrics:
            def getter(unit=None):
                return self._results_obj.get_data(self._variable, metric, unit)
            return getter
        raise AttributeError(f"Metric '{metric}' not found for variable '{self._variable}'.")

    def __dir__(self):
        return self._results_obj.metrics


class CoreInspectionResults(BaseInspectionResults):
    DEFAULT_UNITS = {
        "exc_rate": "Hz",
        "inh_rate": "Hz",
        "exc_voltage": "mV",
        "inh_voltage": "mV",
        "exc_adaptation": "nA",
        "inh_adaptation": "nA",
        "exc_x": "",
        "exc_y": "",
        "exc_u": "",
        "inh_x": "",
        "inh_y": "",
        "inh_u": "",
        "ee_conductance": "nS",
        "ei_conductance": "nS",
        "ie_conductance": "nS",
        "ii_conductance": "nS",
    }

    ALLOWED_VARIABLES = list(DEFAULT_UNITS.keys())

    DEFINED_METRICS = []

    def __init__(self, 
                 inspected_param: str, 
                 inspected_values: list | np.ndarray, 
                 network_names: list[str], 
                 network_params: BaseModel,
                 stimulus_params: BaseModel,
                 variables: list[str] = None,
                 metrics: list[str] = None
                 ):
        
        self.inspected_param = inspected_param
        self.param_values = np.array(inspected_values)
        self.network_names = network_names
        self.network_params = network_params
        self.stimulus_params = stimulus_params

        if variables is not None:
            var_difference = set(variables) - set(self.ALLOWED_VARIABLES)
            if var_difference:
                raise ValueError(f"Variables {var_difference} are not allowed. Allowed variables are: {self.ALLOWED_VARIABLES}")
            self.variables = variables
        else:
            self.variables = self.ALLOWED_VARIABLES
        
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = self.DEFINED_METRICS

        self._finalized = False

        # Initialize nested list containers for data collection
        self._data = {
            var: {metric: [] for metric in self.metrics}
            for var in self.variables
        }

    def add_inspection_data(self, extracted_metrics: list[dict[str, float]]):
        """
        Adds one step of the parameter sweep.

        Parameters
        ----------
        extracted_metrics : list[dict[str, float]]
            A list of extracted metrics. Must match the exact length and order 
            of `self.network_names` (e.g., [SNN_dict, DiVolo_dict, Zerlaut_dict]).
            Keys are of the form f"{variable}_{metric}" (e.g., "exc_rate_time_mean").
        """
        if self._finalized:
            raise RuntimeError("Cannot add data to a finalized InspectionResult.")

        if len(extracted_metrics) != len(self.network_names):
            raise ValueError("Length of extracted_metrics must match network_names.")

        for var in self.variables:
            for metric in self.metrics:
                key = f"{var}_{metric}"
                # Collect the calculated value from each network
                step_values = [net_metrics[key] for net_metrics in extracted_metrics]
                self._data[var][metric].append(step_values)

    def freeze(self):
        """
        Converts internal lists into NumPy arrays and locks the data structure.
        - For scalar metrics (2D): shape is (number_of_networks, number_of_parameters)
        - For time-series metrics (3D): shape is (number_of_networks, number_of_parameters, number_of_timepoints)
        """
        for var in self.variables:
            for metric in self.metrics:
                raw_list = self._data[var][metric]
                arr = np.array(raw_list)

                if arr.ndim == 2:
                    self._data[var][metric] = arr.T
                elif arr.ndim == 3:
                    # Shape was (N_params, N_networks, N_time) -> transpose to (N_networks, N_params, N_time)
                    self._data[var][metric] = np.swapaxes(arr, 0, 1)
                else:
                    self._data[var][metric] = arr

        self._finalized = True

    def get_data(self, variable: str, metric: str, unit: str = None) -> np.ndarray:

        if not self._finalized:
            raise RuntimeError("Cannot access data before freezing. Call freeze() first.")

        if variable not in self.variables or metric not in self.metrics:
            raise ValueError(f"No data for variable '{variable}' and metric '{metric}'.")
        
        internal_data = self._data[variable][metric]
        default_unit = self.DEFAULT_UNITS.get(variable)
        target_unit = default_unit if unit is None else unit
        return self._get_scaled(internal_data, default_unit, target_unit)

    def __getattr__(self, name):

        if '_finalized' not in self.__dict__:
            raise AttributeError(f"Attribute '{name}' not found. Object not initialized.")

        # Dot-notation access (eg. results.exc_rate.time_mean())
        if name in self.variables:
            return VariableResultGroup(self, name)

        # Backward compatibility layer: parses "exc_rate_time_mean" -> variable="exc_rate", metric="time_mean"
        for var in self.variables:
            if name.startswith(var):
                metric = name[len(var):].lstrip('_')
                if metric in self.metrics:
                    def getter_method(unit=None):
                        return self.get_data(var, metric, unit)
                    return getter_method

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def measured_variables(self):
        """Allows legacy plotting code to check 'variable in results.measured_variables'."""
        return [f"{var}_{metric}" for var in self.variables for metric in self.metrics]


class ModelSummaryInspectionResults(CoreInspectionResults):
    """
    Data structure for spontaneous activity inspections.
    Incrementally collects data and freezes it into NumPy arrays.
    """

    DEFINED_METRICS = [
        "time_mean", 
        "time_std",
        "pop_mean",
    ]

class ModelComparisonInspectionResults(CoreInspectionResults):
    """Data structure for dynamic stimulus comparisons (SNN vs MF)."""

    from ..analysis.comparison_metrics import METRIC_REGISTRY
    DEFINED_METRICS = list(METRIC_REGISTRY.keys())

