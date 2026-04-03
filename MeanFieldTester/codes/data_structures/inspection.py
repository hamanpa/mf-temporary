"""
Each InspectionResults class stores the results of inspecting a specific parameter
and for single network and stimulus configuration.

- Having multiple networks/stimuli inspected at once would complicate working with/filtering the data

- Different types of inspections (e.g., spontaneous activity, response to stimuli)
may require different data to be stored, hence different classes.

"""


import codes.data_structures.base as base
import numpy as np

def _create_array_property(var_name):
    """Creates a property that returns the NumPy array conversion of the internal list."""
    # Helper function to generate a property method dynamically
    # The internal list name (e.g., '_exc_rate_mean')
    internal_name = f"_{var_name}"
    
    # Define the getter method for the property
    @property
    def getter(self):
        # Access the internal list attribute (e.g., self._exc_rate_mean)
        return np.array(getattr(self, internal_name))
    
    # Return the property (which is the getter method decorated with @property)
    return getter




class SpontaneousActivityInspectionResults:
    """Inspection results for spontaneous activity simulations.

    Spontaneous activity allows meaningful time-averaging of rates, voltages and
    adaptation variables over a steady-state period of the simulation.
    
    Here the 'mean' means time average over the steady-state period of the simulation.
    The 'std' means standard deviation over time in the same period.
    
    One network per inspection! (the same as with simulations)
    such that I iterate it like a list as previously.

    Attributes:
        inspected_network_name (str): Name of the inspected network.
        inspected_network_params (dict): Parameters of the inspected network.
        inspected_stimulus_name (str): Name of the inspected stimulus.
        inspected_stimulus_params (dict): Parameters of the inspected stimulus.
        inspected_param (str): The name of the inspected parameter.
        param_values (list): The list of inspected parameter values.
        measured_variables (list): List of measured variable names stored as attributes.
    
    """
    ALLOWED_MEASURED_VARS = [
        "exc_rate_time_mean",
        "exc_rate_time_std",
        "inh_rate_time_mean",
        "inh_rate_time_std",
        "exc_voltage_time_mean",
        "exc_voltage_time_std",
        "inh_voltage_time_mean",
        "inh_voltage_time_std",
        "exc_adaptation_time_mean",
        "exc_adaptation_time_std",
        "inh_adaptation_time_mean",
        "inh_adaptation_time_std",
    ]

    def __init__(self, 
                 inspected_network_name:str,
                 inspected_network_params:dict,
                 inspected_stimulus_name:str, 
                 inspected_stimulus_params:dict,
                 inspected_param:str,
                 measured_variables:list[str]=None,
                 param_values:list|float|int=None,
                 **kwargs):

        # Save the information about the network, stimulus and inspection parameter
        self.inspected_network_name = inspected_network_name
        self.inspected_network_params = inspected_network_params
        
        self.inspected_stimulus_name = inspected_stimulus_name
        self.inspected_stimulus_params = inspected_stimulus_params
        
        self.inspected_param = inspected_param

        # Setup the information about measured variables
        if measured_variables is not None:
            for var in measured_variables:
                if var not in self.ALLOWED_MEASURED_VARS:
                    raise ValueError(f"Measured variable '{var}' is not allowed. Allowed variables are: {self.ALLOWED_MEASURED_VARS}")
            self.measured_variables = measured_variables
        else:
            self.measured_variables = self.ALLOWED_MEASURED_VARS
        
        # Init data containers
        self.param_values = []
        for var in self.measured_variables:
            setattr(self, var, [])

        if isinstance(param_values, (float, int)):
            self.add_result_point(param_values, **kwargs)
        elif isinstance(param_values, list):
            self.add_result_list(param_values, **kwargs)
        elif param_values is None:
            pass  # None is valid value (no data provided upon initialization)
        else:
            raise ValueError(f"param_values must be a list, float, or int, not {type(param_values).__name__}")

    def add_result_point(self,
                   param_value:float|int,
                   **kwargs):
        
        # 1. VALIDATION: Check for data consistency before modifying internal state
        self._verify_kwargs(**kwargs)
        self._verify_data_type(param_value, mode="point", **kwargs)

        # 2. ADD RESULTS: the new data to the internal state
        self.param_values.append(param_value)
        for key, value in kwargs.items():
            getattr(self, key).append(value)

    def add_result_list(self,
                   param_value:list,
                   **kwargs):

        # 1. VALIDATION: Check for data consistency before modifying internal state
        self._verify_kwargs(**kwargs)
        self._verify_data_type(param_value, mode="list", **kwargs)

        # 2. ADD RESULTS: the new data to the internal state
        self.param_values.extend(param_value)
        for key, value in kwargs.items():
            getattr(self, key).extend(value)
    
    def add_result(self, param_value:float|int, results:base.NetworkResults, start_time:float=0.0):
        """"""

        if not isinstance(results, base.NetworkResults):
            raise TypeError(f"The value for 'results' must be 'base.NetworkResults' not {type(results).__name__}")

        self.param_values.append(param_value)

        mask = results.times >= start_time
        for var in self.measured_variables:
            match var:
                case "exc_rate_time_mean":
                    getattr(self, var).append(results.exc_rate_mean[mask].mean())
                case "exc_rate_time_std":
                    getattr(self, var).append(results.exc_rate_mean[mask].std())
                case "inh_rate_time_mean":
                    getattr(self, var).append(results.inh_rate_mean[mask].mean())
                case "inh_rate_time_std":
                    getattr(self, var).append(results.inh_rate_mean[mask].std())
                case "exc_voltage_time_mean":
                    getattr(self, var).append(results.exc_voltage_mean[mask].mean())
                case "exc_voltage_time_std":
                    getattr(self, var).append(results.exc_voltage_mean[mask].std())
                case "inh_voltage_time_mean":
                    getattr(self, var).append(results.inh_voltage_mean[mask].mean())
                case "inh_voltage_time_std":
                    getattr(self, var).append(results.inh_voltage_mean[mask].std())
                case "exc_adaptation_time_mean":
                    getattr(self, var).append(results.exc_adaptation_mean[mask].mean())
                case "exc_adaptation_time_std":
                    getattr(self, var).append(results.exc_adaptation_mean[mask].std())
                case "inh_adaptation_time_mean":
                    getattr(self, var).append(results.inh_adaptation_mean[mask].mean())
                case "inh_adaptation_time_std":
                    getattr(self, var).append(results.inh_adaptation_mean[mask].std())
                case _:
                    raise ValueError(f"Variable '{var}' not recognized for inspection.")

    def to_numpy_data(self):
        """Returns the collected results as a dictionary of NumPy arrays."""
        data = {
            self.inspected_param: np.array(self.param_values)
        }
        for var in self.measured_variables:
            data[var] = np.array(getattr(self, var))
        return data

    def _verify_kwargs(self, **kwargs):
        provided_vars = set(kwargs.keys())
        required_vars = set(self.measured_variables)
        if provided_vars != required_vars:
            missing = required_vars - provided_vars
            extra = provided_vars - required_vars
            error_message = f"Provided arguments are inconsistent with measured variables.\n"
            if missing:
                error_message += f"Missing required variables: {sorted(list(missing))}."
            if extra:
                error_message += f"Extra, unexpected variables provided: {sorted(list(extra))}."
            raise ValueError(error_message)

    def _verify_data_type(self, param_value, mode:str, **kwargs):
        """Verifies the type and structure of the data based on the mode."""

        if mode == 'point':
            if not isinstance(param_value, (float, int)):
                raise TypeError(f"'param_value' must be a float or int, not {type(param_value).__name__}")
            for key, value in kwargs.items():
                if not isinstance(value, (float, int)):
                    raise TypeError(f"Value for '{key}' must be a single number (float or int), not {type(value).__name__}")

        elif mode == 'list':
            if not isinstance(param_value, list):
                raise TypeError(f"'param_value' must be a list, not {type(param_value).__name__}")
            N = len(param_value)
            for key, value in kwargs.items():
                if not isinstance(value, list):
                    raise TypeError(f"Value for '{key}' must be a list when using 'add_result_list', not {type(value).__name__}")
                if len(value) != N:
                     raise ValueError(f"Length mismatch: 'param_value' has {N} items, but '{key}' has {len(value)}.")
        else:
            raise ValueError(f"Internal error: Unknown verification mode '{mode}'.")
