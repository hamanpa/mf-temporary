"""
Module for defining the base data structure for storing simulation results.

This module provides abstract base classes for results storage.
It is intended to be extended by specific simulation result classes.

This module also define constants
"""

import pickle

BIN_SIZE = 5  # [ms], size for making histograms
WINDOW_SIZE = 50  # [ms], size of the window for calculating floating averages


class Results:
    """
    Data structure for storing results from the SNN and MF simulations.

    Units:
    - time: [ms]
    - rate, frequency: [Hz]
    - adaptation, current: [pA]
    - conductance: [nS]
    - voltage: [mV] 
    """

    def save(self, filepath):
        """
        Save the results to a file.
        """
        print(f"Saving results to {filepath}")
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        file_size = filepath.stat().st_size / 1024 / 1024  # size in MB
        print(f"WARNING: File size: {int(file_size)} MB")


class NetworkResults(Results):
    """
    Data structure for storing results from the SNN and MF simulations.

    Units:
    - time: [ms]
    - rate, frequency: [Hz]
    - adaptation, current: [pA]
    - conductance: [nS]
    - voltage: [mV] 
    """

    def __init__(self, times, stim_params, net_params):
        self.times = times
        self.dt = times[1] - times[0]
        self.duration = times[-1] - times[0]
        
        self.stim_params = stim_params
        self.net_params = net_params

    @property
    def exc_rate_mean(self):
        """Get the excitatory activity."""
        if not hasattr(self, '_exc_rate_mean'):
            raise NotImplementedError("exc_rate_mean is not set. Please implement this in the subclass.")
        return self._exc_rate_mean

    @exc_rate_mean.setter
    def exc_rate_mean(self, value):
        self._exc_rate_mean = value

    @property
    def inh_rate_mean(self):
        """Get the inhibitory activity."""
        if not hasattr(self, '_inh_rate_mean'):
            raise NotImplementedError("inh_rate_mean is not set. Please implement this in the subclass.")
        return self._inh_rate_mean

    @inh_rate_mean.setter
    def inh_rate_mean(self, value):
        self._inh_rate_mean = value

    @property
    def exc_adaptation_mean(self):
        """Get the excitatory adaptation."""
        if not hasattr(self, '_exc_adaptation_mean'):
            raise NotImplementedError("exc_adaptation_mean is not set. Please implement this in the subclass.")
        return self._exc_adaptation_mean

    @exc_adaptation_mean.setter
    def exc_adaptation_mean(self, value):
        self._exc_adaptation_mean = value

    @property
    def inh_adaptation_mean(self):
        """Get the inhibitory adaptation."""
        if not hasattr(self, '_inh_adaptation_mean'):
            raise NotImplementedError("inh_adaptation_mean is not set. Please implement this in the subclass.")
        return self._inh_adaptation_mean

    @inh_adaptation_mean.setter
    def inh_adaptation_mean(self, value):
        self._inh_adaptation_mean = value   

    @property
    def exc_voltage_mean(self):
        """Get the excitatory membrane potential."""
        if not hasattr(self, '_exc_voltage_mean'):
            raise NotImplementedError("exc_voltage_mean is not set. Please implement this in the subclass.")
        return self._exc_voltage_mean

    @exc_voltage_mean.setter
    def exc_voltage_mean(self, value):
        self._exc_voltage_mean = value

    @property
    def inh_voltage_mean(self):
        """Get the inhibitory membrane potential."""
        if not hasattr(self, '_inh_voltage_mean'):
            raise NotImplementedError("inh_voltage_mean is not set. Please implement this in the subclass.")
        return self._inh_voltage_mean

    @inh_voltage_mean.setter
    def inh_voltage_mean(self, value):
        self._inh_voltage_mean = value

    def print_time_averaged(self, start_time=None, end_time=None):
        """Print the time averages of the results."""
        raise NotImplementedError("This method should be implemented in subclasses.")
