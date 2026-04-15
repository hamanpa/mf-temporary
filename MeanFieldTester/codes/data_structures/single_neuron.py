"""
This module defines the data structure for storing results from a single neuron simulation.

"""

import numpy as np
import warnings

from dataclasses import dataclass
import pickle

import codes.data_structures.base as base
from codes.transfer_function import MPF_with_nu_out


@dataclass(frozen=True)
class DataclassResults:

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

@dataclass(frozen=True)
class SingleNeuronResults(DataclassResults):
    # All the results are 2D arrays with the indexing (exc_rate, inh_rate)

    simulator_name: str = None
    neuron_name: str = None
    neuron_params: dict = None
    neuron_sim_params: dict = None

    spikes: np.ndarray = None
    exc_rate_grid: np.ndarray = None
    inh_rate_grid: np.ndarray = None
    out_rate_mean: np.ndarray = None
    out_rate_std: np.ndarray = None
    adaptation_mean: np.ndarray = None
    adaptation_std: np.ndarray = None
    voltage_mean: np.ndarray = None
    voltage_std: np.ndarray = None
    voltage_tau: np.ndarray = None
    exc_conductance_mean: np.ndarray = None
    exc_conductance_std: np.ndarray = None
    inh_conductance_mean: np.ndarray = None
    inh_conductance_std: np.ndarray = None


    @property
    def exc_drive_mean(self):
        warnings.warn(
            "The attribute 'exc_drive_mean' is deprecated and will be removed in a future version. "
            "Please use 'exc_rate_grid' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.exc_rate_grid

    @property
    def nu_e(self):
        warnings.warn(
            "The attribute 'nu_e' is deprecated and will be removed in a future version. "
            "Please use 'exc_rate_grid' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.exc_rate_grid

    @property
    def nu_i(self):
        warnings.warn(
            "The attribute 'nu_i' is deprecated and will be removed in a future version. "
            "Please use 'inh_rate_grid' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.inh_rate_grid

    @property
    def inh_drive_mean(self):
        warnings.warn(
            "The attribute 'inh_drive_mean' is deprecated and will be removed in a future version. "
            "Please use 'inh_rate_grid' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.inh_rate_grid

    @property
    def nu_out_mean(self):
        warnings.warn(
            "The attribute 'nu_out_mean' is deprecated and will be removed in a future version. "
            "Please use 'out_rate_mean' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.out_rate_mean
    
    @property
    def nu_out_std(self):
        warnings.warn(
            "The attribute 'nu_out_std' is deprecated and will be removed in a future version. "
            "Please use 'out_rate_std' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.out_rate_std

    @property
    def w_mean(self):
        warnings.warn(
            "The attribute 'w_mean' is deprecated and will be removed in a future version. "
            "Please use 'adaptation_mean' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.adaptation_mean

    @property
    def w_std(self):
        warnings.warn(
            "The attribute 'w_std' is deprecated and will be removed in a future version. "
            "Please use 'adaptation_std' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.adaptation_std
 
    @property
    def v_mean(self):
        warnings.warn(
            "The attribute 'v_mean' is deprecated and will be removed in a future version. "
            "Please use 'voltage_mean' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.voltage_mean

    @property
    def v_std(self):
        warnings.warn(
            "The attribute 'v_std' is deprecated and will be removed in a future version. "
            "Please use 'voltage_std' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.voltage_std

    @property
    def v_tau(self):
        warnings.warn(
            "The attribute 'v_tau' is deprecated and will be removed in a future version. "
            "Please use 'voltage_tau' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.voltage_tau

    @property
    def gsyn_e_mean(self):
        warnings.warn(
            "The attribute 'gsyn_e_mean' is deprecated and will be removed in a future version. "
            "Please use 'exc_conductance_mean' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.exc_conductance_mean

    @property
    def gsyn_e_std(self):
        warnings.warn(
            "The attribute 'gsyn_e_std' is deprecated and will be removed in a future version. "
            "Please use 'exc_conductance_std' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.exc_conductance_std

    @property
    def gsyn_i_mean(self):
        warnings.warn(
            "The attribute 'gsyn_i_mean' is deprecated and will be removed in a future version. "
            "Please use 'inh_conductance_mean' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.inh_conductance_mean

    @property
    def gsyn_i_std(self):
        warnings.warn(
            "The attribute 'gsyn_i_std' is deprecated and will be removed in a future version. "
            "Please use 'inh_conductance_std' instead.",
            category=FutureWarning,
            stacklevel=2
        )
        return self.inh_conductance_std








class AdExNeuronTheoreticalResults(base.Results):
    """Data structure for storing theoretical results from an AdEx neuron simulation.
    
    This is used to compare with the results from the single neuron simulation.

    Units
    -----
    - nu_e, nu_i : [Hz]
    - nu_out : [Hz]
    - w_mean, w_std : [pA]
    - v_mean, v_std : [mV]
    - v_tau : [ms]
    """

    def __init__(self, neuron_name, neuron_params, 
                 exc_drive, inh_drive, out_rate):
        self.neuron_name = neuron_name
        self.neuron_params = neuron_params

        self.exc_drive_mean = exc_drive
        self.inh_drive_mean = inh_drive
        self.out_rate_mean = out_rate
        
 
    @property
    def exc_conductance_mean(self):
        rate = self.exc_drive_mean
        syn_num = self.neuron_params['exc_synapses']['number']
        tau = self.neuron_params['neuron_params']['tau_syn_E'] *1e-3
        weight = self.neuron_params['exc_synapses']['syn_params']['weight']

        return rate * syn_num * tau * weight  # Hz * dimless * s * nS

    @property
    def exc_conductance_std(self):
        rate = self.exc_drive_mean
        syn_num = self.neuron_params['exc_synapses']['number']
        tau = self.neuron_params['neuron_params']['tau_syn_E'] *1e-3
        weight = self.neuron_params['exc_synapses']['syn_params']['weight']

        return np.sqrt(rate* syn_num * tau / 2) * weight

    @property
    def inh_conductance_mean(self):
        rate = self.inh_drive_mean
        syn_num = self.neuron_params['inh_synapses']['number']
        tau = self.neuron_params['neuron_params']['tau_syn_I'] *1e-3
        weight = self.neuron_params['inh_synapses']['syn_params']['weight']

        return rate * syn_num * tau * weight
    
    @property
    def inh_conductance_std(self):
        rate = self.inh_drive_mean
        syn_num = self.neuron_params['inh_synapses']['number']
        tau = self.neuron_params['neuron_params']['tau_syn_I'] *1e-3
        weight = self.neuron_params['inh_synapses']['syn_params']['weight']

        return np.sqrt(rate* syn_num * tau / 2) * weight
    
    @property
    def adaptation_mean(self):
        a = self.neuron_params['neuron_params']['a']
        b = self.neuron_params['neuron_params']['b']
        v_rest = self.neuron_params['neuron_params']['v_rest']
        tau_w = self.neuron_params['neuron_params']['tau_w']
        
        # adaptation should be in pa
        # [b] = nA
        # out_rate_mean is in Hz
        # [tau_w] = ms
        # [a] = pA/mV = nS
        # [v_rest] = mV

        return b*self.out_rate_mean*tau_w + a * (self.voltage_mean - v_rest)

    @property
    def conductance_mean(self):
        # conductance is in nS
        g = self.neuron_params['neuron_params']['cm'] / self.neuron_params['neuron_params']['tau_m']*1e3
        return self.exc_conductance_mean + self.inh_conductance_mean + g

    @property
    def tau_eff(self):
        return self.neuron_params['neuron_params']['cm'] / self.conductance_mean * 1e3  # convert to ms

    @property
    def voltage_mean(self):
        method = "implicit"

        exc_voltage = self.exc_conductance_mean*self.neuron_params['neuron_params']['e_rev_E']
        inh_voltage = self.inh_conductance_mean*self.neuron_params['neuron_params']['e_rev_I']
        
        g_rest = self.neuron_params['neuron_params']['cm'] / self.neuron_params['neuron_params']['tau_m']*1e3
        v = g_rest*self.neuron_params['neuron_params']['v_rest']

        if method == "explicit":
            return (exc_voltage + inh_voltage + v - self.adaptation_mean) / self.conductance_mean
        if method == "implicit":
            a = self.neuron_params['neuron_params']['a']
            b = self.neuron_params['neuron_params']['b']
            v_rest = self.neuron_params['neuron_params']['v_rest']
            tau_w = self.neuron_params['neuron_params']['tau_w']

            return (exc_voltage + inh_voltage + v - b*self.out_rate_mean*tau_w + a*v_rest)/(self.conductance_mean + a)

    @property
    def voltage_std(self):
        exc_weight = self.neuron_params['exc_synapses']['syn_params']['weight']
        exc_voltage = self.neuron_params['neuron_params']['e_rev_E']
        exc_num = self.neuron_params['exc_synapses']['number']
        exc_tau = self.neuron_params['neuron_params']['tau_syn_E']

        exc_u = exc_weight / self.conductance_mean * (exc_voltage - self.voltage_mean)
        exc_std = exc_num * self.exc_drive_mean*1e3 * ((exc_u * exc_tau )**2)/(2*(self.tau_eff + exc_tau))

        inh_weight = self.neuron_params['inh_synapses']['syn_params']['weight']
        inh_voltage = self.neuron_params['neuron_params']['e_rev_I']
        inh_num = self.neuron_params['inh_synapses']['number']
        inh_tau = self.neuron_params['neuron_params']['tau_syn_I']

        inh_u = inh_weight / self.conductance_mean * (inh_voltage - self.voltage_mean)
        inh_std = inh_num * self.inh_drive_mean*1e3 * ((inh_u * inh_tau )**2)/(2*(self.tau_eff + inh_tau))

        return np.sqrt(exc_std + inh_std)

    @property
    def voltage_tau(self):
        exc_weight = self.neuron_params['exc_synapses']['syn_params']['weight']
        exc_voltage = self.neuron_params['neuron_params']['e_rev_E']
        exc_num = self.neuron_params['exc_synapses']['number']
        exc_tau = self.neuron_params['neuron_params']['tau_syn_E']

        exc_u = exc_weight / self.conductance_mean * (exc_voltage - self.voltage_mean)
        exc_voltage_tau = exc_num * self.exc_drive_mean*1e3 * ((exc_u * exc_tau )**2)

        inh_weight = self.neuron_params['inh_synapses']['syn_params']['weight']
        inh_voltage = self.neuron_params['neuron_params']['e_rev_I']
        inh_num = self.neuron_params['inh_synapses']['number']
        inh_tau = self.neuron_params['neuron_params']['tau_syn_I']

        inh_u = inh_weight / self.conductance_mean * (inh_voltage - self.voltage_mean)
        inh_voltage_tau = inh_num * self.inh_drive_mean*1e3 * ((inh_u * inh_tau )**2)

        return (exc_voltage_tau + inh_voltage_tau) / (exc_voltage_tau/(self.tau_eff + exc_tau) + inh_voltage_tau/(self.tau_eff + inh_tau))
