"""
This file introduces the transfer function fitting workflow for the mean-field model.

The fitting is based on the following papers
[1] Y. Zerlaut, S. Chemla, F. Chavane, and A. Destexhe, 
“Modeling mesoscopic cortical dynamics using a mean-field model of conductance-based
 networks of adaptive exponential integrate-and-fire neurons,” 
J Comput Neurosci, vol. 44, no. 1, pp. 45–61, Feb. 2018, doi: 10.1007/s10827-017-0668-2.

[2] M. di Volo, A. Romagnoni, C. Capone, and A. Destexhe, 
“Biologically Realistic Mean-Field Models of Conductance-Based Networks of Spiking Neurons with Adaptation,” 
Neural Computation, vol. 31, no. 4, pp. 653–680, Apr. 2019, doi: 10.1162/neco_a_01173.

These papers suggest 'semi-analytical' approach to the transfer function fitting, 
which is based on the mean potential fluctuations. That is, the input rates 
(nu_e, nu_i) are used to compute the mean potential fluctuations 
(mu_V, sigma_V, tau_V, tau_VN, mu_G), by analytical methods. Then, the fluctuations
are fitted to the output rate (nu_out) of the neuron, using polynomial fitting and
complementary error function (erfc) to compute the transfer function.

The fitting goes in two steps:
1. There is fitting of the effective potential (V_eff) based on the mean potential fluctuations.
2. Then, the transfer function is fitted based on the effective potential and the output rate.

"""

from pathlib import Path
import numpy as np
from scipy.special import erfc, erfcinv
from scipy.optimize import minimize

# NOTE: old imports, to be removed after the refactor
# NOTE: only issue is the removing of nans, not sure where they come from but they break the optimization, so we need to keep an eye on that and maybe add some checks in the data loading phase
# from ..utils.array_helpers import convert_to_array, convert_to_arrays, flatten_and_remove_nans, move_and_rescale

from .base import BaseTransferFunction
from .config import TransferFunctionConfig
from ..network_params.models import BiologicalParameters
from ..data_structures.single_neuron import SingleNeuronResults


class NeuroPSICustomTF(BaseTransferFunction):

    def __init__(self, neuron_name: str, network_params: BiologicalParameters, tf_params: TransferFunctionConfig):
        super().__init__(neuron_name, network_params, tf_params)
        
        # Instantiate the standalone physics calculator
        self.mpf = MembranePotentialFluctuations(neuron_name, network_params)
        self.g_L = network_params.neurons[neuron_name].neuron_params.g_L

    def required_inputs(self) -> list[str]:
        """Dynamically declares required inputs based on the configuration."""
        inputs = ["exc_rate", "inh_rate"]
        if self.tf_params.tf_model.adaptation:
            inputs.append("adaptation")
        return inputs

    def evaluate(self, **kwargs) -> np.ndarray:
        """
        The core mapping function: F(v_e, v_i, [w]) -> v_out.
        """
        exc_rate = kwargs["exc_rate"]
        inh_rate = kwargs["inh_rate"]
        adaptation = kwargs.get("adaptation", None)

        # 1. Compute theoretical subthreshold fluctuations
        mu_V, sigma_V, tau_V, tau_VN, mu_G = self.mpf.evaluate(
            exc_rate=exc_rate, 
            inh_rate=inh_rate, 
            adaptation=adaptation
        )

        # 2. Compute effective threshold potential
        v_eff = self._evaluate_effective_potential(mu_V, sigma_V, tau_VN, mu_G)

        # 3. Prevent division by zero
        sigma_V_safe = np.clip(sigma_V, 1e-9, None)

        # 4. Final transfer function mapping (return rate in Hz)
        return 1 / (2 * tau_V * 1e-3) * erfc((v_eff - mu_V) / (np.sqrt(2) * sigma_V_safe))

    def _evaluate_effective_potential(
        self, 
        voltage_mean: np.ndarray, 
        voltage_std: np.ndarray, 
        voltage_tau_n: np.ndarray, 
        conductance_mean: np.ndarray, 
        coefs: dict = None
    ) -> np.ndarray:
        """
        Computes the phenomenological (effective) threshold V_eff using the polynomial expansion


        Accepts optional `coefs` for use during the fitting loop; otherwise uses self.fitted_params.
        """

        if coefs is None:
            coefs = self.fitted_params

        point = self.tf_params.expansion_point
        norm = self.tf_params.expansion_norm
        model_flags = self.tf_params.tf_model

        x_mean = (voltage_mean - point.voltage_mean) / norm.voltage_mean
        x_std = (voltage_std - point.voltage_std) / norm.voltage_std
        x_tau = (voltage_tau_n - point.voltage_tau) / norm.voltage_tau

        v_eff = (
            coefs["P_0"] + 
            coefs["P_mean"] * x_mean + 
            coefs["P_std"] * x_std + 
            coefs["P_tau"] * x_tau
        )

        if getattr(model_flags, "square_terms", False):
            v_eff += (
                coefs["P_mean_mean"] * (x_mean ** 2) +
                coefs["P_std_std"] * (x_std ** 2) +
                coefs["P_tau_tau"] * (x_tau ** 2) +
                coefs["P_mean_std"] * (x_mean * x_std) +
                coefs["P_mean_tau"] * (x_mean * x_tau) +
                coefs["P_std_tau"] * (x_std * x_tau)
            )

        if getattr(model_flags, "log_term", False):
            v_eff += coefs["P_log"] * np.log(conductance_mean / self.g_L)

        return v_eff

    def _get_target_v_eff(
        self, 
        out_rate: np.ndarray, 
        voltage_mean: np.ndarray, 
        voltage_std: np.ndarray, 
        voltage_tau: np.ndarray
    ) -> np.ndarray:
        """
        Computes the target V_eff directly from output rate data. 
        Used strictly during the fitting process.
        """
        out_rate_safe = np.clip(out_rate, 1e-9, None)
        return np.sqrt(2) * voltage_std * erfcinv(2 * voltage_tau * out_rate_safe * 1e-3) + voltage_mean

    def fit(self, single_neuron_results: SingleNeuronResults, **kwargs) -> dict:
        """
        Calibrates the transfer function using a two-step optimization process.
        """
        tf_model_params = self.tf_params.tf_model
        out_rate_min = self.tf_params.out_rate_min
        out_rate_max = self.tf_params.out_rate_max

        # 1. Extract and flatten SNN data
        exc_rates = single_neuron_results.exc_rate_grid("Hz").flatten()
        inh_rates = single_neuron_results.inh_rate_grid("Hz").flatten()
        out_rates = single_neuron_results.out_rate_mean("Hz").flatten()

        if getattr(tf_model_params, "adaptation", False):
            adaptation = single_neuron_results.adaptation_mean("nA").flatten()
        else:
            adaptation = None

        voltage_mean, voltage_std, voltage_tau, voltage_tau_n, conductance_mean = self.mpf.evaluate(exc_rates, inh_rates, adaptation)

        keys = ["P_0", "P_mean", "P_std", "P_tau"]
        if getattr(tf_model_params, "log_term", False):
            keys.append("P_log")
        if getattr(tf_model_params, "square_terms", False):
            keys.extend(["P_mean_mean", "P_std_std", "P_tau_tau", "P_mean_std", "P_mean_tau", "P_std_tau"])

        def array_to_dict(x: np.ndarray) -> dict:
            coefs = dict(zip(keys, x))
            for k in ["P_0", "P_mean", "P_std", "P_tau", "P_log", "P_mean_mean", "P_std_std", "P_tau_tau", "P_mean_std", "P_mean_tau", "P_std_tau"]:
                coefs.setdefault(k, 0.0)
            return coefs

        # ==========================================
        # STEP 1: Fit Effective Potential (V_eff)
        # ==========================================

        mask1 = (out_rates > out_rate_min) & (out_rates < out_rate_max)
        
        v_eff_target = self._get_target_v_eff(
            out_rate=out_rates[mask1], 
            voltage_mean=voltage_mean[mask1], 
            voltage_std=voltage_std[mask1], 
            voltage_tau=voltage_tau[mask1]
        )

        # Initial guess: [Mean V_eff] + [1.0 for linear] + [0.0 for squares]
        x0 = [v_eff_target.mean()] + [1.0] * 3
        if getattr(tf_model_params, "log_term", False): x0 += [1.0]
        if getattr(tf_model_params, "square_terms", False): x0 += [0.0] * 6

        def obj_veff(x):
            guess_coefs = array_to_dict(x)
            v_eff_pred = self._evaluate_effective_potential(
                voltage_mean=voltage_mean[mask1], 
                voltage_std=voltage_std[mask1], 
                voltage_tau_n=voltage_tau_n[mask1], 
                conductance_mean=conductance_mean[mask1], 
                coefs=guess_coefs
            )
            return np.mean((v_eff_target - v_eff_pred) ** 2)

        opts1 = self.tf_params.V_eff_fitting
        res1 = minimize(obj_veff, x0, method=opts1.method, options=opts1.options)

        # ==========================================
        # STEP 2: Fit Transfer Function (nu_out)
        # ==========================================

        mask2 = out_rates < out_rate_max

        def obj_tf(x):
            self.fitted_params = array_to_dict(x)
            
            out_rate_pred = self.evaluate(
                exc_rate=exc_rates[mask2], 
                inh_rate=inh_rates[mask2], 
                adaptation=adaptation[mask2] if adaptation is not None else None
            )
            return np.mean((out_rates[mask2] - out_rate_pred) ** 2)

        opts2 = self.tf_params.TF_fitting
        res2 = minimize(obj_tf, res1.x, method=opts2.method, options=opts2.options)

        # ==========================================
        # Finalize and Return
        # ==========================================
        self.fitted_params = array_to_dict(res2.x)
        self.is_fitted = True

        # Compute final MSE across all valid points for the metric report
        final_error = obj_tf(res2.x) 

        return {
            "V_eff_MSE": res1.fun,
            "TF_MSE": final_error,
            "V_eff_Success": res1.success,
            "TF_Success": res2.success
        }


# TODO: add Tsodyks synapse
class MembranePotentialFluctuations:
    """
    This class should be used to compute the subthreshold membrane potential fluctuations.
    
    
    """
    def __init__(
        self, 
        neuron_name: str,
        network_params: BiologicalParameters,

    ):
        self.neuron_name = neuron_name

        self.exc_syn_tau = network_params.neurons[neuron_name].neuron_params.tau_syn_E
        self.exc_syn_v = network_params.neurons[neuron_name].neuron_params.e_rev_E

        self.inh_syn_tau = network_params.neurons[neuron_name].neuron_params.tau_syn_I
        self.inh_syn_v = network_params.neurons[neuron_name].neuron_params.e_rev_I

        self.tau_m = network_params.neurons[neuron_name].neuron_params.tau_m
        self.cm = network_params.neurons[neuron_name].neuron_params.cm
        self.g_L = network_params.neurons[neuron_name].neuron_params.g_L

        self.v_rest = network_params.neurons[neuron_name].neuron_params.v_rest

        self.a = network_params.neurons[neuron_name].neuron_params.a
        self.b = network_params.neurons[neuron_name].neuron_params.b
        self.tau_w = network_params.neurons[neuron_name].neuron_params.tau_w


        self.exc_syn_num = int(network_params.network.size[network_params.exc_neuron_name] * network_params.network.connectivity[neuron_name][network_params.exc_neuron_name])
        self.inh_syn_num = int(network_params.network.size[network_params.inh_neuron_name] * network_params.network.connectivity[neuron_name][network_params.inh_neuron_name])

        self.exc_syn_type = network_params.synapses[network_params.exc_neuron_name].syn_type
        self.inh_syn_type = network_params.synapses[network_params.inh_neuron_name].syn_type

        self.exc_syn_weight = network_params.synapses[network_params.exc_neuron_name].syn_params.weight
        self.inh_syn_weight = network_params.synapses[network_params.inh_neuron_name].syn_params.weight


        if self.exc_syn_type == 'tsodyks_synapse':
            raise NotImplementedError("Tsodyks synapse is not yet implemented in the mean potential fluctuations calculation")
            # self.u_e = params['exc_synapses']['syn_params']['U']
            # self.tau_rec_e = params['exc_synapses']['syn_params']['tau_rec']
        if self.inh_syn_type == 'tsodyks_synapse':
            raise NotImplementedError("Tsodyks synapse is not yet implemented in the mean potential fluctuations calculation")
            # self.u_i = params['inh_synapses']['syn_params']['U']
            # self.tau_rec_i = params['inh_synapses']['syn_params']['tau_rec']


    def exc_conductance_mean(self, exc_rate):
        """Calculates the mean excitatory conductance in [nS]."""
        # [Hz * number * ms * nS] = [pS], thus factor 1e-3 to convert to [nS]
        return exc_rate * self.exc_syn_num * (self.exc_syn_tau * 1e-3) * self.exc_syn_weight

    def exc_conductance_std(self, exc_rate):
        """Calculates the standard deviation of the excitatory conductance in [nS]."""
        # factor 1e-3 to have the term inside square root unitless, thus the result is in [nS]
        return np.sqrt(exc_rate * self.exc_syn_num * (self.exc_syn_tau * 1e-3) / 2) * self.exc_syn_weight

    def inh_conductance_mean(self, inh_rate):
        """Calculates the mean inhibitory conductance in [nS]."""
        # [Hz * number * ms * nS] = [pS], thus factor 1e-3 to convert to [nS]
        return inh_rate * self.inh_syn_num * (self.inh_syn_tau * 1e-3) * self.inh_syn_weight

    def inh_conductance_std(self, inh_rate):
        """Calculates the standard deviation of the inhibitory conductance in [nS]."""
        # factor 1e-3 to have the term inside square root unitless, thus the result is in [nS]
        return np.sqrt(inh_rate * self.inh_syn_num * (self.inh_syn_tau * 1e-3) / 2) * self.inh_syn_weight

    def conductance_mean(self, exc_rate, inh_rate):
        """Calculates the mean total conductance in [nS]."""
        return self.exc_conductance_mean(exc_rate) + self.inh_conductance_mean(inh_rate) + self.g_L

    def tau_eff(self, exc_rate, inh_rate):
        """Calculates the effective time constant of the neuron in [ms]."""
        # [nF / nS] = [s], thus factor 1e3 to convert to [ms]
        return self.cm / self.conductance_mean(exc_rate, inh_rate) * 1e3

    def voltage_mean(self, exc_rate, inh_rate, out_rate=None, adaptation=None):
        """Calculates the mean voltage of the neuron in [mV]."""
        if out_rate is None and adaptation is None:
            return self._voltage_mean_without_adaptation(exc_rate, inh_rate)
        elif out_rate is None and adaptation is not None:
            return self._voltage_mean_with_adaptation(exc_rate, inh_rate, adaptation)
        elif out_rate is not None and adaptation is None:
            return self._voltage_mean_with_out_rate(exc_rate, inh_rate, out_rate)
        else:
            raise ValueError("out_rate and adaptation cannot be both not None")
    
    def _voltage_mean_without_adaptation(self, exc_rate, inh_rate):
        """Calculates the mean voltage of the neuron without adaptation in [mV]."""
        exc_voltage = self.exc_conductance_mean(exc_rate) * self.exc_syn_v
        inh_voltage = self.inh_conductance_mean(inh_rate) * self.inh_syn_v
        return (exc_voltage + inh_voltage + self.g_L * self.v_rest) / self.conductance_mean(exc_rate, inh_rate)


    def _voltage_mean_with_adaptation(self, exc_rate, inh_rate, adaptation):
        """Calculates the mean voltage of the neuron with adaptation in [mV]."""
        exc_voltage = self.exc_conductance_mean(exc_rate) * self.exc_syn_v
        inh_voltage = self.inh_conductance_mean(inh_rate) * self.inh_syn_v
        return (exc_voltage + inh_voltage + self.g_L * self.v_rest - adaptation) / self.conductance_mean(exc_rate, inh_rate)

    def _voltage_mean_with_out_rate(self, exc_rate, inh_rate, out_rate):
        """Calculates the mean voltage of the neuron with nu_out in [mV]."""
        exc_voltage = self.exc_conductance_mean(exc_rate) * self.exc_syn_v
        inh_voltage = self.inh_conductance_mean(inh_rate) * self.inh_syn_v
        
        numerator = exc_voltage + inh_voltage + self.g_L * self.v_rest - out_rate * self.b * self.tau_w + self.a * self.v_rest
        denominator = self.conductance_mean(exc_rate, inh_rate)+ self.a
        return numerator / denominator

    def voltage_std(self, exc_rate, inh_rate, out_rate=None, adaptation=None):
        """Calculates the standard deviation of the voltage of the neuron in [mV]."""
        voltage_mean = self.voltage_mean(exc_rate, inh_rate, out_rate=out_rate, adaptation=adaptation)
        conductance_mean = self.conductance_mean(exc_rate, inh_rate)
        tau_eff = self.tau_eff(exc_rate, inh_rate)

        exc_syn_u = self.exc_syn_weight * (self.exc_syn_v - voltage_mean) / conductance_mean  # [mV]
        inh_syn_u = self.inh_syn_weight * (self.inh_syn_v - voltage_mean) / conductance_mean  # [mV]

        exc_term = self.exc_syn_num * (exc_rate * 1e-3) * (exc_syn_u * self.exc_syn_tau)**2 / (2 * (tau_eff + self.exc_syn_tau))
        inh_term = self.inh_syn_num * (inh_rate * 1e-3) * (inh_syn_u * self.inh_syn_tau)**2 / (2 * (tau_eff + self.inh_syn_tau))
        voltage_std = np.sqrt(exc_term + inh_term)
        return voltage_std

    def voltage_tau(self, exc_rate, inh_rate, out_rate=None, adaptation=None):
        """Calculates the effective time constant of the voltage fluctuations of the neuron in [ms].
        """
        voltage_mean = self.voltage_mean(exc_rate, inh_rate, out_rate=out_rate, adaptation=adaptation)
        conductance_mean = self.conductance_mean(exc_rate, inh_rate)
        tau_eff = self.tau_eff(exc_rate, inh_rate)

        exc_syn_u = self.exc_syn_weight * (self.exc_syn_v - voltage_mean) / conductance_mean  # [mV]
        inh_syn_u = self.inh_syn_weight * (self.inh_syn_v - voltage_mean) / conductance_mean  # [mV]

        exc_term = self.exc_syn_num * (exc_rate * 1e-3) * (exc_syn_u * self.exc_syn_tau)**2
        inh_term = self.inh_syn_num * (inh_rate * 1e-3) * (inh_syn_u * self.inh_syn_tau)**2
        
        # NOTE: following is equivallent to:
        # exc_term[exc_term<1e-9]=1e-9  
        # inh_term[inh_term<1e-9]=1e-9
        # but allows a float as input, goal is to avoid division by zero
        exc_term = np.clip(exc_term, 1e-9, None)
        inh_term = np.clip(inh_term, 1e-9, None)

        voltage_tau = (exc_term + inh_term) / (exc_term / (tau_eff + self.exc_syn_tau) + inh_term / (tau_eff + self.inh_syn_tau))
        return voltage_tau

    def evaluate(self, exc_rate, inh_rate, out_rate=None, adaptation=None) -> tuple:
        voltage_mean = self.voltage_mean(exc_rate, inh_rate, out_rate=out_rate, adaptation=adaptation)
        voltage_std = self.voltage_std(exc_rate, inh_rate, out_rate=out_rate, adaptation=adaptation)
        voltage_tau = self.voltage_tau(exc_rate, inh_rate, out_rate=out_rate, adaptation=adaptation)
        voltage_tau_n = voltage_tau / self.tau_m
        conductance_mean = self.conductance_mean(exc_rate, inh_rate)
        return voltage_mean, voltage_std, voltage_tau, voltage_tau_n, conductance_mean

