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
import copy
from typing import Dict

# NOTE: old imports, to be removed after the refactor
# NOTE: only issue is the removing of nans, not sure where they come from but they break the optimization, so we need to keep an eye on that and maybe add some checks in the data loading phase
# from ..utils.array_helpers import convert_to_array, convert_to_arrays, flatten_and_remove_nans, move_and_rescale

from .base import BaseTransferFunction
from .config import TransferFunctionConfig
from ..network_params.models import BiologicalParameters
from ..data_structures.neuron_simulation import SingleNeuronResults


class NeuroPSICustomTF(BaseTransferFunction):

    def __init__(self, neuron_name: str, network_params: BiologicalParameters, tf_params: TransferFunctionConfig):
        super().__init__(neuron_name, network_params, tf_params)
        
        # Instantiate the standalone physics calculator
        self.mpf = MembranePotentialFluctuations(neuron_name, network_params, ignore_stp=tf_params.tf_model.static_synapses)
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
            rates={
                "exc_neuron": exc_rate,
                "inh_neuron": inh_rate
            },
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
        tf_model_params = self.tf_params.tf_model

        x_mean = (voltage_mean - point.voltage_mean) / norm.voltage_mean
        x_std = (voltage_std - point.voltage_std) / norm.voltage_std
        x_tau = (voltage_tau_n - point.voltage_tau) / norm.voltage_tau

        v_eff = (
            coefs["P_0"] + 
            coefs["P_mean"] * x_mean + 
            coefs["P_std"] * x_std + 
            coefs["P_tau"] * x_tau
        )

        if tf_model_params.log_term:
            v_eff += coefs["P_log"] * np.log(conductance_mean / self.g_L)

        if tf_model_params.square_terms:
            v_eff += (
                coefs["P_mean_mean"] * (x_mean ** 2) +
                coefs["P_std_std"] * (x_std ** 2) +
                coefs["P_tau_tau"] * (x_tau ** 2) +
                coefs["P_mean_std"] * (x_mean * x_std) +
                coefs["P_mean_tau"] * (x_mean * x_tau) +
                coefs["P_std_tau"] * (x_std * x_tau)
            )


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
        rates = {
            "exc_neuron" : exc_rates,
            "inh_neuron" : inh_rates
        }

        if tf_model_params.adaptation:
            adaptation = single_neuron_results.adaptation_mean("nA").flatten()
        else:
            adaptation = None

        voltage_mean, voltage_std, voltage_tau, voltage_tau_n, conductance_mean = self.mpf.evaluate(rates, adaptation=adaptation)

        keys = ["P_0", "P_mean", "P_std", "P_tau"]
        if tf_model_params.log_term:
            keys.append("P_log")
        if tf_model_params.square_terms:
            keys.extend(["P_mean_mean", "P_std_std", "P_tau_tau", "P_mean_std", "P_mean_tau", "P_std_tau"])

        def array_to_dict(x: np.ndarray) -> dict:
            coefs = {}
            # coefs = dict(zip(keys, x))
            for k in ["P_0", "P_mean", "P_std", "P_tau", "P_log", "P_mean_mean", "P_std_std", "P_tau_tau", "P_mean_std", "P_mean_tau", "P_std_tau"]:
                coefs[k] = x[keys.index(k)] if k in keys else 0.0
                # coefs.setdefault(k, 0.0)
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


class MembranePotentialFluctuations:
    """
    This class should be used to compute the subthreshold membrane potential fluctuations.
    
    
    """
    def __init__(
        self, 
        neuron_name: str,
        network_params: BiologicalParameters,
        ignore_stp: bool = False
    ):
        self.neuron_name = neuron_name
        self.ignore_stp = ignore_stp

        self.tau_m = network_params.neurons[neuron_name].neuron_params.tau_m
        self.cm = network_params.neurons[neuron_name].neuron_params.cm
        self.g_L = network_params.neurons[neuron_name].neuron_params.g_L

        self.v_rest = network_params.neurons[neuron_name].neuron_params.v_rest

        self.a = network_params.neurons[neuron_name].neuron_params.a
        self.b = network_params.neurons[neuron_name].neuron_params.b
        self.tau_w = network_params.neurons[neuron_name].neuron_params.tau_w

        self.synapse_params = {}
        for source_neuron_name in network_params.synapses:

            syn_weight = network_params.synapses[source_neuron_name].syn_params.weight
            if not self.ignore_stp and network_params.synapses[source_neuron_name].syn_type == 'tsodyks_synapse':
                u = network_params.synapses[source_neuron_name].syn_params.U
                tau_rec = network_params.synapses[source_neuron_name].syn_params.tau_rec
                tau_fac = network_params.synapses[source_neuron_name].syn_params.tau_fac
            else:
                u = 1.0
                tau_rec = 0.0
                tau_fac = 0.0

            if network_params.neurons[source_neuron_name].neuron_type == "excitatory":
                e_rev = network_params.neurons[neuron_name].neuron_params.e_rev_E
                syn_tau = network_params.neurons[neuron_name].neuron_params.tau_syn_E
            elif network_params.neurons[source_neuron_name].neuron_type == "inhibitory":
                e_rev = network_params.neurons[neuron_name].neuron_params.e_rev_I
                syn_tau = network_params.neurons[neuron_name].neuron_params.tau_syn_I
            else:
                raise ValueError(f"Unknown neuron type: {network_params.neurons[source_neuron_name].neuron_type}")

            syn_num = int(network_params.network.size[source_neuron_name] * network_params.network.connectivity[neuron_name][source_neuron_name])

            self.synapse_params[source_neuron_name] = {
                'syn_weight': syn_weight,
                'u': u,
                'tau_rec': tau_rec,
                'tau_fac': tau_fac,
                'E_rev': e_rev,
                'syn_tau': syn_tau,
                'syn_num': syn_num
            }


    def _weight_effective(self, rate, syn_weight, u, tau_rec, tau_fac, **kwargs):
        """Calculates the effective synaptic weight considering STP."""

        # synaptic facilitation
        if tau_fac > 0:
            exp = np.zeros_like(rate)
            mask = rate > 0.
            exp[mask] = np.exp(-1 / (rate[mask]*1e-3 * tau_fac))
            u_steady = u / (1 - (1-u)*exp)
        else:
            u_steady = u * np.ones_like(rate)

        # synaptic depression
        if tau_rec > 0:
            exp = np.zeros_like(rate)
            mask = rate > 0.
            exp[mask] = np.exp(-1 / (rate[mask]*1e-3 * tau_rec))
            x_steady = (1-exp) / (1 -(1-u_steady)*exp)
        else:
            x_steady = np.ones_like(rate)
        
        # steady-state effective synaptic weight 
        return syn_weight * u_steady * x_steady

    def _conductance_mean(self, rate, effective_weight, syn_num, syn_tau, **kwargs):
        return rate * syn_num * (syn_tau * 1e-3) * effective_weight

    def _conductance_std(self, rate, effective_weight, syn_num, syn_tau, **kwargs):
        return np.sqrt(rate * syn_num * (syn_tau * 1e-3))* effective_weight

    def conductance_mean(self, rates, effective_weights):
        pop_conductances = []
        for neuron_name in rates:
            pop_conductances.append(self._conductance_mean(
                rate=rates[neuron_name],
                effective_weight=effective_weights[neuron_name],
                **self.synapse_params[neuron_name],
            ))
        return sum(pop_conductances) + self.g_L

    def tau_eff(self, rates, effective_weights):
        """Calculates the effective time constant of the neuron in [ms]."""
        # [nF / nS] = [s], thus factor 1e3 to convert to [ms]
        return self.cm / self.conductance_mean(rates, effective_weights) * 1e3

    def voltage_mean(self, rates, effective_weights, out_rate=None, adaptation=None):
        """Calculates the mean voltage of the neuron in [mV]."""
        if out_rate is None and adaptation is None:
            return self._voltage_mean_without_adaptation(rates, effective_weights)
        elif out_rate is None and adaptation is not None:
            return self._voltage_mean_with_adaptation(rates, effective_weights, adaptation)
        elif out_rate is not None and adaptation is None:
            return self._voltage_mean_with_out_rate(rates, effective_weights, out_rate)
        else:
            raise ValueError("out_rate and adaptation cannot be both not None")

    def _voltage_mean_without_adaptation(self, rates, effective_weights):
        """Calculates the mean voltage of the neuron without adaptation in [mV]."""
        pop_voltages = []
        for neuron_name in rates:
            pop_voltages.append(self._conductance_mean(
                rate=rates[neuron_name],
                effective_weight=effective_weights[neuron_name],
                **self.synapse_params[neuron_name],
            ) * self.synapse_params[neuron_name]['E_rev'])
        return (sum(pop_voltages) + self.g_L * self.v_rest) / self.conductance_mean(rates, effective_weights)

    def _voltage_mean_with_adaptation(self, rates, effective_weights, adaptation):
        """Calculates the mean voltage of the neuron with adaptation in [mV]."""
        pop_voltages = []
        for neuron_name in rates:
            pop_voltages.append(self._conductance_mean(
                rate=rates[neuron_name],
                effective_weight=effective_weights[neuron_name],
                **self.synapse_params[neuron_name],
            ) * self.synapse_params[neuron_name]['E_rev'])
        return (sum(pop_voltages) + self.g_L * self.v_rest - adaptation*1e3) / self.conductance_mean(rates, effective_weights)

    def _voltage_mean_with_out_rate(self, rates, effective_weights, out_rate):
        """Calculates the mean voltage of the neuron with nu_out in [mV]."""
        pop_voltages = []
        for neuron_name in rates:
            pop_voltages.append(self._conductance_mean(
                rate=rates[neuron_name],
                effective_weight=effective_weights[neuron_name],
                **self.synapse_params[neuron_name],
            ) * self.synapse_params[neuron_name]['E_rev'])

        numerator = sum(pop_voltages) + self.g_L * self.v_rest - out_rate * self.b * self.tau_w + self.a * self.v_rest
        denominator = self.conductance_mean(rates, effective_weights)+ self.a
        return numerator / denominator

    def voltage_std(self, rates, effective_weights, out_rate=None, adaptation=None):
        """Calculates the standard deviation of the voltage of the neuron in [mV]."""
        voltage_mean = self.voltage_mean(rates, effective_weights, out_rate=out_rate, adaptation=adaptation)
        conductance_mean = self.conductance_mean(rates, effective_weights)
        tau_eff = self.tau_eff(rates, effective_weights)

        terms = []
        for neuron_name in rates:
            syn_u = effective_weights[neuron_name] * (self.synapse_params[neuron_name]['E_rev'] - voltage_mean) / conductance_mean
            syn_tau = self.synapse_params[neuron_name]['syn_tau']
            syn_num = self.synapse_params[neuron_name]['syn_num']

            terms.append(syn_num * (rates[neuron_name] * 1e-3) * (syn_u * syn_tau)**2 / (2 * (tau_eff + syn_tau)))

        return np.sqrt(sum(terms))

    def voltage_tau(self, rates, effective_weights, out_rate=None, adaptation=None):
        """Calculates the effective time constant of the voltage fluctuations of the neuron in [ms].
        """
        voltage_mean = self.voltage_mean(rates, effective_weights, out_rate=out_rate, adaptation=adaptation)
        conductance_mean = self.conductance_mean(rates, effective_weights)
        tau_eff = self.tau_eff(rates, effective_weights)

        numerator_terms = []
        denominator_terms = []
        for neuron_name in rates:
            syn_u = effective_weights[neuron_name] * (self.synapse_params[neuron_name]['E_rev'] - voltage_mean) / conductance_mean
            syn_tau = self.synapse_params[neuron_name]['syn_tau']
            syn_num = self.synapse_params[neuron_name]['syn_num']

            mask = rates[neuron_name] > 0
            rates[neuron_name][~mask] = 1e-9  # Avoid division by zero for zero rates
            term = syn_num * (rates[neuron_name] * 1e-3) * (syn_u * syn_tau)**2

            # term = syn_num * (rates[neuron_name] * 1e-3 + 1e-9) * (syn_u * syn_tau)**2

            numerator_terms.append(term)
            denominator_terms.append(term / (tau_eff + syn_tau))


        return sum(numerator_terms) / sum(denominator_terms)

    def evaluate(
            self, 
            rates: Dict[str, np.ndarray|float], 
            effective_weights: Dict[str, np.ndarray|float]=None,
            out_rate: np.ndarray|float=None, 
            adaptation: np.ndarray|float=None,
            ) -> tuple:

        # 1. update effective weights
        effective_weights = effective_weights or {}
        for neuron_name in rates:
            if neuron_name not in effective_weights:
                effective_weights[neuron_name] = self._weight_effective(rates[neuron_name], **self.synapse_params[neuron_name])

        voltage_mean = self.voltage_mean(rates, effective_weights, out_rate=out_rate, adaptation=adaptation)
        voltage_std = self.voltage_std(rates, effective_weights, out_rate=out_rate, adaptation=adaptation)
        voltage_tau = self.voltage_tau(rates, effective_weights, out_rate=out_rate, adaptation=adaptation)
        voltage_tau_n = voltage_tau / self.tau_m
        conductance_mean = self.conductance_mean(rates, effective_weights)
        return voltage_mean, voltage_std+1e-9, voltage_tau, voltage_tau_n, conductance_mean

