import numpy as np
import tvb.simulator.lab as lab

from ....network_params.models import BiologicalParameters
from ....network_params.translators import TranslationRule, translate_params
from ...config import MeanFieldSimulationConfig
from .neuropsi_models import Zerlaut_adaptation_first_order, Zerlaut_adaptation_second_order, ZerlautMatteo_adaptation_first_order, ZerlautMatteo_adaptation_second_order
# from .stp_models import DiVolo_STP_second_order
from ....utils.array_helpers import convert_to_array



MODELS_REGISTRY = {
    "zerlaut2018.first_order": Zerlaut_adaptation_first_order,
    "zerlaut2018.second_order": Zerlaut_adaptation_second_order,
    "divolo2019.first_order": ZerlautMatteo_adaptation_first_order,
    "divolo2019.second_order": ZerlautMatteo_adaptation_second_order,

    # asymptotic STP, no dynamics, just a static effective synaptic strength
    "stp_stationary.first_order": None,  
    "stp_stationary.second_order": None,  

    "stp_dynamical.first_order": None,
    "stp_dynamical.second_order": None,
}

LEGACY_MODELS_REGISTRY = {
    "zerlaut2018.first_order",
    "zerlaut2018.second_order",
    "divolo2019.first_order",
    "divolo2019.second_order",
}

TVB_NEUROPSI_EXC_NEURON_MAPPING = {
    "g_L": TranslationRule("g_L", sim_unit="nS"),
    "C_m": TranslationRule("cm", sim_unit="pF"),
    "E_L_e": TranslationRule("v_rest", sim_unit="mV"),
    "b_e": TranslationRule("b", sim_unit="pA"),
    "a_e": TranslationRule("a", sim_unit="nS"),
    "tau_w_e": TranslationRule("tau_w", sim_unit="ms"),
    "E_e": TranslationRule("e_rev_E", sim_unit="mV"),
    "E_i": TranslationRule("e_rev_I", sim_unit="mV"),
    "tau_e": TranslationRule("tau_syn_E", sim_unit="ms"),
    "tau_i": TranslationRule("tau_syn_I", sim_unit="ms"),
}

TVB_NEUROPSI_INH_NEURON_MAPPING = {
    "E_L_i": TranslationRule("v_rest", sim_unit="mV"),
    "b_i": TranslationRule("b", sim_unit="pA"),
    "a_i": TranslationRule("a", sim_unit="nS"),
    "tau_w_i": TranslationRule("tau_w", sim_unit="ms"),
    "E_L_i": TranslationRule("v_rest", sim_unit="mV"),
}

TVB_NEUROPSI_EXC_SYNAPSE_MAPPING = {
    "Q_e": TranslationRule("weight", sim_unit="nS"),
}

TVB_NEUROPSI_INH_SYNAPSE_MAPPING = {
    "Q_i": TranslationRule("weight", sim_unit="nS"),
}

TVB_STATE_VARIABLES_MAPPING = {
    "E": TranslationRule("exc_rate_mean", sim_unit="kHz"),
    "I": TranslationRule("inh_rate_mean", sim_unit="kHz"),
    "C_ee": TranslationRule("exc_rate_var", sim_unit="kHz^2"),
    "C_ei": TranslationRule("rate_cov", sim_unit="kHz^2"),
    "C_ii": TranslationRule("inh_rate_var", sim_unit="kHz^2"),
    "W_e": TranslationRule("exc_adaptation_mean", sim_unit="pA"),
    "W_i": TranslationRule("inh_adaptation_mean", sim_unit="pA"),
    "noise": TranslationRule("noise_rate", sim_unit="kHz"),
    "stimulus": TranslationRule("stim_rate_mean", sim_unit="kHz"),
}

NEUROPSI_TF_FIT_ORDER = ["P_0", "P_mean", "P_std", "P_tau", "P_mean_mean", "P_std_std", "P_tau_tau", "P_mean_std", "P_mean_tau", "P_std_tau"]



# translate_params(single_neuron_params.neuron_params, PYNN_ADEX_MAPPING)

def setup_tvb_model(network_params: BiologicalParameters, mf_sim_params: MeanFieldSimulationConfig) -> lab.models.Model:
    """
    Factory function to instantiate and configure a TVB model directly 
    from the universal BiologicalParameters.
    """
    model_name = mf_sim_params.model
    if model_name not in MODELS_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODELS_REGISTRY.keys())}")
    model_class = MODELS_REGISTRY.get(model_name)
    state_variables = model_class.state_variables
    model = model_class(variables_of_interest=state_variables)

    if model_name in LEGACY_MODELS_REGISTRY:
        # TODO: add assertions, because the legacy models have plenty of assumptions on parameters!!!

        exc_neuron_name = network_params.exc_neuron_name
        inh_neuron_name = network_params.inh_neuron_name
        drive_neuron_name = "drive_neuron"
        stim_neuron_name = "stim_neuron"

        mf_model_params = {
            "N_tot": network_params.internal_size,
            "p_connect_e": network_params.network.connectivity[exc_neuron_name][exc_neuron_name],
            "p_connect_i": network_params.network.connectivity[exc_neuron_name][inh_neuron_name],
            "g": network_params.g,
            "T": mf_sim_params.resolution_time,  # Translation rule
            "P_e": np.array([getattr(mf_sim_params.transfer_function.tf_fits[exc_neuron_name], param) for param in NEUROPSI_TF_FIT_ORDER])*1e-3,
            "P_i": np.array([getattr(mf_sim_params.transfer_function.tf_fits[inh_neuron_name], param) for param in NEUROPSI_TF_FIT_ORDER])*1e-3,
            # This is drive
            "K_ext_e": int(network_params.network.size[drive_neuron_name] * network_params.network.connectivity[exc_neuron_name][drive_neuron_name]),
            "K_ext_i": 0,

            # This is based on the stimulus, has to be updated at each stimulus!
            "external_input_ex_ex": 0,  # [kHz]
            "external_input_ex_in": 0,
            "external_input_in_ex": 0,
            "external_input_in_in": 0,
            "stim_target_ratio": 1.0,

            "tau_OU": 5.0,
            "weight_noise": 0,
        }

        model_params = {
            **translate_params(network_params.neurons[exc_neuron_name].neuron_params, TVB_NEUROPSI_EXC_NEURON_MAPPING),
            **translate_params(network_params.neurons[inh_neuron_name].neuron_params, TVB_NEUROPSI_INH_NEURON_MAPPING),
            **translate_params(network_params.synapses[exc_neuron_name].syn_params, TVB_NEUROPSI_EXC_SYNAPSE_MAPPING),
            **translate_params(network_params.synapses[inh_neuron_name].syn_params, TVB_NEUROPSI_INH_SYNAPSE_MAPPING),
            **mf_model_params
        }

        init_values = {key: TVB_STATE_VARIABLES_MAPPING[key] for key in state_variables}
        init_values = translate_params(mf_sim_params.init_values, init_values)

    else:
        raise NotImplementedError(f"Model '{model_name}' is not in the legacy registry, so we don't have a predefined mapping for it yet. Please implement the parameter translation for this model in build_mf_model().")

    for param_name, param_value in model_params.items():
        setattr(model, param_name, convert_to_array(param_value))

    for state_var in state_variables:
        model.state_variable_range[state_var] = convert_to_array(init_values[state_var])

    return model