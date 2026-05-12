"""
Collection of translation mappings to work with different simulators and models in a unified way. 

"""
from .translators import TranslationRule

# NOTE: Units based on this
# https://neuralensemble.org/docs/PyNN/units.html


PYNN_ADEX_MAPPING = {
    "v_rest": TranslationRule("v_rest", sim_unit="mV"),
    "v_reset": TranslationRule("v_reset", sim_unit="mV"),
    "tau_refrac": TranslationRule("tau_refrac", sim_unit="ms"),
    "tau_m": TranslationRule("tau_m", sim_unit="ms"),
    "cm": TranslationRule("cm", sim_unit="nF"),
    "e_rev_E": TranslationRule("e_rev_E", sim_unit="mV"),
    "e_rev_I": TranslationRule("e_rev_I", sim_unit="mV"),
    "tau_syn_E": TranslationRule("tau_syn_E", sim_unit="ms"),
    "tau_syn_I": TranslationRule("tau_syn_I", sim_unit="ms"),
    "a": TranslationRule("a", sim_unit="nS"),
    "b": TranslationRule("b", sim_unit="nA"),
    "delta_T": TranslationRule("delta_T", sim_unit="mV"),
    "tau_w": TranslationRule("tau_w", sim_unit="ms"),
    "v_thresh": TranslationRule("v_thresh", sim_unit="mV"),
}

PYNN_STATIC_SYNAPSE_MAPPING = {
    "weight": TranslationRule("weight", sim_unit="uS"),
    "delay": TranslationRule("delay", sim_unit="ms"),
}

NEST_STATIC_SYNAPSE_MAPPING = {
    "weight": TranslationRule("weight", sim_unit="nS"),
    "delay": TranslationRule("delay", sim_unit="ms"),
}

NEST_TSODYKS_SYNAPSE_MAPPING = {
    "weight": TranslationRule("weight", sim_unit="nS"),
    "delay": TranslationRule("delay", sim_unit="ms"),
    "tau_psc": TranslationRule("tau_rp", sim_unit="ms"),
    "tau_fac": TranslationRule("tau_psc", sim_unit="ms"),
    "tau_rec": TranslationRule("tau_rec", sim_unit="ms"),
    "U": TranslationRule("U", sim_unit="")
}

PYNN_INITIAL_VALUES_MAPPING = {
    "v": TranslationRule("voltage", sim_unit="mV"),
    "w": TranslationRule("adaptation", sim_unit="nA"),
}