"""
This script contains the parameters of various cell types.

By default neuron params are specified as expected by PyNN

Parameters of the neurons are saved in ../params/neurons.yaml

"""

import json
import fnmatch
from pathlib import Path
import copy
import pickle

from codes.utils.dict_helpers import get_items_recursive
from codes.tvb_models import models


PARAM_DIR = Path(__file__).resolve().parent.parent / 'params'
NEURON_PARAMS = PARAM_DIR / 'neurons.yaml'
SYNAPSE_PARAMS = PARAM_DIR / 'synapses.yaml'
CSNG_PARAMS = PARAM_DIR / 'csng_parameters.json'

# Functions to load/scrap parameters of neurons

class TVBParameterClass():
    def __init__(self, net_pars, sim_pars, path):
        exc_neuron = net_pars["exc_neuron"]
        inh_neuron = net_pars["inh_neuron"]
        # assert exc_neuron["neuron_params"]["cm"] == inh_neuron["neuron_params"]["cm"]
        # assert exc_neuron["neuron_params"]["tau_m"] == inh_neuron["neuron_params"]["tau_m"]
        model_class = eval(f"models.{sim_pars['tvb_model']}")
        svars = model_class.state_variables

        try:
            stp_pars = {
                "U_e" : exc_neuron["exc_synapses"]["syn_params"]["U"],
                "tau_rec_e" : exc_neuron["exc_synapses"]["syn_params"]["tau_rec"],
                "tau_fac_e" : exc_neuron["exc_synapses"]["syn_params"]["tau_fac"],
                "U_i" : exc_neuron["inh_synapses"]["syn_params"]["U"],
                "tau_rec_i" : exc_neuron["inh_synapses"]["syn_params"]["tau_rec"],
                "tau_fac_i" : exc_neuron["inh_synapses"]["syn_params"]["tau_fac"],
            }
        except KeyError:
            stp_pars = {}

        self.parameter_simulation = {
            'path_result' : path,
            'seed': sim_pars["seed"], # the seed for the random generator
            'save_time': 1000.0, # [ms], the time of simulation in each file
        }

        self.parameter_model = {
            'model' : sim_pars['tvb_model'],
            'g_L_e' :exc_neuron["neuron_params"]["cm"]/exc_neuron["neuron_params"]["tau_m"]*1e3,        # [ns]            
            'g_L_i' :inh_neuron["neuron_params"]["cm"]/inh_neuron["neuron_params"]["tau_m"]*1e3,        # [ns]            
            'C_m_e' : exc_neuron["neuron_params"]["cm"]*1e3,                                   # [uF]
            'C_m_i' : inh_neuron["neuron_params"]["cm"]*1e3,                                   # [uF]
            'E_L_e' : exc_neuron["neuron_params"]["v_rest"],                                # [mV]
            'E_L_i' : inh_neuron["neuron_params"]["v_rest"],                                # [mV]
            # adaptive parameters
            'b_e' : exc_neuron["neuron_params"]["b"]*1e3,                                    # [pA]
            'a_e' : exc_neuron["neuron_params"]["a"],                                        # [nS]
            'tau_w_e' : exc_neuron["neuron_params"]["tau_w"],                                # [ms]
            'b_i' : inh_neuron["neuron_params"]["b"]*1e3,                                    # [pA]
            'a_i' : inh_neuron["neuron_params"]["a"],                                        # [nS]
            'tau_w_i' : inh_neuron["neuron_params"]["tau_w"],
            # synaptic parameters
            'E_e' : exc_neuron["neuron_params"]["e_rev_E"],
            'E_i' : exc_neuron["neuron_params"]["e_rev_I"],
            'Q_e' : exc_neuron["exc_synapses"]["syn_params"]["weight"],     # [nS]
            'Q_i' : exc_neuron["inh_synapses"]["syn_params"]["weight"],     # [nS]
            'tau_e' : exc_neuron["neuron_params"]["tau_syn_E"],
            'tau_i' : exc_neuron["neuron_params"]["tau_syn_I"],
            'N_tot' : net_pars["network"]["total_pop_size"],
            **stp_pars,
            # connectivity parameters (for the excitatory and inhibitory populations)
            'p_connect_e' : net_pars["network"]["p_connect_exc"],
            'p_connect_i' : net_pars["network"]["p_connect_inh"],
            'g' : net_pars["network"]["g"],
            # MF parameters
            'T' : sim_pars["T"],                                                # [ms]
            'P_e' : [x*1e-3 for x in net_pars["transfer_function"]["exc_neuron"]],
            'P_i' : [x*1e-3 for x in net_pars["transfer_function"]["inh_neuron"]],
            # external drive
            # probably {target_type}_{source_type}
            # NOTE: These are updated with the specification of the stimulus
            # in meanfield_simulation.run_single_node_mf
            'external_input_ex_ex' : 0.000,                    # [kHz]
            'external_input_ex_in' : 0.000,
            'external_input_in_ex' : 0.000,                    # [kHz]
            'external_input_in_in' : 0.000,
            'stim_target_ratio' : 1.00,                  # fraction of inhibitory neurons receiving external input
            # drive population size
            'K_ext_e' : int(net_pars["network"]["drive_pop_size"]*net_pars["network"]["p_connect_drive"]),                  # number of connection coming from drive population
            'K_ext_i' : 0,
            # noise parameters
            'tau_OU' : 5.0,
            'weight_noise': 0, # 1e-4, #10.5*1e-5,
            # Initial condition [exc_pop, inh_pop]
            'initial_condition':{
                "E": [0.000, 0.000],
                "I": [0.00, 0.00],
                "C_ee": [0.0,0.0],
                "C_ei": [0.0,0.0],
                "C_ii": [0.0,0.0],
                "W_e": [100.0, 100.0],
                "W_i": [0.0,0.0],
                "noise":[0.0,0.0],
            }
        }
        self.parameter_model["initial_condition"].update(sim_pars["mf_init"])

        if "stimulus" in svars:
            self.parameter_model["initial_condition"]["stimulus"] = [0.0, 0.0]

        self.parameter_connection_between_region={
            'default': False,
            'from_file': False,          #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_folder': True,
            'from_h5': False,
            'path' : path,              # the files
            'number_of_regions': 0,     # number of regions
            'tract_lengths': [],        # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'weights': [],              # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'speed': 4.0,               # speed of along long range connection
            'normalised': False
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter':{'a':0.3,   #Peut etre modifie
                         'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                'nsig':[(var=="noise")*1.0 for var in svars],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':list(range(len(svars))),
                'period':self.parameter_integrator['dt']*10.0
            },
            'Bold':False,
            'parameter_Bold':{
                'variables_of_interest':[0],
                'period':self.parameter_integrator['dt']*2000.0
            },
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1
            }
        }

        self.parameter_stimulation = {
            'stim_var' : svars.index('stimulus') if 'stimulus' in svars else None
        } 
        
        

def update_params(pars: dict) -> dict:
    """Converts params to desired form"""
    ntw_pars = pars["network"]
    ntw_pars["exc_pop_size"] = int(ntw_pars["total_pop_size"]*(1-ntw_pars["g"]))
    ntw_pars["inh_pop_size"] = int(ntw_pars["total_pop_size"]*ntw_pars["g"])
 
    for neuron in ["exc_neuron", "inh_neuron"]:
        match pars[neuron]:
            case dict():
                neuron_params = pars[neuron]["neuron_params"]
                init_vals = pars[neuron]["init_values"]
            case "zerlaut":
                raise NotImplementedError
            case "divolo":
                raise NotImplementedError
            case _:
                raise NotImplementedError

        pars[neuron] = {
            "neuron_params" : neuron_params,
            "init_values" : init_vals,
        }

        for syn_name, synapse in copy.deepcopy(pars['synapses']).items():
            pars[neuron][syn_name] = synapse
            pop = syn_name.split("_")[0]
            number = int(ntw_pars[f"{pop}_pop_size"]*ntw_pars[f"p_connect_{pop}"])
            pars[neuron][syn_name]["number"] = number


def scrap_csng_model_parameters(param_file : Path) -> dict:
    """Parse the parameters of the CSNG model from the given file."""
    params = {}
    with open(param_file, 'r') as f:
        full_params = json.load(f)

    for sheet in full_params['sheets'].values():
        if 'name' not in sheet['params']:
            continue
        sheet_name = sheet['params']['name']
        params[sheet_name] = {
            'model': sheet['params']['cell']['model'],
            'params': sheet['params']['cell']['params'],
        }
        if 'K' in sheet:
            params[sheet_name]['K']= sheet['K']
        conn_key = ''.join(sheet_name.replace('/','').split('_')[:0:-1])
        conn_key += 'Connection'

        params[sheet_name]['conns'] = get_items_recursive(full_params, conn_key)

    neurons = dict()
    for cell_name, neuron in params.items():
        layer = cell_name.replace('/', '').split("_")[2]
        conns = [k for k in neuron['conns'].keys() if k.startswith(layer)]
        neurons[cell_name] = {
            'neuron_params' : neuron['params'],
            'synapse_type': 'tsodyks_synapse',
            'simulation_time': 1000.0, 
            'poisson_input': True,
        }
        for conn in conns:
            source = conn.split(layer)[1].lower()
            neurons[cell_name][f'{source}_syn_params'] = {
                'weight': neuron['conns'][conn]['base_weight'] * 1000,  # nS to pS
                'delay': 1.0,
                **neuron['conns'][conn]['short_term_plasticity'],
            }
            neurons[cell_name][f'{source}_syn_number'] = int(neuron['conns'][conn]['num_samples'])
    return neurons

def convert_zerlaut_params(pars):
    new_pars = {
        'neuron_params': {
            'v_rest': pars['El']*1e3,
            'v_reset': pars['Vreset']*1e3,
            'tau_refrac': pars['Trefrac']*1e3,
            'tau_m': pars['Cm']/pars['Gl']*1e3,
            'cm': pars['Cm']*1e9,
            'e_rev_E': pars['Ee']*1e3,
            'e_rev_I': pars['Ei']*1e3,
            'tau_syn_E': pars['Te']*1e3,
            'tau_syn_I': pars['Ti']*1e3,
            'a': pars['a']*1e9,
            'b': pars['b']*1e9,
            'tau_w': pars['tauw']*1e3,
            'delta_T': pars['delta_v']*1e3,
            'v_thresh': pars['Vthre']*1e3
        },
        'exc_syn_params': {'weight': pars['Qe']*1e9, 'delay': 1.0},
        'inh_syn_params': {'weight': pars['Qi']*1e9, 'delay': 1.0},
        'exc_syn_number': pars['Ntot']*pars['pconnec']*(1-pars['gei']),
        'inh_syn_number': pars['Ntot']*pars['pconnec']*pars['gei'],
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }

    return new_pars

def load_zerlaut_model_parameters():
    # taken from Zerlaut2018_ModelingMesoscopic
    zerlaut_exc_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 15.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.150,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 4.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.02,  # Spike-triggered adaptation increment (nA)
            'delta_T': 2.0,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }

    zerlaut_inh_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 15.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.150,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 0.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.0,  # Spike-triggered adaptation increment (nA)
            'delta_T': 0.5,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }
    return {'zerlaut_exc': zerlaut_exc_neuron, 'zerlaut_inh': zerlaut_inh_neuron}

def load_divolo_model_parameters():
    # taken from diVolo2019_BiologicallyRealistic
    divolo_exc_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 20.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.200,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 4.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.02,  # Spike-triggered adaptation increment (nA)
            'delta_T': 2.0,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }

    divolo_inh_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 20.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.200,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 0.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.0,  # Spike-triggered adaptation increment (nA)
            'delta_T': 0.5,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }
    return {'divolo_exc': divolo_exc_neuron, 'divolo_inh': divolo_inh_neuron}

def load_all_neurons() -> dict:
    neurons = dict()

    csng_neurons = scrap_csng_model_parameters(CSNG_PARAMS)
    neurons.update(**csng_neurons)

    zerlaut_neurons = load_zerlaut_model_parameters()
    neurons.update(**zerlaut_neurons)

    divolo_neurons = load_divolo_model_parameters()
    neurons.update(**divolo_neurons)

    return neurons

def load_neurons_from_file(neuron_file : Path) -> dict:
    """Load the parameters of the neurons from the given file.
    
    Assumes parameters are saved in the Python-readable format.
    """
    with open(neuron_file, 'r') as f:
        neurons = eval(f.read())
    return neurons

def load_neurons(neurons : list) -> dict:
    """
    Load the parameters of the neurons specified in the list.
    """
    all_neurons = load_neurons_from_file(NEURON_PARAMS)

    # all_neurons = load_all_neurons()
    return {neuron: all_neurons[neuron] for neuron in neurons}

# Functions to adjust the parameters of neurons

def enforce_synapse(neurons : dict, synapse_types : list[str]) -> dict:
    """Enforce synapse types on the neurons specified in the list."""

    new_neurons = dict()
    for neuron, params in neurons.items():
        if params['synapse_type'] in synapse_types:
            new_neurons[neuron] = params
    return new_neurons

def with_static_synapse(neuron):
    """Makes a copy of dicionary and updates the synaptic parameters to static synapses.
    
    """
    new_neuron = copy.deepcopy(neuron)

    new_neuron['synapse_type'] = 'static_synapse'
    # Remove parameters for short-term plasticity
    for source in ['exc', 'inh']:
        for key in ['U', 'tau_fac', 'tau_rec', 'tau_psc']:
            new_neuron[f'{source}_syn_params'].pop(key, None)
    return new_neuron

def with_tsodyks_synapse(neuron, U, tau_rec, tau_fac=0):
    """Updates the neuron parameters to use static synapses.
    """
    new_neuron = copy.deepcopy(neuron)

    new_neuron['synapse_type'] = 'tsodyks_synapse'
    for source in ['exc', 'inh']:
        new_neuron[f'{source}_syn_params']['U'] = U
        new_neuron[f'{source}_syn_params']['U'] = U

    new_neuron['exc_syn_params']['U'] = U

    return new_neuron

# TODO: print table of parameters
# to see the differences between the models
def printout_neuron_parameters(neurons):
    pass

# TODO: function to change units

# TODO: conversion PyNN <--> NEST
def pynn_to_nest(params):
    pass

def nest_to_pynn(params):
    pass

