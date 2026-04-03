"""
Script that automates the mean-field simulations




"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import time

import codes.tvb_models.nuu_tools_simulation_human as tools
from codes.tvb_models.plot_human import multiview, multiview_one, prepare_surface_regions_human, animation_nuu
from codes.tvb_models import models

from codes.data_structures.network import MFResults
from codes.cell_library import TVBParameterClass

import tvb.simulator.lab as lab
from tvb.basic.neotraits.api import Attr, Final


################################################################################
# Stimulus patterns
class TwoSidedGaussian(lab.equations.FiniteSupportEquation):
    """Asymmetric Gaussian equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = Final(label="Gaussian Equation",
        default="""where((var-midpoint) < 0, amp * exp(-((var-midpoint)**2 / (2.0 * sigma1**2)))+offset, amp * exp(-((var-midpoint)**2 / (2.0 * sigma2**2)))+offset)""",
        doc=""":math:`(amp \\exp\\left(-\\left(\\left(x-midpoint\\right)^2 /
        \\left(2.0 \\sigma1^2\\right)\\right)\\right))\\Theta(midpoint-x) + /
         amp \\exp\\left(-\\left(\\left(x-midpoint\\right)^2 /
        \\left(2.0 \\sigma2^2\\right)\\right)\\right))\\Theta(x-midpoint) offset`"""
        )

    parameters = Attr(
        field_type=dict,
        label="Asymmetric Gaussian Parameters",
        default=lambda: {"amp": 1.0, "sigma1": 1.0, "sigma2": 2.0, "midpoint": 0.0, "offset": 0.0})


class CustomSinusoid(lab.equations.FiniteSupportEquation):

    equation = Final(label="Custom sinusoid equation",
        default="""where((var>=start_time) & (var<end_time), amp * sin(6.283185307179586 * freq * (var-start_time))+offset, offset)""",
        doc=""":math:`amp \\sin(2.0 \\pi freq x) + offset`""")

    parameters = Attr(
        field_type=dict,
        label="Custom Sinusoid Parameters",
        default=lambda: {"amp": 1.0, "freq": 1.0, "start_time": 2.0, "end_time": 0.0, "offset": 0.0})


class GradualConstant(lab.equations.FiniteSupportEquation):
    """Custom drive equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    equation = Final(label="Custom Drive Equation",
        default="""where(var<duration, var*amp/duration+offset, amp+offset)""",
        doc=""":math:`amp/duration * x + offset`"""
        )

    parameters = Attr(
        field_type=dict,
        label="Custom Drive Parameters",
        default=lambda: {"amp": 1.0, "duration": 100.0, "offset": 0.0})

################################################################################

def prepare_pulse_stimulus(stim_start, stim_duration, stim_magnitude, num_nodes,
                           stim_nodes, variables=[8], stim_period=1e9):
    # NOTE: units of the stimulus are the units of the variable to which it is applied to
    # NOTE: the strength of the stimulus is product of the amplitude and the weight
    # NOTE: the weight is set to 1.0

    parameter_stimulus = {}
    eqn_t = lab.equations.PulseTrain()
    stim_pars = {
        "onset": np.array(stim_start),              # [ms] Time of the first pusle
        "tau": np.array(stim_duration),             # [ms] Stimulus duration
        "T": np.array(stim_period),                 # [ms] Interstimulus interval, pulse repetition period
        "amp": np.array(stim_magnitude)*1e-2             # [Hz] Amplitude of the pulse
    }

    eqn_t.parameters.update(stim_pars)

    weight = list(np.zeros(num_nodes))
    weight[stim_nodes] = 1.0 

    parameter_stimulus['eqn_t'] = eqn_t
    parameter_stimulus["variables"] = variables     # index of the variable to which the stimulus is applied
    parameter_stimulus["weights"]= weight
    parameter_stimulus['name'] = 'PulseStimulus'

    return parameter_stimulus

def prepare_twosidedgaussian_stimulus(center, sigma_left, sigma_right, stim_magnitude, num_nodes,
                           stim_nodes, offset=0, variables=[8]):
    # NOTE: units of the stimulus are the units of the variable to which it is applied to
    # NOTE: the strength of the stimulus is product of the amplitude and the weight
    # NOTE: the weight is set to 1.0
    parameter_stimulus = {}

    eqn_t = TwoSidedGaussian()
    stim_pars = {
        "amp": np.array(stim_magnitude)*1e-2,            # [nS] Amplitude of the pulse
        "sigma1": np.array(sigma_left),             # [ms] Standard deviation of the first Gaussian
        "sigma2": np.array(sigma_right),            # [ms] Standard deviation of the second Gaussian
        "midpoint": np.array(center),          # [ms] Midpoint of the pulse
        "offset": np.array(offset)*1e-2             # [nS] Additive constant to the pulse
    }

    eqn_t.parameters.update(stim_pars)

    weight = list(np.zeros(num_nodes))
    weight[stim_nodes] = 1.0 

    parameter_stimulus['eqn_t'] = eqn_t
    parameter_stimulus["variables"] = variables     # index of the variable to which the stimulus is applied
    parameter_stimulus["weights"]= weight
    parameter_stimulus['name'] = 'TwoSidedGaussian'

    return parameter_stimulus

def prepare_sinusoidal_stimulus(stim_magnitude, freq,
                                stim_start, stim_end, num_nodes,
                           stim_nodes, offset=0, variables=[8]):
    # NOTE: units of the stimulus are the units of the variable to which it is applied to

    parameter_stimulus = {}
    eqn_t = CustomSinusoid()
    stim_pars = {
        "amp": np.array(stim_magnitude)*1e-2,            # [nS] Amplitude of the pulse
        "freq": np.array(freq)*1e-3,             # [Hz] Frequency of the sinusoidal
        "offset": np.array(offset)*1e-2,             # [nS] Additive constant to the pulse
        "start_time" : np.array(stim_start),         # [ms] Start time of the sinusoidal
        "end_time" : np.array(stim_end)              # [ms] End time of the sinusoidal
    }

    eqn_t.parameters.update(stim_pars)

    weight = list(np.zeros(num_nodes))
    weight[stim_nodes] = 1.0

    parameter_stimulus['eqn_t'] = eqn_t
    parameter_stimulus["variables"] = variables     # index of the variable to which the stimulus is applied
    parameter_stimulus["weights"]= weight
    parameter_stimulus['name'] = 'Sinusoidal'

    return parameter_stimulus

def prepare_constant_stimulus(stim_magnitude, init_duration,
                              num_nodes,
                                stim_nodes, offset=0, variables=[8]):
    # NOTE: units of the stimulus are the units of the variable to which it is applied to
    parameter_stimulus = {}
    eqn_t = GradualConstant()
    stim_pars = {
        "amp": np.array(stim_magnitude)*1e-2,            # [nS] Amplitude of the pulse
        "duration": np.array(init_duration),             # [ms] Duration of the pulse
        "offset": np.array(offset)*1e-2,             # [nS] Additive constant to the pulse
    }

    eqn_t.parameters.update(stim_pars)

    weight = list(np.zeros(num_nodes))
    weight[stim_nodes] = 1.0

    parameter_stimulus['eqn_t'] = eqn_t
    parameter_stimulus["variables"] = variables     # index of the variable to which the stimulus is applied
    parameter_stimulus["weights"]= weight
    parameter_stimulus['name'] = 'GradualConstant'

    return parameter_stimulus

################################################################################
# Grid connectivity patterns

def create_gaussian_connection_matrix(data_path, nodes=1, amplitude=0.0001, sigma=1):
    """Creates a connection matrix with a Gaussian profile.

    The function generates two matrices:
    - W_matrix: Weight of connections between nodes, with a Gaussian profile.
    - tract_lengths: Distance between nodes in mm.
    The matrices are saved as 'weights.txt' and 'tract_lengths.txt' in the specified path.
    Creates a connection matrix with a gaussian profile.


    Parameters:
        data_path (str): The path where the matrices will be saved.
    Optional:
        nodes (int): The length of the 2D network (for a square network of NxN).
        amplitude (float): Amplitude of Gaussian distance-dependent weight.
        sigma (float): [mm] Standard deviation of Gaussian weight.
    Returns:
        None
    """
    # TODO: write the units of the parameters!!!
    # TODO: tract_length are in mm, two neighboring nodes are 1 mm apart!!!
    # TODO: should add a parameter for the distance between the nodes!!!

    W_matrix = np.zeros((nodes ** 2, nodes ** 2))
    tract_lengths = np.zeros((nodes ** 2, nodes ** 2))
    coords = np.zeros((nodes**2, 2))

    # TODO: should be rescaled!!!
    for i in range(nodes**2):
        row, col = divmod(i,nodes)
        coords[i] = [row, col]

    # Definition of the tract_lengths matrix (euclidian distance between nodes)
    for i in range(0, len(W_matrix)):
        for j in range(0, len(W_matrix)):
            tract_lengths[i, j] = np.sqrt(((coords[i]-coords[j])**2).sum())

    # Definition of the weight matrix (distance dependent - gaussian profile)
    for i in range(0, len(W_matrix)):
        for j in range(0, len(W_matrix)):
            W_matrix[j, i] = amplitude * np.exp(-0.5 * (tract_lengths[i, j] / sigma) ** 2)

    # Set the diagonal to zero to remove self-connections
    for i in range(0, len(W_matrix)):
        W_matrix[i, i] = 0

    # Save weights and tract_lengths matrices in the right
    np.savetxt(data_path / 'weights.txt', W_matrix)
    np.savetxt(data_path / 'tract_lengths.txt', tract_lengths)

################################################################################
# Helpers



def load_results(results_path, sim_time, cut_transient=0):
    return tools.get_result(results_path, cut_transient, sim_time)

################################################################################
# Simulations

def run_single_node_mf(simulation_params, network_params, stimulus, results_path, file_name=f'MFSimulations.pickle'):

    tvb_parameters = TVBParameterClass(
                net_pars=network_params, 
                sim_pars=simulation_params, 
                path=results_path
        )

    grid_size = 1
    num_nodes = grid_size**2
    create_gaussian_connection_matrix(results_path, nodes=grid_size, amplitude=1, sigma=1)

    tvb_parameters.parameter_model['external_input_ex_ex'] = stimulus["drive_rate"]*1e-3                    # [kHz]
    tvb_parameters.parameter_model['external_input_ex_in'] = 0.000
    tvb_parameters.parameter_model['external_input_in_ex'] = stimulus["drive_rate"]*1e-3                    # [kHz]
    tvb_parameters.parameter_model['external_input_in_in'] = 0.000
    tvb_parameters.parameter_model['stim_target_ratio'] = stimulus["stim_target_ratio"]                  # fraction of inhibitory neurons receiving external input

    if stimulus['direct_stimulation']:
        stimulus['direct_stim'] = [0]
    else:
        stimulus['direct_stim'] = [tvb_parameters.parameter_stimulation['stim_var']]

    match stimulus['pattern']:
        case 'PulseTrain':
            parameter_stimulus = prepare_pulse_stimulus(
                                        **stimulus['stim_pars'],
                                        num_nodes=num_nodes, 
                                        stim_nodes=stimulus['target_nodes'],
                                        variables=stimulus['direct_stim'],
                                        )        
        case 'TwoSidedGaussian':
            parameter_stimulus = prepare_twosidedgaussian_stimulus(
                                        **stimulus['stim_pars'],
                                        num_nodes=num_nodes, 
                                        stim_nodes=stimulus['target_nodes'],
                                        variables=stimulus['direct_stim'],
                                        )
        case "Sinusoidal":
            parameter_stimulus = prepare_sinusoidal_stimulus(
                                        **stimulus['stim_pars'],
                                        num_nodes=num_nodes, 
                                        stim_nodes=stimulus['target_nodes'],
                                        variables=stimulus['direct_stim'],                                      
                                        )
        case "GradualConstant":
            parameter_stimulus = prepare_constant_stimulus(
                                        **stimulus['stim_pars'],
                                        num_nodes=num_nodes, 
                                        stim_nodes=stimulus['target_nodes'],
                                        variables=stimulus['direct_stim'],                                      
                                        )
        case "NoStimulus":
            parameter_stimulus = None
        case _:
            raise NotImplementedError(f"Unknown type of stimulus: {stimulus['pattern']}")

    tvb_parameters.parameter_stimulus = parameter_stimulus
    
    simulator = tools.init(tvb_parameters.parameter_simulation,
                            tvb_parameters.parameter_model,
                            tvb_parameters.parameter_connection_between_region,
                            tvb_parameters.parameter_coupling,
                            tvb_parameters.parameter_integrator,
                            tvb_parameters.parameter_monitor,
                            parameter_stimulation=tvb_parameters.parameter_stimulus)

    tools.run_simulation(simulator,
                    stimulus['simulation_time'],                            
                    tvb_parameters.parameter_simulation,
                    tvb_parameters.parameter_monitor)
    
    tvb_results = load_results(results_path, stimulus['simulation_time'])
    vars = eval(f"models.{simulation_params['tvb_model']}.state_variables")
    vars = {var : tvb_results[0][1][:,i,:] for i, var in enumerate(vars)}
    # 'E I C_ee C_ei C_ii W_e W_i noise stimulus'.split()
    results = {
        "time" : tvb_results[0][0].astype(float),      # [ms]
        "drive_rate" : np.ones_like(tvb_results[0][0].astype(float))*stimulus["drive_rate"]*1e-3,  # [kHz] External drive
        **{var : tvb_results[0][1][:,i,:] for i, var in enumerate(vars)}
    }

    results = MFResults(results, stimulus, network_params)
    return results
