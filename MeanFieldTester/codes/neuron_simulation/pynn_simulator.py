"""
This script makes single neuron simulations.


each simulator should implement the same interface, defined in BaseNeuronSimulator,
which is used by the workflow function run_neuron_simulation_workflow to run the simulations.

PyNNSimulator is implemented at the end of the script



Design choice:
- the simulator function asks for the parameters in the form of a dictionary, 
  to make sure multiprocessing works correctly (since multiprocessing requires 
  picklable arguments, and dictionaries are easily picklable).
  Thus provided parameters in the default form of pydantic models has to be 
  converted to dictionaries before being passed to the simulator 
  (use method params.model_dump() before passing to the simulator).
  
  


Phase 1: Core Architecture & Unification (High Priority)

These tasks will finalize the "Unified Flow" so you don't have overlapping, confusing functions doing the same thing.
    [ ] Finish PyNNSimulator.simulate: Update this method to strictly follow the new flow: 1) Resolve the grid for a neuron, 2) Pass the exact 2D arrays to a unified simulation function.
    [ ] Create simulate_adex_batch_unified: Replace the old simulate_adex_batch and simulate_batch_fix_out with a single, clean function that blindly accepts exc_grid and inh_grid and runs them.
    [ ] Delete Dead Code: Once the unified runner works, delete simulate_batch_fix_out, _determine_nu_e_grid, and the giant block of commented-out obsolete code in resolve_grid (around line 522). This will drastically reduce the file size and cognitive load.

Phase 2: Multiprocessing & Performance
    [ ] Parallelize the Unified Batch Runner: Create simulate_adex_batch_unified_multiprocess to handle the heavy lifting of the N statistical runs across the resolved 2D grids.
    [ ] Parallelize Grid Resolution (Your question): Yes, this is absolutely possible and highly recommended! Since resolving the adaptive grid for inh_rate = 10 Hz is completely independent from inh_rate = 20 Hz, we can use mp.Pool to run the coarse interpolation simulations concurrently.

Phase 3: Mathematical Completeness (Medium Priority)

These are hidden TODOs I found inside your code that are necessary for accurate Mean-Field fitting.
    [ ] Implement tau_V calculation: Currently, tau_V is set to 0 or left as a TODO (Lines 115, 230). The Di Volo TF fit relies heavily on the membrane voltage autocorrelation time (τV​).
    [ ] Remove Hardcoded Neuron Names: In resolve_adaptive_grid (Line 414), "exc_neuron" and "inh_neuron" are hardcoded. We should make this dynamic based on the configuration rather than string matching.
    [ ] Adaptive Grid for Inhibitory Neurons: Implement the logic to allow inh_rate to be the adaptive variable. This will require carefully handling the interpolation since the roles of the axes are flipped.

Phase 4: Cleanup & Quality of Life (Low Priority)
    [ ] Unit Conversion: Handle the TODO at Line 65: Convert internal PyNN units (nA, mV) to standard MFT units (pA, V) directly as they come out of the simulation, so the rest of your pipeline doesn't have to guess.
    [ ] Naming Conventions: Align variable names (rate vs nu, activity, firing) according to your todo_ideas.txt master plan.
    [ ] Documentation: Add docstrings to all functions, especially the main workflow and the unified batch runner, to clarify their purpose and expected inputs/outputs.
    [ ] SingleNeuronResults: should I provide the results as a dictionory in the instance or keyword arguments?

"""


from pathlib import Path
import importlib
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import NeuronSimulationConfig
from ..data_structures.single_neuron import SingleNeuronResults
from .base import BaseNeuronSimulator


def simulate_adex_neuron_single_point(exc_rate : float, inh_rate : float ,
                                neuron_params : dict, 
                                init_values : dict, 
                                exc_synapses : dict, inh_synapses : dict,
                                simulation_time=1000.0, dt=0.1, seed=1,
                                **kwargs) -> dict:
    """Simulates a single AdEx neuron with Poisson synaptic input.

    Parameters:
        exc_rate (float): Rate of excitatory Poisson input (Hz).
        inh_rate (float): Rate of inhibitory Poisson input (Hz).
        neuron_params (dict): Parameters for the AdEx neuron model.
        exc_syn_params (dict): Parameters for the excitatory synapse model.
        inh_syn_params (dict): Parameters for the inhibitory synapse model.
        exc_syn_number (int): Number of excitatory synapses.
        inh_syn_number (int): Number of inhibitory synapses.
        synapse_type (str): Type of synapse to use (e.g., 'static_synapse').

        simulation_time (float, optional): Total simulation time in milliseconds
                                           Default is 1000.0 ms.
        dt (float, optional): Simulation time step in milliseconds. 
                              Default is 0.1 ms.
        seed (int, optional): Seed for random number generator. 
                              Default is 0.
    
    Returns:
    dict: A dictionary containing the recorded data with keys 
        'v' (membrane potential) 
        'spikes' (spike times)
        'w' (adaptation variable) 
        'gsyn_exc' (excitatory conductance) 
        'gsyn_inh' (inhibitory conductance)
    """

    # TODO: Direct external input
    # NOTE: synaptic input is Poisson! 

    simulator_backend = kwargs["simulator"].split(".")[1]  # e.g. "nest"
    sim = importlib.import_module(f"pyNN.{simulator_backend}")
    sim.setup(timestep=dt, rng_seed=seed)

    # Test that the seed is set correctly
    # alternative way, something like this
    # I dont know what is the difference between the two though
    # from pyNN.random import NumpyRNG
    # pynn_rng = NumpyRNG(seed=pynn_seed)
    # rng = np.random.RandomState(mozaik_seed)


    neuron = sim.Population(1, sim.EIF_cond_exp_isfa_ista(**neuron_params),initial_values=init_values)
    synapse_exc = sim.native_synapse_type(exc_synapses["syn_type"])(**exc_synapses["syn_params"])
    synapse_inh = sim.native_synapse_type(inh_synapses["syn_type"])(**inh_synapses["syn_params"])

    # Synaptic input
    connector = sim.AllToAllConnector()  # Connect all-to-all
    poisson_input_exc = sim.Population(exc_synapses["number"], sim.SpikeSourcePoisson(rate=exc_rate))
    sim.Projection(poisson_input_exc, neuron, connector, synapse_type=synapse_exc, receptor_type='excitatory')
    poisson_input_inh = sim.Population(inh_synapses["number"], sim.SpikeSourcePoisson(rate=inh_rate))
    sim.Projection(poisson_input_inh, neuron, connector, synapse_type=synapse_inh, receptor_type='inhibitory')

    neuron.record(['v', 'spikes', 'w', 'gsyn_exc', 'gsyn_inh'])

    sim.run(simulation_time)

    data = {name: neuron.get_data().segments[0].filter(name=name)[0] for name in ['v', 'w', 'gsyn_exc', 'gsyn_inh']}
    data['spikes'] = neuron.get_data().segments[0].spiketrains[0]

    # TODO: I think I should also convert the units here to be consistent with
    #  the rest of the code, e.g. convert from nA to pA, mV to V, etc. 
    # But I will do it later when I have the rest of the code working, 
    # for now I just want to get the basic simulation working and then 
    # I will clean up the details later.

    try:
        sim.end()
    except FileNotFoundError:
        # Already cleaned up or missing temp files, it's fine
        pass

    return data

def simulate_adex_neuron_full_grid(neuron_name: str, neuron_params: dict, 
                                exc_rate_grid: np.ndarray, inh_rate_grid: np.ndarray, 
                                neuron_sim_params: NeuronSimulationConfig) -> SingleNeuronResults:
    """
    Simulates a single AdEx neuron across a grid of excitatory and inhibitory input rates.
    """
    sim_time = neuron_sim_params.simulation_time
    dt = neuron_sim_params.time_step
    avg_window = neuron_sim_params.averaging_window
    avg_start = sim_time - avg_window  
    n_bins = int(avg_window / dt)
    seed = neuron_sim_params.seed

    exc_n_points, inh_n_points = exc_rate_grid.shape

    # Initialize result arrays
    out_rate = np.zeros((exc_n_points, inh_n_points, neuron_sim_params.n_runs))
    adaptation_mean = np.zeros_like(out_rate)
    adaptation_std = np.zeros_like(out_rate)
    voltage_mean = np.zeros_like(out_rate)
    voltage_std = np.zeros_like(out_rate)
    tau_V = np.zeros_like(out_rate)
    exc_conductace_mean = np.zeros_like(out_rate)
    exc_conductance_std = np.zeros_like(out_rate)
    inh_conductance_mean = np.zeros_like(out_rate)
    inh_conductance_std = np.zeros_like(out_rate)

    for inh_idx in range(inh_n_points):
        for exc_idx in range(exc_n_points):
            exc_rate = exc_rate_grid[exc_idx, inh_idx]
            inh_rate = inh_rate_grid[exc_idx, inh_idx]
            
            print(f"Simulating {neuron_name} [Point {exc_idx},{inh_idx}]: exc_rate={exc_rate:.2f} Hz, inh_rate={inh_rate:.2f} Hz")

            for n_run in range(neuron_sim_params.n_runs):
                # Run single simulation
                sim_data = simulate_adex_neuron_single_point(
                    exc_rate, inh_rate, **neuron_params, **neuron_sim_params.model_dump(), seed=(seed + n_run)
                )

                spikes = sim_data['spikes']
                adaptation = sim_data['w']
                voltage = sim_data['v']
                exc_conductance = sim_data['gsyn_exc']
                inh_conductance = sim_data['gsyn_inh']

                # Compute metrics
                out_rate[exc_idx,inh_idx,n_run] = spikes[spikes > avg_start].size / (avg_window * 1e-3)

                adaptation_steady = adaptation[-n_bins:]
                adaptation_mean[exc_idx,inh_idx,n_run] = adaptation_steady.mean()
                adaptation_std[exc_idx,inh_idx,n_run] = adaptation_steady.std()

                voltage_steady = voltage[-n_bins:]
                voltage_mean[exc_idx,inh_idx,n_run] = voltage_steady.mean()
                voltage_std[exc_idx,inh_idx,n_run] = voltage_steady.std()

                exc_conductance_steady = exc_conductance[-n_bins:]
                exc_conductace_mean[exc_idx,inh_idx,n_run] = exc_conductance_steady.mean()
                exc_conductance_std[exc_idx,inh_idx,n_run] = exc_conductance_steady.std()
                
                inh_conductance_steady = inh_conductance[-n_bins:]
                inh_conductance_mean[exc_idx,inh_idx,n_run] = inh_conductance_steady.mean()
                inh_conductance_std[exc_idx,inh_idx,n_run] = inh_conductance_steady.std()

    return SingleNeuronResults(
        neuron_name=neuron_name,
        neuron_params=neuron_params,
        results={
            'nu_e': exc_rate_grid,
            'nu_i': inh_rate_grid,
            'nu_out': out_rate.mean(axis=2),
            'nu_out_std': out_rate.std(axis=2),
            'mu_w': adaptation_mean.mean(axis=2) * 1e3,      # convert from nA to pA
            'sigma_w': adaptation_std.mean(axis=2) * 1e3, # convert from nA to pA
            'mu_V': voltage_mean.mean(axis=2),
            'sigma_V': voltage_std.mean(axis=2),
            'tau_V': tau_V.mean(axis=2),
            'mu_ge': exc_conductace_mean.mean(axis=2),
            'sigma_ge': exc_conductance_std.mean(axis=2),
            'mu_gi': inh_conductance_mean.mean(axis=2),
            'sigma_gi': inh_conductance_std.mean(axis=2),
        }
    )

# Multiprocessing 

def _adex_neuron_worker(task_data):
    """Top-level worker to allow pickling across processes.
    
    This is a helper function so that we can use multiprocessing.Pool to run
    simulations in parallel. 
    
    It unpacks the task data, runs a single simulation, 
    computes the metrics, and returns the results along with the indices 
    for where to store them in the result arrays.
    
    """
    exc_rate, inh_rate, exc_rate_idx, inh_rate_idx, n_run_idx, neuron_name, neuron_params, neuron_sim_params = task_data
    
    # Run simulation
    sim_data = simulate_adex_neuron_single_point(
        exc_rate, inh_rate, **neuron_params, **neuron_sim_params
    )
    
    # Extract
    spikes = sim_data['spikes']
    voltage = sim_data['v']
    adaptation = sim_data['w']
    exc_conductance = sim_data['gsyn_exc']
    inh_conductance = sim_data['gsyn_inh']
    
    # Get parameters for calculations
    sim_time = neuron_sim_params['simulation_time']
    dt = neuron_sim_params['time_step']
    avg_window = neuron_sim_params['averaging_window']
    avg_start = sim_time - avg_window
    n_bins = int(avg_window / dt)
    
    # Compute metrics
    out_rate = spikes[spikes > avg_start].size / (avg_window * 1e-3)
    
    adaptation_steady = adaptation[-n_bins:]
    voltage_steady = voltage[-n_bins:]
    exc_conductance_steady = exc_conductance[-n_bins:]
    inh_conductance_steady = inh_conductance[-n_bins:]
    
    return (exc_rate_idx, inh_rate_idx, n_run_idx, {
        'nu_out': out_rate,
        'mu_w': adaptation_steady.mean(),
        'sigma_w': adaptation_steady.std(),
        'mu_V': voltage_steady.mean(),
        'sigma_V': voltage_steady.std(),
        'tau_V': 0,  # TODO: implement tau_V calculation
        'mu_ge': exc_conductance_steady.mean(),
        'sigma_ge': exc_conductance_steady.std(),
        'mu_gi': inh_conductance_steady.mean(),
        'sigma_gi': inh_conductance_steady.std()
    })

def simulate_adex_neuron_full_grid_multiprocess(neuron_name: str, neuron_params: dict, 
                                             exc_rate_grid: np.ndarray, inh_rate_grid: np.ndarray, 
                                             neuron_sim_params: NeuronSimulationConfig) -> SingleNeuronResults:
    """Parallelized execution of the unified batch runner using an un-ordered Pool."""
    exc_n_points, inh_n_points = exc_rate_grid.shape
    seed = neuron_sim_params.seed
    cpus = neuron_sim_params.cpus

    # Initialize result arrays
    nu_out = np.zeros((exc_n_points, inh_n_points, neuron_sim_params.n_runs))
    adaptation_mean = np.zeros_like(nu_out)
    adaptation_std = np.zeros_like(nu_out)
    voltage_mean = np.zeros_like(nu_out)
    voltage_std = np.zeros_like(nu_out)
    voltage_tau = np.zeros_like(nu_out)  # TODO: implement tau_V calculation
    exc_conductance_mean = np.zeros_like(nu_out)
    exc_conductance_std = np.zeros_like(nu_out)
    inh_conductance_mean = np.zeros_like(nu_out)
    inh_conductance_std = np.zeros_like(nu_out)

    # 1. Build the Task List
    tasks = []
    for inh_rate_idx in range(inh_n_points):
        for exc_rate_idx in range(exc_n_points):
            exc_rate = exc_rate_grid[exc_rate_idx, inh_rate_idx]
            inh_rate = inh_rate_grid[exc_rate_idx, inh_rate_idx]
            for n_run_idx in range(neuron_sim_params.n_runs):
                neuron_sim_params_dict = neuron_sim_params.model_dump()
                neuron_sim_params_dict['seed'] = seed + n_run_idx 
                tasks.append((exc_rate, inh_rate, exc_rate_idx, inh_rate_idx, n_run_idx, neuron_name, neuron_params, neuron_sim_params_dict))

    print(f"Starting multiprocessing for {neuron_name}: {len(tasks)} tasks across {cpus} CPUs...")

    # 2. Execute via Pool
    # We use imap_unordered because we don't care about the order they finish, 
    # we just map them directly into our pre-allocated numpy arrays via their indices (e, i, n).
    with mp.Pool(processes=cpus) as pool:
        for result in pool.imap_unordered(_adex_neuron_worker, tasks):
            exc_rate_idx, inh_rate_idx, n_run_idx, res_dict = result
            
            nu_out[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['nu_out']
            adaptation_mean[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['mu_w']
            adaptation_std[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['sigma_w']
            voltage_mean[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['mu_V']
            voltage_std[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['sigma_V']
            exc_conductance_mean[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['mu_ge']
            exc_conductance_std[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['sigma_ge']
            inh_conductance_mean[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['mu_gi']
            inh_conductance_std[exc_rate_idx,inh_rate_idx,n_run_idx] = res_dict['sigma_gi']

    print(f"Finished {neuron_name} multiprocessing batch.")

    return SingleNeuronResults(
        neuron_name=neuron_name,
        neuron_params=neuron_params,
        results={
            'nu_e': exc_rate_grid,
            'nu_i': inh_rate_grid,
            'nu_out': nu_out.mean(axis=2),
            'nu_out_std': nu_out.std(axis=2),
            'mu_w': adaptation_mean.mean(axis=2) * 1e3,      # convert from nA to pA
            'sigma_w': adaptation_std.mean(axis=2) * 1e3, # convert from nA to pA
            'mu_V': voltage_mean.mean(axis=2),
            'sigma_V': voltage_std.mean(axis=2),
            'tau_V': voltage_tau.mean(axis=2),
            'mu_ge': exc_conductance_mean.mean(axis=2),
            'sigma_ge': exc_conductance_std.mean(axis=2),
            'mu_gi': inh_conductance_mean.mean(axis=2),
            'sigma_gi': inh_conductance_std.mean(axis=2),
        }
    )

# Dealing with the grid

def find_exc_rate_max_for_out_rate_target(neuron_params, neuron_sim_params_dict, inh_rate, out_rate_target, rel_tol=0.1, max_iter=100):
    """Finds the upper boundary nu_e using a fast geometric expansion and rough bisection."""
    # (Same helper to get the rate)
    def get_rate(exc_rate):
        data = simulate_adex_neuron_single_point(exc_rate, inh_rate, **neuron_params, **neuron_sim_params_dict)
        spikes = data['spikes']
        avg_window = neuron_sim_params_dict['averaging_window']
        return spikes[spikes > (neuron_sim_params_dict['simulation_time'] - avg_window)].size / (avg_window * 1e-3)

    exc_rate_high = 1.0
    while get_rate(exc_rate_high) < out_rate_target:
        exc_rate_high *= 2.0
        if exc_rate_high > 1000: return exc_rate_high # Safety

    # Only a few bisections just to get the "roof" reasonably close
    exc_rate_low = exc_rate_high / 2.0
    i = 0
    out_rate_last_high = get_rate(exc_rate_high)
    while abs(out_rate_last_high - out_rate_target) > rel_tol * out_rate_target and i < max_iter:
        if i%10 == 0:
            print(f"Finding exc_rate_max: iteration {i}")
        mid = (exc_rate_low + exc_rate_high) / 2.0
        out_rate_new = get_rate(mid)
        if out_rate_new < out_rate_target:
            exc_rate_low = mid
        else:
            exc_rate_high = mid
            out_rate_last_high = out_rate_new
        i += 1
    print(f"Found exc_rate_max: {exc_rate_high:.2f} Hz after {i} iterations")


    return exc_rate_high

def _resolve_adaptive_grid_worker(task_data):
    """Worker function to resolve a single column of the adaptive grid in parallel."""
    (inh_rate_idx, inh_rate, out_rate_targets, out_rate_max, n_coarse_points,
     single_neuron_params, neuron_sim_params_dict) = task_data

    # Find the maximum excitatory rate needed to reach out_rate_max
    exc_rate_max = find_exc_rate_max_for_out_rate_target(
        single_neuron_params, neuron_sim_params_dict, inh_rate, out_rate_max
    )

    exc_rate_grid_coarse = np.linspace(0, exc_rate_max, n_coarse_points)
    out_rate_values_coarse = np.zeros(n_coarse_points)

    sim_time = neuron_sim_params_dict['simulation_time']
    avg_window = neuron_sim_params_dict['averaging_window']
    
    # Run coarse simulations
    for exc_rate_idx, exc_rate_test in enumerate(exc_rate_grid_coarse):
        data = simulate_adex_neuron_single_point(exc_rate_test, inh_rate, **single_neuron_params, **neuron_sim_params_dict)
        spikes = data['spikes']
        out_rate_values_coarse[exc_rate_idx] = spikes[spikes > (sim_time - avg_window)].size / (avg_window * 1e-3)

    # Reversal to capture the last zero activity before firing begins
    out_rate_values_coarse_rev = out_rate_values_coarse[::-1]
    unique_indices_rev = np.unique(out_rate_values_coarse_rev, return_index=True)[1]
    unique_indices = n_coarse_points - 1 - unique_indices_rev

    out_rate_unique = out_rate_values_coarse[unique_indices]
    exc_rate_unique = exc_rate_grid_coarse[unique_indices]
    
    # Prepare the output columns for this specific inh_rate
    out_n_points = len(out_rate_targets)
    exc_rate_column = np.zeros(out_n_points)
    
    if len(out_rate_unique) > 1:
        inverse_f_I_curve = PchipInterpolator(out_rate_unique, exc_rate_unique)
        safe_targets = np.clip(out_rate_targets, out_rate_unique.min(), out_rate_unique.max())
        exc_rate_column[:] = inverse_f_I_curve(safe_targets)
    else:
        # Fallback if the neuron is completely dead (never spiked)
        exc_rate_column[:] = 0.0
        
    inh_rate_column = np.full(out_n_points, inh_rate)
    
    return inh_rate_idx, exc_rate_column, inh_rate_column

def resolve_adaptive_grid(neuron_name, neuron_params, neuron_sim_params):
    """
    The fastest, most robust method. Simulates a coarse grid, then 
    interpolates to find the exact inputs needed for the target outputs.

    Parameters
    ----------
    neuron_name : str
        Name of the neuron type being simulated (e.g., 'exc_neuron', 'inh_neuron').
        Has to correspond to keys in neuron_params.
    neuron_params : dict
        Dictionary of parameters for the neuron models.
    neuron_sim_params : NeuronSimulationConfig
        Configuration object containing grid specifications and other simulation parameters.
    Returns
    -------
    exc_rate_grid : np.ndarray
        2D array of excitatory input rates corresponding to the grid.
        indexing (exc_rate, inh_rate) with shape (n_exc_rates, n_inh_rates)
    inh_rate_grid : np.ndarray
        2D array of inhibitory input rates corresponding to the grid.
        indexing (exc_rate, inh_rate) with shape (n_exc_rates, n_inh_rates)
    """
    grid_params = getattr(neuron_sim_params.grid, neuron_name)

    # TODO: there are hardcoded names here,
    # would be nice to make it more flexible, but for now it works and is clear enough
    # change later if we add more neuron types or want to do something more fancy with the grids
    # TODO: would be nice to make it work for both exc and inh adaptive grids
    # but at the moment I only implemented the exc one because it is more relevant

    if grid_params.inh_rate_grid == "adaptive":
        raise NotImplementedError("Adaptive grid for inhibitory rates not implemented yet")
        # TODO:
        # this is a bit tricky, and I did not have time to implement it
        # requires abstraction + careful handling of the interpolation because
        # the function is monotonic but not strictly and it is lowering instead 
        # of increasing, so we have to be careful with the edge cases and the 
        # interpolation method

    # NOTE: This following assumes 
    # exc_rate_grid == "adaptive"
    
    inh_rate_min, inh_rate_max, inh_n_points = grid_params.inh_rate_grid
    out_rate_min, out_rate_max, out_n_points = grid_params.out_rate_grid
    inh_n_points = int(inh_n_points)
    out_n_points = int(out_n_points)


    inh_rates = np.linspace(inh_rate_min, inh_rate_max, inh_n_points)
    out_rate_targets = np.linspace(out_rate_min, out_rate_max, out_n_points)


    # NOTE: once implementing general adaptive grid, this part needs to be refactored
    # because I require grid to be indexed (exc_rate, inh_rate) for the rest of the code,
    # so if we want to do inh adaptive grid we have to flip the indexing and 
    # be careful with the interpolation and the way we fill the grids, etc.
    # I guess I could keep this and then just transpose or reorder based on the neuron types

    exc_rate_grid = np.zeros((out_n_points, inh_n_points))
    inh_rate_grid = np.zeros((out_n_points, inh_n_points))
    
    n_coarse_points = grid_params.n_coarse_interpolation_points
    cpus = neuron_sim_params.cpus

    print(f"Resolving adaptive grid for {neuron_name} using {cpus} CPUs...")

    # Build tasks
    tasks = []
    for inh_rate_idx, inh_rate in enumerate(inh_rates):
        tasks.append((
            inh_rate_idx, inh_rate, out_rate_targets, out_rate_max, n_coarse_points,
            neuron_params[neuron_name], neuron_sim_params.model_dump()
        ))


    with mp.Pool(processes=cpus) as pool:
        for result in pool.imap_unordered(_resolve_adaptive_grid_worker, tasks):
            inh_rate_idx, exc_rate_col, inh_rate_col = result
            exc_rate_grid[:, inh_rate_idx] = exc_rate_col
            inh_rate_grid[:, inh_rate_idx] = inh_rate_col
            print(f"    Finished interpolation for inh_rate = {inh_rate_col[0]:.2f} Hz")

    return exc_rate_grid, inh_rate_grid

class PyNNSimulator(BaseNeuronSimulator):
    """PyNN implementation of the single neuron simulator."""


    def resolve_grid(self, neuron_name: str, neuron_params: dict, neuron_sim_params: NeuronSimulationConfig) -> tuple[np.ndarray, np.ndarray]:
        """Resolves the 2D grid for a specific neuron based on the configuration.
        
        Parameters
        ----------
        neuron_name : str
            Name of the neuron type being simulated (e.g., 'exc_neuron', 'inh_neuron').
            Has to correspond to keys in neuron_params.
        neuron_params : dict
            Dictionary of parameters for the neuron models.
        neuron_sim_params : NeuronSimulationConfig
            Configuration object containing grid specifications and other simulation parameters.

        Returns
        -------
        exc_rate_grid : np.ndarray
            2D array of excitatory input rates corresponding to the grid.
            indexing (exc_rate, inh_rate) with shape (n_exc_rates, n_inh_rates)
        inh_rate_grid : np.ndarray
            2D array of inhibitory input rates corresponding to the grid.
            indexing (exc_rate, inh_rate) with shape (n_exc_rates, n_inh_rates)
        """
        grid_params = getattr(neuron_sim_params.grid, neuron_name)
        
        match grid_params.grid_type:
            case "linear":
                exc_rate_min, exc_rate_max, exc_n_points = grid_params.exc_rate_grid
                inh_rate_min, inh_rate_max, inh_n_points = grid_params.inh_rate_grid

                exc_rate_grid = np.linspace(exc_rate_min, exc_rate_max, int(exc_n_points))
                inh_rate_grid = np.linspace(inh_rate_min, inh_rate_max, int(inh_n_points))

                exc_rate_grid, inh_rate_grid = np.meshgrid(exc_rate_grid, inh_rate_grid, sparse=False, indexing='ij')
                
            case "custom":
                # NOTE: custom grid is automatically loaded and converted into ndarray
                # based on the config file, so we just need to check that it is valid
                exc_rate_grid = grid_params.exc_rate_grid
                inh_rate_grid = grid_params.inh_rate_grid
                if exc_rate_grid.shape != inh_rate_grid.shape:
                    raise ValueError("Custom grid exc_rate_grid and inh_rate_grid must have the same shape")
 
            case "adaptive":
                print(f"Resolving adaptive grid for {neuron_name}...")
                exc_rate_grid, inh_rate_grid = resolve_adaptive_grid(neuron_name, neuron_params, neuron_sim_params)

            case _:
                raise ValueError(f"Unknown grid type: {grid_params.grid_type}")

        return exc_rate_grid, inh_rate_grid


    def simulate(self, neuron_params: dict, neuron_sim_params: NeuronSimulationConfig) -> dict:
        """Routes to the correct PyNN execution method based on neuron_sim_params.
        
        Parameters
        ----------
        neuron_params : dict
            Dictionary of parameters for the neuron models.
            items are: (neuron_name, dict of neuron parameters, synapses etc.)
        neuron_sim_params : NeuronSimulationConfig
            Configuration object containing grid specifications and other simulation parameters.
        results_path : str
            Path to the directory where simulation results will be saved.

        Returns
        -------
        results : dict
            Dictionary containing the simulation results for each neuron type.

        """
        results = {}

        for neuron_name, single_neuron_params in neuron_params.items():
            print(f"\n{'='*50}\nPreparing simulation for {neuron_name}\n{'='*50}")

            exc_rate_grid, inh_rate_grid = self.resolve_grid(neuron_name, neuron_params, neuron_sim_params)

            if neuron_sim_params.cpus > 1:
                neuron_result = simulate_adex_neuron_full_grid_multiprocess(
                    neuron_name, single_neuron_params, exc_rate_grid, inh_rate_grid, neuron_sim_params
                )
            else:
                neuron_result = simulate_adex_neuron_full_grid(
                    neuron_name, single_neuron_params, exc_rate_grid, inh_rate_grid, neuron_sim_params
                )

            results[neuron_name] = neuron_result

        return results