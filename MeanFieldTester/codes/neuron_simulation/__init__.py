"""
This script makes single neuron simulations.
"""


import numpy as np

from pathlib import Path
import pyNN.nest as sim
import pickle
from functools import partial

import multiprocessing as mp

from codes.data_structures.single_neuron import SingleNeuronResults

def load_data(neurons, fname_pattern, data_path, full=False):
    results = dict()
    for neuron_name in neurons:
        with open(data_path / fname_pattern.format(neuron_name), 'rb') as f:
            data = pickle.load(f)
        for key, val in data.items():
            if key not in results:
                results[key] = dict()
            results[key][neuron_name] = val
    return results

def load_results(path, neuron_names, fname_pattern):
    results = dict()
    for neuron_name in neuron_names:
        with open(path / fname_pattern.format(neuron_name), 'rb') as f:
            data = pickle.load(f)
        results[neuron_name] = data
    return results

# Simulations

def simulate_single_adex_neuron(nu_e : float, nu_i : float ,
                                neuron_params : dict, 
                                init_values : dict, 
                                exc_synapses : dict, inh_synapses : dict,
                                simulation_time=1000.0, dt=0.1, seed=1,
                                **kwargs) -> dict:
    """Simulates a single AdEx neuron with Poisson synaptic input.

    Parameters:
        nu_e (float): Rate of excitatory Poisson input (Hz).
        nu_i (float): Rate of inhibitory Poisson input (Hz).
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
    poisson_input_exc = sim.Population(exc_synapses["number"], sim.SpikeSourcePoisson(rate=nu_e))
    sim.Projection(poisson_input_exc, neuron, connector, synapse_type=synapse_exc, receptor_type='excitatory')
    poisson_input_inh = sim.Population(inh_synapses["number"], sim.SpikeSourcePoisson(rate=nu_i))
    sim.Projection(poisson_input_inh, neuron, connector, synapse_type=synapse_inh, receptor_type='inhibitory')

    neuron.record(['v', 'spikes', 'w', 'gsyn_exc', 'gsyn_inh'])

    sim.run(simulation_time)

    data = {name: neuron.get_data().segments[0].filter(name=name)[0] for name in ['v', 'w', 'gsyn_exc', 'gsyn_inh']}
    data['spikes'] = neuron.get_data().segments[0].spiketrains[0]

    try:
        sim.end()
    except FileNotFoundError:
        # Already cleaned up or missing temp files, it's fine
        pass

    return data

def simulate_adex_batch(nu_e_range, nu_i_range, neurons, data_path, sim_pars):
    """

    Parameters
    ----------
    nu_e_range : 1D np.ndarray
        Range of excitatory input rates.
    nu_i_range : 1D np.ndarray
        Range of inhibitory input rates.
    neurons : dictionary
        Dictionary with neuron parameters.
    data_path : str or Path
        Path to save the data.

    Returns
    -------
    results : dict
        Dictionary with results for each neuron.
        Keys are neuron names, values are dictionaries with keys:
        'nu_e', 'nu_i', 'nu_out', 'nu_out_std', 'mu_w', 'mu_V', 'sigma_V',
    """

    sim_time = sim_pars['simulation_time']  # ms
    dt = sim_pars['time_step']  # ms
    avg_window = sim_pars['averaging_window']  # ms
    avg_start = sim_time - avg_window  
    n_bins = int(sim_pars['averaging_window']/dt)
    seed = sim_pars['seed']

    nu_e_grid, nu_i_grid = np.meshgrid(nu_e_range, nu_i_range, indexing='ij')

    results = {}

    if type(data_path) == str:
        data_path = Path(data_path)

    for neuron_name, neuron_pars in neurons.items():
        print(neuron_name)

        tau_w = neuron_pars['neuron_params']['tau_w']
        if sim_time < tau_w*5:  # static solution may not be reached
            print(f"WARNING: Simulation time ({sim_time} ms)"
                  + f" too short for {neuron_name}")
        if avg_start < tau_w*5:  # starting averaging too early
            print(f"WARNING: Averaging window ({avg_window} ms)"
                  + f" too long for {neuron_name}")

        nu_out = np.zeros((nu_e_range.size, nu_i_range.size, sim_pars['n_runs']))
        mu_w = np.zeros_like(nu_out)
        mu_V = np.zeros_like(nu_out)
        sigma_V = np.zeros_like(nu_out)
        tau_V = np.zeros_like(nu_out)
        mu_ge = np.zeros_like(nu_out)
        sigma_ge = np.zeros_like(nu_out)
        mu_gi = np.zeros_like(nu_out)
        sigma_gi = np.zeros_like(nu_out)

        # Simulate for each combination of nu_e and nu_i
        for i, nu_i in enumerate(nu_i_range):
            for e, nu_e in enumerate(nu_e_range):
                print(f"Simulating {neuron_name} nu_e={nu_e} Hz, nu_i={nu_i} Hz")

                for n in range(sim_pars["n_runs"]):
                    sim_pars['seed'] = seed + n
                    sim_data = simulate_single_adex_neuron(nu_e, nu_i, **neuron_pars, **sim_pars)

                    spikes = sim_data['spikes']
                    membrane_potential = sim_data['v']
                    w = sim_data['w']
                    gsyn_exc = sim_data['gsyn_exc']
                    gsyn_inh = sim_data['gsyn_inh']

                    nu_out[e,i,n] = spikes[spikes > avg_start].size/(avg_window*1e-3)

                    mu_w[e,i,n] = w[-n_bins:].mean()

                    membrane_potential = membrane_potential[-n_bins:]
                    mu_V[e,i,n] = membrane_potential.mean()
                    sigma_V[e,i,n] = membrane_potential.std()
                    # TODO: add tau_V calculation (check cell below)

                    gsyn_exc = gsyn_exc[-n_bins:]
                    mu_ge[e,i,n] = gsyn_exc.mean()
                    sigma_ge[e,i,n] = gsyn_exc.std()
                    
                    gsyn_inh = gsyn_inh[-n_bins:]
                    mu_gi[e,i,n] = gsyn_inh.mean()
                    sigma_gi[e,i,n] = gsyn_inh.std()

        neuron_results = SingleNeuronResults(
            neuron_name=neuron_name,
            neuron_params=neuron_pars,
            results={
                'nu_e' : nu_e_grid,
                'nu_i' : nu_i_grid,
                'nu_out' : nu_out.mean(axis=2),
                'nu_out_std' : nu_out.std(axis=2),
                'mu_w' : mu_w.mean(axis=2)*1e3,  # convert from nA to pA
                'sigma_w' : mu_w.std(axis=2)*1e3,  # convert from nA to pA
                'mu_V' : mu_V.mean(axis=2),
                'sigma_V' : sigma_V.mean(axis=2),
                'tau_V' : tau_V.mean(axis=2),
                'mu_ge' : mu_ge.mean(axis=2),
                'sigma_ge' : sigma_ge.mean(axis=2),
                'mu_gi' : mu_gi.mean(axis=2),
                'sigma_gi' : sigma_gi.mean(axis=2),
            }
        )

        results[neuron_name] = neuron_results

    return results

def simulate_batch_fix_out(nu_out_range, nu_i_range, neurons, data_path, sim_pars, max_e_step=None, adaptive_step=True):
    """

    Parameters
    ----------
    nu_out_range : 1D np.ndarray
        Range of output rates.
    nu_i_range : 1D np.ndarray
        Range of inhibitory input rates.
    neurons : dictionary
        Dictionary with neuron parameters.
    data_path : str or Path
        Path to save the data.
    max_e_step : 1D np.ndarray, optional
        Range of adaptation values.
    """

    sim_time = sim_pars['simulation_time']  # ms
    dt = sim_pars['time_step']  # ms
    avg_window = sim_pars['averaging_window']  # ms
    avg_start = sim_time - avg_window  
    n_bins = int(sim_pars['averaging_window']/dt)
    seed = sim_pars['seed']

    results = dict()

    if type(data_path) == str:
        data_path = Path(data_path)

    for neuron_name, neuron_pars in neurons.items():
        print(neuron_name)

        tau_w = neuron_pars['neuron_params']['tau_w']
        if sim_time < tau_w*5:  # static solution may not be reached
            print(f"WARNING: Simulation time ({sim_time} ms)"
                  + f" too short for {neuron_name}")
        if avg_start < tau_w*5:  # starting averaging too early
            print(f"WARNING: Averaging window ({avg_window} ms)"
                  + f" too long for {neuron_name}")

        nu_e_mesh = np.zeros((nu_out_range.size, nu_i_range.size))
        nu_out = np.zeros((nu_out_range.size, nu_i_range.size, sim_pars['n_runs']))
        mu_w = np.zeros_like(nu_out)
        mu_V = np.zeros_like(nu_out)
        sigma_V = np.zeros_like(nu_out)
        tau_V = np.zeros_like(nu_out)
        mu_ge = np.zeros_like(nu_out)
        sigma_ge = np.zeros_like(nu_out)
        mu_gi = np.zeros_like(nu_out)
        sigma_gi = np.zeros_like(nu_out)

        for i, nu_i in enumerate(nu_i_range):
            print(nu_i)
            filled = np.zeros(nu_out_range.size, dtype=bool)
            while not filled.all():
                # print(filled)
                e = filled.argmin()  # lowest index of unfilled
                f = None
                if np.any(filled[e:]):
                    # There is already a filled value after e
                    # Chose nu_e such that it is in between the filled values
                    f = filled[e:].argmax() + e  # first filled value after e
                    nu_e = (nu_e_mesh[e-1,i] + nu_e_mesh[f,i])/2
                elif e == 0:
                    # First value
                    nu_e = 0
                else:
                    # There is no filled value after e
                    # Make a step from the previous filled value
                    nu_e = nu_e_mesh[e-1,i] + max_e_step
                while True:
                    sim_data = simulate_single_adex_neuron(nu_e, nu_i, **neuron_pars, **sim_pars)

                    spikes = sim_data['spikes']
                    activity = spikes[spikes > avg_start].size/(avg_window*1e-3)

                    # print(f"Simulating {neuron_name} nu_e={nu_e} Hz, nu_i={nu_i} Hz, nu_out={activity} Hz")
                    if (activity <= nu_out_range[e:f]).any():
                        # print("Good candidate")

                        # computed activity is good candidate for filling
                        # we have to determine the right slot to fill
                        e = (activity <= nu_out_range[e:f]).argmax() + e
                        filled[e] = True
                        break
                    else:
                        nu_e = (nu_e + nu_e_mesh[e-1,i])/2
                nu_e_mesh[e, i] = nu_e
                n = 0
                while True:  # For loop to repeat n times
                    spikes = sim_data['spikes']
                    activity = spikes[spikes > avg_start].size/(avg_window*1e-3)
                    nu_out[e,i, n] = activity
                
                    membrane_potential = sim_data['v']
                    w = sim_data['w']
                    gsyn_exc = sim_data['gsyn_exc']
                    gsyn_inh = sim_data['gsyn_inh']

                    mu_w[e,i,n] = w[-n_bins:].mean()
                    membrane_potential = membrane_potential[-n_bins:]
                    mu_V[e,i,n] = membrane_potential.mean()
                    sigma_V[e,i,n] = membrane_potential.std()
                    # TODO: add tau_V calculation (check cell below)

                    gsyn_exc = gsyn_exc[-n_bins:]
                    mu_ge[e,i,n] = gsyn_exc.mean()
                    sigma_ge[e,i,n] = gsyn_exc.std()
                    
                    gsyn_inh = gsyn_inh[-n_bins:]
                    mu_gi[e,i,n] = gsyn_inh.mean()
                    sigma_gi[e,i,n] = gsyn_inh.std()
                    if n == sim_pars['n_runs'] - 1:
                        sim_pars['seed'] = seed
                        break
                    n += 1
                    sim_pars['seed'] = seed + n
                    sim_data = simulate_single_adex_neuron(nu_e, nu_i, **neuron_pars, **sim_pars)

        nu_i_grid = np.stack([nu_i_range for _ in range(nu_out_range.size)], axis=0)

        neuron_results = SingleNeuronResults(
            neuron_name=neuron_name,
            neuron_params=neuron_pars,
            results={
                'nu_e' : nu_e_mesh,
                'nu_i' : nu_i_grid,
                'nu_out' : nu_out.mean(axis=2),
                'nu_out_std' : nu_out.std(axis=2),
                'mu_w' : mu_w.mean(axis=2)*1e3,  # convert from nA to pA
                'sigma_w' : mu_w.std(axis=2)*1e3,  # convert from nA to pA
                'mu_V' : mu_V.mean(axis=2),
                'sigma_V' : sigma_V.mean(axis=2),
                'tau_V' : tau_V.mean(axis=2),
                'mu_ge' : mu_ge.mean(axis=2),
                'sigma_ge' : sigma_ge.mean(axis=2),
                'mu_gi' : mu_gi.mean(axis=2),
                'sigma_gi' : sigma_gi.mean(axis=2),
            }
        )

        results[neuron_name] = neuron_results

    return results


# Multiprocessing 

# Define worker function outside of the main function
def adex_simulation_worker(task, neuron_pars, sim_pars, n_bins, avg_start, avg_window):
    nu_e, nu_i, e, i, n, run_seed, neuron_name = task
    print(f"Simulating {neuron_name} nu_e={nu_e} Hz, nu_i={nu_i} Hz, run {n+1}/{sim_pars['n_runs']}")
    
    # Create a copy of sim_pars with the specific seed
    run_sim_pars = sim_pars.copy()
    run_sim_pars['seed'] = run_seed
    
    # Run the simulation
    sim_data = simulate_single_adex_neuron(nu_e, nu_i, **neuron_pars, **run_sim_pars)
    
    # Extract data
    spikes = sim_data['spikes']
    membrane_potential = sim_data['v']
    w = sim_data['w']
    gsyn_exc = sim_data['gsyn_exc']
    gsyn_inh = sim_data['gsyn_inh']
    
    # Calculate metrics
    run_nu_out = spikes[spikes > avg_start].size/(avg_window*1e-3)
    run_mu_w = w[-n_bins:].mean()
    
    membrane_potential = membrane_potential[-n_bins:]
    run_mu_V = membrane_potential.mean()
    run_sigma_V = membrane_potential.std()
    run_tau_V = 0  # TODO: implement tau_V calculation
    
    gsyn_exc = gsyn_exc[-n_bins:]
    run_mu_ge = gsyn_exc.mean()
    run_sigma_ge = gsyn_exc.std()
    
    gsyn_inh = gsyn_inh[-n_bins:]
    run_mu_gi = gsyn_inh.mean()
    run_sigma_gi = gsyn_inh.std()
    
    # Return results with indices
    return (e, i, n, {
        'nu_out': run_nu_out,
        'mu_w': run_mu_w,
        'mu_V': run_mu_V,
        'sigma_V': run_sigma_V,
        'tau_V': run_tau_V,
        'mu_ge': run_mu_ge,
        'sigma_ge': run_sigma_ge,
        'mu_gi': run_mu_gi,
        'sigma_gi': run_sigma_gi
    })

def _determine_nu_e_grid(nu_out_range, i, nu_i, neuron_pars, sim_pars):
    filled = np.zeros(nu_out_range.size, dtype=bool)
    nu_e_range = np.zeros(nu_out_range.size)

    sim_time = sim_pars['simulation_time']  # ms
    avg_window = sim_pars['averaging_window']  # ms
    avg_start = sim_time - avg_window  
    
    max_e_step = sim_pars['max_nu_e_step']
    if max_e_step is None:
        max_e_step = nu_out_range[1] - nu_out_range[0]
    while not filled.all():
        print(f"For nu_i={nu_i} Hz, {filled.sum()}/{nu_out_range.size} nu_e determined")
        e = filled.argmin()  # lowest index of unfilled
        f = None
        if np.any(filled[e:]):
            # There is already a filled value after e
            # Chose nu_e such that it is in between the filled values
            f = filled[e:].argmax() + e  # first filled value after e
            nu_e = (nu_e_range[e-1] + nu_e_range[f])/2
        elif e == 0:
            # First value
            nu_e = 0
        else:
            # There is no filled value after e
            # Make a step from the previous filled value
            nu_e = nu_e_range[e-1] + max_e_step
        while True:
            sim_data = simulate_single_adex_neuron(nu_e, nu_i, **neuron_pars, **sim_pars)
            spikes = sim_data['spikes']
            activity = spikes[spikes > avg_start].size/(avg_window*1e-3)
            # print(f"Simulating {neuron_name} nu_e={nu_e} Hz, nu_i={nu_i} Hz, nu_out={activity} Hz")
            if (activity <= nu_out_range[e:f]).any():
                # print("Good candidate")
                # computed activity is good candidate for filling
                # we have to determine the right slot to fill
                e = (activity <= nu_out_range[e:f]).argmax() + e
                filled[e] = True
                break
            else:
                nu_e = (nu_e + nu_e_range[e-1])/2
        nu_e_range[e] = nu_e
    return i, nu_e_range

def simulate_adex_batch_multiprocess(neurons, data_path, sim_pars):
    """
    Parallelized version of the AdEx batch simulation.

    Parameters
    ----------
    nu_e_range : 1D np.ndarray
        Range of excitatory input rates.
    nu_i_range : 1D np.ndarray
        Range of inhibitory input rates.
    neurons : dictionary
        Dictionary with neuron parameters.
    data_path : str or Path
        Path to save the data.
    """
    sim_time = sim_pars['simulation_time']  # ms
    dt = sim_pars['time_step']  # ms
    avg_window = sim_pars['averaging_window']  # ms
    avg_start = sim_time - avg_window  
    n_bins = int(sim_pars['averaging_window']/dt)
    seed = sim_pars['seed']

    results = dict()

    if type(data_path) == str:
        data_path = Path(data_path)

    for neuron_name, neuron_pars in neurons.items():
        print(neuron_name)

        tau_w = neuron_pars['neuron_params']['tau_w']
        if sim_time < tau_w*5:  # static solution may not be reached
            print(f"WARNING: Simulation time ({sim_time} ms)"
                  + f" too short for {neuron_name}")
        if avg_start < tau_w*5:  # starting averaging too early
            print(f"WARNING: Averaging window ({avg_window} ms)"
                  + f" too long for {neuron_name}")

        nu_i_range = np.linspace(*sim_pars['nu_i_range'])
        nu_e_range = np.linspace(*sim_pars['nu_e_range'])
        nu_out_range = np.linspace(*sim_pars['nu_out_range'])


        if sim_pars['fix_nu_out']:
            nu_i_grid = np.stack([nu_i_range for _ in range(nu_out_range.size)], axis=0)
            nu_e_grid = np.zeros((nu_out_range.size, nu_i_range.size))
            tasks = []
            for i, nu_i in enumerate(nu_i_range):
                tasks.append((nu_out_range, i, nu_i, neuron_pars, sim_pars))
            with mp.Pool(processes=sim_pars["cpus"]) as pool:
                for i, nu_e_range in pool.starmap(_determine_nu_e_grid, tasks):
                    nu_e_grid[:,i] = nu_e_range
            print("Grid of nu_e determined")
        else:
            nu_e_grid, nu_i_grid = np.meshgrid(nu_e_range, nu_i_range, indexing='ij')

        # Define shape of result arrays
        nu_out = np.zeros((*nu_e_grid.shape, sim_pars['n_runs']))
        mu_w = np.zeros_like(nu_out)
        mu_V = np.zeros_like(nu_out)
        sigma_V = np.zeros_like(nu_out)
        tau_V = np.zeros_like(nu_out)
        mu_ge = np.zeros_like(nu_out)
        sigma_ge = np.zeros_like(nu_out)
        mu_gi = np.zeros_like(nu_out)
        sigma_gi = np.zeros_like(nu_out)

        # Create a task list to parallelize
        tasks = []
        for i, nu_i in enumerate(nu_i_range):
            for e, nu_e in enumerate(nu_e_grid[:,i]):
                for n in range(sim_pars["n_runs"]):
                    # Store indices along with parameters
                    tasks.append((nu_e, nu_i, e, i, n, seed + n, neuron_name))

        # Create a partial function with fixed parameters
        worker_partial = partial(
            adex_simulation_worker,
            neuron_pars=neuron_pars,
            sim_pars=sim_pars,
            n_bins=n_bins,
            avg_start=avg_start,
            avg_window=avg_window
        )
        
        # Execute tasks in parallel
        with mp.Pool(processes=sim_pars["cpus"]) as pool:
            for e, i, n, result in pool.map(worker_partial, tasks):
                # Store results back in the appropriate arrays
                nu_out[e, i, n] = result['nu_out']
                mu_w[e, i, n] = result['mu_w']
                mu_V[e, i, n] = result['mu_V']
                sigma_V[e, i, n] = result['sigma_V']
                tau_V[e, i, n] = result['tau_V']
                mu_ge[e, i, n] = result['mu_ge']
                sigma_ge[e, i, n] = result['sigma_ge']
                mu_gi[e, i, n] = result['mu_gi']
                sigma_gi[e, i, n] = result['sigma_gi']
        
        neuron_results = SingleNeuronResults(
            neuron_name=neuron_name,
            neuron_params=neuron_pars,
            results={
                'nu_e' : nu_e_grid,
                'nu_i' : nu_i_grid,
                'nu_out' : nu_out.mean(axis=2),
                'nu_out_std' : nu_out.std(axis=2),
                'mu_w' : mu_w.mean(axis=2)*1e3,  # convert from nA to pA
                'sigma_w' : mu_w.std(axis=2)*1e3,  # convert from nA to pA
                'mu_V' : mu_V.mean(axis=2),
                'sigma_V' : sigma_V.mean(axis=2),
                'tau_V' : tau_V.mean(axis=2),
                'mu_ge' : mu_ge.mean(axis=2),
                'sigma_ge' : sigma_ge.mean(axis=2),
                'mu_gi' : mu_gi.mean(axis=2),
                'sigma_gi' : sigma_gi.mean(axis=2),
            }
        )

        results[neuron_name] = neuron_results

    return results

def run_single_neuron_workflow(single_sim_pars, neurons, file_name, results_path, data_path):
    neuron_names = list(neurons.keys())
    if single_sim_pars["load_single_neuron"]:         # Load neuron simulation results
        nu_e_size = single_sim_pars["nu_e_range"][2]
        nu_i_size = single_sim_pars["nu_i_range"][2]
        if single_sim_pars["fix_nu_out"]:
            grid = "grid-irregular"
        else:
            grid = "grid-regular"
        file_pattern = f'{file_name}_{{}}_{nu_e_size}x{nu_i_size}_{grid}.pickle'
        try:
            # neuron_results = load_data(neuron_names, file_pattern, data_path)
            neuron_results = load_results(data_path, neuron_names, file_pattern)
            neurons = {neuron_name: result.neuron_params for neuron_name, result in neuron_results.items()}
            single_sim_pars["simulate_single_neuron"] = False
            print("Loaded neurons successfully.")
        except FileNotFoundError:
            print(f"File {file_pattern} not found in {data_path}.")
            print("Will run the simulations.")
            single_sim_pars["simulate_single_neuron"] = True

    if single_sim_pars["simulate_single_neuron"]:  # Do single neuron simulations
        if single_sim_pars["cpus"] > 1:  # Multiprocess simulation
            neuron_results = simulate_adex_batch_multiprocess(neurons, 
                                            results_path, single_sim_pars)

        elif single_sim_pars["fix_nu_out"]:  
            # Adjusting nu_e, so that nu_out is reasonably distributed
            nu_out_range = np.linspace(*single_sim_pars['nu_out_range'])
            nu_i_range = np.linspace(*single_sim_pars['nu_i_range'])
            neuron_results = simulate_batch_fix_out(nu_out_range, nu_i_range, neurons, 
                                                results_path, single_sim_pars, max_e_step=single_sim_pars["max_nu_e_step"])

        else:  # Fixed grid of nu_e, nu_i
            nu_e_range = np.linspace(*single_sim_pars['nu_e_range'])
            nu_i_range = np.linspace(*single_sim_pars['nu_i_range'])
            neuron_results = simulate_adex_batch(nu_e_range, nu_i_range, neurons, 
                                                results_path, single_sim_pars)

    return neuron_results



