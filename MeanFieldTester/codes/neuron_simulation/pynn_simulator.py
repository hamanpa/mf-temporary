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

"""


from pathlib import Path
import importlib
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import NeuronSimulationConfig
from ..data_structures.neuron_simulation import SingleNeuronResults
from .base import BaseNeuronSimulator
from ..network_params.translators import TranslationRule, translate_params
from ..network_params.mappings import PYNN_ADEX_MAPPING, PYNN_STATIC_SYNAPSE_MAPPING, NEST_STATIC_SYNAPSE_MAPPING, NEST_TSODYKS_SYNAPSE_MAPPING, PYNN_INITIAL_VALUES_MAPPING
from ..network_params.models import BiologicalParameters

def simulate_adex_neuron_single_point(
                                exc_rate : float, 
                                inh_rate : float,
                                neuron_params : dict, 
                                init_values : dict, 
                                exc_synapses : dict,
                                inh_synapses : dict,
                                drive_rate : float = 0.0,
                                drive_synapses : dict | None = None,
                                simulation_time=1000.0,
                                time_step=0.1,
                                seed=1,
                                **kwargs) -> dict:
    """Simulates a single AdEx neuron with Poisson synaptic input.

    Parameters:
        exc_rate (float): Rate of excitatory Poisson input (Hz).
        inh_rate (float): Rate of inhibitory Poisson input (Hz).
        neuron_params (dict): Parameters for the AdEx neuron model.
        init_values (dict): Initial values for the neuron state variables.
        exc_synapses (dict): Parameters for the excitatory synapse model.
        inh_synapses (dict): Parameters for the inhibitory synapse model.
        drive_rate (float): Rate of each drive Poisson source (Hz). 0 means no drive input.
        drive_synapses (dict): Synapse model, number of sources and receptor type of the drive input
                               (required if drive_rate > 0). Its conductance is recorded in 'gsyn_exc'
                               (or 'gsyn_inh') together with the other inputs of that receptor.
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
    sim.setup(timestep=time_step, rng_seed=seed)

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

    if drive_rate > 0:
        if drive_synapses is None:
            raise ValueError("drive_rate > 0 requires drive_synapses (the neuron has no drive connection).")
        synapse_drive = sim.native_synapse_type(drive_synapses["syn_type"])(**drive_synapses["syn_params"])
        poisson_input_drive = sim.Population(drive_synapses["number"], sim.SpikeSourcePoisson(rate=drive_rate))
        sim.Projection(poisson_input_drive, neuron, connector, synapse_type=synapse_drive, receptor_type=drive_synapses["receptor_type"])

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

GRID_METRICS = (
    "out_rate",
    "adaptation_mean", "adaptation_std",
    "voltage_mean", "voltage_std", "voltage_tau",
    "exc_conductance_mean", "exc_conductance_std",
    "inh_conductance_mean", "inh_conductance_std",
)


def _single_point_sim_params(neuron_sim_params: NeuronSimulationConfig, n_run: int) -> dict:
    """Picklable simulation parameters of one run (seed shifted by the run index)."""
    return {
        "simulator": neuron_sim_params.simulator,
        "seed": neuron_sim_params.seed + n_run,
        "simulation_time": neuron_sim_params.simulation_time,
        "time_step": neuron_sim_params.time_step,
        "averaging_window": neuron_sim_params.averaging_window,
    }


def _grid_tasks(neuron_name: str, neuron_params: dict,
                exc_rate_grid: np.ndarray, inh_rate_grid: np.ndarray, drive_rate_grid: np.ndarray,
                neuron_sim_params: NeuronSimulationConfig) -> list:
    """One task per grid point (exc, inh, drive) and run."""
    tasks = []
    for grid_idx in np.ndindex(exc_rate_grid.shape):
        rates = (exc_rate_grid[grid_idx], inh_rate_grid[grid_idx], drive_rate_grid[grid_idx])
        for n_run in range(neuron_sim_params.n_runs):
            tasks.append((rates, grid_idx, n_run, neuron_name, neuron_params, _single_point_sim_params(neuron_sim_params, n_run)))
    return tasks


def _adex_neuron_worker(task_data):
    """Top-level worker to allow pickling across processes.
    
    This is a helper function so that we can use multiprocessing.Pool to run
    simulations in parallel. 
    
    It unpacks the task data, runs a single simulation, 
    computes the metrics, and returns the results along with the indices 
    for where to store them in the result arrays.
    
    """
    (exc_rate, inh_rate, drive_rate), grid_idx, n_run, neuron_name, neuron_params, neuron_sim_params = task_data
    
    # Run simulation
    sim_data = simulate_adex_neuron_single_point(
        exc_rate, inh_rate, drive_rate=drive_rate, **neuron_params, **neuron_sim_params
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
    
    return (grid_idx, n_run, {
        'out_rate': out_rate,
        'adaptation_mean': adaptation_steady.mean(),
        'adaptation_std': adaptation_steady.std(),
        'voltage_mean': voltage_steady.mean(),
        'voltage_std': voltage_steady.std(),
        'voltage_tau': 0,  # not computed yet (see todo.md)
        'exc_conductance_mean': exc_conductance_steady.mean(),
        'exc_conductance_std': exc_conductance_steady.std(),
        'inh_conductance_mean': inh_conductance_steady.mean(),
        'inh_conductance_std': inh_conductance_steady.std()
    })


def _collect_grid_results(neuron_name: str, neuron_params: dict,
                          exc_rate_grid: np.ndarray, inh_rate_grid: np.ndarray, drive_rate_grid: np.ndarray,
                          neuron_sim_params: NeuronSimulationConfig, worker_results) -> SingleNeuronResults:
    """Stores the worker results (in any order) into (exc, inh, drive, run) arrays and averages over runs."""
    per_run = {metric: np.zeros(exc_rate_grid.shape + (neuron_sim_params.n_runs,)) for metric in GRID_METRICS}
    for grid_idx, n_run, metrics in worker_results:
        for metric, value in metrics.items():
            per_run[metric][grid_idx + (n_run,)] = value

    run_mean = {metric: values.mean(axis=-1) for metric, values in per_run.items()}
    return SingleNeuronResults(
        simulator_name=neuron_sim_params.simulator,
        neuron_name=neuron_name,
        neuron_params=neuron_params,
        exc_rate_grid=exc_rate_grid,
        inh_rate_grid=inh_rate_grid,
        drive_rate_grid=drive_rate_grid,
        out_rate_mean=run_mean["out_rate"],
        out_rate_std=per_run["out_rate"].std(axis=-1),
        **{metric: run_mean[metric] for metric in GRID_METRICS if metric != "out_rate"},
        input_units = {
            # PyNN records conductances in [uS]
            "exc_conductance_mean" : "uS",
            "exc_conductance_std" : "uS",
            "inh_conductance_mean" : "uS",
            "inh_conductance_std" : "uS",
        },
    )


def simulate_adex_neuron_full_grid(neuron_name: str, neuron_params: dict, 
                                exc_rate_grid: np.ndarray, inh_rate_grid: np.ndarray, drive_rate_grid: np.ndarray,
                                neuron_sim_params: NeuronSimulationConfig) -> SingleNeuronResults:
    """
    Simulates a single AdEx neuron across a 3D grid of excitatory, inhibitory and drive input rates (serially).
    """
    tasks = _grid_tasks(neuron_name, neuron_params, exc_rate_grid, inh_rate_grid, drive_rate_grid, neuron_sim_params)
    print(f"Simulating {neuron_name}: {len(tasks)} tasks serially...")
    return _collect_grid_results(
        neuron_name, neuron_params, exc_rate_grid, inh_rate_grid, drive_rate_grid, neuron_sim_params,
        (_adex_neuron_worker(task) for task in tasks),
    )


def simulate_adex_neuron_full_grid_multiprocess(neuron_name: str, neuron_params: dict, 
                                             exc_rate_grid: np.ndarray, inh_rate_grid: np.ndarray, drive_rate_grid: np.ndarray,
                                             neuron_sim_params: NeuronSimulationConfig) -> SingleNeuronResults:
    """Parallel version of `simulate_adex_neuron_full_grid` (un-ordered Pool; results are placed by index)."""
    tasks = _grid_tasks(neuron_name, neuron_params, exc_rate_grid, inh_rate_grid, drive_rate_grid, neuron_sim_params)
    print(f"Starting multiprocessing for {neuron_name}: {len(tasks)} tasks across {neuron_sim_params.cpus} CPUs...")

    with mp.Pool(processes=neuron_sim_params.cpus) as pool:
        results = _collect_grid_results(
            neuron_name, neuron_params, exc_rate_grid, inh_rate_grid, drive_rate_grid, neuron_sim_params,
            pool.imap_unordered(_adex_neuron_worker, tasks),
        )

    print(f"Finished {neuron_name} multiprocessing batch.")
    return results

# Dealing with the grid

def find_exc_rate_max_for_out_rate_target(neuron_params, neuron_sim_params_dict, inh_rate, out_rate_target, max_input_rate=500.0, rel_tol=0.1, max_iter=100, drive_rate=0.0):
    """Finds the upper boundary nu_e using a fast geometric expansion and rough bisection."""
    # (Same helper to get the rate)
    def get_rate(exc_rate):
        data = simulate_adex_neuron_single_point(exc_rate, inh_rate, drive_rate=drive_rate, **neuron_params, **neuron_sim_params_dict)
        spikes = data['spikes']
        avg_window = neuron_sim_params_dict['averaging_window']
        return spikes[spikes > (neuron_sim_params_dict['simulation_time'] - avg_window)].size / (avg_window * 1e-3)

    exc_rate_high = 1.0
    while get_rate(exc_rate_high) < out_rate_target:
        exc_rate_high *= 2.0
        if exc_rate_high >= max_input_rate:
            return max_input_rate

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
    (inh_rate_idx, inh_rate, drive_rate_idx, drive_rate, out_rate_targets, out_rate_max, max_input_rate, skip_zeros, n_coarse_points,
     single_neuron_params, neuron_sim_params_dict) = task_data

    try:
        # Find the maximum excitatory rate needed to reach out_rate_max
        exc_rate_max = find_exc_rate_max_for_out_rate_target(
            single_neuron_params, neuron_sim_params_dict, inh_rate, out_rate_max, max_input_rate=max_input_rate,
            drive_rate=drive_rate,
        )

        exc_rate_grid_coarse = np.linspace(0, exc_rate_max, n_coarse_points)
        out_rate_values_coarse = np.zeros(n_coarse_points)

        sim_time = neuron_sim_params_dict['simulation_time']
        avg_window = neuron_sim_params_dict['averaging_window']
        
        # Run coarse simulations
        for exc_rate_idx, exc_rate_test in enumerate(exc_rate_grid_coarse):
            data = simulate_adex_neuron_single_point(exc_rate_test, inh_rate, drive_rate=drive_rate, **single_neuron_params, **neuron_sim_params_dict)
            spikes = data['spikes']
            out_rate_values_coarse[exc_rate_idx] = spikes[spikes > (sim_time - avg_window)].size / (avg_window * 1e-3)

        # Find rheobase_exc: highest input rate with out_rate == 0.0 in coarse data
        zero_indices = np.where(out_rate_values_coarse == 0.0)[0]
        if len(zero_indices) > 0 and zero_indices[-1] < len(exc_rate_grid_coarse) - 1:
            rheobase_idx = zero_indices[-1]
            rheobase_exc = exc_rate_grid_coarse[rheobase_idx]
        else:
            rheobase_idx = 0
            rheobase_exc = exc_rate_grid_coarse[0]

        if skip_zeros:
            # Mode 1 (skip_zeros=True): Start grid at the activity threshold (rheobase_exc)
            out_rate_unique_list = [0.0]
            exc_rate_unique_list = [rheobase_exc]
            start_scan_idx = rheobase_idx + 1
        else:
            # Mode 2 (skip_zeros=False): Start grid at 0.0 Hz
            out_rate_unique_list = [out_rate_values_coarse[0]]
            exc_rate_unique_list = [exc_rate_grid_coarse[0]]
            start_scan_idx = 1

        for i in range(start_scan_idx, len(out_rate_values_coarse)):
            if out_rate_values_coarse[i] > out_rate_unique_list[-1]:
                out_rate_unique_list.append(out_rate_values_coarse[i])
                exc_rate_unique_list.append(exc_rate_grid_coarse[i])

        out_rate_unique = np.array(out_rate_unique_list)
        exc_rate_unique = np.array(exc_rate_unique_list)

        out_n_points = len(out_rate_targets)
        exc_rate_column = np.zeros(out_n_points)
        max_step = max_input_rate / max(1, out_n_points - 1)

        if skip_zeros:
            # Mode 1: All out_n_points are allocated to target output rates starting at rheobase_exc
            if len(out_rate_unique) > 1:
                inverse_f_I_curve = PchipInterpolator(out_rate_unique, exc_rate_unique)
                safe_targets = np.clip(out_rate_targets, out_rate_unique.min(), out_rate_unique.max())
                ideal_exc_rates = inverse_f_I_curve(safe_targets)

                exc_rate_column[0] = ideal_exc_rates[0]
                for i in range(1, out_n_points):
                    exc_rate_column[i] = max(
                        exc_rate_column[i - 1],
                        min(ideal_exc_rates[i], exc_rate_column[i - 1] + max_step)
                    )
                exc_rate_column = np.clip(exc_rate_column, 0.0, max_input_rate)
            else:
                exc_rate_column[:] = rheobase_exc
        else:
            # Mode 2: Step through sub-threshold region with max_step up to rheobase_exc,
            # then allocate all remaining points to target output rates (no gaps in nu_out)
            if len(out_rate_unique) > 1:
                sub_thresh_points = []
                curr = 0.0
                while curr < rheobase_exc - 1e-6 and len(sub_thresh_points) < out_n_points - 1:
                    sub_thresh_points.append(curr)
                    curr += max_step

                if rheobase_exc > 0.0 and len(sub_thresh_points) < out_n_points:
                    sub_thresh_points.append(rheobase_exc)

                M = len(sub_thresh_points)
                exc_rate_column[:M] = sub_thresh_points

                rem_points = out_n_points - M
                if rem_points > 0:
                    inverse_f_I_curve = PchipInterpolator(out_rate_unique, exc_rate_unique)
                    out_min, out_max = out_rate_targets[0], out_rate_targets[-1]
                    active_targets = np.linspace(out_min, out_max, rem_points)
                    safe_active = np.clip(active_targets, out_rate_unique.min(), out_rate_unique.max())
                    ideal_active_rates = inverse_f_I_curve(safe_active)

                    for j in range(rem_points):
                        idx = M + j
                        ideal = ideal_active_rates[j]
                        prev = exc_rate_column[idx - 1] if idx > 0 else 0.0
                        exc_rate_column[idx] = max(prev, min(ideal, prev + max_step))

                exc_rate_column = np.clip(exc_rate_column, 0.0, max_input_rate)
            else:
                # Fallback if the neuron is completely dead (never spiked)
                exc_rate_column[:] = 0.0
        
        inh_rate_column = np.full(out_n_points, inh_rate)

        return inh_rate_idx, drive_rate_idx, exc_rate_column, inh_rate_column
    except Exception as e:
        import traceback
        err_msg = f"Error in adaptive grid worker for inh_rate={inh_rate:.2f} Hz, drive_rate={drive_rate:.2f} Hz: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        raise RuntimeError(err_msg) from None


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
    exc_rate_grid, inh_rate_grid, drive_rate_grid : np.ndarray
        3D arrays indexed (exc_rate, inh_rate, drive_rate), shape (n_out_rates, n_inh_rates, n_drive_rates).
        The adaptive exc axis is resolved separately for every (inh_rate, drive_rate) pair.
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
    drive_rate_min, drive_rate_max, drive_n_points = grid_params.drive_rate_grid
    drive_rates = np.linspace(drive_rate_min, drive_rate_max, int(drive_n_points))


    # NOTE: once implementing general adaptive grid, this part needs to be refactored
    # because I require grid to be indexed (exc_rate, inh_rate) for the rest of the code,
    # so if we want to do inh adaptive grid we have to flip the indexing and 
    # be careful with the interpolation and the way we fill the grids, etc.
    # I guess I could keep this and then just transpose or reorder based on the neuron types

    exc_rate_grid = np.zeros((out_n_points, inh_n_points, drive_rates.size))
    inh_rate_grid = np.zeros((out_n_points, inh_n_points, drive_rates.size))
    drive_rate_grid = np.broadcast_to(drive_rates, exc_rate_grid.shape).copy()

    n_coarse_points = grid_params.n_coarse_interpolation_points
    cpus = neuron_sim_params.cpus
    max_input_rate = grid_params.max_input_rate
    skip_zeros = grid_params.skip_zeros

    neuron_sim_params_dict = {
        "simulator": neuron_sim_params.simulator,
        "seed": neuron_sim_params.seed,
        "simulation_time": neuron_sim_params.simulation_time,
        "time_step": neuron_sim_params.time_step,
        "averaging_window": neuron_sim_params.averaging_window,
    }
    print(f"Resolving adaptive grid for {neuron_name} using {cpus} CPUs...")
    # Build tasks
    tasks = []
    for drive_rate_idx, drive_rate in enumerate(drive_rates):
        for inh_rate_idx, inh_rate in enumerate(inh_rates):
            tasks.append((
                inh_rate_idx, inh_rate, drive_rate_idx, drive_rate,
                out_rate_targets, out_rate_max, max_input_rate, skip_zeros, n_coarse_points,
                neuron_params, neuron_sim_params_dict
            ))


    with mp.Pool(processes=cpus) as pool:
        for result in pool.imap_unordered(_resolve_adaptive_grid_worker, tasks):
            inh_rate_idx, drive_rate_idx, exc_rate_col, inh_rate_col = result
            exc_rate_grid[:, inh_rate_idx, drive_rate_idx] = exc_rate_col
            inh_rate_grid[:, inh_rate_idx, drive_rate_idx] = inh_rate_col
            print(f"    Finished interpolation for inh_rate = {inh_rate_col[0]:.2f} Hz, drive_rate = {drive_rates[drive_rate_idx]:.2f} Hz")

    return exc_rate_grid, inh_rate_grid, drive_rate_grid

class PyNNSimulator(BaseNeuronSimulator):
    """PyNN implementation of the single neuron simulator."""


    def resolve_grid(self, neuron_name: str, neuron_params: dict, neuron_sim_params: NeuronSimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolves the 3D input grid for a specific neuron based on the configuration.
        
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
        exc_rate_grid, inh_rate_grid, drive_rate_grid : np.ndarray
            3D arrays of the input rates [Hz], indexed (exc_rate, inh_rate, drive_rate).
        """
        grid_params = getattr(neuron_sim_params.grid, neuron_name)
        
        match grid_params.grid_type:
            case "linear":
                axes = [
                    np.linspace(rate_min, rate_max, int(n_points))
                    for rate_min, rate_max, n_points in (grid_params.exc_rate_grid, grid_params.inh_rate_grid, grid_params.drive_rate_grid)
                ]
                exc_rate_grid, inh_rate_grid, drive_rate_grid = np.meshgrid(*axes, sparse=False, indexing='ij')
                
            case "custom":
                # NOTE: the custom grids are loaded and validated (3D, same shape) by the config
                exc_rate_grid = grid_params.exc_rate_grid
                inh_rate_grid = grid_params.inh_rate_grid
                drive_rate_grid = grid_params.drive_rate_grid
 
            case "adaptive":
                print(f"Resolving adaptive grid for {neuron_name}...")
                exc_rate_grid, inh_rate_grid, drive_rate_grid = resolve_adaptive_grid(neuron_name, neuron_params, neuron_sim_params)

            case _:
                raise ValueError(f"Unknown grid type: {grid_params.grid_type}")

        return exc_rate_grid, inh_rate_grid, drive_rate_grid


    def simulate(self, network_params: BiologicalParameters, neuron_sim_params: NeuronSimulationConfig) -> dict:
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
        for neuron_name in network_params.internal_neurons:
            single_neuron_params = network_params.neurons[neuron_name]
            print(f"\n{'='*50}\nPreparing simulation for {neuron_name}\n{'='*50}")


            exc_conn = network_params.network.connectivity[neuron_name][network_params.exc_neuron_name]
            inh_conn = network_params.network.connectivity[neuron_name][network_params.inh_neuron_name]

            exc_syn_num = exc_conn.conn_num
            inh_syn_num = inh_conn.conn_num

            exc_synapse_mapping = NEST_TSODYKS_SYNAPSE_MAPPING if exc_conn.syn_type == "tsodyks_synapse" else NEST_STATIC_SYNAPSE_MAPPING
            inh_synapse_mapping = NEST_TSODYKS_SYNAPSE_MAPPING if inh_conn.syn_type == "tsodyks_synapse" else NEST_STATIC_SYNAPSE_MAPPING

            legacy_neuron_params = {
                'neuron_params' : translate_params(single_neuron_params.neuron_params, PYNN_ADEX_MAPPING),
                'init_values' : translate_params(neuron_sim_params.init_values[neuron_name], PYNN_INITIAL_VALUES_MAPPING),
                'exc_synapses' : {
                    'syn_type' : exc_conn.syn_type,
                    'syn_params' : translate_params(exc_conn.syn_params, exc_synapse_mapping),
                    'number' : exc_syn_num
                },
                'inh_synapses' : {
                    'syn_type' : inh_conn.syn_type,
                    'syn_params' : translate_params(inh_conn.syn_params, inh_synapse_mapping),
                    'number' : inh_syn_num
                },
            }

            # Drive input (used for drive_rate > 0); its synapses and number of sources are as in the network
            # NOTE: the drive population is still identified by name (see todo.md, hard-coded population names)
            drive_conn = network_params.network.connectivity[neuron_name].get("drive_neuron")
            if drive_conn is not None:
                drive_synapse_mapping = NEST_TSODYKS_SYNAPSE_MAPPING if drive_conn.syn_type == "tsodyks_synapse" else NEST_STATIC_SYNAPSE_MAPPING
                legacy_neuron_params['drive_synapses'] = {
                    'syn_type' : drive_conn.syn_type,
                    'syn_params' : translate_params(drive_conn.syn_params, drive_synapse_mapping),
                    'number' : drive_conn.conn_num,
                    'receptor_type' : network_params.neurons["drive_neuron"].neuron_type,
                }

            exc_rate_grid, inh_rate_grid, drive_rate_grid = self.resolve_grid(neuron_name, legacy_neuron_params, neuron_sim_params)
            if drive_conn is None and np.any(drive_rate_grid > 0):
                raise ValueError(f"The grid of {neuron_name} has drive rates > 0, but {neuron_name} has no drive_neuron connection.")

            if neuron_sim_params.cpus > 1:
                neuron_result = simulate_adex_neuron_full_grid_multiprocess(
                    neuron_name, legacy_neuron_params, exc_rate_grid, inh_rate_grid, drive_rate_grid, neuron_sim_params
                )
            else:
                neuron_result = simulate_adex_neuron_full_grid(
                    neuron_name, legacy_neuron_params, exc_rate_grid, inh_rate_grid, drive_rate_grid, neuron_sim_params
                )

            results[neuron_name] = neuron_result

        return results