"""
Execution workflows for coordinating single-step simulations across SNN and MF backends.
Serves as the stateless execution engine for controllers and scripts.
"""

from typing import Dict, List, Union, Any
import os
import json

from ..neuron_simulation import run_neuron_simulation_workflow
from ..transfer_function import run_tf_fitting_workflow
from ..snn_simulation import run_snn_simulation_workflow, run_snn_batch_parallel, _snn_simulation_worker, SpikingNeuralNetworkSimulationConfig
from ..mf_simulation import run_mf_simulation_workflow, run_mf_batch_parallel, _mf_simulation_worker, MeanFieldSimulationConfig
import multiprocessing as mp

from ..network_params.models import BiologicalParameters
from ..stimuli.config import StimuliCollection, BaseStimulusConfig
from .config import WorkflowConfig

from ..data_structures.base import BaseSNNResults, BaseMFResults, BaseSingleNeuronResults


def run_basic_workflow(
    network_params: BiologicalParameters,
    stimulus_config: StimuliCollection,
    sim_params: WorkflowConfig,
) -> Dict[str, BaseSingleNeuronResults| BaseSNNResults | List[BaseMFResults]]:
    
    basic_results= {}


    basic_results["neuron_results"] = run_neuron_simulation_workflow(sim_params.neuron_simulation, network_params)

    tf_results_dict = {neuron_name: [] for neuron_name in network_params.internal_neurons}
    
    for mf_model_name, mf_sim_params in sim_params.mf_models.items():
        tf_results = run_tf_fitting_workflow(mf_sim_params.transfer_function, network_params, basic_results["neuron_results"])
        for neuron_name in tf_results:
            tf_results_dict[neuron_name].append(tf_results[neuron_name])

    basic_results["tf_results"] = tf_results_dict

    basic_results["snn_results"] = run_snn_simulation_workflow(
        sim_params.snn_simulation, network_params, stimulus_config
    )

    mf_results_dict = {stim_name: [] for stim_name in stimulus_config}
    for mf_model_name, mf_sim_params in sim_params.mf_models.items():
        mf_results_full = run_mf_simulation_workflow(
            mf_sim_params, network_params, stimulus_config
        )
        for stim_name, mf_results in mf_results_full.items():
            mf_results_dict[stim_name].append(mf_results)

    basic_results["mf_results"] = mf_results_dict

    return basic_results


def _unified_simulation_worker(task_tuple: tuple) -> Dict[str, Any]:
    """
    Top-level router worker function executed by worker processes in mp.Pool.
    Delegates to _snn_simulation_worker or _mf_simulation_worker based on worker_type.
    """
    (task_id, net_idx, worker_type, network_params, stim_idx, stim_name, stim_params, sim_name, sim_params, output_dir) = task_tuple

    worker_args = (task_id, net_idx, network_params, stim_idx, stim_name, stim_params, sim_name, sim_params, output_dir)

    if worker_type == "snn":
        return _snn_simulation_worker(worker_args)
    elif worker_type == "mf":
        return _mf_simulation_worker(worker_args)
    else:
        raise ValueError(f"Unknown worker_type '{worker_type}'. Expected 'snn' or 'mf'.")


def run_unified_batch_parallel(
    network_params: BiologicalParameters,
    stimuli: Dict[str, BaseStimulusConfig],
    snn_sim_params: SpikingNeuralNetworkSimulationConfig = None,
    mf_sim_params_dict: Dict[str, MeanFieldSimulationConfig] = None,
    net_idx: str = None,
    output_dir: str = "results/batch_run",
    cpus: int = None
) -> List[Dict[str, Any]]:
    """
    Unified parallel workflow orchestrator for executing SNN and/or Mean Field simulation batches.
    Creates a single Cartesian product queue of [Networks] x [Stimuli] x [SimConfigs] and dispatches
    all tasks concurrently across a process pool of size `cpus`.

    Parameters
    ----------
    network_params : BiologicalParameters
        Biological network parameter configuration.
    stimuli : Dict[str, BaseStimulusConfig], List[BaseStimulusConfig], or BaseStimulusConfig
        Stimulus configuration(s).
    network_sim_params : SpikingNeuralNetworkSimulationConfig, MeanFieldSimulationConfig, List, Dict, or WorkflowConfig
        One or multiple simulation configuration objects (SNN or MF). Can also be a full WorkflowConfig object.
    snn_sim_params : SpikingNeuralNetworkSimulationConfig, optional
        Legacy/explicit parameter for SNN simulation backend.
    mf_sim_params_dict : Dict[str, MeanFieldSimulationConfig], optional
        Legacy/explicit parameter for Mean Field simulation backend.
    output_dir : str
        Directory to store output .npz arrays and manifest.json.
    cpus : int, optional
        Number of CPU worker processes.

    Returns
    -------
    List[Dict[str, Any]]
        Full list of execution metadata records for all executed tasks.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sim_items = []  # tuples of (Name, worker_type, sim_cfg)
    if snn_sim_params is not None:
        sim_items.append(("SNN", "snn", snn_sim_params))
    if mf_sim_params_dict is not None:
        for name, cfg in mf_sim_params_dict.items():
            sim_items.append((name, "mf", cfg))
        
    if not sim_items:
        raise ValueError("No valid simulation parameters provided. Pass snn_sim_params, or mf_sim_params.")

    stim_items = list(stimuli.items())

    if cpus is None:
        print("No CPU count specified. Defaulting to 1 worker process for debugging.")
        cpus = 1

    # 4. Construct single Cartesian Product of Tasks: [Networks] x [Stimuli] x [SimConfigs]
    tasks = []
    task_id = 0

    for stim_idx, (stim_name, stim_params) in enumerate(stim_items):
        for sim_name, worker_type, sim_params in sim_items:
            tasks.append((
                task_id, net_idx, worker_type, 
                network_params,
                stim_idx, stim_name, stim_params, 
                sim_name, sim_params, 
                output_dir
            ))
            task_id += 1

    print(f"Launching Unified Parallel Batch: {len(tasks)} task(s) across {cpus} CPU worker(s)...")

    # 5. Multiprocess all tasks concurrently across `cpus` worker processes
    all_metadata = []
    with mp.Pool(processes=cpus) as pool:
        iterator = pool.imap_unordered(_unified_simulation_worker, tasks)
        for res in iterator:
            all_metadata.append(res)

    # 6. Write execution manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_data = {
        "output_dir": output_dir,
        "total_tasks": len(all_metadata),
        "tasks": all_metadata
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2, default=str)

    print(f"Batch execution complete. Full manifest saved to '{manifest_path}'.")
    return all_metadata