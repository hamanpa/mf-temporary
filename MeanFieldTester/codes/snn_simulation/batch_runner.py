import os
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Any, Union

from .pynn_simulator import PyNNSNNSimulator
from .config import SpikingNeuralNetworkSimulationConfig
from ..stimuli.config import BaseStimulusConfig
from ..network_params.models import BiologicalParameters


def _snn_simulation_worker(task_tuple: tuple) -> Dict[str, Any]:
    """
    Top-level standalone worker function for executing a single SNN simulation task in a process pool.
    
    Parameters
    ----------
    task_tuple : tuple
        (task_id, net_idx, network_params, stim_idx, stim_name, stim_params, sim_name, sim_params, output_dir)
        
    Returns
    -------
    dict
        Lightweight metadata summary of the task run status and disk output location.
    """
    (task_id, net_idx, network_params, stim_idx, stim_name, stim_params, sim_name, sim_params, output_dir) = task_tuple

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if net_idx:
        snn_output_dir = os.path.join(output_dir, str(net_idx))
    else:
        snn_output_dir = os.path.join(output_dir, sim_name.lower())
    os.makedirs(snn_output_dir, exist_ok=True)
    
    safe_stim_name = str(stim_name).replace(" ", "_") if stim_name else f"stim{stim_idx}"
    file_name = f"{sim_name.lower()}_results_{safe_stim_name}.npz"
    file_path = os.path.join(snn_output_dir, file_name)

    metadata = {
        "task_id": task_id,
        "net_idx": net_idx,
        "stim_idx": stim_idx,
        "stim_name": stim_name,
        "sim_name": sim_name,
        "type": "snn",
        "status": "FAILED",
        "output_path": None,
        "error": None
    }

    simulator = PyNNSNNSimulator()
    try:
        simulator.build_network(network_params=network_params, snn_sim_params=sim_params)
        results = simulator.run_stimulus(stim_params=stim_params)

        # 1. Resolve time average window [start_time, end_time]
        window = getattr(sim_params, "time_average_window", [0.0, None])
        if isinstance(window, (list, tuple)):
            t_start = window[0] if window[0] is not None else 0.0
            t_end = window[1] if (len(window) > 1 and window[1] is not None) else np.inf
        else:
            t_start, t_end = 0.0, np.inf

        save_dict = {
            "times": results.times(),
            "drive_rate": results.drive_rate_mean(),
            "stim_rate": results.stim_rate_mean(),
            }

        saved_metrics = sim_params.saved_metrics
        saved_variables = sim_params.saved_variables
        saved_extra_keys = sim_params.saved_extra_keys

        # 2. Main metric reduction loop over saved_metrics x saved_variables
        for metric in saved_metrics:
            getter = getattr(results, f"get_{metric}", None)
            if not callable(getter):
                raise AttributeError(f"SNNResults has no metric method 'get_{metric}'. Valid metrics: pop_mean, pop_std, time_mean, time_std, full_mean, all.")
            
            for var in saved_variables:
                if "spikes" in var:
                    continue
                arr = getter(var, start_time=t_start, end_time=t_end) if metric in ["time_mean", "time_std", "full_mean"] else getter(var)
                if arr is not None:
                    save_dict[f"{var}_{metric}"] = arr
                    if metric == "pop_mean" and f"{var}_mean" not in save_dict:
                        save_dict[f"{var}_mean"] = arr  # Backwards compatibility alias

        # 3. Save Spikes Arrays for Raster Plots (~250 KB)
        if "exc_spikes" in saved_variables or "spikes" in saved_variables:
            save_dict["exc_spikes"] = np.array(results.exc_spikes_all(), dtype=object)
        if "inh_spikes" in saved_variables or "spikes" in saved_variables:
            save_dict["inh_spikes"] = np.array(results.inh_spikes_all(), dtype=object)

        # 4. Explicit Extra Key Overrides (e.g. "exc_rate_all")
        for extra_key in saved_extra_keys:
            if hasattr(results, extra_key) and callable(getattr(results, extra_key)):
                val = getattr(results, extra_key)()
                if val is not None:
                    save_dict[extra_key] = val
            else:
                found_override = False
                for m in ["all", "pop_mean", "pop_std", "time_mean", "time_std", "full_mean"]:
                    if extra_key.endswith(f"_{m}"):
                        var_name = extra_key[:-len(f"_{m}")]
                        getter = getattr(results, f"get_{m}", None)
                        if callable(getter):
                            arr = getter(var_name, start_time=t_start, end_time=t_end) if m in ["time_mean", "time_std", "full_mean"] else getter(var_name)
                            if arr is not None:
                                save_dict[extra_key] = arr
                                found_override = True
                                break
                if not found_override and extra_key not in save_dict:
                    raise KeyError(f"Could not resolve extra_key '{extra_key}' on SNNResults.")

        np.savez_compressed(file_path, **save_dict)

        metadata["status"] = "SUCCESS"
        metadata["output_path"] = file_path

    except Exception as e:
        metadata["error"] = str(e)
    finally:
        try:
            simulator.end()
        except Exception:
            pass

    return metadata


def run_snn_batch_parallel(*args, **kwargs):
    """Placeholder function to maintain backwards compatibility during imports."""
    raise NotImplementedError("run_snn_batch_parallel is currently deprecated. Use run_unified_batch_parallel from codes.controller instead.")

# def run_snn_batch_parallel(
#     network_params_list: Union[BiologicalParameters, List[BiologicalParameters]],
#     stimuli: Union[Dict[str, BaseStimulusConfig], List[BaseStimulusConfig], BaseStimulusConfig],
#     snn_sim_params: SpikingNeuralNetworkSimulationConfig,
#     output_dir: str = "results/snn_batch",
#     cpus: int = None
# ) -> List[Dict[str, Any]]:
#     """
#     Executes a parallel batch grid of SNN network simulations across networks and stimuli lists.

#     Parameters
#     ----------
#     network_params_list : BiologicalParameters or List[BiologicalParameters]
#         One or multiple biological network configurations.
#     stimuli : Dict[str, BaseStimulusConfig], List[BaseStimulusConfig], or BaseStimulusConfig
#         Target stimulus configurations to simulate.
#     snn_sim_params : SpikingNeuralNetworkSimulationConfig
#         SNN simulation configuration settings.
#     output_dir : str
#         Directory where task metadata and .npz array files will be saved.
#     cpus : int, optional
#         Number of worker processes. Defaults to max(1, mp.cpu_count() - 1).

#     Returns
#     -------
#     List[Dict[str, Any]]
#         List of metadata records for all executed tasks.
#     """
#     if isinstance(network_params_list, BiologicalParameters):
#         net_list = [network_params_list]
#     else:
#         net_list = list(network_params_list)

#     if isinstance(stimuli, dict):
#         stim_items = list(stimuli.items())
#     elif isinstance(stimuli, list):
#         stim_items = [(f"stim_{i}", s) for i, s in enumerate(stimuli)]
#     else:
#         stim_items = [("stim_0", stimuli)]

#     if cpus is None:
#         cpus = max(1, mp.cpu_count() - 1)

#     tasks = []
#     task_id = 0
#     for net_idx, net_params in enumerate(net_list):
#         for stim_idx, (stim_name, stim_params) in enumerate(stim_items):
#             tasks.append((
#                 task_id, net_idx, stim_idx, stim_name,
#                 net_params, stim_params, snn_sim_params,
#                 output_dir
#             ))
#             task_id += 1

#     print(f"Launching SNN parallel batch: {len(tasks)} task(s) across {cpus} CPU worker(s)...")

#     try:
#         from tqdm import tqdm
#         has_tqdm = True
#     except ImportError:
#         has_tqdm = False

#     results_metadata = []
#     with mp.Pool(processes=cpus) as pool:
#         iterator = pool.imap_unordered(_snn_simulation_worker, tasks)
#         if has_tqdm:
#             iterator = tqdm(iterator, total=len(tasks), desc="SNN Simulations")
#         for res in iterator:
#             results_metadata.append(res)

#     print(f"Finished SNN parallel batch. Saved outputs to '{output_dir}/snn/'.")
#     return results_metadata
