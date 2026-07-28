import os
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Any, Union

from .tvb_simulator.simulator import TVBMFSimulator
from .config import MeanFieldSimulationConfig
from ..stimuli.config import BaseStimulusConfig
from ..network_params.models import BiologicalParameters


def _mf_simulation_worker(task_tuple: tuple) -> Dict[str, Any]:
    """
    Top-level standalone worker function for executing a single MF simulation task in a process pool.
    
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
        mf_output_dir = os.path.join(output_dir, str(net_idx))
    else:
        mf_output_dir = os.path.join(output_dir, sim_name.lower())
    os.makedirs(mf_output_dir, exist_ok=True)
    
    safe_stim_name = str(stim_name).replace(" ", "_") if stim_name else f"stim{stim_idx}"
    file_name = f"{sim_name.lower()}_results_{safe_stim_name}.npz"
    file_path = os.path.join(mf_output_dir, file_name)

    metadata = {
        "task_id": task_id,
        "net_idx": net_idx,
        "stim_idx": stim_idx,
        "stim_name": stim_name,
        "sim_name": sim_name,
        "type": "mf",
        "status": "FAILED",
        "output_path": None,
        "error": None
    }

    simulator = TVBMFSimulator()
    try:
        simulator.build_network(network_params=network_params, mf_sim_params=sim_params)
        results = simulator.run_stimulus(stim_params=stim_params)

        save_dict = {
            "times": results.times(),
            "drive_rate": results.drive_rate_mean(),
            "stim_rate": results.stim_rate_mean(),
        }
        
        # Save optional fields if present
        measurement_fields = {
            "exc_rate_mean": "exc_rate_pop_mean",
            "exc_rate_std": "exc_rate_pop_std",
            "inh_rate_mean": "inh_rate_pop_mean",
            "inh_rate_std": "inh_rate_pop_std",
            "exc_adaptation_mean": "exc_adaptation_pop_mean",
            "inh_adaptation_mean": "inh_adaptation_pop_mean",
            "rate_cov": "rate_cov",
            "exc_x_mean": "exc_x_pop_mean",
            "exc_y_mean": "exc_y_pop_mean",
            "exc_u_mean": "exc_u_pop_mean",
            "inh_x_mean": "inh_x_pop_mean",
            "inh_y_mean": "inh_y_pop_mean",
            "inh_u_mean": "inh_u_pop_mean",
            "exc_voltage_mean": "exc_voltage_pop_mean",
            "inh_voltage_mean": "inh_voltage_pop_mean",
        }
        for field_name, save_name in measurement_fields.items():
            val = getattr(results, field_name, None)
            if callable(val):
                arr = val()
                if arr is not None:
                    save_dict[save_name] = arr
            elif val is None:
                save_dict[save_name] = np.full((len(save_dict["times"]),), np.nan)
            else:
                raise ValueError(f"Unexpected type for {field_name}: {type(val)}")

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

def run_mf_batch_parallel(*args, **kwargs):
    """Placeholder function to maintain backwards compatibility during imports."""
    raise NotImplementedError("run_mf_batch_parallel is currently deprecated. Use run_unified_batch_parallel from codes.controller instead.")

# def run_mf_batch_parallel(
#     network_params_list: Union[BiologicalParameters, List[BiologicalParameters]],
#     stimuli: Union[Dict[str, BaseStimulusConfig], List[BaseStimulusConfig], BaseStimulusConfig],
#     mf_sim_params: MeanFieldSimulationConfig,
#     output_dir: str = "results/mf_batch",
#     cpus: int = None
# ) -> List[Dict[str, Any]]:
#     """
#     Executes a parallel batch grid of Mean Field simulations across networks and stimuli lists.

#     Parameters
#     ----------
#     network_params_list : BiologicalParameters or List[BiologicalParameters]
#         One or multiple biological network configurations.
#     stimuli : Dict[str, BaseStimulusConfig], List[BaseStimulusConfig], or BaseStimulusConfig
#         Target stimulus configurations to simulate.
#     mf_sim_params : MeanFieldSimulationConfig
#         Mean Field simulation configuration settings.
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
#                 net_params, stim_params, mf_sim_params,
#                 output_dir
#             ))
#             task_id += 1

#     print(f"Launching MF parallel batch: {len(tasks)} task(s) across {cpus} CPU worker(s)...")

#     try:
#         from tqdm import tqdm
#         has_tqdm = True
#     except ImportError:
#         has_tqdm = False

#     results_metadata = []
#     with mp.Pool(processes=cpus) as pool:
#         iterator = pool.imap_unordered(_mf_simulation_worker, tasks)
#         if has_tqdm:
#             iterator = tqdm(iterator, total=len(tasks), desc="MF Simulations")
#         for res in iterator:
#             results_metadata.append(res)

#     print(f"Finished MF parallel batch. Saved outputs to '{output_dir}/mf/'.")
#     return results_metadata
