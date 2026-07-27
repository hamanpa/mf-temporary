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

    snn_output_dir = os.path.join(output_dir, sim_name.lower())
    os.makedirs(snn_output_dir, exist_ok=True)
    
    safe_stim_name = str(stim_name).replace(" ", "_") if stim_name else f"stim{stim_idx}"
    file_name = f"{net_idx}_{sim_name}_results_{safe_stim_name}_task_{task_id}.npz"
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

        np.savez_compressed(
            file_path,
            times=results.times(),
            exc_spikes=np.array(results.exc_spikes_all(), dtype=object),
            inh_spikes=np.array(results.inh_spikes_all(), dtype=object),
            exc_rate=results.exc_rate_all(),
            inh_rate=results.inh_rate_all(),
            exc_voltage=results.exc_voltage_all(),
            inh_voltage=results.inh_voltage_all(),
            exc_adaptation=results.exc_adaptation_all(),
            inh_adaptation=results.inh_adaptation_all(),
            ee_conductance=results.ee_conductance_all(),
            ei_conductance=results.ei_conductance_all(),
            ie_conductance=results.ie_conductance_all(),
            ii_conductance=results.ii_conductance_all(),
            exc_x=results.exc_x_all(),
            exc_u=results.exc_u_all(),
            inh_x=results.inh_x_all(),
            inh_u=results.inh_u_all(),
            drive_rate_mean=results.drive_rate_mean(),
            stim_rate_mean=results.stim_rate_mean()
        )

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


# TODO: following has to be updated to work properly, commented out till then
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
