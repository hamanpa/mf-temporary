import os
import sys
from pathlib import Path
import csv
import copy
import argparse
import pickle

# Setup repository paths dynamically
worker_dir = Path(__file__).resolve().parent
repo_path = worker_dir.parent
mean_field_path = repo_path / "MeanFieldTester"

if str(mean_field_path) not in sys.path:
    sys.path.append(str(mean_field_path))

from codes.controller.workflows import (run_basic_workflow,
                                         run_neuron_simulation_workflow,
                                         run_tf_fitting_workflow,
                                         run_mf_simulation_workflow,
                                         run_snn_simulation_workflow)

from codes.controller.config import load_workflow_config
from codes.stimuli.loader import load_stimuli_config
from codes.network_params.loader import load_network_parameters
from codes.network_params.models import StaticSynapseDefinition, StaticSynapseParams

from codes.controller.inspectors import inject_pydantic_param, ParameterInspector, ModelSummaryExtractor
import codes.plotting.hooks as plt_hooks

DELIMETER = ';'


def parse_val(val_str: str):
    """Attempt to parse string values from CSV to float, int, bool, or original str."""
    val_str = val_str.strip()
    if val_str.lower() == 'true':
        return True
    if val_str.lower() == 'false':
        return False
    try:
        val_float = float(val_str)
        if val_float.is_integer():
            return int(val_float)
        return val_float
    except ValueError:
        return val_str

def convert_tsodyks_to_static_if_zero(network_params, syn_key: str, tau_value):
    try:
        val_float = float(tau_value)
    except (ValueError, TypeError):
        return False

    if val_float == 0.0 and syn_key in network_params.synapses:
        current_syn = network_params.synapses[syn_key]
        if getattr(current_syn, "syn_type", None) == "tsodyks_synapse":
            w = current_syn.syn_params.weight
            u = current_syn.syn_params.U
            d = current_syn.syn_params.delay
            network_params.synapses[syn_key] = StaticSynapseDefinition(
                syn_type="static_synapse",
                syn_params=StaticSynapseParams(
                    weight=w * u,
                    delay=d,
                )
            )
            print(f"Converted synapse '{syn_key}' to static_synapse (weight={w * u:.4f}, delay={d:.2f}) because tau_rec = 0.")
            return True
    return False


def apply_parameter_update(network_params, sim_params, stimuli_config, param_path: str, value):
    """
    Updates configuration objects based on dot-separated parameter path.
    Supported prefixes:
    - 'network.': updates network_params
    - 'workflow.' or 'sim.': updates sim_params
    - 'stimulus.' or 'stimuli.': updates stimuli_config
    """
    if param_path.startswith("network."):
        path = param_path[len("network."):]
        network_params = inject_pydantic_param(network_params, path, value)
    elif param_path.startswith("workflow."):
        path = param_path[len("workflow."):]
        sim_params = inject_pydantic_param(sim_params, path, value)
    elif param_path.startswith("sim."):
        path = param_path[len("sim."):]
        sim_params = inject_pydantic_param(sim_params, path, value)
    elif param_path.startswith("stimulus.") or param_path.startswith("stimuli."):
        prefix = "stimulus." if param_path.startswith("stimulus.") else "stimuli."
        path = param_path[len(prefix):]
        # If path starts with stimulus key e.g. 'SpontActivity.drive_rate'
        parts = path.split('.', 1)
        if len(parts) == 2 and parts[0] in stimuli_config:
            stim_name, subpath = parts[0], parts[1]
            stimuli_config[stim_name] = inject_pydantic_param(stimuli_config[stim_name], subpath, value)
        else:
            # Apply to all stimuli in dict if specific stimulus name not matched
            for stim_name in stimuli_config:
                stimuli_config[stim_name] = inject_pydantic_param(stimuli_config[stim_name], path, value)
    else:
        # Default fallback to network_params
        network_params = inject_pydantic_param(network_params, param_path, value)

    return network_params, sim_params, stimuli_config


def run_worker_workflow(network_params, sim_params, stimuli_config, sim_id: str, results_dir: Path):
    """
    Modular execution entry point for running inspection / simulation for a single ID.
    Can be customized as needed.
    """
    worker_data_dir = results_dir / "data" 
    worker_imgs_dir = results_dir / "imgs" 

    print(f"[{sim_id}] Running worker workflow...")
    print(f"[{sim_id}] Data output directory: {worker_data_dir}")
    print(f"[{sim_id}] Imgs output directory: {worker_imgs_dir}")
    print(network_params.model_dump())

    # Example setup for ParameterInspector run if desired
    # Stimulus setup (default to first stimulus in config if SpontActivity not found)
    stim_key = "SpontActivity" if "SpontActivity" in stimuli_config else list(stimuli_config.keys())[0]
    base_stimulus = stimuli_config[stim_key]

    if sim_params.neuron_simulation.execution_mode in  ["try_load", "load"]:
        sim_params.neuron_simulation.exc_neuron_data_path = worker_data_dir / f"{sim_id}_exc_neuron_results.pkl"
        sim_params.neuron_simulation.inh_neuron_data_path = worker_data_dir / f"{sim_id}_inh_neuron_results.pkl"


    neuron_results = run_neuron_simulation_workflow(sim_params.neuron_simulation, network_params)
    for neuron_name, neuron_result in neuron_results.items():
        with open(worker_data_dir / f"{sim_id}_{neuron_name}_results.pkl", "wb") as f:
            pickle.dump(neuron_result, f)

    neuron_activity_plotter = plt_hooks.NeuronActivityHook(
        savefig_dir=worker_imgs_dir,
        fig_file_prefix="neuron_activity",
        neuron_names = ["exc_neuron", "inh_neuron"],
        fig_params={},
        common_params={},
    )


    neuron_activity_plotter(
        identifier=sim_id, 
        neuron_results=neuron_results,
        tf_funcs_results=None,
        snn_results=None,
        network_results_list=None,
    )

    tf_results_dict = {neuron_name: [] for neuron_name in neuron_results}

    for mf_model_name, mf_sim_params in sim_params.mf_models.items():
        tf_results = run_tf_fitting_workflow(mf_sim_params.transfer_function, network_params, neuron_results)
        for neuron_name in tf_results:
            tf_results_dict[neuron_name].append(tf_results[neuron_name])

    tf_plotter = plt_hooks.TransferFunctionPlottingHook(
        savefig_dir=worker_imgs_dir,
        fig_file_prefix="neuron_activity_tf_fit",
        neuron_names = ["exc_neuron", "inh_neuron"],
        fig_params={
            'figsize': (15, 10),  # width, height

        },
        common_params={
            "labels" : list(sim_params.mf_models.keys()),
            "ylim": (None, 30),
            'linestyles' : ["--", "-.", ":"],
        },
    )


    tf_plotter(
        identifier=sim_id, 
        neuron_results=neuron_results,
        tf_funcs_results=tf_results_dict,
        snn_results=None,
        network_results_list=None,
    )



    snn_results = run_snn_simulation_workflow(sim_params.snn_simulation, network_params, stimuli_config)

    mf_results_dict = {stim_name: [] for stim_name in stimuli_config}
    for mf_model_name, mf_sim_params in sim_params.mf_models.items():
        mf_results = run_mf_simulation_workflow(mf_sim_params, network_params, stimuli_config)
        for stimulus_name in stimuli_config:
            mf_results_dict[stimulus_name].append(mf_results[stimulus_name])



    network_overview_plotter = plt_hooks.NetworkOverviewPlottingHook(
        savefig_dir=worker_imgs_dir,
        fig_file_prefix="network_overview",
        fig_params={
            'axsize' : (20,5),
            # 'figsize': (20, 10),  # width, height
            'constrained_layout' : True,
            'gridspec_kw' : {'hspace': 0.065},
            'title': f"Network Overview",
            'bbox_inches': None,
        },
        common_params={
            'xmargin': 0.0,
            'ymargin': 0.0,
            'labels': ['SNN'] + list(sim_params.mf_models.keys()),
            'legend': {'loc': 'upper left'},
            'xlim' : (0, 4000.0),

        },
        subplot_params = {
            (3,0) : {'y_unit': 'pA', 'ylim': (0, 100)}
        }
    )

    for stimulus_name in stimuli_config:
        network_overview_plotter(
            identifier=stimulus_name+f"_{sim_id}", 
            neuron_results=neuron_results,
            tf_funcs_results=tf_results_dict,
            snn_results=snn_results[stimulus_name],
            network_results_list=[snn_results[stimulus_name]]+mf_results_dict[stimulus_name]
        )



    variables = [
        "exc_rate", "inh_rate", 
        "exc_voltage", "inh_voltage",
        "exc_adaptation", "inh_adaptation",
        "ee_conductance", "ei_conductance", "ie_conductance", "ii_conductance", 
        "exc_u", "exc_x", "inh_u", "inh_x", 
    ]
    metrics = ["pop_mean", "pop_std"]

    network_results = dict()


    for stim_name, snn_results in snn_results.items():
        network_results[stim_name] = {}
        for variable in variables:
            network_results[stim_name][variable] = {}
            for metric in metrics:
                suffix = metric.split("_")[-1]
                method_name = f"{variable}_{suffix}"
                network_results[stim_name][variable][metric] = getattr(snn_results, method_name)() 
        network_results[stim_name].update({
            "times": snn_results.times(),
            "drive_rate": snn_results.drive_rate_mean(),
            "stim_rate": snn_results.stim_rate_mean(),
            "exc_spikes" : snn_results.exc_spikes_all(),
            "inh_spikes" : snn_results.inh_spikes_all(),
        })


    with open(worker_data_dir / f"{sim_id}_snn_results.pkl", "wb") as f:
        pickle.dump(network_results, f)
    with open(worker_data_dir / f"{sim_id}_mf_results.pkl", "wb") as f:
        pickle.dump(mf_results_dict, f)




def main():
    parser = argparse.ArgumentParser(description="Worker script for multi-inspection runs.")
    parser.add_argument("--id", type=str, required=True, help="Simulation Hash ID")
    parser.add_argument("--project_dir", type=str, required=True, help="Path to project directory")
    parser.add_argument('--test', action='store_true', help="Enable test mode")
    args = parser.parse_args()

    project_path = Path(args.project_dir)
    params_dir = project_path / "params"

    if not params_dir.exists():
        raise FileNotFoundError(f"Params directory not found at {params_dir}")

    csv_path = project_path / "param_combinations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Param combinations file not found at {csv_path}")

    # Read CSV and find matching row
    row_found = None
    header = None
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=DELIMETER)
        header = next(reader)
        for row in reader:
            if row and row[0] == args.id:
                row_found = row
                break

    if not row_found:
        raise ValueError(f"Simulation ID {args.id} not found in {csv_path}")

    param_names = header[1:]
    param_values = [parse_val(v) for v in row_found[1:]]
    param_dict = dict(zip(param_names, param_values))

    print(f"[{args.id}] Loaded parameters: {param_dict}")

    # Load base parameters
    if args.test:
        network_params = load_network_parameters(params_dir / "test_network_params.yaml")
        sim_params = load_workflow_config(params_dir / "test_workflow_params.yaml")
        stimuli_config = load_stimuli_config(params_dir / "test_stimuli.yaml")
    else:
        network_params = load_network_parameters(params_dir / "network_params.yaml")
        sim_params = load_workflow_config(params_dir / "workflow_params.yaml")
        stimuli_config = load_stimuli_config(params_dir / "default_stimuli.yaml")

    # Apply updates
    for p_name, p_val in param_dict.items():
        network_params, sim_params, stimuli_config = apply_parameter_update(
            network_params, sim_params, stimuli_config, p_name, p_val
        )

    # Post-process: Convert any tsodyks_synapse with tau_rec == 0.0 to static_synapse after all updates
    for syn_key in list(network_params.synapses.keys()):
        syn_def = network_params.synapses[syn_key]
        if getattr(syn_def, "syn_type", None) == "tsodyks_synapse":
            tau_rec = getattr(syn_def.syn_params, "tau_rec", None)
            if tau_rec is not None and float(tau_rec) == 0.0:
                convert_tsodyks_to_static_if_zero(network_params, syn_key, 0.0)

    # Run workflow
    run_worker_workflow(network_params, sim_params, stimuli_config, args.id, project_path)


if __name__ == "__main__":
    main()
