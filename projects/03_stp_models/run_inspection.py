import sys
import os
from pathlib import Path
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import json
import datetime

sys.path.append('/home/haman/mf-temporary/MeanFieldTester')
repo_path = Path('/home/haman/mf-temporary')

from codes.controller.config import load_workflow_config
from codes.stimuli.loader import load_stimuli_config
from codes.network_params.loader import load_network_parameters

from codes.neuron_simulation import run_neuron_simulation_workflow
from codes.transfer_function import run_tf_fitting_workflow
from codes.controller.inspectors import ParameterInspector

from codes.plotting import fig_plots



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, required=True)
    parser.add_argument("--values", type=str, required=True)
    parser.add_argument("--stim", type=str, required=True)
    
    args = parser.parse_args()


    project_path = repo_path / "projects" / "03_stp_models"
    os.chdir(project_path)

    dir_name = f"inspection_{args.param}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_path = project_path / "results" / dir_name
    results_path.mkdir(parents=True, exist_ok=True)

    network_params = load_network_parameters(project_path / "params" / "network_params.yaml")
    sim_params = load_workflow_config(project_path / "params" / "workflow_params.yaml")
    stimuli_config = load_stimuli_config(project_path / "params" / "default_stimuli.yaml")



    # 1. neuron simulation

    neuron_results_file = project_path / "results" / "neuron_results.pkl"

    with open(neuron_results_file, "rb") as f:
        neuron_results = pickle.load(f)

    # neuron_results = run_neuron_simulation_workflow(sim_params.neuron_simulation, network_params)
    # neuron_results_file = results_path / "neuron_results.pkl"
    # with open(neuron_results_file, "wb") as f:
    #     pickle.dump(neuron_results, f)

    fig_name = f"neuron_activity.png"
    fig_path = results_path / fig_name
    fig_plots.fig_neuron_activity(
        neuron_results, 
        common_params={}, 
        fig_params={
            'tight_layout': True,
            'savefig': True,
            'savefig_path': fig_path,
            'title': f"Neuron Activity"
        }
    )

    fig_name = f"neuron_activity_std.png"
    fig_path = results_path / fig_name
    fig_plots.fig_tf_fits_together(
        neuron_results, 
        {"exc_neuron": [],
        "inh_neuron": []
        }, 
        common_params={
            'labels' : [],
            'linestyles' : [],
            # 'xlim' : (0,7),
            'ylim' : (0, 60),
        }, 
        fig_params={
            'fontsize': 14,
            'figsize': (15, 10),  # width, height
            'tight_layout': True,
            'savefig': True,
            'savefig_path': fig_path,
            'title': f"Neuron Activity (STD)"
    })    

    # 2. Transfer function
    tf_results_dict = {}
    tf_results_legacy = {
        "exc_neuron": [],
        "inh_neuron": []
    }

    for mf_model_name, mf_sim_params in sim_params.mf_models.items():
        tf_results = run_tf_fitting_workflow(mf_sim_params.transfer_function, network_params, neuron_results)
        tf_results_dict[mf_model_name] = tf_results
        tf_results_legacy["exc_neuron"].append(tf_results["exc_neuron"])
        tf_results_legacy["inh_neuron"].append(tf_results["inh_neuron"])

    fig_name = f"neuron_activity_tf.png"
    fig_path = results_path / fig_name
    fig_plots.fig_tf_fits_together(
        neuron_results,
        tf_results_legacy,
        common_params={
            'labels' : list(sim_params.mf_models.keys()),
            'linestyles' : ["--", "-.", ":"],
            # 'xlim' : (0,7),
            'ylim' : (0, 30),
        }, 
        fig_params={
            'fontsize': 14,
            'figsize': (15, 10),  # width, height
            'tight_layout': True,
            'savefig': True,
            'savefig_path': fig_path,
            'title': f"Neuron Activity (TF)"
    })

    # 3. Inspection of SNN and MF models
    inspected_param = args.param
    inspected_values = np.array(json.loads(args.values))
    inspected_stimulus = stimuli_config[args.stim]
    values = np.array(json.loads(args.values))

    inspector = ParameterInspector(
        base_network_params=network_params,
        base_stimulus_params=inspected_stimulus,
        base_sim_params=sim_params,
    )


    inspection_results = inspector.run_inspection(
        inspected_param=inspected_param,
        inspected_values=inspected_values,
        measured_variables=[
            "exc_rate_time_mean",
            "exc_rate_time_std",
            "inh_rate_time_mean",
            "inh_rate_time_std",
            "exc_adaptation_time_mean",
            "exc_adaptation_time_std",
            "exc_rate_rmse",
            "exc_rate_error_mean",
            "exc_rate_error_std",
            "exc_rate_pearson",
            "inh_rate_rmse",
            "inh_rate_error_mean",
            "inh_rate_error_std",
            "inh_rate_pearson",
            "exc_adaptation_rmse",
            "exc_adaptation_error_mean",
            "exc_adaptation_error_std",
            "exc_adaptation_pearson",
        ],
        start_time=2000.0,
        end_time=4000.0,
        plot=True,
        project_path=results_path
    )
    
    spont_results = inspection_results["spont"]
    for measured_variable in spont_results.measured_variables:
        if measured_variable.endswith("_time_std"):
            continue
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.errorbar(
            spont_results.param_values, 
            getattr(spont_results, measured_variable)()[0], 
            yerr=getattr(spont_results, measured_variable.replace("_mean", "_std"))()[0], 
            fmt='o', 
            label = "SNN"
        )
        ax.plot(spont_results.param_values, getattr(spont_results, measured_variable)()[1:].T, label=spont_results.network_names[1:])

        ax.set_title(f"inspected parameter: {spont_results.inspected_param}")
        ax.set_xlabel(f"{spont_results.inspected_param} (Hz)")
        ax.set_ylabel(measured_variable)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_path / f"StaticInspection_{measured_variable}_vs_{inspected_param}.png")

    dynamic_results = inspection_results["dynamic"]
    for measured_variable in dynamic_results.measured_variables:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(dynamic_results.param_values, getattr(dynamic_results, measured_variable)().T, label=dynamic_results.network_names)

        ax.set_title(f"inspected parameter: {dynamic_results.inspected_param}")
        ax.set_xlabel(f"{dynamic_results.inspected_param} (Hz)")
        ax.set_ylabel(measured_variable)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_path / f"DynamicInspection_{measured_variable}_vs_{inspected_param}.png")

if __name__ == "__main__":
    main()
