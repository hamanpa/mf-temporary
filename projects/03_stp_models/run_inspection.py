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

from codes.controller.inspectors import ParameterInspector, ModelComparisonExtractor, ModelSummaryExtractor

import codes.plotting as ax_plt

import codes.plotting.hooks as plt_hooks






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, required=True)
    parser.add_argument("--values", type=str, required=True)
    parser.add_argument("--stim", type=str, required=True)
    
    args = parser.parse_args()


    project_path = repo_path / "projects" / "03_stp_models"
    os.chdir(project_path)

    dir_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_inspection_{args.param}"
    results_path = project_path / "results" / dir_name
    results_path.mkdir(parents=True, exist_ok=True)

    network_params = load_network_parameters(project_path / "params" / "network_params.yaml")
    sim_params = load_workflow_config(project_path / "params" / "workflow_params.yaml")
    stimuli_config = load_stimuli_config(project_path / "params" / "default_stimuli.yaml")

    inspected_param = args.param
    inspected_values = np.array(json.loads(args.values))
    inspected_stimulus = stimuli_config[args.stim]

    inspector = ParameterInspector(
        base_network_params=network_params,
        base_stimulus_params=inspected_stimulus,
        base_sim_params=sim_params,
        project_path=results_path,
    )

    single_step_hooks = [
        plt_hooks.NeuronActivityHook(
            savefig_dir=results_path,
            fig_file_prefix="neuron_activity",
            neuron_names = ["exc_neuron", "inh_neuron"],
            fig_params={},
            common_params={},
        ),
        plt_hooks.TransferFunctionPlottingHook(
            savefig_dir=results_path,
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
        ),
        plt_hooks.NetworkOverviewPlottingHook(
            savefig_dir=results_path,
            fig_file_prefix="network_overview",
            fig_params={
                'axsize' : (20,4),
                # 'figsize': (20, 10),  # width, height
                'gridspec_kw' : {'hspace': 0.0},
                'title': f"Network Overview"
            },
            common_params={
                'xmargin': 0.0,
                'ymargin': 0.0,
                'labels': ['SNN'] + list(sim_params.mf_models.keys()),
                'legend': {'loc': 'upper left'},
                'xlim' : (0, 4000.0),

            },
        ),
        plt_hooks.NetworkHistogramPlottingHook(
            savefig_dir=results_path,
            fig_file_prefix="network_histogram",
            fig_params={
                'figsize': (20, 5),  # width, height
                'title': f"Network Histogram"
            },
            common_params={
                'start_time': 2000.0,
                'bins' : 10,
                'labels': ['SNN'],
                'legend': False,

            },
        ),
    ]

    extractors = [
        ModelSummaryExtractor(
            measured_variables=[
                "exc_rate",
                "inh_rate",
                "exc_adaptation",
            ],
            metrics=[
                "time_mean",
                "time_std",
            ],
            start_time = 2000.0, 
            end_time = 4000.0
        ),
        ModelComparisonExtractor(
            measured_variables=[
                "exc_rate",
                "inh_rate",
                "exc_adaptation",
            ],
            metrics=[
                "mse",
                "rmse",
                "error_mean",
                "error_std",
                "pearson",
                "spearman",
                "lag",
                "max_corr",
                "psd_similarity",
            ],
            start_time = 2000.0, 
            end_time = 4000.0
        ),
    ]

    inspection_hooks = [
        plt_hooks.ModelSummaryInspectionPlottingHook(
            savefig_dir=results_path,
            fig_file_prefix="steady_state_inspection",
            fig_params={
                'figsize': (20, 10),  # width, height
            },
            common_params = {
                "linestyles": [""] + [ ':', '-.', '--'],
                "legend": True,
                "xlabel": "Drive Rate (Hz)"
            },
        ),
        plt_hooks.ModelComparisonInspectionPlottingHook(
            savefig_dir=results_path,
            fig_file_prefix="dynamic_inspection",
            fig_params={
                'axsize': (4, 4),  # width, height
            },
            common_params = {
                "linestyles": [ ':', '-.', '--'],
                "legend": True,
                "xlabel": "Drive Rate (Hz)"
            }
        ),
    ]


    inspection_results = inspector.run_inspection(
        inspected_param=inspected_param,
        inspected_values=inspected_values,
        single_step_hooks=single_step_hooks,
        inspection_hooks=inspection_hooks,
        extractors=extractors,
    )

    with open(results_path / f"inspection_results_{inspected_param}.pkl", "wb") as f:
        pickle.dump(inspection_results, f)

if __name__ == "__main__":
    main()
