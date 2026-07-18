"""
This is a testing script to check the workflow works

"""


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

import codes.plotting.hooks as plt_hooks
from codes.stimuli.config import NoStimulusConfig


TEST_STIMULUS = NoStimulusConfig({
    "pattern" : "NoStimulus",
    "stim_params" : {},
    "drive_rate" : 1.5,
    "initial_increase_duration" : 400,
    "simulation_duration" : 1500,
    "stim_target_ratio" : 1.0,
    "target_nodes" : 0,
    "direct_stimulation" : False
})

TEST_CONFIG = {"test_stimulus" : TEST_STIMULUS}

INSPECTED_PARAM = "stimulus.drive_rate"
INSPECTED_VALUES = np.array([1.0, 1.5])



def main():

    project_path = repo_path / "projects" / "03_stp_models"
    os.chdir(project_path)

    dir_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_testing"
    results_path = project_path / "results" / dir_name
    results_path.mkdir(parents=True, exist_ok=True)

    network_params = load_network_parameters(project_path / "params" / "network_params.yaml")
    sim_params = load_workflow_config(project_path / "params" / "workflow_params.yaml")


    inspector = ParameterInspector(
        base_network_params=network_params,
        base_stimulus_params=TEST_STIMULUS,
        base_sim_params=sim_params,
        project_path=results_path,
    )

    single_step_hooks = [
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
                'start_time': 1000.0,
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
            start_time = 1000.0, 
            end_time = TEST_STIMULUS.simulation_duration
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
            start_time = 1000.0, 
            end_time = TEST_STIMULUS.simulation_duration
        ),
    ]

    inspection_hooks = [
        plt_hooks.SteadyStateInspectionPlottingHook(
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
        plt_hooks.ComparisonInspectionPlottingHook(
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
        inspected_param=INSPECTED_PARAM,
        inspected_values=INSPECTED_VALUES,
        single_step_hooks=single_step_hooks,
        inspection_hooks=inspection_hooks,
        extractors=extractors,
    )

    with open(results_path / f"testing_inspection_results.pkl", "wb") as f:
        pickle.dump(inspection_results, f)

if __name__ == "__main__":
    main()
