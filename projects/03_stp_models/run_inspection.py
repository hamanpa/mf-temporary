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

from codes.controller.inspectors import ParameterInspector

from codes.plotting import fig_plots
import codes.plotting as ax_plt



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



    
    
    inspected_param = args.param
    inspected_values = np.array(json.loads(args.values))
    inspected_stimulus = stimuli_config[args.stim]

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

    fig, axes = plt.subplots(ncols=3, figsize=(16, 8))

    plot = ax_plt.FiringRateInspectionPlot({
        "linestyles": [""] + [ ':', '-.', '--'],
        "legend": True,
        "xlabel": "Drive Rate (Hz)"
    })
    plot.draw(axes[0], inspection_results["spont"])

    plot = ax_plt.VoltageInspectionPlot({
        "linestyles": [""] + [ ':', '-.', '--'],
        "legend": True,
        "xlabel": "Drive Rate (Hz)"
    })
    plot.draw(axes[1], inspection_results["spont"])

    plot = ax_plt.AdaptationInspectionPlot({
        "linestyles": [""] + [ ':', '-.', '--'],
        "legend": True,
        "xlabel": "Drive Rate (Hz)"
    })
    plot.draw(axes[2], inspection_results["spont"])

    fig.suptitle("Spontaneous Activity Inspection Results")
    fig.tight_layout()
    fig.savefig(results_path / f"SpontaneousInspection_{inspected_param}.png")

    dynamic_results = inspection_results["dynamic"]
    dynamic_measures = dynamic_results.measured_variables
    variables = set("_".join(var.split("_")[:2]) for var in dynamic_measures)
    measures = set("_".join(var.split("_")[2:]) for var in dynamic_measures)
    fig, axes = plt.subplots(ncols=4, nrows=len(variables), figsize=(24, 8)*len(variables))

    for i, variable in enumerate(variables):
        if variable + "_rmse" in dynamic_measures:
            plot = ax_plt.CustomInspectionPlot({
                "legend": True,
                "linestyles": [':', '-.', '--'],
                "title": r"$RMSE : \sqrt{1/T\int (SNN-MF)^2}$",
                "ylabel": variable,
            })
            plot.draw(axes[i, 0], inspection_results["dynamic"], "exc_rate_rmse")

        if variable + "_error_mean" in dynamic_measures:
            plot = ax_plt.CustomInspectionPlot({
                "legend": True,
                "linestyles": [':', '-.', '--'],
                "title": r"$Error : (SNN-MF)$",
            })
            plot.draw(axes[i, 1], inspection_results["dynamic"], "exc_rate_error_mean")

        if variable + "_error_std" in dynamic_measures:
            plot = ax_plt.CustomInspectionPlot({
                "legend": True,
                "linestyles": [':', '-.', '--'],
                "title": r"$Error : (SNN-MF)$",
            })
            plot.draw(axes[i, 2], inspection_results["dynamic"], "exc_rate_error_std")

        if variable + "_pearson" in dynamic_measures:
            plot = ax_plt.CustomInspectionPlot({
                "legend": True,
                "linestyles": [':', '-.', '--'],
                "title": r"$Pearson$",
            })
            plot.draw(axes[i, 3], inspection_results["dynamic"], "exc_rate_pearson")

    fig.suptitle("Spontaneous Activity Inspection Results")
    fig.tight_layout()
    fig.savefig(results_path / f"DynamicInspection_{inspected_param}.png")

if __name__ == "__main__":
    main()
