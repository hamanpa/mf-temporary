"""
This script should run the whole workflow of the mean-field model.

0. Load the parameters of the model
    - It looks in the params folder for the following files
        - workflow_params.json
        - custom_network.yaml
0.1. Creates a directory for the results in the results folder
1. Simulate neuron
    - specify number of runs, seed (to have statistics)
    - make a plot of the neuron (activity, adaptation)
    - (optional) make a plot of fluctuations (mu_V, sigma_V, tau_V)
    - possible simulations: 
        - fixed nu_e, nu_i
        - fixed nu_out, nu_i
        - (?) fixed nu_e, nu_i, adaptation
2. Make TF fit
    - make plot of the fit (activity, adaptation, fit)
    - (optional) make a plot of fluctuations (mu_V, sigma_V, tau_V)
3. Simulate the spiking network
4. Simulate MF
5. Compare the results

Options:

1. TF fit of csng
2. add synaptic plasticity
3. add connectivity rules
4. add delays (distances)
"""
import numpy as np
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
from copy import deepcopy

import os
# os.chdir('/home/pavel/academia/mf-temporary/mean-field-CSNG/codes')

import codes.neuron_simulation as ns
import codes.meanfield_simulation as mfs
import codes.network_simulation as nets
import codes.transfer_function as tf
import codes.cell_library as cells

from codes.data_structures.network import SNNFullResults, MFResults
from codes.data_structures.single_neuron import SingleNeuronResults
from codes.data_structures.inspection import SpontaneousActivityInspectionResults
from codes.utils.file_helpers import prepare_result_dir, load_json, load_with_eval
from codes.utils import compare_mf_snn_results
from codes.utils.dict_helpers import deep_union, flatten_dict, unflatten_dict

from codes import plotting
from codes.plotting import fig_plots


WRK_PARAM_FILE = "workflow_params.json"                     # Name of the file with workflow parameters
NEURON_NAMES = ["exc_neuron", "inh_neuron"]                 # This should be read from network parameters!
CODES_PATH = Path(__file__).resolve().parent.parent
PROJ_PATH = CODES_PATH.parent
PARAMS_PATH = PROJ_PATH / 'params'
DATA_PATH = PROJ_PATH / 'data'
TIME_STAMP = time.strftime("%Y%m%d-%H%M%S")

SETUPS = {
    "divolo-static" : {
        "sim_name" : "DiVolo-Static",
        "net_param_file" : "DiVolo_network-static.yaml",
        "data_file_name" : "divolo-static",
    },
    "divolo-stp" : {
        "sim_name" : "DiVolo-STP",
        "net_param_file" : "DiVolo_network-stp.yaml",
        "data_file_name" : "divolo-stp",
    },
    "csng-static" : {
        "sim_name" : "CSNG-Static",
        "net_param_file" : "CSNG_network-static.yaml",
        "data_file_name" : "csng-static",
    },
    "csng-stp" : {
        "sim_name" : "CSNG-STP",
        "net_param_file" : "CSNG_network-stp.yaml",
        "data_file_name" : "csng-stp",
    },
    "custom" : {
        "sim_name" : "custom_network.yaml",
        "net_param_file" : "",
        "data_file_name" : "",
        "time_stamp" : "",
    },
}
# Specifying a setup:
# net_param_file        - Name of the file with network parameters
# sim_name              - Name of the simulation --> Folder name
# data_file_name        - Starting name of the file with neuron simulation results
# time_stamp (optional) - Time stamp for the results folder

############################################################################

class WorkflowRunner:
    """This class runs the whole workflow of the mean-field model testing.
    
    It provides API for running the whole workflow, including:
    - loading parameters
    - simulating single neurons
    - fitting transfer functions
    - simulating the spiking network
    - simulating mean-field models
    - comparing results

    The workflow is designed as follows:

        1. Load parameters for the Spiking Neural Network (SNN) 
    
        2. Create a results folder / use existing one
    
        3. Optionally add mean-field models to the workflow 
            (possible to add multiple models with different parameters)
        
        4. Simulate single neurons based on the parameters
        
        5. Fit transfer functions based on the single neuron simulations and parameters
            (Each SNN, MF model has its own transfer function fit)

        6. Simulate the SNN network based on the parameters and transfer functions

        7. Simulate mean-field models based on the provided parameters in step 3.

    """

    # NOTE: naming convention for saving files
    param_file_pattern = "_network_pars.json"
    snn_param_file_saved = "snn" + param_file_pattern
    workflow_param_file_saved = "workflow_pars.json"
    stimuli_param_file_saved = "stimuli_pars.json"

    def __init__(self, 
                 sim_name:str, 
                 snn_params:str|Path|dict, 
                 workflow_params:str|Path|dict,
                 stimuli_params:str|Path|dict,
                 neuron_data_file_mark:str=None, 
                 results_dir:str|Path=None, 
                 ):
        """Initialize the workflow runner with the given setup of simulation and Spiking Neural Network.

        Parameters
        ----------
        sim_name : str
            Name of the simulation, used for creating the results directory.
            Created directory will be named as '{timestamp}_{sim_name}'.
        snn_params : str or Path or dict
            Spiking Neural Network parameters, either as a file name or a dictionary.
            If a string (file name) is provided, the file should be in Python-evaluatable format.
            If a Path is provided, it should point to a Python-evaluatable file.
            If a dictionary is provided, it will be used directly.
        workflow_params : str or Path or dict
            Workflow parameters, either as a file name or path to parameter or a dictionary.
            If a string (file name) is provided, the file should be in JSON format and located at PARAMS_PATH
            If a Path is provided, it should point to a JSON file.
            If a dictionary is provided, it should be a single dictionary 
                with subdictionaries for each part of the workflow:
                    - "single_neuron_simulations"
                    - "transfer_functions"
                    - "network_simulations"
                    - "mf_model"
        stimuli_params : str or Path or dict
            Stimuli parameters, either as a file name or a dictionary.
            If a string (file name) is provided, the file should be in Python-evaluatable format.
            If a Path is provided, it should point to a Python-evaluatable file.
            If a dictionary is provided, the key:value pairs should follow:
                stimulus_name : stimulus_specs
        neuron_data_file_mark : str, optional
            Prefix for the neuron data file name.
            If provided, it will be used as a prefix identifier for the neuron simulation results.
            If not provided, or no matching data file is found, single neuron simulation IS ENFORCED.
        results_dir : str or Path, optional
            Directory name where the simulation results will be saved.
            If not provided, a new directory will be created in the results folder.
        """

        self.sim_name = sim_name

        workflow_params = resolve_workflow_parameters(workflow_params)
        self.neuron_sim_pars = workflow_params["single_neuron_simulations"]
        self.default_tf_sim_pars = workflow_params["transfer_functions"]
        self.network_sim_pars = workflow_params["network_simulations"]
        self.default_mf_model = workflow_params["mf_model"]

        # Resolve stimuli parameters
        self.stimuli = resolve_stimuli_parameters(stimuli_params)

        # Resolve SNN parameters
        self.snn_network_pars = resolve_network_parameters(snn_params)
        cells.update_params(self.snn_network_pars)
        self.snn_neurons = {neuron: self.snn_network_pars[neuron] for neuron in NEURON_NAMES}

        self.data_file_mark = neuron_data_file_mark

        # Initialize the containers for mean-field models
        self.mf_names = []              # List of mean-field model names (str)
        self.mf_neurons = []            # List of mean-field model neurons (dict: {neuron_name: neuron_params})
        self.mf_net_pars = []           # List of mean-field model network parameters (dict: network_params)
        self.mf_tf_sim_pars = []        # List of mean-field model transfer function simulation parameters
        self.mf_models = []             # List of mean-field models (dict: mf_model)

        # Prepare the results directory
        if results_dir is None:
            self.results_path = prepare_result_dir(dir_name=sim_name, parent_path=PROJ_PATH / 'results')
        elif isinstance(results_dir, Path):
            self.results_path = results_dir.resolve()
            self.results_path.mkdir(exist_ok=True, parents=True)
        elif isinstance(results_dir, str):
            assert "/" not in results_dir, "If the whole path is provided, it should be a Path object."
            self.results_path = prepare_result_dir(dir_name=results_dir, parent_path=PROJ_PATH / 'results', time_stamp="")
        else:
            raise ValueError("results_path should be a Path or a string")

        # Names of the files where parameters will be saved
        self.snn_param_file_saved = self.results_path / self.snn_param_file_saved 
        self.workflow_param_file_saved = self.results_path / self.workflow_param_file_saved
        self.stimuli_param_file_saved = self.results_path / self.stimuli_param_file_saved
        self.save_parameters()

    def load_mf_model(self, mf_name):
        """Loads the mean-field model parameters from the results folder."""

        param_file = self.results_path / (mf_name + self.param_file_pattern)
        if not param_file.exists():
            raise ValueError(f"Mean-field model {mf_name} does not exist in the results folder")
        with param_file.open('r') as f:
            mf_network_pars = json.load(f)

        self.mf_names.append(mf_name)
        self.mf_neurons.append({neuron: mf_network_pars[neuron] for neuron in NEURON_NAMES})
        self.mf_net_pars.append(mf_network_pars)
        self.mf_tf_sim_pars.append(mf_network_pars["transfer_functions"])
        self.mf_models.append(mf_network_pars["mf_model"])

    def save_mf_model(self, mf_name):
        """Saves the mean-field model parameters to the results folder."""

        param_file = self.results_path / (mf_name + self.param_file_pattern)
        i = self.mf_names.index(mf_name)
        with param_file.open('w') as f:
            json.dump((self.mf_net_pars[i] | self.mf_models[i] | self.mf_tf_sim_pars[i]), f, indent=4)

    def save_parameters(self):
        """Saves the workflow, stimuli, SNN, and mean-field model parameters to the results folder.

        This function creates the necessary files in the results directory to store the parameters.
        The files are:
        - workflow_params.json: Contains the workflow parameters.
        - snn_network_pars.json: Contains the SNN network parameters.
        - stimuli_pars.json: Contains the stimuli parameters.
        - {mf_name}_network_pars.json: Contains the mean-field model parameters for each model.
        """
        workflow_pars = {
            "single_neuron_simulations": self.neuron_sim_pars,
            "transfer_functions": self.default_tf_sim_pars,
            "network_simulations": self.network_sim_pars,
            "mf_model": self.default_mf_model,
        }
        self.workflow_param_file_saved.write_text(json.dumps(workflow_pars, indent=4))
        self.snn_param_file_saved.write_text(json.dumps(self.snn_network_pars, indent=4))
        self.stimuli_param_file_saved.write_text(json.dumps(self.stimuli, indent=4))

        for mf_name in self.mf_names:
            self.save_mf_model(mf_name)

    def add_mf_model(self, mf_name:str, network_params:str, mf_model_pars:dict=None, tf_sim_pars:dict=None):
        """Adds a mean-field model to the workflow.

        This function adds a mean-field model to the workflow by loading the necessary parameters
        and updating the internal state of the controller.

        Parameters
        ----------
        mf_name : str
            The name of the mean-field model.
        network_params : str
            The name of the file with the network parameters used for the mean-field model.
            network_params may and may not be the same as snn_param_file.
        mf_model_pars : dict, optional
            Dictionary containing workflow parameters for the mean-field model.
            If provided, it will UPDATE the default mean-field model parameters.
            If not provided, it will use the default mean-field model parameters.
        tf_sim_pars : dict, optional
            Dictionary containing the workflow parameters for the transfer function simulation.
            If provided, it will UPDATE the default transfer function simulation parameters.
            If not provided, it will use the default transfer function simulation parameters.
        """

        if mf_model_pars is None:
            mf_model_pars = {}
        if tf_sim_pars is None:
            tf_sim_pars = {}

        mf_network_pars = resolve_network_parameters(network_params)
        cells.update_params(mf_network_pars)

        mf_model_pars = deep_union(self.default_mf_model, mf_model_pars)

        tf_sim_pars = deep_union(self.default_tf_sim_pars, tf_sim_pars)

        self.mf_names.append(mf_name)
        self.mf_net_pars.append(mf_network_pars)
        self.mf_neurons.append({neuron: mf_network_pars[neuron] for neuron in NEURON_NAMES})
        self.mf_models.append(mf_model_pars)
        self.mf_tf_sim_pars.append(tf_sim_pars)

        self.save_mf_model(mf_name)

    def add_default_mf_model(self, mf_name):
        """Adds a default mean-field model to the workflow."""
        self.add_mf_model(mf_name, self.snn_network_pars)

    def simulate_neurons(self, method="pynn", plot=False, save=False):
        """Runs single neurons simulations based on snn network setup.
        
        This is high-level API that calls function from neuron_simulation module.
        
        Parameter `method` can be used to specify the method for simulating neurons.

        Each method has to return a dictionary with keys as neuron names
        and values as instances of SingleNeuronResults.

        Parameters
        ----------
        method : str, optional
            The method to use for simulating neurons. Default is "pynn".
            Other options can be added in the future.
        save : bool, optional
            If True, it will save the results to the results directory.
            Default is False.
        plot : bool, optional
            If True, it will generate plots of the neuron activity and adaptation.
            Default is False.
        """
        match method:
            case "pynn":
                neuron_results = ns.run_single_neuron_workflow(
                                        self.neuron_sim_pars, 
                                        self.snn_neurons, 
                                        self.data_file_mark, 
                                        self.results_path, 
                                        DATA_PATH
                                        )
            case _:
                raise NotImplementedError(f"Method {method} is not implemented for simulating neurons. Feel free to add it!")

        # Test the results have the expected structure
        assert isinstance(neuron_results, dict), "The result of neuron simulation should be a dictionary."
        for neuron_name, neuron_result in neuron_results.items():
            assert neuron_name in self.snn_neurons, f"Neuron {neuron_name} is not in the SNN neurons."
            assert isinstance(neuron_result, SingleNeuronResults), f"The result for {neuron_name} should be an instance of SingleNeuronResults."
        self.neuron_results = neuron_results

        if save:
            if self.neuron_sim_pars["fix_nu_out"]:
                suffix = "grid-irregular"
                e_range = self.neuron_sim_pars["nu_out_range"][-1]
            else:
                suffix = "grid-regular"
                e_range = self.neuron_sim_pars["nu_e_range"][-1]
            i_range = self.neuron_sim_pars["nu_i_range"][-1]

            filename = f"{self.data_file_mark}_{{}}_{e_range}x{i_range}_{suffix}.pickle"
            for neuron_name, neuron_result in neuron_results.items():
                neuron_result.save(self.results_path / filename.format(neuron_name))

        if plot:
            fig_plots.fig_neuron_activity(
                self.neuron_results, 
                common_params={}, 
                fig_params={
                    'tight_layout': True,
                    'savefig': True,
                    'savefig_path': self.results_path / f"neuron_activity.png",
                }
            )

    def fit_transfer_functions(self, plot=False, save=False, method=""):
        """Fits transfer functions based on results of neuron simulations.

        This function runs the transfer function fitting workflow for each mean-field model
        and saves the results to the results directory.

        Parameters
        ----------
        plot : bool, optional
            If True, it will generate plots of the transfer function fitting results.
            Default is False.
        save : bool, optional
            If True, it will save the transfer function fitting results to the results directory.
            Default is False.
        method : str, optional
            The method to use for fitting transfer functions. Default is "" (empty string).
            Other options can be added in the future.

        """
        self.tf_funcs = {neuron_name : [] for neuron_name in NEURON_NAMES}

        if method != "":
            raise NotImplementedError(f"Method {method} is not implemented for fitting transfer functions. Feel free to add it!")

        for i, mf_network_pars in enumerate(self.mf_net_pars):
            tf_sim_pars = self.mf_tf_sim_pars[i]
            mf_tf_funcs = tf.run_fitting_workflow(
                                self.mf_tf_sim_pars[i], 
                                mf_network_pars, 
                                self.neuron_results, 
                                self.results_path / (self.mf_names[i] + self.param_file_pattern))
            for neuron_name, tf_func in mf_tf_funcs.items():
                self.tf_funcs[neuron_name].append(tf_func)
                mf_network_pars["transfer_function"][neuron_name] = list(tf_func.v_eff.coefs)

        if save:
            for mf_name in self.mf_names:
                self.save_mf_model(mf_name)

        if plot:
            fig_plots.fig_tf_fits_together(
                self.neuron_results,
                self.tf_funcs,
                common_params={
                    'labels' : self.mf_names,
                    'linestyles' : ['--', '-.', ':', '-'],
                },
                fig_params={
                    'savefig': True,
                    'tight_layout': True,
                    'savefig_path': self.results_path / f"tf_fits.png",
                }
            )

    def simulate_snn_network(self, stim_name, stimulus, try_load=True):
        """Simulates or loads the spiking network results for a given stimulus.

        Args:
            stim_name (str): Name of the stimulus.
            stimulus (dict): Stimulus parameters.
            try_load (bool): Whether to try loading existing results.

        Returns:
            dict: Results of the spiking network simulation.
        """
        file_name = f"SNN-{self.data_file_mark}-{stim_name}.pickle"
        file_path = self.results_path / file_name

        if try_load and file_path.exists():
            print(f"Loading the spiking network results from {file_name}")
            with file_path.open("rb") as f:
                net_results = pickle.load(f)
        else:
            print(f"Simulating the spiking network for stimulus: {stim_name}")
            net_results = nets.run_network(
                self.network_sim_pars,
                self.snn_neurons,
                self.snn_network_pars["network"],
                stimulus,
                self.results_path,
                file_name=file_name,
                save=True
            )
        return net_results

    def simulate_mfs(self, stim_name, stimulus, try_load=True):
        """Simulates or loads the mean-field results for a given stimulus.

        Args:
            stim_name (str): Name of the stimulus.
            stimulus (dict): Stimulus parameters.
            try_load (bool): Whether to try loading existing results.

        Returns:
            list[dict]: List of results for each mean-field model.
        """
        mf_results = []
        for i, mf_name in enumerate(self.mf_names):
            file_name = f"{mf_name}-{self.data_file_mark}-{stim_name}.pickle"
            file_path = self.results_path / file_name

            if try_load and file_path.exists():
                print(f"Loading the mean-field results from {file_name}")
                with file_path.open("rb") as f:
                    mf_results.append(pickle.load(f))
            else:
                print(f"Simulating the mean-field for model: {mf_name}, stimulus: {stim_name}")
                mf_results.append(
                    mfs.run_single_node_mf(
                        self.network_sim_pars | self.mf_models[i], 
                        self.mf_net_pars[i], 
                        stimulus, 
                        self.results_path, 
                        file_name=file_name
                    )
                )
        return mf_results

    def simulate_single_stimulus(self, stimulus_name, stimulus, try_load=True, plot=True):
        """Simulates or loads results for a single stimulus for both SNN and MF.

        Args:
            stimulus_name (str): Name of the stimulus.
            stimulus (dict): Stimulus parameters.
            try_load (bool): Whether to try loading existing results.
            plot (bool): Whether to generate plots for the results.

        Returns:
            tuple: (SNN results, list of MF results)
        """
        # Simulate or load SNN results
        snn_results = self.simulate_snn_network(stimulus_name, stimulus, try_load=try_load)

        # Simulate or load MF results
        mf_results = self.simulate_mfs(stimulus_name, stimulus, try_load=try_load)

        # Optionally plot the results
        if plot:
            fig_plots.fig_full_network_overview_together(
                snn_results, 
                mf_results,
                common_params={
                    'xmargin': 0.0,
                    'ymargin': 0.0,
                    'labels': ['SNN'] + self.mf_names,
                    'legend': {'loc': 'upper right'},
                },
                fig_params={
                    'figsize': (20, 10),  # width, height
                    'tight_layout': True,
                    'savefig': True,
                    'savefig_path': self.results_path / f"{stimulus_name}_Full_network_overview_together.png",
                    'title' : f"Network overview for '{self.data_file_mark}' with stimulus: '{stimulus_name}' and drive rate: {stimulus['drive_rate']} Hz",
                }
            )

        compare_mf_snn_results(
            snn_results, 
            mf_results, 
            self.mf_names, 
            start_time=None,
        )

        return snn_results, mf_results

    def simulate_stimuli(self, try_load=True, plot=True):    
        """Iterates through all stimuli, simulates SNN and MF, and compares results."""
        for stimulus_name, stimulus in self.stimuli.items():
            print(f"Processing stimulus: {stimulus_name}")
            snn_results, mf_results = self.simulate_single_stimulus(stimulus_name, stimulus, try_load=try_load, plot=plot)

    def inspect_spont_activity(self, 
                               param_values:list,
                               param_to_inspect:str="stimulus.drive_rate",
                               stimulus:dict=None,
                               inspection_variables:list=None,
                               start_time:float=1000.0,
                               try_load=True, 
                               plot=True
                               ):
        """Inspect a network/stimulus parameter by running spontaneous-activity simulations.

        Spontaneous activity allows meaningful time-averaging of rates, voltages and
        adaptation, enabling direct comparison between the SNN and MF models.

        WARNING: start_time is used to discard initial transient activity before averaging.
        There is on check whether the steady state is reached, so the user should 
        ensure that the start_time is sufficient and time averaging is meaningful.
        
        
        Parameters
        ----------
        param_values : list
            Sequence of values to test for the inspected parameter.
        param_to_inspect : str, optional
            Parameter to vary, given in dot notation describing the section and key,
            e.g. "stimulus.drive_rate" or "network.exc_neuron.neuron_pars.b".
            Default is "stimulus.drive_rate".
        stimulus : dict, optional
            Stimulus specification to use for simulations. If None, a default spontaneous
            stimulus is used. Provided dicts are merged with the default stimulus.
        inspection_variables : list, optional
            List of variable names to extract from each result. If None, a sensible
            default set is used.
        start_time : float, optional
            Time (ms) after which signals are averaged for the inspection. Default 1000.0.
        try_load : bool, optional
            If True, attempt to load existing simulation results instead of re-simulating.
            Default True.
        plot : bool, optional
            If True, generate and save comparison plots for each tested value. Default True.

        Returns
        -------
        dict
            Mapping from inspection variable name to a numpy array of shape
            (1 + n_mf_models, len(param_values)). The first row corresponds to the SNN,
            subsequent rows correspond to the added MF models.
        """
        
        DEFAULT_SPONT_STIMULUS = {
            "pattern" : "NoStimulus",
            "stim_pars" : {},
            "drive_rate" : 2.0,
            "drive_increase_duration" : 400,
            "stim_target_ratio" : 1.0,
            "simulation_time" : 2000,
            "target_nodes" : 0,
            "direct_stimulation" : False
        } 

        if stimulus is None:
            stimulus = DEFAULT_SPONT_STIMULUS
        if isinstance(stimulus, dict):
            stimulus = deep_union(DEFAULT_SPONT_STIMULUS, stimulus)

        if inspection_variables is None:
            inspection_variables = [
                "exc_rate_time_mean",
                "exc_rate_time_std",
                "inh_rate_time_mean",
                "inh_rate_time_std",
                "exc_voltage_time_mean",
                "exc_voltage_time_std",
                "inh_voltage_time_mean",
                "inh_voltage_time_std",
                "exc_adaptation_time_mean",
                "exc_adaptation_time_std",
            ]

        param_section, _, param_key = param_to_inspect.partition(".")

        inspection_results_list = []

        original_value = None  # stores original value of the inspected param

        for i, value in enumerate(param_values):
            print(f"Processing {param_to_inspect} = {value}")
            stimulus_name = f"SpontStimulus_{param_key}_{value}"
            match param_section:
                case "stimulus":
                    if original_value is None:
                        original_value = flatten_dict(stimulus)[param_key]
                    stimulus = flatten_dict(stimulus)
                    stimulus[param_key] = value
                    stimulus = unflatten_dict(stimulus)
                case "network":
                    if original_value is None:
                        original_value = []
                        original_value.append(flatten_dict(self.snn_network_pars)[param_key])
                        for mf_pars in self.mf_net_pars:
                            original_value.append(flatten_dict(mf_pars)[param_key])
                    self.update_network_parameter(param_key, value)

                case _:
                    raise NotImplementedError(f"Parameter section '{param_section}' is not implemented for inspection.")

            # Run simulation for the current parameter value
            snn_results, mf_results_list = self.simulate_single_stimulus(stimulus_name, stimulus, try_load=try_load, plot=plot)
            results_list = [snn_results] + mf_results_list

            # Extract inspection variables
            for j, (results, network_name) in enumerate(zip(results_list, ["SNN"] + self.mf_names)):
                print(f"Processing model: {network_name}")

                if i==0:  # init the inspection storage class
                    inspection_results_list.append(
                        SpontaneousActivityInspectionResults(
                            inspected_network_name=network_name,
                            inspected_network_params=[self.snn_network_pars, *self.mf_net_pars][j],
                            inspected_stimulus_name=stimulus_name,
                            inspected_stimulus_params=stimulus,
                            inspected_param=param_to_inspect,
                            measured_variables=inspection_variables
                        ))

                inspection_results_list[j].add_result(
                    param_value=value,
                    results=results,
                    start_time=start_time
                )

        # Restore original parameter value
        if original_value is not None:
            match param_section:
                case "stimulus":
                    stimulus = flatten_dict(stimulus)
                    stimulus[param_key] = original_value
                    stimulus = unflatten_dict(stimulus)
                case "network":
                    self.update_network_parameter(param_key, original_value, unique=True)

        return inspection_results_list

    def update_network_parameter(self, param_key_to_update:str, new_value, unique=False):
        """Update a network parameter in both SNN and MF models.

        Parameters
        ----------
        param_key_to_update : str
            The parameter to update, specified in dot notation (e.g., "exc_neuron.neuron_pars.b").
        new_value : any
            The new value to set for the specified parameter.
        unique : bool, optional
            If True, new_value is expected to be a list with unique values for each network model.
        """
        if unique:
            assert isinstance(new_value, list), "new_value should be a list when unique=True."
            assert len(new_value) == 1 + len(self.mf_net_pars), "new_value list length should match number of network models (SNN + MF)."


        self.snn_network_pars = flatten_dict(self.snn_network_pars)

        if unique:
            self.snn_network_pars[param_key_to_update] = new_value[0]
        else:
            self.snn_network_pars[param_key_to_update] = new_value
        self.snn_network_pars = unflatten_dict(self.snn_network_pars)

        # Update MF parameters
        updated_mf_net_pars = []
        for i, mf_pars in enumerate(self.mf_net_pars):
            mf_pars = flatten_dict(mf_pars)
            if unique:
                mf_pars[param_key_to_update] = new_value[i+1]
            else:
                mf_pars[param_key_to_update] = new_value
            mf_pars = unflatten_dict(mf_pars)
            updated_mf_net_pars.append(mf_pars)
        self.mf_net_pars = updated_mf_net_pars

    def inspect_stimulus(self):
        pass

    def inspect_parameter(self):
        pass



def resolve_workflow_parameters(workflow_params:str|Path|dict):
    """Resolves the workflow parameters from a given input.

    This function takes the workflow parameters as input, which can be in different formats (str, Path, dict),
    and resolves them into the internal state of the controller. It updates the neuron simulation parameters,
    transfer function simulation parameters, network simulation parameters, and default mean-field model parameters.

    Parameters
    ----------
    workflow_params : str or Path or dict
        Workflow parameters, either as a file name or path to parameter or a dictionary.
        If a string (file name) is provided, the file should be in JSON format and located at PARAMS_PATH
        If a Path is provided, it should point to a JSON file.

    Returns
    -------
    dict : A dictionary containing the resolved workflow parameters, with keys:
        - "single_neuron_simulations": Parameters for single neuron simulations.
        - "transfer_functions": Parameters for transfer function fitting.
        - "network_simulations": Parameters for network simulations.
        - "mf_model": Default parameters for the mean-field model.
    """

    if isinstance(workflow_params, str):
        workflow_params = load_json(PARAMS_PATH / workflow_params)
    elif isinstance(workflow_params, Path):
        workflow_params = load_json(workflow_params)
    elif isinstance(workflow_params, dict):
        workflow_params = workflow_params
    else:
        raise ValueError("workflow_params should be a str, Path, or dict")
    
    return workflow_params

def resolve_stimuli_parameters(stimuli_params:str|Path|dict):
    """Resolves the stimuli parameters from a given input.

    This function takes the stimuli parameters as input, which can be in different formats (str, Path, dict),
    and resolves them into the internal state of the controller. It updates the stimuli parameters.

    Parameters
    ----------
    stimuli_params : str or Path or dict
        Stimuli parameters, either as a file name or a dictionary.
        If a string (file name) is provided, the file should be in Python-evaluatable format.
        If a Path is provided, it should point to a Python-evaluatable file.
        If a dictionary is provided, the key:value pairs should follow:
            stimulus_name : stimulus_specs
    Returns
    -------
    dict : A dictionary containing the resolved stimuli parameters, with keys as stimulus names and values as stimulus specifications.
    """

    if isinstance(stimuli_params, str):
        stimuli = load_with_eval(PARAMS_PATH / stimuli_params)
    elif isinstance(stimuli_params, Path):
        stimuli = load_with_eval(stimuli_params)
    elif isinstance(stimuli_params, dict):
        stimuli = stimuli_params
    else:
        raise ValueError("stimuli_params should be a str, Path, or dict")
    
    return stimuli

def resolve_network_parameters(snn_params:str|Path|dict):
    """Resolves the network parameters from a given input.

    This function takes the SNN parameters as input, which can be in different formats (str, Path, dict),
    and resolves them into the internal state of the controller. It updates the SNN network parameters
    and neuron parameters.

    Parameters
    ----------
    snn_params : str or Path or dict
        Spiking Neural Network parameters, either as a file name or a dictionary.
        If a string (file name) is provided, the file should be in Python-evaluatable format.
        If a Path is provided, it should point to a Python-evaluatable file.
        If a dictionary is provided, it will be used directly.
    Returns
    -------
    dict : A dictionary containing the resolved SNN network parameters. 
            The dictionary should have keys corresponding to neuron names 
            (e.g., "exc_neuron", "inh_neuron") and a key "network" for network parameters.
    """

    if isinstance(snn_params, str):
        snn_network_pars = load_with_eval(PARAMS_PATH / snn_params)
    elif isinstance(snn_params, Path):
        snn_network_pars = load_with_eval(snn_params)
    elif isinstance(snn_params, dict):
        snn_network_pars = snn_params
    else:
        raise ValueError("snn_params should be a str, Path, or dict")
    
    return snn_network_pars
