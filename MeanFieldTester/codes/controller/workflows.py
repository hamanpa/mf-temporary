"""
Execution workflows for coordinating single-step simulations across SNN and MF backends.
Serves as the stateless execution engine for controllers and scripts.
"""

from typing import Dict, List

from ..neuron_simulation import run_neuron_simulation_workflow
from ..transfer_function import run_tf_fitting_workflow
from ..snn_simulation import run_snn_simulation_workflow
from ..mf_simulation import run_mf_simulation_workflow

from ..network_params.models import BiologicalParameters
from ..stimuli.config import StimuliCollection
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

    mf_results_dict = {stim_name: [] for stim_name in stimulus_config.stimuli.keys()}
    for mf_model_name, mf_sim_params in sim_params.mf_models.items():
        mf_results_full = run_mf_simulation_workflow(
            mf_sim_params, network_params, stimulus_config
        )
        for stim_name, mf_results in mf_results_full.items():
            mf_results_dict[stim_name].append(mf_results)

    basic_results["mf_results"] = mf_results_dict

    return basic_results