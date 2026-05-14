from pathlib import Path
import pickle
from .base import BaseSNNSimulator

from .pynn_simulator import PyNNSNNSimulator
from .config import SpikingNeuralNetworkSimulationConfig

from ..stimuli.config import BaseStimulusConfig
from ..network_params.models import BiologicalParameters

from ..data_structures.snn_simulation import SNNResults

SIMULATOR_REGISTRY = {
    "pynn.nest": PyNNSNNSimulator,
}

def get_simulator(method_name: str) -> BaseSNNSimulator:
    """Factory function to get the correct simulator class."""
    if method_name not in SIMULATOR_REGISTRY:
        raise ValueError(f"Simulator method '{method_name}' not found. Available: {list(SIMULATOR_REGISTRY.keys())}")
    return SIMULATOR_REGISTRY[method_name]()

def run_snn_simulation_workflow(
        snn_sim_params: SpikingNeuralNetworkSimulationConfig, 
        network_params: BiologicalParameters,
        stimuli_dict: dict[str, BaseStimulusConfig]
    ) -> dict[str, SNNResults]:
    """High-level orchestrator for network simulation."""
    
    match snn_sim_params.execution_mode:
        case "load":
            raise NotImplementedError("Loading SNN simulation results is not implemented yet. Please run the simulation first.")

        case "run":
            simulator_name = snn_sim_params.simulator
            snn_results = {}
            
            for stim_name, stim_params in stimuli_dict.items():
                print(f"Running SNN simulation for stimulus: {stim_name}")
                simulator = get_simulator(simulator_name)

                print("Building SNN network...")
                simulator.build_network(network_params, snn_sim_params)

                snn_results[stim_name] = simulator.run_stimulus(stim_params)

                simulator.end()

        case _:
            raise NotImplementedError(f"Execution mode '{snn_sim_params.execution_mode}' is not implemented. Use 'run' or 'load'.")

    return snn_results