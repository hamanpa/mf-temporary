# from pathlib import Path
# import pickle
# from .base import BaseSNNSimulator

# from .pynn_simulator import PyNNSNNSimulator
# from .config import NetworkSimulationConfig

# from ..stimuli.config import BaseStimulusConfig
# from ..network_params.models import BiologicalParameters

# from ..data_structures.network import SNNResults
# from ..data_structures.single_neuron import SingleNeuronResults

# # The Registry: Add new simulators here in the future
# SIMULATOR_REGISTRY = {
#     "pynn.nest": PyNNSNNSimulator,
# }

# def get_simulator(method_name: str) -> BaseSNNSimulator:
#     """Factory function to get the correct simulator class."""
#     if method_name not in SIMULATOR_REGISTRY:
#         raise ValueError(f"Simulator method '{method_name}' not found. Available: {list(SIMULATOR_REGISTRY.keys())}")
#     return SIMULATOR_REGISTRY[method_name]()

# def run_snn_simulation_workflow(
#         snn_sim_params: NetworkSimulationConfig, 
#         network_params: BiologicalParameters,
#         stimuli_dict: dict[str, BaseStimulusConfig]
#     ) -> dict[str, SNNResults]:
#     """High-level orchestrator for single neuron simulation."""
    
#     match snn_sim_params.execution_mode:
#         case "load":
#             raise NotImplementedError("Loading SNN simulation results is not implemented yet. Please run the simulation first.")
#             neuron_results = dict()
#             for neuron_name in network_params.internal_neurons:
#                 attribute_name = f"{neuron_name}_data_path"
#                 data_path = Path(getattr(snn_sim_params, attribute_name))
#                 with open(data_path, 'rb') as f:
#                     data = pickle.load(f)
#                 if not isinstance(data, SingleNeuronResults):
#                     raise ValueError(f"Loaded data for {neuron_name} is not of type SingleNeuronResults. Got {type(data)} instead.")
#                 neuron_results[neuron_name] = data
#             print("Loaded neurons successfully.")

#         case "run":
#             simulator_name = snn_sim_params.simulator
#             simulator = get_simulator(simulator_name)
#             # snn_simulator = simu
#             snn_results = {}
#             for stim_name, stim_params in stimuli_dict.items():
#                 print(f"Running SNN simulation for stimulus: {stim_name}")
#                 snn_results[stim_name] = simulator.simulate(network_params, snn_sim_params, stim_params)

#         case _:
#             # NOTE: this should never happen due to Pydantic validation, 
#             # but we include it for safety and clarity, e.g. if someone allows
#             # new execution modes in the future, but forgets to implement them
#             # here, this will raise a clear error instead of silently doing
#             # nothing or crashing in an obscure way.
#             raise NotImplementedError(f"Execution mode '{neuron_sim_params.execution_mode}' is not implemented. Use 'run' or 'load'.")

#     return neuron_results

from pathlib import Path
import pickle
from .base import BaseSNNSimulator

from .pynn_simulator import PyNNSNNSimulator
from .config import NetworkSimulationConfig

from ..stimuli.config import BaseStimulusConfig
from ..network_params.models import BiologicalParameters

from ..data_structures.network import SNNResults

SIMULATOR_REGISTRY = {
    "pynn.nest": PyNNSNNSimulator,
}

def get_simulator(method_name: str) -> BaseSNNSimulator:
    """Factory function to get the correct simulator class."""
    if method_name not in SIMULATOR_REGISTRY:
        raise ValueError(f"Simulator method '{method_name}' not found. Available: {list(SIMULATOR_REGISTRY.keys())}")
    return SIMULATOR_REGISTRY[method_name]()

def run_snn_simulation_workflow(
        snn_sim_params: NetworkSimulationConfig, 
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