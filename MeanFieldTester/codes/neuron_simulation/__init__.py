from pathlib import Path
import pickle
from .base import BaseNeuronSimulator
from .pynn_simulator import PyNNSimulator
from .config import NeuronSimulationConfig
from ..data_structures.single_neuron import SingleNeuronResults

# The Registry: Add new simulators here in the future
SIMULATOR_REGISTRY = {
    "pynn.nest": PyNNSimulator,
}

def get_simulator(method_name: str) -> BaseNeuronSimulator:
    """Factory function to get the correct simulator class."""
    if method_name not in SIMULATOR_REGISTRY:
        raise ValueError(f"Simulator method '{method_name}' not found. Available: {list(SIMULATOR_REGISTRY.keys())}")
    return SIMULATOR_REGISTRY[method_name]()

def run_neuron_simulation_workflow(neuron_sim_params: NeuronSimulationConfig, neuron_params:dict) -> dict[str, SingleNeuronResults]:
    """High-level orchestrator for single neuron simulation."""
    neuron_names = list(neuron_params.keys())
    
    match neuron_sim_params.execution_mode:
        case "load":
            neuron_results = dict()
            for neuron_name in neuron_names:
                attribute_name = f"{neuron_name}_data_path"
                data_path = Path(getattr(neuron_sim_params, attribute_name))
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                if not isinstance(data, SingleNeuronResults):
                    raise ValueError(f"Loaded data for {neuron_name} is not of type SingleNeuronResults. Got {type(data)} instead.")
                neuron_results[neuron_name] = data
            print("Loaded neurons successfully.")

        case "run":
            simulator_name = neuron_sim_params.simulator
            simulator = get_simulator(simulator_name)
            neuron_results = simulator.simulate(neuron_params, neuron_sim_params)

        case _:
            # NOTE: this should never happen due to Pydantic validation, 
            # but we include it for safety and clarity, e.g. if someone allows
            # new execution modes in the future, but forgets to implement them
            # here, this will raise a clear error instead of silently doing
            # nothing or crashing in an obscure way.
            raise NotImplementedError(f"Execution mode '{neuron_sim_params.execution_mode}' is not implemented. Use 'run' or 'load'.")

    return neuron_results