from pathlib import Path
import pickle
from .base import BaseNeuronSimulator
from .pynn_simulator import PyNNSimulator
from .zerlaut2018_simulator import Zerlaut2018Simulator
from .config import NeuronSimulationConfig
from ..data_structures.neuron_simulation import SingleNeuronResults
from ..network_params.models import BiologicalParameters

# The Registry: Add new simulators here in the future
SIMULATOR_REGISTRY = {
    "pynn.nest": PyNNSimulator,
    "zerlaut2018": Zerlaut2018Simulator,
}

def get_simulator(method_name: str) -> BaseNeuronSimulator:
    """Factory function to get the correct simulator class."""
    if method_name not in SIMULATOR_REGISTRY:
        raise ValueError(f"Simulator method '{method_name}' not found. Available: {list(SIMULATOR_REGISTRY.keys())}")
    return SIMULATOR_REGISTRY[method_name]()

def run_neuron_simulation_workflow(neuron_sim_params: NeuronSimulationConfig, network_params: BiologicalParameters) -> dict[str, SingleNeuronResults]:
    """High-level orchestrator for single neuron simulation."""
    
    match neuron_sim_params.execution_mode:
        case "load":
            neuron_results = dict()
            for neuron_name in network_params.internal_neurons:
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
            neuron_results = simulator.simulate(network_params, neuron_sim_params)

        case "skip":
            neuron_results = {neuron_name: None for neuron_name in network_params.internal_neurons}

        case "try_load":
            valid_exc_path = Path(neuron_sim_params.exc_neuron_data_path).exists() if neuron_sim_params.exc_neuron_data_path else False
            valid_inh_path = Path(neuron_sim_params.inh_neuron_data_path).exists() if neuron_sim_params.inh_neuron_data_path else False

            if valid_exc_path and valid_inh_path:
                print("Found existing neuron simulation data, loading...")
                neuron_results = dict()
                for neuron_name in network_params.internal_neurons:
                    attribute_name = f"{neuron_name}_data_path"
                    data_path = Path(getattr(neuron_sim_params, attribute_name))
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    if not isinstance(data, SingleNeuronResults):
                        raise ValueError(f"Loaded data for {neuron_name} is not of type SingleNeuronResults.")
                    neuron_results[neuron_name] = data
            else:
                print("Simulation data not found. Running simulation...")
                simulator_name = neuron_sim_params.simulator
                simulator = get_simulator(simulator_name)
                neuron_results = simulator.simulate(network_params, neuron_sim_params)

                for neuron_name, result in neuron_results.items():
                    attribute_name = f"{neuron_name}_data_path"
                    data_path = Path(getattr(neuron_sim_params, attribute_name))
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(data_path, 'wb') as f:
                        pickle.dump(result, f)
                print("Saved simulation data successfully.")
        case _:
            # NOTE: this should never happen due to Pydantic validation, 
            # but we include it for safety and clarity, e.g. if someone allows
            # new execution modes in the future, but forgets to implement them
            # here, this will raise a clear error instead of silently doing
            # nothing or crashing in an obscure way.
            raise NotImplementedError(f"Execution mode '{neuron_sim_params.execution_mode}' is not implemented. Use 'run' or 'load'.")

    return neuron_results