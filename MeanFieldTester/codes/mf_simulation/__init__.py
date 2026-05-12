from typing import Dict
from .base import BaseMFSimulator
from .tvb_simulator.simulator import TVBMFSimulator # We will create this next
from .config import MeanFieldSimulationConfig

from ..stimuli.config import BaseStimulusConfig
from ..network_params.models import BiologicalParameters
from ..data_structures.network import MFResults

# --- Registry ---
SIMULATOR_REGISTRY = {
    "tvb": TVBMFSimulator,
}

def get_simulator(method_name: str) -> BaseMFSimulator:
    """Factory function to get the correct MF simulator class."""
    if method_name not in SIMULATOR_REGISTRY:
        raise ValueError(f"Simulator method '{method_name}' not found. Available: {list(SIMULATOR_REGISTRY.keys())}")
    return SIMULATOR_REGISTRY[method_name]()

# --- Workflow Orchestrator ---
def run_mf_simulation_workflow(
        mf_sim_params: MeanFieldSimulationConfig, 
        network_params: BiologicalParameters,
        stimuli_dict: Dict[str, BaseStimulusConfig]
    ) -> Dict[str, MFResults]:
    """High-level orchestrator for mean-field network simulation."""
    
    match mf_sim_params.execution_mode:
        case "skip":
            print("Skipping Mean-Field simulation as per configuration.")
            return {}

        case "load":
            raise NotImplementedError("Loading MF simulation results is not implemented yet.")

        case "run":
            simulator_name = mf_sim_params.simulator.value # .value because it's an Enum
            mf_results = {}
            
            for stim_name, stim_params in stimuli_dict.items():
                print(f"Running MF simulation for stimulus: {stim_name}")
                simulator = get_simulator(simulator_name)

                print("Building Mean-Field network/integrator...")
                simulator.build_network(network_params, mf_sim_params)

                mf_results[stim_name] = simulator.run_stimulus(stim_params)

                simulator.end()
                
            return mf_results

        case _:
            raise NotImplementedError(f"Execution mode '{mf_sim_params.execution_mode}' is not valid.")