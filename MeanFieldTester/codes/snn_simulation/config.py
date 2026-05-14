from enum import Enum
from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from ..neuron_simulation.config import NeuronInitialValuesConfig


class SmoothingConfig(BaseModel):
    function: Literal["sliding_window", "gaussian"]
    time_constant: float = Field(description="Time constant for the smoothing function in [ms].")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional keyword arguments for the smoothing function.")


class SpikingNeuralNetworkSimulationConfig(BaseModel):
    """
    Core configuration for the Spiking Neural Network simulation.
    Ensures that parameters like recorded populations are not hardcoded.
    """
    network_name: str = Field(description="Name of the network being simulated, used for logging and saving results.")
    execution_mode: Literal["run", "load", "skip"] = "run"

    simulator: str = Field(default="pynn.nest", description="The backend simulator to use.")
    time_step: float = Field(default=0.1, gt=0, description="Integration time step in [ms].")
    seed: int = Field(default=42, description="Random number generator seed.")
    n_runs: int = Field(default=1, ge=1, description="Number of statistical runs.")
    cpus: int = Field(default=1, ge=1, description="Number of CPU cores for multiprocessing.")
    
    recorded_samples: Dict[str, int] = Field(default_factory=lambda: {"exc_neuron": 500, "inh_neuron": 500}, description="Number of recorded samples for each neuron type during the simulation.")
 
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)

    init_values: Dict[str, NeuronInitialValuesConfig] = Field(
        default_factory=dict, 
        description=("Initial values for each neuron type. Keys should match the internal neuron names defined in the network parameters.")
    )