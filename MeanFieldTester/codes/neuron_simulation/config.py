from enum import Enum
from typing import List, Annotated, Literal, Any
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import yaml


class SimulatorType(str, Enum):
    PYNN_NEST = "pynn.nest"
    ZERLAUT2018 = "zerlaut2018"


class SingleNeuronCustomGrid(BaseModel):
    """Sub-model to handle the grids for a specific neuron type."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_type: Literal["custom"]

    exc_rate_grid: Path | str | np.ndarray = Field(
        description=(
            "One of the following \n" +
            "- a 2D array of meshgrid \n" +
            "- a path to a .npy file containing the grid \n"
        )
    )
    inh_rate_grid: Path | str | np.ndarray = Field(
        description=(
            "One of the following \n" +
            "- a 2D array of meshgrid \n" +
            "- a path to a .npy file containing the grid \n"
        )
    )

    @field_validator('exc_rate_grid', 'inh_rate_grid')
    @classmethod
    def load_mesh_if_path(cls, value: Any) -> np.ndarray:
        if isinstance(value, (str, Path)):
            file_path = Path(value)
            if not file_path.exists():
                raise FileNotFoundError(f"Custom mesh file not found: {file_path}")
            try:
                value = np.load(file_path)
            except Exception as e:
                raise ValueError(f"Failed to load numpy array from {file_path}. Error: {e}")

        if isinstance(value, np.ndarray):
            if value.ndim != 2:
                raise ValueError(f"Mesh array must be 2D, but got {value.ndim}D")
            return value
                
        raise ValueError("Must be a valid path string, Path object, or numpy array.")


class SingleNeuronLinearGrid(BaseModel):
    grid_type: Literal["linear"]
    exc_rate_grid: List[float] = Field(description="[min, max, n_points]")
    inh_rate_grid: List[float] = Field(description="[min, max, n_points]")


class SingleNeuronAdaptiveGrid(BaseModel):
    grid_type: Literal["adaptive"]
    exc_rate_grid: List[float] | Literal["adaptive"] = Field(description="[min, max, n_points] OR 'adaptive' to automatically determine based on the data")
    inh_rate_grid: List[float] | Literal["adaptive"] = Field(description="[min, max, n_points] OR 'adaptive' to automatically determine based on the data")
    out_rate_grid: List[float] = Field(description="[min, max, n_points] defining the grid for the output firing rate, used to determine where to place more points in the adaptive input grid")

    n_coarse_interpolation_points: int | None = Field(
            default=None, 
            description="Number of internal simulations for interpolation. If omitted, defaults to 1.5x the out_rate_grid n_points."
        )

    @model_validator(mode='after')
    def validate_adaptive_logic(self):

        exc_is_adaptive = self.exc_rate_grid == "adaptive"
        inh_is_adaptive = self.inh_rate_grid == "adaptive"
            
        if exc_is_adaptive and inh_is_adaptive:
            raise ValueError("Both grids cannot be 'adaptive'. Choose one.")
        if not exc_is_adaptive and not inh_is_adaptive:
            raise ValueError("If grid_type is 'adaptive', either exc_rate_grid or inh_rate_grid must be set to 'adaptive'.")
        if self.n_coarse_interpolation_points is None:
            target_n_points = int(self.out_rate_grid[2])
            self.n_coarse_interpolation_points = max(10, int(target_n_points)*1.5)
        return self


SingleNeuronGridConfigType = Annotated[
    SingleNeuronLinearGrid | SingleNeuronAdaptiveGrid | SingleNeuronCustomGrid, 
    Field(discriminator="grid_type")
]
class GridConfig(BaseModel):
    exc_neuron: SingleNeuronGridConfigType
    inh_neuron: SingleNeuronGridConfigType

class LoadSimulationConfig(BaseModel):
    execution_mode: Literal["load"] 
    exc_neuron_data_path: str | Path = Field(description="Path to the saved simulation data for excitatory neuron")
    inh_neuron_data_path: str | Path = Field(description="Path to the saved simulation data for inhibitory neuron")

    @field_validator('exc_neuron_data_path', 'inh_neuron_data_path')  # Pydantic will run this function twice, for each input
    @classmethod
    def check_path_exists(cls, value: Any):
        file_path = Path(value)
        if not file_path.exists():
            raise FileNotFoundError(f"Neuron simulation data file not found: {file_path}")

class RunSimulationConfig(BaseModel):
    execution_mode: Literal["run"]  
    
    simulator: SimulatorType
    grid: GridConfig
    simulation_time: float = Field(default=5000.0, description="Total simulation time in ms")
    averaging_window: float = Field(default=2000.0, description="Time window for calculating the mean output rate, starting from the end of the simulation")
    time_step: float = Field(default=0.1, description="Time step for the simulation in ms")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    n_runs: int = Field(default=5, description="Number of simulation runs")
    cpus: int = Field(default=16, description="Number of CPU cores to use, if parallel simulation is supported by the simulator, value of 1 means no parallelization")


class SkipSimulationConfig(BaseModel):
    execution_mode: Literal["skip"]


NeuronSimulationConfig = Annotated[
    RunSimulationConfig | LoadSimulationConfig | SkipSimulationConfig, 
    Field(discriminator='execution_mode')
]