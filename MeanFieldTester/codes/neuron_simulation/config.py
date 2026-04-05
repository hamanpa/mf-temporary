from enum import Enum
from typing import List, Annotated, Literal, Any
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator
import yaml


class SimulatorType(str, Enum):
    PYNN_NEST = "pynn.nest"


class LinearGrid(BaseModel):
    grid_type: Literal["linear"]
    exc_rate_grid: List[float] = Field(description="[min, max, n_points]")
    inh_rate_grid: List[float] = Field(description="[min, max, n_points]")


class AdaptiveGrid(BaseModel):
    grid_type: Literal["adaptive"]
    adaptive_input: str = Field(description="Which input to adapt to, e.g., 'exc_rate'")
    exc_rate_grid: List[float] = Field(description="[min, max, n_points]")
    inh_rate_grid: List[float] = Field(description="[min, max, n_points]")
    max_out_rate_step: float = Field(description="Maximum allowed step in output firing rate for grid adaptation")


class CustomGrid(BaseModel):
    grid_type: Literal["custom"]
    model_config = ConfigDict(arbitrary_types_allowed=True)
    exc_rate_grid: Path | str | np.ndarray = Field(description="Either a 2D array of meshgrid (indexed: (exc_rate, inh_rate)) OR a path to a .npy file containing the grid")
    inh_rate_grid: Path | str | np.ndarray = Field(description="Either a 2D array of meshgrid (indexed: (exc_rate, inh_rate)) OR a path to a .npy file containing the grid")

    @field_validator('exc_rate_grid', 'inh_rate_grid')  # Pydantic will run this function twice, for each input
    @classmethod
    def load_mesh_if_path(cls, value: Any) -> np.ndarray:
        
        if isinstance(value, np.ndarray):
            if value.ndim != 2:
                raise ValueError(f"Mesh array must be 2D, but got {value.ndim}D")
            return value
        
        if isinstance(value, (str, Path)):
            file_path = Path(value)
            if not file_path.exists():
                raise FileNotFoundError(f"Custom mesh file not found: {file_path}")
            try:
                return np.load(file_path)
            except Exception as e:
                raise ValueError(f"Failed to load numpy array from {file_path}. Error: {e}")
        raise ValueError("Must be a valid path string, Path object, or numpy array.")

GridConfigType = LinearGrid | AdaptiveGrid | CustomGrid

class LoadSimulationConfig(BaseModel):
    execution_mode: Literal["load"] 
    data_path: str = Field(description="Path to the saved simulation data")


class RunSimulationConfig(BaseModel):
    execution_mode: Literal["run"]  
    
    simulator: SimulatorType
    grid: GridConfigType = Field(discriminator='grid_type')
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