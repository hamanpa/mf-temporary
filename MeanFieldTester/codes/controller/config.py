"""
python -m codes.controller.config --template
    running this will generate a default_workflow.yaml file in the current directory, 
    which you can then edit to specify your workflow parameters.

python -m codes.controller.config --schema
    running this will generate a workflow_schema.json file in the current directory,
    which contains the full JSON schema of the WorkflowConfig, including all descriptions and options.

"""

import yaml
import json
import argparse
from pathlib import Path
from pydantic import BaseModel

from ..neuron_simulation.config import NeuronSimulationConfig, RunSimulationConfig, GridConfig, SingleNeuronLinearGrid, SimulatorType
from ..transfer_function.config import TransferFunctionConfig

# from codes.meanfield_simulation.config import MeanFieldConfig
# from codes.network_simulation.config import NetworkConfig

class WorkflowConfig(BaseModel):
    neuron_simulation: NeuronSimulationConfig

    # TODO: transfer_function will be part of meanfield_simulation, not a separate module
    # it is here for testing (and because MeanFieldConfig is not yet defined)
    transfer_function: TransferFunctionConfig
    
    # meanfield_simulation: MeanFieldConfig 
    # network_simulation: NetworkConfig


def load_workflow_config(source: str | Path | dict) -> WorkflowConfig:
    """Loads and validates the master workflow YAML file."""
    if isinstance(source, (str, Path)):
        with open(source, "r") as f:
            raw_data = yaml.safe_load(f)
        print(f"Workflow configuration successfully loaded from {source}")
    elif isinstance(source, dict):
        raw_data = source
        print("Workflow configuration successfully loaded from dictionary.")
    else:
        raise TypeError("Source must be a file path (str or Path) or a dictionary.")

    config = WorkflowConfig(**raw_data)
    return config

def generate_default_yaml(output_path: str = "default_workflow.yaml"):
    """Generates a complete, blank template for the user to fill out."""
    
    # Create safe default dummies for the modules
    default_single_neuron_grid = SingleNeuronLinearGrid(
        grid_type="linear",
        exc_rate_grid=[0, 60, 16], 
        inh_rate_grid=[0, 60, 16]
    )
    default_grid = GridConfig(
        exc_neuron=default_single_neuron_grid,
        inh_neuron=default_single_neuron_grid
    )
    GridConfig
    default_neuron_run = RunSimulationConfig(
        execution_mode="run",
        simulator=SimulatorType.PYNN_NEST,
        grid=default_grid, 
        simulation_time=5000.0
    )
    
    # Snap them into the master config
    master_config = WorkflowConfig(
        neuron_simulation=default_neuron_run
    )

    # Convert to standard dictionary, ignoring unset optional fields
    config_dict = master_config.model_dump(exclude_unset=True)

    # Write the beautiful YAML file
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
    print(f"Generated default YAML template at: {output_path}")


def generate_json_schema(output_path: str = "workflow_schema.json"):
    """Exports the complete Pydantic schema with all descriptions and options."""
    # .model_json_schema() is a built-in Pydantic V2 method
    schema = WorkflowConfig.model_json_schema()
    
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
        
    print(f"Full JSON schema successfully generated at: {output_path}")
    print("Tip: Users can link this schema in VSCode for YAML auto-complete!")

# --- 5. The Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MeanFieldTester Configuration Manager")
    
    # Notice we now have two distinct flags!
    parser.add_argument("--template", action="store_true", help="Generate a default workflow.yaml template")
    parser.add_argument("--schema", action="store_true", help="Generate the full JSON schema of all available options")
    
    args = parser.parse_args()
    
    if args.template:
        generate_default_yaml()
    elif args.schema:
        generate_json_schema()
    else:
        print("Please specify an action. Try:")
        print("  python -m codes.controller.config --template")
        print("  python -m codes.controller.config --schema")