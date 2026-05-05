from .config import StimulusConfig, StimuliCollection
from pathlib import Path
import yaml
from typing import Dict


def load_stimuli_config(source: str | Path | dict) -> Dict[str, StimulusConfig]:
    """Loads and validates the stimuli YAML file."""
    if isinstance(source, (str, Path)):
        with open(source, "r") as f:
            raw_data = yaml.safe_load(f)
    elif isinstance(source, dict):
        raw_data = source
    else:
        raise TypeError("Source must be a file path (str or Path) or a dictionary.")

    # Validate the data using our collection model
    collection = StimuliCollection(raw_data)
    
    # Return the inner dictionary for easy access in your runner
    return collection.root