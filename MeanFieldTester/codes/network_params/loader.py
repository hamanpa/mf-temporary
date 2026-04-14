from pathlib import Path
import yaml

from .models import BiologicalParameters



def load_network_parameters(yaml_path: str | Path) -> BiologicalParameters:
    """Loads the YAML file, resolves anchors, and strictly validates the data."""
    with open(yaml_path, 'r') as f:
        # yaml.safe_load automatically handles all your & and * anchors!
        raw_dict = yaml.safe_load(f)
        
    return BiologicalParameters(**raw_dict)