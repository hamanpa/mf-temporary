from pathlib import Path
import yaml

from .models import BiologicalParameters
from .normalization import normalize_synapses



def load_network_parameters(source: str | Path | dict) -> BiologicalParameters:
    """
    Loads network parameters from a YAML file (or a raw dict), normalises synapse types
    (see `normalization.normalize_synapses`) and strictly validates the data.
    """
    if isinstance(source, (str, Path)):
        with open(source, 'r') as f:
            # yaml.safe_load resolves the & / * anchors
            raw_dict = yaml.safe_load(f)
    elif isinstance(source, dict):
        raw_dict = source
    else:
        raise TypeError("Source must be a file path (str or Path) or a dictionary.")

    return BiologicalParameters(**normalize_synapses(raw_dict))
