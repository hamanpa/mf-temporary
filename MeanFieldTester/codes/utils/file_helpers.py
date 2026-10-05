"""
Module containing utility functions for file and directory handling
"""

import json
import pickle
from enum import Enum
from pathlib import Path
import time

import numpy as np
import yaml
from pydantic import BaseModel


def to_builtin(obj):
    """
    Recursively converts pydantic models, numpy objects, Paths and Enums into plain
    Python types (dict, list, str, float, ...) so they can be written to YAML/JSON.
    """
    if isinstance(obj, BaseModel):
        return to_builtin(obj.model_dump())
    if isinstance(obj, dict):
        return {to_builtin(key): to_builtin(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return obj


# Units of the arrays in a results .npz file are stored inside the same file, under this key,
# as a JSON string {array_key: unit}. (Per-file, because files are written by parallel workers.)
NPZ_UNITS_KEY = "units"


def encode_npz_units(units: dict[str, str]) -> np.ndarray:
    """Encodes a {array_key: unit} dict as a 0-d string array storable in an .npz file."""
    return np.array(json.dumps(units))


def decode_npz_units(npz_data) -> dict[str, str]:
    """Returns the {array_key: unit} dict of a loaded .npz file ({} for files saved without units)."""
    if NPZ_UNITS_KEY not in npz_data:
        return {}
    return json.loads(str(npz_data[NPZ_UNITS_KEY]))


def save_yaml(data, filepath: str | Path):
    """Writes `data` (pydantic models, dicts of models, numpy values, ...) to a YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.safe_dump(to_builtin(data), f, sort_keys=False)


def save_to_pickle(filepath : str | Path, **kwargs):
    """Save multiple objects to a pickle file as a dictionary.
    
    Parameters
    ----------
    filepath : str or Path
        The path where the pickle file will be saved.
    **kwargs
        Key-value pairs of objects to save in the pickle file.
    """
    filepath = Path(filepath).resolve()
    print(f"Saving objects to {filepath}")
    
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the objects to the pickle file
    if not kwargs:
        raise ValueError("No objects provided to save. Please provide at least one object.")

    if filepath.is_file():
        print(f"WARNING: File {filepath} already exists. It will be overwritten.")

    objects_dict = kwargs
    with open(filepath, 'wb') as file:
        pickle.dump(objects_dict, file)

def prepare_result_dir(dir_name="TestSimulation", parent_path:str|Path='./results', 
                       time_stamp:str=time.strftime('%Y%m%d-%H%M%S')) -> Path:
    """Prepare a directory for saving simulation results.

    This function creates a directory for saving simulation results. 
    The directory is named after the simulation name and includes a timestamp, unless time_stamp is specified otherwise.

    If the directory already exists, it will not be overwritten. The function returns the path to the created directory.
    
    Parameters
    ----------
    dir_name : str, optional
        The name of the directory. Default is "TestSimulation".
    parent_path : str or Path, optional
        The base path where the results directory will be created. Default is './results'.
    time_stamp : str, optional
        A timestamp to append to the results directory name. Default is current time in 'YYYYMMDD-HHMMSS' format.
        If `time_stamp` is an empty string, the directory will not include a timestamp.
    
    Returns
    -------
    Path
        The path to the created results directory.
    """

    results_path = Path(parent_path).resolve()
    if time_stamp:
        results_path = results_path / f"{time_stamp}_{dir_name}"
    else:
        results_path = results_path / dir_name
    results_path.mkdir(exist_ok=True, parents=True)
    return results_path

def load_json(file_path:Path|str):
    """Loads a JSON file and returns the content as a dictionary."""

    if type(file_path) is str:
        file_path = Path(file_path)

    print(f"Loading parameters from {file_path.resolve()}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_with_eval(file_path:Path|str):
    """Loads a file and evaluates its content as a Python expression, returning the resulting object."""
    if type(file_path) is str:
        file_path = Path(file_path)

    print(f"Loading parameters from {file_path.resolve()}")
    
    with open(file_path, 'r') as f:
        data = eval(f.read())
    return data

