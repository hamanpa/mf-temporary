"""
Module containing utility functions for file and directory handling
"""

import json
import pickle
from pathlib import Path
import time

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

