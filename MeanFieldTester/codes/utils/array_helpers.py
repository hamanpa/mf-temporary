"""
Module containing utility functions for handling arrays and numerical data.
"""

import numpy as np



def convert_to_array(arg):
    """Converts the argument to a numpy array.

    This function handles different types of input:
    - If the input is an integer or float, it converts it to a 1D numpy array.
    - If the input is already a numpy array, it returns it unchanged.
    - If the input is a list, it converts it to a numpy array.
    - If the input is of an unsupported type, it raises a TypeError.

    Parameters
    ----------
    arg : int, float, np.ndarray, or list
        The argument to convert.

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """


    if type(arg) in {int, float, np.float64}:
        return np.array([arg])
    elif type(arg) == np.ndarray:
        return arg
    elif type(arg) == list:
        return np.array(arg)
    else:
        raise TypeError(f"Invalid type {type(arg)}")

def convert_to_arrays(*args):
    """Converts all arguments to numpy arrays.

    This function applies `convert_to_array` to each argument in `args`.

    Parameters
    ----------
    *args : int, float, np.ndarray, or list
        The arguments to convert.

    Returns
    -------
    list of np.ndarray
        A list containing the converted numpy arrays.
    """

    return [convert_to_array(arg) for arg in args]


def flatten_and_remove_nans(*vals: np.ndarray):
    """Returns flattened version of given numpy arrays with nan and inf values removed

    This function takes multiple numpy arrays, flattens them, and removes any NaN or inf values.

    Parameters
    ----------
    *vals : np.ndarray
        The numpy arrays to flatten and process.

    Returns
    -------
    list of np.ndarray
        A list containing the flattened arrays with NaN values removed.
    
    Raises
    ------
    AssertionError
        If the input arrays do not have the same shape.
    """
    assert np.all([val.shape == vals[0].shape for val in vals])
    
    # 1. flatten the values
    new_vals = [val.flatten() for val in vals]
    

    # 2. mask for finite values (all input arrays must be finite for given index)
    finite_mask = np.ones_like(new_vals[0],dtype=bool)
    for val in new_vals:
        finite_mask = finite_mask & np.isfinite(val)  # previous finites AND current ones

    # 3. remove nans and infs
    new_vals = [val[finite_mask] for val in new_vals]

    return new_vals

def move_and_rescale(array: np.ndarray, expansion_point: list | np.ndarray, expansion_norm: list | np.ndarray, axis=-1):
    """Moves the given array by expansion_point and rescales it by expansion_norm along a given axis.

    The abstract operation is:
    new_array = (array - expansion_point) / expansion_norm
    over the specified axis.

    This function is useful for normalizing data or transforming it into a different scale.

    Parameters
    ----------
    array : np.ndarray
        The input array to be moved and rescaled.
    expansion_point : list or np.ndarray
        The point by which to move the values in the array.
    expansion_norm : list or np.ndarray
        The normalization factor by which to rescale the values in the array.
    axis : int, optional
        The axis along which to perform the operation. Default is -1 (the last axis).

    Returns
    -------
    np.ndarray
        The modified array after moving and rescaling.
    """

    if type(expansion_point) is list:
        expansion_point = convert_to_array(expansion_point)
        expansion_norm = convert_to_array(expansion_norm)

    assert expansion_point.size == expansion_norm.size == array.shape[axis], "expansion_point and expansion_norm must have the same size as the specified axis of the array."

    new_array = np.moveaxis(array, axis, -1)
    new_array = (new_array - expansion_point)/expansion_norm
    new_array = np.moveaxis(new_array, -1, axis)

    return new_array