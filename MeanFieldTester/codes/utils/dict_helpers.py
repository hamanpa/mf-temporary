"""
Module containing helpers for working with dictionaries.

"""
from copy import deepcopy


def get_items_recursive(dictionary, key, match_method='endswith', items=None):
    """Recursively search for items in a dictionary of dictionaries.
    
    It matches the key based on the specified match method.
    The match methods can be:
    - 'endswith': matches keys that end with the specified key.
    - 'startswith': matches keys that start with the specified key.
    - 'contains': matches keys that contain the specified key.
    - 'exact': matches keys that are exactly equal to the specified key.

    If `items` is provided, it will be used to store found items; otherwise, a new dictionary will be created.
    
    Parameters
    ----------
    dictionary : dict
        The dictionary to search.
    key : str
        The key to search for.
    match_method : str, optional
        The method to match the key. Options are 'endswith', 'startswith', 'contains', 'exact'.
        Default is 'endswith'.
    items : dict, optional
        A dictionary to store found items. If None, a new dictionary is created.

    Returns
    -------
    dict
        A dictionary containing all items that match the key.
    """

    if items is None:
        items = {}
    for k,v in dictionary.items():
        if match_method == 'endswith' and k.endswith(key):
            items[k] = v
        elif match_method == 'startswith' and k.startswith(key):
            items[k] = v
        elif match_method == 'contains' and key in k:
            items[k] = v
        elif match_method == 'exact' and k == key:
            items[k] = v
        elif isinstance(v, dict):  # If the value is a dictionary, recurse into it
            items = get_items_recursive(v, key, match_method, items=items)
    return items

def flatten_dict(dictionary, parent_key='', sep='.'):
    """Flatten a nested dictionary.

    Parameters
    ----------
    dictionary : dict
        The dictionary to flatten.
    parent_key : str, optional
        The base key to prepend to the keys in the flattened dictionary.
    sep : str, optional
        The separator to use between keys. Default is '.'.

    Returns
    -------
    dict
        A flattened dictionary with keys as concatenated strings.
    """
    items = []
    for k, v in dictionary.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(dictionary, sep='.'):
    """Unflatten a flattened dictionary.

    Parameters
    ----------
    dictionary : dict
        The flattened dictionary to unflatten.
    sep : str, optional
        The separator used in the keys of the flattened dictionary. Default is '.'.

    Returns
    -------
    dict
        An unflattened dictionary with nested structure.
    """
    result = {}
    for key, value in dictionary.items():
        parts = key.split(sep)
        current = result  # pointer to the nested dictionary
        for part in parts[:-1]:  # Traverse the parts and create nested dictionaries
            if part not in current:  # if the part does not exist, create a new dictionary
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result

def deep_update(dict_to_update: dict, updates: dict):
    """Recursively update a dictionary with another dictionary.

    Parameters
    ----------
    dict_to_update : dict
        The original dictionary to be updated.
    updates : dict
        The dictionary with updates.

    Returns
    -------
    None
        The function updates the original dictionary in place.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and k in dict_to_update and isinstance(dict_to_update[k], dict):
            dict_to_update[k] = deep_update(dict_to_update[k], v)
        else:
            dict_to_update[k] = v

def deep_union(dict1: dict, dict2: dict) -> dict:
    """Recursively merge two dictionaries, combining their keys and values.

    Works as a union operation: dict1 | dict2.
    With the exception if a key exists in both dictionaries and the corresponding
    values are also dictionaries, it merges them recursively. Otherwise, the value from
    the second dictionary overwrites the value from the first.

    Parameters
    ----------
    dict1 : dict
        The first dictionary.
    dict2 : dict
        The second dictionary.

    Returns
    -------
    dict
        A new dictionary that is the union of the two input dictionaries.
    """
    result = deepcopy(dict1)
    for k, v in dict2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_union(result[k], v)
        else:
            result[k] = v
    return result