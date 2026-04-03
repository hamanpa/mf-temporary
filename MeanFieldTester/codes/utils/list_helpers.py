"""
Module containing utility functions for working with lists.
"""


def indexed_linear_sample(lst, num_samples):
    """Sample elements from a list in a linearly distributed manner.
    
    Parameters
    ----------
    lst : list
        The input list to sample from.
    num_samples : int
        The number of samples to take.
    
    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains the index and the sampled element.
    
    Notes
    -----
    This function samples elements from the input list at linearly spaced indices.
    If `num_samples` is greater than or equal to the length of `lst`, it returns the whole list.
    If `num_samples` is 1, it returns the middle element of the list.
    """
    
    if num_samples >= len(lst):
        print("WARNING: num_samples is greater than or equal to the length of the list. Returning the whole list.")
        return [(i, lst[i]) for i in range(len(lst))]
    
    # Calculate the step size for even distribution
    step = (len(lst) - 1) / (num_samples - 1) if num_samples > 1 else 0
    
    # Create linearly spaced indices
    indices = [int(round(step * i)) for i in range(num_samples)]
    
    # Handle edge case for num_samples=1
    if num_samples == 1:
        indices = [len(lst) // 2]  # Take middle element if only one sample
    
    # Return the elements at the calculated indices
    return [(i, lst[i]) for i in indices]