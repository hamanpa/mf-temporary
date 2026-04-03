"""
This module defines helper functions for processing SNN results
"""

import numpy as np

def activity_from_spikes_histogram(spikes:list[list], times:np.array, bin_size:float|int):
    """Calculate the mean activity from population spikes using a histogram.

    Parameters
    ----------    
    spikes : list of lists of float, [ms]
        spike times for each neuron in the network, 
        first list indexes the neuron and the second indexes the spike times
    times : 1D array, [ms]
        time points at which to calculate the activity
    bin_size : float, [ms]
        size of the bins for the histogram

    Returns
    -------
    activity : 1D array, [Hz]
        activity of spike counts per bin
    """

    duration = times[-1] - times[0]
    cells = len(spikes)
    bins_num = int(duration/bin_size)
    all_spike_times = np.concatenate(spikes)
    hist, bin_edges = np.histogram(all_spike_times, bins=bins_num, range=tuple(times[[0,-1]]))
    activity = hist / bin_size / cells * 1000

    # histogram is interpolated to the times array
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    activity = np.interp(times, bin_centers, activity) 

    return activity

def activity_from_spikes_sliding_window(spikes, times, window_size):
    """
    Calculate the mean activity from population spikes using a sliding window.
    The sliding window is a simple moving average.

    The window is centered at the given time point, 
    i.e. for time t, the window is [t-window_size/2, t+window_size/2].
    
    Parameters
    ----------
    spikes : list of lists of float, [ms]
        spike times for each neuron in the network, 
        first list indexes the neuron and the second indexes the spike times
    times : 1D array, [ms]
        time points at which to calculate the activity
    window_size : float, [ms]
        size of the sliding window

    Returns
    -------
    activity : 1D array, [Hz]
        mean activity at each time point
    """

    cells = len(spikes)
    spikes = np.concatenate(spikes)
    activity = np.zeros_like(times)

    for i, t in enumerate(times):
        valid_spikes = ((spikes >= t-window_size/2) & (spikes < t + window_size/2)).sum()
        activity[i] = valid_spikes / cells / (window_size / 1000)

    return activity

def activity_from_spikes_alpha_window(spikes, times, alpha_tau, method="numpy"):
    """
    Calculate the mean activity from population spikes using an alpha window.
    The alpha window is a function of time that decays exponentially.
    
    Parameters
    ----------
    spikes : list of lists of float, [ms]
        spike times for each neuron in the network, 
        first list indexes the neuron and the second indexes the spike times
    times : 1D array, [ms]
        time points at which to calculate the activity
    alpha_tau : float, [ms]
        time constant for the alpha window
    method : str, optional
        method to use for calculating activity, options are "for-loop", "numpy", "convolve"

    Returns
    -------
    activity : 1D array, [Hz]
        mean activity at each time point
    """
    options = {"for-loop", "numpy", "convolve"}
    if method not in options:
        raise ValueError(f"Method must be one of {options}, got {method}")

    cells = len(spikes)
    spikes = np.concatenate(spikes)
    activity = np.zeros_like(times)

    match method:
        case "for-loop":
            for i, t in enumerate(times):
                past_spikes = spikes[spikes < t]
                t_diff = t - past_spikes
                activity[i] = ((t_diff) / (alpha_tau**2) * np.exp(-(t_diff) / alpha_tau)).sum()
        case "numpy":
            # NOTE: might be RAM heavy for large number of spikes
            t_diff = times[:, None] - spikes[None, :]
            valid_diff = np.where(t_diff > 0, t_diff, 0)
            activity = (valid_diff / (alpha_tau**2) * np.exp(-valid_diff / alpha_tau)).sum(axis=1)
        case "convolve":
            # NOTE: this method undershoots a bit, due to discretization error!
            import scipy.signal
            hist, bin_edges = np.histogram(spikes, bins=times)
            kernel_times = np.arange(0, 5*alpha_tau, times[1]-times[0])  # up to 5 tau
            alpha_kernel = (kernel_times / (alpha_tau**2)) * np.exp(-kernel_times / alpha_tau)

            activity = scipy.signal.fftconvolve(hist, alpha_kernel, mode='full')[:len(times)]
    activity = activity / cells *1000
    return activity

def spike_counts(spikes, start_time=0, end_time=None):
    """Get the number of spikes for each neuron after a certain time.
    
    Parameters
    ----------
    spikes : list of lists of float, [ms]
        spike times for each neuron in the network, 
        first list indexes the neuron and the second indexes the spike times
    start_time : float, [ms]
        time after which to count spikes
    end_time : float, [ms], optional
        time before which to count spikes, if None, counts until the end of the simulation

    Returns
    -------
    spike_counts : 1D array
        number of spikes for each neuron after start_time
    """
    if end_time is None:
        end_time = np.inf
    spike_counts = np.array([((spike_train >= start_time) & (spike_train <= end_time)).sum() for spike_train in spikes])
    return spike_counts

