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

def activity_from_spikes_sliding_window(spikes: list[np.ndarray], times: np.ndarray, window_size: float|int):
    """
    Calculate the mean activity from population spikes using a sliding window.
    The sliding window is a simple moving average.

    The window is centered at the given time point, 
    i.e. for time t, the window is [t-window_size/2, t+window_size/2].
    
    Parameters
    ----------
    spikes : list of np.ndarray of float, [ms]
        spike times for each neuron in the network, 
        first list indexes the neuron and the second indexes the spike times
    times : 1D np.ndarray, [ms]
        time points at which to calculate the activity
    window_size : float, [ms]
        size of the sliding window

    Returns
    -------
    activity : 2D np.ndarray, [Hz]
        mean activity of each neuron at each time point, shape is (time points, cells)
    """

    cells = len(spikes)
    activity = np.zeros((times.size, cells), dtype=float)

    for cell_idex, spike_train in enumerate(spikes):
        for time_idx, time in enumerate(times):
            valid_spikes = ((spike_train >= time-window_size/2) & (spike_train < time + window_size/2)).sum()
            activity[time_idx, cell_idex] = valid_spikes / (window_size *1e-3)

    return activity

def activity_from_spikes_alpha_window(spikes: list[np.ndarray], times: np.ndarray, alpha_tau: float, method: str="numpy"):
    """
    Calculate the mean activity from population spikes using an alpha window.
    The alpha window is a function of time that decays exponentially.
    
    Parameters
    ----------
    spikes : list of np.ndarray of float, [ms]
        spike times for each neuron in the network, 
        first list indexes the neuron and the second indexes the spike times
    times : 1D np.ndarray, [ms]
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

def _decay(h, tau: float):
    """exp(-h / tau); tau == 0 means instantaneous decay (returns 0)."""
    h = np.asarray(h, dtype=float)
    if tau == 0:
        return np.zeros_like(h)
    return np.exp(-h / tau)


def _tsodyks_propagators(h, tau_rec: float, tau_fac: float, tau_psc: float):
    """Exact propagators of NEST's `tsodyks_synapse` over an interval h (same formulas as NEST)."""
    Puu = _decay(h, tau_fac)
    Pyy = _decay(h, tau_psc)
    Pzz = _decay(h, tau_rec)
    Pxy = ((Pzz - 1.0) * tau_rec - (Pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)
    Pxz = 1.0 - Pzz
    return Puu, Pyy, Pxy, Pxz


def reconstruct_stp_dynamics(spike_times: list, U: float, tau_rec: float, tau_fac: float, tau_psc: float, times: np.ndarray):
    """
    Reconstructs offline the state of NEST's `tsodyks_synapse` driven by presynaptic spike trains.

    Follows NEST's update exactly (Tsodyks et al. 1998 three-state model; resources x -> y -> z -> x).
    At each spike, after propagating the state over the inter-spike interval h:

        z = 1 - x - y
        u <- u * exp(-h/tau_fac)          (u decays to 0; tau_fac = 0 means u = 0 before the jump)
        x <- x + Pxy*y + Pxz*z
        y <- y * exp(-h/tau_psc)
        u <- u + U*(1 - u)
        x <- x - u*x ;  y <- y + u*x

    The initial state is u = 0, x = 1, y = 0 (NEST defaults, last spike at t = 0).

    Parameters
    ----------
    spike_times : list of array-like
        Presynaptic spike times [ms], one array per presynaptic neuron.
    U, tau_rec, tau_fac, tau_psc : float
        `tsodyks_synapse` parameters (times in [ms]); tau_rec must differ from tau_psc (as in NEST).
    times : np.ndarray
        Time grid [ms] on which the state is returned.

    Returns
    -------
    u, x, y : np.ndarray
        Arrays of shape (len(times), len(spike_times)):
        - u(t): utilisation a spike at time t would use, u_dec(t) + U*(1 - u_dec(t)), with u_dec
          decaying from the last post-spike value. u(t)*x(t)*weight is the efficacy of a spike at t,
          which corresponds to the MF models' X * u.
        - x(t): available resources; y(t): active (released) resources.
    """
    if tau_rec == tau_psc:
        raise ValueError("tsodyks_synapse requires tau_rec != tau_psc (singular propagator, as in NEST).")

    times = np.asarray(times, dtype=float)
    neurons_num = len(spike_times)
    u_all = np.empty((times.size, neurons_num))
    x_all = np.empty((times.size, neurons_num))
    y_all = np.empty((times.size, neurons_num))

    for neuron_idx, neuron_spike_times in enumerate(spike_times):
        neuron_spike_times = np.asarray(neuron_spike_times, dtype=float)
        n_spikes = neuron_spike_times.size

        # Post-spike states; index 0 is the initial state at t = 0.
        t_states = np.insert(neuron_spike_times, 0, 0.0)
        u_states = np.zeros(n_spikes + 1)
        x_states = np.ones(n_spikes + 1)
        y_states = np.zeros(n_spikes + 1)

        for k in range(1, n_spikes + 1):
            h = t_states[k] - t_states[k - 1]
            Puu, Pyy, Pxy, Pxz = _tsodyks_propagators(h, tau_rec, tau_fac, tau_psc)
            u, x, y = u_states[k - 1], x_states[k - 1], y_states[k - 1]
            z = 1.0 - x - y

            u = u * Puu
            x = x + Pxy * y + Pxz * z
            y = y * Pyy
            u = u + U * (1.0 - u)
            delta = u * x
            u_states[k], x_states[k], y_states[k] = u, x - delta, y + delta

        # Propagate the last post-spike state to every point of the time grid.
        idx = np.clip(np.searchsorted(t_states, times, side="right") - 1, 0, None)
        h = times - t_states[idx]
        Puu, Pyy, Pxy, Pxz = _tsodyks_propagators(h, tau_rec, tau_fac, tau_psc)
        x_last, y_last = x_states[idx], y_states[idx]

        u_decayed = u_states[idx] * Puu
        u_all[:, neuron_idx] = u_decayed + U * (1.0 - u_decayed)
        x_all[:, neuron_idx] = x_last + Pxy * y_last + Pxz * (1.0 - x_last - y_last)
        y_all[:, neuron_idx] = y_last * Pyy

    return u_all, x_all, y_all