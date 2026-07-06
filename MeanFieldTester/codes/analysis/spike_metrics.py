import numpy as np

def calc_per_cell_rates(spikes: list[np.ndarray], start_time: float, end_time: float) -> np.ndarray:
    """
    Calculates the average firing rate for each cell in the given time window.
    """
    duration_s = (end_time - start_time) / 1000.0
    rates = []
    
    for spike_times in spikes:
        spike_times = np.array(spike_times)  # just for safety, in case spike_times is not a numpy array

        valid_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
        rates.append(len(valid_spikes) / duration_s)
        
    return np.array(rates)


