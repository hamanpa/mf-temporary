import numpy as np

def calc_per_cell_rates(spikes: list[np.ndarray], start_time: float, end_time: float) -> np.ndarray:
    """
    Calculates the average firing rate for each cell in the given time window.

    spikes, start_time, and end_time are in milliseconds. The firing rate is returned in Hz.
    """
    duration_s = (end_time - start_time) / 1000.0
    rates = []
    
    for spike_times in spikes:
        spike_times = np.array(spike_times)  # just for safety, in case spike_times is not a numpy array

        valid_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
        rates.append(len(valid_spikes) / duration_s)
        
    return np.array(rates)





# following are computations that were part of the SNNResults class, 
# but are now moved here for clarity and reusability
    def calculate_synchrony(self, population:str|list[str], start_time=0, end_time=None, spikes_threshold=5, time_bin=10):
        """Calculate the synchrony measure for the excitatory population.

        This method calculates synchrony based on pairwise correlations of spike trains.

        
        
        Parameters
        ----------
        population (str or list of str): Population(s) for which to calculate synchrony. 
            Can be 'exc', 'inh', or a list containing any combination of these.
        start_time (float): Time (in milliseconds) to start counting spikes. 
            Defaults to the constant START_TIME.
        end_time (float): Time (in milliseconds) to end counting spikes. 
            Defaults to the end of the simulation.
        spikes_threshold (int): Minimum number of spikes within the spike train to consider for statistics.

        Returns
        -------
        float or list of float: Synchrony measure(s) for the specified population(s).
        """
        if spikes_threshold < 2:
            raise ValueError("spikes_threshold must be at least 2 to calculate ISI.")

        if (end_time is None) or (end_time > self.times[-1]):
            end_time = self.times[-1]
        if start_time < self.times[0]:
            start_time = self.times[0]

        num_bins = int(round((end_time - start_time)/time_bin))
        r = start_time, end_time

        if isinstance(population, str):
            population = [population]
            unpack = True
        else:
            unpack = False

        synchrony = []
        for pop in population:
            if pop.lower() == "exc":
                population_spiketrains = self.exc_spikes_all
            elif pop.lower() == "inh":
                population_spiketrains = self.inh_spikes_all
            else:
                raise ValueError(f"Unknown population: {pop}. Valid options are 'exc' and 'inh'.")

            selected_spiketrains = []
            for spiketrain in population_spiketrains:
                spikes = []
                for spike in spiketrain:
                    if start_time <= spike <= end_time:
                        spikes.append(spike)
                selected_spiketrains.append(np.array(spikes))

            psths = [np.histogram(spikes, bins=num_bins, range=r)[0] for spikes in selected_spiketrains if len(spikes) >= spikes_threshold]
            corrs = np.nan_to_num(np.corrcoef(np.squeeze(psths)))
            synchrony.append(np.mean(corrs[np.triu_indices(len(psths), 1)]))

        if unpack:
            return synchrony[0]
        return synchrony

    def calculate_regularity(self, population:str|list[str], start_time=0, end_time=None, spikes_threshold=5):
        """Calculate the regularity measure for the excitatory population.

        This method calculates regularity based on the coefficient of variation (CV)
        of inter-spike intervals (ISIs) for each neuron in the excitatory population.

        values close to 0 -> regular firing
        values close to 1 -> Poisson firing (irregular - independent)
        values > 1 -> bursty firing

        Parameters
        ----------
        population (str or list of str): Population(s) for which to calculate regularity. 
            Can be 'exc', 'inh', or a list containing any combination of these.
        start_time (float): Time (in milliseconds) to start counting spikes. 
            Defaults to the constant START_TIME.
        end_time (float): Time (in milliseconds) to end counting spikes. 
            Defaults to the end of the simulation.
        spikes_threshold (int): Minimum number of spikes within the spike train to consider for statistics.

        Returns
        -------
        float or list of float: Regularity measure(s) for the specified population(s).
        """
        if spikes_threshold < 2:
            raise ValueError("spikes_threshold must be at least 2 to calculate ISI.")

        if (end_time is None) or (end_time > self.times[-1]):
            end_time = self.times[-1] 
        if start_time < self.times[0]:
            start_time = self.times[0]

        if isinstance(population, str):
            population = [population]
            unpack = True
        else:
            unpack = False

        regularity = []

        for pop in population:
            if pop.lower() == "exc":
                population_spiketrains = self.exc_spikes_all
            elif pop.lower() == "inh":
                population_spiketrains = self.inh_spikes_all
            else:
                raise ValueError(f"Unknown population: {pop}. Valid options are 'exc' and 'inh'.")

            selected_spiketrains = []
            for spiketrain in population_spiketrains:
                spikes = []
                for spike in spiketrain:
                    if start_time <= spike <= end_time:
                        spikes.append(spike)
                selected_spiketrains.append(np.array(spikes))
            isis = [np.diff(spikes) for spikes in selected_spiketrains if len(spikes) >= spikes_threshold]
            cvs = [np.std(isi) / np.mean(isi) for isi in isis]
            regularity.append(np.mean(cvs))                

        if unpack:
            return regularity[0]
        return regularity