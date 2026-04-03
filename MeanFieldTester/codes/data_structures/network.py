"""
This module defines the data structure for storing network results.

It defines separate classes for results from SNN and mean-field simulations.

"""


import codes.data_structures.base as base
import codes.utils.snn_helpers as snn_utils
import numpy as np



class SNNResults(base.NetworkResults):
    """Meta class for storing SNN results."""
    smoothing_options = {
        "histogram": snn_utils.activity_from_spikes_histogram,
        "sliding_window": snn_utils.activity_from_spikes_sliding_window,
        "alpha_window": snn_utils.activity_from_spikes_alpha_window
    }
    
    def __init__(self, times, stim_params, net_params, smoothing_function="histogram", smoothing_constant=base.BIN_SIZE, smoothing_kwargs=None):
        super().__init__(times, stim_params, net_params)
        self.set_smoothing_function(smoothing_function, smoothing_constant, **(smoothing_kwargs or {}))

    def set_smoothing_function(self, smoothing_function:str, smoothing_constant:float, **kwargs):
        if smoothing_function not in self.smoothing_options:
            raise ValueError(f"Unknown smoothing function: {smoothing_function}. "
                             f"Available options: {list(self.smoothing_options.keys())}")
        self.smoothing_setup = {
            'smoothing_function': smoothing_function,
            'smoothing_constant': smoothing_constant,
            'smoothing_kwargs': kwargs
        }
        function = self.smoothing_options[smoothing_function]
        match smoothing_function:
            case "histogram":
                self._smoothing_function = lambda x: function(x, self.times, bin_size=smoothing_constant, **kwargs)
            case "sliding_window":
                self._smoothing_function = lambda x: function(x, self.times, window_size=smoothing_constant, **kwargs)
            case "alpha_window":
                self._smoothing_function = lambda x: function(x, self.times, alpha_tau=smoothing_constant, **kwargs)

        # every time we set a new smoothing function, we reset the cached rates
        self._exc_rate_mean = None
        self._inh_rate_mean = None

    @property
    def exc_rate_mean(self):
        """Get the excitatory firing rate."""
        if self._exc_rate_mean is None:
            self._exc_rate_mean = self._smoothing_function(self.exc_spikes_all)
        return self._exc_rate_mean

    @property
    def inh_rate_mean(self):
        """Get the inhibitory firing rate."""
        if self._inh_rate_mean is None:
            self._inh_rate_mean = self._smoothing_function(self.inh_spikes_all)
        return self._inh_rate_mean


class SNNFullResults(SNNResults):
    def __init__(self, exc_neurons, inh_neurons, drive_rate, 
                 stim_rate, stim_params, net_params, 
                 smoothing_function="histogram", 
                 smoothing_constant=base.BIN_SIZE, 
                 smoothing_kwargs=None):
        super().__init__(exc_neurons.filter(name='v')[0].times.magnitude, stim_params, net_params, 
                         smoothing_function=smoothing_function, 
                         smoothing_constant=smoothing_constant, 
                         smoothing_kwargs=smoothing_kwargs)

        # list of lists with spike times
        self.exc_spikes_all = exc_neurons.spiketrains
        self.inh_spikes_all = inh_neurons.spiketrains

        # ndarray, shape = (time, neuron)
        self.exc_voltage_all = exc_neurons.filter(name='v')[0].magnitude
        self.inh_voltage_all = inh_neurons.filter(name='v')[0].magnitude
        self.exc_adaptation_all = exc_neurons.filter(name='w')[0].magnitude*1e3  # convert to pA

        # FromTo
        self.ee_conductance_all = exc_neurons.filter(name='gsyn_exc')[0].magnitude
        self.ei_conductance_all = inh_neurons.filter(name='gsyn_exc')[0].magnitude
        self.ie_conductance_all = exc_neurons.filter(name='gsyn_inh')[0].magnitude
        self.ii_conductance_all = inh_neurons.filter(name='gsyn_inh')[0].magnitude

        # ndarray, shape = (time)
        self.drive_rate = drive_rate
        self.stim_rate = stim_rate

    # Following properties are neuron-averaged values
    @property
    def exc_voltage_mean(self):
        return self.exc_voltage_all.mean(axis=1) 

    @property
    def exc_voltage_std(self):
        return self.exc_voltage_all.std(axis=1)

    @property
    def inh_voltage_mean(self):
        return self.inh_voltage_all.mean(axis=1) 

    @property
    def inh_voltage_std(self):
        return self.inh_voltage_all.std(axis=1)

    @property
    def exc_adaptation_mean(self):
        return self.exc_adaptation_all.mean(axis=1) 

    @property
    def exc_adaptation_std(self):
        return self.exc_adaptation_all.std(axis=1)

    @property
    def ee_conductance_mean(self):
        return self.ee_conductance_all.mean(axis=1) 

    @property
    def ee_conductance_std(self):
        return self.ee_conductance_all.std(axis=1)

    @property
    def ei_conductance_mean(self):
        return self.ei_conductance_all.mean(axis=1) 

    @property
    def ei_conductance_std(self):
        return self.ei_conductance_all.std(axis=1)

    @property
    def ie_conductance_mean(self):
        return self.ie_conductance_all.mean(axis=1) 

    @property
    def ie_conductance_std(self):
        return self.ie_conductance_all.std(axis=1)

    @property
    def ii_conductance_mean(self):
        return self.ii_conductance_all.mean(axis=1) 

    @property
    def ii_conductance_std(self):
        return self.ii_conductance_all.std(axis=1)


    def per_cell_average_rates(self, start_time=0, end_time=None):
        """Calculate the average firing rates for each excitatory and inhibitory cell.

            start_time (float): Time (in milliseconds) to start counting spikes. 
                    Defaults to the constant START_TIME.

            tuple: A tuple containing two numpy arrays:
            - exc_rates (1D numpy.ndarray): Average firing rates of excitatory cells in Hz.
            - inh_rates (1D numpy.ndarray): Average firing rates of inhibitory cells in Hz.

        Notes:
            - The averaging window is calculated as the simulation duration minus the start time.
            - Spike counts are converted to firing rates by dividing by the averaging window 
              (in milliseconds) and then converting to Hz.
        """
        if (end_time is None) or (end_time > self.times[-1]):
            end_time = self.times[-1]
        averaging_window = (end_time - start_time) / 1000  # in seconds
        exc_rates = snn_utils.spike_counts(self.exc_spikes_all, start_time=start_time, end_time=end_time) / averaging_window 
        inh_rates = snn_utils.spike_counts(self.inh_spikes_all, start_time=start_time, end_time=end_time) / averaging_window
        return exc_rates, inh_rates

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

    def print_time_averaged(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.times[0]
        if end_time is None:
            end_time = self.times[-1]
        mask = (self.times >= start_time) & (self.times <= end_time)

        print("="*50)
        print("SNN NETWORK TIME-AVERAGED RESULTS")
        print("="*50)
        print(f"Time window: [{start_time:.2f} ms;{end_time:.2f} ms]")
        print(f'Smoothing function: {self.smoothing_setup["smoothing_function"]}')

        exc_rates_naive, inh_rates_naive = self.per_cell_average_rates(start_time=start_time, end_time=end_time)
        exc_rates_smoothed = self.exc_rate_mean[mask]
        inh_rates_smoothed = self.inh_rate_mean[mask]

        print(f"exc rate = {exc_rates_naive.mean():.2f} +- {exc_rates_naive.std():.2f} Hz")
        print(f"exc rate (smoothed) = {exc_rates_smoothed.mean():.2f} Hz")
        print(f"inh rate = {inh_rates_naive.mean():.2f} +- {inh_rates_naive.std():.2f} Hz")
        print(f"inh rate (smoothed) = {inh_rates_smoothed.mean():.2f} Hz")

        snn_exc_voltage = self.exc_voltage_all[mask].mean(axis=0)
        snn_exc_adaptation = self.exc_adaptation_all[mask].mean(axis=0)*1e3
        snn_inh_voltage = self.inh_voltage_all[mask].mean(axis=0)

        print(f"exc w = {snn_exc_adaptation.mean():.2f} +- {snn_exc_adaptation.std():.2f} pA")
        print(f"exc V_m = {snn_exc_voltage.mean():.2f} +- {snn_exc_voltage.std():.2f} Hz")
        print(f"inh V_m = {snn_inh_voltage.mean():.2f} +- {snn_inh_voltage.std():.2f} Hz")

        print("="*50)


class MFResults(base.NetworkResults):
    def __init__(self, mf_results, stim_params, net_params):
        # Assuming mf_results are from TVB simulation run, the following does
        # conversion to appropriate units

        super().__init__(mf_results["time"], stim_params, net_params)

        self.drive_rate = mf_results["drive_rate"].flatten()*1e3
        self.stim_rate = mf_results["stimulus"].flatten()*1e3

        # shape = (time, node)
        self.exc_rate_mean = mf_results["E"].flatten()*1e3
        self.inh_rate_mean = mf_results["I"].flatten()*1e3
        self.exc_adaptation_mean = mf_results["W_e"].flatten()
        self.inh_adaptation_mean = np.zeros_like(self.exc_adaptation_mean)
        
        try:
            self.exc_rate_std = np.sqrt(mf_results["C_ee"]).flatten()*1e3
            self.inh_rate_std = np.sqrt(mf_results["C_ii"]).flatten()*1e3
            self.covariance = mf_results["C_ei"].flatten()*1e6
        except KeyError:
            print("Warning: MFResults does not contain C_ee, C_ii, C_ei (First order model). Skipping standard deviations and covariance.")
            self.exc_rate_std = None
            self.inh_rate_std = None
            self.covariance = None

        try:
            self.exc_x_mean = mf_results["X_e"].flatten()
            self.exc_y_mean = mf_results["Y_e"].flatten()
            self.exc_u_mean = mf_results["U_dyn_e"].flatten()
            self.inh_x_mean = mf_results["X_i"].flatten()
            self.inh_y_mean = mf_results["Y_i"].flatten()
            self.inh_u_mean = mf_results["U_dyn_i"].flatten()

        except KeyError:
            print("Warning: MFResults does not contain X_e, Y_e, X_i, Y_i. Skipping spatial coordinates.")
            self.exc_x_mean = None
            self.exc_y_mean = None
            self.exc_u_mean = None
            self.inh_x_mean = None
            self.inh_y_mean = None
            self.inh_u_mean = None
        
        self.exc_voltage_mean = None
        self.inh_voltage_mean = None

        self.ee_conductance_mean = None
        self.ei_conductance_mean = None
        self.ie_conductance_mean = None
        self.ii_conductance_mean = None

    @property
    def exc_voltage_mean(self):
        if self._exc_voltage_mean is None or not hasattr(self, '_exc_voltage_mean'):
            self.exc_voltage_mean = self.calculate_v_mean_tvb(
                                            self.net_params['exc_neuron'], 
                                            self.stim_params,
                                            self.net_params['network'],
                                            self.exc_rate_mean, 
                                            self.inh_rate_mean, 
                                            self.stim_rate, 
                                            self.exc_adaptation_mean)

        return self._exc_voltage_mean
    
    @exc_voltage_mean.setter
    def exc_voltage_mean(self, value):
        self._exc_voltage_mean = value

    @property
    def inh_voltage_mean(self):
        if self._inh_voltage_mean is None or not hasattr(self, '_inh_voltage_mean'):
            self._inh_voltage_mean = self.calculate_v_mean_tvb(
                                            self.net_params['inh_neuron'], 
                                            self.stim_params,
                                            self.net_params['network'],
                                            self.exc_rate_mean, 
                                            self.inh_rate_mean, 
                                            self.stim_rate*self.stim_params['stim_target_ratio'], 
                                            w=0.)
        return self._inh_voltage_mean

    @inh_voltage_mean.setter
    def inh_voltage_mean(self, value):
        self._inh_voltage_mean = value

    @staticmethod
    def calculate_v_mean_tvb(neuron_params, stim_params, conn_params, exc_rate, inh_rate, stim_rate, w):
        """
        Calculate the mean membrane potential for a given neuron type.
        
        This is a placeholder function, as the actual implementation depends on the
        specific neuron model and parameters.
        """
        from codes.tvb_models.models import Zerlaut_adaptation_first_order
        Fe_ext = stim_params["drive_rate"] + stim_rate
        Fi_ext = 0.

        v_mean, *_ = Zerlaut_adaptation_first_order.get_fluct_regime_vars(
                        exc_rate*1e-3, 
                        inh_rate*1e-3, 
                        Fe_ext*1e-3, 
                        Fi_ext*1e-3, 
                        w, 
                        neuron_params['exc_synapses']['syn_params']['weight'], 
                        neuron_params["neuron_params"]["tau_syn_E"], 
                        neuron_params["neuron_params"]["e_rev_E"], 
                        neuron_params['inh_synapses']['syn_params']['weight'], 
                        neuron_params["neuron_params"]["tau_syn_I"],
                        neuron_params["neuron_params"]["e_rev_I"], 
                        neuron_params["neuron_params"]['cm']/ neuron_params["neuron_params"]['tau_m']*1e3,
                        neuron_params["neuron_params"]['cm']*1e3,
                        neuron_params["neuron_params"]['v_rest'],
                        conn_params['total_pop_size'], 
                        conn_params['p_connect_exc'],
                        conn_params['p_connect_inh'], 
                        conn_params['g'], 
                        int(conn_params["drive_pop_size"]*conn_params["p_connect_drive"]), 
                        0)
        return v_mean

    @staticmethod
    def calculate_v_mean_tf(neuron_params, stim_params, conn_params, exc_rate, inh_rate, stim_rate, w):
        import codes.transfer_function as tf
        Fe_ext = stim_params["drive_rate"] + stim_rate
        Fi_ext = 0.
        v_mean, *_ = tf.MPF_with_adaptation(neuron_params)(
                        (exc_rate+Fe_ext),
                        (inh_rate+Fi_ext),
                        w*1e-3, 
                        flattened=True)
        return v_mean

    def print_time_averaged(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.times[0]
        if end_time is None:
            end_time = self.times[-1]
        mask = (self.times >= start_time) & (self.times <= end_time)

        print("="*50)
        print("MF NETWORK TIME-AVERAGED RESULTS")
        print("="*50)
        print(f"Time window: [{start_time:.2f} ms;{end_time:.2f} ms]")

        print(f"exc rate = {self.exc_rate_mean[mask].mean():.2f} +- {self.exc_rate_std[mask].mean():.2f} Hz")
        print(f"inh rate = {self.inh_rate_mean[mask].mean():.2f} +- {self.inh_rate_std[mask].mean():.2f} Hz")
        print(f"exc w = {self.exc_adaptation_mean[mask].mean():.2f} pA")
        print(f"inh w = {self.inh_adaptation_mean[mask].mean():.2f} pA")

        exc_vv = self.calculate_v_mean_tf(self.net_params["exc_neuron"], self.stim_params, self.net_params["network"], self.exc_rate_mean, self.inh_rate_mean, self.stim_rate, self.exc_adaptation_mean)
        inh_vv = self.calculate_v_mean_tf(self.net_params["inh_neuron"], self.stim_params, self.net_params["network"], self.exc_rate_mean, self.inh_rate_mean, self.stim_rate, 0.)

        print(f"exc V_m = {self.exc_voltage_mean[mask].mean():.2f} +- {self.exc_voltage_mean[mask].std():.2f} mV")
        print(f"inh V_m = {self.inh_voltage_mean[mask].mean():.2f} +- {self.inh_voltage_mean[mask].std():.2f} mV")
        print(f"My exc V_m = {exc_vv[mask].mean():.2f} +- {exc_vv[mask].std():.2f} mV")
        print(f"My inh V_m = {inh_vv[mask].mean():.2f} +- {inh_vv[mask].std():.2f} mV")        


