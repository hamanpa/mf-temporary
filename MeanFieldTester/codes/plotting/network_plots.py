
import numpy as np
from typing import List

from ..analysis.spike_metrics import calc_per_cell_rates
from ..data_structures.base import BaseMFResults, BaseSNNResults
from .base import BaseNetworkPlot, BaseNetworkHistogramPlot


class FiringRatePlot(BaseNetworkPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Firing Rate over Time',
        'xlabel': 'Time (ms)',
        'ylabel': 'Firing Rate (Hz)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            alpha = 0.5 if isinstance(results, BaseSNNResults) else 1.0
            ax.plot(results.times(), results.exc_rate_mean(), label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'], alpha=alpha)
            ax.plot(results.times(), results.inh_rate_mean(), label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'], alpha=alpha)
            if isinstance(results, BaseMFResults) and ls != 'None':
                ax.fill_between(results.times(), 
                                results.exc_rate_mean() - results.exc_rate_std(),
                                results.exc_rate_mean() + results.exc_rate_std(), 
                                color=self.full_params['exc_color'], alpha=0.3)
                ax.fill_between(results.times(), 
                                results.inh_rate_mean() - results.inh_rate_std(),
                                results.inh_rate_mean() + results.inh_rate_std(), 
                                color=self.full_params['inh_color'], alpha=0.3)


class FiringRateAndStimulusPlot(BaseNetworkPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time with stimulus."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Firing Rate and Stimulus over Time',
        'xlabel': 'Time (ms)',
        'ylabel': 'Firing Rate (Hz)',
        'height_ratios' : [5, 1],
        'hspace': 0.0,  
    }

    LOWER_PLOT_PARAMS = {
        'title': None, 
        'ylabel': None, 
        'legend': False, 
        'ylim': (None, None),
        'yticks': [], 
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)
 
        # Spliting the axis into two nested subplots
        gs_nested_fr_stim = ax.get_subplotspec().subgridspec(
                nrows=2, ncols=1,
                height_ratios=self.full_params['height_ratios'],
                hspace=self.full_params['hspace']
            )
        ax.set_visible(False) # Make the parent axis invisible

        # Create the two nested axes
        ax_upper = ax.figure.add_subplot(gs_nested_fr_stim[0, 0])
        ax_lower = ax.figure.add_subplot(gs_nested_fr_stim[1, 0], sharex=ax_upper)

        FiringRatePlot(self.full_params).draw(ax_upper, network_results_list)
        ax_upper.tick_params(axis='x', which='both', bottom=False, labelbottom=False)


        self.apply_preplot_params(ax_lower, self.full_params | self.LOWER_PLOT_PARAMS)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            ax_lower.plot(results.times(), results.drive_rate_mean() + results.stim_rate_mean(), label=label, ls=ls, color='black')
            # NOTE: for some reason the loop take ridiculous amount of time,
            # so we plot only the first one
            break

        self.apply_postplot_params(ax_lower, self.full_params | self.LOWER_PLOT_PARAMS)

class StimulusPlot(BaseNetworkPlot):
    """Plot the stimulus over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Stimulus',
        'xlabel': 'Time (ms)',
        'ylabel': 'Rate (Hz)',
        'stim_linestyle': '--',
        'drive_linestyle': ':',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        times = network_results_list[0].times()
        stimulus_rate = network_results_list[0].stim_rate_mean("Hz")
        drive_rate = network_results_list[0].drive_rate_mean("Hz")

        ax.plot(times, stimulus_rate, self.full_params['stim_linestyle'], color='black', label='Stimulus rate')
        ax.plot(times, drive_rate, self.full_params['drive_linestyle'], color='black', label='Drive rate')


class AdaptationPlot(BaseNetworkPlot):
    """Plot the adaptation current over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Exc Adaptation',
        'xlabel': 'Time (ms)',
        'ylabel': 'Adaptation (nA)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            ax.plot(results.times(), results.exc_adaptation_mean(), label=label, ls=ls, color='blue')

class StimulusWithAdaptationPlot(BaseNetworkPlot):
    """Plot the stimulus with adaptation over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Stimulus with Adaptation',
        'xlabel': 'Time (ms)',
        'ylabel': 'Rate (Hz)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        axtwin = ax.twinx()  

        ax.plot(network_results_list[0].times(), network_results_list[0].drive_rate_mean(), "--", color='black', label='Drive rate')
        ax.plot(network_results_list[0].times(), network_results_list[0].stim_rate_mean(), "-.", color='black', label='Stimulus rate')
        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            axtwin.plot(results.times(), results.exc_adaptation_mean(), ls=ls, color='blue', label=label)

        axtwin.set_ylabel("Adaptation current (nA)", color="blue")
        axtwin.tick_params(axis ='y', labelcolor = "blue")
        if self.full_params['legend'] is True:
            axtwin.legend()


class VoltagePlot(BaseNetworkPlot):
    """Plot the voltage of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Mean Membrane Potential',
        'xlabel': 'Time (ms)',
        'ylabel': 'Membrane potential (mV)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            ax.plot(results.times(), results.exc_voltage_mean(), label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'])
            ax.plot(results.times(), results.inh_voltage_mean(), label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'])

class STPVariableXPlot(BaseNetworkPlot):
    """Plot the STP variable x of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Mean STP Variable x',
        'xlabel': 'Time (ms)',
        'ylabel': 'STP variable x',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.exc_x_mean() is not None:
                ax.plot(results.times(), results.exc_x_mean(), label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'])
            if results.inh_x_mean() is not None:
                ax.plot(results.times(), results.inh_x_mean(), label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'])

class STPVariableUPlot(BaseNetworkPlot):
    """Plot the STP variable u of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Mean STP Variable u',
        'xlabel': 'Time (ms)',
        'ylabel': 'STP variable u',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)


        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.exc_u_mean() is not None:
                ax.plot(results.times(), results.exc_u_mean(), label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'])
            if results.inh_u_mean() is not None:
                ax.plot(results.times(), results.inh_u_mean(), label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'])



class FiringRateHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the firing rate histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Firing Rate Histogram',
        'xlabel': 'Firing Rate (Hz)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("FiringRateHistogramPlot only works for no stimulus simulations.")
            
            if isinstance(results, BaseSNNResults):
                exc_rates = calc_per_cell_rates(results.exc_spikes_all(), start_time=self.full_params['start_time'], end_time=self.full_params['end_time'])
                inh_rates = calc_per_cell_rates(results.inh_spikes_all(), start_time=self.full_params['start_time'], end_time=self.full_params['end_time'])
                exc_neuron_count = len(results.exc_spikes_all())
                inh_neuron_count = len(results.inh_spikes_all())


                bin_edges = self.get_bin_edges([exc_rates, inh_rates])

                ax.hist(exc_rates, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls, 
                        density=self.full_params['density'] ,
                        weights=np.ones(exc_neuron_count) / exc_neuron_count  if self.full_params['normalization'] else None  # Normalize by number of excitatory neurons
                )
                ax.hist(inh_rates, bins=bin_edges, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls, 
                        density=self.full_params['density'] ,
                        weights=np.ones(inh_neuron_count) / inh_neuron_count  if self.full_params['normalization'] else None  # Normalize by number of inhibitory neurons
                )

            elif isinstance(results, BaseMFResults):
                mask = results.times() >= self.full_params['start_time']
                exc_mean = np.mean(results.exc_rate_mean()[mask])
                inh_mean = np.mean(results.inh_rate_mean()[mask])
                if ((results.exc_rate_std() is not None) 
                            and (results.exc_rate_std().size > 0)
                            and (results.inh_rate_std() is not None) 
                            and (results.inh_rate_std().size > 0)):
                    exc_std = np.mean(results.exc_rate_std()[mask])
                    inh_std = np.mean(results.inh_rate_std()[mask])
                    
                    x = np.linspace(0, max(exc_mean + 4*exc_std, inh_mean + 4*inh_std), 100)
                    dx = x[1] - x[0]

                    exc_gauss = np.exp( -0.5 * ((x - exc_mean)/exc_std)**2)
                    if self.full_params['normalization']:
                        exc_gauss /= exc_gauss.max()
                    if self.full_params['density']:
                        exc_gauss /= exc_gauss.sum()*dx

                    inh_gauss = np.exp( -0.5 * ((x - inh_mean)/inh_std)**2)
                    if self.full_params['normalization']:
                        inh_gauss /= inh_gauss.max()
                    if self.full_params['density']:
                        inh_gauss /= inh_gauss.sum()*dx
                    
                    ax.plot(x, exc_gauss, label=f'Exc {label}', color=self.full_params['exc_color'], linestyle=ls)
                    ax.plot(x, inh_gauss, label=f'Inh {label}', color=self.full_params['inh_color'], linestyle=ls)
                else:
                    ax.axvline(exc_mean, label=f'Exc {label}', color=self.full_params['exc_color'], linestyle=ls)
                    ax.axvline(inh_mean, label=f'Inh {label}', color=self.full_params['inh_color'], linestyle=ls)

    def plot_bell_curve(self, ax, mean, std, color, linestyle):
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        y = np.exp( -0.5 * ((x - mean)/std)**2)
        ax.plot(x, y, color=color, linestyle=linestyle)

    def plot_mf_hist(self, ax):
        pass

class VoltageHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the voltage histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Voltage Histogram',
        'xlabel': 'Membrane potential (mV)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("VoltageHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times() >= self.full_params['start_time']) & (results.times() <= self.full_params['end_time'])

                exc_voltage = results._exc_voltage_all[mask].mean(axis=0)
                inh_voltage = results._inh_voltage_all[mask].mean(axis=0)
                
                bin_edges = self.get_bin_edges([exc_voltage, inh_voltage])

                ax.hist(exc_voltage, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)
                ax.hist(inh_voltage, bins=bin_edges, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)


class AdaptationHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the adaptation histogram of excitatory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Adaptation Histogram',
        'xlabel': 'Adaptation current (nA)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("AdaptationHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times() >= self.full_params['start_time']) & (results.times() <= self.full_params['end_time'])

                exc_adaptation = results._exc_adaptation_all[mask].mean(axis=0)

                bin_edges = self.get_bin_edges([exc_adaptation])

                ax.hist(exc_adaptation, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor='blue', color='blue', linestyle=ls)


class ExcitatoryNeuronConductanceHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the conductance histogram of excitatory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Exc Neuron Conductances',
        'xlabel': 'Conductance (nS)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("ExcitatoryNeuronConductanceHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times() >= self.full_params['start_time']) & (results.times() <= self.full_params['end_time'])

                exc_conductance = results._ee_conductance_all[mask].mean(axis=0)
                inh_conductance = results._ie_conductance_all[mask].mean(axis=0)

                bin_edges = self.get_bin_edges([exc_conductance, inh_conductance])

                ax.hist(exc_conductance, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)
                ax.hist(inh_conductance, bins=bin_edges, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)

class InhibitoryNeuronConductanceHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the conductance histogram of inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Inh Neuron Conductances',
        'xlabel': 'Conductance (nS)',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("InhibitoryNeuronConductanceHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times() >= self.full_params['start_time']) & (results.times() <= self.full_params['end_time'])

                exc_conductance = results._ei_conductance_all[mask].mean(axis=0)
                inh_conductance = results._ii_conductance_all[mask].mean(axis=0)

                bin_edges = self.get_bin_edges([exc_conductance, inh_conductance])

                ax.hist(exc_conductance, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)
                ax.hist(inh_conductance, bins=bin_edges, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)


class STPVariableXHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the STP variable x histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'STP Variable x Histogram',
        'xlabel': 'STP variable x',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("STPVariableXHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times() >= self.full_params['start_time']) & (results.times() <= self.full_params['end_time'])

                results.exc_x_mean() # call just to ensure that the exc_x_mean is computed and cached
                results.inh_x_mean() # call just to ensure that the inh_x_mean is computed and cached

                exc_x = results._exc_x_all[mask].mean(axis=0)
                inh_x = results._inh_x_all[mask].mean(axis=0)

                bin_edges = self.get_bin_edges([exc_x, inh_x])

                ax.hist(exc_x, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)
                ax.hist(inh_x, bins=bin_edges, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)


class STPVariableUHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the STP variable u histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'STP Variable u Histogram',
        'xlabel': 'STP variable u',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("STPVariableUHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times() >= self.full_params['start_time']) & (results.times() <= self.full_params['end_time'])

                results.exc_u_mean() # call just to ensure that the exc_u_mean is computed and cached
                results.inh_u_mean() # call just to ensure that the inh_u_mean is computed and cached

                exc_u = results._exc_u_all[mask].mean(axis=0)
                inh_u = results._inh_u_all[mask].mean(axis=0)

                bin_edges = self.get_bin_edges([exc_u, inh_u])

                ax.hist(exc_u, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)
                ax.hist(inh_u, bins=bin_edges, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)