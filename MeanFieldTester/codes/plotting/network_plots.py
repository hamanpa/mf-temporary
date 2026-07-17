
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
        'x_unit': 'ms',
        'y_unit': 'Hz',
        'xlabel': 'Time',
        'ylabel': 'Firing Rate',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        x_unit = self.full_params['x_unit']
        y_unit = self.full_params['y_unit']

        for results, ls, label in self.iter_results(network_results_list):
            self.plot_pair_series(
                ax,
                results,
                ls,
                label,
                exc_getter=lambda result: result.exc_rate_mean(y_unit),
                inh_getter=lambda result: result.inh_rate_mean(y_unit),
                exc_color=self.full_params['exc_color'],
                inh_color=self.full_params['inh_color'],
                alpha=0.5 if isinstance(results, BaseSNNResults) else 1.0,
                exc_std_getter=lambda result: result.exc_rate_std(y_unit),
                inh_std_getter=lambda result: result.inh_rate_std(y_unit),
            )

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
        print("WARNING: FiringRateAndStimulusPlot does not have implemented unit handling, thus it is assumed that the rates are in Hz and time in ms.")
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

        for results, ls, label in self.iter_results(network_results_list):
            self.plot_single_series(
                ax_lower,
                results,
                ls,
                label,
                getter=lambda result: result.drive_rate_mean() + result.stim_rate_mean(),
                color='black',
            )
            # NOTE: for some reason the loop take ridiculous amount of time,
            # so we plot only the first one
            break

        self.apply_postplot_params(ax_lower, self.full_params | self.LOWER_PLOT_PARAMS)

class StimulusPlot(BaseNetworkPlot):
    """Plot the stimulus over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Stimulus',
        'x_unit': 'ms',
        'y_unit': 'Hz',
        'xlabel': 'Time',
        'ylabel': 'Rate',
        'stim_linestyle': '--',
        'drive_linestyle': ':',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        y_unit = self.full_params['y_unit']

        results = network_results_list[0]
        times = results.times(x_unit)
        stimulus_rate = results.stim_rate_mean(y_unit)
        drive_rate = results.drive_rate_mean(y_unit)

        ax.plot(times, stimulus_rate, self.full_params['stim_linestyle'], color='black', label='Stimulus rate')
        ax.plot(times, drive_rate, self.full_params['drive_linestyle'], color='black', label='Drive rate')


class AdaptationPlot(BaseNetworkPlot):
    """Plot the adaptation current over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Exc Adaptation',
        'x_unit': 'ms',
        'y_unit': 'nA',
        'xlabel': 'Time',
        'ylabel': 'Adaptation',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        self.update_params(network_results_list)

        x_unit = self.full_params['x_unit']
        y_unit = self.full_params['y_unit']

        for results, ls, label in zip(network_results_list, self.full_params['linestyles'], self.full_params['labels']):
            ax.plot(results.times(x_unit), results.exc_adaptation_mean(y_unit), label=label, ls=ls, color='blue')

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

        print("WARNING: StimulusWithAdaptationPlot does not have implemented unit handling! Unit may be wrong in the plot.")

        axtwin = ax.twinx()  

        self.plot_single_series(
            ax,
            network_results_list[0],
            '--',
            'Drive rate',
            getter=lambda result: result.drive_rate_mean(),
            color='black',
        )
        self.plot_single_series(
            ax,
            network_results_list[0],
            '-.',
            'Stimulus rate',
            getter=lambda result: result.stim_rate_mean(),
            color='black',
        )
        for results, ls, label in self.iter_results(network_results_list):
            self.plot_single_series(
                axtwin,
                results,
                ls,
                label,
                getter=lambda result: result.exc_adaptation_mean(),
                color='blue',
            )

        axtwin.set_ylabel("Adaptation current (nA)", color="blue")
        axtwin.tick_params(axis ='y', labelcolor = "blue")
        if self.full_params['legend'] is True:
            axtwin.legend()


class VoltagePlot(BaseNetworkPlot):
    """Plot the voltage of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Mean Membrane Potential',
        'x_unit': 'ms',
        'y_unit': 'mV',
        'xlabel': 'Time',
        'ylabel': 'Membrane potential',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        y_unit = self.full_params['y_unit']
            
        for results, ls, label in self.iter_results(network_results_list):
            self.plot_pair_series(
                ax,
                results,
                ls,
                label,
                exc_getter=lambda result: result.exc_voltage_mean(y_unit),
                inh_getter=lambda result: result.inh_voltage_mean(y_unit),
                exc_color=self.full_params['exc_color'],
                inh_color=self.full_params['inh_color'],
            )

class STPVariableXPlot(BaseNetworkPlot):
    """Plot the STP variable x of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Mean STP Variable x',
        'x_unit': 'ms',
        'y_unit': None,
        'xlabel': 'Time',
        'ylabel': 'STP variable x',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        y_unit = self.full_params['y_unit']

        for results, ls, label in self.iter_results(network_results_list):
            self.plot_pair_series(
                ax,
                results,
                ls,
                label,
                exc_getter=lambda result: result.exc_x_mean(y_unit),
                inh_getter=lambda result: result.inh_x_mean(y_unit),
                exc_color=self.full_params['exc_color'],
                inh_color=self.full_params['inh_color'],
            )

class STPVariableUPlot(BaseNetworkPlot):
    """Plot the STP variable u of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Mean STP Variable u',
        'x_unit': 'ms',
        'y_unit': None,
        'xlabel': 'Time',
        'ylabel': 'STP variable u',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        y_unit = self.full_params['y_unit']

        for results, ls, label in self.iter_results(network_results_list):
            self.plot_pair_series(
                ax,
                results,
                ls,
                label,
                exc_getter=lambda result: result.exc_u_mean(y_unit),
                inh_getter=lambda result: result.inh_u_mean(y_unit),
                exc_color=self.full_params['exc_color'],
                inh_color=self.full_params['inh_color'],
            )



class FiringRateHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the firing rate histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Firing Rate Histogram',
        'xlabel': 'Firing Rate',
        'x_unit': 'Hz',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("FiringRateHistogramPlot only works for no stimulus simulations.")
            
            if isinstance(results, BaseSNNResults):
                exc_rates = calc_per_cell_rates(
                    results.exc_spikes_all("ms"), 
                    start_time=results._get_scaled(self.full_params['start_time'], time_unit, "ms"), 
                    end_time=results._get_scaled(self.full_params['end_time'],time_unit, "ms")
                )
                inh_rates = calc_per_cell_rates(
                    results.inh_spikes_all("ms"), 
                    start_time=results._get_scaled(self.full_params['start_time'], time_unit, "ms"), 
                    end_time=results._get_scaled(self.full_params['end_time'], time_unit, "ms")
                )
                exc_neuron_count = len(results.exc_spikes_all("ms"))
                inh_neuron_count = len(results.inh_spikes_all("ms"))

                exc_rates = results._get_scaled(exc_rates, "Hz", x_unit)
                inh_rates = results._get_scaled(inh_rates, "Hz", x_unit)

                self.plot_hist_pair(
                    ax,
                    exc_rates,
                    inh_rates,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                    exc_weights=np.ones(exc_neuron_count) / exc_neuron_count if self.full_params['normalization'] else None,
                    inh_weights=np.ones(inh_neuron_count) / inh_neuron_count if self.full_params['normalization'] else None,
                    density=self.full_params['density'],
                )

            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.exc_rate_mean(x_unit)[mask])
                inh_mean = np.mean(results.inh_rate_mean(x_unit)[mask])
                if ((results.exc_rate_std() is not None) 
                            and (results.exc_rate_std().size > 0)
                            and (results.inh_rate_std() is not None) 
                            and (results.inh_rate_std().size > 0)):
                    self.plot_mf_hist_pair(
                        ax,
                        exc_mean,
                        inh_mean,
                        np.mean(results.exc_rate_std(x_unit)[mask]),
                        np.mean(results.inh_rate_std(x_unit)[mask]),
                        label,
                        ls,
                        exc_color=self.full_params['exc_color'],
                        inh_color=self.full_params['inh_color'],
                        normalization=self.full_params['normalization'],
                        density=self.full_params['density'],
                    )
                else:
                    self.plot_hist_lines(
                        ax,
                        exc_mean,
                        inh_mean,
                        label,
                        ls,
                        exc_color=self.full_params['exc_color'],
                        inh_color=self.full_params['inh_color'],
                    )

class VoltageHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the voltage histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Voltage Histogram',
        'xlabel': 'Membrane potential',
        'x_unit': 'mV',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("VoltageHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])

                exc_voltage = results.exc_voltage_all(x_unit)[mask].mean(axis=0)
                inh_voltage = results.inh_voltage_all(x_unit)[mask].mean(axis=0)

                self.plot_hist_pair(
                    ax,
                    exc_voltage,
                    inh_voltage,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )

            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.exc_voltage_mean(x_unit)[mask])
                inh_mean = np.mean(results.inh_voltage_mean(x_unit)[mask])
                self.plot_hist_lines(
                    ax,
                    exc_mean,
                    inh_mean,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )


class AdaptationHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the adaptation histogram of excitatory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Adaptation Histogram',
        'xlabel': 'Adaptation current',
        'x_unit': 'pA',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("AdaptationHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])

                exc_adaptation = results.exc_adaptation_all(x_unit)[mask].mean(axis=0)

                bin_edges = self.get_bin_edges([exc_adaptation])

                ax.hist(exc_adaptation, bins=bin_edges, alpha=0.5, label=f'Exc {label}', edgecolor='blue', color='blue', linestyle=ls)

            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.exc_adaptation_mean(x_unit)[mask])
                ax.axvline(exc_mean, label=f'Exc {label}', color='blue', linestyle=ls)

class ExcitatoryNeuronConductanceHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the conductance histogram of excitatory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Exc Neuron Conductances',
        'xlabel': 'Conductance',
        'x_unit': 'nS',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("ExcitatoryNeuronConductanceHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])

                exc_conductance = results.ee_conductance_all(x_unit)[mask].mean(axis=0)
                inh_conductance = results.ie_conductance_all(x_unit)[mask].mean(axis=0)

                self.plot_hist_pair(
                    ax,
                    exc_conductance,
                    inh_conductance,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )

            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.ee_conductance_mean(x_unit)[mask])
                inh_mean = np.mean(results.ie_conductance_mean(x_unit)[mask])
                self.plot_hist_lines(
                    ax,
                    exc_mean,
                    inh_mean,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )

class InhibitoryNeuronConductanceHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the conductance histogram of inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'Inh Neuron Conductances',
        'xlabel': 'Conductance',
        'x_unit': 'nS',
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("InhibitoryNeuronConductanceHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])

                exc_conductance = results.ei_conductance_all(x_unit)[mask].mean(axis=0)
                inh_conductance = results.ii_conductance_all(x_unit)[mask].mean(axis=0)

                self.plot_hist_pair(
                    ax,
                    exc_conductance,
                    inh_conductance,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )
            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.ei_conductance_mean(x_unit)[mask])
                inh_mean = np.mean(results.ii_conductance_mean(x_unit)[mask])
                self.plot_hist_lines(
                    ax,
                    exc_mean,
                    inh_mean,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )

class STPVariableXHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the STP variable x histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'STP Variable x Histogram',
        'xlabel': 'STP variable x',
        'x_unit': None,
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("STPVariableXHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])

                exc_x = results.exc_x_all(x_unit)[mask].mean(axis=0)
                inh_x = results.inh_x_all(x_unit)[mask].mean(axis=0)

                self.plot_hist_pair(
                    ax,
                    exc_x,
                    inh_x,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )

            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.exc_x_mean(x_unit)[mask])
                inh_mean = np.mean(results.inh_x_mean(x_unit)[mask])
                self.plot_hist_lines(
                    ax,
                    exc_mean,
                    inh_mean,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )
class STPVariableUHistogramPlot(BaseNetworkHistogramPlot):
    """Plot the STP variable u histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkHistogramPlot.DEFAULT_PARAMS,
        'title': 'STP Variable u Histogram',
        'xlabel': 'STP variable u',
        'x_unit': None,
    }

    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        x_unit = self.full_params['x_unit']
        time_unit = self.full_params['time_unit']

        for results, ls, label in self.iter_results(network_results_list):
            if results.stim_params.pattern != 'NoStimulus':
                raise ValueError("STPVariableUHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, BaseSNNResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])


                exc_u = results.exc_u_all(x_unit)[mask].mean(axis=0)
                inh_u = results.inh_u_all(x_unit)[mask].mean(axis=0)

                self.plot_hist_pair(
                    ax,
                    exc_u,
                    inh_u,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )
            elif isinstance(results, BaseMFResults):
                mask = (results.times(time_unit) >= self.full_params['start_time']) & (results.times(time_unit) <= self.full_params['end_time'])
                exc_mean = np.mean(results.exc_u_mean(x_unit)[mask])
                inh_mean = np.mean(results.inh_u_mean(x_unit)[mask])
                self.plot_hist_lines(
                    ax,
                    exc_mean,
                    inh_mean,
                    label,
                    ls,
                    exc_color=self.full_params['exc_color'],
                    inh_color=self.full_params['inh_color'],
                )