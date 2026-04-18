"""
In this part we define the plotting functions that can be used to plot
each of the function takes an axis object and a list/class of results

Nomenclature for plotting:
    _and_ - two subplots 
    _with_ - one plot with two axes (twin ax on the right)
    _by_row - each row represents one neuron, MF instance, etc.
    _by_col - each column represents one neuron, MF instance, etc.

    draw_ - make predefined plot on a given axis
    plot_ - make predefined figure plot


Design notes:
    - each plot has it predefined parameters (beginning of the function)
    - global parameters are set in the beginning of the file
        - should be hinted in default_params

        
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import copy

from codes import transfer_function as tf

from codes.utils.list_helpers import indexed_linear_sample

from codes.data_structures.network import MFResults, SNNResults
from codes.data_structures.single_neuron import SingleNeuronResults

from codes.utils.snn_helpers import activity_from_spikes_histogram

LINESTYLES = ['-', '--', '-.', ':']
MAX_NTW_ACTIVITY = 200  # Hz
EXC_COLOR = "green"
INH_COLOR = "red"
RASTER_EXC_CELLS = 400
RASTER_INH_CELLS = 100

COMMON_FIG_PARAMS = {}
COMMON_PLOT_PARAMS = {}

BIN_SIZE = 5  # [ms], size for making histograms

NEURON_NAMES = ["exc_neuron", "inh_neuron"]


################################################################################
#                           META CLASSES                                       # 
################################################################################

class BasePlot:
    # Class-level default parameters for common axis settings
    DEFAULT_PARAMS = {
        'title': None,
        'xlabel': None,
        'ylabel': None,
        'xlim': (None, None),
        'ylim': (None, None),
        'yticks': None,  # None means default ticks
        'yticks_labels': None,  # None means default labels
        'xticks': None,  # None means default ticks
        'xticks_labels': None,  # None means default labels
        'legend': False,
        'grid': False,
        'xmargin': 0.05, # Default x-margin
        'ymargin': 0.05, # Default y-margin
        'exc_color': EXC_COLOR,  # Default color for excitatory neurons
        'inh_color': INH_COLOR,  # Default color for inhibitory neurons
    }

    def __init__(self, params=None):
        self.params = params or {}
        self.full_params = copy.deepcopy(self.DEFAULT_PARAMS)
        if params:
            self.full_params.update(params) # Instance params override defaults    

    def draw(self, ax, *data):
        self.apply_preplot_params(ax, self.full_params)
        im = self._draw(ax, *data)
        self.apply_postplot_params(ax, self.full_params)
        return im

    def _draw(self, ax, *data):
        """This method should be implemented in subclasses to draw the actual plot."""
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def apply_preplot_params(ax, params):
        """Apply parameters before drawing the plot."""
        ax.set_xmargin(params['xmargin'])
        ax.set_ymargin(params['ymargin'])
    
    @staticmethod
    def apply_postplot_params(ax, params):
        """Apply parameters after drawing the plot."""

        ax.set_title(params['title'])
        ax.set_xlabel(params['xlabel'])
        ax.set_ylabel(params['ylabel'])

        legend_config = params['legend']
        if legend_config is True:
            ax.legend()
        elif isinstance(legend_config, dict):
            ax.legend(**legend_config)

        if params['grid']:
            ax.grid(True)
        else:
            ax.grid(False)

        if params['xticks'] is not None:
            ax.set_xticks(params['xticks'], labels=params['xticks_labels'])
        if params['yticks'] is not None:
            ax.set_yticks(params['yticks'], labels=params['yticks_labels'])

        ax.set_xlim(params['xlim'])
        ax.set_ylim(params['ylim'])


################################################################################
#                           NEURON PLOTS                                       # 
# EXPECTED INPUT: (ax, neuron_results)                                         #
################################################################################

class BaseSingleNeuronPlot(BasePlot):
    """This is a meta class for single neuron plots, can be used to plot single neuron results."""
    pass


class SingleNeuronActivityPlot(BaseSingleNeuronPlot):
    """Plot the activity of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Activity',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$\nu_{{out}}$ [Hz]',
        'curves_num': 5,  # Number of curves to plot for each neuron
        'linestyle': 'None',
        'marker': 'o',
        'markersize': 5,
        'labels': None,  # Labels for the curves
        'yerrorbar': False,
        'capsize': 3,  # Error bar cap size
    }

    def _draw(self, ax, neuron_results:SingleNeuronResults):
        plt.gca().set_prop_cycle(None)
            
        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(neuron_results.nu_i[0], self.full_params['curves_num'])):
            if self.full_params['labels'] is None:
                label = fr'$\nu_i$={nu_i:.0f} Hz'
            else:
                label = self.full_params['labels'][j]
            if self.full_params['yerrorbar']:
                yerr = neuron_results.nu_out_std[:, nu_i_idx]
            else:
                yerr = None
            ax.errorbar(neuron_results.nu_e[:,nu_i_idx],
                        neuron_results.nu_out_mean[:,nu_i_idx],
                        yerr= yerr,
                        marker=self.full_params['marker'],
                        linestyle=self.full_params['linestyle'],
                        markersize=self.full_params['markersize'], 
                        capsize=self.full_params['capsize'],
                        label=label,
                        )


class SingleNeuronAdaptationPlot(BaseSingleNeuronPlot):
    """Plot the activity of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Adaptation',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$w$ [pA]',
        'curves_num': 5,  # Number of curves to plot for each neuron
        'linestyle': 'None',
        'marker': 'o',
        'markersize': 5,
        'labels': None,  # Labels for the curves
        'yerrorbar': False,
        'capsize': 3,  # Error bar cap size
    }

    def _draw(self, ax, neuron_results:SingleNeuronResults):
        plt.gca().set_prop_cycle(None)
            
        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(neuron_results.nu_i[0], self.full_params['curves_num'])):
            if self.full_params['labels'] is None:
                label = fr'$\nu_i$={nu_i:.0f} Hz'
            else:
                label = self.full_params['labels'][j]
            if self.full_params['yerrorbar']:
                yerr = neuron_results.w_std[:, nu_i_idx]
            else:
                yerr = None
            ax.errorbar(neuron_results.nu_e[:,nu_i_idx],
                        neuron_results.w_mean[:,nu_i_idx],
                        yerr= yerr,
                        marker=self.full_params['marker'],
                        linestyle=self.full_params['linestyle'],
                        markersize=self.full_params['markersize'], 
                        capsize=self.full_params['capsize'],
                        label=label,
                        )


class SingleNeuronAdaptationHeatmapPlot(BaseSingleNeuronPlot):
    """Plot the adaptation current of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Adaptation Heatmap',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$\nu_i$ [Hz]',
        'vmin': None,  # Minimum value for the heatmap
        'vmax': None,  # Maximum value for the heatmap
        'levels': 10,  # Number of levels in the heatmap
        'cmap': 'viridis',  # Colormap for the heatmap
        'extend': 'neither',  # Extend the colorbar to the maximum value
    }

    def _draw(self, ax, neuron_results:SingleNeuronResults):
        im = ax.contourf(neuron_results.nu_e,
                         neuron_results.nu_i,
                         neuron_results.w_mean,
                         levels=self.full_params['levels'],
                         extend=self.full_params['extend'],
                         vmin=self.full_params['vmin'],
                         vmax=self.full_params['vmax'],
                         cmap=self.full_params['cmap']
                         )
        return im


class SingleNeuronActivityHeatmapPlot(BaseSingleNeuronPlot):
    """Plot the activity of a single neuron as a heatmap."""
    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Adaptation Heatmap',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$\nu_i$ [Hz]',
        'vmin': None,  # Minimum value for the heatmap
        'vmax': None,  # Maximum value for the heatmap
        'levels': 10,  # Number of levels in the heatmap
        'cmap': 'viridis',  # Colormap for the heatmap
        'extend': 'max',  # Extend the colorbar to the maximum value
    }

    def _draw(self, ax, neuron_results:SingleNeuronResults):
        im = ax.contourf(neuron_results.nu_e,
                         neuron_results.nu_i,
                         neuron_results.nu_out_mean,
                         levels=self.full_params['levels'],
                         extend=self.full_params['extend'],
                         vmin=self.full_params['vmin'],
                         vmax=self.full_params['vmax'],
                         cmap=self.full_params['cmap']
                         )
        return im


# TODO:
# Voltage plot (mean, std, tau)
# plot_fluctuations_simulations
# it should be possible to unify this, its very similar!
# Place heatmap plot and activity plot in base class and these would just specify which data to plot


################################################################################
#                           TRANSFER FUNCTION PLOTS                            #
# EXPECTED INPUT: (ax, neuron_results, tf_funcs_list)                          #   
################################################################################

class BaseTransferFunctionPlot(BasePlot):
    """Base class for plots based on transfer function results."""
    pass


class TransferFunctionFitPlot(BaseTransferFunctionPlot):
    """Plot the activity of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Activity',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$\nu_{{out}}$ [Hz]',
        'curves_num': 5,  # Number of curves to plot for each neuron
        'linestyle': 'None',
        'linestyles': LINESTYLES,
        'marker': 'o',
        'markersize': 5,
        'labels': None,  # Labels for the curves
        'yerrorbar': False,
        'capsize': 3,  # Error bar cap size
        'colors': None,
    }

    # NOTE: this one should follow plot_multiple_tf_fits

    def _draw(self, ax, neuron_results:SingleNeuronResults, 
              tf_funcs_list):
            #   tf_funcs_list:list[tf.TransferFunction]):
        if self.full_params['colors'] is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']            
        else:
            colors = self.full_params['colors']

        if self.full_params['labels'] is None:
            self.full_params['labels'] = [f'TF {i+1}' for i in range(len(tf_funcs_list))]

        self.full_params['linestyles'] = self.full_params['linestyles'][:len(tf_funcs_list)]

        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(neuron_results.inh_rate_grid[0], self.full_params['curves_num'])):

            if self.full_params['yerrorbar']:
                yerr = neuron_results.out_rate_std[:, nu_i_idx]
            else:
                yerr = None

            color = colors[j % len(colors)]

            ax.errorbar(neuron_results.exc_rate_grid[:,nu_i_idx],
                        neuron_results.out_rate_mean[:,nu_i_idx],
                        yerr= yerr,
                        marker=self.full_params['marker'],
                        linestyle=self.full_params['linestyle'],
                        markersize=self.full_params['markersize'], 
                        capsize=self.full_params['capsize'],
                        color=color,
                        )

            for tf_funcs, ls in zip(tf_funcs_list, self.full_params['linestyles']):
                if "adaptation"  in tf_funcs.required_inputs():
                    adaptation = neuron_results.adaptation_mean[:, nu_i_idx]
                else:
                    adaptation = None

                nu_out_fit = tf_funcs(
                    exc_rate = neuron_results.exc_rate_grid[:,nu_i_idx], 
                    inh_rate = neuron_results.inh_rate_grid[:,nu_i_idx], 
                    adaptation = adaptation)

                ax.plot(neuron_results.exc_rate_grid[:,nu_i_idx], 
                        nu_out_fit, 
                        color=color, 
                        linestyle=ls
                        )

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='Data', 
                                  markerfacecolor='black', markersize=self.full_params['markersize'], linestyle='None')]
        legend_elements += [Line2D([0], [0], color='black', label=tf_name, linestyle=ls) for tf_name, ls in zip(self.full_params['labels'], self.full_params['linestyles'])]

        self.full_params['legend'] = {'handles': legend_elements}


# TODO:
# Voltage plot (mean, std, tau)
# TF Fitting params plot (V_eff, mu_v etc)
# theoretical values vs measured ones
#   mu_V, sigma_V, tau_V, mu_G, sigma_G, v_eff
#   plot_fluctuation_theoretical, plot_fluctuation_comparison, plot_tf_fitting_steps

################################################################################
#                           SNN PLOTS                                          #
# EXPECTED INPUT: (ax, snn_results)                                            #
################################################################################

class BaseSNNPlot(BasePlot):    
    """Base class for plots base solely on results of Spiking Neural Network.

    This class works only with data in helper.SNNResults
    """
    pass


class SpikeRasterPlot(BaseSNNPlot):
    """Plot the spike raster of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseSNNPlot.DEFAULT_PARAMS,
        'title': 'Spike Raster Plot',
        'xlabel': 'Time (ms)',
        'ylabel': 'Neuron Index',
        'marker': 'o',
        'markersize': 7,
        'exc_cells': RASTER_EXC_CELLS,
        'inh_cells': RASTER_INH_CELLS,
    }

    def _draw(self, ax, snn_results:SNNResults):
        exc_cells = self.full_params['exc_cells']
        inh_cells = self.full_params['inh_cells']
        for i, spiketrain in enumerate(snn_results.exc_spikes_all[:exc_cells], start=1):
            ax.scatter(spiketrain, i * np.ones_like(spiketrain), color=self.full_params['exc_color'], 
                       marker=self.full_params['marker'], s=self.full_params['markersize'], lw=0)
        for i, spiketrain in enumerate(snn_results.inh_spikes_all[:inh_cells], start=exc_cells + 1):
            ax.scatter(spiketrain, i * np.ones_like(spiketrain), color=self.full_params['inh_color'],
                       marker=self.full_params['marker'], s=self.full_params['markersize'], lw=0)


class ActivityHistogramPlot(BaseSNNPlot):
    """Plot the activity histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseSNNPlot.DEFAULT_PARAMS,
        'title': 'Activity Histogram',
        'xlabel': 'Time (ms)',
        'ylabel': 'Firing Rate (Hz)',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
    }

    def _draw(self, ax, snn_results:SNNResults):
        exc_activity = activity_from_spikes_histogram(snn_results.exc_spikes_all, snn_results.times, self.full_params['binsize'])
        inh_activity = activity_from_spikes_histogram(snn_results.inh_spikes_all, snn_results.times, self.full_params['binsize'])
        
        ax.plot(snn_results.times, exc_activity, label='Excitatory', color=self.full_params['exc_color'])
        ax.plot(snn_results.times, inh_activity, label='Inhibitory', color=self.full_params['inh_color'])


################################################################################
#                           NETWORK PLOTS                                      #
# EXPECTED INPUT: (ax, results_list)                                           #
################################################################################

class BaseNetworkPlot(BasePlot):
    """Base class for plots based on network result, can be used to plot network results."""

    DEFAULT_PARAMS = {
    **BasePlot.DEFAULT_PARAMS,
    'labels' : None,
    'linestyles' : None
    }
    # NOTE: None params are updated later in the update_params method

    def update_params(self, results_list:list):
        """Some parameters cannot be generater until the results_list is known, so we update them here."""
        
        if self.full_params['labels'] is None:
            self.full_params['labels'] = [f'Results {i+1}' for i in range(len(results_list))]
       
        if self.full_params['linestyles'] is None:
            # cycles through the predefined LINESTYLES
            self.full_params['linestyles'] = [LINESTYLES[i % len(LINESTYLES)] for i in range(len(results_list))]

        if len(results_list) > 2:
            legend_elements = [Line2D([0], [0], color='black', label=label, linestyle=ls) for label, ls in zip(self.full_params['labels'], self.full_params['linestyles'])]
            if self.full_params['legend'] is True:
                self.full_params['legend'] = {'handles': legend_elements}
            elif type(self.full_params['legend'] ) is dict:
                self.full_params['legend']['handles'] = legend_elements


class FiringRatePlot(BaseNetworkPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Firing Rate over Time',
        'xlabel': 'Time (ms)',
        'ylabel': 'Firing Rate (Hz)',
    }

    def _draw(self, ax, results_list:list):

        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            alpha = 0.5 if isinstance(results, SNNResults) else 1.0
            ax.plot(results.times, results.exc_rate_mean, label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'], alpha=alpha)
            ax.plot(results.times, results.inh_rate_mean, label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'], alpha=alpha)
            if isinstance(results, MFResults) and ls != 'None':
                ax.fill_between(results.times, 
                                results.exc_rate_mean - results.exc_rate_std,
                                results.exc_rate_mean + results.exc_rate_std, 
                                color=self.full_params['exc_color'], alpha=0.3)
                ax.fill_between(results.times, 
                                results.inh_rate_mean - results.inh_rate_std,
                                results.inh_rate_mean + results.inh_rate_std, 
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

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)
 
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

        FiringRatePlot(self.full_params).draw(ax_upper, results_list)
        ax_upper.tick_params(axis='x', which='both', bottom=False, labelbottom=False)


        self.apply_preplot_params(ax_lower, self.full_params | self.LOWER_PLOT_PARAMS)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            ax_lower.plot(results.times, results.drive_rate+results.stim_rate, label=label, ls=ls, color='black')
            # NOTE: for some reason the loop take ridiculous amount of time,
            # so we plot only the first one
            break

        self.apply_postplot_params(ax_lower, self.full_params | self.LOWER_PLOT_PARAMS)


class StimulusWithAdaptationPlot(BaseNetworkPlot):
    """Plot the stimulus with adaptation over time."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Stimulus with Adaptation',
        'xlabel': 'Time (ms)',
        'ylabel': 'Rate (Hz)',
    }

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        axtwin = ax.twinx()  

        ax.plot(results_list[0].times, results_list[0].drive_rate, "--", color='black', label='Drive rate')
        ax.plot(results_list[0].times, results_list[0].stim_rate, "-.", color='black', label='Stimulus rate')
        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            axtwin.plot(results.times, results.exc_adaptation_mean, ls=ls, color='blue', label=label)

        axtwin.set_ylabel("Adaptation current (pA)", color="blue")
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

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            ax.plot(results.times, results.exc_voltage_mean, label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'])
            ax.plot(results.times, results.inh_voltage_mean, label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'])


class FiringRateHistogramPlot(BaseNetworkPlot):
    """Plot the firing rate histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Firing Rate Histogram',
        'xlabel': 'Firing Rate (Hz)',
        'ylabel': 'Count',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
        'start_time': 0,  # Start time for the histogram
    }



    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params['pattern'] != 'NoStimulus':
                raise ValueError("FiringRateHistogramPlot only works for no stimulus simulations.")
            
            if isinstance(results, SNNResults):
                exc_rates, inh_rates = results.per_cell_average_rates(start_time=self.full_params['start_time'])
                exc_bins = int(np.ceil(((exc_rates.max() - exc_rates.min()) / self.full_params['binsize'])))
                inh_bins = int(np.ceil(((inh_rates.max() - inh_rates.min()) / self.full_params['binsize'])))

                ax.hist(exc_rates, bins=exc_bins, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)
                ax.hist(inh_rates, bins=inh_bins, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)

            elif isinstance(results, MFResults):
                # plots gaussian distributioon based on mean and std
                mask = results.times >= self.full_params['start_time']
                exc_mean = np.mean(results.exc_rate_mean[mask])
                inh_mean = np.mean(results.inh_rate_mean[mask])
                if ((results.exc_rate_std is not None) 
                            and (results.exc_rate_std.size > 0)
                            and (results.inh_rate_std is not None) 
                            and (results.inh_rate_std.size > 0)):
                    exc_std = np.mean(results.exc_rate_std[mask])
                    inh_std = np.mean(results.inh_rate_std[mask])
                    
                    x = np.linspace(0, max(exc_mean + 4*exc_std, inh_mean + 4*inh_std), 100)
                    # exc_gauss = (1/(exc_std * np.sqrt(2 * np.pi))) * np.exp( -0.5 * ((x - exc_mean)/exc_std)**2)
                    exc_gauss = np.exp( -0.5 * ((x - exc_mean)/exc_std)**2)
                    exc_gauss /= exc_gauss.sum()  # normalize  
                    # inh_gauss = (1/(inh_std * np.sqrt(2 * np.pi))) * np.exp( -0.5 * ((x - inh_mean)/inh_std)**2)
                    inh_gauss = np.exp( -0.5 * ((x - inh_mean)/inh_std)**2)
                    inh_gauss /= inh_gauss.sum()  # normalize
                    
                    ax.plot(x, 500*exc_gauss, label=f'Exc {label}', color=self.full_params['exc_color'], linestyle=ls)
                    ax.plot(x, 500*inh_gauss, label=f'Inh {label}', color=self.full_params['inh_color'], linestyle=ls)
                else:
                    ax.axvline(exc_mean, label=f'Exc {label}', color=self.full_params['exc_color'], linestyle=ls)
                    ax.axvline(inh_mean, label=f'Inh {label}', color=self.full_params['inh_color'], linestyle=ls)


class VoltageHistogramPlot(BaseNetworkPlot):
    """Plot the voltage histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Voltage Histogram',
        'xlabel': 'Membrane potential (mV)',
        'ylabel': 'Count',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
        'start_time': 0,  # Start time for the histogram
    }

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params['pattern'] != 'NoStimulus':
                raise ValueError("VoltageHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, SNNResults):
                mask = results.times >= self.full_params['start_time']

                exc_voltage = results.exc_voltage_all[mask].mean(axis=0)
                exc_bins = int(np.ceil(((exc_voltage.max() - exc_voltage.min()) / self.full_params['binsize'])))
                ax.hist(exc_voltage, bins=exc_bins, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)

                inh_voltage = results.inh_voltage_all[mask].mean(axis=0)
                inh_bins = int(np.ceil(((inh_voltage.max() - inh_voltage.min()) / self.full_params['binsize'])))
                ax.hist(inh_voltage, bins=inh_bins, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)


class AdaptationHistogramPlot(BaseNetworkPlot):
    """Plot the adaptation histogram of excitatory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Adaptation Histogram',
        'xlabel': 'Adaptation current (pA)',
        'ylabel': 'Count',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
        'start_time': 0,  # Start time for the histogram
    }

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params['pattern'] != 'NoStimulus':
                raise ValueError("AdaptationHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, SNNResults):
                mask = results.times >= self.full_params['start_time']

                exc_adaptation = results.exc_adaptation_all[mask].mean(axis=0)
                exc_bins = int(np.ceil(((exc_adaptation.max() - exc_adaptation.min()) / self.full_params['binsize'])))
                ax.hist(exc_adaptation, bins=exc_bins, alpha=0.5, label=f'Exc {label}', edgecolor='blue', color='blue', linestyle=ls)


class ExcitatoryNeuronConductanceHistogramPlot(BaseNetworkPlot):
    """Plot the conductance histogram of excitatory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Exc Neuron Conductances',
        'xlabel': 'Conductance (nS)',
        'ylabel': 'Count',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
        'start_time': 0,  # Start time for the histogram
    }

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params['pattern'] != 'NoStimulus':
                raise ValueError("ExcitatoryNeuronConductanceHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, SNNResults):
                mask = results.times >= self.full_params['start_time']

                exc_conductance = results.ee_conductance_all[mask].mean(axis=0)
                exc_bins = int(np.ceil(((exc_conductance.max() - exc_conductance.min()) / self.full_params['binsize'])))
                ax.hist(exc_conductance, bins=exc_bins, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)

                inh_conductance = results.ie_conductance_all[mask].mean(axis=0)
                inh_bins = int(np.ceil(((inh_conductance.max() - inh_conductance.min()) / self.full_params['binsize'])))
                ax.hist(inh_conductance, bins=inh_bins, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)

class InhibitoryNeuronConductanceHistogramPlot(BaseNetworkPlot):
    """Plot the conductance histogram of inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'title': 'Inh Neuron Conductances',
        'xlabel': 'Conductance (nS)',
        'ylabel': 'Count',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
        'start_time': 0,  # Start time for the histogram
    }

    def _draw(self, ax, results_list:list):
        self.update_params(results_list)

        for results, ls, label in zip(results_list, self.full_params['linestyles'], self.full_params['labels']):
            if results.stim_params['pattern'] != 'NoStimulus':
                raise ValueError("InhibitoryNeuronConductanceHistogramPlot only works for no stimulus simulations.")
            if isinstance(results, SNNResults):
                mask = results.times >= self.full_params['start_time']

                exc_conductance = results.ei_conductance_all[mask].mean(axis=0)
                exc_bins = int(np.ceil(((exc_conductance.max() - exc_conductance.min()) / self.full_params['binsize'])))
                ax.hist(exc_conductance, bins=exc_bins, alpha=0.5, label=f'Exc {label}', edgecolor=self.full_params['exc_color'], color=self.full_params['exc_color'], linestyle=ls)

                inh_conductance = results.ii_conductance_all[mask].mean(axis=0)
                inh_bins = int(np.ceil(((inh_conductance.max() - inh_conductance.min()) / self.full_params['binsize'])))
                ax.hist(inh_conductance, bins=inh_bins, alpha=0.5, label=f'Inh {label}', edgecolor=self.full_params['inh_color'], color=self.full_params['inh_color'], linestyle=ls)
# TODO:
# conductance plot
# short term plasticity plot

################################################################################
#                         INSPECTION PLOTS                                     #
# EXPECTED INPUT: (ax, results_list)                                           #
################################################################################

class BaseInspectionPlot(BasePlot):
    """Base class for plots based on inspection results, can be used to plot inspection results."""

    DEFAULT_PARAMS = {
    **BasePlot.DEFAULT_PARAMS,
    'labels' : None,
    'markers' : None,
    'linestyles' : None
    }
    # NOTE: None params are updated later in the update_params method

    def update_params(self, results_list:list):
        """Some parameters cannot be generater until the results_list is known, so we update them here."""
        
        if self.full_params['labels'] is None:
            self.full_params['labels'] = [f'Results {i+1}' for i in range(len(results_list))]
       
        if self.full_params['linestyles'] is None:
            # cycles through the predefined LINESTYLES
            self.full_params['linestyles'] = [LINESTYLES[i % len(LINESTYLES)] for i in range(len(results_list))]

        if self.full_params['markers'] is None:
            self.full_params['markers'] = ["None" for i in range(len(results_list))]

        if len(results_list) > 2:
            legend_elements = [Line2D([0], [0], color='black', label=label, linestyle=ls, marker=marker) for label, ls, marker in zip(self.full_params['labels'], self.full_params['linestyles'], self.full_params['markers'])]
            if self.full_params['legend'] is True:
                self.full_params['legend'] = {'handles': legend_elements}
            elif type(self.full_params['legend'] ) is dict:
                self.full_params['legend']['handles'] = legend_elements
        
        if self.full_params['xlabel'] is None:
            self.full_params['xlabel'] = results_list[0].inspected_param


class FiringRateInspectionPlot(BaseInspectionPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseInspectionPlot.DEFAULT_PARAMS,
        'title': 'Time averaged Firing Rate\n vs Parameter',
        'xlabel': None,  # Will be set to inspected_param
        'ylabel': 'Firing Rate (Hz)',
    }

    def _draw(self, ax, results_list:list):

        self.update_params(results_list)

        for results,  ls, marker, label in zip(results_list, self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels']):
            if results.inspected_network_name.startswith("SNN"):
                if hasattr(results, 'exc_rate_time_std') and hasattr(results, 'inh_rate_time_std'):
                    ax.errorbar(results.param_values, results.exc_rate_time_mean, yerr=results.exc_rate_time_std, label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                    ax.errorbar(results.param_values, results.inh_rate_time_mean, yerr=results.inh_rate_time_std, label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
                else:
                    ax.plot(results.param_values, results.exc_rate_time_mean, label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                    ax.plot(results.param_values, results.inh_rate_time_mean, label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
            # elif results.inspected_network_name.startswith("MF"):
            else:
                ax.plot(results.param_values, results.exc_rate_time_mean, label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                ax.plot(results.param_values, results.inh_rate_time_mean, label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
                if hasattr(results, 'exc_rate_time_std') and hasattr(results, 'inh_rate_time_std'):
                    ax.fill_between(results.param_values, 
                                    np.array(results.exc_rate_time_mean) - np.array(results.exc_rate_time_std),
                                    np.array(results.exc_rate_time_mean) + np.array(results.exc_rate_time_std), 
                                    color=self.full_params['exc_color'], alpha=0.3)
                    ax.fill_between(results.param_values, 
                                    np.array(results.inh_rate_time_mean) - np.array(results.inh_rate_time_std),
                                    np.array(results.inh_rate_time_mean) + np.array(results.inh_rate_time_std), 
                                    color=self.full_params['inh_color'], alpha=0.3)


class VoltageInspectionPlot(BaseInspectionPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseInspectionPlot.DEFAULT_PARAMS,
        'title': 'Time averaged Voltage\n vs Parameter',
        'xlabel': None,  # Will be set to inspected_param
        'ylabel': 'Voltage (mV)',
    }


    def _draw(self, ax, results_list:list):

        self.update_params(results_list)

        for results, ls, marker, label in zip(results_list, self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels']):
            if results.inspected_network_name.startswith("SNN") and hasattr(results, 'exc_voltage_time_std') and hasattr(results, 'inh_voltage_time_std'):
                    ax.errorbar(results.param_values, results.exc_voltage_time_mean, yerr=results.exc_voltage_time_std, label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                    ax.errorbar(results.param_values, results.inh_voltage_time_mean, yerr=results.inh_voltage_time_std, label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
            else:
                ax.plot(results.param_values, results.exc_voltage_time_mean, label=f'Exc {label}', ls=ls, color=self.full_params['exc_color'], marker=marker)
                ax.plot(results.param_values, results.inh_voltage_time_mean, label=f'Inh {label}', ls=ls, color=self.full_params['inh_color'], marker=marker)



class AdaptationInspectionPlot(BaseInspectionPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseInspectionPlot.DEFAULT_PARAMS,
        'title': 'Time averaged Adaptation\n vs Parameter',
        'xlabel': None,  # Will be set to inspected_param
        'ylabel': 'Adaptation (pA)',
    }


    def _draw(self, ax, results_list:list):

        self.update_params(results_list)

        for results, ls, marker, label in zip(results_list, self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels']):
            if results.inspected_network_name.startswith("SNN") and hasattr(results, 'exc_adaptation_time_std'):
                    ax.errorbar(results.param_values, results.exc_adaptation_time_mean, yerr=results.exc_adaptation_time_std, label=f'Exc {label}', ls=ls, marker=marker, color='blue')
            else:
                ax.plot(results.param_values, results.exc_adaptation_time_mean, label=f'Exc {label}', ls=ls, color='blue', marker=marker)