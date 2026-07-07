import matplotlib.pyplot as plt
from ..utils.list_helpers import indexed_linear_sample
from ..data_structures.base import BaseSingleNeuronResults
from .base import BaseSingleNeuronPlot
from typing import Dict


class SingleNeuronActivityPlot(BaseSingleNeuronPlot):
    """Plot the activity of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BaseSingleNeuronPlot.DEFAULT_PARAMS,
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

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        single_neuron_result = neuron_results[self.full_params['neuron_name']]
        plt.gca().set_prop_cycle(None)
            
        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(single_neuron_result.inh_rate_grid()[0], self.full_params['curves_num'])):
            if self.full_params['labels'] is None:
                label = fr'$\nu_i$={nu_i:.0f} Hz'
            else:
                label = self.full_params['labels'][j]
            if self.full_params['yerrorbar']:
                yerr = single_neuron_result.out_rate_std()[:, nu_i_idx]
            else:
                yerr = None
            ax.errorbar(single_neuron_result.exc_rate_grid()[:,nu_i_idx],
                        single_neuron_result.out_rate_mean()[:,nu_i_idx],
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
        **BaseSingleNeuronPlot.DEFAULT_PARAMS,
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

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        single_neuron_result = neuron_results[self.DEFAULT_PARAMS['neuron_name']]
        plt.gca().set_prop_cycle(None)
            
        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(single_neuron_result.inh_rate_grid()[0], self.full_params['curves_num'])):
            if self.full_params['labels'] is None:
                label = fr'$\nu_i$={nu_i:.0f} Hz'
            else:
                label = self.full_params['labels'][j]
            if self.full_params['yerrorbar']:
                yerr = single_neuron_result.adaptation_std()[:, nu_i_idx]
            else:
                yerr = None
            ax.errorbar(single_neuron_result.exc_rate_grid()[:,nu_i_idx],
                        single_neuron_result.adaptation_mean()[:,nu_i_idx],
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
        **BaseSingleNeuronPlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Adaptation Heatmap',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$\nu_i$ [Hz]',
        'vmin': None,  # Minimum value for the heatmap
        'vmax': None,  # Maximum value for the heatmap
        'levels': 10,  # Number of levels in the heatmap
        'cmap': 'viridis',  # Colormap for the heatmap
        'extend': 'neither',  # Extend the colorbar to the maximum value
        'colorbar_label': 'adaptation [nA]',  # Label for the colorbar
    }

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        single_neuron_result = neuron_results[self.DEFAULT_PARAMS['neuron_name']]
        im = ax.contourf(single_neuron_result.exc_rate_grid(),
                         single_neuron_result.inh_rate_grid(),
                         single_neuron_result.adaptation_mean(),
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
        **BaseSingleNeuronPlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Activity Heatmap',
        'xlabel': r'$\nu_e$ [Hz]',
        'ylabel': r'$\nu_i$ [Hz]',
        'vmin': None,  # Minimum value for the heatmap
        'vmax': None,  # Maximum value for the heatmap
        'levels': 10,  # Number of levels in the heatmap
        'cmap': 'viridis',  # Colormap for the heatmap
        'extend': 'max',  # Extend the colorbar to the maximum value
        'colorbar_label': r'$\nu_{{out}}$ [Hz]',  # Label for the colorbar
    }

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        single_neuron_result = neuron_results[self.full_params['neuron_name']]
        im = ax.contourf(single_neuron_result.exc_rate_grid(),
                         single_neuron_result.inh_rate_grid(),
                         single_neuron_result.out_rate_mean(),
                         levels=self.full_params['levels'],
                         extend=self.full_params['extend'],
                         vmin=self.full_params['vmin'],
                         vmax=self.full_params['vmax'],
                         cmap=self.full_params['cmap']
                         )
        return im

