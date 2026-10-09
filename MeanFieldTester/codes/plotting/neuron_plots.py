import matplotlib.pyplot as plt
from ..utils.list_helpers import indexed_linear_sample
from ..data_structures.base import BaseSingleNeuronResults
from .base import BaseSingleNeuronPlot, HEATMAP_PARAMS, draw_grid_heatmap
from typing import Dict


class SingleNeuronActivityPlot(BaseSingleNeuronPlot):
    """Plot the activity of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BaseSingleNeuronPlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Activity',
        'x_unit': 'Hz',
        'y_unit': 'Hz',
        'xlabel': r'$\nu_e$',
        'ylabel': r'$\nu_{{out}}$',
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
        single_neuron_result = self.get_neuron_results(neuron_results)
        plt.gca().set_prop_cycle(None)

        x_unit = self.full_params.get('x_unit', None)
        y_unit = self.full_params.get('y_unit', None)


        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(single_neuron_result.inh_rate_grid(x_unit)[0], self.full_params['curves_num'])):
            if self.full_params['labels'] is None:
                label = fr'$\nu_i$={nu_i:.0f} {x_unit}'
            else:
                label = self.full_params['labels'][j]
            if self.full_params['yerrorbar']:
                yerr = single_neuron_result.out_rate_std(y_unit)[:, nu_i_idx]
            else:
                yerr = None
            ax.errorbar(single_neuron_result.exc_rate_grid(x_unit)[:,nu_i_idx],
                        single_neuron_result.out_rate_mean(y_unit)[:,nu_i_idx],
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
        'x_unit': 'Hz',
        'y_unit': 'pA',
        'xlabel': r'$\nu_e$',
        'ylabel': r'$w$',
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
        single_neuron_result = self.get_neuron_results(neuron_results)
        plt.gca().set_prop_cycle(None)

        x_unit = self.full_params.get('x_unit', None)
        y_unit = self.full_params.get('y_unit', None)

        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(single_neuron_result.inh_rate_grid(x_unit)[0], self.full_params['curves_num'])):
            if self.full_params['labels'] is None:
                label = fr'$\nu_i$={nu_i:.0f} {x_unit}'
            else:
                label = self.full_params['labels'][j]
            if self.full_params['yerrorbar']:
                yerr = single_neuron_result.adaptation_std(y_unit)[:, nu_i_idx]
            else:
                yerr = None
            ax.errorbar(single_neuron_result.exc_rate_grid(x_unit)[:,nu_i_idx],
                        single_neuron_result.adaptation_mean(y_unit)[:,nu_i_idx],
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
        'xlabel': r'$\nu_e$',
        'ylabel': r'$\nu_i$',
        'x_unit': 'Hz',
        'y_unit': 'Hz',
        'z_unit': 'pA',
        **HEATMAP_PARAMS,
        'extend': 'neither',  # Extend the colorbar to the maximum value
        'colorbar_label': 'adaptation',  # Label for the colorbar
    }

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        single_neuron_result = self.get_neuron_results(neuron_results)

        x_unit = self.full_params.get('x_unit', None)
        y_unit = self.full_params.get('y_unit', None)
        z_unit = self.full_params.get('z_unit', None)

        im = draw_grid_heatmap(ax,
                               single_neuron_result.exc_rate_grid(x_unit),
                               single_neuron_result.inh_rate_grid(y_unit),
                               single_neuron_result.adaptation_mean(z_unit),
                               self.full_params)
        return im


class SingleNeuronActivityHeatmapPlot(BaseSingleNeuronPlot):
    """Plot the activity of a single neuron as a heatmap."""
    DEFAULT_PARAMS = {
        **BaseSingleNeuronPlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Activity Heatmap',
        'x_unit': 'Hz',
        'y_unit': 'Hz',
        'z_unit': 'Hz',
        'xlabel': r'$\nu_e$',
        'ylabel': r'$\nu_i$',
        **HEATMAP_PARAMS,
        'extend': 'max',  # Extend the colorbar to the maximum value
        'colorbar_label': r'$\nu_{{out}}$',  # Label for the colorbar
    }

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        single_neuron_result = self.get_neuron_results(neuron_results)

        x_unit = self.full_params.get('x_unit', None)
        y_unit = self.full_params.get('y_unit', None)
        z_unit = self.full_params.get('z_unit', None)

        im = draw_grid_heatmap(ax,
                               single_neuron_result.exc_rate_grid(x_unit),
                               single_neuron_result.inh_rate_grid(y_unit),
                               single_neuron_result.out_rate_mean(z_unit),
                               self.full_params)
        return im

