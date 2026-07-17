import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Dict

from ..utils.list_helpers import indexed_linear_sample


from ..data_structures.base import BaseSingleNeuronResults
from ..transfer_function.base import BaseTransferFunction
from ..network_params.translators import get_unit_multiplier

from .base import BaseTransferFunctionPlot, LINESTYLES


class TransferFunctionFitPlot(BaseTransferFunctionPlot):
    """Plot the activity of a single neuron over time."""
    DEFAULT_PARAMS = {
        **BaseTransferFunctionPlot.DEFAULT_PARAMS,
        'title': 'Single Neuron Activity',
        'x_unit': 'Hz',
        'y_unit': 'Hz',
        'xlabel': r'$\nu_e$',
        'ylabel': r'$\nu_{{out}}$',
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

    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            tf_funcs_results: Dict[str, List[BaseTransferFunction]],
            ) -> None:

        neuron_results = neuron_results[self.full_params['neuron_name']]
        tf_funcs_list = tf_funcs_results[self.full_params['neuron_name']]

        x_unit = self.full_params.get('x_unit', None)
        y_unit = self.full_params.get('y_unit', None)

        if self.full_params['colors'] is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']            
        else:
            colors = self.full_params['colors']

        if self.full_params['labels'] is None:
            self.full_params['labels'] = [f'TF {i+1}' for i in range(len(tf_funcs_list))]

        self.full_params['linestyles'] = self.full_params['linestyles'][:len(tf_funcs_list)]

        for j, (nu_i_idx, nu_i) in enumerate(indexed_linear_sample(neuron_results.inh_rate_grid(x_unit)[0], self.full_params['curves_num'])):

            if self.full_params['yerrorbar']:
                yerr = neuron_results.out_rate_std(y_unit)[:, nu_i_idx]
            else:
                yerr = None

            color = colors[j % len(colors)]

            ax.errorbar(neuron_results.exc_rate_grid(x_unit)[:,nu_i_idx],
                        neuron_results.out_rate_mean(y_unit)[:,nu_i_idx],
                        yerr= yerr,
                        marker=self.full_params['marker'],
                        linestyle=self.full_params['linestyle'],
                        markersize=self.full_params['markersize'], 
                        capsize=self.full_params['capsize'],
                        color=color,
                        )

            # NOTE: Here a bit of hack, since transfer_function module does not 
            # unit handling and expects Hz for rates and nA for adaptation,
            # thus hardcoded unit conversion here. This should be fixed in the future.
            for tf_funcs, ls in zip(tf_funcs_list, self.full_params['linestyles'], strict=True):
                if "adaptation"  in tf_funcs.required_inputs():
                    adaptation = neuron_results.adaptation_mean("nA")[:, nu_i_idx]
                else:
                    adaptation = None

                nu_out_fit = tf_funcs(
                    exc_rate = neuron_results.exc_rate_grid("Hz")[:,nu_i_idx], 
                    inh_rate = neuron_results.inh_rate_grid("Hz")[:,nu_i_idx], 
                    adaptation = adaptation)*get_unit_multiplier("Hz", y_unit)

                ax.plot(neuron_results.exc_rate_grid(x_unit)[:,nu_i_idx], 
                        nu_out_fit, 
                        color=color, 
                        linestyle=ls,
                        linewidth=self.full_params['linewidth'],
                        )

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='Data', 
                                  markerfacecolor='black', markersize=self.full_params['markersize'], linestyle='None')]
        legend_elements += [Line2D([0], [0], color='black', label=tf_name, linestyle=ls, linewidth=self.full_params['linewidth']) for tf_name, ls in zip(self.full_params['labels'], self.full_params['linestyles'])]

        if isinstance(self.full_params['legend'], dict):
            self.full_params['legend']['handles'] = legend_elements
        elif self.full_params['legend'] is True:
            self.full_params['legend'] = {'handles': legend_elements}