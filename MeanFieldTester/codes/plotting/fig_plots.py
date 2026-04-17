"""
This script is a collection of predefined fig plots for the MeanFieldTester project.


Naming conventions:
_together   - multiple plots are together in one figure
_per_column - multiple plots are arranged in a column
_per_row    - multiple plots are arranged in a row


"""


import matplotlib.pyplot as plt
import numpy as np

from codes import plotting as ax_plt

from codes.data_structures.network import MFResults, SNNResults
from codes.data_structures.single_neuron import SingleNeuronResults
# from codes.transfer_function import TransferFunction

DEFAULT_FIG_PARAMS = {
    'fontsize': 14,
    'dpi': 100,
    'axsize': (8, 5),  # Default size for each subplot
    'figsize': None,  # If not specified, it will be calculated from 'axsize'
    'title': None,  # Default title is None
    'tight_layout': True,  # Use tight layout by default
    'savefig': False,  # Do not save figure by default
    'savefig_path': None,  # Path to save the figure
    'gridspec_kw': {},
}

def prepare_fig(fig_params):
    """Prepare the figure with default parameters."""
    plt.rcParams['font.size'] = fig_params['fontsize']

    rows, cols = fig_params['layout']

    if fig_params['figsize'] is None:
        fig_params['figsize'] = (fig_params['axsize'][0] * cols, fig_params['axsize'][1] * rows)
    fig, axes = plt.subplots(rows, cols, 
                             figsize=fig_params['figsize'], 
                             squeeze=False, 
                             gridspec_kw=fig_params['gridspec_kw'])

    return fig, axes

def finish_fig(fig, fig_params):
    """Finalize the figure with tight layout and saving options."""
    if fig_params['title']:
        fig.suptitle(fig_params['title'])
    if fig_params['tight_layout']:
        fig.tight_layout()
    if fig_params['savefig']:
        fig.savefig(fig_params['savefig_path'], dpi=fig_params['dpi'])
    plt.close(fig)

################################################################################
#                           SingleNeuronResults plots                          #
################################################################################

def fig_neuron_activity(neuron_results:dict[SingleNeuronResults], 
                        common_params:dict, 
                        fig_params:dict):
    """Plot neuron activity as a heatmap and individual neuron activity.

    Each row in the figure corresponds to a different neuron.
    First column is a heatmap of activity.
    Second column is the individual neuron activity plot.

    Parameters
    ----------
    neuron_results : dict[SingleNeuronResults]
        Dictionary containing results for each neuron.
        Keys are neuron names, values are instances
        of SingleNeuronResults containing neuron activity data.
    common_params : dict
        Common parameters for the plots.
    fig_params : dict
        Parameters for the figure, including size, layout, and saving options.
    """ 

    fig_params = {**DEFAULT_FIG_PARAMS, **fig_params}

    plots = [
        ax_plt.SingleNeuronActivityHeatmapPlot({
            'levels' : 10,
            **common_params,
        }),
        ax_plt.SingleNeuronActivityPlot({
            'xmargin': 0.0,
            'ymargin': 0.0,
            'legend': True,
            'curves_num' : 7,
            'linestyle' : 'None',
            'yerrorbar' : True,
            'capsize' : 3,
            **common_params,
        }),
    ]

    fig_params['layout'] = (len(neuron_results), len(plots))

    fig, axes = prepare_fig(fig_params)

    for row, (neuron_name, results) in enumerate(neuron_results.items()):
        for col, plot in enumerate(plots):
            im = plot.draw(axes[row, col], results)
            if isinstance(plot, ax_plt.SingleNeuronActivityHeatmapPlot):
                fig.colorbar(im, ax=axes[row, col], label='activity (Hz)')
                axes[row, col].set_title(f"{neuron_name} activity")

    finish_fig(fig, fig_params)

# TODO:
# Fig with adaptation
# Fig of membrane potential fluctuations

# old_plots.plot_simulated_computed_adaptation(self.snn_neurons, self.neuron_results, self.results_path)
# old_plots.plot_fluctuations_simulations(self.snn_neurons, self.neuron_results, self.results_path)

################################################################################
#                           TransferFunction plots                             #
################################################################################

def fig_tf_fits_together(neuron_results:dict[SingleNeuronResults], 
                tf_funcs, 
                # tf_funcs:dict[TransferFunction], 
                common_params:dict, 
                fig_params:dict): 
    """Plot transfer function fits for each neuron.

    Each row in the figure corresponds to a different neuron.
    All transfer function fits for the neuron are plotted in the same plot.

    Parameters
    ----------
    neuron_results : dict[SingleNeuronResults]
        Dictionary containing results for each neuron.
        Keys are neuron names, values are instances
        of SingleNeuronResults containing neuron activity data.
    tf_funcs : dict
        Dictionary of transfer function fit results for each neuron.
    common_params : dict
        Common parameters for the plots.
    fig_params : dict
        Parameters for the figure, including size, layout, and saving options.
    """

    fig_params = {**DEFAULT_FIG_PARAMS, 'axsize': (15, 5), **fig_params}

    plots = [
        ax_plt.TransferFunctionFitPlot({
            'markersize' : 5,
            'ylim' : (None, 60),
            'xmargin' : 0.0,
            'ymargin' : 0.0,
            'legend' : True,
            'curves_num' : 10,
            'xmargin' : 0.0,
            'ymargin' : 0.0,
            'yerrorbar' : True,
            **common_params,
        }),
    ]
    fig_params['layout'] = (len(tf_funcs), len(plots))

    fig, axes = prepare_fig(fig_params)

    for row, (neuron_name, tf_results) in enumerate(tf_funcs.items()):
        for col, plot in enumerate(plots):
            plot.draw(axes[row, col], neuron_results[neuron_name], tf_results)
            axes[row, col].set_title(f"{neuron_name}")

    finish_fig(fig, fig_params)

# TODO:
# Fig with fluctuations theoretical
# fig with Fitting procedure

# old_plots.plot_tf_fitting_output
# old_plots.plot_single_neuron_activity
# old_plots.plot_fluctuation_theoretical
# old_plots.plot_fluctuation_comparison
# old_plots.plot_tf_fitting_steps
# old_plots.plot_multiple_tf_fits

################################################################################
#                           MF & SNN results plots                             #
################################################################################

def fig_full_network_overview_together(snn_results: SNNResults, 
                                       mf_results_list: list[MFResults], 
                                       common_params: dict, 
                                       fig_params: dict): 
    """Plot full network overview for SNN and MF results together.

    Each row in the figure corresponds to a different plot type.
    First row is the spike raster plot.
    Second row is the firing rate plot.
    Third row is the stimulus and adaptation plot.
    Fourth row is the voltage plot.

    Parameters
    ----------
    snn_results : SNNResults
        Results of the SNN simulation.
    mf_results_list : list[MFResults]
        List of results for each mean-field network.
    common_params : dict
        Common parameters for the plots.
    fig_params : dict
        Parameters for the figure, including size, layout, and saving options.
    """
    
    fig_params = {**DEFAULT_FIG_PARAMS, 'gridspec_kw' : {'hspace': 0.0}, **fig_params}

    plots = [
        ax_plt.SpikeRasterPlot({
            'markersize': 7,
            **common_params,
            'xticks' : [],
            'xticks_labels' : None,
            'xlabel' : None,
            'title' : None,
            'legend' : False,
        }),
        ax_plt.FiringRatePlot({
            **common_params,
            'ylim': (0, 15),
            'xticks_labels' : None,
            'xticks' : [],
            'xlabel' : None,
            'title' : None,
        }),
        ax_plt.StimulusWithAdaptationPlot({
            **common_params,
            'xticks_labels' : None,
            'xticks' : [],
            'xlabel' : None,
            'title' : None,
        }),
        ax_plt.VoltagePlot({
            **common_params,
            'title' : None,
            'ylim' : (-60, -54)
        })
    ]

    fig_params['layout'] = (len(plots), 1)
    fig, axes = prepare_fig(fig_params)

    for row, plot in enumerate(plots):
        match plot:
            case ax_plt.BaseSNNPlot():
                plot.draw(axes[row, 0], snn_results)
            case ax_plt.BaseNetworkPlot():
                plot.draw(axes[row, 0], [snn_results] + mf_results_list)

    finish_fig(fig, fig_params)

def fig_full_network_overview_per_column(snn_results: SNNResults, 
                                         mf_results_list: list[MFResults], 
                                         common_params: dict, 
                                         fig_params: dict): 
    """Plot full network overview for SNN and MF results per column.

    Each column in the figure corresponds to a different mean-field network.
    First row is the spike raster plot.
    Second row is the firing rate plot.
    Third row is the stimulus and adaptation plot.
    Fourth row is the voltage plot.

    Parameters
    ----------
    snn_results : SNNResults
        Results of the SNN simulation.
    mf_results_list : list[MFResults]
        List of results for each mean-field network.
    common_params : dict
        Common parameters for the plots.
    fig_params : dict
        Parameters for the figure, including size, layout, and saving options.
    """

    fig_params = {**DEFAULT_FIG_PARAMS, **fig_params}

    plots = [
        ax_plt.SpikeRasterPlot({
            **common_params,
            'markersize': 7,
        }),
        ax_plt.FiringRatePlot({
            **common_params,
            'ylim': (0, 15),
            'labels': ["SNN", "MF"],
            'legend': True,
        }),
        ax_plt.StimulusWithAdaptationPlot({
            **common_params,
            'legend' : True,
            'labels' : ["SNN", "MF"],
        }),
        ax_plt.VoltagePlot({
            **common_params,
            'legend' : True,
            'labels' : ["SNN", "MF"],
            'ylim' : (-60, -54)
        })
    ]

    fig_params['layout'] = (len(plots), len(mf_results_list))
    fig, axes = prepare_fig(fig_params)    

    for col, mf_results in enumerate(mf_results_list):
        for row, plot in enumerate(plots):
            match plot:
                case ax_plt.BaseSNNPlot():
                    plot.draw(axes[row, col], snn_results)
                case ax_plt.BaseNetworkPlot():
                    plot.draw(axes[row, col], [snn_results, mf_results])
        if common_params['labels']:
            axes[0, col].set_title(common_params['labels'][col])

    finish_fig(fig, fig_params)

def fig_network_activity_together(snn_results: SNNResults, 
                                  mf_results_list: list[MFResults], 
                                  common_params: dict, 
                                  fig_params: dict):
    """Plot network activity for SNN and MF results together.
    
    Each row in the figure corresponds to a different plot type.
    First row is the spike raster plot.
    Second row is the firing rate plot with stimulus plot.
    Third row is the voltage plot.
    
    Parameters
    ----------
    snn_results : SNNResults
        Results of the SNN simulation.
    mf_results_list : list[MFResults]
        List of results for each mean-field network.
    common_params : dict
        Common parameters for the plots.
    fig_params : dict
        Parameters for the figure, including size, layout, and saving options.
    """
    
    fig_params = {**DEFAULT_FIG_PARAMS, 'gridspec_kw' : {'hspace': 0.0}, **fig_params}

    plots = [
        ax_plt.SpikeRasterPlot({
            'markersize': 7,
            **common_params,
            'xticks' : [],
            'xticks_labels' : None,
            'xlabel' : None,
            'title' : None,
            'legend' : False,
        }),
        ax_plt.FiringRateAndStimulusPlot({
            **common_params,
            'ylim': (0, 15),
            'xticks_labels' : None,
            'xticks' : [],
            'xlabel' : None,
            'title' : None,
        }),
        ax_plt.VoltagePlot({
            **common_params,
            'title' : None,
            'ylim' : (-60, -54),
        })
    ]

    fig_params['layout'] = (len(plots), 1)
    fig, axes = prepare_fig(fig_params)

    for row, plot in enumerate(plots):
        match plot:
            case ax_plt.BaseSNNPlot():
                plot.draw(axes[row, 0], snn_results)
            case ax_plt.BaseNetworkPlot():
                plot.draw(axes[row, 0], [snn_results] + mf_results_list)

    finish_fig(fig, fig_params)

def fig_network_activity_overview_per_column(snn_results: SNNResults, 
                                             mf_results_list: list[MFResults], 
                                             common_params: dict, 
                                             fig_params: dict): 
    """Plot network activity overview for SNN and MF results per column.

    Each column in the figure corresponds to a different mean-field network.
    First row is the spike raster plot.
    Second row is the firing rate plot with stimulus plot.
    Third row is the voltage plot.

    Parameters
    ----------
    snn_results : SNNResults
        Results of the SNN simulation.
    mf_results_list : list[MFResults]
        List of results for each mean-field network.
    common_params : dict
        Common parameters for the plots.
    fig_params : dict
        Parameters for the figure, including size, layout, and saving options.
    """
    fig_params = {**DEFAULT_FIG_PARAMS, 'gridspec_kw' : {'hspace': 0.0}, **fig_params}

    plots = [
        ax_plt.SpikeRasterPlot({
            **common_params,
            'markersize': 7,
        }),
        ax_plt.FiringRateAndStimulusPlot({
            **common_params,
            'ylim': (0, 15),
            'labels': ["SNN", "MF"],
            'legend': True,
        }),
    ]

    fig_params['layout'] = (len(plots), len(mf_results_list))
    fig, axes = prepare_fig(fig_params)

    for col, mf_results in enumerate(mf_results_list):
        for row, plot in enumerate(plots):
            match plot:
                case ax_plt.BaseSNNPlot():
                    plot.draw(axes[row, col], snn_results)
                case ax_plt.BaseNetworkPlot():
                    plot.draw(axes[row, col], [snn_results, mf_results])
        if common_params['labels']:
            axes[0, col].set_title(common_params['labels'][col])

    finish_fig(fig, fig_params)

# TODO:
# inspections
# spont_densities

# old_plots.plot_spont_inspection
# old_plots.plot_spont_densities
################################################################################


################################################################################
# The following are drafts of a new figure constructor that can be used to create figures
# based on a list of plots. It is not integrated into the existing codebase.
################################################################################


class FigureConstructor:
    """A class to construct figures based on provided plots and parameters."""
    DEFAULT_PARAMS = {
        'fontsize': 12,
        'axsize': (5, 3),  # Default size for each subplot
        'figsize': None,  # If not specified, it will be calculated from 'axsize'
        'title': None,  # Default title is None
        'tight_layout': True,  # Use tight layout by default
        'savefig': False,  # Do not save figure by default
    }

    def __init__(self, plots:list, fig_params=None):
        """Initialize the FigureConstructor with plots and figure parameters.
        
        Parameters
        ----------
        plots : list
            A list of lists of BasePlot objects or subclasses
            The outer list indexes rows, the nested list indexes columns.
            For example: [[plot11, plot12], [plot21, plot22]]
            will create a figure with 2 rows and 2 columns.
        fig_params : dict, optional
            A dictionary of figure parameters to override the defaults.
        """

        assert isinstance(plots, list), \
            "plots must be a list of lists of BasePlot objects or subclasses."
        assert all(isinstance(col, list) for col in plots), \
            "plots must be a list of lists of BasePlot objects or subclasses."
        assert all(len(col) == len(plots[0]) for col in plots), \
            "All columns must have the same number of plots."
        assert all(isinstance(plot, ax_plt.BasePlot) for col in plots for plot in col), \
            "All elements in plots must be instances of BasePlot or its subclasses."

        self.plots = plots
        self.fig_params = {**self.DEFAULT_PARAMS, **(fig_params or {})}
        self.layout = (len(plots), len(plots[0]))  # (rows, cols)

    def make_figure(self, neuron_results, tf_funcs, snn_results, mf_results_list):
        """Create a figure with the specified plots."""

        plt.rcParams['font.size'] = self.fig_params['fontsize']

        rows, cols = self.layout

        if self.fig_params['figsize'] is None:
            self.fig_params['figsize'] = (self.fig_params['axsize'][0] * cols, self.fig_params['axsize'][1] * rows)

        fig, axes = plt.subplots(rows, cols, figsize=self.fig_params['figsize'], squeeze=False)

        for row in range(rows):
            for col in range(cols):
                plot = self.plots[row][col]
                match plot:
                    case ax_plt.BaseSingleNeuronPlot():
                        plot.draw(axes[row, col], neuron_results)
                    case ax_plt.BaseTransferFunctionPlot():
                        plot.draw(axes[row, col], neuron_results, tf_funcs)
                    case ax_plt.BaseSNNPlot():
                        plot.draw(axes[row, col], snn_results)
                    case ax_plt.BaseNetworkPlot():
                        plot.draw(axes[row, col], [snn_results] + mf_results_list)

        if self.fig_params['title']:
            fig.suptitle(self.fig_params['title'])
        if self.fig_params['tight_layout']:
            fig.tight_layout()
        if self.fig_params['savefig']:
            fig.savefig(self.fig_params['savefig_path'])
        return fig



def make_fig(neuron_results, tf_funcs, snn_results, mf_results_list, plots, fig_params):
    """Make a basic figure with the given plots."""
    plt.rcParams['font.size'] = fig_params['fontsize']

    rows = len(plots)
    cols = 1 

    figsize = (10, 3 * rows) if fig_params['figsize'] is None else fig_params['figsize']
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for row, plot in enumerate(plots):
        match plot:
            case ax_plt.BaseSingleNeuronPlot():
                plot.draw(axes[row, 0], neuron_results)
            case ax_plt.BaseTransferFunctionPlot():
                plot.draw(axes[row, 0], neuron_results, tf_funcs)
            case ax_plt.BaseSNNPlot():
                plot.draw(axes[row, 0], snn_results)
            case ax_plt.BaseNetworkPlot():
                plot.draw(axes[row, 0], [snn_results] + mf_results_list)

    if fig_params['tight_layout']:
        fig.tight_layout()
    if fig_params['savefig']:
        fig.savefig(fig_params['savefig_path'])

"""
I can iterate through 
- neuron_results (exc_neuron, inh_neuron)
- tf_funcs or mf_results_list


"""


def make_fig_per_col(neuron_results, tf_funcs, snn_results, mf_results_list, plots, fig_params):
    """Make a basic figure with the given plots."""
    plt.rcParams['font.size'] = fig_params['fontsize']

    rows = len(plots)
    cols = 1 

    figsize = (10, 3 * rows) if fig_params['figsize'] is None else fig_params['figsize']
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for row, plot in enumerate(plots):
        match plot:
            case ax_plt.BaseSingleNeuronPlot():
                plot.draw(axes[row, 0], neuron_results)
            case ax_plt.BaseTransferFunctionPlot():
                plot.draw(axes[row, 0], neuron_results, tf_funcs)
            case ax_plt.BaseSNNPlot():
                plot.draw(axes[row, 0], snn_results)
            case ax_plt.BaseNetworkPlot():
                plot.draw(axes[row, 0], [snn_results] + mf_results_list)

    if fig_params['tight_layout']:
        fig.tight_layout()
    if fig_params['savefig']:
        fig.savefig(fig_params['savefig_path'])

def make_fig_per_row(neuron_results, tf_funcs, snn_results, mf_results_list, plots, fig_params):
    """Make a basic figure with the given plots."""
    plt.rcParams['font.size'] = fig_params['fontsize']

    rows = len(plots)
    cols = 1 

    figsize = (10, 3 * rows) if fig_params['figsize'] is None else fig_params['figsize']
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for row, plot in enumerate(plots):
        match plot:
            case ax_plt.BaseSingleNeuronPlot():
                plot.draw(axes[row, 0], neuron_results)
            case ax_plt.BaseTransferFunctionPlot():
                plot.draw(axes[row, 0], neuron_results, tf_funcs)
            case ax_plt.BaseSNNPlot():
                plot.draw(axes[row, 0], snn_results)
            case ax_plt.BaseNetworkPlot():
                plot.draw(axes[row, 0], [snn_results] + mf_results_list)

    if fig_params['tight_layout']:
        fig.tight_layout()
    if fig_params['savefig']:
        fig.savefig(fig_params['savefig_path'])
