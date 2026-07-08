from abc import ABC, abstractmethod
from matplotlib.lines import Line2D
import copy

from typing import List, Dict

from ..data_structures.base import BaseMFResults, BaseSNNResults, BaseSingleNeuronResults, BaseInspectionResults
from ..transfer_function.base import BaseTransferFunction

EXC_COLOR = "green"
INH_COLOR = "red"
LINESTYLES = ['-', '--', '-.', ':']
MAX_NTW_ACTIVITY = 200  # Hz
RASTER_EXC_CELLS = 400
RASTER_INH_CELLS = 100


BIN_SIZE = 5  # [ms], size for making histograms

NEURON_NAMES = ["exc_neuron", "inh_neuron"]


class BasePlot(ABC):
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
        'linewidth': 2.0,
    }

    def __init__(self, params=None):
        self.params = copy.deepcopy(params) or {}
        self.full_params = copy.deepcopy(self.DEFAULT_PARAMS)
        if self.params:
            self.full_params.update(self.params) # Instance params override defaults    

    def draw(self, ax, **data):
        self.apply_preplot_params(ax, self.full_params)
        im = self._draw(ax, **data)
        self.apply_postplot_params(ax, self.full_params)
        return im

    @abstractmethod
    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            tf_funcs_results: Dict[str, List[BaseTransferFunction]],
            snn_results: BaseSNNResults,
            network_results_list: List[BaseSNNResults | BaseMFResults],
            inspection_results: BaseInspectionResults,
            ) -> None:
        pass

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

    def add_colorbar(self, fig, ax, im):
        cbar = fig.colorbar(im, ax=ax)

        label = self.full_params.get('colorbar_label', None)
        if label:
            cbar.set_label(label)

class BaseSingleNeuronPlot(BasePlot, ABC):
    """This is a meta class for single neuron plots, can be used to plot single neuron results."""

    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'neuron_name' : None,
    }

    @abstractmethod
    def _draw(
            self, 
            ax, 
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            ) -> None:
        pass


class BaseTransferFunctionPlot(BasePlot, ABC):
    """Base class for plots based on transfer function results."""
    
    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        'labels' : None,
        'linestyles' : None,
        'neuron_name' : None,
    }

    @abstractmethod
    def _draw(
            self,
            ax,
            neuron_results: Dict[str, BaseSingleNeuronResults],
            tf_funcs_results: Dict[str, List[BaseTransferFunction]],
            ) -> None:
        pass

class BaseSNNPlot(BasePlot, ABC):
    """Base class for plots base solely on results of Spiking Neural Network.

    This class works only with data in helper.BaseSNNResults
    """

    @abstractmethod
    def _draw(
            self, 
            ax, 
            snn_results: BaseSNNResults,
            ) -> None:
        pass

class BaseNetworkPlot(BasePlot, ABC):
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
            self.full_params['linestyles'] = [LINESTYLES[i % len(LINESTYLES)] for i in range(len(results_list))]

        if len(results_list) > 2:
            legend_elements = [Line2D([0], [0], color='black', label=label, linestyle=ls) for label, ls in zip(self.full_params['labels'], self.full_params['linestyles'])]
            if self.full_params['legend'] is True:
                self.full_params['legend'] = {'handles': legend_elements}
            elif type(self.full_params['legend'] ) is dict:
                self.full_params['legend']['handles'] = legend_elements


    @abstractmethod
    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        pass


class BaseNetworkHistogramPlot(BaseNetworkPlot, ABC):
    """Base class for histogram plots based on network result, can be used to plot network results."""
    DEFAULT_PARAMS = {
        **BaseNetworkPlot.DEFAULT_PARAMS,
        'ylabel': 'Count',
        'bins' : None,
        'binsize': None,  # Size of the bins for the histogram
        'start_time': None,  # Start time for the histogram
        'end_time': None,  # End time for the histogram
    }

    def update_params(self, results_list:list):
        super().update_params(results_list)

        if self.full_params['start_time'] is None:
            self.full_params['start_time'] = max([results.times()[0] for results in results_list])

        if self.full_params['end_time'] is None:
            self.full_params['end_time'] = min([results.times()[-1] for results in results_list])

        if self.full_params['bins'] is None and self.full_params['binsize'] is None:
            print(f"Warning: Neither 'bins' nor 'binsize' specified for histogram plots. Using default binsize of {BIN_SIZE} ms.")
            self.full_params['binsize'] = BIN_SIZE
        elif self.full_params['bins'] is not None and self.full_params['binsize'] is not None:
            raise ValueError("Only one of 'bins' or 'binsize' should be specified for histogram plots.")

    @abstractmethod
    def _draw(
            self, 
            ax, 
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        pass

class BaseInspectionPlot(BasePlot, ABC):
    """Base class for plots based on inspection results, can be used to plot inspection results."""

    DEFAULT_PARAMS = {
    **BasePlot.DEFAULT_PARAMS,
    'labels' : None,
    'markers' : None,
    'linestyles' : None
    }
    # NOTE: None params are updated later in the update_params method

    def update_params(self, inspection_results:BaseInspectionResults):
        """Some parameters cannot be generater until the results_list is known, so we update them here."""
        num_networks = len(inspection_results.network_names)

        if self.full_params['labels'] is None:
            self.full_params['labels'] = inspection_results.network_names
       
        if self.full_params['linestyles'] is None:
            # cycles through the predefined LINESTYLES
            self.full_params['linestyles'] = [LINESTYLES[i % len(LINESTYLES)] for i in range(num_networks)]

        if self.full_params['markers'] is None:
            self.full_params['markers'] = ['o' if name.startswith("SNN") else "None" for name in inspection_results.network_names]

        if num_networks > 2:
            legend_elements = [
                Line2D([0], [0], color='black', label=label, linestyle=ls, marker=marker) 
                for label, ls, marker in zip(self.full_params['labels'], self.full_params['linestyles'], self.full_params['markers'])
            ]
            if self.full_params['legend'] is True:
                self.full_params['legend'] = {'handles': legend_elements}
            elif type(self.full_params['legend'] ) is dict:
                self.full_params['legend']['handles'] = legend_elements
        
        if self.full_params['xlabel'] is None:
            self.full_params['xlabel'] = inspection_results.inspected_param

    @abstractmethod
    def _draw(
            self, 
            ax, 
            inspection_results: BaseInspectionResults,
            ) -> None:
        pass