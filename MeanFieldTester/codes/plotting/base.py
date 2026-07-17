from abc import ABC, abstractmethod
from matplotlib.lines import Line2D
import copy

from typing import List, Dict
import numpy as np

from ..data_structures.base import BaseMFResults, BaseSNNResults, BaseSingleNeuronResults, BaseInspectionResults
from ..transfer_function.base import BaseTransferFunction

EXC_COLOR = "green"
INH_COLOR = "red"
LINESTYLES = ['-', '--', '-.', ':']
MAX_NTW_ACTIVITY = 200  # Hz
RASTER_EXC_CELLS = 400
RASTER_INH_CELLS = 100


BINS = 10  # [ms], size for making histograms
BIN_SIZE = 5  # [ms], size for averiging activity in histograms

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
        'x_unit': None,  # Default x-axis unit
        'y_unit': None,  # Default y-axis unit
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
            inspection_results: List[BaseInspectionResults],
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

        xlabel = BasePlot.format_label_with_unit(params['xlabel'], params['x_unit'])
        ylabel = BasePlot.format_label_with_unit(params['ylabel'], params['y_unit'])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

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

        zlabel = self.full_params.get('colorbar_label', None)
        zunit = self.full_params.get('z_unit', None)
        label = BasePlot.format_label_with_unit(zlabel, zunit)

        if label:
            cbar.set_label(label)

@staticmethod
def format_label_with_unit(label: str, unit: str) -> str:
    if not label or not unit:
        return label
    import re
    # Matches suffixes like "(Hz)", "[ms]", " (nA)" at the end of the string
    pattern = r'\s*[\(\[][a-zA-Z\^\d/_-]+[\)\]]$'
    if re.search(pattern, label):
        return re.sub(pattern, f" [{unit}]", label)
    return f"{label} [{unit}]"

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

    def iter_results(self, results_list: list):
        """Update dynamic params and yield each result with its style metadata."""
        self.update_params(results_list)
        return zip(results_list, self.full_params['linestyles'], self.full_params['labels'])

    def plot_single_series(self, ax, results, linestyle, label, getter, *, color, prefix=None, alpha=1.0):
        values = getter(results)
        if values is None:
            return

        series_label = f"{prefix} {label}" if prefix is not None else label
        ax.plot(results.times(self.full_params['x_unit']), values, label=series_label, ls=linestyle, color=color, alpha=alpha)

    def plot_pair_series(
            self,
            ax,
            results,
            linestyle,
            label,
            exc_getter,
            inh_getter,
            *,
            exc_color,
            inh_color,
            exc_prefix='Exc',
            inh_prefix='Inh',
            alpha=1.0,
            exc_std_getter=None,
            inh_std_getter=None,
            band_alpha=0.3,
        ):
        times = results.times(self.full_params['x_unit'])
        exc_values = exc_getter(results)
        if exc_values is not None:
            ax.plot(times, exc_values, label=f'{exc_prefix} {label}', ls=linestyle, color=exc_color, alpha=alpha)

        inh_values = inh_getter(results)
        if inh_values is not None:
            ax.plot(times, inh_values, label=f'{inh_prefix} {label}', ls=linestyle, color=inh_color, alpha=alpha)

        if (
            exc_std_getter is not None
            and inh_std_getter is not None
            and isinstance(results, BaseMFResults)
            and linestyle != 'None'
        ):
            exc_std = exc_std_getter(results)
            inh_std = inh_std_getter(results)

            if exc_values is not None and exc_std is not None:
                ax.fill_between(times, exc_values - exc_std, exc_values + exc_std, color=exc_color, alpha=band_alpha)
            if inh_values is not None and inh_std is not None:
                ax.fill_between(times, inh_values - inh_std, inh_values + inh_std, color=inh_color, alpha=band_alpha)


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
        'time_unit': 'ms',  # Time unit for the histogram
    }

    def update_params(self, results_list:list):
        super().update_params(results_list)

        if self.full_params['start_time'] is None:
            self.full_params['start_time'] = max([results.times(self.full_params['time_unit'])[0] for results in results_list])

        if self.full_params['end_time'] is None:
            self.full_params['end_time'] = min([results.times(self.full_params['time_unit'])[-1] for results in results_list])

        if self.full_params['bins'] is None and self.full_params['binsize'] is None:
            print(f"Warning: Neither 'bins' nor 'binsize' specified for histogram plots. Using default number of bins {BINS}.")
            self.full_params['bins'] = BINS
        elif self.full_params['bins'] is not None and self.full_params['binsize'] is not None:
            raise ValueError("Only one of 'bins' or 'binsize' should be specified for histogram plots.")

    def get_bin_edges(self, data: List[np.ndarray]) -> np.ndarray:
        """Calculate bin edges based on the data and the specified bins or binsize."""

        x_min, x_max = self.full_params['xlim']

        if x_min is not None:
            data_min = x_min
        else:
            data_min = min(np.min(d) for d in data)

        if x_max is not None:
            data_max = x_max
        else:
            data_max = max(np.max(d) for d in data)
        
        if self.full_params['binsize'] is not None:
            num_bins = int(np.ceil((data_max - data_min) / self.full_params['binsize']))
        else:
            num_bins = self.full_params['bins']

        bin_edges = np.linspace(data_min, data_max, num_bins + 1)

        return bin_edges

    def plot_hist_pair(
            self,
            ax,
            exc_values,
            inh_values,
            label,
            linestyle,
            *,
            exc_color,
            inh_color,
            exc_weights=None,
            inh_weights=None,
            density=False,
        ):
        bin_edges = self.get_bin_edges([exc_values, inh_values])

        ax.hist(
            exc_values,
            bins=bin_edges,
            alpha=0.5,
            label=f'Exc {label}',
            edgecolor=exc_color,
            color=exc_color,
            linestyle=linestyle,
            density=density,
            weights=exc_weights,
        )
        ax.hist(
            inh_values,
            bins=bin_edges,
            alpha=0.5,
            label=f'Inh {label}',
            edgecolor=inh_color,
            color=inh_color,
            linestyle=linestyle,
            density=density,
            weights=inh_weights,
        )

    def plot_hist_lines(self, ax, exc_mean, inh_mean, label, linestyle, *, exc_color, inh_color):
        ax.axvline(exc_mean, label=f'Exc {label}', color=exc_color, linestyle=linestyle)
        ax.axvline(inh_mean, label=f'Inh {label}', color=inh_color, linestyle=linestyle)

    def plot_mf_hist_pair(
            self,
            ax,
            exc_mean,
            inh_mean,
            exc_std,
            inh_std,
            label,
            linestyle,
            *,
            exc_color,
            inh_color,
            normalization=False,
            density=False,
        ):
        x = np.linspace(0, max(exc_mean + 4 * exc_std, inh_mean + 4 * inh_std), 100)
        dx = x[1] - x[0]

        exc_gauss = np.exp(-0.5 * ((x - exc_mean) / exc_std) ** 2)
        if normalization:
            exc_gauss /= exc_gauss.max()
        if density:
            exc_gauss /= exc_gauss.sum() * dx

        inh_gauss = np.exp(-0.5 * ((x - inh_mean) / inh_std) ** 2)
        if normalization:
            inh_gauss /= inh_gauss.max()
        if density:
            inh_gauss /= inh_gauss.sum() * dx

        ax.plot(x, exc_gauss, label=f'Exc {label}', color=exc_color, linestyle=linestyle)
        ax.plot(x, inh_gauss, label=f'Inh {label}', color=inh_color, linestyle=linestyle)

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
    'linestyles' : None,
    'normalization' : False,
    'density' : False,
    }
    # NOTE: None params are updated later in the update_params method

    inspection_results_type = BaseInspectionResults  # Default type, can be overridden in subclasses

    def filter_results(self, inspection_results_list: List[BaseInspectionResults]) -> List[BaseInspectionResults]:
        """Filter the inspection results to only include those that are instances of BaseInspectionResults."""
        filtered_results = [result for result in inspection_results_list if isinstance(result, self.inspection_results_type)]
        if len(filtered_results) != 1:
            raise ValueError(f"Expected exactly one {self.inspection_results_type.__name__}, but found {len(filtered_results)}.")
        return filtered_results

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
            inspection_results_list: List[BaseInspectionResults],
            ) -> None:
        pass