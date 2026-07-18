import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from ..data_structures.base import BaseSingleNeuronResults, BaseMFResults, BaseSNNResults, BaseInspectionResults
from ..data_structures.inspection import ModelComparisonInspectionResults
from ..transfer_function.base import BaseTransferFunction

from ..controller.interfaces import BasicWorkflowHook, InspectionWorkflowHook

from .base import (BasePlot, BaseSingleNeuronPlot, BaseTransferFunctionPlot, BaseSNNPlot, 
                   BaseNetworkPlot, BaseNetworkHistogramPlot, BaseInspectionPlot)

from . import neuron_plots, tf_plots, snn_plots, network_plots, inspection_plots


class GridFigureHook:
    """
    A generic workflow hook that constructs a multi-panel figure based on a 
    provided 2D grid of BasePlot objects. Satisfies the BasicWorkflowHook protocol.
    """
    
    DEFAULT_FIG_PARAMS = {
        'fontsize': 14,
        'dpi': 100,
        'axsize': (8, 5),  # Default size for each subplot
        'figsize': None,  # If not specified, it will be calculated from 'axsize'
        'title': None,  # Default title is None
        'savefig': True,  # Save figure by default
        'savefig_path': None,  # Path to save the figure
        'gridspec_kw': {},
        # 'tight_layout': True,  # Deprecated in favor of constrained_layout
        'constrained_layout': True,  # Use constrained layout by default
        'bbox_inches': None,  # Save figure with tight bounding box
    }

    def __init__(
            self, 
            plot_grid: List[List[BasePlot]], 
            savefig_dir: Path,
            fig_file_prefix: str,
            fig_params: dict = None,
            common_params: dict = None,
            subplot_params: dict = None,
            ):
        """
        Parameters
        ----------
        plot_grid : List[List[BasePlot]]
            A 2D list representing the row/column layout of the figure.
        savefig_dir : Path
            Directory to save the resulting figure.
        fig_prefix : str
            Prefix for the filename (e.g., 'network_overview').
        fig_params : dict
            Overrides for figure-level parameters.
        common_params : dict
            Common parameters for all plots in the grid.
        subplot_params : dict
            Parameters for each subplot.
        """
        
        # Validate the grid structure
        assert all(len(row) == len(plot_grid[0]) for row in plot_grid), "All rows in plot_grid must have the same number of columns."
        
        self.plot_grid = plot_grid
        self.savefig_dir = Path(savefig_dir)
        self.fig_file_prefix = fig_file_prefix
        self.fig_params = {**self.DEFAULT_FIG_PARAMS, **(fig_params or {})}
        self.common_params = common_params or {}
        self.subplot_params = subplot_params or {}


        if 'tight_layout' in self.fig_params:
            print("Warning: 'tight_layout' is deprecated. 'constrained_layout' is used instead.")
        

    def __call__(
            self,
            identifier: str,
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            tf_funcs_results: Dict[str, List[BaseTransferFunction]],
            snn_results: BaseSNNResults,
            network_results_list: List[BaseSNNResults | BaseMFResults],
            inspection_results_list: List[BaseInspectionResults],
            ) -> None:
        
        self.rows = len(self.plot_grid)
        self.cols = len(self.plot_grid[0])

        plt.rcParams['font.size'] = self.fig_params['fontsize']
        
        col_size, row_size= self.fig_params['axsize']
        figsize = self.fig_params['figsize'] or (col_size * self.cols, row_size * self.rows)
        
        fig, axes = plt.subplots(
            self.rows, self.cols, 
            figsize=figsize, 
            squeeze=False, 
            constrained_layout=self.fig_params['constrained_layout'],
            gridspec_kw=self.fig_params.get('gridspec_kw', {})
        )


        # 2. Route Data to Subplots via Pattern Matching
        for row in range(self.rows):
            for col in range(self.cols):
                plot = self.plot_grid[row][col]
                ax = axes[row, col]

                # Apply subplot-specific overrides
                class_name = plot.__class__.__name__
                if class_name in self.subplot_params:
                    plot.full_params.update(self.subplot_params[class_name])
                coord = (row, col)
                if coord in self.subplot_params:
                    plot.full_params.update(self.subplot_params[coord])

                if isinstance(plot, BaseSingleNeuronPlot):
                    im = plot.draw(ax, neuron_results=neuron_results)
                    if im is not None:
                        plot.add_colorbar(fig, ax, im)
                elif isinstance(plot, BaseTransferFunctionPlot):
                    plot.draw(ax, neuron_results=neuron_results, tf_funcs_results=tf_funcs_results)
                elif isinstance(plot, BaseSNNPlot):
                    plot.draw(ax, snn_results=snn_results)
                elif isinstance(plot, (BaseNetworkPlot, BaseNetworkHistogramPlot)):
                    plot.draw(ax, network_results_list=network_results_list)
                elif isinstance(plot, BaseInspectionPlot):
                    plot.draw(ax, inspection_results_list=inspection_results_list)
                else:
                    raise TypeError(f"Unknown plot type in grid: {type(plot)}")

        # 3. Finalize and Save
        if self.fig_params['title']:
            fig.suptitle(self.fig_params['title'] + f" - {identifier}")

        if self.fig_params['savefig']:
            safe_identifier = identifier.replace(" ", "_")
            filepath = self.savefig_dir / f"{self.fig_file_prefix}_{safe_identifier}.png"
            fig.savefig(filepath, dpi=self.fig_params['dpi'], bbox_inches=self.fig_params['bbox_inches'])
            
        plt.close(fig)


class BasicWorkflowPlottingHook(GridFigureHook):
    
    
    def __call__(
            self,
            identifier: str,
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            tf_funcs_results: Dict[str, List[BaseTransferFunction]],
            snn_results: BaseSNNResults,
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:
        super().__call__(
            identifier=identifier,
            neuron_results=neuron_results,
            tf_funcs_results=tf_funcs_results,
            snn_results=snn_results,
            network_results_list=network_results_list,
            inspection_results_list=None  # Not used in basic workflow
        )

class InspectionWorkflowPlottingHook(GridFigureHook):

    def __call__(
            self,
            identifier: str,
            inspection_results_list: List[BaseInspectionResults],
            ) -> None:
        super().__call__(
            identifier=identifier,
            neuron_results=None,
            tf_funcs_results=None,
            snn_results=None,
            network_results_list=None,
            inspection_results_list=inspection_results_list
        )


class NeuronActivityHook(BasicWorkflowPlottingHook):

    def __init__(
            self, 
            savefig_dir: Path,
            fig_file_prefix: str,
            neuron_names: List[str],
            fig_params: dict = None,
            common_params: dict = None,
            subplot_params: dict = None,
            ):

        plot_grid = [
            [
                neuron_plots.SingleNeuronActivityHeatmapPlot({
                    'levels' : 10,
                    **common_params,
                    'neuron_name': neuron_name,
                    'title': f"{neuron_name} Activity Heatmap",
                }),
                neuron_plots.SingleNeuronActivityPlot({
                    'xmargin': 0.0,
                    'ymargin': 0.0,
                    'legend': True,
                    'curves_num' : 7,
                    'linestyle' : 'None',
                    'yerrorbar' : True,
                    'capsize' : 3,
                    **common_params,
                    'title': f"{neuron_name} Activity vs. Input Rate",
                    'neuron_name': neuron_name,
                })
            ] for neuron_name in neuron_names
        ]

        super().__init__(
            plot_grid=plot_grid, 
            savefig_dir=savefig_dir, 
            fig_file_prefix=fig_file_prefix, 
            fig_params=fig_params,
            common_params=common_params,
            subplot_params=subplot_params
        )

class TransferFunctionPlottingHook(BasicWorkflowPlottingHook):
    def __init__(
            self, 
            savefig_dir: Path,
            fig_file_prefix: str,
            neuron_names: List[str],
            fig_params: dict = None,
            common_params: dict = None,
            subplot_params: dict = None,
            ):

        plot_grid = [
            [
                tf_plots.TransferFunctionFitPlot({
                    'markersize' : 5,
                    'ylim' : (None, 30),
                    'xmargin' : 0.0,
                    'ymargin' : 0.0,
                    'legend' : True,
                    'curves_num' : 10,
                    'xmargin' : 0.0,
                    'ymargin' : 0.0,
                    'yerrorbar' : True,
                    **common_params,
                    'neuron_name': neuron_name,
                    'title': f"{neuron_name} Transfer Function"
                })
            ] for neuron_name in neuron_names
        ]

        super().__init__(
            plot_grid=plot_grid, 
            savefig_dir=savefig_dir, 
            fig_file_prefix=fig_file_prefix, 
            fig_params=fig_params,
            common_params=common_params,
            subplot_params=subplot_params
        )


class NetworkOverviewPlottingHook(BasicWorkflowPlottingHook):
    """
    A hook for plotting an overview of the network during the simulation workflow.
    """

    def __init__(
            self, 
            savefig_dir: Path,
            fig_file_prefix: str,
            fig_params: dict = None,
            common_params: dict = None,
            subplot_params: dict = None,
            ):



        plot_grid = [
            [
                snn_plots.SpikeRasterPlot({
                    'markersize': 7,
                    **common_params,
                    'xticks' : [],
                    'xticks_labels' : None,
                    'xlabel' : None,
                    'title' : None,
                    'legend' : False,
                })
            ],
            [
                network_plots.FiringRatePlot({
                    'ylim': (0, 15),
                    **common_params,
                    'xticks_labels' : None,
                    'xticks' : [],
                    'xlabel' : None,
                    'title' : None,
                })
            ],
            [
                network_plots.StimulusPlot({
                    **common_params,
                    'ylim': (0, 4),
                    'labels': None,
                    'xticks_labels' : None,
                    'xticks' : [],
                    'xlabel' : None,
                    'title' : None,
                })
            ],
            [
                network_plots.AdaptationPlot({
                    **common_params,
                    'xticks_labels' : None,
                    'xticks' : [],
                    'xlabel' : None,
                    'title' : None,
                })
            ],
            [
                network_plots.VoltagePlot({
                    **common_params,
                    'xticks_labels' : None,
                    'xticks' : [],
                    'xlabel' : None,
                    'title' : None,
                    'ylim' : (-60, -54)
                    
                }),                
            ],
            [
                network_plots.STPVariableXPlot({
                    **common_params,
                    'xticks_labels' : None,
                    'xticks' : [],
                    'xlabel' : None,
                    'title' : None,
                    'ylim' : (0, 1)
                })
            ],
            [
                network_plots.STPVariableUPlot({
                    **common_params,
                    'title' : None,
                    'ylim' : (0, 1)
                })
            ]
        ]

        super().__init__(
            plot_grid=plot_grid, 
            savefig_dir=savefig_dir, 
            fig_file_prefix=fig_file_prefix, 
            fig_params=fig_params,
            common_params=common_params,
            subplot_params=subplot_params
        )

class NetworkHistogramPlottingHook(BasicWorkflowPlottingHook):
    def __init__(
            self, 
            savefig_dir: Path,
            fig_file_prefix: str,
            fig_params: dict = None,
            common_params: dict = None,
            subplot_params: dict = None,
            ):



        plot_grid = [
            [
                network_plots.FiringRateHistogramPlot({
                    **common_params,
                    # 'binsize': 0.5,
                }),
                network_plots.VoltageHistogramPlot({
                    **common_params,
                    # 'binsize': 0.4,
                }),
                network_plots.AdaptationHistogramPlot({
                    **common_params,
                    # 'binsize': 0.002,
                }),
                network_plots.ExcitatoryNeuronConductanceHistogramPlot({
                    **common_params,
                    # 'binsize': 0.0001,
                }),
                network_plots.InhibitoryNeuronConductanceHistogramPlot({
                    **common_params,
                    # 'binsize': 0.0001,
                    
                }),
                network_plots.STPVariableXHistogramPlot({
                    **common_params,
                }),
                network_plots.STPVariableUHistogramPlot({
                    **common_params,
                }),
            ]
        ]

        super().__init__(
            plot_grid=plot_grid, 
            savefig_dir=savefig_dir, 
            fig_file_prefix=fig_file_prefix, 
            fig_params=fig_params,
            common_params=common_params,
            subplot_params=subplot_params
        )



class ModelSummaryInspectionPlottingHook(InspectionWorkflowPlottingHook):

    def __init__(
            self, 
            savefig_dir: Path,
            fig_file_prefix: str,
            fig_params: dict,
            common_params: dict,
            subplot_params: dict = None,
            ):

        plot_grid=[
            [
                inspection_plots.FiringRateInspectionPlot({
                    **common_params,
                }),
                inspection_plots.VoltageInspectionPlot({
                    **common_params,
                }),
                inspection_plots.AdaptationInspectionPlot({
                    **common_params,
                }),
            ]
        ]

        super().__init__(
            plot_grid=plot_grid, 
            savefig_dir=savefig_dir, 
            fig_file_prefix=fig_file_prefix, 
            fig_params=fig_params,
            common_params=common_params,
            subplot_params=subplot_params
        )

class ModelComparisonInspectionPlottingHook(InspectionWorkflowPlottingHook):
    MEASURE_SUFFIXES = ('rmse', 'error_mean', 'error_std', 'pearson')
    MEASURE_TITLES = {
        'rmse': r'RMSE : $\sqrt{1/T\int (SNN-MF)^2}$',
        'error_mean': r'Error mean : $1/T\int (SNN-MF))$',
        'error_std': 'Error std',
        'pearson': 'Pearson',
    }

    def _fotmat_label(self, label: str, mapping: dict=None) -> str:
        if mapping is None:
            mapping = {}
        return mapping.get(label, label.replace('_', ' ').title())

    @staticmethod
    def _format_variable_label(variable_name: str) -> str:
        return variable_name.replace('_', ' ').title()

    @classmethod
    def _format_measure_label(cls, measure_name: str) -> str:
        return cls.MEASURE_TITLES.get(measure_name, measure_name.replace('_', ' ').title())

    def __init__(
            self, 
            savefig_dir: Path,
            fig_file_prefix: str,
            fig_params: dict,
            common_params: dict,
            subplot_params: dict = None,
            ):

        super().__init__(
            plot_grid=[[]], 
            savefig_dir=savefig_dir, 
            fig_file_prefix=fig_file_prefix, 
            fig_params=fig_params,
            common_params=common_params,
            subplot_params=subplot_params
        )

    def prepare_plot_grid(self, inspection_results: BaseInspectionResults):

        row_measures: list[str] = inspection_results.metrics
        column_variables: list[str] = inspection_results.variables


        plot_grid = [
            [
                inspection_plots.MetricCustomInspectionPlot(
                    f'{variable_name}_{measure_name}',
                    {
                        'title': self._fotmat_label(variable_name) if row_idx == 0 else None,
                        'ylabel': self._fotmat_label(measure_name, self.MEASURE_TITLES) if col_idx == 0 else None,
                        # 'xlabel': inspection_results.inspected_param if row_idx == len(row_measures) - 1 else None,
                        **self.common_params,
                        
                    },
                )
                for col_idx, variable_name in enumerate(column_variables)
            ]
            for row_idx, measure_name in enumerate(row_measures)
        ]

        self.plot_grid = plot_grid

    def __call__(
            self,
            identifier: str,
            inspection_results_list: List[BaseInspectionResults],
            ) -> None:

        inspection_results_list = [result for result in inspection_results_list if isinstance(result, ModelComparisonInspectionResults)]

        if len(inspection_results_list) != 1:
            raise ValueError(f"Expected exactly one ModelComparisonInspectionResults, but found {len(inspection_results_list)}.")

        inspection_results = inspection_results_list[0]

        self.prepare_plot_grid(inspection_results)
            
        GridFigureHook.__call__(
            self,
            identifier=identifier,
            neuron_results=None,
            tf_funcs_results=None,
            snn_results=None,
            network_results_list=None,
            inspection_results_list=[inspection_results],
        )

