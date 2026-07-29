import os
import copy
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Union, Callable
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

from .base import BasePlot, EXC_COLOR, INH_COLOR, LINESTYLES


class BaseAggregatorPlot(BasePlot, ABC):
    """
    Abstract base class for plotters that draw ResultsAggregator simulation data onto a single ax.
    Extends BasePlot, maintaining full compatibility with DEFAULT_PARAMS, full_params,
    apply_preplot_params, and apply_postplot_params.
    """

    DEFAULT_VARIABLES = []

    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        "colors": None,      # dict or list of colors
        "linestyles": None,  # dict or list of linestyles
        "labels": None,      # list of model/curve labels
        "alphas": None,      # dict or list of alpha values, if None uses default_alpha
        "default_alpha": 1.0,
        "default_color": "black",
    }

    def __init__(
        self,
        stim_name: str,
        variables: str | List[str] | None = None,
        models: List[str] | None = None,
        params: dict = None,
    ):
        """
        
        Parameters
        -----------
        stim_name: str
            Name of the stimulus to plot.
        variables: str or list of str or None
            Variable names to plot. 
            If None, uses DEFAULT_VARIABLES.
        models: list of str
            List of model names to plot. 
            If None, plots all available models.
        params: dict
            Dictionary of plotting parameters to override defaults.
        """
        super().__init__(params=params)

        if variables is not None:
            self.variables = [variables] if isinstance(variables, str) else list(variables)
        else:
            self.variables = list(self.DEFAULT_VARIABLES)

        self.models = models
        self.stim_name = stim_name

    def draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs):
        self.apply_preplot_params(ax, self.full_params)
        im = self._draw(ax, sim_id=sim_id, aggregator=aggregator)
        self.apply_postplot_params(ax, self.full_params)
        return im

    @abstractmethod
    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        pass


class AggregatorTracePlot(BaseAggregatorPlot):
    """Plotter for 1D time-series trace variables across models for a single aggregator run."""

    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        "colors": None,      # dict or list of colors
        "linestyles": None,  # dict or list of linestyles
        "labels": None,      # list of model/curve labels
        "alphas": None,      # dict or list of alpha values, if None uses default_alpha
        "linewidths": None,  # dict or list of linewidths
        "default_alpha": 1.0,
        "default_color": "black",
        "default_linewidth": 1.5,
    }

    def update_params(self, target_models: List[str]):
        """Ensures labels and linestyles are populated for target_models.
        
        Updates
        - labels
        - linestyles
        - alphas
        - linewidths

        """
        num_models = len(target_models)

        if self.full_params["labels"] is None:
            self.full_params["labels"] = {m : f"{m}" for m in target_models}
        elif isinstance(self.full_params["labels"], (list, tuple)):
            if len(self.full_params["labels"]) != num_models:
                raise ValueError(f"Length of 'labels' ({len(self.full_params['labels'])}) does not match number of target models ({num_models}).")
            self.full_params["labels"] = dict(zip(target_models, self.full_params["labels"]))
        elif isinstance(self.full_params["labels"], dict):
            for m in target_models:
                if m not in self.full_params["labels"]:
                    self.full_params["labels"][m] = f"{m}"
        else:
            raise TypeError(f"'labels' must be a list, tuple, or dict, got {type(self.full_params['labels'])}.")

        if self.full_params["linestyles"] is None:
            self.full_params["linestyles"] = {m: LINESTYLES[i % len(LINESTYLES)] for i, m in enumerate(target_models)}
        elif isinstance(self.full_params["linestyles"], (list, tuple)):
            if len(self.full_params["linestyles"]) != num_models:
                raise ValueError(f"Length of 'linestyles' ({len(self.full_params['linestyles'])}) does not match number of target models ({num_models}).")
            self.full_params["linestyles"] = dict(zip(target_models, self.full_params["linestyles"]))
        elif isinstance(self.full_params["linestyles"], dict):
            for m in target_models:
                if m not in self.full_params["linestyles"]:
                    idx = target_models.index(m)
                    self.full_params["linestyles"][m] = LINESTYLES[idx % len(LINESTYLES)]
        else:
            raise TypeError(f"'linestyles' must be a list, tuple, or dict, got {type(self.full_params['linestyles'])}.")

        if self.full_params["alphas"] is None:
            self.full_params["alphas"] = {m: self.full_params["default_alpha"] for m in target_models}
        elif isinstance(self.full_params["alphas"], (list, tuple)):
            if len(self.full_params["alphas"]) != num_models:
                raise ValueError(f"Length of 'alphas' ({len(self.full_params['alphas'])}) does not match number of target models ({num_models}).")
            self.full_params["alphas"] = dict(zip(target_models, self.full_params["alphas"]))
        elif isinstance(self.full_params["alphas"], dict):
            for m in target_models:
                if m not in self.full_params["alphas"]:
                    self.full_params["alphas"][m] = self.full_params["default_alpha"]
        else:
            raise TypeError(f"'alphas' must be a list, tuple, or dict, got {type(self.full_params['alphas'])}.")

        if self.full_params["linewidths"] is None:
            self.full_params["linewidths"] = {m: self.full_params["default_linewidth"] for m in target_models}
        elif isinstance(self.full_params["linewidths"], (list, tuple)):
            if len(self.full_params["linewidths"]) != num_models:
                raise ValueError(f"Length of 'linewidths' ({len(self.full_params['linewidths'])}) does not match number of target models ({num_models}).")
            self.full_params["linewidths"] = dict(zip(target_models, self.full_params["linewidths"]))
        elif isinstance(self.full_params["linewidths"], dict):
            for m in target_models:
                if m not in self.full_params["linewidths"]:
                    self.full_params["linewidths"][m] = self.full_params["default_linewidth"]
        else:
            raise TypeError(f"'linewidths' must be a list, tuple, or dict, got {type(self.full_params['linewidths'])}.")


        if num_models > 2:
            legend_elements = [Line2D([0], [0], color='black', label=self.full_params["labels"][model], linestyle=self.full_params["linestyles"][model]) for model in target_models]
            if self.full_params['legend'] is True:
                self.full_params['legend'] = {'handles': legend_elements}
            elif type(self.full_params['legend'] ) is dict:
                self.full_params['legend']['handles'] = legend_elements


    def get_variable_color(self, var_name: str) -> str:
        """Determines the color for a given variable name or index."""

        colors = self.full_params["colors"]
        if isinstance(colors, (list, tuple)) and colors:
            return dict(zip(self.variables, colors)).get(var_name)

        if isinstance(colors, dict):
            if var_name in colors:
                return colors[var_name]

        var_lower = var_name.lower()
        if var_lower.startswith("exc"):
            return self.full_params["exc_color"]
        elif var_lower.startswith("inh"):
            return self.full_params["inh_color"]

        return self.full_params["default_color"]


    def iter_models(self, aggregator, sim_id: str):
        """Yields (model, times, model_style_dict) for available models."""
        available_models = list(aggregator.get_available_variables().keys())
        if self.models is None:
            target_models = available_models
        else:
            target_models = [m for m in self.models if m in available_models]
            if len(target_models) == 0:
                raise ValueError(f"No requested models {self.models} are available in the aggregator for sim_id '{sim_id}'. Available models: {available_models}")
            missing_models = [m for m in self.models if m not in available_models]
            if missing_models:
                raise ValueError(f"The following requested models are not available in the aggregator for sim_id '{sim_id}': {missing_models}. Only plotting available models: {target_models}")
            
        self.update_params(target_models)

        for model in target_models:
            times = aggregator._load_variable(sim_id, model, self.stim_name, "times")

            model_style = {
                "linestyle": self.full_params["linestyles"][model],
                "linewidth": self.full_params["linewidths"][model],
                "alpha": self.full_params["alphas"][model],
                "label": self.full_params["labels"][model],
            }

            yield model, times, model_style

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        if aggregator is None or sim_id is None:
            return None

        for model, times, model_style in self.iter_models(aggregator, sim_id):
            for var_name in self.variables:
                data = aggregator._load_variable(sim_id, model, self.stim_name, var_name)
                color = self.get_variable_color(var_name)

                ax.plot(
                    times,
                    data,
                    label=model_style["label"],
                    color=color,
                    linestyle=model_style["linestyle"],
                    linewidth=model_style["linewidth"],
                    alpha=model_style["alpha"],
                )

                if np.isnan(data).any() or np.isinf(data).any():
                    print(f"Warning: NaN or Inf values detected in variable '{var_name}' for model '{model}' and sim_id '{sim_id}'. These values will not be plotted.")
                plotted_any = True

        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                f"No Data\n({sim_id})",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
        return None


class AggregatorRateTracePlotter(AggregatorTracePlot):
    DEFAULT_VARIABLES = ["exc_rate_mean", "inh_rate_mean"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "Firing Rate",
        "xlabel": "Time",
        "x_unit": "ms",
        "ylabel": "Firing Rate",
        "y_unit": "Hz",
    }


class AggregatorVoltageTracePlotter(AggregatorTracePlot):
    DEFAULT_VARIABLES = ["exc_voltage_mean", "inh_voltage_mean"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "Membrane Voltage",
        "xlabel": "Time",
        "x_unit": "ms",
        "ylabel": "Membrane potential",
        "y_unit": "mV",
    }


class AggregatorSTPTracePlotter(AggregatorTracePlot):
    DEFAULT_VARIABLES = ["exc_x_mean", "exc_u_mean"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "STP Adaptation Variables",
        "xlabel": "Time",
        "x_unit": "ms",
        "ylabel": "STP Adaptation Variables",
        "y_unit": None,
        "colors": {
            "exc_x_mean": "blue",
            "exc_u_mean": "purple",
            "inh_x_mean": "cyan",
            "inh_u_mean": "magenta",
        },
    }


class AggregatorHeatmapPlotter(BaseAggregatorPlot):
    """Plotter for 2D contourf / heatmaps of single neuron data loaded via aggregator."""

    DEFAULT_VARIABLES = ["out_rate_mean"]
    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        "title": "Single Neuron Activity Heatmap",
        "xlabel": r"$\nu_e$",
        "ylabel": r"$\nu_i$",
        "x_unit": "Hz",
        "y_unit": "Hz",
        "z_unit": "Hz",
        "vmin": None,
        "vmax": None,
        "levels": 10,
        "cmap": "viridis",
        "extend": "max",
        "colorbar_label": r"$\nu_{out}$",
    }

    def __init__(
        self,
        variables: Union[str, List[str]] = None,
        models: List[str] = None,
        stim_name: str = "SpontActivity0_5",
        model: str = "single_neuron",
        params: dict = None,
    ):
        super().__init__(variables=variables, models=models, stim_name=stim_name, params=params)
        self.model = model

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        if aggregator is None or sim_id is None:
            return None

        var_name = self.variables[0] if self.variables else "out_rate_mean"

        try:
            exc_grid = aggregator._load_variable(sim_id, self.model, self.stim_name, "exc_rate_grid")
            inh_grid = aggregator._load_variable(sim_id, self.model, self.stim_name, "inh_rate_grid")
            data = aggregator._load_variable(sim_id, self.model, self.stim_name, var_name)
        except Exception:
            ax.text(0.5, 0.5, f"No Data\n({sim_id})", ha="center", va="center", transform=ax.transAxes, color="gray")
            return None

        im = ax.contourf(
            exc_grid,
            inh_grid,
            data,
            levels=self.full_params["levels"],
            extend=self.full_params["extend"],
            vmin=self.full_params["vmin"],
            vmax=self.full_params["vmax"],
            cmap=self.full_params["cmap"],
        )
        return im


class AggregatorActivityHeatmapPlotter(AggregatorHeatmapPlotter):
    DEFAULT_VARIABLES = ["out_rate_mean"]
    DEFAULT_PARAMS = {
        **AggregatorHeatmapPlotter.DEFAULT_PARAMS,
        "title": "Neuron Activity Heatmap",
        "z_unit": "Hz",
        "colorbar_label": r"$\nu_{out}$",
    }


class AggregatorAdaptationHeatmapPlotter(AggregatorHeatmapPlotter):
    DEFAULT_VARIABLES = ["adaptation_mean"]
    DEFAULT_PARAMS = {
        **AggregatorHeatmapPlotter.DEFAULT_PARAMS,
        "title": "Neuron Adaptation Heatmap",
        "z_unit": "pA",
        "colorbar_label": "adaptation",
        "cmap": "viridis",
        "extend": "neither",
    }


class AggregatorSNNRasterPlotter(BaseAggregatorPlot):
    """Plotter for SNN spike raster plots loaded via aggregator."""

    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        "title": "Spike Raster",
        "xlabel": "Time",
        "ylabel": "Neuron Index",
        "x_unit": "ms",
        "y_unit": None,
        "marker": "o",
        "markersize": 5,
        "exc_cells": 400,
        "inh_cells": 100,
        "legend": False,
        "xmargin": 0.0,
        "ymargin": 0.0,
    }

    def __init__(
        self,
        stim_name: str,
        model: str = "snn",
        params: dict = None,
    ):
        super().__init__(variables=["exc_spikes", "inh_spikes"], models=[model], stim_name=stim_name, params=params)
        self.model = model

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        if aggregator is None or sim_id is None:
            return None

        exc_spikes = aggregator._load_variable(sim_id, self.model, self.stim_name, "exc_spikes")
        inh_spikes = aggregator._load_variable(sim_id, self.model, self.stim_name, "inh_spikes")

        exc_cells = self.full_params["exc_cells"]
        inh_cells = self.full_params["inh_cells"]
        exc_col = self.full_params["exc_color"]
        inh_col = self.full_params["inh_color"]
        ms = self.full_params["markersize"]
        marker = self.full_params["marker"]

        exc_x, exc_y = [], []
        if exc_spikes is not None and len(exc_spikes) > 0:
            for i, spiketrain in enumerate(exc_spikes[:exc_cells], start=1):
                if len(spiketrain) > 0:
                    exc_x.extend(spiketrain)
                    exc_y.extend([i] * len(spiketrain))

        inh_x, inh_y = [], []
        if inh_spikes is not None and len(inh_spikes) > 0:
            for i, spiketrain in enumerate(inh_spikes[:inh_cells], start=exc_cells + 1):
                if len(spiketrain) > 0:
                    inh_x.extend(spiketrain)
                    inh_y.extend([i] * len(spiketrain))

        lw = 0.8 if marker == "|" else 0
        if exc_x:
            ax.scatter(exc_x, exc_y, color=exc_col, marker=marker, s=ms, lw=lw)
        if inh_x:
            ax.scatter(inh_x, inh_y, color=inh_col, marker=marker, s=ms, lw=lw)

        return None


class AggregatorNeuronIOCurvePlotter(BaseAggregatorPlot):
    """Plotter for single neuron I/O response curves loaded via aggregator."""

    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        "title": "Single Neuron Activity",
        "xlabel": r"$\nu_e$",
        "ylabel": r"$\nu_{out}$",
        "x_unit": "Hz",
        "y_unit": "Hz",
        "curves_num": 5,
        "linestyle": "None",
        "marker": "o",
        "markersize": 5,
        "yerrorbar": False,
        "capsize": 3,
    }

    def __init__(
        self,
        variables: Union[str, List[str]] = None,
        models: List[str] = None,
        stim_name: str = "SpontActivity0_5",
        model: str = "single_neuron",
        params: dict = None,
    ):
        super().__init__(variables=variables, models=models, stim_name=stim_name, params=params)
        self.model = model

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        if aggregator is None or sim_id is None:
            return None

        try:
            exc_grid = aggregator._load_variable(sim_id, self.model, self.stim_name, "exc_rate_grid")
            inh_grid = aggregator._load_variable(sim_id, self.model, self.stim_name, "inh_rate_grid")
            out_mean = aggregator._load_variable(sim_id, self.model, self.stim_name, "out_rate_mean")
            out_std = aggregator._load_variable(sim_id, self.model, self.stim_name, "out_rate_std") if self.full_params["yerrorbar"] else None
        except Exception:
            ax.text(0.5, 0.5, f"No Data\n({sim_id})", ha="center", va="center", transform=ax.transAxes, color="gray")
            return None

        inh_slice_indices = np.linspace(0, inh_grid.shape[1] - 1, self.full_params["curves_num"], dtype=int)

        for j, nu_i_idx in enumerate(inh_slice_indices):
            nu_i_val = inh_grid[0, nu_i_idx]
            label = self.full_params["labels"][j] if (self.full_params["labels"] and j < len(self.full_params["labels"])) else fr"$\nu_i$={nu_i_val:.0f} Hz"
            yerr = out_std[:, nu_i_idx] if out_std is not None else None

            ax.errorbar(
                exc_grid[:, nu_i_idx],
                out_mean[:, nu_i_idx],
                yerr=yerr,
                marker=self.full_params["marker"],
                linestyle=self.full_params["linestyle"],
                markersize=self.full_params["markersize"],
                capsize=self.full_params["capsize"],
                color=self.get_variable_color(f"curve_{j}", j),
                label=label,
            )
        return None


class AggregatorGridPlottingHook:
    """
    2D Grid Plotting Hook for ResultsAggregator datasets.

    Generates an nrows x ncols grid of subplots where:
      - nrows = len(y_param_values) (row labels on LEFT margin)
      - ncols = len(x_param_values) (col titles on TOP margin)
    """

    DEFAULT_FIG_PARAMS = {
        "axsize": (4.5, 3.5),
        "figsize": None,
        "dpi": 100,
        "title": None,  # Auto-generated as "{plotter_title}: {plotter.stim_name}" if None
        "sharex": True,
        "sharey": True,
        "constrained_layout": True,
        "savefig": False,
        "savefig_path": None,
        "show_row_col_labels": True,
        "hide_inner_ticks": True,
    }

    def __init__(
        self,
        aggregator,
        x_param: str,
        y_param: str,
        plotter: BaseAggregatorPlot,
        fig_params: dict = None,
        common_params: dict = None,
        subplot_params: dict = None,
        param_filters: dict = None,
        filters: dict = None,
        **kwargs_filters,
    ):
        self.agg = aggregator
        self.x_param = x_param
        self.y_param = y_param
        self.plotter = plotter or AggregatorRateTracePlotter()

        self.fig_params = {**self.DEFAULT_FIG_PARAMS, **(fig_params or {})}
        self.common_params = common_params or {}
        self.subplot_params = subplot_params or {}

        # Merge dict filters and keyword filters
        combined_filters = {}
        if param_filters:
            combined_filters.update(param_filters)
        if filters:
            combined_filters.update(filters)
        combined_filters.update(kwargs_filters)
        self.param_filters = combined_filters

    def __call__(self) -> Tuple[plt.Figure, np.ndarray]:
        # 1. Resolve exact parameter names
        x_full_name, x_col = self.agg.resolve_param_column(self.x_param)
        y_full_name, y_col = self.agg.resolve_param_column(self.y_param)

        # 2. Query filtered results matrix
        _, param_mat, p_names, sim_ids = self.agg.get_results(
            variable="times", **self.param_filters
        )

        if len(sim_ids) == 0:
            raise ValueError(f"No simulation runs match the filters: {self.param_filters}")

        # 3. Extract unique x and y parameter values
        x_vals = list(dict.fromkeys(param_mat[:, x_col]))
        y_vals = list(dict.fromkeys(param_mat[:, y_col]))

        nrows = len(y_vals)
        ncols = len(x_vals)

        # 4. Determine figure size
        ax_w, ax_h = self.fig_params["axsize"]
        figsize = self.fig_params["figsize"] or (ncols * ax_w, nrows * ax_h)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=self.fig_params["sharex"],
            sharey=self.fig_params["sharey"],
            dpi=self.fig_params["dpi"],
            constrained_layout=self.fig_params["constrained_layout"],
        )

        # Normalize axes array to 2D
        if nrows == 1 and ncols == 1:
            axes_grid = np.array([[axes]])
        elif nrows == 1:
            axes_grid = np.array([axes])
        elif ncols == 1:
            axes_grid = np.array([[ax] for ax in axes])
        else:
            axes_grid = np.array(axes)

        # 5. Render subplots cell by cell
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                ax = axes_grid[i, j]

                # Match sim_id for (x_val, y_val)
                cell_mask = (param_mat[:, x_col] == x_val) & (param_mat[:, y_col] == y_val)
                matching_indices = np.where(cell_mask)[0]

                cell_overrides = self.subplot_params.get((i, j), {})

                # Deep copy plotter per cell to isolate parameter updates (matching GridFigureHook in hooks.py)
                if isinstance(self.plotter, BasePlot):
                    cell_plotter = copy.deepcopy(self.plotter)
                    cell_plotter.full_params.update(self.common_params)
                    cell_plotter.full_params.update(cell_overrides)
                else:
                    cell_plotter = self.plotter

                # Column Headers (top row only)
                if i == 0 and self.fig_params.get("show_row_col_labels", True):
                    cell_title = cell_overrides.get("title", f"{self.x_param} = {x_val}")
                elif "title" in cell_overrides:
                    cell_title = cell_overrides["title"]
                else:
                    cell_title = None

                # Outer Axis Labels (shared axes control)
                cell_xlabel = self.common_params.get("xlabel") or cell_plotter.full_params.get("xlabel", "Time")
                if not (i == nrows - 1 or not self.fig_params["sharex"]):
                    cell_xlabel = None

                cell_ylabel = self.common_params.get("ylabel") or cell_plotter.full_params.get("ylabel")
                if not (j == 0 or not self.fig_params["sharey"]):
                    cell_ylabel = None

                # Legend control (top-left subplot or when legend_all=True)
                cell_legend = self.common_params.get("legend", True)
                if cell_legend:
                    if (i == 0 and j == 0) or cell_overrides.get("legend_all", False) or self.common_params.get("legend_all", False):
                        if cell_legend is True:
                            cell_legend = {"fontsize": 8, "loc": "upper right"}
                    else:
                        cell_legend = False

                if isinstance(cell_plotter, BasePlot):
                    cell_plotter.full_params["title"] = cell_title
                    cell_plotter.full_params["xlabel"] = cell_xlabel
                    cell_plotter.full_params["ylabel"] = cell_ylabel
                    cell_plotter.full_params["legend"] = cell_legend

                if len(matching_indices) > 0:
                    sim_id = sim_ids[matching_indices[0]]
                    if isinstance(cell_plotter, BasePlot):
                        im = cell_plotter.draw(ax, sim_id=sim_id, aggregator=self.agg)
                        if im is not None:
                            cell_plotter.add_colorbar(fig, ax, im)
                    else:
                        cell_plotter(ax, sim_id, self.agg, cell_overrides)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="lightgray",
                    )
                    if isinstance(cell_plotter, BasePlot):
                        cell_plotter.apply_preplot_params(ax, cell_plotter.full_params)
                        cell_plotter.apply_postplot_params(ax, cell_plotter.full_params)

                # Hide inner tick labels for shared axes grid
                if self.fig_params.get("hide_inner_ticks", True):
                    if self.fig_params.get("sharex", True) and i < nrows - 1:
                        ax.tick_params(labelbottom=False)
                    if self.fig_params.get("sharey", True) and j > 0:
                        ax.tick_params(labelleft=False)

                # Row Labels (LEFT margin on first column j == 0)
                if j == 0 and self.fig_params.get("show_row_col_labels", True):
                    row_text = f"{self.y_param} = {y_val}"
                    ax.text(
                        -0.22,
                        0.5,
                        row_text,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=11,
                        # fontweight="bold",
                    )

        # Automatic Suptitle format: "{plotter_title}: {plotter.stim_name}"
        title = self.fig_params.get("title")
        if title is None and hasattr(self.plotter, "stim_name"):
            plotter_params = getattr(self.plotter, "full_params", {})
            plotter_title = plotter_params.get("title") or getattr(self.plotter, "title_type", "Trace")
            title = f"{plotter_title}: {self.plotter.stim_name}"

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        # Save Figure if requested
        if self.fig_params.get("savefig"):
            save_path = self.fig_params.get("savefig_path", "grid_plot.png")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.fig_params.get("dpi", 100), bbox_inches="tight")
            print(f"Saved grid figure to '{save_path}'")

        return fig, axes
