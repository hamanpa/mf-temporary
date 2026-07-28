import os
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Callable
import numpy as np
import matplotlib.pyplot as plt

from .base import BasePlot, EXC_COLOR, INH_COLOR, LINESTYLES


class GridTracePlotter(BasePlot):
    """
    Unified time-series trace plotter for multi-model comparison across 2D grid subplots.
    Handles firing rates, membrane voltages, conductances, and STP state variables.
    """

    DEFAULT_VARIABLES = None

    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        "title": None,
        "xlabel": "Time",
        "ylabel": "Trace",
        "x_unit": "ms",
        "y_unit": None,
        "linewidth": 2.0,
        "alpha": 1.0,
        "labels": None,     # Optional list of model labels
        "linestyles": None, # Optional list or dict of model linestyles
        "colors": None,     # Optional list or dict of variable/series colors
    }

    def __init__(
        self,
        variables: Union[str, List[str]] = None,
        models: List[str] = None,
        stim_name: str = "SpontActivity0_5",
        params: dict = None,
    ):
        super().__init__(params=params)

        if variables is not None:
            self.variables = [variables] if isinstance(variables, str) else list(variables)
        elif self.DEFAULT_VARIABLES is not None:
            self.variables = list(self.DEFAULT_VARIABLES)
        else:
            raise ValueError("No variables specified for GridTracePlotter.")

        self.models = models
        self.stim_name = stim_name

    def get_variable_color(self, var_name: str, var_idx: int) -> str:
        """Determines the color for a given variable name or index."""
        colors = self.full_params.get("colors")
        if isinstance(colors, dict):
            if var_name in colors:
                return colors[var_name]
            var_lower = var_name.lower()
            if var_lower in colors:
                return colors[var_lower]
        elif isinstance(colors, (list, tuple)) and colors:
            return colors[var_idx % len(colors)]

        var_lower = var_name.lower()
        if "exc" in var_lower:
            return self.full_params.get("exc_color", EXC_COLOR)
        elif "inh" in var_lower:
            return self.full_params.get("inh_color", INH_COLOR)

        return "black"

    def get_model_linestyle(self, model: str, model_idx: int) -> str:
        """Determines the linestyle for a given model or index."""
        linestyles = self.full_params.get("linestyles")
        if isinstance(linestyles, dict):
            if model in linestyles:
                return linestyles[model]
            model_lower = model.lower()
            if model_lower in linestyles:
                return linestyles[model_lower]
        elif isinstance(linestyles, (list, tuple)) and linestyles:
            return linestyles[model_idx % len(linestyles)]

        return LINESTYLES[model_idx % len(LINESTYLES)]

    def _format_series_label(self, model: str, var_name: str, model_idx: int) -> str:
        """Formats clean series labels matching network_plots conventions."""
        labels = self.full_params.get("labels")
        model_label = labels[model_idx] if (labels and model_idx < len(labels)) else model

        if len(self.variables) == 1:
            return f"{model_label}"

        prefix_map = {
            "exc_rate_mean": "Exc",
            "exc_rate": "Exc",
            "exc_rate_pop_mean": "Exc",
            "inh_rate_mean": "Inh",
            "inh_rate": "Inh",
            "inh_rate_pop_mean": "Inh",
            "exc_voltage_mean": "Exc",
            "exc_voltage": "Exc",
            "inh_voltage_mean": "Inh",
            "inh_voltage": "Inh",
            "exc_x_mean": "Exc x",
            "exc_u_mean": "Exc u",
            "inh_x_mean": "Inh x",
            "inh_u_mean": "Inh u",
        }

        prefix = prefix_map.get(var_name)
        if prefix is None:
            var_lower = var_name.lower()
            if "exc" in var_lower:
                prefix = "Exc"
            elif "inh" in var_lower:
                prefix = "Inh"
            else:
                prefix = var_name

        return f"{prefix} {model_label}"

    def draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs):
        self.apply_preplot_params(ax, self.full_params)
        im = self._draw(ax, sim_id=sim_id, aggregator=aggregator)
        self.apply_postplot_params(ax, self.full_params)
        return im

    def __call__(self, ax: plt.Axes, sim_id: str = None, aggregator=None):
        return self.draw(ax, sim_id=sim_id, aggregator=aggregator)

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        """
        Plots trace variables for the specified simulation ID on subplot ax.
        """
        if aggregator is None or sim_id is None:
            return None

        if self.models is None:
            available = aggregator.get_available_variables()
            target_models = list(available.keys())
        else:
            target_models = self.models

        plotted_any = False
        lw = self.full_params.get("linewidth", 2.0)
        alpha_val = self.full_params.get("alpha", 1.0)

        for model_idx, model in enumerate(target_models):
            ls = self.get_model_linestyle(model, model_idx)

            try:
                times = aggregator._load_variable(sim_id, model, self.stim_name, "times")
            except Exception:
                continue

            for var_idx, var_name in enumerate(self.variables):
                try:
                    data = aggregator._load_variable(sim_id, model, self.stim_name, var_name)
                    color = self.get_variable_color(var_name, var_idx)
                    label_str = self._format_series_label(model, var_name, model_idx)

                    ax.plot(
                        times,
                        data,
                        label=label_str,
                        color=color,
                        linestyle=ls,
                        linewidth=lw,
                        alpha=alpha_val,
                    )
                    plotted_any = True
                except Exception:
                    pass

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


class GridRateTracePlotter(GridTracePlotter):
    DEFAULT_VARIABLES = ["exc_rate_mean", "inh_rate_mean"]
    DEFAULT_PARAMS = {
        **GridTracePlotter.DEFAULT_PARAMS,
        "title": "Firing Rate",
        "xlabel": "Time",
        "x_unit": "ms",
        "ylabel": "Firing Rate",
        "y_unit": "Hz",
    }


class GridVoltageTracePlotter(GridTracePlotter):
    DEFAULT_VARIABLES = ["exc_voltage_mean", "inh_voltage_mean"]
    DEFAULT_PARAMS = {
        **GridTracePlotter.DEFAULT_PARAMS,
        "title": "Membrane Voltage",
        "xlabel": "Time",
        "x_unit": "ms",
        "ylabel": "Membrane potential",
        "y_unit": "mV",
    }


class GridSTPTracePlotter(GridTracePlotter):
    DEFAULT_VARIABLES = ["exc_x_mean", "exc_u_mean"]
    DEFAULT_PARAMS = {
        **GridTracePlotter.DEFAULT_PARAMS,
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
    }

    DEFAULT_COMMON_PARAMS = {
        "xlabel": "Time",
        "x_unit": "ms",
        "ylabel": "Firing Rate",
        "y_unit": "Hz",
        "xlim": (None, None),
        "ylim": (None, None),
        "grid": False,  # Default NO grid lines
        "legend": True,
    }

    def __init__(
        self,
        aggregator,
        x_param: str,
        y_param: str,
        plotter: Union[Callable, Any] = None,
        fig_params: dict = None,
        common_params: dict = None,
        subplot_params: dict = None,
        **param_filters,
    ):
        self.agg = aggregator
        self.x_param = x_param
        self.y_param = y_param
        self.plotter = plotter or GridRateTracePlotter()

        self.fig_params = {**self.DEFAULT_FIG_PARAMS, **(fig_params or {})}
        self.common_params = {**self.DEFAULT_COMMON_PARAMS, **(common_params or {})}
        self.subplot_params = subplot_params or {}
        self.param_filters = param_filters

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

        # Short aliases for row/col headers
        x_alias = self.x_param.split(".")[-1]
        y_alias = self.y_param.split(".")[-1]

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
                    cell_title = cell_overrides.get("title", f"{x_alias} = {x_val}")
                elif "title" in cell_overrides:
                    cell_title = cell_overrides["title"]
                else:
                    cell_title = None

                # Outer Axis Labels (shared axes control)
                cell_xlabel = self.common_params.get("xlabel", "Time")
                if not (i == nrows - 1 or not self.fig_params["sharex"]):
                    cell_xlabel = None

                cell_ylabel = self.common_params.get("ylabel")
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
                        cell_plotter(ax, sim_id=sim_id, aggregator=self.agg)
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

                # Row Labels (LEFT margin on first column j == 0)
                if j == 0 and self.fig_params.get("show_row_col_labels", True):
                    row_text = f"{y_alias} = {y_val}"
                    ax.text(
                        -0.22,
                        0.5,
                        row_text,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
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
