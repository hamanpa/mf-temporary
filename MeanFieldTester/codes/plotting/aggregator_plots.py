import os
import copy
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Union, Callable
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

from .base import BasePlot, EXC_COLOR, INH_COLOR, LINESTYLES
from ..data_structures.neuron_simulation import drive_index, drive_values
from ..network_params.models import INTERNAL_PROJECTIONS


from ..network_params.translators import get_unit_multiplier
from ..transfer_function.base import BaseTransferFunction
from .base import HEATMAP_PARAMS, draw_grid_heatmap


def load_neuron_grid_slice(
        aggregator, sim_id: str, model: str, stim_name: str, variables: List[str],
        drive_rate: float | None = None, units: Dict[str, str] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Loads single-neuron grid arrays of a run as 2D (exc_rate, inh_rate) slices at `drive_rate` [Hz]
    (None: the first drive value), plus the "drive_rate_grid" slice itself [Hz].
    `units`: {variable: unit} to read variables in (others as stored).
    Files saved before the drive axis existed are 2D (drive = 0) and returned as they are, without "drive_rate_grid".
    """
    units = units or {}
    results = aggregator.results(sim_id, model, stim_name)
    arrays = {variable: results.get(variable, metric=None, unit=units.get(variable)) for variable in variables}
    if not results.has("drive_rate_grid", metric=None):
        return arrays  # older 2D data (drive = 0)
    arrays["drive_rate_grid"] = results.get("drive_rate_grid", metric=None, unit="Hz")
    index = drive_index(arrays["drive_rate_grid"], drive_rate)
    return {variable: values[..., index] for variable, values in arrays.items()}


class BaseAggregatorPlot(BasePlot, ABC):
    """
    Abstract base class for plotters that draw ResultsAggregator simulation data onto a single ax.
    Extends BasePlot, maintaining full compatibility with DEFAULT_PARAMS, full_params,
    apply_preplot_params, and apply_postplot_params.

    Shared helpers for plots of several models of one run (resolved per draw, `full_params` is not modified):
    per-model styles (`model_styles`), per-variable colours (`variable_color`), the model legend
    (`model_legend_handles`), the models of a run (`models_for`) and the "No Data" message (`show_no_data`).
    A `_draw` that builds its own legend sets `self._legend_kwargs` (e.g. {"handles": ..., "title": ...}),
    which `draw` merges into a copy of the `legend` param.
    """

    DEFAULT_VARIABLES = []

    DEFAULT_PARAMS = {
        **BasePlot.DEFAULT_PARAMS,
        # Per-model styles: one value for all models, a list (in the order of `models`) or a dict {model: value}.
        # None: labels = model names, linestyles cycle through LINESTYLES, `default_alpha`, `default_linewidth`.
        "labels": None,
        "linestyles": None,
        "alphas": None,
        "linewidths": None,
        "default_alpha": 1.0,
        "default_linewidth": 1.5,
        # Per-variable colours: a list (in the order of `variables`) or a dict {variable: colour}. Variables without
        # one get `default_colors` (set by presets), then their population colour (`exc_color`/`inh_color`, see
        # `population_of`), then `default_color`.
        "colors": None,
        "default_colors": {},
        "default_color": "black",
        # Legend: one entry per model (label and line style) drawn in `legend_color`
        # (None: the variable's colour when one variable is plotted); `variable_legend` adds one entry per variable.
        "legend_color": "black",
        "variable_legend": False,
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

        self._explicit_variables = variables is not None
        self.models = models
        self.stim_name = stim_name
        self._legend_kwargs = None

    def draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs):
        self._legend_kwargs = None
        self.apply_preplot_params(ax, self.full_params)
        im = self._draw(ax, sim_id=sim_id, aggregator=aggregator, **kwargs)
        params = self.full_params
        if self._legend_kwargs and self._legend_kwargs.get("handles") and params["legend"]:
            legend = params["legend"] if isinstance(params["legend"], dict) else {}
            params = {**params, "legend": {**legend, **self._legend_kwargs}}
        self.apply_postplot_params(ax, params)
        return im

    def plot_variables(self) -> List[str]:
        """The variables drawn (presets may derive them from params, see `AggregatorTracePlot`)."""
        return self.variables

    def param_values(self, param: str, aggregator, sim_id: str) -> list | None:
        """
        Values that the plot parameter `param` (a `full_params` key) can take for run `sim_id`,
        e.g. the drive rates of a neuron grid. Used by `AggregatorGridPlottingHook` for `plot.*` axes
        without explicit values. None: this plotter cannot discover the values of `param`.
        """
        return None

    @abstractmethod
    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def models_for(self, aggregator, sim_id: str) -> Tuple[List[str], List[str]]:
        """
        (ordered, present): `ordered` are the requested models (`models`, or all models of the run with this
        stimulus), the order that list-valued style params refer to; `present` are those with results in this run.
        Requested models without results are skipped with a warning.
        """
        stim_file_name = str(self.stim_name).replace(" ", "_")
        available = [model for model, stims in aggregator.available_results(sim_id).items() if stim_file_name in stims]
        ordered = list(self.models) if self.models is not None else available
        present = [model for model in ordered if model.lower() in available]
        missing = [model for model in ordered if model.lower() not in available]
        if missing:
            warnings.warn(f"Run '{sim_id}': no '{self.stim_name}' results for models {missing}; not plotted. Available: {available}")
        return ordered, present

    def _per_model(self, name: str, models: List[str], default: Callable[[int, str], Any]) -> Dict[str, Any]:
        """{model: value} of the per-model param `name` (one value, list in the order of `models`, or dict)."""
        value = self.full_params[name]
        if value is None:
            return {model: default(i, model) for i, model in enumerate(models)}
        if isinstance(value, dict):
            return {model: value.get(model, default(i, model)) for i, model in enumerate(models)}
        if isinstance(value, (list, tuple)):
            if len(value) != len(models):
                raise ValueError(f"'{name}' has {len(value)} values, but there are {len(models)} models: {models}.")
            return dict(zip(models, value))
        return {model: value for model in models}

    def model_styles(self, models: List[str]) -> Dict[str, dict]:
        """{model: {"label", "linestyle", "alpha", "linewidth"}}."""
        labels = self._per_model("labels", models, lambda i, model: model)
        linestyles = self._per_model("linestyles", models, lambda i, model: LINESTYLES[i % len(LINESTYLES)])
        alphas = self._per_model("alphas", models, lambda i, model: self.full_params["default_alpha"])
        linewidths = self._per_model("linewidths", models, lambda i, model: self.full_params["default_linewidth"])
        return {
            model: {"label": labels[model], "linestyle": linestyles[model], "alpha": alphas[model], "linewidth": linewidths[model]}
            for model in models
        }

    @staticmethod
    def population_of(variable: str) -> str | None:
        """
        "exc"/"inh" by the naming convention: the `exc_`/`inh_` prefix, or the *source* of a projection code
        (target-source, e.g. `ei_x` = synapses onto E from I → "inh"); None otherwise (e.g. `drive_rate`).
        """
        prefix = variable.split("_", 1)[0]
        if prefix in ("exc", "inh"):
            return prefix
        if prefix in INTERNAL_PROJECTIONS:
            return {"e": "exc", "i": "inh"}[prefix[1]]
        return None

    def variable_color(self, variable: str) -> str:
        """Colour of a variable (see the `colors` param)."""
        colors = self.full_params["colors"]
        variables = self.plot_variables()
        if isinstance(colors, (list, tuple)):
            if len(colors) != len(variables):
                raise ValueError(f"'colors' has {len(colors)} values, but there are {len(variables)} variables: {variables}.")
            colors = dict(zip(variables, colors))
        if isinstance(colors, dict) and variable in colors:
            return colors[variable]
        if variable in self.full_params["default_colors"]:
            return self.full_params["default_colors"][variable]
        population = self.population_of(variable)
        if population is not None:
            return self.full_params[f"{population}_color"]
        return self.full_params["default_color"]

    def model_legend_handles(self, styles: Dict[str, dict], plotted_variables: List[str]) -> List[Line2D]:
        """One handle per model (see the `legend_color` and `variable_legend` params)."""
        color = self.full_params["legend_color"]
        if color is None:
            color = self.variable_color(plotted_variables[0]) if len(plotted_variables) == 1 else self.full_params["default_color"]
        handles = [
            Line2D([0], [0], color=color, linestyle=style["linestyle"], linewidth=style["linewidth"], alpha=style["alpha"], label=style["label"])
            for style in styles.values()
        ]
        if self.full_params["variable_legend"]:
            handles += [Line2D([0], [0], color=self.variable_color(variable), label=variable) for variable in plotted_variables]
        return handles

    @staticmethod
    def show_no_data(ax: plt.Axes, sim_id: str) -> None:
        ax.text(0.5, 0.5, f"No Data\n({sim_id})", ha="center", va="center", transform=ax.transAxes, color="gray")


class AggregatorTracePlot(BaseAggregatorPlot):
    """
    Time traces of `{variable}_{metric}` (see `SavedResults.get`) for several models of one run:
    one colour per variable, one line style per model. Data are read in `x_unit`/`y_unit`.
    Variables a model has not saved are skipped with a warning.
    """

    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        "xlabel": "Time",
        "x_unit": "ms",
        "metric": "pop_mean",
        # Band mean ± `{variable}_pop_std` per model: one bool, a list (in the order of `models`) or a dict {model: bool}.
        # Drawn only where the std is saved. Note: for the MF it is √C (fluctuation of the population rate),
        # for the SNN (if `pop_std` is saved) the spread across neurons.
        "std_bands": False,
        "band_alpha": 0.3,
        # Per-projection variables onto one target population: with `projection_variable` set (e.g. "x", "u",
        # "conductance") and no explicit `variables`, the plot draws `{target}e_{v}` and `{target}i_{v}`
        # (target-source codes, coloured by source), e.g. target "exc", "x" → ee_x, ei_x.
        "projection_variable": None,
        "target": "exc",
    }

    def plot_variables(self) -> List[str]:
        variable = self.full_params["projection_variable"]
        if self._explicit_variables or variable is None:
            return self.variables
        targets = {"exc": "e", "inh": "i"}
        if self.full_params["target"] not in targets:
            raise ValueError(f"'target' must be one of {list(targets)}, got '{self.full_params['target']}'.")
        target = targets[self.full_params["target"]]
        return [f"{target}{source}_{variable}" for source in ("e", "i")]

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        if aggregator is None or sim_id is None:
            return None

        ordered, present = self.models_for(aggregator, sim_id)
        styles = self.model_styles(ordered)
        std_bands = self._per_model("std_bands", ordered, lambda i, model: False)
        x_unit, y_unit, metric = self.full_params["x_unit"], self.full_params["y_unit"], self.full_params["metric"]

        plotted_variables = []
        for model in present:
            results = aggregator.results(sim_id, model, self.stim_name)
            times = results.times(x_unit)
            style = styles[model]

            for variable in self.plot_variables():
                if not results.has(variable, metric):
                    # (no sim_id in the message, so the warning is shown once per model and key, not per run)
                    warnings.warn(f"Model '{model}' has no '{results.key(variable, metric)}'; not plotted.")
                    continue
                data = results.get(variable, metric, unit=y_unit)
                if not np.all(np.isfinite(data)):
                    warnings.warn(f"Run '{sim_id}', model '{model}': NaN or Inf in '{results.key(variable, metric)}' (not drawn there).")
                color = self.variable_color(variable)
                ax.plot(times, data, label=style["label"], color=color,
                        linestyle=style["linestyle"], linewidth=style["linewidth"], alpha=style["alpha"])

                if std_bands[model]:
                    if metric == "pop_mean" and results.has(variable, "pop_std"):
                        std = results.get(variable, "pop_std", unit=y_unit)
                        ax.fill_between(times, data - std, data + std, color=color, alpha=self.full_params["band_alpha"], linewidth=0)
                    else:
                        warnings.warn(f"Model '{model}' has no '{results.key(variable, 'pop_std')}' for the std band.")

                if variable not in plotted_variables:
                    plotted_variables.append(variable)

        if not plotted_variables:
            self.show_no_data(ax, sim_id)
            return None

        self._legend_kwargs = {"handles": self.model_legend_handles({model: styles[model] for model in present}, plotted_variables)}
        return None


class AggregatorRateTracePlotter(AggregatorTracePlot):
    DEFAULT_VARIABLES = ["exc_rate", "inh_rate"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "Firing Rate",
        "ylabel": "Firing Rate",
        "y_unit": "Hz",
    }


class AggregatorVoltageTracePlotter(AggregatorTracePlot):
    DEFAULT_VARIABLES = ["exc_voltage", "inh_voltage"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "Membrane Voltage",
        "ylabel": "Membrane potential",
        "y_unit": "mV",
    }


class AggregatorAdaptationTracePlotter(AggregatorTracePlot):
    # Only E adapts in the usual setups, so E alone in blue (add "inh_adaptation" to `variables` for I)
    DEFAULT_VARIABLES = ["exc_adaptation"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "Adaptation",
        "ylabel": "Adaptation",
        "y_unit": "pA",
        "default_colors": {"exc_adaptation": "blue"},
    }


class AggregatorSTPTracePlotter(AggregatorTracePlot):
    """
    STP variable `projection_variable` ("x" or "u") of the projections onto `target` ("exc"/"inh"),
    e.g. target "exc", "x" → ee_x (from E, exc colour) and ei_x (from I, inh colour).
    """
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "STP Variables",
        "ylabel": "STP variable",
        "y_unit": None,
        "projection_variable": "x",
    }

    def plot_variables(self) -> List[str]:
        if not self._explicit_variables and self.full_params["projection_variable"] not in ("x", "u"):
            # y (active resources) is not needed in the MF and will be dropped (see todo.md)
            raise ValueError(f"'projection_variable' must be 'x' or 'u', got '{self.full_params['projection_variable']}'.")
        return super().plot_variables()


class AggregatorConductanceTracePlotter(AggregatorTracePlot):
    """Mean conductance of the projections onto `target` ("exc"/"inh"). Only the SNN saves conductances."""
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "Synaptic Conductance",
        "ylabel": "Conductance",
        "y_unit": "nS",
        "projection_variable": "conductance",
    }


class AggregatorInputTracePlotter(AggregatorTracePlot):
    """External inputs: drive and stimulus rate per source (saved without a metric)."""
    DEFAULT_VARIABLES = ["drive_rate", "stim_rate"]
    DEFAULT_PARAMS = {
        **AggregatorTracePlot.DEFAULT_PARAMS,
        "title": "External Inputs",
        "ylabel": "Input rate",
        "y_unit": "Hz",
        "metric": None,
        "default_colors": {"drive_rate": "gray", "stim_rate": "purple"},
        "variable_legend": True,
    }


class AggregatorHeatmapPlotter(BaseAggregatorPlot):
    """
    Heatmap of a single-neuron grid variable (first of `variables`, a grid key such as "out_rate_mean")
    over the (exc, inh) input rates at one `drive_rate`, read in `x_unit`/`y_unit`/`z_unit`. See HEATMAP_PARAMS.
    """

    DEFAULT_VARIABLES = ["out_rate_mean"]
    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        **HEATMAP_PARAMS,
        "title": "Single Neuron Activity Heatmap",
        "xlabel": r"$\nu_e$",
        "ylabel": r"$\nu_i$",
        "x_unit": "Hz",
        "y_unit": "Hz",
        "z_unit": "Hz",
        "extend": "max",
        "colorbar_label": r"$\nu_{out}$",
        "drive_rate": None,  # drive rate [Hz] of the (exc, inh) slice; None = first drive value of the grid
    }

    def __init__(
        self,
        variables: Union[str, List[str]] = None,
        models: List[str] = None,
        stim_name: str = "steady_state",
        model: str = "exc_neuron",
        params: dict = None,
    ):
        super().__init__(variables=variables, models=models, stim_name=stim_name, params=params)
        self.model = model

    def _draw(self, ax: plt.Axes, sim_id: str = None, aggregator=None, **kwargs) -> None:
        if aggregator is None or sim_id is None:
            return None

        var_name = self.variables[0] if self.variables else "out_rate_mean"
        units = {"exc_rate_grid": self.full_params["x_unit"], "inh_rate_grid": self.full_params["y_unit"], var_name: self.full_params["z_unit"]}

        try:
            arrays = load_neuron_grid_slice(
                aggregator, sim_id, self.model, self.stim_name,
                ["exc_rate_grid", "inh_rate_grid", var_name], self.full_params["drive_rate"], units=units,
            )
        except (FileNotFoundError, KeyError) as error:
            warnings.warn(str(error))
            self.show_no_data(ax, sim_id)
            return None

        return draw_grid_heatmap(ax, arrays["exc_rate_grid"], arrays["inh_rate_grid"], arrays[var_name], self.full_params)


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

        results = aggregator.results(sim_id, self.model, self.stim_name)
        exc_spikes = results.get("exc_spikes", metric=None, unit=self.full_params["x_unit"])
        inh_spikes = results.get("inh_spikes", metric=None, unit=self.full_params["x_unit"])

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
    """Plot single-neuron I/O curves and optionally overlay transfer-function fits."""

    DEFAULT_PARAMS = {
        **BaseAggregatorPlot.DEFAULT_PARAMS,
        "title": "Single Neuron Activity",
        "xlabel": r"Firing Rate $r_{E}$",
        "ylabel": r"Firing Rate $r_{out}$",
        "x_unit": "Hz",
        "y_unit": "Hz",
        "curves_num": 5,
        "inh_rate_values": None,
        "inh_rate_indices": None,
        "linestyle": "None",
        "marker": "o",
        "markersize": 5,
        "yerrorbar": False,
        "capsize": 3,
        "curve_legend": False,
        "curve_legend_title": None,
        "tf_funcs": None,
        "tf_labels": None,
        "tf_linestyles": LINESTYLES,
        "drive_rate": None,  # drive rate [Hz] of the (exc, inh) slice; None = first drive value of the grid
    }

    def __init__(
        self,
        variables: Union[str, List[str]] = None,
        models: List[str] = None,
        stim_name: str = "steady_state",
        model: str = "exc_neuron",
        tf_funcs: List[BaseTransferFunction] | Dict[str, List[BaseTransferFunction]] | None = None,
        neuron_name: str | None = None,
        mf_model_names: str | List[str] | None = None,
        params: dict = None,
    ):
        """
        TF curves are overlaid either from `tf_funcs` (given explicitly, same for every run) or from
        `mf_model_names`: the TFs of these MF models as fitted during each plotted run
        (`ResultsAggregator.load_transfer_functions`, read from `data/<id>/params/`; no refitting).
        `neuron_name` selects the TF (default: `model`, the neuron results file prefix).
        """
        super().__init__(variables=variables, models=models, stim_name=stim_name, params=params)
        self.model = model
        self.tf_funcs = tf_funcs
        self.neuron_name = neuron_name or model
        self.mf_model_names = [mf_model_names] if isinstance(mf_model_names, str) else mf_model_names

    def param_values(self, param: str, aggregator, sim_id: str) -> list | None:
        """`drive_rate`: the drive rates [Hz] of the run's neuron grid (older 2D data: [0.0]; no grid file: [])."""
        if param != "drive_rate":
            return None
        try:
            results = aggregator.results(sim_id, self.model, self.stim_name)
        except FileNotFoundError:
            return []
        if not results.has("drive_rate_grid", metric=None):
            return [0.0]  # older 2D data (drive = 0)
        return drive_values(results.get("drive_rate_grid", metric=None)).tolist()

    def _get_tf_funcs(self, aggregator, sim_id: str, tf_funcs=None) -> Tuple[List[BaseTransferFunction], List[str]]:
        """
        The TFs to overlay and their default labels: explicit `tf_funcs` (draw argument, then constructor),
        else the run's own fitted TFs of `mf_model_names` (labelled by MF model name).
        """
        tf_funcs = self.tf_funcs if tf_funcs is None else tf_funcs
        if tf_funcs is None and self.mf_model_names:
            loaded = [aggregator.load_transfer_functions(sim_id, name)[self.neuron_name] for name in self.mf_model_names]
            return loaded, list(self.mf_model_names)
        if isinstance(tf_funcs, dict):
            tf_funcs = tf_funcs.get(self.neuron_name, [])
        if isinstance(tf_funcs, BaseTransferFunction):
            tf_funcs = [tf_funcs]
        tf_funcs = list(tf_funcs or [])
        return tf_funcs, [f"TF {i + 1}" for i in range(len(tf_funcs))]

    def _draw(
        self,
        ax: plt.Axes,
        sim_id: str = None,
        aggregator=None,
        tf_funcs: List[BaseTransferFunction] | Dict[str, List[BaseTransferFunction]] | None = None,
        **kwargs,
    ) -> None:
        if aggregator is None or sim_id is None:
            return None

        x_unit, y_unit = self.full_params["x_unit"], self.full_params["y_unit"]
        variables = ["exc_rate_grid", "inh_rate_grid", "out_rate_mean"] + (["out_rate_std"] if self.full_params["yerrorbar"] else [])
        units = {"exc_rate_grid": x_unit, "inh_rate_grid": x_unit, "out_rate_mean": y_unit, "out_rate_std": y_unit}
        try:
            arrays = load_neuron_grid_slice(aggregator, sim_id, self.model, self.stim_name, variables, self.full_params["drive_rate"], units=units)
        except (FileNotFoundError, KeyError) as error:
            warnings.warn(str(error))
            self.show_no_data(ax, sim_id)
            return None
        exc_grid, inh_grid, out_mean = arrays["exc_rate_grid"], arrays["inh_rate_grid"], arrays["out_rate_mean"]
        out_std = arrays.get("out_rate_std")

        inh_values = np.asarray(inh_grid[0, :] if inh_grid.ndim == 2 else inh_grid)
        requested_indices = self.full_params["inh_rate_indices"]
        requested_values = self.full_params["inh_rate_values"]

        if requested_indices is not None and requested_values is not None:
            raise ValueError("Specify only one of 'inh_rate_indices' or 'inh_rate_values'.")
        if requested_indices is not None:
            inh_slice_indices = np.asarray(requested_indices, dtype=int)
            if np.any((inh_slice_indices < 0) | (inh_slice_indices >= len(inh_values))):
                raise IndexError(
                    f"'inh_rate_indices' must be between 0 and {len(inh_values) - 1}."
                )
        elif requested_values is not None:
            requested_values = np.atleast_1d(requested_values)
            inh_slice_indices = np.asarray(
                [np.argmin(np.abs(inh_values - value)) for value in requested_values],
                dtype=int,
            )
        else:
            inh_slice_indices = np.linspace(
                0, len(inh_values) - 1, self.full_params["curves_num"], dtype=int
            )

        # Preserve the requested order while avoiding duplicate nearest-grid slices.
        _, unique_positions = np.unique(inh_slice_indices, return_index=True)
        inh_slice_indices = inh_slice_indices[np.sort(unique_positions)]

        colors = self.full_params["colors"]
        if colors is None:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        elif isinstance(colors, dict):
            colors = [colors.get(f"curve_{j}", self.full_params["default_color"]) for j in range(len(inh_slice_indices))]

        curve_labels = []
        for j, nu_i_idx in enumerate(inh_slice_indices):
            nu_i_val = inh_grid[0, nu_i_idx]
            label = self.full_params["labels"][j] if (self.full_params["labels"] and j < len(self.full_params["labels"])) else fr"$r_i$={nu_i_val:.0f} {x_unit}"
            curve_labels.append(label)
            yerr = out_std[:, nu_i_idx] if out_std is not None else None

            ax.errorbar(
                exc_grid[:, nu_i_idx],
                out_mean[:, nu_i_idx],
                yerr=yerr,
                marker=self.full_params["marker"],
                linestyle=self.full_params["linestyle"],
                markersize=self.full_params["markersize"],
                capsize=self.full_params["capsize"],
                color=colors[j % len(colors)],
                label=label if self.full_params["curve_legend"] else "_nolegend_",
            )

        fit_funcs, default_fit_labels = self._get_tf_funcs(aggregator, sim_id, tf_funcs)

        if fit_funcs:
            fit_labels = self.full_params["tf_labels"]
            if fit_labels is None:
                fit_labels = default_fit_labels
            fit_linestyles = self.full_params["tf_linestyles"][:len(fit_funcs)]

            # The TFs take rates in Hz and adaptation in nA (see transfer_function), independent of the plot units
            tf_variables = ["exc_rate_grid", "inh_rate_grid"]
            if any("adaptation" in tf.required_inputs() for tf in fit_funcs):
                tf_variables.append("adaptation_mean")
            try:
                tf_inputs = load_neuron_grid_slice(
                    aggregator, sim_id, self.model, self.stim_name, tf_variables, self.full_params["drive_rate"],
                    units={"exc_rate_grid": "Hz", "inh_rate_grid": "Hz", "adaptation_mean": "nA"},
                )
            except KeyError:
                raise ValueError(f"Transfer-function fits require {tf_variables}, not all saved for model '{self.model}'.") from None

            for j, nu_i_idx in enumerate(inh_slice_indices):
                adaptation = tf_inputs["adaptation_mean"][:, nu_i_idx] if "adaptation_mean" in tf_inputs else None
                drive_grid = tf_inputs.get("drive_rate_grid")

                for tf, linestyle in zip(fit_funcs, fit_linestyles, strict=True):
                    nu_out_fit = tf(
                        exc_rate=tf_inputs["exc_rate_grid"][:, nu_i_idx],
                        inh_rate=tf_inputs["inh_rate_grid"][:, nu_i_idx],
                        drive_rate=None if drive_grid is None else drive_grid[:, nu_i_idx],  # the plotted drive slice
                        adaptation=adaptation,
                    ) * get_unit_multiplier("Hz", y_unit)
                    ax.plot(
                        exc_grid[:, nu_i_idx],
                        nu_out_fit,
                        color=colors[j % len(colors)],
                        linestyle=linestyle,
                        linewidth=self.full_params["linewidth"],
                    )

            legend_elements = []
            if self.full_params["curve_legend"]:
                legend_elements.extend(
                    Line2D(
                        [0], [0], marker=self.full_params["marker"],
                        color=colors[j % len(colors)], label=label,
                        markerfacecolor=colors[j % len(colors)],
                        markersize=self.full_params["markersize"], linestyle="None",
                    )
                    for j, label in enumerate(curve_labels)
                )
            else:
                legend_elements.append(
                    Line2D(
                        [0], [0], marker=self.full_params["marker"], color="black",
                        label="Data", markerfacecolor="black",
                        markersize=self.full_params["markersize"], linestyle="None",
                    )
                )
            if len(fit_funcs) > 1 or self.full_params["tf_labels"] is not None:
                legend_elements += [
                    Line2D(
                        [0], [0], color="black", label=label,
                        linestyle=linestyle, linewidth=self.full_params["linewidth"],
                    )
                    for label, linestyle in zip(fit_labels, fit_linestyles, strict=True)
                ]
            self._legend_kwargs = {"handles": legend_elements}
        elif self.full_params["curve_legend"]:
            legend_elements = [
                Line2D(
                    [0], [0], marker=self.full_params["marker"],
                    color=colors[j % len(colors)], label=label,
                    markerfacecolor=colors[j % len(colors)],
                    markersize=self.full_params["markersize"], linestyle="None",
                )
                for j, label in enumerate(curve_labels)
            ]
            self._legend_kwargs = {"handles": legend_elements}
        if self._legend_kwargs and self.full_params["curve_legend_title"]:
            self._legend_kwargs["title"] = self.full_params["curve_legend_title"]
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
        # Values of the x axis (columns, left to right) and y axis (rows, top to bottom).
        # None: all values of the runs left by `run_filters` (run parameter), or the values the plotter
        # discovers (`plot.*` parameter, see `BaseAggregatorPlot.param_values`); a list: these values, in this order
        # (cells without a matching run, or whose run lacks the plot value, show "N/A").
        "x_values": None,
        "y_values": None,
        # Order of values that are not given explicitly: "ascending", "descending",
        # or None (order of the runs, as in param_combinations.csv).
        "x_order": "ascending",
        "y_order": "ascending",
    }

    # Axis parameters with this prefix are plot parameters (`full_params` keys set per row/column on the
    # same run, e.g. "plot.drive_rate"); any other axis parameter is a run parameter (selects the run).
    PLOT_PARAM_PREFIX = "plot."

    def __init__(
        self,
        aggregator,
        x_param: str,
        y_param: str,
        plotter: BaseAggregatorPlot,
        fig_params: dict = None,
        common_params: dict = None,
        subplot_params: dict = None,
        run_filters: dict = None,
    ):
        """
        Parameters
        ----------
        x_param, y_param : str
            Parameter spanning the columns / rows: a run parameter (as in param_combinations.csv, may be
            abbreviated, see `ResultsAggregator.resolve_param_column`) or a plot parameter "plot.<full_params key>".
        run_filters : dict
            Run parameter filters, {param: value or list of values} (see `ResultsAggregator.filter_runs`).
            Each cell must match exactly one of the remaining runs.
        """
        self.agg = aggregator
        self.x_param = x_param
        self.y_param = y_param
        self.plotter = plotter or AggregatorRateTracePlotter()

        self.fig_params = {**self.DEFAULT_FIG_PARAMS, **(fig_params or {})}
        self.common_params = common_params or {}
        self.subplot_params = subplot_params or {}

        self.run_filters = dict(run_filters or {})
        plot_keys = [key for key in self.run_filters if key.startswith(self.PLOT_PARAM_PREFIX)]
        if plot_keys:
            raise ValueError(
                f"run_filters only select runs, got plot parameters {plot_keys}. "
                "Use x_param/y_param with fig_params['x_values'/'y_values'] for an axis, or common_params for a fixed value."
            )

    @staticmethod
    def _order_values(values: list, order) -> list:
        """Orders the unique values of one grid axis: "ascending", "descending" or None (as given)."""
        if order is None:
            return values
        if order not in ("ascending", "descending"):
            raise ValueError(f"Unknown order '{order}'. Use 'ascending', 'descending' or None.")
        return sorted(values, reverse=(order == "descending"))

    @staticmethod
    def _contains(values: list, value) -> bool:
        """`value in values`, with numbers compared by np.isclose (plot values are often floats, e.g. drive rates)."""
        for candidate in values:
            if isinstance(candidate, (int, float)) and isinstance(value, (int, float)) and not isinstance(value, bool):
                if np.isclose(candidate, value):
                    return True
            elif candidate == value:
                return True
        return False

    def _resolve_axis(self, axis: str, param_mat: np.ndarray, sim_ids: List[str]) -> dict:
        """
        Resolves the "x" or "y" axis into {"kind", "label", "values", ...}:
        run axis: "col" (param_mat column); plot axis: "key" (full_params key) and "run_values"
        ({sim_id: values the plotter discovers for that run, or None}).
        """
        param = getattr(self, f"{axis}_param")
        values = self.fig_params[f"{axis}_values"]
        order = self.fig_params[f"{axis}_order"]

        if param.startswith(self.PLOT_PARAM_PREFIX):
            key = param[len(self.PLOT_PARAM_PREFIX):]
            if not isinstance(self.plotter, BaseAggregatorPlot):
                raise TypeError(f"{axis}_param='{param}' (a plot parameter) needs a BaseAggregatorPlot plotter.")
            run_values = {sim_id: self.plotter.param_values(key, self.agg, sim_id) for sim_id in sim_ids}
            if values is None:
                if any(run_value is None for run_value in run_values.values()):
                    raise ValueError(
                        f"{axis}_param='{param}': {type(self.plotter).__name__} cannot discover the values of '{key}'. "
                        f"Give them in fig_params['{axis}_values']."
                    )
                discovered = []
                for run_value in run_values.values():
                    discovered.extend(value for value in run_value if not self._contains(discovered, value))
                values = self._order_values(discovered, order)
            return {"kind": "plot", "label": key, "key": key, "values": list(values), "run_values": run_values}

        _, col = self.agg.resolve_param_column(param)
        if values is None:
            values = self._order_values(list(dict.fromkeys(param_mat[:, col])), order)
        return {"kind": "run", "label": param, "col": col, "values": list(values)}

    def _cell_sim_id(self, cell_axes: list, param_mat: np.ndarray, sim_ids: List[str]) -> str | None:
        """
        The run of one cell, given [(axis, value), ...] for its x and y; None if there is no matching run
        or the run lacks a plot-axis value. Raises ValueError if several runs match.
        """
        mask = np.ones(len(sim_ids), dtype=bool)
        for axis, value in cell_axes:
            if axis["kind"] == "run":
                mask &= param_mat[:, axis["col"]] == value
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return None
        if indices.size > 1:
            cell = ", ".join(f"{axis['label']} = {value}" for axis, value in cell_axes)
            raise ValueError(
                f"{indices.size} runs match the cell ({cell}): {[sim_ids[i] for i in indices]}. "
                "Narrow them with run_filters."
            )
        sim_id = sim_ids[indices[0]]
        for axis, value in cell_axes:
            if axis["kind"] == "plot":
                run_value = axis["run_values"][sim_id]
                if run_value is not None and not self._contains(run_value, value):
                    return None
        return sim_id

    def __call__(self) -> Tuple[plt.Figure, np.ndarray]:
        # 1. Runs left by the filters (no data is loaded)
        param_mat, sim_ids = self.agg.filter_runs(self.run_filters)
        if len(sim_ids) == 0:
            raise ValueError(f"No simulation runs match the filters: {self.run_filters}")

        # 2. Values of the x (columns) and y (rows) axes
        x_axis = self._resolve_axis("x", param_mat, sim_ids)
        y_axis = self._resolve_axis("y", param_mat, sim_ids)
        x_vals, y_vals = x_axis["values"], y_axis["values"]
        if not x_vals or not y_vals:
            raise ValueError(f"Empty grid axis: {self.x_param} = {x_vals}, {self.y_param} = {y_vals}.")

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

                # Match the run of (x_val, y_val)
                cell_axes = [(x_axis, x_val), (y_axis, y_val)]
                sim_id = self._cell_sim_id(cell_axes, param_mat, sim_ids)

                cell_overrides = self.subplot_params.get((i, j), {})

                # Deep copy plotter per cell to isolate parameter updates (matching GridFigureHook in hooks.py)
                if isinstance(self.plotter, BasePlot):
                    cell_plotter = copy.deepcopy(self.plotter)
                    cell_plotter.full_params.update(self.common_params)
                    for axis, value in cell_axes:
                        if axis["kind"] == "plot":
                            cell_plotter.full_params[axis["key"]] = value
                    cell_plotter.full_params.update(cell_overrides)
                else:
                    cell_plotter = self.plotter

                # Column Headers (top row only)
                if i == 0 and self.fig_params.get("show_row_col_labels", True):
                    cell_title = cell_overrides.get("title", f"{x_axis['label']} = {x_val}")
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

                if sim_id is not None:
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
                    row_text = f"{y_axis['label']} = {y_val}"
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
