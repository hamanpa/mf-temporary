"""
Sweep-level summaries of a multi-inspection project (`controller.ResultsAggregator`).

One scalar per run: a time reduction (`TIME_REDUCTIONS`) of a saved series `{variable}_{metric}` within a
time window, arranged along one swept run parameter (x) or two (x, y). Plots draw these summaries, they
compute nothing themselves.

Each value of the swept parameter(s) must select exactly one run of those left by `run_filters`
(see `ResultsAggregator.filter_runs`): every other varying parameter has to be fixed by the filters.
Joint parameters (comma keys in `inspected_params.yaml`) move together, so they count as one.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

TIME_REDUCTIONS = {
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
}

# The last 2000 ms: the relaxation time is not known in advance, the end of the trace is the most likely to be relaxed
DEFAULT_TIME_WINDOW = (-2000.0, None)

# Reduced scalars {(file, modification time, array key, reduction, window, unit, time unit): value}
_CACHE_SIZE = 100_000
_cache: Dict[tuple, float] = {}


@dataclass(frozen=True)
class SweepSummary:
    """
    Time reductions of one saved series of one model along one or two swept run parameters.

    Attributes
    ----------
    x_param, y_param : str
        Full names (param_combinations.csv columns) of the swept parameters; `y_param` is None in 1D.
    x_values, y_values : np.ndarray
        Sorted values of the swept parameters, as in the configs (MFT units); `y_values` is None in 1D.
    values : np.ndarray
        Reduced values [unit], shape (n_x,) or (n_y, n_x); NaN where no run matches or its data are missing.
    sim_ids : np.ndarray
        Run of each value (object array of the same shape; None where no run matches).
    missing : list of str
        Runs whose results file or array is missing (their values are NaN).
    unit : str or None
        Unit of `values` (the requested one, or as stored; None if unknown).
    """
    x_param: str
    x_values: np.ndarray
    y_param: str | None
    y_values: np.ndarray | None
    values: np.ndarray
    sim_ids: np.ndarray
    missing: List[str]
    unit: str | None


def window_mask(times: np.ndarray, time_window: Tuple[float | None, float | None]) -> np.ndarray:
    """
    Mask of `times` inside `time_window` = (start, end), in the unit of `times`.
    None: the start / end of the trace; a negative value counts back from the last time
    (e.g. (-2000, None) with times in ms: the last 2000 ms).
    """
    start, end = time_window
    last = times[-1]
    start = times[0] if start is None else (last + start if start < 0 else start)
    end = last if end is None else (last + end if end < 0 else end)
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        raise ValueError(f"Time window {time_window} contains no samples (times from {times[0]} to {last}).")
    return mask


def time_reduce(
        results, variable: str, metric: str | None = "pop_mean", reduction: str = "mean",
        time_window: Tuple[float | None, float | None] = DEFAULT_TIME_WINDOW,
        unit: str | None = None, time_unit: str = "ms",
) -> float:
    """
    Time reduction of `{variable}_{metric}` of one saved results file within `time_window` [time_unit].

    Parameters
    ----------
    results : SavedResults
        View of one results file (`ResultsAggregator.results`).
    reduction : str
        Key of `TIME_REDUCTIONS`. NaN in the window gives NaN.
    unit : str or None
        Unit of the returned value (None: as stored).

    Raises
    ------
    KeyError
        If the file has no such array (see `SavedResults.get`).
    """
    if reduction not in TIME_REDUCTIONS:
        raise ValueError(f"Unknown reduction '{reduction}'. Use one of {list(TIME_REDUCTIONS)}.")
    time_window = tuple(time_window)
    cache_key = (
        results.file_path, results.file_path.stat().st_mtime_ns, results.key(variable, metric),
        reduction, time_window, unit, time_unit,
    )
    if cache_key in _cache:
        return _cache[cache_key]

    data = results.get(variable, metric, unit=unit)
    times = results.times(time_unit)
    if data.shape[:1] != times.shape:
        raise ValueError(f"'{results.key(variable, metric)}' in '{results.file_path}' has shape {data.shape}, times {times.shape}.")
    value = float(TIME_REDUCTIONS[reduction](data[window_mask(times, time_window)]))

    if len(_cache) >= _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    _cache[cache_key] = value
    return value


def _axis_values(values, param: str) -> np.ndarray:
    """Sorted unique values of a swept parameter; only numbers can span an axis."""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        raise ValueError(f"Only numeric parameters can span a sweep axis; '{param}' has values {sorted(map(str, values))}.")
    return np.array(sorted(values), dtype=float)


def runs_by_values(aggregator, params: List[str], run_filters: dict | None = None) -> Tuple[List[str], Dict[tuple, str]]:
    """
    Runs left by `run_filters` keyed by their values of `params`.

    Returns
    -------
    Tuple[List[str], Dict[tuple, str]]
        - the full names of `params` (see `ResultsAggregator.resolve_param_column`);
        - {(value of each param): sim_id}.

    Raises
    ------
    ValueError
        If no run matches, or if one combination of values matches several runs (another parameter still varies;
        the message names it).
    """
    param_mat, sim_ids = aggregator.filter_runs(run_filters)
    if not sim_ids:
        raise ValueError(f"No runs match run_filters {run_filters}.")
    columns = [aggregator.resolve_param_column(param) for param in params]
    names = [name for name, _ in columns]
    if len(set(names)) != len(names):
        raise ValueError(f"The swept parameters {params} resolve to the same column(s): {names}.")

    by_values: Dict[tuple, List[int]] = {}
    for index, row in enumerate(param_mat):
        by_values.setdefault(tuple(row[col] for _, col in columns), []).append(index)

    clashes = {values: indices for values, indices in by_values.items() if len(indices) > 1}
    if clashes:
        values, indices = next(iter(clashes.items()))
        varying = aggregator.analyze_parameter_grid(param_mat[indices])["varying_params"]
        raise ValueError(
            f"{len(clashes)} value(s) of {names} match several runs, e.g. {values}: {[sim_ids[i] for i in indices]}. "
            f"Parameters that still vary there: {list(varying)}. Fix them with run_filters."
        )
    return names, {values: sim_ids[indices[0]] for values, indices in by_values.items()}


def sweep_summary(
        aggregator, variable: str, model: str, stim_name: str, x_param: str, y_param: str | None = None,
        metric: str | None = "pop_mean", reduction: str = "mean",
        time_window: Tuple[float | None, float | None] = DEFAULT_TIME_WINDOW,
        unit: str | None = None, time_unit: str = "ms", run_filters: dict | None = None,
) -> SweepSummary:
    """
    Time reduction (see `time_reduce`) of `{variable}_{metric}` of `model` for stimulus `stim_name`,
    for every run left by `run_filters`, along `x_param` (1D) or `x_param` and `y_param` (2D).

    Parameters
    ----------
    x_param, y_param : str
        Swept run parameters, as in param_combinations.csv (may be abbreviated, see
        `ResultsAggregator.resolve_param_column`). Each value (1D) or cell (2D) must match exactly one run.
    time_window : (start, end)
        In `time_unit`; see `window_mask`.

    Returns
    -------
    SweepSummary
        Runs without the results file or the array are listed in `missing` (NaN values), not raised.
    """
    params = [x_param] if y_param is None else [x_param, y_param]
    names, runs = runs_by_values(aggregator, params, run_filters)
    x_values = _axis_values({values[0] for values in runs}, names[0])
    y_values = None if y_param is None else _axis_values({values[1] for values in runs}, names[1])

    shape = (len(x_values),) if y_values is None else (len(y_values), len(x_values))
    values = np.full(shape, np.nan)
    cell_sim_ids = np.full(shape, None, dtype=object)
    missing = []
    stored_unit = None

    for run_values, sim_id in runs.items():
        index = (int(np.searchsorted(x_values, run_values[0])),)
        if y_values is not None:
            index = (int(np.searchsorted(y_values, run_values[1])),) + index
        cell_sim_ids[index] = sim_id
        try:
            results = aggregator.results(sim_id, model, stim_name)
            values[index] = time_reduce(results, variable, metric, reduction, time_window, unit, time_unit)
        except (FileNotFoundError, KeyError):
            missing.append(sim_id)
            continue
        if stored_unit is None:
            stored_unit = results.unit(variable, metric)

    return SweepSummary(
        x_param=names[0], x_values=x_values,
        y_param=None if y_param is None else names[1], y_values=y_values,
        values=values, sim_ids=cell_sim_ids, missing=missing,
        unit=unit if unit is not None else stored_unit,
    )
