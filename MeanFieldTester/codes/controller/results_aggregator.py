"""
Access to the results of a multi-inspection sweep (`projects/<name>/`).

- `ResultsAggregator` reads `param_combinations.csv` (one row per run), filters runs by parameter values
  and loads the saved results of each run from `data/<sim_id>/`.
- `SavedResults` is a lazy, unit-aware view of one saved results file (`{model}_results_{stim}.npz`):
  its keys and units are read up front, arrays only when accessed (`get`).

Saved array keys follow `{variable}_{metric}` (e.g. `exc_rate_pop_mean`); keys without a metric
(e.g. `times`, `drive_rate`, `exc_spikes`, the neuron grids `exc_rate_grid`, `out_rate_mean`) are read with `metric=None`.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Callable

import numpy as np

from .run_params import load_run_params
from ..transfer_function import get_transfer_function
from ..network_params.translators import get_unit_multiplier
from ..utils.file_helpers import NPZ_UNITS_KEY, decode_npz_units

DELIMITER = ";"
RESULTS_FILE_INFIX = "_results_"


class SavedResults:
    """
    Lazy, unit-aware view of one saved results file (`data/<sim_id>/{model}_results_{stim}.npz`).

    Arrays are returned in the unit stored in the file (MFT units, see `units`) or converted to `unit`
    with `get_unit_multiplier`, the same conversion as the results classes. Returned arrays are read-only
    (they are shared through the aggregator's cache).
    """

    def __init__(self, file_path: Path, load_array: Callable[[Path, str], np.ndarray]):
        self.file_path = Path(file_path)
        self._load_array = load_array
        with np.load(self.file_path, allow_pickle=True) as data:
            self.keys: Tuple[str, ...] = tuple(key for key in data.files if key != NPZ_UNITS_KEY)
            self.units: Dict[str, str] = decode_npz_units(data)

    @staticmethod
    def key(variable: str, metric: str | None = "pop_mean") -> str:
        """Array key of `variable` and `metric`: `{variable}_{metric}`, or `variable` itself if `metric` is None."""
        return variable if metric is None else f"{variable}_{metric}"

    def has(self, variable: str, metric: str | None = "pop_mean") -> bool:
        return self.key(variable, metric) in self.keys

    def unit(self, variable: str, metric: str | None = "pop_mean") -> str | None:
        """Stored unit ("" for unitless quantities, None if the file has no unit for this key)."""
        return self.units.get(self.key(variable, metric))

    def get(self, variable: str, metric: str | None = "pop_mean", unit: str | None = None) -> np.ndarray:
        """
        Array of `variable` and `metric` (see `key`).

        Parameters
        ----------
        unit : str or None
            Target unit. None returns the array as stored (see `units`).

        Raises
        ------
        KeyError
            If the file has no such key (the message lists the available keys).
        ValueError
            If `unit` is requested but the file stores no unit for the key (files saved before units were
            stored), or the quantity is unitless; nothing is converted on assumptions.
        """
        key = self.key(variable, metric)
        if key not in self.keys:
            raise KeyError(f"'{key}' not in '{self.file_path}'. Available keys: {sorted(self.keys)}")
        data = self._load_array(self.file_path, key)
        if unit is None:
            return data

        source_unit = self.units.get(key)
        if source_unit is None:
            raise ValueError(f"'{key}' in '{self.file_path}' has no stored unit; cannot convert to '{unit}'.")
        if source_unit == unit:
            return data
        if source_unit == "":
            raise ValueError(f"'{key}' in '{self.file_path}' is unitless; cannot convert to '{unit}'.")
        return data * get_unit_multiplier(source_unit, unit)

    def times(self, unit: str | None = None) -> np.ndarray:
        return self.get("times", metric=None, unit=unit)


class ResultsAggregator:
    """
    Lightweight, pure-NumPy aggregator for multi-inspection project results.

    Reads param_combinations.csv, builds a 2D parameter matrix with unique alias and partial path resolution,
    and gives lazy, unit-aware access to the saved results of each run (`results`), with an LRU array cache.
    """

    def __init__(self, project_dir: Union[str, Path], cache_size: int = 256):
        self.project_dir = Path(project_dir)
        self.csv_path = self.project_dir / "param_combinations.csv"
        self.data_dir = self.project_dir / "data"
        self.cache_size = cache_size
        self._array_cache: Dict[Tuple[Path, str], np.ndarray] = {}
        self._views: Dict[Path, SavedResults] = {}

        self.sim_ids: List[str] = []
        self.param_names: List[str] = []
        self.param_col_map: Dict[str, int] = {}
        self.param_matrix: np.ndarray = None

        self._load_param_combinations()

    def _load_param_combinations(self):
        """Loads param_combinations.csv into a 2D NumPy array and maps headers."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"param_combinations.csv not found in '{self.project_dir}'")

        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=DELIMITER)
            header = next(reader)
            rows = [r for r in reader if r]

        self.sim_ids = [r[0] for r in rows]
        self.param_names = header[1:]

        for idx, p_name in enumerate(self.param_names):
            self.param_col_map[p_name] = idx

        parsed_rows = []
        for r in rows:
            parsed_row = []
            for val_str in r[1:]:
                val_str = val_str.strip()
                if val_str.lower() == 'true':
                    parsed_row.append(True)
                elif val_str.lower() == 'false':
                    parsed_row.append(False)
                else:
                    try:
                        f_val = float(val_str)
                        parsed_row.append(int(f_val) if f_val.is_integer() else f_val)
                    except ValueError:
                        parsed_row.append(val_str)
            parsed_rows.append(parsed_row)

        self.param_matrix = np.array(parsed_rows, dtype=object)

    def resolve_param_column(self, param_key: str) -> Tuple[str, int]:
        """
        Resolves a short alias or partial sub-path parameter key to exact CSV header column index.
        Matches if all dot-separated tokens in param_key appear in order inside the full CSV column header.
        Raises ValueError if ambiguous across multiple columns.
        """
        # 1. Exact match
        if param_key in self.param_col_map:
            return param_key, self.param_col_map[param_key]

        # 2. Token sub-sequence search
        key_parts = [p.strip() for p in param_key.split('.') if p.strip()]

        matches = []
        for full_name in self.param_names:
            full_parts = full_name.split('.')
            curr_idx = 0
            is_match = True
            for part in key_parts:
                try:
                    found_idx = full_parts.index(part, curr_idx)
                    curr_idx = found_idx + 1
                except ValueError:
                    is_match = False
                    break

            if is_match:
                matches.append(full_name)

        if len(matches) == 1:
            matched_name = matches[0]
            return matched_name, self.param_col_map[matched_name]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous parameter key '{param_key}'. Matches {len(matches)} columns:\n" +
                "\n".join([f"  - {m}" for m in matches]) +
                f"\nPlease specify a more specific parameter path."
            )
        else:
            raise KeyError(f"Parameter '{param_key}' not found in CSV headers. Available headers: {self.param_names}")

    # ------------------------------------------------------------------
    # Saved results of a run
    # ------------------------------------------------------------------

    def available_results(self, sim_id: str) -> Dict[str, List[str]]:
        """{model: [stimulus names]} of the results files saved for a run (from the file names, nothing is loaded)."""
        available = {}
        for file_path in sorted((self.data_dir / sim_id).glob(f"*{RESULTS_FILE_INFIX}*.npz")):
            model, _, stim_name = file_path.stem.partition(RESULTS_FILE_INFIX)
            available.setdefault(model, []).append(stim_name)
        return available

    def results_path(self, sim_id: str, model: str, stim_name: str) -> Path:
        """Path of `data/<sim_id>/{model}_results_{stim}.npz` (model name case-insensitive, spaces in stim_name → _)."""
        safe_stim_name = str(stim_name).replace(" ", "_")
        file_path = self.data_dir / sim_id / f"{model.lower()}{RESULTS_FILE_INFIX}{safe_stim_name}.npz"
        if not file_path.exists():
            raise FileNotFoundError(
                f"No results file '{file_path.name}' for run '{sim_id}'. "
                f"Available {{model: stimuli}}: {self.available_results(sim_id)}"
            )
        return file_path

    def results(self, sim_id: str, model: str, stim_name: str) -> SavedResults:
        """Lazy, unit-aware view of the saved results of (run, model, stimulus), see `SavedResults`."""
        file_path = self.results_path(sim_id, model, stim_name)
        if file_path not in self._views:
            self._views[file_path] = SavedResults(file_path, self._load_array)
        return self._views[file_path]

    def _load_array(self, file_path: Path, key: str) -> np.ndarray:
        """Loads one array of a results file (LRU-cached, read-only)."""
        cache_key = (file_path, key)
        if cache_key in self._array_cache:
            return self._array_cache[cache_key]

        with np.load(file_path, allow_pickle=True) as data:
            arr = data[key]
        arr.flags.writeable = False

        if len(self._array_cache) >= self.cache_size:
            self._array_cache.pop(next(iter(self._array_cache)))
        self._array_cache[cache_key] = arr
        return arr

    def get_available_variables(self, sim_ids: str | List[str] = None, full_iter=False) -> Dict[str, Dict[str, set]]:
        """
        {model: {"variables": set of array keys, "stimuli": set of stimulus names}} of the saved results
        of the first run (`full_iter=True`: of all runs; or of the given `sim_ids`).
        """
        if sim_ids is None:
            sim_ids = self.sim_ids if full_iter else [self.sim_ids[0]]
        elif isinstance(sim_ids, str):
            sim_ids = [sim_ids]

        available_models = {}
        for sim_id in sim_ids:
            for model, stim_names in self.available_results(sim_id).items():
                entry = available_models.setdefault(model, {"variables": set(), "stimuli": set()})
                for stim_name in stim_names:
                    entry["stimuli"].add(stim_name)
                    entry["variables"].update(self.results(sim_id, model, stim_name).keys)
        return available_models

    def get_units(self, sim_id: str, sim_name: str, stim_name: str) -> Dict[str, str]:
        """
        Units {array_key: unit} stored in a run's results file ({} for files saved before units were stored).
        For neuron grids use sim_name='exc_neuron', stim_name='steady_state'.
        """
        return dict(self.results(sim_id, sim_name, stim_name).units)

    # ------------------------------------------------------------------
    # Configs of a run
    # ------------------------------------------------------------------

    def run_params_dir(self, sim_id: str) -> Path:
        """Folder with the validated configs a run was simulated with (`data/<sim_id>/params/`)."""
        return self.data_dir / sim_id / "params"

    def load_run_params(self, sim_id: str):
        """
        Loads the configs a run was simulated with: (network_params, workflow_params, stimuli).
        The workflow params contain the fitted TF coefficients (`mf_models.*.transfer_function.tf_fits`).
        """
        params_dir = self.run_params_dir(sim_id)
        if not params_dir.exists():
            raise FileNotFoundError(
                f"No saved run params for '{sim_id}' ({params_dir}). Runs made before run params were saved cannot be reloaded."
            )
        return load_run_params(params_dir)

    def load_transfer_functions(self, sim_id: str, mf_model_name: str) -> Dict[str, Any]:
        """
        Transfer functions of one MF model of a run, with the coefficients fitted during the run
        (no refitting): {neuron_name: BaseTransferFunction}.
        """
        network_params, workflow_params, _ = self.load_run_params(sim_id)
        if mf_model_name not in workflow_params.mf_models:
            raise KeyError(f"MF model '{mf_model_name}' not in run '{sim_id}'. Available: {list(workflow_params.mf_models)}")
        tf_params = workflow_params.mf_models[mf_model_name].transfer_function

        transfer_functions = {}
        for neuron_name in network_params.internal_neurons:
            if neuron_name not in tf_params.tf_fits:
                raise KeyError(f"Run '{sim_id}', model '{mf_model_name}': no fitted TF coefficients for '{neuron_name}'.")
            tf = get_transfer_function(tf_params.tf_model.model_name, neuron_name, network_params, tf_params)
            tf.set_fitted_parameters(tf_params.tf_fits[neuron_name].model_dump())
            transfer_functions[neuron_name] = tf
        return transfer_functions

    # ------------------------------------------------------------------
    # Queries over runs
    # ------------------------------------------------------------------

    def analyze_parameter_grid(self, params_matrix: np.ndarray, param_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyzes a parameter matrix (from get_results) to determine:
          - degrees_of_freedom: number of parameters that vary across the filtered set
          - varying_params: dict mapping varying parameter names -> list of unique values
          - constant_params: dict mapping constant parameter names -> constant value
          - x_param: name of 1st varying parameter (for plotting X-axis)
          - y_param: name of 2nd varying parameter (for plotting Y-axis grid)
        """
        if param_names is None:
            param_names = self.param_names

        if params_matrix is None or params_matrix.size == 0 or len(param_names) == 0:
            return {
                "degrees_of_freedom": 0,
                "varying_params": {},
                "constant_params": {},
                "x_param": None,
                "y_param": None
            }

        varying = {}
        constant = {}

        for j, p_name in enumerate(param_names):
            col = params_matrix[:, j]
            # Distinct values preserving insertion order
            unique_vals = list(dict.fromkeys(col))
            if len(unique_vals) > 1:
                varying[p_name] = unique_vals
            else:
                constant[p_name] = unique_vals[0] if len(unique_vals) > 0 else None

        var_names = list(varying.keys())
        return {
            "degrees_of_freedom": len(varying),
            "varying_params": varying,
            "constant_params": constant,
            "x_param": var_names[0] if len(var_names) >= 1 else None,
            "y_param": var_names[1] if len(var_names) >= 2 else None
        }

    def filter_runs(self, run_filters: dict | None = None) -> Tuple[np.ndarray, List[str]]:
        """
        Runs matching the parameter filters, without loading any data.

        Parameters
        ----------
        run_filters : dict or None
            {param: value or list of values}; keys as in param_combinations.csv, may be abbreviated
            (see `resolve_param_column`), e.g. {"exc_neuron.exc_neuron.tau_rec": [0, 200]}.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            - filtered_param_matrix: 2D array of parameter values of the matching runs
            - filtered_sim_ids: matching simulation hash IDs
        """
        mask = np.ones(len(self.sim_ids), dtype=bool)

        for param_key, target_val in (run_filters or {}).items():
            full_name, col_idx = self.resolve_param_column(param_key)
            column_vals = self.param_matrix[:, col_idx]

            if isinstance(target_val, (list, tuple, set, np.ndarray)):
                match_mask = np.isin(column_vals, list(target_val))
            else:
                match_mask = (column_vals == target_val)

            mask = mask & match_mask

        filtered_indices = np.where(mask)[0]
        return self.param_matrix[filtered_indices, :], [self.sim_ids[i] for i in filtered_indices]

    def get_results(
        self,
        variable: str,
        sim_name: str,
        stim_name: str,
        metric: str | None = "pop_mean",
        unit: str | None = None,
        run_filters: dict | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Queries and retrieves stacked array results for filtered parameter combinations.

        Parameters
        ----------
        variable, metric : str
            Array `{variable}_{metric}` (e.g. 'exc_rate', 'pop_mean'); metric=None for keys without a metric
            (e.g. 'times', 'drive_rate'). See `SavedResults.get`.
        sim_name : str
            Name of simulator/model as in the file names ('snn', 'stp_dynamic', 'exc_neuron', ...).
        stim_name : str
            Name of stimulus ('SpontActivity0_5', 'Pulse', 'steady_state', ...).
        unit : str or None
            Target unit (None: as stored).
        run_filters : dict or None
            Parameter filters, see `filter_runs`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]
            - data_array: Stacked NumPy array of shape (N_filtered_sims, T_time, ...)
            - filtered_param_matrix: 2D NumPy array of parameter values for matching runs
            - param_names: List of parameter names corresponding to the columns
            - filtered_sim_ids: List of matching simulation hash IDs
        """
        filtered_params, filtered_sim_ids = self.filter_runs(run_filters)

        if len(filtered_sim_ids) == 0:
            return np.array([]), filtered_params, self.param_names, filtered_sim_ids

        loaded_arrays = [
            self.results(sim_id, sim_name, stim_name).get(variable, metric=metric, unit=unit)
            for sim_id in filtered_sim_ids
        ]
        data_array = np.array(loaded_arrays)

        # Warn user if any returned arrays contain NaN values
        if data_array.size > 0:
            try:
                nan_mask = np.isnan(data_array.astype(float))
                if np.any(nan_mask):
                    nan_sims = [filtered_sim_ids[i] for i in range(len(filtered_sim_ids)) if np.any(nan_mask[i])]
                    print(
                        f"[ResultsAggregator Warning] Query for '{SavedResults.key(variable, metric)}' ({sim_name}) contains NaN values "
                        f"in {len(nan_sims)}/{len(filtered_sim_ids)} simulation run(s): {nan_sims}"
                    )
            except (ValueError, TypeError):
                pass

        return data_array, filtered_params, self.param_names, filtered_sim_ids
