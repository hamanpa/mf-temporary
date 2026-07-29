import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import copy
import gc
import pickle

from .workflows import run_basic_workflow
from .interfaces import BasicWorkflowHook

from ..data_structures.base import BaseResults, BaseMFResults, BaseSNNResults, BaseInspectionResults
from ..data_structures.inspection import ModelSummaryInspectionResults, ModelComparisonInspectionResults
from ..analysis.comparison_metrics import METRIC_REGISTRY


from pydantic import BaseModel

INSPECTION_PARMAS_WITHOUT_UPDATE = {
    "network.neurons.exc_neuron.neuron_params.a",
    "network.neurons.exc_neuron.neuron_params.b",
}
DELIMETER = ";"

class ModelSummaryExtractor:
    """
    Strategy class to extract steady-state metrics from raw simulation results.

    Steady state means that the system has reached a point where its properties do not change over time.
    This extractor computes time-averaged values and standard deviations for specified measured variables.

    """
    input_mode = "unary"

    DEFAULT_UNITS = ModelSummaryInspectionResults.DEFAULT_UNITS
    ALLOWED_VARIABLES = ModelSummaryInspectionResults.ALLOWED_VARIABLES
    DEFINED_METRICS = ModelSummaryInspectionResults.DEFINED_METRICS

    results_container_class = ModelSummaryInspectionResults

    def __init__(
            self, 
            measured_variables: list[str], 
            metrics: list[str],
            start_time: float = 0.0, 
            end_time: float = np.inf):
        
        self.measured_variables = measured_variables
        self.metrics = metrics
        
        self.start_time = start_time
        self.end_time = end_time

    def _calc_time_mean(self, data: np.ndarray) -> float:
        if data is None:
            return None
        return np.mean(data)

    def _calc_time_std(self, data: np.ndarray) -> float:
        if data is None:
            return None
        return np.std(data)
    
    def _calc_pop_mean(self, data: np.ndarray) -> np.ndarray:
        if data is None:
            return None
        return data

    def _get_results_data(self, result: BaseResults, var: str) -> np.ndarray:
        """
        Smart lookup: handles short names like 'exc_rate' mapping to 'exc_rate_mean',
        or accepts exact method names like 'exc_rate_std'.
        """
        if hasattr(result, f"{var}_mean"):
            return getattr(result, f"{var}_mean")()
        elif hasattr(result, var):
            return getattr(result, var)()
        else:
            raise AttributeError(f"Results object does not have variable '{var}_mean' or '{var}'.")

    def extract(self, result: BaseMFResults | BaseSNNResults | None) -> dict[str, float | np.ndarray]:
        """
        Slices the time array and computes time-averages and standard deviations.
        """

        if result is None:
            # In case the simulation was skipped or failed, return a dictionary with NaN values for all metrics.
            return {f"{var}_{metric}": np.nan for var in self.measured_variables for metric in self.metrics}        

        extracted_data = {}
        mask = (result.times() >= self.start_time) & (result.times() <= self.end_time)
        
        for metric in self.metrics:
            for variable in self.measured_variables:
                # Output a flat dictionary with f"{variable}_{metric}" keys 
                # (perfectly matching what add_inspection_data expects)
                if metric == "pop_mean":
                    # For pop_mean, we want to return the full time series (or the sliced time series)
                    raw_data = self._get_results_data(result, variable)
                    extracted_data[f"{variable}_{metric}"] = self._calc_pop_mean(raw_data)
                elif (isinstance(result, BaseMFResults)
                        and "rate" in variable
                        and metric == "time_std"):
                    # NOTE: For MF results, the std of rates are computed separately
                    # thus make sense to call the dedicated method for std instead of computing it from the mean.
                    raw_data = self._get_results_data(result, variable + "_std")
                    masked_data = raw_data[mask]
                    extracted_data[f"{variable}_{metric}"] = self._calc_time_mean(masked_data)
                else:
                    raw_data = self._get_results_data(result, variable)
                    calculator_method = getattr(self, f"_calc_{metric}")
                    masked_data = raw_data[mask]
                    extracted_data[f"{variable}_{metric}"] = calculator_method(masked_data)
        return extracted_data


class ModelComparisonExtractor:
    """
    Extracts error metrics (RMSE, Bias, Variance, Pearson) by comparing 
    a target result (MF) against a ground-truth result (SNN).
    
    Assumes dt is identical for both simulators.
    
    This extractor is used when comparing dynamic stimulus responses, 
    where the time averaging does not make sense and we want to compare the full time series.
    """

    input_mode = "pairwise"

    DEFINED_METRICS = list(METRIC_REGISTRY.keys())

    def __init__(
            self, 
            measured_variables: list[str], 
            metrics: list[str],
            custom_metrics: dict[str, callable] = None,
            start_time: float = 0.0, 
            end_time: float = np.inf
            ):
        self.start_time = start_time
        self.end_time = end_time
        self.measured_variables = measured_variables
        
        defined_metrics = {metric: METRIC_REGISTRY[metric] for metric in metrics if metric in self.DEFINED_METRICS}
        undefined_metrics = [metric for metric in metrics if metric not in self.DEFINED_METRICS]
        if undefined_metrics:
            raise ValueError(f"Unknown metrics: {undefined_metrics}. Available metrics: {self.DEFINED_METRICS}")
        
        self.metrics = {**defined_metrics, **(custom_metrics or {})}

    results_container_class = ModelComparisonInspectionResults

    def _get_results_data(self, result: BaseResults, var: str) -> np.ndarray:
        """
        Smart lookup: handles short names like 'exc_rate' mapping to 'exc_rate_mean',
        or accepts exact method names like 'exc_rate_std'.
        """
        if hasattr(result, f"{var}_mean"):
            return getattr(result, f"{var}_mean")()
        elif hasattr(result, var):
            return getattr(result, var)()
        else:
            raise AttributeError(f"Results object does not have variable '{var}_mean' or '{var}'.")

    def extract(self, ground_truth: BaseResults, target: BaseResults) -> dict[str, float]:
        """
        Computes error metrics by comparing target results against ground truth.

        Parameters
        ----------
        ground_truth : BaseResults
            The reference results (typically SNN).
        target : BaseResults
            The target results (typically MF).
        Returns
        -------
        dict[str, float]
            A dictionary containing the computed metrics for each measured variable.
        """

        if ground_truth is None or target is None:
            # In case the simulation was skipped or failed, return a dictionary with NaN values for all metrics.
            return {f"{var}_{metric}": np.nan for var in self.measured_variables for metric in self.metrics}        

        extracted_data = {}
        
        gt_times = ground_truth.times()
        target_times = target.times()

        if len(gt_times) >= 2:
            dt = gt_times[1] - gt_times[0]
        else:
            dt = 0.1

        start_time = max(self.start_time, gt_times[0], target_times[0])
        end_time = min(self.end_time, gt_times[-1], target_times[-1])

        gt_mask = (gt_times >= start_time) & (gt_times <= end_time)
        target_mask = (target_times >= start_time) & (target_times <= end_time)

        assert gt_mask.sum() == target_mask.sum(), "Time masks for ground truth and target must be of the same size."

        for var in self.measured_variables:
            gt_data = self._get_results_data(ground_truth, var)
            target_data = self._get_results_data(target, var)
            
            gt_masked = gt_data[gt_mask]
            target_masked = target_data[target_mask]

            for metric, func in self.metrics.items():
                extracted_data[f"{var}_{metric}"] = func(gt_masked, target_masked, dt=dt)

        return extracted_data


class ParameterInspector:
    """
    Master controller for parameter inspections. 

    Runs SNN and MF simulations ONCE per parameter step, and dynamically 
    routes data to multiple extractors (Spont and/or Dynamic) to save memory and time.
    """

    def __init__(self, 
                 base_network_params: BaseModel, 
                 base_stimulus_params: BaseModel, 
                 base_sim_params: BaseModel,
                 project_path: Path | str = None,
                 ): 
        
        self.base_network_params = base_network_params
        self.base_stimulus_params = base_stimulus_params
        self.base_sim_params = base_sim_params

        self.project_path = Path(project_path)


    def run_single_step(
            self,
            inspected_param: str,
            inspected_value: float | int,
            sim_params,
            network_params,
            stimulus_config,
            hooks: list[BasicWorkflowHook] = None,
            extractors: dict = None,
            ):

        hooks = hooks or []
        extractors = extractors or {}

        stimulus_config = {"InspectionStimulus": stimulus_config}  # Wrap in dict for workflow

        basic_results = run_basic_workflow(
            network_params=network_params,
            stimulus_config=stimulus_config,
            sim_params=sim_params
        )

        neuron_results = basic_results["neuron_results"]
        tf_results_dict = basic_results["tf_results"]
        snn_results = basic_results["snn_results"]["InspectionStimulus"]
        mf_results_list = basic_results["mf_results"]["InspectionStimulus"]

        # logic for hooks
        for hook in hooks:
            hook(
                identifier=f"{inspected_param.split('.')[-1]} {inspected_value}",
                neuron_results=neuron_results,
                tf_funcs_results=tf_results_dict,
                snn_results=snn_results,
                network_results_list=[snn_results] + mf_results_list
            )

        extracted_data = []
        for i, extractor in enumerate(extractors):
            extracted_data.append([])
            if extractor.input_mode == "unary":
                extracted_data[i].append(extractor.extract(snn_results))
                extracted_data[i].extend([extractor.extract(mf_results) for mf_results in mf_results_list])
                if "pop_mean" in extractor.metrics:
                    start_times = [snn_results.times()[0]] + [mf_results.times()[0] for mf_results in mf_results_list]
                    end_times = [snn_results.times()[-1]] + [mf_results.times()[-1] for mf_results in mf_results_list]
                    start_time = max(start_times)
                    end_time = min(end_times)
                    time_masks = [(snn_results.times() >= start_time) & (snn_results.times() <= end_time)] + \
                                [(mf_results.times() >= start_time) & (mf_results.times() <= end_time) for mf_results in mf_results_list]
                    for j, extracted_dict in enumerate(extracted_data[i]):
                        for var in extractor.measured_variables:
                            extracted_dict[f"{var}_pop_mean"] = extracted_dict[f"{var}_pop_mean"][time_masks[j]]

            elif extractor.input_mode == "pairwise":
                extracted_data[i].extend([
                    extractor.extract(
                        ground_truth=snn_results,
                        target=mf_results
                    ) 
                    for mf_results in mf_results_list
                ])
            else:
                raise ValueError(f"Unknown extractor input_mode: {extractor.input_mode}")

        # NOTE: if memory issues arise, consider using `del` and `gc.collect()`
        # since Python may store references to large objects in memory even after they go out of scope.

        # del snn_results
        # del mf_results 
        # gc.collect() 

        return extracted_data

    def run_inspection(
            self, 
            inspected_param: str, 
            inspected_values: list | np.ndarray, 
            single_step_hooks: list[BasicWorkflowHook] = None,
            inspection_hooks: list = None,
            extractors: dict = None,
            ) -> Dict[str, BaseInspectionResults]:

        single_step_hooks = single_step_hooks or []
        inspection_hooks = inspection_hooks or []
        extractors = extractors or []

        mf_names = list(self.base_sim_params.mf_models.keys())

        inspection_results = []
        for extractor in extractors:

            results_container = extractor.results_container_class
            if extractor.input_mode == "unary":
                network_names = ["SNN"] + mf_names
            elif extractor.input_mode == "pairwise":
                network_names = mf_names
            else:
                raise ValueError(f"Unknown extractor input_mode: {extractor.input_mode}")

            inspection_results.append(results_container(
                inspected_param=inspected_param, 
                inspected_values=inspected_values,
                network_names=network_names, 
                variables=extractor.measured_variables,  
                metrics=extractor.metrics,               
                network_params=self.base_network_params,
                stimulus_params=self.base_stimulus_params,
            ))

        is_network = inspected_param.startswith("network.")
        is_stimulus = inspected_param.startswith("stimulus.")
        if not (is_network or is_stimulus):
            raise ValueError("inspected_param must start with 'network.' or 'stimulus.'")
        inspected_param_path = inspected_param.split(".", maxsplit=1)[-1]  

        for value in inspected_values:
            print(f"\n--- Inspecting {inspected_param} = {value} ---")
            
            current_network_params = copy.deepcopy(self.base_network_params)
            current_stimulus_params = copy.deepcopy(self.base_stimulus_params)
            current_sim_params = copy.deepcopy(self.base_sim_params)
            
            if is_network:
                current_network_params = inject_pydantic_param(current_network_params, inspected_param_path, value)
            else:
                current_stimulus_params = inject_pydantic_param(current_stimulus_params, inspected_param_path, value)
                
            extracted_data = self.run_single_step(
                inspected_param=inspected_param,
                inspected_value=value,
                sim_params=current_sim_params,
                network_params=current_network_params,
                stimulus_config=current_stimulus_params,
                hooks=single_step_hooks,
                extractors=extractors,
            )

            for i, data in enumerate(extracted_data):
                inspection_results[i].add_inspection_data(data)
        
        print("\nInspection Complete. Freezing results...")

        for container in inspection_results:
            container.freeze()

        for hook in inspection_hooks:
            hook(
                identifier=f"{inspected_param.split('.')[-1]} inspection",
                inspection_results_list=inspection_results
            )
        
        return inspection_results


def inject_pydantic_param(base_model: BaseModel, param_path: str, value: str|float|int) -> BaseModel:
    """
    Returns a new instance of a Pydantic model with a specified parameter updated to a new value
    
    Parameters
    ----------
    base_model : pydantic.BaseModel
        The root configuration model (e.g., network_params or stimulus_params).
    param_path : str
        Dot notation path to the parameter (e.g., 'neurons.exc_neuron.neuron_params.a').
    value : Any
        The new value to assign.
        
    Returns
    -------
    pydantic.BaseModel
        A new, deep-copied instance of the model with the updated parameter.
    """

    model_copy = copy.deepcopy(base_model)
    
    keys = param_path.split('.')
    current_obj = model_copy
    
    for key in keys[:-1]:
        if isinstance(current_obj, BaseModel):
            current_obj = getattr(current_obj, key)
        elif isinstance(current_obj, dict):
            current_obj = current_obj[key]
        else:
            raise ValueError(f"Cannot traverse into {type(current_obj)} for key '{key}'")
        
    setattr(current_obj, keys[-1], value)
    
    return model_copy


class ResultsAggregator:
    """
    Lightweight, pure-NumPy aggregator for multi-inspection project results.
    
    Reads param_combinations.csv, builds a 2D parameter matrix with unique alias and partial path resolution,
    and lazily loads variable arrays from project_dir/data/{sim_id}/ with LRU memory caching.
    """

    def __init__(self, project_dir: Union[str, Path], cache_size: int = 256):
        self.project_dir = Path(project_dir)
        self.csv_path = self.project_dir / "param_combinations.csv"
        self.data_dir = self.project_dir / "data"
        self.cache_size = cache_size
        self._cache: Dict[Tuple, np.ndarray] = {}

        self.sim_ids: List[str] = []
        self.param_names: List[str] = []
        self.param_col_map: Dict[str, int] = {}
        self.param_matrix: np.ndarray = None

        self._load_param_combinations()

        self.available_models = self.get_available_variables()

    def _load_param_combinations(self):
        """Loads param_combinations.csv into a 2D NumPy array and maps headers."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"param_combinations.csv not found in '{self.project_dir}'")

        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=DELIMETER)
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

    def get_available_variables(self, sim_ids: str|List[str] = None, full_iter=False) -> Dict[str, List[str]]:
        """
        Inspects saved .npz files in the first available simulation directory
        and returns a dictionary of available variable keys per model category.
        """
        available_models = {}

        if sim_ids is None:
            if full_iter:
                sim_ids = self.sim_ids
            else:
                sim_ids = [self.sim_ids[0]]
        elif isinstance(sim_ids, str):
            sim_ids = [sim_ids]
        
        for sim_id in sim_ids:
            sim_dir = self.data_dir / sim_id

            for npz_file in sim_dir.glob("*.npz"):
                cat_name = npz_file.stem.split("results")
                model_name = cat_name[0].rstrip("_")
                if model_name not in available_models:
                    available_models[model_name] = {
                        "variables": set(),
                        "stimuli": set(),
                    }
                if len(cat_name) > 1:
                    stim_name = cat_name[1].lstrip("_")
                    available_models[model_name]["stimuli"].add(stim_name)


                with np.load(npz_file, allow_pickle=True) as data:
                    available_models[model_name]["variables"].update(data.keys())

        return available_models

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

    def get_results(
        self,
        variable: str,
        sim_name: str = "SNN",
        stim_name: str = "SpontActivity",
        **param_filters
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Queries and retrieves stacked array results for filtered parameter combinations.

        Parameters
        ----------
        variable : str
            Variable name to extract (e.g., 'exc_rate_mean', 'exc_voltage_all', 'drive_rate_mean').
        sim_name : str
            Name of simulator/model ('SNN', 'divolo_second_order', etc.).
        stim_name : str
            Name of stimulus ('SpontActivity', etc.).
        **param_filters : dict
            Parameter equality or list filters (e.g., b=5.0, drive_rate=[0.0, 5.0]).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]
            - data_array: Stacked NumPy array of shape (N_filtered_sims, T_time, ...)
            - filtered_param_matrix: 2D NumPy array of parameter values for matching runs
            - param_names: List of parameter names corresponding to the columns
            - filtered_sim_ids: List of matching simulation hash IDs
        """
        mask = np.ones(len(self.sim_ids), dtype=bool)

        for param_key, target_val in param_filters.items():
            full_name, col_idx = self.resolve_param_column(param_key)
            column_vals = self.param_matrix[:, col_idx]

            if isinstance(target_val, (list, tuple, set, np.ndarray)):
                match_mask = np.isin(column_vals, list(target_val))
            else:
                match_mask = (column_vals == target_val)

            mask = mask & match_mask

        filtered_indices = np.where(mask)[0]
        filtered_sim_ids = [self.sim_ids[i] for i in filtered_indices]
        filtered_params = self.param_matrix[filtered_indices, :]

        if len(filtered_sim_ids) == 0:
            return np.array([]), filtered_params, self.param_names, filtered_sim_ids

        loaded_arrays = []
        for sim_id in filtered_sim_ids:
            arr = self._load_variable(sim_id, sim_name, stim_name, variable)
            loaded_arrays.append(arr)

        data_array = np.array(loaded_arrays)

        # Warn user if any returned arrays contain NaN values
        if data_array.size > 0:
            try:
                nan_mask = np.isnan(data_array.astype(float))
                if np.any(nan_mask):
                    nan_sims = [filtered_sim_ids[i] for i in range(len(filtered_sim_ids)) if np.any(nan_mask[i])]
                    print(
                        f"[ResultsAggregator Warning] Query for variable '{variable}' ({sim_name}) contains NaN values "
                        f"in {len(nan_sims)}/{len(filtered_sim_ids)} simulation run(s): {nan_sims}"
                    )
            except (ValueError, TypeError):
                pass

        return data_array, filtered_params, self.param_names, filtered_sim_ids

    def _load_variable(self, sim_id: str, sim_name: str, stim_name: str, variable: str) -> np.ndarray:
        cache_key = (sim_id, sim_name.lower(), stim_name, variable)
        if cache_key in self._cache:
            return self._cache[cache_key]

        sim_dir = self.data_dir / sim_id
        safe_stim_name = str(stim_name).replace(" ", "_")
        file_name = f"{sim_name.lower()}_results_{safe_stim_name}.npz"
        file_path = sim_dir / file_name

        if not file_path.exists():
            alt_file_name = f"{sim_name.lower()}_results.npz"
            if (sim_dir / alt_file_name).exists():
                file_path = sim_dir / alt_file_name
            else:
                candidates = list(sim_dir.glob(f"*{sim_name.lower()}*.npz")) if sim_dir.exists() else []
                if candidates:
                    file_path = candidates[0]
                else:
                    raise FileNotFoundError(f"Result file '{file_name}' not found in '{sim_dir}'")

        npz_data = np.load(file_path, allow_pickle=True)
        target_key = variable
        if target_key not in npz_data:
            for cand in [f"{variable}_pop_mean", f"{variable}_mean", f"{variable}_all"]:
                if cand in npz_data:
                    target_key = cand
                    break

        if target_key not in npz_data:
            available_keys = list(npz_data.keys())
            raise KeyError(f"Variable '{variable}' not found in '{file_path}'. Available variables in file: {available_keys}")

        arr = npz_data[target_key]

        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = arr

        return arr

