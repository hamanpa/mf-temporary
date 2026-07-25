from pathlib import Path
from typing import Dict
from unittest import result
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
