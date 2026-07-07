from pathlib import Path
from typing import Dict
import numpy as np
import copy
import gc
import pickle

from .workflows import run_basic_workflow
from .interfaces import BasicWorkflowHook

from ..data_structures.base import BaseResults, BaseMFResults, BaseSNNResults, BaseInspectionResults
from ..data_structures.inspection import SpontInspectionResults, DynamicStimulusInspectionResults

from ..plotting import fig_plots

from pydantic import BaseModel

INSPECTION_PARMAS_WITHOUT_UPDATE = {
    "network.neurons.exc_neuron.neuron_params.a",
    "network.neurons.exc_neuron.neuron_params.b",
}


class SpontActivityExtractor:
    """
    Strategy class to extract steady-state metrics from raw simulation results.
    """

    DEFAULT_UNITS = {
        "exc_rate_time_mean" : "Hz",
        "exc_rate_time_std" : "Hz",
        "inh_rate_time_mean" : "Hz",
        "inh_rate_time_std" : "Hz",
        "exc_voltage_time_mean" : "mV",
        "exc_voltage_time_std" : "mV",
        "inh_voltage_time_mean" : "mV",
        "inh_voltage_time_std" : "mV",
        "exc_adaptation_time_mean" : "nA",
        "exc_adaptation_time_std" : "nA",
        "inh_adaptation_time_mean" : "nA",
        "inh_adaptation_time_std" : "nA",
    }

    ALLOWED_MEASURED_VARS = list(DEFAULT_UNITS.keys())  # just for explicitness


    def __init__(self, measured_variables: list[str], start_time: float = 0.0, end_time: float = np.inf):
        self.start_time = start_time
        self.end_time = end_time
        self.measured_variables = measured_variables

    def extract(self, result: BaseMFResults | BaseSNNResults) -> dict[str, float]:
        """
        Slices the time array and computes time-averages and standard deviations.
        """
        extracted_data = {}
        
        mask = (result.times() >= self.start_time) & (result.times() <= self.end_time)
        
        for variable in self.measured_variables:
            base_var = variable.replace("_time_mean", "").replace("_time_std", "")
            raw_data = getattr(result, f"{base_var}_mean")() # e.g., result.exc_rate_mean
            
            steady_state_data = raw_data[mask]
            
            if variable.endswith("_time_mean"):
                extracted_data[variable] = steady_state_data.mean()
            elif variable.endswith("_time_std"):
                extracted_data[variable] = steady_state_data.std()
            else:
                raise ValueError(f"Variable '{variable}' must end in '_time_mean' or '_time_std'.")
                
        return extracted_data


class ComparisonExtractor:
    """
    Extracts error metrics (RMSE, Bias, Variance, Pearson) by comparing 
    a target result (MF) against a ground-truth result (SNN).
    
    Assumes dt is identical for both simulators.
    
    This extractor is used when comparing dynamic stimulus responses, 
    where the time averaging does not make sense and we want to compare the full time series.
    """

    def __init__(self, measured_variables: list[str], start_time: float = 0.0, end_time: float = np.inf):
        self.start_time = start_time
        self.end_time = end_time
        self.measured_variables = measured_variables

    # Helper methods for computing metrics

    def _calc_error_mean(self, error, gt, target):
        return np.mean(error)

    def _calc_error_std(self, error, gt, target):
        return np.std(error)

    def _calc_rmse(self, error, gt, target):
        return np.sqrt(np.mean(error**2))

    def _calc_pearson(self, error, gt, target):
        if np.std(gt) == 0 or np.std(target) == 0:
            return np.nan
        return np.corrcoef(gt, target)[0, 1]

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

        extracted_data = {}
        
        gt_times = ground_truth.times()
        target_times = target.times()

        start_time = max(self.start_time, gt_times[0], target_times[0])
        end_time = min(self.end_time, gt_times[-1], target_times[-1])

        gt_mask = (gt_times >= start_time) & (gt_times <= end_time)
        target_mask = (target_times >= start_time) & (target_times <= end_time)

        assert gt_mask.sum() == target_mask.sum(), "Time masks for ground truth and target must be of the same size."

        for var in self.measured_variables:
            base_var = var.replace("_rmse", "").replace("_error_mean", "").replace("_error_std", "").replace("_pearson", "")
            metric_suffix = var.replace(f"{base_var}_", "")
            
            gt_data = getattr(ground_truth, f"{base_var}_mean")()
            target_data = getattr(target, f"{base_var}_mean")()
            
            gt_masked = gt_data[gt_mask]
            target_masked = target_data[target_mask]
            error = target_masked - gt_masked

            method_name = f"_calc_{metric_suffix}"
            
            if hasattr(self, method_name):
                calculator_method = getattr(self, method_name)
                extracted_data[var] = calculator_method(error, gt_masked, target_masked)
            else:
                raise ValueError(f"Extractor does not know how to compute metric: {metric_suffix}")
                
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
                tf_results=tf_results_dict,
                snn_results=snn_results,
                mf_results=mf_results_list
            )

        extracted_data = {}
        for key, extractor in extractors.items():
            if isinstance(extractor, SpontActivityExtractor):
                extracted_data.get(key, []).append(extractor.extract(snn_results))
                extracted_data.get(key, []).extend([extractor.extract(mf_results) for mf_results in mf_results_list])
            elif isinstance(extractor, ComparisonExtractor):
                extracted_data.get(key, []).extend([
                    extractor.extract(
                        ground_truth=snn_results,
                        target=mf_results
                    ) 
                    for mf_results in mf_results_list
                ])
            else:
                raise ValueError(f"Unknown extractor type: {type(extractor)}")

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
        extractors = extractors or {}

        mf_names = list(self.base_sim_params.mf_models.keys())
        network_names = ["SNN"] + mf_names

        inspection_results = {}
        for key, extractor in extractors.items():

            if isinstance(extractor, SpontActivityExtractor):
                results_container = SpontInspectionResults
            elif isinstance(extractor, ComparisonExtractor):
                results_container = DynamicStimulusInspectionResults
            else:
                raise ValueError(f"Unknown extractor type: {type(extractor)}")

            inspection_results[key] = results_container(
                inspected_param=inspected_param, 
                inspected_values=inspected_values,
                network_names=network_names, 
                measured_variables=extractor.measured_variables,
                network_params=self.base_network_params,
                stimulus_params=self.base_stimulus_params,
            )

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
                
            current_stimulus_config = {"InspectionStimulus" : current_stimulus_params}

            extracted_data = self.run_single_step(
                sim_params=current_sim_params,
                network_params=current_network_params,
                stimulus_config=current_stimulus_config,
                hooks=single_step_hooks,
                extractors=extractors,
            )

            for key, data in extracted_data.items():
                inspection_results[key].add_inspection_data(data)
        
        for hook in inspection_hooks:
            hook(
                identifier=f"{inspected_param.split('.')[-1]} inspection",
                inspection_results=inspection_results
            )
        
        print("\nInspection Complete. Freezing results...")

        for container in inspection_results.values():
            container.freeze()
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
