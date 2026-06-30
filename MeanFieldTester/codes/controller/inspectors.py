from typing import Dict
import numpy as np
import copy
import gc

from ..mf_simulation import run_mf_simulation_workflow
from ..snn_simulation import run_snn_simulation_workflow
from ..data_structures.base import BaseResults, BaseMFResults, BaseSNNResults, BaseInspectionResults
from ..data_structures.inspection import SpontInspectionResults, DynamicStimulusInspectionResults

from ..plotting import fig_plots

from pydantic import BaseModel


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
                 base_sim_params: BaseModel): 
        
        self.base_network_params = base_network_params
        self.base_stimulus_params = base_stimulus_params
        self.base_sim_params = base_sim_params

    def run_inspection(
            self, 
            inspected_param: str, 
            inspected_values: list | np.ndarray, 
            measured_variables: list[str], 
            start_time: float = 1000.0,
            end_time: float = np.inf,
            plot: bool = False,
            project_path: str | None = None,
            ) -> Dict[str, BaseInspectionResults]:

        spont_vars = [v for v in measured_variables if v in SpontInspectionResults.DEFAULT_UNITS]
        dynamic_vars = [v for v in measured_variables if v in DynamicStimulusInspectionResults.DEFAULT_UNITS]
        
        unknown_vars = set(measured_variables) - set(spont_vars) - set(dynamic_vars)
        if unknown_vars:
            raise ValueError(f"Unknown variables requested: {unknown_vars}")

        mf_names = list(self.base_sim_params.mf_models.keys())
        network_names = ["SNN"] + mf_names

        extractors = {}
        results_containers = {}

        if spont_vars:
            extractors["spont"] = SpontActivityExtractor(
                measured_variables=spont_vars,
                start_time=start_time, 
                end_time=end_time,
            )
            
            results_containers["spont"] = SpontInspectionResults(
                inspected_param=inspected_param, 
                inspected_values=inspected_values,
                network_names=network_names, 
                measured_variables=spont_vars,
                network_params=self.base_network_params,
                stimulus_params=self.base_stimulus_params,
            )
            
        if dynamic_vars:
            extractors["dynamic"] = ComparisonExtractor(
                measured_variables=dynamic_vars,
                start_time=start_time, 
                end_time=end_time, 
            )

            results_containers["dynamic"] = DynamicStimulusInspectionResults(
                inspected_param=inspected_param, 
                inspected_values=inspected_values,
                network_names=mf_names, 
                measured_variables=dynamic_vars,
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
            
            if is_network:
                current_network_params = inject_pydantic_param(current_network_params, inspected_param_path, value)
            else:
                current_stimulus_params = inject_pydantic_param(current_stimulus_params, inspected_param_path, value)
                
            current_stimulus_config = {"InspectionStimulus" : current_stimulus_params}


            print("Running SNN Simulation...")
            snn_results = run_snn_simulation_workflow(
                self.base_sim_params.snn_simulation, 
                current_network_params, 
                current_stimulus_config
            )
            snn_data = snn_results["InspectionStimulus"]
            
            if spont_vars:
                extracted_spont_data = [extractors["spont"].extract(snn_data)]
            if dynamic_vars:
                exctracted_dyn_data =  []

            mf_results_dict = {}
            for mf_model_name, mf_sim_params in self.base_sim_params.mf_models.items():
                print(f"Running MF Simulation: {mf_model_name}...")
                mf_results = run_mf_simulation_workflow(mf_sim_params, current_network_params, current_stimulus_config)
                mf_data = mf_results["InspectionStimulus"]
                mf_results_dict[mf_model_name] = mf_data

                if spont_vars:
                    extracted_spont_data.append(extractors["spont"].extract(mf_data))
                if dynamic_vars:
                    exctracted_dyn_data.append(extractors["dynamic"].extract(ground_truth=snn_data, target=mf_data))

            if spont_vars:
                results_containers["spont"].add_inspection_data(extracted_spont_data)
            if dynamic_vars:
                results_containers["dynamic"].add_inspection_data(exctracted_dyn_data)


            if plot:
                fig_plots.fig_full_network_overview_together(
                    snn_data, 
                    list(mf_results_dict.values()),
                    common_params={
                        'xmargin': 0.0,
                        'ymargin': 0.0,
                        'labels': network_names,
                        'legend': {'loc': 'upper left'},
                        'xlim' : (0, snn_data.times()[-1])
                    },
                    fig_params={
                        'figsize': (20, 10),  # width, height
                        'tight_layout': True,
                        'savefig': True,
                        'savefig_path': project_path /f"{inspected_param}_{value}_Full_network_overview_together.png",
                        'title' : f"Network overview for: '{inspected_param}={value}'",
                    }
                )

            # NOTE: if memory issues arise, consider using `del` and `gc.collect()`
            # since Python may store references to large objects in memory even after they go out of scope.

            # del snn_results
            # del mf_results 
            # gc.collect() 

        print("\nInspection Complete. Freezing results...")
        for container in results_containers.values():
            container.freeze()
            
        return results_containers



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
