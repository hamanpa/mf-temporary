from pathlib import Path
import pickle
from typing import Dict

from .base import BaseTransferFunction
from .config import TransferFunctionConfig

from ..network_params.models import BiologicalParameters
from ..data_structures.single_neuron import SingleNeuronResults

from .zerlaut2018_tf import Zerlaut2018TF
from .divolo2019_tf import DiVolo2019TF
from .neuropsi_tf import NeuroPSICustomTF


TF_REGISTRY = {
    "zerlaut2018": Zerlaut2018TF,
    "divolo2019": DiVolo2019TF,
    "neuropsi.custom": NeuroPSICustomTF,
}

def get_transfer_function(
    tf_method_name: str, 
    neuron_name: str, 
    network_params: BiologicalParameters, 
    tf_params: TransferFunctionConfig
) -> BaseTransferFunction:
    """Factory function to instantiate the correct Transfer Function class."""
    
    if tf_method_name not in TF_REGISTRY:
        raise ValueError(f"Transfer Function '{tf_method_name}' not found. Available: {list(TF_REGISTRY.keys())}")
    
    return TF_REGISTRY[tf_method_name](neuron_name, network_params, tf_params)


def run_tf_fitting_workflow(
    tf_params: TransferFunctionConfig, 
    network_params: BiologicalParameters, 
    neuron_results: Dict[str, SingleNeuronResults] = None
) -> Dict[str, BaseTransferFunction]:
    """High-level orchestrator for Transfer Function fitting and loading."""
    
    fitted_tfs = {}
    
    for neuron_name in network_params.internal_neurons:
        print(f"\n{'='*40}\nPreparing Transfer Function for {neuron_name}\n{'='*40}")
        
        tf_instance = get_transfer_function(
            tf_method_name=tf_params.tf_model.model_name,
            neuron_name=neuron_name,
            network_params=network_params,
            tf_params=tf_params
        )
        
        if not tf_params.fit_transfer_function:  # Load fit
            if neuron_name not in tf_params.tf_fits:
                raise ValueError(f"Missing 'tf_fits' data for {neuron_name}!")
            
            loaded_params = tf_params.tf_fits[neuron_name].model_dump()
            tf_instance.set_fitted_parameters(loaded_params)
            
            print(f"Successfully loaded pre-calculated coefficients for {neuron_name}.")
            
        else:  # make fit
            if neuron_name not in neuron_results:
                raise ValueError(f"Cannot fit TF for {neuron_name}: Missing SNN simulation results!")
                
            print(f"Fitting {tf_params.tf_model.model_name}...")
            
            metrics = tf_instance.fit(single_neuron_results=neuron_results[neuron_name])
            print(f"Fit complete. Metrics: {metrics}")
            
        fitted_tfs[neuron_name] = tf_instance
        
    return fitted_tfs