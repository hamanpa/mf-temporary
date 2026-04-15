"""
This part defines the abstract base class for all Transfer Functions. 

Design choices:
- The `fit` method can take optional `simulation_results` for TFs that need fitting
- We go with class so that we can store parameters of the neuron
- We add a `__call__` method to make it easy to use the TF as a functor


"""



from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

from ..data_structures.single_neuron import SingleNeuronResults
from ..transfer_function.config import TransferFunctionConfig
from ..network_params.models import BiologicalParameters

class BaseTransferFunction(ABC):
    """
    Abstract base class for all Mean-Field Transfer Functions.
    Implements the Functor pattern to store state (fitted parameters)
    while remaining callable like a standard math function.
    """
    
    def __init__(
        self, 
        neuron_name: str, 
        network_params: BiologicalParameters,
        tf_params: TransferFunctionConfig
    ):
        self.neuron_name = neuron_name
        self.network_params = network_params
        self.tf_params = tf_params

        # State tracking
        self.is_fitted: bool = False
        self.fitted_params: Dict[str, float] = {}

    @abstractmethod
    def required_inputs(self) -> List[str]:
        """
        CLASS METHOD: Tells the workflow exactly what this TF needs to evaluate.
        Example: return ["exc_rate", "inh_rate", "adaptation"]
        """
        pass

    @abstractmethod
    def fit(self, single_neuron_results: SingleNeuronResults, **kwargs) -> dict:
        """
        Calibrates the model based on single neuron grid data.
        Must set `self.is_fitted = True` and populate `self.fitted_params`.
        
        Returns:
            dict: Metrics assessing the fit quality (e.g., R-squared, MSE).
        """
        pass

    def set_fitted_parameters(self, params: Dict[str, float]) -> None:
        """
        Injects pre-calculated parameters directly from the LoadTFFittingConfig, 
        completely bypassing the optimization phase.
        """

        self.fitted_params = params
        self.is_fitted = True

    @abstractmethod
    def evaluate(self, **kwargs) -> np.ndarray:
        """
        The core mapping function: F(v_e, v_i, ...) -> v_out.
        MUST be vectorized using numpy for fast ODE execution.
        """
        pass

    def __call__(self, **kwargs) -> np.ndarray:
        """
        The callable interface. 
        Adds a safety validation layer before running the fast numpy math.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Transfer Function for '{self.neuron_name}' "
                f"has no parameters! You must call fit() or set_fitted_parameters() first."
            )
        
        # Pre-flight Validation: Did the ODE solver pass everything we declared we need?
        missing = [req for req in self.required_inputs() if req not in kwargs]
        if missing:
            raise ValueError(
                f"[{self.__class__.__name__}] Missing required inputs for evaluation: {missing}. "
                f"Provided kwargs: {list(kwargs.keys())}"
            )
            
        return self.evaluate(**kwargs)