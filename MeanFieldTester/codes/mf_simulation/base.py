from abc import ABC, abstractmethod
from typing import Any

# TODO: We will need to create MFResults in the data_structures module
# from ..data_structures.meanfield import MFResults 
from ..network_params.models import BiologicalParameters
from .config import MeanFieldSimulationConfig
from ..stimuli.config import BaseStimulusConfig

class BaseMFSimulator(ABC):
    """
    Abstract base class for Mean-Field simulators.
    
    Any new MF simulator (e.g., TVB, custom SciPy solver, PyTorch) must inherit 
    from this class and implement these core lifecycle methods.
    """

    @abstractmethod
    def build_network(self, network_params: BiologicalParameters, mf_sim_params: MeanFieldSimulationConfig) -> None:
        """
        Builds the mean-field mathematical model and integration engine.
        This should only be called once per simulation workflow.
        
        Implementation note: 
        Subclasses should store these parameters as instance attributes 
        so they can be accessed during the run_stimulus phase.
        """
        pass

    @abstractmethod
    def run_stimulus(self, stim_params: BaseStimulusConfig) -> Any: # Replace Any with MFResults
        """
        Executes the Mean-Field simulation for a single stimulus.

        Parameters
        ----------
        stim_params : BaseStimulusConfig
            The configuration for the stimulus to be applied.

        Returns
        -------
        MFResults
            A standardized data structure containing the results (time, rates, 
            covariances, adaptation) for the entire network during this stimulus.
        """
        pass

    @abstractmethod
    def end(self) -> None:
        """
        Cleans up the simulator environment after the simulation is done.
        Critical for freeing memory (especially if integrating with C/C++ backends).
        """
        pass