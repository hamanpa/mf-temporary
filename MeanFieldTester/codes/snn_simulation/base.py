from abc import ABC, abstractmethod
from typing import Any

from ..data_structures.network import SNNResults
from ..network_params.models import BiologicalParameters
from .config import NetworkSimulationConfig
from ..stimuli.config import BaseStimulusConfig

class BaseSNNSimulator(ABC):
    """
    Abstract base class for spiking neural network simulators.
    
    Any new simulator (e.g., PyNN, Brian2, NEST) must inherit from this class 
    and implement the core lifecycle methods.
    """

    @abstractmethod
    def build_network(self, network_params: BiologicalParameters, snn_sim_params: NetworkSimulationConfig) -> None:
        """
        Builds the network structure (populations, projections) based on the parameters.
        This should only be called once per simulation workflow.
        
        Implementation note: 
        Subclasses should store these parameters as instance attributes 
        (e.g., `self.network_params = network_params`) so they can be accessed 
        during the run_stimulus phase.
        """
        pass

    @abstractmethod
    def run_stimulus(self, stim_params: BaseStimulusConfig) -> SNNResults:
        """
        Executes the SNN simulation for a single stimulus.

        Parameters
        ----------
        stim_params : BaseStimulusConfig
            The configuration for the stimulus to be applied.

        Returns
        -------
        SNNResults
            A standardized data structure containing the results (spikes, voltages, etc.) 
            for the entire network during this stimulus.
        """
        pass

    # @abstractmethod
    # def reset(self) -> None:
    #     """
    #     Resets the simulator state (time, membrane potentials, adaptation) 
    #     to prepare for the next stimulus without rebuilding the network.
    #     """
    #     pass

    @abstractmethod
    def end(self) -> None:
        """Cleans up the simulator environment (e.g., closing PyNN) after all simulations are done."""
        pass