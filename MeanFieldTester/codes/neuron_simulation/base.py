from abc import ABC, abstractmethod
from typing import Any
from codes.data_structures.single_neuron import SingleNeuronResults

class BaseNeuronSimulator(ABC):
    """
    Abstract base class for single neuron simulators.
    Any new simulator (e.g., Brian2, NEST) must inherit from this class 
    and implement the `simulate` method.
    """

    @abstractmethod
    def simulate(self, neurons: dict[str, Any], sim_pars: dict[str, Any]) -> dict[str, SingleNeuronResults]:
        """
        Executes the neuron simulation.

        Parameters
        ----------
        neurons : dict
            Dictionary with neuron parameters.
        sim_pars : dict
            Dictionary with simulation parameters (e.g., simulation_time, dt, ranges).

        Returns
        -------
        Dict[str, SingleNeuronResults]
            A dictionary where keys are neuron names and values are the standardized 
            SingleNeuronResults objects.
        """
        pass