from typing import Protocol, Dict, List
from ..data_structures.base import BaseSingleNeuronResults, BaseMFResults, BaseSNNResults, BaseInspectionResults
from ..transfer_function.base import BaseTransferFunction


class BasicWorkflowHook(Protocol):
    """
    Protocol for a basic workflow hook that can be called with step parameters and results.
    """

    def __call__(
            self, 
            identifier: str,
            neuron_results: Dict[str, BaseSingleNeuronResults], 
            tf_funcs_results: Dict[str, List[BaseTransferFunction]],
            snn_results: BaseSNNResults,
            network_results_list: List[BaseSNNResults | BaseMFResults],
            ) -> None:

        pass


class InspectionWorkflowHook(Protocol):
    """
    Protocol for a basic workflow hook that can be called with step parameters and results.
    """

    def __call__(
            self,
            identifier: str,
            inspection_results_list: List[BaseInspectionResults],
            ) -> None:
        
        pass