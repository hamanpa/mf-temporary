from typing import Protocol, Dict, List
from ..data_structures.base import BaseResults, BaseMFResults, BaseSNNResults, BaseInspectionResults
from ..transfer_function.base import BaseTransferFunction


class BasicWorkflowHook(Protocol):
    """
    Protocol for a basic workflow hook that can be called with step parameters and results.
    """

    def __call__(
            self, 
            neuron_results: BaseResults,
            tf_results: Dict[str, Dict[str, BaseTransferFunction]],
            snn_results: BaseSNNResults,
            mf_results: List[BaseMFResults],
            ) -> None:

        pass


class InspectionWorkflowHook(Protocol):
    """
    Protocol for a basic workflow hook that can be called with step parameters and results.
    """

    def __call__(
            self,
            identifier: str,
            inspection_results: Dict[str, BaseInspectionResults],
            ) -> None:
        
        pass