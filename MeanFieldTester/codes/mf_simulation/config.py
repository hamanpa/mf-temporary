from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict, AliasChoices
from typing import Dict, Any, Optional, Annotated, List, Literal
from enum import Enum
from pathlib import Path
from ..transfer_function.config import TransferFunctionConfig

class SimulatorType(str, Enum):
    TVB = "tvb"


class ModelType(str, Enum):
    ZERLAUT2018_FO = "zerlaut2018.first_order"
    ZERLAUT2018_SO = "zerlaut2018.second_order"

    DIVOLO2019_FO = "divolo2019.first_order"
    DIVOLO2019_SO = "divolo2019.second_order"

    STP_ASYPTOTIC_FO = "stp_asymptotic.first_order"
    STP_ASYPTOTIC_SO = "stp_asymptotic.second_order"

    STP_DYNAMIC_FO = "stp_dynamic.first_order"
    STP_DYNAMIC_SO = "stp_dynamic.second_order"

    CUSTOMNEUROPSI = "custom.neuropsi"

class BaseInitValues(BaseModel):
    """Base state variables common to all second-order MF models."""
    
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    
    exc_rate_mean: List[float] = Field(
        validation_alias=AliasChoices('E', 'exc_rate_mean', 'exc_rate'),
        serialization_alias='exc_rate_mean',
        description="Initial mean firing rate of the excitatory population in [Hz]."
    )
    
    inh_rate_mean: List[float] = Field(
        validation_alias=AliasChoices('I', 'inh_rate_mean', 'inh_rate'),
        serialization_alias='inh_rate_mean',
        description="Initial mean firing rate of the inhibitory population in [Hz]."
    )

    exc_rate_var: List[float] = Field(
        validation_alias=AliasChoices('C_ee', 'exc_rate_var'),
        serialization_alias='exc_rate_var',
        description="Initial variance of the excitatory population firing rate in [Hz^2]."
    )

    inh_rate_var: List[float] = Field(
        validation_alias=AliasChoices('C_ii', 'inh_rate_var'),
        serialization_alias='inh_rate_var',
        description="Initial variance of the inhibitory population firing rate in [Hz^2]."
    )

    rate_cov: List[float] = Field(
        validation_alias=AliasChoices('C_ei', 'rate_cov'),
        serialization_alias='rate_cov',
        description="Initial covariance between excitatory and inhibitory population firing rates in [Hz^2]."
    )

    noise_rate: List[float] | None = Field(
        validation_alias=AliasChoices('noise', 'noise_rate'),
        serialization_alias='noise_rate',
        description="Initial noise level in the mean-field model in [Hz]."
    )

    stim_rate_mean: List[float] | None = Field(
        validation_alias=AliasChoices('stimulus', 'stim_rate_mean'),
        serialization_alias='stim_rate_mean',
        description="Initial external stimulus level in the mean-field model in [Hz]."
    )

class Zerlaut2018InitialValuesConfig(BaseInitValues):
    pass

class Divolo2019InitialValuesConfig(Zerlaut2018InitialValuesConfig):
    exc_adaptation_mean: List[float] = Field(
        validation_alias=AliasChoices('W_e', 'exc_adaptation_mean', 'adaptation_mean'),
        serialization_alias='exc_adaptation_mean',
        description="Initial mean adaptation current for the excitatory population in [nA]."
    )
    inh_adaptation_mean: List[float] = Field(
        validation_alias=AliasChoices('W_i', 'inh_adaptation_mean'),
        serialization_alias='inh_adaptation_mean',
        description="Initial mean adaptation current for the inhibitory population in [nA]."
    )

_STP_INIT_DESCRIPTIONS = {
    "X": "available resources x",
    "Y": "active resources y",
    "U_dyn": "facilitation variable U_dyn (efficacy u = U_dyn*(1-U) + U)",
}


# Older population-based init names refer to the SOURCE population (e.g. X_e = synapses from E)
# and are expanded to both projections from that source (X_e -> X_ee and X_ie).
_STP_SOURCE_BASED_INIT_NAMES = {
    "X_e": ("X", "e"), "exc_stp_x_mean": ("X", "e"),
    "Y_e": ("Y", "e"), "exc_stp_y_mean": ("Y", "e"),
    "U_e": ("U_dyn", "e"), "U_dyn_e": ("U_dyn", "e"), "exc_stp_u_mean": ("U_dyn", "e"),
    "X_i": ("X", "i"), "inh_stp_x_mean": ("X", "i"),
    "Y_i": ("Y", "i"), "inh_stp_y_mean": ("Y", "i"),
    "U_i": ("U_dyn", "i"), "U_dyn_i": ("U_dyn", "i"), "inh_stp_u_mean": ("U_dyn", "i"),
}


def _stp_init_field(state: str, projection: str):
    """Initial value field of a dynamic-STP state variable; accepts the TVB name (e.g. 'X_ei') or the field name."""
    field_name = f"{projection}_{state.lower()}_mean"
    return Field(
        validation_alias=AliasChoices(f"{state}_{projection}", field_name),
        serialization_alias=field_name,
        description=f"Initial mean of the STP {_STP_INIT_DESCRIPTIONS[state]} of projection '{projection}' (target-source).",
    )


class CustomNeuroPSIInitialValuesConfig(Divolo2019InitialValuesConfig):
    """
    Initial values for the dynamic STP models: one X, Y, U_dyn per projection (target-source code,
    'ei' = onto E from I), matching the TVB state variables X_ee, Y_ee, U_dyn_ee, ...
    """
    ee_x_mean: List[float] = _stp_init_field("X", "ee")
    ee_y_mean: List[float] = _stp_init_field("Y", "ee")
    ee_u_dyn_mean: List[float] = _stp_init_field("U_dyn", "ee")
    ei_x_mean: List[float] = _stp_init_field("X", "ei")
    ei_y_mean: List[float] = _stp_init_field("Y", "ei")
    ei_u_dyn_mean: List[float] = _stp_init_field("U_dyn", "ei")
    ie_x_mean: List[float] = _stp_init_field("X", "ie")
    ie_y_mean: List[float] = _stp_init_field("Y", "ie")
    ie_u_dyn_mean: List[float] = _stp_init_field("U_dyn", "ie")
    ii_x_mean: List[float] = _stp_init_field("X", "ii")
    ii_y_mean: List[float] = _stp_init_field("Y", "ii")
    ii_u_dyn_mean: List[float] = _stp_init_field("U_dyn", "ii")

    @model_validator(mode="before")
    @classmethod
    def _expand_source_based_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for old_name, (state, source) in _STP_SOURCE_BASED_INIT_NAMES.items():
            if old_name not in data:
                continue
            value = data.pop(old_name)
            for target in ("e", "i"):
                projection = f"{target}{source}"
                names = (f"{state}_{projection}", f"{projection}_{state.lower()}_mean")
                if not any(name in data for name in names):
                    data[names[0]] = value
        return data




ModelTypeInitialValuesConfig = Zerlaut2018InitialValuesConfig | Divolo2019InitialValuesConfig | CustomNeuroPSIInitialValuesConfig


class LoadSimulationConfig(BaseModel):
    execution_mode: Literal["load"] 

    # TODO: update the loading!

class RunSimulationConfig(BaseModel):
    execution_mode: Literal["run"]  

    simulator: SimulatorType
    model: ModelType    
    
    time_step: float = Field(..., gt=0.0, description="Integration time step in [ms].")
    resolution_time: float = Field(..., gt=0.0, description="Time scale for Mean-Field to be Markovian [ms].")
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    init_values: ModelTypeInitialValuesConfig
    # init_values: Dict[str, List[float]] = Field(
    #     default_factory=dict,
    #     description="Initial conditions for the state variables of the mean-field model. Keys should be population names."
    # )


    transfer_function: TransferFunctionConfig 

    # @model_validator(mode='after')
    # def validate_init_values(self):
    #     pass


    @model_validator(mode='before')
    @classmethod
    def validate_and_cast_init_values(cls, data: Any) -> Any:
        """
        Dynamically casts the 'mf_init' dictionary into the correct Schema based
        on the selected 'model'.
        """

        if not isinstance(data, dict):
            return data
            
        model_type = data.get("model")
        init_data = data.get("init_values", {})
        
        if not isinstance(init_data, dict):
            return data

        match model_type:
            case ModelType.ZERLAUT2018_FO | ModelType.ZERLAUT2018_SO:
                data["init_values"] = Zerlaut2018InitialValuesConfig(**init_data)
            case ModelType.DIVOLO2019_FO | ModelType.DIVOLO2019_SO | ModelType.STP_ASYPTOTIC_FO | ModelType.STP_ASYPTOTIC_SO:
                data["init_values"] = Divolo2019InitialValuesConfig(**init_data)
            case ModelType.CUSTOMNEUROPSI | ModelType.STP_DYNAMIC_FO | ModelType.STP_DYNAMIC_SO:
                data["init_values"] = CustomNeuroPSIInitialValuesConfig(**init_data)
            case _:
                raise ValueError(f"Unknown model type: {model_type}")

        return data




class SkipSimulationConfig(BaseModel):
    execution_mode: Literal["skip"]



MeanFieldSimulationConfig = Annotated[
    RunSimulationConfig | LoadSimulationConfig | SkipSimulationConfig, 
    Field(discriminator='execution_mode')
]
