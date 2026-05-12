from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict, AliasChoices
from typing import Dict, Any, Optional, Annotated, List, Literal
from enum import Enum
from pathlib import Path
from ..transfer_function.config import TransferFunctionConfig

class SimulatorType(str, Enum):
    TVB = "tvb"


class ModelType(str, Enum):
    ZERLAUT2018 = "zerlaut2018"
    DIVOLO2019_SO = "divolo2019.second_order"
    DIVOLO2019_FO = "divolo2019.first_order"
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

class CustomNeuroPSIInitialValuesConfig(Divolo2019InitialValuesConfig):
    exc_stp_x_mean: List[float] = Field(
        validation_alias=AliasChoices('X_e', 'exc_stp_x_mean'),
        serialization_alias='exc_stp_x_mean',
        description="Initial mean of the STP variable X for the excitatory population."
    )
    exc_stp_y_mean: List[float] = Field(
        validation_alias=AliasChoices('Y_e', 'exc_stp_y_mean'),
        serialization_alias='exc_stp_y_mean',
        description="Initial mean of the STP variable Y for the excitatory population."
    )
    inh_stp_x_mean: List[float] = Field(
        validation_alias=AliasChoices('X_i', 'inh_stp_x_mean'),
        serialization_alias='inh_stp_x_mean',
        description="Initial mean of the STP variable X for the inhibitory population."
    )
    inh_stp_y_mean: List[float] = Field(
        validation_alias=AliasChoices('Y_i', 'inh_stp_y_mean'),
        serialization_alias='inh_stp_y_mean',
        description="Initial mean of the STP variable Y for the inhibitory population."
    )




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
            case ModelType.ZERLAUT2018:
                data["init_values"] = Zerlaut2018InitialValuesConfig(**init_data)
            case ModelType.DIVOLO2019_FO | ModelType.DIVOLO2019_SO:
                data["init_values"] = Divolo2019InitialValuesConfig(**init_data)
            case ModelType.CUSTOMNEUROPSI:
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
