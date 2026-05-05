import yaml
from enum import Enum
from typing import Dict, Literal, Union, Any, Annotated
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, RootModel
from numpy import pi

class StimulusPatternType(str, Enum):
    PULSE_TRAIN = "PulseTrain"
    SINUSOIDAL = "Sinusoidal"
    TWO_SIDED_GAUSSIAN = "TwoSidedGaussian"
    NO_STIMULUS = "NoStimulus"


class SinusoidalParams(BaseModel):
    """
    stim_rate = offset + magnitude * sin(2 * pi * freq * (t - stim_start))
    """

    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., description="Start time of the stimulus in [ms].")
    end: float = Field(..., description="End time of the stimulus in [ms].")
    magnitude: float = Field(..., description="Magnitude of the stimulus in [Hz].")
    offset: float = Field(..., description="Offset of the stimulus in [Hz].")

    freq: float = Field(..., description="Frequency of the sinusoidal stimulus in [Hz].")
    phase: float = Field(0.0, description="Phase of the sinusoidal stimulus in radians.", ge=0.0, lt=2*pi)

class PulseTrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., description="Start time of the stimulus in [ms].")
    end: float = Field(..., description="End time of the stimulus in [ms].")
    magnitude: float = Field(..., description="Magnitude of the stimulus in [Hz].")
    offset: float = Field(..., description="Offset of the stimulus in [Hz].")

    pulse_duration: float = Field(..., description="Duration of the pulse train in [ms].")
    pulse_period: float = Field(..., description="Period of the pulse train in [ms].")

class TwoSidedGaussianParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., description="Start time of the stimulus in [ms].")
    end: float = Field(..., description="End time of the stimulus in [ms].")
    magnitude: float = Field(..., description="Magnitude of the stimulus in [Hz].")
    offset: float = Field(..., description="Offset of the stimulus in [Hz].")

    center: float = Field(..., description="Center of the Gaussian in [ms].")
    sigma_left: float = Field(..., description="Width of the left side of the Gaussian in [ms].")
    sigma_right: float = Field(..., description="Width of the right side of the Gaussian in [ms].")


class BaseStimulusConfig(BaseModel):
    """Base class holding shared parameters for all stimuli."""
    model_config = ConfigDict(extra="forbid") # Will strictly throw errors if you type `stim_pars` instead of `stim_params`
    
    drive_rate: float = Field(..., description="Base drive rate in [Hz].")
    drive_increase_duration: float = Field(..., description="Duration of the drive increase in [ms].")
    stim_target_ratio: float = Field(..., description="Ratio of stimulated nodes to total nodes.")
    simulation_duration: float = Field(..., gt=0, description="Total simulation duration in [ms].")
    target_nodes: int = Field(..., description="Number of target nodes for stimulation.")
    direct_stimulation: bool = Field(..., description="Whether to directly stimulate nodes.")


class NoStimulusConfig(BaseStimulusConfig):
    pattern: Literal[StimulusPatternType.NO_STIMULUS]
    stim_params: Dict[str, Any] = Field(default_factory=dict)

class SinusoidalConfig(BaseStimulusConfig):
    pattern: Literal[StimulusPatternType.SINUSOIDAL]
    stim_params: SinusoidalParams

class PulseTrainConfig(BaseStimulusConfig):
    pattern: Literal[StimulusPatternType.PULSE_TRAIN]
    stim_params: PulseTrainParams

class TwoSidedGaussianConfig(BaseStimulusConfig):
    pattern: Literal[StimulusPatternType.TWO_SIDED_GAUSSIAN]
    stim_params: TwoSidedGaussianParams

StimulusConfig = Annotated[
    (
        NoStimulusConfig | 
        SinusoidalConfig |
        PulseTrainConfig |
        TwoSidedGaussianConfig
    ),
    Field(discriminator='pattern')
]

class StimuliCollection(RootModel):
    """
    Represents the entire YAML file, which is a dictionary where 
    keys are custom names (e.g., 'SpontActivity') and values are configurations.
    """
    root: Dict[str, StimulusConfig]

