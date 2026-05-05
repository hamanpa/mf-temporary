# MeanFieldTester/codes/stimuli/__init__.py

from .config import BaseStimulusConfig, StimulusPatternType
from .models import (
    BaseRateProfile,
    NoStimulusRateProfile, 
    PulseTrainRateProfile, 
    SinusoidalRateProfile,
    TwoSidedGaussianRateProfile
)

# The Registry maps the config string to the actual Python class
PROFILE_REGISTRY = {
    StimulusPatternType.NO_STIMULUS: NoStimulusRateProfile,
    StimulusPatternType.SINUSOIDAL: SinusoidalRateProfile,
    StimulusPatternType.PULSE_TRAIN: PulseTrainRateProfile,
    StimulusPatternType.TWO_SIDED_GAUSSIAN: TwoSidedGaussianRateProfile,
}

def create_rate_profile(stim_config: BaseStimulusConfig) -> BaseRateProfile:
    """
    Factory function: Takes a validated configuration and returns 
    the instantiated mathematical rate profile model.
    """
    if stim_config.pattern not in PROFILE_REGISTRY:
        raise ValueError(f"Stimulus pattern '{stim_config.pattern}' is not registered.")
    
    profile_class = PROFILE_REGISTRY[stim_config.pattern]
    return profile_class(stim_config)