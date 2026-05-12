from ...stimuli.config import BaseStimulusConfig
from ...stimuli.config import StimulusPatternType
from ...network_params.translators import TranslationRule, translate_params
from pydantic import BaseModel


from tvb.datatypes.equations import FiniteSupportEquation
from tvb.basic.neotraits.api import Attr, Final

class TwoSidedGaussianRateProfile(FiniteSupportEquation):
    """Asymmetric Gaussian equation.
    offset: parameter to extend the behaviour of this function
    when spatializing model parameters.
    """

    STIMULUS_PARAMETERS_MAPPING = {
        "start": TranslationRule("start", sim_unit="ms"),
        "end": TranslationRule("end", sim_unit="ms"),
        "magnitude": TranslationRule("magnitude", sim_unit="nA"),
        "sigma_left": TranslationRule("sigma_left", sim_unit="ms"),
        "sigma_right": TranslationRule("sigma_right", sim_unit="ms"),
        "center": TranslationRule("center", sim_unit="ms"),
        "offset": TranslationRule("offset", sim_unit="nA"),
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Gaussian Equation",
        default=(
            "where((var-center) < 0, magnitude * exp(-((var-center)**2 / (2.0 * sigma_left**2)))+offset, magnitude * exp(-((var-center)**2 / (2.0 * sigma_right**2)))+offset)"
            + " * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
            + " * where((var>=start) & (var<end), 1.0, 0.0)"
        ),
        doc=""":math:`(magnitude \\exp\\left(-\\left(\\left(x-center\\right)^2 /
        \\left(2.0 \\sigma_left^2\\right)\\right)\\right))\\Theta(center-x) + /
         magnitude \\exp\\left(-\\left(\\left(x-center\\right)^2 /
        \\left(2.0 \\sigma_right^2\\right)\\right)\\right))\\Theta(x-center) offset`"""
    )

    parameters = Attr(
        field_type=dict,
        label="Asymmetric Gaussian Parameters",
        default=lambda: {
            "start": 1000.0,
            "end": 2000.0,
            "magnitude": 1.0, 
            "sigma_left": 1.0, 
            "sigma_right": 2.0, 
            "center": 0.0, 
            "offset": 0.0,
            "initial_increase_duration": 400.0,
        })


class SinusoidalRateProfile(FiniteSupportEquation):
    
    STIMULUS_PARAMETERS_MAPPING = {
        "start": TranslationRule("start", sim_unit="ms"),
        "end": TranslationRule("end", sim_unit="ms"),
        "magnitude": TranslationRule("magnitude", sim_unit="nA"),
        "freq": TranslationRule("freq", sim_unit="kHz"),
        "offset": TranslationRule("offset", sim_unit="nA"),
        "phase": TranslationRule("phase", sim_unit="radians"),
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Custom sinusoid equation",
        default=(
            "where((var>=start) & (var<end), magnitude * sin(6.283185307179586 * freq * (var-start)+phase)+offset, offset)"
            + " * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
            + " * where((var>=start) & (var<end), 1.0, 0.0)"
        ),
        doc=""":math:`magnitude \\sin(2.0 \\pi freq x+phase) + offset`""")

    parameters = Attr(
        field_type=dict,
        label="Custom Sinusoid Parameters",
        default=lambda: {
            "start": 2.0, 
            "end": 0.0, 
            "magnitude": 1.0, 
            "freq": 1.0, 
            "offset": 0.0, 
            "phase": 0.0,
            "initial_increase_duration": 100.0,
        })


class NoStimulusRateProfile(FiniteSupportEquation):
    equation = Final(label="Custom drive equation",
        default=(
            "0.0"
        ),
        doc=""":math:`magnitude \\sin(2.0 \\pi freq x+phase) + offset`""")

    parameters = Attr(
        field_type=dict,
        label="Custom drive Parameters",
        default=lambda: {
            "initial_increase_duration": 100.0,
        })


class PulseTrainRateProfile(FiniteSupportEquation):

    STIMULUS_PARAMETERS_MAPPING = {
        "start": TranslationRule("start", sim_unit="ms"),
        "end": TranslationRule("end", sim_unit="ms"),
        "magnitude": TranslationRule("magnitude", sim_unit="nA"),
        "offset": TranslationRule("offset", sim_unit="nA"),
        "pulse_duration": TranslationRule("pulse_duration", sim_unit="ms"),
        "pulse_period": TranslationRule("pulse_period", sim_unit="ms"),
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Custom Pulse Train Equation",
        default=(
            "where((var >= start) & (var-start) % pulse_period < pulse_duration, magnitude + offset, offset)"
            + " * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
            + " * where((var>=start) & (var<end), 1.0, 0.0)"
            + " + drive_rate * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
        ),
        doc=""":math:`(magnitude \\exp\\left(-\\left(\\left(x-center\\right)^2 /
        \\left(2.0 \\sigma_left^2\\right)\\right)\\right))\\Theta(center-x) + /
         magnitude \\exp\\left(-\\left(\\left(x-center\\right)^2 /
        \\left(2.0 \\sigma_right^2\\right)\\right)\\right))\\Theta(x-center) offset`"""
    )

    parameters = Attr(
        field_type=dict,
        label="Custom Sinusoid Parameters",
        default=lambda: {
            "start" : 1000,
            "end": 2000,
            "magnitude" : 1,
            "offset": 0,
            "pulse_duration" : 500,
            "pulse_period" : 1000,
            "initial_increase_duration": 100.0,
            "drive_rate": 0.0,
        })


PROFILE_REGISTRY = {
    StimulusPatternType.NO_STIMULUS: NoStimulusRateProfile,
    StimulusPatternType.SINUSOIDAL: SinusoidalRateProfile,
    StimulusPatternType.PULSE_TRAIN: PulseTrainRateProfile,
    StimulusPatternType.TWO_SIDED_GAUSSIAN: TwoSidedGaussianRateProfile,
}


def prepare_stimulus(stim_params: BaseStimulusConfig):
    if stim_params.pattern not in PROFILE_REGISTRY:
        raise ValueError(f"Stimulus pattern '{stim_params.pattern}' is not registered.")
    
    profile_class = PROFILE_REGISTRY[stim_params.pattern]


    if issubclass(profile_class, FiniteSupportEquation):
        stim_model = profile_class()
        
        if isinstance(stim_params.stim_params, BaseModel):
            stim_params_dict = stim_params.stim_params.model_dump()
        elif isinstance(stim_params.stim_params, dict):
            stim_params_dict = stim_params.stim_params
        else:
            raise ValueError(f"stim_params should be either a Pydantic model or a dict, got {type(stim_params.stim_params)}")

        stim_model.parameters.update({
            **stim_params_dict,
            "initial_increase_duration": stim_params.initial_increase_duration,
        })
    else:
        raise NotImplementedError(f"Stimulus pattern '{stim_params.pattern}' is not implemented yet. Please implement the stimulus preparation for this pattern in setup_stimulus().")
    
    return stim_model