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
        "magnitude": TranslationRule("magnitude", sim_unit="kHz"),
        "sigma_left": TranslationRule("sigma_left", sim_unit="ms"),
        "sigma_right": TranslationRule("sigma_right", sim_unit="ms"),
        "center": TranslationRule("center", sim_unit="ms"),
        "offset": TranslationRule("offset", sim_unit="kHz"),
    }

    ADDITIONAL_PARAMETERS_MAPPING = {
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Gaussian Equation",
        default=(
            "where((var-center) < 0, magnitude * exp(-((var-center)**2 / (2.0 * sigma_left**2))), magnitude * exp(-((var-center)**2 / (2.0 * sigma_right**2))))"
            + " * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
            + " * where((var>=start) & (var<end), 1.0, 0.0)"
            + " + offset * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
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
        "magnitude": TranslationRule("magnitude", sim_unit="kHz"),
        "freq": TranslationRule("freq", sim_unit="kHz"),
        "offset": TranslationRule("offset", sim_unit="kHz"),
        "phase": TranslationRule("phase", sim_unit="rad"),
    }

    ADDITIONAL_PARAMETERS_MAPPING = {
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Custom sinusoid equation",
        default=(
            "where((var>=start) & (var<end), magnitude * sin(6.283185307179586 * freq * (var-start)+phase), 0.0)"
            + " * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
            + " * where((var>=start) & (var<end), 1.0, 0.0)"
            + " + offset * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
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
    STIMULUS_PARAMETERS_MAPPING = {}

    ADDITIONAL_PARAMETERS_MAPPING = {
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Custom drive equation",
        default=(
            "var*0.0"
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
        "magnitude": TranslationRule("magnitude", sim_unit="kHz"),
        "offset": TranslationRule("offset", sim_unit="kHz"),
        "pulse_duration": TranslationRule("pulse_duration", sim_unit="ms"),
        "pulse_period": TranslationRule("pulse_period", sim_unit="ms"),
    }

    ADDITIONAL_PARAMETERS_MAPPING = {
        "initial_increase_duration": TranslationRule("initial_increase_duration", sim_unit="ms"),
    }

    equation = Final(label="Custom Pulse Train Equation",
        default=(
            "where((var >= start) & (var<end) & ((var-start) % pulse_period < pulse_duration), magnitude, 0.0)"
            + " * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
            + " * where((var>=start) & (var<end), 1.0, 0.0)"
            + " + offset * where(var < initial_increase_duration, var/initial_increase_duration, 1.0)"
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
        
        stim_model.parameters.update({
            **translate_params(stim_params.stim_params, profile_class.STIMULUS_PARAMETERS_MAPPING),
            **translate_params(stim_params, profile_class.ADDITIONAL_PARAMETERS_MAPPING),
        })
    else:
        raise NotImplementedError(f"Stimulus pattern '{stim_params.pattern}' is not implemented yet. Please implement the stimulus preparation for this pattern in setup_stimulus().")
    
    return stim_model