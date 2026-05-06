"""

"""


from abc import ABC, abstractmethod
import numpy as np
from .config import StimulusConfig, BaseStimulusConfig, NoStimulusConfig, SinusoidalConfig, PulseTrainConfig, TwoSidedGaussianConfig
from ..network_params.translators import get_unit_multiplier



class BaseRateProfile(ABC):
    """
    Abstract base class establishing the Strategy Pattern for stimulus rate generation.
    """
    def __init__(self, stim_params: BaseStimulusConfig):
        self.drive_rate_const = stim_params.drive_rate
        self.drive_increase_duration = stim_params.drive_increase_duration
        self.stim_target_ratio = stim_params.stim_target_ratio
        self.simulation_duration = stim_params.simulation_duration
        self.target_nodes = stim_params.target_nodes
        self.direct_stimulation = stim_params.direct_stimulation


    @abstractmethod
    def stim_rate(self, times: np.ndarray, target_unit: str = "Hz") -> np.ndarray:
        """
        Generates the instantaneous firing rate at each time step.
        
        Args:
            times (np.ndarray): 1D array of time steps in ms.
            
        Returns:
            np.ndarray: 1D array of rates in Hz.
        """
        pass

    def drive_rate(self, times: np.ndarray, target_unit: str = "Hz") -> np.ndarray:
        rate = np.where(times<self.drive_increase_duration, 
                        times*self.drive_rate_const/self.drive_increase_duration, 
                        self.drive_rate_const)
        
        return rate * get_unit_multiplier("Hz", target_unit)

class NoStimulusRateProfile(BaseRateProfile):

    def __init__(self, stim_params: NoStimulusConfig):
        super().__init__(stim_params)

    def stim_rate(self, times: np.ndarray, target_unit: str = "Hz") -> np.ndarray:
        return np.zeros_like(times) * get_unit_multiplier("Hz", target_unit)

class SinusoidalRateProfile(BaseRateProfile):
    """Generates a sinusoidal oscillation for network stimulation."""
    
    def __init__(self, stim_params: SinusoidalConfig):
        super().__init__(stim_params)

        self.stim_start = stim_params.stim_params.start
        self.stim_end = stim_params.stim_params.end
        self.magnitude = stim_params.stim_params.magnitude
        self.offset = stim_params.stim_params.offset

        self.freq = stim_params.stim_params.freq
        self.phase = stim_params.stim_params.phase

    def stim_rate(self, times: np.ndarray, target_unit: str = "Hz") -> np.ndarray:
        rate = np.where(times<self.drive_increase_duration, 
                        times*self.offset/self.drive_increase_duration, 
                        self.offset)

        mask = (times >= self.stim_start) & (times <= self.stim_end)

        rate[mask] += self.magnitude * np.sin(2 * np.pi * self.freq * (times[mask] - self.stim_start) * 1e-3 + self.phase)
        
        return np.maximum(rate, 0.0) * get_unit_multiplier("Hz", target_unit)

class TwoSidedGaussianRateProfile(BaseRateProfile):
    def __init__(self, stim_params):
        super().__init__(stim_params)

        self.stim_start = stim_params.stim_params.start
        self.stim_end = stim_params.stim_params.end
        self.magnitude = stim_params.stim_params.magnitude
        self.offset = stim_params.stim_params.offset

        self.center = stim_params.stim_params.center
        self.sigma_left = stim_params.stim_params.sigma_left
        self.sigma_right = stim_params.stim_params.sigma_right

    def stim_rate(self, times: np.ndarray, target_unit: str = "Hz") -> np.ndarray:
        rate = np.where(times < self.drive_increase_duration, 
                        times * self.offset / self.drive_increase_duration, 
                        self.offset)

        mask = (times >= self.stim_start) & (times <= self.stim_end)
        full_left_mask = mask & (times < self.center)
        full_right_mask = mask & (times >= self.center)

        rate[full_left_mask] += self.magnitude * np.exp(-0.5 * ((times[full_left_mask] - self.center) / self.sigma_left) ** 2)
        rate[full_right_mask] += self.magnitude * np.exp(-0.5 * ((times[full_right_mask] - self.center) / self.sigma_right) ** 2)
        
        return np.maximum(rate, 0.0) * get_unit_multiplier("Hz", target_unit)

class PulseTrainRateProfile(BaseRateProfile):
    def __init__(self, stim_params):
        super().__init__(stim_params)

        self.stim_start = stim_params.stim_params.start
        self.stim_end = stim_params.stim_params.end
        self.magnitude = stim_params.stim_params.magnitude
        self.offset = stim_params.stim_params.offset

        self.pulse_duration = stim_params.stim_params.pulse_duration
        self.pulse_period = stim_params.stim_params.pulse_period

    def stim_rate(self, times: np.ndarray, target_unit: str = "Hz") -> np.ndarray:
        rate = np.where(times < self.drive_increase_duration, 
                        times * self.offset / self.drive_increase_duration, 
                        self.offset)

        window_mask = (times >= self.stim_start) & (times <= self.stim_end)
        pulse_condition = ((times - self.stim_start) % self.pulse_period) < self.pulse_duration

        full_pulse_mask = window_mask & pulse_condition

        rate[full_pulse_mask] += self.magnitude

        return np.maximum(rate, 0.0) * get_unit_multiplier("Hz", target_unit)
    
class CustomStimulusRateProfile(BaseRateProfile):
    """Placeholder for future custom stimulus profiles."""
    pass
