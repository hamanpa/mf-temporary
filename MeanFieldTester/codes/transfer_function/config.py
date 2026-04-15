from typing import Literal, Dict, Any, Optional, Annotated
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
import numpy as np
import json
import pickle

# --- 1. Basic Data Structures ---
class FittingOptions(BaseModel):
    method: str = Field(...)
    options: Dict[str, Any] = Field(default_factory=dict)

class TFCoefficients(BaseModel):
    P_0: float = Field(..., description="Constant term in [mV]")
    P_mean: float = Field(..., description="Coefficient for voltage_mean in [mV]")
    P_std: float = Field(..., description="Coefficient for voltage_std in [mV]")
    P_tau: float = Field(..., description="Coefficient for voltage_tau in [mV]")

    P_log: float = Field(default=0.0, description="Coefficient for log term of conductance_mean in [mV]")

    P_mean_mean: float = Field(default=0.0, description="Coefficient for square term (voltage_mean x voltage_mean) in [mV]")
    P_std_std: float = Field(default=0.0, description="Coefficient for square term (voltage_std x voltage_std) in [mV]")
    P_tau_tau: float = Field(default=0.0, description="Coefficient for square term (voltage_tau x voltage_tau) in [mV]")
    P_mean_std: float = Field(default=0.0, description="Coefficient for cross term (voltage_mean x voltage_std) in [mV]")
    P_mean_tau: float = Field(default=0.0, description="Coefficient for cross term (voltage_mean x voltage_tau) in [mV]")
    P_std_tau: float = Field(default=0.0, description="Coefficient for cross term (voltage_std x voltage_tau) in [mV]")

    @model_validator(mode='before')
    @classmethod
    def load_from_path_or_list(cls, data: Any) -> Any:
        # Route A: The user provided a file path
        if isinstance(data, (str, Path)):
            filepath = Path(data)
            if not filepath.exists():
                raise ValueError(f"TF Fit file not found: {filepath}")
            
            if filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
            elif filepath.suffix == '.pkl':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            elif filepath.suffix == '.npy':
                data = np.load(filepath)
            else:
                raise ValueError(f"Unsupported file extension {filepath.suffix}. Use .pkl or .json.")
        if isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError(f"Expected a 1D array of coefficients, but got {data.ndim}D.")
            data = data.tolist()  # Convert to list for handling below

        # Route B: The user provided a raw list of floats
        if isinstance(data, list):
            all_keys = ['P_0', 'P_mean', 'P_std', 'P_tau', 'P_log', 'P_mean_mean', 'P_std_std', 'P_tau_tau', 'P_mean_std', 'P_mean_tau', 'P_std_tau']
            match len(data):
                case 4:
                    keys = all_keys[:4]
                case 5:
                    keys = all_keys[:5]
                case 10:
                    keys = all_keys[:4] + all_keys[5:]
                case 11:
                    keys = all_keys
                case _:
                    raise ValueError(f"List provided has {len(data)} coefficients, but expected 4, 5, 10 or 11.")
            print(f"WARNING: Interpreting list of {len(data)} coefficients as: {keys}")
            data = dict(zip(keys, data))
            for key in all_keys:
                if key not in data:
                    data[key] = 0.0
        return data


class Point3D(BaseModel):
    voltage_mean: float = Field(..., description="Mean membrane potential in [mV]")
    voltage_std: float = Field(..., description="Standard deviation of membrane potential in [mV]")
    voltage_tau: float = Field(..., description="Time constant of membrane potential in [ms]")

# --- 2. The Nested Physics Models (The Discriminator) ---
class Zerlaut2018ModelParams(BaseModel):
    # This model ignores adaptation
    model_name: Literal["zerlaut2018"]
    square_terms: bool = Field(True, description="Whether to include square terms in the polynomial expansion.")
    log_term: bool = Field(False, description="Whether to include a logarithmic term of conductance_mean")
    # Adaptation is impossible here!

class DiVolo2019ModelParams(BaseModel):
    # This model adds adaptation but ignores the log term
    model_name: Literal["divolo2019"]
    square_terms: bool = Field(True, description="Whether to include square terms in the polynomial expansion.")


class NeuropsiModelParams(BaseModel):
    model_name: Literal["neuropsi.custom"]
    square_terms: bool = Field(True, description="Whether to include square terms in the polynomial expansion.")
    log_term: bool = Field(False, description="Whether to include a logarithmic term of conductance_mean")
    adaptation: bool = Field(True, description="Whether to include adaptation terms.")

TFModelParams = Annotated[
    Zerlaut2018ModelParams | DiVolo2019ModelParams | NeuropsiModelParams,
    Field(discriminator="model_name")
]

# --- 3. The Master Configuration ---

class RunTFFittingConfig(BaseModel):
    fit_transfer_function: Literal[True] = True

    tf_model: TFModelParams
    expansion_point: Optional[Point3D] = Field(default_factory=lambda: Point3D(voltage_mean=-60.0, voltage_std=4.0, voltage_tau=0.5), description="The point in (voltage_mean, voltage_std, voltage_tau) space around which to perform the polynomial expansion.")
    expansion_norm: Optional[Point3D] = Field(default_factory=lambda: Point3D(voltage_mean=10.0, voltage_std=6.0, voltage_tau=1.0), description="Normalization factors for (voltage_mean, voltage_std, voltage_tau) to improve numerical stability during fitting.")
    
    fit_with_w: bool = True
    out_rate_min: float = Field(default=0.0, description="Minimum output firing rate to consider during fitting, in [Hz]. Output rate have to be strictly greater than this value. Used to avoid numerical issues with very low rates.")
    out_rate_max: float = Field(default=200.0, description="Maximum output firing rate to consider during fitting, in [Hz]. Output rate have to be strictly less than this value. Used to exclude saturation regime where TF becomes flat and fitting is unstable.")
    V_eff_fitting: Optional[FittingOptions] = FittingOptions(
                                                    method="SLSQP", 
                                                    options={
                                                        'ftol' : 5e-9,
                                                        'disp' : False,
                                                        'maxiter' : 10000
                                                    })
    TF_fitting: Optional[FittingOptions] = FittingOptions(
                                                    method="nelder-mead", 
                                                    options={
                                                        'fatol' : 5e-9,
                                                        'disp' : False,
                                                        'maxiter' : 10000
                                                    })

class LoadTFFittingConfig(BaseModel):
    fit_transfer_function: Literal[False] = False
    
    tf_model: TFModelParams
    expansion_point: Optional[Point3D] = Field(default_factory=lambda: Point3D(voltage_mean=-60.0, voltage_std=4.0, voltage_tau=0.5), description="The point in (voltage_mean, voltage_std, voltage_tau) space around which to perform the polynomial expansion.")
    expansion_norm: Optional[Point3D] = Field(default_factory=lambda: Point3D(voltage_mean=10.0, voltage_std=6.0, voltage_tau=1.0), description="Normalization factors for (voltage_mean, voltage_std, voltage_tau) to improve numerical stability during fitting.")
    
    tf_fits: Dict[str, TFCoefficients]


TransferFunctionConfig = Annotated[
    RunTFFittingConfig | LoadTFFittingConfig,
    Field(discriminator="fit_transfer_function")
]