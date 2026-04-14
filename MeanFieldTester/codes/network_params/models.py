import yaml
from pathlib import Path
from typing import Dict, Literal, Annotated
from pydantic import BaseModel, Field, computed_field

# ==========================================
# SYNAPSE MODELS
# ==========================================

class StaticSynapseParams(BaseModel):
    """Requirements for a static synapse."""
    weight: float = Field(..., description="Synaptic weight [nS]")
    delay: float = Field(..., description="Synaptic delay [ms]")

class TsodyksSynapseParams(BaseModel):
    """Requirements for a Tsodyks-Markram STP synapse."""
    weight: float = Field(..., description="Synaptic weight [nS]")
    delay: float = Field(..., description="Synaptic delay [ms]")
    U: float = Field(..., ge=0.0, le=1.0, description="Utilization of synaptic efficacy")
    tau_rec: float = Field(..., gt=0, description="Recovery time constant [ms]")
    tau_psc: float = Field(..., gt=0, description="Post-synaptic current time constant [ms]")
    tau_fac: float = Field(..., ge=0, description="Facilitation time constant [ms]")

class StaticSynapseDefinition(BaseModel):
    syn_type: Literal["static_synapse"]
    syn_params: StaticSynapseParams

class TsodyksSynapseDefinition(BaseModel):
    syn_type: Literal["tsodyks_synapse"]
    syn_params: TsodyksSynapseParams

SynapseDefinition = Annotated[
    StaticSynapseDefinition | TsodyksSynapseDefinition,
    Field(discriminator="syn_type")
]

# ==========================================
# NEURON MODELS
# ==========================================
class ConductanceBasedAdExNeuronParams(BaseModel):
    """
    Standard Internal Representation (SIR) for an AdEx Neuron.
    Units: Voltage [mV], Time [ms], Capacitance [nF], Conductance [nS], Current [nA]
    """
    v_rest: float = Field(..., description="Resting membrane potential [mV]")
    v_reset: float = Field(..., description="Reset potential after spike [mV]")
    tau_refrac: float = Field(..., description="Refractory period [ms]")
    tau_m: float = Field(..., description="Membrane time constant [ms]")
    cm: float = Field(..., description="Membrane capacitance [nF]")
    
    e_rev_E: float = Field(..., description="Excitatory reversal potential [mV]")
    e_rev_I: float = Field(..., description="Inhibitory reversal potential [mV]")
    tau_syn_E: float = Field(..., description="Excitatory synaptic time constant [ms]")
    tau_syn_I: float = Field(..., description="Inhibitory synaptic time constant [ms]")
    
    a: float = Field(..., description="Subthreshold adaptation conductance [nS]")
    b: float = Field(..., description="Spike-triggered adaptation increment [nA]")
    delta_T: float = Field(..., description="Slope factor [mV]")
    tau_w: float = Field(..., description="Adaptation time constant [ms]")
    v_thresh: float = Field(..., description="Spike threshold [mV]")

    @computed_field(description="Leak conductance [nS]. Calculated as (cm / tau_m).")
    @property
    def g_L(self) -> float:
        """Derived Leak conductance [nS] calculated as (cm / tau_m) * 1000."""
        return (self.cm / self.tau_m) * 1000.0

class PoissonParams(BaseModel):
    """Parameters for a Poisson source."""
    rate: float = Field(default=10.0, description="Mean firing rate [kHz]")


class AdExDefinition(BaseModel):
    neuron_type: Literal["excitatory", "inhibitory"] = Field(..., description="The nature of the synapses the neuron makes (excitatory or inhibitory)")
    neuron_model: Literal["adex"]
    is_external: Literal[False] = False  # Automatically False for AdEx!
    neuron_params: ConductanceBasedAdExNeuronParams

class PoissonDefinition(BaseModel):
    neuron_type: Literal["excitatory", "inhibitory"] = Field("excitatory", description="The nature of the synapses the neuron makes (excitatory or inhibitory)")
    neuron_model: Literal["poisson_generator"]
    is_external: Literal[True] = True  # Poisson sources are always external, so we can set this as a fixed value.
    neuron_params: PoissonParams | None = None

NeuronDefinition = Annotated[
    AdExDefinition | PoissonDefinition,
    Field(discriminator="neuron_model")
]


# ==========================================
# 3. NETWORK TOPOLOGY MODELS
# ==========================================
class NetworkTopology(BaseModel):
    """Handles the programmatic adjacency matrix and sizes."""
    size: Dict[str, int] = Field(
        ..., 
        description="Population sizes. Map of pop_name -> N"
    )
    connectivity: Dict[str, Dict[str, float]] = Field(
        ..., 
        description="Nested mapping: {target_pop: {source_pop: probability}}"
    )

# ==========================================
# 4. MASTER ROOT MODEL
# ==========================================
class BiologicalParameters(BaseModel):
    """
    The root model representing the entire biological setup.
    Matches the master structure of the YAML file.
    """
    neurons: Dict[str, NeuronDefinition]
    network: NetworkTopology
    synapses: Dict[str, SynapseDefinition]


    @property
    def internal_neurons(self) -> list[str]:
        """Returns a list of population names that are internal (not external drives)."""
        return [name for name, definition in self.neurons.items() if not definition.is_external]

    # @computed_field
    @property
    def total_size(self) -> int:
        """The sum of ALL populations (Internal + External Drives/Stimuli)."""
        return sum(self.network.size.values())

    # @computed_field
    @property
    def internal_size(self) -> int:
        """
        Data-driven logic! We don't guess by name.
        We check if the physics model is 'adex' (internal) vs 'poisson' (external).
        """

        return sum(self.network.size[pop_name] for pop_name in self.internal_neurons)

    # @computed_field
    @property
    def g(self) -> float:
        """
        Ratio of inhibitory neurons to the INTERNAL network size.
        Identifies inhibitory populations by looking for 'inh' in their name.
        """
        if self.internal_size == 0:
            return 0.0
            
        inh_size = sum(
            self.network.size[name]
            for name in self.internal_neurons
            if self.neurons[name].neuron_type == "inhibitory"
        )

        return inh_size / self.internal_size



