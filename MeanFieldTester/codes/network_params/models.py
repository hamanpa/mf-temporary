import yaml
from pathlib import Path
from typing import Dict, Literal, Annotated, Union
from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

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
    syn_type: Literal["static_synapse"] = "static_synapse"
    syn_params: StaticSynapseParams

class TsodyksSynapseDefinition(BaseModel):
    syn_type: Literal["tsodyks_synapse"] = "tsodyks_synapse"
    syn_params: TsodyksSynapseParams

SynapseDefinition = Annotated[
    Union[StaticSynapseDefinition, TsodyksSynapseDefinition],
    Field(discriminator="syn_type")
]

# ==========================================
# CONNECTIVITY / PROJECTION MODELS
# ==========================================
ConnectivityRule = Literal["fixed_prob", "fixed_in", "fixed_out"]

class ConnectionDefinition(BaseModel):
    """
    Defines a projection from source_neuron to target_neuron.
    Combines topology rule, count/probability, and synapse parameters.
    """
    rule: ConnectivityRule = Field(default="fixed_prob", description="Connectivity rule: fixed_prob, fixed_in, or fixed_out")
    val: float = Field(..., gt=0.0, description="Probability (0..1) or connection count K (fixed_in/fixed_out)")
    syn_type: Literal["static_synapse", "tsodyks_synapse"]
    syn_params: Union[StaticSynapseParams, TsodyksSynapseParams]

    # Private context attached automatically by BiologicalParameters
    _source_size: int | None = PrivateAttr(default=None)
    _target_size: int | None = PrivateAttr(default=None)
    _source_name: str | None = PrivateAttr(default=None)
    _target_name: str | None = PrivateAttr(default=None)

    @property
    def conn_num(self) -> int:
        """Returns the exact number of synaptic connections (K)."""
        if self.rule in ("fixed_in", "fixed_out"):
            return int(self.val)
        elif self.rule == "fixed_prob":
            if self._source_size is None:
                raise ValueError("ConnectionDefinition context is uninitialized (_source_size missing).")
            return int(round(self.val * self._source_size))
        raise ValueError(f"Unknown rule: {self.rule}")

    @property
    def conn_prob(self) -> float:
        """Returns the connection probability p in (0, 1]."""
        if self.rule == "fixed_prob":
            return self.val
        elif self.rule in ("fixed_in", "fixed_out"):
            if self._source_size is None or self._source_size == 0:
                raise ValueError("ConnectionDefinition context is uninitialized (_source_size missing).")
            return self.val / self._source_size
        raise ValueError(f"Unknown rule: {self.rule}")


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
    Union[AdExDefinition, PoissonDefinition],
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
    connectivity: Dict[str, Dict[str, ConnectionDefinition]] = Field(
        ..., 
        description="Nested mapping: {target_pop: {source_pop: ConnectionDefinition}}"
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

    @model_validator(mode="after")
    def _attach_connection_contexts_and_validate(self):
        """Strictly validate target neurons and attach metadata to ConnectionDefinitions."""
        internal_set = set(self.internal_neurons)
        for target_name, sources in self.network.connectivity.items():
            if target_name not in internal_set:
                raise ValueError(
                    f"Invalid connectivity target '{target_name}'. Target populations in network.connectivity "
                    f"must be internal neurons (found internal neurons: {self.internal_neurons}). "
                    f"External sources (e.g. drive_neuron, stim_neuron) cannot receive incoming connections."
                )
            target_size = self.network.size[target_name]
            for source_name, conn_def in sources.items():
                source_size = self.network.size[source_name]
                conn_def._source_name = source_name
                conn_def._target_name = target_name
                conn_def._source_size = source_size
                conn_def._target_size = target_size
        return self

    @property
    def internal_neurons(self) -> list[str]:
        """Returns a list of population names that are internal (not external drives)."""
        return [name for name, definition in self.neurons.items() if not definition.is_external]

    @property
    def total_size(self) -> int:
        """The sum of ALL populations (Internal + External Drives/Stimuli)."""
        return sum(self.network.size.values())

    @property
    def internal_size(self) -> int:
        """
        Data-driven logic! We don't guess by name.
        We check if the physics model is 'adex' (internal) vs 'poisson' (external).
        """

        return sum(self.network.size[pop_name] for pop_name in self.internal_neurons)

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

    @property
    def exc_neuron_name(self) -> str:
        """Returns the name of the excitatory population. Assumes there is exactly one."""
        exc_neurons = [name for name in self.internal_neurons if self.neurons[name].neuron_type == "excitatory"]
        if len(exc_neurons) != 1:
            raise ValueError(f"Expected exactly one excitatory population, but found {len(exc_neurons)}: {exc_neurons}")
        return exc_neurons[0]

    @property
    def inh_neuron_name(self) -> str:
        """Returns the name of the inhibitory population. Assumes there is exactly one."""
        inh_neurons = [name for name in self.internal_neurons if self.neurons[name].neuron_type == "inhibitory"]
        if len(inh_neurons) != 1:
            raise ValueError(f"Expected exactly one inhibitory population, but found {len(inh_neurons)}: {inh_neurons}")
        return inh_neurons[0]
