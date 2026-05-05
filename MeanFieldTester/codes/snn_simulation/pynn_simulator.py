import importlib
import numpy as np
from typing import Dict, Any

from .config import NetworkSimulationConfig
from .base import BaseSNNSimulator
from ..stimuli.config import BaseStimulusConfig
from ..data_structures.network import SNNFullResults
from ..network_params.models import BiologicalParameters
from ..network_params.translators import TranslationRule, translate_params
from ..neuron_simulation.pynn_simulator import PYNN_ADEX_MAPPING, PYNN_STATIC_SYNAPSE_MAPPING, NEST_STATIC_SYNAPSE_MAPPING, NEST_TSODYKS_SYNAPSE_MAPPING, PYNN_INITIAL_VALUES_MAPPING
from ..stimuli import create_rate_profile

class PyNNSNNSimulator(BaseSNNSimulator):
    """
    Object-oriented manager for executing PyNN Spiking Neural Network simulations.
    Implements a Build -> Run Multiple -> Reset lifecycle for maximum efficiency.
    """

    def __init__(self):
        self.sim = None
        self.populations: Dict[str, Any] = {}
        self.projections: list = []
        self._network_built = False
        self.network_params = None
        self.sim_params = None


    def build_network(self, network_params: BiologicalParameters, snn_sim_params: NetworkSimulationConfig) -> None:
        """Phase 1: Setup environment, build neurons, and wire connections."""
        self.network_params = network_params
        self.sim_params = snn_sim_params

        backend_name = snn_sim_params.simulator.split(".")[-1]
        self.sim = importlib.import_module(f"pyNN.{backend_name}")
        self.sim.setup(
            timestep=snn_sim_params.time_step, 
            rng_seed=snn_sim_params.seed,
        )

        for pop_name in network_params.internal_neurons:
            pop_size = network_params.network.size[pop_name]
            neuron_params = network_params.neurons[pop_name].neuron_params
            init_values = snn_sim_params.init_values[pop_name]
            
            cell_model = getattr(self.sim, "EIF_cond_exp_isfa_ista")
            self.populations[pop_name] = self.sim.Population(
                pop_size, 
                cell_model(**translate_params(neuron_params, PYNN_ADEX_MAPPING)),
                initial_values=translate_params(init_values, PYNN_INITIAL_VALUES_MAPPING),
                label=pop_name
            )
            
            sample_size = snn_sim_params.recorded_samples[pop_name]
            sample_size = min(sample_size, pop_size)
            self.populations[pop_name].sample(sample_size).record(['spikes', 'v', 'w', 'gsyn_exc', 'gsyn_inh'])

        for pop_name, pop_size in network_params.network.size.items():
            if pop_name not in network_params.internal_neurons:
                empty_spikes = [[] for _ in range(pop_size)]
                self.populations[pop_name] = self.sim.Population(
                    pop_size, 
                    self.sim.SpikeSourceArray(spike_times=empty_spikes),
                    label=pop_name
                )

        for target_pop, sources in network_params.network.connectivity.items():
            for source_pop, prob in sources.items():
                if prob <= 0.0:
                    continue
                    
                connector = self.sim.FixedProbabilityConnector(p_connect=prob)
                synapse_param = network_params.synapses[source_pop]
                match synapse_param.syn_type:
                    case "static_synapse":
                        synapse_mapping = NEST_STATIC_SYNAPSE_MAPPING
                    case "tsodyks_synapse":
                        synapse_mapping = NEST_TSODYKS_SYNAPSE_MAPPING
                    case _:
                        raise ValueError(f"Unknown synapse type: {synapse_param.syn_type}")
                
                synapse_type = self.sim.native_synapse_type(synapse_param.syn_type)(**translate_params(synapse_param.syn_params, synapse_mapping))
                receptor = network_params.neurons[source_pop].neuron_type
               
                proj = self.sim.Projection(
                    self.populations[source_pop], 
                    self.populations[target_pop], 
                    connector, 
                    synapse_type, 
                    receptor_type=receptor
                )
                self.projections.append(proj)
        self._network_built = True

    def run_stimulus(self, stim_params: BaseStimulusConfig) -> SNNFullResults:
        """Phase 2: Inject specific stimulus, run the simulation, extract data."""

        if not self._network_built:
            raise RuntimeError("Network not built. Call build_network() first.")

        times = np.arange(0, stim_params.simulation_duration+self.sim_params.time_step, self.sim_params.time_step)
        
        stim_profile = create_rate_profile(stim_params)
        drive_rate = stim_profile.drive_rate(times)
        stim_rate = stim_profile.stim_rate(times)

        for pop_name, pop_size in self.network_params.network.size.items():
            if pop_name not in self.network_params.internal_neurons:
                if pop_name.startswith("stim"):
                    rate_to_use = stim_rate
                elif pop_name.startswith("drive"):
                    rate_to_use = drive_rate
                else:
                    raise ValueError(f"Unknown external population name: {pop_name}")
                new_spikes = self._generate_nhpp_spikes(rate_to_use, times, pop_size)
                
                self.populations[pop_name].set(spike_times=new_spikes)

        self.sim.run(stim_params.simulation_duration)

        exc_data = self.populations["exc_neuron"].get_data().segments[0]
        inh_data = self.populations["inh_neuron"].get_data().segments[0]

        results = SNNFullResults(
            exc_data,
            inh_data,
            drive_rate,
            stim_rate,
            stim_params,
            self.network_params
        )
        return results

    def end(self) -> None:
        """Phase 4: Teardown and cleanup."""
        if self.sim:
            self.sim.end()

        self.sim = None
        self.populations.clear() 
        self.projections.clear()
        self.network_params = None
        self.sim_params = None
        self._network_built = False


    def _generate_nhpp_spikes(self, rate_array: np.ndarray, times: np.ndarray, pop_size: int) -> list:
        """Helper: Generates Nonhomogeneous Poisson Process spike times."""
        rng = np.random.default_rng(self.sim_params.seed)
        max_rate = np.max(rate_array)
        
        if max_rate <= 0:
            return [[] for _ in range(pop_size)]
            
        spikes = []
        for _ in range(pop_size):
            n_events = rng.poisson(max_rate * (times[-1] - times[0]) / 1000.0) 
            homo_spikes = np.sort(rng.uniform(times[0], times[-1], n_events))
            
            spike_indices = np.searchsorted(times, homo_spikes) - 1
            spike_indices = np.clip(spike_indices, 0, len(rate_array) - 1)
            acceptance_probs = rate_array[spike_indices] / max_rate
            
            accepted_spikes = homo_spikes[rng.random(len(homo_spikes)) < acceptance_probs]
            decimals = int(-np.log10(self.sim_params.time_step))
            spikes.append(np.round(accepted_spikes, decimals).tolist())
            
        return spikes