"""



"""


import numpy as np
import pyNN.nest as sim

from codes.data_structures.network import SNNFullResults

"""
Simulation parameters
    simulation time
    dt (time step)
    simulator seed
    number of simulations (for statistical analysis)
    results folder

Network parameters
    number of neurons
    connections
    synapse parameters
    neuron parameters
    const external input?

External input (stimulus)
    design (population, spike times etc.)
    time variable input

Design assumptions
two populations of neurons
    excitatory
    inhibitory

    
"""
EXC_RECORDED = 500
INH_RECORDED = 500

# Functions to generate the external input

# TODO: write the source patterns as classes
# class NonhomogeneousPoissonGenerator:

def nonhomogeneous_poisson_generator(rate : np.array, t_stop : int, t_start : int = 0, seed=42):
    """Nonhomogeneous Poisson process generator for one source neuron."""
    assert rate.ndim == 1
    np.random.seed(seed)

    max_rate = np.max(rate)
    dt = (t_stop - t_start)/(len(rate)-1)

    spikes = []
    t = t_start
    while t < t_stop:
        # Generate the next candidate spike time.
        t += np.random.exponential(1.0 / max_rate) * 1000.0
        if t >= t_stop:
            break
        t_idx = int((t-t_start) / dt)
        # Thinning - only accept the spike if a random number is less than the rate at time t divided by the maximum rate
        if np.random.rand() < rate[t_idx] / max_rate:
            spikes.append(t)
    return spikes

def nhpp_many(rate : np.array, pop_size : int, t_stop : int, t_start : int = 0, seed=42):
    """Nonhomogeneous Poisson process generator for multiple source neurons."""
    assert rate.ndim == 1
    np.random.seed(seed)

    max_rate = max(np.max(rate),1e-6)
    dt = (t_stop - t_start)/(len(rate)-1)

    spikes = [[] for _ in range(pop_size)]
    t = np.ones(pop_size) * t_start
    while True:
        # Generate the next candidate spike times.
        t += np.random.exponential(1.0 / max_rate, size=pop_size) * 1000.0
        reached_end = (t >= t_stop)
        if np.all(reached_end):
            break

        # Thinning - only accept the spike if a random number is less than the rate at time t divided by the maximum rate
        for i in np.where(reached_end == False)[0]:
            t_idx = int((t[i]-t_start) / dt)
            if np.random.rand() < rate[t_idx] / max_rate:
                spikes[i].append(t[i])
    return spikes

# Function generating the rate arrays

def generate_drive_rate(times, stim_magnitude, init_duration, offset=0):
    """Generate a constant rate with linear increase to the maximum rate."""
    rate = np.where(times<init_duration, times*stim_magnitude/init_duration+offset, stim_magnitude+offset)
    return rate 
    rate = np.ones_like(times) * stim_magnitude + offset
    if init_duration:
        mask = times <= init_duration
        rate[mask] = np.linspace(0, stim_magnitude, sum(mask)) + offset
    return rate
    bins = int((t_end-t_start) / dt) + 1
    bins_gradual = int(init_duration / dt) + 1
    rates = stim_magnitude * np.ones(bins)
    rates[:bins_gradual] = np.linspace(0, stim_magnitude, bins_gradual)
    return rates

def generate_double_gaussian_rate(times, stim_magnitude, center, sigma_left, sigma_right, offset=0):
    rate = np.zeros_like(times)
    left_mask = times < center
    right_mask = times >= center
    rate[left_mask] = np.exp(-0.5 * ((times[left_mask] - center) / sigma_left) ** 2)
    rate[right_mask] = np.exp(-0.5 * ((times[right_mask] - center) / sigma_right) ** 2)
    return stim_magnitude*rate

def generate_sinusoidal_rate(times, stim_start, stim_end, stim_magnitude, freq, offset):
    rate = stim_magnitude * np.sin(2*np.pi*freq*1e-3 * (times-stim_start)) + offset
    mask = (times < stim_start) | (times >= stim_end)
    rate[mask] = offset
    return rate

def generate_null_rate(times):
    return np.zeros_like(times)

def generate_pulse_train(times, stim_start, stim_duration, stim_magnitude, stim_period=1e9):
    """Generate a pulse train rate profile."""
    rate = np.zeros_like(times)
    for t in range(stim_start, int(times[-1]), int(stim_period)):
        pulse_start = t
        pulse_end = t + stim_duration
        rate[(times >= pulse_start) & (times < pulse_end)] = stim_magnitude
    return rate


# Function to simulate the network

def run_network(sim_pars, neurons, net_pars, stimulus, results_path, file_name='NetworkSimulation.pickle', save=False):
    print("Setting up the network simulation...")
    decimal_places = -int(np.floor(np.log10(sim_pars["timestep"])))
    exc_neuron = neurons["exc_neuron"]
    inh_neuron = neurons["inh_neuron"]

    times = np.linspace(0, stimulus["simulation_time"], int(stimulus["simulation_time"]/sim_pars["timestep"])+1)

    seed = sim_pars["seed"]

    drive_rate_array = generate_drive_rate(times, stimulus["drive_rate"], stimulus["drive_increase_duration"])
    drive_spikes = nhpp_many(drive_rate_array, net_pars['drive_pop_size'], stimulus["simulation_time"], seed=seed)
    drive_spikes = [list(np.round(np.array(spikes), decimals=decimal_places)) for spikes in drive_spikes]
    # sorted([x[0] for x in drive_spikes])

    match stimulus["pattern"]:
        case "NoStimulus":
            stim_rate_array = generate_null_rate(times)
        case "PulseTrain":
            stim_rate_array = generate_pulse_train(times, **stimulus['stim_pars'])
        case "TwoSidedGaussian":
            stim_rate_array = generate_double_gaussian_rate(times, **stimulus['stim_pars'])
        case "Sinusoidal":
            stim_rate_array = generate_sinusoidal_rate(times, **stimulus['stim_pars'])
        case "GradualConstant":
            stim_rate_array = generate_drive_rate(times, **stimulus['stim_pars'])
        case _:
            raise NotImplementedError
    
    exc_stim_spikes = nhpp_many(stim_rate_array, net_pars['stim_pop_size'], stimulus["simulation_time"], seed=seed+1)
    exc_stim_spikes = [list(np.round(np.array(spikes), decimals=decimal_places)) for spikes in exc_stim_spikes]

    inh_stim_spikes = nhpp_many(stim_rate_array*stimulus['stim_target_ratio'], net_pars['stim_pop_size'], stimulus["simulation_time"], seed=seed+2)
    inh_stim_spikes = [list(np.round(np.array(spikes), decimals=decimal_places)) for spikes in inh_stim_spikes]

    # Throw away spikes that are too early
    for spike_array in [drive_spikes, exc_stim_spikes, inh_stim_spikes]:
        for i, st in enumerate(spike_array):
            while st and (st[0] <= sim_pars["timestep"]):
                st = st[1:]
            spike_array[i] = st

    # TODO: set the seed
    sim.setup(timestep=sim_pars["timestep"], rng_seed=seed)

    # Setup network
    exc_neurons = sim.Population(net_pars['exc_pop_size'], sim.EIF_cond_exp_isfa_ista(**exc_neuron['neuron_params']), initial_values=exc_neuron["init_values"])
    inh_neurons = sim.Population(net_pars['inh_pop_size'], sim.EIF_cond_exp_isfa_ista(**inh_neuron['neuron_params']), initial_values=inh_neuron["init_values"])
    drive_neurons = sim.Population(net_pars['drive_pop_size'], sim.SpikeSourceArray(spike_times=drive_spikes))
    exte_neurons = sim.Population(net_pars['stim_pop_size'], sim.SpikeSourceArray(spike_times=exc_stim_spikes))
    exti_neurons = sim.Population(net_pars['stim_pop_size'], sim.SpikeSourceArray(spike_times=inh_stim_spikes))

    # conn_SourceTarget
    conn_ee = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_exc"])
    conn_ei = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_exc"])
    conn_ie = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_inh"])
    conn_ii = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_inh"])

    conn_de = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_drive"])
    conn_di = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_drive"])

    conn_exte = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_stim"])  # ext --> exc
    conn_exti = sim.FixedProbabilityConnector(p_connect=net_pars["p_connect_stim"])  # ext --> inh

    synapse_exc = sim.native_synapse_type(exc_neuron['exc_synapses']['syn_type'])(**exc_neuron['exc_synapses']['syn_params'])
    synapse_inh = sim.native_synapse_type(exc_neuron['inh_synapses']['syn_type'])(**exc_neuron['inh_synapses']['syn_params'])
    synapse_drive = sim.native_synapse_type(exc_neuron['drive_synapses']['syn_type'])(**exc_neuron['drive_synapses']['syn_params'])
    synapse_stim = sim.native_synapse_type(exc_neuron['stim_synapses']['syn_type'])(**exc_neuron['stim_synapses']['syn_params'])

    sim.Projection(exc_neurons, exc_neurons, conn_ee, synapse_type=synapse_exc, receptor_type='excitatory')
    sim.Projection(exc_neurons, inh_neurons, conn_ei, synapse_type=synapse_exc, receptor_type='excitatory')
    sim.Projection(inh_neurons, exc_neurons, conn_ie, synapse_type=synapse_inh, receptor_type='inhibitory')
    sim.Projection(inh_neurons, inh_neurons, conn_ii, synapse_type=synapse_inh, receptor_type='inhibitory')

    sim.Projection(drive_neurons, exc_neurons, conn_de, synapse_type=synapse_drive, receptor_type='excitatory')
    sim.Projection(drive_neurons, inh_neurons, conn_di, synapse_type=synapse_drive, receptor_type='excitatory')

    sim.Projection(exte_neurons, exc_neurons, conn_exte, synapse_type=synapse_stim, receptor_type='excitatory')
    sim.Projection(exti_neurons, inh_neurons, conn_exti, synapse_type=synapse_stim, receptor_type='excitatory')

    exc_neurons.sample(EXC_RECORDED).record(['v', 'spikes', 'w', 'gsyn_exc', 'gsyn_inh'])
    inh_neurons.sample(INH_RECORDED).record(['v', 'spikes', 'w', 'gsyn_exc', 'gsyn_inh'])

    print("Running the simulation...")
    sim.run(stimulus["simulation_time"])
    print("Simulation finished.")

    print("Retrieving exc data")
    exc_data = exc_neurons.get_data().segments[0]
    print("Retrieving inh data")
    inh_data = inh_neurons.get_data().segments[0]

    results = SNNFullResults(exc_data, inh_data, drive_rate_array, stim_rate_array, stimulus, net_pars)
    return results


    print("Making list of spikes")
    exc_spikes = exc_data.spiketrains
    exc_spikes = [list(sp.magnitude) for sp in exc_spikes]
    inh_spikes = inh_data.spiketrains
    inh_spikes = [list(sp.magnitude) for sp in inh_spikes]

    print("Retrieving voltages")    
    exc_v = exc_data.filter(name='v')[0].magnitude
    inh_v = inh_data.filter(name='v')[0].magnitude
    exc_w = exc_data.filter(name='w')[0].magnitude

    print("Done retrieving data")
    try:
        sim.end()
    except FileNotFoundError:
        # Already cleaned up or missing temp files, it's fine
        pass

    results = {
        'times' : times,
        'exc_spikes' : exc_spikes,
        'exc_v' : exc_v,
        'inh_spikes' : inh_spikes,
        'inh_v' : inh_v,
        'drive_rate' : drive_rate_array,
        'stim_rate' : stim_rate_array,
        'exc_w' : exc_w,
        'gsyn_ee' : exc_data.filter(name='gsyn_exc')[0].magnitude,
        'gsyn_ei' : exc_data.filter(name='gsyn_inh')[0].magnitude,
        'gsyn_ie' : inh_data.filter(name='gsyn_exc')[0].magnitude,
        'gsyn_ii' : inh_data.filter(name='gsyn_inh')[0].magnitude,
    }
    if save:
        save_to_pickle(results_path / file_name, **results)

    return results
