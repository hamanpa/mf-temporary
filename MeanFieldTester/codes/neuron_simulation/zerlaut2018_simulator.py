"""
Simulator for the adex neuron model based on the repository
https://github.com/ModelDBRepository/234992/tree/master

For the paper

Y. Zerlaut, S. Chemla, F. Chavane, and A. Destexhe, “Modeling mesoscopic 
cortical dynamics using a mean-field model of conductance-based networks of 
adaptive exponential integrate-and-fire neurons,” 
NeuroJ Comput Neurosci, vol. 44, no. 1, pp. 45–61, Feb. 2018, 
doi: 10.1007/s10827-017-0668-2.


"""


from .base import BaseNeuronSimulator
from .config import NeuronSimulationConfig
from ..data_structures.single_neuron import SingleNeuronResults
from pathlib import Path
from ..network_params.translators import TranslationRule, translate_params
from pydantic import BaseModel

ZERLAUT2018_ADEX_MAPPING = {
    'Gl': TranslationRule("g_L", sim_unit="nS"),
    'Cm': TranslationRule("cm", sim_unit="pF"),
    'Trefrac': TranslationRule("tau_refrac", sim_unit="ms"),
    'El': TranslationRule("v_rest", sim_unit="mV"),
    'Vthre': TranslationRule("v_thresh", sim_unit="mV"),
    'Vreset': TranslationRule("v_reset", sim_unit="mV"),
    'delta_v': TranslationRule("delta_T", sim_unit="mV"),
    'a': TranslationRule("a", sim_unit="nS"),
    'b': TranslationRule("b", sim_unit="pA"),
    'tauw': TranslationRule("tau_w", sim_unit="ms"),
}

################################################################################
# What follows is the code from the original repository, which we will use to generate data
# The code is left mostly as is, any modifications or indicated by comments


# NOTE: imports across code based were moved here directly
import itertools, string, sys
import numpy as np
import argparse
    
def get_neuron_params(NAME, name='', number=1, SI_units=False):

    if NAME=='LIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':0.,\
                  'a':0., 'b': 0., 'tauw':1e9}
    elif NAME=='EIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':2.,\
                  'a':0., 'b':0., 'tauw':1e9}
    elif NAME=='AdExp':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':2.,\
                  'a':4., 'b':20., 'tauw':500.}
    elif NAME=='FS-cell':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':0.5,\
                  'a':0., 'b': 0., 'tauw':1e9}
    elif NAME=='RS-cell':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                  'a':4., 'b':20., 'tauw':500.}
    elif NAME=='RS-cell2':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                  'a':0., 'b':0., 'tauw':500.}
    else:
        print('====================================================')
        print('------------ CELL NOT RECOGNIZED !! ---------------')
        print('====================================================')


    if SI_units:
        print('cell parameters in SI units')
        # mV to V
        params['El'], params['Vthre'], params['Vreset'], params['delta_v'] =\
            1e-3*params['El'], 1e-3*params['Vthre'], 1e-3*params['Vreset'], 1e-3*params['delta_v']
        # ms to s
        params['Trefrac'], params['tauw'] = 1e-3*params['Trefrac'], 1e-3*params['tauw']
        # nS to S
        params['a'], params['Gl'] = 1e-9*params['a'], 1e-9*params['Gl']
        # pF to F and pA to A
        params['Cm'], params['b'] = 1e-12*params['Cm'], 1e-12*params['b']
    else:
        print('cell parameters --NOT-- in SI units')
        
    return params.copy()

def get_connectivity_and_synapses_matrix(NAME, number=2, SI_units=False):


    # creating empty arry of objects (future dictionnaries)
    M = np.empty((number, number), dtype=object)

    if NAME=='Vogels-Abbott':
        exc_pop = {'p_conn':0.02, 'Q':7., 'Tsyn':5., 'Erev':0.}
        inh_pop = {'p_conn':0.02, 'Q':67., 'Tsyn':10., 'Erev':-80.}
        M[:,0] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : exc
        M[:,1] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : inh
        M[0,0]['name'], M[1,0]['name'] = 'ee', 'ie'
        M[0,1]['name'], M[1,1]['name'] = 'ei', 'ii'

        # in the first element we put the network number and connectivity information
        M[0,0]['Ntot'], M[0,0]['gei'] = 5000, 0.2
        
    elif NAME=='CONFIG1':
        exc_pop = {'p_conn':0.05, 'Q':1., 'Tsyn':5., 'Erev':0.}
        inh_pop = {'p_conn':0.05, 'Q':5., 'Tsyn':5., 'Erev':-80.}
        M[:,0] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : exc
        M[:,1] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : inh
        M[0,0]['name'], M[1,0]['name'] = 'ee', 'ie'
        M[0,1]['name'], M[1,1]['name'] = 'ei', 'ii'
        
        # in the first element we put the network number and connectivity information
        M[0,0]['Ntot'], M[0,0]['gei'] = 10000, 0.2
        M[0,0]['ext_drive'] = 4. # we also store here the choosen excitatory drive 
        M[0,0]['afferent_exc_fraction'] = 1. # we also store here the choosen excitatory drive 
        
        
    else:
        print('====================================================')
        print('------------ NETWORK NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        print('synaptic network parameters in SI units')
        for m in M.flatten():
            m['Q'] *= 1e-9
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3
    else:
        print('synaptic network parameters --NOT-- in SI units')

    return M

### ================================================
### ========== Reformat parameters  ================
### ======= for single cell simulation =============
### ================================================

def reformat_syn_parameters(params, M):
    """
    valid only of no synaptic differences between excitation and inhibition
    """

    params['Qe'], params['Te'], params['Ee'] = M[0,0]['Q'], M[0,0]['Tsyn'], M[0,0]['Erev']
    params['Qi'], params['Ti'], params['Ei'] = M[1,1]['Q'], M[1,1]['Tsyn'], M[1,1]['Erev']
    params['pconnec'] = M[0,0]['p_conn']
    params['Ntot'], params['gei'] = M[0,0]['Ntot'], M[0,0]['gei']
    
### ================================================
### ======== Conductance Time Trace ================
### ====== Poisson + Exponential synapses ==========
### ================================================

def generate_conductance_shotnoise(freq, t, N, Q, Tsyn, g0=0, seed=0):
    """
    generates a shotnoise convoluted with a waveform
    frequency of the shotnoise is freq,
    K is the number of synapses that multiplies freq
    g0 is the starting value of the shotnoise
    """
    if freq==0:
        freq=1e-9
    upper_number_of_events = max([int(3*freq*t[-1]*N),1]) # at least 1 event
    np.random.seed(seed=seed)
    spike_events = np.cumsum(np.random.exponential(1./(N*freq),\
                             upper_number_of_events))
    g = np.ones(t.size)*g0 # init to first value
    dt, t = t[1]-t[0], t-t[0] # we need to have t starting at 0
    # stupid implementation of a shotnoise
    event = 0 # index for the spiking events
    for i in range(1,t.size):
        g[i] = g[i-1]*np.exp(-dt/Tsyn)
        while spike_events[event]<=t[i]:
            g[i]+=Q
            event+=1
    return g

### ================================================
### ======== AdExp model (with IaF) ================
### ================================================

def pseq_adexp(cell_params):
    """ function to extract all parameters to put in the simulation"""

    # those parameters have to be set
    El, Gl = cell_params['El'], cell_params['Gl']
    Ee, Ei = cell_params['Ee'], cell_params['Ei']
    Cm = cell_params['Cm']
    a, b, tauw = cell_params['a'],\
                     cell_params['b'], cell_params['tauw']
    trefrac, delta_v = cell_params['Trefrac'], cell_params['delta_v']
    
    vthresh, vreset =cell_params['Vthre'], cell_params['Vreset']

    # then those can be optional
    if 'vspike' not in cell_params.keys():
        vspike = vthresh+5*delta_v # as in the Brian simulator !

    return El, Gl, Cm, Ee, Ei, vthresh, vreset, vspike,\
                     trefrac, delta_v, a, b, tauw


def adexp_sim(t, I, Ge, Gi,
              El, Gl, Cm, Ee, Ei, vthresh, vreset, vspike, trefrac, delta_v, a, b, tauw):
    """ functions that solve the membrane equations for the
    adexp model for 2 time varying excitatory and inhibitory
    conductances as well as a current input
    returns : v, spikes
    """

    if delta_v==0: # i.e. Integrate and Fire
        one_over_delta_v = 0
    else:
        one_over_delta_v = 1./delta_v
        
    vspike=vthresh+5.*delta_v # practival threshold detection
            
    last_spike = -np.inf # time of the last spike, for the refractory period
    ############################################################################
    # NOTE: PH - Change compared to the original code
    # np.float is deprecated, using built-in float instead
    
    # V, spikes = El*np.ones(len(t), dtype=np.float), []
    V, spikes = El*np.ones(len(t), dtype=float), [] 
    ############################################################################


    dt = t[1]-t[0]

    w, i_exp = 0., 0. # w and i_exp are the exponential and adaptation currents

    for i in range(len(t)-1):
        w = w + dt/tauw*(a*(V[i]-El)-w) # adaptation current
        i_exp = Gl*delta_v*np.exp((V[i]-vthresh)*one_over_delta_v) 
        
        if (t[i]-last_spike)>trefrac: # only when non refractory
            ## Vm dynamics calculus
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                 Gl*(El-V[i]) + Ge[i]*(Ee-V[i]) + Gi[i]*(Ei-V[i]) )

        if V[i+1] > vspike:

            V[i+1] = vreset # non estethic version
            w = w + b # then we increase the adaptation current
            last_spike = t[i+1]
            spikes.append(t[i+1])

    return V, np.array(spikes)


### ================================================
### ========== Single trace experiment  ============
### ================================================

def single_experiment(t, fe, fi, params, seed=0):
    ## fe and fi total synaptic activities, they include the synaptic number
    ge = generate_conductance_shotnoise(fe, t, 1, params['Qe'], params['Te'], g0=0, seed=seed)
    gi = generate_conductance_shotnoise(fi, t, 1, params['Qi'], params['Ti'], g0=0, seed=seed)
    I = np.zeros(len(t))
    v, spikes = adexp_sim(t, I, ge, gi, *pseq_adexp(params))
    return len(spikes)/t.max() # finally we get the output frequency


### ================================================
### ========== Transfer Functions ==================
### ================================================

### generate a transfer function's data
def generate_transfer_function(params,\
                               MAXfexc=40., MAXfinh=30., MINfinh=2.,\
                               discret_exc=9, discret_inh=8, MAXfout=35.,\
                               SEED=3,\
                               verbose=False,
                               filename='data/example_data.npy',
                               dt=5e-5, tstop=10):
    """ Generate the data for the transfer function  """
    
    t = np.arange(int(tstop/dt))*dt

    # this sets the boundaries (factor 20)
    dFexc = MAXfexc/discret_exc
    fiSim=np.linspace(MINfinh,MAXfinh, discret_inh)
    feSim=np.linspace(0, MAXfexc, discret_exc) # by default
    MEANfreq = np.zeros((fiSim.size,feSim.size))
    SDfreq = np.zeros((fiSim.size,feSim.size))
    Fe_eff = np.zeros((fiSim.size,feSim.size))
    JUMP = np.linspace(0,MAXfout,discret_exc) # constrains the fout jumps

    for i in range(fiSim.size):
        Fe_eff[i][:] = feSim # we try it with this scaling
        e=1 # we start at fe=!0
        while (e<JUMP.size):
            vec = np.zeros(SEED)
            vec[0]= single_experiment(t,\
                Fe_eff[i][e]*(1-params['gei'])*params['pconnec']*params['Ntot'],
                fiSim[i]*params['gei']*params['pconnec']*params['Ntot'], params, seed=0)

            if (vec[0]>JUMP[e-1]): # if we make a too big jump
                # we redo it until the jump is ok (so by a small rescaling of fe)
                # we divide the step by 2
                Fe_eff[i][e] = (Fe_eff[i][e]-Fe_eff[i][e-1])/2.+Fe_eff[i][e-1]
                if verbose:
                    print("we rescale the fe vector [...]")
                # now we can re-enter the loop as the same e than entering..
            else: # we can run the rest
                if verbose:
                    print("== the excitation level :", e+1," over ",feSim.size)
                    print("== ---- the inhibition level :", i+1," over ",fiSim.size)
                for seed in range(1,SEED):
                    params['seed'] = seed
                    vec[seed] = single_experiment(t,\
                            Fe_eff[i][e]*(1-params['gei'])*params['pconnec']*params['Ntot'],\
                            fiSim[i]*params['gei']*params['pconnec']*params['Ntot'], params, seed=seed)
                    if verbose:
                        print("== ---- _____________ seed :",seed)
                MEANfreq[i][e] = vec.mean()
                SDfreq[i][e] = vec.std()
                if verbose:
                    print("== ---- ===> Fout :",MEANfreq[i][e])
                if e<feSim.size-1: # we set the next value to the next one...
                    Fe_eff[i][e+1] = Fe_eff[i][e]+dFexc
                e = e+1 # and we progress in the fe loop
                
        # now we really finish the fe loop

    # then we save the results
    ############################################################################
    # NOTE: this has been changed compared to the original code,
    # we do not want to save the data, we just return them
    return [MEANfreq, SDfreq, Fe_eff, fiSim, params]

    # np.save(filename, np.array([MEANfreq, SDfreq, Fe_eff, fiSim, params], dtype=object))
    # print('numerical TF data saved in :', filename)
    ############################################################################


################################################################################
# follows Simulator class which was not in the original repository, 
# but is what we will use to generate data in our pipeline, and which calls the above code


class Zerlaut2018Simulator(BaseNeuronSimulator):

    def get_connectivity_and_synapses_matrix(self, network_params, si_units=False):
        # This is the implementation of the get_connectivity_and_synapses_matrix function from the original code
        # here we derive the matrix from the parameters in the expected format

        # TODO: make it use the network_params!!!!!
        # NOTE: to implement it I have to pass network_params to the simulate function, 
        # which is not the case in the current design, so I will have to change 
        # the design a bit, or just pass it as part of neuron_sim_params for now, 
        # and then change it later when we will have the network simulation module ready
        number = len(network_params.internal_neurons)

        exc_neuron_name = network_params.exc_neuron_name
        inh_neuron_name = network_params.inh_neuron_name

        conn_matrix = np.empty((number, number), dtype=object)
        
        if network_params.network.connectivity[exc_neuron_name][exc_neuron_name] != network_params.network.connectivity[inh_neuron_name][exc_neuron_name]:
            print("WARNING: the connectivity from excitatory to excitatory and from excitatory to inhibitory neurons are different, which is not supported by the current implementation, using the one from excitatory to excitatory neurons")

        if network_params.network.connectivity[exc_neuron_name][inh_neuron_name] != network_params.network.connectivity[inh_neuron_name][inh_neuron_name]:
            print("WARNING: the connectivity from inhibitory to excitatory and from inhibitory to inhibitory neurons are different, which is not supported by the current implementation, using the one from inhibitory to excitatory neurons")

        exc_pop = {
            **translate_params(
                network_params.neurons[exc_neuron_name].neuron_params,
                {
                    'Tsyn': TranslationRule("tau_syn_E", sim_unit="ms"),
                    'Erev': TranslationRule("e_rev_E", sim_unit="mV"),
                }),
            **translate_params(
                network_params.synapses[exc_neuron_name].syn_params,
                {'Q': TranslationRule("weight", sim_unit="nS")}),
            "p_conn" : network_params.network.connectivity[exc_neuron_name][exc_neuron_name]
        }

        inh_pop = {
            **translate_params(
                network_params.neurons[inh_neuron_name].neuron_params,
                {
                    'Tsyn': TranslationRule("tau_syn_I", sim_unit="ms"),
                    'Erev': TranslationRule("e_rev_I", sim_unit="mV"),
                }),
            **translate_params(
                network_params.synapses[inh_neuron_name].syn_params,
                {'Q': TranslationRule("weight", sim_unit="nS")}),
            "p_conn" : network_params.network.connectivity[inh_neuron_name][inh_neuron_name]
        }

        conn_matrix[:,0] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : exc
        conn_matrix[:,1] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : inh
        conn_matrix[0,0]['name'], conn_matrix[1,0]['name'] = 'ee', 'ie'
        conn_matrix[0,1]['name'], conn_matrix[1,1]['name'] = 'ei', 'ii'
        
        # in the first element we put the network number and connectivity information
        conn_matrix[0,0]['Ntot'], conn_matrix[0,0]['gei'] = network_params.internal_size, network_params.g
        conn_matrix[0,0]['ext_drive'] = 4. # we also store here the choosen excitatory drive 
        conn_matrix[0,0]['afferent_exc_fraction'] = 1. # we also store here the choosen excitatory drive 
    
        if si_units:
            print('synaptic network parameters in SI units')
            for m in conn_matrix.flatten():
                m['Q'] *= 1e-9
                m['Erev'] *= 1e-3
                m['Tsyn'] *= 1e-3
        else:
            print('synaptic network parameters --NOT-- in SI units')
        return conn_matrix

    def get_neuron_params(self, single_neuron_params, name='', number=1, si_units=False):

        if isinstance(single_neuron_params, BaseModel):
            params = {
                'name': name,
                'N': number,
                **translate_params(single_neuron_params, ZERLAUT2018_ADEX_MAPPING)
            }
        elif isinstance(single_neuron_params, str):
            if single_neuron_params=='LIF':
                params = {'name':name, 'N':number,\
                        'Gl':10., 'Cm':150.,'Trefrac':5.,\
                        'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':0.,\
                        'a':0., 'b': 0., 'tauw':1e9}
            elif single_neuron_params=='EIF':
                params = {'name':name, 'N':number,\
                        'Gl':10., 'Cm':150.,'Trefrac':5.,\
                        'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':2.,\
                        'a':0., 'b':0., 'tauw':1e9}
            elif single_neuron_params=='AdExp':
                params = {'name':name, 'N':number,\
                        'Gl':10., 'Cm':150.,'Trefrac':5.,\
                        'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':2.,\
                        'a':4., 'b':20., 'tauw':500.}
            elif single_neuron_params=='FS-cell':
                params = {'name':name, 'N':number,\
                        'Gl':10., 'Cm':200.,'Trefrac':5.,\
                        'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':0.5,\
                        'a':0., 'b': 0., 'tauw':1e9}
            elif single_neuron_params=='RS-cell':
                params = {'name':name, 'N':number,\
                        'Gl':10., 'Cm':200.,'Trefrac':5.,\
                        'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                        'a':4., 'b':20., 'tauw':500.}
            elif single_neuron_params=='RS-cell2':
                params = {'name':name, 'N':number,\
                        'Gl':10., 'Cm':200.,'Trefrac':5.,\
                        'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                        'a':0., 'b':0., 'tauw':500.}
        else:
            print('====================================================')
            print('------------ CELL NOT RECOGNIZED !! ---------------')
            print('====================================================')


        if si_units:
            print('cell parameters in SI units')
            # mV to V
            params['El'], params['Vthre'], params['Vreset'], params['delta_v'] =\
                1e-3*params['El'], 1e-3*params['Vthre'], 1e-3*params['Vreset'], 1e-3*params['delta_v']
            # ms to s
            params['Trefrac'], params['tauw'] = 1e-3*params['Trefrac'], 1e-3*params['tauw']
            # nS to S
            params['a'], params['Gl'] = 1e-9*params['a'], 1e-9*params['Gl']
            # pF to F and pA to A
            params['Cm'], params['b'] = 1e-12*params['Cm'], 1e-12*params['b']
        else:
            print('cell parameters --NOT-- in SI units')
        
        return params.copy()

    def simulate(self, network_params: dict, neuron_sim_params: NeuronSimulationConfig) -> dict:
        """Routes to the correct PyNN execution method based on neuron_sim_params.
        
        Parameters
        ----------
        network_params : dict
            Dictionary of parameters for the network models.
            items are: (neuron_name, dict of neuron parameters, synapses etc.)
        neuron_sim_params : NeuronSimulationConfig
            Configuration object containing grid specifications and other simulation parameters.
        results_path : str
            Path to the directory where simulation results will be saved.

        Returns
        -------
        results : dict
            Dictionary containing the simulation results for each neuron type.

        """


        results = {}
        # possible Network_Model are 'Vogels-Abbott' and 'CONFIG1'

        for neuron_name in network_params.internal_neurons:
            single_neuron_params = network_params.neurons[neuron_name].neuron_params
            
            
            conn_matrix = self.get_connectivity_and_synapses_matrix(network_params, si_units=True)
            params = self.get_neuron_params(single_neuron_params, name=neuron_name, si_units=True)
            reformat_syn_parameters(params, conn_matrix) # merging those parameters
    
            grid_params = getattr(neuron_sim_params.grid, neuron_name)
            if grid_params.grid_type == "adaptive":
                if grid_params.exc_rate_grid != "adaptive":
                    raise ValueError("For adaptive grid, exc_rate_grid must be set to 'adaptive'")
                
                file_name = Path('data').resolve() / f"{neuron_name}_zerlaut2018simulator_tmp.npy"
                zerlaut_results = generate_transfer_function(
                    params,
                    MAXfexc = grid_params.out_rate_grid[1],  # Maximum excitatory frequency (default=30.)
                    MAXfinh = grid_params.inh_rate_grid[1],  # Limits for inhibitory frequency (default=[1.,20.])
                    MINfinh = grid_params.inh_rate_grid[0],  # Limits for inhibitory frequency (default=[1.,20.])
                    discret_exc = int(grid_params.out_rate_grid[2]),  # Discretization (number of points) of excitatory frequencies (default=9)
                    discret_inh = int(grid_params.inh_rate_grid[2]),  # Discretization (number of points) of inhibitory frequencies (default=8)
                    MAXfout = grid_params.out_rate_grid[1],  # Maximum output frequency (default=30.)
                    SEED = neuron_sim_params.seed,
                    verbose = False,
                    filename = file_name,
                    dt = neuron_sim_params.time_step/1000.,  # converting ms to s (default 5e-5 s)
                    tstop = neuron_sim_params.simulation_time/1000. # converting ms to s (default 10 s)
                )

                results[neuron_name] = SingleNeuronResults(
                    simulator_name='Zerlaut2018Simulator',
                    neuron_name=neuron_name,
                    neuron_params=single_neuron_params,
                    neuron_sim_params=neuron_sim_params,
                    exc_rate_grid=zerlaut_results[2].T,  # Excitatory input (Hz)
                    inh_rate_grid=np.stack([zerlaut_results[3]]*10, axis=1).T,  # Inhibitory input (Hz)
                    out_rate_mean=zerlaut_results[0].T,  # Mean output firing rate (Hz)
                    out_rate_std=zerlaut_results[1].T,  # STD firing rate (Hz)
                )
                continue

            if grid_params.grid_type == "linear":
                exc_rate_min, exc_rate_max, exc_n_points = grid_params.exc_rate_grid
                inh_rate_min, inh_rate_max, inh_n_points = grid_params.inh_rate_grid

                exc_n_points, inh_n_points = int(exc_n_points), int(inh_n_points)

                exc_rate_grid = np.linspace(exc_rate_min, exc_rate_max, exc_n_points)
                inh_rate_grid = np.linspace(inh_rate_min, inh_rate_max, inh_n_points)

                exc_rate_grid, inh_rate_grid = np.meshgrid(exc_rate_grid, inh_rate_grid, sparse=False, indexing='ij')
            elif grid_params.grid_type == "custom":
                exc_rate_grid = grid_params.exc_rate_grid
                inh_rate_grid = grid_params.inh_rate_grid

                exc_n_points, inh_n_points = exc_rate_grid.shape
            else:
                raise ValueError(f"Unknown grid type: {grid_params.grid_type}")


            n_runs = int(neuron_sim_params.n_runs)
            out_rate = np.zeros((exc_n_points, inh_n_points, n_runs))
            for inh_idx in range(inh_n_points):
                for exc_idx in range(exc_n_points):
                    exc_rate = exc_rate_grid[exc_idx, inh_idx]
                    inh_rate = inh_rate_grid[exc_idx, inh_idx]

                    for n_run in range(n_runs):
                        out_rate[exc_idx, inh_idx, n_run] = single_experiment(
                            t=np.arange(0, neuron_sim_params.simulation_time/1000., neuron_sim_params.time_step/1000.),  # converting ms to s
                            fe=exc_rate*(1-params['gei'])*params['pconnec']*params['Ntot'],
                            fi=inh_rate*params['gei']*params['pconnec']*params['Ntot'],
                            params=params,
                            seed=neuron_sim_params.seed + n_run
                        )

            results[neuron_name] = SingleNeuronResults(
                simulator_name='Zerlaut2018Simulator',
                neuron_name=neuron_name,
                neuron_params=single_neuron_params,
                neuron_sim_params=neuron_sim_params,
                exc_rate_grid=exc_rate_grid,
                inh_rate_grid=inh_rate_grid,
                out_rate_mean=out_rate.mean(axis=2),
                out_rate_std=out_rate.std(axis=2),
            )

        return results


################################################################################



if __name__=='__main__':

    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ Runs two types of protocols on a given neuronal and network model
        1)  ==> Preliminary transfer function protocol ===
           to find the fixed point (with possibility to add external drive)
        2)  =====> Full transfer function protocol ==== 
           i.e. scanning the (fe,fi) space and getting the output frequency""",
              formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("Neuron_Model",help="Choose a neuronal model from 'neuronal_models.py'")
    parser.add_argument("Network_Model",help="Choose a network model (synaptic and connectivity properties)"+\
                        "\n      from 'network_models'.py")

    parser.add_argument("--max_Fe",type=float, default=30.,\
                        help="Maximum excitatory frequency (default=30.)")
    parser.add_argument("--discret_Fe",type=int, default=10,\
                        help="Discretization of excitatory frequencies (default=9)")
    parser.add_argument("--lim_Fi", type=float, nargs=2, default=[0.,20.],\
                help="Limits for inhibitory frequency (default=[1.,20.])")
    parser.add_argument("--discret_Fi",type=int, default=8,\
               help="Discretization of inhibitory frequencies (default=8)")
    parser.add_argument("--max_Fout",type=float, default=30.,\
                         help="Minimum inhibitory frequency (default=30.)")
    parser.add_argument("--tstop",type=float, default=10.,\
                         help="tstop in s")
    parser.add_argument("--dt",type=float, default=5e-5,\
                         help="dt in ms")
    parser.add_argument("--SEED",type=int, default=1,\
                  help="Seed for random number generation (default=1)")

    parser.add_argument("-s", "--save", help="save with the right name",
                         action="store_true")
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                         action="store_true")

    args = parser.parse_args()

    params = get_neuron_params(args.Neuron_Model, SI_units=True)
    conn_matrix = get_connectivity_and_synapses_matrix(args.Network_Model, SI_units=True)

    reformat_syn_parameters(params, conn_matrix) # merging those parameters

    if args.save:
        FILE = 'data/'+args.Neuron_Model+'_'+args.Network_Model+'.npy'
    else:
        FILE = 'data/example_data.npy'
        
    generate_transfer_function(params,\
                               verbose=args.verbose,
                               MAXfexc=args.max_Fe, 
                               MINfinh=args.lim_Fi[0], MAXfinh=args.lim_Fi[1],\
                               discret_exc=args.discret_Fe,discret_inh=args.discret_Fi,\
                               filename=FILE,
                               dt=args.dt, tstop=args.tstop,
                               MAXfout=args.max_Fout, SEED=args.SEED)
