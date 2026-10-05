import numpy as np
from pathlib import Path
from typing import Any

from ..base import BaseMFSimulator
from ..config import MeanFieldSimulationConfig
from ...network_params.models import BiologicalParameters, INTERNAL_PROJECTIONS
from ...stimuli.config import BaseStimulusConfig
from .models.factory import setup_tvb_model, MODEL_STP_MODES
from .stimuli import prepare_stimulus
from ...utils.array_helpers import convert_to_array
from ...utils.stp_helpers import calculate_steady_state_stp_variables
from ...data_structures.mf_simulation import MFResults


from tvb.simulator.simulator import Simulator
from tvb.simulator.coupling import Linear
from tvb.simulator.noise import Additive
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion




class TVBMFSimulator(BaseMFSimulator):
    """
    Adapter class that wraps the legacy TVB Nuu Tools and TVBParameterClass 
    to fit the universal BaseMFSimulator interface.
    """

    def __init__(self):
        self.network_params = None
        self.mf_sim_params = None
        
        self.model = None
        self.connectivity = None
        self.coupling = None
        self.integrator = None
        self.monitors = None
        self.stimulus = None

        # NOTE: each node is accommodated with a single MF simulator
        # At the moment we hardcode a single node
        self.grid_size = 1
        self.num_nodes = self.grid_size ** 2

    def build_network(self, network_params: BiologicalParameters, mf_sim_params: MeanFieldSimulationConfig) -> None:
        """
        Prepares the static TVB parameters and connectivity matrix.
        """

        self.network_params = network_params
        self.mf_sim_params = mf_sim_params

        
        # NOTE: 
        # model is the single node MF
        # connectivity is the connection between nodes
        # coupling is
        # integrator is
        # monitor is
        

        self.model = setup_tvb_model(network_params, mf_sim_params)

        self.setup_connectivity()
        self.setup_coupling()
        self.setup_integrator()
        self.setup_monitors()


    def setup_connectivity(self) -> None:

        assert self.grid_size == 1, "Currently we only support a single node (grid_size=1) for the TVB simulator. Please set grid_size=1 in the configuration."


        self.connectivity = Connectivity(
            weights=np.array([[0.0]]), 
            tract_lengths=np.array([[0.0]]),
            region_labels=np.array([], dtype=np.dtype('<U128')),
            centres=np.array([]),
            cortical=None
        )
        self.connectivity.speed = convert_to_array(4.0)

    def _create_gaussian_connection_matrix(self, nodes=1, amplitude=0.0001, sigma=1):
        """Legacy helper copied directly from __init__.py"""
        W_matrix = np.zeros((nodes ** 2, nodes ** 2))
        tract_lengths = np.zeros((nodes ** 2, nodes ** 2))
        coords = np.zeros((nodes**2, 2))

        for i in range(nodes**2):
            row, col = divmod(i,nodes)
            coords[i] = [row, col]

        for i in range(0, len(W_matrix)):
            for j in range(0, len(W_matrix)):
                tract_lengths[i, j] = np.sqrt(((coords[i]-coords[j])**2).sum())

        for i in range(0, len(W_matrix)):
            for j in range(0, len(W_matrix)):
                W_matrix[j, i] = amplitude * np.exp(-0.5 * (tract_lengths[i, j] / sigma) ** 2)

        for i in range(0, len(W_matrix)):
            W_matrix[i, i] = 0

        return W_matrix, tract_lengths

    def setup_coupling(self) -> None:
        self.coupling = Linear(
            a=convert_to_array(0.3), 
            b=convert_to_array(0.0)
        )

    def setup_integrator(self) -> None:
        noise = Additive(
            nsig=np.array([(var=="noise")*1.0 for var in self.model.state_variables]),
            ntau=0.0,
        )
        noise.random_stream.seed(self.mf_sim_params.seed)

        self.integrator = HeunStochastic(
            noise=noise,
            dt=self.mf_sim_params.time_step
        )

    def setup_monitors(self) -> None:
        self.monitors = []
        self.monitors.append(Raw())        


        # parameter_monitor= {
        #     'Raw':True,
        #     'TemporalAverage':False,
        #     'parameter_TemporalAverage':{
        #         'variables_of_interest':list(range(len(svars))),
        #         'period':self.mf_sim_params.time_step*10.0
        #     },
        #     'Bold':False,
        #     'parameter_Bold':{
        #         'variables_of_interest':[0],
        #         'period':self.parameter_integrator['dt']*2000.0
        #     },
        #     'Ca':False,
        #     'parameter_Ca':{
        #         'variables_of_interest':[0,1,2],
        #         'tau_rise':0.01,
        #         'tau_decay':0.1
        #     }
        # }
        # pass


    def run_stimulus(self, stim_params: BaseStimulusConfig) -> MFResults:
        
        self.setup_stimulus(stim_params)
        

        print(f"Booting TVB engine...")
        sim = Simulator(
            model=self.model,
            connectivity=self.connectivity,
            coupling=self.coupling,
            integrator=self.integrator,
            monitors=self.monitors,
            stimulus=self.stimulus
        )
        sim.configure()
        
        # 3. Execute
        duration = stim_params.simulation_duration
        print(f"Integrating for {duration} ms...")
        
        # TODO: monitor part in progress, for now only raw monitor

        times, results_raw = [], []
        for result in sim(simulation_length=duration):
            # TVB returns a generator that yields results at each time step; 
            # we will need to accumulate these results and then map them to the MFResults structure.

            # results in in the shape [[time, np.array(state_vars)]]
            # probably first index is monitor index, second index labels time and state variables

            time = result[0][0]  
            data = result[0][1]
            if data.shape != (len(self.model.state_variables), 1, 1):
                # shape is (len(state_vars), 1, 1)  (I expect the ones are for nodes and trials, which we do not have at the moment)
                raise ValueError(f"Unexpected data shape from TVB: {data.shape}. Expected ({len(self.model.state_variables)}, 1, 1).")
            else:
                data = data.flatten()  # shape becomes (len(state_vars),)

            times.append(time)
            results_raw.append(data)

        times = np.array(times)
        results_raw = np.array(results_raw)

        keys = self.model.state_variables
        results_dict = {key: results_raw[:, i] for i, key in enumerate(keys)}

        # First-order models have no covariances (C_ee, C_ii, C_ei)
        exc_rate_var = results_dict.get("C_ee")
        inh_rate_var = results_dict.get("C_ii")

        result = MFResults(
            label_name = "MFResults",
            mf_sim_params = self.mf_sim_params,
            network_params = self.network_params,
            stim_name = "test",
            stim_params = stim_params,
            times = times,
            exc_rate_mean = results_dict.get("E", None),
            exc_rate_std = None if exc_rate_var is None else np.sqrt(exc_rate_var),
            inh_rate_mean = results_dict.get("I", None),
            inh_rate_std = None if inh_rate_var is None else np.sqrt(inh_rate_var),
            stim_rate_mean = results_dict.get("stimulus", None),
            drive_rate_mean = np.ones_like(times.astype(float))*stim_params.drive_rate,
            exc_adaptation_mean = results_dict.get("W_e", None),
            inh_adaptation_mean = results_dict.get("W_i", None),
            rate_cov = results_dict.get("C_ei", None),
            stp_means = self._stp_means(results_dict),
            input_units = {
                "times" : "ms",
                "exc_rate_mean" : "kHz",
                "exc_rate_std" : "kHz",
                "inh_rate_mean" : "kHz",
                "inh_rate_std" : "kHz",
                "stim_rate_mean" : "kHz",
                "drive_rate_mean" : "Hz",  # this is not typo! we compute drive_rate_mean above directly in MFT units, not TVB units
                "exc_adaptation_mean" : "pA",
                "inh_adaptation_mean" : "pA",
                "rate_cov" : "kHz^2",  # TVB C_ei is in kHz^2 (see TVB_STATE_VARIABLES_MAPPING)
            },
        )

        return result

    def _stp_means(self, results_dict: dict) -> dict[str, np.ndarray]:
        """
        STP time courses per internal projection ("ee_x", "ee_u", ...), consistent with how the
        current model treats the synapses (see MODEL_STP_MODES):

        - dynamic:       x = X, y = Y, u = the model's utilisation (U if tau_fac == 0, else U_dyn*(1-U) + U);
                         the efficacy factor used by the model is u*x
        - asymptotic:    steady-state x, u at the source rate (y not modelled)
        - static_weight: x = 1, u = U of the projection onto E from the same source (legacy models use
                         static weights weight*U and the projections onto E for both targets)
        """
        model_name = getattr(self.mf_sim_params.model, "value", self.mf_sim_params.model)
        mode = MODEL_STP_MODES[model_name]
        network = self.network_params
        exc_name = network.exc_neuron_name
        rates_hz = {
            exc_name: results_dict["E"] * 1e3,  # TVB rates are in kHz
            network.inh_neuron_name: results_dict["I"] * 1e3,
        }

        stp_means = {}
        for projection in INTERNAL_PROJECTIONS:
            _, source_name = network.projection_populations(projection)
            if mode == "static_weight":
                conn = network.network.connectivity[exc_name].get(source_name)
            else:
                conn = network.connection(projection)
            if conn is None:
                continue

            U = getattr(conn.syn_params, "U", 1.0)
            if mode == "dynamic":
                stp_means[f"{projection}_x"] = results_dict[f"X_{projection}"]
                stp_means[f"{projection}_y"] = results_dict[f"Y_{projection}"]
                stp_means[f"{projection}_u"] = self.model._utilization(
                    results_dict[f"U_dyn_{projection}"], U, getattr(conn.syn_params, "tau_fac", 0.0)
                )
            elif mode == "asymptotic":
                x_steady, u_steady = calculate_steady_state_stp_variables(
                    rate=rates_hz[source_name],
                    u=U,
                    tau_rec=getattr(conn.syn_params, "tau_rec", 0.0),
                    tau_fac=getattr(conn.syn_params, "tau_fac", 0.0),
                )
                stp_means[f"{projection}_x"] = x_steady
                stp_means[f"{projection}_u"] = u_steady
            elif mode == "static_weight":
                stp_means[f"{projection}_x"] = np.ones_like(rates_hz[source_name])
                stp_means[f"{projection}_u"] = np.full_like(rates_hz[source_name], U)
            else:
                raise ValueError(f"Unknown STP mode '{mode}' for model '{model_name}'.")

        return stp_means

    def setup_stimulus(self, stim_params: BaseStimulusConfig) -> None:


        tvb_stimulus = prepare_stimulus(stim_params)

        if stim_params.direct_stimulation:
            variables = [0]  # Assuming the first variable is the one to stimulate; adjust as needed
        else:
            variables = [self.model.state_variables.index("stimulus")]

        weight = list(np.zeros(self.num_nodes))
        # TODO: in the future make this part dynamic (with initial period by making another state variable)
        # Not sure why the weight need this weird unit!!!!!!!!
        # HACK
        weight[stim_params.target_nodes] = 10.

        parameter_stimulus = {}
        parameter_stimulus['eqn_t'] = tvb_stimulus
        parameter_stimulus["variables"] = variables     # index of the variable to which the stimulus is applied
        parameter_stimulus["weights"]= weight
        parameter_stimulus['name'] = stim_params.pattern

        # TODO: in the future make this part dynamic (with initial period by making another state variable)
        # NOTE: drive by default goes to all internal populations
        # stimulus may go just to one population!


        # DO ONE THING AT A MOMENT!!!
        # More project managment approach
        # do not try to factorize and improve at the same time!
        # first make it work, then make it better!

        self.model.external_input_ex_ex = convert_to_array(stim_params.drive_rate)*1e-3
        self.model.external_input_in_ex = convert_to_array(stim_params.drive_rate)*1e-3
        self.model.external_input_ex_in = convert_to_array(0.0)
        self.model.external_input_in_in = convert_to_array(0.0)

        self.model.stim_target_ratio = convert_to_array(stim_params.stim_target_ratio)

        self.stimulus = StimuliRegion(
            temporal=tvb_stimulus,
            connectivity=self.connectivity,
            weight=np.array(weight)
        )
        self.model.stvar = variables


    def end(self) -> None:
        """Tear down the simulation environment."""
        self.model = None
        self.connectivity = None
        self.coupling = None
        self.integrator = None
        self.monitors = None
        self.stimulus = None

