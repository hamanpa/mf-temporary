# Author:  Kusch Lionel
# Contact: lionel.kusch@grenoble-inp.org
""" Mean Ad Ex with extra variables """

import numpy as np
from  tvb_models.models import Zerlaut_adaptation_second_order #tvb.simulator.models.zerlaut import ZerlautAdaptationSecondOrder
import numpy
from tvb.basic.neotraits.api import List, Final


class ZerlautAdaptationSecondOrder_Record(Zerlaut_adaptation_second_order):
    #  Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 0.0]), # actually the 100Hz should be replaced by 1/T_refrac
                 "I": numpy.array([0.0, 0.0]),
                 "C_ee": numpy.array([0.0, 0.0]),  # variance is positive or null
                 "C_ei": numpy.array([0.0, 0.0]),  # the co-variance is in [-c_ee*c_ii,c_ee*c_ii]
                 "C_ii": numpy.array([0.0, 0.0]),  # variance is positive or null
                 "W_e":numpy.array([0.0, 0.0]),
                 "W_i":numpy.array([0.0, 0.0]),
                 "noise":numpy.array([0.0, 0.0]),
                 "ex_f_e": numpy.array([0.0, 0.0]),
                 "ex_f_i": numpy.array([0.0, 0.0]),
                 "ex_mu_V": numpy.array([0.0, 0.0]),
                 "ex_mu_Ge": numpy.array([0.0, 0.0]),
                 "ex_mu_Gi": numpy.array([0.0, 0.0]),
                 "in_f_e": numpy.array([0.0, 0.0]),
                 "in_f_i": numpy.array([0.0, 0.0]),
                 "in_mu_V": numpy.array([0.0, 0.0]),
                 "in_mu_Ge": numpy.array([0.0, 0.0]),
                 "in_mu_Gi": numpy.array([0.0, 0.0])
                 },
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.\n
        E: firing rate of excitatory population in KHz\n
        I: firing rate of inhibitory population in KHz\n
        C_ee: the variance of the excitatory population activity \n
        C_ei: the covariance between the excitatory and inhibitory population activities (always symetric) \n
        C_ie: the variance of the inhibitory population activity \n
        W: level of adaptation
        ex_f_e: mean firing rate excitatory input of excitatory neurons 
        ex_f_i:mean firing rate inhibitory input of excitatory neurons 
        ex_mu_V: mean voltage of excitatory neurons 
        ex_mu_Ge: mean conductance of excitatory synapses to excitatory neurons
        ex_mu_Gi: mean conductance of inhibitory synapses to excitatory neurons
        in_f_e: mean firing rate excitatory input of inhibitory neurons 
        in_f_i:mean firing rate inhibitory input of inhibitory neurons 
        in_mu_V: mean voltage of inhibitory neurons 
        in_mu_Ge: mean conductance of excitatory synapses to inhibitory neurons
        in_mu_Gi: mean conductance of inhibitory synapses to inhibitory neurons

        """)
    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(
        "E", "I", "C_ee","C_ei","C_ii","W_e", "W_i","noise", "ex_f_e", "ex_f_i", "ex_mu_V", "ex_mu_Ge", "ex_mu_Gi", "in_f_e",
        "in_f_i", "in_mu_V", "in_mu_Ge", "in_mu_Gi"),
        default=(
        "E", "I", "C_ee","C_ei","C_ii","W_e", "W_i","noise", "ex_f_e", "ex_f_i", "ex_mu_V", "ex_mu_Ge", "ex_mu_Gi", "in_f_e",
        "in_f_i", "in_mu_V", "in_mu_Ge", "in_mu_Gi"),
        doc="""This represents the default state-variables of this Model to be
               monitored. It can be overridden for each Monitor if desired. The
               corresponding state-variable indices for this model are :math:`E = 0`,
               :math:`I = 1`, :math:`C_ee = 2`, :math:`C_ei = 3`, :math:`C_ii = 4` and :math:`W = 5`.""")

    state_variables = 'E I C_ee C_ei C_ii W_e W_i noise ex_f_e ex_f_i ex_mu_V ex_mu_Ge ex_mu_Gi in_f_e in_f_i in_mu_V in_mu_Ge in_mu_Gi'.split()
    non_integrated_variables = 'ex_f_e ex_f_i ex_mu_V ex_mu_Ge ex_mu_Gi in_f_e in_f_i in_mu_V in_mu_Ge in_mu_Gi'.split()
    _nvar = 18

    def get_MF_variables(super, Fe, Fi, Fe_ext, Fi_ext, W, Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L, N_tot,
                         p_connect_i, p_connect_e, g, K_ext_e, K_ext_i):
        """
        Compute the mean characteristic of neurons.
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param Fe: firing rate of excitatory population
        :param Fi: firing rate of inhibitory population
        :param Fe_ext: external excitatory input
        :param Fi_ext: external inhibitory input
        :param W: level of adaptation
        :param Q_e: excitatory quantal conductance
        :param tau_e: excitatory decay
        :param E_e: excitatory reversal potential
        :param Q_i: inhibitory quantal conductance
        :param tau_i: inhibitory decay
        :param E_i: inhibitory reversal potential
        :param E_L: leakage reversal voltage of neurons
        :param g_L: leak conductance
        :param C_m: membrane capacitance
        :param E_L: leak reversal potential
        :param N_tot: cell number
        :param p_connect: connectivity probability
        :param g: fraction of inhibitory cells
        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # firing rate
        # 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
        fe = (Fe + 1.0e-6) * (1. - g) *  p_connect_e * N_tot + Fe_ext * K_ext_e
        fi = (Fi + 1.0e-6) * g * p_connect_i * N_tot + Fi_ext * K_ext_i

        # conductance fluctuation and effective membrane time constant
        mu_Ge, mu_Gi = Q_e * tau_e * fe, Q_i * tau_i * fi  # Eqns 5 from [MV_2018]
        mu_G = g_L + mu_Ge + mu_Gi  # Eqns 6 from [MV_2018]

        # membrane potential
        mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G  # Eqns 7 from [MV_2018]

        return fe, fi, mu_V, mu_Ge, mu_Gi

    def update_state_variables_before_integration(self, state_variables, coupling, local_coupling=0.0, stimulus=0.0):
        E = state_variables[0, :]
        I = state_variables[1, :]
        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_E = local_coupling * E
        lc_I = local_coupling * I

        noise = state_variables[7,:]	

        # external firing rate for the different population
        E_input_excitatory = c_0+lc_E+self.external_input_ex_ex #+ self.weight_noise * noise
        index_bad_input = numpy.where( E_input_excitatory < 0)
        E_input_excitatory[index_bad_input] = 0.0
        E_input_inhibitory = c_0+lc_E+self.external_input_in_ex #+ self.weight_noise * noise
        index_bad_input = numpy.where( E_input_inhibitory < 0)
        E_input_inhibitory[index_bad_input] = 0.0
        I_input_excitatory = lc_I+self.external_input_ex_in
        I_input_inhibitory = lc_I+self.external_input_in_in
        W_e = state_variables[5, :]
        W_i = state_variables[6, :]


        # extra variable for excitatory neurons
        state_variables[8:8 + 5] = self.get_MF_variables(
            E, I,
            E_input_excitatory,
            E_input_inhibitory,
            W_e, self.Q_e, self.tau_e, self.E_e,
            self.Q_i, self.tau_i, self.E_i,
            self.g_L, self.C_m, self.E_L_e, self.N_tot,
            self.p_connect_i, self.p_connect_e, self.g, self.K_ext_e, self.K_ext_i)

        # extra variable for inhibitory neurons
        state_variables[8 + 5:] = self.get_MF_variables(
            E, I,
            I_input_excitatory,
            I_input_inhibitory,
            W_i, self.Q_e, self.tau_e, self.E_e,
            self.Q_i, self.tau_i, self.E_i,
            self.g_L, self.C_m, self.E_L_i, self.N_tot,
            self.p_connect_i, self.p_connect_e, self.g, self.K_ext_e, self.K_ext_i)
        # self.tmp = np.copy(state_variables[7:])
        return state_variables

    # def dfun(self, state_variables, coupling, local_coupling=0.00):
    #     derivative = super().dfun(state_variables[:7],coupling,local_coupling)
    #     return np.concatenate((derivative,numpy.zeros_like(state_variables[7:,:])))

    # def update_state_variables_after_integration(self, state_variables):
    #     state_variables[7:] = self.tmp
    #     return state_variables

