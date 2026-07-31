from tvb.simulator.models.base import Model, numpy
from tvb.basic.neotraits.api import NArray, Range, Final, List
import scipy.special as sp_spec
from numba import jit


class BaseNeuroPSI_STP(Model):
    r"""
    Base class containing common traited parameters, static transfer functions,
    and helper routines for NeuroPSI Short-Term Plasticity (STP) Mean-Field models.
    """

    # Traited attributes representing model parameters
    g_L_e = NArray(
        label=":math:`g_{L}`",
        default=numpy.array([10.]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="""leak conductance [nS] of excitatory neuron"""
    )

    g_L_i = NArray(
        label=":math:`g_{L}`",
        default=numpy.array([10.]),
        domain=Range(lo=0.1, hi=100.0, step=0.1),
        doc="""leak conductance [nS] of inhibitory neuron"""
    )

    E_L_e = NArray(
        label=":math:`E_{L}`",
        default=numpy.array([-65.0]),
        domain=Range(lo=-90.0, hi=-60.0, step=0.1),
        doc="""leak reversal potential for excitatory [mV]"""
    )

    E_L_i = NArray(
        label=":math:`E_{L}`",
        default=numpy.array([-65.0]),
        domain=Range(lo=-90.0, hi=-60.0, step=0.1),
        doc="""leak reversal potential for inhibitory [mV]"""
    )

    C_m_e = NArray(
        label=":math:`C_{m}`",
        default=numpy.array([200.0]),
        domain=Range(lo=10.0, hi=500.0, step=10.0),
        doc="""membrane capacitance [pF] of excitatory neuron"""
    )

    C_m_i = NArray(
        label=":math:`C_{m}`",
        default=numpy.array([200.0]),
        domain=Range(lo=10.0, hi=500.0, step=10.0),
        doc="""membrane capacitance [pF] of inhibitory neuron"""
    )

    b_e = NArray(
        label=":math:`Excitatory b`",
        default=numpy.array([60.0]),
        domain=Range(lo=0.0, hi=150.0, step=1.0),
        doc="""Excitatory adaptation current increment [pA]"""
    )

    a_e = NArray(
        label=":math:`Excitatory a`",
        default=numpy.array([4.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.1),
        doc="""Excitatory adaptation conductance [nS]"""
    )

    b_i = NArray(
        label=":math:`Inhibitory b`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=100.0, step=0.1),
        doc="""Inhibitory adaptation current increment [pA]"""
    )

    a_i = NArray(
        label=":math:`Inhibitory a`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.1),
        doc="""Inhibitory adaptation conductance [nS]"""
    )

    tau_w_e = NArray(
        label=":math:`tau_w_e`",
        default=numpy.array([500.0]),
        domain=Range(lo=1.0, hi=1000.0, step=1.0),
        doc="""Adaptation time constant of excitatory neurons [ms]"""
    )

    tau_w_i = NArray(
        label=":math:`tau_w_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=1000.0, step=1.0),
        doc="""Adaptation time constant of inhibitory neurons [ms]"""
    )

    E_e = NArray(
        label=r":math:`E_e`",
        default=numpy.array([0.0]),
        domain=Range(lo=-20., hi=20., step=0.01),
        doc="""excitatory reversal potential [mV]"""
    )

    E_i = NArray(
        label=":math:`E_i`",
        default=numpy.array([-80.0]),
        domain=Range(lo=-100.0, hi=-60.0, step=1.0),
        doc="""inhibitory reversal potential [mV]"""
    )

    tau_e = NArray(
        label=":math:`\tau_e`",
        default=numpy.array([5.0]),
        domain=Range(lo=1.0, hi=10.0, step=1.0),
        doc="""excitatory decay [ms]"""
    )

    tau_i = NArray(
        label=":math:`\tau_i`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.5, hi=10.0, step=0.01),
        doc="""inhibitory decay [ms]"""
    )

    # exc -> exc conns
    K_ee = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of exc -> exc connections"""
    )

    Q_ee = NArray(
        label=r":math:`Q_ee`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS] for exc -> exc connections"""
    )

    U_ee = NArray(
        label=":math:`U_ee`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Synaptic efficacy for exc -> exc connections"""
    )

    tau_rec_ee = NArray(
        label=":math:`\tau_d_ee`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic recovery time constant for exc -> exc connections [ms]"""
    )

    tau_fac_ee = NArray(
        label=":math:`\tau_f_ee`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic facilitation time constant for exc -> exc connections [ms]"""
    )

    # inh -> exc conns
    K_ei = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of inh -> exc connections"""
    )

    Q_ei = NArray(
        label=r":math:`Q_ei`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""inhibitory quantal conductance [nS] for inh -> exc connections"""
    )

    U_ei = NArray(
        label=":math:`U_ei`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Synaptic efficacy for inh -> exc connections"""
    )

    tau_rec_ei = NArray(
        label=":math:`\tau_d_ei`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic recovery time constant for inh -> exc connections [ms]"""
    )

    tau_fac_ei = NArray(
        label=":math:`\tau_f_ei`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic facilitation time constant for inh -> exc connections [ms]"""
    )

    # exc -> inh conns
    K_ie = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of exc -> inh connections"""
    )

    Q_ie = NArray(
        label=r":math:`Q_ie`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS] for exc -> inh connections"""
    )

    U_ie = NArray(
        label=":math:`U_ie`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Synaptic efficacy for exc -> inh connections"""
    )

    tau_rec_ie = NArray(
        label=":math:`\tau_d_ie`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic recovery time constant for exc -> inh connections [ms]"""
    )

    tau_fac_ie = NArray(
        label=":math:`\tau_f_ie`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic facilitation time constant for exc -> inh connections [ms]"""
    )

    # inh -> inh conns
    K_ii = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of inh -> inh connections"""
    )

    Q_ii = NArray(
        label=r":math:`Q_ii`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""inhibitory quantal conductance [nS] for inh -> inh connections"""
    )

    U_ii = NArray(
        label=":math:`U_ii`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Synaptic efficacy for inh -> inh connections"""
    )

    tau_rec_ii = NArray(
        label=":math:`\tau_d_ii`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic recovery time constant for inh -> inh connections [ms]"""
    )

    tau_fac_ii = NArray(
        label=":math:`\tau_f_ii`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="""Synaptic facilitation time constant for inh -> inh connections [ms]"""
    )

    # drive -> exc conns
    K_ed = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of drive -> exc connections"""
    )

    Q_ed = NArray(
        label=r":math:`Q_ed`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS] for drive -> exc connections"""
    )

    # stim -> exc conns
    K_es = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of stim -> exc connections"""
    )

    Q_es = NArray(
        label=r":math:`Q_es`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS] for stim -> exc connections"""
    )

    # drive -> inh conns
    K_id = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of drive -> inh connections"""
    )

    Q_id = NArray(
        label=r":math:`Q_id`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS] for drive -> inh connections"""
    )

    # stim -> inh conns
    K_is = NArray(
        label=":math:`\\epsilon`",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=10000, step=1),
        doc="""Number of stim -> inh connections"""
    )

    Q_is = NArray(
        label=r":math:`Q_is`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=5.0, step=0.1),
        doc="""excitatory quantal conductance [nS] for stim -> inh connections"""
    )

    T = NArray(
        label=":math:`T`",
        default=numpy.array([20.0]),
        domain=Range(lo=1., hi=20.0, step=0.1),
        doc="""time scale of describing network activity"""
    )

    P_e = NArray(
        label=":math:`P_e`",
        default=numpy.array([
            -0.04983106, 0.005063550882777035, -0.023470121807314552,
            0.0022951513725067503, -0.0004105302652029825, 0.010547051343547399,
            -0.03659252821136933, 0.007437487505797858, 0.001265064721846073,
            -0.04072161294490446
        ]),
        doc="""Polynome of excitatory phenomenological threshold (order 9)"""
    )

    P_i = NArray(
        label=":math:`P_i`",
        default=numpy.array([
            -0.05149122024209484, 0.004003689190271077, -0.008352013668528155,
            0.0002414237992765705, -0.0005070645080016026, 0.0014345394104282397,
            -0.014686689498949967, 0.004502706285435741, 0.0028472190352532454,
            -0.015357804594594548
        ]),
        doc="""Polynome of inhibitory phenomenological threshold (order 9)"""
    )

    external_input_ex_ex = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive"""
    )

    external_input_ex_in = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive"""
    )

    external_input_in_ex = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive"""
    )

    external_input_in_in = NArray(
        label=":math:`\nu_e^{drive}`",
        default=numpy.array([0.000]),
        domain=Range(lo=0.00, hi=0.1, step=0.001),
        doc="""external drive"""
    )

    tau_OU = NArray(
        label=":math:`\ntau noise`",
        default=numpy.array([5.0]),
        domain=Range(lo=0.10, hi=10.0, step=0.01),
        doc="""time constant noise"""
    )

    weight_noise = NArray(
        label=":math:`\nweight noise`",
        default=numpy.array([10.5]),
        domain=Range(lo=0., hi=50.0, step=1.0),
        doc="""weight noise"""
    )

    stim_target_ratio = NArray(
        label=":math:`stim_target_ratio`",
        default=numpy.array([1.0]),
        doc="""Ratio of stimulated nodes to total nodes"""
    )

    @staticmethod
    def convert_to_array(lst: list, axis: int = 0):
        shape = lst[0].shape
        new_array = []
        for item in lst:
            if type(item) == numpy.ndarray:
                if item.shape == shape:
                    new_array.append(item)
                else:
                    assert item.size == 1
                    new_array.append(item.reshape(shape))
            else:
                new_array.append(numpy.ones(shape) * item)
        new_array = numpy.stack(new_array, axis=axis)
        return new_array

    @staticmethod
    def _steady_state_stp(rate, u, tau_rec, tau_fac):
        "Returns stationary limit of short-term plasticity."
        u = u.reshape(1)
        tau_rec = tau_rec.reshape(1)
        tau_fac = tau_fac.reshape(1)

        flat_rate = rate.flatten()
        u_steady = numpy.ones_like(flat_rate) * u
        x_steady = numpy.ones_like(flat_rate)
        mask = flat_rate >= 1e-9

        if tau_fac != 0:
            exp = numpy.exp(-1 / (flat_rate[mask] * tau_fac))
            u_steady[mask] = u / (1 - (1 - u) * exp)

        if tau_rec != 0:
            exp = numpy.exp(-1 / (flat_rate[mask] * tau_rec))
            x_steady[mask] = (1 - exp) / (1 - (1 - u_steady[mask]) * exp)

        flat_stp = u_steady * x_steady
        return flat_stp.reshape(rate.shape)

    @staticmethod
    def get_fluct_regime_vars(inputs_exc, inputs_inh, W, weights_exc, taus_exc, conns_exc, weights_inh, taus_inh, conns_inh, E_e, E_i, g_L, C_m, E_L):
        """
        Compute the mean characteristic of neurons.
        """
        mu_Ge = (weights_exc * taus_exc * conns_exc * inputs_exc).sum(axis=0)[numpy.newaxis]
        mu_Gi = (weights_inh * taus_inh * conns_inh * inputs_inh).sum(axis=0)[numpy.newaxis]
        mu_G = g_L + mu_Ge + mu_Gi
        tau_eff = C_m / mu_G

        mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W[numpy.newaxis]) / mu_G

        U_e = weights_exc / mu_G * (E_e - mu_V)
        U_i = weights_inh / mu_G * (E_i - mu_V)

        s_e = (conns_exc * inputs_exc * (U_e * taus_exc) ** 2 / (2. * (taus_exc + tau_eff))).sum(axis=0)
        s_i = (conns_inh * inputs_inh * (U_i * taus_inh) ** 2 / (2. * (taus_inh + tau_eff))).sum(axis=0)
        sigma_V = numpy.sqrt(s_e + s_i)

        t_e = conns_exc * inputs_exc * (U_e * taus_exc) ** 2
        t_i = conns_inh * inputs_inh * (U_i * taus_inh) ** 2
        T_V_numerator = t_e.sum(axis=0) + t_i.sum(axis=0)
        T_V_denominator = (t_e / (taus_exc + tau_eff)).sum(axis=0) + (t_i / (taus_inh + tau_eff)).sum(axis=0)
        T_V = T_V_numerator / T_V_denominator

        return mu_V[0], sigma_V, T_V

    @staticmethod
    @jit(nopython=True, cache=True)
    def threshold_func(muV, sigmaV, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
        """The threshold function of the neurons."""
        muV0, DmuV0 = -60.0, 10.0
        sV0, DsV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.
        V = (muV - muV0) / DmuV0
        S = (sigmaV - sV0) / DsV0
        T = (TvN - TvN0) / DTvN0
        return P0 + P1*V + P2*S + P3*T + P4*V**2 + P5*S**2 + P6*T**2 + P7*V*S + P8*V*T + P9*S*T

    @staticmethod
    def estimate_firing_rate(muV, sigmaV, Tv, Vthre):
        """The firing rate estimation function of the neurons."""
        return sp_spec.erfc((Vthre - muV) / (numpy.sqrt(2) * sigmaV)) / (2 * Tv)

    def TF_excitatory(self, fe, fi, W, weights_e, taus_e, conns_e, weights_i, taus_i, conns_i):
        """Transfer function for excitatory population."""
        return self.TF(fe, fi, W, weights_e, taus_e, conns_e, weights_i, taus_i, conns_i, self.P_e, self.E_L_e, self.g_L_e, self.C_m_e)

    def TF_inhibitory(self, fe, fi, W, weights_e, taus_e, conns_e, weights_i, taus_i, conns_i):
        """Transfer function for inhibitory population."""
        return self.TF(fe, fi, W, weights_e, taus_e, conns_e, weights_i, taus_i, conns_i, self.P_i, self.E_L_i, self.g_L_i, self.C_m_i)

    def TF(self, fe, fi, W, weights_e, taus_e, conns_e, weights_i, taus_i, conns_i, P, E_L, g_L, C_m):
        """Transfer function main routine."""
        mu_V, sigma_V, T_V = self.get_fluct_regime_vars(
            fe, fi, W, weights_e, taus_e, conns_e, weights_i, taus_i, conns_i,
            self.E_e, self.E_i, g_L, C_m, E_L
        )

        V_thre = self.threshold_func(
            mu_V, sigma_V, T_V * g_L / C_m,
            P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9]
        )
        V_thre *= 1e3  # Convert threshold to mV
        return self.estimate_firing_rate(mu_V, sigma_V, T_V, V_thre)

    def _prepare_inputs_and_weights(self, E, I, stimulus, local_coupling, coupling_0, 
                                    stp_ee_custom=None, stp_ei_custom=None,
                                    stp_ie_custom=None, stp_ii_custom=None):
        """Prepares input populations, connection counts, and synaptic weights."""
        
        # NOTE: no idea what that is, I believe it is zero and it is self 
        # connection on the node, but that is in equations thus here left to zero.
        if local_coupling > 0:
            print(f"Warning: local_coupling == {local_coupling} is not implemented in this model, setting to zero.")
        if coupling_0 > 0:
            print(f"Warning: coupling_0 == {coupling_0} is not implemented in this model, setting to zero.")

        lc_E = local_coupling * E
        lc_I = local_coupling * I

        stim_ratio = getattr(self, 'stim_target_ratio', 1.0)

        # excitatory sources:
        # (self conns, drive input, stimulus, coupling inside node, coupling with other nodes)
        # inhibitory sources:
        # (self cons, inhibitory external, coupling inside node)
        input_ee = self.convert_to_array([E, self.external_input_ex_ex, stimulus, lc_E, coupling_0], axis=0)
        input_ie = self.convert_to_array([E, self.external_input_in_ex, stim_ratio * stimulus, lc_E, coupling_0], axis=0)
        input_ei = self.convert_to_array([I, self.external_input_ex_in, lc_I], axis=0)
        input_ii = self.convert_to_array([I, self.external_input_in_in, lc_I], axis=0)

        conns_ee = numpy.array([self.K_ee, self.K_ed, self.K_es, 0, 0]).reshape((-1, 1, 1))
        conns_ei = numpy.array([self.K_ei, 0, 0]).reshape((-1, 1, 1))
        conns_ie = numpy.array([self.K_ie, self.K_id, self.K_is, 0, 0]).reshape((-1, 1, 1))
        conns_ii = numpy.array([self.K_ii, 0, 0]).reshape((-1, 1, 1))

        weights_ee = numpy.array([self.Q_ee, self.Q_ed, self.Q_es, 0, 0]).reshape((-1, 1, 1))
        weights_ei = numpy.array([self.Q_ei, 0, 0]).reshape((-1, 1, 1))
        weights_ie = numpy.array([self.Q_ie, self.Q_id, self.Q_is, 0, 0]).reshape((-1, 1, 1))
        weights_ii = numpy.array([self.Q_ii, 0, 0]).reshape((-1, 1, 1))

        if stp_ee_custom is not None:
            stp_ee = stp_ee_custom
        else:
            # NOTE: drive and stimulus are expected to be static synapses
            stp_ee = [self._steady_state_stp(E, self.U_ee, self.tau_rec_ee, self.tau_fac_ee), 1., 1., 1., 1.]
        stp_ee = self.convert_to_array(stp_ee, axis=0)

        if stp_ei_custom is not None:
            stp_ei = stp_ei_custom
        else:
            stp_ei = [self._steady_state_stp(I, self.U_ei, self.tau_rec_ei, self.tau_fac_ei), 1., 1.]
        stp_ei = self.convert_to_array(stp_ei, axis=0)

        if stp_ie_custom is not None:
            stp_ie = stp_ie_custom
        else:
            stp_ie = [self._steady_state_stp(E, self.U_ie, self.tau_rec_ie, self.tau_fac_ie), 1., 1., 1., 1.]
        stp_ie = self.convert_to_array(stp_ie, axis=0)

        if stp_ii_custom is not None:
            stp_ii = stp_ii_custom
        else:
            stp_ii = [self._steady_state_stp(I, self.U_ii, self.tau_rec_ii, self.tau_fac_ii), 1., 1.]
        stp_ii = self.convert_to_array(stp_ii, axis=0)

        weights_ee = weights_ee * stp_ee
        weights_ei = weights_ei * stp_ei
        weights_ie = weights_ie * stp_ie
        weights_ii = weights_ii * stp_ii

        taus_e = numpy.array([self.tau_e, self.tau_e, self.tau_e, self.tau_e, self.tau_e]).reshape((-1, 1, 1))
        taus_i = numpy.array([self.tau_i, self.tau_i, self.tau_i]).reshape((-1, 1, 1))

        return (input_ee, input_ei, input_ie, input_ii, 
                weights_ee, conns_ee, weights_ei, conns_ei, weights_ie, conns_ie, weights_ii, conns_ii,
                taus_e, taus_i)

    def _compute_second_order_terms(
        self, input_ee, input_ei, W_e, input_ie, input_ii, W_i,
        weights_ee, conns_ee, weights_ei, conns_ei, weights_ie, conns_ie, weights_ii, conns_ii,
        taus_e, taus_i,
        E, I, C_ee, C_ei, C_ii, N_e, N_i, _TF_e, _TF_i, df=1e-7
    ):
        """Computes 2nd-order numerical partial derivatives and covariance ODE terms."""
        args_exc = (weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei)
        args_inh = (weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii)

        dfe_ee = numpy.zeros_like(input_ee)
        dfe_ee[0] = df
        dfi_ei = numpy.zeros_like(input_ei)
        dfi_ei[0] = df

        dfe_ie = numpy.zeros_like(input_ie)
        dfe_ie[0] = df
        dfi_ii = numpy.zeros_like(input_ii)
        dfi_ii[0] = df

        h = df * 1e3
        h2 = h ** 2

        # First partial derivatives
        d_fe_TF_e = (self.TF_excitatory(input_ee + dfe_ee, input_ei, W_e, *args_exc) - self.TF_excitatory(input_ee - dfe_ee, input_ei, W_e, *args_exc)) / (2 * h)
        d_fi_TF_e = (self.TF_excitatory(input_ee, input_ei + dfi_ei, W_e, *args_exc) - self.TF_excitatory(input_ee, input_ei - dfi_ei, W_e, *args_exc)) / (2 * h)

        d_fe_TF_i = (self.TF_inhibitory(input_ie + dfe_ie, input_ii, W_i, *args_inh) - self.TF_inhibitory(input_ie - dfe_ie, input_ii, W_i, *args_inh)) / (2 * h)
        d_fi_TF_i = (self.TF_inhibitory(input_ie, input_ii + dfi_ii, W_i, *args_inh) - self.TF_inhibitory(input_ie, input_ii - dfi_ii, W_i, *args_inh)) / (2 * h)

        # Second partial derivatives
        d2_fefe_TF_e = (self.TF_excitatory(input_ee + dfe_ee, input_ei, W_e, *args_exc) - 2 * _TF_e + self.TF_excitatory(input_ee - dfe_ee, input_ei, W_e, *args_exc)) / h2
        d2_fefe_TF_i = (self.TF_inhibitory(input_ie + dfe_ie, input_ii, W_i, *args_inh) - 2 * _TF_i + self.TF_inhibitory(input_ie - dfe_ie, input_ii, W_i, *args_inh)) / h2

        d2_fifi_TF_e = (self.TF_excitatory(input_ee, input_ei + dfi_ei, W_e, *args_exc) - 2 * _TF_e + self.TF_excitatory(input_ee, input_ei - dfi_ei, W_e, *args_exc)) / h2
        d2_fifi_TF_i = (self.TF_inhibitory(input_ie, input_ii + dfi_ii, W_i, *args_inh) - 2 * _TF_i + self.TF_inhibitory(input_ie, input_ii - dfi_ii, W_i, *args_inh)) / h2

        # Mixed partial derivatives
        d_fi_TF_e_plus = (self.TF_excitatory(input_ee + dfe_ee, input_ei + dfi_ei, W_e, *args_exc) - self.TF_excitatory(input_ee + dfe_ee, input_ei - dfi_ei, W_e, *args_exc)) / (2 * h)
        d_fi_TF_e_minus = (self.TF_excitatory(input_ee - dfe_ee, input_ei + dfi_ei, W_e, *args_exc) - self.TF_excitatory(input_ee - dfe_ee, input_ei - dfi_ei, W_e, *args_exc)) / (2 * h)
        d2_fefi_TF_e = (d_fi_TF_e_plus - d_fi_TF_e_minus) / (2 * h)

        d_fi_TF_i_plus = (self.TF_inhibitory(input_ie + dfe_ie, input_ii + dfi_ii, W_i, *args_inh) - self.TF_inhibitory(input_ie + dfe_ie, input_ii - dfi_ii, W_i, *args_inh)) / (2 * h)
        d_fi_TF_i_minus = (self.TF_inhibitory(input_ie - dfe_ie, input_ii + dfi_ii, W_i, *args_inh) - self.TF_inhibitory(input_ie - dfe_ie, input_ii - dfi_ii, W_i, *args_inh)) / (2 * h)
        d2_fefi_TF_i = (d_fi_TF_i_plus - d_fi_TF_i_minus) / (2 * h)

        # Rate derivatives 2nd order corrections
        dE = (_TF_e - E + 0.5 * C_ee * d2_fefe_TF_e + C_ei * d2_fefi_TF_e + 0.5 * C_ii * d2_fifi_TF_e) / self.T
        dI = (_TF_i - I + 0.5 * C_ee * d2_fefe_TF_i + C_ei * d2_fefi_TF_i + 0.5 * C_ii * d2_fifi_TF_i) / self.T

        # Covariance derivatives
        dC_ee = (_TF_e * (1. / self.T - _TF_e) / N_e + (_TF_e - E) ** 2 + 2. * C_ee * d_fe_TF_e + 2. * C_ei * d_fi_TF_i - 2. * C_ee) / self.T
        dC_ei = ((_TF_e - E) * (_TF_i - I) + C_ee * d_fe_TF_e + C_ei * d_fe_TF_i + C_ei * d_fi_TF_e + C_ii * d_fi_TF_i - 2. * C_ei) / self.T
        dC_ii = (_TF_i * (1. / self.T - _TF_i) / N_i + (_TF_i - I) ** 2 + 2. * C_ii * d_fi_TF_i + 2. * C_ei * d_fe_TF_e - 2. * C_ii) / self.T

        return dE, dI, dC_ee, dC_ei, dC_ii


class NeuroPSI_STP_asymptotic_first_order(BaseNeuroPSI_STP):
    r"""
    First-order mean-field model with asymptotic short-term plasticity (STP).
    """
    _ui_name = "NeuroPSI_STP_asymptotic_first_order"
    ui_configurable_parameters = [
        'g_L', 'E_L_e', 'E_L_i', 'C_m', 'b', 'tau_w',
        'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i',
        'N_tot', 'p_connect', 'g', 'T', 'external_input'
    ]

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "E": numpy.array([0.0, 0.0]),
            "I": numpy.array([0.0, 0.0]),
            "W_e": numpy.array([0.0, 0.0]),
            "W_i": numpy.array([0.0, 0.0]),
            "noise": numpy.array([0.0, 0.0]),
            "stimulus": numpy.array([0.0, 0.0]),
        },
        doc="""State-variable dynamic ranges."""
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "W_e", "W_i", "noise", "stimulus"),
        default=("E",),
        doc="""Default state-variables of this Model to be monitored."""
    )

    state_variable_boundaries = Final(
        label="Firing rate of population is always positive",
        default={
            "E": numpy.array([0.0, None]),
            "I": numpy.array([0.0, None])
        },
        doc="""Boundaries of state-variables."""
    )

    state_variables = 'E I W_e W_i noise stimulus'.split()
    _nvar = 6
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        E = state_variables[0, :]
        I = state_variables[1, :]
        W_e = state_variables[2, :]
        W_i = state_variables[3, :]
        noise = state_variables[4, :]
        stimulus = state_variables[5, :]
        derivative = numpy.empty_like(state_variables)

        c_0 = coupling[0, :]

        (input_ee, input_ei, input_ie, input_ii, 
        weights_ee, conns_ee, weights_ei, conns_ei, weights_ie, conns_ie, weights_ii, conns_ii,
        taus_e, taus_i) = self._prepare_inputs_and_weights(E, I, stimulus, local_coupling, c_0)


        # Firing rates
        _TF_e = self.TF_excitatory(input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei)
        _TF_i = self.TF_inhibitory(input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii)

        derivative[0] = (_TF_e - E) / self.T
        derivative[1] = (_TF_i - I) / self.T

        # Adaptation
        mu_V_e, _, _ = self.get_fluct_regime_vars(
            input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei,
            self.E_e, self.E_i, self.g_L_e, self.C_m_e, self.E_L_e
        )
        derivative[2] = -W_e / self.tau_w_e + self.b_e * E + self.a_e * (mu_V_e - self.E_L_e) / self.tau_w_e

        mu_V_i, _, _ = self.get_fluct_regime_vars(
            input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii,
            self.E_e, self.E_i, self.g_L_i, self.C_m_i, self.E_L_i
        )
        derivative[3] = -W_i / self.tau_w_i + self.b_i * I + self.a_i * (mu_V_i - self.E_L_i) / self.tau_w_i

        # Noise & stimulus
        derivative[4] = -noise / self.tau_OU
        derivative[5] = 0.0
        state_variables[5, :] = 0.0

        return derivative


class NeuroPSI_STP_asymptotic_second_order(NeuroPSI_STP_asymptotic_first_order):
    r"""
    Second-order mean-field model with asymptotic short-term plasticity (STP).
    """
    _ui_name = "NeuroPSI_STP_asymptotic_second_order"

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "E": numpy.array([0.0, 0.0]),
            "I": numpy.array([0.0, 0.0]),
            "C_ee": numpy.array([0.0, 0.0]),
            "C_ei": numpy.array([0.0, 0.0]),
            "C_ii": numpy.array([0.0, 0.0]),
            "W_e": numpy.array([0.0, 0.0]),
            "W_i": numpy.array([0.0, 0.0]),
            "noise": numpy.array([0.0, 0.0]),
            "stimulus": numpy.array([0.0, 0.0]),
        },
        doc="""State-variable dynamic ranges."""
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "C_ee", "C_ei", "C_ii", "W_e", "W_i", "noise", "stimulus"),
        default=("E",),
        doc="""Default state-variables of this Model to be monitored."""
    )

    state_variables = 'E I C_ee C_ei C_ii W_e W_i noise stimulus'.split()
    _nvar = 9

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        N_e = self.N_tot * (1 - self.g)
        N_i = self.N_tot * self.g

        E = state_variables[0, :]
        I = state_variables[1, :]
        C_ee = state_variables[2, :]
        C_ei = state_variables[3, :]
        C_ii = state_variables[4, :]
        W_e = state_variables[5, :]
        W_i = state_variables[6, :]
        noise = state_variables[7, :]
        stimulus = state_variables[8, :]
        derivative = numpy.empty_like(state_variables)

        c_0 = coupling[0, :]

        (input_ee, input_ei, input_ie, input_ii, 
        weights_ee, conns_ee, weights_ei, conns_ei, weights_ie, conns_ie, weights_ii, conns_ii,
        taus_e, taus_i) = self._prepare_inputs_and_weights(E, I, stimulus, local_coupling, c_0)

        _TF_e = self.TF_excitatory(input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei)
        _TF_i = self.TF_inhibitory(input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii)

        dE, dI, dC_ee, dC_ei, dC_ii = self._compute_second_order_terms(
            input_ee, input_ei, W_e, input_ie, input_ii, W_i,
            weights_ee, conns_ee, weights_ei, conns_ei,
            weights_ie, conns_ie, weights_ii, conns_ii,
            taus_e, taus_i,
            E, I, C_ee, C_ei, C_ii, N_e, N_i, _TF_e, _TF_i
        )

        derivative[0] = dE
        derivative[1] = dI
        derivative[2] = dC_ee
        derivative[3] = dC_ei
        derivative[4] = dC_ii

        # Adaptation
        mu_V_e, _, _ = self.get_fluct_regime_vars(
            input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei,
            self.E_e, self.E_i, self.g_L_e, self.C_m_e, self.E_L_e
        )
        derivative[5] = -W_e / self.tau_w_e + self.b_e * E + self.a_e * (mu_V_e - self.E_L_e) / self.tau_w_e

        mu_V_i, _, _ = self.get_fluct_regime_vars(
            input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii,
            self.E_e, self.E_i, self.g_L_i, self.C_m_i, self.E_L_i
        )
        derivative[6] = -W_i / self.tau_w_i + self.b_i * I + self.a_i * (mu_V_i - self.E_L_i) / self.tau_w_i

        # Noise & stimulus
        derivative[7] = -noise / self.tau_OU
        derivative[8] = 0.0
        state_variables[8, :] = 0.0

        return derivative


class NeuroPSI_STP_dynamic_first_order(BaseNeuroPSI_STP):
    r"""
    First-order mean-field model with dynamic short-term plasticity (STP).
    """
    _ui_name = "NeuroPSI_STP_dynamic_first_order"
    ui_configurable_parameters = [
        'g_L_e', 'g_L_i', 'C_m_e', 'C_m_i', 'E_L_e', 'E_L_i', 'b', 'tau_w',
        'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i',
        'tau_rec_e', 'U_e', 'tau_rec_i', 'U_i',
        'N_tot', 'p_connect', 'g', 'T', 'external_input'
    ]

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "E": numpy.array([0.0, 0.0]),
            "I": numpy.array([0.0, 0.0]),
            "W_e": numpy.array([0.0, 0.0]),
            "W_i": numpy.array([0.0, 0.0]),
            "X_ee": numpy.array([1.0, 1.0]),
            "Y_ee": numpy.array([0.0, 0.0]),
            "U_dyn_ee": numpy.array([1.0, 1.0]),
            "X_ei": numpy.array([1.0, 1.0]),
            "Y_ei": numpy.array([0.0, 0.0]),
            "U_dyn_ei": numpy.array([1.0, 1.0]),
            "X_ie": numpy.array([1.0, 1.0]),
            "Y_ie": numpy.array([0.0, 0.0]),
            "U_dyn_ie": numpy.array([1.0, 1.0]),
            "X_ii": numpy.array([1.0, 1.0]),
            "Y_ii": numpy.array([0.0, 0.0]),
            "U_dyn_ii": numpy.array([1.0, 1.0]),
            "noise": numpy.array([0.0, 0.0]),
            "stimulus": numpy.array([0.0, 0.0]),
        },
        doc="""State-variable dynamic ranges."""
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "W_e", "W_i", "X_ee", "Y_ee", "U_dyn_ee", "X_ei", "Y_ei", "U_dyn_ei", "X_ie", "Y_ie", "U_dyn_ie", "X_ii", "Y_ii", "U_dyn_ii", "noise", "stimulus"),
        default=("E",),
        doc="""Default state-variables of this Model to be monitored."""
    )

    state_variable_boundaries = Final(
        label="Firing rate of population is always positive",
        default={
            "E": numpy.array([0.0, None]),
            "I": numpy.array([0.0, None])
        },
        doc="""Boundaries of state-variables."""
    )

    state_variables = 'E I W_e W_i X_ee Y_ee U_dyn_ee X_ei Y_ei U_dyn_ei X_ie Y_ie U_dyn_ie X_ii Y_ii U_dyn_ii noise stimulus'.split()
    _nvar = 18
    cvar = numpy.array([0], dtype=numpy.int32)

    def tsodyks_markram_stp(self, X, Y, U_dyn, rate, tau_rec, tau_fac, tau_syn, U):
        """Tsodyks-Markram short-term plasticity model."""
        if tau_rec:
            u = U_dyn * (1 - U) + U
            dX = (1 - X) / tau_rec - u * X * rate
            dY = -Y / tau_syn + u * X * rate
        else:
            dX = 0.
            dY = 0.

        if tau_fac:
            dU_dyn = -U_dyn / tau_fac + U * (1. - U_dyn) * rate
        else:
            dU_dyn = 0.

        return dX, dY, dU_dyn

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        E = state_variables[0, :]
        I = state_variables[1, :]
        W_e = state_variables[2, :]
        W_i = state_variables[3, :]
        X_ee = state_variables[4, :]
        Y_ee = state_variables[5, :]
        U_dyn_ee = state_variables[6, :]
        X_ei = state_variables[7, :]
        Y_ei = state_variables[8, :]
        U_dyn_ei = state_variables[9, :]
        X_ie = state_variables[10, :]
        Y_ie = state_variables[11, :]
        U_dyn_ie = state_variables[12, :]
        X_ii = state_variables[13, :]
        Y_ii = state_variables[14, :]
        U_dyn_ii = state_variables[15, :]
        noise = state_variables[16, :]
        stimulus = state_variables[17, :]
        derivative = numpy.empty_like(state_variables)

        c_0 = coupling[0, :]

        stp_ee_custom = [X_ee * (U_dyn_ee*(1-self.U_ee) + self.U_ee), 1., 1., 1., 1.]
        stp_ei_custom = [X_ei * (U_dyn_ei*(1-self.U_ei) + self.U_ei), 1., 1.]
        stp_ie_custom = [X_ie * (U_dyn_ie*(1-self.U_ie) + self.U_ie), 1., 1., 1., 1.]
        stp_ii_custom = [X_ii * (U_dyn_ii*(1-self.U_ii) + self.U_ii), 1., 1.]

        (input_ee, input_ei, input_ie, input_ii, 
        weights_ee, conns_ee, weights_ei, conns_ei, weights_ie, conns_ie, weights_ii, conns_ii,
        taus_e, taus_i) = self._prepare_inputs_and_weights(E, I, stimulus, local_coupling, c_0, stp_ee_custom=stp_ee_custom, stp_ei_custom=stp_ei_custom, stp_ie_custom=stp_ie_custom, stp_ii_custom=stp_ii_custom)

        # Firing rates
        _TF_e = self.TF_excitatory(input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei)
        _TF_i = self.TF_inhibitory(input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii)

        derivative[0] = (_TF_e - E) / self.T
        derivative[1] = (_TF_i - I) / self.T

        # Adaptation
        mu_V_e, _, _ = self.get_fluct_regime_vars(
            input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei,
            self.E_e, self.E_i, self.g_L_e, self.C_m_e, self.E_L_e
        )
        derivative[2] = -W_e / self.tau_w_e + self.b_e * E + self.a_e * (mu_V_e - self.E_L_e) / self.tau_w_e

        mu_V_i, _, _ = self.get_fluct_regime_vars(
            input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii,
            self.E_e, self.E_i, self.g_L_i, self.C_m_i, self.E_L_i
        )
        derivative[3] = -W_i / self.tau_w_i + self.b_i * I + self.a_i * (mu_V_i - self.E_L_i) / self.tau_w_i

        # Dynamic Synaptic Plasticity ODEs (exc -> exc)
        dX_ee, dY_ee, dU_dyn_ee = self.tsodyks_markram_stp(X_ee, Y_ee, U_dyn_ee, E, self.tau_rec_ee, self.tau_fac_ee, self.tau_e, self.U_ee)
        derivative[4] = dX_ee
        derivative[5] = dY_ee
        derivative[6] = dU_dyn_ee
        
        # Dynamic Synaptic Plasticity ODEs (inh -> exc)
        dX_ei, dY_ei, dU_dyn_ei = self.tsodyks_markram_stp(X_ei, Y_ei, U_dyn_ei, I, self.tau_rec_ei, self.tau_fac_ei, self.tau_e, self.U_ei)
        derivative[7] = dX_ei
        derivative[8] = dY_ei
        derivative[9] = dU_dyn_ei

        # Dynamic Synaptic Plasticity ODEs (exc -> inh)
        dX_ie, dY_ie, dU_dyn_ie = self.tsodyks_markram_stp(X_ie, Y_ie, U_dyn_ie, E, self.tau_rec_ie, self.tau_fac_ie, self.tau_i, self.U_ie)
        derivative[10] = dX_ie
        derivative[11] = dY_ie
        derivative[12] = dU_dyn_ie

        # Dynamic Synaptic Plasticity ODEs (inh -> inh)
        dX_ii, dY_ii, dU_dyn_ii = self.tsodyks_markram_stp(X_ii, Y_ii, U_dyn_ii, I, self.tau_rec_ii, self.tau_fac_ii, self.tau_i, self.U_ii)
        derivative[13] = dX_ii
        derivative[14] = dY_ii
        derivative[15] = dU_dyn_ii

        # Noise & stimulus
        derivative[16] = -noise / self.tau_OU
        derivative[17] = 0.0
        state_variables[17, :] = 0.0

        return derivative


class NeuroPSI_STP_dynamic_second_order(NeuroPSI_STP_dynamic_first_order):
    r"""
    Second-order mean-field model with dynamic short-term plasticity (STP).
    """
    _ui_name = "NeuroPSI_STP_dynamic_second_order"

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "E": numpy.array([0.0, 0.0]),
            "I": numpy.array([0.0, 0.0]),
            "C_ee": numpy.array([0.0, 0.0]),
            "C_ei": numpy.array([0.0, 0.0]),
            "C_ii": numpy.array([0.0, 0.0]),
            "W_e": numpy.array([0.0, 0.0]),
            "W_i": numpy.array([0.0, 0.0]),
            "X_ee": numpy.array([1.0, 1.0]),
            "Y_ee": numpy.array([0.0, 0.0]),
            "U_dyn_ee": numpy.array([1.0, 1.0]),
            "X_ei": numpy.array([1.0, 1.0]),
            "Y_ei": numpy.array([0.0, 0.0]),
            "U_dyn_ei": numpy.array([1.0, 1.0]),
            "X_ie": numpy.array([1.0, 1.0]),
            "Y_ie": numpy.array([0.0, 0.0]),
            "U_dyn_ie": numpy.array([1.0, 1.0]),
            "X_ii": numpy.array([1.0, 1.0]),
            "Y_ii": numpy.array([0.0, 0.0]),
            "U_dyn_ii": numpy.array([1.0, 1.0]),
            "noise": numpy.array([0.0, 0.0]),
            "stimulus": numpy.array([0.0, 0.0]),
        },
        doc="""State-variable dynamic ranges."""
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("E", "I", "C_ee", "C_ei", "C_ii", "W_e", "W_i", "X_ee", "Y_ee", "U_dyn_ee", "X_ei", "Y_ei", "U_dyn_ei", "X_ie", "Y_ie", "U_dyn_ie", "X_ii", "Y_ii", "U_dyn_ii", "noise", "stimulus"),
        default=("E",),
        doc="""Default state-variables of this Model to be monitored."""
    )

    state_variables = 'E I C_ee C_ei C_ii W_e W_i X_ee Y_ee U_dyn_ee X_ei Y_ei U_dyn_ei X_ie Y_ie U_dyn_ie X_ii Y_ii U_dyn_ii noise stimulus'.split()
    _nvar = 21

    def dfun(self, state_variables, coupling, local_coupling=0.00):
        N_e = self.N_tot * (1 - self.g)
        N_i = self.N_tot * self.g

        E = state_variables[0, :]
        I = state_variables[1, :]
        C_ee = state_variables[2, :]
        C_ei = state_variables[3, :]
        C_ii = state_variables[4, :]
        W_e = state_variables[5, :]
        W_i = state_variables[6, :]
        X_ee = state_variables[7, :]
        Y_ee = state_variables[8, :]
        U_dyn_ee = state_variables[9, :]
        X_ei = state_variables[10, :]
        Y_ei = state_variables[11, :]
        U_dyn_ei = state_variables[12, :]
        X_ie = state_variables[13, :]
        Y_ie = state_variables[14, :]
        U_dyn_ie = state_variables[15, :]
        X_ii = state_variables[16, :]
        Y_ii = state_variables[17, :]
        U_dyn_ii = state_variables[18, :]
        noise = state_variables[19, :]
        stimulus = state_variables[20, :]
        derivative = numpy.empty_like(state_variables)

        c_0 = coupling[0, :]

        stp_ee_custom = [X_ee * (U_dyn_ee*(1-self.U_ee) + self.U_ee), 1., 1., 1., 1.]
        stp_ei_custom = [X_ei * (U_dyn_ei*(1-self.U_ei) + self.U_ei), 1., 1.]
        stp_ie_custom = [X_ie * (U_dyn_ie*(1-self.U_ie) + self.U_ie), 1., 1., 1., 1.]
        stp_ii_custom = [X_ii * (U_dyn_ii*(1-self.U_ii) + self.U_ii), 1., 1.]

        (input_ee, input_ei, input_ie, input_ii, 
        weights_ee, conns_ee, weights_ei, conns_ei, weights_ie, conns_ie, weights_ii, conns_ii,
        taus_e, taus_i) = self._prepare_inputs_and_weights(E, I, stimulus, local_coupling, c_0, stp_ee_custom=stp_ee_custom, stp_ei_custom=stp_ei_custom, stp_ie_custom=stp_ie_custom, stp_ii_custom=stp_ii_custom)

        _TF_e = self.TF_excitatory(input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei)
        _TF_i = self.TF_inhibitory(input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii)

        dE, dI, dC_ee, dC_ei, dC_ii = self._compute_second_order_terms(
            input_ee, input_ei, W_e, input_ie, input_ii, W_i,
            weights_ee, conns_ee, weights_ei, conns_ei,
            weights_ie, conns_ie, weights_ii, conns_ii,
            taus_e, taus_i,
            E, I, C_ee, C_ei, C_ii, N_e, N_i, _TF_e, _TF_i
        )

        derivative[0] = dE
        derivative[1] = dI
        derivative[2] = dC_ee
        derivative[3] = dC_ei
        derivative[4] = dC_ii

        # Adaptation
        mu_V_e, _, _ = self.get_fluct_regime_vars(
            input_ee, input_ei, W_e, weights_ee, taus_e, conns_ee, weights_ei, taus_i, conns_ei,
            self.E_e, self.E_i, self.g_L_e, self.C_m_e, self.E_L_e
        )
        derivative[5] = -W_e / self.tau_w_e + self.b_e * E + self.a_e * (mu_V_e - self.E_L_e) / self.tau_w_e

        mu_V_i, _, _ = self.get_fluct_regime_vars(
            input_ie, input_ii, W_i, weights_ie, taus_e, conns_ie, weights_ii, taus_i, conns_ii,
            self.E_e, self.E_i, self.g_L_i, self.C_m_i, self.E_L_i
        )
        derivative[6] = -W_i / self.tau_w_i + self.b_i * I + self.a_i * (mu_V_i - self.E_L_i) / self.tau_w_i

        # Dynamic Synaptic Plasticity ODEs (exc -> exc)
        dX_ee, dY_ee, dU_dyn_ee = self.tsodyks_markram_stp(X_ee, Y_ee, U_dyn_ee, E, self.tau_rec_ee, self.tau_fac_ee, self.tau_e, self.U_ee)
        derivative[7] = dX_ee
        derivative[8] = dY_ee
        derivative[9] = dU_dyn_ee
        
        # Dynamic Synaptic Plasticity ODEs (inh -> exc)
        dX_ei, dY_ei, dU_dyn_ei = self.tsodyks_markram_stp(X_ei, Y_ei, U_dyn_ei, I, self.tau_rec_ei, self.tau_fac_ei, self.tau_e, self.U_ei)
        derivative[10] = dX_ei
        derivative[11] = dY_ei
        derivative[12] = dU_dyn_ei

        # Dynamic Synaptic Plasticity ODEs (exc -> inh)
        dX_ie, dY_ie, dU_dyn_ie = self.tsodyks_markram_stp(X_ie, Y_ie, U_dyn_ie, E, self.tau_rec_ie, self.tau_fac_ie, self.tau_i, self.U_ie)
        derivative[13] = dX_ie
        derivative[14] = dY_ie
        derivative[15] = dU_dyn_ie

        # Dynamic Synaptic Plasticity ODEs (inh -> inh)
        dX_ii, dY_ii, dU_dyn_ii = self.tsodyks_markram_stp(X_ii, Y_ii, U_dyn_ii, I, self.tau_rec_ii, self.tau_fac_ii, self.tau_i, self.U_ii)
        derivative[16] = dX_ii
        derivative[17] = dY_ii
        derivative[18] = dU_dyn_ii

        # Noise & stimulus
        derivative[19] = -noise / self.tau_OU
        derivative[20] = 0.0
        state_variables[20, :] = 0.0

        return derivative
