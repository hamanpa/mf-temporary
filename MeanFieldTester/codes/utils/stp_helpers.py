"""
Tsodyks-Markram short-term plasticity (STP): steady states and reconstruction of NEST's `tsodyks_synapse`.

Conventions (as NEST's `tsodyks_synapse`): between spikes the available resources x recover to 1 with tau_rec
and the facilitation variable decays to 0 with tau_fac; at a spike u⁺ = u⁻ + U(1 − u⁻) is the utilisation used
and u⁺·x⁻ the fraction released (the efficacy: the spike adds weight·u⁺·x⁻ to the conductance).
Without facilitation (tau_fac = 0) u⁺ = U.

Steady states depend on the statistics of the presynaptic spike train (`SPIKE_STATISTICS`). Both follow from the
spike-to-spike map averaged in the stationary state, x̄ = (1 − ⟨E⟩)/(1 − (1 − ū⁺)⟨E⟩) with ⟨E⟩ = ⟨exp(−Δ/tau_rec)⟩
over the inter-spike intervals Δ:
- "regular": every interval is 1/rate, ⟨E⟩ = exp(−1/(rate·tau_rec)).
- "poisson": exponential intervals, ⟨E⟩ = rate·tau_rec/(1 + rate·tau_rec), so x̄ = 1/(1 + ū⁺·rate·tau_rec).
  Exact without facilitation; with facilitation ū⁺ is exact, x̄ and the efficacy moments use the closure
  ⟨u⁺·x⁻⟩ ≈ ū⁺·x̄ (u⁺ and x⁻ are negatively correlated, so the true x̄ is somewhat higher).
Rates are in [Hz], times in [ms].
"""

import numpy as np
from numba import njit

SPIKE_STATISTICS = ("regular", "poisson")


def _check_spike_statistics(spike_statistics: str) -> None:
    if spike_statistics not in SPIKE_STATISTICS:
        raise ValueError(f"Unknown spike_statistics '{spike_statistics}'. Use one of {SPIKE_STATISTICS}.")


def _regular_decay(rate: np.ndarray, tau: float) -> np.ndarray:
    """exp(−1/(rate·tau)) of a regular train [rate in Hz, tau in ms]; 0 for rate = 0 or tau = 0 (full recovery)."""
    decay = np.zeros_like(rate)
    if tau > 0:
        mask = rate > 0
        decay[mask] = np.exp(-1 / (rate[mask] * 1e-3 * tau))
    return decay


def _poisson_decay_moments(rate: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """⟨exp(−Δ/tau)⟩ and ⟨exp(−2Δ/tau)⟩ over exponential intervals Δ [rate in Hz, tau in ms]."""
    rho = rate * 1e-3 * tau
    return rho / (1 + rho), rho / (rho + 2)


def steady_state_stp(rate, U: float, tau_rec: float, tau_fac: float, spike_statistics: str = "poisson") -> tuple[np.ndarray, np.ndarray]:
    """
    Steady-state x⁻ (resources just before a spike) and u⁺ (utilisation used by a spike), averaged over spikes.

    Parameters
    ----------
    rate : array-like
        Presynaptic rate [Hz].
    U : float
        Utilisation increment [unitless].
    tau_rec, tau_fac : float
        Recovery and facilitation time constants [ms]; 0 means no depression / no facilitation.
    spike_statistics : str
        "regular" or "poisson", see the module docstring.

    Returns
    -------
    x, u : np.ndarray
        Mean x⁻ and u⁺ [unitless], shaped like `rate`.
    """
    _check_spike_statistics(spike_statistics)
    rate = np.asarray(rate, dtype=float)
    if spike_statistics == "regular":
        decay_fac = _regular_decay(rate, tau_fac)
        u = U / (1 - (1 - U) * decay_fac)
        decay_rec = _regular_decay(rate, tau_rec)
        x = (1 - decay_rec) / (1 - (1 - u) * decay_rec)
        return x, u

    mean_fac, _ = _poisson_decay_moments(rate, tau_fac)
    u = U / (1 - (1 - U) * mean_fac)
    mean_rec, _ = _poisson_decay_moments(rate, tau_rec)
    x = (1 - mean_rec) / (1 - (1 - u) * mean_rec)
    return x, u


def steady_state_efficacy_moments(rate, U: float, tau_rec: float, tau_fac: float, spike_statistics: str = "poisson") -> tuple[np.ndarray, np.ndarray]:
    """
    Steady-state first and second moments of the efficacy u⁺·x⁻ over spikes: ⟨u⁺x⁻⟩ and ⟨(u⁺x⁻)²⟩ [unitless].

    "regular": every spike has the same efficacy, so the second moment is the squared mean.
    "poisson": second moments from the averaged squared spike-to-spike maps, e.g. without facilitation
    ⟨x²⟩ = (⟨(1 − E)²⟩ + 2(1 − U)⟨(1 − E)E⟩⟨x⟩)/(1 − (1 − U)²⟨E²⟩) with ⟨E²⟩ = rate·tau_rec/(rate·tau_rec + 2);
    exact without facilitation, the closure ⟨u·x⟩ ≈ ⟨u⟩⟨x⟩ (and for the squares) with facilitation.
    Parameters as in `steady_state_stp`.
    """
    x, u = steady_state_stp(rate, U, tau_rec, tau_fac, spike_statistics)
    if spike_statistics == "regular":
        efficacy = u * x
        return efficacy, efficacy ** 2

    rate = np.asarray(rate, dtype=float)
    mean_fac, sq_fac = _poisson_decay_moments(rate, tau_fac)
    u_sq = (U ** 2 + 2 * U * (1 - U) * mean_fac * u) / (1 - (1 - U) ** 2 * sq_fac)
    mean_rec, sq_rec = _poisson_decay_moments(rate, tau_rec)
    one_minus_u_sq = 1 - 2 * u + u_sq
    x_sq = (1 - 2 * mean_rec + sq_rec + 2 * (mean_rec - sq_rec) * (1 - u) * x) / (1 - one_minus_u_sq * sq_rec)
    return u * x, u_sq * x_sq


def effective_synapse_weights(rate, syn_weight: float, U: float, tau_rec: float, tau_fac: float,
                              spike_statistics: str = "poisson") -> tuple[np.ndarray, np.ndarray]:
    """
    Steady-state effective weights of a Tsodyks synapse: weight·⟨u⁺x⁻⟩ (for the mean conductance) and
    weight·√⟨(u⁺x⁻)²⟩ (for the conductance fluctuations), in the unit of `syn_weight`.
    Parameters as in `steady_state_stp`.
    """
    mean, mean_sq = steady_state_efficacy_moments(rate, U, tau_rec, tau_fac, spike_statistics)
    if spike_statistics == "regular":
        return syn_weight * mean, syn_weight * mean  # equal efficacies: exactly the mean (no sqrt round-off)
    return syn_weight * mean, syn_weight * np.sqrt(mean_sq)


# ----------------------------------------------------------------------
# Reconstruction of NEST's tsodyks_synapse from presynaptic spike times
# ----------------------------------------------------------------------

@njit(cache=True)
def _decay(h: float, tau: float) -> float:
    """exp(-h / tau); tau == 0 means instantaneous decay (returns 0)."""
    return 0.0 if tau == 0 else np.exp(-h / tau)


@njit(cache=True)
def _propagators(h: float, tau_rec: float, tau_fac: float, tau_psc: float):
    """Exact propagators of NEST's `tsodyks_synapse` over an interval h (same formulas as NEST)."""
    Puu = _decay(h, tau_fac)
    Pyy = _decay(h, tau_psc)
    Pzz = _decay(h, tau_rec)
    Pxy = ((Pzz - 1.0) * tau_rec - (Pyy - 1.0) * tau_psc) / (tau_psc - tau_rec)
    Pxz = 1.0 - Pzz
    return Puu, Pyy, Pxy, Pxz


@njit(cache=True)
def _spike_states(spike_times: np.ndarray, U: float, tau_rec: float, tau_fac: float, tau_psc: float):
    """
    NEST's `tsodyks_synapse` update at each spike of one train (Tsodyks et al. 1998 three-state model;
    resources x -> y -> z -> x). At each spike, after propagating the state over the interval h:

        z = 1 - x - y
        u <- u * exp(-h/tau_fac)          (u decays to 0; tau_fac = 0 means u = 0 before the jump)
        x <- x + Pxy*y + Pxz*z
        y <- y * exp(-h/tau_psc)
        u <- u + U*(1 - u)                (u⁺, the utilisation used)
        x <- x - u*x ;  y <- y + u*x      (x before this line is x⁻)

    The initial state is u = 0, x = 1, y = 0 (NEST defaults, last spike at t = 0).

    Returns u⁺ and x⁻ per spike, and the post-spike states (u, x, y) with index 0 the initial state.
    """
    n_spikes = spike_times.size
    u_plus = np.empty(n_spikes)
    x_minus = np.empty(n_spikes)
    u_post = np.empty(n_spikes + 1)
    x_post = np.empty(n_spikes + 1)
    y_post = np.empty(n_spikes + 1)
    u_post[0], x_post[0], y_post[0] = 0.0, 1.0, 0.0
    t_last = 0.0
    for k in range(n_spikes):
        Puu, Pyy, Pxy, Pxz = _propagators(spike_times[k] - t_last, tau_rec, tau_fac, tau_psc)
        u, x, y = u_post[k], x_post[k], y_post[k]
        z = 1.0 - x - y
        u = u * Puu
        x = x + Pxy * y + Pxz * z
        y = y * Pyy
        u = u + U * (1.0 - u)
        u_plus[k], x_minus[k] = u, x
        delta = u * x
        u_post[k + 1], x_post[k + 1], y_post[k + 1] = u, x - delta, y + delta
        t_last = spike_times[k]
    return u_plus, x_minus, u_post, x_post, y_post


@njit(cache=True)
def _states_on_grid(spike_times, u_post, x_post, y_post, U, tau_rec, tau_fac, tau_psc, times):
    """u (the utilisation a spike at t would use) and x on the time grid, propagated from the last post-spike state."""
    u_grid = np.empty(times.size)
    x_grid = np.empty(times.size)
    k = 0  # number of spikes at or before times[i] (times are sorted)
    for i in range(times.size):
        while k < spike_times.size and spike_times[k] <= times[i]:
            k += 1
        t_last = spike_times[k - 1] if k > 0 else 0.0
        Puu, _, Pxy, Pxz = _propagators(times[i] - t_last, tau_rec, tau_fac, tau_psc)
        u_decayed = u_post[k] * Puu
        u_grid[i] = u_decayed + U * (1.0 - u_decayed)
        x_grid[i] = x_post[k] + Pxy * y_post[k] + Pxz * (1.0 - x_post[k] - y_post[k])
    return u_grid, x_grid


def _check_tau_psc(tau_rec: float, tau_psc: float) -> None:
    if tau_rec == tau_psc:
        raise ValueError("tsodyks_synapse requires tau_rec != tau_psc (singular propagator, as in NEST).")


def reconstruct_stp_dynamics(spike_times: list, U: float, tau_rec: float, tau_fac: float, tau_psc: float, times: np.ndarray):
    """
    Reconstructs offline the state of NEST's `tsodyks_synapse` driven by presynaptic spike trains,
    on a time grid. Follows NEST's update exactly (see `_spike_states`).

    Parameters
    ----------
    spike_times : list of array-like
        Presynaptic spike times [ms], one sorted array per presynaptic neuron.
    U, tau_rec, tau_fac, tau_psc : float
        `tsodyks_synapse` parameters (times in [ms]); tau_rec must differ from tau_psc (as in NEST).
    times : np.ndarray
        Sorted time grid [ms] on which the state is returned.

    Returns
    -------
    u, x : np.ndarray
        Arrays of shape (len(times), len(spike_times)):
        - u(t): utilisation a spike at time t would use, u_dec(t) + U*(1 - u_dec(t)), with u_dec
          decaying from the last post-spike value. u(t)*x(t)*weight is the efficacy of a spike at t,
          which corresponds to the MF models' X * u.
        - x(t): available resources.
    """
    _check_tau_psc(tau_rec, tau_psc)
    times = np.asarray(times, dtype=float)
    u_all = np.empty((times.size, len(spike_times)))
    x_all = np.empty((times.size, len(spike_times)))
    for neuron_idx, neuron_spike_times in enumerate(spike_times):
        neuron_spike_times = np.asarray(neuron_spike_times, dtype=float)
        _, _, u_post, x_post, y_post = _spike_states(neuron_spike_times, U, tau_rec, tau_fac, tau_psc)
        u_all[:, neuron_idx], x_all[:, neuron_idx] = _states_on_grid(
            neuron_spike_times, u_post, x_post, y_post, U, tau_rec, tau_fac, tau_psc, times
        )
    return u_all, x_all


def efficacy_at_spikes(spike_times: list, U: float, tau_rec: float, tau_fac: float, tau_psc: float,
                       t_start: float = 0.0, t_end: float = np.inf) -> dict:
    """
    Statistics of NEST's `tsodyks_synapse` state at the spikes in [t_start, t_end], pooled over all spikes of all
    sources (each spike counts once, as in the conductance). The reconstruction runs from t = 0 (see `_spike_states`).

    Parameters
    ----------
    spike_times : list of array-like
        Presynaptic spike times [ms], one sorted array per source.
    U, tau_rec, tau_fac, tau_psc : float
        `tsodyks_synapse` parameters (times in [ms]).
    t_start, t_end : float
        Window [ms] of the spikes that are averaged.

    Returns
    -------
    dict
        {"x_mean", "x_std", "u_mean", "u_std", "ux_mean", "ux_std"} of x⁻, u⁺ and u⁺·x⁻ [unitless] (std over spikes).
        Without spikes in the window: the state a spike would meet after a long silence (x = 1, u = U), std 0.
    """
    _check_tau_psc(tau_rec, tau_psc)
    u_parts, x_parts = [], []
    for source_spike_times in spike_times:
        source_spike_times = np.asarray(source_spike_times, dtype=float)
        u_plus, x_minus, _, _, _ = _spike_states(source_spike_times, U, tau_rec, tau_fac, tau_psc)
        in_window = (source_spike_times >= t_start) & (source_spike_times <= t_end)
        u_parts.append(u_plus[in_window])
        x_parts.append(x_minus[in_window])
    u = np.concatenate(u_parts) if u_parts else np.empty(0)
    x = np.concatenate(x_parts) if x_parts else np.empty(0)
    if u.size == 0:
        return {"x_mean": 1.0, "x_std": 0.0, "u_mean": float(U), "u_std": 0.0, "ux_mean": float(U), "ux_std": 0.0}
    ux = u * x
    return {
        "x_mean": float(x.mean()), "x_std": float(x.std()),
        "u_mean": float(u.mean()), "u_std": float(u.std()),
        "ux_mean": float(ux.mean()), "ux_std": float(ux.std()),
    }
