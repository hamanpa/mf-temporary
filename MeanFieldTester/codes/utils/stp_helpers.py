import numpy as np


def calculate_steady_state_stp_variables(rate, u, tau_rec, tau_fac) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the steady-state variables x and u for a given firing rate."""

    if tau_fac > 0:
        exp = np.zeros_like(rate)
        mask = rate > 0.
        exp[mask] = np.exp(-1 / (rate[mask]*1e-3 * tau_fac))
        u_steady = u / (1 - (1-u)*exp)
    else:
        u_steady = u * np.ones_like(rate)

    if tau_rec > 0:
        exp = np.zeros_like(rate)
        mask = rate > 0.
        exp[mask] = np.exp(-1 / (rate[mask]*1e-3 * tau_rec))
        x_steady = (1-exp) / (1 -(1-u_steady)*exp)
    else:
        x_steady = np.ones_like(rate)
    
    return x_steady, u_steady

def calculate_effective_synapse_weight(rate, syn_weight, u, tau_rec, tau_fac):
    """Calculates the effective synaptic weight considering STP."""

    # Calculate steady-state variables
    x_steady, u_steady = calculate_steady_state_stp_variables(rate, u, tau_rec, tau_fac)

    # steady-state effective synaptic weight 
    return syn_weight * u_steady * x_steady
