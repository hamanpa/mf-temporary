"""
Normalisation of raw (YAML-level) network parameter dicts before validation.

Synapse type switching is a property of the parameter values, not of the code path,
so it lives here and is applied by `load_network_parameters` and by sweep workers alike.

Rules
-----
- Demotion: a `tsodyks_synapse` with `tau_rec == 0` becomes a `static_synapse` with
  `weight = weight * U` (NEST's `tsodyks_synapse` requires tau_rec > 0; tau_rec = 0 is the static baseline).
- Promotion: a `static_synapse` whose `syn_params` contain STP keys (e.g. set by a sweep) with
  `tau_rec > 0` becomes a `tsodyks_synapse` with `weight = weight / U`, so that the efficacy
  weight * U equals the static weight (inverse of demotion).
  `U` and `tau_rec` are required, `tau_fac` defaults to 0, and `tau_psc` is by definition the
  target neuron's `tau_syn_E` / `tau_syn_I` (chosen by the source neuron type).
  With `tau_rec == 0` the synapse stays static and the STP keys are dropped.
"""

import copy

STP_KEYS = ("U", "tau_rec", "tau_psc", "tau_fac")


def unshare(obj):
    """
    Recursively copies nested dicts/lists without preserving shared references.

    `yaml.safe_load` returns ONE object for every alias of an anchor (`*conn_ee`), so changing a
    value through one path would silently change it everywhere the anchor is used.
    (`copy.deepcopy` preserves such sharing, hence this helper.)
    """
    if isinstance(obj, dict):
        return {key: unshare(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [unshare(value) for value in obj]
    return copy.copy(obj)


def _demote_to_static(conn: dict) -> None:
    params = conn["syn_params"]
    conn["syn_type"] = "static_synapse"
    conn["syn_params"] = {
        "weight": params["weight"] * params["U"],
        "delay": params["delay"],
    }


def _promote_to_tsodyks(conn: dict, raw: dict, target_name: str, source_name: str) -> None:
    params = conn["syn_params"]
    missing = [key for key in ("U", "tau_rec") if key not in params]
    if missing:
        raise ValueError(
            f"Cannot promote static synapse {source_name} -> {target_name} to tsodyks_synapse: "
            f"missing {missing} (U and tau_rec must be given; tau_fac defaults to 0)."
        )
    if params["U"] <= 0:
        raise ValueError(f"Cannot promote {source_name} -> {target_name}: U must be > 0 (weight = weight / U), got {params['U']}.")

    source_type = raw["neurons"][source_name].get("neuron_type", "excitatory")
    target_neuron_params = raw["neurons"][target_name]["neuron_params"]
    tau_psc = target_neuron_params["tau_syn_E" if source_type == "excitatory" else "tau_syn_I"]
    if "tau_psc" in params and params["tau_psc"] != tau_psc:
        raise ValueError(
            f"Promotion of {source_name} -> {target_name}: tau_psc={params['tau_psc']} differs from the target's "
            f"synaptic time constant {tau_psc}. tau_psc is defined by the target neuron; do not set it."
        )

    conn["syn_type"] = "tsodyks_synapse"
    conn["syn_params"] = {
        "weight": params["weight"] / params["U"],
        "delay": params["delay"],
        "U": params["U"],
        "tau_rec": params["tau_rec"],
        "tau_psc": tau_psc,
        "tau_fac": params.get("tau_fac", 0.0),
    }


def normalize_synapses(raw: dict) -> dict:
    """
    Returns an un-shared copy of a raw network parameter dict with synapse types normalised
    (see module docstring). The input is not modified.
    """
    raw = unshare(raw)
    connectivity = raw.get("network", {}).get("connectivity", {})

    for target_name, sources in connectivity.items():
        for source_name, conn in sources.items():
            params = conn.get("syn_params", {})
            if conn.get("syn_type") == "tsodyks_synapse":
                if float(params.get("tau_rec", 1.0)) == 0.0:
                    _demote_to_static(conn)
            elif conn.get("syn_type") == "static_synapse" and any(key in params for key in STP_KEYS):
                if float(params.get("tau_rec", 0.0)) > 0.0:
                    _promote_to_tsodyks(conn, raw, target_name, source_name)
                else:
                    conn["syn_params"] = {"weight": params["weight"], "delay": params["delay"]}

    return raw
