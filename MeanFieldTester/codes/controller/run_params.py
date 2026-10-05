"""
Materialisation of per-run parameters for parameter sweeps.

A sweep run is defined by the base configs (YAML files) plus a set of dotted-path overrides
(one row of `param_combinations.csv`). The overrides are applied to the RAW YAML dicts, the
synapse types are normalised, and only then are the configs validated. The validated configs
are saved to `<run_dir>/params/` and loaded back from there, so what is saved is exactly what
is simulated.

Dotted paths
------------
`<prefix>.<key>.<key>...`, where the prefix selects the config:

- `network.`                -> network params
- `workflow.` / `sim.`      -> workflow params
- `stimulus.` / `stimuli.`  -> stimuli. If the first key is a stimulus name, only that stimulus
                               is changed; otherwise the change applies to all stimuli.
- no known prefix           -> network params (legacy fallback)

Keys are the YAML keys. For aliased fields (e.g. MF `init_values`: `E` vs `exc_rate_mean`) use the
key written in the YAML file. A key that does not exist in the base config is an error, except a
synapse parameter (`...syn_params.<key>`), which may be added (static -> Tsodyks promotion).
"""

from pathlib import Path
from typing import Any

import yaml

from ..network_params.models import BiologicalParameters
from ..network_params.normalization import unshare
from ..network_params.loader import load_network_parameters
from ..stimuli.loader import load_stimuli_config
from .config import WorkflowConfig, load_workflow_config
from ..utils.file_helpers import save_yaml


CONFIG_FILES = {
    "network": "network_params.yaml",
    "workflow": "workflow_params.yaml",
    "stimuli": "default_stimuli.yaml",
}
TEST_CONFIG_FILES = {
    "network": "test_network_params.yaml",
    "workflow": "test_workflow_params.yaml",
    "stimuli": "test_stimuli.yaml",
}

_PREFIXES = {
    "network": "network",
    "workflow": "workflow",
    "sim": "workflow",
    "stimulus": "stimuli",
    "stimuli": "stimuli",
}


def base_config_paths(params_dir: str | Path, test: bool = False) -> dict[str, Path]:
    """Paths of the base config files in a project's `params/` folder (`test_*` variants if `test`)."""
    file_names = TEST_CONFIG_FILES if test else CONFIG_FILES
    return {key: Path(params_dir) / file_name for key, file_name in file_names.items()}


def load_raw_configs(params_dir: str | Path, test: bool = False) -> dict[str, dict]:
    """Loads the base configs as raw dicts with YAML anchors un-shared (see `unshare`)."""
    raw_configs = {}
    for key, path in base_config_paths(params_dir, test).items():
        with open(path, "r") as f:
            raw_configs[key] = unshare(yaml.safe_load(f))
    return raw_configs


def _split_prefix(path: str) -> tuple[str, str]:
    prefix, _, subpath = path.partition(".")
    if prefix in _PREFIXES and subpath:
        return _PREFIXES[prefix], subpath
    return "network", path


def _stimulus_targets(stimuli: dict, subpath: str) -> list[tuple[dict, list[str]]]:
    """Resolves a stimuli subpath into (stimulus dict, keys) pairs (one stimulus or all of them)."""
    keys = subpath.split(".")
    if len(keys) > 1 and keys[0] in stimuli:
        return [(stimuli[keys[0]], keys[1:])]
    return [(stimulus, keys) for stimulus in stimuli.values()]


def _parent_and_key(root: dict, keys: list[str], path: str, allow_new_key: bool) -> tuple[dict, str]:
    current = root
    for depth, key in enumerate(keys[:-1]):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Parameter path '{path}': key '{key}' not found (at depth {depth}).")
        current = current[key]
    last = keys[-1]
    if not isinstance(current, dict):
        raise KeyError(f"Parameter path '{path}': '{keys[-2]}' is not a mapping.")
    if last not in current and not (allow_new_key and len(keys) > 1 and keys[-2] == "syn_params"):
        raise KeyError(f"Parameter path '{path}': key '{last}' not found in the base config.")
    return current, last


def get_by_path(raw_configs: dict[str, dict], path: str) -> Any:
    """
    Returns the base value of a dotted parameter path.

    Returns None for a synapse parameter that is absent in the base config (e.g. `U` of a
    static synapse), meaning "not set". For stimuli paths without a stimulus name, the value
    of the first stimulus is returned.
    """
    config_key, subpath = _split_prefix(path)
    if config_key == "stimuli":
        root, keys = _stimulus_targets(raw_configs["stimuli"], subpath)[0]
    else:
        root, keys = raw_configs[config_key], subpath.split(".")
    parent, last = _parent_and_key(root, keys, path, allow_new_key=True)
    return parent.get(last)


def set_by_path(raw_configs: dict[str, dict], path: str, value: Any) -> None:
    """Sets a dotted parameter path in the raw configs (in place)."""
    config_key, subpath = _split_prefix(path)
    if config_key == "stimuli":
        targets = _stimulus_targets(raw_configs["stimuli"], subpath)
    else:
        targets = [(raw_configs[config_key], subpath.split("."))]
    for root, keys in targets:
        parent, last = _parent_and_key(root, keys, path, allow_new_key=True)
        parent[last] = value


def apply_updates(raw_configs: dict[str, dict], updates: dict[str, Any]) -> dict[str, dict]:
    """
    Returns a copy of the raw configs with the dotted-path updates applied.
    Values that are None or "" mean "not set" and are skipped (base value kept).
    """
    raw_configs = unshare(raw_configs)
    for path, value in updates.items():
        if value is None or value == "":
            continue
        set_by_path(raw_configs, path, value)
    return raw_configs


def save_run_params(
        run_params_dir: str | Path,
        network_params: BiologicalParameters = None,
        workflow_params: WorkflowConfig = None,
        stimuli: dict = None,
        ) -> None:
    """Saves validated configs to `run_params_dir` under the standard file names (only those given)."""
    run_params_dir = Path(run_params_dir)
    for key, obj in (("network", network_params), ("workflow", workflow_params), ("stimuli", stimuli)):
        if obj is not None:
            save_yaml(obj, run_params_dir / CONFIG_FILES[key])


def load_run_params(run_params_dir: str | Path) -> tuple[BiologicalParameters, WorkflowConfig, dict]:
    """Loads (and validates) the configs saved by `save_run_params`."""
    paths = {key: Path(run_params_dir) / file_name for key, file_name in CONFIG_FILES.items()}
    return (
        load_network_parameters(paths["network"]),
        load_workflow_config(paths["workflow"]),
        load_stimuli_config(paths["stimuli"]),
    )


def materialize_run_params(
        params_dir: str | Path,
        updates: dict[str, Any],
        run_params_dir: str | Path,
        test: bool = False,
        ) -> tuple[BiologicalParameters, WorkflowConfig, dict]:
    """
    Builds, validates and saves the configs of one sweep run, then loads them back.

    Parameters
    ----------
    params_dir : str or Path
        Project `params/` folder with the base configs.
    updates : dict
        Dotted path -> value (one row of `param_combinations.csv`).
    run_params_dir : str or Path
        Where the validated configs are saved (e.g. `data/<sim_id>/params`).
    test : bool
        Use the `test_*` base configs.

    Returns
    -------
    tuple
        (network_params, workflow_params, stimuli) loaded from `run_params_dir`.
    """
    network_params, workflow_params, stimuli = build_run_params(load_raw_configs(params_dir, test), updates)
    save_run_params(run_params_dir, network_params, workflow_params, stimuli)
    return load_run_params(run_params_dir)


def build_run_params(
        raw_configs: dict[str, dict],
        updates: dict[str, Any],
        ) -> tuple[BiologicalParameters, WorkflowConfig, dict]:
    """
    Applies the updates to the raw configs and validates them (network synapse types are normalised
    inside `load_network_parameters`). Nothing is saved; raises on invalid combinations.
    """
    raw_configs = apply_updates(raw_configs, updates)
    return (
        load_network_parameters(raw_configs["network"]),
        load_workflow_config(raw_configs["workflow"]),
        load_stimuli_config(raw_configs["stimuli"]),
    )
