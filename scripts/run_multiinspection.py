import os
import sys
from pathlib import Path
import csv
import copy
import json
import shutil
import hashlib
import subprocess
from itertools import product
import argparse
import yaml


# Setup repository paths dynamically
script_dir = Path(__file__).resolve().parent
repo_path = script_dir.parent
mean_field_path = repo_path / "MeanFieldTester"

if str(mean_field_path) not in sys.path:
    sys.path.append(str(mean_field_path))

from codes.controller.config import load_workflow_config
from codes.stimuli.loader import load_stimuli_config
from codes.network_params.loader import load_network_parameters

DELIMETER = ';'

def generate_param_combinations(inspected_params):
    """
    Generates flattened parameter names list and list of parameter value combinations.
    Handles single parameter key strings and tuple keys.
    """
    keys = list(inspected_params.keys())
    values = list(inspected_params.values())

    param_names = []
    for k in keys:
        if isinstance(k, tuple):
            param_names.extend(k)
        else:
            param_names.append(k)

    param_combinations = []
    for combo in product(*values):
        combo_list = []
        for k, v in zip(keys, combo):
            if isinstance(k, tuple):
                if isinstance(v, tuple) and len(k) == len(v):
                    combo_list.extend(v)
                else:
                    combo_list.extend([v] * len(k))
            else:
                combo_list.append(v)
        param_combinations.append(combo_list)

    return param_names, param_combinations


def generate_unique_sim_id(combination, existing_ids, length=8) -> str:
    """Generates an 8-character unique hash ID from a parameter combination."""
    combo_string = str(tuple(combination)).encode('utf-8')
    hash_str = hashlib.md5(combo_string).hexdigest()
    for i in range(0, len(hash_str) - length + 1):
        short_hash = hash_str[i:i + length]
        if short_hash not in existing_ids:
            existing_ids.add(short_hash)
            return short_hash
    raise ValueError("Unable to generate a unique simulation ID. Consider increasing hash length.")


def get_default_param_value(network_params, sim_params, stimuli_config, param_path: str):
    """Extracts default value for a parameter path from default config objects."""
    subpath = param_path
    if param_path.startswith("network."):
        subpath = param_path[len("network."):]
        obj = network_params
    elif param_path.startswith("workflow.") or param_path.startswith("sim."):
        prefix = "workflow." if param_path.startswith("workflow.") else "sim."
        subpath = param_path[len(prefix):]
        obj = sim_params
    elif param_path.startswith("stimulus.") or param_path.startswith("stimuli."):
        prefix = "stimulus." if param_path.startswith("stimulus.") else "stimuli."
        subpath = param_path[len(prefix):]
        parts = subpath.split('.', 1)
        if len(parts) == 2 and parts[0] in stimuli_config:
            obj = stimuli_config[parts[0]]
            subpath = parts[1]
        else:
            obj = stimuli_config[list(stimuli_config.keys())[0]]
    else:
        obj = network_params

    keys = subpath.split('.')
    curr = obj
    for k in keys:
        if isinstance(curr, dict):
            curr = curr[k]
        elif hasattr(curr, k):
            curr = getattr(curr, k)
        else:
            raise AttributeError(f"Could not find default parameter '{k}' in path '{param_path}'")
    return curr


def parse_val(val_str: str):
    """Parse string value from CSV row."""
    val_str = val_str.strip()
    if val_str.lower() == 'true':
        return True
    if val_str.lower() == 'false':
        return False
    try:
        val_float = float(val_str)
        if val_float.is_integer():
            return int(val_float)
        return val_float
    except ValueError:
        return val_str


def sync_param_combinations_csv(csv_path: Path, inspected_params, network_params, sim_params, stimuli_config):
    """
    Syncs param_combinations.csv:
    1. If CSV exists, reads header and existing rows.
    2. Backfills default values for any new parameter columns not present in existing header.
    3. Finds which combinations are already present in the CSV and skips them.
    4. Adds new missing combinations with newly generated 8-char hash IDs.
    5. Saves updated CSV and returns list of new (id, param_dict) jobs to submit.
    """
    param_names, param_combinations = generate_param_combinations(inspected_params)

    existing_header = []
    existing_rows = []
    existing_ids = set()

    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=DELIMETER)
            try:
                existing_header = next(reader)
                for row in reader:
                    if row:
                        existing_rows.append(row)
                        existing_ids.add(row[0])
            except StopIteration:
                existing_header = []

    if not existing_header:
        header = ['id'] + param_names
        all_rows = []
        new_jobs = []
        for combo in param_combinations:
            sim_id = generate_unique_sim_id(combo, existing_ids)
            row = [sim_id] + [str(v) for v in combo]
            all_rows.append(row)
            param_dict = dict(zip(param_names, combo))
            new_jobs.append((sim_id, param_dict))

        # Write new CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=DELIMETER)
            writer.writerow(header)
            writer.writerows(all_rows)

        return new_jobs, header, all_rows

    # Handle case where CSV already exists:
    existing_params = existing_header[1:]
    new_params_to_add = [p for p in param_names if p not in existing_params]

    # Backfill default values for existing rows if new parameter columns are added
    if new_params_to_add:
        print(f"New parameters detected in inspected_params: {new_params_to_add}")
        print("Backfilling default values for existing combinations in CSV...")
        updated_header = existing_header + new_params_to_add
        updated_existing_rows = []
        for row in existing_rows:
            new_defaults = [str(get_default_param_value(network_params, sim_params, stimuli_config, p)) for p in new_params_to_add]
            updated_existing_rows.append(row + new_defaults)
        existing_header = updated_header
        existing_rows = updated_existing_rows
        existing_params = existing_header[1:]

    # Build lookup of existing combination values
    existing_combos_lookup = set()
    for row in existing_rows:
        row_params = dict(zip(existing_params, [parse_val(v) for v in row[1:]]))
        # Key combination signature
        sig = tuple(str(row_params.get(p, "")) for p in param_names)
        existing_combos_lookup.add(sig)

    all_rows = list(existing_rows)
    new_jobs = []

    for combo in param_combinations:
        sig = tuple(str(v) for v in combo)
        if sig in existing_combos_lookup:
            continue

        sim_id = generate_unique_sim_id(combo, existing_ids)
        row_dict = {p: get_default_param_value(network_params, sim_params, stimuli_config, p) for p in existing_params}
        for p_name, p_val in zip(param_names, combo):
            row_dict[p_name] = p_val

        row = [sim_id] + [str(row_dict[p]) for p in existing_params]
        all_rows.append(row)
        existing_combos_lookup.add(sig)
        new_jobs.append((sim_id, row_dict))

    # Write updated CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=DELIMETER)
        writer.writerow(existing_header)
        writer.writerows(all_rows)

    return new_jobs, existing_header, all_rows


def submit_slurm_job(sim_id: str, results_dir: Path, script_dir: Path, dry_run=False, testing_mode=False):
    """Submits worker job to Slurm via sbatch."""
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    slurm_script = "\n".join([
        "#!/bin/bash",
        f"#SBATCH --error={log_dir / f'inspection_{sim_id}_%j.log'}",
        f"#SBATCH --output={log_dir / f'inspection_{sim_id}_%j.log'}",
        f"#SBATCH --job-name=inspection_{sim_id}",
        "#SBATCH --nodes=1",
        "#SBATCH --mem=62G",
        "#SBATCH --time=168:00:00",
        "#SBATCH --exclude=w[9,11,13-17]",
        "#SBATCH --cpus-per-task=16",
        "",
        "source /home/haman/virt_env/mf-csng/bin/activate",
        f"cd {script_dir}",
        f"python -u inspection_worker.py --id {sim_id} --project_dir {results_dir}" + (" --test" if testing_mode else ""),
    ])

    if dry_run or shutil.which("sbatch") is None:
        print(f"[DRY-RUN / Local] Would submit Slurm job for ID '{sim_id}':")
        print(f"  Command: python -u inspection_worker.py --id {sim_id} --project_dir {results_dir}")
        return None

    p = subprocess.Popen(
        ['sbatch'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = p.communicate(input=slurm_script)
    if p.returncode == 0:
        print(f"Submitted Slurm job for ID {sim_id}: {stdout.strip()}")
    else:
        print(f"Failed to submit Slurm job for ID {sim_id}: {stderr.strip()}")
    return stdout.strip()


def load_inspected_params(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    parsed_params = {}
    for key, val in data.get("inspected_params", {}).items():
        # 1. Convert comma-separated string keys into tuple keys
        if "," in key:
            key = tuple(k.strip() for k in key.split(","))
        
        # 2. Convert nested list values into tuples if they represent tuples
        if isinstance(val, list):
            # If it's a list of lists (e.g. [[0.6, 0.2], [0.8, 0.4]]), convert inner lists to tuples
            val = [tuple(item) if isinstance(item, list) else item for item in val]
            
        parsed_params[key] = val
            
    return parsed_params


def main():
    parser = argparse.ArgumentParser(description="Worker script for multi-inspection runs.")
    parser.add_argument('--test', action='store_true', help="Enable test mode to use test parameter files.")
    parser.add_argument("--project_dir", type=str, required=True, help="Path to project directory")

    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    results_path = project_dir
    os.chdir(results_path)

    params_save_dir = project_dir / "params"
    params_save_dir.mkdir(parents=True, exist_ok=True)

    data_save_dir = project_dir / "data"
    data_save_dir.mkdir(parents=True, exist_ok=True)

    imgs_save_dir = project_dir / "imgs"
    imgs_save_dir.mkdir(parents=True, exist_ok=True)

    # Load base parameters
    if args.test:
        network_params = load_network_parameters(params_save_dir / "test_network_params.yaml")
        sim_params = load_workflow_config(params_save_dir / "test_workflow_params.yaml")
        stimuli_config = load_stimuli_config(params_save_dir / "test_stimuli.yaml")
    else:
        network_params = load_network_parameters(params_save_dir / "network_params.yaml")
        sim_params = load_workflow_config(params_save_dir / "workflow_params.yaml")
        stimuli_config = load_stimuli_config(params_save_dir / "default_stimuli.yaml")

    # Define inspected parameters dictionary
    inspected_params = load_inspected_params(params_save_dir / "inspected_params.yaml")

    csv_path = project_dir / "param_combinations.csv"
    if args.test and csv_path.exists():
        csv_path.unlink()  # Remove existing CSV in testing mode to start fresh

    new_jobs, header, all_rows = sync_param_combinations_csv(
        csv_path=csv_path,
        inspected_params=inspected_params,
        network_params=network_params,
        sim_params=sim_params,
        stimuli_config=stimuli_config,
    )

    print(f"Total parameter combinations in CSV: {len(all_rows)}")
    print(f"New combinations to submit: {len(new_jobs)}")

    for sim_id, p_dict in new_jobs:
        submit_slurm_job(sim_id=sim_id, results_dir=results_path, script_dir=script_dir, testing_mode=args.test)


if __name__ == "__main__":
    main()
