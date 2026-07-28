#!/usr/bin/env python3
"""
CLI Script for executing parallel SNN and Mean Field network simulations across networks and stimuli.
"""

import sys
import os
import argparse
from pathlib import Path

# Setup repository path
script_dir = Path(__file__).resolve().parent
repo_path = script_dir.parent
mean_field_path = repo_path / "MeanFieldTester"

if str(mean_field_path) not in sys.path:
    sys.path.append(str(mean_field_path))

from codes.network_params.loader import load_network_parameters
from codes.stimuli.loader import load_stimuli_config
from codes.controller.config import load_workflow_config
from codes.controller.workflows import run_unified_batch_parallel


def main():
    parser = argparse.ArgumentParser(description="Parallel SNN & MF Network Simulation Batch Runner")
    parser.add_argument("--network", type=str, required=True, help="Path to network parameters file (JSON/YAML)")
    parser.add_argument("--stimulus", type=str, required=True, help="Path to stimulus config file (YAML)")
    parser.add_argument("--workflow", type=str, required=True, help="Path to workflow config file (YAML)")
    parser.add_argument("--output-dir", type=str, default="results/batch_run", help="Output directory for results and manifest")
    parser.add_argument("--run-types", nargs="+", choices=["snn", "mf"], default=["snn", "mf"], help="Simulation types to run")
    parser.add_argument("--cpus", type=int, default=None, help="Number of worker CPU processes")

    args = parser.parse_args()

    print(f"Loading network parameters from: {args.network}")
    net_params = load_network_parameters(args.network)

    print(f"Loading stimuli from: {args.stimulus}")
    stimuli = load_stimuli_config(args.stimulus)

    print(f"Loading workflow config from: {args.workflow}")
    workflow_cfg = load_workflow_config(args.workflow)

    print(f"Starting parallel batch execution in '{args.output_dir}'...")
    results = run_unified_batch_parallel(
        network_params_list=net_params,
        stimuli=stimuli,
        network_sim_params=workflow_cfg,
        output_dir=args.output_dir,
        cpus=args.cpus
    )

    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"\n[DONE] Batch complete: {successful}/{len(results)} tasks completed successfully.")


if __name__ == "__main__":
    main()
