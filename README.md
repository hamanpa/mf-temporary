# mf-temporary: MeanFieldTester

Tools for testing **mean-field (MF) models** of AdEx E/I networks against the **spiking network (SNN)** they approximate. The focus is extending the Zerlaut 2018 / Di Volo 2019 MF models with Tsodyks–Markram **short-term plasticity (STP)**.

The pipeline runs from a single network description:

```
single-neuron simulation → transfer-function fit → SNN (PyNN/NEST) + MF (TVB) simulations → saved results → analysis & plots
```

Most runs are parameter sweeps on the **wintermute** SLURM cluster: one job per parameter combination, with results collected afterwards by `ResultsAggregator`.

## Repository layout

```
MeanFieldTester/codes/   the package (imported as `codes`); module overview in design.md §3
scripts/                 cluster entry points: run_multiinspection.py (submits a sweep), inspection_worker.py (one combination)
params/                  template config files (network, workflow, stimuli; test_* variants)
projects/NN_name/        one folder per study: params/, param_combinations.csv, data/, imgs/, logs/, notebooks
drafts/                  local scratch notebooks (gitignored)
```

## Documentation

| File | Contents |
|---|---|
| [design.md](design.md) | Architecture, design principles, conventions (naming, target-source projection codes), known deviations |
| [workflow.md](workflow.md) | How to run things: configs, sweeps, results, plotting *(being written)* |
| [units.md](units.md) | Internal (MFT) units and conversions to PyNN/NEST/TVB *(being rewritten)* |
| [todo.md](todo.md) | Research and code tasks, with priorities |
| `notes.md` | Research notes and reading *(planned)* |

## Environment

The code runs on the wintermute cluster, in the venv `mf-csng` (`/home/haman/virt_env/mf-csng`).

**Required by MeanFieldTester:**
- Python ≥ 3.10 (the code uses `match` and `X | Y` type hints);
- **pydantic v2**;
- numpy, scipy, matplotlib, PyYAML, numba;
- PyNN with the **NEST 3.4** backend;
- TVB (`tvb-library`).

**Used only by drafts, notebooks or mozaik data loading:** mozaik, Brian2, jupyter, sympy, moviepy.

How the venv was set up:
1. Follow the [mozaik](https://github.com/csng-mff/mozaik) installation instructions: pip packages, Imagen, PyNN, NEST 3.4, mozaik.
2. Add the extra packages:
   ```bash
   pip3 install pydantic jupyter sympy brian2 moviepy tvb-library
   ```

Pinned versions (`requirements.txt` / `pyproject.toml`) are a task in `todo.md`.

The package is not installed. Scripts and notebooks add `MeanFieldTester/` to `sys.path` and import `codes.*`.

## Quick start: run a sweep

Full instructions will be in `workflow.md`. In short:

1. Create `projects/NN_name/params/` containing `network_params.yaml`, `workflow_params.yaml`, `default_stimuli.yaml` and `inspected_params.yaml`. Copy them from `params/` or from an existing project.
2. Submit the master script from the project folder. **Use an absolute `--project_dir`.** Existing projects do this in `run_multiinspection.sbatch`:
   ```bash
   python -u /home/haman/mf-temporary/scripts/run_multiinspection.py --project_dir /home/haman/mf-temporary/projects/NN_name --cpus 16
   ```
   Options:
   - `--test` uses the `test_*.yaml` configs;
   - `--override` resubmits combinations that already exist in `param_combinations.csv`.
3. Explore the results with `codes.controller.ResultsAggregator`, as in `projects/05_DiVolo-STP/explore_results.ipynb`.

## Cluster notes (wintermute)

- Small nodes: `#SBATCH --exclude=w[9,11,13-17]` (the worker jobs use this).
- Big nodes: `#SBATCH --exclude=w[1-8,10,12]`.
- `.git/config` has an `nbstripout` filter that points to the cluster venv. On other machines it fails, and `git status` can then look misleading.
