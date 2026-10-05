# CLAUDE.md

Working guide for this repo. Keep it short and stable:
- architecture lives in `design.md`;
- tasks and known bugs live in `todo.md`;
- the units table lives in `units.md`;
- how to run things will live in `workflow.md` (not written yet).

Read the relevant doc before changing code. Don't re-derive what is written there.

## What this is
MeanFieldTester (`MeanFieldTester/codes/`, imported as `codes`) is PhD research code by one developer, possibly shared with NeuroPSI later. It tests mean-field (MF) models of AdEx E/I networks against the spiking network (SNN), which is the ground truth.

The pipeline: single neuron → transfer-function (TF) fit → SNN (PyNN/NEST) + MF (TVB) → `.npz` → `ResultsAggregator` → plots.

The active research is **MF models with Tsodyks–Markram STP**.

## Environment: read before running or editing anything
- **This checkout is the cluster's filesystem, mounted locally with sshfs.** `/home/pavel/academia/wintermute/mf-temporary` is the same directory as `/home/haman/mf-temporary` on wintermute.
  - Every edit is immediately live on the cluster.
  - GitHub is only a backup.
- **Code runs on wintermute (SLURM)**, in the venv `/home/haman/virt_env/mf-csng`. A sweep takes 1–2 h there, versus about half a day locally.
  - Don't run simulations here, and don't probe or trust the local venv (it is stale).
  - Verification is static. Never claim to have run code. When a run is needed, give the user the exact command or `sbatch` to run on the cluster.
- **Running and queued SLURM jobs import `codes/` when they start.** Editing library code while a sweep is queued or running changes what those jobs execute.
  - Ask before editing `MeanFieldTester/codes/` or `scripts/` if a sweep might be in flight.
  - Prefer a separate dev worktree for code changes.
- **Over sshfs, file scans are slow.** Never grep or find through `projects/*/data/` (05 alone has ~10k files). Exclude `data/`, `imgs/` and `logs/`.
- **The `nbstripout` git filter points to the cluster venv, so it fails locally.** Use `git -c filter.nbstripout.clean=cat -c filter.nbstripout.smudge=cat status`.

## Repo map
- `MeanFieldTester/codes/` is the **API**: stages (`neuron_simulation`, `transfer_function`, `snn_simulation`, `mf_simulation`), plus `network_params`, `stimuli`, `data_structures`, `controller`, `analysis`, `plotting` and `utils`. See `design.md` §3–4.
- `scripts/` holds **reusable runnable entry points** built on that API and meant for `sbatch`. The current main path is `run_multiinspection.py` → `inspection_worker.py`.
  - Simulations run from scripts. Notebooks are for small tests and for exploring results.
  - Logic too specific for `controller/` goes in `scripts/`.
- `params/` holds templates. It will later hold a standard small test network and the configs behind stable/published results.
- `projects/NN_name/` holds work-in-progress studies. Each has `params/` with fixed names (`network_params.yaml`, `workflow_params.yaml`, `default_stimuli.yaml`, `inspected_params.yaml`, plus `test_*` variants) and `param_combinations.csv`. Its `data/`, `imgs/` and `logs/` are generated and never committed.
- `drafts/` and `.notes/` are local and gitignored.

## Conventions (binding)
- **The design in `design.md` is binding for new code.** That means:
  - pydantic configs with discriminated unions;
  - stage = ABC + `*_REGISTRY` + `run_*_workflow`;
  - units declared as `[unit]` in each `Field` description;
  - results accessed only through unit-aware getters;
  - no computation in plots;
  - data-driven, not name-driven, logic.

  If the design is impractical for a change, **stop and discuss it with the user** before deviating.
- **Projection codes are target-source:** `ei` = onto E from I (I→E), matching `W @ rate`. External sources: `d` (drive), `s` (stimulus). In configs, connectivity is `{target: {source: ...}}`.
- **STP:** use `tsodyks_synapse` only, never `tsodyks2_synapse` (NEST bug, projects/09, reported upstream). NEST requires `tau_rec > 0`.
  - `tau_rec = 0` means "static baseline": demoted to static with weight·U.
  - Static + STP keys with `tau_rec > 0` is promoted to Tsodyks with weight/U.
  - This happens in `network_params/normalization.py`, applied by every loader.
  - STP variables are **per projection** (`ee_x`, `ei_u`, …). The SNN reconstruction follows NEST exactly; without facilitation, u = U.
- **Units:** MFT units are ms, Hz, mV, nA, nS, nF. The code must match `units.md`. Conversions happen only at simulator boundaries, through `TranslationRule` mappings.
- **Names:** `rate` (not nu/activity), `params` (not pars), `mean`/`std`, `exc_`/`inh_` prefixes, `{variable}_{metric}` keys, snake_case, NumPy-style docstrings.
- **Model roles:**
  - `stp_asymptotic.*` / `stp_dynamic.*` are the actively developed models.
  - Legacy `zerlaut2018.*` / `divolo2019.*` MF models, and the ported Zerlaut/Di Volo TFs and neuron simulator, are **comparison references**: keep them as close to the original code as possible.
  - `neuropsi.custom` is the TF for all new work.
  - Second order is the main comparison model, but **first-order models must keep working** (they are used in analyses).
- **Scope:** two internal populations (E, I) plus drive and stimulus, single MF node. A spatial grid of MF nodes (one per SNN tile) is a long-term goal. Don't add generality for it now, but don't make it harder.

## Process
- **Docs:** when code changes, update `design.md` (if architecture changes) and `todo.md` (tick, add, restate). Keep one source of truth: no TODO comments in code, they go in `todo.md`.
- **Deleting:** obsolete code is deleted, not archived. **Ask the user to confirm before deleting.**
- **Git:** the user commits and pushes. Don't commit unless asked. Working on `main` is fine; suggest a branch or worktree for larger refactors.
- **Configs:** the same seed is used across sweep combinations, deliberately (for now).
- **Known open problems:** see `todo.md`, starting with the (1) items. Don't assume features work just because a config option exists for them; several options are declared but not implemented (listed in `todo.md`).

## Quick facts that save re-reading code
- **Sweep:**
  - `python -u /home/haman/mf-temporary/scripts/run_multiinspection.py --project_dir <ABSOLUTE path> --cpus 16 [--test] [--override]`. A relative `--project_dir` breaks.
  - `inspected_params.yaml` keys are dotted paths with prefixes `network.`, `workflow.`/`sim.`, `stimulus.`. A comma-joined key moves several parameters together.
- **Results:**
  - `projects/X/data/<id>/{model}_results_{stim}.npz` (each with a `units` entry); `{neuron}_results_steady_state.npz` holds the neuron grid.
  - `data/<id>/params/` holds the validated configs the run used, including `tf_fits`.
  - Load them with `codes.controller.ResultsAggregator(project_dir)`: `get_results`, `get_units`, `load_run_params`, `load_transfer_functions`. See `projects/05_DiVolo-STP/explore_results.ipynb`.
  - Runs made before iteration 1 (2026-10-05) have no `params/`, no units, and NaN SNN STP.
- **Sweep runs** are built by `controller.run_params`: CSV values are applied to the raw YAML (dotted path = YAML keys), then normalised, validated and saved. The master validates every combination before submitting.
- **Checking `.npz` files locally** (no numpy here): read them with `zipfile` and parse the `.npy` headers.
- **TF → MF hand-off:** fitted coefficients are written into `mf_sim_params.transfer_function.tf_fits` (config mutation). The TVB models re-implement the TF internally, with a hard-coded expansion point and no `P_log` term.
- **Legacy MF models** read only the projections onto E.
- **Not used on the main path:** `ParameterInspector` and its extractors (obsolete, scheduled for deletion). `run_basic_workflow` is an in-memory API pipeline that the main path doesn't use.
