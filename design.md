# Design & Architecture

This file describes how the codebase is built and the rules it follows.
How to run things → `workflow.md`. Units table → `units.md`. Open work → `todo.md`.
Everything here was checked against the code (Oct 2026). If the code and this file disagree, fix one of them.

---

## 1. Purpose

MeanFieldTester (`MeanFieldTester/codes/`, imported as `codes`) tests mean-field (MF) models of AdEx E/I networks against the spiking network (SNN) they approximate. Its main focus is extending the Zerlaut 2018 / Di Volo 2019 MF models with Tsodyks–Markram short-term plasticity (STP).

```
network_params.yaml ─┐
stimuli.yaml ────────┼──► validated pydantic objects
workflow_params.yaml ┘
        │
 [1] neuron_simulation  single AdEx neuron on a (exc_rate, inh_rate) Poisson-input grid → SingleNeuronResults
        │
 [2] transfer_function  fit V_eff polynomial + erfc to the grid, per population, per MF model → BaseTransferFunction
        │                (the fitted coefficients are written back into the MF config, see §6.3)
        ├──────────────────────────┐
 [3a] snn_simulation (PyNN/NEST)   [3b] mf_simulation (TVB, one per entry of mf_models)
        │  → SNNResults              │  → MFResults
        └──────────┬───────────────┘
 [4] saved to .npz per (parameter combination × model × stimulus)
 [5] ResultsAggregator → analysis + plotting
```

---

## 2. Design principles

These are the rules the code follows. Known places where it doesn't are in §10.

- **Modularity / swappable backends.** Each stage hides its backend behind an abstract base class and a registry (§4). Changing the MF model or simulator is a config change, not a code change.
- **Open/closed (in practice).** A new backend is a new class plus one line in the stage's `*_REGISTRY`. The orchestrators never `match` on the backend name.
- **Single responsibility.** Config models hold and validate data. Simulators run. Results classes store data and serve it. Plots draw.
- **Data-driven, not "stringly typed".** Physical meaning comes from data, not from names. Examples:
  - whether a population is excitatory comes from `neuron_type`;
  - whether it is external comes from `neuron_model`/`is_external`;
  - units come from the `Field` description (§5).
- **One set of user-facing units.** Everything the user reads or writes (params, results, plot axes) uses MFT units. Conversions to simulator units happen only at module boundaries.
- **Separation of concerns.** Simulation, analysis and plotting are separate. Plots ask results objects for values and do no physics.
- **Stateless orchestration.** Workflow functions return results or write them to disk. They keep no data between calls.
- **YAGNI.** Implement what the current research needs. Extension points exist (registries), but speculative backends are not written ahead of time.

---

## 3. Package layout

| Package | Role |
|---|---|
| `network_params/` | `BiologicalParameters`: the single description of neurons, populations, connectivity and synapses. Also `translators.py` (unit engine) and `mappings.py` (MFT→PyNN/NEST names and units). |
| `stimuli/` | Stimulus configs (`StimuliCollection`) and numpy rate profiles used by the SNN. |
| `neuron_simulation/` | Stage 1: single-neuron grid simulations. |
| `transfer_function/` | Stage 2: TF classes, fitting, and `MembranePotentialFluctuations` (the subthreshold μV, σV, τV calculator). |
| `snn_simulation/` | Stage 3a: network simulation and its parallel worker. |
| `mf_simulation/` | Stage 3b: TVB adapter, TVB model classes, TVB stimulus equations, parallel worker. |
| `data_structures/` | Results classes returned by every stage (§6). |
| `controller/` | `WorkflowConfig`, workflow orchestrators, `ResultsAggregator` (§7). |
| `analysis/` | Comparison metrics (`METRIC_REGISTRY`) and spike statistics. |
| `plotting/` | Plot classes and figure "hooks" (§8). |
| `utils/` | Array, dict, file, spike→rate and STP helpers. |

Outside the package:
- `scripts/` holds the cluster entry points (§7.1).
- `params/` holds template configs.
- `projects/NN_name/` holds one folder per study (params, data, imgs, logs, notebooks).

**Imports:** the package is not installed. Scripts and notebooks add `MeanFieldTester/` to `sys.path` and use `from codes.<module> import …`. Inside the package, all imports are relative.

---

## 4. Stage pattern

Each stage has the same structure:

```
<stage>/
  config.py     pydantic config; `execution_mode` is a discriminated union
  base.py       ABC defining the backend interface
  <backend>.py  concrete backend(s)
  __init__.py   *_REGISTRY + get_simulator()/get_transfer_function() factory + run_<stage>_workflow()
```

| Stage | Interface | Registry keys | `execution_mode` |
|---|---|---|---|
| neuron_simulation | `simulate(network_params, sim_params) → {neuron: SingleNeuronResults}` | `pynn.nest`, `zerlaut2018` | `run`, `load`, `try_load`, `skip` |
| transfer_function | functor: `fit(results)`, `set_fitted_parameters(dict)`, `required_inputs()`, `evaluate(**kw)`, `__call__` (checks it is fitted and has the inputs) | `neuropsi.custom`, `zerlaut2018`, `divolo2019` | `fit_transfer_function: true/false` |
| snn_simulation | lifecycle: `build_network → run_stimulus → end` | `pynn.nest` | `run` |
| mf_simulation | lifecycle: `build_network → run_stimulus → end` | `tvb` (model chosen by `model`, §6.4) | `run`, `skip` |

- `zerlaut2018`/`divolo2019` TFs and the `zerlaut2018` neuron simulator are ports of the original authors' code. They are kept to validate the new code (projects/01).
- `try_load` loads pickled neuron results if both paths exist. Otherwise it runs the simulation and saves the results.

---

## 5. Configuration and units

### 5.1 Config files
There are three YAML files per project, all validated by pydantic. YAML anchors (`&`/`*`/`<<:`) are used to share blocks.

- **`network_params.yaml` → `BiologicalParameters`**
  - `neurons`: name → `AdExDefinition` (internal) or `PoissonDefinition` (external, `is_external=True`).
  - `network.size`: population sizes, including external sources.
  - `network.connectivity`: nested **`{target: {source: ConnectionDefinition}}`**. Each `ConnectionDefinition` has `rule` (`fixed_prob`/`fixed_in`/`fixed_out`), `val`, and `syn_type` (`static_synapse` | `tsodyks_synapse`) with `syn_params`. Only internal populations can be targets.
  - The root validator attaches population sizes to each connection, so `conn_num` (K) and `conn_prob` (p) are both available whichever rule was used.
  - Derived properties: `internal_neurons`, `exc_neuron_name`, `inh_neuron_name`, `internal_size`, `g`.
- **`workflow_params.yaml` → `WorkflowConfig`**: `neuron_simulation`, `snn_simulation`, and `mf_models: {name: MeanFieldSimulationConfig}`. Each MF model carries its own `transfer_function` config, so models can differ in their TF.
- **`stimuli.yaml` → `{name: StimulusConfig}`**: discriminated on `pattern` (`NoStimulus`, `PulseTrain`, `Sinusoidal`, `TwoSidedGaussian`). Uses `extra="forbid"`, so typos are errors.

Sweeps add `inspected_params.yaml` (§7.1).

### 5.2 Units mechanism
- **MFT units** are the internal standard. Table in `units.md`.
- **Units live in the schema.** Every physical `Field` states its unit in its description, e.g. `description="Synaptic weight [nS]"`. `translate_params(model, mapping)` parses the `[unit]`, so a field with no unit cannot be converted.
- **Mappings** are dicts of `TranslationRule(mft_name, sim_unit)` per simulator. PyNN/NEST mappings are in `network_params/mappings.py`; TVB mappings are in `mf_simulation/tvb_simulator/models/factory.py`. Conversion happens only when building a simulator.
- **`get_unit_multiplier`** handles SI prefixes (k, m, u, n, p) over the bases V, A, F, S, s, Hz, rad, and powers like `Hz^2`. An empty string means unitless.
- **Results**: each results class stores data in its `DEFAULT_UNITS`. Backends declare non-default input units through `input_units={...}`; `_ingest` converts them on construction. Getters take an optional unit: `res.exc_rate_mean("kHz")`.

### 5.3 Naming conventions
- `rate`, not nu/activity/fr. `params`, not pars. `mean`/`std`, not mu/sigma, in the public API. Greek-letter names (`mu_V`, `sigma_V`, `T_V`) appear only inside formula code.
- **Two-letter projection codes are target-source.** `ei` is the projection *onto E from I* (I→E). This matches `W @ rate` and is used throughout: `ee/ei/ie/ii_conductance`, TVB `K_ei`, `Q_ei`, `X_ei`. The external sources are `d` (drive) and `s` (stimulus), e.g. `K_ed`.
- Population prefixes: `exc_` and `inh_`. Population names in configs: `exc_neuron`, `inh_neuron`, `drive_neuron`, `stim_neuron`.
- Variable/metric keys: `{variable}_{metric}`, e.g. `exc_rate_pop_mean` or `exc_voltage_time_std`.
  - `_all` means the raw per-neuron array, shape (time, neuron).
  - `_mean`/`_std` in getters are population statistics over time.
- snake_case everywhere. NumPy-style docstrings (most of the code).
- STP: only `tsodyks_synapse` is used. `tsodyks2_synapse` is avoided because of a NEST bug, investigated in projects/09 and reported upstream.

---

## 6. Data model and stage internals

### 6.1 Results classes (`data_structures/`)
`BaseResults` has four branches: `BaseSingleNeuronResults`, `BaseSNNResults`, `BaseMFResults`, `BaseInspectionResults`. Orchestrators and plots type-check against these bases, not against concrete classes.

- **Storage:** data is stored in private attributes (`_exc_rate_mean`) in default units and read through getter methods. Setting a field listed in `DEFAULT_UNITS` after construction raises an error (instances are frozen).
- **Derived quantities are computed lazily inside results objects:**
  - **SNN rates** are computed from spikes using the smoothing function configured in `snn_simulation.smoothing` (`histogram`, `sliding_window`, `alpha_window`).
  - **SNN STP variables** (`exc_x`, `exc_u`, …) are reconstructed from spike trains (`utils.snn_helpers.reconstruct_stp_dynamics`).
  - **SNN statistics** come from generic accessors `get_all`, `get_pop_mean`, `get_pop_std`, `get_time_mean`, `get_time_std`, `get_full_mean`. Workers rely on these.
  - **MF voltage and conductance** come from `transfer_function.MembranePotentialFluctuations`, applied to the simulated rates. This means `data_structures` depends on `transfer_function`.
  - **MF STP variables** fall back to the steady-state values for the current rate when the model has no dynamic STP state.
- **Persistence:** `BaseResults.save()` pickles. The sweep path saves plain `.npz` instead (§7.1).

### 6.2 Single neuron and TF fitting
- **The neuron grid** is 2D, indexed (exc_rate, inh_rate). It can be:
  - `linear` (meshgrid);
  - `custom` (array or `.npy`);
  - `adaptive`: for each inh rate, choose exc rates so the output rates land on `out_rate_grid`. The upper exc bound is found by doubling and bisection. A coarse scan is then simulated, and PCHIP interpolation gives the exc rates. Only an adaptive *exc* axis is implemented.
- **Multiprocessing:** the PyNN neuron simulator runs with `cpus > 1` as a process pool. Parameters are passed as plain dicts (`translate_params` output) so they pickle.
- **`NeuroPSICustomTF`** fits in two steps:
  1. Fit V_eff with SLSQP against the V_eff obtained by inverting erfc on the data.
  2. Fit the full TF output rate with Nelder–Mead.

  Step 1 uses only points with `out_rate_min < out_rate < out_rate_max`. Step 2 uses points with `out_rate < out_rate_max`.
- **Flags:** `square_terms`, `log_term`, `adaptation` (pass the measured adaptation to μV), and `static_synapses` (ignore STP when computing effective weights).
- **Expansion:** the polynomial is expanded around `expansion_point` and scaled by `expansion_norm`.
- **STP in the TF:** handled through effective weights (`utils.stp_helpers.calculate_effective_synapse_weight`, the steady-state u·x at the presynaptic rate).

### 6.3 TF → MF hand-off
`run_tf_fitting_workflow` writes the fitted coefficients into `mf_sim_params.transfer_function.tf_fits[neuron_name]`. In other words, it mutates the config object. The TVB factory then reads `tf_fits` and passes `P_e`/`P_i` to the model, converting to TVB units (×1e-3, mV→V).

Because of this, the TF must be fitted, or `tf_fits` loaded, **before** the same config object is sent to the MF simulation.

The TF formula exists **twice**: in `NeuroPSICustomTF` (used for fitting and plots) and inside each TVB model's `TF`/`threshold_func` (used during integration). The TVB copy hard-codes the expansion point/norm and the 10 polynomial coefficients without `P_log`.

### 6.4 MF models (TVB)
- **Adapter:** `TVBMFSimulator` is a single node (`grid_size=1`, enforced). It uses Heun stochastic integration; noise applies only to the `noise` state variable. It has a Raw monitor. For each stimulus it rebuilds `Simulator`, sets `external_input_*` from `drive_rate`, and attaches the stimulus as a TVB `StimuliRegion` on the `stimulus` state variable.
- **Factory:** `models/factory.setup_tvb_model` translates `BiologicalParameters` into model attributes. It handles two families:
  - **Legacy** (`zerlaut2018.*`, `divolo2019.*`, in `neuropsi_models.py`):
    - takes only the projections onto E and assumes I receives the same inputs;
    - converts Tsodyks weights to static weights as `weight·U`.
  - **STP** (`stp_asymptotic.*`, `stp_dynamic.*`, in `stp_models.py`):
    - takes the full projection set `K/Q/U/tau_rec/tau_fac` for `ee, ei, ed, es, ie, ii, id, is`, plus `N_e`/`N_i`;
    - `asymptotic` scales weights by the steady-state u·x at the current rate;
    - `dynamic` integrates X/Y/U_dyn per projection (`X_ee, X_ei, X_ie, X_ii`, …) as extra state variables.
- **Order:** `.first_order` has rates + adaptation. `.second_order` adds covariances `C_ee, C_ei, C_ii`, with corrections from numerical TF derivatives.
- **Initial values:** `init_values` in the config accept TVB names (`E`, `C_ee`, `W_e`, …) or MFT names (`exc_rate_mean`, …) through aliases. They are validated against the schema for the chosen model.

### 6.5 SNN
- PyNN `EIF_cond_exp_isfa_ista` populations for internal neurons, built from `BiologicalParameters`.
- External populations are `SpikeSourceArray`s, refilled for each stimulus with inhomogeneous Poisson spike trains. The trains come from the numpy rate profiles in `stimuli/`.
- Connections use NEST native synapses (`static_synapse`/`tsodyks_synapse`); the receptor type is taken from the source's `neuron_type`.
- Records `recorded_samples` neurons per population: spikes, v, w, gsyn_exc, gsyn_inh.
- **Decision: "clean slate" per stimulus.** Each stimulus gets a fresh kernel: build → run → `end()`, rebuilt with the same seed.
  - *Why:* PyNN/NEST `reset(t_flush=…)` desynchronises clocks and breaks `get_data()`. Clean slates also guarantee identical initial conditions (v, w, u, x) for every trial, so the SNN is an uncontaminated ground truth for the MF comparison.
  - *Rejected alternative:* one continuous run with blank gaps between stimuli. It's faster, but slow variables (τ_w = 500 ms, STP) carry over between stimuli and create order effects.
  - *Revisit if:* sequence-dependent effects become a research topic, or rebuild time becomes the bottleneck.

### 6.6 Stimuli
- **One config, two implementations:**
  - numpy `*RateProfile` classes in `stimuli/models.py` (SNN);
  - TVB `FiniteSupportEquation` subclasses in `mf_simulation/tvb_simulator/stimuli.py` (MF).

  The two must stay mathematically identical.
- Every profile ramps the drive (and stimulus offset) linearly over `initial_increase_duration`, and clips rates at ≥ 0.

---

## 7. Orchestration

### 7.1 Main path: multi-dimensional sweeps on SLURM
1. **`scripts/run_multiinspection.py --project_dir P [--test] [--override] [--cpus N]`**
   - Reads `P/params/{network_params,workflow_params,default_stimuli,inspected_params}.yaml`. `--test` switches to the `test_*` files.
   - Builds the Cartesian product of `inspected_params`. A comma-joined key moves several dotted paths together: one value is applied to all of them, or a list of tuples gives per-path values.
   - Syncs `P/param_combinations.csv` (`;`-separated, columns `id` + full dotted paths, 8-character md5 IDs):
     - new parameter columns are backfilled with defaults;
     - existing combinations are skipped unless `--override`.
   - Submits one `sbatch` job per new combination. It prints instead when `sbatch` is missing.
2. **`scripts/inspection_worker.py --id ID --project_dir P --cpus N`**
   - Loads the base configs and applies the row's values. Prefixes: `network.`, `workflow.`/`sim.`, `stimulus.`/`stimuli.`.
   - Converts any `tsodyks_synapse` with `tau_rec == 0` to a static synapse with `weight·U`.
   - Runs the neuron simulation (`try_load` caches it as `.pkl` in `data/`), saves `data/ID/{neuron}_results_steady_state.npz`, and plots it.
   - Fits the TFs for every MF model and plots them.
   - Calls `controller.run_unified_batch_parallel`: an `mp.Pool(cpus, maxtasksperchild=1)` over [stimuli] × [SNN + MF models], with BLAS threads pinned to 1. Each task writes `data/ID/{model}_results_{stim}.npz` with `{variable}_{metric}` keys, chosen by `snn_simulation.saved_variables/saved_metrics/saved_extra_keys`. Each task returns status metadata; a failed task is recorded, it does not crash the batch. A `manifest.json` is written.
3. **`controller.ResultsAggregator(P)`**
   - Loads the CSV into a parameter matrix.
   - Resolves short parameter aliases: any ordered sub-sequence of the dotted path, e.g. `exc_neuron.b`. An ambiguous alias is an error.
   - `get_results(variable, sim_name, stim_name, **filters)` stacks arrays across matching runs, with lazy cached `.npz` loading.
   - `analyze_parameter_grid` finds the varying parameters.

Project folder layout: `params/`, `param_combinations.csv`, `data/<id>/`, `imgs/<id>/`, `logs/`, `explore_results.ipynb`.

### 7.2 Other entry points
- **`controller.run_basic_workflow(network_params, stimuli, workflow_config)`**: an in-memory, single-process pipeline. It runs neuron simulation → TF fit per MF model → SNN per stimulus → each MF model per stimulus, and returns a dict of results objects. It writes nothing to disk. Useful in notebooks; the sweep path does not use it.
- **`controller.inspectors.ParameterInspector`** with `ModelSummaryExtractor`/`ModelComparisonExtractor` and the `Inspection*Results` classes: an earlier 1-D sweep design. **Obsolete**, replaced by §7.1, and **to be deleted** together with `BaseInspectionPlot`, `inspection_plots.py` and the `Inspection*` hooks. Obsolete code is deleted, not archived; git history keeps it.

---

## 8. Plotting

- **`BasePlot`** draws onto one `ax`. Its params come from a merged `DEFAULT_PARAMS` dict. Shared pre/post handling covers title, labels, limits, ticks, legend and grid; `x_unit`/`y_unit` are passed to the results getters and added to the axis labels.
- **Plot families by data input:** `BaseSingleNeuronPlot`, `BaseTransferFunctionPlot`, `BaseSNNPlot`, `BaseNetworkPlot`/`BaseNetworkHistogramPlot` (SNN and MF together), `BaseInspectionPlot` (obsolete path).
- **Hooks** build figures:
  - `GridFigureHook` takes a 2D grid of plots and passes each one the data its family needs, using `isinstance`.
  - Params are layered: defaults < `common_params` < `subplot_params`. `subplot_params` is keyed by plot class name or by `(row, col)`.
  - Ready-made hooks: `NeuronActivityHook`, `TransferFunctionPlottingHook`, `NetworkOverviewPlottingHook`, `NetworkHistogramPlottingHook`.
  - The hook call signature is the `BasicWorkflowHook` protocol in `controller/interfaces.py`.
- **Aggregator plots** (sweep path): `BaseAggregatorPlot.draw(ax, sim_id, aggregator)`, e.g. traces, heatmaps, rasters, neuron I/O curves. `AggregatorGridPlottingHook` lays them out on a 2D grid over two swept parameters (columns = x, rows = y), with optional filters on the others.

---

## 9. Parallelism and reproducibility

- **Three levels:**
  - SLURM job per parameter combination;
  - `mp.Pool` over stimulus × model inside a job;
  - `mp.Pool` over the neuron grid inside the neuron simulation.
- `run_multiinspection` refuses to run when `neuron_simulation.cpus` is larger than `--cpus`. SLURM requests `2·cpus` CPUs per task.
- **Seeds** come from the configs (`seed` in each stage). NEST, the TVB noise stream and the Poisson spike generation are all seeded explicitly.

---

## 10. Known deviations from these principles

Tracked in `todo.md`. Listed here so the design text is not mistaken for a description of every line of code.

- **Stringly typed:**
  - external populations are recognised by the name prefixes `drive`/`stim` (SNN), and the TVB factory looks up `"drive_neuron"`/`"stim_neuron"`;
  - several places use `"exc_neuron"`/`"inh_neuron"` directly instead of `exc_neuron_name`/`inh_neuron_name`.
- **Computation in plotting:** `AggregatorNeuronIOCurvePlotter` can fit a TF itself, and TF plots evaluate TFs.
- **Hidden side effect:** TF fitting writes into the MF config (§6.3).
- **Duplicated formulas:** the TF/μV formulas (TF class vs TVB models) and the stimulus profiles (numpy vs TVB) each exist twice, with no automated check that they agree.
- **Two exc/inh populations assumed:** E/I-specific code (`exc_neuron_name` raises unless there is exactly one of each, legacy MF models, results getters) assumes one E and one I population.
- **No automated tests** (`MeanFieldTester/tests/` is empty).
