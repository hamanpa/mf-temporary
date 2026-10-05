# TODO

Task template:
```
- [ ] (Priority) **module.part**: *name of the task*
  - additional info
```
Keep the task line short. Details go in the sub-bullets.

Priority
- **(1) Critical/Blocker:** prevents the code from running or gives wrong results; or an architectural dependency for other work.
- **(2) Essential/High:** needed for the next stage of the research.
- **(3) Important/Medium:** non-blocking; usability, code quality, supplementary data.
- **(4) Nice-to-have/Low:** polish, optimisation, experimental ideas.

Last full review against the code: 2026-10-04.

---

# ACTIVE

- [ ] (3) **docs**: *Rewrite the documentation set*
  - Done: `design.md`, `todo.md`, `README.md`, `CLAUDE.md`. `MeanFieldTester/README.md` was deleted; its content was migrated.
  - When `workflow.md` exists, update `CLAUDE.md` (Quick facts → point to it) and `README.md` (quick start).
  - Next: `units.md` (include the PyNN `EIF_cond_exp_isfa_ista` units reference: https://pynn.readthedocs.io/en/latest/reference/neuronmodels.html), `workflow.md` (how to run things, and what can be run), `notes.md`.

---

# Todos (research)

> Statuses below were inferred from the project folders on 2026-10-04 and have not been confirmed. Update them when you return to each topic.

- [ ] (2) **research.stp**: *Analyse the DiVolo-STP sweep (projects/05)*
  - 160 combinations of τ_rec (E-source × I-source, 0–600 ms), U = 0.6, τ_fac = 0.
  - What changes in the network when STP is introduced? Where do MF and SNN disagree?
  - Is Di Volo's statement (TF coefficients are independent of the adaptation parameters) still true with STP?
- [ ] (2) **research.stp**: *Facilitation*
  - All sweeps so far use τ_fac = 0. Inhibitory synapses are typically facilitating.
- [ ] (2) **research.stp**: *Choose a working drive rate, then run the non-spontaneous stimuli*
  - Inspect `drive_rate` first. Then run PulseTrain / Sinusoidal / TwoSidedGaussian at a reasonable drive.
- [ ] (2) **research.divolo**: *Finish replicating Di Volo 2019 (static synapses)*
  - Sanity check: the new STP models must reproduce it with STP off. Projects 02 and 08.
  - Fig. 1 is done (projects/02 README). Fig. 2 (spontaneous activity) is still open.
  - Verify the paper's statement about fitting with b = 0.
- [ ] (2) **research.tf_fitting**: *Voltage fitting in the suprathreshold regime*
  - Theoretical μV and σV are subthreshold quantities. Above about 5 Hz output, the measured μV saturates near −53 mV because of reset, while the theoretical value keeps growing (projects/02).
  - Decide which range of membrane-fluctuation values to fit on.
- [ ] (2) **research.dynamics**: *DiVolo-STP explosions*
  - Explosions show up as high activity in the first ~1000 ms. Does the explosion return once adaptation decays (longer runs)?
  - See also the analysis tasks: explosion and steady-state detection.
- [ ] (3) **research.csng**: *CSNG setups, static and STP (projects 06, 07)*
  - Only one combination each has been run so far.
- [ ] (3) **research.tvb**: *Minimal example of the TVB direct-stimulus behaviour*
  - Related: the stimulus weight `10.` HACK in `TVBMFSimulator.setup_stimulus`, and the `direct_stimulation` option.
- [ ] (4) **research.csng**: *CSNG with non-homogeneous connectivity*
  - Needs a multi-node MF (see mf_simulation).

---

# Todos (code)

## Blockers / wrong results

- [ ] (1) **projects**: *Update the project YAMLs to the iteration-1 conventions (user)*
  - `snn_simulation.saved_variables`: `exc_x/exc_u/inh_x/inh_u` are now rejected at load. Use per-projection names `ee_x, ee_u, ei_x, ei_u, ie_x, ie_u, ii_x, ii_u` (and `*_y` if wanted). Affects projects 05 and 06.
  - `stp_dynamic` `init_values`: set `U_e`/`U_i` (U_dyn) to `[0.0, 0.0]`. The model now uses u = U when τ_fac = 0 regardless, but the init still matters when τ_fac > 0.
  - The commented examples in `inspected_params.yaml` still use `network.synapses.*`.
- [ ] (2) **research.stp**: *Re-run the `stp_dynamic` results with τ_rec > 0 (projects 05, 07)*
  - Until iteration 1, `stp_dynamic` used u = 1 instead of U whenever τ_fac = 0 (U_dyn initialised to 1 and never decaying). The efficacy was too strong: +17% to +67% depending on rate and τ_rec, with too much depletion.
  - Unaffected: SNN, `divolo2019`, `stp_asymptotic`, TF fits, τ_rec = 0 combinations, fully static networks.
  - Also unaffected by the fix, but missing in old data: SNN STP variables (they were NaN), `stp_dynamic` x/u (they were steady-state values), and per-run params, `tf_fits` and units (not saved before).
- [ ] (3) **research.stp**: *Check the effective-synapse steady state at r·τ_rec ≈ 1 (a few Hz)*
  - `stp_asymptotic` and the TF effective weights (`utils.stp_helpers`) use the regular-spike-train steady state, x* = (1 − e)/(1 − (1 − U)e) with e = exp(−1/(r·τ_rec)).
  - This is validated for many Poisson synapses at 15/30/80 Hz, τ_rec = 450 ms (projects/09 `tsodyks_inspection_notebook.ipynb`), i.e. r·τ_rec ≈ 7–36.
  - For one Poisson synapse the exact mean is x* = 1/(1 + U·r·τ_rec) (averaging e^(−Δ/τ_rec) over exponential intervals). The two converge for r·τ_rec ≫ 1, but differ by up to ~20% at r·τ_rec ≈ 0.5–2 (U = 0.6). That is ~1–10 Hz for τ_rec 200–600 ms, the spontaneous regime.
  - Open question: does the mean over many i.i.d. Poisson synapses approach the regular-train value? Check: repeat the notebook comparison at 1–10 Hz.

## mf_simulation

- [ ] (2) **mf_simulation.tvb_simulator.models**: *The TF inside the TVB models ignores the TF config*
  - `threshold_func` hard-codes the expansion point/norm (−60, 10, 4, 6, 0.5, 1) and uses 10 coefficients without `P_log`.
  - A fit with a different `expansion_point`/`expansion_norm`, or with `log_term: true`, is silently evaluated wrongly in the MF.
  - Minimum fix: validate in `setup_tvb_model` and raise an error. Better fix: pass the expansion parameters and `P_log` to the model.
- [ ] (2) **mf_simulation.tvb_simulator**: *Drive ramp missing in the MF*
  - The SNN ramps the drive over `initial_increase_duration`. The MF sets `external_input_*` to a constant `drive_rate` from t = 0 (and `MFResults.drive_rate_mean` is constant too).
  - This affects comparisons that include the first ~400 ms (the default `time_average_window` starts at 0).
  - Avoid packing the drive into the `stimulus` state variable: drive and stimulus can have different targets once there is a grid.
- [ ] (2) **mf_simulation**: *Test the first-order models*
  - First-order models are used in analyses, so they must be maintained, not just second order.
  - Only `stp_*.first_order` was run, once, in projects/04. `divolo2019.first_order` has never been run.
  - Iteration 1 fixed a crash when building results (`np.sqrt(None)` on the missing `C_ee`/`C_ii`), so they can't have worked in the sweep path before. Still untested on the cluster.
- [ ] (3) **mf_simulation**: *Single source of truth for the TF formula*
  - It is implemented twice: `NeuroPSICustomTF`/`MembranePotentialFluctuations`, and `get_fluct_regime_vars`/`TF` in the TVB models.
  - At least add a test that evaluates both with the same coefficients.
- [ ] (3) **mf_simulation.tvb_simulator.models**: *Legacy models take only the projections onto E*
  - `zerlaut2018.*`/`divolo2019.*` assume I receives the same inputs as E. Document this, or raise an error when the connectivity is asymmetric.
- [ ] (4) **mf_simulation.tvb_simulator**: *Make connectivity/coupling/integrator/monitors configurable*
  - All four are hard-coded: single node, `Linear(a=0.3)`, Heun stochastic, Raw monitor.
  - A multi-node grid (`grid_size > 1`) is a prerequisite for CSNG non-homogeneous connectivity.
  - Long-term goal: one MF node per spatial tile of a spatially distributed SNN (TVB nodes). This will need non-trivial extensions to `data_structures` (a node dimension in results and npz files).
- [ ] (4) **mf_simulation.config**: *Remove or implement the `load` mode and the `custom.neuropsi` ModelType*
  - `load` raises `NotImplementedError`. `custom.neuropsi` is in the enum but has no registry entry.

## snn_simulation

- [ ] (2) **snn_simulation.pynn_simulator**: *Drive and stimulus spike trains share one RNG stream*
  - `_generate_nhpp_spikes` creates `default_rng(seed)` on every call, so the drive and stimulus populations get correlated spike trains.
  - Use one generator per simulator, or derive a per-population seed.
- [ ] (3) **snn_simulation**: *`n_runs` is ignored*
  - The config field exists, but only one run is made. Implement it (seed per run, mean over runs) or remove it.
- [ ] (3) **snn_simulation.config**: *Smoothing options do not match the implementation*
  - The config allows `sliding_window | gaussian`. `SNNResults` implements `histogram | sliding_window | alpha_window`.
- [ ] (4) **snn_simulation.config**: *Configurable recorders*
  - The number of recorded neurons is configurable (`recorded_samples`). The recorded variables are hard-coded (`spikes, v, w, gsyn_exc, gsyn_inh`).
- [ ] (4) **snn_simulation**: *Remove or implement the `load`/`skip` modes*
  - The config accepts them, but `run_snn_simulation_workflow` only handles `run`.

## neuron_simulation

- [ ] (3) **neuron_simulation**: *Compute `voltage_tau`*
  - It is currently zeros (pynn_simulator.py:258, :279).
- [ ] (3) **neuron_simulation**: *Execution mode `validate`*
  - Compare stored neuron data with a fresh simulation.
- [ ] (3) **neuron_simulation**: *Unclear: "weird results" in projects/04_debug*
  - The data is in `projects/04_debug`. Check whether this is still relevant; if not, delete the project.
- [ ] (4) **neuron_simulation.pynn_simulator**: *Replace `legacy_neuron_params`*
  - Now built from `BiologicalParameters`, but still passed around as a hand-made dict with hard-coded keys.
- [ ] (4) **neuron_simulation**: *Adaptive grid for inhibitory rates*
  - Only an adaptive *exc* axis is implemented; the inh axis raises `NotImplementedError`. The interpolation roles of the axes are swapped.
- [ ] (4) **neuron_simulation**: *Share grid resolving between simulators*
  - `PyNNSimulator.resolve_grid` and the Zerlaut simulator each implement it.
- [ ] (4) **neuron_simulation**: *Allow neuron models other than AdEx*
  - `EIF_cond_exp_isfa_ista` is hard-coded, and `neuron_model` only accepts `adex`.

## transfer_function

- [ ] (3) **transfer_function.neuropsi_tf**: *`MembranePotentialFluctuations.voltage_tau` mutates its input*
  - `rates[neuron_name][~mask] = 1e-9` writes into the caller's arrays.
- [ ] (3) **transfer_function**: *Make the TF → MF hand-off explicit*
  - `run_tf_fitting_workflow` writes the coefficients into `mf_sim_params.transfer_function.tf_fits` as a hidden side effect.
  - Return the coefficients instead, and have the caller put them into the MF config. (Since iteration 1 the worker also saves them in `data/<id>/params/workflow_params.yaml`.)
- [ ] (4) **transfer_function**: *Update the Zerlaut2018/DiVolo2019 TF ports to the `connectivity` format*
  - `_get_legacy_params_dict` reads the removed `network_params.synapses`, so they crash.
  - They are comparison references (projects/01 validated our implementation against them, and that check may need repeating), so keep them close to the original code. Same treatment as the legacy MF models. Not urgent.
  - Once the TF input grid has a drive axis: these ports (and the `zerlaut2018` neuron simulator) only support drive = 0. Their wrapper should take the drive = 0 slice of the 3-D grid, and raise if the grid has no drive = 0 value (the Zerlaut simulator should raise if a non-zero drive grid is requested).
- [ ] (4) **transfer_function**: *Rename `run_tf_fitting_workflow` or the module*
  - The workflow function also loads fits; it doesn't only fit.

## data_structures / storage

- [ ] (3) **data_structures**: *Unit rescaling for all results*
  - Unit-aware ingestion and getters exist for some results and simulators only. Elsewhere the units are hard-coded (e.g. the TVB `run_stimulus` `input_units`, `MFResults._conductance_mean` "draft" warning, the plot-side assumptions). Make every results class and backend go through `DEFAULT_UNITS` + `input_units`, consistent with `units.md`.
- [ ] (3) **storage**: *Stop using pickle for the neuron-results cache*
  - `try_load` caches `SingleNeuronResults` as `.pkl`, which is brittle when classes are renamed. The worker already writes `{neuron}_results_steady_state.npz`, so load from that instead.
- [ ] (4) **data_structures**: *Rename `_mean`/`_std` getters to `_pop_mean`/`_pop_std`*
  - This makes the names unambiguous next to `_time_mean`, and the npz keys already use the `_pop_` form.

## controller / scripts

- [ ] (2) **analysis**: *Compute SNN-vs-MF comparison metrics in the sweep path*
  - `analysis/comparison_metrics.METRIC_REGISTRY` (RMSE, Pearson, lag, PSD, …) is only used by the obsolete `ParameterInspector`. The main path saves raw traces and computes no errors.
  - Add a helper on top of `ResultsAggregator`.
- [ ] (3) **analysis**: *Detect explosions and steady state*
  - Flag runaway activity (e.g. high rate in the first 1000 ms) and check whether a steady state was reached before time averaging.
- [ ] (3) **controller**: *Move `ResultsAggregator` into its own module*
  - It currently lives in `controller/inspectors.py` next to obsolete code.
- [ ] (3) **scripts**: *Remove duplicated helpers*
  - `parse_val`, `normalize_val` and `DELIMETER` (sic) are copied in `run_multiinspection.py`, `inspection_worker.py` and `ResultsAggregator`.
- [ ] (4) **controller.config**: *Template and schema generation*
  - `--template` and `--schema` raise `NotImplementedError`, and the code after the `raise` is dead. Either implement them (`WorkflowConfig.model_json_schema()` is nearly free) or delete them.

## plotting

- [ ] (3) **plotting**: *No computation inside plots*
  - `AggregatorNeuronIOCurvePlotter` can fit a TF itself (`_fit_tf_funcs`), and it refits from the project's *base* YAMLs, not from the run's parameters (wrong for swept τ_rec with STP TFs).
  - Use `ResultsAggregator.load_transfer_functions(sim_id, mf_model)` instead: the run's own fitted TFs, no refitting. This only works for runs made after iteration 1.
- [ ] (3) **plotting**: *Handle missing data gracefully*
  - `None` when a variable wasn't measured, `None` instead of a results object when a run was skipped, and NaN arrays when an MF field is missing. See `.notes/none_data_handling.md`.
- [ ] (3) **plotting**: *Diagnostic plots for the TF approach*
  - Theoretical vs measured μV, σV, τV, μG, σG, and V_eff. Fitting-step plots: Di Volo eq. (10)–(11).
- [ ] (4) **plotting**: *Overview figures that adapt to the recorded variables*
  - Skip panels for variables that weren't measured.

## codebase

- [ ] (2) **codebase**: *Delete obsolete code (delete, don't archive)*
  - `ParameterInspector`, both extractors, `inject_pydantic_param`, `INSPECTION_PARMAS_WITHOUT_UPDATE`.
  - `data_structures/inspection.py`, `plotting/inspection_plots.py`, `BaseInspectionPlot`, `InspectionWorkflowPlottingHook`, `ModelSummary`/`ModelComparisonInspectionPlottingHook`, `InspectionWorkflowHook`.
  - The `run_snn_batch_parallel`/`run_mf_batch_parallel` stubs and their commented-out bodies.
  - `utils/result_helpers.py` and `compare_mf_snn_results` in `utils/__init__.py` (used only in a projects/02 notebook).
  - `TVBMFSimulator._create_gaussian_connection_matrix`, and the commented-out blocks in `inspection_worker.py`.
  - Decide whether `run_basic_workflow` stays (the sweep path doesn't use it).
- [ ] (3) **codebase**: *Remove hard-coded population names*
  - `"exc_neuron"`/`"inh_neuron"` are used directly (SNN `run_stimulus`, TF `evaluate`, `MFResults`, plotting `NEURON_NAMES`, the worker). External populations are recognised by the name prefixes `drive`/`stim`.
  - Use `exc_neuron_name`/`inh_neuron_name`, and give external populations an explicit role (drive or stimulus) in the config.
- [ ] (3) **codebase**: *Move in-code TODO/HACK comments here (single source of truth)*
  - There are 17 in total. Most are covered above. Remaining: pynn_simulator.py:77 (direct external input) and :110 (units), and neuropsi_models.py:272 (P_e/P_i array size).
- [ ] (3) **codebase**: *Basic tests*
  - `MeanFieldTester/tests/` is empty.
  - Start with:
    - unit conversion (`get_unit_multiplier`, `translate_params`);
    - config loading of the `params/` templates;
    - TF class vs TVB TF consistency;
    - numpy vs TVB stimulus profiles.
  - Add a dummy-results generator so plots can be tested without running simulations.
- [ ] (3) **codebase**: *Pinned requirements*
  - Create `requirements.txt` from the cluster venv (`/home/haman/virt_env/mf-csng/bin/pip freeze`). Separate what MeanFieldTester needs (pydantic v2, numpy, scipy, matplotlib, PyYAML, numba, PyNN, NEST 3.4, tvb-library) from what only drafts use (mozaik, Brian2, …).
  - Then update the Environment section of `README.md`.
- [ ] (4) **codebase**: *Packaging*
  - Add a `pyproject.toml` and install with `pip install -e`, so the `sys.path.append` hacks in scripts and notebooks can go.
- [ ] (3) **codebase**: *Make the repo presentable for NeuroPSI*
  - Commit projects 04–09: params, notebooks, scripts.
  - Never commit generated results. `.gitignore` currently covers `*data/`, `logs/` and `*.png`; add `projects/*/imgs/`. Note that `*.sbatch` is ignored, so the project launchers are not in git — decide whether that is intended.
  - Clean the stale config comments (e.g. `zerlaut2018_simulator`/`divolo2019_simulator` in `workflow_params.yaml`).
- [ ] (3) **params**: *Make root `params/` the home of standard configs*
  - It should hold a small, fast standard test network (move the `test_*` variants here from the projects) and the configs behind stable/published results.
  - `projects/*/params` stay as work in progress. Broader test setups go in `MeanFieldTester/tests/` once testing exists.
- [ ] (4) **codebase**: *Logging instead of `print`*
- [ ] (4) **codebase**: *Unify docstrings (NumPy style)*
  - Priority: the workflow functions, `run_unified_batch_parallel`, the workers, and `ResultsAggregator`.
- [ ] (4) **codebase**: *Tutorial notebook*
  - `projects/05_DiVolo-STP/explore_results.ipynb` is a good base for the aggregator part. Also add a notebook showing what each comparison metric captures, visually.
- [ ] (4) **codebase**: *Decide on the neuron_simulation API*
  - SNN and MF use `build_network → run_stimulus → end`. Neuron simulation uses `simulate()` because NEST objects can't be pickled for the grid multiprocessing. Keep it, unless `build/run/end` can live inside each worker.

---

# Ideas

- [ ] **analysis**: *Network state classification*
  - `analysis/spike_metrics` already has synchrony and regularity. Save them in the sweep and use them to label AI / UP-DOWN states.
- [ ] **analysis**: *Phase-plane / fixed-point / nullcline analysis of the MF models*
- [ ] **plotting**: *"Style plot" system*
  - One generic plot class with styles (line, errorbar, fill_between) instead of one class per quantity.
- [ ] **plotting**: *Default axis units from the results' `DEFAULT_UNITS`*
  - Not from variable names, to avoid stringly typed code.
- [ ] **snn_simulation**: *Continuous-epoch stimulation*
  - Only if sequence effects become a topic. See the "clean slate" decision in `design.md` §6.5.
- [ ] **research**: *QIF neuron models with STP* (Montbrió-type / Helmut Schmidt)
- [ ] **controller**: *Seed strategy for sweeps*
  - Currently the same seed is used for every combination (common random numbers). Consider several seeds per combination to estimate SNN variability.

---

# Notes (to move to notes.md)

- **STP biology:** inhibitory synapses are often facilitating (PV interneurons are mentioned; SST may be depressing). Excitatory synapses are depressing. Look into ISN + STP.
- **Reading backlog:** NeuroPSI MF papers; Tsodyks–Markram (network-level effects); ISN and STP; QIF neurons and STP (Helmut Schmidt).

---

# DONE

Iteration 1 (2026-10-05). Checked on the cluster with projects/10_iteration1_check, except the `U_dyn` fix, which was applied after that run:
- [x] (1) **network_params**: *Strict synapse models*: `extra="forbid"` plus a `syn_type` ↔ `syn_params` check (a Tsodyks dict could validate as static and drop STP).
- [x] (1) **network_params**: *Synapse normalisation*: demotion (τ_rec = 0 → static, w·U) and promotion (static + STP keys → Tsodyks, w/U), in `load_network_parameters` and in sweeps; YAML anchors un-shared.
- [x] (1) **controller/scripts**: *Run materialisation*: the worker applies CSV values to the raw YAML, normalises, validates, saves `data/<id>/params/` and runs from it; `tf_fits` re-saved after fitting. The master validates all new combinations before submitting; new CSV columns default to "" when not set. A relative `--project_dir` works.
- [x] (1) **data_structures**: *STP per projection* (`ee_x`, `ei_u`, …; `stp_mean(projection, variable)`) in SNN and MF results, the workers and the plots. The `saved_variables` names are validated at load.
- [x] (1) **utils.snn_helpers**: *SNN STP reconstruction follows NEST `tsodyks_synapse` exactly* (the old one double-applied the u jump; with τ_fac = 0 it used u = U(2 − U)).
- [x] (1) **data_structures**: *No more silent NaN*: `get_pop_mean`/`get_pop_std`/`_get_raw_all` no longer swallow exceptions (this is what made the SNN STP variables NaN).
- [x] (1) **mf_simulation**: *`stp_dynamic` STP state reported* per projection (it used to fall back to steady state); legacy/asymptotic report what their model uses.
- [x] (1) **mf_simulation.models**: *`stp_dynamic` used u = 1 with τ_fac = 0*: `_utilization()` now gives u = U without facilitation; `U_dyn` defaults and templates are initialised to 0.
- [x] (2) **storage**: *Units in every `.npz`* (`units` entry); `ResultsAggregator.get_units`, `load_run_params`, `load_transfer_functions`.
- [x] (2) **mf_simulation**: *First-order results crash* (`np.sqrt(None)`) fixed; *`rate_cov` units* (kHz², was off by 1e6) fixed; *swapped Y time constants* (ei/ie) in the dynamic models fixed.
- [x] (3) **controller**: *Manifest per run* (`data/<id>/manifest.json`; parallel jobs used to overwrite one file).

Verified in the code or in the project READMEs on 2026-10-04:
- [x] (2) **controller**: *Multi-dimensional inspection*: `run_multiinspection.py` (Cartesian product), joint/tuple parameters (comma keys), adding new combinations to an existing CSV, SLURM submission, and `ResultsAggregator`.
- [x] (2) **mf_simulation**: *STP models*: `stp_asymptotic.*` and `stp_dynamic.*`, with full ee/ei/ie/ii/ed/es projections.
- [x] (2) **network_params**: *Connectivity format with rules and all projections* (`fixed_prob`/`fixed_in`/`fixed_out`; target → source).
- [x] (3) **storage**: *Reduced saving*: `.npz` per model × stimulus with `saved_variables` × `saved_metrics` and `saved_extra_keys`, instead of pickled full results.
- [x] (3) **snn/mf_simulation**: *Parallelisation over stimuli × models* (`run_unified_batch_parallel`).
- [x] (3) **neuron_simulation**: *Execution modes* `run`/`load`/`try_load`/`skip`; `init_values`; adaptive grid with a subthreshold option (`skip_zeros: false`).
- [x] (3) **transfer_function**: *Ports of the Zerlaut 2018 and Di Volo 2019 TFs, plus the separate NeuroPSI custom TF*.
- [x] (2) **research.tf_fitting**: *Verified the TF implementation against Zerlaut/Di Volo (projects/01)*.
  - Neuron data matches. MPF matches to ~1e-4. Identical coefficients give identical TF curves.
  - The published fits don't match their own data, so we use our own fits.
- [x] (3) **controller**: *Test runs*: `--test` with `test_*.yaml` in each project.
- [x] (3) **config**: *All configs are YAML, with one fixed set of file names per project*
- [x] (1) **controller**: *Full workflow config and loading* (`WorkflowConfig`)
- [x] (2) **controller**: *High-level API instead of a god-like class*
- [x] (1) **data_structures**: *Results classes with unit handling* (`BaseResults`, `SNNResults` rewrite, `MFResults` voltage and conductance)
- [x] **plotting**: *Generic grid figure hooks; conductance and STP plots; plot logic by results type instead of name*
