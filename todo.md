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
  - Done: `design.md`, `todo.md`, `README.md`. `MeanFieldTester/README.md` was deleted; its content was migrated.
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

- [ ] (1) **network_params**: *Finish the migration to the `connectivity` format*
  - These still read the removed `network_params.synapses`, so they crash:
    - `SNNResults._compute_stp_variables` (data_structures/snn_simulation.py:318), which breaks the SNN `exc_x`/`exc_u` getters;
    - `Zerlaut2018TF` and `DiVolo2019TF` (`_get_legacy_params_dict`).
  - The commented examples in each project's `inspected_params.yaml` still use `network.synapses.*`.
- [ ] (1) **mf_simulation.tvb_simulator**: *`stp_dynamic` STP state is not passed to MFResults*
  - The model has `X_ee, Y_ee, U_dyn_ee, X_ei, …`, but `run_stimulus` reads `X_e` / `U_dyn_e`. Those are missing, so `MFResults` silently falls back to steady-state STP. x/u plots for the dynamic model show the asymptotic values.
  - The `X_e/U_e/X_i/…` aliases in `CustomNeuroPSIInitialValuesConfig` and the workflow YAMLs are left over from the old model.
- [ ] (1) **data_structures**: *Store STP variables per projection (ee, ei, ie, ii)*
  - STP is a property of a projection, not of a population. Since the connectivity refactor, ee and ie can have different τ_rec.
  - `SNNResults` (one `syn_params` per neuron) and `MFResults` (`exc_x`, `inh_u`, …) both assume one STP per population.
  - Do this together with the fix above. Rename the keys to target-source codes (`ee_x`, …), and update the npz keys and plots.

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
  - Only `stp_*.first_order` was run, once, in projects/04. `divolo2019.first_order` has never been run.
- [ ] (3) **mf_simulation**: *Single source of truth for the TF formula*
  - It is implemented twice: `NeuroPSICustomTF`/`MembranePotentialFluctuations`, and `get_fluct_regime_vars`/`TF` in the TVB models.
  - At least add a test that evaluates both with the same coefficients.
- [ ] (3) **mf_simulation.tvb_simulator.models**: *Legacy models take only the projections onto E*
  - `zerlaut2018.*`/`divolo2019.*` assume I receives the same inputs as E. Document this, or raise an error when the connectivity is asymmetric.
- [ ] (4) **mf_simulation.tvb_simulator**: *Make connectivity/coupling/integrator/monitors configurable*
  - All four are hard-coded: single node, `Linear(a=0.3)`, Heun stochastic, Raw monitor.
  - A multi-node grid (`grid_size > 1`) is a prerequisite for CSNG non-homogeneous connectivity.
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
  - Return the coefficients instead, and have the caller put them into the MF config.
- [ ] (4) **transfer_function**: *Rename `run_tf_fitting_workflow` or the module*
  - The workflow function also loads fits; it doesn't only fit.

## data_structures / storage

- [ ] (2) **storage**: *Store units with the saved results*
  - The `.npz` files hold arrays in default units with no unit metadata. Save a `units` entry, e.g. from `DEFAULT_UNITS`.
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
- [ ] (3) **scripts.run_multiinspection**: *Relative `--project_dir` breaks*
  - `main()` does `os.chdir(project_dir)` and then builds `project_dir / "params"` again, so a relative path resolves twice. Resolve the path before the `chdir` (`Path(...).resolve()`), or drop the `chdir`.
- [ ] (3) **scripts**: *Remove duplicated helpers*
  - `parse_val`, `normalize_val` and `DELIMETER` (sic) are copied in `run_multiinspection.py`, `inspection_worker.py` and `ResultsAggregator`.
- [ ] (4) **controller.config**: *Template and schema generation*
  - `--template` and `--schema` raise `NotImplementedError`, and the code after the `raise` is dead. Either implement them (`WorkflowConfig.model_json_schema()` is nearly free) or delete them.

## plotting

- [ ] (3) **plotting**: *No computation inside plots*
  - `AggregatorNeuronIOCurvePlotter` can fit a TF itself (`_fit_tf_funcs`). Fit outside the plot and pass the fitted TFs in.
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
  - Commit or ignore projects 04–09; clean the stale config comments (e.g. `zerlaut2018_simulator`/`divolo2019_simulator` in `workflow_params.yaml`).
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

---

# Notes (to move to notes.md)

- **STP biology:** inhibitory synapses are often facilitating (PV interneurons are mentioned; SST may be depressing). Excitatory synapses are depressing. Look into ISN + STP.
- **Reading backlog:** NeuroPSI MF papers; Tsodyks–Markram (network-level effects); ISN and STP; QIF neurons and STP (Helmut Schmidt).

---

# DONE

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
