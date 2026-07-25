STP inspection
- what tau_rec should I set up?
- inspect various stimulus
- first inspect drive_rate
- find some reasonable drive and run other stimuli

# Todos (research)
- [ ] **TVB direct stimulus issue**
  - [ ] Create a minimal working example to isolate the weird behavior of direct stimulus in TVB.

- [ ] **Voltage TF fitting**
  - ISSUE: theoretical values are based on subthreshold activity, once the activity becomes high, these data are not reflecting much
  - [ ] Determine what range of membrane fluctuation parameters should be used.

- [ ] **DiVolo TF fitting**
  - ISSUE: TF fit params in repo and paper are completely off, where is the issue?
  - [ ] Generate neuron data using Zerlaut and my MFT scripts
  - [ ] Compare data with Di Volo TF fit (use my MFT, Zerlaut and DiVolo TF function implementation)

- [ ] **Verify TF Implementation**
  - [ ] Compare neuron data (Zerlaut x My script)
  - [ ] Compare membrane potential fluctuation computation
  - [ ] Do not compare the fitting (different units results in different results), Compare that TF functions give the same curves if each script has the same TF fit coeffs

- [ ] **Replicate DiVolo paper**
  - Basically a sanity check, that the new models can do that (static synapse, there should be no changes)
  - [ ] Verify the $b=0$ fitting statement.

- [ ] **MF models with STP**
  - [ ] Run sanity checks on diVolo with static synapses.
  - [ ] Test diVolo params with STP.
  - [ ] Identify the network changes after introduction of STP
  - [ ] inspect the network behavior
  - [ ] is there any equivalence to statement of divolo - TF fit coeffs are independent of adaptation params changes?
  - [ ] Test CSNG with STP synapses.

- [ ] **Test CSNG architecture**
  - [ ] Test CSNG with nonhomogeneous connectivity.

# Todos (code)
Task pattern:
- [ ] (Priority) **module part**: *name of the task*
  - additional info

Task should be short and clear, to get the idea quickly, details in additional info which can be folded/unfolded

Priority
(1) Critical/Blocker: Immediate action required. These tasks prevent the code from running (bugs/crashes) or are architectural dependencies for all other planned features.
(2) Essential/High: Important for research progress. These are core features or experiments that are necessary for the next stage of your thesis.
(3) Important/Medium: Non-blocking features that improve usability, code quality, or provide supplementary data.
(4) Nice-to-have/Low: Minor optimizations, documentation polish, or experimental ideas that don't have a specific deadline.




- [ ] (3) **codebase**: *update README.md files and other files providing explanations*
- [ ] (3) **codebase**: *Check all the commented notes and todos within codes and put them on todo.md, readme.md or similarly instead*
  -  Keep a single source of truth for all the todos and ideas, and they are not lost in the code comments.
- [ ] (4) **codebase**: *Unify documentation and docstrings*
  - Add docstrings to all functions, especially the main workflow and the unified batch runner, to clarify their purpose and expected inputs/outputs.
- [ ] (4) **codebase**: *Migrate away from Pickle*
  - Pickle is notoriously brittle if you rename classes (SingleNeuronResults). For long-term PhD research, storing simulation metrics in HDF5 or Parquet is much safer.
- [ ] (3) **codebase**: *Write description of units handling setup in `units.md`*
- [ ] (4) **codebase**: *Implement logging*
- [ ] (4) **codebase**: *Create a tutorial notebook*
- [ ] (3) **codebase**: *Clean up - Remove dead, unused, or commented-out code across the repository.*
- [ ] (4) **codebase**: *Setup Testing - Write basic tests to check core functions quickly.*

- [ ] (4) **controller**: *Generation of template config params - Make this work nicer codes.controller.config --template --schema*
  - or create some other way of generating and providing example templates
  - so the user can check what are allowed params in config files, the structure and expected format/values/types

- [ ] (3) **data_structures**: *SingleNeuronResults should keep spikes as data with default units*
- [ ] (3) **data_structures**: *Create a method for all results classes listing measured values (what is not None)*
  - usecase?
- [ ] (3) **data_structures**: *Other methods of saving (due to spike data) np.save_compressed() or the h5py*
  - save all and save means (time means, pop means etc, so that we do not keep 40000x500 arrays, but rather the reasonable averages)
- [ ] (4) **data_structures**: *Add load method (e.g., a @classmethod for load(cls, filepath))*














# Ideas
- [ ] **"Style Plot" system:** Instead of specific classes like `FiringRatePlot`, pass a generic results object to a single class that applies "styles" (errorbar, line, fill_between).
- [ ] **Automatic units:** Try to determine the plot unit automatically from the variable name.
- [ ] **Network State Classification:** Compute regularity and synchrony parameters for SNN to classify network states.
- [ ] **Phase-plane analysis:** Explore nullclines and behavior when changing rates.
- [ ] **Caching:** Implement caching mechanisms for storing `Results`.
- [ ] **QIF neuron replication:** Look into replicating Alain lab or Helmut Schmidt QIF neuron papers.

# Other notes
- **Results Class Safety:** Ensure that we strictly use predefined methods when interacting with the `Results` class.
- **New Variables to Measure:** Consider formally tracking inhibitory adaptation, conductances (ee, ei, ie, ii), and STP variables (u, x, y).
- **Biology Notes on STP:** Inhibitory neurons are generally Facilitatory (e.g., PV interneurons), while excitatory neurons are Depressing (SST interneurons might be depressive). Look into observing ISN and STP.
- **Reading Backlog:**
  - NeuroPSI MF papers
  - Markram Tsodyks STP model (find network-level influence)
  - ISN and STP
  - QIF neuron and STP (Helmut Schmidt)


### API Refactoring & Parallelization
- [ ] Evaluate `neuron_simulation` API against the new `network_simulation` lifecycle.
    - **Context:** `network_simulation` now uses `build() -> run_stimulus() -> reset()`. 
    - **Blocker:** Single neuron simulations heavily exploit parallelization (e.g., TF fitting). NEST simulator objects cannot be pickled, and global resets might interfere with multiprocessing.
    - **Task:** Investigate if we can safely use the `build/run/reset` pattern *inside* the parallel worker functions to maintain API consistency without breaking multiprocessing. Until then, keep `simulate()` for neurons.

# Plan

- [ ] Implement transfer function zerlaut2018, divolo2019
- [ ] test it
- [ ] issue 1 presentation
  - [ ] Data generation
    - [ ] Zerlaut script vs Zerlaut MFT
    - [ ] Zerlaut MFT vs PyNN MFT
    - [ ] result --> we are confident to use PyNN MFT data
  - [ ] Transfer function
    - [ ] zerlaut script vs zerlaut MFT
    - [ ] divolo script vs divolo MFT
    - [ ] zerlaut MFT, divolo MFT vs NeuroPSI cutom MFT
    - [ ] result --> we are confident to use NeuroPSI cutom MFT
  - [ ] Issue 1 : DiVolo published fit does not hold!
    - [ ] inspect various parameters set up (check adaptation parameters)
    - [ ] sources of error - come up what could it be (network parameter)
  - [ ] replicate diVolo paper
    - [ ] update network simulation, meanfield simulation (or make it usable with new params structure and refactor it later)
    - [ ] with MFT I can investigate edge cases etc

- [ ] snn_simulation recorders options

TODO
- 2D inspection
    - adding inspection params as dict
    - possibility of cut (tuple_params) : [(list_of_tuple,values)]
- merge inspection results (additional inspection)
- saving full and reduced results (later I can remake the plots for details)
- runable scripts (can run sbatch on the wintermute)

- explosion detection within first 1000 ms high activity
    - give it longer time, will the explosion happen again once the adaptation drops?  
- control/detection of steady state?

- Extractor do not have check on measured variables
- there is no control of units (I have the whole system of working with units, use it)
    - or use the unit conversion when public API, but internally expect the units (but what if I decide to change the units? will I?)

- inspection neural data simulation choice

- add neuron_simulation execution modes (run, load (alias for one of the following?), load-try (makes simulation if nothing found), load-strict (raises error), skip)

- change the snn_results and mf_result to have variable and dot notation method to access metrics (eg. all, pop_mean, time_mean, future position_mean?)
- maybe rename "_mean", "_std" to "_pop_mean" and "_pop_std"? so it is clear what mean is meant (I have "_time_mean")

- Make myself tutorial notebook for comparison metrics to see what the metrics catch visually, so I understand the numbers


plotting to inspect Zerlaut approach (neuron_simulation, transfer_function level)
- neuron_simulation - voltage tau from simulation
- Voltage plot (mean, std, tau)
- TF Fitting params plot (V_eff, mu_v etc)
- theoretical values vs measured ones
- mu_V, sigma_V, tau_V, mu_G, sigma_G, v_eff
- plot_fluctuation_theoretical, plot_fluctuation_comparison, plot_tf_fitting_steps

- analysis/spike_metrics: Regularity, Synchrony
    - as measures of AI state, also measures of UP/DOWN states

- Overview plots dynamically adjusting to measured variables and plotting accordingly? (if conductance measure plot conductance etc.)
  - if variable not measured do not include the plot?

## [Architecture Decision Record] SNN Simulation Reset Paradigm

**Context:** 
When running multiple stimuli sequentially in PyNN with the NEST backend, using the native `.reset(t_flush=...)` method causes internal clock desynchronization, leading to `AssertionError`s during data retrieval (`get_data()`). We evaluated two architectural paradigms to solve this.

### Option A: The "Continuous Epoch" Paradigm (Rejected for now)
**Idea:** Instead of resetting the simulator, run a single continuous simulation from $t=0$ to $t=T_{total}$. Separate different stimuli using "blank" (0 Hz) spontaneous activity windows of ~1000ms+ to allow the network to relax back to its attractor state.

*   **Pros:**
    *   **Computationally Fast:** Bypasses the overhead of rebuilding the 10,000+ neuron network and wiring matrix for every stimulus.
    *   **Backend Safe:** Avoids PyNN's buggy `reset()` logic entirely.
    *   **Biological Realism:** Mimics continuous *in vivo* recording sessions where an animal is simply shown a blank screen between visual stimuli.
*   **Cons (Scientific Risks):**
    *   **Hysteresis (History Dependence):** Slow biological variables (like AdEx adaptation $w$ with $\tau_w=500ms$, or STP facilitation/depression variables $u/x$) decay exponentially but never perfectly reach $0$.
    *   **Order Effects:** The exact numerical response to "Stimulus B" will change depending on whether "Stimulus A" preceded it, making debugging and isolated Mean-Field comparisons extremely difficult.
    *   **Data Parsing Complexity:** The PyNN recorder will yield one massive, continuous Neo block that must be meticulously sliced using time-masks during the analysis phase.

### Option B: The "Clean Slate" Paradigm (Current Decision)
**Idea:** The Build-Run-Teardown pattern. For every stimulus trial, completely destroy the NEST kernel, rebuild the network with the exact same `rng_seed`, run the stimulus, and extract the data.

*   **Pros:**
    *   **Absolute State Isolation:** Guarantees absolute mathematical certainty that every trial starts with the exact same initial conditions ($v$, $w$, $u$, $x$). 
    *   **Ground-Truth Reliability:** Ensures that SNN data provides a perfectly clean "ground truth" to compare against Mean-Field analytical equations, free from hidden cross-contamination.
*   **Cons:**
    *   Slower runtime due to the overhead of rebuilding the network graph for each stimulus in the dictionary.

**Verdict & Future Action:** 
We stick to **Option B (Clean Slate)** for all Mean-Field transfer-function fitting and validations, as trial independence is strictly required. 

*Future Todo:* **Option A (Continuous Epochs)** should only be implemented if we start researching sequence-dependent network effects (e.g., how the network responds to a high-frequency train of different stimuli) or if simulation times become a critical bottleneck.


----


# TODO:




- [ ] (2) **neuron_simulation**: *Subthreshold grid: allow adaptive grid to also cover subthreshold region*
- [ ] (3) **neuron_simulation**: *implement execution modes 'skip", 'validate' (comparison of existing data and newly generated ones)* 
- [ ] (3) **neuron_simulation.pynn_simulator**: *Make it work with init_values*
- [ ] (3) **neuron_simulation.pynn_simulator**: *inspect what happened and debug it, extremely weir results!*
  - the data in `project/04_debug`
- [ ] (4) **neuron_simulation.pynn_simulator**: *Redo the `legacy_neuron_params`*
  - currently relies on hardcoded string names in legace neuron_params format!
- [ ] (4) **neuron_simulation**: *Adaptive Grid for Inhibitory Neurons*
  - Implement the logic to allow inh_rate to be the adaptive variable. This will require carefully handling the interpolation since the roles of the axes are flipped.
- [ ] (4) **neuron_simulation**: *Implement computation of `voltage_tau`*
- [ ] (4) **neuron_simulation**: *Implement computation of `adaptation_std`*
- [ ] (4) **neuron_simulation**: *grid resolving (at least linear) could be in some helper function, not necessary to copy to each simulator*
- [ ] (4) **neuron_simulation**: *PyNN simulator, load the units from the model used? Such that I do not have to hard code the units and do not have to make mapping for each neuron model*
- [ ] (4) **neuron_simulation**: *Option to pick neuron model*

- [ ] (4) **transfer_function**: *Rename to tf_fitting? or rename the `run_tf_fittinf_workflow`*
- [ ] (4) **transfer_function.neuropsi_tf**: *MembranePotentialFluctuations allows adaptation only for 'exc_neuron', could be optional the same as effective_weights...*

- [ ] (4) **snn_simulation**: *implement parallelization (or use PyNN methods of parallelization)*
- [ ] (4) **snn_simulation.config**: *Add recorders to allow specify what is measured in SNN and from how many neurons*

- [ ] (4) **mf_simulation**: *implement parallelization? at least with various stimuli*
- [ ] (2) **mf_simulation**: *add tsodyks models (models handling STP)*
- [ ] (3) **mf_simulation.tvb_simulator**: *Make a way for drive rate to increase gradually*
  -  solution of having stimulus together with drive is dangerous once I move to grid, also drive and stimulus can have different targets!
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_connectivity()`, currently hardcoded*
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_coupling()`, currently hardcoded*
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_integrator()`, currently hardcoded*
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_monitors()`, currently hardcoded*
- [ ] (2) **mf_simulation**: *test first order model, added models handling STP but works only for second order and I did not even test the first order divolo*


- [ ] **plotting**: *Handle missing variables (when returns None) gracefully*
  - None when not measured, None instead of Results class when skipped simulation
  - np.nan when inspection metrics applied 
- [ ] **plotting.inspection_plots**: *Add unit handling option*

- [ ] (3) **utils.snn_helpers**: *Update activity calculation so that all method return 2D array*

- [ ] (2) **research**: **


- [ ] (2) **data_structures.inspection**: *Merge results, so that I can run two inspections in parallel and merge results*
- [ ] (2) **data_structures.inspection**: *Change VariableResultGroup such that it represents raw data for single variable*
  - and allows eg variable.pop_mean(), variable.time_mean(start_time, end_time), variable() -> returns all data
  - now it just wraps the CoreInspectionResults, to provide the dot notation from self._data dictionary
  - idea is to make ResultClasses represent measured variables as subclasses

- [ ] (2) **controller.inspectors**: *Inspector for multidim inspection*
  - e.g. inspection_dict = {param1: [values], param2: [values], ...}
- [ ] (2) **controller.inspectors**: *Inspector for inspection with two params sharing value*
  - i.e. not making 2D grid, but a slice
  - e.g. inspection_dict = {(param1,param2): [(value11, value21), (value12, value22), ...]}
- [ ] (3) **controller.inspectors**: *Extractor can extract data in nested dictionary or use dot notation for the var.metric instead*

- [ ] (3) **testing**: *Small scripts for various optionalities, workflows, individual plots etc.*
- [ ] (3) **testing**: *with each update run a script that generates dummy results to use for testing*
  - so that for testing of plotting or calculations simulations are not needed to be run

# ACTIVE:


# DONE:
- [x] (1) **controller**: *Full workflow config and loading*
- [x] (2) **controller**: *Make some high level API instead of the god-like class*
- [x] (1) **controller**: *make it runable start to finish based on param files* (with plotting)
- [x] (1) **data_structures**: *Clean up old code and update simulators*
- [x] **data_structures**: *rewrite SNNResults*
- [x] **BaseResults**: *create new Base Results class that would contain the the unit handling* 
- [x] (3) **data_structures**: *MFResults voltage and conductance data calculation*
  - implementation issues:
    1. MPF does not differentiate drive, stimulus, exc_neuron inputs
    2. exc_neuron has adaptation, but drive and stimulus do not
  - make it a subclass?
  - I can add different input sources in MPF
- [x] **plotting**: *Implement a generic Figure Plot Controller ("Style Plot" system).*
- [x] **plotting**: *Create predefined inspection plots (e.g., `SpontRateHistogramPlot`, `ActivityInspectionPlot`)*
- [x] **plotting**: *Create synaptic conductivity plot.*
- [x] **plotting**: *Create STP plots*
- [x] (3) **plotting** : *Remove naming convention reliance (Refactor plotting logic to use `.results_type` instead of `.startswith("SNN")`)*
