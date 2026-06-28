
# Todos (code base)


# Todos (research)
- [ ] **TF fitting replication**
  - [ ] Replicate diVolo figures.
  - [ ] Compare with Zerlaut data points.
  - [ ] Verify the $b=0$ fitting statement.
- [ ] **Add ModelDB TF fit**
  - [ ] Add a TF fit line based on the ModelDB repo to verify parameter reliance independently.
- [ ] **MF models with STP**
  - [ ] Run sanity checks on diVolo with static synapses.
  - [ ] Test diVolo params with STP.
  - [ ] Test CSNG with STP synapses.
- [ ] **TVB direct stimulus issue**
  - [ ] Create a minimal working example to isolate the weird behavior of direct stimulus in TVB.
- [ ] **Spont activity inspection**
  - [ ] Automate inspection over parameter ranges.
  - [ ] Create histogram plots for activity, voltage, and adaptation.
- [ ] **General stimuli inspection**
  - [ ] Compute difference and correlation between SNN and MF over varying parameters.
- [ ] **Test CSNG architecture**
  - [ ] Test CSNG with nonhomogeneous connectivity.
- [ ] **Voltage TF fitting**
  - [ ] Determine what range of membrane fluctuation parameters should be used.

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
- **Proposed Inspection Attributes:** `exc_rate_time_mean`, `exc_rate_diff_time_mean`, `exc_rate_time_std`, `exc_rate_diff_time_std`.
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

Task pattern:
- [ ] Priority **module part**: *name of the task*
  - additional info
Priority
(1) Critical/Blocker: Immediate action required. These tasks prevent the code from running (bugs/crashes) or are architectural dependencies for all other planned features.
(2) Essential/High: Important for research progress. These are core features or experiments that are necessary for the next stage of your thesis.
(3) Important/Medium: Non-blocking features that improve usability, code quality, or provide supplementary data.
(4) Nice-to-have/Low: Minor optimizations, documentation polish, or experimental ideas that don't have a specific deadline.

# TODO:
- [ ] () **module**: *task*

- [ ] (3) **codebase**: *update README.md files and other files providing explanations*
- [ ] (3) **codebase**: *Check all the commented notes and todos and put them on todo.md instead*
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

- [ ] (3) **data_structures**: *MFResults voltage and conductance data calculation*
  - implementation issues:
    1. MPF does not differentiate drive, stimulus, exc_neuron inputs
    2. exc_neuron has adaptation, but drive and stimulus do not
  - make it a subclass?
  - I can add different input sources in MPF
- [ ] (3) **data_structures**: *SingleNeuronResults should keep spikes as data with default units*
- [ ] (3) **data_structures**: *Create a method for all results classes listing measured values (what is not None)*
- [ ] (3) **data_structures**: *Other methods of saving (due to spike data) np.save_compressed() or the h5py*
- [ ] (3) **data_structures**: *InspectionResults write in accordance with the new setup*
- [ ] (4) **data_structures**: *Add load method (e.g., a @classmethod for load(cls, filepath))*

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

- [ ] (4) **snn_simulation**: *implement parallelization (or use PyNN methods of parallelization)*

- [ ] (4) **mf_simulation**: *implement parallelization? at least with various stimuli*
- [ ] (2) **mf_simulation**: *add tsodyks models (models handling STP)*
- [ ] (3) **mf_simulation.tvb_simulator**: *Make a way for drive rate to increase gradually*
  -  solution of having stimulus together with drive is dangerous once I move to grid, also drive and stimulus can have different targets!
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_connectivity()`, currently hardcoded*
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_coupling()`, currently hardcoded*
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_integrator()`, currently hardcoded*
- [ ] (4) **mf_simulation.tvb_simulator**: *add options for `self.setup_monitors()`, currently hardcoded*
- [ ] (2) **mf_simulation**: *test first order model, added models handling STP but works only for second order and I did not even test the first order divolo*


- [ ] (3) **plotting** : *Remove naming convention reliance (Refactor plotting logic to use `.results_type` instead of `.startswith("SNN")`)*
- [ ] **plotting**: *Handle missing variables gracefully*
- [ ] **plotting**: *Implement a generic Figure Plot Controller ("Style Plot" system).*
- [ ] **plotting**: *Create predefined inspection plots (e.g., `SpontRateHistogramPlot`, `ActivityInspectionPlot`)*
- [ ] **plotting**: *Create synaptic conductivity plot.*
- [ ] **plotting**: *Create STP plots*

- [ ] (3) **utils.snn_helpers**: *Update activity calculation so that all method return 2D array*

- [ ] (2) **research**: **




# ACTIVE:

- [ ] (1) **controller**: *Full workflow config and loading*
- [ ] (2) **controller**: *Make some high level API instead of the god-like class*

# DONE:
- [x] (1) **controller**: *make it runable start to finish based on param files* (with plotting)
- [x] (1) **data_structures**: *Clean up old code and update simulators*
- [x] **data_structures**: *rewrite SNNResults*
- [x] **BaseResults**: *create new Base Results class that would contain the the unit handling* 
