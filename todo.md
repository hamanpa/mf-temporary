# Design choice
- **Modularity:** Modules should be easily swappable (e.g., changing the MF model should not require changing the rest of the workflow).
- **Separation of Concerns (Plotting):** No computations inside plotting modules. Results generation, analysis, and plotting must be strictly separate.
- **Data Handling:** The runner/controller should not store data internally; data must be explicitly returned to or saved for the user.
- **Consistent User-Facing Units:** Units that the user interacts with (network params, results) should always be identical. Any necessary conversions must happen internally within modules.
- **Naming Conventions:** Use underscores instead of hyphens for attribute names (e.g., `my_attribute`, not `my-attribute`).
- **Code Style Standard:** Use standard Python conventions (PEP 8) and a consistent docstring style (e.g., Google or NumPy style) across the codebase.
- **Testing Framework:** Use a standard testing framework (e.g., `pytest`) to ensure updates do not break existing functionality.


- **The Open/Closed Principle (OCP)** 
states that software entities (classes, modules, functions) should be open for extension (you can add new behavior) but closed for modification (you don't have to alter existing code to add that behavior).
If a PhD student in your lab writes a custom simulator for a new type of hardware, with the match statement, they must edit your WorkflowRunner.py to add case "custom_hardware":. If you use an interface, they can just pass their custom class to your runner, and your runner will execute it blindly because it trusts the abstract class structure. Your core code remains untouched.


Single Responsibility Principle (SRP)


# Todos (code base)

- [ ] neuron_simulation - make it more modular
- [ ] neuron_simulation - allow custom grid
- [ ] neuron_simulation - better naming workflow_params
- [ ] neuron_simulation - add zerlaut2018_simulator.py

### General MFT Code
- [ ] **Units handling setup**
  - [ ] Correct adaptation conversion logic.
  - [ ] Establish a central overview/handler module for units.
  - [ ] Ensure units for user-facing network parameters match the results outputs.
- [ ] **Implement logging**
  - [ ] Set up a base logger config for the MFT package.
  - [ ] Replace `print()` statements with appropriate `logger.info()`, `logger.debug()`, or `logger.warning()` calls.
  - [ ] Expand logging (so that tracking issues is easier)
- [ ] **Documentation**
  - [ ] Add module-level docstrings/help strings to core files.
  - [ ] Complete the tutorial notebook (`workflow.ipynb`).
  - [ ] Update the `README.md` with basic setup and usage instructions.
- [ ] **Parameter validation**
  - [ ] Implement an `__init__` reader/validator that checks for required parameters.
  - [ ] Add backend compatibility checks (e.g., "does TVB/NEST support these params?").
- [ ] **Clean up**
  - [ ] Remove dead, unused, or commented-out code across the repository.
- [ ] **Standardize naming convention**
  - [ ] Enforce consistent variable names (e.g., use `rate` over `nu/activity`, `params` over `pars`).
  - [ ] Ensure all class attributes use underscores instead of hyphens.
- [ ] **Setup Testing**
  - [ ] Initialize a `pytest` suite infrastructure.
  - [ ] Write basic tests to check core functions quickly.

### Controller & Workflow
- [ ] **`workflow_params` improvements**
  - [ ] Allow passing custom grids for neuron simulation.
  - [ ] Fix naming inconsistencies (e.g., rename to `fix_nu_out`).
  - [ ] Add support for custom data output paths.
- [ ] **Model selection**
  - [ ] Update `workflow_params` to accept MF model name strings.
  - [ ] Update `workflow_params` to accept SNN model name strings.
- [ ] **`run_workflow` shortcut**
  - [ ] Create a high-level API method to execute the entire workflow based on a single parameter dictionary/file.
- [ ] **Parallelization**
  - [ ] Implement parallel simulation capabilities for SNN (PyNN).
  - [ ] Implement parallelization for MF simulations.
  - [ ] Implement parallelization for stimuli generation.

### Cell Library
- [ ] **Refactor parameter handling**
  - [ ] Create a unified class/structure that holds raw biological parameters.
  - [ ] Write a method to generate a proper dictionary for NEST.
  - [ ] Write a method to generate a proper dictionary for PyNN.
  - [ ] Write a method to generate a proper dictionary for TVB.
- [ ] **Unit conversions**
  - [ ] Add robust unit conversion methods specifically within the cell library.

### Neuron Simulation & Transfer Function
- [ ] **Clarify loading vs. simulating**
  - [ ] Resolve the intended behavior between `simulate_single_neuron` and `load_single_neuron`.
  - [ ] Implement a check comparing loaded params against current network params.
- [ ] **Update results format**
  - [ ] Update `neuron_simulation.py` to correctly output/use `helper.SingleNeuronResults`.
  - [ ] Update `transfer_function.py` to correctly output/use `helper.SingleNeuronResults`.
- [ ] **Zerlaut/DiVolo examples**
  - [ ] Create clear code examples showing how to initialize and pick between these two modules.
- [ ] **Implement specific computations**
  - [ ] Add computation for `tau_V`.
  - [ ] Add computation for membrane voltage fluctuations.
  - [ ] Add computation for `sigma_w`.

### Plotting
- [ ] **Remove naming convention reliance**
  - [ ] Add an internal `.results_type` property to all results data classes (e.g., `neuron`, `snn`, `mf`).
  - [ ] Refactor plotting logic to use `.results_type` instead of `.startswith("SNN")`.
- [ ] **Handle missing variables gracefully**
  - [ ] Update plots to handle missing adaptation variables (e.g., in first-order MF).
  - [ ] Update simulation result plots to handle other conditionally absent variables.
- [ ] **New plots implementation**
  - [ ] Implement a generic Figure Plot Controller ("Style Plot" system).
  - [ ] Create predefined inspection plots (e.g., `SpontRateHistogramPlot`, `ActivityInspectionPlot`).
  - [ ] Create synaptic conductivity plot.
  - [ ] Create STP plots.

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