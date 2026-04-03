
Working Modules
- **cell_library**
- **neuron_simulation**
- **transfer_function**
- **network_simulation**
- **meanfield_simulation**

Workflow Modules
- **controller**
- **data_structures** - *each of the modules should return appropriate instance from this module*. That ensures the common API and naming across the package, even when there are changes in working modules, or working modules are extended by another backend simulators.
- **plotting**
- **utils**




Units:
- time: [ms]
- rate, frequency: [Hz]
- adaptation, current: [pA]
- conductance: [nS]
- voltage: [mV] 

units used throughout in the code (unit used when specifying the models!)

conversions should be inside functions! input and output of ALL FUNCTIONS should be standardized!


Nomenclature:
- times
- rate (activity, $\nu$, $r$)
    - reason : drive_rate, stim_rate, rate models, I didn't see drive_activity or activity models 
- adaptation ($w$)
- voltage (membrane potential, $V$)
- conductance (gsyn, g)
- FromTo : eg. ei means from Exc to Inh
    - reason : I like it more
- mean, std (instead of mu, sigma)


To decide
- pars x params x parameters
- net x network
- stim x stimulus
- const x constant, func x function (USE shortcuts everywhere or nowhere, be consistent!)


# TODO
- DiVolo-stp MF-Tsodyks returns nan values
- Membrane Fluctuations, adaptations, TF fitting plots
    - DiVolo eq (10), (11).
    - plotting
- `AdExNeuronTheoreticalResults` class should be more directly connected to the transfer_function module! (At least the formulas should be separate function called on these two places)
    - reuse MPF class better
    - Unite MPF classes into one. Don't go with call (makes sense for TF), have methods to pick
- Ensure the new result classes are given in network/mf simulator
- plotting 
    - accept new data classes (specifically single neuron class, and the units!)
    - figure_maker
    - conductance plot
    - STP plot
- calculate error (diff between snn and mf)
- inspect param


### MF testing
- with the new code - generate new neuron data!
- TF fitting
- Voltage fitting
- inspect parameter/stimulus
- STP facilitation
- DiVolo STP - explosion inspect
- CSNG setup - inspect
- CSNG-static - update TVB Zerlaut model such that it can take Cm for exc and inh population

### Modules
- run workflow with parsing argument (eg. python run_workflow.py divolo-static), I dont need to have special script for standardized runs
- Naming convention (few lines before I have some notes)
- class for parameters, so that I can generate PyNN, TVB etc. the conversion would be inside
- clean up dead code
- clean up repo (make it representable and committable to NeuroPSI)
- Ensure unit-handling, setup strict convention ()
- Tutorial notebook
- setup logging
- determine neuron names (exc_neuron, inh_neuron) from the param file, it is hardcoded in (cell_library, transfer_function...)
- unite naming of files! (DiVolo_network-stp.yaml, divolo-stp-exc_neuron...)
    - after this update controller so that just the name is enough (divolo-stp, etc)
- import parent directory https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder?rq=3
- no computation INSIDE plotting, just call/ask for the values!
- fixed point, nullcline analysis? 
- make a able for options with single neuron simulation what is more important etc
- create a test run with test workflow_params etc, such that it runs quickly
- pip requirements.txt
##### controller
- update plotting
- make it possible to continue from a directory (in init update the NotImplementedErrors)
- make workflow and network params the same format style (either json or yaml)
##### cell_library
- clean up, plenty of old unused code
##### neuron_simulation
- compute and return $tau_V$ (ATM: returns zeroes)
##### transfer_function
- documentation
- change the naming convention
- change the units so it follows my choice (ideally first make parameter class)
- separate the current TF fit into neuropsi tf part, it would allow possibly different TF approaches to test
##### data_structures
##### network_simulation
##### meanfield_simulation
##### plotting
- update with being careful about units!
##### utils


# Design of the repo

MF are added separately so that for each MF transfer function fit can be chosen, 
also separate model can be chosen and separate network params

Maybe this repo could be extended for other MF models and Spiking Neural Networks!


# pynn models here:
# https://pynn.readthedocs.io/en/latest/reference/neuronmodels.html
# EIF_cond_exp_isfa_ista, there are units and parameters






to change:
---

get_fluct_regime_vars
- shape of the input of the external sources
- STP correction



Scripts
---
`run_workflow.py`
- 


