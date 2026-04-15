# Units 
Here is a table of units used in different simulators, because of this we set
the units as they are indicated in the in the table inside MFT.

Be aware that PyNN is not consistent with the rules of units used! Also when PyNN 
uses native simulator, it uses the units of the simulator. Be cautious!


| Physical quantity | MFT Units | PyNN Units    | NEST Units    | TVB Units |
| ----------------- | --------- | ------------- | ------------- | --------- |
| time              | ms        | ms            | ms            | ms        |
| rate              | Hz        | Hz            | Hz            | kHz       |
| voltage           | mV        | mV            | mV            |  mV       |
| current           | nA        | nA            | pA            | --        |
| conductance       | nS        | uS*           | nS            |           |
| capacitance       | nF        | nF            | pF            |           |
| phase/angle       |           | deg           |               |           |

\* PyNN: Conductance for adaptation parameter 'a' is in [nS]!

Sources
- [PyNN Units](https://neuralensemble.org/docs/PyNN/units.html), [PyNN synapses part](https://pynn.readthedocs.io/en/latest/connections.html) 
- [NEST Units](https://nest-simulator.readthedocs.io/en/v2.18.0/getting_started.html)
- [TVB Units]()

Additional notes
- PyNN: Synaptic weights are in [uS] or [nA], depending on whether the post-synaptic mechanism implements a change in conductance or current.
- PyNN: when using native synapse models within PyNN (NEST) be sure to convert the units! (nb in connectors in mozaik there is this conversion)
- PyNN: these units does not hold everywhere!!!!! E.g. neuron model in PyNN [EIF_cond_exp_isfa_ista](https://pynn.readthedocs.io/en/latest/reference/neuronmodels.html#pyNN.standardmodels.cells.EIF_cond_exp_isfa_ista) has units for adaptation 'a': 'nS', 'b': 'nA'




