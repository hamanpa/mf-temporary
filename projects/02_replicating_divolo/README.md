# Overview

This project aims to replicate Di Volo results within my implementation of MeanFieldTester

Relevant papers:
- [diVolo2019_paper] M. di Volo, A. Romagnoni, C. Capone, and A. Destexhe, “Biologically Realistic Mean-Field Models of Conductance-Based Networks of Spiking Neurons with Adaptation,” Neural Computation, vol. 31, no. 4, pp. 653–680, Apr. 2019, doi: 10.1162/neco_a_01173.
- [Zerlaut2018_paper] Y. Zerlaut, S. Chemla, F. Chavane, and A. Destexhe, “Modeling mesoscopic cortical dynamics using a mean-field model of conductance-based networks of adaptive exponential integrate-and-fire neurons,” J Comput Neurosci, vol. 44, no. 1, pp. 45–61, Feb. 2018, doi: 10.1007/s10827-017-0668-2.


Relevant repos:
- [diVolo2019_repo] https://github.com/ModelDBRepository/263236/tree/master
    - this is a repo for [diVolo2019_paper]
- [MFT] MeanFieldTester code
    - this is my code 


### Transfer Function (Fig 1)

Fig 1A 
- My and Zerlaut repo data are similar. None corresponds to di Volo data
- inhibitory neurons functional threshold higher (roughly $\nu_e=4\, Hz$) for my and Zerlaut repo data
- inhibitory neurons do not even reach $\nu_{out}=35\, Hz$ for my and Zerlaut repo data
- excitatory neurons functional threshold higher (roughly $\nu_e=5\, Hz$) for my and Zerlaut repo data
- excitatory neurons do not even reach $\nu_{out}=10\, Hz$ for my and Zerlaut repo data
- TF fit params of di Volo are much slower than data points. It is lower when used with MFT code as well as di Volo repo code (and both TFs are the same)

Fig 1B
- My and di Volo computation give the same results (up to rounding error ~1e-14)
- it fits well until the neurons start spiking (for inhibitory and excitatory it splits as the firing increases, the limit seems to be around ~5 Hz)
    - neuron simulations reach roughly steady mean voltage (~-53 mV) due to voltage reset mechanism
    - computed voltage fluctuations continue to increase
- in di Volo paper it is not that visible...
- ! di Volo paper does not show inhibitory neuron

Notes
- [Zerlaut2018_paper] has AdEx neurons, excitatory neurons have adaptations! so that is not different from di Volo!
- The only difference in between [Zerlaut2018_paper] and [diVolo2019_paper] setup is $c_m$, they have $c_m=150\,pF$ and $c_m=200\,pF$ respectively
- [Zerlaut2018_paper] has external drive $\nu_{ext}=4\,Hz$ while [diVolo2019_paper] goes with just $\nu_{ext}=2\,Hz$
- [Zerlaut2018_paper] simulates data by explicit Euler method of the AdEx equation with dt = 5e-5 s.

### Spontaneous activity (Fig 2)



### Results

### Issues
- Voltage fit seem poor
    - [diVolo2019_paper] has voltage only for excitatory spont activity in Fig 2B

