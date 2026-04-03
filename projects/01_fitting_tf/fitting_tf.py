"""


"""


import numpy as np
import sys
from pathlib import Path
import pickle


PROJECT_DIR = Path(__file__).absolute().parent

# Add MeanFieldTester directory to the system path to import modules
sys.path.append(str(PROJECT_DIR.parent.parent / "MeanFieldTester"))

import codes.controller as rw
from codes.plotting import fig_plots

def main():
    # PART 1: compare neural data
    #   Zerlaut repo data, Zerlaut code generated data, MFT code generated data
    zerlaut_repo_data_path = PROJECT_DIR / "modeldb_zerlaut2018" / "data"
    rs_data_repo = np.load(zerlaut_repo_data_path / "RS-cell_CONFIG1.npy", allow_pickle=True)


    # Allow the custom grid!
    # data path, storing data into results
    # workflow name networks

    # PART 2: Compare transfer functions
    #   1. DiVolo params, MFT generated neural data
    #       - are the neural data similar to paper?
    #   2. DiVolo code TF, MFT code TF (using paper params)
    #       - are the TFs similar to paper, or to data, or completely different?
    #   3. DiVolo code TF, MFT code TF (using new fit params)
    #      - are the TFs similar to data now?


    pass


workflow_params = {
    "neuron_simulation" : {  # the same name as module
        "run_simulation" : True,
        "load_simulation" : False,
        # if both True - simulation will run, then loaded data and compared to check if they are the same
        # if they are not the same - warning will be printed, but the workflow will continue with simulated data

        "adaptive_grid" : True,
        # if True - the grid will be adapted to the data, so that we have more points in the interesting regions (e.g. where the TF changes rapidly)
        # if False - the grid will be fixed, and defined by the following parameters:
        "adaptive_input" : "exc_rate",  # This specifies which input to adapt the grid to. It can be "exc_rate", "inh_rate"
        "exc_rate_range" : [0, 60, 16],
        "inh_rate_range" : [0, 60, 16],
        "out_rate_range" : [0, 60, 16],
        # Choice is either list of 3 numbers [min, max, n_points]
        # or 2D mesh

        "max_grid_step" : 4,  # TODO: cut it out, can be purely automatic 

        "simulation_time" : 5000,  # [ms], total simulation time
        "averaging_window" : 2000,  # [ms], time window for calculating the mean output rate, starting from the end of the simulation
        "time_step" : 0.1,  # [ms], time step for the simulation
        "seed" : 42,
        "n_runs" : 5,  # [int], number of runs for the simulation
        "cpus" : 16  # [int], number of CPUs to use for parallel simulations
    },
    "transfer_function" : {  # the same name as module

    },
    "network_simulation" : {  # the same name as module

        "network_models" : {
            "SNN" : {
                "init_values" : {
                    "exc_neuron" : {
                        "w" : 0.00,                         # [nA], Default value of adaptation current
                        "v" : -65.0,                        # [mV], Default value of membrane potential
                    },
                    "inh_neuron" : {
                        "w" : 0.00,                         # [nA], Default value of adaptation current
                        "v" : -65.0,                        # [mV], Default value of membrane potential
                    },
                },
            },
            "DiVolo2019-MF" : {
                "T" : 40.0,
                "tvb_model" : "DiVolo_Tsodyks_second_order",
                "init_values" : {
                    "E" : [0.000, 0.000],
                    "I" : [0.00, 0.00],
                    "C_ee" : [0.0000005, 0.0000005],
                    "C_ei" : [0.0000005, 0.0000005],
                    "C_ii" : [0.0000005, 0.0000005],
                    "W_e" : [50.0, 50.0],
                    "W_i" : [0.0, 0.0],
                    "X_e" : [1.0, 1.0],
                    "Y_e" : [0.0, 0.0],
                    "X_i" : [1.0, 1.0],
                    "Y_i" : [0.0, 0.0],
                    "noise" : [0.0, 0.0],
                    "stimulus" : [0.0, 0.0]
                }
            }
        },
    }
}





if __name__ == "__main__":
    main()