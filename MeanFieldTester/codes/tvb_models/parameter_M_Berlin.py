import numpy as np
import os

from pathlib import Path

class Parameter :
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation={
            'path_result':'./results/synch/',
            'seed':10, # the seed for the random generator
            'save_time': 1000.0, # the time of simulation in each file
        }

        self.parameter_model ={
            'matteo':False,
            # true ==> imports Zerlaut_matteo.py
            # false ==> imports Zerlaut.py
            # Basically no difference! It does not makes the difference between
            # Zerlaut2018 paper without adaptation and diVolo2019 paper with adaptation
            # They both include adaptation they just deal differently with
            # the zero activity (to avoid errors in calculations)
            
            #order of the model
            'order':2,
            #parameter of the model
            'g_L':10.0,
            'C_m':200.0,
            'E_L_e':-63.0,
            'E_L_i':-65.0,
            # adaptive parameters
            'b_e':60.0,
            'a_e':0.0,
            'tau_w_e':500.0,
            'b_i':0.0,
            'a_i':0.0,
            'tau_w_i':1.0,
            # synaptic parameters
            'E_e':0.0,
            'E_i':-80.0,
            'Q_e':1.5,
            'Q_i':5.0,
            'tau_e':5.0,
            'tau_i':5.0,
            'N_tot':10000,
            # connectivity parameters (for the excitatory and inhibitory populations)
            # 
            'p_connect_e':0.05,
            'p_connect_i':0.05,
            'g':0.2,
            # MF parameters
            'T':40.0,
            'P_e':[-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
            'P_i':[-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
            # external drive
            # probably {target_type}_{source_type}
            'external_input_ex_ex':0.315*1e-3,
            'external_input_ex_in':0.000,
            'external_input_in_ex':0.315*1e-3,
            'external_input_in_in':0.000,
            # drive population size
            'K_ext_e':400,
            'K_ext_i':0,
            # noise parameters
            'tau_OU':5.0,
            'weight_noise': 0, # 1e-4, #10.5*1e-5,
            # Initial condition [exc_pop, inh_pop]
            'initial_condition':{
                "E": [0.000, 0.000],"I": [0.00, 0.00],"C_ee": [0.0,0.0],"C_ei": [0.0,0.0],"C_ii": [0.0,0.0],"W_e": [100.0, 100.0],"W_i": [0.0,0.0],"noise":[0.0,0.0], "stimulus": [0.0, 0.0]}
        }

        self.parameter_connection_between_region={
            ## CONNECTIVITY
            # connectivity by default
            'default':False,
            #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_file':False,
            'from_folder':True,
            'from_h5':False,
            # 'path':path+'/../../data/QL_20120814/', #the files
            'path':path+'/../data/', #the files
            # File description
            'number_of_regions':0, # number of regions
            # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'tract_lengths':[],
            # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'weights':[],
            # speed of along long range connection
            'speed':4.0,
            #normalised':True
            'normalised':False
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter':{'a':0.3,   #Peut etre modifie
                         'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                'nsig':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':[0,1,2,3,4,5,6,7,8],
                'period':self.parameter_integrator['dt']*10.0
            },
            'Bold':False,
            'parameter_Bold':{
                'variables_of_interest':[0],
                'period':self.parameter_integrator['dt']*2000.0
            },
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1
            }
        }


        self.parameter_stimulus = {
            'onset': 99.0,
            "tau": 9.0,
            "T": 99.0,
            "weights": None,
            "variables":[0]
        }



class CSNGParameter :
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation={
            'path_result':'./result/synch/',
            'seed':10, # the seed for the random generator
            'save_time': 1000.0, # the time of simulation in each file
        }

        self.parameter_model ={
            'matteo':False,
            #order of the model
            'order':2,
            #parameter of the model
            'g_L':4.0,
            'E_L_e':-63.0,
            'E_L_i':-65.0,
            'C_m':32.0,
            'b_e':80.0,
            'a_e':-0.8,
            'b_i':0.0,
            'a_i':0.0,
            'tau_w_e':1.0,
            'tau_w_i':1.0,
            'E_e':0.0,
            'E_i':-80.0,
            'Q_e':0.18,
            'Q_i':1.0,
            'tau_e':1.5,
            'tau_i':4.2,
            'N_tot':10000,
            'p_connect_e':0.084,
            'p_connect_i':0.054,
            'g':0.2,
            'T':40.0,
            'P_e':[-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
            'P_i':[-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
            'external_input_ex_ex':0.315*1e-3,
            'external_input_ex_in':0.000,
            'external_input_in_ex':0.315*1e-3,
            'external_input_in_in':0.000,
            'tau_OU':5.0,
            'weight_noise': 0, # 1e-4, #10.5*1e-5,
            'K_ext_e':400,
            'K_ext_i':0,
            #Initial condition :
            'initial_condition':{
                "E": [0.000, 0.000],"I": [0.00, 0.00],"C_ee": [0.0,0.0],"C_ei": [0.0,0.0],"C_ii": [0.0,0.0],"W_e": [100.0, 100.0],"W_i": [0.0,0.0],"noise":[0.0,0.0]}
        }

        self.parameter_connection_between_region={
            ## CONNECTIVITY
            # connectivity by default
            'default':False,
            #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_file':False,
            'from_folder':True,
            'from_h5':False,
            #'path':path+'/../../data/QL_20120814/', #the files
            'path':path+'/../../data/QL_20120814', #the files
            # File description
            'number_of_regions':0, # number of regions
            # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'tract_lengths':[],
            # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'weights':[],
            # speed of along long range connection
            'speed':4.0,
            #normalised':True
            'normalised':False
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter':{'a':0.3,   #Peut etre modifie
                         'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                'nsig':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':[0,1,2,3,4,5,6,7],
                'period':self.parameter_integrator['dt']*10.0
            },
            'Bold':False,
            'parameter_Bold':{
                'variables_of_interest':[0],
                'period':self.parameter_integrator['dt']*2000.0
            },
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1
            }
        }


        self.parameter_stimulus = {
            'onset': 99.0,
            "tau": 9.0,
            "T": 99.0,
            "weights": None,
            "variables":[0]
        }


class DiVoloParameter :
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation={
            'path_result':'./result/synch/',
            'seed':10, # the seed for the random generator
            'save_time': 1000.0, # the time of simulation in each file
        }

        self.parameter_model ={
            'matteo':False,
            #order of the model
            'order':2,
            #parameter of the model
            'g_L':10.0,
            'E_L_e':-65.0,
            'E_L_i':-65.0,
            'C_m':200.0,
            'b_e':20.0,
            'a_e':4.0,
            'b_i':0.0,
            'a_i':0.0,
            'tau_w_e':500.0,
            'tau_w_i':1.0,
            'E_e':0.0,
            'E_i':-80.0,
            'Q_e':1.5,
            'Q_i':5.0,
            'tau_e':5.0,
            'tau_i':5.0,
            'N_tot':10000,
            'p_connect_e':0.05,
            'p_connect_i':0.05,
            'g':0.2,
            'T':40.0,
            'P_e':[-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
            'P_i':[-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
            'external_input_ex_ex':0.315*1e-3,
            'external_input_ex_in':0.000,
            'external_input_in_ex':0.315*1e-3,
            'external_input_in_in':0.000,
            'tau_OU':5.0,
            'weight_noise': 0, # 1e-4, #10.5*1e-5,
            'K_ext_e':400,
            'K_ext_i':0,
            #Initial condition :
            'initial_condition':{
                "E": [0.000, 0.000],"I": [0.00, 0.00],"C_ee": [0.0,0.0],"C_ei": [0.0,0.0],"C_ii": [0.0,0.0],"W_e": [100.0, 100.0],"W_i": [0.0,0.0],"noise":[0.0,0.0]}
        }

        self.parameter_connection_between_region={
            ## CONNECTIVITY
            # connectivity by default
            'default':False,
            #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_file':False,
            'from_folder':True,
            'from_h5':False,
            #'path':path+'/../../data/QL_20120814/', #the files
            'path':path+'/../../data/QL_20120814', #the files
            # File description
            'number_of_regions':0, # number of regions
            # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'tract_lengths':[],
            # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'weights':[],
            # speed of along long range connection
            'speed':4.0,
            #normalised':True
            'normalised':False
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter':{'a':0.3,   #Peut etre modifie
                         'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                'nsig':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':[0,1,2,3,4,5,6,7],
                'period':self.parameter_integrator['dt']*10.0
            },
            'Bold':False,
            'parameter_Bold':{
                'variables_of_interest':[0],
                'period':self.parameter_integrator['dt']*2000.0
            },
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1
            }
        }


        self.parameter_stimulus = {
            'onset': 99.0,
            "tau": 9.0,
            "T": 99.0,
            "weights": None,
            "variables":[0]
        }