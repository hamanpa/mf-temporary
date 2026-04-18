"""
This is integration of the Zerlaut2018 TF code into our framework.

"""

from typing import List
import numpy as np
from .base import BaseTransferFunction
from ..network_params.translators import translate_params, TranslationRule


ZERLAUT2018_ADEX_MAPPING_SI = {
    'Gl': TranslationRule("g_L", sim_unit="S"),
    'Cm': TranslationRule("cm", sim_unit="F"),
    'Trefrac': TranslationRule("tau_refrac", sim_unit="s"),
    'El': TranslationRule("v_rest", sim_unit="V"),
    'Vthre': TranslationRule("v_thresh", sim_unit="V"),
    'Vreset': TranslationRule("v_reset", sim_unit="V"),
    'delta_v': TranslationRule("delta_T", sim_unit="V"),
    'a': TranslationRule("a", sim_unit="S"),
    'b': TranslationRule("b", sim_unit="A"),
    'tauw': TranslationRule("tau_w", sim_unit="s"),
    'Te': TranslationRule(mft_name='tau_syn_E', sim_unit='s'),
    'Ee': TranslationRule(mft_name='e_rev_E', sim_unit='V'),
    'Ti': TranslationRule(mft_name='tau_syn_I', sim_unit='s'),
    'Ei': TranslationRule(mft_name='e_rev_I', sim_unit='V'),
}

class Zerlaut2018TF(BaseTransferFunction):
    """
    Adapter class for the Zerlaut 2018 Transfer Function.
    Wraps the original published code without modifying its underlying logic.
    """

    def required_inputs(self) -> List[str]:
        return ["exc_rate", "inh_rate"]

    def _get_legacy_params_dict(self) -> dict:
        """
        Translates the modern MFT network_params into the dictionary 
        format expected by the legacy Zerlaut code.
        
        TODO for user: Ensure the attribute names here exactly match the 
        attributes in your BiologicalParameters class. Perform any unit 
        conversions here to keep the original code untouched.
        """
        exc_neuron_name = self.network_params.exc_neuron_name
        inh_neuron_name = self.network_params.inh_neuron_name

        params = {
            **translate_params(
                self.network_params.neurons[self.neuron_name].neuron_params, 
                ZERLAUT2018_ADEX_MAPPING_SI),
            **translate_params(
                self.network_params.synapses[exc_neuron_name].syn_params,
                {'Qe': TranslationRule("weight", sim_unit="S")}),
            **translate_params(
                self.network_params.synapses[inh_neuron_name].syn_params,
                {'Qi': TranslationRule("weight", sim_unit="S")}),
            'pconnec': self.network_params.network.connectivity[exc_neuron_name][exc_neuron_name],
            'Ntot': self.network_params.internal_size,
            'gei': self.network_params.g,
        }

        return params

    def fit(self, single_neuron_results, **kwargs) -> dict:
        """
        Extracts grid data and uses the original Zerlaut fitting logic.
        """

        Fe_eff = single_neuron_results.exc_rate_grid.T
        fiSim = single_neuron_results.inh_rate_grid.T
        Fout = single_neuron_results.out_rate_mean.T
        
        params = self._get_legacy_params_dict()
        
        with_square_terms = self.tf_params.tf_model.square_terms
        
        P_array = fitting_Vthre_then_Fout(
            Fout=Fout, 
            Fe_eff=Fe_eff, 
            fiSim=fiSim, 
            params=params,
            with_square_terms=with_square_terms
        )
        
        self.fitted_params = {f"P{i}": P_array[i] for i in range(len(P_array))}
        self.is_fitted = True
        
        return {"status": "success", "num_params": len(P_array)}

    def evaluate(self, **kwargs) -> np.ndarray:
        """
        Maps standard kwargs to the massive argument list required by TF_my_template.
        """
        fe = kwargs["exc_rate"]
        fi = kwargs["inh_rate"]
        
        p = self._get_legacy_params_dict()
        
        P_coeffs = [self.fitted_params.get(f"P{i}", 0.0) for i in range(11)]
        
        return TF_my_template(
            fe, fi, 
            p['Qe'], p['Te'], p['Ee'], 
            p['Qi'], p['Ti'], p['Ei'], 
            p['Gl'], p['Cm'], p['El'], 
            p['Ntot'], p['pconnec'], p['gei'], 
            *P_coeffs
        )

################################################################################
# Below is the original, published code from Zerlaut2018, which we will adapt into the above class.
# We do not touch it if not necessary, to preserve the original logic and fitting procedure.
# original code source: https://github.com/ModelDBRepository/234992/tree/master
################################################################################

import numpy as np
import scipy.special as sp_spec
import scipy.integrate as sp_int
from scipy.optimize import minimize, curve_fit
import sys


def pseq_params(params):
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    for key, dval in zip(['Ntot', 'pconnec', 'gei'], [1, 2., 0.5]):
        if not key in params.keys():
            params[key] = dval

    if 'P' in params.keys():
        P = params['P']
    else: # no correction
        P = [-45e-3]
        for i in range(1,11):
            P.append(0)

    # return Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10
    return params['Qe'], params['Te'], params['Ee'], params['Qi'], params['Ti'], params['Ei'], params['Gl'], params['Cm'], params['El'], params['Ntot'], params['pconnec'], params['gei'], P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]

def get_fluct_regime_vars(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    fi = Fi*gei*pconnec*Ntot
    
    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    muV = (muGe*Ee+muGi*Ei+Gl*El)/muG
    muGn, Tm = muG/Gl, Cm/muG
    
    Ue, Ui = Qe/muG*(Ee-muV), Qi/muG*(Ei-muV)

    sV = np.sqrt(\
                 fe*(Ue*Te)**2/2./(Te+Tm)+\
                 fi*(Qi*Ui)**2/2./(Ti+Tm))

    fe, fi = fe+1e-9, fi+1e-9 # just to insure a non zero division, 
    Tv = ( fe*(Ue*Te)**2 + fi*(Qi*Ui)**2 ) /( fe*(Ue*Te)**2/(Te+Tm) + fi*(Qi*Ui)**2/(Ti+Tm) )
    TvN = Tv*Gl/Cm

    return muV, sV+1e-12, muGn, TvN

def mean_and_var_conductance(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    fi = Fi*gei*pconnec*Ntot
    return Qe*Te*fe, Qi*Ti*fi, Qe*np.sqrt(Te*fe/2.), Qi*np.sqrt(Ti*fi/2.)


### FUNCTION, INVERSE FUNCTION
def erfc_func(muV, sV, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*\
      sp_spec.erfc((Vthre-muV)/np.sqrt(2)/sV)

def effective_Vthre(Y, muV, sV, TvN, Gl, Cm):
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff

def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    setting by default to True the square
    because when use by external modules, coeff[5:]=np.zeros(3)
    in the case of a linear threshold
    """
    muV0, DmuV0 = -60e-3,10e-3
    sV0, DsV0 =4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.
    return P0+P1*(muV-muV0)/DmuV0+\
        P2*(sV-sV0)/DsV0+P3*(TvN-TvN0)/DTvN0+\
        P4*np.log(muGn)+P5*((muV-muV0)/DmuV0)**2+\
        P6*((sV-sV0)/DsV0)**2+P7*((TvN-TvN0)/DTvN0)**2+\
        P8*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
        P9*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
        P10*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0
      
# final transfer function template :
def TF_my_template(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    muV, sV, muGn, TvN = get_fluct_regime_vars(fe, fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    return Fout_th

def make_loop(t, nu, vm, nu_aff_exc, nu_aff_inh, BIN,\
              Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    dt = t[1]-t[0]
    # constructing the Euler method for the activity rate
    for i_t in range(len(t)-1): # loop over time
        
        fe = (nu_aff_exc[i_t]+nu[i_t]+Fdrive) # afferent+recurrent excitation
        fi = nu[i_t]+nu_aff_inh[i_t] # recurrent inhibition
        W[i_t+1] = W[i_t] + dt/Tw*(b*nu[i_t]*Tw - W[i_t])

        nu[i_t+1] = nu[i_t] +\
               dt/BIN*(\
                TF_my_template(fe, fi, W[i_t], Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)\
                -nu[i_t])

        vm[i_t], _, _, _ = get_fluct_regime_vars(fe, fi, W[i_t], Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)

    return nu, vm, W


################################################################
##### Now fitting to Transfer Function data
################################################################


def fitting_Vthre_then_Fout(Fout, Fe_eff, fiSim, params,\
                            maxiter=10000, xtol=1e-5,
                            verbose=False,
                            with_square_terms=False):

    Fout, Fe_eff, fiSim = Fout.flatten(), Fe_eff.flatten(), fiSim.flatten()
    
    muV, sV, muGn, TvN = get_fluct_regime_vars(Fe_eff, fiSim, *pseq_params(params))
    i_non_zeros = np.where(Fout>0)

    Vthre_eff = effective_Vthre(Fout[i_non_zeros], muV[i_non_zeros],\
                sV[i_non_zeros], TvN[i_non_zeros], params['Gl'], params['Cm'])
    
    if with_square_terms:
        P = np.zeros(11)
    else:
        P = np.zeros(5)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    def Res(p):
        if not with_square_terms:
            pp = np.concatenate([p, np.zeros(6)])
        else:
            pp=p
        vthre = threshold_func(muV[i_non_zeros], sV[i_non_zeros],\
                               TvN[i_non_zeros], muGn[i_non_zeros], *pp)
        return np.mean((Vthre_eff-vthre)**2)
    
    plsq = minimize(Res, P, method='SLSQP',\
                    options={'ftol': 1e-8, 'disp': True, 'maxiter':40000})

    if verbose:
        print(plsq)

    P = plsq.x
    
    def Res(p):
        if not with_square_terms:
            params['P'] = np.concatenate([p, np.zeros(6)])
        else:
            params['P'] = p
        return np.mean((Fout-\
                        TF_my_template(Fe_eff, fiSim, *pseq_params(params)))**2)

    plsq = minimize(Res, P, method='nelder-mead',\
            options={'xtol': xtol, 'disp': True, 'maxiter':maxiter})

    if verbose:
        print(plsq)
    
    if with_square_terms:
        return plsq.x
    else:
        return np.concatenate([plsq.x, np.zeros(6)])

def make_fit_from_data(DATA, with_square_terms=False,
                       verbose=False):

    MEANfreq, SDfreq, Fe_eff, fiSim, params = np.load(DATA)

    Fe_eff, Fout = np.array(Fe_eff), np.array(MEANfreq)
    levels = fiSim # to store for colors
    fiSim = np.meshgrid(np.zeros(Fe_eff.shape[1]), fiSim)[1]

    P = fitting_Vthre_then_Fout(Fout, Fe_eff, fiSim, params,\
                                with_square_terms=with_square_terms,
                                verbose=verbose)
                            
    print('==================================================')
    print(1e3*np.array(P), 'mV')

    # then we save it:
    filename = DATA.replace('.npy', '_fit.npy')
    print('coefficients saved in ', filename)
    np.save(filename, np.array(P))

    return P

    
# import argparse
# if __name__=='__main__':
#     # First a nice documentation 
#     parser=argparse.ArgumentParser(description=
#      """ 
#      '=================================================='
#      '=====> FIT of the transfer function =============='
#      '=== and theoretical objects for the TF relation =='
#      '=================================================='
#      """,
#               formatter_class=argparse.RawTextHelpFormatter)

#     parser.add_argument('-f', "--FILE",help="file name of numerical TF data",\
#                         default='data/example_data.npy')
#     parser.add_argument("--With_Square",help="Add the square terms in the TF formula"+\
#                         "\n then we have 7 parameters",\
#                          action="store_true")
#     args = parser.parse_args()

#     make_fit_from_data(args.FILE, with_square_terms=args.With_Square)