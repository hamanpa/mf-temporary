################################################################################
# What follows is old code, I keep it for now until I have working new code


class TransferFunction:

    def __init__(self, coefs_fit, expansion_point, expansion_norm, params, square_terms=False, log_term=False):
        g_L = params['neuron_params']['cm'] / params['neuron_params']['tau_m'] * 1e3
        self.v_eff = V_eff(expansion_point, expansion_norm, coefs=coefs_fit, g_L=g_L,
                           square_terms=square_terms, log_term=log_term)
        self.params = params

    def __call__(self, nu_e, nu_i, flattened=False):
        mu_V, sigma_V, tau_V, tau_VN, mu_G = MeanPotentialFluctuations(self.params)(nu_e, nu_i, flattened=flattened)
        v_eff = self.v_eff(mu_V, sigma_V, tau_VN, mu_G)
        sigma_V[sigma_V<1e-9]=1e-9  # to avoid division by zero
        return 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))
    
    def make_fit(self, nu_out, nu_e, nu_i, method='nelder-mead', options={'fatol': 5e-9, 'disp': True, 'maxiter':10000}, flattened=False):
        def func_to_minimize(coefs, *args):
            nu_out, nu_e, nu_i, params, expansion_point, expansion_norm, g_L, square_terms, log_term = args
            v_eff = V_eff(expansion_point, expansion_norm, coefs=coefs, g_L=g_L,
                           square_terms=square_terms, log_term=log_term)
            mu_V, sigma_V, tau_V, tau_VN, mu_G = MeanPotentialFluctuations(self.params)(nu_e, nu_i, flattened=flattened)
            v_eff = v_eff(mu_V, sigma_V, tau_VN, mu_G)
            sigma_V[sigma_V<1e-9]=1e-9  # to avoid division by zero
            tf = 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))
            return np.mean((nu_out - tf)**2)

        args = (nu_out, nu_e, nu_i, self.params, self.v_eff.expansion_point, 
                self.v_eff.expansion_norm, self.v_eff.g_L, 
                self.v_eff.square_terms, self.v_eff.log_term)
        # method='SLSQP'
        # options={'ftol': 1e-2, 'disp': False, 'maxiter':4000}
        res = minimize(func_to_minimize, self.v_eff.coefs, args=args, method=method, options=options)
        self.v_eff.coefs = res.x


class TransferFunctionAdaptation(TransferFunction):
    def __call__(self, nu_e, nu_i, w, flattened=False):
        mu_V, sigma_V, tau_V, tau_VN, mu_G = MPF_with_adaptation(self.params)(nu_e, nu_i, w, flattened=flattened)
        v_eff = self.v_eff(mu_V, sigma_V, tau_VN, mu_G)
        return 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))

    def make_fit(self, nu_out, nu_e, nu_i, method='nelder-mead', options={'xatol': 5e-4, 'disp': False, 'maxiter':10000}, flattened=False):
        def func_to_minimize(coefs, *args):
            nu_out, nu_e, nu_i, params, expansion_point, expansion_norm, g_L, square_terms, log_term = args
            v_eff = V_eff(expansion_point, expansion_norm, coefs=coefs, g_L=g_L,
                           square_terms=square_terms, log_term=log_term)
            mu_V, sigma_V, tau_V, tau_VN, mu_G = MPF_with_nu_out(params)(nu_e, nu_i, nu_out, flattened=flattened)
            v_eff = v_eff(mu_V, sigma_V, tau_VN, mu_G)
            sigma_V[sigma_V<1e-9]=1e-9  # to avoid division by zero
            tf = 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))
            return np.mean((nu_out - tf)**2)

        args = (nu_out, nu_e, nu_i, self.params, self.v_eff.expansion_point, 
                self.v_eff.expansion_norm, self.v_eff.g_L, 
                self.v_eff.square_terms, self.v_eff.log_term)
        res = minimize(func_to_minimize, self.v_eff.coefs, args=args,
                       method=method, options=options)
        self.v_eff.coefs = res.x


class TransferFunctionAdaptation_FitWithW(TransferFunctionAdaptation):
    def make_fit(self, nu_out, nu_e, nu_i, w, method='nelder-mead', options={'xatol': 5e-4, 'disp': False, 'maxiter':10000}, flattened=False):
        def func_to_minimize(coefs, *args):
            nu_out, nu_e, nu_i, params, expansion_point, expansion_norm, g_L, square_terms, log_term = args
            v_eff = V_eff(expansion_point, expansion_norm, coefs=coefs, g_L=g_L,
                           square_terms=square_terms, log_term=log_term)
            mu_V, sigma_V, tau_V, tau_VN, mu_G = MPF_with_adaptation(params)(nu_e, nu_i, w, flattened=flattened)
            v_eff = v_eff(mu_V, sigma_V, tau_VN, mu_G)
            sigma_V[sigma_V<1e-9]=1e-9  # to avoid division by zero
            tf = 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))
            return np.mean((nu_out - tf)**2)

        args = (nu_out, nu_e, nu_i, self.params, self.v_eff.expansion_point, 
                self.v_eff.expansion_norm, self.v_eff.g_L, 
                self.v_eff.square_terms, self.v_eff.log_term)
        res = minimize(func_to_minimize, self.v_eff.coefs, args=args,
                       method=method, options=options)
        self.v_eff.coefs = res.x


class V_eff:

    def __init__(self, expansion_point, expansion_norm, coefs=None, g_L=None, square_terms=False, log_term=False):
        self.expansion_point = expansion_point
        self.expansion_norm = expansion_norm
        self.square_terms = square_terms
        self.log_term = log_term
        if log_term:
            assert g_L is not None, "g_L has to be provided for log term"
        self.g_L = g_L
        if coefs is not None:
            assert len(coefs) == 1+3+1*log_term+6*square_terms, "Invalid number of coefs"
        self.coefs = coefs

    def __call__(self, mu_V, sigma_V, tau_VN, mu_G):
        if self.coefs is None:
             raise RuntimeError("You must call the 'make_fit' method before calling this instance or provide coefs.") 
        mu_V, sigma_V, tau_VN, mu_G = convert_to_arrays(mu_V, sigma_V, tau_VN, mu_G)
        x_data = self._x_data(mu_V, sigma_V, tau_VN, mu_G)
        return self._bare_function(x_data, *self.coefs, square_terms=self.square_terms, log_term=self.log_term)

    @staticmethod
    def _bare_function(x_data, *coefs, square_terms=False, log_term=False):
        """
        x_data : 2D np.ndarray
        coefs : list
        """
        # Add constant term to the beginning
        x_vals = np.concatenate([np.ones_like(x_data[0])[np.newaxis], x_data], axis=0)

        if square_terms:  # quadratic expansion
            x1, x2, x3 = x_data[:3]
            point_quad = np.stack([x1*x1, x2*x2, x3*x3, x1*x2, x1*x3, x2*x3])
            x_vals = np.concatenate([x_vals, point_quad], axis=0)

        # Moves the first axis at the end so that it can be matrix-multiplied with coefs
        x_vals = np.moveaxis(x_vals, 0, -1)
        return x_vals @ np.array(coefs)


    @staticmethod
    def v_eff_from_data(nu_out, mu_V, sigma_V, tau_V, tau_VN):
        """Computes and return V_eff from the data (nu_out, mu_V, sigma_V, tau_V, tau_VN)"""
        mu_V, sigma_V, tau_V, tau_VN = convert_to_arrays(mu_V, sigma_V, tau_V, tau_VN)
        # nu_out[nu_out<1e-9]=1e-9
        # tau_V[tau_V<1e-9]=1e-9
        return np.sqrt(2)*sigma_V * erfcinv(2*tau_V * nu_out * 1e-3) + mu_V

    def make_curvefit(self, nu_out, mu_V, sigma_V, tau_V, tau_VN, mu_G, coefs_init=None):
        y_data = self.v_eff_from_data(nu_out, mu_V, sigma_V, tau_V, tau_VN)
        y_data, mu_V, sigma_V, tau_VN, mu_G = flatten_and_remove_nans(y_data, mu_V, sigma_V, tau_VN, mu_G)
        if coefs_init is None:
            coefs_init = [y_data.mean()] +[1.]*(3+1*self.log_term) + [0.]*6*self.square_terms
        x_data = self._x_data(mu_V, sigma_V, tau_VN, mu_G)

        function = lambda x_data, *coefs: self._bare_function(x_data, *coefs, square_terms=self.square_terms, log_term=self.log_term)

        coefs, pcov = curve_fit(function, x_data, y_data, p0=coefs_init)
        self.coefs = coefs

    def _x_data(self, mu_V, sigma_V, tau_VN, mu_G):
        mu_V, sigma_V, tau_VN, mu_G = convert_to_arrays(mu_V, sigma_V, tau_VN, mu_G)
        x_data = np.stack([mu_V, sigma_V, tau_VN], axis=0)
        x_data = move_and_rescale(x_data, self.expansion_point, self.expansion_norm, axis=0)
        mu_V, sigma_V, tau_VN = x_data
        if self.log_term:
            mu_G = np.log(mu_G/self.g_L)
            x_data = np.concatenate([x_data, mu_G[np.newaxis]], axis=0)
        return x_data


    def make_gradual_fit(self, nu_out, mu_V, sigma_V, tau_V, tau_VN, mu_G, coefs_init=None, method='minimize'):
        # This should make a fit of first order parameters and only after that
        # fit the second order parameters while keeping the first order parameters fixed

        y_data = self.v_eff_from_data(nu_out, mu_V, sigma_V, tau_V, tau_VN)
        y_data, mu_V, sigma_V, tau_VN, mu_G = flatten_and_remove_nans(y_data, mu_V, sigma_V, tau_VN, mu_G)
        if coefs_init is None:
            coefs_init = [y_data.mean()] +[1.]*(3+self.log_term) + [0.]*6*self.square_terms
        x_data = self._x_data(mu_V, sigma_V, tau_VN, mu_G)

        # Make the fit (with picking the method)
        if method == 'minimize':
            # func = func(x : 1d nd.array, *args) -> float
            # minimize looks for x such that func(x, *args) is minimal
            # thus x should correspond to params

            # Fit first order parameters
            func_1st_order = lambda pars: np.mean((y_data - self._bare_function(x_data, *pars, square_terms=False, log_term=self.log_term))**2)
            res = minimize(func_1st_order, coefs_init[:4+self.log_term], method='nelder-mead', tol=1e-14, options={'disp':False,'maxiter':20000})
            # res = minimize(func_1st_order, coefs_init[:4+self.log_term], method='SLSQP', tol=1e-32, options={'disp':False,'maxiter':20000})
            coefs = res.x

            if self.square_terms:
                # Fit second order parameters
                func_2nd_order = lambda pars: np.mean((y_data - self._bare_function(x_data, *coefs, *pars, square_terms=self.square_terms, log_term=True))**2)
                res = minimize(func_2nd_order, coefs_init[4+self.log_term:], method='nelder-mead', tol=1e-9, options={'disp':False,'maxiter':20000})
                # res = minimize(func_2nd_order, coefs_init[4+self.log_term:], method='SLSQP', options={'ftol':1e-14,'disp':False,'maxiter':20000})
                coefs = np.concatenate([coefs, res.x])

        elif method == 'curve_fit':
            # func = func(x : (k,M) nd.array, *params ) -> (M,) nd.array
            # curve_fit searches nd.array params 
            # such that func(x, *params) is as close as possible to y_data

            raise NotImplementedError
        else:
            raise NotImplementedError

        self.coefs = coefs

    def make_minimize_fit(self, nu_out, mu_V, sigma_V, tau_V, tau_VN, mu_G, 
                          coefs_init=None, method='SLSQP', 
                          options={'ftol': 1e-2, 'disp': False, 'maxiter':4000}):
        y_data = self.v_eff_from_data(nu_out, mu_V, sigma_V, tau_V, tau_VN)
        y_data, mu_V, sigma_V, tau_VN, mu_G = flatten_and_remove_nans(y_data, mu_V, sigma_V, tau_VN, mu_G)
        if coefs_init is None:
            coefs_init = [y_data.mean()] +[1.]*(3+self.log_term) + [0.]*6*self.square_terms
        x_data = self._x_data(mu_V, sigma_V, tau_VN, mu_G)

        # func = func(x : 1d nd.array, *args) -> float
        # minimize looks for x such that func(x, *args) is minimal
        # thus x should correspond to params

        def func_to_min(coefs):
            vthre = self._bare_function(x_data, *coefs, square_terms=self.square_terms, log_term=self.log_term)
            return np.mean((y_data - vthre)**2)
        res = minimize(func_to_min, coefs_init, method=method, options=options)
        self.coefs = res.x


class MeanPotentialFluctuations:
    """This is a base class for calculating mean potential fluctuations.
    
    This class ignores adaptation in the calculations, so it correspond to the
    Zerlaut model.
    """

    def __init__(self, params):

        # Saving neuron parameters
        self.tau_e = params['neuron_params']['tau_syn_E']
        self.tau_i = params['neuron_params']['tau_syn_I']
        self.tau_m = params['neuron_params']['tau_m']
        self.cm = params['neuron_params']['cm']
        self.g_L = self.cm / self.tau_m * 1e3
        self.e_rev_E = params['neuron_params']['e_rev_E']
        self.e_rev_I = params['neuron_params']['e_rev_I']
        self.v_rest = params['neuron_params']['v_rest']
        
        # Saving synaptic parameters
        self.num_e = params['exc_synapses']['number']
        self.num_i = params['inh_synapses']['number']
        self.exc_syn_type = params['exc_synapses']['syn_type']
        self.inh_syn_type = params['inh_synapses']['syn_type']
        if self.exc_syn_type == 'tsodyks_synapse':
            self.u_e = params['exc_synapses']['syn_params']['U']
            self.tau_rec_e = params['exc_synapses']['syn_params']['tau_rec']
        if self.inh_syn_type == 'tsodyks_synapse':
            self.u_i = params['inh_synapses']['syn_params']['U']
            self.tau_rec_i = params['inh_synapses']['syn_params']['tau_rec']
        self.weight_e = params['exc_synapses']['syn_params']['weight']
        self.weight_i = params['inh_synapses']['syn_params']['weight']

    def __call__(self, nu_e, nu_i, flattened=False):

        nu_e, nu_i = self.ensure_grid(nu_e, nu_i, flattened=flattened)
        stp_e, stp_i = self.short_term_plasticity(nu_e, nu_i)

        mu_V, mu_G = self.calculate_muV_muG(nu_e, nu_i, stp_e, stp_i)
        sigma_V, tau_V = self.calculate_sigmaV_tauV(nu_e, nu_i, stp_e, stp_i, mu_V, mu_G)
        tau_VN = tau_V / self.tau_m

        return mu_V.squeeze(), sigma_V.squeeze(), tau_V.squeeze(), tau_VN.squeeze(), mu_G.squeeze()

    def calculate_muV_muG(self, nu_e, nu_i, stp_e, stp_i):
        mu_Ge = nu_e * (self.tau_e*1e-3) * self.num_e * self.weight_e * stp_e 
        mu_Gi = nu_i * (self.tau_i*1e-3) * self.num_i * self.weight_i * stp_i  
        mu_G = mu_Ge + mu_Gi + self.g_L

        # NOTE: not very precise naming 
        mu_Ve = mu_Ge * self.e_rev_E
        mu_Vi = mu_Gi * self.e_rev_I
        mu_Vl = self.g_L * self.v_rest

        mu_V = (mu_Ve + mu_Vi + mu_Vl) / mu_G
        
        return mu_V, mu_G

    def calculate_sigmaV_tauV(self, nu_e, nu_i, stp_e, stp_i, mu_V, mu_G):
        tau_eff = self.cm / mu_G * 1e3  # 1e3 to convert to ms
        u_e = self.weight_e * stp_e *(self.e_rev_E - mu_V) / mu_G
        u_i = self.weight_i * stp_i *(self.e_rev_I - mu_V) / mu_G

        s_e = self.num_e * (nu_e * 1e-3) * ((u_e*self.tau_e)**2) / (2.*(tau_eff+self.tau_e))
        s_i = self.num_i * (nu_i * 1e-3) * ((u_i*self.tau_i)**2) / (2.*(tau_eff+self.tau_i))
        sigma_V = np.sqrt(s_e + s_i)

        t_e = self.num_e * nu_e * 1e-3 * (u_e*self.tau_e)**2 
        t_i = self.num_i * nu_i * 1e-3 * (u_i*self.tau_i)**2
        t_e[t_e<1e-9]=1e-9  # to avoid division by zero
        t_i[t_i<1e-9]=1e-9  # to avoid division by zero
        tau_V = (t_e + t_i) / (t_e/(tau_eff+self.tau_e) + t_i/(tau_eff+self.tau_i))
        return sigma_V, tau_V

    def short_term_plasticity(self, nu_e, nu_i) -> tuple:
        if self.exc_syn_type == 'tsodyks_synapse':
            stp_e = self._short_term_plasticity(nu_e, self.u_e, self.tau_rec_e)
        else:
            stp_e = 1.
        if self.inh_syn_type == 'tsodyks_synapse':
            stp_i = self._short_term_plasticity(nu_i, self.u_i, self.tau_rec_i)
        else:
            stp_i = 1.
        return stp_e, stp_i

    @staticmethod
    def _short_term_plasticity(rate, u, tau_rec):
        "Returns stationary limit of short-term plasticity."
        rate[rate<1e-9]=1e-9  # to avoid division by zero
        exp = np.exp(-1/(rate*tau_rec*1e-3))
        return u*(1-exp)/(1-(1-u)*exp)

    @staticmethod
    def ensure_grid(nu_e, nu_i, flattened=False):
        nu_e = convert_to_array(nu_e)
        nu_i = convert_to_array(nu_i)

        if flattened is True:
            return nu_e, nu_i

        # NOTE: could also be sparse meshgrid!!!!!!  --> then it will raise an error
        if (nu_e.ndim, nu_i.ndim) == (1,1):
            nu_e, nu_i = np.meshgrid(nu_e, nu_i, sparse=True, indexing='ij')
        elif (nu_e.ndim, nu_i.ndim) == (1,2) and nu_e.size == nu_i.shape[0]:
            nu_e = nu_e[:,np.newaxis]
        elif (nu_e.ndim, nu_i.ndim) == (2,1) and nu_e.shape[1] == nu_i.size:
            nu_i = nu_i[np.newaxis]
        elif (nu_e.ndim, nu_i.ndim) == (2,2) and nu_e.shape == nu_i.shape:
            pass  # no action needed
        else:
            raise ValueError("Invalid shape of nu_e and nu_i")
        return nu_e, nu_i


class MPF_with_nu_out(MeanPotentialFluctuations):

    def __init__(self, params):
        super().__init__(params)
        self.tau_w = params['neuron_params']['tau_w']
        self.a = params['neuron_params']['a']
        self.b = params['neuron_params']['b']

    def __call__(self, nu_e, nu_i, nu_out, flattened=False):
        # NOTE: check the correct shape of nu_out
        # assert nu_out.shape == nu_e.size, nu_i.size
        nu_e, nu_i = self.ensure_grid(nu_e, nu_i, flattened=flattened)
        stp_e, stp_i = self.short_term_plasticity(nu_e, nu_i)

        mu_V, mu_G = self.calculate_muV_muG(nu_e, nu_i, nu_out, stp_e, stp_i)
        sigma_V, tau_V = self.calculate_sigmaV_tauV(nu_e, nu_i, stp_e, stp_i, mu_V, mu_G)
        tau_VN = tau_V / self.tau_m

        return mu_V.squeeze(), sigma_V.squeeze(), tau_V.squeeze(), tau_VN.squeeze(), mu_G.squeeze()

    def calculate_muV_muG(self, nu_e, nu_i, nu_out, stp_e, stp_i):
        mu_Ge = nu_e * (self.tau_e*1e-3) * self.num_e * self.weight_e * stp_e 
        mu_Gi = nu_i * (self.tau_i*1e-3) * self.num_i * self.weight_i * stp_i  
        mu_G = mu_Ge + mu_Gi + self.g_L

        # NOTE: not very precise naming 
        mu_Ve = mu_Ge * self.e_rev_E  # nS*mV
        mu_Vi = mu_Gi * self.e_rev_I  # nS*mV
        mu_Vl = self.g_L * self.v_rest  # nS*mV
        adaptation = nu_out*(self.tau_w*1e-3)*self.b - self.a*self.v_rest*1e-3  # nA (V*nS)

        mu_V = (mu_Ve + mu_Vi + mu_Vl - adaptation*1e3) / (mu_G + self.a)  # nS*mV/nS
        return mu_V, mu_G


class MPF_with_adaptation(MeanPotentialFluctuations):

    def __init__(self, params):
        super().__init__(params)
        self.tau_w = params['neuron_params']['tau_w']
        self.a = params['neuron_params']['a']
        self.b = params['neuron_params']['b']

    def __call__(self, nu_e, nu_i, w, flattened=False):
        # NOTE: expected unit of w is nA
        # NOTE: check the correct shape of w
        # assert w.shape == nu_e.size, nu_i.size
        nu_e, nu_i = self.ensure_grid(nu_e, nu_i, flattened=flattened)
        stp_e, stp_i = self.short_term_plasticity(nu_e, nu_i)

        mu_V, mu_G = self.calculate_muV_muG(nu_e, nu_i, w, stp_e, stp_i)
        sigma_V, tau_V = self.calculate_sigmaV_tauV(nu_e, nu_i, stp_e, stp_i, mu_V, mu_G)
        tau_VN = tau_V / self.tau_m

        return mu_V.squeeze(), sigma_V.squeeze(), tau_V.squeeze(), tau_VN.squeeze(), mu_G.squeeze()

    def calculate_muV_muG(self, nu_e, nu_i, w, stp_e, stp_i):
        mu_Ge = nu_e * (self.tau_e*1e-3) * self.num_e * self.weight_e * stp_e 
        mu_Gi = nu_i * (self.tau_i*1e-3) * self.num_i * self.weight_i * stp_i  
        mu_G = mu_Ge + mu_Gi + self.g_L

        # NOTE: not very precise naming 
        mu_Ve = mu_Ge * self.e_rev_E  # nS*mV
        mu_Vi = mu_Gi * self.e_rev_I  # nS*mV
        mu_Vl = self.g_L * self.v_rest  # nS*mV

        mu_V = (mu_Ve + mu_Vi + mu_Vl - w*1e3) / mu_G   # nS*mV/nS
        return mu_V, mu_G


def fit_tf(nu_e, nu_i, nu_out, neuron_pars, fit_pars, expansion_point, expansion_norm):
    # compute theoretical membrane potential fluctuations
    mu_V, sigma_V, tau_V, tau_VN, mu_G = MeanPotentialFluctuations(neuron_pars)(nu_e, nu_i)

    # create V_eff instance (to compute V_eff from data)
    if fit_pars['log_term']:
        g_L = neuron_pars['neuron_params']['cm'] / neuron_pars['neuron_params']['tau_m'] * 1e3
    else:
        g_L = None
    v_eff = V_eff(expansion_point, expansion_norm, 
                  square_terms=fit_pars['square_terms'], 
                  log_term=fit_pars['log_term'], g_L=g_L)

    # 1st fit - fitting V_eff function

    # NOTE: To get nicer fit we can discard the zero activity values
    nu_out_min = fit_pars['nu_out_min']
    nu_out_max = fit_pars['nu_out_max']

    nonzero_mask = (nu_out.flatten() > nu_out_min) & (nu_out.flatten() < nu_out_max)
    nu_out_masked = nu_out.flatten()[nonzero_mask]
    mu_V = mu_V.flatten()[nonzero_mask]
    sigma_V = sigma_V.flatten()[nonzero_mask]
    tau_V = tau_V.flatten()[nonzero_mask]
    tau_VN = tau_VN.flatten()[nonzero_mask]
    mu_G = mu_G.flatten()[nonzero_mask]

    v_eff.make_minimize_fit(nu_out_masked, mu_V, sigma_V, tau_V, tau_VN, mu_G, **fit_pars['V_eff_fitting'])
    print("V_eff coefs:", v_eff.coefs)

    # 2nd fit - fitting transfer function

    mask = nu_out.flatten() < nu_out_max

    tf = TransferFunction(v_eff.coefs, expansion_point, expansion_norm,
                          neuron_pars, square_terms=fit_pars['square_terms'], log_term=fit_pars['log_term'])
    tf.make_fit(nu_out.flatten()[mask], nu_e.flatten()[mask], nu_i.flatten()[mask], 
                flattened=True, **fit_pars['TF_fitting'])
    return tf

def fit_tf_adaptation(nu_e, nu_i, nu_out, w, neuron_pars, fit_pars, expansion_point, expansion_norm):
    # compute theoretical membrane potential fluctuations
    mu_V, sigma_V, tau_V, tau_VN, mu_G = MPF_with_nu_out(neuron_pars)(nu_e, nu_i, nu_out)

    # create V_eff instance (to compute V_eff from data)
    if fit_pars['log_term']:
        g_L = neuron_pars['neuron_params']['cm'] / neuron_pars['neuron_params']['tau_m'] * 1e3
    else:
        g_L = None
    v_eff = V_eff(expansion_point, expansion_norm, 
                  square_terms=fit_pars['square_terms'], 
                  log_term=fit_pars['log_term'], g_L=g_L)

    # 1st fit - fitting V_eff function

    # NOTE: To get nicer fit we can discard the zero activity values
    nu_out_min = fit_pars['nu_out_min']
    nu_out_max = fit_pars['nu_out_max']

    nonzero_mask = (nu_out.flatten() > nu_out_min) & (nu_out.flatten() < nu_out_max)
    nu_out_masked = nu_out.flatten()[nonzero_mask]
    mu_V = mu_V.flatten()[nonzero_mask]
    sigma_V = sigma_V.flatten()[nonzero_mask]
    tau_V = tau_V.flatten()[nonzero_mask]
    tau_VN = tau_VN.flatten()[nonzero_mask]
    mu_G = mu_G.flatten()[nonzero_mask]

    v_eff.make_minimize_fit(nu_out_masked, mu_V, sigma_V, tau_V, tau_VN, mu_G, **fit_pars['V_eff_fitting'])

    # 2nd fit - fitting transfer function

    mask = nu_out.flatten() < nu_out_max
    if fit_pars['fit_with_w']:
        tf = TransferFunctionAdaptation_FitWithW(v_eff.coefs, expansion_point, expansion_norm,
                            neuron_pars, square_terms=fit_pars['square_terms'], log_term=fit_pars['log_term'])
        tf.make_fit(nu_out.flatten()[mask], nu_e.flatten()[mask], nu_i.flatten()[mask], w.flatten()[mask],
                    flattened=True, **fit_pars['TF_fitting'])
        pass
    else:
        tf = TransferFunctionAdaptation(v_eff.coefs, expansion_point, expansion_norm,
                            neuron_pars, square_terms=fit_pars['square_terms'], log_term=fit_pars['log_term'])
        tf.make_fit(nu_out.flatten()[mask], nu_e.flatten()[mask], nu_i.flatten()[mask], 
                    flattened=True, **fit_pars['TF_fitting'])
    return tf

def run_fitting_workflow(tf_sim_pars:dict, network_pars:dict, neuron_results:dict, param_file:Path):
    """This function runs the transfer function fitting workflow. 
    
    Returns a dictionary of fitted transfer functions for each neuron type 
    and updates network_pars.



    Parameters
    ----------
    tf_sim_pars : dict
        Workflow parameters for the transfer function simulation.
        It should contain:
        - 'fit_transfer_function': bool, whether to fit the transfer function or not.
        - 'adaptation': bool, whether to use adaptation in the transfer function fitting.
        - 'square_terms': bool, whether to include square terms in the transfer function.
        - 'log_term': bool, whether to include log term in the transfer function.
        - 'nu_out_min': float, minimum value of nu_out for fitting.
        - 'nu_out_max': float, maximum value of nu_out for fitting.
        - 'V_eff_fitting': dict, parameters for V_eff fitting.
        - 'TF_fitting': dict, parameters for transfer function fitting.
    network_pars : dict
        Network parameters including neuron parameters and transfer function parameters (expansion point, normalization).
    neuron_results : dict
        Results from the neuron simulations, containing firing rates and other relevant data.
    param_file : Path
        Path to the file where the updated network parameters will be saved.
    
    Returns
    -------
    tf_funcs : dict
        Dictionary containing the fitted transfer functions for each neuron type.
    """

    neuron_names = list(neuron_results.keys())
    neurons = {neuron: network_pars[neuron] for neuron in neuron_names}
    tf_funcs = {}

    if tf_sim_pars["fit_transfer_function"]:
        for neuron_name, neuron_pars in neurons.items():
            nu_e = neuron_results[neuron_name].exc_drive_mean  # [Hz]
            nu_i = neuron_results[neuron_name].inh_drive_mean  # [Hz]
            nu_out = neuron_results[neuron_name].out_rate_mean  # [Hz]
            if tf_sim_pars['adaptation']:
                w = neuron_results[neuron_name].adaptation_mean *1e-3 # [nA]
                tf_funcs[neuron_name] = fit_tf_adaptation(nu_e, nu_i, nu_out, w, neuron_pars, tf_sim_pars,
                                            network_pars["transfer_function"]["expansion_point"],
                                            network_pars["transfer_function"]["expansion_norm"])
            else:
                tf_funcs[neuron_name] = fit_tf(nu_e, nu_i, nu_out, neuron_pars, tf_sim_pars,
                                            network_pars["transfer_function"]["expansion_point"],
                                            network_pars["transfer_function"]["expansion_norm"])

    else:
        # If transfer function is not fitted, we just create the transfer function objects
        # with the parameters from the network_pars
        for neuron, neuron_pars in neurons.items():
            if tf_sim_pars['adaptation']:
                tf_funcs[neuron] = TransferFunctionAdaptation(
                                        network_pars["transfer_function"][neuron],
                                        network_pars["transfer_function"]["expansion_point"],
                                        network_pars["transfer_function"]["expansion_norm"],
                                        network_pars[neuron],
                                        square_terms=tf_sim_pars["square_terms"], 
                                        log_term=tf_sim_pars["log_term"])
            else:
                tf_funcs[neuron] = TransferFunction(
                                        network_pars["transfer_function"][neuron],
                                        network_pars["transfer_function"]["expansion_point"],
                                        network_pars["transfer_function"]["expansion_norm"],
                                        network_pars[neuron],
                                        square_terms=tf_sim_pars["square_terms"], 
                                        log_term=tf_sim_pars["log_term"])

    print("Fit parameters:")
    print("-------------------")
    names = ['P_0', 'P_mu', 'P_sigma', 'P_tau']
    if tf_sim_pars['log_term']:
        names += ['P_log']
    if tf_sim_pars["square_terms"]:
        names += ['P_mu-mu', 'P_sigma-sigma', 'P_tau-tau', 'P_mu-sigma', 'P_mu-tau', 'P_sigma-tau']
    names += ['Error']
    print(f"{'Param':13} | " +" | ".join(f"{fn:10}" for fn in neuron_names))

    err = []
    for neuron_name in neurons:
        nu_out = neuron_results[neuron_name].out_rate_mean.flatten()
        nu_e = neuron_results[neuron_name].exc_drive_mean.flatten()
        nu_i = neuron_results[neuron_name].inh_drive_mean.flatten()
        if tf_sim_pars['adaptation']:
            w = neuron_results[neuron_name].adaptation_mean.flatten() * 1e-3
            err.append(((nu_out - tf_funcs[neuron_name](nu_e, nu_i, w, flattened=True))**2).mean())
        else:
            err.append(((nu_out - tf_funcs[neuron_name](nu_e, nu_i, flattened=True))**2).mean())
    fit_pars = [list(tf_funcs[neuron_name].v_eff.coefs) +[err[j]] for j, neuron_name in enumerate(neurons)]

    for val, *fits in zip(names, *fit_pars):
        print(f"{val:13} | " + " | ".join(f"{fit:10.3f}" for fit in fits))


    return tf_funcs
