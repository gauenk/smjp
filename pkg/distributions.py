from easydict import EasyDict as edict
import copy
import numpy as np
import numpy.random as npr
from scipy.special import gammaln as log_gamma_fx # for gamma distribution
from scipy.stats import invgamma,weibull_min
from pkg.utils import *

class Distribution():
    """
    a template class used to define functions associated with probability distributions
    """
    
    def __init__(self,name):
        self.name = name
        self.params = None
        self.params_keywords = None # the keywords associated with the updated parameters for an mcmc sampler; e.g. the state of the transition operator
        self.params_transforms = None # the transformation functions associated with the input sample for use in an mcmc sampler
        # to use when the Distribution is a likelihood function: p(x|\theta)
        self.proir = None
        self.posterior = None
        self.is_hazard_rate = False

    @property
    def p(self):
        return self.prior

    def set_prior(self,prior):
        self.prior = prior

    def log_likelihood(self,x, params = None,  **kwargs):
        raise NotImplemented("Please define in sub-class")

    def ll(self,x, params = None,  **kwargs):
        return self.log_likelihood(x, params = params, **kwargs)

    def nll(self,x, params = None,  **kwargs):
        return -self.log_likelihood(x, params = params, **kwargs)

    def likelihood(self,x, params = None,  **kwargs):
        raise NotImplemented("Please define in sub-class")

    def l(self,x, params = None,  **kwargs):
        return self.likelihood(x, params = params, **kwargs)
    
    def sample(self,n = 1, params = None,  **kwargs):
        raise NotImplemented("Please define in sub-class")

    def s(self,n = 1, params = None, **kwargs):
        return self.sample(n = n, params = params, **kwargs)
    
    def params_handler(self,state,params=None,**kwargs):
        if state == 'load':
            if params is None:
                return self.params
            params_edict = self._construct_params_edict(params,**kwargs)
            return params_edict
        elif state == 'set':
            if params is None:
                print("[params_handler] no parameters to set. Quittin' this prog.")
                exit()
            else:
                params_edict = self._construct_params_edict(params,**kwargs) # make sure the parameters are in dictionary format
                return params_edict
        else:
            print("[class Distribution: params_handler] Undefined state [{}]".format(state))
            exit()

    def set_parameters(self,params,**kwargs):
        params_edict = self._construct_params_edict(params,**kwargs)
        self.set_parameters_edict(params_edict)
         
    def set_parameters_edict(self,params_edict): # only assigment to "self" of subclass; the parameters should be computed correctly by here
        raise NotImplemented("Please define in sub-class")        
        
    def _construct_params_edict(self,params,**kwargs):
        # check if we have raw parameters or they are in a dictionary
        # if params is None:
        #     print("set_parameters error: can't send me a None type.")
        #     exit()
        if params is None:
            params = edict(copy.copy(self.params))
        params_edict = edict(copy.copy(self.params)) # only "copy.copy" needed, not a "deepcopy". we dont edit the paramters; just replace them with new ones.
        if type(params) not in [dict,edict]: # we actually have raw data
            if self.params_keywords is not None:
                for param_name,param_value in zip(self.params_keywords,params):
                    params_edict[param_name] = param_value
        else:
            params_edict = edict(params_edict)
            for param_name,param_value in params.items():
                params_edict[param_name] = param_value
        params_edict = self.apply_param_transformations(params_edict,**kwargs) # prepare the input parameters for replacement in the distribution definition.
        return params_edict

    def apply_param_transformations(self,params_edict,**kwargs):
        """
        for use in an mcmc sampler;
        e.g. 
         -> x* ~ S(x*|x)
         -> computing S(x|x*) may not be just a parameter replacement.
        """
        if self.params_transforms == None:
            return params_edict
        for key,transform_function in self.params_transforms.items():
            params_edict[key] = transform_function(params_edict,**kwargs)
        return params_edict

    def __str__(self):
        return "Distribution type [{}] with params: ".format(self.name) + str(self.params)

class Normal(Distribution):

    def __init__(self,params,params_keywords=None,params_transforms=None):
        super().__init__('Normal')
        self.params = edict(params)
        self.params_keywords = params_keywords
        self.params_transforms = params_transforms
        self.set_parameters(params)

    def set_parameters_edict(self,params):
        for key,value in params.items():
            if 'mu' == key:
                self.mu = value
            elif 'sigma2' == key:
                self.sigma2 = value
            else:
                print("[{}] unknown parameter key [{}]".format(self.name,key))
                exit()

    def log_likelihood(self,x_list,params = None, **kwargs):
        params = self.params_handler('load',params,**kwargs)
        ll = 0
        for x in x_list:
            ll += -(x - params.mu)**2 / ( 2 * params.sigma2 ) - np.log( params.sigma2 ) / 2 - np.log( 2 * np.pi ) / 2
        return ll
        
    def likelihood(self,x,params = None, **kwargs):
        return np.exp(self.log_likelihood(x,params=params,**kwargs))
    
    def sample(self,n = 1,params = None, **kwargs):
        params = self.params_handler('load',params,**kwargs)
        s = npr.normal(params.mu,params.sigma2,n)
        return s

class InverseGamma(Distribution):

    def __init__(self,params,params_keywords=None,params_transforms=None):
        super().__init__('InverseGamma')
        self.params = edict(params)
        self.params_keywords = params_keywords
        self.params_transforms = params_transforms
        self.set_parameters(params)

    def set_parameters_edict(self,params):
        for key,value in params.items():
            if 'shape' == key:
                self.shape = value
            elif 'scale' == key:
                self.scale = value
            else:
                print("[{}] unknown parameter key [{}]".format(self.name,key))
                exit()
    
    def log_likelihood(self,x_list,params = None, **kwargs):
        params = self.params_handler('load',params,**kwargs)
        ll = 0
        for x in x_list:
            ll += invgamma.logpdf(x,params.shape,0.0,params.scale)
        return ll
        
    def likelihood(self,x,params = None, **kwargs):
        return np.exp(self.log_likelihood(x,params=params,**kwargs))
    
    def sample(self,n = 1,params = None):
        params = self.params_handler('load',params,**kwargs)
        s = invgamma.rvs(params.shape,scale=params.scale,size=n)
        return s

class Gamma(Distribution):

    def __init__(self,params,params_keywords=None,params_transforms=None,**kwargs):
        super().__init__('Gamma')
        self.params = edict(params)
        self.params_keywords = params_keywords
        self.params_transforms = params_transforms
        self.set_parameters(params,**kwargs)

    def set_parameters_edict(self,params):
        for key,value in params.items():
            self.params[key] = value
            if 'shape' == key:
                self.shape = value
            elif 'rate' == key:
                self.rate = value
            else:
                print("[{}] unknown parameter key [{}]".format(self.name,key))
                exit()
    
    def log_likelihood(self,x_list,params = None, **kwargs):
        params = self.params_handler('load',params,**kwargs)
        ll = 0
        for x in x_list:
            ll += params.shape * np.log(params.rate) - log_gamma_fx(params.rate) +\
                 (params.shape - 1) * np.ma.log(x).filled(0) - params.rate * x
        return ll
        
    def likelihood(self,x,params = None,**kwargs):
        """
        l = self.rate**self.shape / gamma_fx(self.rate) * \
             x**(self.shape - 1) * np.exp(-self.rate * x)
        """
        return np.exp(self.log_likelihood(x,params=params,**kwargs))
    
    def sample(self,n = 1,params = None,**kwargs):
        params = self.params_handler('load',params,**kwargs)
        s = npr.gamma(params.shape,1/params.rate,n) # numpy.random wants "scale" not "rate"
        return s

class ExpPiecewiseLinearDistribution(Distribution):

    def __init__(self,init_samples):
        pass
    
    def sample(self,n=1,params=None):
        params = self.params_handler('load',params)
        
    def update_linear_regions(new_sample,gradient_log_p):
        self.samples += [new_sample]
        # compute the new linear regions
        self.samples_grads += [gradient_log_p(new_sample)]
        

class WeibullDistribution(Distribution):

    def __init__(self,params,params_keywords=None,params_transforms=None,is_hazard_rate=False,**kwargs):
        super().__init__('Weibull')
        self.params = edict(params)
        self.params_keywords = params_keywords
        self.params_transforms = params_transforms
        self.is_hazard_rate = is_hazard_rate
        self.set_parameters(params,**kwargs)
        
    def set_parameters_edict(self,params):
        for key,value in params.items():
            self.params[key] = value
            if 'shape' == key:
                self.shape = value
            elif 'scale' == key:
                self.scale = value
            else:
                print("[{}] unknown parameter key [{}]".format(self.name,key))
                exit()
    
    def log_likelihood(self,x_list,params = None, **kwargs):
        params = self.params_handler('load',params,**kwargs)
        ll = 0
        if not isiterable(x_list):
            x = x_list
            """
            NOTE:
            we have just one event in one dimension for now 
            so the value happens to correspond to the index
            """
            if (x < 0): return 0 # only non-negative "x"
            if self.is_hazard_rate:
                ll += np.log(params.shape) - np.log(params.scale) + \
                      (params.shape - 1) * ( np.ma.log([x]).filled(0) - \
                                             np.log(params.scale) )
            else:
                ll += np.log(params.shape) - np.log(params.scale) + \
                      (params.shape - 1) * ( np.ma.log([x]).filled(0) - \
                                             np.log(params.scale) ) +\
                      -(x/params.scale)**params.shape
            ll = ll[0]
        else:
            for x in x_list:
                if x < 0: continue # only non-negative "x"
                if self.is_hazard_rate:
                    ll += np.log(params.shape) - np.log(params.scale) + \
                          (params.shape - 1) * ( np.ma.log([x]).filled(0) - \
                                                 np.log(params.scale) )
                else:
                    ll += np.log(params.shape) - np.log(params.scale) + \
                          (params.shape - 1) * ( np.ma.log(x).filled(0) - \
                                                 np.log(params.scale) ) +\
                          -(x/params.scale)**params.shape
        return ll
        
    def likelihood(self,x,params = None,**kwargs):
        return np.exp(self.log_likelihood(x,params=params,**kwargs))
    
    def sample(self,n = 1,params = None,hold_time=None,**kwargs):
        params = self.params_handler('load',params,**kwargs)
        if hold_time is None: # ordinary sampling from weibull
            s = npr.weibull(params.shape,size=n)
        else: # conditional sampling from weibull
            shape = params['shape']
            scale = params['scale']
            cdf_at_m = 1 - np.exp( - (hold_time / scale ) ** shape )
            s = []
            for i in range(n):
                u = npr.uniform(cdf_at_m,1)[0]
                sample = scale * np.power( -np.log(1 - u), 1./shape )
                s.append(sample)
        return s


class MultinomialDistribution(Distribution):

    def __init__(self,params,params_keywords=None,params_transforms=None,**kwargs):
        super().__init__('Multinomial')
        self.prob_vector,self.translation = None,None
        self.params = edict(params)
        self.params_keywords = params_keywords
        self.params_transforms = params_transforms
        self.set_parameters(params,**kwargs)
        
    def set_parameters_edict(self,params):
        for key,value in params.items():
            self.params[key] = value
            if 'prob_vector' == key:
                self.prob_vector = value
            elif 'translation' == key:
                self.translation = value
            else:
                print("[{}] unknown parameter key [{}]".format(self.name,key))
                exit()
    
    def log_likelihood(self,x_list,params = None, **kwargs):
        params = self.params_handler('load',params,**kwargs)
        p = params.prob_vector
        n = len(p)
        if isiterable(x_list) is False:
            x = x_list
            """
            NOTE:
            we have just one event in one dimension for now 
            so the value happens to correspond to the index
            """
            # if self.translation:
            #     #x_index = self.translation.index(x)
            #     ll = np.log(p[x])
            # else:
            ll = np.ma.log([p[x]]).filled(-np.infty)[0]
        else:
            log_p = np.ma.log(p).filled(-np.infty)
            log_x = np.ma.log(x_list).filled(-1)
            ll = 0
            for x in x_list:
                if x < 0: continue # only non-negative "x"
                raise NotImplemented("Mulitnomial Likeilhood is not implemented")
                ll += log_p[x]
        return ll
        
    def likelihood(self,x,params = None,**kwargs):
        return np.exp(self.log_likelihood(x,params=params,**kwargs))
    
    def sample(self,n = 1,params = None,**kwargs):
        params = self.params_handler('load',params,**kwargs)
        if 'size' not in kwargs.keys():
            size = n
        if n == 1:
            sample = np.where(npr.multinomial(1,params.prob_vector) == 1)[0][0]
        elif size == n:
            sample = np.where(npr.multinomial(1,params.prob_vector,n) == 1)[1]
        else:
            sample = npr.multinomial(n,params.prob_vector,size)            
        if self.translation is not None:
            if isiterable(sample):
                tmp = []
                for s in sample:
                    tmp += [self.translation[s]]
                sample = tmp
            else:
                sample = self.translation[sample]
        return sample


