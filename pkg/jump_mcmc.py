import numpy as np
import numpy.random as npr

def jump_mcmc(*args,**kwargs):
    """
    cts time; discrete state
    """
    number_of_steps = 1000
    V = [None for _ in range(number_of_steps)]
    L = [None for _ in range(number_of_steps)]
    W = [None for _ in range(number_of_steps)]
    V[0],L[0],W[0] = sample_smjp_trajectory(**kwargs) # get initial sample (no data)
    for i in range(number_of_steps):
        V,L = sample_new_path(V,L,W,X,**kwargs)
        W = sample_random_discretization(V,L,W,X,**kwargs)
    return V,L,W
        
def sample_new_path(**kwargs):
    """
    ~~ forward-backward alg for \lambda\{ p(v_i,l_i|v_{i-1},l_{i-1}), p(x_i|v_i) \} ~~ 
    Hidden: (v_i,l_i)
    Observed: (x_i,w_i)
    """
    pass

def sample_random_discretization(V,L,W,X,**kwargs):
    pass

        


