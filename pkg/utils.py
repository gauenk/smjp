def logsumexp(nd_array):
    return np.ma.log([np.sum(np.exp(nd_array))])[0]


def np_log(number):
    return np.ma.log([number]).filled(-np.infty)

def write_ndarray_list_to_debug_file(params,uuid_str=None,debug_fn='debug_params'):
    w_str = ''
    for key,ndarray in params.items():
        w_str += '-------\n'
        w_str += key + '\n'
        w_str += str(ndarray)
        w_str += '\n\n'
    if uuid_str is not None:
        debug_fn += "_"+str(uuid_str)
    with open(debug_fn+".txt",'w') as f:
        f.write(w_str)
    return
    
def isiterable(p_object):
    try:
        it = iter(p_object)
    except:
        return False
    return True

import numpy as np
from numpy import transpose as npt

def compute_autocorrelation(samples):
    from scipy.signal import fftconvolve
    sample_mean = np.mean(samples)
    x = samples - sample_mean
    n = len(x)
    result = fftconvolve(samples,samples[::-1],mode='full')
    acorr = result[len(result) // 2:] # symmetric so remove second half
    acorr /= np.arange(n,0,-1) # we normalize the auto-corr via # of samples: *(#samples)^{-1}
    acorr /= acorr[0] # divide by variance per the definition of auto-corr (autocov -> autocorr)
    print(acorr[:15])
    return acorr

def compute_ess(samples,autocorr_threshold = 0.01):
    """
    Compute the effective sample size for a list
    """
    sample_mean = np.mean(samples)
    sample_mean_2 = sample_mean**2
    sample_std = np.std(samples,ddof=1) # use 1/(N-1) for unbiased estimate
    sample_var = sample_std**2
    autocorr = compute_autocorrelation(samples)
    
    
def use_filepicker(start,ft):
    pass
