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
    
    
