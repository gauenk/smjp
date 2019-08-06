def isiterable(p_object):
    try:
        it = iter(p_object)
    except:
        return False
    return True

import numpy as np

def compute_autocorrelation(samples):
    from scipy.signal import fftconvolve
    sample_mean = np.mean(samples)
    x = samples - sample_mean
    n = len(x)
    result = fftconvolve(samples,samples[::-1],mode='full')
    acorr = result[len(result) // 2:] # symmetric so remove second half
    acorr /= np.arange(n,0,-1) # we need to normalize the auto-corr via # of samples
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
    
    
    trace_value = x.T
    nchain, n_samples = trace_value.shape

    acov = np.asarray([autocov(trace_value[chain]) for chain in range(nchain)])

    chain_mean = trace_value.mean(axis=1)
    chain_var = acov[:, 0] * n_samples / (n_samples - 1.) # correctiion from 1/n to 1/(n-1)
    acov_t = acov[:, 1] * n_samples / (n_samples - 1.) # correction again
    mean_var = np.mean(chain_var) # across all chains
    var_plus = mean_var * (n_samples - 1.) / n_samples # correcting _back_ from 1/(n-1) to 1/n
    var_plus += np.var(chain_mean, ddof=1)
    # var_plus = \hat{var}(\hat{x}) + \frac{1}{n} \sum_i^N \hat{cov}(x_0,x_0)
    # = \hat{var}(\hat{x})

    rho_hat_t = np.zeros(n_samples)
    rho_hat_even = 1.
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1. - (mean_var - np.mean(acov_t)) / var_plus
    rho_hat_t[1] = rho_hat_odd
    # Geyer's initial positive sequence
    max_t = 1
    t = 1
    while t < (n_samples - 2) and (rho_hat_even + rho_hat_odd) >= 0.:
        rho_hat_even = 1. - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1. - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
            max_t = t + 2
            t += 2

    # Geyer's initial monotone sequence
    t = 3
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
            t += 2
            ess = nchain * n_samples
            ess = ess / (-1. + 2. * np.sum(rho_hat_t))
    return ess
