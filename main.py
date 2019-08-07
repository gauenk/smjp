"""
This code is an implementation of "MCMC for continuous-time
discrete-state systems" by Vinayak Rao and Yee Whye Teh for
exact MCMC on continuous-time discrete-state systems (surprise).
"""
import matplotlib
matplotlib.use('tkagg')
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
from pkg.distributions import *
from pkg.mh_mcmc import mh_mcmc

from experiments import *

if __name__ == "__main__":
    np.seterr(all='print')
    experiment_2()
    
