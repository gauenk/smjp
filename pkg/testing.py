import numpy as np
from numpy import transpose as npt

import pickle,uuid,re
import numpy as np
import numpy.random as npr
import seaborn as sns
from pkg.distributions import *
from pkg.jump_mcmc import *
from pkg.mh_mcmc import mh_mcmc
from pkg.hidden_markov_model import *
from pkg.smjp_utils import *
from pkg.mcmc_utils import *
import matplotlib.pyplot as plt
from matplotlib import cm


def test_1():
    omega = 2
    states = [1,2]
    shape = np.array([[3./2,1./2],[2.,3./4]])
    scale = np.array([[1,1],[1,1]])
    scale_tilde = np.array([[1./np.power(omega,2./3),1./np.power(omega,2.)],
                            [1./np.power(omega,1./2),1./np.power(omega,4./3)]
    ])
    grid = np.array([0.,1./3,2./3,1.])
    aug_state_space = np.array([[1,0],[1,1./3],[1,2./3],
                                [2,0],[2,1./3],[2,2./3],
    ])
    pi_0 = np.array([1./2,0,0,0,1./2,0,0,0])
    asize = len(aug_state_space)
    
    #
    # load in the computed alpha terms
    #

    alpha_0 = np.zeros(asize)
    alpha_0[0] = ( (3./2)*(1./3)**(1./2) + (1./2)*(1./3)**(-1./2) ) \
                 * np.exp(-2*(1./3)**(3./2) - 2*(1./3)**(1./2))
    alpha_0[3] = (2./3 + 3./4*(1./3)**(-1./4)) * np.exp(-2*(1./3)**2 - 2*(1./3)**(3./4))
    
    alpha_1 = np.zeros(asize)
    alpha_1_helper_term_1 = 2*(3./2)*(1./3)**(1./2) + 2*(1./2)*(1./3)**(-1./2)
    alpha_1_helper_term_2 = 2*(2.)*(1./3)**(1.) + 2*(3./4)*(1./3)**(-1./4)

    alpha_1[0] = ( (3./2)*(2./3)**(1./2) + (1./2)*(2./3)**(-1./2) )\
                 * np.exp( -2*(2./3)**(3./2) + 2*(1./3)**(3./2) \
                           -2*(2./3)**(1./2) + 2*(1./3)**(1./2) ) \
                 * alpha_0[0]
    alpha_1[1] = alpha_1_helper_term_1 \
                 * np.exp( -2*(1./3)**(3./2) - 2*(1./3)**(1./2) ) \
                 * ( (3./2)*(1./3)**(1./2) / alpha_1_helper_term_1 * alpha_0[0] + \
                     (2.)*(1./3)**(1.) / alpha_1_helper_term_2 * alpha_0[3] )
    alpha_1[3] = ( 2 * (2./3)**(1) + (3./4) * (2./3)**(-1./4) ) \
                 * np.exp( -2*(2./3)**(2) + 2*(1./3)**(2)
                           -2*(2./3)**(3./4) + 2*(1./3)**(3./4) ) \
                 * alpha_0[3]
    alpha_1[4] = alpha_1_helper_term_2 \
                 * np.exp( -2*(1./3)**(2.) - 2*(1./3)**(3./4) ) \
                 * ( (1./2)*(1./3)**(-1./2) / alpha_1_helper_term_1 * alpha_0[0] +
                     (3./4)*(1./3)**(-1./4) / alpha_1_helper_term_2 * alpha_0[3] )

    alpha_2 = np.zeros(asize)

    a_11_23 = (3./2)*(2./3)**(1./2)
    a_11_13 = (3./2)*(1./3)**(1./2)
    a_21_23 = (2.)*(2./3)**(1.)
    a_21_13 = (2.)*(1./3)**(1.)

    a_12_23 = (1./2)*(2./3)**(-1./2)
    a_12_13 = (1./2)*(1./3)**(-1./2)
    a_22_23 = (3./4)*(2./3)**(-1./4)
    a_22_13 = (3./4)*(1./3)**(-1./4)

    b_1_23 = 2 * ( (3./2)*(2./3)**(1./2) + (1./2)*(2./3)**(-1./2) )
    b_1_13 = 2 * ( (3./2)*(1./3)**(1./2) + (1./2)*(1./3)**(-1./2) )
    b_2_23 = 2 * ( (2.)*(2./3)**(1.) + (3./4)*(2./3)**(-1./4) )
    b_2_13 = 2 * ( (2.)*(1./3)**(1.) + (3./4)*(1./3)**(-1./4) )

    alpha_2[0] = (1./2) * np.exp( -2 * (1.)**(3./2) + 2 * (2./3)**(3./2)
                                  -2 * (1.)**(1./2) + 2 * (2./3)**(1./2)) \
                                  * alpha_1[0]
    alpha_2[1] = (1./2) * np.exp( -2 * (2./3)**(3./2) + 2 * (1./3)**(3./2)
                                  -2 * (2./3)**(1./2) + 2 * (1./3)**(1./2)) \
                                  * alpha_1[1]
    alpha_2[2] = np.exp( -2 * (1./3)**(3./2) - 2 * (1./3)**(1./2) ) * (
        a_11_23 / b_1_23 * alpha_1[0] +
        a_11_13 / b_1_13 * alpha_1[1] +
        a_21_23 / b_2_23 * alpha_1[3] +
        a_21_13 / b_2_13 * alpha_1[4]
    )
    alpha_2[3] = (1./2) * np.exp( -2 * (1.)**(2.) + 2 * (2./3)**(2.)
                                  -2 * (1.)**(3./4) + 2 * (2./3)**(3./4)) \
                                  * alpha_1[3]
    alpha_2[4] = (1./2) * np.exp( -2 * (2./3)**(2.) + 2 * (1./3)**(2.)
                                  -2 * (2./3)**(3./4) + 2 * (1./3)**(3./4)) \
                                  * alpha_1[4]
    alpha_2[5] =  np.exp( -2 * (1./3)**(2.) - 2 * (1./3)**(3./4) ) * (
        a_12_23 / b_1_23 * alpha_1[0] +
        a_12_13 / b_1_13 * alpha_1[1] +
        a_22_23 / b_2_23 * alpha_1[3] +
        a_22_13 / b_2_13 * alpha_1[4] 
    )
    alphas = np.array([alpha_0,alpha_1,alpha_2])
    alphas = npt(npt(alphas) / np.sum(alphas,axis=1))
    print(alphas)
    run_test(states,shape,scale,scale_tilde,grid,aug_state_space,pi_0,omega)
    print(alphas)

def run_test(states,shape,scale,scale_tilde,grid,aug_state_space,pi_0,omega):
    # ---------------
    #
    # sMJP parameters
    #
    # ---------------

    state_space = states
    obs_space = state_space
    s_size = len(state_space)
    time_length = 1. # for time t \in [0,time_length]

    # experiment info
    number_of_observations = 3
    likelihood_power = 1.

    # ------------------------------------------
    #
    # create hazard functions defining the sMJP
    #
    # ------------------------------------------

    shape_mat = shape
    scale_mat = scale
    scale_mat_tilde = scale_tilde
    
    # hazard A needs a sampler for the prior
    weibull_hazard_create_A = partial(weibull_hazard_create_unset,shape_mat,scale_mat,state_space)
    weibull_hazard_sampler_A = partial(smjp_hazard_sampler_unset,state_space,weibull_hazard_create_A)
    weibull_hazard_create_B = partial(weibull_hazard_create_unset,shape_mat,scale_mat_tilde,state_space)
    weibull_hazard_sampler_B = partial(smjp_hazard_sampler_unset,state_space,weibull_hazard_create_B)

    # hazard B
    weibull_hazard_create_B = partial(weibull_hazard_create_unset,shape_mat,scale_mat_tilde,state_space)

    hazard_A = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create_A,sampler=weibull_hazard_sampler_A)
    hazard_B = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create_B,sampler=weibull_hazard_sampler_B)

    # ------------------------------------------------------------
    #
    # instantiate the Poisson process to sample the thinned events
    #
    # ------------------------------------------------------------

    pp_mean_params_A = {'shape':shape_mat,'scale':scale_mat}
    poisson_process_A = PoissonProcess(state_space,None,hazard_A,pp_mean_params_A)
    pp_mean_params_B = {'shape':shape_mat,'scale':scale_mat_tilde}
    poisson_process_B = PoissonProcess(state_space,None,hazard_B,pp_mean_params_B)
    
    # -------------------------------------------------------------------------
    #
    # generate observations from data
    #
    # -------------------------------------------------------------------------

    # data
    smjp_emission_create = partial(smjp_emission_multinomial_create_unset,state_space)
    emission_sampler = sMJPWrapper(smjp_emission_sampler,state_space,smjp_emission_create)
    emission_likelihood = sMJPWrapper(smjp_emission_likelihood,state_space,smjp_emission_create)
    data_samples,data_times = create_toy_data(state_space,time_length,number_of_observations,emission_sampler)
    data = sMJPDataWrapper(data=data_samples,time=data_times)
    data_sampler_info = [state_space,time_length,emission_sampler]
    
    # likelihood of obs
    emission_info = [emission_likelihood,poisson_process_B,likelihood_power,state_space]
    smjp_emission = partial(smjp_emission_unset,*emission_info)


    # ----------------------------------------------------------------------
    #
    # gibbs sampling for-loop
    #
    # ----------------------------------------------------------------------

    aggregate = {'W':[],'V':[],'T':[],'prob':[]}
    aggregate_prior = {'V':[],'T':[]}
    number_of_samples = 1000
    smjp_sampler_input = [state_space,hazard_A,hazard_B,smjp_emission,time_length]
    pi_0 = MultinomialDistribution({'prob_vector': np.ones(s_size)/s_size,\
                                    'translation':state_space})
    V,T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)
    _V,_T = V,T
    print("--------------")
    print(V)
    print(T)
    print("|T| = {}".format(len(T)))
    print("--------------")

    W = np.array([0,1./3,2./3,1.])
    # print(W)

    counts = np.zeros((len(W)-1)**2 * s_size).reshape(len(W)-1,s_size*(len(W)-1))
    print_iter = 100
    for i in range(number_of_samples):
        if (i % print_iter) == 0:
            print("iteration: {}/{}".format(i,number_of_samples))

        # ~~ the main gibbs sampler for mcmc of posterior sample paths ~~
        p_u_str = 'none'
        V,T,prob = sample_smjp_trajectory_posterior(W,data,*smjp_sampler_input,p_u_str)
        # print("---------")
        # print(V)
        # print(T)
        # print("|T| = {}".format(len(T)))
        # print("---------")

        w_start = 0
        T_idx = -1
        for v,t in zip(V,T):
            if np.isclose(t,1.0):
                break
            T_idx += 1 # start @ 0
            t_idx = np.where(np.isclose(W,t))[0]
            v_idx = state_space.index(v)
            aug_idx = v_idx*(len(W)-1) + t_idx
            for w_idx,w in enumerate(W[:-1]):
                if w_idx < w_start:
                    continue
                if (w - T[T_idx+1]) > 0 or np.isclose(w,T[T_idx+1]):
                    continue
                # print('a',T[T_idx+1],t,w,w_start,aug_idx,aug_state_space[aug_idx])
                counts[w_idx,aug_idx] += 1
                w_start += 1
            # print("--")

        # aggregate['V'].append(V)
        # aggregate['T'].append(T)
        # aggregate['prob'].append(prob)
        # print('posterior',np.c_[V,T])

        
    # V_post,T_post = aggregate['V'],aggregate['T']
    
    """
    V_post [num_of_samples,??]; T_post [num_of_samples,??]
    """
    #
    # compute monte carlo estimate of each state
    #
    counts = npt(npt(counts) / np.sum(counts,axis=1))
    print("monte carlo estimate")
    print(counts)

if __name__ == "__main__":
    print("HI")
    test_1()
