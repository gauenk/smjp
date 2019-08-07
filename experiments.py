"""
-- Skeleton --
W = random discretization of time
T = un-thinned events (keeping these)
U = thinned events (removing these)
|W| = |T| + |U|  and  W = T \cup U


-- State Assignments --
V = the states for each element of W
L = the time since the last jump transition ( w_i - max_{t \in T,t \leq w_i} t )

"""

import numpy as np
import numpy.random as npr
import seaborn as sns
from pkg.distributions import *
from pkg.jump_mcmc import *
from pkg.mh_mcmc import mh_mcmc
from pkg.hidden_markov_model import *
from pkg.smjp_utils import *
import matplotlib.pyplot as plt
from matplotlib import cm

#tmp
from scipy.stats import weibull_min

def experiment_0():
    """
    check if we have some basics correct
    """
    params_list = [[1,.5],[1,1],[1,1.5],[1,5]]
    quantile_list = [.1,.25,.5,.75,.9,.95]
    x_grid = np.arange(0.01,5,.05)
    color_list = ['b','k','g','r']
    for index,params in enumerate(params_list):
        scale,shape = params[0],params[1]
        label_str = 'lambda = {}, k = {}'.format(scale,shape)
        p = WeibullDistribution({'scale':scale,'shape':shape})
        data = p.sample(2000)
        print("~ {} ~".format(label_str))
        for quantile in quantile_list:
            quant = np.quantile(data,quantile)
            print("quantile @ {}: {}".format(quantile,quant))
        print("---")
        # sns.scatterplot(x_grid,weibull_min.pdf(x_grid,shape,scale=scale),label=label_str)
        if np.isclose(shape,0.5):
            t = sns.kdeplot(data, bw=0.005, shade = True, label=label_str, gridsize=1000, cut=3)
            # plt.vlines(data,ymin=0,ymax=1,color=color_list[index],alpha=0.4)
        else:
            t = sns.kdeplot(data, shade = True, label=label_str, gridsize=1000, cut=3)  
    plt.xlim(0,5)
    plt.show()

def experiment_1():

    # plotting parameters
    sns.set(font_scale=1.25)
    style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
                  'font.name':'courier new', 'legend.frameon': True}
    sns.set_style('white', style_kwds)

    # emission: ( |states| x |size of observed space| )
    emission = np.array(
        [
            [0.25,0.25,.25,.25],
            [0.0,0.35,.40,.25],
            [0.5,0.0,.5,.0],
        ])
    # emission = np.eye(3) # a special case of hmm; perfect obs, identified by state (too easy?)
    # transition: ( |states| x |states| )
    transition = np.array(
        [
            [0.10,0.40,.40],
            [0.40,0.10,.40],
            [0.40,0.40,.10],
        ])
    # state_alphabet: |states| (purpose is vague currently... maybe just use "number_of_states")
    state_alphabet = ['0','1','2',]
    observed_alphabet = ['0','1','2','3',]
    # data: ( |# of entries| x |size of element from observed space| )
    data = np.array([2,3,1,2,2,3,1,1,1,2,2,2,2,2,2,1,2,3,1,1,])

    """
    NOTE [size of observed value space] :
    1.
    - emission and data use "|size of observed values|" as part of their construction.
    - this only works when the observation space is DISCRETE and FINITE.
    - sometimes the space of observations are referred to as the STATES;
      I think this is a special case of HMM, and the case used in the Rao-Teh alg.
    2.
    - what about handling dimensions > 1?... idk. (**)
    """
    
    pi_0 = MultinomialDistribution({'prob_vector': np.ones(3)/3})
    time_grid = np.arange(len(data))
    hmm_init = {'emission': HMMWrapper(emission,True), 
                'transition': HMMWrapper(transition,False),
                'data': data,
                'state_alphabet': state_alphabet,
                'pi_0': pi_0,
                'time_grid': time_grid,
                'sample_dimension': 1,
                }
    hmm = HiddenMarkovModel([],**hmm_init)
    alphas,prob = hmm.likelihood()

    plot_colors = cm.rainbow(np.linspace(0, 1, len(transition)))
    sns.lineplot(data=np.ma.log(alphas).filled(+1),
                 legend='full',
                 palette=plot_colors).set_title("Log Prob. of States and Previous Data")

    samples = hmm.backward_sampling()
    plot_hmm_path(samples,pi_0)
    plt.show()


def plot_hmm_path(samples,pi_0):
    init_sample = np.where(pi_0.sample(1) == 1)[0][0]
    all_samples = np.r_[init_sample,samples[0]]
    time = np.arange(len(all_samples))
    fig,ax = plt.subplots(1,1)
    # first we plot the trace
    ax.step(time,all_samples,where='post',label="sample_path")
    ax.plot(time,all_samples,'C0o',alpha=0.5)
    ax.set_title("HMM Sample Path")



def sample_alpha_parameters(number_of_states):
    alphas = []
    for i in range(number_of_states):
        alphas_row = []
        for j in range(number_of_states):
            alphas_row += [npr.uniform(0.6,1.2)]
        alphas += [alphas_row]
    return alphas


def experiment_2( likelihood_power = 1. ):
    """ 
    Run "inner-loop" of experiment 1 from Rao-Teh alg. (loop is to generate plots)
    """

    # ---------------
    #
    # sMJP parameters
    #
    # ---------------

    state_space = [1,2,3]
    obs_space = state_space
    s_size = len(state_space)
    time_length = 2 # for time t \in [0,time_length]
    omega = 2

    # experiment info
    likelihood_power = 0.
    number_of_observations = 3

    # ------------------------------------------
    #
    # create hazard functions defining the sMJP
    #
    # ------------------------------------------

    shape_mat = npr.uniform(0.6,3.0,s_size**2).reshape((s_size,s_size))
    scale_mat = np.ones((s_size,s_size)) 
    scale_mat_tilde = create_upperbound_scale(shape_mat,scale_mat,omega)

    # hazard A needs a sampler for the prior
    weibull_hazard_create_A = partial(weibull_hazard_create_unset,shape_mat,scale_mat,state_space)
    weibull_hazard_sampler_A = partial(smjp_hazard_sampler_unset,state_space,weibull_hazard_create_A)

    # hazard B
    weibull_hazard_create_B = partial(weibull_hazard_create_unset,shape_mat,scale_mat_tilde,state_space)

    hazard_A = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create_A,sampler=weibull_hazard_sampler_A)
    hazard_B = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create_B)


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
    smjp_sampler_input = [state_space,hazard_A,hazard_B,smjp_emission,data]
    pi_0 = MultinomialDistribution({'prob_vector': np.ones(s_size)/s_size,\
                                    'translation':state_space})
    T,V = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)
    for i in range(number_of_samples):

        # ~~ the main gibbs sampler for mcmc of posterior sample paths ~~
        W = sample_smjp_event_times(poisson_process_A,V,T)
        # just for testing
        # W = np.arange(10)
        # while len(W) > 9:
        #     W = sample_smjp_event_times(poisson_process_A,V,T)
        
        V,T,prob = sample_smjp_trajectory_posterior(W,*smjp_sampler_input)
        aggregate['W'].append(W)
        aggregate['V'].append(V)
        aggregate['T'].append(T)
        aggregate['prob'].append(prob)
        
        # take some samples from the prior for de-bugging the mcmc sampler
        _V,_T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)
        aggregate_prior['V'].append(_V)
        aggregate_prior['T'].append(_T)
        print("i = {}".format(i))
        

    # --------------------------------------------------
    #
    # compute some metrics for evaluation of the sampler
    # 
    # -------------------------------------------------

    # 1.) time per state: [# of samples, # of states] each entry is \sum of time in each state, 
    times_per_state = compute_time_per_state(aggregate)
    times_per_state_prior = compute_time_per_state(aggregate_prior)

    # 2.) number of transitions: [# of samples, 1] each entry is the size of T
    num_of_transitions = compute_num_of_transitions(aggregate)
    num_of_transitions_prior = compute_num_of_transitions(aggregate_prior)

    # compact the results for returning
    metrics_posterior = {'state_times':[times_per_state],
                         'transitions':[num_of_transitions]
                         }
    metrics_prior = {'state_times':[times_per_state],
                     'transitions':[num_of_transitions]
    }

    # computing effective sample size

    return metrics_posterior,metrics_prior,aggregate,aggregate_prior



def experiment_3():
    """
    Run experiment 1 from Rao-Teh
    """
    import matplotlib.pyplot as plt
    aggregate_metrics = []
    inv_temp_grid = np.arange(0,1,.1)
    for inv_temp in inv_temp_grid:
        m_posterior,m_prior,agg_posterior,agg_prior = experiment_2(inv_temp)

        
