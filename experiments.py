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
from pkg.pmcmc import pmcmc
from pkg.raoteh import raoteh
import matplotlib.pyplot as plt
from matplotlib import cm

def experiment_0():
    """
    check if we have some basics correct
    """
    from scipy.stats import weibull_min
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


def experiment_2( likelihood_power = 1. , inference = ['trajectory']):
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
    time_final = 2.0
    omega = 2
    uuid_str = uuid.uuid4()

    # experiment info
    # obs_times = [1./3,2./3,4./3,5./3]
    obs_times = np.arange(0.1,time_final,0.1)
    num_of_obs = len(obs_times)

    # ------------------------------------------
    #
    # create hazard functions defining the sMJP
    #
    # ------------------------------------------

    shape_mat = np.array([[1.606, 1.933, 0.865],
                          [1.938, 0.869, 1.751],
                          [1.69,  0.64,  0.696]])
    # shape_mat = npr.uniform(0.6,3.0,s_size**2).reshape((s_size,s_size))
    scale_mat = np.ones((s_size,s_size)) 
    scale_mat_tilde = create_upperbound_scale(shape_mat,scale_mat,omega)
    scale_mat_hat = create_upperbound_scale(shape_mat,scale_mat,omega-1)
    debug_params = {'shape_mat':shape_mat,
                    'scale_mat':scale_mat,
                    'scale_mat_tilde':scale_mat_tilde,
                    'scale_mat_hat':scale_mat_hat,
    }

    write_uuid_str = None
    if False: #True:
        write_uuid_str = uuid_str
    write_ndarray_list_to_debug_file(debug_params,write_uuid_str)

    hazard_A = smjpHazardFunction(state_space,shape_mat,scale_mat)
    hazard_B = smjpHazardFunction(state_space,shape_mat,scale_mat_tilde,omega=omega)
    hazard_A_hat = smjpHazardFunction(state_space,shape_mat,scale_mat_hat)

    # ------------------------------------------------------------
    #
    # instantiate the Poisson process to sample the thinned events
    #
    # ------------------------------------------------------------

    pp_mean_params_A = {'shape':shape_mat,'scale':scale_mat}
    poisson_process_A = PoissonProcess(state_space,None,hazard_A,pp_mean_params_A)
    pp_mean_params_A_hat = {'shape':shape_mat,'scale':scale_mat_hat}
    poisson_process_A_hat = PoissonProcess(state_space,None,hazard_A_hat,pp_mean_params_A_hat)
    pp_mean_params_B = {'shape':shape_mat,'scale':scale_mat_tilde}
    poisson_process_B = PoissonProcess(state_space,None,hazard_B,pp_mean_params_B)
    
    # -------------------------------------------------------------------------
    #
    # generate observations from data
    #
    # -------------------------------------------------------------------------

    # data
    emission = smjpEmission(state_space,poisson_process_B,time_final,likelihood_power)
    # data_samples = create_toy_data(state_space,time_final,num_of_obs,emission)
    # data_samples = [1,3,2,1] # deterministic for testing
    data_samples = np.array([1,1,1,1,1,1,2,2,2,1,1,1,1,3,3,2,1,1,3])
    data = sMJPDataWrapper(data=data_samples,time=obs_times)

    # initial state prior
    pi_0 = MultinomialDistribution({'prob_vector': np.ones(s_size)/s_size,\
                                    'translation':state_space})

    print("-- data --")
    print(data)
    # ----------------------------------------------------------------------
    #
    # sampler
    #
    # ----------------------------------------------------------------------
    number_of_samples = 6000
    save_iter = 300

    # --------------------
    # --- rao-teh (rt) ---
    # --------------------
    filename = "results_raoteh_3059de65-2596-40a7-9395-e1e059351c16_final.pkl"
    load_file = True
    raoteh_input = [inference,
                    number_of_samples,
                    save_iter,
                    state_space,
                    time_final,
                    emission,
                    data,
                    pi_0,
                    hazard_A,
                    hazard_B,
                    poisson_process_A_hat,
                    poisson_process_B,
                    uuid_str,
                    omega,
                    obs_times,
                    filename,
                    load_file]
    rt_aggregate,rt_uuid_str,rt_omega = raoteh(*raoteh_input)

    # -------------------
    # --- pmcmc (pm) ----
    # -------------------
    number_of_particles = 10
    # filename = "results_pmcmc_66120de8-d9f2-4e1c-8fc0-2317456c0768_final.pkl"
    filename = "results_pmcmc_0f9ee403-c4aa-4637-8f66-5387ba4bda34_4800.pkl"
    load_file = True
    pmcmc_input = [inference,
                   number_of_particles,
                   number_of_samples,
                   save_iter,
                   state_space,
                   hazard_A,
                   emission,
                   time_final,
                   data,
                   pi_0,
                   uuid_str,
                   omega,
                   filename,
                   load_file]
    pm_aggregate,pm_uuid_str = pmcmc(*pmcmc_input)

    # --------------------------------------------------
    #
    # compute some metrics for evaluation of the sampler
    # 
    # -------------------------------------------------

    # ~~ change experiment uuid if both uuid's from raoteh and pmcmcm are equal ~~ 
    print(rt_uuid_str)
    print(pm_uuid_str)
    if rt_uuid_str == pm_uuid_str:
        uuid_str = rt_uuid_str

    # generate_sample_report_twochainz(rt_aggregate,rt_aggregate['prior'],
    #                                  'posterior','prior',
    #                                  state_space,uuid_str)
    # generate_sample_report_twochainz(rt_aggregate_prior,pm_aggregate_prior,'raoteh-pr','pm-pr')
    generate_sample_report_twochainz(rt_aggregate,pm_aggregate,'raoteh','pm',
                                     state_space,uuid_str)


def experiment_3():
    """
    Run experiment 1 from Rao-Teh
    """
    import matplotlib.pyplot as plt
    aggregate_metrics = []
    inv_temp_grid = np.arange(0,1,.1)
    for inv_temp in inv_temp_grid:
        m_posterior,m_prior,agg_posterior,agg_prior = experiment_2(inv_temp)

        
def get_sample_frequency_multiplier(cond_x_st,states):
    """
    we resample the posterior samples given the frequency
    associated with the data sample.

    The max number of repeats is associated with the precision of the 
    likelihood... Having something super unlikely occur _once_ or have the 
    possibility of occuring can make number of repeats of a sample increase
    Dramatically; we have to repeat the common events several times.

    f_mult = "frequency based multiplier"
    """
    probs = [cond_x_st(s) for s in states]
    probs -= np.min(probs)
    probs /= np.min(probs)
    print(probs)
    return probs
    
def verify_hazard_function(h_create,state_space,aug_state_space):
    import seaborn as sns
    import matplotlib.pyplot
    n_samples = 1000
    nss = len(state_space)
    samples = np.zeros(nss**2*n_samples).reshape(nss,nss,n_samples)
    for idx_c,state_c in enumerate(state_space):
        for idx_n,state_n in enumerate(state_space):
            for idx_s in range(n_samples):
                samples[idx_c,idx_n,idx_s] = h_create(state_c,state_n).sample()

    for idx_c,state_c in enumerate(state_space):
        for idx_n,state_n in enumerate(state_space):
            print('{}->{}'.format(state_c,state_n),h_create(state_c,state_n).params)
            sns.distplot(samples[idx_c,idx_n],hist=True,rug=False,\
                         label='{}->{}'.format(state_c,state_n)).set(xlim=(0))

    plt.legend()
    plt.show()

def verify_conditional_weibull():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as sss
    p = Weibull({'shape':1.0,'scale':1.0})
    regular_samples = p.sample(1000)
    conditional_samples_A = np.array(p.sample(1000,hold_time=.5))
    conditional_samples_B = np.array(p.sample(1000,hold_time=10))
    sns.distplot(regular_samples,hist=True,rug=False,\
                 label='P( \tau )').set(xlim=(0))
    sns.distplot(conditional_samples_A,hist=True,rug=False,\
                 label='P( \tau | \tau > 0.5)').set(xlim=(0))
    sns.distplot(conditional_samples_B,hist=True,rug=False,\
                 label='P( \tau | \tau > 10)').set(xlim=(0,20))

    ks_result = sss.ks_2samp(conditional_samples_A - .5,regular_samples)
    print(ks_result)
    ks_result = sss.ks_2samp(conditional_samples_B - 10,regular_samples)
    print(ks_result)
    ks_result = sss.ks_2samp(conditional_samples_B,regular_samples)
    print(ks_result)
    # plt.show()


    
    


"""
TODO: 

-=-=- 0. Verify the alpha shape -=-=-
- 

-=-=- 1. verify the W's -=-=-=-
-generate grid via A and B
- use A_hat on both grids
-take likelihoods

"""
