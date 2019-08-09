import numpy as np
import numpy.random as npr
from functools import partial
from pkg.distributions import WeibullDistribution as Weibull
from pkg.distributions import MultinomialDistribution as Multinomial,MultinomialDistribution
from pkg.utils import *
from pkg.hidden_markov_model import HMMWrapper,HiddenMarkovModel
from pkg.fast_hmm import fast_HiddenMarkovModel
from pkg.timer import Timer
from pkg.smjp_utils import *

def smjp_setup_exp():
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
    number_of_samples = 6000
    smjp_sampler_input = [state_space,hazard_A,hazard_B,smjp_emission,data,time_length]
    pi_0 = MultinomialDistribution({'prob_vector': np.ones(s_size)/s_size,\
                                    'translation':state_space})
    V,T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)

    if True:
        for i in range(number_of_samples):

            # ~~ the main gibbs sampler for mcmc of posterior sample paths ~~
            W = sample_smjp_event_times(poisson_process_A,V,T,time_length)
            # just for testing
            # W = np.arange(10)
            # while len(W) > 9:
            #     W = sample_smjp_event_times(poisson_process_A,V,T)

            V,T,prob = sample_smjp_trajectory_posterior_speed_test(W,*smjp_sampler_input)
            aggregate['W'].append(W)
            aggregate['V'].append(V)
            aggregate['T'].append(T)
            aggregate['prob'].append(prob)
            print(np.c_[V,T])

            # take some samples from the prior for de-bugging the mcmc sampler
            _V,_T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)
            aggregate_prior['V'].append(_V)
            aggregate_prior['T'].append(_T)
            print("i = {}".format(i))

        # save to memory
        pickle_mem_dump = {'agg':aggregate,'agg_prior':aggregate_prior}
        with open('results.pkl','wb') as f:
            pickle.dump(pickle_mem_dump,f)
    else:
        # load to memory
        with open('results.pkl','rb') as f:
            pickle_mem_dump = pickle.load(f)
        aggregate = pickle_mem_dump['agg']
        aggregate_prior = pickle_mem_dump['agg_prior']
        
    print(aggregate,aggregate_prior)

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

    

def sample_smjp_trajectory_posterior_speed_test(W,state_space,hazard_A,hazard_B,smjp_e,data,t_end):
    # we might need to mangle the "data" to align with the time intervals;
    # 1.) what about no observations in a given time? O[t_i,t_{i+1}] = []
    # 2.) what about multiple obs for a given time? O[t_i,t_{i+1}] = [o_k,o_k+1,...]
    

    augmented_state_space = enumerate_state_space(W,state_space)
    aug_ss_size = len(augmented_state_space)

    # ----------------------------------------------------------------------
    #
    # data likelihood defined over augmented state space
    #
    # ----------------------------------------------------------------------
    emission = sMJPWrapper(smjp_e,augmented_state_space,obs_is_iterable=True)

    # ----------------------------------------------------------------------
    #
    # pi_0 should actually picks from state_space; {1,2,3} BUT in the augmented_state_space
    # That is, we pick only from {[1,0],[2,0],[3,0]}
    #
    # ----------------------------------------------------------------------
    start_indices = np.where(np.isclose(augmented_state_space[:,1],0) == 1)[0]
    prob_vector = np.zeros(aug_ss_size)
    prob_vector[start_indices] = 1/len(start_indices)
    pi_0 = Multinomial({'prob_vector': prob_vector,'translation':state_space})

    # ----------------------------------------------------------------------
    #
    # transition probability over augmented state space
    #
    # ----------------------------------------------------------------------

    pi = sMJPWrapper(smjp_transition,augmented_state_space,hazard_A,hazard_B,obs_is_iterable=True)

    # ----------------------------------------------------------------------
    #
    # Run the HMM 
    #
    # ----------------------------------------------------------------------

    hmm_init = {'emission': emission,
                'transition': pi,
                'data': data,
                'state_alphabet': augmented_state_space,
                'pi_0': pi_0,
                'time_grid': W,
                'sample_dimension': 1,
                }
    
    _t = {'ordinary':Timer(),'fast':Timer()}
    hmm = HiddenMarkovModel([],**hmm_init)
    for i in range(1):
        _t['ordinary'].tic()
        alphas,prob = hmm.likelihood() # only for dev.
        samples,t_samples = hmm.backward_sampling(alphas = alphas)
        _t['ordinary'].toc()

    hmm = fast_HiddenMarkovModel([],**hmm_init)
    for i in range(1):
        _t['fast'].tic()
        alphas_f,prob_f = hmm.likelihood() # only for dev.
        samples_f,t_samples_f = hmm.backward_sampling(alphas = alphas_f)
        _t['fast'].toc()

    if np.all(np.isclose(alphas,alphas_f)):
         print("both are the same.")
    else:
         print("they are the different!")
         print(alphas)
         print(alphas_f)
    print(t_samples_f)

    for key,val in _t.items():
        print(key,val)
    exit()
    fast_HiddenMarkovModel

    # print(alphas)
    # print(augmented_state_space)
    # print(np.exp(alphas))
    samples,t_samples = hmm.backward_sampling(alphas = alphas)


    # get unique thinned values
    thinned_samples = np.unique(t_samples[0],axis=0) # ignored thinned samples
    thinned_samples = thinned_samples[thinned_samples[:,1].argsort()] # its out of order now.
    
    # ensure [t_end] is included; copy the findal state to time t_end
    thinned_samples = include_endpoint_in_thinned_samples(thinned_samples,t_end)

    print(thinned_samples)
    # print(samples)
    # print(t_samples)
    s_states = thinned_samples[:,0]
    s_times = thinned_samples[:,1]
    return s_states,s_times,prob


#
# Testing.
#



def check_weibull():
    # the two plots should overlap
    
    import matplotlib.pyplot as plt
    States = [1,2,3]
    s_size = len(States)
    W = np.arange(20) # random_grid()
    augmented_state_space = enumerate_state_space(W,States)
    aug_ss_size = len(augmented_state_space)

    # shape_mat = npr.uniform(0.6,3,s_size**2).reshape((s_size,s_size))
    shape_mat = np.ones((s_size,s_size)) * .8
    scale_mat = np.ones((s_size,s_size)) 
    weibull_hazard_create = partial(weibull_hazard_create_unset,shape_mat,scale_mat,States)
    hazard_A = sMJPWrapper(smjp_hazard_functions,States,weibull_hazard_create)

    scale_mat_tilde = create_upperbound_scale(shape_mat,scale_mat,2)
    weibull_hazard_create_B = partial(weibull_hazard_create_unset,shape_mat,\
                                      scale_mat_tilde,States)
    hazard_B = sMJPWrapper(smjp_hazard_functions,States,weibull_hazard_create_B)

    x_values = np.arange(0,20.1)
    y_values_A = [hazard_A(x)[1,1] for x in x_values]
    y_values_B = [hazard_B(x)[1,1] for x in x_values]
    # y_values_ratio = [A([x])[60] / B([x])[60] for x in x_values] # ratio is constant b/c def B
    print(y_values_A)
    print(y_values_B)

    plt.plot(x_values,y_values_A,'k+-',label="A")
    plt.plot(x_values,y_values_B,'g+-',label="B")
    # plt.plot(x_values,y_values_ratio,'m+-',label="A/B")
    w_rv = Weibull({'shape':1.0,'scale':1.0})
    y_values_wb = [w_rv.l(x) for x in x_values]
    # plt.plot(x_values,y_values_wb,'m+-',label="wb")
    plt.legend()
    plt.title("Inpecting Weibull Functions")
    plt.show()

    print(hazard_A(5)[1,2])
    print(hazard_A(2)[1,3])
    print(hazard_A(5)[1,3])
    print(hazard_A(2)[1,3])
    print(hazard_A(5)[1])
    print(hazard_A(5)[2])


def check_transition():

    States = [1,2,3]
    # reset each Gibbs iteration
    s_size = len(States)
    W = np.arange(20) # random_grid()
    augmented_state_space = enumerate_state_space(W,States)
    aug_ss_size = len(augmented_state_space)
          
    shape_mat = npr.uniform(0.6,1.1,s_size**2).reshape((s_size,s_size))
    # shape_mat = np.ones((s_size,s_size))
    scale_mat = np.ones((s_size,s_size)) 
    weibull_hazard_create_A = partial(weibull_hazard_create_unset,shape_mat,scale_mat,States)

    scale_mat_tilde = create_upperbound_scale(shape_mat,scale_mat,2)
    print(scale_mat_tilde[0,:5],scale_mat[0,:5])
    weibull_hazard_create_B = partial(weibull_hazard_create_unset,shape_mat,scale_mat_tilde,States)

    hazard_A = sMJPWrapper(smjp_hazard_functions,States,weibull_hazard_create_A)
    hazard_B = sMJPWrapper(smjp_hazard_functions,States,weibull_hazard_create_B)
    
    print(hazard_A(1)[1,2])
    print(hazard_B(1)[1,2])
    print(hazard_A(1)[1])
    print(hazard_B(1)[1])

    pi = sMJPWrapper(smjp_transition,augmented_state_space,hazard_A,hazard_B)
    
    import matplotlib.pyplot as plt

    # plot over augmented_state_space
    x_values = np.arange(0,aug_ss_size**2,1)
    print("_1_")
    y_values_1 = [pi([0,5])[i,j] for i in range(aug_ss_size) for j in range(aug_ss_size)]
    y_values_h = [hazard_A(5)[i,j] for i in States for j in States]
    # print("_2_")
    # y_values_2 = [pi(2)[0,1] for x in state_space]
    # print("_3_")
    # y_values_3 = [pi(3)[1,1] for x in state_space]
    # print("_4_")
    # y_values_4 = [pi(4)[1,2] for x in state_space]

    # plt.plot(x_values,y_values_0,'k*',label='(0,0)')
    plt.plot(x_values,y_values_1,'k*',label='(0,0)')
    # plt.plot(x_values,y_values_h,'g+',label='(hazard)')
    # plt.plot(x_values,y_values_2,'g+',label='(0,1)')
    # plt.plot(x_values,y_values_3,'rx',label='(1,1)')
    # plt.plot(x_values,y_values_4,'c^',label='(1,2)')
    plt.title("Transition matrix")
    plt.ylabel("Log-Likelihood")
    plt.xlabel("State Values")
    plt.legend(title="(Current,Next)")
    print("alphas")
    print(shape_mat)
    print("hazardA")
    print(y_values_h)

    plt.show()

    exit()

    # plot over delta_w

    """
    Note that when we leave the state, the probability of leaving a state S at any time is the 
    same. 
    """
    print(augmented_state_space)
    x_values = W
    print("_00_")
    y_values_00 = -np.ma.log([pi(x)[0,2] for x in W]).filled(0)
    print("_01_")
    y_values_01 = -np.ma.log([pi(x)[0,3] for x in W]).filled(0)
    print("_11_")
    y_values_11 = -np.ma.log([pi(x)[5,6] for x in W]).filled(0)
    print("_12_")
    y_values_12 = -np.ma.log([pi(x)[23,29] for x in W]).filled(0)

    print(y_values_00)
    print(y_values_01)
    print(y_values_11)
    print(y_values_12)

    plt.plot(x_values,y_values_00,'k*',label='(0,0)')
    plt.plot(x_values,y_values_01,'g+',label='(0,1)')
    plt.plot(x_values,y_values_11,'rx',label='(1,1)')
    plt.plot(x_values,y_values_12,'c^',label='(1,2)')
    plt.title("Transition matrix")
    plt.ylabel("Likelihood")
    plt.xlabel("Time Steps")
    plt.legend(title="(Current,Next)")
    plt.show()


if __name__ == "__main__":
    # check_weibull() # passsed.
    # check_transition()
    smjp_setup_exp()


