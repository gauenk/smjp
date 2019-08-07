import numpy as np
import numpy.random as npr
from functools import partial
from pkg.distributions import WeibullDistribution as Weibull
from pkg.distributions import MultinomialDistribution as Multinomial
from pkg.utils import *
from pkg.hidden_markov_model import HMMWrapper,HiddenMarkovModel

def smjp_transition(s_curr_idx,s_next_idx,observation,augmented_state_space,A,B):
    """
    "augmented_state_space" includes the time discretization
    The "s_curr_idx,s_next_idx" here refers to the ~state indices~!

    returns P(v_i,l_i | v_{i-1}, l_{i-1}, \delta w_i)

    P(v_i,l_i | v_{i-1}, l_{i-1}, \delta w_i) = 
    1. P(l_i | l_{i-1}, \delta w_i)
    \times
    2. P(v_i,l_i | v_{i-1}, l_i, \delta w_i) 
    """
    log_probability = 0
    time_next = observation
    v_curr,l_curr = augmented_state_space[s_curr_idx]
    v_next,l_next = augmented_state_space[s_next_idx]
    t_hold = time_next - l_curr
    # (T,?,?)
    # (F,T,T)

    # print("dw",delta_w,"th",t_hold,"l_n",l_next,"l_c",l_curr)
    # print(v_curr,v_next)
    # print((l_next != 0),(l_next != t_hold))
    # print((l_next == t_hold), (v_next != v_curr))

    # Note: we can return immmediately if the "l_next" is not possible.
    # -> this is common since l_next can only be {0, l_curr + delta_w }
    # print("smjp_pi",t_hold,l_next,l_curr,v_next,v_curr, l_next == t_hold)
    if (l_next > time_next):
        return -np.infty

    # if we thin the current time-step, we can not transition state values.
    if (l_next < time_next) and (v_next != v_curr):
        return -np.infty

    # print(delta_w)
    # print(s_curr,s_next)
    # print(augmented_state_space[s_curr],augmented_state_space[s_next])
    # print(l_next == 0,l_next == t_hold,v_next == v_curr)

    # P(l_i | l_{i-1}, \delta w_i)
    l_ratio = A(t_hold)[v_curr] / B(t_hold)[v_curr]
    assert l_ratio <= 1, "the ratio of hazard functions should be <= 1"
    if l_next == time_next:
        log_probability += np.ma.log( [l_ratio] ).filled(-np.infty)[0]
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) : note below is _not_ a probability.
        log_probability += np.ma.log([ A(t_hold)[v_curr,v_next] ]).filled(-np.infty)[0]
    else:
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) = 1 if v_i == v_{i-1}
        # print(l_ratio)
        log_probability += np.ma.log( [1. - l_ratio] ).filled(-np.infty)[0]
    # print(log_probability)
    return log_probability # np.exp(log_probability)

def smjp_hazard_functions(s_curr,s_next,observation,state_space,h_create):
    """
    The State_Space here refers to the ~actual state values~!
    This is different from the entire sMJP "state space" which includes 
    the randomized grid.
    """
    t_hold = observation
    if s_next is None: # return Prob of leaving s_curr for ~any state~
        # possible error here for A_s
        # this "log" doesn't help much:
        ## its a sum-of-probs;
        ## "log" method is helpful when its a prod-of-probs
        # (e.g. not time for "log-sum-exp")
        rate = 0
        for s_next in state_space:
            # skip the same state (goal of this code snippet)
            if s_next == s_curr: continue 
            hazard_rate = h_create(s_curr,s_next)
            rate += hazard_rate.l(t_hold)
        # we don't normalized; if we do the result is incorrect.
        # hazard_rate = h_create(s_curr,s_curr)
        # current_rate =  hazard_rate.l(t_hold)
        # rate = rate / (rate + current_rate) # normalize over ~all~
        return rate
    else:  # return Prob of leaving s_curr for s_next; normalized over s_next
        hazard_rate = h_create(s_curr,s_next)
        rate = hazard_rate.l(t_hold)
        normalization_rate = rate
        for s_next in state_space:
            # skip the same state (goal of this code snippet)
            if s_next == s_curr: continue 
            hazard_rate = h_create(s_curr,s_next)
            normalization_rate += hazard_rate.l(t_hold)
        rate /= normalization_rate
        return rate
        

def compute_likelihood_obs(x,p_x,state_space,v_curr,inv_temp):
    # P(x | v_i )
    if isiterable(x):
        likelihood_x = 0
        for sample in x:
            x_state_index = state_space.index(sample)
            likelihood_x += p_x(x_state_index)[v_curr]
    else:
        x_state_index = state_space.index(x)
        likelihood_x = p_x(x_state_index)[v_curr]
    likelihood_x = likelihood_x**inv_temp
    return likelihood_x

def smjp_emission_unset(p_x,p_w,inv_temp,state_space,
                        s_curr,s_next,observation,aug_state_space):
    """
    P( x_i, \delta w_i | v_i, l_i )
    =
    P( x_i | v_i ) * P( \delta w_i | v_i, l_i )
    """

    x,time_current = observation
    # s_next not used; kept for compatability with the smjpwrapper
    v_curr,l_curr = aug_state_space[s_curr]
    t_hold = time_current - l_curr
    
    # P(x | v_i )
    if len(x) == 0: 
        # return no information when the observation is 
        likelihood_x = 1
    else:
        likelihood_x = compute_likelihood_obs(x,p_x,state_space,v_curr,inv_temp)

    # P( \delta w_i | v_i, l_i )
    likelihood_delta_w = p_w.l([l_curr, time_current],v_curr)

    likelihood = likelihood_x * likelihood_delta_w
    return likelihood

def smjp_emission_sampler(s_curr,s_next,observation,state_space,d_create):
    # s_next,observation not used; kept for compatability with the smjpwrapper
    distribution = d_create(s_curr)
    sampled_state = distribution.s(1)
    return sampled_state

def smjp_emission_likelihood(s_curr,s_next,observation,state_space,d_create):
    # s_next,observation not used; kept for compatability with the smjpwrapper
    # print("@@")
    # print(state_space)
    # print(state_space.index(observation))
    # print(observation)
    # print("--")
    distribution = d_create(s_curr)
    likelihood = distribution.l(observation)
    return likelihood

class sMJPWrapper(object):
    """
    We want to be able to call the transition matrix 
    for sMJP in a readable, nicely fashioned format.

    ex:
       pi(\delta w_i)[current_state]
       pi(\delta w_i)[current_state, next_state]

    ex:
      hazard_funcion(\tau_{hold})[current_state,next_state] # pseudo-"probability"
      hazard_funcion.sampler()[current_state] # sample over \tau_{hold}
      hazard_funcion.sampler(\tau_{hold})[current_state] # sample over next_state
    
    Note: "sampler" is aliased by "s", e.g. hazard_function.sampler = hazard_function.s

    This way it looks something similar to how it looks in the paper. Cool.

    Why not just pi(current_stat, next_state, \delta w_i)?
    I want to emphasize that each location is a ~function~ of
    the discrete, finite state space. Thus, a function call is 
    use for the proper function argument, \delta w_i.
    """

    def __init__(self,state_function,state_space,*args,\
                 observation_index=0,sampler=None,obs_is_iterable=False):
        self.state_function = state_function
        self.state_space = state_space
        self.observation = None
        # its always just delta_t... I was originally confused during implementation.
        # Todo: remove the "observation_index" and replace with more explicit delta_t or.
        # dependance of more generally (t_curr,t_next).
        self.observation_is_iterable = obs_is_iterable
        self.state_function_args = args
        self.sampler_bool = False
        self.sampler = sampler


    def s(self,hold_time = None, n = 1):
        # alias for "sampler"
        return self.sample(hold_time, n = n)

    def sample(self,hold_time = None, n = 1):
        """
        WARNING: This function is writting with the hazard functions in mind.
        As in, the transition matrix is not in mind when writing this function.
        """
        self.sampler_bool = True
        self.sampler_hold_time = hold_time
        self.sampler_n = n
        return self

    def __call__(self,observation):
        if self.sampler_bool:
            self.sampler_bool = False # no reason to have this here.
            raise SyntaxError("The object can not be called until it is sampled.")
        if isiterable(observation) and not self.observation_is_iterable:
            print(observation)
            raise TypeError("observation is iterable.")
        self.observation = observation
        return self

    def __getitem__(self,a_slice):
        if not isiterable(a_slice):
            return self._slice_1d(a_slice)
        if len(a_slice) > 2:
            raise IndexError("only a matrix; no more than 2 arguments please.")
        elif len(a_slice) == 2:
            return self._slice_2d(a_slice)
        else:
            raise NotImplemented("We've only implemented 2d for this.")
        
    def _slice_1d(self,a_slice):
        assert isiterable(a_slice) is False, "we can only accept non-iterables; maybe just ints?"
        s_curr = a_slice # current state
        if self.sampler_bool:
            # reset "sampler" bool so we don't mis-interpret this later
            self.sampler_n = None
            self.sampler_bool = False
            return self.sampler(self.sampler_hold_time,s_curr,None,self.sampler_n)
        result = self.state_function(s_curr,None,self.observation,
                                   self.state_space,*self.state_function_args)
        return result

    def _slice_2d(self,a_slice):
        assert len(a_slice) == 2, "we can only do 2d"
        s_curr,s_next = a_slice # (current,next) state
        if type(s_curr) is slice or type(s_next) is slice:
            raise TypeError("we can only accept integer inputs")
        # logic for functions based on current & next state (maybe 
        if self.sampler_bool: 
            # reset "sampler" bool so we don't mis-interpret this later
            self.sampler_n = None
            self.sampler_bool = False 
            sample = self.sampler(self.sampler_hold_time,s_curr,s_next,self.sampler_n)
            return sample
        result = self.state_function(s_curr,s_next,self.observation,
                                   self.state_space,*self.state_function_args)
        return result

"""
A(delta w_i)[v_curr] = the hazard function associated with 
                        leaving the state v_curr for any other state according

B(delta w_i)[v_curr] = same as above, but B > A (? B >= A?)

A(delta w_i)[v_curr,v_next] = the _normalized_ (w.r.t. v_next, fixed delta w_i and v_curr) 
                              hazard function
"""
    
class sMJPDataWrapper(object):
    """
    We need to be able to access all the data within a given interval for the 
    forward-algorithm in HMM

    API:
    "observation_data[step]" returns a list of elements between the previously
    indexed time and the current time.
    """
    def __init__(self,data,time):
        self.data = data
        self.time = time

    def __getitem__(self,a_slice):
        if not isiterable(a_slice):
            return self._slice_1d(a_slice)
        if len(a_slice) > 2:
            raise IndexError("only a matrix; two arguments please.")
        elif len(a_slice) == 2:
            return self._slice_2d(a_slice)
        else:
            raise NotImplemented("We've only implemented 2d for this.")
        
    def _slice_1d(self,a_slice):
        raise NotImplemented("Why are you using only a 1d slice for observations?")
        # if self.time == a_slice: 
        #     return self.data[a_slice]
        
    def _slice_2d(self,a_slice):
        data = self.data
        t_curr,t_next = a_slice
        data_in_interval = []
        for index,time in enumerate(self.time):
            if time < t_curr or time > t_next: continue
            data_in_interval += [data[index]]
        data_in_interval = np.array(data_in_interval)
        return data_in_interval

    def __str__(self):
        return str(np.c_[self.data,self.time])
    
class PoissonProcess(object):

    def __init__(self,state_space,mean_function,hazard_function,mean_function_params=None):
        self.state_space = state_space
        self.mean_function = self.weibull_mean
        self.mean_function_params = mean_function_params
        self.hazard_function = hazard_function

    def sample(self,interval,current_state):
        tau = interval[1] - interval[0]
        # print('tau',tau)
        samples = []
        for state in self.state_space:
            mean = self.mean_function( 0, tau, current_state, state )
            # print('mean',mean)
            N = npr.poisson( lam = mean )
            # print('N',N)
            samples_by_state = []
            i = 0
            while (i < N):
                sample = self.hazard_function.sample( n = 1 )[current_state,state]
                if sample <= tau:
                    samples_by_state.append(sample)
                    i += 1
            samples.extend(samples_by_state)
        samples = np.array(sorted(samples))
        # print('interval',interval)
        if len(samples) > 0:
            samples = samples + interval[0]
        # print('samples',samples)
        return samples

    def l(self,*args):
        return self.likelihood(*args)

    def likelihood(self,interval,current_state):
        tau = interval[1] - interval[0]
        # print('a',interval,interval[1])
        likelihood = self.hazard_function(interval[1])[current_state]
        log_likelihood = 0
        for state in self.state_space:
            mean = self.mean_function( interval[0], interval[1], current_state, state)
            #log_likelihood += self.poisson_likelihood(mean,0) #mistake i bet
            log_likelihood += mean
        likelihood *= np.exp(-log_likelihood)
        return likelihood

    def weibull_mean(self,t_start,t_end,current_state,state):
        # compute the Poisson process mean for weibull hazard function
        curr_state_index = self.state_space.index(current_state)
        next_state_index = self.state_space.index(state)
        shape = self.mean_function_params['shape'][curr_state_index][next_state_index]
        scale = self.mean_function_params['scale'][curr_state_index][next_state_index]
        # print(scale,t_end,t_start,shape)
        # print(scale**shape)
        mean = ( t_end**shape - t_start**shape ) / scale**shape
        return mean

    def poisson_likelihood(self,mean,k):
        return mean**k * np.exp(-mean) / np.math.factorial(k)
                
# weibull hazard rate experiment
def weibull_hazard_create_unset(shape_mat,scale_mat,state_space,state_curr,state_next):
    assert state_curr in state_space, "must have the state in state space"
    assert state_next in state_space, "must have the state in state space"
    state_curr_index = state_space.index(state_curr)
    state_next_index = state_space.index(state_next)
    shape = shape_mat[state_curr_index][state_next_index]
    scale = scale_mat[state_curr_index][state_next_index]
    rv = Weibull({'shape':shape,'scale':scale},is_hazard_rate=True)
    return rv

# sampler for sampling (1) prior trajectories & (2) discretizations
def smjp_hazard_sampler_unset(tate_space,h_create,hold_time,current_state,next_state,n=1):
    if hold_time is None: # sampling over hold_time "\tau" given (current_state,next_state)
        sample = h_create(current_state,next_state).sample(n)
        return sample
    else: # sampling over next state given (hold_time, current_state)
        state_rate = []
        for next_state in state_space:
            rate = h_create(current_state,next_state).l(hold_time)
            state_rate.append(rate)
        state_rate /= np.sum(state_rate)
        # could also do "argmax"
        # TODO: verify multinomial over "next_state"
        state_index_l = np.where(npr.multinomial(n,state_rate,n) == 1)[0]
        sampled_state = []
        for state_index in state_index_l:
            sampled_state += [state_space[state_index]]
        return sampled_state

def enumerate_state_space(grid,states):
    if 0 not in grid: # it shouldn't be i think
        grid = [0] + grid
    # # create all the "delta w_i" terms
    # deltas = []
    # i = 0
    # for time_a in grid:
    #     for time_b in grid:
    #         diff = time_a - time_b
    #         if diff > 0:
    #             deltas += [diff]
    #             i += 1
    # deltas += [0]
    # deltas = np.array(sorted(deltas))
    mesh = np.meshgrid(grid,states)
    state_space = np.c_[mesh[1].ravel(),mesh[0].ravel()]
    return state_space

def create_upperbound_scale(shape_mat,scale_mat,constant):
    shapes = shape_mat.ravel()
    const_mat = np.ones(len(shapes))
    for index,shape in enumerate(shapes):
        const_mat[index] = np.power(constant,1./shape)
    const_mat = const_mat.reshape(scale_mat.shape)
    scales_mat_tilde = scale_mat / const_mat
    return scales_mat_tilde


#
# sampling the random grid via a Poisson process
# 

def sample_smjp_event_times(poisson_process,V,T):
    W = None
    # sample thinned events
    v_curr,t_curr = V[0],T[0]
    T_tilde_list = []
    for v_next,t_next in zip(V[1:],T[1:]):
        interval = [t_curr,t_next]
        T_tilde_list += [poisson_process.sample(interval,v_curr)]
        v_curr,t_curr = v_next,t_next # reset
    T_tilde = np.concatenate(T_tilde_list)
    W = np.r_[T_tilde,T]
    W = sorted(W)
    return W

#
# sample from the prior 
#

def sample_smjp_trajectory_prior(pi, pi_0, state_space, t_end, t_start = 0):
    """
    ~~ Sampling from the prior ~~
    """
    # init the variables
    v = [pi_0.s()]
    w = [t_start]
    v_curr,w_curr = v[0],w[0]

    # sample until the end of the time period
    while(w_curr < t_end):
        
        # sample all holding times for the next state
        hold_time_samples = []
        for state in state_space:
            hold_time_samples += [ pi.sample()[v_curr,state] ]

        # take the smallest hold-time and move to that state
        t_hold = np.min(hold_time_samples)
        w_next = w_curr + t_hold
        v_next = state_space[np.argmin(hold_time_samples)]

        # append values to the vectors
        w += [w_next]
        v += [v_next]

        # update
        w_curr = w_next
        v_curr = v_next
        
    # correct final values
    w[-1] = t_end
    v[-1] = v[-2]

    return w,v

def sample_smjp_trajectory_prior_2(A, B, pi_0, pi, t_start = 0):
    """
    DOES NOT WORK
    ~~ Implementation of Algorithm 1 ~~
    ** does not include observations (duh) **
    """
    # init the variables
    v = [pi_0.s()]
    w = [t_start]
    l = [0]
    i = 0
    v_curr,w_curr,l_curr = v[0],w[0],l[0]

    # sample until the end of the time period
    while(w_curr < t_end):

        # alg. 1 [steps 1 - 9]
        t_hold = B[v_curr].sample() # conditional is the same as the ordinary.
        delta_w = t_hold - l_curr
        w_next = w_curr + delta_w
        accept_prob = A(t_hold)[v_curr] / B(t_hold)[v_curr]
        coin_flip = npr.randn(0,1,1)
        if(accept_prob > coin_flip):
            l_next = 0
            v_next = v_distribution.s() # what is the sampler for "v"?; it's part of "pi"
        else:
            l_next = l_curr + delta_w # l_next = w_{current index} - w_{previous jump index}
            v_next = v_curr
        i = i + 1

        # append values to the vectors
        w += [w_next]
        v += [v_next]
        l += [l_next]

    # correct final values
    w[-1] = t_end
    v[-1] = v[-2]
    l[-1] = l[-1] + w[-1] - w[-2]
    return w,v,l

#
# posterior smjp trajectory sampler; main function in Gibbs loop
#

def sample_smjp_trajectory_posterior(W,state_space,hazard_A,hazard_B,smjp_e,data):
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

    pi = sMJPWrapper(smjp_transition,augmented_state_space,hazard_A,hazard_B)

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
    
    hmm = HiddenMarkovModel([],**hmm_init)
    alphas,prob = hmm.likelihood() # only for dev.
    # print(alphas)
    # print(augmented_state_space)
    # print(np.exp(alphas))
    samples,t_samples = hmm.backward_sampling(alphas = alphas)
    # print(samples)
    # print(t_samples)
    s_states = t_samples[0][:,0]
    s_times = t_samples[0][:,1]
    return s_states,s_times,prob


#
# Misc
#

def create_toy_data(state_space,time_length,number_of_observations,emission_sampler):
    obs_times = sorted(npr.uniform(0,time_length,number_of_observations)) # random times
    gen_state_prob = np.ones(len(state_space)) / len(state_space) # all states equally likely
    states_index = np.where(npr.multinomial(1,gen_state_prob,number_of_observations) == 1)[1]
    data = []
    for state_index in states_index:
        state = state_space[state_index]
        # this is pretty un-necessary here since its 100 v 1
        data += [emission_sampler(0)[state]] 
    return data,obs_times
    
def smjp_emission_multinomial_create_unset(state_space,state_curr):
    mn_probs = np.ones(len(state_space))
    state_curr_index = state_space.index(state_curr)
    mn_probs[state_curr_index] = 100
    mn_probs /= np.sum(mn_probs)
    distribution = Multinomial({'prob_vector':mn_probs,'translation':state_space})
    return distribution
    
    
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
    y_values_1 = [pi(5)[i,j] for i in range(aug_ss_size) for j in range(aug_ss_size)]
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
    check_transition()

    
