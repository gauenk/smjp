import copy
import numpy as np
import numpy.random as npr
from functools import partial
from pkg.distributions import WeibullDistribution as Weibull
from pkg.distributions import MultinomialDistribution as Multinomial
from pkg.utils import *
from pkg.hidden_markov_model import HMMWrapper,HiddenMarkovModel
from pkg.fast_hmm import fast_HiddenMarkovModel
from pkg.timer import Timer

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
        """
        interval = [time_current,time_next]
        """
        # print('tau',tau)
        samples = []
        for state in self.state_space:
            if tau < 0:
                print(tau)
                print(self.state_space)
                exit()
            mean = self.mean_function( 0, tau, current_state, state )
            N = npr.poisson( lam = mean )
            # print('(mean, tau, N): ({},{},{})'.format(mean,tau,N))
            samples_by_state = []

            # --> version 1: force the # samples = N ~ Poisson ( mean ) <--
            # this is rejection sampling over the truncated distribution
            i = 0
            while (i < N):
                sample = self.hazard_function.sample( current_state,state, None, n = 1 )
                if sample <= tau:
                    samples_by_state.append(sample)
                    i += 1
            samples.extend(samples_by_state)
            
            # --> version 2: sample N; keep samples within interval <-- WRONG!
            # for i in range(N):
            #     sample = self.hazard_function.sample( n = 1 )[current_state,state]
            #     if sample <= tau:
            #         samples_by_state.append(sample)
            # samples.extend(samples_by_state)
            
        samples = np.array(sorted(samples))
        # print('interval',interval)
        if len(samples) > 0:
            samples = samples + interval[0] # handles the offset by interval
        return samples

    def l(self,*args,**kwargs):
        return self.likelihood(*args,**kwargs)

    def likelihood(self,interval,current_state,is_poisson_event=True):
        tau = interval[1]
        """
        interval = ["current_hold_time", "next_hold_time"]
        tau \neq delta w.... we

        delta w_i = interval[1] - interval[0]
        we want the total hold time though...
        """
        # A_{s,s'}(\tau) is the hazard function
        # = [\sum_{s'\in S} A_{s,s'}(\tau)] * exp\{ \int_{l_i}^{l_i + \delta w_i} A_s(\tau) \}
        # -----------------------------
        # -=-=-=-> version 1 <-=-=-=-=-
        # -----------------------------
        if not is_poisson_event:
            log_exp_term = 0
            for next_state in self.state_space:
                log_exp_term += self.mean_function(interval[0], interval[1], current_state,next_state)
            return np.exp(-log_exp_term)

        front_term = 0
        log_exp_term = 0
        for next_state in self.state_space:
            front_term += self.hazard_function(tau,current_state,next_state)
            log_exp_term += self.mean_function( interval[0], interval[1], current_state, \
                                                next_state)
        # print("front_term",front_term)
        # print("exp_term",np.exp(-log_exp_term))
        likelihood = front_term * np.exp(-log_exp_term)
        return likelihood

    def weibull_mean(self,t_start,t_end,current_state,state):
        # compute the Poisson process mean for weibull hazard function
        curr_state_index = self.state_space.index(current_state)
        next_state_index = self.state_space.index(state)
        shape = self.mean_function_params['shape'][curr_state_index][next_state_index]
        scale = self.mean_function_params['scale'][curr_state_index][next_state_index]
        # print(scale,t_end,t_start,shape)
        # print(scale**shape)
        mean = (t_end/scale)**shape - (t_start/scale)**shape
        # print(mean)
        return mean

    def update_parameters(self,parameters):
        if 'shape' in parameters.keys():
            self.mean_function_params['shape'] = parameters['shape']
        if 'scale' in parameters.keys():
            self.mean_function_params['scale'] = parameters['scale']

    def poisson_likelihood(self,mean,k):
        return mean**k * np.exp(-mean) / np.math.factorial(k)
                
def enumerate_state_space(grid,states):
    """
    --> we do not include the final jump time <--
    it is not a poisson event
    """
    
    grid = grid[:-1]
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

def sample_smjp_event_times(poisson_process,V,T,time_length):
    W = None
    # sample thinned events
    v_curr,t_curr = V[0],T[0]
    T_tilde_list = []
    if False:
        print("-=-=-=-DOWN-=-=-=-")
        print(np.c_[V,T])
        print("-=-=-=- UP -=-=-=-")
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

def sample_smjp_trajectory_prior(hazard_A, pi_0, state_space,t_end, t_start = 0, v_0 = None):
    """
    ~~ Sampling from the prior ~~
    """
    # init the variables
    if v_0 is None:
        v = [pi_0.s()]
    else:
        v = [v_0]        
    w = [t_start]
    v_curr,w_curr = v[0],w[0]

    # sample until the end of the time period
    while(w_curr < t_end):
        
        # sample all holding times for the next state
        hold_time_samples = []
        for state in state_space:
            hold_time_samples += [ hazard_A.sample(v_curr,state,None) ]

        # take the smallest hold-time and move to that state
        t_hold = np.min(hold_time_samples)
        v_next = state_space[np.argmin(hold_time_samples)]
        w_next = w_curr + t_hold

        # append values to the vectors
        v += [v_next]
        w += [w_next]

        # update
        v_curr = v_next
        w_curr = w_next

        
    # correct final values
    v[-1] = v[-2]
    w[-1] = t_end

    return v,w

#
# posterior smjp trajectory sampler; main function in Gibbs loop
#

# "another name for the same thing" ~ DCFC
def smjp_ffbs(W,data,state_space,hazard_A,hazard_B,smjp_e,t_end,u_str):
    return sample_smjp_trajectory_posterior(W,data,state_space,hazard_A,hazard_B,smjp_e,t_end,u_str)

def sample_smjp_trajectory_posterior(W,data,state_space,hazard_A,hazard_B,smjp_e,t_end,u_str):
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
    smjp_e.augmented_state_space = augmented_state_space

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
    pi = smjpTransitionFunction(augmented_state_space,hazard_A,hazard_B)

    # ----------------------------------------------------------------------
    #
    # Run the HMM 
    #
    # ----------------------------------------------------------------------

    hmm_init = {'emission': smjp_e,
                'transition': pi,
                'data': data,
                'state_alphabet': augmented_state_space,
                'pi_0': pi_0,
                'time_grid': W,
                'sample_dimension': 1,
                'uuid_str':u_str,
                }
    
    hmm = fast_HiddenMarkovModel([],**hmm_init)
    alphas,prob = hmm.likelihood() 
    samples,t_samples = hmm.backward_sampling(alphas = alphas)

    # get unique thinned values
    thinned_samples = np.unique(t_samples[0],axis=0) # ignored thinned samples
    thinned_samples = thinned_samples[thinned_samples[:,1].argsort()] # its out of order now.
    
    # ensure [t_end] is included; copy the findal state to time t_end
    thinned_samples = include_endpoint_in_thinned_samples(thinned_samples,t_end)

    # print(thinned_samples)
    # print(samples)
    # print(t_samples)
    s_states = thinned_samples[:,0]
    s_times = thinned_samples[:,1]
    return s_states,s_times,prob


#
# Misc
#


def include_endpoint_in_thinned_samples(thinned_samples,t_end):
    """
    needing this might be wrong...
    """
    if np.isclose(thinned_samples[-1,1],t_end):
        return thinned_samples
    else:
        final_state,final_time = thinned_samples[-1,:]
        end = np.array([final_state,t_end])[np.newaxis,:]
        complete_thinned_samples = np.r_[thinned_samples,end]
        return complete_thinned_samples

def sample_data_posterior(V,T,state_space,emission,obs_times):
    # -- sample via emission probability @ fixed times --
    ss = len(state_space)
    data_states = []
    data_times = []
    for obs_time in obs_times:
        time_c = T[0]
        for state,time_n in zip(V[:-1],T[1:]):
            if time_c <= obs_time < time_n: 
                data_states += [emission.sample(state,1)] 
                data_times += [ obs_time ]
                break
            time_c = time_n
    data = sMJPDataWrapper(data=data_states,time=data_times)
    return data

def create_toy_data(state_space,time_length,number_of_observations,emission):
    data = []
    for state_index in states_index:
        state = state_space[state_index]
        # this is pretty un-necessary here since its 100 v 1
        data += [emission.sample(state,1)] 
    return data
    
class smjpTransitionFunction(object):

    def __init__(self,augmented_state_space,hazard_A,hazard_B):
        self.state_space = augmented_state_space
        self.augmented_state_space = augmented_state_space
        self.hazard_A = hazard_A
        self.hazard_B = hazard_B

    def __call__(self,s_prev_idx,s_curr_idx,time_p,time_c):
        log_probability = 0
        v_prev,l_prev = self.augmented_state_space[s_prev_idx]
        v_curr,l_curr = self.augmented_state_space[s_curr_idx]
        t_hold = time_c - l_prev
        error_conditions = (time_p >= time_c)
        invalid_conditions = (l_prev > l_curr) or (t_hold <= 0)
        # the "jump" condition
        true_condition_1 = (l_curr == time_c) and (l_prev <= time_p)
        # the "thin" condition
        true_condition_2 = (l_curr == l_prev) and (v_curr == v_prev) and (l_prev <= time_p)
        # the "self-transition" condition
        true_condition_3 = (l_curr == l_prev) and (v_curr == v_prev) and (time_p == time_c)
        # print(time_p == time_c,time_p,time_c)
        any_true_conditions = true_condition_1 or true_condition_2 or true_condition_3
        if error_conditions or invalid_conditions or (not any_true_conditions):
            if error_conditions:
                raise ValueError("We can not have time_p >= time_c")
            return -np.infty

        # P(l_i | l_{i-1}, \delta w_i)
        A = self.hazard_A
        B = self.hazard_B
        l_ratio_A = np.ma.log([ A(t_hold,v_prev,None) ]).filled(-np.infty)[0]
        l_ratio_B = np.ma.log([ B(t_hold,v_prev,None) ]).filled(-np.infty)[0]
        l_ratio = l_ratio_A - l_ratio_B
        assert l_ratio <= 0, "the ratio of hazard functions should be <= 1"
        if l_curr == time_c:
            # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) : note below is _not_ a probability.
            log_A = np.ma.log( [ A(t_hold,v_prev,v_curr) ] ).filled(-np.infty)[0]
            log_probability = log_A - l_ratio_B
        else:
            # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) = 1 if v_i == v_{i-1}
            log_probability = np.ma.log( [1. - np.exp(l_ratio)] ).filled(-np.infty)[0]
        return log_probability # np.exp(log_probability)


class smjpHazardFunction(object):

    """
    hazard function is weibull for now.
    """

    def __init__(self,state_space,shape_mat,scale_mat,omega=None):
        self.state_space = state_space
        self.shape_mat = shape_mat
        self.scale_mat = scale_mat
        self.omega = omega

    def update_parameters(self,parameters):
        if 'shape' in parameters.keys():
            self.shape_mat = parameters['shape']
        if 'scale' in parameters.keys():
            self.scale_mat = parameters['scale']
            
    def h_create(self,state_curr,state_next):
        assert state_curr in self.state_space, "must have the state in state space"
        assert state_next in self.state_space, "must have the state in state space"
        state_curr_index = self.state_space.index(state_curr)
        state_next_index = self.state_space.index(state_next)
        shape = self.shape_mat[state_curr_index][state_next_index]
        scale = self.scale_mat[state_curr_index][state_next_index]
        rv = Weibull({'shape':shape,'scale':scale},is_hazard_rate=True)
        return rv

    def __call__(self,observation,s_curr,s_next):
        t_hold = observation
        if s_next is None: # return Prob of leaving s_curr for ~any state~
            rate = 0
            for s_next in self.state_space:
                hazard_rate = self.h_create(s_curr,s_next)
                rate += hazard_rate.l(t_hold)
        else:  # return Prob of leaving s_curr for s_next; normalized over s_next
            hazard_rate = self.h_create(s_curr,s_next)
            rate = hazard_rate.l(t_hold)
        return rate

    def sample(self,current_state,next_state,hold_time,n=1):
        if hold_time is None: # sampling over hold_time "\tau" given (current_state,next_state)
            # print(" [[ hold_time ]]")
            sample = self.h_create(current_state,next_state).sample(n)
            return sample
        elif next_state is None: # sampling over next state given (hold_time, current_state)
            # print(" [[ next state ]]")
            assert next_state is None,"next state must be None"
            state_rate = []
            for next_state in self.state_space:
                rate = self.h_create(current_state,next_state).l(hold_time)
                state_rate.append(rate)
            state_rate /= np.sum(state_rate)
            state_index_l = np.where(npr.multinomial(n,state_rate,n) == 1)[0]
            sampled_state = []
            for state_index in state_index_l:
                sampled_state += [self.state_space[state_index]]
            return sampled_state
        else: # sampling over hold_time conditioned on current holdtime
            # print("[[ conditional ]]")
            sample = self.h_create(current_state,next_state).sample(n,hold_time=hold_time)
            return sample

class smjpEmission(object):

    def __init__(self,state_space,p_w,time_final,likelihood_power,ll_compare=10):
        self.state_space = state_space
        self.p_x = self.likelihood #?
        self.p_w = p_w
        self.augmented_state_space = None
        self.likelihood_power = likelihood_power
        self.how_likely_is_current_state_compared_to_others = ll_compare
        self.final_grid_time = time_final

    def compute_likelihood_obs(self,x,v_curr):
        # P(x | v_i )
        if isiterable(x):
            likelihood_x = 0
            for sample in x:
                x_state_index = self.state_space.index(sample)
                likelihood_x += self.p_x(x_state_index,v_curr)
        else:
            x_state_index = self.state_space.index(x)
            likelihood_x = self.p_x(x_state_index,v_curr)
        likelihood_x = likelihood_x**self.likelihood_power
        return likelihood_x

    def __call__(self,obs,s_curr):
        """
        this is called inside of the FFBS algorithm
        """
        if self.augmented_state_space is None:
            raise ValueError("smjp emission not using augmented state space")
        """
        p_x,p_w,final_grid_time,inv_temp,state_space,
        s_curr,s_next,observation,aug_state_space
        """
        # obs = [O[0,time_next],[0,time_next]]
        x,times = obs
        time_c,time_n = times
        # s_next not used; kept for compatability with the smjpwrapper
        v_curr,l_curr = self.augmented_state_space[s_curr]
        t_hold = time_n - l_curr
        t_hold_c = time_c - l_curr

        invalid_conditions = (time_c >= time_n)
        if invalid_conditions:
            return -np.infty

        if t_hold <= 0: # we want t_hold >= 0
            return -np.infty

        if t_hold_c < 0:
            return -np.infty
            print(times)
            print(t_hold_c)
            print(l_curr)
            exit()
        # P(x | v_i )
        if len(x) == 0: 
            # return no information when the observation is 
            likelihood_x = 1
        else:
            likelihood_x = self.compute_likelihood_obs(x,v_curr)

        if not np.isclose(self.final_grid_time,time_n):
            likelihood_delta_w = self.p_w.l([t_hold_c,t_hold],v_curr)
        else: # the final event is not a poisson process
            likelihood_delta_w = self.p_w.l([t_hold_c,t_hold],v_curr,is_poisson_event=False)
        likelihood = likelihood_x * likelihood_delta_w
        log_likelihood = np.ma.log([likelihood]).filled(-np.infty)[0]
        return log_likelihood
        
    def d_create(self,state_curr):
        mn_probs = np.ones(len(self.state_space))
        state_curr_index = self.state_space.index(state_curr)
        mn_probs[state_curr_index] = self.how_likely_is_current_state_compared_to_others
        mn_probs /= np.sum(mn_probs)
        distribution = Multinomial({'prob_vector':mn_probs,'translation':self.state_space})
        return distribution

    def sample(self,s_curr,n=1):
        distribution = self.d_create(s_curr)
        sampled_state = distribution.s(n=n)
        return sampled_state

    def likelihood(self,observation,s_curr):
        distribution = self.d_create(s_curr)
        likelihood = distribution.l(observation)
        return likelihood

if __name__ == "__main__":
    print("HI")
    
