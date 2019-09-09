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

def smjp_transition(s_prev_idx,s_curr_idx,observation,augmented_state_space,A,B):
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
    time_p,time_c = observation
    v_prev,l_prev = augmented_state_space[s_prev_idx]
    v_curr,l_curr = augmented_state_space[s_curr_idx]
    t_hold = time_c - l_prev

    # if t_hold < 0:
    #     return -np.infty
    # (T,?,?)
    # (F,T,T)

    # print("dw",delta_w,"th",t_hold,"l_n",l_next,"l_c",l_curr)
    # print(v_curr,v_next)
    # print((l_next != 0),(l_next != t_hold))
    # print((l_next == t_hold), (v_next != v_curr))

    # we can't hold for negative time; this happens when
    # l_next > time_c (good to go) 
    # ~but~
    # l_current > time_c (current state is greater than current time)
    # --> this only happens in backward-sampling since we can sample "backward in time"
    # resulting in l_currents that are actually greater than the time we are at.
    """
    Needed when saying: "l_curr" is the time ~of~ the most recent jump
    
    This does not happen in "forward-filter" since we consider each state in sequence with
    "time_c". Iow, we have l_curr <= time_c since we have the "if l_next < time_c"
    condition. 

    The key, l_next becomes l_curr.

    For backward sampling, we the case when we sample a time futher in the past
    than the current timestep. This means the sample, which becomes l_next on the next 
    iteration, can be further in the past than the current time.

    The key, l_curr becomes l_next.

    The issue with allowing is that we allow impossible transitions to occur. 
    For example, at time (t) we might sample (v,t-2). 
    This says: the most recent transition was at time "t-2" and it was to state "v".
    Then at time (t-1), we look at the future state of (v,t-2). If we allow transitions
    at time (t-1), then the sampled state of (v,t-2) is a lie. Without the following 
    condition, that is exactly what happens.
    """
    # if (l_prev > l_curr): 
    #     return -np.infty
    # # if (l_curr == l_next) then we've thinned an event. This is okay.

    # # says: we can not "jump" into the past [very confident]
    # # if we are holding a state, we can not change the most recent jump time.
    # # if we do change the jump time, it must happen ~not before~ time_p
    # if (l_curr < time_p) and (s_prev_idx != s_curr_idx):
    #     return -np.infty

    # # Note: we can return immmediately if the "l_next" is not possible.
    # # -> this is common since l_next can only be {0, l_curr + delta_w }
    # # print("smjp_pi",t_hold,l_next,l_curr,v_next,v_curr, l_next == t_hold)
    # if (l_curr > time_p):
    #     return -np.infty

    # # if we thin the current time-step,
    # # 1.) we can not transition state values.
    # if (l_curr < time_c) and (s_prev_idx != s_curr_idx):
    #     return -np.infty

    
    error_conditions = (time_p >= time_c)
    invalid_conditions = (l_prev > l_curr) or (t_hold <= 0)
    # the "jump" condition
    true_condition_1 = (l_curr == time_c) and (l_prev <= time_p)
    # the "thin" condition
    true_condition_2 = (l_curr == l_prev) and (v_curr == v_prev) and (l_prev <= time_p)
    # the "self-transition" condition
    true_condition_3 = (l_curr == l_prev) and (v_curr == v_prev) and (time_p == time_c)
    # print(time_p == time_c,time_p,time_c)
    
    # print(error_conditions,invalid_conditions,\
        # true_condition_1,true_condition_2,true_condition_3)
    # "<="? or "<" for (l_prev ?? time_p)
    """
    I think we need the allow the equality due to self transitions.
    E.g. We need to be able "transition" from (1,0) -> (1,0) when computing 
    the forward probabilities
    
    But on the other, more concrete, hand we have that l_prev cannot be equal to
    time_p since "the most recent jump" cannot happen in the future.

    BUT we might be having a translation issue between "time_p" 
    and the "delta w_i" terms from the paper. 
    """
    
    any_true_conditions = true_condition_1 or true_condition_2 or true_condition_3
    # print(invalid_conditions)
    # print((time_p >= time_c), (l_prev > l_curr))
    # print(l_prev, l_curr)
    # print(true_condition_1)
    # print(true_condition_2)
    if error_conditions or invalid_conditions or (not any_true_conditions):
        if error_conditions:
            raise ValueError("We can not have time_p >= time_c")
        return -np.infty
    
    # P(l_i | l_{i-1}, \delta w_i)
    l_ratio_A = np.ma.log([ A(t_hold)[v_prev] ]).filled(-np.infty)[0]
    l_ratio_B = np.ma.log([ B(t_hold)[v_prev] ]).filled(-np.infty)[0]
    l_ratio = l_ratio_A - l_ratio_B
    assert l_ratio <= 0, "the ratio of hazard functions should be <= 1"
    if l_curr == time_c:
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) : note below is _not_ a probability.
        log_A = np.ma.log( [ A(t_hold)[v_prev,v_curr] ] ).filled(-np.infty)[0]
        log_probability = log_A - l_ratio_B
        # print("(time_p,time_c,t_hold): ({:.3f},{:.3f},{:.3f})".format(time_p,time_c,t_hold))
        # print("-------- (A_n,B,lp): ({:.3f},{:.3f},{:.3f}) ---------".format(np.exp(log_A),np.exp(l_ratio_B),np.exp(log_probability)))

        # = A(t_hold)[v_curr,v_next] / B(t_hold)[v_curr]
    else:
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) = 1 if v_i == v_{i-1}
        # print(l_ratio)
        log_probability = np.ma.log( [1. - np.exp(l_ratio)] ).filled(-np.infty)[0]
    # print(log_probability)
    return log_probability # np.exp(log_probability)

def smjp_hazard_functions(s_curr,s_next,observation,state_space,h_create,normalized=True):
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
            hazard_rate = h_create(s_curr,s_next)
            rate += hazard_rate.l(t_hold)
        return rate
        # rate_of_leaving = 0
        # leaving_rates_by_state = {state:0 for state in state_space}
        # for index,s_curr_prime in enumerate(state_space):
        #     rate_of_leaving = 0
        #     for s_next in state_space:
        #         hazard_rate = h_create(s_curr_prime,s_next)
        #         rate_of_leaving += hazard_rate.l(t_hold)
        #     leaving_rates_by_state[s_curr_prime] = rate_of_leaving
        # lrbs = list(leaving_rates_by_state.values())
        # rate = leaving_rates_by_state[s_curr]
        # print(rate)
        
        """
        We used to _not_ normalize, but now we normalize because of the transition probability
        """
        # hazard_rate = h_create(s_curr,s_curr)
        # current_rate =  hazard_rate.l(t_hold)
        # rate = rate / (rate + current_rate) # normalize over ~all~
        return rate
    else:  # return Prob of leaving s_curr for s_next; normalized over s_next
        hazard_rate = h_create(s_curr,s_next)
        rate = hazard_rate.l(t_hold)
        # if normalized:
        #     normalization_rate = rate
        #     for s_next in state_space:
        #         # skip the same state (goal of this code snippet)
        #         if s_next == s_curr: continue 
        #         hazard_rate = h_create(s_curr,s_next)
        #         normalization_rate += hazard_rate.l(t_hold)
        #     rate /= normalization_rate
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

def get_final_grid_time_from_state_space(state_space,time_col):
    return np.max(state_space[:,time_col])

def smjp_emission_unset(p_x,p_w,final_grid_time,inv_temp,state_space,
                        s_curr,s_next,observation,aug_state_space):
    """
    P( x_i, \delta w_i | v_i, l_i )
    =
    P( x_i | v_i ) * P( \delta w_i | v_i, l_i )
    """

    x,times = observation
    time_c,time_n = times
    # s_next not used; kept for compatability with the smjpwrapper
    v_curr,l_curr = aug_state_space[s_curr]
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
        likelihood_x = compute_likelihood_obs(x,p_x,state_space,v_curr,inv_temp)
    
    """
    The issue now is that we've got to compute the integral after equation 3
    but this integral requires knowing how long we've held our current state value.
    
    For example,

    say at time 1 we jump to state 3. Say the grid includes 1.1. Then at 1.1 if we hold
    we went from holding 0 sec to .1 sec. Say the grid includes 1.5. Then at 1.1 if we hold 
    then we move from holding .1 to .5 sec. The information about .1 is contained in the pdf
    in the "L" values "time hold". But since l_curr says: "time of jump", we need a way to 
    include the delta_w term as the input

    The integral is not shift-invariant (duh) so we really need to original "l_i" term,
    or hold time term from the previous state.

    delta_w_i is not a problem to give as an observation

    the previous hold time is the challenging part...

    "l_i" (from pdf) = time_c - l_curr?

    """
    
    
    if not np.isclose(final_grid_time,time_n):
        likelihood_delta_w = p_w.l([t_hold_c,t_hold],v_curr)
    else: # the final event is not a poisson process
        likelihood_delta_w = p_w.l([t_hold_c,t_hold],v_curr,is_poisson_event=False)
    likelihood = likelihood_x * likelihood_delta_w
    log_likelihood = np.ma.log([likelihood]).filled(-np.infty)[0]
    # print("Pw",likelihood_delta_w)
    # print("Px",likelihood_x)
    # print("Pobs",likelihood)

    # print('LL',log_likelihood)
    return log_likelihood

def smjp_emission_sampler(s_curr,s_next,observation,state_space,d_create):
    # s_next,observation not used; kept for compatability with the smjpwrapper
    distribution = d_create(s_curr)
    sampled_state = distribution.s(1)
    return sampled_state

def smjp_emission_likelihood_nosmjpwrapper(d_create,s_curr,obs_emission):
    distribution = d_create(s_curr)
    likelihood = distribution.l(obs_emission)
    return likelihood

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
                 observation_index=0,sampler=None,obs_is_iter=False):
        self.state_function = state_function
        self.state_space = state_space
        self.observation = None
        # its always just delta_t... I was originally confused during implementation.
        # Todo: remove the "observation_index" and replace with more explicit delta_t or.
        # dependance of more generally (t_curr,t_next).
        self.observation_is_iterable = obs_is_iter
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
                sample = self.hazard_function.sample( n = 1 )[current_state,state]
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
        # print('samples',samples)
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


        # --> version 2: normalized across current_state <---
        # intvl = interval
        # front_term = 0
        # log_exp_term = 0
        # pp_ll = np.zeros(len(self.state_space))
        # for index_c,state_c in enumerate(self.state_space):
        #     for state_n in self.state_space:
        #         front_term += self.hazard_function(tau)[state_c,state_n]
        #         log_exp_term += self.mean_function(intvl[0],intvl[1],state_c,state_n)
        #     ll = np_log(front_term) - log_exp_term
        #     pp_ll[index_c] = ll
        # pp_ll -= logsumexp(pp_ll)
        # state_select = self.state_space.index(current_state)
        # return np.exp(pp_ll[state_select])
        # --> version 1: not normalized across current_state <---
        front_term = 0
        log_exp_term = 0
        for next_state in self.state_space:
            front_term += self.hazard_function(tau)[current_state,next_state]
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
def smjp_hazard_sampler_unset(state_space,h_create,hold_time,current_state,next_state,n=1):
    if hold_time is None: # sampling over hold_time "\tau" given (current_state,next_state)
        sample = h_create(current_state,next_state).sample(n)
        return sample
    else: # sampling over next state given (hold_time, current_state)
        assert next_state is None,"next state must be None"
        state_rate = []
        for next_state in state_space:
            rate = h_create(current_state,next_state).l(hold_time)
            state_rate.append(rate)
        state_rate /= np.sum(state_rate)
        # could also do "argmax"
        state_index_l = np.where(npr.multinomial(n,state_rate,n) == 1)[0]
        sampled_state = []
        for state_index in state_index_l:
            sampled_state += [state_space[state_index]]
        return sampled_state

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

def sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, t_end, t_start = 0):
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
            hold_time_samples += [ hazard_A.sample()[v_curr,state] ]

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
    emission = sMJPWrapper(smjp_e,augmented_state_space,obs_is_iter=True)

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
    pi = sMJPWrapper(smjp_transition,augmented_state_space,hazard_A,hazard_B,obs_is_iter=True)

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

def sample_data_posterior(V,T,state_space,emission_sampler,obs_times):
    # -- sample via emission probability @ fixed times --
    ss = len(state_space)
    data_states = []
    data_times = []
    for obs_time in obs_times:
        time_c = T[0]
        for state,time_n in zip(V[:-1],T[1:]):
            if time_c <= obs_time < time_n:
                data_states += [emission_sampler(None)[state]] 
                data_times += [ obs_time ]
                break
            time_c = time_n
    data = sMJPDataWrapper(data=data_states,time=data_times)
    return data

def create_toy_data(state_space,time_length,number_of_observations,emission_sampler):
    gen_state_prob = np.ones(len(state_space)) / len(state_space) # all states equally likely
    states_index = np.where(npr.multinomial(1,gen_state_prob,number_of_observations) == 1)[1]
    data = []
    for state_index in states_index:
        state = state_space[state_index]
        # this is pretty un-necessary here since its 100 v 1
        data += [emission_sampler(None)[state]] 
    return data
    
def smjp_emission_multinomial_create_unset(state_space,state_curr):
    mn_probs = np.ones(len(state_space))
    state_curr_index = state_space.index(state_curr)
    mn_probs[state_curr_index] = 1
    mn_probs /= np.sum(mn_probs)
    distribution = Multinomial({'prob_vector':mn_probs,'translation':state_space})
    return distribution
    
    

if __name__ == "__main__":
    print("HI")
    
