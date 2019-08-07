import numpy as np
import numpy.random as npr
from numpy import transpose as npt


class HiddenMarkovModel(object):
    """
    Finite and Discrete Time & State
    """
    def __init__(self,*args,**kwargs):
        self.emission = kwargs['emission']
        self.transition = kwargs['transition']
        self.data = kwargs['data']
        self.state_alphabet = kwargs['state_alphabet'] # we currently only use the alphabet size.
        self.pi_0 = kwargs['pi_0'] # initial distribution
        self.time_grid = kwargs['time_grid']
        self.sample_dimension = kwargs['sample_dimension']

    def likelihood(self):
        """
        aka "forward" or P(data | HMM_params)
        """
        alphas,output_prob,viterbi,backpath = self._run_likelihood_and_decoding_hmm()
        return alphas,output_prob
        
    def decoding(self):
        """
        aka "backward" or Viterbi or Find *best* path, Q ~ P(Q | data, HMM_params)
        """
        alphas,output_prob,viterbi,backpath = self._run_likelihood_and_decoding_hmm()
        return viterbi,backpath

    def learning(self):
        """
        aka "forward-backward" or Baum-Welch or HMM_params ~ P(HMM_params | data, state_alphabet )
        """
        pass

    def backward_sampling(self,num_of_samples = 1,alphas=None):
        """
        think: "probabilistic decoding". Instead of taking the "max", we sample according
        to the probability of each state.
        """
        if alphas is None:
            alphas,output_prob = self.likelihood()
        kwargs = {'num_of_samples': num_of_samples,
                  'num_of_states': len(self.state_alphabet),
                  'time_grid': self.time_grid,
                  'sample_dimension': self.sample_dimension,
                  'alphas': alphas,
                  'pi': self.transition,
        }
        samples,t_samples = backward_sampling_hmm([],**kwargs)
        return samples,t_samples

    def _run_likelihood_and_decoding_hmm(self):
        kwargs = {'num_of_states': len(self.state_alphabet),
                  'time_grid': self.time_grid,
                  'init_probs': self.pi_0,
                  'pi': self.transition,
                  'e': self.emission,
                  'O': self.data,
        }
        alphas,output_prob,viterbi,backpath = likelihood_and_decoding_hmm([],**kwargs)
        return alphas,output_prob,viterbi,backpath

def backward_sampling_hmm(*args,**kwargs):
    """
    sampling a trajectory from T to 1 given the alphas from "forward"
    """
    num_of_samples = 1 #kwargs['num_of_samples'] # fix @ one for now.
    num_of_states = kwargs['num_of_states']
    time_grid = kwargs['time_grid']
    num_of_steps = len(time_grid) # |W| - 1
    sample_dimension = kwargs['sample_dimension'] # not used in this B.S. function
    unnormalized_alphas = kwargs['alphas']
    pi = kwargs['pi'] # state transition; "params" of HMM (rows x cols ~ origin x destination)
    log_alphas = unnormalized_alphas
    # alphas = unnormalized_alphas / np.sum(unnormalized_alphas,axis=0)
    samples = np.zeros((num_of_samples,num_of_steps),dtype=np.int)

    # init; sample from P(q_T | data_{1:T} )
    mn_prob = np.exp(log_alphas[-1,:]) / np.sum(np.exp(log_alphas[-1,:]))
    samples[:,-1] = np.where(npr.multinomial(num_of_samples,mn_prob) == 1)[0][0]

    # run loop for (T-1, 1) out of (T,1); e.g. T is not included;
    # note: the range() is "-2" since python is zero indexed.
    index_iter = reversed(range(1,len(time_grid)))
    time_iter = reversed(time_grid[:-1])
    #print([i for i in index_iter])
    # print(time_grid)
    i = 0
    for time_index_future,time_present in zip(index_iter,time_iter):
        i += 1
        time_index_present = time_index_future - 1
        time_future = time_grid[time_index_future]
        delta_w = time_future - time_present

        # print(delta_w,time_future,time_present)
        alpha_at_future = np.exp(log_alphas[time_index_future-1,:])
        # p( q_t | data_{1:t} ) \propto p( q_t, data_{1:t} )
        prob_likelihood = alpha_at_future / np.sum(alpha_at_future)

        #  p( q_t | q_{t+1} ) for each (delta_w,state_current)
        prob_transition = np.zeros(len(pi.state_space))
        #delta_w_enumeration = create_delta_w_enumeration(time_present,time_grid)
        future_sample = samples[:,time_index_future][0]
        print(delta_w,pi.state_space[future_sample])
        print('pl',prob_likelihood)
        for state_index in range(len(pi.state_space)):
            l_trans = pi(delta_w)[state_index,future_sample]
            prob_transition[state_index] += np.exp(l_trans)
            #for delta_w in delta_w_enumeration:
        # explainatory names
        print('pt',prob_transition)
        prob_state_given_future_state = prob_transition
        prob_state_given_data = prob_likelihood
        mn_prob = prob_state_given_data * prob_state_given_future_state
        if np.all(np.isclose(mn_prob,0)):
            print("==> ~~mn_prob is zero~~ <==")
            print('i',i)
            print("future_state",pi.state_space[future_sample])
            print('s',samples)
            print("tlen",len(time_grid))
            print('ti',time_index_present,time_present)
            print('pl',prob_state_given_data)
            print('pt',prob_state_given_future_state)
            exit()
        mn_prob /= np.sum(mn_prob)
        print('mn',mn_prob)

        s = np.where(npr.multinomial(num_of_samples,mn_prob) == 1)[0][0]
        samples[:,time_index_present] = s
    
    # translate the sample back
    t_samples = [None for _ in range(len(samples))]
    for index,sample_index in enumerate(samples):
        t_samples[index] = pi.state_space[sample_index]

    return samples,t_samples

def create_delta_w_enumeration(time_next,time_grid):
    enum = [0]
    for time in time_grid:
        diff = time_next - time
        if diff <= 0:
            break
        enum += [ time_next - time ]
    return enum

def likelihood_and_decoding_hmm(*args,**kwargs):
    """
    Forward algorithm & Viterbi algorithm;
    """
    num_of_states = kwargs['num_of_states']
    time_grid = kwargs['time_grid']
    num_of_steps = len(time_grid) # |W| - 1
    log_alphas = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    log_viterbi = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    backpointers = np.zeros((num_of_steps),dtype=np.float)
    init_probs = kwargs['init_probs'] # distribution over initial states
    pi = kwargs['pi'] # state transition; "parameters" HMM (rows x cols ~ origin x destination)
    e = kwargs['e'] # list of functions size (num_of_states); likelihood of obs from each state
    O = kwargs['O'] # list of observations size (num_of_steps)

    # init forward pass
    time_next = time_grid[0] # the first point in the time_grid, |W|, is _not_ zero in general.
    # but it is zero currently.... not great.
    for state_idx in range(num_of_states):
        obs = [O[0,time_next],time_next-0]
        log_alphas[0,state_idx] = np.ma.log([init_probs.l(state_idx)]).filled(-np.infty) +\
                                  np.ma.log([e(obs)[state_idx]]).filled(-np.infty)
        log_viterbi[0,state_idx] = np.ma.log([init_probs.l(state_idx)]).filled(-np.infty) +\
                                   np.ma.log([e(obs)[state_idx]]).filled(-np.infty)
    
    """
    The augmented state space is actually the issue; we don't need to iterate over delta_w_enum
    What we actually need is for the augmented state space to _include_ all the different
    enumerations of delta_w. E.g. NOT [STATE,GRID] rather [STATE,DELTA_W]. 
    I do ~not~ think this should impact "smjp_transition()", but let's watch out.
    """
    # run the forward
    # print("---")
    index_range = range(1,len(time_grid))
    print(len(index_range),len(time_grid[1:-1]),len(time_grid[0:-1]),len(time_grid))
    for alpha_index,time in zip(index_range,time_grid[0:-1]): # {w_0,...,w_{|W|-1}}
        # print(alpha_index)
        time_next = time_grid[alpha_index] # update the next time.
        delta_w = time_next - time 
        delta_w_l = [delta_w]
        # delta_w_enumeration = create_delta_w_enumeration(time_next,time_grid)
        for delta_w in delta_w_l: # todo: remove this loop
            obs = [O[time,time_next],delta_w]
            for state_idx in range(num_of_states):
                for state_prime_idx in range(num_of_states):
                    #print(pi(delta_w)[state_prime_idx,state_idx],e(obs)[state_idx])
                    # HMM: pi(t,t_next) = pi; does not depend on time difference
                    # sMJP: pi(t,t_next) =/= pi; depends on time difference
                    log_alpha = log_alphas[alpha_index-1,state_prime_idx]
                    log_transition = pi(delta_w)[state_prime_idx,state_idx]
                    log_obs = np.ma.log([e(obs)[state_idx]]).filled(-np.infty)[0]
                    #print(log_alpha,log_transition,log_obs)
                    alpha_new = np.exp(log_alpha + log_transition + log_obs)
                    #print(alpha_new)
                    if state_prime_idx == 0 and state_idx == 1 and alpha_index > 2 and False:
                        print("[[state_prime_idx == 0 and state_idx == 1]]")
                        print(log_alpha,log_transition,log_obs)
                        print(log_alphas[alpha_index-1,:])
                        print(log_alphas[alpha_index-1,state_prime_idx])
                        print('dw',delta_w,time,time_next)
                        print(pi.state_space[state_prime_idx])
                        print(pi.state_space[state_idx])
                        print("==========================")
                        exit()
                    if do_we_thin(pi.state_space,state_prime_idx,state_idx,delta_w) and False:
                        print(pi.state_space)
                        print(alpha_index-1)
                        print(log_alphas[alpha_index-1,:])
                        print(state_prime_idx,state_idx)
                        print(log_alpha)
                        print("THN")
                        v,l = pi.state_space[state_prime_idx]
                        vp,lp = pi.state_space[state_idx]
                        print(log_alphas[alpha_index-1,state_prime_idx])
                        print(delta_w)
                        print(pi.state_space[state_prime_idx])
                        print(pi.state_space[state_idx])
                    if np.any(np.isnan(alpha_new)):
                        print("found a nan")
                        quit()
                    #print('alpha_new',alpha_new,log_alpha,state_idx)
                    log_alphas[alpha_index,state_idx] += alpha_new
        log_alphas[alpha_index,:] = np.ma.log([log_alphas[alpha_index,:]]).filled(-np.infty)
        # print(log_alphas)
        # exit()
        # print(alpha_index,time)
        # print(time,alphas[time,:])
    backpath = None # trace_backward_path(backpointers)
       
    # print(log_alphas)
    # print(len(index_range),len(time_grid[1:-1]),len(time_grid[0:-1]),len(time_grid))
    # print(time_grid)
    # exit()
    backpath = None
    output_prob = np.sum(np.exp(log_alphas[-1,:]))
    
    return log_alphas,output_prob,log_viterbi,backpath


def do_we_thin(state_space,sc_idx,sn_idx,delta_w):
    v_curr,l_curr = state_space[sc_idx]
    v_next,l_next = state_space[sn_idx]
    t_hold = delta_w + l_curr
    if (l_next == t_hold) and (v_next == v_curr):
        return True
    else:
        return False

class HMMWrapper(object):
    """
    A wrapper for HMM information:
    - state transition probabilities
    - emission probabilities

    In the FFBS algs we use pi(observation_t)[current_state][next_state].
    The observation information is ~too much man~ for some HMM parameters (e.g. transition). 
    This wrapper handles using/throwing away the information neatly.
    """
    def __init__(self,state_info_array,use_observation):
        self.state_info_array = state_info_array
        self.observation = None
        self.use_observation = use_observation

    def __call__(self,observation):
        self.observation = observation
        return self

    def __getitem__(self,a_slice):
        val = None
        if self.use_observation:
            val = self.state_info_array[a_slice][self.observation]
        else:
            val = self.state_info_array[a_slice]
        self.observation = None # ensure we are required to get another obs before __getitem__
        return val
        

