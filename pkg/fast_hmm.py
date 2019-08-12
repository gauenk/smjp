import numpy as np
import numpy.random as npr
from numpy import transpose as npt


class fast_HiddenMarkovModel(object):
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
                  'e': self.emission,
                  'O': self.data,
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

def compute_beta_term(pi,beta_next,e,obs,time_c,time_n):
    states = pi.state_space
    ss_size = len(states)
    betas = np.zeros(ss_size)
    for state_n in range(ss_size):
        likelihood_data = np.ma.log([e(obs)[state_n]]).filled(-np.infty)
        # beta_n = np.ma.log(beta_next[state_n]).filled(-np.infty)
        beta_n = beta_next[state_n] # our beta terms are already log
        if np.isinf(likelihood_data) or np.isinf(beta_n):
            continue
        for state_c in range(ss_size):
            if is_pi_zero(time_c,time_n,state_c,state_n,states):
                continue
            transition = pi([time_c,time_n])[state_c,state_n]
            # print('comp_betas',transition,likelihood_data, beta_n)
            betas[state_c] += np.exp(transition + likelihood_data + beta_n)
    betas = np.ma.log(betas).filled(-np.infty)
    return betas

def compute_transition_vector(pi,state_n,time_c,time_n):
    states = pi.state_space
    ss_size = len(states)
    transition = np.ones(ss_size) * (-np.infty)
    for state_c in range(ss_size):
        if is_pi_zero(time_c,time_n,state_c,state_n,states):
            continue
        transition[state_c] = pi([time_c,time_n])[state_c,state_n]
    return transition

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
    O = kwargs['O']
    e = kwargs['e']
    pi = kwargs['pi'] # state transition; "params" of HMM (rows x cols ~ origin x destination)
    log_alphas = unnormalized_alphas
    betas = np.zeros(log_alphas.shape)
    # alphas = unnormalized_alphas / np.sum(unnormalized_alphas,axis=0)
    samples = np.zeros((num_of_samples,num_of_steps),dtype=np.int)

    # init; sample from P(q_T | data_{1:T} )
    sampling_prob = np.exp(log_alphas[-1,:]) / np.sum(np.exp(log_alphas[-1,:]))
    samples[:,-1] = np.where(npr.multinomial(num_of_samples,sampling_prob) == 1)[0][0]
    betas[:,-1] = np.zeros(len(betas[:,-1])) # log_betas plz (1 -> 0)

    index_iter = reversed(range(1,len(time_grid)))
    time_iter = reversed(time_grid[:-1])
    i = 0
    iterable = [[i,j] for i,j in enumerate(time_grid[:-1])]
    # print('time_grid',time_grid)
    # print('(sample,time_current):',pi.state_space[samples[0,-1]],time_grid[-1])
    for time_index_current,time_current in reversed(iterable):
        i += 1
        time_index_next = time_index_current + 1
        time_next = time_grid[time_index_next]
        obs = [O[time_current,time_next],time_next]

        # --> sample_{t+1} [get the previous sample index] <--
        next_sample = samples[:,time_index_next][0]

        # --> compute beta terms (we might need these) <--
        beta_term_args = [pi,betas[time_index_next,:],e,obs,time_current,time_next]
        betas[time_index_current,:] = compute_beta_term(*beta_term_args)
        beta_n = betas[time_index_next,next_sample]

        # --> alpha_{t+1}(next_sample), a scalar <--
        alpha_at_next = log_alphas[time_index_next,:]
        alpha_n = alpha_at_next[next_sample]

        # --> alpha_t [not normalized] <--
        alpha_at_current = log_alphas[time_index_current,:]
        alpha_c = alpha_at_current

        # --> transition vector <--
        transition = compute_transition_vector(pi,next_sample,time_current,time_next)
        
        # --> likelihood of data, a scalar <--
        likelihood_data = np.ma.log([e(obs)[next_sample]]).filled(-np.infty)

        # altogether.
        # print(alpha_c, transition,likelihood_data,beta_n,alpha_n)
        # sampling_prob = (alpha_c * transition) * (likelihood_data) * (beta_n / alpha_n)
        sampling_prob = np.exp(alpha_c + transition + likelihood_data + beta_n - alpha_n)
        sampling_prob /= np.sum(sampling_prob)
        # print('time_current',time_current)
        # print('samp_prob',sampling_prob)

        time_current_p,time_next_p = round(time_current,3),round(time_next,3)
        time_str = "({},{})".format(time_current_p,time_next_p)

        s = np.where(npr.multinomial(num_of_samples,sampling_prob) == 1)[0][0]
        samples[:,time_index_current] = s
        # print('(sample,time_current):',pi.state_space[s],time_current)

    
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
    num_of_steps = len(time_grid) # |W|
    log_alphas = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    log_viterbi = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    backpointers = np.zeros((num_of_steps),dtype=np.float)
    init_probs = kwargs['init_probs'] # distribution over initial states
    pi = kwargs['pi'] # state transition; "parameters" HMM (rows x cols ~ origin x destination)
    e = kwargs['e'] # list of functions size (num_of_states); likelihood of obs from each state
    O = kwargs['O'] # list of observations size (num_of_steps)
    states = pi.state_space

    # init forward pass
    time_next = time_grid[0] # == 0 (or t_start)
    for state_idx in range(num_of_states):
        obs = [O[0,time_next],time_next]
        ll_init =  np.ma.log([init_probs.l(state_idx)]).filled(-np.infty)
        ll_obs = np.ma.log([e(obs)[state_idx]]).filled(-np.infty)
        log_alphas[0,state_idx] = ll_init + ll_obs
        log_viterbi[0,state_idx] = ll_init + ll_obs
    
    print("|W| = {}".format(len(time_grid)))
    for time_index,time_c in enumerate(time_grid[0:-1]): # {w_0,...,w_{|W|-1}}
        alpha_index = time_index + 1
        time_n = time_grid[alpha_index] # update the next time.
        obs = [O[time_c,time_n],time_n]
        for state_n in range(num_of_states):
            log_obs = np.ma.log([e(obs)[state_n]]).filled(-np.infty)[0]
            for state_c in range(num_of_states):
                if is_pi_zero(time_c,time_n,state_c,state_n,states):
                    continue
                log_alpha = log_alphas[alpha_index-1,state_c]
                log_transition = pi([time_c,time_n])[state_c,state_n]
                alpha_new = np.exp(log_alpha + log_transition + log_obs)
                log_alphas[alpha_index,state_n] += alpha_new
        log_alphas[alpha_index,:] = np.ma.log([log_alphas[alpha_index,:]]).filled(-np.infty)
    backpath = None
    output_prob = np.sum(np.exp(log_alphas[-1,:]))
    return log_alphas,output_prob,log_viterbi,backpath

def is_pi_zero(time_c,time_n,state_c,state_n,states):
    v_c,l_c = states[state_c] # (state,time_of_jump)
    v_n,l_n = states[state_n]
    if (l_c > l_n): 
        return True
    if (l_n <= time_c) and (state_c != state_n):
        return True
    if (l_n < time_n) and (state_c != state_n):
        return True
    if (l_n > time_n):
        return True
    if (time_n < l_c):
        return True
    return False
    
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
        

