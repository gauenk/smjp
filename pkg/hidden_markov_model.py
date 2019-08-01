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
        self.path_length = kwargs['path_length']
        self.sample_dimension = kwargs['sample_dimension']
        self.current_path = [None for _ in range(self.path_length)]
        assert (len(self.data) == self.path_length), \
            "the data should be the same size as the inferred path length"

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

    def backward_sampling(self,num_of_samples = 1):
        """
        think: "probabilistic decoding". Instead of taking the "max", we sample according
        to the probability of each state.
        """
        alphas,output_prob = self.likelihood()
        kwargs = {'num_of_samples': num_of_samples,
                  'num_of_states': len(self.state_alphabet),
                  'num_of_steps': self.path_length,
                  'sample_dimension': self.sample_dimension,
                  'alphas': alphas,
                  'pi': self.transition,
        }
        samples = backward_sampling_hmm([],**kwargs)
        return samples

    def _run_likelihood_and_decoding_hmm(self):
        kwargs = {'num_of_states': len(self.state_alphabet),
                  'num_of_steps': self.path_length,
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
    num_of_samples = kwargs['num_of_samples'] # fix @ one for now.
    num_of_states = kwargs['num_of_states']
    num_of_steps = kwargs['num_of_steps']
    sample_dimension = kwargs['sample_dimension'] # not used in this B.S. function
    unnormalized_alphas = kwargs['alphas']
    pi = kwargs['pi'] # state transition; "parameters" of HMM (rows x cols ~ origin x destination)
    alphas = unnormalized_alphas / np.sum(unnormalized_alphas,axis=0)
    samples = np.zeros((num_of_samples,num_of_steps),dtype=np.int)

    # init; sample from P(q_T | data_{1:T} )
    mn_prob = alphas[-1,:] / np.sum(alphas[-1,:])
    samples[:,-1] = np.where(npr.multinomial(num_of_samples,mn_prob) == 1)[0][0]

    # run loop for (T-1, 1) out of (T,1); e.g. T is not included;
    # note: the range() is "-2" since python is zero indexed.
    for index in reversed(range(0,num_of_steps-2)):
        mn_prob = alphas[index,:] / np.sum(alphas[index,:]) # p( q_t | data_{1:t} )
        # print(mn_prob)
        mn_prob = mn_prob * pi[:,samples[0,index+1]]  #  p( q_t | q_{t+1} )
        # print(mn_prob)
        samples[:,index] = np.where(npr.multinomial(num_of_samples,mn_prob) == 1)[0][0]

    return samples

def likelihood_and_decoding_hmm(*args,**kwargs):
    """
    Forward algorithm & Viterbi algorithm;
    """
    num_of_states = kwargs['num_of_states']
    num_of_steps = kwargs['num_of_steps']
    alphas = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    viterbi = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    backpointers = np.zeros((num_of_steps),dtype=np.float)
    init_probs = kwargs['init_probs'] # distribution over initial states
    pi = kwargs['pi'] # state transition; "parameters" HMM (rows x cols ~ origin x destination)
    e = kwargs['e'] # list of functions size (num_of_states); likelihood of obs from each state
    O = kwargs['O'] # list of observations size (num_of_steps)

    # init forward pass
    for state_idx in range(num_of_states):
        alphas[0,state_idx] = init_probs.l(state_idx) * e(O[0])[state_idx]
        viterbi[0,state_idx] = init_probs.l(state_idx) * e(O[0])[state_idx]
    
    # run the forward
    # print("---")
    for step in range(1,num_of_steps):
        for state_idx in range(num_of_states):
            alphas[step,state_idx] = 0
            for state_prime_idx in range(num_of_states):
                # print(alphas[step-1,state_prime_idx],
                #       pi[state_prime_idx,state_idx],
                #       e[state_idx][O[step]])
                # HMM: pi(O_t) = pi; does not depend on obs
                # sMJP: pi(O_t) =/= pi; depends on obs
                alphas[step,state_idx] += alphas[step-1,state_prime_idx] *\
                                          pi(O[step])[state_prime_idx,state_idx] *\
                                          e(O[step])[state_idx]
        # print(step,alphas[step,:])
    # backpath = trace_backward_path(backpointers)
    backpath = None
    output_prob = np.sum(alphas[-1,:])
    return alphas,output_prob,viterbi,backpath


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
        

    
