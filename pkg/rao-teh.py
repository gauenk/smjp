from .hidden_markov_model import HiddenMarkovModel


class C(object):

    def __init__(self,functions):
        self.functions = functions

    def __getitem__(self,val):
        print(val)

class HMM2d(HiddenMarkovModel):

    def __init__(self,*args,**kwargs):
        HiddenMarkovModel.__init__(*args,**kwargs)

    def _run_likelihood_and_decoding_hmm(self):
        # change to 2d
        kwargs = {'num_of_states': len(self.state_alphabet),
                  'num_of_steps': self.path_length,
                  'init_probs': self.pi_0,
                  'A': self.transition,
                  'B': self.emission,
                  'O': self.data,
        }
        alphas,output_prob,viterbi,backpath = likelihood_and_decoding_hmm_2d([],**kwargs)
        return alphas,output_prob,viterbi,backpath

    def backward_sampling(self,num_of_samples = 1):
        """
        change to 2d.

        think: "probabilistic decoding". Instead of taking the "max", we sample according
        to the probability of each state.
        """
        alphas,output_prob = self.likelihood()
        kwargs = {'num_of_samples': num_of_samples,
                  'num_of_states': len(self.state_alphabet),
                  'num_of_steps': self.path_length,
                  'sample_dimension': self.sample_dimension,
                  'alphas': alphas,
                  'A': self.transition,
        }
        samples = backward_sampling_hmm_2d([],**kwargs)
        return samples
    

def likelihood_and_decoding_hmm_2d(*args,**kwargs):
    """
    Forward algorithm & Viterbi algorithm;
    """
    num_of_states = kwargs['num_of_states']
    num_of_steps = kwargs['num_of_steps']
    alphas = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    viterbi = np.zeros((num_of_steps,num_of_states),dtype=np.float)
    backpointers = np.zeros((num_of_steps),dtype=np.float)
    init_probs = kwargs['init_probs'] # distribution over initial states
    A = kwargs['A'] # state transition; "parameters" of HMM (rows x cols ~ origin x destination)
    B = kwargs['B'] # list of functions size (num_of_states); likelihood of obs from each state
    O = kwargs['O'] # list of observations size (num_of_steps)

    # init forward pass
    for state_idx in range(num_of_states):
        alphas[0,state_idx] = init_probs.l(state_idx) * B[state_idx][O[0]]
        viterbi[0,state_idx] = init_probs.l(state_idx) * B[state_idx][O[0]]
    
    # run the forward
    # print("---")
    for step in range(1,num_of_steps):
        for state_idx in range(num_of_states):
            alphas[step,state_idx] = 0
            for state_prime_idx in range(num_of_states):
                # print(alphas[step-1,state_prime_idx],
                #       A[state_prime_idx,state_idx],
                #       B[state_idx][O[step]])
                alphas[step,state_idx] += alphas[step-1,state_prime_idx] *\
                                          A[state_prime_idx,state_idx] * B[state_idx][O[step]]
        # print(step,alphas[step,:])
    backpath = trace_backward_path(backpointers)
    output_prob = np.sum(alphas[-1,:])
    return alphas,output_prob,viterbi,backpath

def backward_sampling_hmm_2d(*args,**kwargs):
    """
    sampling a trajectory from T to 1 given the alphas from "forward"
    now in 2 dimensions
    """
    num_of_samples = kwargs['num_of_samples'] # fix @ one for now.
    num_of_states = kwargs['num_of_states']
    num_of_steps = kwargs['num_of_steps']
    sample_dimension = kwargs['sample_dimension'] # not used in this B.S. function
    unnormalized_alphas = kwargs['alphas']
    A = kwargs['A'] # state transition; "parameters" of HMM (rows x cols ~ origin x destination)
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
        mn_prob = mn_prob * A[:,samples[0,index+1]]  #  p( q_t | q_{t+1} )
        # print(mn_prob)
        samples[:,index] = np.where(npr.multinomial(num_of_samples,mn_prob) == 1)[0][0]

    return samples




def sample_smjp_trajectory(**kwargs):
    """
    ~~ Implementation of Algorithm 1 ~~
    ** does not include observations **
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
        t_hold = B.s_cond(l_curr)
        delta_w = t_hold - l_curr
        w_next = w_curr + delta_w
        accept_prob = A.l(v_curr,t_hold) / B.l(v_curr,t_hold)
        coin_flip = uniform.s()
        if(accept_prob > coin_flip):
            l_next = 0
            v_next = v_distribution.s() # what is the sampler for "v"?
        else:
            l_next = l_curr + delta_w
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



