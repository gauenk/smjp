import numpy as np
import numpy.random as npr
from pkg.utils import *
from numpy import transpose as npt
from pkg.smjp_utils import sample_smjp_trajectory_prior
from pkg.mcmc_utils import *

def pmcmc(number_of_particles,number_of_samples,save_iter,states,
          hazard_A,emission,time_final,
          data,pi_0,uuid_str,omega,filename,load_file):

    # load samples from file if we want
    if load_file:
        aggregate,_,_,uuid_str = load_samples_in_pickle(filename)
        return aggregate,uuid_str

    # smc input
    smc_input = {
        'pi_0' : pi_0,
        'time_final': time_final,
        'N': number_of_particles,
        'data': data,
        'A' : hazard_A,
        'emission' : emission,
        'states' : states,
        'uuid' : uuid_str,
    }

    # print info to user
    print("----> Running pMCMC Algorithm <----")
    print("number of particles: {}".format(number_of_particles))
    print("number of samples: {}".format(number_of_samples))

    # print init sample
    print("--------------")
    p_u_str = '{}_init'.format(uuid_str)
    V,T,prob = smjp_smc(**smc_input)
    print("--- Likelihood {} ---".format(prob))
    print(np.c_[V,T])
    print("--------------")

    # create collection bins
    aggregate = {'V':[],'T':[],'prob':[]}
    print_iter = 300

    # run MH mcmc with proposals from SMC
    for i in range(number_of_samples):
        # regular,old mh-mcmc
        V_prop,T_prop,prob_prop = smjp_smc(**smc_input)
        coin_flip = np.log(npr.uniform(0,1,1))
        if coin_flip > ( prob_prop / prob ):
            V,T,prob = V_prop,T_prop,prob_prop

        # -- print summary & status --
        p_u_str = '{}_{}'.format(uuid_str,i)
        if i % print_iter == 0:
            print("[{} / {}] samples".format(i,number_of_samples))
            save_samples_in_pickle(aggregate,None,None,uuid_str,i)

        # -- collection samples --
        aggregate['V'].append(V)
        aggregate['T'].append(T)
        aggregate['prob'].append(prob)
    save_samples_in_pickle(aggregate,None,None,uuid_str)

    return aggregate,uuid_str

def sample_smjp_smc(data,state_space,hazard_A,hazard_B,smjp_e,t_end,u_str):
        
    path_states,path_times,prob = smjp_smc(*args,**kwargs)
    return path_states,path_times,prob
    
def sample_smjp_smc_v2(data,state_space,hazard_A,hazard_B,smjp_e,t_end,u_str):

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
    # Run SMC
    #
    # ----------------------------------------------------------------------

    smc_input = {'emission': emission,
                   'transition': pi,
                   'data': data,
                   'state_alphabet': augmented_state_space,
                   'pi_0': pi_0,
                   'time_grid': W,
                   'uuid_str':u_str,
                   'q': hazard_B,
                   'N': 10, # hyperparameter
                   'T': len(W)-1,
    }
    
    sample = smjp_smc(**smc_input)
    print("smjp smc sample")
    print(sample)
    exit()
    return sample

class sMJPSMCHazardWrapper(object):

    def __init__(self,hazard,pi,pi_0,e,state_space):
        self.hazard = hazard
        self.pi = pi
        self.pi_0
        self.e = e
        self.state_space = state_space

    def sample(self,time_c,time_n,V_A,N):
        if time_n is None: # init sample [[ 0,t[0],None,N ]]
            samples = self.pi_0.sample(N)
        else: # [[ t_c, t_n, V_A, N ]]
            prior_sampler_input = [
                self.hazard, self.pi_0, self.state_space, time_n, time_c
            ]
            samples = []
            for i in range(N):
                v_curr,w_curr = V_A[i],time_c
                while(True):
                    # sample all holding times for the next state
                    hold_time_samples = []
                    for state in state_space:
                        hold_time_samples += [ hazard_A.sample()[v_curr,state] ]

                    # take the smallest hold-time and move to that state
                    t_hold = np.min(hold_time_samples)
                    v_next = state_space[np.argmin(hold_time_samples)]
                    w_next = w_curr + t_hold
                    
                    # check with observation time
                    if t_next < w_next: # we are beyond the observation
                        log_obs_lik = self.e(t_hold)
                    
                    # append values to the vectors
                    v += [v_next]
                    w += [w_next]

                    # update
                    v_curr = v_next
                    w_curr = w_next
                samples.append([v,w])
        return samples
    
    def l(self,obs,t_c,t_n,V_c,V_n):
        return self.likelihood(obs,t_c,t_n,V_c,V_n)

    def likelihood(self,obs,t_c,t_n,V_c,V_n):
        """
        TODO: resolve issues via augmented state space vs original state space.

        identify the role of augmented_state_space vs
        original_state_space for the p v.s. q in SMC...

        like how do we know tau? Are we running SMC over the 
        augmented state space? Would that make sense?
        
        -> I am thinking yes since right now we have each state
        specificied for an ordinary HMM.

        -> By reducing a sMJP to a HMM with a growing state space,
        we have the orignal algorithm function just fine.

        -> However, by allowing this we may need to smarter with the sampling
        of V and A_x....

        """
        ll = 0
        for i in range(N):
            state_c = V_c[i]
            state_n = V_n[i]
            log_obs = self.e(obs)[state_n]
            log_trans = self.pi([t_c,t_n])[state_c,state_n]
            ll += log_trans + log_obs 
        return np.exp(ll)

def smjp_smc(*args,**kwargs):
    init_probs = kwargs['pi_0'] # distribution over initial states
    t_final = kwargs['time_final']
    N = kwargs['N'] # number of particles
    Obs = kwargs['data'] # observations
    y = Obs.data
    t = [0] + Obs.time # includes "0" by convention. required for proper looping below.
    D = len(y)
    hazard_A = kwargs['A'] # target distribution (hazard_A; assume we can't sample hazard_A)
    emission = kwargs['emission']
    states = kwargs['states']
    u_str = kwargs['uuid']

    """
    the state space used in SMC for sMJP
    is the augmented state space that grows with # of (( observations / time. )) ??
    -> which one is actually more correct? say  |O| >> |T| or |T| >> |O|. what happens??
    """
    
    # first sample
    _V = init_probs.sample(N)
    V = [ [_V[i]] for i in range(N) ]
    T = [ [0] for i in range(N) ]
    Z = np.zeros((D))
    W = np.zeros((N,D))
    A = []

    # ---------------------------------
    # particle filtering for smjp
    # ---------------------------------
    obs_index = 0
    for tidx in range(D):
        t_next_obs = t[tidx+1]
        t_curr = t[tidx]
        # --------------------------------------------------
        # run the particle filter step for each particle (p)
        # --------------------------------------------------
        for p in range(N):
            v_p = V[p][-1]
            t_p = T[p][-1]
            while(True):
                # sample all holding times for the next state
                current_hold_time = t_curr-t_p
                hold_time_samples = []
                for state in states:
                    hold_time_samples += [ hazard_A.sample(current_hold_time)[v_p,state] ]
                # take the smallest hold-time and move to that state
                t_hold = np.min(hold_time_samples)
                v_next = states[np.argmin(hold_time_samples)]
                w_next = t_p + t_hold

                if w_next >= t_next_obs: # achieved!, compute some weights
                    y_index = states.index(y[tidx])
                    W[p][tidx] = emission(y_index)[v_p]
                    break # break the "while(True)"
                else: # not run long enough; keep on samplin'
                    t_curr = w_next
                    t_p = w_next
                    v_p = v_next
                    # append values to the vectors
                    V[p] += [v_next]
                    T[p] += [w_next]
            
        # ------------------------------------------
        # normalize the weights for the observation
        # ------------------------------------------
        Z[tidx] = logsumexp(W[:,tidx]) # across all particles
        W[:,tidx] -= Z[tidx]
        W[:,tidx] = np.exp(W[:,tidx])
        Z[tidx] -= np.log(N) # Z = (1/N)\hat{Z}

        # ------------------------------------------
        # resample according to the weights
        # ------------------------------------------
        resample = sampleDiscrete(W[:,tidx],N)
        V = [ V[i] for i in resample ]
        T = [ T[i] for i in resample ]

    final_path_index = sampleDiscrete(W[:,tidx],1)
    likelihood = np.sum(Z)
    # -- Rao says: sample at first index --
    V[0] = V[0] + [V[0][-1]]
    T[0] += [t_final]
    path_states = V[0]
    path_times = V[0]
    return path_states,path_times,likelihood


def smc(*args,**kwargs):
    """
    UNTESTED

    vanilla implementation of smc
    """

    N = kwargs['N'] # number of particles
    T = kwargs['T'] # number of time discrete steps
    obs = kwargs['obs'] # observations
    mu = kwargs['mu'] # prior on X; X_i ~ \mu
    q = kwargs['q'] # envelope for IS
    f = kwargs['f'] # transition probability
    g = kwargs['g'] # emission probability
    
    # first sample
    _X = q.sample(y[0],None,N)
    X = [_X]
    w = mu(X[0]) * g(y[0],X[0]) / q.l(X[0],y[0])
    _W = w / np.sum(w)
    W = [_W]
    A = []

    # sample for each t \in {2,..,T}
    for t in range(1,T):
        _A = npr.multinomial(W[t-1])
        A += [_A] # set A_{t-1}
        _X = q.sample(y[t],X[t-1],A[t],N)
        X += [_X]
        w = f(X[t],X[t-1],A) * g(y[t],X[t]) / q.l(X[t],y[t],X[t-1],A)
        _W = w / np.sum(w)
        W += [_W]


    final_path_index = npr.multinomial(W[T])
    path = construct_path(final_path_index,A,X,N,T)
    return path

def constuct_path(k,A,X,N,T):
    """
    Create the path (X_1,...,X_T) given the 
    (A,X) pair and the particle index for time T;
    We move backward through the graph generated by smc
    """

    # backward ancestry; trace the smc graph backward
    B = [ [None for _ in range(N)] for _ in range(T) ]
    B[T] = np.arange(N)
    for t in range(T-1,1):
        B[t] = A[t][B[t+1]]

    # grab our sample path
    sample_path = []
    for t in range(T,1):
        sample_path += [ X[B[k]] ]
    sample_path = [s for s in reversed(sample_path)] # {T,..,1} -> {1,..,T}
    return sample_path
