import numpy as np
import numpy.random as npr
from functools import partial
from pkg.distributions import WeibullDistribution as Weibull
from pkg.utils import *

def smjp_transition(s_curr,s_next,observation,augmented_state_space,A,B):
    """
    "augmented_state_space" includes the time discretization

    returns P(v_i,l_i | v_{i-1}, l_{i-1}, \delta w_i)

    P(v_i,l_i | v_{i-1}, l_{i-1}, \delta w_i) = 
    1. P(l_i | l_{i-1}, \delta w_i)
    \times
    2. P(v_i,l_i | v_{i-1}, l_i, \delta w_i) 
    """
    log_probability = 0
    delta_w = observation
    v_curr,l_curr = augmented_state_space[s_curr]
    v_next,l_next = augmented_state_space[s_next]
    t_hold = delta_w + l_curr
    # (T,?,?)
    # (F,T,T)

    # Note: we can return immmediately if the "l_next" is not possible.
    # -> this is common since l_next can only be {0, l_curr + delta_w }
    if (l_next != 0) and (l_next != t_hold):
        return 0

    # if we thin the current time-step, we can not transition state values.
    if (l_next == t_hold) and (v_next != v_curr):
        return 0

    # print(delta_w)
    # print(s_curr,s_next)
    # print(augmented_state_space[s_curr],augmented_state_space[s_next])
    # print(l_next == 0,l_next == t_hold,v_next == v_curr)

    # P(l_i | l_{i-1}, \delta w_i)
    l_ratio = A([t_hold])[v_curr] / B([t_hold])[v_curr]
    assert l_ratio <= 1, "the ratio of hazard functions should be <= 1"
    if l_next == 0:
        log_probability += np.ma.log( [l_ratio] ).filled(0)[0]
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) : note below is _not_ a probability.
        log_probability += np.ma.log([ A([t_hold])[v_curr,v_next] ]).filled(0)[0]
    else:
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) = 1 if v_i == v_{i-1}
        # print(l_ratio)
        log_probability += np.ma.log( [1. - l_ratio] ).filled(0)[0]
    print(log_probability)
    return log_probability # np.exp(log_probability)

def smjp_hazard_functions(s_curr,s_next,observation,state_space,h_create):
    """
    The State_Space here refers to the actual state values!
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
        likelihood = 0
        for s_next in state_space:
            # skip the same state (goal of this code snippet)
            if s_next == s_curr: continue 
            hazard_rate = h_create(s_curr,s_next)
            likelihood += hazard_rate.l(t_hold)
        # we don't normalized; if we do the result is incorrect.
        # hazard_rate = h_create(s_curr,s_curr)
        # current_likelihood =  hazard_rate.l(t_hold)
        # likelihood = likelihood / (likelihood + current_likelihood) # normalize over ~all~
        return likelihood
    else:  # return Prob of leaving s_curr for s_next; normalized over s_next
        hazard_rate = h_create(s_curr,s_next)
        likelihood = hazard_rate.l(t_hold)
        normalization_likelihood = likelihood
        for s_next in state_space:
            # skip the same state (goal of this code snippet)
            if s_next == s_curr: continue 
            hazard_rate = h_create(s_curr,s_next)
            normalization_likelihood += hazard_rate.l(t_hold)
        likelihood /= normalization_likelihood
        return likelihood
        
class sMJPWrapper(object):
    """
    We want to be able to call the transition matrix 
    for sMJP in a readable, nicely fashioned format.

    ex:
       pi(\delta w_i)[current_state]
       pi(\delta w_i)[current_state, next_state]

    This way it looks something similar to how it looks in the paper. Cool.

    Why not just pi(current_stat, next_state, \delta w_i)?
    I want to emphasize that each location is a ~function~ of
    the discrete, finite state space. Thus, a function call is 
    use for the proper function argument, \delta w_i.
    """

    def __init__(self,state_function,state_space,*args,observation_index=0):
        self.state_function = state_function
        self.state_space = state_space
        self.observation = 0
        self.observation_index = observation_index
        self.state_function_args = args


    def __call__(self,observation):
        self.observation = observation
        return self

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
        assert isiterable(a_slice) is False, "we can only accept ints"
        s_curr = a_slice # current state
        return self.state_function(s_curr,None,self.observation[self.observation_index],
                                   self.state_space,*self.state_function_args)

    def _slice_2d(self,a_slice):
        assert len(a_slice) == 2, "we can only do 2d"
        s_curr = a_slice[0] # current state
        s_next = a_slice[1] # next state
        if type(s_curr) is slice or type(s_next) is slice:
            raise TypeError("we can only accept integer inputs")
        # logic for functions based on current & next state (maybe 
        return self.state_function(s_curr,s_next,self.observation[self.observation_index],
                                   self.state_space,*self.state_function_args)

"""
A(delta w_i)[v_curr] = the hazard function associated with 
                        leaving the state v_curr for any other state according

B(delta w_i)[v_curr] = same as above, but B > A (? B >= A?)

A(delta w_i)[v_curr,v_next] = the _normalized_ (w.r.t. v_next, fixed delta w_i and v_curr) 
                              hazard function
"""


# weibull hazard rate experiment
def weibull_hazard_create_unset(shape_mat,scale_mat,state_space,state_curr,state_next):
    assert state_curr in state_space, "must have the state in state space"
    assert state_next in state_space, "must have the state in state space"
    shape = shape_mat[state_curr-1][state_next-1]
    scale = scale_mat[state_curr-1][state_next-1]
    rv = Weibull({'shape':shape,'scale':scale},is_hazard_rate=True)
    return rv

def enumerate_state_space(grid,states):
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
# main function used in a Gibbs sampler
#

def sample_smjp_trajectory(W,state_space,hazard_A,hazard_B,emission,data,pi_0):
    augmented_state_space = enumerate_state_space(W,state_space)
    aug_ss_size = len(augmented_state_space)
    pi = sMJPWrapper(smjp_transition,augmented_state_space,hazard_A,hazard_B)
    hmm_init = {'emission': HMMWrapper(emission,True), 
                'transition': transition,
                'data': data,
                'state_alphabet': state_space,
                'pi_0': pi_0,
                'path_length': len(W),
                'sample_dimension': 1,
                }
    hmm = HiddenMarkovModel([],**hmm_init)
    alphas,prob = hmm.likelihood() # only for dev.
    samples = hmm.backward_sampling()
    return samples,alphas,prob


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
    y_values_A = [hazard_A([x])[1,1] for x in x_values]
    y_values_B = [hazard_B([x])[1,1] for x in x_values]
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

    print(hazard_A([5])[1,2])
    print(hazard_A([2])[1,3])
    print(hazard_A([5])[1,3])
    print(hazard_A([2])[1,3])
    print(hazard_A([5])[1])
    print(hazard_A([5])[2])


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
    weibull_hazard_create_B = partial(weibull_hazard_create_unset,shape_mat,\
                                      scale_mat_tilde,States)

    hazard_A = sMJPWrapper(smjp_hazard_functions,States,weibull_hazard_create_A)
    hazard_B = sMJPWrapper(smjp_hazard_functions,States,weibull_hazard_create_B)
    
    print(hazard_A([1])[1,2])
    print(hazard_B([1])[1,2])
    print(hazard_A([1])[1])
    print(hazard_B([1])[1])

    pi = sMJPWrapper(smjp_transition,augmented_state_space,hazard_A,hazard_B)
    
    import matplotlib.pyplot as plt

    # plot over augmented_state_space
    x_values = np.arange(0,aug_ss_size**2,1)
    print("_1_")
    y_values_1 = [pi([5])[i,j] for i in range(aug_ss_size) for j in range(aug_ss_size)]
    y_values_h = [hazard_A([5])[i,j] for i in States for j in States]
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
    y_values_00 = -np.ma.log([pi([x])[0,2] for x in W]).filled(0)
    print("_01_")
    y_values_01 = -np.ma.log([pi([x])[0,3] for x in W]).filled(0)
    print("_11_")
    y_values_11 = -np.ma.log([pi([x])[5,6] for x in W]).filled(0)
    print("_12_")
    y_values_12 = -np.ma.log([pi([x])[23,29] for x in W]).filled(0)

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

    
