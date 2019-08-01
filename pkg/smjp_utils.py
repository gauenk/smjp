import numpy as np
import numpy.random as npr
from functools import partial
from distributions import WeibullDistribution as Weibull
from utils import *

def smjp_transition(s_curr,s_next,observation,state_space,A,B):
    """
    returns P(v_i,l_i | v_{i-1}, l_{i-1}, \delta w_i)

    P(v_i,l_i | v_{i-1}, l_{i-1}, \delta w_i) = 
    1. P(l_i | l_{i-1}, \delta w_i)
    \times
    2. P(v_i,l_i | v_{i-1}, l_i, \delta w_i) 
    """
    log_probability = 0
    delta_w = observation[0]
    v_curr,l_curr = state_space[s_curr]
    v_next,l_next = state_space[s_next]

    # Note: we can return immmediately if the "l_next" is not possible.
    # -> this is common since l_next can only be {0, l_curr + delta_w }
    if (l_next != 0) and (l_next != l_curr + delta_w):
        return 0

    # if we thin the current time-step, we can not transition state values.
    if (l_next == l_curr + delta_w) and (v_next != v_curr):
        return 0

    # P(l_i | l_{i-1}, \delta w_i)
    print(A([delta_w])[0])
    print("v_curr",v_curr)
    ratio = A([delta_w])[v_curr] / B([delta_w])[v_curr] # TODO: define A, B
    if l_next == 0:
        log_probability += np.ma.log( l_ratio ).filled(0)
    else:
        # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) = 1 if v_i == v_{i-1}
        return (1 - ratio) 

    # P(v_i,l_i | v_{i-1}, l_i, \delta w_i) 
    log_probability += np.ma.log( A(delta_w)[v_curr,v_next] ).filled(0)
    return np.exp(log_probability)

def smjp_hazard_functions(s_curr,s_next,observation,state_space,h_create):
    delta_w = observation[0]
    if s_next is None: # return Prob of leaving s_curr for ~any state~
        # possible error here for A_s
        # this "log" doesn't help much:
        ## its a sum-of-probs;
        ## "log" method is helpful when its a prod-of-probs
        # (e.g. not time for "log-sum-exp")
        v_curr,_ = state_space[s_curr]
        all_v = [v for v,l in state_space]
        likelihood = 0
        for v_next in all_v:
            if v_next == v_curr: continue # skip the same state (goal of this code snippet)
            hazard_rate = h_create(v_curr,v_next)
            likelihood += hazard_rate.l(delta_w)
        return likelihood
    else:  # return Prob of leaving s_curr for s_next
        v_curr,_ = state_space[s_curr]
        v_next,_ = state_space[s_next]
        hazard_rate = h_create(v_curr,v_next)
        likelihood = hazard_rate.l(delta_w) # possibly un-normalized!!!
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

    def __init__(self,state_function,state_space,*args):
        self.state_function = state_function
        self.state_space = state_space
        self.state_function_args = args
        self.observation = 0

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
        return self.state_function(s_curr,None,self.observation,
                                   self.state_space,*self.state_function_args)

    def _slice_2d(self,a_slice):
        assert len(a_slice) == 2, "we can only do 2d"
        s_curr = a_slice[0] # current state
        s_next = a_slice[1] # next state
        if type(s_curr) is slice or type(s_next) is slice:
            raise TypeError("we can only accept integer inputs")
        # logic for functions based on current & next state (maybe 
        self.state_function(s_curr,s_next,self.observation,
                                   self.state_space,*self.state_function_args)

"""
A(delta w_i)[v_curr] = the hazard function associated with 
                        leaving the state v_curr for any other state according

B(delta w_i)[v_curr] = same as above, but B > A (? B >= A?)

A(delta w_i)[v_curr,v_next] = the _normalized_ (w.r.t. v_next, fixed delta w_i and v_curr) 
                              hazard function
"""


# weibull hazard rate experiment
def weibull_hazard_create_unset(shape_mat,scale_mat,state_curr,state_next):
    shape = shape_mat[state_curr][state_next]
    scale = scale_mat[state_curr][state_next]
    rv = Weibull({'shape':shape,'scale':scale})
    return rv

def enumerate_state_space(grid,states):
    mesh = np.meshgrid(grid,states)
    state_space = np.c_[mesh[0].ravel(),mesh[1].ravel()]
    return state_space
    



#
# Testing.
#



def check_weibull():
    # the two plots should overlap
    
    import matplotlib.pyplot as plt
    States = [1,2,3]
    W = np.arange(20) # random_grid()
    state_space = enumerate_state_space(W,States)
    ss_size = len(state_space)

    # shape_mat = npr.uniform(0.6,3,ss_size**2).reshape((ss_size,ss_size))
    shape_mat = np.ones((ss_size,ss_size))
    scale_mat = np.ones((ss_size,ss_size)) 
    weibull_hazard_create = partial(weibull_hazard_create_unset,shape_mat,scale_mat)

    hazard_A = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create)

    x_values = np.arange(0,20.1)
    y_values = [hazard_A([x])[1,1] for x in x_values]

    plt.plot(x_values,y_values,'k+-')
    w_rv = Weibull({'shape':1.0,'scale':1.0})
    y_values = [w_rv.l(x) for x in x_values]

    plt.plot(x_values,y_values,'g+-')
    plt.show()

    print(hazard_A([5])[1,4])
    print(hazard_A([2])[1,4])
    print(hazard_A([5])[1,3])
    print(hazard_A([2])[1,3])
    print(hazard_A([5])[1])
    print(hazard_A([5])[6])


def check_transition():

    States = [1,2,3]

    # reset each Gibbs iteration
    W = np.arange(20) # random_grid()
    state_space = enumerate_state_space(W,States)
    ss_size = len(state_space)

    shape_mat = npr.uniform(0.6,3,ss_size**2).reshape((ss_size,ss_size))
    # shape_mat = np.ones((ss_size,ss_size))
    scale_mat = np.ones((ss_size,ss_size)) 
    weibull_hazard_create = partial(weibull_hazard_create_unset,shape_mat,scale_mat)

    hazard_A = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create)
    hazard_B = sMJPWrapper(smjp_hazard_functions,state_space,weibull_hazard_create)
    pi = sMJPWrapper(smjp_transition,state_space,hazard_A,hazard_B)
    
    print(pi([0])[1,2])
    import matplotlib.pyplot as plt

    x_values = np.arange(0,ss_size)
    print(x_values)

    print(pi([1.])[0,0])
    print("new")
    y_values_00 = [pi([x])[0,0] for x in x_values]
    y_values_01 = [pi([x])[0,1] for x in x_values]
    y_values_11 = [pi([x])[1,1] for x in x_values]
    y_values_12 = [pi([x])[1,2] for x in x_values]

    plt.plot(x_values,y_values_00,'k+-',label='00')
    plt.plot(x_values,y_values_01,'g+-',label='01')
    plt.plot(x_values,y_values_11,'r+-',label='11')
    plt.plot(x_values,y_values_12,'c+-',label='12')

    plt.show()


if __name__ == "__main__":
    
    # check_weibull() # passsed.
    check_transition()
