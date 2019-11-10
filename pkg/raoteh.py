import pickle,uuid,re
import numpy.random as npr
from pkg.smjp_utils import *
from pkg.mcmc_utils import *


def raoteh(inference,number_of_samples,save_iter,state_space,time_final,emission,\
           data,pi_0,hazard_A,hazard_B,poisson_process_A_hat,poisson_process_B,\
           uuid_str,omega,obs_times,filename,load_file):
    """
    Implements the RaoTeh algorithm.

    We also allow for Gibbs sampling of parameters. This option is available if the 
    'parameters' word inclued in the Python list "inference".
    """

    # some error checking
    inference_error_checking(inference)

    # print info to user
    print("----> Running Rao-Teh Algorithm <----")
    print("number of samples: {}".format(number_of_samples))

    # prior sampling
    V,T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_final)

    # print init sample
    print("--------------")
    print(V)
    print(T)
    print("|T| = {}".format(len(T)))
    print("--------------")

    # create collection bins
    if 'trajectory' in inference:
        aggregate = {'W':[],'T':[],'V':[],'prob':[]}
    if 'parameters' in inference:
        aggregate['theta'] = []
    if 'prior' in inference:
        aggregate_prior = {'W':[],'T':[],'V':[]}
        aggregate['prior'] = aggregate_prior

    # create a list for posterior sampling
    smjp_sampler_input = [state_space,hazard_A,hazard_B,emission,time_final]
    
    # start the rao-teh gibbs loop
    if not load_file:
        for i in range(number_of_samples):

            # ~~ sample the data given the sample path ~~
            if 'data' in inference:
                data = sample_data_posterior(V,T,state_space,emission,obs_times)
                print('data',data)
                aggregate['data'].append(data)

            # ~~ the main gibbs sampler for mcmc of posterior sample paths ~~
            p_u_str = 'raoteh_{}_{}'.format(uuid_str,i)
            W = sample_smjp_event_times(poisson_process_A_hat,V,T,time_final)
            V,T,prob = smjp_ffbs(W,data,*smjp_sampler_input,p_u_str)

            print("---------")
            print(W)
            print(V)
            print(T)
            print("|T| = {}".format(len(T)))
            print("---------")
            aggregate['W'].append(W)
            aggregate['V'].append(V)
            aggregate['T'].append(T)
            aggregate['prob'].append(prob)
            print('posterior',np.c_[V,T])

            # ~~ gibbs sampler parameter inference ~~
            if 'parameters' in inference:
                theta = sample_smjp_parameters(data,V,T,state_space)
                aggregate['theta'].append(theta)
                update_parameters(hazard_A,hazard_B,poisson_process_A_hat,\
                                  poisson_process_B,theta)

            # ~~ gibbs sampling prior for debugging ~~
            if 'prior' in inference:
                _V,_T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)
                aggregate['prior']['V'].append(_V)
                aggregate['prior']['T'].append(_T)
                print('prior',np.c_[_V,_T])
                
            # ~~ loop info ~~
            print("i = {}".format(i))
            if (i % save_iter) == 0 and ( i > 0 ):
                print("saving current samples to file.")
                save_samples_in_pickle(aggregate,omega,'raoteh',uuid_str,i)

        # save to memory
        p_u_str = 'raoteh_{}'.format(uuid_str)
        save_samples_in_pickle(aggregate,omega,'raoteh',uuid_str)
    else:
        # load to memory
        fn = "results_45c2b10d-0052-4a9d-a210-e85e58edfe7e_1800.pkl"
        #fn = "final_results/results_1b584b67-cfd4-442b-a737-64d30c296e91_1200.pkl"
        #fn = "final_results/results_eeb56a2f-cbf9-4aba-bd4d-f68fe2c1dd0f_8t_final.pkl"
        #fn = "results_043b94d0-56b9-4d68-847d-e52148d2401e_final.pkl" # no omega
        #fn = "results_541f8ddf-1342-41db-8daa-855a7041081e_final.pkl"
        if filename:
            fn = filename
        aggregate,uuid_str,omega = load_samples_in_pickle(fn)
    print("omega: {}".format(omega))
    return aggregate,uuid_str,omega

def sample_smjp_parameters(data,V,T,state_space,sigma=1.0):
    sss = len(state_space)
    means = esimtate_weibull_shape_mat(V,T,state_space)
    cov = sigma * np.identity(sss**2)
    proposal = npr.multivariate_normal(means.ravel(),cov).reshape(sss,sss)
    print(proposal)
    return {'shape':npr.uniform(0.6,3.0,sss**2).reshape(sss,sss)}


def esimtate_weibull_shape_mat(V,T,state_space):
    sss = len(state_space)
    shapes = np.zeros((sss,sss))
    for curr_idx,curr_s in enumerate(state_space):
        for next_idx,next_s in enumerate(state_space):
            hold_times = filter_T_by_state(T,V,curr_s,next_s)
            np.savetxt("hold_times.txt",hold_times,fmt='%3.4f',delimiter=',')
            if len(hold_times) == 0:
                khat = npr.uniform(0.6,3)
            else:
                khat,ksum = esimtate_weibull_shape(hold_times)
                # print("ksum: {}".format(ksum)) # how good is our "k"
            shapes[curr_idx,next_idx] = khat
    return shapes

def filter_T_by_state(T,V,curr_s,next_s):
    curr_indices = np.where(V == curr_s)[0]
    next_indices = np.where(V == next_s)[0]
    curr_i = []
    next_i = []
    # we need to loop b/c different lengths
    for c_i in curr_indices: 
        for n_i in next_indices:
            if c_i + 1 == n_i:
                curr_i += [c_i]
                next_i += [n_i]
    hold_times = []
    for c,n in zip(curr_i,next_i):
        hold_times += [ T[n] - T[c] ]
    return hold_times
    
def esimtate_weibull_shape(T,grid=0.001,grid_max=5):
    lnT = np.log(T)
    mean = np.mean(T)
    ln_mean = np.mean(lnT)
    optimal_k = -1
    optimal_ksum = np.inf
    for k in np.arange(grid,grid_max,grid): # find k via grid search
        Tk = np.power(T,k)
        sumTk = np.sum(Tk)
        ksum = np.sum(Tk * lnT) / sumTk - 1/k - ln_mean
        # print(k,ksum)
        if np.abs(ksum) < np.abs(optimal_ksum):
            optimal_k = k
            optimal_ksum = ksum
    return optimal_k,optimal_ksum
        
def inference_error_checking(inference):
    if len(inference) == 0:
        raise ValueError("We need to infer something.")
    if 'trajectory' not in inference:
        raise ValueError("We always need to sample trajectory.")
    if 'parameters' in inference and 'trajectory' not in inference:
        raise ValueError("We can't have parameter inference without trajectory inference.")
    
def update_parameters(hazard_A,hazard_B,poisson_proc_A_hat,poisson_proc_B,theta):
    # pi just for testing
    omega = hazard_B.omega

    # A
    print(hazard_A.shape_mat)
    hazard_A.update_parameters(theta)

    # B
    scale_mat_tilde = create_upperbound_scale(hazard_A.shape_mat,hazard_A.scale_mat,omega)
    params = {'shape':hazard_A.shape_mat,'scale':scale_mat_tilde}
    hazard_B.update_parameters(params)

    # A_hat
    scale_mat_hat = create_upperbound_scale(hazard_A.shape_mat,hazard_A.scale_mat,omega-1)
    params = {'shape':hazard_A.shape_mat,'scale':scale_mat_hat}
    poisson_proc_A_hat.update_parameters(params)

    # poissonProcessB
    params = {'shape':hazard_A.shape_mat,'scale':scale_mat_tilde}    
    poisson_proc_B.update_parameters(params)
