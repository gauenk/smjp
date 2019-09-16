import pickle,uuid,re
from pkg.smjp_utils import *
from pkg.mcmc_utils import *

def raoteh(number_of_samples,save_iter,smjp_sampler_input,data,pi_0,\
            poisson_process_A_hat,V,T,uuid_str,omega,filename,load_file):

    # some unpacking
    state_space = smjp_sampler_input[0]
    hazard_A = smjp_sampler_input[1]
    time_length = smjp_sampler_input[-1]

    # print info to user
    print("----> Running Rao-Teh Algorithm <----")
    print("number of samples: {}".format(number_of_samples))

    # print init sample
    print("--------------")
    print(V)
    print(T)
    print("|T| = {}".format(len(T)))
    print("--------------")

    # create collection bins
    aggregate = {'W':[],'T':[],'V':[],'prob':[]}
    aggregate_prior = {'W':[],'T':[],'V':[]}

    # start the rao-teh gibbs loop
    if not load_file:
        for i in range(number_of_samples):
            # ~~ sample the data given the sample path ~~
            # data = sample_data_posterior(V,T,*data_sampler_info)
            # print('data',data)

            # ~~ the main gibbs sampler for mcmc of posterior sample paths ~~
            p_u_str = 'raoteh_{}_{}'.format(uuid_str,i)
            W = sample_smjp_event_times(poisson_process_A_hat,V,T,time_length)
            V,T,prob = sample_smjp_trajectory_posterior(W,data,*smjp_sampler_input,p_u_str)
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
            #aggregate['data'].append(data) # when running Gibbs for P(x|\theta) & P(\theta|x)
            print('posterior',np.c_[V,T])

            # take some samples from the prior for de-bugging the mcmc sampler
            _V,_T = sample_smjp_trajectory_prior(hazard_A, pi_0, state_space, time_length)
            aggregate_prior['V'].append(_V)
            aggregate_prior['T'].append(_T)
            print('prior',np.c_[_V,_T])
            print("i = {}".format(i))

            if (i % save_iter) == 0 and ( i > 0 ):
                print("saving current samples to file.")
                save_samples_in_pickle(aggregate,aggregate_prior,omega,uuid_str,i)
        # save to memory
        p_u_str = 'raoteh_{}'.format(uuid_str)
        save_samples_in_pickle(aggregate,aggregate_prior,omega,p_u_str,None)
    else:
        # load to memory
        fn = "results_45c2b10d-0052-4a9d-a210-e85e58edfe7e_1800.pkl"
        #fn = "final_results/results_1b584b67-cfd4-442b-a737-64d30c296e91_1200.pkl"
        #fn = "final_results/results_eeb56a2f-cbf9-4aba-bd4d-f68fe2c1dd0f_8t_final.pkl"
        #fn = "results_043b94d0-56b9-4d68-847d-e52148d2401e_final.pkl" # no omega
        #fn = "results_541f8ddf-1342-41db-8daa-855a7041081e_final.pkl"
        if filename:
            fn = filename
        aggregate,aggregate_prior,uuid_str,omega = load_samples_in_pickle(fn)
    print("omega: {}".format(omega))
    return aggregate,aggregate_prior,uuid_str,omega
