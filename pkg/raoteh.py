import pickle,uuid,re
from pkg.smjp_utils import *
from pkg.mcmc_utils import *


def raoteh(inference,number_of_samples,save_iter,state_space,smjp_emission,time_final,\
           data,pi_0,hazard_A,hazard_B,poisson_process_A_hat,uuid_str,omega,filename,\
           load_file):

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
    smjp_sampler_input = [state_space,hazard_A,hazard_B,smjp_emission,time_final]
    
    # start the rao-teh gibbs loop
    if not load_file:
        for i in range(number_of_samples):

            # ~~ sample the data given the sample path ~~
            if 'data' in inference:
                data = sample_data_posterior(V,T,*data_sampler_info)
                print('data',data)
                aggregate['data'].append(data)

            # ~~ the main gibbs sampler for mcmc of posterior sample paths ~~
            p_u_str = 'raoteh_{}_{}'.format(uuid_str,i)
            W = sample_smjp_event_times(poisson_process_A_hat,V,T,time_final)
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
            print('posterior',np.c_[V,T])

            # ~~ gibbs sampler parameter inference ~~
            if 'parameters' in inference:
                theta = sample_smjp_parameters(data,V,T)
                aggregate['theta'].append(theta)
                smjp_sampler_input[1].update_parameters(theta) # hazard_A
                smjp_sampler_input[2].update_parameters(theta) # hazard_B
                poisson_process_A_hat.update_parameters(theta) # poisson process A_hat

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

def inference_error_checking(inference):
    if len(inference) == 0:
        raise ValueError("We need to infer something.")
    if 'trajectory' not in inference:
        raise ValueError("We always need to sample trajectory.")
    if 'parameters' in inference and 'trajectory' not in inference:
        raise ValueError("We can't have parameter inference without trajectory inference.")
    
