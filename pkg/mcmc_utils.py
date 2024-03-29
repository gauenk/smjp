import re,pickle
import numpy as np
import numpy.random as npr
from pkg.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sss

def compute_ks(x,y,alpha=0.05):
    """
    WARNING: THIS FUNCTION IS INCOMPLETE. NO ATTEMPT WAS MADE TO HAVE THIS CODE PROPER.

    Compute the Kolmogorov-Smirnov test for two empirical CDFs
    """
    print("ERROR: DO NOT USE")
    exit()
    n = len(x)
    m = len(y)
    c_table = np.array([[0.10,1.073],
                  [0.05,1.224],
                  [0.025,1.358],
                  [0.01,1.517],
                  [0.005,1.628],
                  [0.001,1.858],
    ])
    c = c_table[np.where(np.isclose(c_table[:,0],alpha))[0],1]
    reject_value = c * np.sqrt( (n+m) / (n*m) )

    #
    # compare the empirical distribution functions
    #

    sx = sorted(x)
    sy = sorted(y)
    d_nm = np.max(np.abs(sx - sy))

    is_recjected = d_nm > reject_value
    return is_recjected,d_nm,reject_values
    
def compute_evaluation_chain_metrics(agg,states):
    V_list = agg['V']
    T_list = agg['T']
    ss = len(states)
    n_samples = len(T_list)
    
    # create a differently formatted "aggreate" object
    state_times = {key:[0 for _ in range(n_samples)] for key in states}
    state_jumps = {key:[0 for _ in range(n_samples)] for key in states}
    for s_index,V,T in zip(range(n_samples),V_list,T_list):
        for state in states:
            state_bool = np.isclose(V, state)
            state_jumps[state][s_index] = np.sum(state_bool)
        for index,state in enumerate(V[:-1]):
            state_times[state][s_index] += T[index+1] - T[index]
            
    return state_times,state_jumps
            
def compute_metric_summaries(samples,states):
    # compute summary statistics
    info = {'median':{state:0 for state in states},
                 'mean':{state:0 for state in states},
                 'var':{state:0 for state in states},
    }
    for state in states:
        info['median'][state] = np.median(samples[state])
        info['mean'][state] = np.mean(samples[state])
        info['var'][state] = np.var(samples[state])
    return info

def compute_metric_summaries_trajectory(state_times,state_jumps,states):

    # compute summary statistics
    time_info = {'median':{state:0 for state in states},
                 'mean':{state:0 for state in states},
                 'var':{state:0 for state in states},
    }
    jump_info = {'median':{state:0 for state in states},
                 'mean':{state:0 for state in states},
                 'var':{state:0 for state in states},
    }
    for state in states:

        time_info['median'][state] = np.median(state_times[state])
        time_info['mean'][state] = np.mean(state_times[state])
        time_info['var'][state] = np.var(state_times[state])
        
        jump_info['median'][state] = np.median(state_jumps[state])
        jump_info['mean'][state] = np.mean(state_jumps[state])
        jump_info['var'][state] = np.var(state_jumps[state])


    return time_info, jump_info

def plot_metric_traces(samples,name,title,states,uuid_str,file_id=None):
    # format data
    x_grid = {state:None for state in states}
    for state in states:
        x_grid[state] = np.arange(len(samples[state]))
    
    # plot it
    if file_id:
        plot_filename = "trace_{}_{}_{}.png".format(name,uuid_str,file_id)
    else:
        plot_filename = "trace_{}_{}.png".format(name,uuid_str)        
    plot_metric_helper(x_grid,samples,states,title,plot_filename)
    
def plot_metric_traces_time_jump(time_info,jump_info,states,uuid_str,file_id=None):
    title = "Traces of [Total Time] by State"
    plot_metric_traces(time_info,"time",title,states,uuid_str,file_id)
    title = "Traces of [Number of Jumps] by State"
    plot_metric_traces(jump_info,"jump",title,states,uuid_str,file_id)

def toy_correlationed_data(nsamples=1000):
    x = np.zeros(nsamples)
    x[0] = npr_unif(-1,1)
    x[1] = npr_unif(0,1) * x[0]
    x[2] = npr_unif(0,1) * x[1] + npr_unif(-1,0) * x[0]
    x[3] = npr_unif(0,1) * x[2] + npr_unif(-1,0) * x[1] + npr_unif(-1,-1) * x[0]
    for i in range(3,nsamples):
        x[i] = npr_unif(0,1) * x[i-1] + npr_unif(-1,0) * x[i-2] + npr_unif(-1,-1) * x[i-3]
    x /= np.sum(x)
    return x

def npr_unif(b,e):
    return npr.uniform(b,e,1)[0]

def plot_metric_autocorrelation(samples,name,title,states,uuid_str,file_id=None):
    # compute autocorrelation
    max_acorr = 50
    ss = len(states)
    acorr = {state:0 for state in states}
    x_grid = {state:None for state in states}
    for state in states:
        # data = toy_correlationed_data()
        # acorr[state] = compute_autocorrelation(data)
        acorr[state] = compute_autocorrelation(samples[state])
        acorr[state] = acorr[state][:max_acorr]
        x_grid[state] = np.arange(len(acorr[state]))
    # plot results
    if file_id:
        plot_filename = "autocorrelation_{}_{}_{}.png".format(name,uuid_str,file_id)
    else:
        plot_filename = "autocorrelation_{}_{}.png".format(name,uuid_str)

    plot_metric_helper(x_grid,acorr,states,title,plot_filename)
    
def plot_metric_autocorrelation_time_jump(time_info,jump_info,states,uuid_str,file_id=None):
    title = "Autocorrelation of [Total Time]"
    plot_metric_autocorrelation(time_info,'time',title,states,uuid_str,file_id)
    title = "Autocorrelation of [Number of Jumps]"
    plot_metric_autocorrelation(jump_info,'jump',title,states,uuid_str,file_id)

def plot_metric_helper(x_grid,y_data,states,plot_title,plot_filename):
    import matplotlib.pyplot as plt
    # handle plotting
    ss = len(states)
    fig = plt.figure(figsize=(6,3))
    axes = {}
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, ss)]
    for index,state in enumerate(states):
        axes[state] = fig.add_subplot(ss,1,index+1)
        axes[state].plot(x_grid[state],y_data[state],label=state,color=colors[index])

    lgd = fig.legend(bbox_to_anchor=(0,1.10,1,0.2), loc="upper left",mode="expand", borderaxespad=0, ncol=3, title="State")
    fig.tight_layout()
    fig.suptitle(plot_title,fontsize=20,y=1.10)
    fig.savefig("./figs/" + plot_filename,dpi=500,quality=100,bbox_inches='tight',bbox_extra_artists=(lgd,))
    plt.close(fig)
    plt.clf()
    plt.cla()
    

def create_summary_image(uuid_str,metrics,fields,file_id = None):
    import glob,imageio
    
    import matplotlib.pyplot as plt

    file_list = []
    for fn in glob.glob('./figs/*{}*png'.format(uuid_str)):
        file_list += [fn]
    # we want a plot for each dim:
    n_metrics = len(metrics)
    n_fields = len(fields)
    n_plots = n_metrics * n_fields
    fig,ax_mat = plt.subplots(figsize=(6 * n_metrics,3 * n_fields),ncols=n_metrics,nrows=n_fields,gridspec_kw = {'wspace':-0.3, 'hspace':0.05})
    #fig = plt.figure(figsize=(8 * n_metrics,8 * n_fields))
    axes = {}
    for m_index,metric in enumerate(metrics):
        axes[metric] = {}
        for f_index,field in enumerate(fields):
            number = (f_index+1) + n_fields * m_index
            #axes[metric][field] = fig.add_subplot(n_metrics,n_fields,number,gridspec_kw = {'wspace':0, 'hspace':0})
            axes[metric][field] = ax_mat[m_index][f_index]
            axes[metric][field].grid('off')
            axes[metric][field].set_xticklabels([])
            axes[metric][field].set_yticklabels([])
            image_fn = get_image_name(file_list,metric,field,file_id)
            image = imageio.imread(image_fn)
            axes[metric][field].imshow(image)

    if file_id:
        plot_filename = 'summary_{}_{}'.format(file_id,uuid_str)
        plot_title = 'Summary [{}]'.format(file_id)
    else:
        plot_filename = 'summary_{}'.format(uuid_str)
        plot_title = 'Summary'
        
    # fig.tight_layout()
    fig.suptitle(plot_title,fontsize=20)
    fig.savefig("./figs/" + plot_filename,dpi=300,quality=100,bbox_inches='tight')

def get_image_name(file_list,metric,field,file_id):
    regexp_metric = re.compile(metric)
    regexp_field = re.compile(field)
    regexp_id = re.compile(file_id)
    for name in file_list:
        find_metric = regexp_metric.search(name)
        find_field = regexp_field.search(name)
        find_id = regexp_id.search(name)
        # print(name,find_metric,find_field,find_id,metric,field,file_id)
        if find_metric and find_field and find_id:
            return name
    print("No match.")
    return None


def generate_sample_report_twochainz(aggA,aggB,nameA,nameB,state_space,uuid_str):
    states = state_space # alias
    time_info_A,jump_info_A = compute_evaluation_chain_metrics(aggA,state_space)
    time_info_B,jump_info_B = compute_evaluation_chain_metrics(aggB,state_space)

    agg_time_info_A,agg_jump_info_A = compute_metric_summaries_trajectory(time_info_A,jump_info_A,state_space)
    agg_time_info_B,agg_jump_info_B = compute_metric_summaries_trajectory(time_info_B,jump_info_B,state_space)

    print("-=-=-=- Aggregate Information -=-=-=-")
    print(" ------------ ")
    print(" --> Time <-- ")
    print(" ------------ ")
    print('--> {} <--'.format(nameA))
    print(agg_time_info_A)
    print('--> {} <--'.format(nameB))
    print(agg_time_info_B)

    print(" ------------ ")
    print(" --> Jumps <-- ")
    print(" ------------ ")
    print('--> {} <--'.format(nameA))
    print(agg_jump_info_A)
    print('--> {} <--'.format(nameB))
    print(agg_jump_info_B)

    print("-"*30)

    # compact the results for returning
    metrics_A = {'time':time_info_A,
                         'jump':jump_info_A,
                         'agg_time':agg_time_info_A,
                         'agg_jump':agg_jump_info_A,
                         }
    metrics_B = {'time':time_info_B,
                     'jump':jump_info_B,
                     'agg_time':agg_time_info_B,
                     'agg_jump':agg_jump_info_B,
    }

    # create density plots of values
    # plot_sample_densities(time_info_A,time_info_B,jump_info_A,jump_info_B,state_space)
    compute_ks_twosample_time_jump(time_info_A,time_info_B,jump_info_A,jump_info_B,state_space)

    # create plots of metrics
    if False:
        file_id = nameA
        plot_metric_traces_time_jump(time_info_A,jump_info_A,state_space,uuid_str,file_id)
        plot_metric_autocorrelation_time_jump(time_info_A,jump_info_A,state_space,uuid_str,file_id)
        create_summary_image(uuid_str,['trace','autocorr'],['time','jump'],file_id)

        file_id = nameB
        plot_metric_traces_time_jump(time_info_B,jump_info_B,state_space,uuid_str,file_id)
        plot_metric_autocorrelation_time_jump(time_info_B,jump_info_B,state_space,uuid_str,file_id)
        create_summary_image(uuid_str,['trace','autocorr'],['time','jump'],file_id)
        print("Finished computing metrics for experiment id {}".format(uuid_str))

    # compate parameter inference
    if 'theta' in aggA.keys() and 'theta' in aggB.keys():
        print("-"*50)
        print("--- Computing KS Test for Parameter Inference ---")
        print("-"*50)
        print(aggA['theta'][0])
        print(aggB['theta'][0])
        print(aggB['theta'][5])
        print(aggB['theta'][10])
        print(type(aggA['theta']))
        print(type(aggB['theta']))
        print(type(aggA['theta'][0]))
        paramsA,paramsB = listdict_to_dictlist(aggA['theta']),listdict_to_dictlist(aggB['theta'])
        common_inference = [key for key in paramsA.keys() if key in paramsB.keys()]
        for key in common_inference:
            paramA_raw,paramB_raw = paramsA[key],paramsB[key]
            paramA,paramB,pstates = compute_params_by_state(paramA_raw,paramB_raw,state_space)
            compute_ks_twosample(paramA,paramB,pstates)
            # traces
            file_id = nameA + '_{}'.format(key)
            title = "{} Traces of [Parameter {}] by State Pairs".format(nameA,key)
            plot_metric_traces(paramA,"param",title,pstates,uuid_str,file_id)
            file_id = nameB + '_{}'.format(key)
            title = "{} Traces of [Parameter {}] by State Pairs".format(nameB,key)
            plot_metric_traces(paramB,"param",title,pstates,uuid_str,file_id)
            # autocorrelation
            file_id = nameA + '_{}'.format(key)
            title = "{} Autocorrelation of [Parameter {}] by State Pairs".format(nameB,key)
            plot_metric_autocorrelation(paramA,key,title,pstates,uuid_str,file_id)
            file_id = nameA + '_{}'.format(key)
            title = "{} Autocorrelation of [Parameter {}] by State Pairs".format(nameB,key)
            plot_metric_autocorrelation(paramA,key,title,pstates,uuid_str,file_id)
            # summary statistics
            summaryA = compute_metric_summaries(paramA,pstates)
            summaryB = compute_metric_summaries(paramB,pstates)
            print("--- summary of [{}] parameter inference ---".format(key))
            print(summaryA)
            print(summaryB)

        create_summary_image(uuid_str,['trace','autocorr'],common_inference,file_id)
    return 

def listdict_to_dictlist(listdict):
    dictkeys = listdict[0].keys()
    dictlist = {key:[] for key in dictkeys}
    for item in listdict:
        for key,value in item.items():
            dictlist[key].append(value)
    # numpyify
    for key in dictkeys:
        dictlist[key] = np.array(dictlist[key])
    return dictlist

def compute_params_by_state(paramsA,paramsB,state_space):
    pair_states = []
    paramsA_dict = {}
    paramsB_dict = {}
    for sci,sc in enumerate(state_space):
        for sni,sn in enumerate(state_space):
            pair_state = '{}_{}'.format(sc,sn)
            pair_states.append(pair_state)
            paramsA_dict[pair_state] = paramsA[:,sci,sni]
            paramsB_dict[pair_state] = paramsB[:,sci,sni]
    return paramsA_dict,paramsB_dict,pair_states

def compute_ks_twosample(samplesA,samplesB,state_space):
    skip = 500
    n = len(samplesA[state_space[0]][::skip])
    m = len(samplesB[state_space[0]][::skip])
    alpha = 0.05
    c_alpha = 1.224
    ub = c_alpha * np.sqrt( ( n + m ) / ( n * m ) )

    print("[skip: {}]".format(skip))
    ssize = len(state_space)
    ks = np.zeros(ssize*2).reshape(ssize,2)
    for state_idx,state in enumerate(state_space):
        # times = npr.rand(len(samplesA[state]))
        sA = samplesA[state][::skip]
        sB = samplesB[state][::skip]
        ks_result = sss.ks_2samp(sA,sB)
        ks[state_idx,0] = ks_result.statistic
        ks[state_idx,1] = ks_result.pvalue
    
    print(" ---> KS Results <--- ")
    print("Reject if D_{{n,m}} > {}".format(ub))
    print("--> ks")
    print("[state,stat,pvalue,do_we_reject?]")
    is_reject = ks[:,0] > ub
    print(np.c_[np.arange(len(state_space)),ks,is_reject])
    print("-"*30)

    
def compute_ks_twosample_time_jump(time_info,time_info_prior,jump_info,jump_info_prior,state_space):
    # not used but for records
    skip = 30
    n = len(time_info[state_space[0]][::skip])
    m = len(time_info_prior[state_space[0]][::skip])
    alpha = 0.05
    c_alpha = 1.224
    ub = c_alpha * np.sqrt( ( n + m ) / ( n * m ) )

    print("[skip: {}]".format(skip))
    ssize = len(state_space)
    times_ks = np.zeros(ssize*2).reshape(ssize,2)
    for state_idx,state in enumerate(state_space):
        # times = npr.rand(len(time_info[state]))
        times = time_info[state][::skip]
        times_pr = time_info_prior[state][::skip]
        ks_result = sss.ks_2samp(times,times_pr)
        times_ks[state_idx,0] = ks_result.statistic
        times_ks[state_idx,1] = ks_result.pvalue
    
    jumps_ks = np.zeros(ssize*2).reshape(ssize,2)
    for state_idx,state in enumerate(state_space):
        jumps = jump_info[state][::skip]
        jumps_pr = jump_info_prior[state][::skip]
        ks_result = sss.ks_2samp(jumps,jumps_pr)
        jumps_ks[state_idx,0] = ks_result.statistic
        jumps_ks[state_idx,1] = ks_result.pvalue

    print(" ---> KS Results <--- ")
    print("Reject if D_{{n,m}} > {}".format(ub))
    print("--> times_ks")
    print("[stat,pvalue,do_we_reject?]")
    is_reject = times_ks[:,0] > ub
    print(np.c_[times_ks,is_reject])
    print("--> jumps_ks")
    print("[stat,pvalue,do_we_reject?]")
    is_reject = jumps_ks[:,0] > ub
    print(np.c_[jumps_ks,is_reject])
    print("-"*30)


def plot_sample_densities(time_info,time_info_prior,jump_info,jump_info_prior,state_space):
    for state in state_space:
        times = time_info[state]
        times_pr = time_info_prior[state]
        sns.distplot(times,hist=False,rug=True,label='{}_post'.format(state))
        sns.distplot(times_pr,hist=False,rug=True,label='{}_prior'.format(state))
    plt.show()
    plt.clf()
    
    for state in state_space:
        jumps = jump_info[state]
        jumps_pr = jump_info_prior[state]
        sns.distplot(jumps,hist=False,rug=True,label='{}_post'.format(state))
        sns.distplot(jumps_pr,hist=False,rug=True,label='{}_prior'.format(state))
    plt.show()
    plt.clf()


def save_samples_in_pickle(aggregate,omega,expName,uuid_str,n_iters=None):
    if n_iters is None:
        fn = 'results_{}_{}_final.pkl'.format(expName,uuid_str)
    else:
        fn = 'results_{}_{}_{}.pkl'.format(expName,uuid_str,n_iters)        
    pickle_mem_dump = {'agg':aggregate,'uuid_str':uuid_str,'omega':omega}
    with open(fn,'wb') as f:
        pickle.dump(pickle_mem_dump,f)

def load_samples_in_pickle(fn):
    with open(fn,'rb') as f:
        pickle_mem_dump = pickle.load(f)
        aggregate = pickle_mem_dump['agg']
        uuid_str = pickle_mem_dump['uuid_str']
        omega = pickle_mem_dump['omega']
    return aggregate,uuid_str,omega


