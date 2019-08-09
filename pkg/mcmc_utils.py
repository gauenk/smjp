import re
import numpy as np
from pkg.utils import *
import matplotlib.pyplot as plt

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
            
def compute_metric_summaries(state_times,state_jumps,states):

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

def plot_metric_traces(time_info,jump_info,states,uuid_str,file_id=None):
    # format data
    x_grid = {state:None for state in states}
    for state in states:
        x_grid[state] = np.arange(len(time_info[state]))
    
    # plot it
    if file_id:
        plot_filename = "trace_time_{}_{}.png".format(uuid_str,file_id)
    else:
        plot_filename = "trace_time_{}.png".format(uuid_str)        
    plot_title = "Traces of [Total Time] by State"
    plot_metric_helper(x_grid,time_info,states,plot_title,plot_filename)

    if file_id:
        plot_filename = "trace_jump_{}_{}.png".format(uuid_str,file_id)
    else:
        plot_filename = "trace_jump_{}.png".format(uuid_str)
    plot_title = "Traces of [Number of Jumps] by State"
    plot_metric_helper(x_grid,jump_info,states,plot_title,plot_filename)

def plot_metric_autocorrelation(time_info,jump_info,states,uuid_str,file_id=None):

    # compute autocorrelation
    ss = len(states)
    time_acorr = {state:0 for state in states}
    jump_acorr = {state:0 for state in states}
    x_grid = {state:None for state in states}
    for state in states:
        time_acorr[state] = compute_autocorrelation(time_info[state])
        time_acorr[state] = time_acorr[state][:50]
        jump_acorr[state] = compute_autocorrelation(jump_info[state])        
        jump_acorr[state] = jump_acorr[state][:50]
        x_grid[state] = np.arange(len(time_acorr[state]))

    # plot results
    if file_id:
        plot_filename = "autocorrelation_time_{}_{}.png".format(uuid_str,file_id)
    else:
        plot_filename = "autocorrelation_time_{}.png".format(uuid_str)
    plot_title = "Autocorrelation of [Total Time]"
    plot_metric_helper(x_grid,time_acorr,states,plot_title,plot_filename)

    if file_id:
        plot_filename = "autocorrelation_jump_{}_{}.png".format(uuid_str,file_id)
    else:
        plot_filename = "autocorrelation_jump_{}.png".format(uuid_str)        
    plot_title = "Autocorrelation of [Number of Jumps]"
    plot_metric_helper(x_grid,jump_acorr,states,plot_title,plot_filename)

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
