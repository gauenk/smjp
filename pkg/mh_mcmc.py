import numpy as np
import numpy.random as npr

def mh_mcmc(p,q,steps,init,data=None,proposal_history_bool = False, debug_print = False, posterior_inference = True):
    params = init
    history = [None for _ in range(steps)]
    proposal_history = [None for _ in range(steps)]
    for index in range(steps):

        params_prop = q.s(params=params,data = data)
        if posterior_inference:
            log_alpha_num = q.ll(params,params_prop,data=data) + p.ll(data,params_prop) + p.prior.ll(params_prop)
            log_alpha_den = q.ll(params_prop,params,data=data) + p.ll(data,params) + p.prior.ll(params)
        else:
            log_alpha_num = q.ll(params,params_prop,data=data) + p.ll(params_prop)
            log_alpha_den = q.ll(params_prop,params,data=data) + p.ll(params)


        log_alpha = log_alpha_num - log_alpha_den
        log_coin_flip = np.log(npr.rand(1)[0])

        if debug_print:
            print("prop: [{:.3f}] | l_num: [{:.3f}] | l_den: [{:.3f}] | l_alpha: [{:.3f}] | l_coin: [{:.3f}]".format(params_prop[0],log_alpha_num,log_alpha_den,log_alpha,log_coin_flip))

        if (log_alpha > log_coin_flip): # accept
            params = params_prop
        history[index] = params

        if proposal_history_bool:
            proposal_history[index] = params_prop

    history = np.squeeze(np.array(history).reshape(len(params),-1))
    proposal_history = np.squeeze(np.array(proposal_history).reshape(len(params),-1))
    if debug_print:
        print(history)
    return history,proposal_history
