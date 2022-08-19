import numpy as np
from scipy.special import digamma
from scipy.optimize import brentq
from scipy.stats import norm, gamma

### root-finding equation
def fn(alpha, target):
    return np.log(alpha) - digamma(alpha) - target


### update parameters in gamma distribution
def update_gmm_pars(x, wt):
    x = np.array(x)
    wt = np.array(wt)
    tp_s = np.sum(wt)
    tp_t = np.sum(wt * x)
    tp_u = np.sum(wt * np.log(x))
    tp_v = -tp_u / tp_s - np.log(tp_s / tp_t)
    alpha = brentq(fn, 0.01, 5, args=tuple([tp_v]))    
    beta = tp_s / tp_t * alpha
    return alpha, beta

### estimate parameters in the mixture distribution
def get_mix(allscores, ratio_init=0.7, min_eps = 0.5):
    # regard all zero as clean
    scores = [s for s in allscores if s>0] # remove zero to fit better
    # estimate parameters
    inits = np.zeros(5)
    min_eps = 0.5
    ratio_init = 0.5
    inits[0] = ratio_init # propotion of gamma (clean)
    if inits[0] == 0: 
        inits[0] = 0.01
    inits[1], inits[2] = 0.5, 0.1 
    scores_rm = [s for s in scores if s > np.quantile(scores, ratio_init)]
    inits[3], inits[4] = np.mean(scores_rm), 1/np.std(scores_rm) # gaussian parameter
    paramt = inits
    print(paramt)
    eps = 10
    iter = 0
    loglik_old = 0
    eps_list = []
    while eps > min_eps:
        wt0, wt1 = calculate_weight(scores, paramt)
        paramt[0] = np.mean(wt0) # ratio
        paramt[3] = np.sum(np.array(wt1) * np.array(scores))/np.sum(wt1)
        paramt[4] = 1/np.sqrt(np.sum(np.array(wt1) * (np.array(scores) - paramt[3])**2)/np.sum(wt1))
        paramt[1], paramt[2] = update_gmm_pars(x=scores, wt=wt0)
        # compute log likelihood
        loglik = np.sum(np.log10(dmix(scores, paramt)))
        eps = (loglik - loglik_old)**2 
        eps_list.append(eps)
        loglik_old = loglik
        iter = iter + 1
        if iter > 100: 
            break
    return paramt # "rate", "alpha", "beta", "mu", "sigma"
    
def rmix(pars, n):
    n1 = int(np.ceil(n * pars[0]))
    n2 = n - n1
    x1 = np.random.gamma(shape = pars[1], scale = 1/pars[2], size=n1)
    x2 = np.random.normal(loc = pars[3], scale = 1/pars[4], size=n2)
    return list(x1) + list(x2), ['gamma' for i in range(n1)] + ['gauss' for i in range(n2)]

def dmix(x, pars):
    return pars[0] * gamma.pdf(x, a = pars[1], scale = 1/pars[2]) + (1 - 
        pars[0]) * norm.pdf(x, loc = pars[3], scale = 1/pars[4])


def calculate_weight(x, paramt):
    pz1 = paramt[0] * gamma.pdf(x, a = paramt[1], scale = 1/paramt[2])
    pz2 = (1 - paramt[0]) * norm.pdf(x, loc = paramt[3], scale = 1/paramt[4])
    pz = pz1/(pz1 + pz2)
    return pz, 1 - pz


def get_cutoff(allscores, recall = 0.7, thred = 0, usef1 = True):
    # maximize the precision give recall>0.7
    '''
    recall  = p(x<cutoff | x is gauss)
            = gauss.cdf(cutoff) + \ 

    precision  = p(x is gauss | x<cutoff) 
               = p(x<cutoff|x is gauss)p(x is gauss) / p(x<cutoff)  
               = recall * p(x is gauss) / p(x<cutoff)                       
    ''' 
    # if np.max(scores)<0:
    #     scores = [-s for s in scores]
    scores = [s for s in allscores if s>thred]    
    paramt = get_mix(scores)
    cutoff = norm.ppf(1-recall, loc=paramt[3], scale=1/paramt[4])       
    def pre(c):
        recall = 1-norm.cdf(c, loc = paramt[3], scale = 1/paramt[4])
        precision = recall * (1-paramt[0]) / (paramt[0]*(1-gamma.cdf(c, a = paramt[1], scale = 1/paramt[2])) + (1-paramt[0])*(1-norm.cdf(c, loc = paramt[3], scale = 1/paramt[4])))
        f1 = 2*recall*precision / (recall + precision) if recall*precision!=0 else 0
        #print(recall, precision)
        return precision, recall, f1


    precision, recall, f1 = pre(cutoff)
    scores = np.array(scores)
    # find the best cutoff
    if usef1:
        res_scores = sorted(scores[scores<cutoff])
        f1max = f1
        for r in res_scores:
          _,_,f1 = pre(r)
          if f1>f1max:
            f1max=f1
            cutoff = r    
    else:    
        if precision>recall:
            res_scores = sorted(scores[scores<cutoff])
            left = 0
            right = len(res_scores)
            # binary-search for cutoff
            while left<right:
                mid = left+(right-left)//2
                cutoff = res_scores[mid]
                precision, recall,_ = pre(cutoff)
                if precision > recall:
                    right = mid
                else:
                    left = mid+1        
    print('best precision %.1f, recall %.1f'%(precision, recall))
    return cutoff  
