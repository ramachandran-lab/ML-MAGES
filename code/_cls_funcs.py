import numpy as np
import scipy as sp
import sys
from scipy.stats import multivariate_normal
from collections import Counter
import pandas as pd

def exp_log_Wishart(nu, L):
    """
    Compute the expected logarithm of the Wishart distribution.
    """
    p = L.shape[0]
    (sign, logabsdet) = np.linalg.slogdet(L)
    if p > 1:
        return np.sum([sp.special.psi((nu + 1 - i) / 2) for i in range(1, p + 1)]) + p * np.log(2) + logabsdet
    else:
        return sp.special.psi(nu / 2) + p * np.log(2) + np.log(L[0, 0])

def exp_log_Beta(a, b):
    """
    Compute the expected logarithm of the Beta distribution.
    """
    return sp.special.psi(a) - sp.special.psi(a + b)

def exp_log_Beta_minus(a, b):
    """
    Compute the expected logarithm of 1 minus the Beta distribution.
    """
    return sp.special.psi(b) - sp.special.psi(a + b)

def Wishart_logW(nu, L):
    """
    Compute the logarithm of the Wishart distribution constant.
    """
    p = L.shape[0]
    (sign, logabsdet) = np.linalg.slogdet(L)
    if sign<0:
        print(sign, logabsdet)
        sys.exit()
    L_det = sign * np.exp(logabsdet)        
    return -nu / 2 * L_det - (nu * p / 2) * np.log(2) - p * (p - 1) / 4 * np.log(np.pi) - np.sum([sp.special.loggamma((nu + 1 - i) / 2) for i in range(1, p + 1)])

def Beta_logB(a, b):
    """
    Compute the logarithm of the Beta distribution constant.
    """
    return sp.special.betaln(a, b)

def set_hyperparameters(p, alpha=0.5):
    """
    Set the hyperparameters for the model.
    """
    a_0 = 1
    b_0 = alpha
    L_0 = np.identity(p)
    nu_0 = 5
    L_0_inv = np.linalg.inv(L_0)
    return a_0, b_0, nu_0, L_0, L_0_inv

def initialize_VEM(p, K):
    """
    Initialize the parameters for the VEM algorithm.
    """
    a, b = np.random.rand(K), np.random.rand(K)
    nu = np.random.randint(1, 10, K)  
    L = list()
    for i in range(p): # initialize with dimension-specific cls
        mat = np.identity(p)*100
        mat[i,i] = 1
        B = mat/nu[i]
        B /= (np.max(B)/2)
        L.append(B) 
    for i in range(p,K): # initialize the rest
        C = (np.random.rand(p, p) - 0.5) * np.random.randint(1, 10, 1)[0]
        B = np.dot(C, C.transpose())
        B /= np.max(B)
        L.append(B)
    L_final = L 
    return a, b, nu, L_final

def run_VEM(X, K=25, alpha=0.5, niter=100, epsilon=1e-2):
    """
    Run the Variational Expectation-Maximization (VEM) algorithm.
    """
    N, p = X.shape
    a_0, b_0, nu_0, L_0, L_0_inv = set_hyperparameters(p, alpha=alpha)
    a, b, nu, L = initialize_VEM(p, K)
    X_outer = np.array([np.outer(X[i, :], X[i, :]) for i in range(N)])
    C1 = -p / 2 * np.log(2 * np.pi)
    ELBO_track = []
    E_log_A_ks = [exp_log_Wishart(nu[k], L[k]) for k in range(K)]
    E_log_v_ks = exp_log_Beta(a, b)
    E_log_1minusv_ks = exp_log_Beta_minus(a, b)
    E_log_1minusv_ks_cumsum = np.zeros_like(E_log_1minusv_ks)
    E_log_1minusv_ks_cumsum[1:] = np.cumsum(E_log_1minusv_ks)[:(K - 1)]
    nu_TrxxL = np.multiply(np.tensordot(X_outer, np.array(L), axes=((1,2),(1,2))),nu)
    for iteration in range(niter):
        # compute rho
        C2_all_logv = E_log_v_ks 
        C2_all_log1minusv = E_log_1minusv_ks_cumsum
        C2_all = C2_all_logv + C2_all_log1minusv
        C3_all = 0.5 * np.array(E_log_A_ks) 
        C4_all = -0.5 * nu_TrxxL
        rho = np.exp(((C1+C2_all+C3_all)[:, np.newaxis]+C4_all.T).T)
        r = rho / (np.sum(rho, axis=1, keepdims=True)+1e-8)
        N_ks = np.sum(r, axis=0)
        a = a_0 + N_ks # update a
        b_update = np.zeros_like(b) 
        N_ks_reverse_cumsum = np.cumsum(N_ks[::-1])[::-1]
        b_update[:(K - 1)] = N_ks_reverse_cumsum[1:]
        b = b_0 + b_update  # update b
        nu = nu_0 + N_ks # update nu
        rxxT_ks = np.tensordot(X_outer,r,axes=(0,0)) 
        for k in range(K): # update L
            L[k] = np.linalg.inv(L_0_inv + rxxT_ks[:,:,k])
        # compute ELBO and check convergence
        E_log_A_ks = [exp_log_Wishart(nu[k], L[k]) for k in range(K)]
        E_log_v_ks = exp_log_Beta(a, b)
        E_log_1minusv_ks = exp_log_Beta_minus(a, b)
        E_log_1minusv_ks_cumsum = np.zeros_like(E_log_1minusv_ks)
        E_log_1minusv_ks_cumsum[1:] = np.cumsum(E_log_1minusv_ks)[:(K - 1)]
        nu_TrxxL = np.multiply(np.tensordot(X_outer, np.array(L), axes=((1,2),(1,2))),nu)
        tr_rxxL = np.tensordot(rxxT_ks,np.array(L),axes=((0,1),(1,2)))
        tr_rxxL = np.diag(tr_rxxL)
        Elog_p_x = -0.5 * K * p * np.log(2 * np.pi) + 0.5 * np.sum(N_ks * np.array(E_log_A_ks)) + np.sum(-0.5 * nu_TrxxL * r) 
        Elog_p_z = np.sum(N_ks * (E_log_v_ks + E_log_1minusv_ks_cumsum))
        Elog_p_v = -(K - 1) * Beta_logB(a_0, b_0) + np.sum(((a_0 - 1) * E_log_v_ks + (b_0 - 1) * E_log_1minusv_ks)[:(K - 1)])
        Elog_p_A = K * Wishart_logW(nu_0, L_0) + 0.5 * (nu_0 - p - 1) * np.sum(E_log_A_ks) - 0.5 * np.sum(nu * np.array([np.trace(np.matmul(L_0_inv, L[k])) for k in range(K)]))
        Elog_q_z = np.sum(r * np.log(r + 1e-8))
        Elog_q_v = np.sum((-Beta_logB(a, b) + (a - 1) * E_log_v_ks + (b - 1) * E_log_1minusv_ks)[:(K - 1)])
        Elog_q_A = np.sum([Wishart_logW(nu[k], L[k]) for k in range(K)]) + np.sum((nu - p - 1) / 2 * E_log_A_ks) - 0.5 * p * np.sum(nu)
        ELBO = Elog_p_x + Elog_p_z + Elog_p_v + Elog_p_A - Elog_q_z - Elog_q_v - Elog_q_A
        ELBO_track.append(ELBO)
        if iteration > 2 and np.abs(ELBO - ELBO_track[-2]) < epsilon:
            break
    return r, a, b, nu, L, iteration

def sort_cls_dec_pi(r, a, b, nu, L, K):
    """
    Sort the results of the VEM algorithm.
    """
    Sigma = np.array([np.linalg.inv(nu[k]*L[k]) for k in range(K)])
    v_final = a/(a+b)
    v_final[-1] = 1
    pi = np.array([v_final[k]*np.prod(1-v_final[:k]) if k>0 else v_final[k] for k in range(K)])
    # sort classes by decreasing pi
    sort_idx = np.argsort(pi)[::-1] 
    Sigma, pi = Sigma[sort_idx], pi[sort_idx]
    r = r[:,sort_idx]
    return Sigma, pi, r

def predict_cls(X, Sigma):
    """
    Predict the component from the mixture.
    """
    p = X.shape[1]
    K = Sigma.shape[0]
    probs = np.array([multivariate_normal.pdf(X, mean=np.zeros(p), cov=Sigma[k], allow_singular=True) for k in range(K)]).T
    cls = np.argmax(probs, axis=1)
    return cls, probs

def get_cls(X, Sigma, pi, r, min_K = 1, pi_sum_thre = 1e-2):
    """
    Get the class labels based on the VEM results.
    """
    pi_cutpoint = np.cumsum(pi)
    pred_K = np.where(pi_cutpoint>1-pi_sum_thre)[0][0]+1
    if pred_K<min_K:
        pred_K = min_K
    truncate_Sigma = Sigma[:pred_K]
    truncate_pi = pi[:pred_K]
    truncate_pi = truncate_pi/np.sum(truncate_pi) # renormalize pi
    r = r[:,:pred_K]
    pred_cls = np.argmax(r,axis=1)
    sort_idx = np.argsort([np.trace(s) for s in truncate_Sigma])[::-1]
    truncate_Sigma, truncate_pi = truncate_Sigma[sort_idx], truncate_pi[sort_idx]
    pred_cls, _ = predict_cls(X,truncate_Sigma)
    return truncate_Sigma, truncate_pi, pred_K, pred_cls

def compute_bic(X, pred_K, pred_cls, truncate_Sigma):
    """
    Compute the AIC and BIC of the model (used for compare same-K models).
    """
    N, p = X.shape
    LL = np.sum([np.sum(multivariate_normal.logpdf(X[pred_cls==k], np.zeros(p), truncate_Sigma[k])) for k in range(pred_K)])
    c = 1+(1+p)*p/2
    BIC = np.log(N)*(c*pred_K)-2*LL
    return BIC

def infmix_clustering(data, K = 30, alpha = 0.5, niter = 1000, eps = 1e-4, n_runs = 10):
    """
    Perform infinite mixture clustering on the given data.
    """
    scale = 1/np.std(data, axis=0).mean()
    X = scale*data
    if X.ndim<2:
        X = np.expand_dims(X, axis=1)
    result_runs = list()
    for run in range(n_runs):
        r, a, b, nu, L, iteration = run_VEM(X, K, alpha=alpha, niter=niter, epsilon=eps)
        Sigma, pi, r = sort_cls_dec_pi(r, a, b, nu, L, K)
        truncate_Sigma, truncate_pi, pred_K, pred_cls = get_cls(X, Sigma, pi, r)
        BIC = compute_bic(X, pred_K, pred_cls, truncate_Sigma)
        truncate_Sigma /= scale**2
        cls_cnt = pd.Series(pred_cls).value_counts()
        cls_perc = np.zeros(pred_K)
        for k in range(pred_K):
            if k in cls_cnt.index:
                cls_perc[k] = cls_cnt.loc[k]/cls_cnt.sum()
        result_runs.append({"Sigma": truncate_Sigma, "pi": truncate_pi, "pred_K": pred_K, "pred_cls": pred_cls, "BIC": BIC})
    # get consensus clustering results with the majority K
    BIC_results = np.array([r['BIC'] for r in result_runs])
    K_results = np.array([r['pred_K'] for r in result_runs])
    pred_K_cnt = Counter(K_results)
    all_pred_K = np.sort(np.unique(K_results))
    all_pred_K_cnt = np.array([pred_K_cnt[k] for k in all_pred_K])
    idx_sort_cnt = np.argsort(-all_pred_K_cnt)
    major_K, major_K_cnt = all_pred_K[idx_sort_cnt[0]], all_pred_K_cnt[idx_sort_cnt[0]]
    if major_K==1:
        K_chosen = np.min([k for k in all_pred_K if k>1])
    else:
        k_candidate = [k for k in all_pred_K if pred_K_cnt[k]==major_K_cnt]
        K_chosen = np.min(k_candidate)
    idx_of_K = np.where(K_results==K_chosen)[0]
    min_BIC_idx = np.argsort(BIC_results[idx_of_K])[0]
    best_results = result_runs[idx_of_K[min_BIC_idx]]
    truncate_Sigma, truncate_pi, pred_K, pred_cls = best_results["Sigma"], best_results["pi"], best_results["pred_K"], best_results["pred_cls"]
    return truncate_Sigma, truncate_pi, pred_K, pred_cls
