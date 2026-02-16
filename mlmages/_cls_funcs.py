import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from collections import Counter
import time
from typing import Tuple
from ._util_funcs import check_finite


def exp_log_Wishart(nu: float, L: np.ndarray) -> float:
    """
    Compute the expected logarithm of the Wishart distribution.

    Parameters:
    - nu (float): Degrees of freedom parameter.
    - L (ndarray): Scale matrix.

    Returns:
    - float: Expected logarithm of the Wishart distribution.
    """
    p = L.shape[0]
    (sign, logabsdet) = np.linalg.slogdet(L)
    assert sign>= 0  
    if p > 1:
        return np.sum([sp.special.psi((nu + 1 - i) / 2) for i in range(1, p + 1)]) + p * np.log(2) + logabsdet
    else:
        return sp.special.psi(nu / 2) + p * np.log(2) + np.log(L[0, 0])


def exp_log_Beta(a: float, b: float) -> float:
    """
    Compute the expected logarithm of the Beta distribution.

    Parameters:
    - a (float): Shape parameter a.
    - b (float): Shape parameter b.

    Returns:
    - float: Expected logarithm of the Beta distribution.
    """
    return sp.special.psi(a) - sp.special.psi(a + b)


def exp_log_Beta_minus(a: float, b: float) -> float:
    """
    Compute the expected logarithm of 1 minus the Beta distribution.

    Parameters:
    - a (float): Shape parameter a.
    - b (float): Shape parameter b.

    Returns:
    - float: Expected logarithm of 1 minus the Beta distribution.
    """
    return sp.special.psi(b) - sp.special.psi(a + b)


def Wishart_logW(nu: float, L: np.ndarray) -> float:
    """
    Compute the logarithm of the Wishart distribution constant.

    Parameters:
    - nu (float): Degrees of freedom parameter.
    - L (ndarray): Scale matrix.

    Returns:
    - float: Logarithm of the Wishart distribution constant.
    """
    p = L.shape[0]
    (sign, logabsdet) = np.linalg.slogdet(L)
    assert sign>=0
    L_det = sign * np.exp(logabsdet)        
    return -nu / 2 * L_det - (nu * p / 2) * np.log(2) - p * (p - 1) / 4 * np.log(np.pi) - np.sum([sp.special.loggamma((nu + 1 - i) / 2) for i in range(1, p + 1)])


def Beta_logB(a: float, b: float) -> float:
    """
    Compute the logarithm of the Beta distribution constant.

    Parameters:
    - a (float): Shape parameter a.
    - b (float): Shape parameter b.

    Returns:
    - float: Logarithm of the Beta distribution constant.
    """
    return sp.special.betaln(a, b)


def set_hyperparameters(p: int, alpha: float=0.5) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Set the hyperparameters for the model.

    Parameters:
    - p (int): Dimensionality of the data.
    - alpha (float): Hyperparameter alpha.

    Returns:
    - tuple: Hyperparameters (a_0, b_0, nu_0, L_0, L_0_inv).
    """
    a_0 = 1.0
    b_0 = alpha
    L_0 = np.identity(p)
    nu_0 = 5.0
    L_0_inv = np.linalg.inv(L_0)
    return a_0, b_0, nu_0, L_0, L_0_inv


def initialize_VEM(p: int, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Initialize the parameters for the VEM algorithm.

    Parameters:
    - p (int): Dimensionality of the data.
    - K (int): Number of components.

    Returns:
    - tuple: Initial parameters (a, b, nu, L).
    """
    a, b = np.random.rand(K), np.random.rand(K)
    nu = np.random.randint(p, 8+p, K)  
    L = list()
    for i in range(p): # initialize with dimension-specific cls
        mat = np.identity(p)*100
        mat[i,i] = 1
        B = mat/nu[i]
        B /= (np.max(B)/2)
        L.append(B) 
    for k in range(p,K):
        C = (np.random.rand(p, p) - 0.5) * np.random.randint(1, 10, 1)[0]
        B = np.dot(C, C.transpose())
        B /= np.max(B)
        L.append(B)
    L_final = L 

    return a, b, nu, L_final


def run_VEM(X: np.ndarray, K: int=20, alpha: float=0.5, niter: int=1000, epsilon: float=1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run the Variational Expectation-Maximization (VEM) algorithm.

    Parameters:
    - X (ndarray): Input data.
    - K (int): Number of components.
    - alpha (float): Hyperparameter alpha.
    - niter (int): Maximum number of iterations.
    - epsilon (float): Convergence threshold.

    Returns:
    - tuple: Results of the VEM algorithm (r, a, b, nu, L) #parameters_track, ELBO_parts_track, ELBO_track.
    """
    N, p = X.shape
    a_0, b_0, nu_0, L_0, L_0_inv = set_hyperparameters(p, alpha=alpha)
    a, b, nu, L = initialize_VEM(p, K)
    X_outer = np.array([np.outer(X[i, :], X[i, :]) for i in range(N)])
    C1 = -p / 2 * np.log(2 * np.pi)
    check_finite("C1", C1)
    
    parameters_track = []
    ELBO_track = []
    ELBO_parts_track = []

    E_log_A_ks = [exp_log_Wishart(nu[k], L[k]) for k in range(K)]
    E_log_v_ks = np.vectorize(exp_log_Beta)(a, b)
    E_log_1minusv_ks = np.vectorize(exp_log_Beta_minus)(a, b)
    E_log_1minusv_ks_cumsum = np.zeros_like(E_log_1minusv_ks)
    E_log_1minusv_ks_cumsum[1:] = np.cumsum(E_log_1minusv_ks)[:(K - 1)]
    nu_TrxxL = np.multiply(np.tensordot(X_outer, np.array(L), axes=((1,2),(1,2))),nu)
    
    for iteration in range(niter):
        # compute rho
        C2_all_logv = E_log_v_ks 
        Beta_minus = E_log_1minusv_ks 
        Beta_minus_cumsum = np.cumsum(Beta_minus)
        C2_all_log1minusv = E_log_1minusv_ks_cumsum
        C2_all = C2_all_logv + C2_all_log1minusv
        C3_all = 0.5 * np.array(E_log_A_ks) 
        C4_all = -0.5 * nu_TrxxL

        check_finite("C2_all", C2_all)
        check_finite("C3_all", C3_all)
        check_finite("C4_all", C4_all)
        
        # rho = np.exp(((C1+C2_all+C3_all)[:, np.newaxis]+C4_all.T).T)
        # r = rho / (np.sum(rho, axis=1, keepdims=True)+1e-8)
        log_rho = ((C1 + C2_all + C3_all)[:, None] + C4_all.T).T   # shape (N, K)
        check_finite("log_rho", log_rho, max_abs=1e6)
        # mx = np.max(log_rho)
        # if mx > 700:
        #     raise FloatingPointError(f"log_rho too large for exp: max={mx}")
        # rho = np.exp(log_rho)
        # r = rho / (np.sum(rho, axis=1, keepdims=True)+1e-8)
        log_r = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        r = np.exp(log_r)

        # log_r = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        # r = np.exp(log_r)   # rows sum to 1, stable
        # eps = 1e-8
        # lse = logsumexp(log_rho, axis=1, keepdims=True)              # log(sum(exp(log_rho)))
        # log_denom = np.logaddexp(lse, np.log(eps))                   # log(exp(lse) + eps)
        # r = np.exp(log_rho - log_denom)
        N_ks = np.sum(r, axis=0)
        
        a = a_0 + N_ks # update a
        b_update = np.zeros_like(b) 
        N_ks_reverse_cumsum = np.cumsum(N_ks[::-1])[::-1]
        b_update[:(K - 1)] = N_ks_reverse_cumsum[1:]
        b = b_0 + b_update  # update b
        nu = nu_0 + N_ks # update nu
        rxxT_ks = np.tensordot(X_outer,r,axes=(0,0)) 
        for k in range(K): # update L
            # L[k] = np.linalg.inv(L_0_inv + rxxT_ks[:,:,k])
            A = L_0_inv + rxxT_ks[:, :, k]
            A = 0.5 * (A + A.T)                 # enforce symmetry
            A = A + 1e-8 * np.eye(p)            # jitter to ensure SPD
            Lk = np.linalg.inv(A)
            L[k] = 0.5 * (Lk + Lk.T)
        parameters_track.append((r, a, b, nu, L))
        
        # compute ELBO
        E_log_A_ks = [exp_log_Wishart(nu[k], L[k]) for k in range(K)]
        E_log_v_ks = np.vectorize(exp_log_Beta)(a, b)
        E_log_1minusv_ks = np.vectorize(exp_log_Beta_minus)(a, b)
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
        Elog_q_v = np.sum((-np.vectorize(Beta_logB)(a, b) + (a - 1) * E_log_v_ks + (b - 1) * E_log_1minusv_ks)[:(K - 1)])
        Elog_q_A = np.sum([Wishart_logW(nu[k], L[k]) for k in range(K)]) + np.sum((nu - p - 1) / 2 * E_log_A_ks) - 0.5 * p * np.sum(nu)
        ELBO_parts = [Elog_p_x, Elog_p_z, Elog_p_v, Elog_p_A, Elog_q_z, Elog_q_v, Elog_q_A]
        ELBO = Elog_p_x + Elog_p_z + Elog_p_v + Elog_p_A - Elog_q_z - Elog_q_v - Elog_q_A
        ELBO_track.append(ELBO)
        ELBO_parts_track.append(ELBO_parts)
        
        if iteration > 2 and np.abs(ELBO - ELBO_track[-2]) < epsilon:
            break

    return r, a, b, nu, L 


def sort_results(r: np.ndarray, a: np.ndarray, b: np.ndarray, nu: np.ndarray, L: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort the results of the VEM algorithm.

    Parameters:
    - r (ndarray): Posterior probabilities.
    - a (ndarray): Shape parameter a.
    - b (ndarray): Shape parameter b.
    - nu (ndarray): Degrees of freedom parameter.
    - L (ndarray): Scale matrix.
    - K (int): Number of components.

    Returns:
    - tuple: Sorted results (Sigma, pi, r).
    """
    Sigma = np.array([np.linalg.inv(nu[k]*L[k]) for k in range(K)])
    v_final = a/(a+b)
    v_final[-1] = 1
    pi = np.array([v_final[k]*np.prod(1-v_final[:k]) if k>0 else v_final[k] for k in range(K)])
    assert(np.abs(1-np.sum(pi))<1e-6)

    # sort classes by pi
    sort_idx = np.argsort(pi)[::-1] 
    Sigma, pi = Sigma[sort_idx], pi[sort_idx]
    r = r[:,sort_idx]

    return Sigma, pi, r


def pred_mixture(X: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the component from the mixture.

    Parameters:
    - X (ndarray): Input data.
    - Sigma (ndarray): Covariance matrices.

    Returns:
    - tuple: Predicted class labels and probabilities.
    """
    p = X.shape[1]
    K = Sigma.shape[0]
    probs = np.array([multivariate_normal.pdf(X, mean=np.zeros(p), cov=Sigma[k], allow_singular=True) for k in range(K)]).T
    cls = np.argmax(probs, axis=1)
    return cls, probs


def get_cls(X: np.ndarray, Sigma: np.ndarray, pi: np.ndarray, r: np.ndarray, min_K: int = 1) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Get the class labels based on the VEM results.

    Parameters:
    - X (ndarray): Input data.
    - Sigma (ndarray): Covariance matrices.
    - pi (ndarray): Mixture proportions.
    - r (ndarray): Posterior probabilities.

    Returns:
    - tuple: (ndarray, ndarray, int, ndarray) representing truncated Sigma, truncated pi, predicted number of components, and class labels.
    """
    # truncate pi to some K with sum(pi_k) large enough
    pi_cutpoint = np.cumsum(pi)
    pred_K = np.where(pi_cutpoint>1-1e-2)[0][0]+1
    if pred_K<min_K:
        pred_K = min_K
    truncate_Sigma = Sigma[:pred_K]
    truncate_pi = pi[:pred_K]
    truncate_pi = truncate_pi/np.sum(truncate_pi) #renormalize Pi
    r = r[:,:pred_K]
    pred_cls = np.argmax(r,axis=1)

    # sort classes by Sigma #linalg.det
    sort_idx = np.argsort([np.trace(s) for s in truncate_Sigma])[::-1]
    truncate_Sigma, truncate_pi = truncate_Sigma[sort_idx], truncate_pi[sort_idx]

    # get cls
    pred_cls, _ = pred_mixture(X,truncate_Sigma)

    return truncate_Sigma, truncate_pi, pred_K, pred_cls


def compute_IC(X: np.ndarray, pred_K: int, pred_cls: np.ndarray, truncate_Sigma: np.ndarray) -> Tuple[float, float]:
    """
    Compute the AIC and BIC of the model (used for compare same-K models).

    Parameters:
    - X (ndarray): Input data.
    - pred_K (int): Predicted number of components.
    - pred_cls (ndarray): Predicted class labels for the data.
    - truncate_Sigma (ndarray): Covariance matrices for the predicted components.

    Returns:
    - tuple: (float, float) representing AIC and BIC.
    """
    N, p = X.shape

    # compute log-likelihood    
    LL = np.sum([np.sum(multivariate_normal.logpdf(X[pred_cls==k], np.zeros(p), truncate_Sigma[k])) for k in range(pred_K)])

    # compute AIC and BIC
    c = 1+(1+p)*p/2
    AIC = 2*(c*pred_K)-2*LL
    BIC = np.log(N)*(c*pred_K)-2*LL

    return AIC, BIC


def infmix_clustering(data: np.ndarray, K: int = 30, alpha: float = 0.5, niter: int = 1000, eps: float = 1e-4, n_runs: int = 100) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Perform infinite mixture model clustering.
    Parameters:
    - data (ndarray): Input data.
    - K (int): Maximum number of components.
    - alpha (float): Concentration parameter.
    - niter (int): Number of iterations.
    - eps (float): Convergence threshold.
    - n_runs (int): Number of random restarts.

    Returns:
    - tuple: (ndarray, ndarray, int, ndarray) representing truncated Sigma, truncated pi, predicted number of components, and class labels.
    """

    result_runs = list()

    for run in range(n_runs):

        # fit model
        scale = 1/np.std(data, axis=0).mean()
        X = scale*data
        # assure X is Nxp
        if X.ndim<2:
            X = np.expand_dims(X, axis=1)

        r, a, b, nu, L = run_VEM(X, K, alpha=alpha, niter=niter, epsilon=eps)

        Sigma, pi, r = sort_results(r, a, b, nu, L, K)
        truncate_Sigma, truncate_pi, pred_K, pred_cls = get_cls(X, Sigma, pi, r)
        AIC, BIC = compute_IC(X, pred_K, pred_cls, truncate_Sigma)

        truncate_Sigma /= scale**2

        cls_cnt = pd.Series(pred_cls).value_counts()
        cls_perc = np.zeros(pred_K)
        for k in range(pred_K):
            if k in cls_cnt.index:
                cls_perc[k] = cls_cnt.loc[k]/cls_cnt.sum()

        result_runs.append({"Sigma": truncate_Sigma,
                            "pi": truncate_pi,
                            "pred_K": pred_K,
                            "pred_cls": pred_cls,
                            "AIC": AIC, "BIC": BIC})

    # choose best run
    BIC_list = [result_runs[run]["BIC"] for run in range(n_runs)]
    pred_K_list = [result_runs[run]["pred_K"] for run in range(n_runs)]
    Sigma_list =[result_runs[run]["Sigma"] for run in range(n_runs)]

    pred_K_list = np.array(pred_K_list)
    BIC_list = np.array(BIC_list)

    # find the majority K
    pred_K_cnt = Counter(pred_K_list)
    all_pred_K = np.sort(np.unique(pred_K_list))
    all_pred_K_cnt = np.array([pred_K_cnt[k] for k in all_pred_K])
    idx_sort_cnt = np.argsort(-all_pred_K_cnt)
    major_K, major_K_cnt = all_pred_K[idx_sort_cnt[0]], all_pred_K_cnt[idx_sort_cnt[0]]

    if major_K==1:
        K_chosen = np.min([k for k in all_pred_K if k>1])
    else:
        # break tie if exist: go for the larger one
        k_candidate = [k for k in all_pred_K if pred_K_cnt[k]==major_K_cnt]
        K_chosen = np.max(k_candidate)

    print("K chosen:", K_chosen)

    # get clustering with smallest BIC within K
    BIC_results = np.array([r['BIC'] for r in result_runs])
    K_results = np.array([r['pred_K'] for r in result_runs])
    idx_of_K = np.where(K_results==K_chosen)[0]
    min_BIC_idx = np.argsort(BIC_results[idx_of_K])[0]
    rep_run_idx = idx_of_K[min_BIC_idx]

    best_results = result_runs[rep_run_idx]
    pred_cls = best_results["pred_cls"]
    pred_K = best_results["pred_K"]
    truncate_Sigma = best_results["Sigma"]
    truncate_pi = best_results["pi"]

    return truncate_Sigma, truncate_pi, pred_K, pred_cls


def extract_multivariate_components(betas_regularized: np.ndarray, K: int=20, n_runs: int=25):
    """
    Extract components from regularized betas using infinite mixture model (multivariate).
    Parameters:
    - betas_regularized (ndarray): Regularized beta values.
    - K (int): Maximum number of components.
    - n_runs (int): Number of random restarts.
    Returns:
    - tuple: (ndarray, ndarray, int, ndarray) representing truncated Sigma, truncated pi, predicted number of components, and class labels.
    """

    # extract component
    tic = time.time()
    truncate_Sigma, truncate_pi, pred_K, pred_cls = infmix_clustering(betas_regularized, K = K, alpha = 0.5, niter = 1000, eps = 1e-3, n_runs = n_runs)
    toc = time.time()
    print("Component extraction takes {:.1f}s.".format((toc-tic)))

    return truncate_Sigma, truncate_pi, pred_K, pred_cls


def threshold_zeros(x: np.ndarray, zero_cutoff: float=1e-4):
    """
    Threshold the input array to zero out small values.
    Parameters:
    - x (ndarray): Input array.
    - zero_cutoff (float): Threshold value to consider as zero.
    Returns:
    - ndarray: Thresholded array.
    """ 
    y = np.zeros_like(x)
    y[np.abs(x)>zero_cutoff] = x[np.abs(x)>zero_cutoff]
    return y


def threshold_beta(b: np.ndarray, zero_cutoff: float=1e-4, any_zero: bool=True) -> np.ndarray:
    """
    Threshold the beta array to remove rows with all or any zeros.
    Parameters:
    - b (ndarray): Input beta array.
    - zero_cutoff (float): Threshold value to consider as zero.
    - any_zero (bool): If True, remove rows with any zeros; if False, remove rows with all zeros.
    Returns:
    - ndarray: Thresholded beta array.
    """
    if b.ndim == 1:
        b = np.expand_dims(b, axis=1)  # or b = b[:, None]
    b_filtered = threshold_zeros(b, zero_cutoff=zero_cutoff)
    is_nz = np.abs(b_filtered)>0
    if any_zero:
        b_nz = b_filtered[np.any(is_nz, axis=1),:] # any nz
    else:
        b_nz = b_filtered[np.all(is_nz, axis=1),:] # all nz
    return np.squeeze(b_nz)


def adjust_zero_threshold(b: np.ndarray, init_zero_cutoff: float=1e-3, any_zero: bool=True, 
                          min_nz: int=1000, max_nz: int=5000, adjust_scale: float=2, max_iter: int=10) -> Tuple[np.ndarray, float]:
    """
    Adjust the zero threshold to ensure the number of non-zero rows is within a specified range.
    Parameters:
    - b (ndarray): Input beta array.
    - init_zero_cutoff (float): Initial threshold value to consider as zero.
    - any_zero (bool): If True, remove rows with any zeros; if False, remove rows with all zeros.
    - min_nz (int): Minimum number of non-zero rows desired.
    - max_nz (int): Maximum number of non-zero rows desired.
    - adjust_scale (float): Scale factor to adjust the threshold.
    - max_iter (int): Maximum number of iterations to adjust the threshold.
    Returns:
    - tuple: (ndarray, float) representing the thresholded beta array and the final zero cutoff value.
    """
    zero_cutoff = init_zero_cutoff
    b_nz = threshold_beta(b, zero_cutoff=zero_cutoff, any_zero=any_zero)
    i_iter = 0
    while len(b_nz.shape)==0 or ((b_nz.shape[0]<min_nz or b_nz.shape[0]>max_nz) and i_iter<max_iter):
        if len(b_nz.shape)==0 or b_nz.shape[0]<min_nz:
            zero_cutoff /= adjust_scale
        else:
            zero_cutoff *= adjust_scale
        b_nz = threshold_beta(b, zero_cutoff=zero_cutoff, any_zero=any_zero)
        i_iter += 1

    return b_nz, zero_cutoff
    