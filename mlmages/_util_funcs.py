import os
import re
import numpy as np
import scipy as sp
import scipy.stats as stats
import itertools
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, r2_score, precision_recall_fscore_support

import pandas as pd
import numpy as np

def binary_combinations(n: int):
    return [comb for comb in itertools.product([0,1], repeat=n) if any(comb)]

def disp_params(args: argparse.Namespace, title: str = ""):
    print(f" {title} ".center(50, '='))
    print("--------------------------------------------------") 
    print("Parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("--------------------------------------------------") 


def parse_file_list(files: list[str]) -> list[str]:
    # Case 1: passed a single file, and it exists and ends with _files.txt
    if len(files) == 1 and files[0].endswith("_files.txt") and os.path.isfile(files[0]):
        with open(files[0]) as f:
            return [line.strip() for line in f if line.strip()]
    # Case 2: passed multiple file names directly
    return files


def load_gwas_file(gwas_file: str, chroms=[]):

    col_map = {
        "CHR": ["CHR", "CHROM", "CHROMOSOME"],
        "BETA": ["BETA", "EFFECT", "B", "LOG_ODDS", "OR"],
        "SE": ["SE", "STDERR", "STD_ERR", "STANDARD_ERROR"]
    }
    gwas_results = pd.read_csv(gwas_file, sep=None, engine="python")
    gwas_results.columns = gwas_results.columns.str.strip().str.upper()

    def find_col(possible_names):
        for name in possible_names:
            if name in gwas_results.columns:
                return name
        raise ValueError(f"Missing any of {possible_names} in file")

    chrom_col = find_col(col_map["CHR"])
    beta_col  = find_col(col_map["BETA"])
    se_col    = find_col(col_map["SE"])

    if len(chroms) > 0:
        gwas_results = gwas_results[gwas_results[chrom_col].isin(chroms)]

    beta = gwas_results[beta_col].astype(float).values
    if beta_col == "OR":
        print("Converting OR to BETA (log OR)")
        beta = np.log(beta)
    se = gwas_results[se_col].astype(float).values

    return gwas_results, beta, se


def parse_model_file(model_file):
    # strip directory + extension
    basename = os.path.splitext(os.path.basename(model_file))[0]

    # regex: top<number>_<number>L
    m = re.search(r"top(\d+)_([0-9]+)L", basename)
    if not m:
        raise ValueError(f"Cannot parse top_r / n_layer from: {basename}")

    top_r = int(m.group(1))
    n_layer = int(m.group(2))
    return top_r, n_layer


def match_distribution(src:np.ndarray, ref:np.ndarray) -> np.ndarray:
    """
    Match the distribution of `src` to that of `ref` using quantile mapping.
    """

    # Compute source percentiles
    src_percentiles = np.linspace(0, 100, len(src))
    
    # Interpolate reference's quantile function
    ref_percentiles = np.linspace(0, 100, len(ref))
    ref_sorted = np.sort(ref)
    ref_values = np.interp(src_percentiles, ref_percentiles, ref_sorted)
    
    # Rank source values, then map to reference quantiles
    src_ranks = np.argsort(np.argsort(src))
    matched = ref_values[src_ranks]
    
    return matched


def scale_by_quantile(x:np.ndarray, ref:np.ndarray, q=0.01) -> np.ndarray:
    
    trans_params = (np.quantile(x,q),np.quantile(x,1-q),np.quantile(ref,q),np.quantile(ref,1-q))
    x_scaled = (x-trans_params[0])/(trans_params[1]-trans_params[0])*(trans_params[3]-trans_params[2])+trans_params[2]
    
    return x_scaled

def scale_laplace(data_to_scale:np.ndarray, loc_ref:float, scale_ref:float) -> np.ndarray:
    # get ECDF of the simulated data
    sim_ecdf = sp.stats.ecdf(data_to_scale.flatten()).cdf
    # scale betas
    data_trans_probs = sim_ecdf.evaluate(data_to_scale)
    data_trans = sp.stats.laplace.ppf(data_trans_probs, loc=loc_ref, scale=scale_ref)
    data_trans[np.isinf(data_trans)] = np.nanmax(data_trans[~np.isinf(data_trans)])
    return data_trans


def scale_true_beta(true_betas:np.ndarray, obs_betas:np.ndarray, scaled_obs_betas:np.ndarray, q: float=0.0) -> np.ndarray:
    if len(true_betas.shape)<2:
        tb = true_betas[:, np.newaxis] 
    else:
        tb = true_betas
    if len(obs_betas.shape)<2:
        ob = obs_betas[:, np.newaxis]   
    else:
        ob = obs_betas
    if len(scaled_obs_betas.shape)<2:
        sob = scaled_obs_betas[:, np.newaxis]   
    else:
        sob = scaled_obs_betas
        
    sob_quantiles_range = np.quantile(sob, 1-q, axis=0)-np.quantile(sob, q, axis=0)
    tb_quantiles_range = np.quantile(tb, 1-q, axis=0)-np.quantile(tb, q, axis=0)
    scales = np.divide(sob_quantiles_range,tb_quantiles_range)
    scaled_true_betas = np.multiply(tb,scales)
    
    return scaled_true_betas.squeeze()    


def evaluate_perf(y_true:np.ndarray, y_pred:np.ndarray) -> dict:
    assert(y_true.shape==y_pred.shape)
    # RMSE
    sqr_err = (y_true-y_pred)**2
    rmse = np.sqrt(np.mean(sqr_err, axis=0))
    # weighted RMSE
    is_true_nz = y_true!=0
    frac_true_nz = np.sum(is_true_nz, axis=0)/y_true.shape[0]
    weighted_mean_sum_sqr_err = [(np.mean(sqr_err[is_true_nz[:,i],i])*(1-frac_true_nz[i]) + np.mean(sqr_err[~is_true_nz[:,i],i])*frac_true_nz[i]) for i in range(y_true.shape[1])]
    wrmse = np.sqrt(weighted_mean_sum_sqr_err)
    # others
    p_corr = [np.corrcoef(y_true[:,idx],y_pred[:,idx])[0,1] for idx in range(y_pred.shape[1])]
    auc = [np.nan for idx in range(y_pred.shape[1])]
    r2 = [np.nan for idx in range(y_pred.shape[1])]
    try:
        auc = [roc_auc_score(y_true[:,idx]!=0,np.abs(y_pred[:,idx])) for idx in range(y_pred.shape[1])]
        r2 = [r2_score(y_true[:,idx],y_pred[:,idx]) for idx in range(y_pred.shape[1])]
    except: 
        print("all labels are the same")
    
    ap = [average_precision_score(y_true[:,idx]!=0,np.abs(y_pred[:,idx])) for idx in range(y_pred.shape[1])]

    return {'rmse': rmse, 'wrmse': wrmse, 'p_corr': np.array(p_corr),
           'auc': np.array(auc), 'r2': np.array(r2), 'ap': np.array(ap)}

def compute_mean_prec(y_true:np.ndarray, y_pred:np.ndarray, base_rec:np.ndarray) -> np.ndarray:
    assert(y_true.shape==y_pred.shape)
    tprs = []
    precs = []
    for idx in range(y_pred.shape[1]):
        true_nz = y_true[:,idx]!=0
        precision, recall, thresholds = precision_recall_curve(true_nz, np.abs(y_pred[:,idx]),drop_intermediate=True)
        prec = np.interp(base_rec, recall[::-1], precision[::-1])
        precs.append(prec)
    precs = np.array(precs)
    mean_precs = precs.mean(axis=0)
    return mean_precs


def transform_beta(real_data:np.ndarray, sim_data:np.ndarray, data_to_trans_list: list=[], asymmetric: bool=False) -> list:

    if asymmetric:
        kappa, loc_real, scale_real = stats.laplace_asymmetric.fit(real_data, floc=0)
    else:
        loc_real, scale_real = stats.laplace.fit(real_data, floc=0)
    sim_ecdf = stats.ecdf(sim_data).cdf

    data_trans_list = list()
    for data_to_trans in data_to_trans_list:
        data_trans_probs = sim_ecdf.evaluate(data_to_trans)
        if asymmetric:
            data_trans = stats.laplace_asymmetric.ppf(data_trans_probs, kappa=kappa, loc=loc_real, scale=scale_real)
        else:
            data_trans = stats.laplace.ppf(data_trans_probs, loc=loc_real, scale=scale_real)
        is_pos_inf = np.logical_and(np.isinf(data_trans),data_trans>0)
        is_neg_inf = np.logical_and(np.isinf(data_trans),data_trans<0)
        data_trans[is_pos_inf] = np.nanmax(data_trans[~is_pos_inf])
        data_trans[is_neg_inf] = np.nanmin(data_trans[~is_neg_inf])       
        data_trans_list.append(data_trans)
    
    return data_trans_list


def transform_data(X: np.ndarray, y: np.ndarray, beta_real: np.ndarray, se_real: np.ndarray, 
                   top_r: int, asymmetric: bool=False):
    
    # scale X
    X_scaled = np.zeros_like(X)
    # scale beta
    beta_sim = X[:,0]
    beta_idx = np.concatenate([[0], np.arange(3,3+top_r)])
    data_to_trans = X[:,beta_idx]
    data_trans_list = transform_beta(beta_real,beta_sim,[data_to_trans],asymmetric=asymmetric)
    data_trans = data_trans_list[0]
    X_scaled[:,beta_idx] = data_trans
    # scale se
    X_scaled[:,1] = scale_by_quantile(X[:,1],se_real,q=0.01)
    # keep LD the same
    ld_idx = np.concatenate([[2], np.arange(3+top_r,3+2*top_r)]) # for LDs
    X_scaled[:,ld_idx] = X[:,ld_idx]

    sob = X_scaled[:,0]
    ob = X[:,0]
    sob_quantiles_range = np.quantile(sob, 0.9, axis=0)-np.quantile(sob, 0.1, axis=0)
    ob_quantiles_range = np.quantile(ob, 0.9, axis=0)-np.quantile(ob, 0.1, axis=0)
    # scales = np.divide(sob_quantiles_range,ob_quantiles_range)
    scale = sob_quantiles_range/(ob_quantiles_range+1e-8)
    y_scaled = y*scale

    return X_scaled, y_scaled


def angle_between(v1, v2, deg=False):
    """Computes the angle in radians between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    if deg:
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_eiginfo(Sigma: list[np.ndarray], comp_to_ref: bool=False):
    assert len(Sigma)>0
    if not comp_to_ref:
        # compute eigen-info, and get the orientation of the cls
        sigma_eiginfo = list()
        for i, covar in enumerate(Sigma):
        
            # Calculate the eigenvectors and eigenvalues
            eigenval, eigenvec = np.linalg.eig(covar)
            idx = eigenval.argsort()[::-1]   
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:,idx]
        
            angles = list()
            for i_eig in range(len(eigenval)):
              eigval = eigenval[i_eig]
              eigvec = eigenvec[:,i_eig]
              angle = np.arctan2(eigvec[1], eigvec[0])
              if (angle < 0):
                angle = angle + 2*np.pi
              angles.append(angle)
        
            sigma_eiginfo.append([eigenval,angles[0]])
    else:
        n = Sigma[0].shape[0]
        ref_vecs = [list(p) for p in itertools.product([0, 1], repeat=n) if any(p)]
        sigma_eiginfo = list()
        for i, covar in enumerate(Sigma):
        
            # Calculate the eigenvectors and eigenvalues
            eigenval, eigenvec = np.linalg.eig(covar)
            idx = eigenval.argsort()[::-1]   
            eigenval = eigenval[idx]
            eigenvec = eigenvec[:,idx]
            
            angles = list()
            for i_eig in range(len(eigenval)):
              eigval = eigenval[i_eig]
              eigvec = eigenvec[:,i_eig]
              angs = list()
              for ref_vec in ref_vecs:
                  ang = angle_between(eigvec,ref_vec)
                  if (ang < 0):  
                      ang = ang + 2*np.pi
                  angs.append(ang)
              angles.append(angs)
        
            sigma_eiginfo.append([eigenval,eigenvec,angles])
    
    return sigma_eiginfo

