import numpy as np
import scipy as sp
import pandas as pd

def load_simulation(sim_files_prefix):
    X = list()
    y = list()
    meta = list()
    for f_prefix in sim_files_prefix:
        X.append(np.loadtxt(f_prefix + ".X", delimiter=','))
        y.append(np.loadtxt(f_prefix + ".y", delimiter=','))
        meta.append(np.loadtxt(f_prefix + ".meta", delimiter=','))
    X = np.vstack(X)
    y = np.concatenate(y)
    meta = np.vstack(meta)
    return X, y, meta

def load_real_data(real_files, beta_col="BETA", se_col="SE"):
    beta_real = list()
    se_real = list()
    for f in real_files:
        df_gwas = pd.read_csv(f)
        beta_real.append(df_gwas[beta_col].values)
        se_real.append(df_gwas[se_col].values)
    beta_real = np.concatenate(beta_real)
    se_real = np.concatenate(se_real)
    return beta_real, se_real


def transform_beta(real_data,sim_data,data_to_trans_list=[],asymmetric=False):

    if asymmetric:
        kappa, loc_real, scale_real = sp.stats.laplace_asymmetric.fit(real_data, floc=0)
    else:
        loc_real, scale_real = sp.stats.laplace.fit(real_data, floc=0)
    sim_ecdf = sp.stats.ecdf(sim_data).cdf

    data_trans_list = list()
    for data_to_trans in data_to_trans_list:
        data_trans_probs = sim_ecdf.evaluate(data_to_trans)
        if asymmetric:
            data_trans = sp.stats.laplace_asymmetric.ppf(data_trans_probs, kappa=kappa, loc=loc_real, scale=scale_real)
        else:
            data_trans = sp.stats.laplace.ppf(data_trans_probs, loc=loc_real, scale=scale_real)
        is_pos_inf = np.logical_and(np.isinf(data_trans),data_trans>0)
        is_neg_inf = np.logical_and(np.isinf(data_trans),data_trans<0)
        data_trans[is_pos_inf] = np.nanmax(data_trans[~is_pos_inf])
        data_trans[is_neg_inf] = np.nanmin(data_trans[~is_neg_inf])       
        data_trans_list.append(data_trans)
    
    return data_trans_list

def scale_by_quantile(x,y,q=0.1):
    
    x_qtl = np.quantile(x,[q,0.5,1-q])
    y_qtl = np.quantile(y,[q,0.5,1-q])
    x_scaled = (x-x_qtl[0])/(x_qtl[2]-x_qtl[0])*(y_qtl[2]-y_qtl[0])+y_qtl[0]
    
    return x_scaled

def transform_data(X_train,y_train,beta_real,se_real,max_r,asymmetric=False):
    
    # scale X
    X_train_scaled = np.zeros_like(X_train)
    # scale beta
    beta_sim = X_train[:,0].copy()
    beta_idx = np.concatenate([[0], np.arange(3,3+max_r)])
    data_to_trans = X_train[:,beta_idx].copy()
    data_trans_list = transform_beta(beta_real,beta_sim,[data_to_trans],asymmetric=asymmetric)
    data_trans = data_trans_list[0]
    X_train_scaled[:,beta_idx] = data_trans
    # scale se
    X_train_scaled[:,1] = scale_by_quantile(X_train[:,1],se_real,q=0.01)
    # keep LD the same
    ld_idx = np.concatenate([[2], np.arange(3+max_r,3+2*max_r)]) # for LDs
    X_train_scaled[:,ld_idx] = X_train[:,ld_idx]
    
    # scale y -- no matching distribution
    # so that the var of nonzero entires = the variance of the observed ones (with outliers removed)
    scaled_obs_betas = X_train_scaled[:,0]
    z = np.abs(sp.stats.zscore(scaled_obs_betas))
    valid_indices = np.where(z <= 3)[0]
    y_train_scale = scaled_obs_betas[valid_indices].std()/y_train[valid_indices][y_train[valid_indices]!=0].std()
    y_train_scaled = y_train*y_train_scale

    return X_train_scaled, y_train_scaled


def subset_X(data_X,max_r,top_r):
    
    idx_features = np.concatenate([np.arange(top_r+3),np.arange(3+max_r,3+max_r+top_r)]) 

    return data_X[:,idx_features]

def scale_and_subset(X, y, beta_real, se_real, max_r, top_r, scale=1, asymmetric=False):
    X_scaled, y_scaled = transform_data(X,y,beta_real,se_real,max_r,asymmetric=True)
    y_scale = np.mean(y[y_scaled!=0]/y_scaled[y_scaled!=0])
    scale_idx = np.concatenate([[0,1], np.arange(3,3+top_r)])
    X_sub = subset_X(X_scaled, max_r, top_r).copy()
    X_sub[:,scale_idx] *= scale
    return X_sub, y_scaled, y_scale

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False