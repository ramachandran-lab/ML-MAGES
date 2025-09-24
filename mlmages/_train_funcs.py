import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import shutil

# for training NN
import torch
import torch.nn as nn
from ._util_funcs import *


def get_n_feature(top_r):
    """
    Calculate the number of features based on the given top_r value.
    (bhat, shat, ldsc, & top_r_beta, top_r_ld for each top_r)

    Parameters:
    top_r (int): The top_r value used in the calculation.

    Returns:
    int: The number of features calculated based on the given top_r value.
    """
    return 2*top_r+3


def load_simulated_by_snp(df_chrom: pd.DataFrame, ld_chrom: np.ndarray, sim_path_prefix: str, top_r:int, nsnp=1000):
    """
    Load simulated data (simulation by SNP).

    Parameters:
    df_chrom (pd.DataFrame): DataFrame containing chromosome information.
    ld_chrom (np.ndarray): LD matrix for the chromosome.
    sim_path_prefix (str): Path prefix for the simulated data files.
    top_r (int): Number of top SNPs to consider.
    nsnp (int): Number of SNPs in each simulation segment.

    Returns:
    tuple: A tuple containing the simulated feature matrix (X_sim) and target vector (y_sim).
    """

    X_sim = []
    y_sim = []
        
    uniq_snp_start = df_chrom['snp_start'].unique().astype(int)

    for snp_start in uniq_snp_start:
        # get LD for the segment
        ld_seg = ld_chrom[snp_start:(snp_start+nsnp),snp_start:(snp_start+nsnp)]
        # compute LD score (within the segment)
        idx_max_ldsc, top_r_val, ldsc = get_topr_idx_from_ld(ld_seg, top_r)

        for i_sim in df_chrom[(df_chrom['snp_start']==snp_start)].index:
            # finish constructing features:  
            df_sim = pd.read_csv("{}{}.csv".format(sim_path_prefix,i_sim), index_col=0)
            assert nsnp==df_sim.shape[0]
            assert 'beta_hat' in df_sim.columns and 'se' in df_sim.columns and 'beta' in df_sim.columns
            bhat = df_sim['beta_hat'].values
            shat = df_sim['se'].values
            top_r_beta = bhat[idx_max_ldsc]
            X = np.concatenate([bhat[:,None],shat[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
            y = df_sim['beta'].values

            X_sim.append(X)
            y_sim.append(y)
            
    return X_sim, y_sim


def get_topr_idx_from_ld(ld: np.ndarray, top_r: int):
    """
    Get the top r indices and values from the LD matrix.

    Parameters:
    ld (np.ndarray): The LD matrix.
    top_r (int): The number of top SNPs to consider.

    Returns:
    tuple: A tuple containing the top r indices, top r values, and LD score.
    """

    nsnp = ld.shape[0]
    assert nsnp==ld.shape[1]

    # compute LD score (within the segment) and get top r idx and val
    ldsc = np.sum(ld,axis=0)-1
    idx_max_ldsc = np.argsort(-ld-np.identity(nsnp), axis=1)[:,1:(top_r+1)]
    top_r_val = ld[idx_max_ldsc][np.arange(nsnp),:,np.arange(nsnp)]
    
    return idx_max_ldsc, top_r_val, ldsc


# def load_simulated_by_snp_seg(df_seg: pd.DataFrame, ld_seg: np.ndarray, sim_path_prefix: str, top_r:int, nsnp=1000):

#     X_sim = []
#     y_sim = []
#     assert nsnp==ld_seg.shape[0]

#     # compute LD score (within the segment)
#     ldsc = np.sum(ld_seg,axis=0)-1
#     # get top r idx and val
#     idx_max_ldsc = np.argsort(-ld_seg-np.identity(nsnp), axis=1)[:,1:(top_r+1)]
#     top_r_val = ld_seg[idx_max_ldsc][np.arange(nsnp),:,np.arange(nsnp)]

#     for ii,i_sim in enumerate(df_seg['index']): # 
#         # df_sim = pd.read_csv("scaled_sim{}.csv".format(i_sim)), index_col=0)
#         df_sim = pd.read_csv("{}{}.csv".format(sim_path_prefix,i_sim), index_col=0)
#         bhat = df_sim['beta_hat'].values
#         shat = df_sim['se'].values
#         top_r_beta = bhat[idx_max_ldsc]
#         X = np.concatenate([bhat[:,None],shat[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
#         y = df_sim['beta'].values

#         X_sim.append(X)
#         y_sim.append(y)
            
#     return X_sim, y_sim

def construct_from_simulated_by_gene(sim_data_block: np.ndarray, ld: np.ndarray, top_r:int):

    X_sim_block = []
    y_sim_block = []
    nsnp = ld.shape[0]

    # compute LD score and get top r (within the block)
    idx_max_ldsc, top_r_val, ldsc = get_topr_idx_from_ld(ld, top_r)

    n_trait = sim_data_block.shape[1]//3

    for i_trait in range(n_trait):
        sim_data_block_bhat = sim_data_block[:,n_trait+i_trait]
        sim_data_block_se = sim_data_block[:,2*n_trait+i_trait]
        sim_data_block_btrue = sim_data_block[:,i_trait]
    
        top_r_beta = sim_data_block_bhat[idx_max_ldsc]
    
        X = np.concatenate([sim_data_block_bhat[:,None],sim_data_block_se[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
        y = sim_data_block_btrue

        X_sim_block.append(X)
        y_sim_block.append(y)

    return X_sim_block, y_sim_block


# NN model    
class FCNN(nn.Module):
    def __init__(self, n_feature:int, n_layer:int, model_label=""):
        super().__init__()
        self.name = model_label
        self.n_feature = n_feature
        self.n_layer = n_layer

        layers = []
        out_dim = n_feature
        for i_layer in range(n_layer):
            if i_layer==0:
                in_dim, out_dim = n_feature, 64
            else:
                in_dim, out_dim = out_dim, 32
            if i_layer==(n_layer-1):
                out_dim = 8
            dor = 0.1 if i_layer==(n_layer-1) else 0.2
                
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(num_features=out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dor))
        
        self.hidden_stack = nn.Sequential(*layers)

        # final layer
        self.output_layer = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.hidden_stack(x)
        x = self.output_layer(x)
        return x


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


class WeightedMSELoss(nn.Module):
    def __init__(self, non_zero_weight=5.0): # Default weight for non-zero values
        super(WeightedMSELoss, self).__init__()
        self.non_zero_weight = non_zero_weight

    def forward(self, predictions, targets):
        # Calculate element-wise squared error
        squared_error = (predictions - targets) ** 2

        # Create a mask for non-zero target values
        non_zero_mask = (targets.abs() > 1e-6).float()

        # Apply weights: higher weight for non-zero targets
        weighted_squared_error = squared_error * (non_zero_mask * self.non_zero_weight + (1 - non_zero_mask) * 1.0)

        # Calculate the mean of the weighted squared errors
        loss = torch.mean(weighted_squared_error)
        return loss
