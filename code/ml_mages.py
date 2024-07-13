import os
import numpy as np
import pandas as pd
import scipy as sp
import itertools 
import time
import torch
import torch.nn as nn
from chiscore import liu_sf, davies_pvalue
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from _cls_funcs import infmix_clustering


def load_gwa(gwa_files, cols=['BETA','SE']):
    """
    Load genetic association results from multiple files.

    Parameters:
    gwa_files (list): A list of file paths to the genetic association result files.
    cols (list, optional): A list of column names to extract from the genetic association result files. Defaults to ['BETA','SE'].

    Returns:
    list: A list of dict (of numpy.ndarray) containing all columns extracted for each trait.
    """
    gwa_loaded = list()
    for gwa_file in gwa_files:
        df_gwa = pd.read_csv(gwa_file)
        res = dict()
        for col in cols:
            res[col] = df_gwa[col].values
        gwa_loaded.append(res)
    return gwa_loaded


def load_gwa_results(gwa_files, beta_col='BETA', se_col='SE'):
    """
    Load genetic association results from multiple files.

    Parameters:
    gwa_files (list): A list of file paths to the genetic association result files.
    beta_col (str, optional): The column name for the beta values. Defaults to 'BETA'.
    se_col (str, optional): The column name for the standard error values. Defaults to 'SE'.

    Returns:
    tuple: A tuple containing two lists - beta values and standard error values.
    """
    beta = list()
    se = list()
    for gwa_file in gwa_files:
        df_gwa = pd.read_csv(gwa_file)
        beta.append(df_gwa[beta_col].values)
        se.append(df_gwa[se_col].values)
    return beta, se


def load_ld_blocks(ld_files, sep=" "):
    """
    Load LD blocks from a list of LD files.

    Parameters:
    ld_files (list): A list of file paths to the LD files.

    Returns:
    list: A list of LD blocks loaded from the LD files.
    """
    ld_list = list()
    for ld_file in ld_files:
        ld = np.loadtxt(ld_file, delimiter=sep)
        ld_list.append(ld)
    return ld_list


def get_n_features(top_r):
    """
    Calculate the number of features based on the given top_r value.

    Parameters:
    top_r (int): The top_r value used in the calculation.

    Returns:
    int: The number of features calculated based on the given top_r value.
    """
    return 2*top_r+3

def load_model(model_path,n_layer,top_r):
    n_features = get_n_features(top_r)
    feature_lb = "top{}".format(top_r)        
    model = Fc3(n_features,feature_lb) if n_layer==3 else Fc2(n_features,feature_lb)
    model_file = os.path.join(model_path,"{}.model".format(model.name))
    state = torch.load(model_file)
    model.load_state_dict(state)
    return model

def load_models(model_path,n_layer,top_r,n_models):
    n_features = get_n_features(top_r)
    feature_lb = "top{}".format(top_r)  
    models = list()      
    for i_model in range(n_models):
        model = Fc3(n_features,feature_lb) if n_layer==3 else Fc2(n_features,feature_lb)
        model_file = os.path.join(model_path,"{}_{}.model".format(model.name,i_model))
        state = torch.load(model_file)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def construct_features(bhat, shat, ld, top_r):
    """
    Constructs features for ML-MAGES algorithm.

    Parameters:
    - bhat (ndarray): Array of shape (n_snps,) representing the observed effect sizes.
    - shat (ndarray): Array of shape (n_snps,) representing the standard errors of the effect sizes.
    - ld (ndarray): Array of shape (n_snps, n_snps) representing the linkage disequilibrium matrix.
    - top_r (int): Number of top LD scores to consider.

    Returns:
    - features (ndarray): Array of shape (n_snps, 2 + top_r + 1 + top_r) representing the constructed features.
                          The columns are: bhat, shat, ldsc, top_r_beta, top_r_ld.

    """
    
    ld_sum = np.sum(ld,axis=0)
    n_snps = ld.shape[0]
    ldsc = ld_sum-1

    idx_max_ldsc = np.argsort(-ld-np.identity(n_snps), axis=1)[:,1:(top_r+1)]
    top_r_beta = bhat[idx_max_ldsc]
    top_r_ld = ld[idx_max_ldsc][np.arange(n_snps),:,np.arange(n_snps)]
    features = np.concatenate([bhat[:,None],shat[:,None],ldsc[:,None],top_r_beta,top_r_ld], axis=1)

    return features


def threshold_vals(x, zero_cutoff=1e-3):
    """
    Apply thresholding to the input array.

    Parameters:
    - x (ndarray): Input array.
    - zero_cutoff (float, optional): Threshold value. Values below this threshold will be set to zero. Default is 1e-3.

    Returns:
    - y (ndarray): Thresholded array, where values below the threshold are set to zero.

    """
    y = np.zeros_like(x)
    y[np.abs(x)>zero_cutoff] = x[np.abs(x)>zero_cutoff]
    return y


def get_nz_effects(beta_reg, fold_min=200, fold_max=10, zero_cutoff=1e-3, adjust_max = 10, adjust_rate = 1.5):
    """
    Get the non-zero effects from a given beta regression.

    Parameters:
    - beta_reg (numpy.ndarray): The regularized effects.
    - zero_cutoff (float, optional): The threshold value to consider a value as zero. Default is 1e-3.
    - adjust_max (int, optional): The maximum number of adjustments to make. Default is 10.

    Returns:
    - beta_nz (numpy.ndarray): The non-zero effects.
    - zero_cutoff (float): The final threshold value used.

    """
    if beta_reg.ndim < 2:
        beta_reg = beta_reg.reshape(-1, 1)
    p = beta_reg.shape[1]

    beta_reg_f = threshold_vals(beta_reg, zero_cutoff=zero_cutoff)
    beta_nz = beta_reg_f[np.any(np.abs(beta_reg_f)>0, axis=1),:] # any nz

    adjust_iter = 0
    while (beta_nz.shape[0]<beta_reg.shape[0]//fold_min or beta_nz.shape[0]>beta_reg.shape[0]//fold_max) and adjust_iter<adjust_max:
        if beta_nz.shape[0]<beta_reg.shape[0]//fold_min:
            zero_cutoff /= adjust_rate
        else:
            zero_cutoff *= adjust_rate
        beta_reg_f = threshold_vals(beta_reg, zero_cutoff=zero_cutoff)
        beta_nz = beta_reg_f[np.any(np.abs(beta_reg_f)>0, axis=1),:] # any nz
        adjust_iter += 1 
    
    return beta_nz, zero_cutoff


def clustering(beta_reg, K=25, n_runs=25):
    """
    Perform clustering using the infmix_clustering function.

    Parameters:
    - beta_reg: The input regularized effects.
    - K: The number of clusters (default is 25).
    - n_runs: The number of runs (default is 25).

    Returns:
    - Sigma: The Sigma value.
    - pi: The pi value.
    - pred_K: The predicted K value.
    - pred_cls: The predicted cls value.
    """
    tic = time.time()
    Sigma, pi, pred_K, pred_cls = infmix_clustering(beta_reg, K=K, n_runs=n_runs, alpha=0.5, niter=1000, eps=1e-3)
    toc = time.time()

    return Sigma, pi, pred_K, pred_cls


def enrichment_test(genes, eps_eff, beta_reg, ld, use_davies=False):
    """
    Perform enrichment test for a list of genes.

    Args:
        genes (pandas.DataFrame): DataFrame containing information about genes.
        eps_eff (float): epsilon effect.
        beta_reg (numpy.ndarray): Array of regularized effects.
        ld (numpy.ndarray): Linkage disequilibrium matrix.
        use_davies (bool, optional): Flag indicating whether to use Davies method for p-value computation. 
            Defaults to False.

    Returns:
        dict: Dictionary containing the p-values, test statistics, and test statistic variances for each gene.
            - 'P': List of p-values.
            - 'STAT': List of test statistics.
            - 'VAR': List of test statistic variances.
    """
    # gene-level test    
    prior_weight = np.ones(len(beta_reg))
    
    p_vals = list()
    test_stats = list()
    test_stat_vars = list()
    
    for i in range(len(genes)):
    
        gene_snps = np.arange(genes.loc[i, "SNP_FIRST"], genes.loc[i, "SNP_LAST"]+1)
        nsnps = genes.loc[i, "N_SNPS"]
        ld_g = ld[gene_snps,:][:,gene_snps]
        weight_matrix = np.diag(prior_weight[gene_snps])
        
        # compute eigenvalues
        mat = eps_eff * ld_g @ weight_matrix
        e_val, e_vec = np.linalg.eig(mat)
        
        #compute test statistics
        betas_g = beta_reg[gene_snps]
        test_stat = betas_g.T @ weight_matrix @ betas_g
        #compute test statistics variance
        t_var = np.diag((ld_g * eps_eff) @ (ld_g * eps_eff)).sum()
        if use_davies:
            (p_val_g, _, _, _) = liu_sf(test_stat, e_val, 1, 0)
        else:
            p_val_g = davies_pvalue(test_stat, mat)
        p_val_g = 1e-20 if p_val_g <= 0.0 else p_val_g
        p_vals.append(p_val_g)
        test_stats.append(test_stat)
        test_stat_vars.append(t_var)

    return {'P':p_vals,'STAT':test_stats,'VAR':test_stat_vars}


def summarize_multivariate_gene(genes, betas, cls_lbs, pred_K):
    """
    Summarizes multivariate gene associations.

    Args:
        genes (DataFrame): A DataFrame containing gene information.
        betas (ndarray): An array of regularized effects.
        cls_lbs (ndarray): An array of cluster labels.
        pred_K (int): The number of classes.

    Returns:
        DataFrame: A DataFrame containing summarized gene-level assocations.

    """
    if betas.ndim<2:
        betas = np.expand_dims(betas, axis=1)
    p = betas.shape[1]
    all_pw_comb = list(itertools.combinations_with_replacement(np.arange(p), r=2))

    genes_cls_cnt = list()
    genes_beta_prod_all_cls = list()
    for i_g in range(len(genes)):
        snps_g = np.arange(genes.loc[i_g, "SNP_FIRST"], genes.loc[i_g, "SNP_LAST"]+1)
        betas_g = betas[snps_g,:]
        betas_cls = cls_lbs[snps_g]
        
        cls_cnt = pd.Series(betas_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
        genes_cls_cnt.append(cls_cnt[np.arange(pred_K)].values)

        s = genes.loc[i_g]['N_SNPS']
        prod_list = list()
        for pw in all_pw_comb:
            prod_list.append(np.sum(betas_g[:,pw[0]]*betas_g[:,pw[1]])/s)
        genes_beta_prod_all_cls.append(prod_list)
        # genes_beta_prod_all_cls.append([np.sum(betas_g[:,0]**2)/s,np.sum(betas_g[:,1]**2)/s,np.sum(betas_g[:,0]*betas_g[:,1])/s])
        
    genes_cls_cnt = np.array(genes_cls_cnt)
    genes_beta_prod_all_cls = np.array(genes_beta_prod_all_cls)

    df_gene = pd.DataFrame(genes)
    df_gene['size'] = genes['N_SNPS']
    for cls in range(pred_K):
        df_gene['cls{}_frac'.format(cls+1)] = genes_cls_cnt[:,cls]/df_gene['size']
    # prod_types = ['b1b1','b2b2','b1b2']
    prod_types = list()
    for pw in all_pw_comb:
        prod_types.append("b{}b{}".format(pw[0]+1,pw[1]+1))
    df_gene[prod_types] = genes_beta_prod_all_cls

    return df_gene


def plot_clustering(beta_nz, pred_cls, pred_K, Sigma, pi, traits, save_file):
    """
    Plot the clustering results.

    Parameters:
    - beta_nz (numpy.ndarray): The non-zero effect values.
    - pred_cls (numpy.ndarray): The predicted cluster labels.
    - pred_K (int): The number of clusters.
    - Sigma (numpy.ndarray): The covariance matrices.
    - pi (numpy.ndarray): The cluster probabilities.
    - traits (list): The names of the traits.
    - save_file (str): The path to save the plot.

    Returns:
    None
    """

    df = pd.DataFrame(beta_nz)
    df['CLS'] = pred_cls
    cls_cnt = pd.Series(pred_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
    cls_perc = (cls_cnt/cls_cnt.sum()).loc[np.arange(pred_K)]    

    xylim = np.ceil(np.max(np.abs(beta_nz))*100)/100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=False, sharey=False, dpi=200)
    # plot data
    palette = sns.color_palette("colorblind", pred_K)
    if beta_nz.shape[1]==1:
        ax = axes[0]
        g = sns.kdeplot(data=df, x=0, hue='CLS', ax=ax,
                    fill=True, common_norm=True, alpha=0.5, palette=palette, 
                legend=True, gridsize=100, bw_adjust=10)
        ax.set_xlabel("$\\beta$ labeled by components")
        ax.set_title("$\\beta$ labeled by clusters")
        leg = ax.legend_ #.legendHandles
        labels = [t.get_text() for t in axes[0].legend_.texts]
        new_labels = ["{}: {:.1%}".format(int(ii)+1,cls_perc.loc[int(ii)]) for ii in labels]
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)
        ax.set_ylim(ymin=-0.1)

        ax = axes[1]
        x_extreme = max(abs(df[0]))
        x = np.linspace(-x_extreme,x_extreme,1000)
        ax = axes[1]
        for k in range(pred_K):
            y = sp.stats.norm.pdf(x, 0, np.sqrt(float(Sigma[k][0][0])))#*pi[k]*3 # gaussian
            ax.plot(x,y,lw=1,label="{}: {:.1%}".format(k+1,pi[k]), color=palette[k])
            
        ax.legend(bbox_to_anchor=(1.01, 1.01), title="CLS: inferred $\\pi_k$", loc='upper left')
        ax.set_title("Gaussian mixtures")
        ax.set_ylim(ymin=-0.1)
    
    elif beta_nz.shape[1]==2:

        ax = axes[0]
        sns.scatterplot(data=df, x=0, y=1, hue='CLS', ax=axes[0], alpha=0.8, s=12, palette=palette)
        ax.legend(bbox_to_anchor=(1.01, 1.01), numpoints=1, markerscale=2,title="CLS: nz-variants", loc='upper left')
        leg = ax.legend_ #.legendHandles
        labels = [t.get_text() for t in axes[0].legend_.texts]
        new_labels = ["{}: {:.1%}".format(int(ii)+1,cls_perc.loc[int(ii)]) for ii in labels]
        ax.set_xlim([-xylim,xylim])
        ax.set_ylim([-xylim,xylim])
        ax.set_aspect('equal', 'box')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)
        ax.set_title("$\\beta$ labeled by clusters")
        ax.set_xlabel(traits[0])
        ax.set_ylabel(traits[1])

        ax = axes[1]
        for i, covar in enumerate(Sigma):

            eigenval, eigenvec = np.linalg.eig(covar)
            max_eigval, min_eigval = max(eigenval), min(eigenval)

            max_eigenvec_idx= np.argwhere(eigenval == max_eigval)[0][0]
            max_eigenvec = eigenvec[:,max_eigenvec_idx]
            angle = np.arctan2(max_eigenvec[1], max_eigenvec[0])
            if (angle < 0):
                angle = angle + 2*np.pi

            # draw the confidence ellipse
            nstd = np.sqrt(9.210) # for 99% CI 
            ell = Ellipse(xy=(0, 0), width=np.sqrt(max_eigval)*nstd*2, height=np.sqrt(min_eigval)*nstd*2, \
                          angle=np.rad2deg(angle), facecolor='none', edgecolor=palette[i], \
                            label="{}: {:.1%}".format(int(i)+1,pi[i]))
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.9)
            ax.add_patch(ell)

        ax.set_xlim([-xylim,xylim])
        ax.set_ylim([-xylim,xylim])
        ax.set_aspect('equal', 'box')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_title("Gaussian mixtures")
        ax.legend(bbox_to_anchor=(1.01, 1.01), title="CLS: inferred $\\pi_k$", loc='upper left')
        ax.set_xlabel(traits[0])
        ax.set_ylabel(traits[1])
    
    axes[1].yaxis.get_label().set_visible(True)
    fig.tight_layout()
    fig.savefig(save_file, bbox_inches='tight', dpi=200)

    return


def plot_shrinkage(beta_obs, beta_reg, beta_chrs, save_file):
    """
    Plot the shrinkage results.

    Parameters:
    - beta_obs (numpy.ndarray): The observed (un-regularized) effect labels.
    - beta_reg (numpy.ndarray): The regularized effect values.
    - beta_chrs (numpy.ndarray): The CHR of each variant.
    - save_file (str): The path to save the plot.

    Returns:
    None
    """
    # create dataframe
    df_results = pd.DataFrame(np.vstack([beta_obs, beta_reg, beta_chrs]).T,columns=["BETA_OBS","BETA_REG","CHR"])

    # CHR switch point
    chr_switch_idx = np.where(np.diff(beta_chrs)>0)[0]
    chr_switch_idx = np.insert(chr_switch_idx,0,0)
    chr_switch_idx = np.insert(chr_switch_idx,len(chr_switch_idx),len(beta_obs))
    chr_in_data = np.sort(np.unique(beta_chrs))
    lb_idx = list()
    lb_name = list()
    for ic, c in enumerate(chr_switch_idx[:(len(chr_switch_idx)-1)]):
        mid_idx = (chr_switch_idx[ic]+chr_switch_idx[ic+1])//2
        lb_idx.append(mid_idx)
        lb_name.append(chr_in_data[ic])
    df_results['is_odd'] = df_results['CHR']%2==1
    df_results['idx'] = np.arange(len(df_results))
    
    fig, axes = plt.subplots(2,1, figsize=(8, 3), sharex=False, sharey=False, dpi=200)
    # plot data
    cols = ["BETA_OBS","BETA_REG"]
    for i,col in enumerate(cols):
        ax = axes[i]
        sf = sns.scatterplot(data=df_results[df_results['is_odd']], 
                             x='idx', y=col, s=4, 
                             color='#999999',linewidth=0,
                             ax=ax)
        sf = sns.scatterplot(data=df_results[~df_results['is_odd']], 
                             x='idx', y=col, s=4, 
                             color='#404040',linewidth=0,
                             ax=ax)
        ax.set_xlim([0,len(df_results)])
        for sw in chr_switch_idx:
            ax.axvline(x=sw, color='lightgray', ls='--', lw=0.5)
        if i==(len(cols)-1):
            ax.set_xlabel("Chromosome")
        else:
            ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_ylabel(col)
    
    fig.tight_layout()
    fig.savefig(save_file, bbox_inches='tight', dpi=200)


def plot_enrichment(df_enrich, save_file, level=0.05):
    """
    Plot the enrichment test results.

    Parameters:
    - df_enrich (pd.DataFrame): A dataframe containing the enrichment test results. Must contain columns 'CHR', 'GENE', and 'P'.
    - save_file (str): The path to save the plot.

    Returns:
    None
    """
    # CHR switch point
    chr_switch_idx = np.where(np.diff(df_enrich['CHR'])>0)[0]
    chr_switch_idx = np.insert(chr_switch_idx,0,0)
    chr_switch_idx = np.insert(chr_switch_idx,len(chr_switch_idx),len(df_enrich))
    chr_in_data = np.sort(df_enrich['CHR'].unique())
    lb_idx = list()
    lb_name = list()
    for ic, c in enumerate(chr_switch_idx[:(len(chr_switch_idx)-1)]):
        mid_idx = (chr_switch_idx[ic]+chr_switch_idx[ic+1])//2
        lb_idx.append(mid_idx)
        lb_name.append(chr_in_data[ic])
    df_enrich['is_odd'] = df_enrich['CHR']%2==1
    df_enrich['idx'] = np.arange(len(df_enrich))
    adjP = sp.stats.false_discovery_control(df_enrich["P"].values, method='bh')
    df_enrich['neglogp'] = -np.log10(adjP)
    
    fig, ax = plt.subplots(1,1, figsize=(7, 2), dpi=200)
    # plot data
    sf = sns.scatterplot(data=df_enrich[df_enrich['is_odd']], 
                         x='idx', y='neglogp', s=3, 
                         color='#999999',linewidth=0,
                         ax=ax)
    sf = sns.scatterplot(data=df_enrich[~df_enrich['is_odd']], 
                         x='idx', y='neglogp', s=3, 
                         color='#404040',linewidth=0,
                         ax=ax)
    ax.axhline(-np.log10(level), color='blue', alpha=0.5, ls='--', lw=1)
    ax.text(2,-np.log10(level)+0.5,r"$p={}$".format(level),
            c='blue', alpha=0.5, ha="left",va="bottom") 
    ax.set_xlim([0,len(df_enrich)])
    # ax.set_ylim([-10,None])
    for sw in chr_switch_idx:
        ax.axvline(x=sw, color='lightgray', ls='--', lw=0.5)
    ax.set_xlabel("Chromosome")
    ax.set_xticks([])
    ax.set_ylabel(r'adjusted $-log_{10}p$')
    
    fig.savefig(save_file, bbox_inches='tight', dpi=200)


def construct_new_model(model_layer,n_features,feature_lb):
    if model_layer==3:
        model = Fc3(n_features,feature_lb)
    else:
        model = Fc2(n_features,feature_lb)
    return model


# 2-layer FFNN
class Fc2(nn.Module):
    def __init__(self,n_features,feature_lb):
        super().__init__()
        self.name = "Fc2{}".format(feature_lb)
        
        self.hidden1 = nn.Linear(n_features, 64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2) 
        self.hidden2 = nn.Linear(64, 8)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1) 
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.dropout1(self.act1(self.bn1(self.hidden1(x))))
        x = self.dropout2(self.act2(self.bn2(self.hidden2(x))))
        x = self.output(x)
        return x
   
# 3-layer FFNN
class Fc3(nn.Module):
    def __init__(self,n_features,feature_lb):
        super().__init__()
        self.name = "Fc3{}".format(feature_lb)
        self.hidden1 = nn.Linear(n_features, 64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2) 
        self.hidden2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(32, 8)
        self.bn3 = nn.BatchNorm1d(num_features=8)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.dropout1(self.act1(self.bn1(self.hidden1(x))))
        x = self.dropout2(self.act2(self.bn2(self.hidden2(x))))
        x = self.dropout3(self.act3(self.bn3(self.hidden3(x))))
        x = self.output(x)
        return x
