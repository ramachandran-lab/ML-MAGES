import numpy as np
import pandas as pd
import scipy as sp
import itertools 
import time
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics import precision_recall_curve, average_precision_score, r2_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from sklearn import preprocessing
# for shrinkage using ENET
from sklearn.linear_model import ElasticNetCV

import _main_funcs as mf

def prepare_cov(n_traits,n_sim_var=50, sig_thre=0.1, pcorr_thre=0.6):
    sim_cov_multi = dict()
    sim_cov_multi[1] = np.random.uniform(0.1,0.5,n_sim_var)
    if n_traits>1:
        for n in range(2,n_traits+1):
            sim_cov_multi[n] = list()
            cov_sim_done = False
            while not cov_sim_done:
                C = np.random.rand(n,n)-0.5  
                Sigma = np.dot(C, C.transpose()) #-np.identity(2)*1e-6
                X = np.random.multivariate_normal(np.zeros(n), Sigma, size=500)
                is_ok = True
                for combo in itertools.combinations(range(n), 2):
                    pscorr = sp.stats.pearsonr(X[:,combo[0]],X[:,combo[1]])[0]
                    sig = sp.stats.pearsonr(X[:,combo[0]],X[:,combo[1]])[1]
                    if sig>sig_thre or (pscorr<pcorr_thre and pscorr>-pcorr_thre):
                        is_ok = False
                if is_ok:
                    sim_cov_multi[n].append(Sigma)
                cov_sim_done = len(sim_cov_multi[n])>n_sim_var
        
    return sim_cov_multi


def load_real_chr_data(real_files, beta_col="BETA", se_col="SE"):
    beta_real = list()
    se_real = list()
    for f in real_files:   
        df_gwas = pd.read_csv(f, sep="\s+") 
        beta_real.append(df_gwas[beta_col].values)
        se_real.append(df_gwas[se_col].values)
    beta_real = np.concatenate(beta_real)
    se_real = np.concatenate(se_real)
    return beta_real, se_real


# performance evaluation
def perf_eval(true,pred,type='rmse'):
    if type=='rmse':
        perf = np.sqrt(np.mean((true-pred)**2))
    elif type=='wrmse':
        is_true_nz = true!=0
        frac_true_nz = np.sum(is_true_nz, axis=0)/true.shape[0]
        sqr_err = (true-pred)**2
        weighted_mean_sum_sqr_err = np.mean(sqr_err[is_true_nz])*(1-frac_true_nz) + np.mean(sqr_err[~is_true_nz])*frac_true_nz
        perf = np.sqrt(weighted_mean_sum_sqr_err)
    elif type=='pcorr':
        perf = np.corrcoef(true,pred)[0,1]
    elif type=='auc':
        perf = roc_auc_score(true!=0,np.abs(pred))
    elif type=='r2':
        perf = r2_score(true,pred)
    elif type=='aps':
        perf = average_precision_score(true!=0,np.abs(pred))
    else: # default to RMSE
        perf = np.sqrt(np.mean((true-pred)**2))
    return perf


def process_gene_list(gene_list,all_pos,chr,min_gene_size=10):

    gene_list['CHR'] = gene_list['CHR'].astype(str)
    gene_list = gene_list[(gene_list['CHR']!='X') & (gene_list['CHR']!='Y')].reset_index(drop=True)
    gene_list['CHR'] = gene_list['CHR'].astype(int)
    
    gene_list = gene_list[gene_list['CHR']==chr].reset_index(drop=True)
    all_genes = gene_list['GENE']
    gene_start = gene_list['START']
    gene_end = gene_list['END']
    
    genes_chr = pd.DataFrame(columns=['CHR','GENE','START','END','N_SNPS','SNP_FIRST','SNP_LAST','SNPS'])
    
    for i_gene in range(len(gene_list)):
        snps_in_range = np.where((all_pos>=gene_start[i_gene]) & (all_pos<=gene_end[i_gene]))[0]
        nsnps = len(snps_in_range)
        if nsnps>1:
            genes_chr.loc[len(genes_chr.index)] = [chr, all_genes[i_gene], gene_start[i_gene], gene_end[i_gene], nsnps, snps_in_range[0],snps_in_range[-1],snps_in_range] 
    # print("{} genes with number of SNPs between {} and {}".format(len(genes_chr), genes_chr['N_SNPS'].min(), genes_chr['N_SNPS'].max()))
    
    genes_chr = genes_chr[genes_chr['N_SNPS']>=min_gene_size].reset_index(drop=True)
    
    return genes_chr



def shrink_enet(betas,ses,chr_ld,brkpts):
    # ENET
    breg_enet = list()
    tic = time.time()
    for i in range(betas.shape[1]):
        
        bhat_all = betas[:,i]
        shat_all = ses[:,i]
        
        breg = list()
        for i_bk in range(len(brkpts)-1):
            # ENET
            lb, ub = brkpts[i_bk], brkpts[i_bk+1]
    
            # old ENET
            X_input = chr_ld[lb:ub,lb:ub]
            y_std = np.std(bhat_all[lb:ub]) 
            y_mean = np.mean(bhat_all[lb:ub]) 
            y_input = (bhat_all[lb:ub]-y_mean)/y_std # standardize input y -y_mean
            regr = ElasticNetCV(cv = 5, n_alphas=10, l1_ratio=0.5, eps=0.1, 
                                random_state=42, fit_intercept=False).fit(X_input,y_input)
            breg_enet_block = regr.coef_*y_std
            
            breg.append(breg_enet_block)
    
        breg = np.concatenate(breg)
        breg_enet.append(breg)
        
    toc = time.time()
    time_used = toc-tic
        # print(time_used)
    breg_enet = np.array(breg_enet)
    return breg_enet, time_used

def shrink_nn(model,scaled_obs_betas,scaled_se,chr_ld,brkpts):
    # NN model
    top_r = int(model.name.split("top")[1])
    breg_NN = list()
    tic = time.time()
    
    for i in range(scaled_obs_betas.shape[1]):
        
        bhat_all = scaled_obs_betas[:,i]
        shat_all = scaled_se[:,i]
        
        breg = list()
        for i_bk in range(len(brkpts)-1):
            
            lb, ub = brkpts[i_bk], brkpts[i_bk+1]
            sim_ld = chr_ld[lb:ub,lb:ub]
            # normalize LD
            sim_ld = preprocessing.normalize(sim_ld)
            data_X = mf.construct_features(bhat_all[lb:ub], shat_all[lb:ub],
                                          sim_ld, top_r = top_r)
            breg.append(model(torch.tensor(data_X, dtype=torch.float32)).detach().numpy().squeeze())
    
        breg = np.concatenate(breg)
        breg_NN.append(breg)
        
    toc = time.time()
    time_used = toc-tic
    breg_NN = np.array(breg_NN)
    
    return breg_NN, time_used


def eval_snp_level(btrue,breg,perf_types=["rmse"], base_rec=np.linspace(0, 1, 101)):
    n_traits = btrue.shape[1]
    perfs_traits = list()
    prec_traits = list()
    for i_trait in range(n_traits):
        true = btrue[:,i_trait]
        pred = breg[:,i_trait]
        true = np.squeeze(true)
        pred = np.squeeze(pred)
        assert(true.shape==pred.shape)
        
        # evaluate
        perfs = list()
        for perf_type in perf_types:
            perf = perf_eval(true,pred,type=perf_type)
            perfs.append(perf)
        perfs_traits.append(perfs)
        
        is_true_nz = true!=0
        precision, recall, thresholds = precision_recall_curve(is_true_nz, np.abs(pred),drop_intermediate=True)
        prec = np.interp(base_rec, recall[::-1], precision[::-1])   
        prec_traits.append(prec)

    return perfs_traits, prec_traits

    

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

def get_cls_assoc(truncate_Sigma, rad_thre, eigval_times_thre):
    n_traits = truncate_Sigma[0].shape[0]
    all_possible_causal_types = list()
    ref_vecs = list()
    for n in range(1,n_traits+1):
        for combo in itertools.combinations(range(1,n_traits+1), n):
            all_possible_causal_types.append(combo)
            ref_vec = np.zeros(n_traits)
            ref_vec[np.array(combo)-1] = 1
            ref_vecs.append(ref_vec.astype(int))
    
    # compute eigen-info, and get the orientation of the clusters
    sigma_eiginfo = list()
    for i, covar in enumerate(truncate_Sigma):
    
        # Calculate the eigenvectors and eigenvalues
        eigenval, eigenvec = np.linalg.eig(covar)
        idx = eigenval.argsort()[::-1]   
        eigenval = eigenval[idx]
        eigenvec = eigenvec[:,idx]
        
        angles = list() # for each eigenvector, its angle to each of the ref vector
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

    # get the assocation type of each inferred clusters
    cls_all_possible_is_assoc = list()
    for eigvals,eigvecs,angs in sigma_eiginfo:
    
        is_specific = [False]*len(all_possible_causal_types)
        # largest eigenval and its corresponding angles to ref vectors
        if np.all([eigvals[0]/e>eigval_times_thre for e in eigvals[1:]]): # large difference in axes
            for i_a,ang in enumerate(angs):
                if angs[0][i_a]<rad_thre or angs[0][i_a]>(2*np.pi-rad_thre): # close to x
                    is_specific[i_a] = True
        
        proj_on_axes = list()
        for i_axis in range(len(angs)):
            proj_on_axes.append([])
            for i_a in range(len(angs)):
                ang_to_axis = angs[i_a][i_axis]
                proj_on_axes[i_axis].append(np.abs(eigvals[i_a]*np.cos(ang_to_axis)))
        max_on_axes = np.max(proj_on_axes,axis=1)
        # True if >>, False if not
        max_on_axes_pw = max_on_axes[:,None]/max_on_axes[None,:]>eigval_times_thre
    
        all_possible_is_assoc = list()
        for ct in all_possible_causal_types:
            is_assoc = True
            large_traits = np.array(ct)
            small_traits = np.array(list(set(np.arange(1,n_traits+1)) - set(large_traits)))
            
            # within the same set: all pw should be False
            for comb in itertools.combinations(large_traits,2):
                is_assoc &= ~max_on_axes_pw[comb[0]-1,comb[1]-1]
                is_assoc &= ~max_on_axes_pw[comb[1]-1,comb[0]-1]
            for comb in itertools.combinations(small_traits,2):
                is_assoc &= ~max_on_axes_pw[comb[0]-1,comb[1]-1]
                is_assoc &= ~max_on_axes_pw[comb[1]-1,comb[0]-1]
            # across two sets: >>
            for comb in itertools.product(large_traits, small_traits):
                is_assoc &= max_on_axes_pw[comb[0]-1,comb[1]-1]        
            all_possible_is_assoc.append(is_assoc)
        cls_all_possible_is_assoc.append(all_possible_is_assoc)
    cls_all_possible_is_assoc = np.array(cls_all_possible_is_assoc)
    cls_all_possible_is_assoc = pd.DataFrame(cls_all_possible_is_assoc, columns=all_possible_causal_types)

    return cls_all_possible_is_assoc


def eval_multi_gene_level(btrue,cls_all,pred_K,cls_all_possible_is_assoc,causal_types,genes_chr, sig_cls_thre):
    
    # count and frac of truly causal variants in genes
    causal_snp_cnt = np.array([np.sum(btrue[np.array(g).astype(int),:]!=0,axis=0) for g in genes_chr['SNPS']])
    causal_snp_frac = causal_snp_cnt/genes_chr['N_SNPS'].values[:,None]
    is_truly_causal = np.sum(causal_snp_cnt>0,axis=1)
    truly_causal = genes_chr[is_truly_causal>0].index
    
    # whether genes are truly causal to each type
    gene_is_causal = list()
    for ct in causal_types:
        ct = np.array(ct)-1
        # causal_snp_frac
        ct_assoc = np.all(causal_snp_frac[:,ct]>0, axis=1)
        gene_is_causal.append(ct_assoc)
    gene_is_causal = np.array(gene_is_causal).T
    
    # get the cluster fractions of each gene
    genes_cls_cnt = list()
    for i_g in range(len(genes_chr)):
        snps_g = np.array(genes_chr.loc[i_g]['SNPS']).astype(int)
        betas_cls = cls_all[snps_g]
        cls_cnt = pd.Series(betas_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
        genes_cls_cnt.append(cls_cnt[np.arange(pred_K)].values)
    genes_cls_cnt = np.array(genes_cls_cnt)
    genes_cls_frac = list()
    for cls in range(pred_K):
        genes_cls_frac.append(genes_cls_cnt[:,cls]/genes_chr['N_SNPS'].values)
    genes_cls_frac = np.array(genes_cls_frac).T    
    
    nz_cls_perc = pd.Series(cls_all[cls_all>=0]).value_counts(normalize=True).reindex(np.arange(pred_K), fill_value=0)
    nz_cls_perc_cumsum = nz_cls_perc.cumsum()
    sig_cls = np.where((nz_cls_perc_cumsum<sig_cls_thre))[0]
    
    meas_types = ['pcorr','prec','rec','fs']
    perf_gene_level_ct = list()
    perf_gene_level_ct_index = list()
    for ct in causal_types:
        cls_assoc = np.where(cls_all_possible_is_assoc[ct])[0]
        sig_cls_assoc = np.array(list(set(sig_cls).intersection(cls_assoc)))
        if len(sig_cls_assoc)>0:
            genes_nz_assoc = np.any(genes_cls_frac[:,sig_cls_assoc]>0,axis=1)
            var_of_interest = np.where(genes_nz_assoc)[0]
            frac_of_interest = np.sum(genes_cls_frac[var_of_interest,:][:,sig_cls_assoc],axis=1)
            truly_frac = np.sum(causal_snp_frac[var_of_interest,:][:,np.array(ct)-1],axis=1)
            n_overlap = np.sum((truly_frac>0) & (frac_of_interest>0))
            
            prec, rec, fscore, _ = precision_recall_fscore_support(gene_is_causal[:,causal_types.index(ct)],
                                                                   genes_nz_assoc, pos_label=True)
            prec, rec, fscore = prec[1], rec[1], fscore[1]
            cor = np.corrcoef(truly_frac,frac_of_interest)[0,1]
            perf_gene_level_ct.append([cor,prec, rec, fscore])
            perf_gene_level_ct_index.append(["{}_{}".format(meas,ct) for meas in meas_types])
    
    df_gene_level = pd.Series(np.hstack(perf_gene_level_ct),index=np.hstack(perf_gene_level_ct_index))
    
    return df_gene_level
    
def eval_uni_gene_level(btrue, breg, method, chr_ld, genes_chr, sig_threshold=0.05, base_rec=np.linspace(0, 1, 101)):
    n_traits = btrue.shape[1]
    perf_unigene_traits = list()
    prec_unigene_traits = list()

    for i_trait in range(n_traits):
    
        beta_true = btrue[:,i_trait]
        beta_reg = breg[:,i_trait]
        
        if method=="ENET":
            zero_cutoff = 0
            beta_nz = beta_reg[np.abs(beta_reg)>0]
        else:
            beta_nz, zero_cutoff = mf.get_nz_effects(beta_reg, fold_min=200, fold_max=10, 
                                                  zero_cutoff=1e-3, adjust_max = 10, adjust_rate = 1.5)

        truncate_Sigma, truncate_pi, pred_K, pred_cls = mf.clustering(beta_nz, K = 20, n_runs=25)
        
        # reproduce cls labels of both zeros and non-zeros
        breg_filtered_uni = mf.threshold_vals(beta_reg, zero_cutoff=zero_cutoff)
        cls_uni = -np.ones(beta_reg.shape[0])
        is_nz_uni = np.abs(breg_filtered_uni)>0
        cls_uni[is_nz_uni] = pred_cls
    
        # gene enrichment
        eps_eff_cls = max(1,np.where(np.cumsum(truncate_pi)>0.05)[0][0])
        epsilon_effect = truncate_Sigma[eps_eff_cls][0][0]
        
        # enrichment analysis
        df_enrich = genes_chr.copy()
        enrich_results = mf.enrichment_test(df_enrich, epsilon_effect, beta_reg, chr_ld, use_davies=False)
        df_test = pd.DataFrame(enrich_results)
        df_enrich = pd.concat([df_enrich[df_enrich.columns[:-1]],df_test],axis=1)
    
        # evaluate
        eff_s_true = [np.sum(beta_true[np.array(g).astype(int)]**2)/len(g) for g in genes_chr['SNPS']]
        is_truly_causal = np.array([np.any(beta_true[np.array(g).astype(int)]>0) for g in genes_chr['SNPS']])
        truly_causal = genes_chr[is_truly_causal].index
        
        # print(method)
        p_enrich = df_enrich["P"].values
        rejected, pval_corrected = fdrcorrection(p_enrich,
                                                 alpha=sig_threshold, 
                                                 method='indep', is_sorted=False)
        idx_sig = np.where(rejected)[0]
        is_sig = pval_corrected<sig_threshold
        
        prc, rec, fs, _ = precision_recall_fscore_support(is_truly_causal,is_sig, 
                                                          labels=[False, True],
                                                          average='binary', pos_label=True)
        
        neglogp = -np.log10(pval_corrected)
        aps = average_precision_score(is_truly_causal,neglogp)
        
        scorr = sp.stats.spearmanr(eff_s_true, neglogp)[0]
        precision, recall, thresholds = precision_recall_curve(is_truly_causal, neglogp, drop_intermediate=True)
        prec = np.interp(base_rec, recall[::-1], precision[::-1])
        perf_unigene_traits.append([fs, aps,scorr])
        prec_unigene_traits.append(prec)
        
    return perf_unigene_traits, prec_unigene_traits