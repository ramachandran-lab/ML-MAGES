import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from tqdm import tqdm
from pandas_plink import read_plink
import itertools
from functools import reduce
import argparse
import time

import ml_mages
import _sim_funcs as sf
import _train_funcs as tf
# from _sim_funcs import prepare_cov, scale_by_quantile

class MultiTraitSimulator:
    def __init__(self, X, n_traits, fcg, fcs, causal_types = [], causal_types_prob = []):
        self.n_inds = X.shape[0]
        self.n_snps = X.shape[1]
        self.X = X
        self.n_traits = n_traits
        self.fcg = fcg
        self.fcs = fcs
        if len(causal_types)==0:
            self.causal_types = list()
            for n in range(1,n_traits+1):
                for combo in itertools.combinations(range(1,n_traits+1), n):
                    self.causal_types.append(combo)
        else:
            self.causal_types = causal_types
        if len(causal_types_prob)==0:
            self.causal_types_prob = np.ones(len(self.causal_types))/len(self.causal_types)
        else:
            self.causal_types_prob = causal_types_prob
            
        
    def simulate_raw_beta(self, sim_cov_multi, gene_sizes, genes_start, genes_end): 
        causal_snps = list()
        true_beta_raw = np.zeros((self.n_snps, self.n_traits))
         
        n_genes = len(gene_sizes)
        n_causal_gene_any = int(self.fcg*n_genes) 
        n_causal_gene_types = (np.array(self.causal_types_prob)*n_causal_gene_any).astype(int)
        
        idx_causal_gene_any = np.random.choice(np.arange(n_genes),
                                               n_causal_gene_any,replace=False)
        idx_causal_gene_type = np.split(idx_causal_gene_any, np.cumsum(n_causal_gene_types)[:-1])
        idx_causal_gene_type = [np.sort(a) for a in idx_causal_gene_type]

        for i_c_g,c_g in enumerate(idx_causal_gene_type):
            causal_snps_type = list()
            for i_g in c_g:
                n_causal_snps_in_g = np.max([2,int(gene_sizes[i_g]*self.fcs)])
                idx_causal_snps = np.random.choice(np.arange(genes_start[i_g],genes_end[i_g]),
                                                  n_causal_snps_in_g, replace=False)
                idx_causal_snps = np.sort(idx_causal_snps)
                causal_snps_type.append(idx_causal_snps)
            
                causal_traits = self.causal_types[i_c_g]
                n = len(causal_traits)
                Sigma = sim_cov_multi[n][np.random.choice(np.arange(len(sim_cov_multi[n])),1)[0]]
            
                if n==1:
                    beta_sim = np.random.normal(0, Sigma, size=n_causal_snps_in_g)[:,None]
                else:
                    beta_sim = np.random.multivariate_normal(np.zeros(n), Sigma, size=n_causal_snps_in_g)
                for ii,ics in enumerate(idx_causal_snps):
                    true_beta_raw[ics,np.array(causal_traits)-1] = beta_sim[ii,:]
            
            causal_snps.append(causal_snps_type)

        df_sim_genes = pd.DataFrame(np.vstack([gene_sizes,genes_start,genes_end]).T, columns=["n_snps","gene_start","gene_end"])
        df_sim_genes["is_causal"] = [s in idx_causal_gene_any for s in np.arange(n_genes)]
        causal_types = np.ones(n_genes)*(-1)
        for i_t, t in enumerate(idx_causal_gene_type):
            causal_types[t] = i_t
        df_sim_genes["causal_type"] = causal_types
        df_sim_genes["causal_type"] = df_sim_genes["causal_type"].astype(int)

        return true_beta_raw, df_sim_genes, causal_snps

    def simulate_true_beta(self, true_beta_raw, h2s): 
        true_betas = np.zeros_like(true_beta_raw)
        y = np.zeros((self.n_inds,self.n_traits))
        for i_trait in range(self.n_traits):
            # adjust h2 of raw betas
            h2 = h2s[i_trait]
            braw = true_beta_raw[:,i_trait]
            # simulate noise
            noise = np.random.normal(0,1,self.n_inds)
            noise = (noise-np.mean(noise))/np.std(noise)
            noise_var = np.var(noise)
            noise_scale = np.sqrt((1-h2)/noise_var)
            noise = noise * noise_scale # rescale to have variance 1-h2
        
            # simulate y's
            tmp = self.X @ braw
            tmp_var = np.var(tmp)
            tmp_scale = np.sqrt(h2/tmp_var)
            btrue = braw * tmp_scale # rescale to let y_g have variance h2
            y_g = self.X @ btrue
            y_i = y_g + noise
            y_i = (y_i-y_i.mean())/y_i.std()
            y[:,i_trait] = y_i
            true_betas[:,i_trait] = btrue
        return true_betas, y
    
    def GWAS(self,y):
        gwas_res_list = list()
        obs_betas = np.zeros((self.X.shape[1],self.n_traits))
        obs_ses = np.zeros((self.X.shape[1],self.n_traits))
        for i_trait in range(self.n_traits):
            # get beta_hat for all SNPs
            gwas_res = pd.DataFrame(columns=['BETA_HAT','SE','P','T'])
            for j in range(self.n_snps):   
                result = sm.OLS(y[:,i_trait], self.X[:,j]).fit()
                gwas_res.loc[len(gwas_res.index)] = [result.params[0], result.bse[0], result.pvalues[0], result.tvalues[0]]
            gwas_res['TRAIT'] = i_trait+1
            gwas_res_list.append(gwas_res)
            obs_betas[:,i_trait] = gwas_res['BETA_HAT'].values
            obs_ses[:,i_trait] = gwas_res['SE'].values
        
        return obs_betas, obs_ses, pd.concat(gwas_res_list)


    
def main(args):
    
    print("-----Required Arguments: ")
    chr = args.chr
    print("chr:", chr) # e.g., 15
    geno_file = args.geno_file
    print("geno_file:", geno_file)
    full_ld_file = args.full_ld_file
    print("full_ld_file:", full_ld_file)
    gwa_files = args.gwa_files.split(",")
    print("gwa_files:", gwa_files)
    gene_list_file = args.gene_list_file
    print("gene_list_file:", gene_list_file)
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("output_path:", output_path)

    print("-----Optional Arguments: ")
    n_inds = args.n_inds 
    print("n_inds (default 10000):", n_inds)
    min_gene_size = args.min_gene_size
    print("min_gene_size (default 10):", min_gene_size)
    n_traits = args.n_traits
    print("n_traits (default 3):", n_traits)
    causal_types = args.causal_types
    causal_types = [tuple([int(e) for e in t.split(",")]) for t in causal_types.split(";")]
    # e.g., "1;2;3;1,2;1,2,3" gives causal_types = [(1,),(2,),(3,),(1,2),(1,2,3)]
    print("causal_types (default 1;2;3;1,2;1,2,3):", causal_types)
    n_sim = args.n_sim
    print("n_sim (default 100):", n_sim)
    
    # fixed settings and random seed
    fcg, fcs = 0.1, 0.4
    h2s = [0.8]*n_traits
    np.random.seed(42)

    print("=====//ML-MAGES// Helper Function: Simulate Data for Performance Evaluation=====")
    start_time = time.time()

    # load genotype data
    (bim, fam, bed) = read_plink(("{}.bed".format(geno_file)),verbose=False)
    print("Genotype data size:", bim.shape, bed.shape)
    
    # load ld
    chr_ld = np.loadtxt(full_ld_file)
    print("Chr {}, LD size: {}x{}".format(chr, chr_ld.shape[0],chr_ld.shape[1]))

    # get n_snps
    n_snps = bim.shape[0]

    # load gene data
    gene_list = pd.read_csv(gene_list_file)
    gene_list['CHR'] = gene_list['CHR'].astype(str)
    gene_list = gene_list[(gene_list['CHR']!='X') & (gene_list['CHR']!='Y')].reset_index(drop=True)
    gene_list['CHR'] = gene_list['CHR'].astype(int)
    
    gene_list = gene_list[gene_list['CHR']==chr].reset_index(drop=True)
    all_genes = gene_list['GENE']
    gene_start = gene_list['START']
    gene_end = gene_list['END']
    genes_chr = pd.DataFrame(columns=['CHR','GENE','START','END','N_SNPS','SNP_START','SNP_END','SNPS'])

    all_pos = bim["pos"].values
    
    for i_gene in range(len(gene_list)):
        snps_in_range = np.where((all_pos>=gene_start[i_gene]) & (all_pos<=gene_end[i_gene]))[0]
        nsnps = len(snps_in_range)
        if nsnps>1:
            genes_chr.loc[len(genes_chr.index)] = [chr, all_genes[i_gene], gene_start[i_gene], gene_end[i_gene], nsnps, snps_in_range[0],snps_in_range[-1],snps_in_range] 
    print("Altogether {} genes with number of SNPs between {} and {}".format(len(genes_chr), genes_chr['N_SNPS'].min(), genes_chr['N_SNPS'].max()))
    
    genes_chr = genes_chr[genes_chr['N_SNPS']>=min_gene_size].reset_index(drop=True)
    gene_sizes = genes_chr['N_SNPS'].values
    genes_start = genes_chr['SNP_START'].values
    genes_end = genes_chr['SNP_END'].values
    n_genes = len(genes_chr)
    print("Considering {} genes with size >={}".format(n_genes,min_gene_size))
    
    # load real data 
    beta_real, se_real = sf.load_real_chr_data(gwa_files)
    # scale sim data by fitting a Laplace distribution to the real data
    loc_real, scale_real = sp.stats.laplace.fit(beta_real, floc=0)
    print("Real gwa size", len(beta_real))

    # generate random settings
    causal_types_prob = np.random.dirichlet([5 if len(e)==1 else 10 for e in causal_types],1)[0]
    print("causal_types_prob:",causal_types_prob)
    sim_cov_multi = sf.prepare_cov(n_traits,n_sim_var=50, sig_thre=0.1, pcorr_thre=0.6)

    # simulate
    for i_sim in tqdm(range(n_sim)):

        sampled_inds = np.sort(np.random.choice(bed.shape[1],n_inds))
        X_sim = bed[:,sampled_inds].compute().T
        
        # fill in NAs
        geno_col_mean = np.nanmean(X_sim, axis=0)
        nan_inds = np.where(np.isnan(X_sim))
        X_sim[nan_inds] = np.take(geno_col_mean, nan_inds[1])
        
        # subset LD
        maf_sim = X_sim.mean(axis=0)/2
        maf_sim[maf_sim>0.5] = 1-maf_sim[maf_sim>0.5]
    
        mts = MultiTraitSimulator(X_sim, n_traits, fcg, fcs, 
                              causal_types = causal_types, 
                              causal_types_prob = causal_types_prob)
        true_beta_raw, df_sim_genes, causal_snps = mts.simulate_raw_beta(sim_cov_multi, gene_sizes, genes_start, genes_end)
        
        causal_snps_any = list(reduce(lambda x, y: x + y, causal_snps, []))
        causal_snps_any = np.concatenate(causal_snps_any)
        n_causal_snps_any = len(causal_snps_any)
        
        # based on raw true beta, simulate true beta, then y for each trait
        true_betas, y = mts.simulate_true_beta(true_beta_raw, h2s)
        
        # GWA on y and X
        obs_betas, obs_ses, gwas_res = mts.GWAS(y)
        
        # scale data
        scaled_obs_betas = np.zeros_like(obs_betas)
            
        # get ECDF of the simulated data
        sim_ecdf = sp.stats.ecdf(obs_betas.flatten()).cdf
        # scale betas
        data_trans_probs = sim_ecdf.evaluate(obs_betas)
        data_trans = sp.stats.laplace.ppf(data_trans_probs, loc=loc_real, scale=scale_real)
        data_trans[np.isinf(data_trans)] = np.nanmax(data_trans[~np.isinf(data_trans)])
        scaled_obs_betas = data_trans
        # scale se
        scaled_se = tf.scale_by_quantile(obs_ses.flatten(),se_real,q=0.01).reshape(-1,n_traits)
        
        scaled_true_betas = np.zeros_like(true_betas)
        true_beta_scales = np.zeros(mts.n_traits)
        for i_trait in range(mts.n_traits):
            z = np.abs(sp.stats.zscore(scaled_obs_betas[:,i_trait]))
            valid_indices = np.where(z <= 3)[0]
            true_beta_scale = scaled_obs_betas[valid_indices,i_trait].std()/true_betas[valid_indices,i_trait][true_betas[valid_indices,i_trait]!=0].std()
            true_beta_scales[i_trait] = true_beta_scale
            scaled_true_betas[:,i_trait] = true_betas[:,i_trait]*true_beta_scale
        
        # save simulation data
        tmp = np.hstack([scaled_true_betas,scaled_obs_betas,scaled_se])
        np.savetxt(os.path.join(output_path,"data_sim{}.txt".format(i_sim)), tmp, delimiter=',')

    end_time = time.time()
    print("Simulating evaluation data takes {:.2f} seconds".format(end_time - start_time))

    print("============DONE============")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide the required arguments')

    # Required argument
    parser.add_argument('--chr', type=int, required=True, help='Chr of the real data to base simulation on')
    parser.add_argument('--geno_file', type=str, required=True, help='path to the genotyped data (in PLINK format)')
    parser.add_argument('--full_ld_file', type=str, required=True, help='path to the full LD file')
    parser.add_argument('--gwa_files', type=str, required=True, help='comma-separated list of GWA files used for matching the simulation to real data')
    parser.add_argument('--gene_list_file', type=str, required=True, help='path to the gene list file (with columns CHR, GENE, START, END)')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the output simulation files')

    # Optional arguments
    parser.add_argument('--n_inds', type=int, required=False, default=10000, help='number of individuals to be sampled for each simulation')
    parser.add_argument('--min_gene_size', type=int, required=False, default=10, help='min number of variants in a gene for gene to be considered as candidate')
    parser.add_argument('--n_traits', type=int, required=False, default=3, help='number of traits to be simulated')
    parser.add_argument('--causal_types', type=str, required=False, default="1;2;3;1,2;1,2,3", help='causal types to be simulated')
    parser.add_argument('--n_sim', type=int, required=False, default=100, help='number of simulations')

    args = parser.parse_args()
    main(args)
