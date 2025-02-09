import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from pandas_plink import read_plink
import ml_mages


class Simulator:
  def __init__(self, X, maf):
    self.n_inds = X.shape[0]
    self.n_snps = X.shape[1]
    self.X = X
    self.maf = maf

  def simulate_beta(self, h2, idx_snps_assoc):
    # simulate noise
    noise = np.random.normal(0,1,self.n_inds)
    noise = (noise-np.mean(noise))/np.std(noise)
    noise_var = np.var(noise)
    noise_scale = np.sqrt((1-h2)/noise_var)
    noise = noise * noise_scale # rescale to have variance 1-h2

    # simulate beta's 
    braw = np.zeros(self.n_snps)
    betas = np.random.normal(0,np.sqrt((2*self.maf*(1.0-self.maf))**0.75))
    braw[idx_snps_assoc] = betas[idx_snps_assoc]

    # simulate y's
    tmp = self.X @ braw
    tmp_var = np.var(tmp)
    tmp_scale = np.sqrt(h2/tmp_var)
    btrue = braw * tmp_scale # rescale to let y_g have variance h2
    y_g = self.X @ btrue
    y = y_g + noise
    y = (y-y.mean())/y.std()

    return y, btrue

  def GWAS(self,y):
    # get beta_hat for all SNPs
    gwas_res = pd.DataFrame(columns=['BETA_HAT','SE','P','T'])
    for j in range(self.n_snps):   
        result = sm.OLS(y, self.X[:,j]).fit()
        gwas_res.loc[len(gwas_res.index)] = [result.params[0], result.bse[0], result.pvalues[0], result.tvalues[0]]
    return gwas_res


def main():
    if len(sys.argv) < 5:
        print("Usage: {} simulate_train.py sim_chrs(separated by comma) geno_path ld_path output_path (n_inds=10000) (n_snps=1000) (n_sim=200) (topr=25)".format(sys.argv[0]))
        sys.exit(1)

    sim_chrs = [int(c) for c in sys.argv[1].split(",")]
    #e.g., [18,19,20,21,22]
    print("sim_chrs:", sim_chrs)
    geno_path = sys.argv[2]
    print("geno_path:", geno_path)
    ld_path = sys.argv[3]
    print("ld_path:", ld_path)
    sim_path = sys.argv[4]
    if not os.path.exists(sim_path):
        os.makedirs(sim_path)
    print("output_path:", sim_path)
    n_inds = int(sys.argv[5]) if len(sys.argv)>5 else 10000
    print("n_inds:", n_inds)
    n_snps = int(sys.argv[6]) if len(sys.argv)>6 else 1000
    print("n_snps:", n_snps)
    n_sim = int(sys.argv[7]) if len(sys.argv)>7 else 200
    print("n_sim:", n_sim)
    top_r = int(sys.argv[8]) if len(sys.argv)>8 else 25
    print("top_r:", top_r)

    max_assoc_frac = 0.05
    max_n_assoc = n_snps*max_assoc_frac
    h2_range = [0.1,0.9] #[0.1,0.9]
    np.random.seed(42)

    

    for chr in sim_chrs:
        sim_label = "ninds{}_nsnps{}_nsim{}_topr{}_chr{}".format(n_inds,n_snps,n_sim,top_r,chr)
        print(sim_label)

        data_sim_X = list()
        data_sim_y = list()
        meta_info_sim = list() # chr,sampled_snps_start,h2,n_snps_assoc
        pheno_sim = list()

        # load genotype data
        (bim, fam, bed) = read_plink(os.path.join(geno_path, "ukb_chr{}.qced.bed".format(chr)),verbose=False)
        print(bim.shape, bed.shape)

        # load ld
        ld_file = os.path.join(ld_path,"ukb_chr{}.qced.ld".format(chr))
        chr_ld = np.loadtxt(ld_file)
        print("Chr {}, LD size: {}x{}".format(chr, chr_ld.shape[0],chr_ld.shape[1]))
          
        for i_sim in tqdm(range(n_sim)):

            # random sample individuals and snps
            sampled_inds = np.sort(np.random.choice(bed.shape[1],n_inds))
            sampled_snps_start = np.random.choice(bed.shape[0]-n_snps)
            sampled_snps = np.arange(sampled_snps_start,sampled_snps_start+n_snps)
            X_sim = bed[:,sampled_inds][sampled_snps,:].compute().T
            
            # fill in NAs
            geno_col_mean = np.nanmean(X_sim, axis=0)
            nan_inds = np.where(np.isnan(X_sim))
            X_sim[nan_inds] = np.take(geno_col_mean, nan_inds[1])
            
            # subset LD
            ld_sim = chr_ld[:,sampled_snps][sampled_snps,:]
            
            maf_sim = X_sim.mean(axis=0)/2
            maf_sim[maf_sim>0.5] = 1-maf_sim[maf_sim>0.5]
            
            # simulate betas
            sim = Simulator(X_sim, maf_sim)
            n_snps_assoc = np.random.randint(1,max_n_assoc)
            h2 = np.random.uniform(h2_range[0], h2_range[1])
            idx_snps_assoc = np.sort(np.random.choice(np.arange(n_snps), n_snps_assoc, replace=False))
            y, btrue = sim.simulate_beta(h2, idx_snps_assoc)
            y = (y-y.mean())/y.std() # standardize y

            # run gwas
            gwas_res = sim.GWAS(y)
            bhat = gwas_res['BETA_HAT'].values
            shat = gwas_res['SE'].values
            data_X = ml_mages.construct_features(bhat, shat, ld_sim, top_r)

            # store data
            data_sim_X.append(data_X)
            data_sim_y.append(btrue)
            meta_info_sim.append([chr,sampled_snps_start,h2,n_snps_assoc])
            pheno_sim.append(y)

        X_sim = np.concatenate(data_sim_X,axis=0)
        y_sim = np.concatenate(data_sim_y,axis=0)
        meta_sim = np.vstack(meta_info_sim)

        print(X_sim.shape, y_sim.shape, meta_sim.shape)

        # save simulated data
        save_file = os.path.join(sim_path,"{}.X".format(sim_label))
        np.savetxt(save_file, X_sim, delimiter=',')
        save_file = os.path.join(sim_path,"{}.y".format(sim_label))
        np.savetxt(save_file, y_sim, delimiter=',')
        save_file = os.path.join(sim_path,"{}.meta".format(sim_label))
        np.savetxt(save_file, meta_sim, delimiter=',')


if __name__ == "__main__":
    main()