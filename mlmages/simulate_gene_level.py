import os
import numpy as np
import pandas as pd
import scipy as sp
from pandas_plink import read_plink
import itertools 
import argparse

from ._sim_funcs import simulate_phenotype, compute_gwas_summary, simulate_covar, simulate_covar_base, scale_var_beta
from ._util_funcs import disp_params, str2bool, load_gwas_file, scale_laplace, scale_by_quantile, scale_true_beta

def simulate_gene_level(plink_file: str,
                        af_file: str,
                        score_file: str,
                        gene_file: str,
                        chroms: list[int],
                        i_sim: int,
                        n_trait: int,
                        sim_path: str,
                        sim_prefix: str="",
                        nind: int=10000,
                        s: float=-0.25,
                        w: float=0,
                        h2: float=0.7,
                        f_cg: float=0.05,
                        f_cs: float=0.3,
                        transform_data: bool=True,
                        asymmetric: bool=False,
                        gwas_files: list[str]=[]
                        ):
    os.makedirs(sim_path, exist_ok=True)
    # load genotype data
    bim, fam, bed = read_plink(plink_file,verbose=False)
    print("Genotype data loaded, #SNPs: {}, #Individuals: {}".format(bed.shape[0], bed.shape[1]))
    assert bed.shape[1]>=nind, "Number of individuals in the genotype data ({}) is less than the requested number of individuals ({}).".format(bed.shape[1], nind)

    # load af
    af = pd.read_csv(af_file, delimiter='\s+', )
    af = af['MAF'].to_numpy()
    # load ld scores
    ld_scores = np.loadtxt(score_file)

    # load gene data
    if chroms is None or len(chroms)==0:
        chroms = list(range(1,23))
    genes_chr_all = pd.read_csv(gene_file)
    required_cols = ['CHR','GENE','N_SNPS','start_idx_chr','chr_start_idx_gw']
    assert all(col in genes_chr_all.columns for col in required_cols), "Gene file is missing some required columns: {}!".format([col for col in required_cols if col not in genes_chr_all.columns])
    df_genes = genes_chr_all[genes_chr_all['CHR'].isin(chroms)].reset_index()
    n_genes = len(df_genes)
    gene_sizes = df_genes['N_SNPS'].values
    genes_start = df_genes['start_idx_chr'].values
    print("Gene file loaded, #genes:", n_genes)

    # set simulation parameters
    list_causal_gene_types = list(itertools.product([0,1], repeat=n_trait))
    f_cg = 0.05 
    f_cs = 0.3 
    n_causal_gene_any = int(f_cg*n_genes)
    if n_trait==1:
        causal_gene_types = [(1,)]
        ratio_causal_gene_types = (1,)
    elif n_trait==2:
        causal_gene_types = [(1,0),(0,1),(1,1)]
        ratio_causal_gene_types = (3,3,2)
    elif n_trait==3:
        causal_gene_types = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,1,1)]
        ratio_causal_gene_types = (1,1,1,2,2)
    else:
        causal_gene_types = [t for t in list_causal_gene_types if sum(t)>0]
        ratio_causal_gene_types = [1]*len(causal_gene_types)    
        

    print("s={}, w={}, h2={}, f_cg={}, f_cs={}".format(s,w,h2,f_cg,f_cs))

    n_sim_cov = 50
    sim_cov_base_all = dict()
    sim_cov_all_ngt3 = list()
    for i_trait in np.arange(1,n_trait+1):
        if i_trait>1 and i_trait<=3:
            sim_cov_base = simulate_covar_base(i_trait, n_sim_cov=50, min_corr=0.6)
            sim_cov_base_all[i_trait] = sim_cov_base
        elif i_trait>3:
            sim_cov = list()
            for i in range(n_sim_cov):
                sim_cov.append(simulate_covar(n_trait = i_trait, sig_thre = 0.01, coeff_thre = 0.6))
            sim_cov_all_ngt3 = sim_cov
        else:
            sim_cov_base_all[i_trait] = np.ones(n_sim_cov) #np.random.uniform(0.25,0.75,n_sim_cov)
    

    print("........................")
    print("......SIM {}......".format(i_sim))
    print("........................")

    sampled_inds = np.sort(np.random.choice(bed.shape[1],nind))
    G = bed[:,sampled_inds].compute().T
    
    snp_mean = np.nanmean(G, axis=0)
    nan_inds = np.where(np.isnan(G))
    G[nan_inds] = np.take(snp_mean, nan_inds[1])
    print("Genotype matrix for simulation generated, size: {}.".format(G.shape))
    nsnp = G.shape[1]
    print("Total #SNPs:", nsnp)

    if n_trait>1:
        frac_causal_gene_types = np.random.dirichlet(50*np.array(ratio_causal_gene_types)/np.sum(ratio_causal_gene_types), 1) #5,5,5,5,5
        n_causal_gene_types = (frac_causal_gene_types*n_causal_gene_any).astype(int)
    else:
        n_causal_gene_types = int(n_causal_gene_any)

    idx_causal_gene_any = np.random.choice(np.where(gene_sizes>=6)[0], #np.arange(n_genes),
                                            n_causal_gene_any,replace=False)
    idx_causal_gene_types = np.split(idx_causal_gene_any, np.cumsum(n_causal_gene_types)[:-1])
    idx_causal_gene_types = [np.sort(a) for a in idx_causal_gene_types]
    print("Causal types:", causal_gene_types)
    print("#genes for each causal type:", [len(a) for a in idx_causal_gene_types])

    df_meta = df_genes[['index','gene','CHR','N_SNPS','start_idx_chr']]

    # simulate true betas
    causal_snps = list()
    true_beta_raw = np.zeros((nsnp, n_trait))
    true_beta = np.zeros((nsnp, n_trait))
    # for each type of causal genes
    for i_c_g, c_g in enumerate(idx_causal_gene_types):
        c_t = causal_gene_types[i_c_g]

        idx_nz_trait = np.where(np.array(c_t)!=0)[0]
        n = len(idx_nz_trait)
        causal_type_index = list_causal_gene_types.index(c_t)
        df_meta.loc[c_g,'causal_type_index'] = causal_type_index

        causal_snps = list()
        for i_g in c_g:
            gene_indices = np.arange(genes_start[i_g],genes_start[i_g]+gene_sizes[i_g])

            n_causal_snps_in_g = np.max([2,int(gene_sizes[i_g]*f_cs)])
            rdm_idx = np.random.choice(np.arange(gene_sizes[i_g]), n_causal_snps_in_g, replace=False)
            idx_causal_snps = genes_start[i_g]+rdm_idx
            idx_causal_snps = np.sort(idx_causal_snps)  
            causal_snps.append(idx_causal_snps)

            af_s = af[idx_causal_snps]
            ld_s = ld_scores[idx_causal_snps]
            het = 2 * af_s * (1 - af_s) + 1e-8
            weight = (het ** s) * (ld_s**w)
            if n==1:
                beta_sim = list()
                for ii in range(n_causal_snps_in_g):
                    beta_sim.append(np.random.normal(0, np.sqrt(weight[ii]), size=1)[0])
                beta_sim = np.array(beta_sim)[:,None]
            elif n<=3:
                Sigma_base = sim_cov_base_all[n][np.random.choice(np.arange(n_sim_cov),1)[0]]
                beta_sim = list()
                for ii in range(n_causal_snps_in_g): 
                    Sigma = Sigma_base*weight[ii]
                    beta_sim.append(np.random.multivariate_normal(np.zeros(n), Sigma, size=1))
                beta_sim = np.vstack(beta_sim)
                assert beta_sim.shape[0]==n_causal_snps_in_g
                assert np.sum(np.isnan(beta_sim)==0)
            else:
                Sigma = sim_cov_all_ngt3[np.random.choice(np.arange(n_sim_cov),1)[0]]
                beta_sim = list()
                for ii in range(n_causal_snps_in_g): 
                    Sigma_w = Sigma*weight[ii]
                    beta_sim.append(np.random.multivariate_normal(np.zeros(n), Sigma_w, size=1))
                beta_sim = np.vstack(beta_sim)
                assert beta_sim.shape[0]==n_causal_snps_in_g
                assert np.sum(np.isnan(beta_sim)==0)
            assert beta_sim.shape[1]==n

            for ii,ics in enumerate(idx_causal_snps):
                true_beta_raw[ics,idx_nz_trait] = beta_sim[ii,:]
                     
        for idx in idx_nz_trait:
            true_beta[:,idx] = scale_var_beta(true_beta_raw[:,idx], af, h2)

        n_causal_snps = sum([len(idx_causal_snps) for idx_causal_snps in causal_snps])
        print("Causal type: {} (index: {}), total #causal snps: {}".format(list_causal_gene_types[causal_type_index],causal_type_index,n_causal_snps))
        
    df_meta = df_meta.fillna(value={'causal_type_index':-1, 'causal_snps':""})
    df_meta['causal_type_index'] = df_meta['causal_type_index'].astype(int)
    out = "{}genes_sim{}.csv".format(sim_prefix,i_sim) if sim_prefix != "" else "genes_sim{}.csv".format(i_sim)
    print("Saving simulation meta to:", os.path.join(sim_path,out))
    df_meta.to_csv(os.path.join(sim_path,out))

    true_beta_raw = true_beta # for new sim

    # simulate Y and GWAS
    Y = np.zeros((nind, n_trait))
    obs_beta_raw = np.zeros_like(true_beta_raw)
    obs_se_raw = np.zeros_like(true_beta_raw)
    for i_trait in range(n_trait):
        Y[:,i_trait] = simulate_phenotype(G, true_beta_raw[:,i_trait], h2)
        df_gwas = compute_gwas_summary(G, Y[:,i_trait])
        obs_beta_raw[:,i_trait] = df_gwas['beta_hat'].values
        obs_se_raw[:,i_trait] = df_gwas['se'].values

    # save simulation data (unscaled)
    tmp = np.hstack([true_beta_raw,obs_beta_raw,obs_se_raw])
    print("Shape of simulated data:", tmp.shape)
    out_file = os.path.join(sim_path,"{}_data_sim{}.txt".format(sim_prefix,i_sim)) if sim_prefix != "" else os.path.join(sim_path,"data_sim{}.txt".format(i_sim))
    print("Saving raw simulation data to:", out_file)
    np.savetxt(out_file, tmp, delimiter=',')

    if transform_data:
        assert len(gwas_files)>0, "No real GWAS files provided for data transformation!"
        print("Transforming simulated data to match real data distribution...")
        # load real data
        beta_real = list()
        se_real = list()
        for gwas_file in gwas_files:
            _, beta, se = load_gwas_file(gwas_file)
            beta_real.append(beta)
            se_real.append(se)
        print("Number of real GWAS files loaded:", len(gwas_files))
        beta_real = np.concatenate(beta_real)
        se_real = np.concatenate(se_real)
        
        if not asymmetric:
            kappa = None
            loc_real, scale_real = sp.stats.laplace.fit(beta_real, floc=0)
            print("Real data: loc={}, scale={}".format(loc_real, scale_real))
        else:
            kappa, loc_real, scale_real = sp.stats.laplace_asymmetric.fit(beta_real, floc=0)
            print("Real data (asymmetric): kappa={}, loc={}, scale={}".format(kappa, loc_real, scale_real))

        sim_data = np.loadtxt(out_file, delimiter=',')
        assert n_trait==sim_data.shape[1]//3
        beta = sim_data[:,:n_trait]
        beta_hat = sim_data[:,n_trait:2*n_trait]
        se = sim_data[:,2*n_trait:]
        scaled_beta_hat = scale_laplace(beta_hat, loc_ref=loc_real, scale_ref=scale_real, kappa=kappa)
        scaled_se = scale_by_quantile(se.flatten(),se_real, q=0.01).reshape(se.shape)
        scaled_beta = scale_true_beta(beta, beta_hat, scaled_beta_hat, q=0.0)
        if len(scaled_beta.shape)<2:
            scaled_beta = scaled_beta[:,None]
        tmp = np.hstack([scaled_beta,scaled_beta_hat,scaled_se])
        scaled_out_file = os.path.join(sim_path,"{}_data_scaled_sim{}.txt".format(sim_prefix,i_sim)) if sim_prefix != "" else os.path.join(sim_path,"data_scaled_sim{}.txt".format(i_sim))
        np.savetxt(scaled_out_file, tmp, delimiter=',')  
        print("Saving scaled simulation data to:", scaled_out_file)



def main():
    print("RUNNING: simulate_gene_level")

    parser = argparse.ArgumentParser(description='Simulate gene-level data for training')
    parser.add_argument('--plink_file', type=str, help='Path to PLINK file (BED)')
    parser.add_argument('--af_file', type=str, help='Path to allele frequency file')
    parser.add_argument('--score_file', type=str, help='Path to LD score file')
    parser.add_argument('--gene_file', type=str, help="Gene file")
    parser.add_argument('--chroms', type=int, nargs="*", default=None, help="Chromosome numbers (0–22). If not provided, all chromosomes will be used.")
    
    parser.add_argument('--i_sim', type=int, help='Simulation index')
    parser.add_argument('--n_trait', type=int, help='Number of traits')
    parser.add_argument('--sim_path', type=str, help='Directory to save simulated data')
    parser.add_argument('--sim_prefix', type=str, default="", help="Result prefix")

    # optional
    parser.add_argument('--nind', type=int, default=10000, help='Sample size')
    parser.add_argument('--s', type=float, default=-0.25, help='s parameter')
    parser.add_argument('--w', type=float, default=0, help='w parameter')
    parser.add_argument('--h2', type=float, default=0.7, help='Heritability')
    parser.add_argument('--f_cg', type=float, default=0.05, help='Fraction of causal genes')
    parser.add_argument('--f_cs', type=float, default=0.3, help='Fraction of causal SNPs within each gene')

    parser.add_argument("--transform_data", type=str2bool, nargs="?", default=True, help="Whether to transform the simulated data to match real data")
    parser.add_argument('--asymmetric', type=str2bool, nargs="?", default=False, help="Whether to use asymmetric Laplace distribution for data transformation")
    parser.add_argument("--gwas_files", nargs="+", default=[], help="List of GWAS summary files (to get real effect sizes)")

    args = parser.parse_args()
    disp_params(args, title="INPUT SETTINGS")
    simulate_gene_level(**vars(args))



if __name__ == "__main__":
    main()

