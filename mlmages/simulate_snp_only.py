import os
import numpy as np
import pandas as pd
import scipy as sp
from pandas_plink import read_plink
import itertools 
import argparse

from ._sim_funcs import simulate_effects, simulate_phenotype, compute_gwas_summary
from ._util_funcs import load_gwas_file, disp_params, parse_file_list, scale_laplace, scale_by_quantile, scale_true_beta, str2bool

def simulate_snp_only(p_list, h_list, s_list, w_list, n_sim, nsnp, nind,
                      plink_file, af_file, score_file, transform_data, asymmetric, gwas_files,
                      sim_path, sim_prefix):

    os.makedirs(sim_path, exist_ok=True)

    param_list = list(itertools.product(p_list, h_list, s_list, w_list))
    print("Total number of parameter combinations:", len(param_list))
    gwas_files = parse_file_list(gwas_files)

    df_params = pd.DataFrame(param_list)
    df_params.columns = ['p_causal','h2','s','w']
    out = "{}_sim_params.csv".format(sim_prefix) if sim_prefix != "" else "sim_params.csv"
    df_params.to_csv(os.path.join(sim_path,out))

    # load genotype data
    bim, fam, bed = read_plink(plink_file,verbose=False)
    print("Genotype data loaded, #SNPs: {}, #Individuals: {}".format(bed.shape[0], bed.shape[1]))
    assert bed.shape[0]>=nsnp, "Number of SNPs in the genotype data ({}) is less than the requested number of SNPs ({}).".format(bed.shape[0], nsnp)
    assert bed.shape[1]>=nind, "Number of individuals in the genotype data ({}) is less than the requested number of individuals ({}).".format(bed.shape[1], nind)

    # load af
    af = pd.read_csv(af_file, delimiter='\s+', )
    af = af['MAF'].values
    # load ld scores
    ld_scores = np.loadtxt(score_file)

    for i_param in range(len(param_list)):
        print(f"Parameter set {i_param}")
        p_causal, h2, s, w = param_list[i_param]

        snp_start_list = list()
        inds_list = list()
        for i_sim in range(n_sim):
            snp_start = np.random.choice(bed.shape[0]-nsnp,1)[0]
            sampled_snps = np.arange(snp_start,snp_start+nsnp)
            sampled_inds = np.sort(np.random.choice(bed.shape[1],nind))
            G = bed[:,sampled_inds][sampled_snps,:].compute().T
            
            snp_mean = np.nanmean(G, axis=0)
            nan_inds = np.where(np.isnan(G))
            G[nan_inds] = np.take(snp_mean, nan_inds[1])

            # simulate
            beta = simulate_effects(af[snp_start:(snp_start+nsnp)], ld_scores[snp_start:(snp_start+nsnp)], 
                                    nsnp, p_causal, s, h2, w)
            Y = simulate_phenotype(G, beta, h2)
            df_gwas = compute_gwas_summary(G, Y)
            df_gwas['beta'] = beta
            df_gwas['ldsc'] = ld_scores[snp_start:(snp_start+nsnp)]

            # save
            out = "{}_param{}_sim{}.csv".format(sim_prefix,i_param,i_sim) if sim_prefix != "" else "param{}_sim{}.csv".format(i_param,i_sim)
            df_gwas.to_csv(os.path.join(sim_path,out))
            snp_start_list.append(snp_start)
            # inds_list.append(",".join([str(si) for si in sampled_inds]))
            
        df = pd.DataFrame({'snp_start': snp_start_list})
        df[['p_causal','h2','s','w']] = p_causal, h2, s, w
        df[['nind', 'nsnp']]  = nind, nsnp
        out = "{}_param{}_meta.csv".format(sim_prefix,i_param) if sim_prefix != "" else "param{}_meta.csv".format(i_param)
        df.to_csv(os.path.join(sim_path,out))

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

        for i_param in range(len(param_list)):
            p_causal, h2, s, w = param_list[i_param]

            for i_sim in range(n_sim):
                out = "{}_param{}_sim{}.csv".format(sim_prefix,i_param,i_sim) if sim_prefix != "" else "param{}_sim{}.csv".format(i_param,i_sim)
                df_sim = pd.read_csv(os.path.join(sim_path,out), index_col=0)
                
                beta = df_sim['beta'].values
                beta_hat = df_sim['beta_hat'].values
                se = df_sim['se'].values
                
                scaled_beta_hat = scale_laplace(beta_hat, loc_ref=loc_real, scale_ref=scale_real, kappa=kappa)
                scaled_se = scale_by_quantile(se.flatten(),se_real, q=0.01)
                scaled_beta = scale_true_beta(beta, beta_hat, scaled_beta_hat, q=0.0)
                
                df_sim_scaled = pd.DataFrame({'beta_hat':scaled_beta_hat,
                                            'se': scaled_se,
                                            'beta': scaled_beta})
                # save
                out = "{}_param{}_scaled_sim{}.csv".format(sim_prefix,i_param,i_sim) if sim_prefix != "" else "param{}_scaled_sim{}.csv".format(i_param,i_sim)
                df_sim_scaled.to_csv(os.path.join(sim_path,out))


def main():
    print("RUNNING: simulate_snp_only")

    parser = argparse.ArgumentParser(description='Simulate SNP-only data for training')

    parser.add_argument('--p_list', type=float, nargs='+', default=[0.01,0.05], help='List of p_causal values')
    parser.add_argument('--h_list', type=float, nargs='+', default=[0.3,0.7], help='List of h2 values')
    parser.add_argument('--s_list', type=float, nargs='+', default=[-0.25,0], help='List of s values')
    parser.add_argument('--w_list', type=float, nargs='+', default=[-1,0], help='List of w values')
    parser.add_argument('--n_sim', type=int, default=100, help='Number of simulations')
    parser.add_argument('--nsnp', type=int, default=1000, help='Number of sampled SNPs')
    parser.add_argument('--nind', type=int, default=10000, help='Number of sampled individuals')

    parser.add_argument('--plink_file', type=str, help='Path to PLINK file (BED)')
    parser.add_argument('--af_file', type=str, help='Path to allele frequency file')
    parser.add_argument('--score_file', type=str, help='Path to LD score file')
    parser.add_argument("--transform_data", type=str2bool, nargs="?", default=True, help="Whether to transform the simulated data to match real data")
    parser.add_argument('--asymmetric', type=str2bool, nargs="?", default=False, help="Whether to use asymmetric Laplace distribution for data transformation")
    parser.add_argument("--gwas_files", nargs="+", default=[], help="List of GWAS summary files (to get real effect sizes)")

    parser.add_argument('--sim_path', type=str, help='Directory to save simulated data')
    parser.add_argument('--sim_prefix', type=str, default="", help="Result prefix")

    args = parser.parse_args()
    disp_params(args, title="INPUT SETTINGS")
    simulate_snp_only(**vars(args))


if __name__ == "__main__":
    main()



