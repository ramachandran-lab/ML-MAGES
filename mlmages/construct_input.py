import os
import numpy as np
import pandas as pd
import argparse


def main(args):
    top_r = args.top_r
    print("using top r variants:", top_r)
    chrom = args.chrom
    print("CHR:", chrom)
    param_str = args.param_filter
    param_filter = {c.split("=")[0]:float(c.split("=")[1]) for c in param_str.split(",")} if param_str!="" else {}
    print("param_filter:", param_filter)
    ld_path = args.ld_path 
    sim_path = args.sim_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("output path:", output_path)
    nsim = args.nsim
    nsnp = args.nsnp

    # Using simulation data that are separated by chrom and param
    # store training data
    X_all = []
    y_all = []

    # load param_meta
    df_params = pd.read_csv(os.path.join(sim_path,"chr{}_params.csv".format(chrom)), index_col=0)
    df_params = df_params.astype(float)
    print(df_params.head)
    for param in param_filter:
        df_params = df_params[df_params[param]==param_filter[param]]
    print("{} set of parameters are being used.".format(len(df_params)))    

    # load LD
    ld_chrom = np.loadtxt(os.path.join(ld_path,"ukb_chr{}.qced.ld".format(chrom)))
    print("Chr {}, LD size: {}x{}".format(chrom, ld_chrom.shape[0],ld_chrom.shape[1]))

    X_list = list()
    y_list = list()
    for i_param in df_params.index:
        df_meta = pd.read_csv(os.path.join(sim_path,"chr{}_param{}_meta.csv".format(chrom,i_param)), index_col=0)
        for i_sim in range(nsim):
            snp_start = int(df_meta.iloc[i_sim]['snp_start'])
            # get LD for the segment
            ld_seg = ld_chrom[snp_start:(snp_start+nsnp),snp_start:(snp_start+nsnp)]
            # compute LD score (within the segment)
            ldsc = np.sum(ld_seg,axis=0)-1
            # get top r idx and val
            idx_max_ldsc = np.argsort(-ld_seg-np.identity(nsnp), axis=1)[:,1:(top_r+1)]
            top_r_val = ld_seg[idx_max_ldsc][np.arange(nsnp),:,np.arange(nsnp)]
            df_sim = pd.read_csv(os.path.join(sim_path,"chr{}_param{}_scaled_sim{}.csv".format(chrom,i_param,i_sim)), index_col=0)
            bhat = df_sim['beta_hat'].values
            shat = df_sim['se'].values
            top_r_beta = bhat[idx_max_ldsc]
            X = np.concatenate([bhat[:,None],shat[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
            y = df_sim['beta'].values
            X_list.append(X)
            y_list.append(y)
            
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"Chr{chrom} constructed data size:", X_all.shape, y_all.shape)

    output_prefix = "chr{}_topr{}_{}".format(chrom,top_r,param_str)
    np.savetxt(os.path.join(output_path,"{}.X".format(output_prefix)),X_all)
    np.savetxt(os.path.join(output_path,"{}.y".format(output_prefix)),y_all)
    print("Saving to:", os.path.join(output_path,output_prefix))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct model input using per-snp simulated data)')
    parser.add_argument('--top_r', type=int, help='Number of top variants to use')
    parser.add_argument('--ld_path', type=str, help='Path to ld files')
    parser.add_argument('--sim_path', type=str, help='Path to simulated data files')
    parser.add_argument('--output_path', type=str, help='Path to output data files')
    parser.add_argument('--chrom', type=int, help='Chromosome')
    # Optional arguments
    parser.add_argument('--param_filter', type=str, default="", help='Comma-separated list of parameter filters, e.g., s=0,w=0')
    parser.add_argument('--nsim', type=int, default=100, help='Number of simulations')
    parser.add_argument('--nsnp', type=int, default=1000, help='Number of SNPs in a simulation')

    args = parser.parse_args()
    main(args)