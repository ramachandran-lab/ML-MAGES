import os
import numpy as np
import pandas as pd
import argparse

from ._util_funcs import disp_params, parse_file_list, load_ld_files, str2bool


def construct_input_from_snp_only(top_r, ld_files, sim_path, res_prefix, output_file,
                                  p_list, h_list, s_list, w_list, n_sim, nsnp, is_transformed):
    
    # construct param str
    ld_files = parse_file_list(ld_files)
    # get dir of output_file and create if not exist:
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # load param list
    param_filters = {'p_causal': p_list, 'h2': h_list, 's': s_list, 'w': w_list}

    # Using simulation data that are separated by chrom and param
    # store training data
    X_all = []
    y_all = []

    # load param_meta
    out = "{}_sim_params.csv".format(res_prefix) if res_prefix != "" else "sim_params.csv"
    df_params = pd.read_csv(os.path.join(sim_path,out), index_col=0)
    df_params = df_params.astype(float)
    print("Total {} sets of parameters loaded.".format(len(df_params)))

    for param in param_filters:
        df_params = df_params[df_params[param].isin(param_filters[param])]
    print("{} set of parameters out of are being used.".format(len(df_params)))

    # load LD
    ld_all = load_ld_files(ld_files)
    print("All LD combined, size: {}x{}".format(ld_all.shape[0],ld_all.shape[1]))
    assert ld_all.shape[0]>=nsnp, "LD matrix size {} is smaller than nsnp {}".format(ld_all.shape[0],nsnp)

    X_list = list()
    y_list = list()
    for i_param in df_params.index:
        out = "{}_param{}_meta.csv".format(res_prefix,i_param) if res_prefix != "" else "param{}_meta.csv".format(i_param)
        df_meta = pd.read_csv(os.path.join(sim_path,out), index_col=0)
        for i_sim in range(n_sim):
            snp_start = int(df_meta.iloc[i_sim]['snp_start'])
            # get LD for the segment
            ld_seg = ld_all[snp_start:(snp_start+nsnp),snp_start:(snp_start+nsnp)]
            # compute LD score (within the segment)
            ldsc = np.sum(ld_seg**2,axis=0)-1
            # get top r idx and val
            idx_max_ldsc = np.argsort(-ld_seg-np.identity(nsnp), axis=1)[:,1:(top_r+1)]
            top_r_val = ld_seg[idx_max_ldsc][np.arange(nsnp),:,np.arange(nsnp)]
            # load sim data
            if is_transformed:
                out = "{}_param{}_scaled_sim{}.csv".format(res_prefix,i_param,i_sim) if res_prefix != "" else "param{}_scaled_sim{}.csv".format(i_param,i_sim)
            else:
                out = "{}_param{}_sim{}.csv".format(res_prefix,i_param,i_sim) if res_prefix != "" else "param{}_sim{}.csv".format(i_param,i_sim)
            df_sim = pd.read_csv(os.path.join(sim_path,out), index_col=0)
            bhat = df_sim['beta_hat'].values
            shat = df_sim['se'].values
            top_r_beta = bhat[idx_max_ldsc]
            X = np.concatenate([bhat[:,None],shat[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
            y = df_sim['beta'].values
            X_list.append(X)
            y_list.append(y)
            
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"Constructed data size:", X_all.shape, y_all.shape)

    np.savetxt("{}.X".format(output_file),X_all)
    np.savetxt("{}.y".format(output_file),y_all)
    print("Saving to:", output_file)


def main():
    print("RUNNING: construct_input_from_snp_only")

    parser = argparse.ArgumentParser(description='Construct model training input using snp-only simulated data)')
    parser.add_argument('--top_r', type=int, help='Number of top variants to use')
    parser.add_argument("--ld_files", nargs="+", default=[], help="List of LD files (in order), or a file containing one filename per line")
    
    parser.add_argument('--sim_path', type=str, help='Directory to saved simulated data')
    parser.add_argument('--res_prefix', type=str, default="", help="Simulation result prefix")
    parser.add_argument('--output_file', type=str, help='Path to output data file')

    parser.add_argument('--p_list', type=float, nargs='+', default=[0.01,0.05], help='List of p_causal values')
    parser.add_argument('--h_list', type=float, nargs='+', default=[0.3,0.7], help='List of h2 values')
    parser.add_argument('--s_list', type=float, nargs='+', default=[-0.25,0], help='List of s values')
    parser.add_argument('--w_list', type=float, nargs='+', default=[-1,0], help='List of w values')
    # parser.add_argument('--param_filter', type=str, default="", help='Comma-separated list of parameter filters, e.g., s=0,w=0')
    parser.add_argument('--n_sim', type=int, default=100, help='Number of simulations')
    parser.add_argument('--nsnp', type=int, default=1000, help='Number of SNPs in a simulation')
    parser.add_argument("--is_transformed", type=str2bool, nargs="?", default=True, help="Whether the data has been transformed to match real data")

    args = parser.parse_args()
    disp_params(args, title="INPUT SETTINGS")
    construct_input_from_snp_only(**vars(args))




if __name__ == "__main__":
    
    main()