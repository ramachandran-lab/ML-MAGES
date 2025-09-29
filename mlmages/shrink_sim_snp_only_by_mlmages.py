import os
import numpy as np
import pandas as pd
import time
import torch
import argparse

from ._util_funcs import disp_params, parse_file_list, parse_model_file, load_ld_files
from ._train_funcs import get_n_feature, FCNN


def shrink_sim_snp_only_by_mlmages(model_files, ld_files, sim_path, sim_prefix, shrinkage_path, 
                      p_list, h_list, s_list, w_list, nsnp):

    os.makedirs(shrinkage_path, exist_ok=True)

    ld_files = parse_file_list(ld_files)
    # load param list
    param_filters = {'p_causal': p_list, 'h2': h_list, 's': s_list, 'w': w_list}

    model_files = parse_file_list(model_files)
    assert len(model_files)>0, "No model files found!"

    # load models
    n_models = len(model_files)
    model_file = model_files[0]
    top_r, n_layer = parse_model_file(model_file)
    model_label = "top{}_{}L".format(top_r,n_layer)
    print("Using top {} variants and {}-layer model".format(top_r,n_layer))
    print("Number of models in ensemble:", n_models)
    print("Model label:", model_label)

    n_feature = get_n_feature(top_r)
    NN_models = list()
    for model_idx in range(n_models):
        model = FCNN(n_feature, n_layer=n_layer, model_label="top{}_{}L".format(top_r,n_layer))
        model_path = os.path.join(model_files[model_idx])
        state = torch.load(model_path)
        model.load_state_dict(state)
        model.eval()
        NN_models.append(model)

    # load LD
    ld = load_ld_files(ld_files)
    print("All LD combined, size: {}x{}".format(ld.shape[0],ld.shape[1]))
    assert ld.shape[0]>=nsnp, "LD matrix size {} is smaller than nsnp {}".format(ld.shape[0],nsnp)

    # load simulation data
    # load param_meta
    out = "{}_sim_params.csv".format(sim_prefix) if sim_prefix != "" else "sim_params.csv"
    df_params = pd.read_csv(os.path.join(sim_path,out), index_col=0)
    df_params = df_params.astype(float)
    print("Total {} sets of parameters loaded.".format(len(df_params)))

    for param in param_filters:
        df_params = df_params[df_params[param].isin(param_filters[param])]
    print("{} set of parameters out of are being used.".format(len(df_params)))

    for i_param in df_params.index: 
        out = "{}_param{}_meta.csv".format(sim_prefix,i_param) if sim_prefix != "" else "param{}_meta.csv".format(i_param)
        df_param = pd.read_csv(os.path.join(sim_path,out), index_col=0)

        snp_starts = df_param['snp_start'].astype(int)
        # get simulated gwas for the segment (multiple params)
        ind_file_prefix = "{}_param{}_scaled_sim".format(sim_prefix,i_param) if sim_prefix != "" else "param{}_scaled_sim".format(i_param)
        # ind_files = [os.path.join(sim_path,"{}_scaled_sim{}.csv".format(ind_file_prefix, int(i_sim))) for i_sim in n_sim]
        print("#simulated seg:",len(snp_starts))
        tot_times = list()
        y_true = list()
        y_pred = list()
        for i_seg in range(len(snp_starts)):
            ptm = time.time()
            snp_start = snp_starts[i_seg]
            # get LD for the segment
            ld_seg = ld[snp_start:(snp_start+nsnp),snp_start:(snp_start+nsnp)]
            # compute LD score (within the segment)
            ldsc = np.sum(ld_seg**2,axis=0)-1
            # get top r idx and val
            idx_max_ldsc = np.argsort(-ld_seg-np.identity(nsnp), axis=1)[:,1:(top_r+1)]
            top_r_val = ld_seg[idx_max_ldsc][np.arange(nsnp),:,np.arange(nsnp)]

            ind_file = os.path.join(sim_path,"{}{}.csv".format(ind_file_prefix, int(i_seg)))
            df_sim = pd.read_csv(ind_file, index_col=0)
            bhat = df_sim['beta_hat'].values
            shat = df_sim['se'].values
            top_r_beta = bhat[idx_max_ldsc]

            X = np.concatenate([bhat[:,None],shat[:,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
            y = df_sim['beta'].values
            y_true.append(y)

            y_obs = df_sim['beta_hat'].values
            y_pred_ensemble = list()
            for model in NN_models:
                with torch.no_grad():
                    y_pred_seg = model(torch.tensor(X, dtype=torch.float32))
                    y_pred_seg = y_pred_seg.numpy().squeeze()
                y_pred_ensemble.append(y_pred_seg)
            y_pred_ensemble = np.stack(y_pred_ensemble).mean(axis=0)
            y_pred.append(y_pred_ensemble)
        y_true = np.vstack(y_true).T
        y_pred = np.vstack(y_pred).T
        assert y_true.shape[1]==len(snp_starts)
        assert y_true.shape[0]==nsnp
        print("True values shape:", y_true.shape)
        print("Predicted values shape:", y_pred.shape)

        y_true_file = os.path.join(shrinkage_path,"true_{}_param{}.txt".format(sim_prefix,i_param))
        y_pred_file = os.path.join(shrinkage_path,"shrinkage_mlmages_{}_{}_param{}.txt".format(model_label,sim_prefix,i_param))
        np.savetxt(y_true_file,y_true)
        np.savetxt(y_pred_file,y_pred)

        tot_time = time.time() - ptm
        tot_times.append(tot_time)

        np.savetxt(os.path.join(shrinkage_path,"time_mlmages_{}_{}.txt".format(model_label,ind_file_prefix)),tot_times)
        print("Saving times to:",os.path.join(shrinkage_path,"time_mlmages_{}_{}.txt".format(model_label,ind_file_prefix)))


def main():
    parser = argparse.ArgumentParser(description="Shrink simulation (SNP-only) using ML-MAGES.")
    parser.add_argument('--model_files', nargs="+", default=[], help="List of model files (in order), or a file containing one filename per line")
    parser.add_argument("--ld_files", nargs="+", default=[], help="List of LD files (in order), or a file containing one filename per line")
    parser.add_argument('--sim_path', type=str, help='Directory to saved simulated data')
    parser.add_argument('--sim_prefix', type=str, default="", help="Simulation result prefix")
    parser.add_argument('--shrinkage_path', type=str, help='Path to shrinkage output')

    parser.add_argument('--p_list', type=float, nargs='+', default=[0.01,0.05], help='List of p_causal values')
    parser.add_argument('--h_list', type=float, nargs='+', default=[0.3,0.7], help='List of h2 values')
    parser.add_argument('--s_list', type=float, nargs='+', default=[-0.25,0], help='List of s values')
    parser.add_argument('--w_list', type=float, nargs='+', default=[-1,0], help='List of w values')
    parser.add_argument('--nsnp', type=int, default=1000, help='Number of sampled SNPs')

    args = parser.parse_args()

    disp_params(args, title="INPUT SETTINGS")
    shrink_sim_snp_only_by_mlmages(**vars(args))


if __name__ == "__main__":
    
    main()