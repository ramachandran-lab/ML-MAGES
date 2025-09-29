import os
import numpy as np
import pandas as pd
import time
import torch
import argparse

from ._util_funcs import disp_params, parse_file_list, parse_model_file, load_ld_files
from ._train_funcs import get_n_feature, FCNN
from ._sim_funcs import construct_features_from_sim_by_gene

def shrink_sim_gene_level_by_mlmages(model_files, ld_files, sim_path, sim_prefix, i_sim, shrinkage_path):

    os.makedirs(shrinkage_path, exist_ok=True)

    ld_files = parse_file_list(ld_files)

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

    sim_data_file = os.path.join(sim_path,"{}_data_scaled_sim{}.txt".format(sim_prefix,i_sim))

    sim_data = np.loadtxt(sim_data_file, delimiter=',')
    n_trait = sim_data.shape[1]//3
    
    tic = time.time()
    breg_all = list()
    btrue_all = list()
    lb = 0
    ub = None
    for idx, ld_file in enumerate(ld_files):

        ld = np.loadtxt(ld_file, delimiter=" ", dtype=float)
        print("LD file {}, size: {}x{}".format(ld_file,ld.shape[0],ld.shape[1]))
        ub = lb + ld.shape[0]
        assert ub<=sim_data.shape[0], "LD matrix size {} is larger than sim_data {}".format(ub,sim_data.shape[0])

        sim_data_block = sim_data[lb:ub,]
        
        X_list, y_list = construct_features_from_sim_by_gene(sim_data_block, ld, top_r)
        
        y_output = list()
        y_true = list()
        for ii in range(len(X_list)):
            X = X_list[ii]
            y = y_list[ii]
            # scale
            y_pred_ensemble = list()
            for model in NN_models:
                with torch.no_grad():
                    y_pred = model(torch.tensor(X, dtype=torch.float32))
                    y_pred = y_pred.numpy().squeeze()
                y_pred_ensemble.append(y_pred)
            y_pred_ensemble = np.stack(y_pred_ensemble).mean(axis=0)
            y_output.append(y_pred_ensemble)
            y_true.append(y)
        y_output = np.vstack(y_output).T  
        y_true = np.vstack(y_true).T
        breg_all.append(y_output)
        btrue_all.append(y_true)
    breg_all = np.vstack(breg_all)
    btrue_all = np.vstack(btrue_all)
    print(breg_all.shape, btrue_all.shape)
    toc = time.time()
    print("time:", toc-tic)

    
    out_file = os.path.join(shrinkage_path,"shrinkage_mlmages_{}_{}_sim{}.txt".format(model_label,sim_prefix,i_sim))
    np.savetxt(out_file,breg_all)
    out_file = os.path.join(shrinkage_path,"true_{}_sim{}.txt".format(sim_prefix,i_sim))
    np.savetxt(out_file,btrue_all)

    for i_trait in range(n_trait):
        np.savetxt(os.path.join(shrinkage_path,"shrinkage_mlmages_{}_{}_sim{}_trait{}.txt".format(model_label, sim_prefix, i_sim, i_trait)),breg_all[:,i_trait])
        np.savetxt(os.path.join(shrinkage_path,"true_{}_sim{}_trait{}.txt".format(sim_prefix,i_sim,i_trait)),btrue_all[:,i_trait])

    # save time
    np.savetxt(os.path.join(shrinkage_path,"time_mlmages_{}_{}_sim{}.txt".format(model_label, sim_prefix, i_sim)),np.array([toc-tic]))
    print("Saving times to:",os.path.join(shrinkage_path,"time_mlmages_{}_{}_sim{}.txt".format(model_label, sim_prefix, i_sim)))


def main():
    parser = argparse.ArgumentParser(description="Shrink simulation (gene-level) using ML-MAGES.")
    parser.add_argument('--model_files', nargs="+", default=[], help="List of model files (in order), or a file containing one filename per line")
    parser.add_argument("--ld_files", nargs="+", default=[], help="List of LD files (in order), or a file containing one filename per line")
    parser.add_argument('--sim_path', type=str, help='Directory to saved simulated data')
    parser.add_argument('--sim_prefix', type=str, default="", help="Simulation result prefix")
    parser.add_argument('--i_sim', type=int, default=0, help="Simulation index")
    parser.add_argument('--shrinkage_path', type=str, help='Path to shrinkage output')

    args = parser.parse_args()

    disp_params(args, title="INPUT SETTINGS")
    shrink_sim_gene_level_by_mlmages(**vars(args))


if __name__ == "__main__":
    
    main()