import os
import numpy as np
import pandas as pd
import time
import argparse
import torch

from ._train_funcs import *
from ._util_funcs import disp_params, parse_file_list, load_gwas_file, parse_model_file


def shrink_by_mlmages(gwas_file, ld_files, model_files, output_file, chroms=None):

    ld_files = parse_file_list(ld_files)
    model_files = parse_file_list(model_files)
    assert len(model_files)>0, "No model files found!"
    if chroms is None or len(chroms)==0:
        chroms = list(range(1,23))
    else:
        chroms = chroms
    print("Chromosomes to be used:", chroms)
    if os.path.isfile(output_file):
        print("Warning: output_file already exists and will be overwritten!")
    if not os.path.isdir(os.path.dirname(output_file)) and os.path.dirname(output_file) != '':
        os.makedirs(os.path.dirname(output_file))
        print("Created output directory:", os.path.dirname(output_file))
    print("--------------------------------------------------")

    n_models = len(model_files)
    model_file = model_files[0]
    top_r, n_layer = parse_model_file(model_file)
    model_label = "top{}_{}L".format(top_r,n_layer)
    print("Using top {} variants and {}-layer model".format(top_r,n_layer))
    print("Number of models in ensemble:", n_models)
    print("Model label:", model_label)

    # load models
    n_feature = get_n_feature(top_r)
    NN_models = list()
    for model_idx in range(n_models):
        model = FCNN(n_feature, n_layer=n_layer, model_label="top{}_{}L".format(top_r,n_layer))
        model_path = os.path.join(model_files[model_idx])
        state = torch.load(model_path)
        model.load_state_dict(state)
        model.eval()
        NN_models.append(model)
    
    # load GWAS results
    print("Loading GWAS results from", gwas_file)
    _, beta, se = load_gwas_file(gwas_file, chroms=chroms)
   
    print("Number of SNPs in the GWAS file:", len(beta))
    
    # load LD files
    bound_list = list()
    ld_list = list()
    lb, ub = 0, None
    for ld_file in ld_files:
        assert os.path.isfile(ld_file), "LD file {} not found!".format(ld_file)
        df = pd.read_csv(ld_file, sep=r"[\s,]+", engine="python", header=None)
        ld = df.values
        ub = lb+ld.shape[0]
        bound_list.append((lb, ub))
        ld_list.append(ld)
        lb = ub
    assert ub==len(beta), "Number of SNPs in LD blocks ({}) does not match that in the GWAS file ({})".format(ub,len(beta))

    # apply ML-MAGES effect size shrinkage
    tic = time.time()
    beta_mlmages = list()
    for i_ld,ld_file in enumerate(ld_files):
        ld = ld_list[i_ld]
        lb, ub = bound_list[i_ld]
        print("Processing SNPs {}-{} (LD file: {})".format(lb,ub,ld_file))

        # compute LD score and get top r (within the block)
        idx_max_ldsc, top_r_val, ldsc = get_topr_idx_from_ld(ld, top_r)
        top_r_beta = beta[idx_max_ldsc]
        # construct features
        X = np.concatenate([beta[lb:ub,None],se[lb:ub,None],ldsc[:,None],top_r_beta,top_r_val], axis=1)
        # predict using ensemble of models
        y_pred_ensemble = list()
        for model in NN_models:
            with torch.no_grad():
                y_pred = model(torch.tensor(X, dtype=torch.float32))
                y_pred = y_pred.numpy().squeeze()
            y_pred_ensemble.append(y_pred)
        y_pred_ensemble = np.stack(y_pred_ensemble).mean(axis=0)
        beta_mlmages.append(y_pred_ensemble)

    beta_mlmages = np.concatenate(beta_mlmages)
    toc = time.time()
    print("Time:",toc-tic)

    print("Saving results to", output_file)
    np.savetxt(output_file, beta_mlmages)



def main():
    print("RUNNING: shrink_by_mlmages")

    parser = argparse.ArgumentParser(description='Apply trained ML-MAGES models to shrink GWA effects')
    # Required arguments
    parser.add_argument('--gwas_file', type=str, help="GWAS file")
    parser.add_argument("--ld_files", nargs="+", default=[], help="List of LD block files (in order), or a file containing one filename per line")
    parser.add_argument('--model_files', nargs="+", default=[], help="List of model files (in order), or a file containing one filename per line")
    parser.add_argument('--output_file', type=str, help='Output file name to save shrinkage results')
    # Optional arguments
    parser.add_argument('--chroms', type=int, nargs="*", default=None, help="Chromosome numbers (0–22). If not provided, all chromosomes will be used.")
    
    args = parser.parse_args()
    args._description = parser.description

    disp_params(args, title="INPUT SETTINGS")
    shrink_by_mlmages(**vars(args))
    # shrink_by_mlmages(args.gwas_file, args.ld_files, args.model_files, args.output_file, chroms=args.chroms)
    


if __name__ == "__main__":
    
    main()