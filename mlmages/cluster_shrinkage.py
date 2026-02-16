import os
import numpy as np
import time
import argparse

from ._cls_funcs import *
from ._util_funcs import disp_params, parse_file_list


def cluster_shrinkage(shrinkage_files, output_file, K=20, n_runs=25, max_nz=0):

    print("Running infinite mixture with K={} and {} runs.".format(K, n_runs))

    # load shrinkage data
    n_trait = len(shrinkage_files)
    print("Number of traits:", n_trait)
    breg_list = []
    for trait_file_name, files in shrinkage_files.items():
        breg_trait = list()
        for file in files:
            breg = np.loadtxt(file, delimiter=",")
            breg_trait.append(breg)
        breg_list.append(np.concatenate(breg_trait))
    beta_reg = np.vstack(breg_list).T
    print("Shape of effect sizes loaded:", beta_reg.shape)
    
    # perform clustering
    if max_nz==0:
        max_nz = beta_reg.shape[0]//5 if beta_reg.shape[0]>15000 else beta_reg.shape[0]//3
    elif max_nz<0:
        max_nz = beta_reg.shape[0]
    else:
        max_nz = max_nz
        assert max_nz<beta_reg.shape[0], "max_nz should be less than the number of SNPs!"
    if beta_reg.shape[0]>50000:
        cutoff = 1e-3
    elif beta_reg.shape[0]>10000:
        cutoff = 1e-4
    else:
        cutoff = 1e-5
    betas_nz, zero_cutoff = adjust_zero_threshold(beta_reg, init_zero_cutoff=cutoff, any_zero=True, 
                                                  min_nz=1000, max_nz=max_nz, adjust_scale=2, max_iter=10)
    print("shape change: {} -> {}".format(beta_reg.shape,betas_nz.shape))
    print("zero cutoff: {}".format(zero_cutoff))
    print("#nz for each trait: {}".format(np.sum(betas_nz!=0, axis=0)))
    
    tic = time.time()
    truncate_Sigma, truncate_pi, pred_K, pred_cls = extract_multivariate_components(betas_nz, K=K, n_runs=n_runs)
    toc = time.time()
    
    print("K={} (time: {:.2f}s)".format(pred_K,toc-tic))
    
    cls_labels = np.ones(beta_reg.shape[0])*(-1)
    cls_labels[np.any(np.abs(beta_reg)>zero_cutoff, axis=1)] = pred_cls
    cls_labels = cls_labels.astype(int)
    
    # save results
    # save Sigma and pi
    np.savetxt(f"{output_file}_Sigma.txt", truncate_Sigma.reshape(truncate_Sigma.shape[0],-1), delimiter=',') 
    np.savetxt(f"{output_file}_pi.txt", truncate_pi.squeeze(), delimiter=',') 
    # save cluster labels
    np.savetxt(f"{output_file}_cls.txt", cls_labels, delimiter=',')
    # save meta (nz_size, zero_cutoff, time)
    np.savetxt(f"{output_file}_meta.txt", np.array([zero_cutoff,betas_nz.shape[0],toc-tic]), delimiter=',')
    print(f"Saving results to {output_file}_x.txt")


def main():

    print("RUNNING: cluster_shrinkage")

    parser = argparse.ArgumentParser(description='Use infinite mixture to cluster effects')
    parser.add_argument('--output_file', type=str, help='Output file name (prefix) to save clustering results')
    # # Optional arguments
    parser.add_argument('--K', type=int, default=20, help='Default K in the infinite mixture')
    parser.add_argument('--n_runs', type=int, default=25, help='Number of runs of the infinite mixture')
    parser.add_argument('--max_nz', type=int, default=0, help='Maximum number of non-zero effects allowed (default: n/5 if n>15000 else n/3); if negative, allowing all SNPs')
    
    
    # Handle dynamic trait file arguments
    args, unknown = parser.parse_known_args()
    trait_files = {}
    key = None
    for token in unknown:
        if token.startswith("--shrinkage_trait"):
            key = token.lstrip("-")
            trait_files[key] = []
        else:
            if key is None:
                raise ValueError(f"Unexpected token {token} without a preceding --shrinkage_traitX_files flag")
            trait_files[key].append(token)
    # Expand possible list-files
    for trait, files in trait_files.items():
        trait_files[trait] = parse_file_list(files)
    args.shrinkage_files = trait_files

    # settings
    disp_params(args, title="INPUT SETTINGS")
    if os.path.isfile(args.output_file):
        print("Warning: output_file already exists and will be overwritten!")
    else:
        # create directory if not exist
        if not os.path.isdir(os.path.dirname(args.output_file)) and os.path.dirname(args.output_file) != '':
            os.makedirs(os.path.dirname(args.output_file))
            print("Created output directory:", os.path.dirname(args.output_file))
    print("--------------------------------------------------")

    cluster_shrinkage(**vars(args))


if __name__ == "__main__":
    
    main()

