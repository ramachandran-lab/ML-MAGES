import os
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import ElasticNetCV
from sklearn import preprocessing

import argparse

from ._util_funcs import disp_params, parse_file_list, load_gwas_file, parse_model_file


def shrink_by_enet(gwas_file, ld_files, output_file, chroms=None):

    ld_files = parse_file_list(ld_files)
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
    beta_enet = list()
    for i_ld,ld_file in enumerate(ld_files):
        ld = ld_list[i_ld]
        lb, ub = bound_list[i_ld]
        print("Shrinking SNPs {}-{} (LD file: {})".format(lb,ub,ld_file))

        X_input = preprocessing.normalize(ld)
        y_input = beta[lb:ub]
        regr = ElasticNetCV(cv = 5, l1_ratio=0.5, n_alphas=10, eps=1e-2, random_state=42, fit_intercept=False).fit(X_input,y_input)
        y_output = regr.coef_ 
        
        beta_enet.append(y_output)

    beta_enet = np.concatenate(beta_enet)
    toc = time.time()
    print("Time:",toc-tic)

    print("Saving results to", output_file)
    np.savetxt(output_file, beta_enet)


def main():
    print("RUNNING: shrink_by_enet")

    parser = argparse.ArgumentParser(description='Apply ENet to shrink GWA effects')
    # Required arguments
    parser.add_argument('--gwas_file', type=str, help="GWAS file")
    parser.add_argument("--ld_files", nargs="+", default=[], help="List of LD block files (in order), or a file containing one filename per line")
    parser.add_argument('--output_file', type=str, help='Output file name to save shrinkage results')
    # Optional arguments
    parser.add_argument('--chroms', type=int, nargs="*", default=None, help="Chromosome numbers (0–22). If not provided, all chromosomes will be used.")
    
    args = parser.parse_args()

    disp_params(args, title="INPUT SETTINGS")
    shrink_by_enet(**vars(args))


if __name__ == "__main__":
    
    main()