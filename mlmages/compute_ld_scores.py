import pandas as pd
import numpy as np
import os
import argparse

from ._util_funcs import disp_params, parse_file_list


def compute_ld_scores(ld_files, ldscore_file: str):
    # actual processing code here
    ld_files = parse_file_list(ld_files)

    # create path for ldscore_file if not exist
    if not os.path.isdir(os.path.dirname(ldscore_file)) and os.path.dirname(ldscore_file) != '':
        os.makedirs(os.path.dirname(ldscore_file))
        print("Created output directory to ldscore file:", os.path.dirname(ldscore_file))
    print("--------------------------------------------------")
    
    score = list()
    for ld_file in ld_files:
        assert os.path.isfile(ld_file), "LD file {} not found!".format(ld_file)
        ld = np.loadtxt(ld_file, dtype=float)
        full_score = np.sum(ld**2,axis=0)-1
        score.append(full_score)
    score = np.concatenate(score)

    np.savetxt(ldscore_file, score, fmt='%f')


def main():
    print("RUNNING: compute_ld_scores")

    parser = argparse.ArgumentParser(description="Compute LD scores")
    parser.add_argument("--ld_files", nargs="+", default=[], help="List of LD block files (in order), or a file containing one filename per line")
    parser.add_argument("--ldscore_file", type=str, help="Output LD score file")
    args = parser.parse_args()

    disp_params(args, title="INPUT SETTINGS")
    compute_ld_scores(**vars(args))


if __name__ == "__main__":
    main()
