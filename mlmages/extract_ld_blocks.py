import pandas as pd
import numpy as np
import os
import argparse

from ._process_funcs import construct_antidiagonal_sum_vec, apply_filter, local_search
from ._util_funcs import disp_params


def extract_ld_blocks(ld_file, ldblock_path, block_meta_file, avg_block_size, res_prefix):
    # actual processing code here
    # print(f"Processing {ld_file} into {ldblock_path}/{res_prefix}")
    os.makedirs(ldblock_path, exist_ok=True)

    assert os.path.isfile(ld_file), "LD file {} not found!".format(ld_file)
    ld = np.loadtxt(ld_file)

    J = ld.shape[0]
    v = construct_antidiagonal_sum_vec(ld)
    n_bpoints = int(np.floor(J/avg_block_size))+1
    print("Total number of break points:", n_bpoints)

    v_filtered, minima_ind, _, filter_width = apply_filter(v, width=200, n_pt = n_bpoints, min_width = avg_block_size)
    new_brkpts, new_block_sizes = local_search(ld, minima_ind)

    # get all blocks' lower and upper bound
    brkpts = np.insert(new_brkpts,0,0)
    brkpts = np.insert(brkpts,len(brkpts),J)
    block_lbs = brkpts[:-1]
    block_ubs = brkpts[1:]
    block_sizes = block_ubs-block_lbs
    df_brkpts = pd.DataFrame([np.arange(len(block_lbs)),block_lbs,block_ubs,block_sizes]).T
    df_brkpts.columns = ['idx_in_chr','block_lb','block_ub','block_size']

    # make directories to save meta file
    if not os.path.isdir(os.path.dirname(block_meta_file)) and os.path.dirname(block_meta_file) != '':
        os.makedirs(os.path.dirname(block_meta_file))
        print("Created output directory to meta file:", os.path.dirname(block_meta_file))
    # save breakpoints
    df_brkpts.to_csv(block_meta_file, index=False)

    # save each ld block
    for i_bk in range(len(df_brkpts)):
        lb, ub = df_brkpts.iloc[i_bk]['block_lb'], df_brkpts.iloc[i_bk]['block_ub']
        block_ld = ld[lb:ub,lb:ub]
        out = "{}_block{}.ld".format(res_prefix,i_bk) if res_prefix != "" else "block{}.ld".format(i_bk)
        print("Saving LD block {} of size {}x{} to {}".format(i_bk, block_ld.shape[0], block_ld.shape[1], os.path.join(ldblock_path,out)))
        np.savetxt(os.path.join(ldblock_path,out), block_ld, delimiter=" ",fmt='%f')


def main():
    print("RUNNING: extract_ld_blocks")

    parser = argparse.ArgumentParser(description="Extract LD blocks from a full LD.")
    # required args
    parser.add_argument('--ld_file', type=str, help="Full LD file")
    parser.add_argument('--ldblock_path', type=str, help="Path to save LD block files")
    parser.add_argument('--block_meta_file', type=str, help="Meta file of block LDs")
    # optional args
    parser.add_argument('--avg_block_size', type=int, default=1000, help="Average number of SNPs per block")
    parser.add_argument('--res_prefix', type=str, default="", help="Result prefix")

    args = parser.parse_args()
    disp_params(args, title="INPUT SETTINGS")
    extract_ld_blocks(**vars(args))
    

if __name__ == "__main__":

    main()