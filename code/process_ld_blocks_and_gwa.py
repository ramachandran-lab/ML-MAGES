import os
import numpy as np
import pandas as pd
import argparse
import time

def main(args):
    # take in command line arguments
    print("-----Required Arguments: ")
    chrs = args.chrs.split(",")
    print("chrs:", [int(c) for c in chrs])
    full_ld_files = args.full_ld_files.split(",")
    print("full_ld_files:", full_ld_files)
    brkpts_files = args.brkpts_files.split(",")
    print("brkpts_files:", brkpts_files)
    
    ld_block_meta_file = args.ld_block_meta_file
    print("ld_block_meta_file:", ld_block_meta_file)
    ld_block_brkpts_file = args.ld_block_brkpts_file
    print("ld_block_brkpts_file:", ld_block_brkpts_file)
    ld_block_path = args.ld_block_path
    print("ld_block_path:", ld_block_path)
    if not os.path.exists(ld_block_path):
        os.makedirs(ld_block_path)
        
    gwa_files = [f.split(",") for f in args.gwa_files.split(";")]
    print("gwa_files:", gwa_files)
    processed_gwa_files = args.processed_gwa_files.split(";")
    print("processed_gwa_files:", processed_gwa_files)

    print("-----Optional Arguments: ")
    chr_col = args.chr_col
    print("chr_col (default: CHR):", chr_col)
    pos_col = args.pos_col
    print("pos_col (default: POS):", pos_col)
    id_col = args.id_col
    print("id_col (default: ID):", id_col)
    beta_col = args.beta_col
    print("beta_col (default: BETA):", beta_col)
    se_col = args.se_col
    print("se_col (default: SE):", se_col)

    print("=====//ML-MAGES// Helper Function: Process LD Blocks and GWA=====")
    start_time = time.time()

    n_files = len(full_ld_files)
    process_gwa = len(processed_gwa_files)>0
    if process_gwa:
        df_gwa_all = list()
        for i_trait in range(len(gwa_files)):
            df_gwa_all.append(list())
    chr_brkpts = list()
    chr_sizes = list()
    meta_info = list()
    ld_cnt = 0
    for i_file in range(n_files):
        chr = chrs[i_file]
        # load LD
        chr_ld = np.loadtxt(full_ld_files[i_file])
        print("LD size: {}x{}".format(chr_ld.shape[0],chr_ld.shape[1]))
        J = chr_ld.shape[0]
        
        # load breakpoints
        brkpts = np.loadtxt(brkpts_files[i_file], delimiter=',').astype(int) 
        print("#brkpts:",len(brkpts))
        brkpts = np.insert(brkpts,0,0)
        brkpts = np.insert(brkpts,len(brkpts),J)
        
        # load gwa
        if process_gwa:
            for i_trait in range(len(gwa_files)):
                file = gwa_files[i_trait][i_file]
                df_gwa_chr = pd.read_csv(file)
                if len(df_gwa_chr.columns)==1: # tab-delimited
                    df_gwa_chr = pd.read_csv(file, sep="\s+")
                assert len(df_gwa_chr)==J, "Size of GWA results does not match that of LD."
                df_gwa_chr["index_chr"] = np.arange(len(df_gwa_chr))
                if id_col in df_gwa_chr.columns:
                    df = df_gwa_chr[["index_chr",chr_col,pos_col,id_col,beta_col,se_col]]
                    df.columns = ["index_chr","CHR","POS","ID","BETA","SE"]
                else:
                    df = df_gwa_chr[["index_chr",chr_col,pos_col,beta_col,se_col]]
                    df.columns = ["index_chr","CHR","POS","BETA","SE"]
                # df.iloc[np.arange(len(df)),"index_chr"] = np.arange(len(df))
                df_gwa_all[i_trait].append(df)
                    
        # save ld
        for i_bk in range(len(brkpts)-1):
            # each block
            lb, ub = brkpts[i_bk], brkpts[i_bk+1]
            block_ld = chr_ld[lb:ub,lb:ub]
            np.savetxt(os.path.join(ld_block_path,"block_{}.ld".format(ld_cnt)), 
                       block_ld, delimiter=",")
            meta_info.append((ld_cnt,chr,i_bk))
            ld_cnt += 1
    
        chr_sizes.append(J)
        chr_brkpts.append(brkpts[1:])

    # save block IDs
    chr_sizes = np.array(chr_sizes)
    chr_cumsize = np.cumsum(chr_sizes)
    chr_brkpts_all = list()
    for i_file in range(n_files):
        brkpts = chr_brkpts[i_file]
        if i_file>=1:
            brkpts += chr_cumsize[i_file-1]
        chr_brkpts_all.append(brkpts)
    chr_brkpts_all = np.concatenate(chr_brkpts_all)
    np.savetxt(ld_block_brkpts_file, chr_brkpts_all[:,None], fmt="%d")
    
    # save meta
    meta_info = np.array(meta_info)
    df_meta = pd.DataFrame(meta_info, columns=["block_id","chr","id_in_chr"])
    df_meta["brkpt"] = chr_brkpts_all
    df_meta.to_csv(ld_block_meta_file, index=False)

    if process_gwa:
        for i_trait in range(len(gwa_files)):
            df_gwa_all[i_trait] = pd.concat(df_gwa_all[i_trait])
            if id_col=="":
                df_gwa_all[i_trait]["ID"] = ['var_{}'.format(i) for i in np.arange(len(df_gwa_all[i_trait]))]
            df = df_gwa_all[i_trait]
            df.to_csv(processed_gwa_files[i_trait], index=False)

    end_time = time.time()
    print("Processing split LD and GWA takes {:.2f} seconds".format(end_time - start_time))

    print("============DONE============")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide the required arguments')

    # Required argument
    parser.add_argument('--chrs', type=str, required=True, help='comma-separated chromosomes')
    parser.add_argument('--full_ld_files', type=str, required=True, help='path to the full LD files (with multiple file names separated by comma)')
    parser.add_argument('--brkpts_files', type=str, required=True, help='path to the breakpoints files of LD splitting (with multiple file names separated by comma)')
    
    parser.add_argument('--ld_block_meta_file', type=str, required=True, help='path to the file with meta info of LD splitting')
    parser.add_argument('--ld_block_brkpts_file', type=str, required=True, help='path to the file with all breakpoints (right boundaries only)')
    parser.add_argument('--ld_block_path', type=str, required=True, help='path to store the split LD blocks')

    parser.add_argument('--gwa_files', type=str, required=True, help='path to the GWA files to be processed (with multiple traits separated by semicolon and multiple chromosomes of the same trait separated by comma; ex. of nested structure: "trait1-chr1,trait1-chr2;trait2-chr1,trait2-chr2")')
    parser.add_argument('--processed_gwa_files', type=str, required=True, help='path to the processed GWA files of all traits (with multiple traits separated by semicolon)')

    # Optional arguments
    parser.add_argument('--chr_col', type=str, required=False, default='CHR', help='column name of chromosome in GWA file')
    parser.add_argument('--pos_col', type=str, required=False, default='POS', help='column name of variant position in GWA file')
    parser.add_argument('--id_col', type=str, required=False, default='ID', help='column name of variant ID in GWA file')
    parser.add_argument('--beta_col', type=str, required=False, default='BETA', help='column name of estimated effect in GWA file')
    parser.add_argument('--se_col', type=str, required=False, default='SE', help='column name of standard error in GWA file')


    args = parser.parse_args()
    main(args)