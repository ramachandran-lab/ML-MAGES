import os
import numpy as np
import pandas as pd
import argparse

from ._gene_funcs import test_gene
from ._util_funcs import disp_params, parse_file_list

def univar_enrich(output_file, gene_file, shrinkage_file, clustering_file_prefix, ld_files, chroms=None):
    # load shrinkage (effects)
    breg = np.loadtxt(shrinkage_file, delimiter=",")
    print("Effect sizes loaded, #SNPs:", len(breg))

    # load clustering
    Sigma = np.loadtxt("{}_Sigma.txt".format(clustering_file_prefix), delimiter=',') 
    pi = np.loadtxt("{}_pi.txt".format(clustering_file_prefix), delimiter=',') 
    pred_K = len(pi)
    assert Sigma.shape[0]==pred_K, "Number of clusters does not match between Sigma and pi!"
    eps_eff_cls = 1 if pi[0]>0.01 else 2
    epsilon_effect = Sigma[eps_eff_cls]
    print("eps Cls: {}, eps={}".format(eps_eff_cls,epsilon_effect))

    # load gene data
    genes_chr_all = pd.read_csv(gene_file)
    required_cols = ['CHR','GENE','N_SNPS','start_idx_chr','chr_start_idx_gw']
    assert all(col in genes_chr_all.columns for col in required_cols), "Gene file is missing some required columns: {}!".format([col for col in required_cols if col not in genes_chr_all.columns])
    print("Gene file loaded, #genes:", len(genes_chr_all))

    # load ld(s)
    chr_start_idx = 0
    chr_end_idx = None
    ld_files = parse_file_list(ld_files)

    df_test_all = list()
    for i,ld_file in enumerate(ld_files):
        assert os.path.isfile(ld_file), "LD file {} not found!".format(ld_file)
        ld = np.loadtxt(ld_file)
        chr_end_idx = chr_start_idx + ld.shape[0]
        chrom = chroms[i]
        genes_selected = genes_chr_all[(genes_chr_all['CHR']==chrom)].reset_index(drop=True)
        print("{} genes in (chr {}) LD file {}".format(len(genes_selected), chrom, ld_file))

        # enrichment analysis
        enrich_test_results = test_gene(genes_selected['start_idx_chr'].values.astype(int), 
                                        genes_selected['N_SNPS'].values.astype(int), 
                                        epsilon_effect, breg, ld) 
        df_test = pd.DataFrame(enrich_test_results)
        df_test['GENE'] = genes_selected['GENE']
        df_test['CHR'] = genes_selected['CHR']
        df_test = df_test[['GENE','CHR'] + list(df_test.columns[:-2])]

        df_test_all.append(df_test)
        chr_start_idx = chr_end_idx

    print("#LD files:", len(ld_files))
    assert chr_end_idx==len(breg), "Number of SNPs in LD files ({}) does not match that in the shrinkage file ({})".format(chr_end_idx,len(breg))

    # save
    df_test_all = pd.concat(df_test_all, axis=0)
    df_test_all.to_csv(output_file, index=False)
    print("Saving results to", output_file)


def main():
    parser = argparse.ArgumentParser(description='Enrichment test for single trait')
    parser.add_argument('--output_file', type=str, help='Output file name to save test results')
    parser.add_argument('--gene_file', type=str, help="Gene file")
    parser.add_argument('--shrinkage_file', type=str, help="Trait's regularized effects file")
    parser.add_argument('--clustering_file_prefix', type=str, help='Clustering file prefix')
    parser.add_argument('--ld_files', nargs="+", default=[], help='List of LD files (in order) of a trait, or a file containing one filename per line. Each file contains an LD matrix.')
    parser.add_argument('--chroms', type=int, nargs="*", default=None, help="Chromosome numbers (0–22). If not provided, all chromosomes will be used.")
    args = parser.parse_args()

    disp_params(args, title="INPUT SETTINGS")
    if os.path.isfile(args.output_file):
        print("Warning: output_file already exists and will be overwritten!")
    else:
        # create directory if not exist
        if not os.path.isdir(os.path.dirname(args.output_file)) and os.path.dirname(args.output_file) != '':
            os.makedirs(os.path.dirname(args.output_file))
            print("Created output directory:", os.path.dirname(args.output_file))
    if args.chroms is None or len(args.chroms)==0:
        chroms = list(range(1,23))
    else:
        chroms = args.chroms
    print("Chromosomes to be used:", chroms)
    print("--------------------------------------------------")

    disp_params(args, title="INPUT SETTINGS")
    univar_enrich(**vars(args))


if __name__ == "__main__":
    
    main()
