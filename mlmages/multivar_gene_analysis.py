import os
import numpy as np
import pandas as pd
import argparse

from ._gene_funcs import test_gene, check_specific_or_shared_bivar, check_specific_trivar
from ._util_funcs import disp_params, parse_file_list, get_eiginfo


def multivar_gene_analysis(output_file, gene_file, clustering_file_prefix, shrinkage_files, chroms=None, prioritized_cls_prop=1):
    # load shrinkage (effects)
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

    # load clustering
    Sigma = np.loadtxt("{}_Sigma.txt".format(clustering_file_prefix), delimiter=',') 
    n_traits = int(np.sqrt(Sigma[0].shape[0]))
    print("n_traits:", n_traits)
    Sigma = [s.reshape(n_traits,n_traits) for s in Sigma]
    pi = np.loadtxt("{}_pi.txt".format(clustering_file_prefix), delimiter=',') 
    pred_K = len(pi)
    assert len(Sigma) == pred_K, "Number of clusters does not match between Sigma and pi!"
    cls_lbs = np.loadtxt("{}_cls.txt".format(clustering_file_prefix), delimiter=',').astype(int)
    meta = np.loadtxt("{}_meta.txt".format(clustering_file_prefix), delimiter=',')
    zero_cutoff = float(meta[0])
    # betas = np.stack([beta_reg_1,beta_reg_2]).T
    is_nz = cls_lbs>=0
    betas_regularized = beta_reg[is_nz,:] # any nz
    pred_cls = cls_lbs[is_nz].astype(int)
    cls_cnt = pd.Series(pred_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
    cls_perc = (cls_cnt/cls_cnt.sum()).loc[np.arange(pred_K)]    
    # print("percentage of nz variants in each cluster:\n", cls_perc)

    # load gene data
    genes_chr_all = pd.read_csv(gene_file)
    required_cols = ['CHR','GENE','N_SNPS','start_idx_chr','end_idx_chr','chr_start_idx_gw']
    assert all(col in genes_chr_all.columns for col in required_cols), "Gene file is missing some required columns: {}!".format([col for col in required_cols if col not in genes_chr_all.columns])
    genes_selected = genes_chr_all[(genes_chr_all['CHR'].isin(chroms))].reset_index(drop=True)
    print("{} genes in chrs {}".format(len(genes_selected), chroms))
    n_genes = len(genes_selected)
    print("Gene file loaded, #genes:", n_genes)

    # compute sum of effect products
    genes_cls_cnt = list()
    genes_cls_sum = list()
    genes_beta_prod = dict()
    genes_beta_prod_abs = dict()

    for i in range(n_trait):
        for j in range(i,n_trait):
            genes_beta_prod[(i,j)] = list()
            genes_beta_prod_abs[(i,j)] = list()

    chrom_min = np.min(chroms)
    min_chr_start_idx_gw = genes_selected[genes_selected['CHR']==chrom_min]['chr_start_idx_gw'].min()
    print("min chr_start_idx_gw for chr {}: {}".format(chrom_min, min_chr_start_idx_gw))

    for i_g in range(n_genes):
        # adjust snp indices to match those in beta_reg
        snps_g = np.arange(genes_selected.loc[i_g]['start_idx_chr'],genes_selected.loc[i_g]['end_idx_chr']+1).astype(int)
        snps_g += int(genes_selected.loc[i_g]['chr_start_idx_gw']-min_chr_start_idx_gw)
        # extract betas and cls for the gene
        betas_g = beta_reg[snps_g,:]
        betas_cls = cls_lbs[snps_g]
        cls_cnt = pd.Series(betas_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
        genes_cls_cnt.append(cls_cnt[np.arange(pred_K)].values)
        cls_sum = [np.abs(np.array(betas_g[betas_cls==ii,:])).sum(axis=0) for ii in np.arange(pred_K)]
        genes_cls_sum.append(np.array(cls_sum))
        
        # choose every combination of two from n_trait then compute the sum of produces
        s = genes_selected.loc[i_g]['N_SNPS']
        for i in range(n_trait):
            for j in range(i,n_trait):
                prods = [betas_g[betas_cls==cls,i].T @ betas_g[betas_cls==cls,j] for cls in np.arange(pred_K)]
                prods_abs = [np.abs(betas_g[betas_cls==cls,i]).T @ np.abs(betas_g[betas_cls==cls,j]) for cls in np.arange(pred_K)]
                prods /= s
                prods_abs /= s
                genes_beta_prod[(i,j)].append(prods)
                genes_beta_prod_abs[(i,j)].append(prods_abs)

    genes_cls_cnt = np.array(genes_cls_cnt)
    genes_cls_frac = np.zeros_like(genes_cls_cnt, dtype=float)
    for cls in range(pred_K):
        genes_cls_frac[:,cls] = genes_cls_cnt[:,cls] / genes_selected['N_SNPS']
    # genes_cls_sum is of shape n_genes x pred_K x n_trait
    genes_beta_prod = {k: np.array(v) for k, v in genes_beta_prod.items()}
    genes_beta_prod_abs = {k: np.array(v) for k, v in genes_beta_prod_abs.items()}

    # save results
    np.savetxt("{}_genes_cls_frac.txt".format(output_file), genes_cls_frac, delimiter=',', fmt='%.6f')
    for key, val in genes_beta_prod.items():
        i,j = key
        np.savetxt("{}_genes_cls_bprod_({},{}).txt".format(output_file,i,j), val, delimiter=',')
    for key, val in genes_beta_prod_abs.items():
        i,j = key
        np.savetxt("{}_genes_cls_bprodabs_({},{}).txt".format(output_file,i,j), val, delimiter=',')
    
    # understand assocation types based on eigenvalues of Sigma
    
    nz_cls_perc = pd.Series(cls_lbs[cls_lbs>=0]).value_counts(normalize=True).reindex(np.arange(pred_K), fill_value=0)
    nz_cls_perc_cumsum = nz_cls_perc.cumsum()
    sig_cls = np.where((nz_cls_perc_cumsum<=prioritized_cls_prop))[0]

    if n_trait==2:
        sigma_eiginfo = get_eiginfo(Sigma, comp_to_ref=False)
        cls_is_trait_specific = list()
        cls_is_shared = list()
        
        # 15 degree
        for eig,ang in sigma_eiginfo:
            is_specific, is_shared = check_specific_or_shared_bivar(ang, eig, angle_tol=15, axes_ratio_specific = 2, axes_ratio_shared = 1.5)
            cls_is_trait_specific.append(is_specific)
            cls_is_shared.append(is_shared)
        
        cls_is_trait_specific = np.array(cls_is_trait_specific)
        cls_is_shared = np.array(cls_is_shared)
        trait_1_specific = np.where(cls_is_trait_specific[:,0])[0]
        trait_2_specific = np.where(cls_is_trait_specific[:,1])[0]
        trait_shared = np.where(cls_is_shared)[0]
        
        sig_trait_1_specific = np.array(list(set(sig_cls).intersection(trait_1_specific)))
        sig_trait_2_specific = np.array(list(set(sig_cls).intersection(trait_2_specific)))
        sig_both = np.array(list(set(sig_cls).intersection(trait_shared)))
        print("Prioritized trait 1 specific clusters:", sig_trait_1_specific+1)
        print("Prioritized trait 2 specific clusters:", sig_trait_2_specific+1)
        print("Prioritized shared clusters:", sig_both+1)

        # save results
        results = dict()
        results["(1,0)"] = (sig_trait_1_specific+1).tolist()
        results["(0,1)"] = (sig_trait_2_specific+1).tolist()
        results["(1,1)"] = (sig_both+1).tolist()

    if n_trait==3:
        sigma_eiginfo = get_eiginfo(Sigma, comp_to_ref=True)
        cls_specific = check_specific_trivar(sigma_eiginfo, rad_thre=np.pi/12, eigval_times_thre=5)
        cls_specific = np.array(cls_specific)

        sig_trait_A = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,0])[0])))
        sig_trait_B = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,1])[0])))
        sig_trait_C = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,2])[0])))
        sig_trait_AB = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,3])[0])))
        sig_trait_AC = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,4])[0])))
        sig_trait_BC = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,5])[0])))
        sig_trait_ABC = np.array(list(set(sig_cls).intersection(np.where(cls_specific[:,6])[0])))
        print("Prioritized trait A specific clusters:", sig_trait_A+1)
        print("Prioritized trait B specific clusters:", sig_trait_B+1)
        print("Prioritized trait C specific clusters:", sig_trait_C+1)
        print("Prioritized trait AB specific clusters:", sig_trait_AB+1)
        print("Prioritized trait AC specific clusters:", sig_trait_AC+1)
        print("Prioritized trait BC specific clusters:", sig_trait_BC+1)
        print("Prioritized trait ABC specific clusters:", sig_trait_ABC+1)

        # save results
        results = dict()
        results["(1,0,0)"] = (sig_trait_A+1).tolist()
        results["(0,1,0)"] = (sig_trait_B+1).tolist()
        results["(0,0,1)"] = (sig_trait_C+1).tolist()
        results["(1,1,0)"] = (sig_trait_AB+1).tolist()
        results["(1,0,1)"] = (sig_trait_AC+1).tolist()
        results["(0,1,1)"] = (sig_trait_BC+1).tolist()
        results["(1,1,1)"] = (sig_trait_ABC+1).tolist()
        
    if n_trait==2 or n_trait==3:
        print("Saving clustering association types to {}_cls_association_types.txt".format(output_file))   
        with open("{}_cls_association_types.txt".format(output_file), 'w') as f:
            f.write("Type\tClusters\n")
            for key, val in results.items():
                f.write("{}\t{}\n".format(key, ",".join(map(str, val))))

        # for each type of association, compute the sum of abs effects in the corresponding clusters
        for assoc_type, cls_of_interest in results.items():
            if len(cls_of_interest)>0:
                # get the indices in tuple that equals 1
                assoc_idx = np.array([int(x) for x in assoc_type.strip("()").split(",")])
                assoc_idx = np.where(assoc_idx==1)[0]
                cls_indices = np.array(cls_of_interest)-1
                print("For association type {}, calculating sum of abs effects in clusters {} for traits {}".format(assoc_type, cls_of_interest, assoc_idx+1))
                beta_abs_sum_genes = list()
                for i_g in range(len(genes_cls_sum)):
                    beta_abs_sum = np.sum([genes_cls_sum[i_g][cls_indices-1,i] for i in assoc_idx])
                    beta_abs_sum_genes.append(beta_abs_sum)
                beta_abs_sum_genes = np.array(beta_abs_sum_genes)
                np.savetxt("{}_genes_beta_abs_sum_{}.txt".format(output_file, assoc_type), beta_abs_sum_genes, delimiter=',')


def main():
    
    parser = argparse.ArgumentParser(description='Gene-level analysis for multiple traits')
    parser.add_argument('--output_file', type=str, help='Output file name (prefix) to save clustering results')
    parser.add_argument('--gene_file', type=str, help="Gene file")
    parser.add_argument('--clustering_file_prefix', type=str, help='Clustering file prefix')
    parser.add_argument('--chroms', type=int, nargs="*", default=None, help="Chromosome numbers (0–22). If not provided, all chromosomes will be used.")
    parser.add_argument('--prioritized_cls_prop', type=float, default=1, help="Proportion of variants to be considered in prioritized clusters (default 1, i.e., all non-zero variants)")

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

    disp_params(args, title="INPUT SETTINGS")
    if os.path.isfile(args.output_file):
        print("Warning: output_file already exists and will be overwritten!")
    else:
        # create directory if not exist
        if not os.path.isdir(os.path.dirname(args.output_file)) and os.path.dirname(args.output_file) != '':
            os.makedirs(os.path.dirname(args.output_file))
            print("Created output directory:", os.path.dirname(args.output_file))
    print("--------------------------------------------------")
    if args.chroms is None or len(args.chroms)==0:
        chroms = list(range(1,23))
    else:
        chroms = args.chroms
    print("Chromosomes to be used:", chroms)
    print("--------------------------------------------------")
    multivar_gene_analysis(**vars(args))

    
if __name__ == "__main__":

    main()
