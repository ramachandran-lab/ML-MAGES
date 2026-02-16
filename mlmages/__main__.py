#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns

from .extract_ld_blocks import extract_ld_blocks
from .shrink_by_mlmages import shrink_by_mlmages
from .cluster_shrinkage import cluster_shrinkage
from .univar_enrich import univar_enrich
from .multivar_gene_analysis import multivar_gene_analysis

from ._util_funcs import disp_params, parse_file_list, binary_combinations, str2bool
from ._plot_funcs import plot_inf_cls, plot_data_cls, plot_inf_cls_bivar, plot_data_cls_bivar, plot_data_cls_trivar, plot_inf_cls_trivar, plot_effects, plot_pvals


def main(args):
    

    disp_params(args, title="INPUT SETTINGS")

    # Make sure output directories exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    shrink_dir = os.path.join(args.output_dir, "shrinkage")
    cluster_dir = os.path.join(args.output_dir, "clustering")
    enrich_dir = os.path.join(args.output_dir, "enrichment")
    genelevel_dir = os.path.join(args.output_dir, "genelevel")
    intermediate_dir = os.path.join(args.output_dir, "intermediate_files")
    fig_path = os.path.join(args.output_dir, "figures")
    for d in [shrink_dir, cluster_dir, enrich_dir, genelevel_dir, intermediate_dir, fig_path]:
        os.makedirs(d, exist_ok=True)

    full_ld_files = parse_file_list(args.full_ld_files)
    if len(args.chroms)==1:
        chrom_label = str(args.chroms[0])
    else:
        chrom_label = "{}-{}".format(args.chroms[0], args.chroms[-1])
    print("Chromosomes to be used:", chrom_label)

    # --- 0. Extract LD blocks per chromosome ---
    if args.split_ld:
        print(full_ld_files)
        assert len(full_ld_files) == len(args.chroms), "Number of full LD files must match number of chromosomes!"
        os.makedirs(args.ldblock_dir, exist_ok=True)
        
        for i_chrom,chrom in enumerate(args.chroms):
            ld_file = full_ld_files[i_chrom]
            block_meta_file = os.path.join(args.ldblock_dir, f"block_meta_chr{chrom}.csv")
            res_prefix = f"chr{chrom}"
            print(f"Extracting LD blocks for chromosome {chrom} from {ld_file} ...")
            print(f"Block meta file: {block_meta_file}")

            extract_ld_blocks(
                ld_file=ld_file,
                ldblock_path=args.ldblock_dir,
                block_meta_file=block_meta_file,
                avg_block_size=args.avg_block_size,
                res_prefix=res_prefix
            )

        # --- 0b. Create LD files list for later use (one per chromosome) ---
        
        if len(args.chroms)==1:
            chrom = args.chroms[0]
            ld_files_list_path = os.path.join(intermediate_dir, f"chr{chrom}_ld_files.txt")
            all_block_files = [os.path.join(args.ldblock_dir, f) for f in os.listdir(args.ldblock_dir)
                            if f.startswith(f"chr{chrom}_block") and f.endswith(".ld")]
            # Sort by block number
            all_block_files.sort(key=lambda x: int(os.path.basename(x).split("_block")[1].split(".ld")[0]))
            with open(ld_files_list_path, "w") as f:
                for file_path in all_block_files:
                    f.write(f"{file_path}\n")
            print(f"Created LD file list: {ld_files_list_path}")
        else:
            ld_files_list_path = os.path.join(intermediate_dir, f"chr{chrom_label}_ld_files.txt")
            with open(ld_files_list_path, "w") as f:
                for chrom in args.chroms:
                    all_block_files = [os.path.join(args.ldblock_dir, f) for f in os.listdir(args.ldblock_dir)
                                    if f.startswith(f"chr{chrom}_block") and f.endswith(".ld")]
                    # Sort by block number
                    all_block_files.sort(key=lambda x: int(os.path.basename(x).split("_block")[1].split(".ld")[0]))
                    for file_path in all_block_files:
                        f.write(f"{file_path}\n")
            print(f"Created LD file list: {ld_files_list_path}")

    
    # --- 1. Shrinkage and univariate clustering per trait ---
    for trait, gwas_file in zip(args.trait_names, args.gwas_files):
        # for chrom in args.chroms:
        if args.split_ld:
            ld_files_list = [os.path.join(intermediate_dir, f"chr{chrom_label}_ld_files.txt")]
        else:
            assert len(full_ld_files) == len(args.chroms), "Number of full LD files must match number of chromosomes!"
            ld_files_list = [full_ld_files[i] for i, c in enumerate(args.chroms)]

        shrink_output = os.path.join(shrink_dir, f"shrink_{trait}_chr{chrom_label}.txt")
        shrink_by_mlmages(gwas_file, ld_files_list, args.model_files, shrink_output, chroms=args.chroms)

        # Univariate clustering
        shrinkage_trait_files = {
            "shrinkage_trait1_files": [shrink_output],
        }
        cluster_output = os.path.join(cluster_dir, f"{trait}_chr{chrom_label}")
        cluster_shrinkage(shrinkage_trait_files, cluster_output)

        # Enrichment analysis
        enrich_output = os.path.join(enrich_dir, f"{trait}_gene_enrich_chr{chrom_label}.txt")
        univar_enrich(
            output_file=enrich_output,
            gene_file=args.gene_file,
            shrinkage_file=shrink_output,
            clustering_file_prefix=cluster_output,
            ld_files=args.full_ld_files,
            chroms=args.chroms
        )
            
    # --- 2. Multivariate clustering for all traits (if available) ---
    if len(args.trait_names) >= 2:
        n_traits = len(args.trait_names)
        multivar_name = "-".join(args.trait_names)
        # for chrom in args.chroms:
        shrinkage_trait_files = {}
        for i in range(n_traits):
            shrinkage_trait_files[f"shrinkage_trait{i+1}_files"] = [os.path.join(shrink_dir, f"shrink_{args.trait_names[i]}_chr{chrom_label}.txt")]
        multivariate_cluster_output = os.path.join(cluster_dir, f"{multivar_name}_chr{chrom_label}")
        cluster_shrinkage(shrinkage_trait_files, multivariate_cluster_output)

        # Gene-level multivariate analysis
        genelevel_output = os.path.join(genelevel_dir, f"{multivar_name}_chr{chrom_label}")
        multivar_gene_analysis(
            output_file=genelevel_output,
            gene_file=args.gene_file,
            clustering_file_prefix=multivariate_cluster_output,
            shrinkage_files=shrinkage_trait_files,
            chroms=args.chroms
        )

    print("Pipeline completed successfully.")

    # --- 3. Visualizations (if --vis True) ---
    if args.vis:
        # # bivariate clustering
        traits = args.trait_names
        
        # load shrinkage
        for i, trait in enumerate(traits):
            suffix = f"{trait}_chr{chrom_label}"
            shrinkage_file = os.path.join(shrink_dir, f"shrink_{suffix}.txt")
            if i==0:
                breg = np.loadtxt(shrinkage_file)[:,None]
            else:
                breg = np.concatenate([breg, np.loadtxt(shrinkage_file)[:,None]], axis=1)
        print(breg.shape)

        if n_traits >=2:

            # load clustering
            clustering_file_prefix = multivariate_cluster_output
            Sigma = np.loadtxt("{}_Sigma.txt".format(clustering_file_prefix), delimiter=',') 
            # n_traits = int(np.sqrt(Sigma[0].shape[0]))
            print("n_traits:", n_traits)
            Sigma = [s.reshape(n_traits,n_traits) for s in Sigma]
            pi = np.loadtxt("{}_pi.txt".format(clustering_file_prefix), delimiter=',') 
            pred_K = len(pi)
            for k in range(pred_K):
                print("Cluster {}: pi={:.3f}, Sigma=\n{}".format(k, pi[k], Sigma[k]))
            cls_lbs = np.loadtxt("{}_cls.txt".format(clustering_file_prefix), delimiter=',').astype(int)
            meta = np.loadtxt("{}_meta.txt".format(clustering_file_prefix), delimiter=',')

            # plot clustering results 
            if n_traits==2:
                
                is_nz = cls_lbs>=0
                breg_nz = breg[is_nz] 
                pred_cls = cls_lbs[is_nz].astype(int)
                cls_cnt = pd.Series(pred_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
                cls_perc = (cls_cnt/cls_cnt.sum()).loc[np.arange(pred_K)]    

                x_extreme = np.max(np.abs(breg))*1.05
                df = pd.DataFrame(breg_nz, columns=traits)
                df['Cluster'] = pred_cls
                ax = plot_data_cls_bivar(df, traits, 'Cluster', cls_perc, x_extreme)
                fig = ax.get_figure()
                fig.savefig(os.path.join(fig_path, "{}_data_cls.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

                ax = plot_inf_cls_bivar(traits, pred_K, Sigma, pi, x_extreme) 
                fig = ax.get_figure()
                fig.savefig(os.path.join(fig_path, "{}_inf_cls.png".format("-".join(traits))), dpi=150, bbox_inches='tight')
            
            elif n_traits==3:
                
                is_nz = cls_lbs>=0
                breg_nz = breg[is_nz] 
                pred_cls = cls_lbs[is_nz].astype(int)
                cls_cnt = pd.Series(pred_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
                cls_perc = (cls_cnt/cls_cnt.sum()).loc[np.arange(pred_K)]    

                x_extreme = np.max(np.abs(breg))*1.05
                df = pd.DataFrame(breg_nz, columns=traits)
                df['Cluster'] = pred_cls
                ax = plot_data_cls_trivar(df, traits, 'Cluster', cls_perc, x_extreme, view=(15, 30))
                fig = ax.get_figure()
                fig.savefig(os.path.join(fig_path, "{}_data_cls.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

                ax = plot_inf_cls_trivar(traits, pred_K, Sigma, pi, x_extreme, view=(15, 30))
                fig = ax.get_figure()
                fig.savefig(os.path.join(fig_path, "{}_inf_cls.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

            # plot gene-level analysis
            genelevel_prefix = genelevel_output
            n_traits = len(traits)
            genes_val_list = list()
            genes_val_names = list()
            for i in range(n_traits):
                for j in range(i, n_traits):
                    genes_val_file = "{}_genes_cls_bprodabs_({},{})".format(genelevel_prefix, i, j)+".txt"
                    if not os.path.isfile(genes_val_file):  # file not exist
                        continue
                    genes_val = np.loadtxt(genes_val_file, delimiter=',')    
                    genes_val_list.append(genes_val.sum(axis=1))
                    genes_val_names.append("$\\beta_{}$$\\beta_{}$".format(i, j))

            fig, ax = plt.subplots(1,1, figsize=(9,3), dpi=150, sharex=True)
            if len(genes_val_list)>20:
                palette = list(sns.color_palette("husl", len(genes_val_list)))
                rng = np.random.default_rng(123)  
                rng.shuffle(palette)
            elif len(genes_val_list)>8:
                palette = sns.color_palette("tab20", len(genes_val_list))   
            else:
                palette = sns.color_palette("colorblind", len(genes_val_list))
            for i in range(len(genes_val_list)):
                ax.scatter(np.arange(len(genes_val_list[i])), genes_val_list[i], color=palette[i], 
                        edgecolor='black', alpha=0.7, s=20, lw=0.1, label=genes_val_names[i])
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            ax.set_xlabel("Gene index")
            ax.set_ylabel("Normalized $\\Sigma|$\\beta_i\\beta_j$| over variants in gene")
            ax.set_title("Gene-level effect size products")
            fig.savefig(os.path.join(fig_path, "{}_genes_beta_prod.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

            cls_frac = np.loadtxt("{}_genes_cls_frac.txt".format(genelevel_prefix), delimiter=',')
            fig, ax = plt.subplots(1,1, figsize=(9,3), dpi=150, sharex=True)
            bottom = np.zeros(cls_frac.shape[0])
            if cls_frac.shape[1]<=8:
                palette = sns.color_palette("colorblind", len(cls_frac))
            else:
                palette = list(sns.color_palette("husl", len(cls_frac)))
                rng = np.random.default_rng(123)   
                rng.shuffle(palette)
            for i in range(cls_frac.shape[1]):
                ax.bar(np.arange(cls_frac.shape[0]), cls_frac[:,i], bottom=bottom, color=palette[i], edgecolor='black', lw=0.1, label="Cls."+str(i+1))
                bottom += cls_frac[:,i]
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            ax.set_xlabel("Gene index")
            ax.set_ylabel("Proportion of variants in cluster")
            ax.set_title("Gene-level cluster membership proportions")
            fig.savefig(os.path.join(fig_path, "{}_genes_cls_frac.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

            bin_combs = binary_combinations(n_traits)
            genes_val_list = list()
            genes_val_names = list()
            for bin_comb in bin_combs:
                genes_val_file = "{}_genes_beta_abs_sum_({}).txt".format(genelevel_prefix,",".join(map(str,bin_comb)))
                if os.path.isfile(genes_val_file):  # file exist
                    genes_val = np.loadtxt(genes_val_file, delimiter=',')
                    genes_val_list.append(genes_val)
                    name = "(prioritized) assoc. with " + ",".join([traits[i] for i in range(n_traits) if bin_comb[i]==1])
                    genes_val_names.append(name)

            fig, axes = plt.subplots(len(genes_val_list), 1, figsize=(9,2.5*len(genes_val_list)), dpi=150, sharex=True)
            for i in range(len(genes_val_list)):
                ax = axes[i] if len(genes_val_list)>1 else axes
                ax.scatter(np.arange(len(genes_val_list[i])), genes_val_list[i], color=palette[i], 
                        edgecolor='black', alpha=0.7, s=20, lw=0.1, )
                # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
                ax.set_xlabel("Gene index")
                ax.set_ylabel("abs sum of effects")
                ax.set_title(genes_val_names[i], loc='left')
            fig.suptitle("Gene-level abs sum of effects in corresponding clusters")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_path, "{}_genes_beta_abs_sum.png".format("-".join(traits))), dpi=150, bbox_inches='tight')
        
        print("Figures saved to", fig_path)
        print("Visualizations completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Master pipeline for ML-MAGES: (LD block extraction), shrinkage, clustering, and enrichment")
    parser.add_argument("--chroms", type=int, nargs="+", required=True, help="Chromosome numbers")
    parser.add_argument("--gwas_files", type=str, nargs="+", required=True, help="GWAS files, one per trait")
    parser.add_argument("--model_files", nargs="+", default=[], help="Model files for shrinkage")
    parser.add_argument("--full_ld_files", nargs="+", default=[], required=True, help="Full LD files for splitting into blocks and enrichment tests")
    parser.add_argument("--gene_file", type=str, required=True, help="Gene file for enrichment and gene-level analysis")
    parser.add_argument("--trait_names", type=str, nargs="+", required=True, help="Trait names (same order as gwas_files)")
    parser.add_argument("--split_ld", type=str2bool, nargs="?", default=True, help="Whether to split LD files by chromosome")
    parser.add_argument("--ldblock_dir", type=str, default="", help="Path to store LD block files")
    parser.add_argument("--avg_block_size", type=int, default=1000, help="Average block size for LD block extraction")
    parser.add_argument("--vis", type=str2bool, nargs="?", default=True, help="Whether to generate visualizations")
    parser.add_argument("--output_dir", type=str, default="../output", help="Output directory")
    args = parser.parse_args()

    main(args)