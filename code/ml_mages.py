import os
import time
import numpy as np
import pandas as pd
import argparse
import torch
import _main_funcs as mf


def main(args):
    # take in command line arguments
    print("-----Required Arguments: ")
    gwa_files = args.gwa_files.split(",")
    print("gwa_files:", gwa_files)
    traits = args.traits.split(",")
    print("traits:", traits)
    ld_path = args.ld_path
    print("ld_path:", ld_path)
    ld_block_file = args.ld_block_file
    print("ld_block_file:", ld_block_file)
    gene_file = args.gene_file
    print("gene_file:", gene_file)
    model_path = args.model_path
    print("model_path:", model_path)
    n_layer = args.n_layer
    print("n_layer:", n_layer)
    top_r = args.top_r
    print("top_r:", top_r)
    n_models = args.n_models
    print("n_models:", n_models)
    output_path = args.output_path
    print("output_path:", output_path)

    scale = 250
    n_traits = len(traits)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.random.seed(42)

    print("==========ML-MAGES==========")

    # load input
    start_time = time.time()

    gwa_loaded = mf.load_gwa(gwa_files, cols=['BETA','SE','CHR'])
    beta = [g["BETA"] for g in gwa_loaded]
    se = [g["SE"] for g in gwa_loaded]
    ld_block_ids = np.loadtxt(ld_block_file, dtype=int).astype(int)

    brkpts = np.insert(ld_block_ids,0,0)
    ld_files = [os.path.join(ld_path,"block_{}.ld".format(i_bk)) for i_bk in range(len(ld_block_ids))]
    ld_list = mf.load_ld_blocks(ld_files, sep=",")
    assert(sum([ld.shape[0] for ld in ld_list])==len(beta[0]))

    # load trained models
    models = mf.load_models(model_path,n_layer,top_r,n_models=n_models)

    end_time = time.time()
    print("Loading takes {:.2f} seconds".format(end_time - start_time))

    # shrinkage and univariate clustering
    start_time = time.time()
    for i_trait, trait in enumerate(traits):
        with torch.no_grad():
            beta_reg = list()
            for i_bk in range(len(brkpts)-1):
                lb, ub = brkpts[i_bk], brkpts[i_bk+1]
                # construct model input
                bhat, shat = beta[i_trait][lb:ub], se[i_trait][lb:ub]
                X = mf.construct_features(bhat, shat, ld_list[i_bk], top_r)
                # scale input
                scale_idx = np.concatenate([[0,1], np.arange(3,3+top_r)])
                X[:,scale_idx] *= scale
                # apply shrinkage models
                breg_bk = list()
                for model in models:
                    breg_bk.append(model(torch.tensor(X, dtype=torch.float32)).detach().numpy().squeeze())
                breg_bk = np.vstack(breg_bk)/scale
                beta_reg.append(np.mean(breg_bk,axis=0))  
        beta_reg = np.concatenate(beta_reg)
        # save results
        reg_file = os.path.join(output_path,"regularized_effects_{}.txt".format(traits[i_trait]))
        np.savetxt(reg_file, beta_reg, delimiter=',')
        # plot
        beta_obs, beta_chrs = gwa_loaded[i_trait]["BETA"], gwa_loaded[i_trait]["CHR"]
        fig_file = os.path.join(output_path,"shrinkage_{}.png".format(trait))
        mf.plot_shrinkage(beta_obs, beta_reg, beta_chrs, fig_file)
        
        # clustering
        beta_nz, zero_cutoff = mf.get_nz_effects(beta_reg, fold_min=500, fold_max=5, zero_cutoff=1e-4, adjust_max = 20, adjust_rate = 1.5)
        Sigma, pi, pred_K, pred_cls = mf.clustering(beta_nz, K=30, n_runs=15)
        cls_labels = -np.ones(len(beta_reg))
        cls_labels[np.where(np.abs(beta_reg)>zero_cutoff)[0]] = pred_cls
        # save results
        np.savetxt(os.path.join(output_path,"univar_{}_Sigma.txt".format(traits[i_trait])), Sigma.squeeze().ravel(), delimiter=',') 
        np.savetxt(os.path.join(output_path,"univar_{}_pi.txt".format(traits[i_trait])), pi.squeeze(), delimiter=',') 
        np.savetxt(os.path.join(output_path,"univar_{}_cls.txt".format(traits[i_trait])), cls_labels, delimiter=',')
        np.savetxt(os.path.join(output_path,"univar_{}_zc.txt".format(traits[i_trait])), np.array([zero_cutoff]), delimiter=',')
        # plot
        fig_file = os.path.join(output_path,"clustering_univar_{}.png".format(traits[i_trait]))
        mf.plot_clustering(beta_nz, pred_cls, pred_K, Sigma, pi, traits, fig_file)

    end_time = time.time()
    print("Shrinkage and univariate clustering take {:.2f} seconds".format(end_time - start_time))

    # multivariate clustering
    start_time = time.time()
    beta_reg_multi = list()
    for i_trait, trait in enumerate(traits):
        # load results
        reg_file = os.path.join(output_path,"regularized_effects_{}.txt".format(traits[i_trait]))
        beta_reg = np.loadtxt(reg_file, delimiter=',')
        beta_reg_multi.append(beta_reg)
    beta_reg_multi = np.vstack(beta_reg_multi).T
    beta_nz_multi, zero_cutoff = mf.get_nz_effects(beta_reg_multi, fold_min=500, fold_max=5, zero_cutoff=1e-4, adjust_max = 20, adjust_rate = 1.5)
    Sigma, pi, pred_K, pred_cls = mf.clustering(beta_nz_multi, K=30, n_runs=25)
    cls_labels = np.ones(beta_reg_multi.shape[0])*(-1)
    is_nz = np.any(np.abs(beta_reg_multi)>zero_cutoff, axis=1) 
    cls_labels[is_nz] = pred_cls
    # save results
    output_lb = "-".join(traits)
    np.savetxt(os.path.join(output_path,"multivar_{}_Sigma.txt".format(output_lb)), Sigma.reshape(Sigma.shape[0],-1), delimiter=',') 
    np.savetxt(os.path.join(output_path,"multivar_{}_pi.txt".format(output_lb)), pi.squeeze(), delimiter=',') 
    np.savetxt(os.path.join(output_path,"multivar_{}_cls.txt".format(output_lb)), cls_labels, delimiter=',')
    np.savetxt(os.path.join(output_path,"multivar_{}_zc.txt".format(output_lb)), np.array([zero_cutoff]), delimiter=',')
    if n_traits==2:
        # plot
        fig_file = os.path.join(output_path,"clustering_multivar_{}.png".format(output_lb))
        mf.plot_clustering(beta_nz_multi, pred_cls, pred_K, Sigma, pi, traits, fig_file)
        # plot with null "zero" effects
        # beta_zero_multi = beta_reg_multi[~is_nz,:]

    end_time = time.time()
    print("Multivariate clustering takes {:.2f} seconds".format(end_time - start_time))

    # gene-level analysis 
    # enrichment test
    start_time = time.time()
    # load gene data
    genes = pd.read_csv(gene_file)
    for i_trait, trait in enumerate(traits):
        # load shrinkage and clustering results
        reg_file = os.path.join(output_path,"regularized_effects_{}.txt".format(traits[i_trait]))
        beta_reg = np.loadtxt(reg_file, delimiter=',')
        Sigma = np.loadtxt(os.path.join(output_path,"univar_{}_Sigma.txt".format(traits[i_trait])), delimiter=',') 
        Sigma = [s.reshape(1,1) for s in Sigma]
        # enrichment analysis null threshold
        eps_eff_cls = 1
        eps_eff = Sigma[eps_eff_cls][0][0]
        # enrichment analysis
        df_enrich = list()
        for i_bk in range(len(brkpts)-1):
            lb, ub = brkpts[i_bk], brkpts[i_bk+1]
            genes_bk = genes[(genes['SNP_FIRST']>=lb) & (genes['SNP_LAST']<ub)]
            genes_bk = genes_bk.reset_index(drop=True)
            genes_bk.loc[:,'SNP_FIRST'] -= lb
            genes_bk.loc[:,'SNP_LAST'] -= lb
            enrich_stats = mf.enrichment_test(genes_bk, eps_eff, beta_reg[lb:ub], ld_list[i_bk])
            df_enrich_bk = pd.DataFrame(enrich_stats)
            df_enrich.append(pd.concat([genes_bk,df_enrich_bk],axis=1))
        df_enrich = pd.concat(df_enrich,axis=0).reset_index(drop=True)
        # save
        df_enrich.to_csv(os.path.join(output_path,"enrichment_{}.csv".format(traits[i_trait])), index=False)
        # plot
        fig_file = os.path.join(output_path,"enrichment_{}.png".format(trait))
        mf.plot_enrichment(df_enrich, fig_file, level=0.05)
    end_time = time.time()
    print("Univariate gene enrichment analysis takes {:.2f} seconds".format(end_time - start_time))

    # multivariate analysis
    start_time = time.time()
    betas = list()
    for i_trait, trait in enumerate(traits):
        # load results
        reg_file = os.path.join(output_path,"regularized_effects_{}.txt".format(traits[i_trait]))
        beta_reg = np.loadtxt(reg_file, delimiter=',')
        betas.append(beta_reg)
    betas = np.vstack(betas).T
    cls_lbs = np.loadtxt(os.path.join(output_path,"multivar_{}_cls.txt".format("-".join(traits))), delimiter=',') 
    Sigma = np.loadtxt(os.path.join(output_path,"multivar_{}_Sigma.txt".format("-".join(traits))), delimiter=',') 
    pred_K = len(Sigma)
    Sigma = [s.reshape(n_traits,n_traits) for s in Sigma]
    genes = pd.read_csv(gene_file)
    df_multi_gene = mf.summarize_multivariate_gene(genes,betas,cls_lbs,pred_K)
    # save
    df_multi_gene.to_csv(os.path.join(output_path,"multivar_gene_{}.csv".format("-".join(traits))), index=False)
    end_time = time.time()
    print("Multivariate gene analysis takes {:.2f} seconds".format(end_time - start_time))

    print("============DONE============")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide the required arguments')

    # Required argument
    parser.add_argument('--gwa_files', type=str, required=True, help='comma-separated list of GWA files')
    parser.add_argument('--traits', type=str, required=True, help='comma-separated list of traits associated with the GWA files')
    parser.add_argument('--ld_path', type=str, required=True, help='path to the LD (block) files')
    parser.add_argument('--ld_block_file', type=str, required=True, help='file containing the LD block IDs')
    parser.add_argument('--gene_file', type=str, required=True, help='file containing gene data')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained models')
    parser.add_argument('--n_layer', type=int, choices=[2,3], required=True, help='number of layers in the model, chosen from [2,3]')
    parser.add_argument('--top_r', type=int, choices=[15], required=True, help='number of top (highest correlation) variants used to construct the features, chosen from [15]')
    parser.add_argument('--n_models', type=int, required=True, help='number of models; note that the number of models should be consistent with those trained (indexed from 0 to n_models-1)')
    parser.add_argument('--output_path', type=str, required=True, help='path to save the output files')

    args = parser.parse_args()
    main(args)

