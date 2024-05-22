import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import ml_mages

def main():
    # take in command line arguments
    if len(sys.argv) != 10:
        print("Usage: {} run_sample.py gwa_files traits ld_path ld_block_file gene_file model_path n_layer top_r output_path".format(sys.argv[0]))
        sys.exit(1)

    gwa_files = sys.argv[1].split(",")
    print("gwa_files:", gwa_files)
    traits = sys.argv[2].split(",")
    print("traits:", traits)
    ld_path = sys.argv[3]
    print("ld_path:", ld_path)
    ld_block_file = sys.argv[4]
    print("ld_block_file:", ld_block_file)
    gene_file = sys.argv[5]
    print("gene_file:", gene_file)
    model_path = sys.argv[6]
    print("model_path:", model_path)
    n_layer = int(sys.argv[7])
    print("n_layer:", n_layer)
    top_r = int(sys.argv[8])
    print("top_r:", top_r)
    output_path = sys.argv[9]
    print("output_path:", output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.random.seed(42)

    print("==========ML-MAGES==========")

    # load input
    start_time = time.time()
    beta, se = ml_mages.load_gwa_results(gwa_files, beta_col='BETA', se_col='SE')
    ld_block_ids = np.loadtxt(ld_block_file, dtype=int).astype(int)
    brkpts = np.insert(ld_block_ids,0,0)
    ld_files = [os.path.join(ld_path,"sample_block{}.ld".format(i_bk+1)) for i_bk in range(len(ld_block_ids))]
    ld_list = ml_mages.load_ld_blocks(ld_files)
    assert(sum([ld.shape[0] for ld in ld_list])==len(beta[0]))

    # load trained models
    model = ml_mages.load_model(model_path,n_layer,top_r)
    model.eval()

    end_time = time.time()
    print("Loading takes {:.2f} seconds".format(end_time - start_time))

    # shrinkage and univariate clustering
    start_time = time.time()
    for i_trait, trait in enumerate(traits):
        beta_reg = list()
        for i_bk in range(len(brkpts)-1):
            lb, ub = brkpts[i_bk], brkpts[i_bk+1]
            # construct model input
            bhat, shat = beta[i_trait][lb:ub], se[i_trait][lb:ub]
            X = ml_mages.construct_features(bhat, shat, ld_list[i_bk], top_r)
            # apply shrinkage model
            breg = model(torch.tensor(X, dtype=torch.float32)).detach().numpy().squeeze()
            beta_reg.append(breg)
        beta_reg = np.concatenate(beta_reg)
        # save results
        reg_file = os.path.join(output_path,"regularized_effects_{}.txt".format(traits[i_trait]))
        np.savetxt(reg_file, beta_reg, delimiter=',')
        
        # clustering
        beta_nz, zero_cutoff = ml_mages.get_nz_effects(beta_reg, zero_cutoff=1e-3, adjust_max = 10)
        Sigma, pi, pred_K, pred_cls = ml_mages.clustering(beta_nz, K=20, n_runs=15)
        cls_labels = -np.ones(len(beta_reg))
        cls_labels[np.where(np.abs(beta_reg)>zero_cutoff)[0]] = pred_cls
        # save results
        np.savetxt(os.path.join(output_path,"univar_{}_Sigma.txt".format(traits[i_trait])), Sigma.squeeze().ravel(), delimiter=',') 
        np.savetxt(os.path.join(output_path,"univar_{}_pi.txt".format(traits[i_trait])), pi.squeeze(), delimiter=',') 
        np.savetxt(os.path.join(output_path,"univar_{}_cls.txt".format(traits[i_trait])), cls_labels, delimiter=',')
        np.savetxt(os.path.join(output_path,"univar_{}_zc.txt".format(traits[i_trait])), np.array([zero_cutoff]), delimiter=',')
        # plot
        fig_file = os.path.join(output_path,"clustering_univar_{}.png".format(traits[i_trait]))
        ml_mages.plot_clustering(beta_nz, pred_cls, pred_K, Sigma, pi, traits, fig_file)
    end_time = time.time()
    print("Shrinkage and univariate clustering take {:.2f} seconds".format(end_time - start_time))

    # bivariate clustering
    start_time = time.time()
    beta_reg_multi = list()
    for i_trait, trait in enumerate(traits):
        # load results
        reg_file = os.path.join(output_path,"regularized_effects_{}.txt".format(traits[i_trait]))
        beta_reg = np.loadtxt(reg_file, delimiter=',')
        beta_reg_multi.append(beta_reg)
    beta_reg_multi = np.vstack(beta_reg_multi).T
    beta_nz_multi, zero_cutoff = ml_mages.get_nz_effects(beta_reg_multi, zero_cutoff=1e-4, adjust_max = 10)
    Sigma, pi, pred_K, pred_cls = ml_mages.clustering(beta_nz_multi, K=20, n_runs=15)
    cls_labels = np.ones(beta_reg_multi.shape[0])*(-1)
    is_nz = np.any(np.abs(beta_reg_multi)>zero_cutoff, axis=1) 
    cls_labels[is_nz] = pred_cls
    # save results
    output_lb = "-".join(traits)
    np.savetxt(os.path.join(output_path,"multivar_{}_Sigma.txt".format(output_lb)), Sigma.reshape(Sigma.shape[0],-1), delimiter=',') 
    np.savetxt(os.path.join(output_path,"multivar_{}_pi.txt".format(output_lb)), pi.squeeze(), delimiter=',') 
    np.savetxt(os.path.join(output_path,"multivar_{}_cls.txt".format(output_lb)), cls_labels, delimiter=',')
    np.savetxt(os.path.join(output_path,"multivar_{}_zc.txt".format(output_lb)), np.array([zero_cutoff]), delimiter=',')
    # plot
    fig_file = os.path.join(output_path,"clustering_multivar_{}.png".format(output_lb))
    ml_mages.plot_clustering(beta_nz_multi, pred_cls, pred_K, Sigma, pi, traits, fig_file)

    end_time = time.time()
    print("Bivariate clustering takes {:.2f} seconds".format(end_time - start_time))


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
            enrich_stats = ml_mages.enrichment_test(genes_bk, eps_eff, beta_reg[lb:ub], ld_list[i_bk])
            df_enrich_bk = pd.DataFrame(enrich_stats)
            df_enrich.append(pd.concat([genes_bk,df_enrich_bk],axis=1))
        df_enrich = pd.concat(df_enrich,axis=0).reset_index(drop=True)
        # save
        df_enrich.to_csv(os.path.join(output_path,"enrichment_{}.csv".format(traits[i_trait])), index=False)
    end_time = time.time()
    print("Univariate gene enrichment analysis takes {:.2f} seconds".format(end_time - start_time))

    # bivariate analysis
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
    Sigma = [s.reshape(2,2) for s in Sigma]
    genes = pd.read_csv(gene_file)
    df_bivar_gene = ml_mages.summarize_bivariate_gene(genes,betas,cls_lbs,pred_K)
    # save
    df_bivar_gene.to_csv(os.path.join(output_path,"bivar_gene_{}.csv".format(traits[i_trait])), index=False)
    end_time = time.time()
    print("Bivariate gene analysis takes {:.2f} seconds".format(end_time - start_time))

    print("============DONE============")

if __name__ == "__main__":
    main()