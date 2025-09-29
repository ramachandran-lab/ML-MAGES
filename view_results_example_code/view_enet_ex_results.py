import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from matplotlib import pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlmages._util_funcs import load_gwas_file, binary_combinations
from mlmages._plot_funcs import plot_inf_cls, plot_data_cls, plot_inf_cls_bivar, plot_data_cls_bivar, plot_data_cls_trivar, plot_inf_cls_trivar, plot_effects, plot_pvals

fig_path = "output/figures"
os.makedirs(fig_path, exist_ok=True)

chrom = 22
# gwas_file = "data/gwas/gwas_HDL.csv"
# shrinkage_file = "output/shrinkage/shrink_HDL_enet.txt"
# 

# df_gwas, beta, se = load_gwas_file(gwas_file, chroms=[chrom])
# beta_mlmages = np.loadtxt(shrinkage_file)

# chr_is_odd = df_gwas['CHR'].values%2==1
# chr_switch_idx = np.where(np.diff(df_gwas["CHR"])>0)[0]
# chr_switch_idx = np.insert(chr_switch_idx,0,0)
# chr_switch_idx = np.insert(chr_switch_idx,len(chr_switch_idx),len(df_gwas))

# fig = plot_effects([beta,beta_mlmages], chr_is_odd=chr_is_odd, chr_switch_idx=chr_switch_idx,
#                    sharey=True, marker_s=5)
# fig.axes[0].set_title("GWAS effect size", loc='left')
# fig.axes[1].set_title("ML-MAGES regularized effect size", loc='left')
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, "mlmages_shrinkage.png"), dpi=300)


# univariate clustering
trait = "LDL"
shrinkage_file = f"output/shrinkage/shrink_{trait}_enet.txt"
clustering_file_prefix = f"output/clustering/{trait}_enet"

# load shrinkage
breg = np.loadtxt(shrinkage_file)
# load clustering
Sigma = np.loadtxt("{}_Sigma.txt".format(clustering_file_prefix), delimiter=',') 
n_traits = 1
Sigma = [s.reshape(n_traits,n_traits) for s in Sigma]
pi = np.loadtxt("{}_pi.txt".format(clustering_file_prefix), delimiter=',') 
pred_K = len(pi)
for k in range(pred_K):
    print("Cluster {}: pi={:.3f}, Sigma=\n{}".format(k+1, pi[k], Sigma[k]))
cls_lbs = np.loadtxt("{}_cls.txt".format(clustering_file_prefix), delimiter=',').astype(int)
meta = np.loadtxt("{}_meta.txt".format(clustering_file_prefix), delimiter=',')
zero_cutoff = float(meta[0])

# plot clustering results 
is_nz = cls_lbs>=0
breg_nz = breg[is_nz] 
pred_cls = cls_lbs[is_nz].astype(int)
cls_cnt = pd.Series(pred_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
cls_perc = (cls_cnt/cls_cnt.sum()).loc[np.arange(pred_K)]    

x_extreme = np.max(np.abs(breg))*1.05
df = pd.DataFrame(breg_nz, columns=[trait])
df['Cluster'] = pred_cls
ax = plot_data_cls(df, trait, 'Cluster', cls_perc, palette=None, ax=None, gridsize=100, bw_adjust=10)
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, f"{trait}_enet_data_cls.png"), dpi=150, bbox_inches='tight')

ax = plot_inf_cls(pred_K, Sigma, pi, x_extreme)    
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, f"{trait}_enet_inf_cls.png"), dpi=150, bbox_inches='tight')

# bivariate clustering
traits = ["HDL", "LDL"]
# load shrinkage
for i_trait, trait in enumerate(traits):
    shrinkage_file = f"output/shrinkage/shrink_{trait}_enet.txt"
    if i_trait==0:
        breg = np.loadtxt(shrinkage_file)[:,None]
    else:
        breg = np.concatenate([breg, np.loadtxt(shrinkage_file)[:,None]], axis=1)
print(breg.shape)
# load clustering
clustering_file_prefix = "output/clustering/{}_enet".format("-".join(traits))
Sigma = np.loadtxt("{}_Sigma.txt".format(clustering_file_prefix), delimiter=',') 
n_traits = int(np.sqrt(Sigma[0].shape[0]))
print("n_traits:", n_traits)
Sigma = [s.reshape(n_traits,n_traits) for s in Sigma]
pi = np.loadtxt("{}_pi.txt".format(clustering_file_prefix), delimiter=',') 
pred_K = len(pi)
for k in range(pred_K):
    print("Cluster {}: pi={:.3f}, Sigma=\n{}".format(k, pi[k], Sigma[k]))
cls_lbs = np.loadtxt("{}_cls.txt".format(clustering_file_prefix), delimiter=',').astype(int)
meta = np.loadtxt("{}_meta.txt".format(clustering_file_prefix), delimiter=',')
zero_cutoff = float(meta[0])

# plot clustering results 
is_nz = cls_lbs>=0
breg_nz = breg[is_nz] 
pred_cls = cls_lbs[is_nz].astype(int)
cls_cnt = pd.Series(pred_cls).value_counts().reindex(np.arange(pred_K), fill_value=0)
cls_perc = (cls_cnt/cls_cnt.sum()).loc[np.arange(pred_K)]    

x_extreme = np.max(np.abs(breg))*1.05
df = pd.DataFrame(breg_nz, columns=traits)
df['Cluster'] = pred_cls
ax = plot_data_cls_bivar(df, traits, 'Cluster', cls_perc, x_extreme, palette=None, ax=None)
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, "{}_enet_data_cls.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

ax = plot_inf_cls_bivar(traits, pred_K, Sigma, pi, x_extreme, palette=None, ax=None) 
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, "{}_enet_inf_cls.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

# # plot enrichment test p-values
# enrichment_file = "output/enrichtest/HDL_gene_enrich_enet.txt"
# df_enrich = pd.read_csv(enrichment_file)
# pvals = df_enrich['P'].to_numpy()
# names = df_enrich['GENE'].to_numpy()
# rejected, pvals_corrected = fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)
# fig = plot_pvals(pvals_corrected, names=names, chr_is_odd=None, chr_switch_idx=None,
#                  sig_line=0.05, marker_s=10)
# fig.savefig(os.path.join(fig_path, "HDL_enet_enrich_.png"), dpi=150, bbox_inches='tight')


# plot gene-level multivariate values
genelevel_prefix = "output/genelevel/{}_enet".format("-".join(traits))
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
palette = plt.get_cmap('tab20').colors
for i in range(len(genes_val_list)):
    ax.scatter(np.arange(len(genes_val_list[i])), genes_val_list[i], color=palette[i], 
               edgecolor='black', alpha=0.7, s=20, lw=0.1, label=genes_val_names[i])
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.set_xlabel("Gene index")
ax.set_ylabel("Normalized $\\Sigma|$\\beta_i\\beta_j$| over variants in gene")
ax.set_title("Gene-level effect size products")
fig.savefig(os.path.join(fig_path, "{}_enet_genes_beta_prod.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

cls_frac = np.loadtxt("{}_genes_cls_frac.txt".format(genelevel_prefix), delimiter=',')
fig, ax = plt.subplots(1,1, figsize=(9,3), dpi=150, sharex=True)
# plot stacked bars
bottom = np.zeros(cls_frac.shape[0])
palette = sns.color_palette("colorblind", 10)
for i in range(cls_frac.shape[1]):
    ax.bar(np.arange(cls_frac.shape[0]), cls_frac[:,i], bottom=bottom, color=palette[i], edgecolor='black', lw=0.1, label="Cls."+str(i+1))
    bottom += cls_frac[:,i]
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax.set_xlabel("Gene index")
ax.set_ylabel("Proportion of variants in cluster")
ax.set_title("Gene-level cluster membership proportions")
fig.savefig(os.path.join(fig_path, "{}_enet_genes_cls_frac.png".format("-".join(traits))), dpi=150, bbox_inches='tight')


bin_combs = binary_combinations(n_traits)
genes_val_list = list()
genes_val_names = list()
for bin_comb in bin_combs:
    genes_val_file = "{}_genes_beta_abs_sum_({}).txt".format(genelevel_prefix,",".join(map(str,bin_comb)))
    if os.path.isfile(genes_val_file):  # file exist
        genes_val = np.loadtxt(genes_val_file, delimiter=',')
        genes_val_list.append(genes_val)
        name = "(prioritized) assoc with " + ",".join([traits[i] for i in range(n_traits) if bin_comb[i]==1])
        genes_val_names.append(name)

fig, axes = plt.subplots(len(genes_val_list), 1, figsize=(9,2.5*len(genes_val_list)), dpi=150, sharex=True)
palette = plt.get_cmap('tab10').colors
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
fig.savefig(os.path.join(fig_path, "{}_enet_genes_beta_abs_sum.png".format("-".join(traits))), dpi=150, bbox_inches='tight')

