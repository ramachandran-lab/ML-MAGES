import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlmages._util_funcs import evaluate_perf, compute_mean_prec
from mlmages._plot_funcs import plot_data_cls_bivar, plot_inf_cls_bivar

fig_path = "output/example_figures"
output_path = "output/example_simulation_performance"
os.makedirs(fig_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

### Example: evaluate simulation shrinkage results from ML-MAGES (snp-only) ###
sim_shrinkage_path = "output/simulation_shrinkage/snp_only"
model_label = "top15_2L"
sim_prefix = "chr22"
n_params = 16

# use i_param to select one set of simulation parameters to evaluate (as example)
i_param = 0
y_true_all = np.loadtxt(os.path.join(sim_shrinkage_path,"true_{}_param{}.txt".format(sim_prefix,i_param)))
y_pred_all = np.loadtxt(os.path.join(sim_shrinkage_path,"shrinkage_mlmages_{}_{}_param{}.txt".format(model_label,sim_prefix,i_param)))
print("True values shape:", y_true_all.shape)
print("Predicted values shape:", y_pred_all.shape)

# evaluate performance
perfs = evaluate_perf(y_true_all, y_pred_all)
df_perfs = pd.DataFrame(perfs)
# print(df_perfs)
df_perfs.to_csv(os.path.join(output_path,"sim_shrinkage_snp_only_{}_param{}_performance.csv".format(model_label,i_param)), index=False)

base_rec = np.linspace(0, 1, 101)
mean_precs = compute_mean_prec(y_true_all,y_pred_all,base_rec)

# plot precision-recall curve
fig, ax = plt.subplots(figsize=(4,4), dpi=200)
ax.set_aspect('equal', 'datalim')
ax.plot(base_rec[1:-1], mean_precs[1:-1], c = 'k', ls = '--',
        label = 'example: mlmages_{}_{}'.format(model_label,sim_prefix))
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=12)
fig.savefig(os.path.join(fig_path,"sim_shrinkage_snp_only_{}_param{}.png".format(model_label,i_param)), dpi=300)


### Example: evaluate simulation shrinkage results from ML-MAGES (gene-level) ###
sim_shrinkage_path = "output/simulation_shrinkage/gene_level"
sim_clustering_path = "output/simulation_clustering"
model_label = "top15_2L"
sim_prefix = "chr22"
y_true_all = list()
y_pred_all = list()
for i_sim in [0,1,2]: # three simulations available as example
    y_true = np.loadtxt(os.path.join(sim_shrinkage_path,"true_{}_sim{}.txt".format(sim_prefix,i_sim)))
    y_pred = np.loadtxt(os.path.join(sim_shrinkage_path,"shrinkage_mlmages_{}_{}_sim{}.txt".format(model_label,sim_prefix,i_sim)))
    y_true_all.append(y_true)
    y_pred_all.append(y_pred)
y_true_all = np.hstack(y_true_all)
y_pred_all = np.hstack(y_pred_all)
print("True values shape:", y_true_all.shape)
print("Predicted values shape:", y_pred_all.shape)

# evaluate performance
perfs = evaluate_perf(y_true_all, y_pred_all)
df_perfs = pd.DataFrame(perfs)
df_perfs.to_csv(os.path.join(output_path,"sim_shrinkage_gene_only_{}_sim{}_performance.csv".format(model_label,i_sim)), index=False)

base_rec = np.linspace(0, 1, 101)
mean_precs = compute_mean_prec(y_true_all,y_pred_all,base_rec)

# plot precision-recall curve
fig, ax = plt.subplots(figsize=(4,4), dpi=200)
ax.set_aspect('equal', 'datalim')
ax.plot(base_rec[1:-1], mean_precs[1:-1], c = 'k', ls = '--',
        label = 'example: mlmages_{}_{}'.format(model_label,sim_prefix))
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=12)
fig.savefig(os.path.join(fig_path,"sim_shrinkage_gene_only_{}_sim{}.png".format(model_label,i_sim)), dpi=300)

# clustering performance
i_sim = 0
clustering_file_prefix = os.path.join(sim_clustering_path,"{}_sim{}".format(sim_prefix,i_sim))
breg = np.loadtxt(os.path.join(sim_shrinkage_path,"shrinkage_mlmages_{}_{}_sim{}.txt".format(model_label,sim_prefix,i_sim)))

# load clustering
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
traits = ["Trait{}".format(i) for i in range(breg.shape[1])]
df = pd.DataFrame(breg_nz, columns=traits)
df['Cluster'] = pred_cls
ax = plot_data_cls_bivar(df, traits, 'Cluster', cls_perc, x_extreme, palette=None, ax=None)
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, "{}_data_cls.png".format("gene_level_sim")), dpi=150, bbox_inches='tight')

ax = plot_inf_cls_bivar(traits, pred_K, Sigma, pi, x_extreme, palette=None, ax=None) 
fig = ax.get_figure()
fig.savefig(os.path.join(fig_path, "{}_inf_cls.png".format("gene_level_sim")), dpi=150, bbox_inches='tight')


