# Number of layers of the model
n_layer=2
# Number of top variants used in features
top_r=15
# Phenotypes (separated by comma) used for matching the simulation to real data
phenotypes=HDL
# Directory path for the genotyped data
geno_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/qc
# Path for the (full) LD data
ld_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/ld
# Path for GWA results (of the phenotypes)
gwas_path=/oscar/data/sramacha/users/xliu293/multi_genee/data/gwas
# Path to the simulation data
sim_path=../data/simulation
# Prefix of the simulation label
sim_label_prefix=ninds10000_nsnps1000_nsim200_topr25
# Output directory path: use a new one for a different simulation, otherwise the result files will be overwritten.
output_base_path=../trained_models
mkdir -p ${output_base_path}
mkdir -p ${output_base_path}/log
# Chromosomes used for training and validation
train_chrs=18,19,21,22
val_chrs=20
# Model index (for labeling multiple models of the same architecture)
# model_idx=0

# Run command (Python 3.9.16 is used for implementation)
for model_idx in {2..3} # train multiple models for ensemble learning
do
    python -u train_model.py \
    --n_layer $n_layer \
    --top_r $top_r \
    --phenotypes $phenotypes \
    --geno_path $geno_path \
    --ld_path $ld_path \
    --gwas_path $gwas_path \
    --sim_path $sim_path \
    --sim_label_prefix $sim_label_prefix \
    --output_base_path $output_base_path \
    --train_chrs $train_chrs \
    --val_chrs $val_chrs \
    --model_idx $model_idx \
    > ${output_base_path}/log/train_Fc${n_layer}top${top_r}_model${model_idx}.log 2>&1 
done

