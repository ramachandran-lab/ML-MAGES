# Number of layers of the model
n_layer=2
# Number of top variants used in features
top_r=15
# Directory path for the genotyped data
geno_path=../data/real
# Path for the (full) LD data
ld_path=../data/real
# Path for GWA results used for matching the simulation to real data (separated by comma)
gwa_files="../data/gwa/gwas_MCV.csv,../data/gwa/gwas_MPV.csv"
# Path to the simulation data
sim_path=../data/simulation/sim_train
# Prefix of the simulation label
sim_label_prefix=ninds10000_nsnps1000_nsim100_topr25
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
for model_idx in {0..9} # train multiple models for ensemble learning
do
    python -u train_model.py \
    --n_layer $n_layer \
    --top_r $top_r \
    --geno_path $geno_path \
    --ld_path $ld_path \
    --gwa_files $gwa_files \
    --sim_path $sim_path \
    --sim_label_prefix $sim_label_prefix \
    --output_base_path $output_base_path \
    --train_chrs $train_chrs \
    --val_chrs $val_chrs \
    --model_idx $model_idx \
    > ${output_base_path}/log/train_Fc${n_layer}top${top_r}_model${model_idx}.log 2>&1 
done

