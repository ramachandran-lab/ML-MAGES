data_dir=../data
# Comma-separated list of GWA files
gwa_files=${data_dir}/gwa/gwas_MCV.csv,${data_dir}/gwa/gwas_MPV.csv,${data_dir}/gwa/gwas_PLC.csv
# Traits associated with the GWA files
traits=MCV,MPV,PLC
# Path for the LD data
ld_path=${data_dir}/block_ld
# File containing LD block IDs
ld_block_file=${ld_path}/block_brkpts.txt
# File containing gene list
gene_file=${data_dir}/genelist.csv
# Path for the trained models
model_path='../trained_models/imputed_simulated_training'
# Number of layers in the neural network, chosen from [2,3]
n_layer=2
# Number of top (highest correlation) variants used to construct the features, chosen from [15]
top_r=15
# Output directory path: use a new one for a different model, otherwise the result files will be overwritten.
output_path=../output/${traits}-model_layer${n_layer}_top${top_r}
mkdir -p ${output_path}
mkdir -p ${output_path}/log

# Run command (Python 3.9.16 is used for implementation)
python -u run_ensemble.py \
--gwa_files $gwa_files \
--traits $traits \
--ld_path $ld_path \
--ld_block_file $ld_block_file \
--gene_file $gene_file \
--model_path $model_path \
--n_layer $n_layer \
--top_r $top_r \
--output_path $output_path \
> ${output_path}/log/run_ensemble.log 2>&1 

