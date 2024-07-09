
# Directory path for the sample data
data_dir=../data
# Comma-separated list of GWA files
gwa_files=${data_dir}/gwa/gwas_HDL.csv,${data_dir}/gwa/gwas_LDL.csv
# Traits associated with the GWA files
traits=HDL,LDL
# Path for the LD data
ld_path=${data_dir}/block_ld
# File containing LD block IDs
ld_block_file=${ld_path}/block_ids.txt
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

# Run command (Python 3.9.16 is used for implementation)
python -u run_ensemble.py $gwa_files $traits $ld_path $ld_block_file $gene_file $model_path $n_layer $top_r $output_path
