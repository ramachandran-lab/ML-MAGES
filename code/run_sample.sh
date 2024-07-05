
# Directory path for the sample data
data_dir='../sample_data'
# Comma-separated list of GWA files
gwa_files=${data_dir}/sample_gwa_HDL.csv,${data_dir}/sample_gwa_LDL.csv
# Traits associated with the GWA files
traits=HDL,LDL
# Path for the LD data
ld_path=${data_dir}
# File containing LD block IDs
ld_block_file=${data_dir}/block_ids.txt
# File containing gene list
gene_file=${data_dir}/sample_genelist.csv
# Path for the trained models
model_path='../trained_models/genotyped_simulated_training'
# Number of layers in the neural network, chosen from [2,3]
n_layer=3
# Number of top (highest correlation) variants used to construct the features, chosen from [5,10,15]
top_r=15
# Output directory path: use a new one for a different model, otherwise the result files will be overwritten.
output_path='../sample_output'

# Run command (Python 3.9.16 is used for implementation)
python -u run_sample.py $gwa_files $traits $ld_path $ld_block_file $gene_file $model_path $n_layer $top_r $output_path
