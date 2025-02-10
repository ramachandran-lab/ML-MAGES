# Run a single model on example data 

# Directory path for the example data
data_dir=../example_data
# Comma-separated list of GWA files
gwa_files=${data_dir}/example_gwa_HDL.csv,${data_dir}/example_gwa_LDL.csv
# Traits associated with the GWA files
traits=HDL,LDL
# Path for the LD data
ld_path=${data_dir}
# File containing LD block IDs
ld_block_file=${data_dir}/block_brkpts.txt
# File containing gene list (with columns CHR,GENE,N_SNPS,SNP_FIRST,SNP_LAST)
gene_file=${data_dir}/example_genelist.csv
# Path for the trained models
model_path=../trained_models/genotyped_simulated_training
# Number of layers in the neural network, chosen from [2,3]
n_layer=3
# Number of top (highest correlation) variants used to construct the features, chosen from [5,10,15]
top_r=15
# Output directory path: use a new one for a different model, otherwise the result files will be overwritten.
output_path=../example_output
mkdir -p $output_path

# log file
log_file=${output_path}/single_example.log
    
# Run command (Python 3.9.16 is used for implementation)
python -u run_single_example.py \
--gwa_files $gwa_files \
--traits $traits \
--ld_path $ld_path \
--ld_block_file $ld_block_file \
--gene_file $gene_file \
--model_path $model_path \
--n_layer $n_layer \
--top_r $top_r \
--output_path $output_path \
> $log_file 2>&1 