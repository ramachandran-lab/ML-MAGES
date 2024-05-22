
data_dir='../sample_data'  # Directory path for the sample data
gwa_files=${data_dir}/sample_gwa_HDL.csv,${data_dir}/sample_gwa_LDL.csv  # Comma-separated list of GWA files
traits=HDL,LDL  # Traits associated with the GWA files
ld_path=${data_dir}  # Path for the LD (Linkage Disequilibrium) data
ld_block_file=${data_dir}/block_ids.txt  # File containing LD block IDs
gene_file=${data_dir}/sample_genelist.csv  # File containing gene list
model_path='../trained_models'  # Path for the trained models
n_layer=3  # Number of layers in the neural network
top_r=15  # Number of top (highest correlation) variants used to construct the features
output_path='../output'  # Output directory path

python -u run_sample.py $gwa_files $traits $ld_path $ld_block_file $gene_file $model_path $n_layer $top_r $output_path
