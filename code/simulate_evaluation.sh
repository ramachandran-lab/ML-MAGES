# A single chromosome used for simulation
chr=15
data_dir=../data/real
# Directory path for the genotyped data (in PLINK format)
geno_file=${data_dir}/ukb_chr${chr}.qced
# Path for the (full) LD data file
full_ld_file=${data_dir}/ukb_chr${chr}.qced.ld
# Comma-separated list of GWA files used for matching the simulation to real data
gwa_files=${data_dir}/gwa/ukb_chr${chr}.LDL.csv,${data_dir}/gwa/ukb_chr${chr}.HDL.csv
# Path to the gene list file (with columns CHR, GENE, START, END)
gene_list_file=../data/genelist.csv
# Output directory path: use a new one for a different simulation, otherwise the result files will be overwritten.
output_path=../data/simulation/sim_gene_mlmt
mkdir -p ${output_path}
mkdir -p ${output_path}/log
# Number of individuals to be sampled
n_inds=10000
# Min number of variants in a gene for gene to be considered as candidate
min_gene_size=10
# Number of traits and causal types to be simulated
n_traits=3
causal_types="1;2;3;1,2;1,2,3"
# Number of simulations
n_sim=10
# log file
log_file=../data/simulation/sim_gene_mlmt/log/simulate_eval.log

# Run command (Python 3.9.16 is used for implementation)
python -u simulate_evaluation.py \
--chr $chr \
--geno_file $geno_file \
--full_ld_file $full_ld_file \
--gwa_files $gwa_files \
--gene_list_file $gene_list_file \
--output_path $output_path \
--n_inds $n_inds \
--min_gene_size $min_gene_size \
--n_traits $n_traits \
--causal_types $causal_types \
--n_sim $n_sim \
> $log_file 2>&1 