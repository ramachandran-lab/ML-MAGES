# Simulate training data

# Chromosomes used for simulation
sim_chrs=21,22
# Directory path for the genotyped data
geno_path=../data/real
# Path for the (full) LD data
ld_path=../data/real
# Output directory path: use a new one for a different simulation, otherwise the result files will be overwritten.
output_path=../data/simulation/sim_train
mkdir -p ${output_path}
# log file
mkdir -p ../data/simulation/log
log_file=../data/simulation/log/simulate_train_chr${sim_chrs}.log

    
# Run command (Python 3.9.16 is used for implementation)
python -u simulate_train.py \
--sim_chrs $sim_chrs \
--geno_path $geno_path \
--ld_path $ld_path \
--output_path $output_path \
--n_sim 10 \
> $log_file 2>&1 