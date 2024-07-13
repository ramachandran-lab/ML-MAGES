
# Chromosomes used for simulation
sim_chrs=18,19,20
# Directory path for the genotyped data
geno_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/qc
# Path for the (full) LD data
ld_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/ld
# Output directory path: use a new one for a different simulation, otherwise the result files will be overwritten.
output_path=../data/simulation
mkdir -p ${output_path}
mkdir -p ${output_path}/log

# Run command (Python 3.9.16 is used for implementation)
python -u simulate_train.py $sim_chrs $geno_path $ld_path $output_path \
> ${output_path}/log/simulate_train_chr${sim_chrs}.log 2>&1 

