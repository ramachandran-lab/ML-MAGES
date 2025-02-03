# A single chromosome used for simulation
chr=15
# Directory path for the genotyped data
geno_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/qc
# Path for the (full) LD data
ld_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/ld
# Path for GWA results (of the phenotypes)
gwas_path=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/gwas
# Phenotypes (separated by comma) used for matching the simulation to real data
phenotypes=LDL_direct,HDL_Cholesterol
# Path to the gene list file (with columns CHR, GENE, START, END)
gene_list_file=/oscar/data/sramacha/users/xliu293/ML-MAGES/ML-MAGES/data/genelist.csv
# Output directory path: use a new one for a different simulation, otherwise the result files will be overwritten.
output_path=../data/simulation/sim_gene_mlmt
mkdir -p ${output_path}
mkdir -p ${output_path}/log

# Run command (Python 3.9.16 is used for implementation)
python -u simulate_evaluation.py $chr $geno_path $ld_path $gwas_path $phenotypes $gene_list_file $output_path \
> ${output_path}/log/simulate_eval.log 2>&1 

