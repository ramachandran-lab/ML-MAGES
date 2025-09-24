# module load python/3.11.0s-ixrhc3q
# source ~/pytorch.venv/bin/activate

python -m mlmages  \
--chroms 20 21 22 \
--gwas_files data/gwas/gwas_MCV.csv data/gwas/gwas_MPV.csv data/gwas/gwas_PLC.csv \
--model_files example_files/model_files.txt \
--full_ld_files example_files/chr20-22_full_ld_files.txt \
--gene_file data/genes/genes.csv \
--trait_names MCV MPV PLC \
--split_ld True \
--ldblock_dir data/ld_blocks/ \
--avg_block_size 1000 \
--output_dir ../output