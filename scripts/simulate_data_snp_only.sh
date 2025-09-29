plink_file=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/qc/ukb_chr22.qced.bed

python -m mlmages.simulate_snp_only \
--plink_file $plink_file \
--af_file data/af/ukb_chr22.qced.frq \
--score_file data/ld_score/chr22.score \
--transform_data True \
--asymmetric False \
--gwas_files data/gwas/gwas_MCV.csv data/gwas/gwas_MPV.csv \
--sim_path output/simulation/snp_only \
--sim_prefix chr22