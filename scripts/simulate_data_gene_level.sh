plink_file=/oscar/data/sramacha/users/xliu293/new_ukb_gwas/data/qc/ukb_chr22.qced.bed

for i in {0..2}
do
    python -m mlmages.simulate_gene_level \
    --plink_file $plink_file \
    --af_file data/af/ukb_chr22.qced.frq \
    --score_file data/ld_score/chr22.score \
    --gene_file data/genes/genes.csv \
    --chroms 22 \
    --transform_data True \
    --asymmetric False \
    --gwas_files data/gwas/gwas_MCV.csv data/gwas/gwas_MPV.csv \
    --n_trait 2 \
    --sim_path output/simulation/gene_level \
    --sim_prefix chr22 \
    --i_sim $i
done