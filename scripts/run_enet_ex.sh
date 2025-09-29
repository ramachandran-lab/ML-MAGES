# NOTE: change to mlmages.shrink_by_mlmages if applying the ML-MAGES shrinkage step-by-step (and provide model_files argument)

### shrink_by_enet (for chr 22 only)
chrom=22
for trait in HDL LDL 
do
    ld_files=example_files/chr${chrom}_ld_files.txt
    gwas_file=data/gwas/gwas_${trait}.csv
    output_file=output/shrinkage/shrink_${trait}_enet.txt
    python -m mlmages.shrink_by_enet \
    --chroms $chrom --gwas_file $gwas_file \
    --ld_files $ld_files \
    --output_file $output_file

    # univariate clustering (for chr 22 only)
    output_file=output/clustering/${trait}_enet
    python -m mlmages.cluster_shrinkage \
    --output_file $output_file \
    --shrinkage_trait1_files output/shrinkage/shrink_${trait}_enet.txt

    ### enrichment analysis (for chr 22 only)
    output_file=output/enrichment/${trait}_gene_enrich_enet.txt
    gene_file=data/genes/genes.csv
    shrinkage_file=output/shrinkage/shrink_${trait}_enet.txt
    clustering_file_prefix=output/clustering/${trait}_enet
    ld_files=example_files/full_ld_files.txt
    python -m mlmages.univar_enrich \
    --output_file $output_file \
    --gene_file $gene_file \
    --shrinkage_file $shrinkage_file \
    --clustering_file_prefix $clustering_file_prefix \
    --ld_files $ld_files \
    --chroms 22
done

# bivariate clustering (for chr 22 only)
output_file=output/clustering/HDL-LDL_enet
python -m mlmages.cluster_shrinkage \
--output_file $output_file \
--shrinkage_trait1_files output/shrinkage/shrink_HDL_enet.txt \
--shrinkage_trait2_files output/shrinkage/shrink_LDL_enet.txt \
--max_nz 0


### gene-level multivariate analysis (for chr 22 only)
output_file=output/genelevel/HDL-LDL_enet
gene_file=data/genes/genes.csv
clustering_file_prefix=output/clustering/HDL-LDL_enet
python -m mlmages.multivar_gene_analysis \
--output_file $output_file \
--gene_file $gene_file \
--shrinkage_trait1_files output/shrinkage/shrink_HDL_enet.txt \
--shrinkage_trait2_files output/shrinkage/shrink_LDL_enet.txt \
--clustering_file_prefix $clustering_file_prefix \
--chroms 22