for i in {0..2}
do
    python -m mlmages.shrink_sim_gene_level_by_mlmages \
    --model_files example_files/chr22_model_files.txt \
    --ld_files example_files/chr22_ld_files.txt \
    --sim_path output/simulation/gene_level \
    --sim_prefix chr22 \
    --shrinkage_path output/simulation_shrinkage/gene_level \
    --i_sim $i 
done

shrinkage_path=output/simulation_shrinkage/gene_level
clustering_path=output/simulation_clustering
sim_prefix=chr22
model_label=top15_2L
i_sim=0
shrinkage_file1=${shrinkage_path}/shrinkage_mlmages_${model_label}_${sim_prefix}_sim${i_sim}_trait0.txt
shrinkage_file2=${shrinkage_path}/shrinkage_mlmages_${model_label}_${sim_prefix}_sim${i_sim}_trait1.txt
python -m mlmages.cluster_shrinkage \
    --shrinkage_trait1_files $shrinkage_file1 \
    --shrinkage_trait2_files $shrinkage_file2 \
    --output_file ${clustering_path}/${sim_prefix}_sim${i_sim}