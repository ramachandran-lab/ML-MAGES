python -m mlmages.construct_input_from_snp_only \
--top_r 15 \
--ld_files example_files/chr22_ld_files.txt \
--sim_path output/simulation/snp_only \
--res_prefix chr22 \
--s_list -0.25 --w_list 0 \
--output_file output/simulation/snp_only_training_input/chr22 