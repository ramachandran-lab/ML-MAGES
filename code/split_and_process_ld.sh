# Split LD (ex. of Chr15, Chr21, and Chr22)
for chr in 15 21 22
do
    # path to the full LD data (in comma-separated matrix format)
    full_ld_file=../data/real/ukb_chr${chr}.qced.ld
    # path to the output file to store indices of the breakpoints to split the LD
    output_file=../data/real/block_ld/chr${chr}_brkpts.txt
    # approximated average LD block size after splitting (default: 1000)
    avg_block_size=1000
    # log file
    log_file=../data/real/block_ld/split_ld_chr${chr}.log
    
    # Run command (Python 3.9.16 is used for implementation)
    python -u split_ld_blocks.py \
    --full_ld_file $full_ld_file \
    --output_file $output_file \
    --avg_block_size $avg_block_size \
    > $log_file 2>&1 
done

# Process split LD breakpoints to prepare LD blocks and meta info of splitting (ex. of Chr21&22)
# input file parameters
chrs=21,22
data_dir=../data/real
full_ld_files=${data_dir}/ukb_chr21.qced.ld,${data_dir}/ukb_chr22.qced.ld
brkpts_files=${data_dir}/block_ld/chr21_brkpts.txt,${data_dir}/block_ld/chr22_brkpts.txt
# output file parameters
ld_block_meta_file=${data_dir}/block_ld/block_meta_chr${chrs}.csv
ld_block_brkpts_file=${data_dir}/block_ld/block_brkpts_chr${chrs}.txt
ld_block_path=${data_dir}/block_ld/block_chr${chrs}.txt
# gwa files (optional, use gwa_files="" if choosing to not process GWA files at the same time.
gwa_files="${data_dir}/gwa/ukb_chr21.LDL.csv,${data_dir}/gwa/ukb_chr22.LDL.csv;${data_dir}/gwa/ukb_chr21.HDL.csv,${data_dir}/gwa/ukb_chr22.HDL.csv"
processed_gwa_files="${data_dir}/gwa/gwas_LDL_chr${chrs}.csv;${data_dir}/gwa/gwas_HDL_chr${chrs}.csv"
# gwa file columns
chr_col="#CHROM"
# log file
log_file=../data/real/block_ld/process_ld_chr${chrs}.log
    
# Run command (Python 3.9.16 is used for implementation)
python -u process_ld_blocks_and_gwa.py \
--chrs $chrs \
--full_ld_files $full_ld_files \
--brkpts_files $brkpts_files \
--ld_block_meta_file $ld_block_meta_file \
--ld_block_brkpts_file $ld_block_brkpts_file \
--ld_block_path $ld_block_path \
--gwa_files $gwa_files \
--processed_gwa_files $processed_gwa_files \
--chr_col $chr_col \
> $log_file 2>&1 