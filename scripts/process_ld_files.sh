# NOTE: filelist needs to end with "_files.txt"
# NOTE: change PATH_TO_LD to the directory where the LD files are stored

### extract_ld_blocks
chrom=22
for chrom in {10..22}
do
    echo "Processing chromosome $chrom"
    ld_file=${PATH_TO_LD}/ukb_chr${chrom}.qced.ld
    ldblock_path=/data/ld_blocks
    block_meta_file=/data/ld_blocks/block_meta_chr${chrom}.csv
    res_prefix=chr${chrom}

    python -m mlmages.extract_ld_blocks.py \
    --ld_file $ld_file \
    --ldblock_path $ldblock_path \
    --block_meta_file $block_meta_file \
    --avg_block_size 1000 \
    --res_prefix $res_prefix
done

### create ld_files.txt for later use
# for chr 22 only
x=22
y=22
ls ${ldblock_path}/chr*_block*.ld \
  | awk -v x=$x -v y=$y 'match($0,/chr([0-9]+)_block([0-9]+)\.ld$/,m) {
      chr = m[1]; block = m[2];
      if (chr >= x && chr <= y) print chr, block, $0
  }' \
  | sort -k1,1n -k2,2n \
  | awk '{print $3}' > example_files/chr22_ld_files.txt
# for multiple chroms, sorted
x=17
y=22
ls ${ldblock_path}/chr*_block*.ld \
  | awk -v x=$x -v y=$y 'match($0,/chr([0-9]+)_block([0-9]+)\.ld$/,m) {
      chr = m[1]; block = m[2];
      if (chr >= x && chr <= y) print chr, block, $0
  }' \
  | sort -k1,1n -k2,2n \
  | awk '{print $3}' > example_files/chr17-22_ld_files.txt


### compute_ld_scores (for chr 22 only)
chrom=22
ld_files=example_files/chr${chrom}_ld_files.txt
ldscore_file=data/ld_score/chr${chrom}.score
python -m mlmages.compute_ld_scores \
--ld_files $ld_files \
--ldscore_file $ldscore_file