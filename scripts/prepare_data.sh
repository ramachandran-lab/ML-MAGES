# This script processes genotype data per chromosome, performs QC, PCA, and GWAS using PLINK2 and FlashPCA.
# Requirements: plink2, flashpca

# Set directories
base_dir="/path/to/project"
plink_dir="${base_dir}/data/plink"
data_dir="${base_dir}/data"

# Under plink_dir: "ukb_chr{}.bim", "ukb_chr{}.bed", "ukb_chr{}.fam" files of genotype data from UK Biobank
# Under data_dir/pheno: "pheno.covar.wHeader": phenotype and covariate file with header (FID IID SEX AGE etc.); "phenotypes": phenotype file (FID IID PHENO1 PHENO2 PHENO3 etc.)
# Other files under data_dir: "WB.FIDIIDs": White British individuals; "removal.FIDIIDs": individuals to remove

for chrom in {1..22}; do

echo "Processing Chr$chrom ..."

# remove ambiguous SNPs
awk '!( ($5=="A" && $6=="T") ||
        ($5=="T" && $6=="A") ||
        ($5=="G" && $6=="C") ||
        ($5=="C" && $6=="G")) {print $2}' $plink_dir/ukb_chr$chrom.bim \
        > $data_dir/noambigSNPs_chr$chrom.txt

# perform QC filtering
mkdir -p $data_dir/qc
plink2 --bed $plink_dir/ukb_chr$chrom.bed \
--bim $plink_dir/ukb_chr$chrom.bim \
--fam $plink_dir/ukb_chr$chrom.fam \
--maf 0.01 --hwe 1e-6 --geno 0.01 --mind 0.05  --snps-only just-acgt \
--keep $data_dir/WB.FIDIIDs \
--remove $data_dir/removal.FIDIIDs \
--extract $data_dir/noambigSNPs_chr$chrom.txt \
--make-bed --out $data_dir/qc/ukb_chr$chrom.qced 

# prune SNPs (for PCA)
plink2 --bfile $data_dir/qc/ukb_chr$chrom.qced --indep-pairwise 100 10 0.1 \
--out $data_dir/qc/ukb_chr$chrom.pruned

plink2 --bfile $data_dir/qc/ukb_chr$chrom.qced \
--extract $data_dir/qc/ukb_chr$chrom.pruned.prune.in \
--make-bed --out $data_dir/qc/ukb_chr$chrom.qced.pruned

# calculate PCA using pruned SNPs
mkdir -p $data_dir/pca
flashpca_x86-64 --bfile $data_dir/qc/ukb_chr$chrom.qced.pruned -d 20 \
--outpc $data_dir/pca/ukb_chr$chrom.pc \
--outvec $data_dir/pca/ukb_chr$chrom.vec \
--outval $data_dir/pca/ukb_chr$chrom.val \
--outpve $data_dir/pca/ukb_chr$chrom.pve

# calculate LD
mkdir -p $data_dir/ld
plink2 --bfile $data_dir/qc/ukb_chr$chrom.qced \
--r square --out $data_dir/ld/ukb_chr$chrom.qced

# prepare PCs
awk 'NR == 1; NR > 1 {print $0 | "sort -n -k 1"}' $data_dir/pca/ukb_chr$chrom.pc \
> $data_dir/pca/ukb_chr$chrom.pc.sort
# create the covariate file
join --header -j 1 $data_dir/pheno/pheno.covar.wHeader $data_dir/pca/ukb_chr$chrom.pc.sort \
| awk '{$7=""; print $0}' | awk '{$6=""; print $0}' \
> $data_dir/covar/ukb.meta.chr$chrom.pc.covar 

# perform GWAS
mkdir -p $data_dir/gwas
plink2 --bfile $data_dir/qc/ukb_chr$chrom.qced \
--pheno $data_dir/pheno/phenotypes \
--pheno-name PHENO1,PHENO2,PHENO3 \
--covar $data_dir/covar/ukb.meta.chr$chrom.pc.covar \
--glm hide-covar --variance-standardize \
--out ${base_dir}/data/gwas/ukb_chr$chrom

done

