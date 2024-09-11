# ML-MAGES

This folder contains sample data and code for ML-MAGES: machine learning approaches for multivariate genetic association analyses with genes and effect size shrinkage.

Python 3.9.16 is used for implementing the method. A list of Python packages used for the implementation is included in the file `Python_packages.txt`. 

*The folders `data` and `output` for running the method on genotyped data are not included in the repository. Data is available upon request.*

## Code Folder

The `code` folder in this repository contains the following files:

- `run_sample.sh`: This file contains the bash script with the command to run the Python code file for the sample data, including a brief description of all the input arguments. The script need to be run from the `code` folder, or input paths in the bash file need to be updated. 
- `run_sample.py`: This file contains the code to apply the method on sample data.
- `ml_mages.py`: This file includes implemented functions for each step of the method.
- `_cls_funcs_.py`: This file contains utility functions for the clustering algorithm.
- `run_ensemble.sh`: This file contains the bash script with the command to run the Python code file for the full genotyped data (not included) using ensembled models, including a brief description of all the input arguments. The script need to be run from the `code` folder, or input paths in the bash file need to be updated. 
- `run_ensemble.py`: This file contains the code to apply the method on the full genotyped data (not included).

## Data Folder
The `sample_data` folder in this repository contains the following files:

- `sample_gwa_HDL.txt` and `sample_gwa_LDL.txt`: These files contain the genome-wide association (GWA) results on a subset of variants on a segment of Chromosome 20 from the UK Biobank dataset for High-Density Lipoprotein (HDL) and Low-Density Lipoprotein (LDL) .
- `sample_block1.ld` and `sample_block2.ld`: These files contain the linkage disequilibrium (LD) matrix, split into two blocks, of the same subset of variants from UK Biobank.
- `block_ids.txt`: This file contains the (0-based-)indices of the boundary points at which the LD matrix correspond to the set of variants is split into blocks. 
- `sample_genelist`: This file contains the (unnamed) gene annotations of the subset of variants. Each gene is marked by the indices of the first and last variants in it. 
These files provide the necessary data for performing the ML-MAGES method described in the paper.

Input data files to `run_ensemble.py` are not included due to large file sizes. The following files (referenced in `run_ensemble.sh`) are needed:

- The `block_ld` folder contains LD data. Suppose there are a total of x LD blocks, then this folder should contain x+1 files, optionally with additional files for reference. 
  - The full LD of splitted segments along all the chromosomes, ordered by chromosome and position, are saved in x files labeled as `block_0.ld`, `block_1.ld`, to `block_x.ld`, each being a comma-delimited matrix. 
  - The file `block_ids.txt` contains x lines, where each line is the index of the last variant in the corresponding block **plus one**. Indices go from 0 to M-1, where M is the total number of variants along all the chromosomes. For instance, if `block_0.ld` is of size 200x200 and `block_1.ld` is of size 210x210, then the first two lines in `block_ids.txt` should be 200 and 410.
  - [Optional] The file `blocks_meta.csv` is a comma-delimited file with three columns: 'block_id', 'chr', 'id_in_chr', and x rows (excluding the header).  The three columns correspond to the block index (as used in the `.ld` file names), the chromosome to which the block belongs, and the index of the block within that chromosome. For instance, a row of `405,15,0` means the `block_405.ld` is the first (indexed by 0) block in CHR15. 
- The `gwa` folder contains GWA result files, labeled as `gwas_TRAIT.csv` where `TRAIT` is the trait name and should be the same as the one used for argument `traits` in the input to `run_ensemble.py`. The file for each trait should have exactly M lines excluding the header, with each line corresponds to a variant, and all variants ordered the same as those in LD blocks. There should be at least three columns: 'BETA','SE', and 'CHR'. 'BETA' is the estimated GWA effect of the variant; 'SE' is the standard error of the estimated effect; 'CHR' is the chromosome of the variant. 

## Model Folder
The `trained_model` folder in this repository contains trained models. 

The subfolder `genotyped_simulated_training` contains the six models, each of a different architecture, trained using genotyped-data-based simulation described in the paper. We do not provide the simulated training data, but simulation and training can be performed following steps described in appendix. 

The subfolder `imputed_simulated_training` contains two set of models, each with 10 models of a same architecture, trained using imputed-data-based simulation. The output of each set of models are averaged to generate an ensemble result of shrinkage, as used in the `run_ensemble.py`. Similarly, simulation and training can be performed following steps described in appendix. 

