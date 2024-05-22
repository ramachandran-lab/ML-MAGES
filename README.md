# ML-MAGES

This folder contains sample data and code for ML-MAGES: machine learning approaches for multivariate genetic association analyses with genes and effect size shrinkage.

Python 3.9.16 is used for implementing the method. A list of Python packages used for the implementation is included in the file `Python_packages.txt`. 

## Code Folder

The `code` folder in this repository contains the following files:

- `run_sample.sh`: This file contains the bash script with the command to run the Python code file for the sample data, including a brief description of all the input arguments. The script need to be run from the `code` folder, or input paths in the bash file need to be updated. 
- `run_sample.py`: This file contains the code to apply the method on sample data.
- `ml_mages.py`: This file includes implemented functions for each step of the method.
- `_cls_funcs_.py`: This file contains utility functions for the clustering algorithm.

## Data Folder
The `sample_data` folder in this repository contains the following files:

- `sample_gwa_HDL.txt` and `sample_gwa_LDL.txt`: These files contain the genome-wide association (GWA) results on a subset of variants on a segment of Chromosome 20 from the UK Biobank dataset for High-Density Lipoprotein (HDL) and Low-Density Lipoprotein (LDL) .
- `sample_block1.ld` and `sample_block2.ld`: These files contain the linkage disequilibrium (LD) matrix, split into two blocks, of the same subset of variants from UK Biobank.
- `block_ids.txt`: This file contains the (0-based-)indices of the boundary points at which the LD matrix correspond to the set of variants is split into blocks. 
- `sample_genelist`: This file contains the (unnamed) gene annotations of the subset of variants. Each gene is marked by the indices of the first and last variants in it. 
These files provide the necessary data for performing the ML-MAGES method described in the paper.

## Model Folder
The `trained_model` folder in this repository contains the six trained models described in the paper. We do not provide the simulated training data, but training can be performed following steps described in appendix. 

