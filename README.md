# ML-MAGES
Updated: 2/11/2025

This folder contains example data and code for __*ML-MAGES*: A machine learning framework for multivariate
genetic association analyses with genes and effect size shrinkage__.


## Requirements  
The method is implemented in *Python 3.9.16*. 

While basic Python familiarity is required, users only need minimal scripting experience to run the provided workflows.  

### Dependency Management  
All required packages are listed in [`requirements.txt`](requirements.txt). For reproducibility, we **strongly recommend** creating a Python virtual environment when using this tool:  
```bash
python -m venv ml-mages-env  # Create virtual environment
source ml-mages-env/bin/activate  # Activate (Linux/Mac)
pip install -r requirements.txt  # Install dependencies
```
Alternatively,
```bash
conda create -n ml-mages-env python=3.9
conda activate ml-mages-env
pip install -r requirements.txt  # Install dependencies
```
This isolates the tool's dependencies from system-wide Python installations, avoiding potential dependency incompatibilities and version conflicts.  

## Quick Start (TODO)
* Install required Python packages if not already [`requirements.txt`](requirements.txt).
* Clone this repository to your local directory.
* The default working directory is assumed to be `ML-MAGES/code`. However, you can switch to your preferred working directory, provided you update the file paths accordingly.
* To run the method using a single pre-trained model (trained using synthetic data based on genotype data) on example data, follow the commands in `run_single_example.sh`.
* To run the method with ensemble of pre-trained models (trained using synthetic data based on imputation data) on real data (partially included), follow the commands in `run_ensemble.sh`.
* To run the method with ensemble of pre-trained models on your own data,
   * Pre-process the data to generate the 1) summary statistics and 2) LD files, as well as 3) the meta information file for genes (see [below](#data-for-running-the-full-method-with-ensembled-models-not-included) for detailed data contents).
   * Format the data as required by the input arguments for `run_ensemble.py` (see [below](#input-arguments-for-run_ensemblepy) for details).
   * Then run the command
   ```bash
   python -u run_ensemble.py \
   --gwa_files $gwa_files \
   --traits $traits \
   --ld_path $ld_path \
   --ld_block_file $ld_block_file \
   --gene_file $gene_file \
   --model_path $model_path \
   --n_layer $n_layer \
   --top_r $top_r \
   --output_path $output_path 
   ```
* To preprocess LD data (and optionally GWA results) (of each individual chromosome), follow [`split_and_process_ld.sh`](code/`split_and_process_ld.sh`).
* To train your own shrinkage models, follow [`simulate_train.sh`](code/simulate_train.sh) to generate synthetic data and [`train_model.sh`](code/train_model.sh) to train the models.
* To generate new synthetic data for performance evaluation, follow [`simulate_evaluation.sh`](code/simulate_evaluation.sh).
* [`demo_visualize_outputs.ipynb`](demo_visualize_outputs.ipynb) provides an example of visualizing multi-trait analysis results, but users are free to explore any possible downstream using the results.
* [`demo_evaluate_perf.ipynb`](demo_evaluate_perf.ipynb) provides an example for performance evaluation using the synthetic data, but users are free to explore any other evaluation metrics. 


## Repository Structure  

```text
ML-MAGES/
├── code/                   
│   ├── ml_mages.py              # Core ML-MAGES algorithm implementation
│   ├── _cls_funcs.py            # Clustering utilities
│   ├── _train_funcs.py          # Model training components
│   ├── _enrich_funcs.py         # Enrichment analysis alternatives
│   ├── _sim_funcs.py            # Synthetic data generation
│   ├── run_single_example.sh    # Single model on example data demo
│   ├── run_ensemble.sh          # Ensemble models on real data demo
│   ├── train_model.sh           # Model training script
│   └── evaluate_perf.ipynb      # Performance evaluation notebook
│
├── example_data/                
│   ├── example_gwa_HDL.txt      # Example GWAS results (HDL)
│   ├── example_gwa_LDL.txt      # Example GWAS results (LDL)
│   ├── example_block*.ld        # LD matrices (blocks 1-2)
│   └── block_brkpts.txt         # LD block boundaries (only the right boundary for each block)
│
├── trained_model/          
│   ├── genotyped_models/        # Models trained using synthetic data based on genotype data
│   └── imputed_models/          # Models trained using synthetic data based on imputation data
│
├── example_output/              # Outputs from run_single_example.sh
│
├── data/                        # (Part of this folder's contents are left out for file size restriction)
│   ├── block_ld/                # LD block matrices
│   ├── gwa/                     # GWAS files (gwas_TRAIT.csv)
│   └── genelist.csv             # Gene metadata
│
└── output/                      # (Empty by default: analysis results directory)
```

### Key Components  
| File/Directory             | Purpose                                       |
|----------------------------|-----------------------------------------------|
| `run_single_example.sh`    | Demo pipeline for running a single model      |
| `run_ensemble.sh`          | Demo pipeline for running ensemble models     |
| `example_data/`            | Contains data used for `run_single_example.sh`|
| `trained_model/`           | Pre-trained effect shrinkage models           |

### Input Requirements  
1. **LD Blocks** (`data/block_ld/`):  
   - Files: `block_0.ld`, `block_1.ld`, etc.  
   - Metadata: `block_ids.txt` (required), `blocks_meta.csv` (optional)  

2. **GWAS Data** (`data/gwa/`):  
   - Format: `gwas_[TRAIT].csv` with `BETA`, `SE`, `CHR` columns  

3. **Gene Metadata** (`data/genelist.csv`):  
   - Required columns: `CHR`, `GENE`, `START`, `END`, `SNP_FIRST`, `SNP_LAST`  

---


## Code Folder

The `code` folder in this repository contains the following files:

**Main functions:**
- `ml_mages.py`: This file includes implemented functions for each step of the method.
- `_cls_funcs.py`: This file contains utility functions for the clustering algorithm.
- `_train_funcs.py`: This file contains functions for training the models.
- `_enrich_funcs.py`: This file contains functions to replace the default enrichment test function if the installation of the required package *chiscore* fails.
- `_sim_funcs.py`: This file contains functions for simulating data for performance evaluation as well as evaluating the performance using the simulated data.

**Run the method using a single trained model for effect size shrinkage on sample data:**
- `run_sample.sh`: This file contains the bash script with the command to run the Python code file for the sample data, including a brief description of all the input arguments. The script need to be run from the `code` folder, or input paths in the bash file need to be updated. 
- `run_sample.py`: This file contains the code to apply the method on sample data.

**Run the full method using ensemble learning on multiple trained models for effect size shrinkage on genotype data (not included):**  
- `run_ensemble.sh`: This file contains the bash script with the command to run the Python code file for the full genotyped data (not included) using ensembled models, including a brief description of all the input arguments. The script need to be run from the `code` folder, or input paths in the bash file need to be updated. 
- `run_ensemble.py`: This file contains the code to apply the method on the full genotyped data (not included).

**Additional files for simulating training data and training models from scratch:**  
- `train_model.sh`: This file contains the bash script with the command to run the Python code file for training the models for effect size shrinkage based on the full genotyped data (not included).
- `train_model.py`: This file contains the code to train the models for effect size shrinkage based on the full genotyped data (not included).
- `simulate_train.sh`: This file contains the bash script with the command to run the Python code file for simulating effects for pseudo-traits based on real genotyped data and LD (not included), which are used for training the models.
- `simulate_train.py`: This file contains the code to simulate effects for pseudo-traits based on real genotyped data and LD (not included).

  **Additional files for simulating evluation data:**  
- `simulate_evaluation.sh`: This file contains the bash script with the command to run the Python code file for simulating synthetic traits based on real genotyped data and LD (not included), which are used for evaluting the performances. 
- `simulate_evaluation.py`: This file contains the code to simulate synthetic traits for performance evluation based on real genotyped data and LD (not included). Multiple traits with various association relationships can be simulated.
  
**Additional files for visualizing results and comparing performances:** 
- `demo_visualize_outputs.ipynb`: This Jupyter notebook demonstrates how to visualize and analyze gene-level output for multi-trait analyses.
- `evaluate_perf.ipynb`: This Jupyter notebook demonstrates how to evaluate the performances of the methods using the simulated data.


## Data Folder
### Sample data
The `sample_data` folder in this repository contains the following files:

- `sample_gwa_HDL.txt` and `sample_gwa_LDL.txt`: These files contain the genome-wide association (GWA) results on a subset of variants on a segment of Chromosome 20 from the UK Biobank dataset for High-Density Lipoprotein (HDL) and Low-Density Lipoprotein (LDL) .
- `sample_block1.ld` and `sample_block2.ld`: These files contain the linkage disequilibrium (LD) matrix, split into two blocks, of the same subset of variants from UK Biobank.
- `block_ids.txt`: This file contains the (0-based-)indices of the boundary points at which the LD matrix correspond to the set of variants is split into blocks. 
- `sample_genelist`: This file contains the (unnamed) gene annotations of the subset of variants. Each gene is marked by the indices of the first and last variants in it. 
These files provide the necessary data for performing the ML-MAGES method described in the paper.

### Data for running the full method with ensembled models (not included)
Input data files to `run_ensemble.py` are not included due to large file sizes. The following files (as used in `run_ensemble.sh`) are needed:

- The `data/block_ld` folder contains LD data. Suppose there are a total of x LD blocks, then this folder should contain x+1 files, optionally with additional files for reference. 
  - The full LD of split segments along all the chromosomes, ordered by chromosome and position, are saved in x files labeled as `block_0.ld`, `block_1.ld`, to `block_x.ld`, each being a comma-delimited matrix. 
  - The file `block_ids.txt` contains x lines, where each line is the index of the last variant in the corresponding block **plus one**. Indices go from 0 to M-1, where M is the total number of variants along all the chromosomes. For instance, if `block_0.ld` is of size 200x200 and `block_1.ld` is of size 210x210, then the first two lines in `block_ids.txt` should be 200 and 410.
  - [Optional] The file `blocks_meta.csv` is a comma-delimited file with three columns: 'block_id', 'chr', 'id_in_chr', and x rows (excluding the header).  The three columns correspond to the block index (as used in the `.ld` file names), the chromosome to which the block belongs, and the index of the block within that chromosome. For instance, a row of `405,15,0` means the `block_405.ld` is the first (indexed by 0) block in CHR15. 
- The `data/gwa` folder contains GWA result files, labeled as `gwas_TRAIT.csv` where `TRAIT` is the trait name and should be the same as the one used for argument `traits` in the input to `run_ensemble.py`. The file for each trait should have exactly M lines excluding the header, with each line corresponds to a variant, and all variants ordered the same as those in LD blocks. There should be at least three columns: 'BETA','SE', and 'CHR'.
  - 'BETA': the estimated GWA effect of the variant
  - 'SE': the standard error of the estimated effect
  - 'CHR': chromosome of the variant 
- `data/genelist.csv` is a comma-delimited file containing gene information used for gene-level analyses. It has 7 columns:
  - 'CHR': chromosome
  - 'GENE': gene symbol
  - 'START': position (in bp) marking the start of the region considered for this gene
  - 'END': position (in bp) marking the end of the region considered for this gene
  - 'N_SNPS': number of variants (out of the total M variants) contained in the marked region for this gene
  - 'SNP_FIRST': index of the first variant considered for the gene
  - 'SNP_LAST': index of the last variant considered for the gene

The following files, contained in the folder `data/real_for_sim` (not included), are used in `simulate_evaluation.sh` and `evaluate_perf.ipynb`:
- `ukb_chr15.qced.bim`, `ukb_chr15.qced.bed`, `ukb_chr15.qced.fam`: the genotype data of Chr15 of UKB European individuals in PLINK format.
- `ukb_chr15.qced.ld`: full LD matrix of Chr15 in UKB European individuals.
- `blocks_chr15_ws1000.txt`: The (0-based-)indices of the boundary points at which the LD matrix of Chr15 is split into blocks.
- 
The simulated training data will be included in the folder `data/simulation` (not shown), and subsequently used for model training.

The simulated evluation data will be included in the folder `data/simulation/sim_gene_mlmt` (not shown), and subsequently used for performance evaluation.

## Model Folder
The `trained_model` folder in this repository contains trained models. 

The subfolder `genotyped_simulated_training` contains the six models, each of a different architecture, trained using genotyped-data-based simulation described in the paper. We do not provide the simulated training data, but simulation and training can be performed following steps described in appendix. 

The subfolder `imputed_simulated_training` contains two set of models, each with 10 models of a same architecture, trained using imputed-data-based simulation. The output of each set of models are averaged to generate an ensemble result of shrinkage, as used in the `run_ensemble.py`. Similarly, simulation and training can be performed following steps described in appendix. 
  * The model files are named as ''Fc*a*top*b*_*c*.model'', where *a* is the number of fully-connected layers in the neural network model, *b* is the number of top variants used to construct the features, and *c* is the index of the model among all models of the same architecture.

## Input arguments for `run_ensemble.py`

The main function `run_ensemble.py` takes in 9 input arguments. These arguments are required for running the script and performing *ML-MAGES* on the given data:

- `--gwa_files`: A comma-separated list of GWA files. These files contain the GWA results for different traits. E.g., `../data/gwa/gwas_MCV.csv,../data/gwa/gwas_MPV.csv,../data/gwa/gwas_PLC.csv`.

- `--traits`: A comma-separated list of traits associated with the GWA files. Each trait corresponds to a GWA file. E.g., `MCV,MPV,PLC`.

- `--ld_path`: The path to the LD (block) files. These files contain the linkage disequilibrium matrix split into blocks. E.g., `../data/block_ld`.

- `--ld_block_file`: The file containing the LD block IDs. This file specifies the indices of the boundary points at which the LD matrix is split into blocks. E.g., `../data/block_ld/block_ids.txt`.

- `--gene_file`: The file containing gene data. This file includes gene annotations for the subset of variants. E.g., `../data/genelist.csv`.

- `--model_path`: The path to the trained models. This folder contains (multiple) trained models used for the ensemble learning by *ML-MAGES*. E.g., `../trained_models/imputed_simulated_training`.

- `--n_layer`: The number of layers in the model. This argument should be chosen from either 2 or 3 for the provided trained models. User may choose other number if providing self-trained models.

- `--top_r`: The number of top variants used to construct the features. This argument should be chosen as 15 for the provided trained models. User may choose other number if providing self-trained models.

- `--output_path`: The path to save the output files. All output files generated by the *ML-MAGES* will be saved in this directory.

## Output of `run_ensemble.py`

The script `run_ensemble.py` outputs several files to the `output_path` folder specified. Suppose two traits are analyzed, X and Y. The output files in this folder include:

1. Shrinkage results:
   * `regularized_effects_X.txt`
   * `regularized_effects_Y.txt`
   
   Each line corresponds to the regularized effect of one variant, for a total of M variants.
2. Visualization of shrinkage results:
   * `shrinkage_X.png`
   * `shrinkage_Y.png`
   
   The plots show the effects before and after shrinkage for the M variants along the genome.
3. Clustering results:
   * `univar_X_cls.txt`, `univar_X_pi.txt`, `univar_X_Sigma.txt`, `univar_X_zc.txt`
   * `univar_Y_cls.txt`, `univar_Y_pi.txt`, `univar_Y_Sigma.txt`, `univar_Y_zc.txt`
   * `multivar_X-Y_cls.txt`, `multivar_X-Y_pi.txt`, `multivar_X-Y_Sigma.txt`, `multivar_X-Y_zc.txt`
   
   The `*_cls.txt` file contains the cluster label for each variant, where a label of -1 denotes the nearly-zero effect that is not considered in the clustering, and the cluster labels start from index 0. 
   The `*_pi.txt` file contains the mixing coefficient $\pi$ of the clusters.
   The `*_Sigma.txt` file contains the covariance matrices of clusters. Each line records the covariance of one cluster, and if it's multivariate, the matrix is flattened in row-major order with entries separated by comma.
   The `*_zc.txt` file contains a single value used as the zero-cutoff for the regularized effect values for determining which variants to be included in the clustering: Only variants with effects greater than this value are considered. This value is dynamically determined by the total number of variants so that a reasonable proportion of them are ''non-zero''.
4. Visualization of clustering results: 
   * `clustering_multivar_X-Y.png`
   * `clustering_univar_X.png`
   * `clustering_univar_Y.png`
   
   Variants are colored by clusters, and Gaussians inferred for each cluster are shown along the side in the same colors (for up to 2D Gaussian). 
5. Enrichment results:
   * `enrichment_X.csv`
   * `enrichment_Y.csv`
   
   Each result file has the same rows as in input  `gene_file`, with 3 additional columns, 'P', 'STAT', and 'VAR', corresponding to the p-value, test statistics, and variance of test statistics of the gene-level enrichment test. If the dependency package *chiscore* can not be installed successfully, that is, gene-level test is not available, then p-values will all be set to 1 by default, and a warning message *"Unable to import chiscore. Please install chiscore package separately"* will pop up when running the program. 
   * `multivar_gene_X-Y.csv`
   
   The multivariate gene-level result file contains the same rows as in input  `gene_file`, with a couple additional columns:
     * 'size' is the same as 'N_SNPS', denoting the number of variants considered for the gene.
     * 'cls1_frac', ..., 'cls*K*_frac' (*K* columns): the fraction of variants in each gene that belong to each cluster, from cluster 1 to cluster *K*.
     * 'b1b1', 'b1b2', 'b2b2', etc. (*K(K+1)/2* columns): sum of the product of regularized effects for each pair of traits (including a trait to itself) for all variants in the gene, divided by the gene size. 
   
6. Visualization of enrichment results:
   * `enrichment_X.png`
   * `enrichment_Y.png` 
   
   The manhattan plot shows negative log of adjusted p-values for each gene along the genome.

Users may generate other visualizations or perform downstream analyses using the result provided in these files. 

----
**Citation**

Liu X, Crawford L, Ramachandran S. ML-MAGES: Machine learning approaches for multivariate genetic association analyses with genes and effect size shrinkage. (accepted at RECOMB 2025)

For questions and comments, please contact Xiran Liu at *xiran_liu1 at brown dot edu*.
