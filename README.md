# ML-MAGES
Last Updated: 2/11/2025

This folder contains example data and code for __*ML-MAGES*: A machine learning framework for multivariate
genetic association analyses with genes and effect size shrinkage__ [1]


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
Alternatively, if using Conda,
```bash
conda create -n ml-mages-env python=3.9
conda activate ml-mages-env
pip install -r requirements.txt  # Install dependencies
```
This isolates the tool's dependencies from system-wide Python installations, avoiding potential dependency incompatibilities and version conflicts.  

## Quick Start 
* Install required Python packages if not already (see [`requirements.txt`](requirements.txt)).
* Clone this repository to your local directory.
  ```bash
  git clone https://github.com/ramachandran-lab/ML-MAGES.git
  ```
* The default working directory is assumed to be `ML-MAGES/code`. However, you can switch to your preferred working directory, provided that you update all the file paths accordingly.
* To run the method using a single pre-trained model (trained using synthetic data based on genotype data) on example data, follow the commands in [`run_single_example.sh`](code/run_single_example.sh). After defining all input arguments, run
  ```bash
   python -u single_example.py --gwa_files $gwa_files --traits $traits --ld_path $ld_path  --ld_block_file $ld_block_file --gene_file $gene_file --model_path $model_path  --n_layer $n_layer  --top_r $top_r --output_path $output_path
  ```
* **[Main]** To run the method with ensemble of pre-trained models (trained using synthetic data based on imputation data) on real data (partially included) or on your own data, follow the commands in [`run_ml_mages.sh`](code/run_ml_mages.sh).
   * Pre-process the data to generate the 1) summary statistics and 2) LD files, as well as 3) the gene annotation metadata file (see [below](#contents-of-data-folder) for detailed data contents).
   * Format the data as required by the input arguments for `ml_mages.py` (see [below](#detailed-input-requirements) for details).
   * Then run the command
   ```bash
   python -u ml_mages.py \
   --gwa_files $gwa_files --traits $traits \
   --ld_path $ld_path --ld_block_file $ld_block_file \
   --gene_file $gene_file \
   --model_path $model_path --n_layer $n_layer --top_r $top_r \
   --output_path $output_path 
   ```
   * Details of output files can be found [here](#detailed-output-information).
* To preprocess LD data (and optionally GWA results) (of each individual chromosome), follow [`split_and_process_ld.sh`](code/`split_and_process_ld.sh`).
* To train your own shrinkage models, follow [`simulate_train.sh`](code/simulate_train.sh) to generate synthetic data and [`train_model.sh`](code/train_model.sh) to train the models.
* To generate new synthetic data for performance evaluation, follow [`simulate_evaluation.sh`](code/simulate_evaluation.sh).
* [`demo_vis_outputs.ipynb`](demo_vis_outputs.ipynb) provides an example of visualizing multi-trait analysis results, but users are free to explore any possible downstream using the results.
* [`demo_eval_perf.ipynb`](demo_eval_perf.ipynb) provides an example for performance evaluation using the synthetic data, but users are free to explore any other evaluation metrics. 

The structure of the rest of this manual goes as follows:
* [Contents of the Repository](#contents-of-the-repository)
* [Usage of Functions](#usage-of-functions)
* [Data](#data)
* [Models](#models)
  

## Contents of the Repository

```text
ML-MAGES/
├── code/                   
│   ├── _main_funcs.py           # Core ML-MAGES functions
│   ├── _cls_funcs.py            # Utility functions for clustering
│   ├── _train_funcs.py          # Utility functions for model training 
│   ├── _enrich_funcs.py         # Utility function for enrichment analysis alternatives, specifically to replace the default enrichment test if installation of the required package 'chiscore' fails.
│   ├── _sim_funcs.py            # Utility functions for synthetic data generation and performance evaluation
│   ├── single_example.sh        # Apply a single model on example data (demo)
│   ├── ml_mages.sh              # [Main] Apply ensemble models on real data 
│   ├── split_and_process_ld.sh  # LD splitting and pre-processing script
│   ├── train_model.sh           # Model training script
│   ├── simulate_train.sh        # Synthetic data generation (for training) script
│   ├── train_model.sh           # Model training script
│   ├── simulate_evaluation.sh   # Synthetic data generation (for evaluation) script
│   ├── demo_vis_outputs.ipynb   # Result visualization notebook
│   └── demo_eval_perf.ipynb     # Performance evaluation notebook
│
├── example_data/                # Input for run_single_example.sh
│   ├── example_gwa_HDL.txt      # Example GWAS results (HDL)
│   ├── example_gwa_LDL.txt      # Example GWAS results (LDL)
│   ├── example_block*.ld        # LD matrices (blocks 1-2)
│   └── block_brkpts.txt         # LD block boundaries (only the right boundary for each block)
│
├── trained_model/          
│   ├── genotyped_models/        # Models trained using synthetic data based on genotype data
│   └── imputed_models/          # Models trained using synthetic data based on imputation data
│
├── example_output/              # Outputs of run_single_example.sh
│
├── data/                        # (Selected files in this folder have been omitted to restrict size: Input for run_ml_mages.sh)
│   ├── block_ld/                # LD block matrices
│   ├── gwa/                     # GWAS files (gwas_TRAIT.csv)
│   └── genelist.csv             # Gene metadata
│
└── output/                      # (Empty by default: Output of run_ml_mages.sh)
```

---

## Usage of Functions
### `single_example.py`

  - Run the core ML-MAGES workflow for a single model on example data.  
  
  - **Usage**:  
  ```bash
   python -u single_example.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --gwa_files      Path to GWAS summary statistics files (CSV format), multiple traits separated by comma
    --traits         Names of traits corresponding to gwa_files, separated by comma
    --ld_path        Directory containing LD matrix blocks (block0.ld, block1.ld, etc.)
    --ld_block_file  Path to LD block boundary indices file
    --gene_file      Path to gene annotation metadata file (CSV format)
    --model_path     Directory containing pre-trained shrinkage models
    --n_layer        Model architecture configuration (chosen from {2,3})
    --top_r          Number of top variants for feature construction (choices: {5,10,15})
    --output_path    Path to save analysis results 
    ```
  
  - **Outputs**:  
    ```text
    output_path
    ├── regularized_effects_TRAIT.txt,                                    # shrinkage results for each trait
    ├── univar_TRAIT_cls.txt, *_pi.txt, *_Sigma.txt, *_zc.txt             # univariate clustering results for each trait
    ├── enrichment_TRAIT.csv                                              # univariate gene enrichment results for each trait
    ├── multivar_TRAIT1-TRAIT2_cls.txt, *_pi.txt, *_Sigma.txt, *_zc.txt   # multivariate clustering results for traits 1 and 2
    ├── bivar_gene_TRAIT1-TRAIT2.csv                                      # bivariate gene anlaysis results for traits 1 and 2
    └── clustering_*.png                                                  # visualization of clustering results (up to bivariate)
    ```
  
  - **Command** (from `run_single_example.sh`):  
    ```bash
    python -u single_example.py \
    --gwa_files $gwa_files --traits $traits \
    --ld_path $ld_path  --ld_block_file $ld_block_file \
    --gene_file $gene_file \
    --model_path $model_path  --n_layer $n_layer  --top_r $top_r \
    --output_path $output_path
    ```

### `ml_mages.py`

  - Run the core ML-MAGES workflow for ensemble models on real data. 
  
  - **Usage**:  
  ```bash
  python -u ml_mages.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --gwa_files      Path to GWAS summary statistics files (CSV format), multiple traits separated by comma
    --traits         Names of traits corresponding to gwa_files, separated by comma
    --ld_path        Directory containing LD matrix blocks (block0.ld, block1.ld, etc.)
    --ld_block_file  Path to LD block boundary indices file
    --gene_file      Path to gene annotation metadata file (CSV format)
    --model_path     Directory containing pre-trained shrinkage models
    --n_layer        Model architecture configuration (chosen from {2,3})
    --top_r          Number of top variants for feature construction (choices: {5,10,15})
    --output_path    Path to save analysis results 
    ```

  - **Outputs**:  
    ```text
    output_path
    ├── regularized_effects_TRAIT.txt,                                    # shrinkage results for each trait
    ├── univar_TRAIT_cls.txt, *_pi.txt, *_Sigma.txt, *_zc.txt             # univariate clustering results for each trait
    ├── enrichment_TRAIT.csv                                              # univariate gene enrichment results for each trait
    ├── multivar_TRAIT1-TRAIT2_cls.txt, *_pi.txt, *_Sigma.txt, *_zc.txt   # multivariate clustering results for traits 1 and 2
    ├── bivar_gene_TRAIT1-TRAIT2.csv                                      # bivariate gene anlaysis results for traits 1 and 2
    └── clustering_*.png                                                  # visualization of clustering results (up to bivariate)
    ```

  - **Command** (from `run_ml_mages.sh`):  
    ```bash
    python -u ml_mages.py \
    --gwa_files $gwa_files --traits $traits \
    --ld_path $ld_path --ld_block_file $ld_block_file \
    --gene_file $gene_file \
    --model_path $model_path --n_layer $n_layer --top_r $top_r \
    --output_path $output_path 
    ```
#### Detailed Input Requirements  
  1. **LD Data** (`data/block_ld/`):  
     - Files: `block0.ld`, `block1.ld`, etc.  
     - Metadata: `block_brkpts.txt` (required, indices of right boundaries of the split LD blocks, with one index on each line), `blocks_meta.csv` (optional)
     - ***If not splitting LD***: use a single LD "block", and a breakpoint file with a single value denoting the number of variants (should be the same as the length of LD).
  
  2. **GWAS Data** (`data/gwa/`):  
     - Files: `gwas_[TRAIT].csv` with columns `BETA` (GWA effect) and `SE` (standard error); other columns are optional  
  
  3. **Gene Annotation** (`data/genelist.csv`):  
     - CSV file with required columns: 
       `CHR`, `GENE`, `START`, `END`, `SNP_FIRST` (index of the first variant considered in this gene), and `SNP_LAST`  (index of the last variant)  

#### Detailed Output Information

  1. Shrinkage results:
     `regularized_effects_X.txt`, `regularized_effects_Y.txt`
     - Each line corresponds to the regularized effect of one variant, for a total of M variants.
  2. Visualization of shrinkage results:
     `shrinkage_X.png`, `shrinkage_Y.png`
     - The plots show the effects before and after shrinkage for the M variants along the genome.
  3. Clustering results:
     `univar_X_cls.txt`, `univar_X_pi.txt`, `univar_X_Sigma.txt`, `univar_X_zc.txt`; `univar_Y_cls.txt`, `univar_Y_pi.txt`, `univar_Y_Sigma.txt`, `univar_Y_zc.txt`; `multivar_X-Y_cls.txt`, `multivar_X-Y_pi.txt`, `multivar_X-Y_Sigma.txt`, `multivar_X-Y_zc.txt`
     - The `*_cls.txt` file contains the cluster label for each variant, where a label of -1 denotes the nearly-zero effect that is not considered in the clustering, and the cluster labels start from index 0. 
     - The `*_pi.txt` file contains the mixing coefficient $\pi$ of the clusters.
     - The `*_Sigma.txt` file contains the covariance matrices of clusters. Each line records the covariance of one cluster, and if it's multivariate, the matrix is flattened in row-major order with entries separated by comma.
     - The `*_zc.txt` file contains a single value used as the zero-cutoff for the regularized effect values for determining which variants to be included in the clustering: Only variants with effects greater than this value are considered. This value is dynamically determined by the total number of variants so that a reasonable proportion of them are ''non-zero''.
  4. Visualization of clustering results: 
     `clustering_multivar_X-Y.png`; `clustering_univar_X.png`, `clustering_univar_Y.png`
     - Variants are colored by clusters, and Gaussians inferred for each cluster are shown along the side in the same colors (for up to 2D Gaussian). 
  5. Enrichment results:
     `enrichment_X.csv`, `enrichment_Y.csv`
     - Each result file has the same rows as in input  `gene_file`, with 3 additional columns, 'P', 'STAT', and 'VAR', corresponding to the p-value, test statistics, and variance of test statistics of the gene-level enrichment test. If the dependency package *chiscore* can not be installed successfully, that is, gene-level test is not available, then p-values will be generated by an alternative testing method adapted from the original code of `liu_sf()` from the Python package [chiscore](https://github.com/limix/chiscore), and the program will show a warning message *"Using alternative method for linear-combination-of-chi-square test for gene-level test."* 
     
     `multivar_gene_X-Y.csv`
     - The multivariate gene-level result file contains the same rows as in input  `gene_file`, with a couple additional columns:
       * 'size' is the same as 'N_SNPS', denoting the number of variants considered for the gene.
       * 'cls1_frac', ..., 'cls*K*_frac' (*K* columns): the fraction of variants in each gene that belong to each cluster, from cluster 1 to cluster *K*.
       * 'b1b1', 'b1b2', 'b2b2', etc. (*K(K+1)/2* columns): sum of the product of regularized effects for each pair of traits (including a trait to itself) for all variants in the gene, divided by the gene size. 
     
  6. Visualization of enrichment results:
     `enrichment_X.png`, `enrichment_Y.png` 
     - The manhattan plot shows negative log of adjusted p-values for each gene along the genome.

  Users may generate other visualizations or perform downstream analyses using the result provided in these files on their own. 
  
### `split_ld_blocks.py`

  - Split LD matrices into block-wise partitions. 
  
  - **Usage**:  
  ```bash
  python -u split_ld_blocks.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --full_ld_file            Path to full LD matrix file (space-delimited)
    --output_file             Directory to save indices of the breakpoints to split the LD
    ```
    
  - **Optional Arguments**:  
    ```text
    --avg_block_size          Approximated average LD block size after splitting [default: 1000]
    ```

  - **Outputs**:  
    ```text
    output_file               # indices of the breakpoints (right-boundary-only) to split the LD, with one index on each line
    ```

  - **Command** (from `split_and_process_ld.sh`):  
    ```bash
    python -u split_ld_blocks.py \
    --full_ld_file $full_ld_file \
    --output_file $output_file \
    --avg_block_size $avg_block_size
    ```

### `process_ld_blocks_and_gwa.py`

  - Process split LD blocks and concatenates the breakpoints as well as GWA results (optional) of all chromosomes for downstream ML-MAGES analysis.
  
  - **Usage**:  
  ```bash
  python -u process_ld_blocks_and_gwa.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --chrs                  Chromosomes corresponding to LD (and GWA) files to be processed, comma-separated
    --full_ld_files         Path to full LD matrix files, multiple chromosomes separated by comma
    --brkpts_files          Path to files containing saved indices of LD splitting breakpoints (i.e., output of ``split_ld_blocks.py``), multiple chromosomes separated by comma
    --ld_block_meta_file    Path to the output file with meta info of LD splitting
    --ld_block_brkpts_file  Path to the output file with all breakpoints (right boundaries only)
    --ld_block_path         Path to save the split LD blocks (block*.ld)
    --gwa_files             Path to GWAS summary statistics files (CSV format) to be processed (use a blank string "" if not processing this), multiple traits separated by semicolon and multiple chromosomes of the same trait separated by comma; ex. of nested structure: "trait1-chr1,trait1-chr2;trait2-chr1,trait2-chr2")
    --processed_gwa_files   Path to save processed GWAS files, multiple traits separated by semicolon
    ```
    
  - **Optional Arguments**:  
    ```text
    --chr_col               Name of the column for chromosome of the genetic variant [default: CHR]
    --pos_col               Name of the column for base pair position of the genetic variant  [default: POS]
    --id_col                Name of the column for ID of the genetic variant [default: ID]
    --beta_col              Name of the column for GWA effect of the genetic variant  [default: BETA]
    --se_col                Name of the column for GWA standard error of the genetic variant  [default: SE]
    ```
    
  - **Outputs**:  
    ```text
    ld_block_meta_file      # CSV file containing information on LD blocks; columns: `block_id` (index in all chromosomes), `chr`, `id_in_chr` (index in each chromosome), `brkpt` (right boundary breakpoint)
    ld_block_brkpts_file    # Text file (txt format) with indices of the breakpoints (right-boundary-only) to split the LD, with one index on each line
    ld_block_path
    └── block*.ld                                     # visualization of clustering results (up to bivariate)
    ```

  - **Command** (from `split_and_process_ld.sh`):  
    ```bash
    python -u process_ld_blocks_and_gwa.py \
    --chrs $chrs \
    --full_ld_files $full_ld_files \
    --brkpts_files $brkpts_files \
    --ld_block_meta_file $ld_block_meta_file \
    --ld_block_brkpts_file $ld_block_brkpts_file \
    --ld_block_path $ld_block_path \
    --gwa_files $gwa_files \
    --processed_gwa_files $processed_gwa_files
    ```

### `simulate_train.py`

  - Simulate synthetic effects for pseudo-traits based on real genotyped data and LD. 
  
  - **Usage**:  
  ```bash
  python -u simulate_train.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --sim_chrs            Chromosomes corresponding to genotype and LD files to be used in the simulation, comma-separated
    --geno_path           Path to genotype data files in PLINK format (`ukb_chr*.qced.bed/bim/fam`), multiple chromosomes separated by comma
    --ld_path             Path to full LD matrix files (`ukb_chr*.qced.ld`), multiple chromosomes separated by comma
    --output_path         Path to save the simulation outputs
    ```

  - **Optional Arguments**:  
    ```text
    --n_inds              Number of individuals to be sampled for each simulation [default: 10000]
    --n_snps              Number of variants to be sampled for each simulation [default: 1000]
    --n_sim               Number of simulations [default: 100]
    --top_r               Number of top (highest correlation) variants to save for future feature construction [default: 25]
    ```

  - **Outputs**:  
    ```text
    output_path
    ├── ninds*_nsnps*_nsim*_topr*_chr*.X          # simulated features (n_sim*n_snps rows); number (and order) of columns correspond to those of features (default: 2*25+3; see [1])
    ├── ninds*_nsnps*_nsim*_topr*_chr*.y          # simulated trait values (n_sim*n_snps lines)
    └── ninds*_nsnps*_nsim*_topr*_chr*.meta       # meta information of simulations (4 columns and n_sim rows); columns record chr, starting index of sampled variants, h2, and number of associated variants
    ```

  - **Command** (from `simulate_train.sh`):  
    ```bash
    python -u simulate_train.py \
    --sim_chrs $sim_chrs \
    --geno_path $geno_path \
    --ld_path $ld_path \
    --output_path $output_path 
    ```

### `train_model.py`

  - Train the models for effect size shrinkage using simulated data based on genotyped data 
  
  - **Usage**:  
  ```bash
  python -u train_model.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --n_layer              Number of fully-connected layers in the model architecture configuration
    --top_r                Number of top variants for feature construction
    --geno_path            Path to genotype data files in PLINK format (`ukb_chr*.qced.bed/bim/fam`) for the chromosomes used in simulation
    --ld_path              Path to full LD matrix file (`ukb_chr*.qced.ld`) for the chromosomes used in simulation
    --gwa_files            Path to GWA result files (CSV format) used for matching the simulation to real data, multiple traits separated by comma; columns: `BETA` and `SE`
    --sim_path             Path to simulated training data files
    --sim_label_prefix     Prefix of the labels for simulation data to be used for training
    --output_base_path     Home directory to save the trained model
    --train_chrs           Chromosomes used in training set, multiple chromosomes separated by comma 
    --val_chrs             Chromosomes used in validation set, multiple chromosomes separated by comma 
    ```

  - **Optional Arguments**:  
    ```text
    --model_idx            Model index for labeling multiple models with the same architecture [default: 0]
    --n_epochs             Maximum training epochs [default: 500]
    --bs                   Batch size [default: 50]
    --lr                   Learning rate [default: 1e-4]
    --reg_lambda           Regularization lambda for model parameters [default:1e-5]
    --es_patience          Early stopping patience [default: 25]
    --es_eps               Early stopping tolerance [default: 1e-9]
    ```

  - **Outputs**:  
    ```text
    output/OUTPUT_PATH
    ├── epochs/                      # directory to store temporary models during training (for early-stopping)
    └── Fcatopb_c.model              # trained model, labeled by a=n_layer, b=top_r, c=model_idx
    ```

  - **Command** (from `train_model.sh`):  
    ```bash
    python -u train_model.py \
    --n_layer $n_layer \
    --top_r $top_r \
    --phenotypes $phenotypes \
    --geno_path $geno_path \
    --ld_path $ld_path \
    --gwas_path $gwas_path \
    --sim_path $sim_path \
    --sim_label_prefix $sim_label_prefix \
    --output_base_path $output_base_path \
    --train_chrs $train_chrs \
    --val_chrs $val_chrs \
    --model_idx $model_idx 
    ```

### `simulate_evaluation.py`

  - Simulate synthetic data for performance evluation based on real genotyped data and LD, including the scenarios of multi-trait associations. 
  
  - **Usage**:  
  ```bash
  python -u simulate_evaluation.py <arguments>
  ```  
  
  - **Required Arguments**:  
    ```text
    --chr               Chromosome corresponding to genotype and LD file to be used in the simulation
    --geno_file         Path to genotype data files in PLINK format (`ukb_chr*.qced.bed/bim/fam`) for the chromosome
    --full_ld_file      Path to full LD matrix file (`ukb_chr*.qced.ld`) for the chromosome
    --gwa_files         Path to GWA result files (CSV format) used for matching the simulation to real data, multiple traits separated by comma; columns: `BETA` and `SE`
    --gene_list_file    Path to the gene annotation file (CSV format) used for gene-level simulation; columns include `CHR`, `GENE`, `START`, and `END`
    --output_path       Path to save the simulation outputs
    ```

  - **Optional Arguments**:  
    ```text
    --n_inds              Number of individuals to be sampled for each simulation [default: 10000]
    --min_gene_size       Minimum number of variants in a gene for the gene to be considered a candidate for causal genes (i.e., those containing variants with non-zero true effects) [default: 10]
    --n_traits            Number of traits to be simulation simultaneously [default: 3]
    --causal_types        Association types of among simulated traits, with each type of association shared by comma-separated trait indices, multiple association types separated by semicolon [default: "1;2;3;1,2;1,2,3"] (Note: the default association types are trait 1-specific, trait 2-specific, trait 3-specific, traits 1&2-shared, all traits-shared.)
    --n_sim               Number of simulations [default: 100]
    ```

  - **Outputs**:
    ```text
    output_path
    └── data_sim*.txt     # simulated data (n_traits*3 columns); number of rows correspond to number of variants in the chromosome used; columns record true effect (for each of n_trait), observed GWA effect (for each of n_trait), and observed GWA standard error (for each of n_trait)
    ``` 
    

  - **Command** (from `simulate_evaluation.sh`):  
    ```bash
    python -u simulate_evaluation.py \
    --chr $chr \
    --geno_file $geno_file \
    --full_ld_file $full_ld_file \
    --gwa_files $gwa_files \
    --gene_list_file $gene_list_file \
    --output_path $output_path \
    --n_inds $n_inds \
    --min_gene_size $min_gene_size \
    --n_traits $n_traits \
    --causal_types $causal_types \
    --n_sim $n_sim
    ```

  
### Additional notebook files for visualizing results and comparing performances:
- `demo_vis_outputs.ipynb`: This Jupyter notebook provides example code on how to visualize and analyze gene-level output for multi-trait analyses.
- `demo_eval_perf.ipynb`: This Jupyter notebook provides example code on how to evaluate the performances of the methods using the simulated data.


## Data
### Contents of ``example_data`` folder
The `example_data` folder in this repository contains the toy data for running a single model:
- `example_gwa_HDL.csv` and `example_gwa_LDL.csv`: These comma-delimited csv files contain the genome-wide association (GWA) results on a subset of variants on a segment of Chromosome 20 from the UK Biobank dataset for High-Density Lipoprotein (HDL) and Low-Density Lipoprotein (LDL). **Required columns for GWA result files include ``BETA`` for GWA effect and ``SE`` for standard error.**
- `example_block_*.ld`: These files contain the linkage disequilibrium (LD) matrix, split into two blocks, of the same subset of variants from UK Biobank. **The matrix is comma-delimited.** 
- `block_brkpts.txt`: This file contains (0-based-)indices of the boundary points at which the LD matrix correspond to the set of variants is split into blocks. **Only the right boundaries are included, with one index on each line.** For instance, two indices ``851 1818`` indicate that the LD is split into blocks that should be indexed by ``[0:851]`` and ``[851:1818]``. Note that in Python indexing ``[start:end]``, the ``start`` index is inclusive, while the ``end`` index is exclusive
- `example_genelist`: This comma-delimited csv file contains the (unnamed) gene annotations of the subset of variants. Each gene is marked by the indices of the first and last variants in it (**columns ``SNP_FIRST`` and ``SNP_LAST``, required**), as well as its chromosome (``CHR``), its name (``GENE``), and the number of variants considered in this gene (``N_SNPS``).
These files provide the necessary data for performing the ML-MAGES method described in the paper.

### Contents of ``data`` folder
The `data` folder in this repository contains real data for running the full method with ensembled models, i.e., input data files to used in `run_ml_mages.sh`. However, only selected data is included in this repository due to storage limitations and privacy concerns for sensitive individual-level genetic data. The following files are required for analysis:

- The `data/block_ld` folder contains LD data. Suppose there are a total of x LD blocks, then this folder should contain x+1 files, optionally with additional files for reference. 
  - The full LD of split segments along all the chromosomes, ordered by chromosome and position, are saved in x files labeled as `block_0.ld`, `block_1.ld`, to `block_x.ld`, each being a comma-delimited matrix. 
  - The file `block_ids.txt` contains x lines, where each line is the index of the last variant in the corresponding block **plus one**. Indices go from 0 to M-1, where M is the total number of variants along all the chromosomes. For instance, if `block_0.ld` is of size 200x200 and `block_1.ld` is of size 210x210, then the first two lines in `block_ids.txt` should be 200 and 410.
  - [Optional] The file `blocks_meta.csv` is a comma-delimited file with three columns: 'block_id', 'chr', 'id_in_chr', and x rows (excluding the header).  The three columns correspond to the block index (as used in the `.ld` file names), the chromosome to which the block belongs, and the index of the block within that chromosome. For instance, a row of `405,15,0` means the `block_405.ld` is the first (indexed by 0) block in CHR15. 
- The `data/gwa` folder contains GWA result files, labeled as `gwas_TRAIT.csv` where `TRAIT` is the trait name and should be the same as the one used for argument `traits` in the input to `ml_mages.py`. The file for each trait should have exactly M lines excluding the header, with each line corresponds to a variant, and all variants ordered the same as those in LD blocks. There should be at least three columns: 'BETA','SE', and 'CHR'.
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

The following files, contained in the folder `data/real` (not included), are used in `split_and_process_ld.sh`:
- `ukb_chr*.qced.ld`: full LD matrix of Chr* in UKB European individuals.
- `gwa/ukb_chr*.TRAIT.csv`: GWA results of TRAIT of Chr* in UKB European individuals, with columns including `#CHROM`, `POS`, `ID`, `BETA`,	`SE` and several other non-used ones.

The following files, contained in the folder `data/real` (not included), are used in `simulate_evaluation.sh` and `demo_eval_perf.ipynb`:
- `ukb_chr*.qced.bim`, `ukb_chr*.qced.bed`, `ukb_chr*.qced.fam`: the genotype data of Chr* of UKB European individuals in PLINK format.
- `ukb_chr*.qced.ld`: full LD matrix of Chr* in UKB European individuals.
- `block_ld/chr*_brkpts.txt`: The (0-based-)indices of the boundary points at which the LD matrix of Chr* is split into blocks.

The simulated training data will be included in the folder `data/simulation/sim_train` (not shown), and subsequently used for model training.

The simulated evluation data will be included in the folder `data/simulation/sim_gene_mlmt` (not shown), and subsequently used for performance evaluation.

## Models
The `trained_model` folder in this repository contains trained models. 

The subfolder `genotyped_simulated_training` contains the six models, each of a different architecture, trained using genotyped-data-based simulation described in the paper. We do not provide the simulated training data, but simulation and training can be performed following steps described in appendix. 

The subfolder `imputed_simulated_training` contains two set of models, each with 10 models of a same architecture, trained using imputed-data-based simulation. The output of each set of models are averaged to generate an ensemble result of shrinkage, as used in the `ml_mages.py`. Similarly, simulation and training can be performed following steps described in appendix. 

Name of the model files follows ''Fc*a*top*b*_*c*.model'', where *a* is the number of fully-connected layers in the neural network model, *b* is the number of top variants used to construct the features, and *c* is the index of the model among all models of the same architecture.


----
## Citation

[1] Liu X, Crawford L, Ramachandran S. ML-MAGES: Machine learning approaches for multivariate genetic association analyses with genes and effect size shrinkage. (accepted at RECOMB 2025)

For questions and comments, please contact Xiran Liu at *xiran_liu1 at brown dot edu*.
