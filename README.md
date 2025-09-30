# ML-MAGES
Last Updated: 9/29/2025

This folder contains example data and code for __*ML-MAGES*: A machine learning framework for multivariate genetic association analyses with genes and effect size shrinkage__


## Requirements  
The method is implemented in *Python 3*. 

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

## Set Up
* Install required Python packages if not already (see [`requirements.txt`](requirements.txt)).
* **Clone this repository to your local directory.**
  ```bash
  git clone https://github.com/ramachandran-lab/ML-MAGES.git
  ```
* The default working directory is assumed to be `ML-MAGES` to run the program ``mlmages``. However, you can switch to your preferred working directory, provided that you update all the file paths accordingly.
* Pre-trained model are provided in [trained_models/](trained_models). 
* Example data are provided in [data/](data). For space considerations, a full set of summary-level data and simulation data used in this project is provided in Zenodo.
* Other files used in running the example (on small subset data) are provided in [example_files/](example_files)

## Usage
To run the full *ML-MAGES* pipeline, use the main module ``mlmages``. See the helper message by running ``python -m mlmages -h``.
````
usage: __main__.py [-h] --chroms CHROMS [CHROMS ...] --gwas_files GWAS_FILES [GWAS_FILES ...]
               [--model_files MODEL_FILES [MODEL_FILES ...]] --full_ld_files FULL_LD_FILES
               [FULL_LD_FILES ...] --gene_file GENE_FILE --trait_names TRAIT_NAMES [TRAIT_NAMES ...]
               [--split_ld [SPLIT_LD]] [--ldblock_dir LDBLOCK_DIR] [--avg_block_size AVG_BLOCK_SIZE]
               [--vis [VIS]] [--output_dir OUTPUT_DIR]

Master pipeline for ML-MAGES: (LD block extraction), shrinkage, clustering, and enrichment

optional arguments:
-h, --help            show this help message and exit
--chroms CHROMS [CHROMS ...]
                      Chromosome numbers
--gwas_files GWAS_FILES [GWAS_FILES ...]
                      GWAS files, one per trait
--model_files MODEL_FILES [MODEL_FILES ...]
                      Model files for shrinkage
--full_ld_files FULL_LD_FILES [FULL_LD_FILES ...]
                      Full LD files for splitting into blocks and enrichment tests
--gene_file GENE_FILE
                      Gene file for enrichment and gene-level analysis
--trait_names TRAIT_NAMES [TRAIT_NAMES ...]
                      Trait names (same order as gwas_files)
--split_ld [SPLIT_LD]
                      Whether to split LD files by chromosome
--ldblock_dir LDBLOCK_DIR
                      Path to store LD block files
--avg_block_size AVG_BLOCK_SIZE
                      Average block size for LD block extraction
--vis [VIS]           Whether to generate visualizations
--output_dir OUTPUT_DIR
                      Output directory
````

## Examples
For the following examples, all required data are provided. Check out the data repo for the full LD file.

**Example 1: run on a single chromosome** 
```bash
bash scripts/run_ex1.sh
```
which contains the following command:
```bash
python -m mlmages \
--chroms 22 \
--gwas_files data/gwas/gwas_HDL.csv data/gwas/gwas_LDL.csv \
--model_files example_files/model_files.txt \
--full_ld_files example_files/full_ld_files.txt \
--gene_file data/genes/genes.csv \
--trait_names HDL LDL \
--split_ld True \
--ldblock_dir data/ld_blocks/ \
--avg_block_size 1000 \
--output_dir output
```

**Example 2: run on multiple chromosomes** 
```bash
bash scripts/run_ex2.sh
```
which contains the following command:
```bash
python -m mlmages  \
--chroms 20 21 22 \
--gwas_files data/gwas/gwas_MCV.csv data/gwas/gwas_MPV.csv data/gwas/gwas_PLC.csv \
--model_files example_files/model_files.txt \
--full_ld_files example_files/chr20-22_full_ld_files.txt \
--gene_file data/genes/genes.csv \
--trait_names MCV MPV PLC \
--split_ld True \
--ldblock_dir data/ld_blocks/ \
--avg_block_size 1000 \
--output_dir ../output
```

**Example using ENet for shrinkage (also an example running the workflow step-by-step)** 
```bash
bash scripts/run_enet_ex.sh
```
This example illustrates how to use the  ``mlmages.shrink_by_enet`` submodule to perform shrinkage using elastic net (benchmark method).

Change the submodule to ``mlmages.shrink_by_mlmages`` if applying the ML-MAGES shrinkage and provide ``--model_files`` argument appropriately (see "Example 1").

## Other Scripts 
In addition, we provide a full list of example scripts and code used for each step in our analysis, including
* Data preparation:
  * scripts/prepare_data.sh (*We provide commands used to generate summary-level data from UKB data.*)
  * scripts/process_ld_files.sh
* Simulation data generation (for both model training and performance evaluation)
  * scripts/simulate_data_snp_only.sh
  * scripts/construct_input_from_snp_only.sh
  * scripts/simulate_data_gene_level.sh
* Application of ML-MAGES on simulation data
  * scripts/shrink_sim_snp_only_by_mlmages.sh
  * scripts/shrink_and_cluster_sim_gene_level_by_mlmages.sh
* View results and performance:
  * (Visualization of ML-MAGES results was embedded in the main example.)
  * view_results_example_code/view_sim_results.py
  * view_results_example_code/view_enet_ex_results.py (can also be easily adapted to view ML-MAGES results)

## Data Availability
Example data are provided in this GitHub repository. See Zenodo for summary-level data and simulation data used in this project.
