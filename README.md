# ML-MAGES
Last Updated: 9/24/2025

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
* Example data are provided in [data/](data).
* Other files used in running the example are provided in [example_files/](example_files)

## Usage
    ```
    usage: __main__.py [-h] --chroms CHROMS [CHROMS ...] --gwas_files GWAS_FILES [GWAS_FILES ...]
                   [--model_files MODEL_FILES [MODEL_FILES ...]] --full_ld_files FULL_LD_FILES [FULL_LD_FILES ...] --gene_file
                   GENE_FILE --trait_names TRAIT_NAMES [TRAIT_NAMES ...] [--split_ld [SPLIT_LD]] [--ldblock_dir LDBLOCK_DIR]
                   [--avg_block_size AVG_BLOCK_SIZE] [--vis [VIS]] [--output_dir OUTPUT_DIR]

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
    ```

## Examples
Run Example 1 (all required data are provided)
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

