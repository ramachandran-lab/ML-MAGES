# ML-MAGES

This folder contains sample data and code for ML-MAGES: machine learning approaches for multivariate genetic association analyses with genes and effect size shrinkage.

Python 3.9.16 is used for implementing the method. A list of Python packages used for the implementation is included in the file `Python_packages.txt`. 

## Code Folder

The `code` folder in this repository contains the following files:

- `run_sample.sh`: This file contains the bash script with the command to run the Python code file for the sample data, including a brief description of all the input arguments. The script need to run from the `code` folder, or input paths in the bash file need to be updated. 
- `run_sample.py`: This file contains the code to apply the method on sample data.
- `ml_mages.py`: This file includes implemented functions for each step of the method.
- `_cls_funcs_.py`: This file contains utility functions for the clustering algorithm.

## Data Folder
The `sample_data` folder in this repository contains the sample data files. The data contains GWA results of traits HDL and LDL as well as LD obtained on a segment of Chr20 from UK Biobank (see manuscript).

## Model Folder
The `trained_model` folder in this repository contains the trained models. We do not provide the simulated training data, but the pre-trained models can be obtained from the code files, and training can be performed following steps described in our manuscript and appendices. 

