This folder contains the models trained using simulation based on imputed UKB data.

## simulation setting
Simulated training set is based on CHR 18, 19, 21, and 22. 
Simulated validation set is based on CHR 20.

A total of 100 simulations each with 10,000 individuals and variants within a maximum of 1000kb range are generated for each chromosome. 100 variants in highest LD to each vaiant are included in the features.

## training and validation data preparation
Simulated BETA and SE are transformed to match the distributions of real BETA and SE from GWA of the trait MCV (in imputed UKB). They are then scaled up by 250, which equals to 1/mean(abs(transformed_BETA)) rounded to the nearest 10. Note that when applying the trained models, input needs to be scaled up by 250 as well, and the corresonding output needs to be divided by 250 afterwards.

To balance the zeros vs. non-zeros in the simulated ground-truth BETA values, a subset of simulated data with zero BETA are sampled randomly, and together with all simulated data with non-zero BETA, used for the training and validation. The size of training data used is 15000, corresponding to 2 x the number of true non-zeros in all simulated training data rounded up to the nearest 5000. The size of validation data used is 5000, corresponding to 2 x the number of true non-zeros in all simulated validation data rounded up to the nearest 5000. 

## model architectures
* 2 fully-connected layer with top 15 variants in the feautres
* 3 fully-connected layer with top 15 variants in the feautres
  
## ensemble learning
For each architecture, 10 models are trained (each time the training and validation data are sub-sampled randomly), and we use the ensemble learning results which average the output of 10 models.

## training parameters
n_epochs = 500
batch_size = 50 
learning_rate = 1e-4 
reg_lambda = 100
es_patience, es_eps = 50, 1e-9 (for early stopping)


