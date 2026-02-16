# module load python/3.11.0s-ixrhc3q
# module load cuda
# source ~/pytorch.venv/bin/activate

python -m mlmages.train_model \
--top_r 15 \
--n_layer 3 \
--train_input_files output/simulation/snp_only_training_input/chr22 \
--val_input_files output/simulation/snp_only_training_input/chr22 \
--model_path output/example_trained_models \
--model_prefix chr22 \
--model_idx 2 \
--n_train 100000 --n_val 20000 
# n_train and n_val should be set larger according to the actual training/validation sample size in the input files