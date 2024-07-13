import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import shutil
import time
# for training NN
import torch
import torch.nn as nn
from sklearn.utils import shuffle

import ml_mages
import _train_funcs as tf


def main():
    if len(sys.argv) < 13:
        tmp = "geno_path ld_path gwas_path sim_path sim_label_prefix output_base_path train_chrs(separated by comma) val_chrs(separated by comma) model_idx"
        tmp2 = "(n_epochs bs lr reg_lambda es_patience es_eps)"
        print("Usage: {} train_model.py model_layer top_r phenotypes(separated by comma)".format(sys.argv[0]))+" "+tmp+" "+tmp2
        sys.exit(1)

    model_layer = int(sys.argv[1])
    print("Using {}-layer model".format(model_layer))
    top_r = int(sys.argv[2])
    print("Using top {} variants".format(top_r))
    phenotypes = sys.argv[3].split(",")
    print("phenotypes",phenotypes)
    geno_path = sys.argv[4]
    print("geno_path:", geno_path)
    ld_path = sys.argv[5]
    print("ld_path:", ld_path)
    gwas_path = sys.argv[6]
    print("gwas_path:", gwas_path)
    sim_path = sys.argv[7]
    print("sim_path:", sim_path)
    sim_label_prefix = sys.argv[8]
    print("sim_label_prefix:", sim_label_prefix)
    max_r = int(sim_label_prefix.split("topr")[1])
    output_base_path = sys.argv[9]
    train_chrs = [int(c) for c in sys.argv[10].split(",")]
    print("train_chrs:", train_chrs)
    val_chrs = [int(c) for c in sys.argv[11].split(",")]
    print("val_chrs:", val_chrs)
    #[18,19,20,21,22]
    model_idx = int(sys.argv[12]) if len(sys.argv)>12 else 0
    print("model_idx:", model_idx)
    output_path = os.path.join(output_base_path,"geno_sim_ensemble_Fc{}top{}".format(model_layer,top_r))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("output_path:", output_path)
    epochs_output_path = os.path.join(output_path,"epochs")
    if not os.path.exists(epochs_output_path):
        os.makedirs(epochs_output_path)

    # training parameters
    print("------TRAINING PARAMETERS------")
    n_epochs = int(sys.argv[13]) if len(sys.argv)>13 else 500
    print("maximum training epochs:", n_epochs)
    bs = int(sys.argv[14]) if len(sys.argv)>14 else 50 # batch_size
    print("batch size:", bs)
    lr = int(sys.argv[15]) if len(sys.argv)>15 else 1e-4 #4 # learning rate
    print("learning rate:", lr)
    reg_lambda = int(sys.argv[16]) if len(sys.argv)>16 else 1e-5
    print("regularization lambda for model parameters:", reg_lambda)
    es_patience = int(sys.argv[17]) if len(sys.argv)>17 else 25 # early stopping patience
    print("early stopping patience:", es_patience)
    es_eps = int(sys.argv[18]) if len(sys.argv)>18 else 1e-9 # early stopping min_delta
    print("early stopping tolerance:", es_eps)
    print("-----------------------------")

    if torch.cuda.is_available():
        print("CUDA:", torch.cuda.get_device_name(0))
    n_features = ml_mages.get_n_features(top_r)
    feature_lb = "top{}".format(top_r)

    # load data
    train_files_prefix = [os.path.join(sim_path,"{}_chr{}".format(sim_label_prefix,chr)) for chr in train_chrs]
    val_files_prefix = [os.path.join(sim_path,"{}_chr{}".format(sim_label_prefix,chr)) for chr in val_chrs]
    X_train, y_train, meta_train = tf.load_simulation(train_files_prefix)
    X_test, y_test, meta_test = tf.load_simulation(val_files_prefix)
    print("loaded training data shape:", X_train.shape, y_train.shape, meta_train.shape)
    print("loaded validation data shape:", X_test.shape, y_test.shape, meta_test.shape)

    # load real data 
    real_files = [os.path.join(gwas_path,'gwas_{}.csv'.format(pheno)) for pheno in phenotypes]
    beta_real, se_real = tf.load_real_data(real_files)

    # transform simulated data to match real distributions and subset for training
    X_tr, y_tr, y_train_scale = tf.scale_and_subset(X_train,y_train, beta_real, se_real, max_r, top_r, scale=1, asymmetric=False)
    X_val, y_val, y_val_scale = tf.scale_and_subset(X_test,y_test, beta_real, se_real, max_r, top_r, scale=1, asymmetric=False)

    # build model
    model = ml_mages.construct_new_model(model_layer,n_features,feature_lb)
    print(model)
    n_param = 0
    for W in model.parameters():
        n_param += np.prod(W.shape)
    print("#parameters:",n_param)

    # save model per epoch 
    epoch_path = os.path.join(epochs_output_path,"model_{}".format(model_idx))
    if os.path.exists(epoch_path):
        shutil.rmtree(epoch_path)
    os.makedirs(epoch_path)
    print(epoch_path)

    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=42)
    X_tr_torch = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_torch = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr_torch = X_tr_torch.to(device)
    y_tr_torch = y_tr_torch.to(device)
    X_val_torch = X_val_torch.to(device)
    y_val_torch = y_val_torch.to(device)
    model.to(device)

    print("------TRAINING MODEL {}------".format(model_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss_fn = nn.MSELoss()
    early_stopper = tf.EarlyStopper(patience=es_patience, min_delta=es_eps)
    
    losses = list()
    tr_times = list()
    for epoch in range(n_epochs):
        t0 = time.time()
        
        # shuffle data
        perm_idx = torch.randperm(len(X_tr_torch))
        X_tr_torch = X_tr_torch[perm_idx]
        y_tr_torch = y_tr_torch[perm_idx]
    
        # train
        # model.train()
        for i in range(0, len(X_tr_torch), bs):
            Xbatch = X_tr_torch[i:i+bs]
            y_pred = model(Xbatch)
            ybatch = y_tr_torch[i:i+bs]
            loss = loss_fn(y_pred, ybatch)
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            loss += reg_lambda * l2_reg/n_param      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # model.eval()
        with torch.no_grad():
            y_tr_pred = model(X_tr_torch)
            y_val_pred = model(X_val_torch)
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
    
        tr_loss = loss_fn(y_tr_pred,y_tr_torch)
        reg_loss = reg_lambda * l2_reg/n_param  
        val_loss = loss_fn(y_val_pred, y_val_torch)
        tr_time = time.time() - t0
     
        print("Epoch {:>2}, time {:.2f}s, losses: train {:.3e}, val {:.3e}, reg {:.3e}".format(epoch,tr_time,tr_loss,val_loss,reg_loss))
        # losses.append([tr_loss.detach().numpy(),val_loss.detach().numpy(),reg_loss.detach().numpy()])
        losses.append([tr_loss.item(),val_loss.item(),reg_loss.item()])
        tr_times.append(tr_time)
        
        # Save the current model (checkpoint) to a file
        model_path = os.path.join(epoch_path,"epoch{}".format(epoch))
        torch.save(model.state_dict(), model_path)
    
        if early_stopper.early_stop(val_loss): 
            break
            
    print("Stopped at epoch {}".format(epoch))
    # get best epoch
    best_epoch = np.argsort(np.array(losses)[:,1])[0]
    print("Epoch with best performance:",best_epoch)
    
    model = ml_mages.construct_new_model(model_layer,n_features,feature_lb)
    model_path = os.path.join(epoch_path,"epoch{}".format(best_epoch))
    state = torch.load(model_path)
    model.load_state_dict(state)

    # save model
    save_file = os.path.join(output_path,"{}_{}.model".format(model.name,model_idx))
    torch.save(model.state_dict(), save_file)

    # evaluate model
    # compute performance (no_grad is optional)
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_tr_pred = model(X_tr_torch)
    y_tr_pred = y_tr_pred.cpu().numpy().squeeze()
    y_tr_true = y_tr_torch.cpu().numpy().squeeze()
    tr_err = np.mean((y_tr_pred-y_tr_true)**2)
    print(f"Training MSE {tr_err}")

    with torch.no_grad():
        y_val_pred = model(X_val_torch)
    y_val_pred = y_val_pred.cpu().numpy().squeeze()
    y_val_true = y_val_torch.cpu().numpy().squeeze()
    val_err = np.mean((y_val_pred-y_val_true)**2)
    print(f"Validation MSE {val_err}")
    y_true, y_pred = y_tr_true, X_tr_torch[:,0].cpu().numpy()
    perf = "TRAIN obs pcorr={:.3f}".format(sp.stats.pearsonr(y_true, y_pred)[0])
    print(perf)
    y_true, y_pred = y_val_true, X_val_torch[:,0].cpu().numpy()
    perf = "VAL obs pcorr={:.3f}".format(sp.stats.pearsonr(y_true, y_pred)[0])
    print(perf)
    y_true, y_pred = y_tr_true, y_tr_pred
    perf = "TRAIN reg pcorr={:.3f}".format(sp.stats.pearsonr(y_true, y_pred)[0])
    print(perf)
    y_true, y_pred = y_val_true, y_val_pred
    perf = "VAL reg pcorr={:.3f}".format(sp.stats.pearsonr(y_true, y_pred)[0])
    print(perf)


if __name__ == "__main__":
    main()