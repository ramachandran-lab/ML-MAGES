import numpy as np
import scipy as sp
import os
import shutil
import time
import argparse

import torch
import torch.nn as nn

from ._util_funcs import disp_params, parse_file_list
from ._train_funcs import get_n_feature, FCNN, EarlyStopper, WeightedMSELoss


def train_model(n_layer, top_r, train_input_files, val_input_files, model_path, model_prefix, model_idx=0,
                n_epochs=500, bs=100, lr=1e-4, reg_lambda=1e-5, es_patience=10, es_eps=1e-9, n_train=400000, n_val=100000, nz_weight=1):
    
    # parse input files
    train_input_files = parse_file_list(train_input_files)
    val_input_files = parse_file_list(val_input_files)
    print(f"Training input files: {train_input_files}")
    print(f"Validation input files: {val_input_files}")

    # save model per epoch 
    model_label = "top{}_{}L".format(top_r,n_layer)
    print("-------------{} Model {} -------------".format(model_label,model_idx))
    epoch_path = os.path.join(model_path,"{}_model_{}_{}_epochs".format(model_prefix,model_label,model_idx)) if model_prefix!="" else os.path.join(model_path,"model_{}_{}_epochs".format(model_label,model_idx))   
    if os.path.exists(epoch_path):
        shutil.rmtree(epoch_path)
    os.makedirs(epoch_path)
    print("epoch_path:", epoch_path)
    print("-----------------------------")

    if torch.cuda.is_available():
        print("CUDA:", torch.cuda.get_device_name(0))
    n_feature = get_n_feature(top_r)
    print("#features:", n_feature)

    # load training data
    X_train = []
    y_train = []
    for train_input_file in train_input_files:
        X = np.loadtxt("{}.X".format(train_input_file))
        y = np.loadtxt("{}.y".format(train_input_file))
        X_train.append(X)
        y_train.append(y)
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print("Training data size:", X_train.shape, y_train.shape)
    assert X_train.shape[1]==n_feature 
    assert X_train.shape[0]>=n_train, "Training data size {} is smaller than n_train {}".format(X_train.shape[0],n_train)
    
    # load validation data
    X_val = []
    y_val = []
    for val_input_file in val_input_files:
        X = np.loadtxt("{}.X".format(val_input_file))
        y = np.loadtxt("{}.y".format(val_input_file))
        X_val.append(X)
        y_val.append(y)
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    print("Validation data size:", X_val.shape, y_val.shape)
    assert X_val.shape[1]==n_feature 
    assert X_val.shape[0]>=n_val, "Validation data size {} is smaller than n_val {}".format(X_val.shape[0],n_val)

    # subset 
    train_idx = np.random.choice(X_train.shape[0],n_train,replace=False)
    val_idx = np.random.choice(X_val.shape[0],n_val,replace=False)
    # construct torch dataset
    X_tr_torch = torch.tensor(X_train[train_idx,:], dtype=torch.float32)
    y_tr_torch = torch.tensor(y_train[train_idx], dtype=torch.float32).reshape(-1, 1)
    X_val_torch = torch.tensor(X_val[val_idx,:], dtype=torch.float32)
    y_val_torch = torch.tensor(y_val[val_idx], dtype=torch.float32).reshape(-1, 1)

    model = FCNN(n_feature, n_layer=n_layer, model_label=model_label)

    print("======Model Structure======")
    print(model)
    n_param = 0
    for W in model.parameters():
        n_param += np.prod(W.shape)
    print("#parameters:",n_param)
    print("===========================")

    # prep training    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr_torch = X_tr_torch.to(device)
    y_tr_torch = y_tr_torch.to(device)
    X_val_torch = X_val_torch.to(device)
    y_val_torch = y_val_torch.to(device)
    model.to(device)

    print("------TRAINING MODEL {} ({})------".format(model_label, model_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss_fn = WeightedMSELoss(non_zero_weight=nz_weight)
    early_stopper = EarlyStopper(patience=es_patience, min_delta=es_eps)
    
    losses = list()
    tr_times = list()
    for epoch in range(n_epochs):
        t0 = time.time()
        
        # shuffle data
        perm_idx = torch.randperm(len(X_tr_torch))
        X_tr_torch = X_tr_torch[perm_idx]
        y_tr_torch = y_tr_torch[perm_idx]
    
        # train
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
        torch.save(model.state_dict(), os.path.join(epoch_path,"epoch{}".format(epoch)))
    
        if early_stopper.early_stop(val_loss): 
            break
            
    print("Stopped at epoch {}".format(epoch))
    best_epoch = np.argsort(np.array(losses)[:,1])[0]
    print("Epoch with best performance:",best_epoch)

    model = FCNN(n_feature, n_layer=n_layer, model_label=model_label)
    best_epoch_model_path = os.path.join(epoch_path,"epoch{}".format(best_epoch))
    state = torch.load(best_epoch_model_path)
    model.load_state_dict(state)
    
    # save model
    save_file = os.path.join(model_path,"{}_{}_{}.model".format(model_prefix,model_label,model_idx)) if model_prefix!="" else os.path.join(model_path,"{}_{}.model".format(model_label,model_idx))
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


def main():
    print("RUNNING: train_model")

    parser = argparse.ArgumentParser(description='Train the model using prepared input data')
    parser.add_argument('--n_layer', type=int, help='Number of hidden layers in the model')
    parser.add_argument('--top_r', type=int, help='Number of top variants to use')
    parser.add_argument("--train_input_files", nargs="+", default=[], help="List of input files for training, or a file containing one filename per line")
    parser.add_argument("--val_input_files", nargs="+", default=[], help="List of input files for validation, or a file containing one filename per line")
    parser.add_argument('--model_path', type=str, help='Path for output trained models')
    parser.add_argument('--model_prefix', type=str, default="", help="Model prefix")
    
    # Optional arguments
    parser.add_argument('--model_idx', type=int, default=0, help='Model index with default as 0 (for training multiple models with the same architecture)')
    parser.add_argument('--n_epochs', type=int, default=500, help='Maximum training epochs')
    parser.add_argument('--bs', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--reg_lambda', type=float, default=1e-5, help='Regularization lambda for model parameters')
    parser.add_argument('--es_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--es_eps', type=float, default=1e-9, help='Early stopping tolerance')
    parser.add_argument('--n_train', type=int, default=400000, help='Number of training samples to use (default 400000)')
    parser.add_argument('--n_val', type=int, default=100000, help='Number of validation samples to use (default 100000)')
    parser.add_argument('--nz_weight', type=float, default=2, help='Weight for the non-zero effects in the loss function (default 2)')

    args = parser.parse_args()
    disp_params(args, title="INPUT SETTINGS")
    train_model(**vars(args))


if __name__ == "__main__":
    main()
