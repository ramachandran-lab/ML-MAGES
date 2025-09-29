import numpy as np
import pandas as pd
import scipy as sp

# for training NN
import torch
import torch.nn as nn
# from ._util_funcs import *


def get_n_feature(top_r):
    """
    Calculate the number of features based on the given top_r value.
    (bhat, shat, ldsc, & top_r_beta, top_r_ld for each top_r)

    Parameters:
    top_r (int): The top_r value used in the calculation.

    Returns:
    int: The number of features calculated based on the given top_r value.
    """
    return 2*top_r+3

# NN model    
class FCNN(nn.Module):
    def __init__(self, n_feature:int, n_layer:int, model_label=""):
        super().__init__()
        self.name = model_label
        self.n_feature = n_feature
        self.n_layer = n_layer

        layers = []
        out_dim = n_feature
        for i_layer in range(n_layer):
            if i_layer==0:
                in_dim, out_dim = n_feature, 64
            else:
                in_dim, out_dim = out_dim, 32
            if i_layer==(n_layer-1):
                out_dim = 8
            dor = 0.1 if i_layer==(n_layer-1) else 0.2
                
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(num_features=out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dor))
        
        self.hidden_stack = nn.Sequential(*layers)

        # final layer
        self.output_layer = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.hidden_stack(x)
        x = self.output_layer(x)
        return x


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class WeightedMSELoss(nn.Module):
    def __init__(self, non_zero_weight=5.0): # Default weight for non-zero values
        super(WeightedMSELoss, self).__init__()
        self.non_zero_weight = non_zero_weight

    def forward(self, predictions, targets):
        # Calculate element-wise squared error
        squared_error = (predictions - targets) ** 2

        # Create a mask for non-zero target values
        non_zero_mask = (targets.abs() > 1e-6).float()

        # Apply weights: higher weight for non-zero targets
        weighted_squared_error = squared_error * (non_zero_mask * self.non_zero_weight + (1 - non_zero_mask) * 1.0)

        # Calculate the mean of the weighted squared errors
        loss = torch.mean(weighted_squared_error)
        return loss
