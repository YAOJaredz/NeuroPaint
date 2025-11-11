import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
import numpy as np

def np2tensor(v):
    return torch.from_numpy(v).type(torch.float)

def np2param(v, grad=True):
    return nn.Parameter(np2tensor(v), requires_grad=grad)

def get_device():
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")
    return device


def poisson_nll_loss(rate, spike_train, eps=1e-6):
    """
    Compute the Poisson negative log-likelihood loss.
    
    Args:
        rate (Tensor): predicted firing rates, shape (T,) or (batch, T)
        spike_train (Tensor): observed spikes, same shape as rate
        eps (float): small constant to avoid log(0)
    
    Returns:
        Tensor: scalar loss
    """
    # Ensure positivity of rate
    rate = torch.clamp(rate, min=eps)

    loss = rate - spike_train * torch.log(rate)
    return loss.mean()


def preprocess_X_and_y(X, y, smooth_w, halfbin_X):
    # X: (K, T, N)
    # y: (K, T, N)
    y = gaussian_filter1d(y, smooth_w, axis=1)  # (K,T, N)
    X_padded = np.pad(X, pad_width=((0,0),(halfbin_X, halfbin_X),(0,0)), mode='reflect')
    X_padded = gaussian_filter1d(X_padded, smooth_w, axis=1)  # (K,T, N)
    ## give X of neighboring time windows as input
    X_output = np.zeros((X.shape[0], X.shape[1], X.shape[2]*(1+2*halfbin_X)))
    for i in range(halfbin_X, X.shape[1] + halfbin_X):
        _ = X_padded[:,i-halfbin_X:i+halfbin_X+1,:]
        _ = np.moveaxis(_, 1, -1) # (K, N, 1+2*halfbin_X)
        X_output[:,i-halfbin_X,:] = _.reshape(X.shape[0], -1)
    return X_output, y
