import torch
import torch.nn as nn
import torch.nn.functional as F
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


# def preprocess_X_and_y(X, y, smooth_w, halfbin_X):
#     # X: (K, T, N)
#     # y: (K, T, N)
#     y = gaussian_filter1d(y, smooth_w, axis=1)  # (K,T, N)
#     X_padded = np.pad(X, pad_width=((0,0),(halfbin_X, halfbin_X),(0,0)), mode='reflect')
#     X_padded = gaussian_filter1d(X_padded, smooth_w, axis=1)  # (K,T, N)
#     ## give X of neighboring time windows as input
#     X_output = np.zeros((X.shape[0], X.shape[1], X.shape[2]*(1+2*halfbin_X)))
#     for i in range(halfbin_X, X.shape[1] + halfbin_X):
#         _ = X_padded[:,i-halfbin_X:i+halfbin_X+1,:]
#         _ = np.moveaxis(_, 1, -1) # (K, N, 1+2*halfbin_X)
#         X_output[:,i-halfbin_X,:] = _.reshape(X.shape[0], -1)
#     return X_output, y



def gaussian_filter1d_torch(x, sigma, axis=1, truncate=4.0):
    """
    1D Gaussian filter along a given axis for a 3D tensor x (K, T, N).
    sigma: standard deviation of the Gaussian.
    axis: axis along which to smooth (default 1 = time).
    """
    if sigma is None or sigma == 0:
        return x

    # Move the target axis to position 1 (time) and channels to 2 if needed
    # We want shape (K, T, N) internally for simplicity
    if axis == 1:
        x_work = x
    else:
        # General axis handling (rarely needed if you keep axis=1)
        perm = list(range(x.ndim))
        perm[1], perm[axis] = perm[axis], perm[1]
        x_work = x.permute(*perm)

    K, T, N = x_work.shape

    # Build 1D Gaussian kernel
    radius = int(truncate * sigma + 0.5)
    kernel_size = 2 * radius + 1
    device = x_work.device
    dtype = x_work.dtype

    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()                          # (kernel_size,)

    # Prepare for depthwise conv1d over time dimension
    # We want input as (batch, channels, length) = (K, N, T)
    x_conv = x_work.permute(0, 2, 1)                        # (K, N, T)

    # Reflect-pad along time (last) dimension
    x_padded = F.pad(x_conv, (radius, radius), mode="reflect")

    # Depthwise conv: same kernel on each channel
    kernel = kernel.view(1, 1, kernel_size)                 # (1, 1, k)
    kernel = kernel.repeat(N, 1, 1)                         # (N, 1, k)

    y_conv = F.conv1d(x_padded, kernel, groups=N)           # (K, N, T)
    y_work = y_conv.permute(0, 2, 1)                        # back to (K, T, N)

    # If axis was not 1, permute back
    if axis == 1:
        return y_work
    else:
        # undo the permutation
        inv_perm = list(range(x.ndim))
        inv_perm[1], inv_perm[axis] = inv_perm[axis], inv_perm[1]
        return y_work.permute(*inv_perm)


def preprocess_y(y: torch.Tensor, smooth_w: float):
    """
    y: (K, T, N) torch tensor
    Gaussian smooth along time axis (axis=1).
    """
    return gaussian_filter1d_torch(y, smooth_w, axis=1)


def preprocess_X(X: torch.Tensor, smooth_w: float, halfbin_X: int, smoothing: bool = True):
    """
    X: (K, T, N) torch tensor

    Steps:
      1) reflect-pad in time by halfbin_X
      2) Gaussian smooth along time
      3) concatenate neighboring time windows

    Returns:
      X_output: (K, T, N * (1 + 2 * halfbin_X))
    """
    K, T, N = X.shape

    # 1) reflect-pad along time axis
    # F.pad pads last dims first, so we move time to last, pad, then move back.
    X_time_last = X.permute(0, 2, 1)                        # (K, N, T)
    X_padded_tlast = F.pad(X_time_last, (halfbin_X, halfbin_X), mode="reflect")
    X_padded = X_padded_tlast.permute(0, 2, 1)              # (K, T + 2*halfbin_X, N)

    # 2) smooth padded X along time axis
    if smoothing:
        X_padded = gaussian_filter1d_torch(X_padded, smooth_w, axis=1)

    # 3) window concatenation
    win_size = 1 + 2 * halfbin_X
    X_output = X.new_zeros(K, T, N * win_size)

    for i in range(halfbin_X, T + halfbin_X):
        # window: (K, win_size, N)
        window = X_padded[:, i - halfbin_X : i + halfbin_X + 1, :]
        # (K, N, win_size)
        window = window.permute(0, 2, 1)
        # (K, N * win_size)
        window_flat = window.reshape(K, -1)
        X_output[:, i - halfbin_X, :] = window_flat

    return X_output


def preprocess_X_and_y(X: torch.Tensor, y: torch.Tensor, smooth_w: float, halfbin_X: int):
    """
    Tensor version of your original preprocess_X_and_y.

    X, y: (K, T, N) torch tensors
    """
    y_proc = preprocess_y(y, smooth_w)
    X_proc = preprocess_X(X, smooth_w, halfbin_X)
    return X_proc, y_proc