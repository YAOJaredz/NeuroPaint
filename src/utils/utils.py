import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utils.metric_utils import r2_score_torch
import torch.distributed as dist


def set_seed(seed):
    rank = dist.get_rank() if dist.is_initialized() else 0
    final_seed = seed + rank

    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
    np.random.seed(final_seed)
    random.seed(final_seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(final_seed))

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def plot_gt_pred(gt, pred, epoch=0, title=None):
    """
    gt: ground truth, shape (d, T), numpy array
    pred: prediction, shape (d, T), numpy array
    """
    # plot Ground Truth and Prediction in the same figure
    vmax = np.quantile(gt, 0.99)
    vmin = np.quantile(gt, 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary', vmin=vmin, vmax=vmax)

    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary', vmin=vmin, vmax=vmax)
    
    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}".format(epoch))

    if title:
        fig.suptitle("Epoch: {}, ".format(epoch) + title)

    return fig

def plot_neurons_r2(gt, pred, epoch=0, neuron_idx=[]):
    # Create one figure and axis for all plots
    fig, ax = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []  # To store R2 values for each neuron
    
    for neuron in neuron_idx:
        r2 = r2_score_torch(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax[neuron_idx.index(neuron)].plot(gt[:, neuron].cpu().numpy(), label="Ground Truth", color="blue")
        ax[neuron_idx.index(neuron)].plot(pred[:, neuron].cpu().numpy(), label="Prediction", color="red")
        ax[neuron_idx.index(neuron)].set_title("Neuron: {}, R2: {:.4f}".format(neuron, r2))
        ax[neuron_idx.index(neuron)].legend()
        # x label
        ax[neuron_idx.index(neuron)].set_xlabel("Time")
        # y label
        ax[neuron_idx.index(neuron)].set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Avg R2: {:.4f}".format(epoch, np.mean(r2_values)))
    return fig


# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["r2", "rsquared", "mse", "mae", "acc"], device="cpu"):
    results = {}
    if "r2" in metrics:
        gt   = gt.to(device).float()
        pred = pred.to(device).float()

        ss_res = ((gt - pred) ** 2).sum(dim=1)             # (B, N)
        gt_mean = gt.mean(dim=1, keepdim=True)             # (B, 1, N)
        ss_tot  = ((gt - gt_mean) ** 2).sum(dim=1)         # (B, N)

        r2 = torch.zeros_like(ss_res)                      # (B, N)
        valid = ss_tot > 0
        r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]

        r2_np = r2.detach().cpu().numpy()                  # (B, N)
        r2_masked_per_sample = np.ma.masked_invalid(r2_np)  # (B,)
        r2_masked_per_sample = r2_masked_per_sample.mean(axis=1).filled(np.nan)  # (B,)
        r2 = float(np.nanmean(r2_masked_per_sample))
        
        results["r2"] = r2
    if "rsquared" in metrics:
        r2_list = []
        for i in range(gt.shape[0]):
            r2 = r2_score_torch(y_true=gt[i], y_pred=pred[i], device=device) 
            r2_list.append(r2)
        r2 = np.mean(r2_list)
        results["rsquared"] = r2
    if "mse" in metrics:
        mse = torch.mean((gt - pred) ** 2)
        results["mse"] = mse
    if "mae" in metrics:
        mae = torch.mean(torch.abs(gt - pred))
        results["mae"] = mae
    if "acc" in metrics:
        acc = accuracy_score(gt.cpu().numpy(), pred.cpu().detach().numpy())
        results["acc"] = acc
    return results


def torch_corrcoef(x1, x2):
    #x1, x2: (# of samples, # of features)

    # mean of x1, x2
    x1_mean = torch.mean(x1, dim=0, keepdim=True)
    x2_mean = torch.mean(x2, dim=0, keepdim=True)

    x1_std = torch.std(x1, dim=0, keepdim=True)
    x2_std = torch.std(x2, dim=0, keepdim=True)

    x1_norm = (x1 - x1_mean) / x1_std
    x2_norm = (x2 - x2_mean) / x2_std

    # correlation coefficient
    corr = x1_norm.T @ x2_norm / (x1.shape[0]-1) # (n_features, n_features)

    return corr