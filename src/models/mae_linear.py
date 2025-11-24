import os, sys
sys.path.append('src')
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pickle

from models.model_output import ModelOutput
from utils.config_utils import DictConfig
from models.mae_with_hemisphere_embed_and_diff_dim_per_area import StitchDecoder
from utils.mask import random_mask
from utils.utils_linear import preprocess_X, preprocess_y, poisson_nll_loss
from constants import IBL_AREAOI

@dataclass
class MAE_Output(ModelOutput):
    loss: Optional[torch.Tensor] = None
    regularization_loss: Optional[torch.Tensor] = None
    preds: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None

class MaskedLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mask_map: list[tuple[np.ndarray, np.ndarray]]):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Create mask for weights
        mask = torch.zeros_like(self.linear.weight)
        output_mask = torch.zeros(output_dim)
        
        for in_inds, out_inds in mask_map:
            if len(in_inds) > 0 and len(out_inds) > 0:
                # Convert to tensors
                out_t = torch.as_tensor(out_inds, dtype=torch.long)
                in_t = torch.as_tensor(in_inds, dtype=torch.long)
                
                # Set weight mask: connect specified inputs to specified outputs
                mask[out_t[:, None], in_t] = 1.0
                
                # Set output mask: these outputs are active
                output_mask[out_t] = 1.0
                
        self.register_buffer('mask', mask)
        self.register_buffer('output_mask', output_mask)
        
        # Initialize weights to respect the mask
        with torch.no_grad():
            mask = getattr(self, 'mask')
            output_mask = getattr(self, 'output_mask')
            self.linear.weight.data *= mask
            self.linear.bias.data *= output_mask

    def forward(self, x: torch.Tensor):
        # Apply mask to weights during forward pass
        mask = getattr(self, 'mask')
        output_mask = getattr(self, 'output_mask')
        masked_weight = self.linear.weight * mask
        masked_bias = self.linear.bias * output_mask
        
        out = F.linear(x, masked_weight, masked_bias)
        return out

class LinearStitcher(nn.Module):
    """ Maps neuron activity to region embeddings for session-area combinations. """
    def __init__(self, 
                session_list: list[str],
                area_ind_list_list: list[list[str]],
                areaoi_ind: np.ndarray,
                config: DictConfig):
        super().__init__()
        self.eids = list(map(str, session_list))
        self.areaoi_ind = np.array(areaoi_ind, dtype=int)
        self.smoothing: bool = config['smoothing']
        self.halfbin_X: int = config['halfbin_X']
        self.smooth_w: float = config['smooth_w']
        self.n_channels_per_region: int = config['n_channels_per_region']
        self.output_dim: int = len(areaoi_ind) * self.n_channels_per_region
        
        self.session_modules = nn.ModuleDict()
        
        for session_ind, area_ind_list in zip(session_list, area_ind_list_list):
            expanded_area_inds = np.repeat(area_ind_list, 1 + 2 * self.halfbin_X)
            total_neurons_expanded = len(expanded_area_inds)
            
            mask_map = []
            for i, area in enumerate(self.areaoi_ind):
                # Input indices: neurons in this area
                in_inds = np.where(expanded_area_inds == area)[0]
                
                # Output indices: channels for this area
                start_idx = i * self.n_channels_per_region
                end_idx = (i + 1) * self.n_channels_per_region
                out_inds = np.arange(start_idx, end_idx)
                
                if len(in_inds) > 0:
                    mask_map.append((in_inds, out_inds))
            
            self.session_modules[str(session_ind)] = MaskedLinear(
                total_neurons_expanded, self.output_dim, mask_map
            )

    def forward(self, x: torch.Tensor, eid: str, neuron_regions: torch.Tensor, is_left: torch.Tensor) -> torch.Tensor:
        
        x = preprocess_X(x.clone(), smooth_w=self.smooth_w, halfbin_X=self.halfbin_X, smoothing=self.smoothing)
        
        return self.session_modules[eid](x)

class LinearEncoder(nn.Module):
    """ Maps neuron activity to latent representations. """
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size: int = config.hidden_size       # type: ignore
        self.r_ratio: float = config.masker.r_ratio             # type: ignore
        
        self.n_regions = len(kwargs['areaoi_ind'])
        if 'pr_max_dict' in kwargs:
            self.n_lat = {k: kwargs['pr_max_dict'][k] for k in kwargs['areaoi_ind']}
        else:
            self.n_lat = {k: 20 for k in kwargs['areaoi_ind']}

        self.stitchers = LinearStitcher(
            session_list=kwargs['eids'], 
            area_ind_list_list=kwargs['area_ind_list_list'], 
            areaoi_ind=kwargs['areaoi_ind'], 
            config=config.stitcher
        )

        self.U_layer = nn.Linear(self.stitchers.output_dim, self.hidden_size)
        self.V_layer = nn.Linear(self.hidden_size, np.sum(list(self.n_lat.values()), dtype=int))
        
    def forward(
            self, 
            spikes:           torch.Tensor,  # (B, T, N)
            neuron_regions:   Optional[torch.Tensor] = None,  # (B, N)
            is_left:         Optional[torch.LongTensor] = None, # (B, N)
            trial_type:       Optional[torch.LongTensor] = None, # (B, )
            masking_mode:     Optional[str] = None,
            eid:              Optional[str] = None,
            force_mask:       Optional[dict] = None
    ) -> torch.FloatTensor:                     # (B, T_kept*R_kept+1, hidden_size)
        B, T, N = spikes.size()
        
        x = self.stitchers(x=spikes, eid=eid, neuron_regions=neuron_regions, is_left=is_left)  # (B, T, num_regions*n_emb + n_emb)
        
        x_region = x.view(x.size(0), x.size(1), self.n_regions, self.stitchers.n_channels_per_region)  # (B, T, R, n_emb)

        x_masked, mask, mask_R, mask_T, ids_restore_R, ids_restore_T  = random_mask(masking_mode, x_region, r_ratio = self.r_ratio)

        R_kept = int(torch.sum(mask_R[0, :] == 0))
        T_kept = int(torch.sum(mask_T[0, :] == 0))
        x_masked = x_masked.view(B, T_kept, R_kept, -1)
        
        if ids_restore_R is not None:
            R_mask = self.n_regions - R_kept
            mask_tokens_region = torch.zeros(B, R_mask, self.stitchers.n_channels_per_region, device=x.device)
            x_masked = torch.cat((x_masked, mask_tokens_region.unsqueeze(1).repeat(1, T_kept, 1, 1)), dim=2)
            x_ = torch.gather(x_masked, dim=2, index=ids_restore_R.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, self.stitchers.n_channels_per_region))

        x_ = x_.view(B, T, -1)  # (B, T, R_kept*n_emb)
        # x = torch.cat([x_, x[:, :, -self.stitcher.n_emb:]], dim=2)  # (B, T, R_kept*n_emb + n_emb)
        x = x_

        x = self.V_layer(self.U_layer(x))  # (B, T, sum(pr_max_per_region))
        return x

class LinearDecoder(nn.Module):
    """ Maps latent representations to neuron activity for session-area combinations. """
    def __init__(self,
                 session_list: list[str],
                 n_latents_dict: dict[int, int],
                 area_ind_list_list: list[list[str]],
                 areaoi_ind: np.ndarray):
        super().__init__()
        self.session_list = session_list
        self.n_latents_dict = {k: int(n_latents_dict[k]) for k in n_latents_dict}
        self.area_ind_list_list = area_ind_list_list
        self.areaoi_ind = np.array(areaoi_ind, dtype=int)
        
        # Latent structure (Input to decoder)
        # Concatenate latents for all regions in order of areaoi_ind
        self.lat_areas = np.concatenate([np.repeat(area, self.n_latents_dict[area]) for area in areaoi_ind])
        input_dim = len(self.lat_areas)
        
        self.session_modules = nn.ModuleDict()
        
        for session_ind, area_ind_list in zip(session_list, area_ind_list_list):
            # Output dim: number of neurons in session
            output_dim = len(area_ind_list)
            
            mask_map = []
            for area in self.areaoi_ind:
                # Input indices: latents for this area
                in_inds = np.where(self.lat_areas == area)[0]
                
                # Output indices: neurons for this area
                out_inds = np.where(area_ind_list == area)[0]
                
                if len(out_inds) > 0:
                    mask_map.append((in_inds, out_inds))
            
            self.session_modules[str(session_ind)] = MaskedLinear(
                input_dim, output_dim, mask_map
            )

    def forward(self, x: torch.Tensor, eid: str, neuron_regions: torch.Tensor) -> torch.Tensor:
        return self.session_modules[eid](x)
    

class Linear_MAE(nn.Module):
    """ Linear MAE model for neural activity reconstruction. """
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()

        self.smooth_w: float = config.encoder.stitcher.smooth_w           # type: ignore

        if 'pr_max_dict' not in kwargs:
            lat_dict = {k: 20 for k in kwargs['areaoi_ind']}
            kwargs['pr_max_dict'] = lat_dict
            
        # Build encoder
        self.encoder = LinearEncoder(config.encoder, **kwargs)
        
        self.decoder = LinearDecoder(
            session_list=kwargs['eids'],
            n_latents_dict=kwargs['pr_max_dict'],
            area_ind_list_list=kwargs['area_ind_list_list'],
            areaoi_ind=kwargs['areaoi_ind'],
        )

        self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=True)
        
    def forward(
        self, 
        spikes:           torch.Tensor,  # (bs, seq_len, N)
        spikes_timestamps: torch.LongTensor,   # (bs, seq_len)
        neuron_regions:   torch.Tensor,   # (bs, N)
        is_left:         Optional[torch.LongTensor] = None, # (bs, N)
        trial_type:       Optional[torch.LongTensor] = None, # (bs, )
        masking_mode:     Optional[str] = None,
        eid:              Optional[torch.Tensor] = None,
        with_reg:       Optional[bool] = False,
        force_mask:       Optional[dict] = None,
        compute_loss:     Optional[bool] = True
    ) -> MAE_Output:  

        B, T, N = spikes.size()
        if eid is None:
            raise ValueError("eid must be provided")
        elif isinstance(eid, torch.Tensor):
            eid_str = str(eid.item())
        else:
            eid_str = str(eid)
        
        spikes_targets = spikes.clone()

        x = self.encoder.forward(
            spikes=spikes, neuron_regions=neuron_regions, is_left=is_left, eid=eid_str,
            trial_type=trial_type, masking_mode=masking_mode, force_mask=force_mask
        )

        outputs = self.decoder.forward(x, eid_str, neuron_regions)
        outputs = torch.clamp(outputs, max=5.3)
        
        if not compute_loss:
            return MAE_Output(
                loss = None,
                regularization_loss = None,
                preds = outputs,
                targets = spikes_targets
            )

        loss = torch.nanmean(self.loss_fn(outputs, spikes_targets))
        
        if with_reg:
            regularization_loss = torch.nanmean(torch.abs(torch.diff(x, dim=1)))
            loss += regularization_loss * 0.1
        
        return MAE_Output(
            loss = loss,
            regularization_loss = regularization_loss,
            preds = outputs,
            targets = spikes_targets
        )