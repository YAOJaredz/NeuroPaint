import os, sys
sys.path.append('src')
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
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
        
        self.region_to_idx = {r: i for i,r in enumerate(areaoi_ind)}
        
        session_area_linears = {}
        for session_ind, area_ind_list in zip(session_list, area_ind_list_list):
            for area in self.areaoi_ind:
                n_neurons = int(np.sum(area_ind_list == area)) * (1 + 2 * self.halfbin_X)
                if n_neurons == 0:
                    continue
                session_area_linears[f"{session_ind}_{area}"] = nn.Linear(n_neurons, self.n_channels_per_region)

        self.session_area_linears = nn.ModuleDict(session_area_linears)

        # self.hemisphere_embed = nn.Embedding(2, self.n_emb)


    def forward(self, x: torch.Tensor, eid: str, neuron_regions: torch.Tensor, is_left: torch.Tensor) -> torch.Tensor:
        B, T, N = x.size()
        
        x = preprocess_X(x.clone(), smooth_w=self.smooth_w, halfbin_X=self.halfbin_X, smoothing=self.smoothing)
        neuron_regions = neuron_regions.repeat_interleave(1 + 2 * self.halfbin_X, dim=1)  # (B, N*(1+2*halfbin_X))

        region_emb_x = torch.zeros(B, T, len(self.areaoi_ind) * self.n_channels_per_region, device=x.device, dtype=torch.float32)
        for area in self.areaoi_ind:
            if f"{eid}_{area}" not in self.session_area_linears:
                continue
            sa_embed = self.session_area_linears[f"{eid}_{area}"]
            neuron_mask = torch.where(neuron_regions[0] == area)[0]
            x_area = x[:, :, neuron_mask]  # (B, T, n_neurons_in_area)

            area_ind = self.region_to_idx[area]
            region_emb_x[:, :, area_ind * self.n_channels_per_region:(area_ind + 1) * self.n_channels_per_region] = sa_embed(x_area)

        # hemi_emb = self.hemisphere_embed(is_left).expand(B, 1, self.n_emb)
        # emb_x = torch.cat([region_emb_x, hemi_emb], dim=1)  # (B, num_regions*n_emb + n_emb, T)
        emb_x = region_emb_x  # (B, T, num_regions*n_emb)

        return emb_x  # (B, T, num_regions*n_emb)

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
        
        area_session_linears = {}
        for session_ind, area_ind_list in zip(session_list, area_ind_list_list):
            for area in self.areaoi_ind:
                n_neurons = int(np.sum(area_ind_list == area))
                if n_neurons == 0:
                    continue
                area_session_linears[f"{session_ind}_{area}"] = nn.Linear(self.n_latents_dict[area], n_neurons)
        self.session_area_linears = nn.ModuleDict(area_session_linears)

        self.lat_areas = np.concatenate([np.repeat(area, self.n_latents_dict[area]) for area in areaoi_ind]).tolist()

    def forward(self, x: torch.Tensor, eid: str, neuron_regions: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        output = torch.zeros(B, T, neuron_regions.size(1), device=x.device)
        for area in self.areaoi_ind:

            if f"{eid}_{area}" not in self.session_area_linears:
                continue

            sa_embed = self.session_area_linears[f"{eid}_{area}"]
            neuro_mask = torch.where(neuron_regions[0] == area)[0]
            x_area = x[:, :, self.lat_areas == area]  # (B, T, n_latents_in_area)
            output[:, :, neuro_mask] = sa_embed(x_area)

        return output
    

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
        force_mask:       Optional[dict] = None
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
        
        if not self.training:
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