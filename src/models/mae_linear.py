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
# from models.mae_with_hemisphere_embed_and_diff_dim_per_area import StitchDecoder
from utils.mask import random_mask
from utils.utils_linear import preprocess_X_and_y

@dataclass
class MAE_Output(ModelOutput):
    loss: Optional[torch.Tensor] = None
    regularization_loss: Optional[torch.Tensor] = None
    preds: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None

class LinearStitcher(nn.Module):
    def __init__(self, 
                session_list: list[str],
                area_ind_list_list: list[list[str]],
                areaoi_ind: np.ndarray,
                config: DictConfig):
        super().__init__()
        self.eids = list(map(str, session_list))
        self.areaoi_ind = np.array(areaoi_ind, dtype=int)
        self.halfbin_X: int = config.halfbin_X         # type: ignore
        self.n_channels_per_region: int = config.n_channels_per_region            # type: ignore
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
        x = x.transpose(2, 1).float()  # (B, N, T)
        
        region_emb_x = torch.zeros(B, len(self.areaoi_ind) * self.n_channels_per_region * 2, T, device=x.device, dtype=torch.float32)
        for area in self.areaoi_ind:
            if f"{eid}_{area}" not in self.session_area_linears:
                continue
            sa_embed = self.session_area_linears[f"{eid}_{area}"]
            neuron_mask = torch.where(neuron_regions == area)[0]
            x_area = x[:, neuron_mask, :]  # (B, n_neurons_in_area, T)

            area_ind = self.region_to_idx[area]
            # FIXME: shape mismatch here
            print(x_area.size(), area_ind, area_ind * self.n_channels_per_region, (area_ind + 1) * self.n_channels_per_region, region_emb_x.size())
            region_emb_x[:, area_ind * self.n_channels_per_region:(area_ind + 1) * self.n_channels_per_region, :] = sa_embed(x_area)

        # hemi_emb = self.hemisphere_embed(is_left).expand(B, 1, self.n_emb)
        # emb_x = torch.cat([region_emb_x, hemi_emb], dim=1)  # (B, num_regions*n_emb + n_emb, T)
        emb_x = region_emb_x  # (B, num_regions*n_emb, T)

        return emb_x.transpose(2, 1)  # (B, T, num_regions*n_emb)

class LinearEncoder(nn.Module):
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

        self.stitcher = LinearStitcher(kwargs['eids'], kwargs['area_ind_list_list'], kwargs['areaoi_ind'], config.stitcher)

        self.U_layer = nn.Linear(self.stitcher.output_dim, self.hidden_size)
        self.V_layer = nn.Linear(self.hidden_size, np.sum(list(self.n_lat.values()), dtype=int))
        
    def forward(
            self, 
            spikes:           torch.Tensor,  # (B, T, N)
            neuron_regions:   Optional[torch.LongTensor] = None,  # (B, N)
            is_left:         Optional[torch.LongTensor] = None, # (B, N)
            trial_type:       Optional[torch.LongTensor] = None, # (B, )
            masking_mode:     Optional[str] = None,
            eid:              Optional[str] = None,
            force_mask:       Optional[dict] = None
    ) -> torch.FloatTensor:                     # (B, T_kept*R_kept+1, hidden_size)
        B, T, N = spikes.size()
        x = self.stitcher(spikes, eid, neuron_regions, is_left)  # (B, T, num_regions*n_emb + n_emb)
        
        x_region = x.view(x.size(0), x.size(1), self.n_regions, self.stitcher.n_channels_per_region)  # (B, T, R, n_emb)

        x_masked, mask, mask_R, mask_T, ids_restore_R, ids_restore_T  = random_mask(masking_mode, x_region, r_ratio = self.r_ratio)

        R_kept = int(torch.sum(mask_R[0, :] == 0))
        T_kept = int(torch.sum(mask_T[0, :] == 0))
        x_masked = x_masked.view(B, T_kept, R_kept, -1)
        
        if ids_restore_R is not None:
            R_mask = self.n_regions - R_kept
            mask_tokens_region = torch.zeros(B, R_mask, self.stitcher.n_channels_per_region, device=x.device)
            x_masked = torch.cat((x_masked, mask_tokens_region.unsqueeze(1).repeat(1, T_kept, 1, 1)), dim=2)
            x_ = torch.gather(x_masked, dim=2, index=ids_restore_R.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, self.stitcher.n_channels_per_region))

        x_ = x_.view(B, T, -1)  # (B, T, R_kept*n_emb)
        # x = torch.cat([x_, x[:, :, -self.stitcher.n_emb:]], dim=2)  # (B, T, R_kept*n_emb + n_emb)
        x = x_

        x = self.V_layer(self.U_layer(x))  # (B, T, sum(pr_max_per_region))
        return x

class LinearDecoder(nn.Module):
    def __init__(self,
                 session_list: list[str],
                 n_latents_dict: dict[int, int],
                 area_ind_list_list: list[list[str]],
                 areaoi_ind: np.ndarray):
        super().__init__()
        self.session_list = session_list
        self.n_latents_dict = n_latents_dict
        self.area_ind_list_list = area_ind_list_list
        self.areaoi_ind = areaoi_ind
        
        area_session_linears = {}
        for session_ind, area_ind_list in zip(session_list, area_ind_list_list):
            for area in self.areaoi_ind:
                n_neurons = int(np.sum(area_ind_list == area))
                if n_neurons == 0:
                    continue
                area_session_linears[f"{session_ind}_{area}"] = nn.Linear(n_latents_dict[area], n_neurons)
        self.session_area_linears = nn.ModuleDict(area_session_linears)
        
        self.lat_areas = np.concatenate([np.repeat(area, n_latents_dict[area]) for area in areaoi_ind]).tolist()

    def forward(self, x: torch.Tensor, eid: str, neuron_regions: torch.LongTensor) -> torch.Tensor:
        B, T, _ = x.size()
        output = torch.zeros(B, T, neuron_regions.size(1), device=x.device)
        for area in self.areaoi_ind:

            if f"{eid}_{area}" not in self.session_area_linears:
                continue

            sa_embed = self.session_area_linears[f"{eid}_{area}"]
            output[:, neuron_regions == area] = sa_embed(
                x[:, :, self.lat_areas == area]
            )

        return output

class Linear_MAE(nn.Module):
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()
        
        self.smooth_w: int = config.smooth_w           # type: ignore
        self.halfbin_X: int = config.encoder.stitcher.halfbin_X         # type: ignore
        
        if 'pr_max_dict' not in kwargs:
            lat_dict = {k: 20 for k in kwargs['areaoi_ind']}
            kwargs['pr_max_dict'] = lat_dict
            
        # Build encoder
        self.encoder = LinearEncoder(config.encoder, **kwargs)
        
        self.decoder = LinearDecoder(
            kwargs['eids'],
            kwargs['pr_max_dict'],
            kwargs['area_ind_list_list'],
            kwargs['areaoi_ind'],
        )

        self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=True)
        
    def forward(
        self, 
        spikes:           torch.Tensor,  # (bs, seq_len, N)
        spikes_timestamps: torch.LongTensor,   # (bs, seq_len)
        neuron_regions:   torch.LongTensor,   # (bs, N)
        is_left:         Optional[torch.LongTensor] = None, # (bs, N)
        trial_type:       Optional[torch.LongTensor] = None, # (bs, )
        masking_mode:     Optional[str] = None,
        eid:              Optional[str] = None,
        with_reg:       Optional[bool] = False,
        force_mask:       Optional[dict] = None
    ) -> MAE_Output:  

        spikes_np = spikes.detach().cpu().numpy()
        targets_np = spikes_np.copy()
        
        spikes_np, targets_np = preprocess_X_and_y(
            spikes_np,
            targets_np,
            smooth_w=self.smooth_w,
            halfbin_X=self.halfbin_X
        )

        spikes = torch.from_numpy(spikes_np).to(spikes.device)
        targets = torch.from_numpy(targets_np).to(spikes.device)
        
        # Expand neuron_regions to match the padded spikes shape
        # FIXME: Type error here
        neuro_regions_targets: torch.LongTensor = neuron_regions.clone().long()
        neuron_regions_expanded = neuron_regions.unsqueeze(-1).repeat(1, 1, 1 + 2 * self.halfbin_X)  # (B, N, 1+2*halfbin_X)
        neuron_regions = neuron_regions_expanded.reshape(neuron_regions.size(0), -1)  # type: ignore # (B, N*(1+2*halfbin_X))

        x = self.encoder.forward(
            spikes=spikes, neuron_regions=neuron_regions, is_left=is_left, eid=eid, 
            trial_type=trial_type, masking_mode=masking_mode, force_mask=force_mask
        )

        regularization_loss = torch.mean(torch.abs(torch.diff(x, dim=1)))*0.1

        outputs = self.decoder.forward(x, str(eid), neuro_regions_targets)

        loss = torch.nanmean(self.loss_fn(outputs, targets))

        if with_reg:
            loss += regularization_loss
        
        return MAE_Output(
            loss = loss,
            regularization_loss = regularization_loss,
            preds = outputs,
            targets = targets
        )