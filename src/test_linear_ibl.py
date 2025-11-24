import sys

sys.path.append('src')
import torch
import numpy as np
import os
from models.mae_with_hemisphere_embed_and_diff_dim_per_area import MAE_with_region_stitcher
from models.mae_linear import Linear_MAE
from loader.data_loader_ibl import make_loader
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed, move_batch_to_device
from accelerate import Accelerator
from utils.metric_utils import Poisson_fraction_deviance_explained, get_deviance_explained
from constants import BASE_PATH, CONFIG_LINEAR_MAE_PATH, FINETURN_SESSIONS_TRAINER_CONFIG_PATH, IBL_N_LATANT_PATH

import argparse
import pickle

def get_pred_fr_and_dfe(
    factors_region: torch.Tensor, 
    spikes_region: torch.Tensor, 
    n_trial_train: int, 
    device: torch.device, 
    eid: str
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    '''
    factors_region: B x T x C
    spikes_region: B x T x N
    n_trial_train: int
    device: torch.device

    return: fr_pred_test, spikes_region_test, dfe_test
    '''

    factors_region_train = factors_region[:n_trial_train]
    spikes_region_train = spikes_region[:n_trial_train]

    factors_region_test = factors_region[n_trial_train:]
    spikes_region_test = spikes_region[n_trial_train:]

    fr_pred_train, weight, bias, dfe_train = get_deviance_explained(factors_region_train, spikes_region_train, device, verbose=True)
    fr_pred_test = torch.exp(factors_region_test @ weight + bias[None, None, :])

    if torch.any(torch.isnan(fr_pred_test)):
        print('nan in fr_pred_test using recorded data', ', session ', eid)
        return fr_pred_test, spikes_region_test, None

    dfe_test = Poisson_fraction_deviance_explained(fr_pred_test.cpu().detach().numpy(), spikes_region_test.cpu().detach().numpy())
    
    return fr_pred_test, spikes_region_test, dfe_test

def main(eids: list[str], with_reg: bool, consistency: bool, override: bool = False):
    print(f"eids: {eids}")
    print(f"with_reg: {with_reg}")
    print(f"consistency: {consistency}")

    torch.cuda.empty_cache()

    base_path = BASE_PATH
    num_train_sessions = len(eids)

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    num_epochs = 500
    batch_size = 64
    lr = 1e-3
    use_wandb = True
    
    kwargs = {
        "model": f"include:{CONFIG_LINEAR_MAE_PATH}",
    }
    
    config = config_from_kwargs(kwargs)
    config = update_config(str(FINETURN_SESSIONS_TRAINER_CONFIG_PATH), config)
    
    config['model']['encoder']['masker']['mask_mode'] = mask_mode
    config['training']['num_epochs'] = num_epochs
    config['wandb']['use'] = use_wandb
    config['wandb']['project'] = 'lin-mae-ibl'
    config['optimizer']['lr'] = lr
    
    meta_data = {}

    dataloader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict = \
        make_loader(eids, batch_size, seed = config['seed'])
    set_seed(config['seed'])
    
    areaoi_ind = np.array(areaoi_ind)
    
    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = [eid_idx for eid_idx, eid in enumerate(eids)]
    
    with open(IBL_N_LATANT_PATH, 'rb') as f:
        pr_max_dict = pickle.load(f)
    
    for k, v in pr_max_dict.items():
        pr_max_dict[k] = int(v)
    meta_data['pr_max_dict'] = pr_max_dict
    
    trial_type_values = list(trial_type_dict.values())
    meta_data['trial_type_values'] = trial_type_values
    
    config = update_config(config, meta_data)
    
    test_dataloader = dataloader['test']
    
    accelerator = Accelerator()
    
    model_path = \
        base_path / "train" / "ibl_linear_mae" / f"with_reg_{with_reg}_consistency_{consistency}" / f"num_session_{num_train_sessions}" / 'model_best_eval_loss.pt'
    model = Linear_MAE(config.model, **meta_data)
    
    state_dict = torch.load(model_path, map_location=accelerator.device)['model']
    model.load_state_dict(state_dict)
    model = accelerator.prepare(model)
    
    model.eval()
    
    save_path = \
        base_path / "eval" / "ibl_linear_mae" / f"with_reg_{with_reg}_consistency_{consistency}" / f"num_session_{num_train_sessions}"
    
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Results saved to: {save_path}")
    
    
    dfe_no_mask_from_record_to_heldout = {}
    dfe_no_mask_pred = {}
    fr_pred_test_save = {}
    fr_pred_test_save_baseline = {}
    spike_test_save = {}
    
    if not override and os.path.exists(save_path / 'mae_no_mask_fr_pred_test_save.pkl'):
        fr_pred_test_save = pickle.load(open(save_path / 'mae_no_mask_fr_pred_test_save.pkl', 'rb'))
        spike_test_save = pickle.load(open(save_path / 'spike_test_save.pkl', 'rb'))
        fr_pred_test_save_baseline = pickle.load(open(save_path / 'baseline_no_mask_fr_pred_test_save.pkl', 'rb'))
        dfe_no_mask_pred = pickle.load(open(save_path / 'dfe_no_mask_pred.pkl', 'rb'))
        dfe_no_mask_from_record_to_heldout = pickle.load(open(save_path / 'dfe_no_mask_from_record_to_heldout.pkl', 'rb'))
    
    with torch.no_grad():
        device = accelerator.device
        n_area = len(areaoi_ind)
        for batch in test_dataloader:
            batch = move_batch_to_device(batch, device)
            B, T, _ = batch['spikes_data'].size()
            eid = batch['eid'][0].item()
            print(eid)
            
            if eid in dfe_no_mask_pred:
                continue
            
            neuron_regions = batch['neuron_regions'][0] #(N,) area_ind_list
            trial_type = batch['trial_type'] # (B,) trial_type
            area_ind_unique_tensor = neuron_regions.unique()
            R = len(area_ind_unique_tensor) #number of regions

            mask_T = torch.zeros([B,T], dtype=torch.bool, device=device)
            mask_R = torch.zeros([B,R], dtype=torch.bool, device=device)
            ids_restore_R = None
            ids_restore_T = None
            mask = torch.zeros([B,T,R], dtype=torch.bool, device=device)
            
            force_mask = {'mask': mask, 
                        'mask_R': mask_R, 
                        'mask_T': mask_T, 
                        'ids_restore_R': ids_restore_R, 
                        'ids_restore_T': ids_restore_T}
            
            area_ind_list_full = batch['neuron_regions_full'][0] # (N_all,) 
            
            model_output = model.forward(
                spikes=batch['spikes_data'],
                spikes_timestamps=batch['spikes_timestamps'],
                neuron_regions=batch['neuron_regions'],
                is_left=batch['is_left'],
                trial_type=trial_type,
                masking_mode=mask_mode,
                eid=eid,
                force_mask=force_mask,
                compute_loss=False
            )
            factors_pred = model_output.preds
            print(factors_pred.size())

            dfe_no_mask_pred[eid] = {}
            dfe_no_mask_from_record_to_heldout[eid] = {}

            fr_pred_test_save[eid] = {}
            spike_test_save[eid] = {}
            fr_pred_test_save_baseline[eid] = {}

            n_trial_train = int(B*0.6)

            for idx, area_ind in enumerate(areaoi_ind):
                #no mask MAE
                factors_region = factors_pred[:,:, neuron_regions == area_ind]
                spikes_region = batch['spikes_data_full'][:,:,area_ind_list_full==area_ind]
                if spikes_region.size(2)<=5:
                    continue
                
                fr_pred_test, spikes_region_test, dfe_test = get_pred_fr_and_dfe(factors_region, spikes_region, n_trial_train, device, eid)

                dfe_no_mask_pred[eid][area_ind] = dfe_test
                fr_pred_test_save[eid][area_ind] = fr_pred_test.cpu().detach().numpy()
                spike_test_save[eid][area_ind] = spikes_region_test.cpu().detach().numpy()

                #baseline model
                factors_region = batch['spikes_data']                
                if factors_region.size(2)<=5 or torch.any(torch.isnan(factors_region)):
                    continue

                fr_pred_test, spikes_region_test, dfe_test = get_pred_fr_and_dfe(factors_region, spikes_region, n_trial_train, device, eid)
                dfe_no_mask_from_record_to_heldout[eid][area_ind] = dfe_test
                fr_pred_test_save_baseline[eid][area_ind] = fr_pred_test.cpu().detach().numpy()
            
            pickle.dump(fr_pred_test_save, open(save_path / 'mae_no_mask_fr_pred_test_save.pkl', 'wb'))
            pickle.dump(spike_test_save, open(save_path / 'spike_test_save.pkl', 'wb'))
            pickle.dump(fr_pred_test_save_baseline, open(save_path / 'baseline_no_mask_fr_pred_test_save.pkl', 'wb'))
            pickle.dump(dfe_no_mask_pred, open(save_path / 'dfe_no_mask_pred.pkl', 'wb'))
            pickle.dump(dfe_no_mask_from_record_to_heldout, open(save_path / 'dfe_no_mask_from_record_to_heldout.pkl', 'wb'))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eids', type=str, nargs='+', help='List of EIDs')
    parser.add_argument('--with_reg', action='store_true', help='Use regularization')
    parser.add_argument('--consistency', action='store_true', help='Use consistency')

    args = parser.parse_args()
    main(args.eids, args.with_reg, args.consistency)
    
    # eids = [
    #     'f312aaec-3b6f-44b3-86b4-3a0c119c0438', 
    #     '51e53aff-1d5d-4182-a684-aba783d50ae5', 
    #     '88224abb-5746-431f-9c17-17d7ef806e6a'
    #     ]
    # with_reg = False
    # consistency = False
    # main(eids, with_reg, consistency, override=True)