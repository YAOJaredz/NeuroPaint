import sys
sys.path.append('src')
import torch
import numpy as np
import os
from models.mae_with_hemisphere_embed_and_diff_dim_per_area import MAE_with_region_stitcher
from loader.data_loader_ibl import *
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed, move_batch_to_device
from accelerate import Accelerator
from utils.metric_utils import Poisson_fraction_deviance_explained, get_deviance_explained

import argparse
import pickle

def main(eids, with_reg):
    print(f"eids: {eids}")
    print(f"with_reg: {with_reg}")

    consistency = True

    base_path = 'results/mae_results'
    num_train_sessions = len(eids)

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    region_channel_num_encoder = 48 # number of region channels in encoder
    unit_embed_dim = 50
    n_layers = 5

    num_epochs = 1000
    batch_size = 16
    use_wandb = False

    #%%
    kwargs = {
        "model": f"include:src/configs/mae_with_hemisphere_embed_and_diff_dim_per_area_ibl.yaml",
    }

    config = config_from_kwargs(kwargs)
    config = update_config("src/configs/finetune_sessions_trainer.yaml", config)

    config['model']['encoder']['masker']['mask_mode'] = mask_mode
    config['model']['encoder']['stitcher']['n_channels_per_region'] = region_channel_num_encoder
    config['model']['encoder']['stitcher']['unit_embed_dim'] = unit_embed_dim
    config['training']['num_epochs'] = num_epochs
    config['wandb']['use'] = use_wandb
    config['wandb']['project'] = 'mae-ibl'
    
    config['model']['encoder']['transformer']['n_layers'] = n_layers

    meta_data = {}

    dataloader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict = make_loader(eids, batch_size, seed = config.seed)
    set_seed(config.seed)  

    areaoi_ind = np.array(areaoi_ind)

    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = [eid_idx for eid_idx, eid in enumerate(eids)]

    #load pr_max_dict
    pr_max_dict_path = '/work/hdd/bdye/jyao7/data/tables_and_infos/pr_max_dict_ibl.pkl'
    with open(pr_max_dict_path, 'rb') as f:
        pr_max_dict = pickle.load(f)

    for k, v in pr_max_dict.items():
        pr_max_dict[k] = int(v)

    meta_data['pr_max_dict'] = pr_max_dict

    trial_type_values = list(trial_type_dict.values())
    meta_data['trial_type_values'] = trial_type_values

    config = update_config(config, meta_data) # so that everything is saved in the config file

    test_dataloader = dataloader['test']

    accelerator = Accelerator()


    model_path = f'{base_path}/train/ibl_wo_trial_type_embed/with_reg_{with_reg}_2check/consistency_{consistency}/n_layers_{n_layers}/encoder_factors_{region_channel_num_encoder}/unit_embed_dim_{unit_embed_dim}/num_session_{num_train_sessions}/model_best_eval_loss.pt'
    model = MAE_with_region_stitcher(config.model, **meta_data)

    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    model = accelerator.prepare(model)

    model.eval()

    save_path = f'{base_path}/eval/ibl_wo_trial_type_embed/with_reg_{with_reg}_2check/consistency_{consistency}/n_layers_{n_layers}/encoder_factors_{region_channel_num_encoder}/unit_embed_dim_{unit_embed_dim}/num_session_{num_train_sessions}/dfe/'
    
    os.makedirs(save_path, exist_ok=True)
    print(save_path)

    def get_pred_fr_and_dfe(factors_region, spikes_region, n_trial_train, device):
        '''
        factors_region: B x T x C
        spikes_region: B x T x N
        n_trial_train: int
        device: torch.device

        return: fr_pred_test, dfe_test
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
    

    #%%
    dfe_no_mask_from_record_to_heldout = {}
    dfe_no_mask_pred = {}
    fr_pred_test_save = {}
    fr_pred_test_save_baseline = {}
    spike_test_save = {}

    R_all = len(areaoi_ind)


    #if files exist in the save path, load the results
    if os.path.exists(save_path+'mae_no_mask_fr_pred_test_save.npy'):
        fr_pred_test_save = np.load(save_path+'mae_no_mask_fr_pred_test_save.npy', allow_pickle=True).item()
        spike_test_save = np.load(save_path+'spike_test_save.npy', allow_pickle=True).item()
        fr_pred_test_save_baseline = np.load(save_path+'baseline_no_mask_fr_pred_test_save.npy', allow_pickle=True).item()
        dfe_no_mask_pred = np.load(save_path+'dfe_no_mask_pred.npy', allow_pickle=True).item()
        dfe_no_mask_from_record_to_heldout = np.load(save_path+'dfe_no_mask_from_record_to_heldout.npy', allow_pickle=True).item()

    set_seed(config.seed)  
    with torch.no_grad():
        for batch in test_dataloader:
            device = accelerator.device
            batch = move_batch_to_device(batch, device)
            B = batch['spikes_data'].size(0)
            T = batch['spikes_data'].size(1)
            eid = batch['eid'][0].item()
            print(eid)

            if eid in dfe_no_mask_pred:
                continue

            neuron_regions = batch['neuron_regions'][0] #(N,) area_ind_list
            trial_type = batch['trial_type'] # (B,) trial_type
            area_ind_unique_tensor = neuron_regions.unique()
            R = len(area_ind_unique_tensor) #number of regions

            #don't mask any data
            mask_T = torch.zeros([B,T], dtype=torch.bool, device=device)
            mask_R = torch.zeros([B,R], dtype=torch.bool, device=device)
            ids_restore_R = None
            ids_restore_T = None
            mask = torch.zeros([B,T,R], dtype=torch.bool, device=device)

            force_mask ={'mask': mask, 
                        'mask_R': mask_R, 
                        'mask_T': mask_T, 
                        'ids_restore_R': ids_restore_R, 
                        'ids_restore_T': ids_restore_T}

            area_ind_list_full = batch['neuron_regions_full'][0] # (N_all,) 

            x, mask, mask_R, mask_T, ids_restore_R, ids_restore_T = model.encoder(batch['spikes_data'], 
                                                                                batch['spikes_timestamps'],
                                                                                area_ind_unique_tensor, 
                                                                                batch['neuron_regions'],
                                                                                batch['is_left'],
                                                                                batch['trial_type'], 
                                                                                mask_mode, 
                                                                                batch['eid'][0], 
                                                                                force_mask)
            
            # Decode
            factors_pred_dict = {}
            for area_ind in areaoi_ind:
                x_tmp = x[:,:, areaoi_ind==area_ind,:].squeeze(dim=2) # (B, T, H)
                factors_pred_dict[str(area_ind)] = model.stitch_decoder.linear_layer_dict[str(area_ind)](x_tmp) # (B, T, n_channels_this_region)

            dfe_no_mask_pred[eid] = {}
            dfe_no_mask_from_record_to_heldout[eid] = {}

            fr_pred_test_save[eid] = {}
            spike_test_save[eid] = {}
            fr_pred_test_save_baseline[eid] = {}

            n_trial_train = int(B*0.6)

            for area_ind_i, area_ind in enumerate(areaoi_ind):
                #no mask MAE
                factors_region = factors_pred_dict[str(area_ind)]
                spikes_region = batch['spikes_data_full'][:,:,area_ind_list_full==area_ind]
                if spikes_region.size(2)<=5:
                    continue
                
                fr_pred_test, spikes_region_test, dfe_test = get_pred_fr_and_dfe(factors_region, spikes_region, n_trial_train, device)

                dfe_no_mask_pred[eid][area_ind] = dfe_test
                fr_pred_test_save[eid][area_ind] = fr_pred_test.cpu().detach().numpy()
                spike_test_save[eid][area_ind] = spikes_region_test.cpu().detach().numpy()

                #baseline model
                factors_region = batch['spikes_data']                
                if factors_region.size(2)<=5:
                    continue

                if np.any(np.isnan(factors_region.cpu().detach().numpy())):
                    print('nan in spikes_data', ', session ', eid)
                    continue

                fr_pred_test, spikes_region_test, dfe_test = get_pred_fr_and_dfe(factors_region, spikes_region, n_trial_train, device)
                dfe_no_mask_from_record_to_heldout[eid][area_ind] = dfe_test
                fr_pred_test_save_baseline[eid][area_ind] = fr_pred_test.cpu().detach().numpy()

                #break

            #break
            np.save(save_path+'mae_no_mask_fr_pred_test_save.npy', fr_pred_test_save)
            np.save(save_path+'spike_test_save.npy', spike_test_save)
            np.save(save_path+'baseline_no_mask_fr_pred_test_save.npy', fr_pred_test_save_baseline)
            np.save(save_path+'dfe_no_mask_pred.npy', dfe_no_mask_pred)
            np.save(save_path+'dfe_no_mask_from_record_to_heldout.npy', dfe_no_mask_from_record_to_heldout)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eids', type=str, nargs='+', help='List of EIDs')
    parser.add_argument('--with_reg', action='store_true', help='Use regularization')

    args = parser.parse_args()
    main(args.eids, args.with_reg)    
