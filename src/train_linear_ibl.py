import torch
import numpy as np
import os
import warnings
from models.mae_with_hemisphere_embed_and_diff_dim_per_area import MAE_with_region_stitcher, NeuralStitcher_cross_att
from loader.data_loader_ibl import make_loader
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make_unbalanced_ibl import make_trainer
import argparse
import pickle
import yaml
from constants import (
    CONFIG_LINEAR_MAE_PATH, 
    BASE_PATH,
    FINETURN_SESSIONS_TRAINER_CONFIG_PATH,
    IBL_N_LATANT_PATH
)

warnings.simplefilter("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_IGNORE_COPY_ERR"] = "true"

def main(eids, with_reg):
    print(f"eids: {eids}")
    print(f"with_reg: {with_reg}")
    
    torch.cuda.empty_cache()
    
    multi_gpu = False
    consistency = True
    load_previous_model = False

    base_path = BASE_PATH
    num_train_sessions = len(eids)
    train = True

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    num_epochs = 1000
    batch_size = 16
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
    
    meta_data = {}
    
    if multi_gpu:
        print("Using multi-gpu training.")
        from accelerate.utils import DistributedDataParallelKwargs
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs]) 

        global_batch_size = batch_size * accelerator.num_processes
        config['optimizer']['lr'] = 1e-3 * global_batch_size / 256

    else:
        accelerator = Accelerator()
        
    print(f"Accelerator device: {accelerator.device}")
    
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    
    dataloader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict = make_loader(
        eids, batch_size, seed=config['seed'], distributed=multi_gpu, rank=rank, world_size=world_size
    )
    set_seed(config['seed'])

    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = [eid_idx for eid_idx, eid in enumerate(eids)]

    with open(IBL_N_LATANT_PATH, 'r') as f:
        pr_max_dict = yaml.safe_load(f)

    meta_data['pr_max_dict'] = pr_max_dict
    
    trial_type_values = list(trial_type_dict.values())
    meta_data['trial_type_values'] = trial_type_values
    
    config = update_config(config, meta_data)
    
    config = update_config(config, meta_data) # so that everything is saved in the config file

    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']
    #test_dataloader = dataloader['test']

    print('check heldout info of dataset')
    print(heldout_info_list)

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--eids', nargs='+', required=True, help='List of eids for training sessions')
    # parser.add_argument('--with_reg', action='store_true', help='Whether to use regularization')
    # args = parser.parse_args()
    
    # main(args.eids, args.with_reg)
    
    eids = [
        'f312aaec-3b6f-44b3-86b4-3a0c119c0438', 
        '51e53aff-1d5d-4182-a684-aba783d50ae5', 
        '88224abb-5746-431f-9c17-17d7ef806e6a', 
        'c7248e09-8c0d-40f2-9eb4-700a8973d8c8', 
        '4b00df29-3769-43be-bb40-128b1cba6d35'
        ]
    with_reg = True
    main(eids, with_reg)
    