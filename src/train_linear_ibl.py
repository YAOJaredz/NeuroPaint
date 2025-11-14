import torch
import numpy as np
import os
import warnings
from models.mae_linear import Linear_MAE, LinearStitcher
from loader.data_loader_ibl import make_loader
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed
from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make_unbalanced_ibl import make_trainer
import argparse
import pickle
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

def main(eids: list[str], with_reg: bool, consistency: bool):
    print(f"eids: {eids}")
    print(f"with_reg: {with_reg}")
    print(f"consistency: {consistency}")

    torch.cuda.empty_cache()
    
    multi_gpu = False
    load_previous_model = False

    base_path = BASE_PATH
    num_train_sessions = len(eids)
    train = True

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    num_epochs = 500
    batch_size = 32
    lr = 0.01
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

    with open(IBL_N_LATANT_PATH, 'rb') as f:
        pr_max_dict = pickle.load(f)

    meta_data['pr_max_dict'] = pr_max_dict
    
    trial_type_values = list(trial_type_dict.values())
    meta_data['trial_type_values'] = trial_type_values
    
    config = update_config(config, meta_data)

    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']
    test_dataloader = dataloader['test']

    print('check heldout info of dataset')
    print(heldout_info_list)
    
    if train:
        log_dir = \
            base_path / "train" / "ibl_linear_mae" / f"with_reg_{with_reg}" / f"consistency_{consistency}" / f"num_session_{num_train_sessions}"
        os.makedirs(log_dir, exist_ok=True)
    
        if not torch.cuda.is_available():
            print("CUDA is not available. Exiting.")
            exit()
        else:
            print("CUDA is available. Using GPU.")
        
        if config.wandb.use and accelerator.is_local_main_process and accelerator.process_index == 0:
            import wandb
            wandb.init(project=config.wandb.project, # type: ignore
                       dir="/root_folder/wandb", 
                    entity=config.wandb.entity, # type: ignore
                    config=config, 
                    name=f"lin_mae-ibl-reg_{with_reg}-consistency_{consistency}-sessions_{num_train_sessions}"
                    )
        
        model = Linear_MAE(config.model, **meta_data)
        
        if load_previous_model:
            previous_model_path = \
                base_path / "finetune" / "ibl_linear_mae" / f"with_reg_{with_reg}" / f"consistency_{consistency}" / \
                    f"num_session_{num_train_sessions}" / str(hash(tuple(eids))) / 'model_best.pt'
            state_dict = torch.load(previous_model_path, map_location=accelerator.device)['model']
            model.load_state_dict(state_dict)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['wd'], eps=config['optimizer']['eps'])
        lr_scheduler = OneCycleLR(
                        optimizer=optimizer,
                        total_steps=config['training']['num_epochs'] * len(train_dataloader) // config['optimizer']['gradient_accumulation_steps'],
                        max_lr=config['optimizer']['lr'],
                        pct_start=config['optimizer']['warmup_pct'],
                        div_factor=config['optimizer']['div_factor'],
                    )
        
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
        
        if consistency:
            encoder_stitcher_ema = LinearStitcher(meta_data['eids'], meta_data['area_ind_list_list'], meta_data['areaoi_ind'], config.model.encoder.stitcher) # type: ignore
            encoder_stitcher_ema = accelerator.prepare(encoder_stitcher_ema)
            for param in encoder_stitcher_ema.parameters():
                param.detach_() 
        
        trainer_kwargs = {
            "log_dir": log_dir,
            "accelerator": accelerator,
            "lr_scheduler": lr_scheduler,
            "config": config,
            "multi_gpu": multi_gpu,
            "with_reg": with_reg,
        }
        
        trainer = make_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            optimizer=optimizer,
            consistency = consistency,
            encoder_stitcher_ema = encoder_stitcher_ema if consistency else None,
            **trainer_kwargs,
            **meta_data
        )
        
        print(accelerator.device)
        
        # train loop
        trainer.train()

        print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eids', nargs='+', required=True, help='List of eids for training sessions')
    parser.add_argument('--with_reg', action='store_true', help='Whether to use regularization')
    parser.add_argument('--consistency', action='store_true', help='Whether to use consistency')
    args = parser.parse_args()

    main(args.eids, args.with_reg, args.consistency)

    # eids = [
    #     'f312aaec-3b6f-44b3-86b4-3a0c119c0438', 
    #     '51e53aff-1d5d-4182-a684-aba783d50ae5', 
    #     '88224abb-5746-431f-9c17-17d7ef806e6a'
    #     ]
    # with_reg = False
    # consistency = False
    # main(eids, with_reg, consistency)
