import torch
from accelerate import Accelerator
import numpy as np
import os
import warnings
from models.mae_with_region_cross_att_stitcher_simple_decoder import MAE_with_region_stitcher, NeuralStitcher_cross_att
from loader.chaotic_rnn_loader import *
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make import make_trainer
import argparse


warnings.simplefilter("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_IGNORE_COPY_ERR"] = "true"

def main(eids, with_reg):
    print(f"eids: {eids}")
    print(f"with_reg: {with_reg}")

    torch.cuda.empty_cache()

    #%% set arguments
    multi_gpu = True
    consistency = False
    load_previous_model = False

    base_path = 'results/'
    num_train_sessions = len(eids)
    train = True

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    region_channel_num = 24 # number of region channels in decoder
    region_channel_num_encoder = 48 # number of region channels in encoder
    unit_embed_dim = 50
    n_layers = 5

    num_epochs = 1000
    batch_size = 16
    use_wandb = True

    #%%
    kwargs = {
        "model": f"include:src/configs/mae_with_hemisphere_embed_and_diff_dim_per_area.yaml",
    }

    config = config_from_kwargs(kwargs)
    config = update_config("src/configs/finetune_sessions_trainer.yaml", config)

    config['model']['encoder']['masker']['mask_mode'] = mask_mode
    config['model']['decoder']['stitcher']['n_channels_per_region'] = region_channel_num
    config['model']['encoder']['stitcher']['n_channels_per_region'] = region_channel_num_encoder
    config['model']['encoder']['stitcher']['unit_embed_dim'] = unit_embed_dim
    config['training']['num_epochs'] = num_epochs
    config['wandb']['use'] = use_wandb
    config['wandb']['project'] = 'mae-chaotic-rnn'
    
    config['model']['encoder']['transformer']['n_layers'] = n_layers

    meta_data = {}
    
    # set accelerator
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


    dataloader, num_neurons, datasets, area_ind_list_list, record_info_list = make_chaotic_rnn_loader(eids, batch_size=batch_size, seed = config.seed, distributed=multi_gpu, rank=rank, world_size=world_size)
    set_seed(config.seed)   

    areaoi_ind = np.array([0,1,2,3,4])

    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = eids

    config = update_config(config, meta_data) # so that everything is saved in the config file

    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']
    #test_dataloader = dataloader['test']

    print('check basic info of dataset')
    print(record_info_list)


    #%%
    if train:
        log_dir = os.path.join(base_path, 
                        "train",
                        "chaotic_rnn_g3_01_log_fr_max_3_min_neg_3",
                        "with_reg_{}".format(with_reg),
                        "consistency_{}".format(consistency),
                        "n_layers_{}".format(n_layers),
                        "encoder_factors_{}".format(region_channel_num_encoder),
                        "unit_embed_dim_{}".format(unit_embed_dim),
                        "num_session_{}".format(num_train_sessions),
                        "region_factors_{}".format(region_channel_num)
                        )
        
        os.makedirs(log_dir, exist_ok=True)

        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU.")
            exit()
        else:
            print("CUDA is available. Using GPU.")

            # set accelerator
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

        if config.wandb.use and accelerator.is_local_main_process and accelerator.process_index == 0:
            import wandb
            wandb.init(project=config.wandb.project, 
                    entity=config.wandb.entity, 
                    config=config, 
                    name="chaotic_rnn_g3_01_log_fr_max_3_min_neg_3_with_reg_{}_consistency_{}_n_layers_{}_num_session_{}_region_factors_encoder_{}_factors_{}".format(with_reg, consistency, n_layers, num_train_sessions, region_channel_num_encoder, region_channel_num)
                    )
        
        model = MAE_with_region_stitcher(config.model, **meta_data)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
        lr_scheduler = OneCycleLR(
                        optimizer=optimizer,
                        total_steps=config.training.num_epochs*len(train_dataloader) //config.optimizer.gradient_accumulation_steps,
                        max_lr=config.optimizer.lr,
                        pct_start=config.optimizer.warmup_pct,
                        div_factor=config.optimizer.div_factor,
                    )
        
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
        

        if consistency:
            encoder_stitcher_ema = NeuralStitcher_cross_att(meta_data['eids'], meta_data['area_ind_list_list'], meta_data['areaoi_ind'], config.model.encoder.stitcher)
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
        # check device
        print(accelerator.device)
        # train loop
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eids', type=int, nargs='+', help='List of EIDs')
    parser.add_argument('--with_reg', action='store_true', help='Use regularization')

    args = parser.parse_args()
    main(args.eids, args.with_reg)       
        

