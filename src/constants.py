import os, sys
from pathlib import Path

BASE_PATH = Path('results/mae_results')

CONFIG_PATH = Path('src/configs')
FINETURN_SESSIONS_TRAINER_CONFIG_PATH = CONFIG_PATH / 'finetune_sessions_trainer.yaml'


# Linear MAE IBL path
IBL_N_LATANT_PATH = CONFIG_PATH / 'pr_max_dict_ibl.yaml'

CONFIG_LINEAR_MAE_PATH = CONFIG_PATH / 'mae_linear_config.yaml'
