import os, sys
from pathlib import Path

BASE_PATH = Path('results/mae_results')

CONFIG_PATH = Path('src/configs')
FINETURN_SESSIONS_TRAINER_CONFIG_PATH = CONFIG_PATH / 'finetune_sessions_trainer.yaml'

DATA_PATH = Path('/work/hdd/bdye/jyao7/data')
DATA_INFO_PATH = DATA_PATH / 'tables_and_infos'



# Linear MAE IBL path
IBL_DATA_PATH = DATA_PATH / 'loaded_ibl_data'

IBL_AREAOI = ["PO", "LP", "DG", "CA1", "VISa", "VPM", "APN", "MRN"]
IBL_N_LATENT_PATH = DATA_INFO_PATH / 'pr_max_dict_ibl.pkl'

CONFIG_LINEAR_MAE_PATH = CONFIG_PATH / 'mae_linear_config.yaml'

make_ibl_linear_model_path = lambda reg, consistency, smooth, n: Path("ibl_linear_mae") / f"with_reg_{reg}_consistency_{consistency}_smooth_{smooth}" / f"num_session_{n}"