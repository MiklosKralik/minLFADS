import random

import numpy as np
import torch
from omegaconf import OmegaConf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('medium')

def get_default_config():
    conf = OmegaConf.create(
        {'log_dir' : './out',
         'model': None,
         'data': None,
         'train': None,
         }
    )
    return conf