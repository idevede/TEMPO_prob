# %%
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from torch.utils.data import Subset
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.TEMPO import TEMPO
# from models.T5 import T54TS
from models.ETSformer import ETSformer
import torch.distributions as dist


import numpy as np
import torch

# %%
import torch.nn as nn
from torch import optim
from numpy.random import choice

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import sys

from omegaconf import OmegaConf

import torch.nn as nn
from torch import optim
from numpy.random import choice

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import sys

from omegaconf import OmegaConf

# %%
def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

# %%
from omegaconf import OmegaConf

# Define the configuration dictionary
config = {
    "datasets": {
        "Solar": {
            "root_path": "/u/dcao1/workspace/Generative-TS-train_all/csdi_data/solar_nips/",
            "data_path": "test/test.csv",
            "data_name": "Solar",
            "data": "solar",
            "lradj": "type4",
            "features": "M",
            "target": "400001",
            "embed": "timeF",
            "freq": 0,
            "percent": 100
        }
    }
}

# Create an OmegaConf object from the dictionary
cfg = OmegaConf.create(config)


# %%

# Save the configuration to a YAML file
with open("./configs/custom_dataset.yml", "w") as f:
    OmegaConf.save(cfg, f)

print("Configuration saved to ./configs/custom_dataset.yml")

# %%
config = {
    "description": "TEMPO",
    "checkpoints": "checkpoints/lora_revin_6domain_checkpoints_1/",
    "task_name": "long_term_forecast",
    "prompt": 1,
    "num_nodes": 1,
    "seq_len": 168,
    "pred_len": 30,
    "label_len": 30,
    "decay_fac": 0.5,
    "learning_rate": 0.001,
    "batch_size": 256,
    "num_workers": 0,
    "train_epochs": 10,
    "lradj": "type3",
    "patience": 5,
    "gpt_layers": 6,
    "is_gpt": 1,
    "e_layers": 3,
    "d_model": 768,
    "n_heads": 4,
    "d_ff": 768,
    "dropout": 0.3,
    "enc_in": 7,
    "c_out": 1,
    "patch_size": 16,
    "kernel_size": 25,
    "loss_func": "mse",
    "pretrain": 1,
    "freeze": 1,
    "model": "TEMPO",
    "stride": 8,
    "max_len": -1,
    "hid_dim": 16,
    "tmax": 20,
    "itr": 3,
    "cos": 1,
    "equal": 1,
    "pool": False,
    "no_stl_loss": False,
    "stl_weight": 0.001,
    "config_path": "./configs/custom_dataset.yml",
    "datasets": "ETTm1,ETTh1,ETTm2,electricity,traffic,weather",
    "target_data": "Solar",
    "use_token": 0,
    "electri_multiplier": 1,
    "traffic_multiplier": 1,
    "embed": "timeF",
    "percent": 100,
    "model_id": "prob_TEMPO_6_prompt_learn_168_30_100_sl336_ll30_pl30_dm768_nh4_el3_gl6_df768_ebtimeF_itr0"
}

# Create an OmegaConf object from the dictionary
cfg = OmegaConf.create(config)

# Save the configuration to a YAML file
with open("configs/etth2_config.yml", "w") as f:
    OmegaConf.save(cfg, f)

print("Configuration written to config.yml")

# %%
config = get_init_config(cfg.config_path)

# %%
SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

# %%
device = torch.device('cpu')

# %%
if cfg.model == 'PatchTST':
    model = PatchTST(cfg, device)
    model.to(device)
elif cfg.model == 'DLinear':
    model = DLinear(cfg, device)
    model.to(device)
elif cfg.model == 'TEMPO':
    model = TEMPO(cfg, device)
    model.to(device)
elif cfg.model == 'T5':
    model = T54TS(cfg, device)
    model.to(device)
elif 'ETSformer' in cfg.model:
    model = ETSformer(cfg, device)
    model.to(device)
else:
    model = GPT4TS(cfg, device)


params = model.parameters()
model_optim = torch.optim.Adam(params, lr=cfg.learning_rate)

early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)
if cfg.loss_func == 'mse':
    criterion = nn.MSELoss()
elif cfg.loss_func == 'smape':
    class SMAPE(nn.Module):
        def __init__(self):
            super(SMAPE, self).__init__()
        def forward(self, pred, true):
            return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
    criterion = SMAPE()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=cfg.tmax, eta_min=1e-8)

# %%
model_path = os.path.join(cfg.checkpoints, cfg.model_id)
best_model_path = model_path + '/checkpoint.pth'
print(best_model_path)
model.load_state_dict(torch.load(best_model_path,  map_location=torch.device(device)), strict=False)
print("------------------------------------")

# %%
cfg.data = config['datasets'][cfg.target_data].data
cfg.root_path = config['datasets'][cfg.target_data].root_path
cfg.data_path = config['datasets'][cfg.target_data].data_path
cfg.data_name = config['datasets'][cfg.target_data].data_name
cfg.features = config['datasets'][cfg.target_data].features
cfg.freq = config['datasets'][cfg.target_data].freq
cfg.target = config['datasets'][cfg.target_data].target
cfg.embed = config['datasets'][cfg.target_data].embed
cfg.percent = config['datasets'][cfg.target_data].percent
cfg.lradj = config['datasets'][cfg.target_data].lradj
if cfg.freq == 0:
    cfg.freq = 'h'
cfg.seq_len = 168 + 24
cfg.pred_len = 24 
test_data, test_loader = data_provider(cfg, 'test')

# %%
preds = []
trues = []
# mases = []

# Initialize accumulators for errors
total_mae = 0
total_mse = 0
n_samples = 0
itr = 0

cfg.seq_len = 168
cfg.pred_len = 24
model.eval()
# import pdb; pdb.set_trace()
with torch.no_grad():
    for i in range(len(test_data)):
        #enumerate(test_loader): #tqdm(enumerate(test_loader), total=len(test_loader)):
        data = test_data[i]
        seq_x, text_embedding, observed_mask = data[0].to(device), data[1].to(device), data[2].to(device)
        seq_x = seq_x.float().to(device)
        batch_x = seq_x[:, :cfg.seq_len].unsqueeze(-1) # 137, 168, 1
        batch_y = seq_x[:,cfg.seq_len:cfg.seq_len+cfg.pred_len]

                        
        outputs, _ = model(batch_x, 0, None, None, None) #+ model(seq_seasonal, ii) + model(seq_resid, ii)

        mu, sigma, nu = outputs[0], outputs[1], outputs[2]
        # Create the Student's t-distribution with the predicted parameters
        student_t = dist.StudentT(df=nu, loc=mu, scale=sigma)

        # Generate 30 samples for each prediction
        num_samples = 35
        probabilistic_forecasts = student_t.rsample((num_samples,))

        # The shape of probabilistic_forecasts will be (num_samples, batch_size, pred_length)
        print(probabilistic_forecasts.shape)


# %%


