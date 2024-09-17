from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from torch.utils.data import Subset
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.TEMPO import TEMPO
from models.T5 import T54TS
from models.ETSformer import ETSformer


import numpy as np
import torch
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

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, default='weather_GTP4TS_multi-debug')
parser.add_argument('--checkpoints', type=str, default='/l/users/defu.cao/checkpoints_multi_dataset/')
parser.add_argument('--task_name', type=str, default='long_term_forecast')


parser.add_argument('--prompt', type=int, default=0)
parser.add_argument('--num_nodes', type=int, default=1)


parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.9)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type3') # for what
parser.add_argument('--patience', type=int, default=5)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='GPT4TS_multi')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
parser.add_argument('--pool', action='store_true', help='whether use prompt pool')
parser.add_argument('--no_stl_loss', action='store_true', help='whether use prompt pool')

parser.add_argument('--stl_weight', type=float, default=0.01)
parser.add_argument('--config_path', type=str, default='./data_config.yml')
parser.add_argument('--datasets', type=str, default='exchange')
parser.add_argument('--target_data', type=str, default='ETTm1')

parser.add_argument('--use_token', type=int, default=0)
parser.add_argument('--electri_multiplier', type=int, default=1)
parser.add_argument('--traffic_multiplier', type=int, default=1)
parser.add_argument('--embed', type=str, default='timeF')

#args = parser.parse_args([])
args = parser.parse_args()
config = get_init_config(args.config_path)

args.itr = 1

print(args)

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




mses = []
maes = []
for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    # if args.freq == 0:
    #     args.freq = 'h'

    device = torch.device('cuda:0')


    
    train_data_name = args.datasets.split(',')
    print(train_data_name)
    train_datas = []
    val_datas = []
    min_sample_num = sys.maxsize
    for dataset_singe in args.datasets.split(','):
        print(dataset_singe)
        args.data = config['datasets'][dataset_singe].data
        args.root_path = config['datasets'][dataset_singe].root_path
        args.data_path = config['datasets'][dataset_singe].data_path
        args.data_name = config['datasets'][dataset_singe].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][dataset_singe].freq
        args.target = config['datasets'][dataset_singe].target
        args.embed = config['datasets'][dataset_singe].embed
        args.percent = config['datasets'][dataset_singe].percent
        args.lradj = config['datasets'][dataset_singe].lradj
        if args.freq == 0:
            args.freq = 'h'
       
        print("dataset: ", args.data)
        train_data, train_loader = data_provider(args, 'train')
        if dataset_singe not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:   
            min_sample_num = min(min_sample_num, len(train_data))
        
        args.percent = 2
        vali_data, vali_loader = data_provider(args, 'val')
        args.percent = 100

        # train_datas.append(train_data)
        val_datas.append(vali_data)

    for dataset_singe in args.datasets.split(','):
        print(dataset_singe)
        args.data = config['datasets'][dataset_singe].data
        args.root_path = config['datasets'][dataset_singe].root_path
        args.data_path = config['datasets'][dataset_singe].data_path
        args.data_name = config['datasets'][dataset_singe].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][dataset_singe].freq
        args.target = config['datasets'][dataset_singe].target
        args.embed = config['datasets'][dataset_singe].embed
        args.percent = config['datasets'][dataset_singe].percent
        args.lradj = config['datasets'][dataset_singe].lradj
        if args.freq == 0:
            args.freq = 'h'
        # if args.freq != 'h':
        #     args.freq = SEASONALITY_MAP[test_data.freq]
        #     print("freq = {}".format(args.freq))

        print("dataset: ", args.data)
        train_data, train_loader = data_provider(args, 'train')
        if dataset_singe not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and args.equal == 1: 
            train_data = Subset(train_data, choice(len(train_data), min_sample_num))
        if args.electri_multiplier>1 and args.equal == 1 and dataset_singe in ['electricity']: 
            train_data = Subset(train_data, choice(len(train_data), int(min_sample_num*args.electri_multiplier)))
        if args.traffic_multiplier>1 and args.equal == 1 and dataset_singe in ['traffic']: 
            train_data = Subset(train_data, choice(len(train_data), int(min_sample_num*args.traffic_multiplier)))
        train_datas.append(train_data)

    if len(train_datas) > 1:
        train_data = torch.utils.data.ConcatDataset([train_datas[0], train_datas[1]])
        vali_data = torch.utils.data.ConcatDataset([val_datas[0], val_datas[1]])
        for i in range(2,len(train_datas)):
            train_data = torch.utils.data.ConcatDataset([train_data, train_datas[i]])
            
            vali_data = torch.utils.data.ConcatDataset([vali_data, val_datas[i]])

        # import pdb; pdb.set_trace()
        print("Way1",len(train_data))
        num_samples = int(len(vali_data) * 0.01)

        # Ensure at least one sample is selected
        num_samples = max(1, num_samples)

        # Use random_split to split the dataset
        subset_indices = torch.utils.data.random_split(vali_data, [num_samples, len(vali_data) - num_samples])[0]

        # Create the final validation subset
        vali_data = torch.utils.data.Subset(vali_data, subset_indices.indices)
        print("Way1, vali_data",len(vali_data))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        args.data = config['datasets'][args.target_data].data
        args.root_path = config['datasets'][args.target_data].root_path
        args.data_path = config['datasets'][args.target_data].data_path
        args.data_name = config['datasets'][args.target_data].data_name
        args.features = config['datasets'][dataset_singe].features
        args.freq = config['datasets'][args.target_data].freq
        args.target = config['datasets'][args.target_data].target
        args.embed = config['datasets'][args.target_data].embed
        args.percent = config['datasets'][args.target_data].percent
        args.lradj = config['datasets'][args.target_data].lradj
        if args.freq == 0:
            args.freq = 'h'
        test_data, test_loader = data_provider(args, 'test')

    time_now = time.time()
    train_steps = len(train_loader) #190470 -52696

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    elif args.model == 'TEMPO':
        model = TEMPO(args, device)
        model.to(device)
    elif args.model == 'T5':
        model = T54TS(args, device)
        model.to(device)
    elif 'ETSformer' in args.model:
        model = ETSformer(args, device)
        model.to(device)
    else:
        model = GPT4TS(args, device)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    elif args.loss_func == 'prob':
        import torch.distributions as dist

        def criterion(y_true, y_pred):
            # import pdb; pdb.set_trace()
            y_true = y_true.squeeze()
            mu, sigma, nu = y_pred[0], y_pred[1], y_pred[2]
            # Create the Student's t-distribution
            student_t = dist.StudentT(df=nu, loc=mu, scale=sigma)
            # Calculate the negative log-likelihood
            nll = -student_t.log_prob(y_true)
            return nll.mean()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(enumerate(train_loader),total = len(train_loader)):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            seq_trend = seq_trend.float().to(device)
            seq_seasonal = seq_seasonal.float().to(device)
            seq_resid = seq_resid.float().to(device)

            # print(seq_seasonal.shape)
            if args.model == 'TEMPO' or 'multi' in args.model:
                outputs, loss_local = model(batch_x, ii, seq_trend, seq_seasonal, seq_resid) #+ model(seq_seasonal, ii) + model(seq_resid, ii)
            elif 'former' in args.model:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, ii)
            # outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device).squeeze()

            loss = criterion(batch_y, outputs) 
            if args.model == 'GPT4TS_multi' or args.model == 'TEMPO':
                if not args.no_stl_loss:
                    loss += args.stl_weight*loss_local
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()
            # break
        
        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
       
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    

#     best_model_path = path + '/' + 'checkpoint.pth'
#     model.load_state_dict(torch.load(best_model_path), strict=False)
#     print("------------------------------------")
#     mse, mae = test(model, test_data, test_loader, args, device, ii)
#     torch.cuda.empty_cache()
#     print('test on the ' + str(args.target_data) + ' dataset: mse:' + str(mse) + ' mae:' + str(mae))
    
#     mses.append(mse)
#     maes.append(mae)
# print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
# print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

