import torch
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from torch.nn import MSELoss, L1Loss
import os
from matplotlib import pyplot as plt
from imputation_metrics import mse_withmask, mae_withmask, calc_quantile_CRPS, calc_quantile_CRPS_sum
import sys

def plot_results(synthetic_data, target_mask, gt_data, seq_len, 
                 low_q, mid_q, high_q, path = '', pred_len=24):
    i = 0 #3000
    for i in range(len(synthetic_data)):
        feat_dim = synthetic_data.shape[2]
        if feat_dim>100:
            feat_dim = 100
        synthetic_data = synthetic_data[:,:,:gt_data.shape[1],:seq_len]
        # import pdb; pdb.set_trace()
        target_mask = target_mask[:,:gt_data.shape[1],:seq_len]
        gt_data = gt_data[:,:,:seq_len]

        fig, axes = plt.subplots(nrows=feat_dim, ncols=1, figsize=(10, 70))
        for feat_idx in range(feat_dim):
            df_x = pd.DataFrame({"x": np.arange(0, seq_len), "val": gt_data[i, feat_idx, :],
                                "y": (1-target_mask)[i, feat_idx, :]})
            df_x = df_x[df_x.y!=0]

            df_o = pd.DataFrame({"x": np.arange(0, seq_len), "val": gt_data[i, feat_idx, :],
                                "y": target_mask[i, feat_idx, :]})
            # df_o = df_o[df_o.y!=0]
            axes[feat_idx].plot(df_o.x, df_o.val, color='b',  linestyle='solid', label='label')
            # axes[feat_idx].plot(df_x.x, df_x.val, color='r', marker='x', linestyle='None')
            axes[feat_idx].plot(range(seq_len-pred_len, seq_len), mid_q[i, feat_idx, -pred_len:], color='g', linestyle='solid', label='Diffusion-TS')
            axes[feat_idx].fill_between(range(seq_len-pred_len, seq_len), low_q[i, feat_idx,-pred_len:],high_q[i,feat_idx, -pred_len:], color='g', alpha = 0.3)
            plt.setp(axes[feat_idx], ylabel='value')
            if feat_idx == feat_dim-1:
                plt.setp(axes[-1], xlabel='time')
            # axes[feat_idx].set_ylim(-3, 3)
        plt.legend()
        plt.savefig(path+str(i) + '.png')
        # plt.show()
        plt.close()


# Solar
# df_raw = pd.read_csv("/home/minghao.fu/df_work/Generative-TS/datasets/requests_minute_train.csv", header=None)

def calculate_results_ecl(data_path, data_type, seq_len, pre_model, imputed_folder, pred_len = 24):
    # df_raw.replace(to_replace=-200, value=np.nan, inplace=True)
    test = True #False#True #False #True

    if test:
        imputed_path = os.path.join(imputed_folder,"test_samples.npy")
        mask_path = os.path.join(imputed_folder,"test_masks.npy")
        gt_path = os.path.join(imputed_folder,"test_gt.npy")
    else:
        imputed_path = os.path.join(imputed_folder,"train_samples.npy")
        mask_path = os.path.join(imputed_folder,"train_masks.npy")
        gt_path = os.path.join(imputed_folder,"train_gt.npy")
    synthetic_data = np.load(imputed_path)
    gt_data = np.load(gt_path)
    target_mask = np.load(mask_path)

    
    synthetic_data = np.reshape(synthetic_data,newshape=(gt_data.shape[0],-1,target_mask.shape[1],gt_data.shape[2]))
    synthetic_data = synthetic_data[:,:,:gt_data.shape[1],:gt_data.shape[2]]
    target_mask = target_mask[:,:gt_data.shape[1],:gt_data.shape[2]]

    low_q = np.quantile(synthetic_data,0.05,axis=1)
    high_q = np.quantile(synthetic_data,0.95,axis=1)
    mid_q = np.quantile(synthetic_data,0.5,axis=1)

    # plot_results(synthetic_data, target_mask, gt_data, seq_len, low_q, mid_q, high_q, 
    #              path = imputed_folder + '/forecasting_examine_together', pred_len = pred_len)
    
    import pickle
    paths='./csdi_data/electricity_nips/meanstd.pkl'
    with open(paths, 'rb') as f:
        mean_data, std_data = pickle.load(f)
    # synthetic_data  = np.swapaxes(synthetic_data, -1, -2)
    # gt_data = np.swapaxes(gt_data, -1, -2)
    # target_mask =np.swapaxes(target_mask, -1, -2)
    # CRPS_sum_1 = calc_quantile_CRPS_sum(torch.Tensor(gt_data),torch.Tensor(synthetic_data[:,:10]),torch.Tensor(target_mask),mean_scaler=mean_data,scaler=std_data)
    # print(CRPS_sum_1)
    unormzalized_gt_data = []
    for g in gt_data:
        unormzalized_gt_data.append(np.transpose(np.transpose(g)*std_data+mean_data))
    unormzalized_gt_data = np.array(unormzalized_gt_data)
    unormalized_synthetic_data = []
    for i in range(len(synthetic_data)):
        for j in range(len(synthetic_data[i])):
            s = synthetic_data[i][j]
            unormalized_synthetic_data.append(np.transpose(np.transpose(s)*std_data+mean_data))
    unormalized_synthetic_data = np.array(unormalized_synthetic_data).reshape(synthetic_data.shape[0],-1,synthetic_data.shape[2],synthetic_data.shape[3])
    
    unormzalized_gt_data = np.swapaxes(unormzalized_gt_data, -1, -2)
    unormalized_synthetic_data = np.swapaxes(unormalized_synthetic_data, -1, -2)
    target_mask = np.swapaxes(target_mask, -1, -2)
    print("all:" + data_type + " CRPS_sum:")
    # import pdb; pdb.set_trace()
    CRPS_sum_1 = calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data[:,:10]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    CRPS_sum_2 = calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data[:,10:20]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    CRPS_sum_3 = calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data[:,20:30]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    
    CRPS_sum_mean = np.mean([CRPS_sum_1, CRPS_sum_2, CRPS_sum_3])
    CRPS_sum_std = np.std([CRPS_sum_1, CRPS_sum_2, CRPS_sum_3])
    print("Mean CRPS_sum:", CRPS_sum_mean)
    print("Standard Deviation CRPS_sum:", CRPS_sum_std)
    return CRPS_sum_1

def calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder, pred_len = 24):
    # df_raw.replace(to_replace=-200, value=np.nan, inplace=True)
    
    if data_type in ['solar', 'traffic', 'exchange', 'Taxi']:
        df_raw = pd.read_csv(data_path,  header=None)
        border = [0,int(len(df_raw)-7*24-5*24),len(df_raw)]
        cols_data = df_raw.columns[:] #change for different data
        df_data = df_raw[cols_data]
        data = df_data.values
    elif data_type in ['Wiki']:
        df_raw = pd.read_csv(data_path, header=None)
        border = [0,int(len(df_raw)-2*30-5*30),len(df_raw)]
        cols_data = df_raw.columns[:] #
        df_data = df_raw[cols_data]
        data = df_data.values
        
    elif data_type not in ['pde']:
        df_raw = pd.read_csv(data_path).set_index('time')
        border = [0,int(len(df_raw)-1*24-14*24),len(df_raw)]
        cols_data = df_raw.columns[:] #change for different data

        df_data = df_raw[cols_data]
        data = df_data.values
    test = True #False#True #False #True
    # if not test:
    #     data_x = data[border[0]:border[1]]
    # else:
    #     data_x = data[border[1]:border[2]]

    # orig_data = []
    # observed_mask = []

    # length = len(data_x)-seq_len+1
    # for i in range(length):
    #     orig_data.append(data_x[i:i+seq_len])
    #     observed_mask.append((~np.isnan(data_x[i:i+seq_len])).astype(int))
    # # imputed_folder = "../Generative-TS/results/beijingpm25_wo_text_onecolumn_cf"

    if test:
        imputed_path = os.path.join(imputed_folder,"test_samples.npy")
        mask_path = os.path.join(imputed_folder,"test_masks.npy")
        gt_path = os.path.join(imputed_folder,"test_gt.npy")
    else:
        imputed_path = os.path.join(imputed_folder,"train_samples.npy")
        mask_path = os.path.join(imputed_folder,"train_masks.npy")
        gt_path = os.path.join(imputed_folder,"train_gt.npy")
    synthetic_data = np.load(imputed_path)
    gt_data = np.load(gt_path)
    target_mask = np.load(mask_path)

    scaler = StandardScaler()
    if data_type in ['solar', 'Wiki', 'exchange']:
        scaler.fit(df_data[border[0]:border[1]].values)
    elif data_type in ['traffic', 'Taxi']:
        scaler.fit(df_data.values)
    elif data_type not in ['pde']:
        scaler.fit(df_data.values)
    # min_max_scaler = MinMaxScaler()
    # min_max_scaler.fit(df_data[border[0]:border[1]].values)

    synthetic_data = np.reshape(synthetic_data,newshape=(gt_data.shape[0],-1,target_mask.shape[1],gt_data.shape[2]))
    # synthetic_data = synthetic_data[:,:,:gt_data.shape[1],:seq_len]
    # # import pdb; pdb.set_trace()
    # target_mask = target_mask[:,:gt_data.shape[1],:seq_len]
    # gt_data = gt_data[:,:,:seq_len]
    
    # seq_len = 192
    synthetic_data = synthetic_data[:,:,:gt_data.shape[1],:seq_len]
    # import pdb; pdb.set_trace()
    target_mask = target_mask[:,:gt_data.shape[1],:seq_len]
    gt_data = gt_data[:,:,:seq_len]
    # print(synthetic_data.shape, gt_data.shape, target_mask.shape)

    low_q = np.quantile(synthetic_data,0.05,axis=1)
    high_q = np.quantile(synthetic_data,0.95,axis=1)
    mid_q = np.quantile(synthetic_data,0.5,axis=1)

    # # # if data_type not in ['Wiki']:
    # plot_results(synthetic_data, target_mask, gt_data, seq_len, low_q, mid_q, high_q, 
    #              path = imputed_folder + '/forecasting_examine_together.png', pred_len = pred_len)
    

    # synthetic_data  = np.swapaxes(synthetic_data, -1, -2)
    # gt_data = np.swapaxes(gt_data, -1, -2)
    # target_mask =np.swapaxes(target_mask, -1, -2)
    # CRPS_sum_1 = calc_quantile_CRPS_sum(torch.Tensor(gt_data),torch.Tensor(synthetic_data[:,:10]),torch.Tensor(target_mask),mean_scaler=scaler.mean_,scaler=scaler.scale_)
    # print(CRPS_sum_1)
    # CRPS_sum_2 = calc_quantile_CRPS_sum(torch.Tensor(gt_data),torch.Tensor(synthetic_data[:,:10]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    # print(CRPS_sum_2)
    # CRPS_sum_1
    # import pdb; pdb.set_trace()
    if data_type in ['solar', 'Wiki', 'Taxi']:
        unormzalized_gt_data = []
        for g in gt_data:
            unormzalized_gt_data.append(np.transpose(np.transpose(g)*scaler.scale_+scaler.mean_))
        unormzalized_gt_data = np.array(unormzalized_gt_data)
        unormalized_synthetic_data = []
        for i in range(len(synthetic_data)):
            for j in range(len(synthetic_data[i])):
                s = synthetic_data[i][j]
                # print(s.shape)
                unormalized_synthetic_data.append(np.transpose(np.transpose(s)*scaler.scale_+scaler.mean_))
        unormalized_synthetic_data = np.array(unormalized_synthetic_data).reshape(synthetic_data.shape[0],-1,synthetic_data.shape[2],synthetic_data.shape[3])
    elif data_type in ['traffic', 'exchange']:
        unormalized_synthetic_data = synthetic_data
        unormzalized_gt_data = gt_data
    else:
        unormalized_synthetic_data = synthetic_data
        unormzalized_gt_data = gt_data
    
    unormzalized_gt_data = np.swapaxes(unormzalized_gt_data, -1, -2)
    unormalized_synthetic_data = np.swapaxes(unormalized_synthetic_data, -1, -2)
    target_mask = np.swapaxes(target_mask, -1, -2)

    print("all:" + data_type +  " CRPS_sum:")
    print(calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data),torch.Tensor(target_mask),mean_scaler=0,scaler=1))
    print("all:" + data_type +  " CRPS_sum:")
    # import pdb; pdb.set_trace()
    CRPS_sum_1 = calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data[:,:10]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    CRPS_sum_2 = calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data[:,10:20]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    CRPS_sum_3 = calc_quantile_CRPS_sum(torch.Tensor(unormzalized_gt_data),torch.Tensor(unormalized_synthetic_data[:,20:30]),torch.Tensor(target_mask),mean_scaler=0,scaler=1)
    
    CRPS_sum_mean = np.mean([CRPS_sum_1, CRPS_sum_2, CRPS_sum_3])
    CRPS_sum_std = np.std([CRPS_sum_1, CRPS_sum_2, CRPS_sum_3])
    print("Mean CRPS_sum:", CRPS_sum_mean)
    print("Standard Deviation CRPS_sum:", CRPS_sum_std)
    return CRPS_sum_1

if __name__ == '__main__':
    pre_model = 'pretrain_aq_ettm1_ecl_more_aq_weather'
    pre_model = 'pretrain_aq_ettm1_m2'
    pre_model = 'pretrain_aq_ettm1_ecl'
    pre_model = 'pretrain_aq_ettm1_ecl_more_aq_weather_with_cloud'
    pre_model = 'pretrain_aq_ettm1_ecl_more_aq_weather_with_cloud_with_text'
    print("all:")
    # pre_model = sys.argv[1]
    # data_path="./csdi_data/solar_nips/train/train.csv"
    # data_type = 'pde'
    # seq_len = 41 #192
    # imputed_folder = "./results/0_pretrain_sample/pde_1d/" + pre_model
    # calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder, pred_len =31)
    
    # data_path="./csdi_data/solar_nips/train/train.csv"
    # data_type = 'solar'
    # seq_len = 192
    # # imputed_folder = "./results/0_pretrain_sample_debug/"+ data_type + "/" + pre_model
    # imputed_folder = "results_july/0_pretrain_sample_no_reshape/solar/forecasting"
    # imputed_folder = "results_july/0_pretrain_sample_fre/solar/forecasting"
    # #0_pretrain_sample_fre
    # # imputed_folder = "/scratch/bcqc/dcao1/goole_drive/solar_wo_text_add_obs/backup"
    # calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder)
    
    data_path="./csdi_data/traffic_nips/train/train.csv"
    data_type = 'traffic'
    seq_len = 192
    # imputed_folder = "./results/0_pretrain_sample/"+ data_type + "/" + pre_model
    #/scratch/bcqc/dcao1/goole_drive/0_pretrain_sample/traffic/pretrain_L_aq_ettm1_ecl_more_aq_weather_with_cloud_traffic
    imputed_folder = 'results_july/0_pretrain_sample_no_reshape/traffic/forecasting'
    imputed_folder = 'results_july/0_pretrain_sample_fre/traffic/forecasting'
    calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder)
    
    # data_path="./csdi_data/electricity/train/train.csv"
    # data_type = 'electricity'
    # seq_len = 192
    # imputed_folder = "results_july/0_pretrain_sample_no_reshape/ecl/forecasting"
    # # imputed_folder = "/scratch/bcqc/dcao1/goole_drive/0_pretrain_sample/electricity/pretrain_aq_ettm1_ecl_more_aq_weather"
    # calculate_results_ecl(data_path, data_type, seq_len, pre_model, imputed_folder)
    
    # data_path="./csdi_data/wiki_2000/train/train.csv"
    # data_type = 'Wiki'
    # seq_len = 120
    # imputed_folder = "results_july/0_pretrain_sample_no_reshape/wiki/forecasting"
    # calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder, pred_len =30)
    
    data_path="./csdi_data/exchange/train/train.csv"
    data_type = 'exchange'
    seq_len = 120
    imputed_folder = "results_july/0_pretrain_sample_reshape/exchange/forecasting"
    #0_pretrain_sample_fre
    # imputed_folder = "results_july/0_pretrain_sample_fre/exchange/forecasting"

    calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder, pred_len =30)
    
    
    data_path = "./csdi_data/taxi_30min/train/train.csv"
    data_type = 'Taxi'
    seq_len = 72
    imputed_folder = "/scratch/bcqc/dcao1/goole_drive/0_pretrain_sample/Taxi/pretrain_L_aq_ettm1_ecl_more_aq_weather_with_cloud_traffic"
    imputed_folder = "results_july/0_pretrain_sample_no_reshape/taxi/forecasting"
    calculate_results(data_path, data_type, seq_len, pre_model, imputed_folder, pred_len =24)
    

    
    
    