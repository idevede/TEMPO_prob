import pickle
import datasets
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from datasets import load_from_disk
import os
period_map = {

    # Jena Weather
    'jena_weather/10T': 144,  # 10分钟数据，一天144个点
    'jena_weather/H': 24,
    'jena_weather/D': 30,
    
    # BizITObs
    'bizitobs_application': 360,  # 10秒数据，一小时360个点
    'bizitobs_service': 360,
    'bizitobs_l2c/5T': 288,   # 5分钟数据，一天288个点
    'bizitobs_l2c/H': 24,
    
    # Bitbrains
    'bitbrains_fast_storage/5T': 288,
    'bitbrains_fast_storage/H': 24,
    'bitbrains_rnd/5T': 288,
    'bitbrains_rnd/H': 24,
    
    # Restaurant
    'restaurant/D': 7,
    
    # ETT
    'ett1/15T': 96,
    'ett1/H': 24,
    'ett1/D': 30,
    'ett1/W': 52,
    'ett2/15T': 96,
    'ett2/H': 24,
    'ett2/D': 30,
    'ett2/W': 52,
    
    # Transport
    'LOOP_SEATTLE/5T': 288,
    'LOOP_SEATTLE/H': 24,
    'LOOP_SEATTLE/D': 7,
    'SZ_TAXI/15T': 96,
    'SZ_TAXI/H': 24,
    'M_DENSE/H': 24,
    'M_DENSE/D': 30,
    
    # Solar
    'solar/10T': 144,
    'solar/H': 24,
    'solar/D': 30,
    'solar/W': 52,
    
    # Sales
    'hierarchical_sales/D': 7,
    'hierarchical_sales/W': 52,
    
    # M4
    'm4_yearly': 2,
    'm4_quarterly': 4,
    'm4_monthly': 12,
    'm4_weekly': 52,
    'm4_daily': 7,
    'm4_hourly': 24,
    
    # Healthcare
    'hospital': 12,
    'covid_deaths': 7,
    'us_births/D': 30,
    'us_births/W': 52,
    'us_births/M': 12,
    
    # Nature
    'saugeenday/D': 30,
    'saugeenday/W': 52,
    'saugeenday/M': 12,
    'temperature_rain': 30,
    'kdd_cup_2018/H': 24,
    'kdd_cup_2018/D': 30,
    
    # Sales
    'car_parts_with_missing': 12,
    
    # Electricity
    'electricity/15T': 96,
    'electricity/H': 24,
    'electricity/D': 30,
    'electricity/W': 52
}


all_names = period_map.keys()
# [ 
#  "electricity/15T"       
# ]

GIFT_EVAL='/home/defucao/workspace/gift-eval/datasets/'

datasets_dict = {}

def read_dataset(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    try:
        ds =  load_from_disk(GIFT_EVAL + dataset_name)
        # ds = datasets.load_dataset("Salesforce/GiftEval", "electricity/15T", split="train")
        ds.set_format("numpy")
        return ds
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return None

def perform_stl_decomposition(series, dataset_name):
    period = period_map.get(dataset_name, 24)
    stl = STL(series, period=period)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

def create_sliding_windows(dataset_name, ds, window_size=432, x_size=336, y_size=96):
    
    i = 0
    for entry in ds:
        windows_dict = {}
        # import pdb; pdb.set_trace()
        data_id = entry['item_id']
    
        if 'target' in entry:
            targets = np.array(entry['target'])

            print(f"Processing data_id: {data_id} with length: {len(targets)}")

            if len(targets) < window_size:
                print(f"Skipping data_id: {data_id} due to insufficient data with length: {len(targets)}")
                continue
            # 计算均值，忽略NaN值
            
            mean = np.nanmean(targets)
            std = np.nanstd(targets)

            # 用均值填充NaN值
            targets = np.where(np.isnan(targets), mean, targets)

            # 进行z-score标准化
            z_scores = (targets - mean) / std
            if np.isnan(z_scores).any():
                print(f"NaN values in z_scores for data_id: {data_id}")
                print(f"targets: {targets}")
                continue
            # print(f"z_scores: {z_scores}")
            
            targets = z_scores #entry['target']

            trend, seasonal, resid = perform_stl_decomposition(targets, dataset_name)

            windows_list = []

            for i in range(0, len(targets) - window_size + 1):
                # window_timestamps = timestamps[i:i + window_size]
                window_targets = targets[i:i + window_size]

                x = {'target': window_targets[:x_size]}
                y = {'target': window_targets[x_size:]}
                x_trend = trend[i:i + x_size]
                x_seasonal = seasonal[i:i + x_size]
                x_resid = resid[i:i + x_size]
                y_trend = trend[i + x_size:i + window_size]
                y_seasonal = seasonal[i + x_size:i + window_size]
                y_resid = resid[i + x_size:i + window_size]
                windows_list.append({
                    'x': x, 'y': y,
                    'x_trend': x_trend, 'x_seasonal': x_seasonal, 'x_resid': x_resid,
                    'y_trend': y_trend, 'y_seasonal': y_seasonal, 'y_resid': y_resid,
                    'mean': mean, 'std': std
                })
            if len(windows_list) > 0:
                windows_dict[data_id] = windows_list
            else:
                print(f"Skipping data_id: {data_id} due to missing target")
                continue
        else:
            for key in entry.keys():
                print(f"Key: {key}")
                targets = np.array(entry[key])
                try:
                    if len(targets) < window_size or key == 'timestamp':
                        print(f"Skipping data_id: {data_id}'s {key} due to insufficient data with length: {len(targets)}")
                        continue
                except:
                    print('error on:' + key)
                    continue
                # 计算均值，忽略NaN值
                if np.isnan(targets).any():
                    print(f"NaN values in targets for data_id: {data_id}")
                    print(f"targets: {len(targets)}")
                    targets = targets[~np.isnan(targets)]
                    print(f"After de-NAN targets: {len(targets)}")

                mean = np.nanmean(targets)
                std = np.nanstd(targets)

                # 用均值填充NaN值
                targets = np.where(np.isnan(targets), mean, targets)

                # 进行z-score标准化
                z_scores = (targets - mean) / std
                if np.isnan(z_scores).any():
                    print(f"NaN values in z_scores for data_id: {data_id}")
                    # print(f"targets: {targets}")
                    continue
                # print(f"z_scores: {z_scores}")
                
                targets = z_scores #entry['target']

                trend, seasonal, resid = perform_stl_decomposition(targets, dataset_name)

                windows_list = []

                for i in range(0, len(targets) - window_size + 1):
                    # window_timestamps = timestamps[i:i + window_size]
                    window_targets = targets[i:i + window_size]

                    x = { 'target': window_targets[:x_size]}
                    y = {'target': window_targets[x_size:]}
                    x_trend = trend[i:i + x_size]
                    x_seasonal = seasonal[i:i + x_size]
                    x_resid = resid[i:i + x_size]
                    y_trend = trend[i + x_size:i + window_size]
                    y_seasonal = seasonal[i + x_size:i + window_size]
                    y_resid = resid[i + x_size:i + window_size]
                    windows_list.append({
                        'x': x, 'y': y,
                        'x_trend': x_trend, 'x_seasonal': x_seasonal, 'x_resid': x_resid,
                        'y_trend': y_trend, 'y_seasonal': y_seasonal, 'y_resid': y_resid,
                        'mean': mean, 'std': std
                    })
                if len(windows_list) > 0:
                    windows_dict[data_id] = windows_list

                # print(f"Skipping data_id: {data_id} due to insufficient data with length: {len(targets)}")
                else:
                    print(f"Skipping data_id: {data_id} due to missing target")
                    continue      
    return windows_dict


all_data_windows = {}

for dataset_name in all_names:
    try:

        print(f"Processing dataset: {dataset_name}")
        print("====================================="*2)
        ds = read_dataset(dataset_name)
        if ds is not None:
            sliding_windows = create_sliding_windows(dataset_name, ds)
            if len(sliding_windows) ==0:
                print(f"Skipping dataset: {dataset_name} due to insufficient data")
                continue
            # all_data_windows[dataset_name] = sliding_windows
            #
            # with open(f"dataset/gift_eval/{dataset_name}.pkl", "wb") as f:
            #     pickle.dump(sliding_windows, f)

            file_path = f"dataset/gift_eval/{dataset_name}.pkl"
        
            # 获取目录路径
            directory = os.path.dirname(file_path)
            
            # 创建所有必要的目录
            os.makedirs(directory, exist_ok=True)
            
            # 保存文件
            with open(file_path, "wb") as f:
                pickle.dump(sliding_windows, f)
    except Exception as e:
        print(f"Failed to process dataset {dataset_name}: {e}")

