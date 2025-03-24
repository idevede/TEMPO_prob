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

all_names = [
    #  'electricity/15T',
    'electricity/H',
    'electricity/D',
    # 'covid_deaths',
    # 'electricity/W': 52
]
# all_names = period_map.keys()
# [ 
# #  "electricity/W" 
#  "car_parts_with_missing"      
# ]
##period_map.keys()

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
    windows_dict = {}
    for entry in ds:
        
        # import pdb; pdb.set_trace()
        data_id = entry['item_id']
    
        if 'target' in entry:
            targets = np.array(entry['target'])
            if len(targets.shape) > 1:
                # continue
                for i in range(targets.shape[0]):
                    target = targets[i]
                    if len(target) < window_size:
                        print(f"Skipping data_id: {data_id} due to insufficient data with length: {len(target)}")
                        # continue
                        padding_length = window_size - len(target)
                        # 在目标序列末尾补充0
                        padded_target = np.pad(target, (0, padding_length), 'constant', constant_values=0)
                        # 将填充后的目标更新回原数组
                        targets[i] = padded_target
                        print(f"Padding target with {padding_length} zeros for data_id: {data_id}")

                    # 计算均值，忽略NaN值
                    mean = np.nanmean(target)
                    std = np.nanstd(target)
                    # 用均值填充NaN值
                    target = np.where(np.isnan(target), mean, target)
                    # 进行z-score标准化
                    z_scores = (target - mean) / std
                    if np.isnan(z_scores).any():
                        print(f"NaN values in z_scores for data_id: {data_id}")
                        print(f"targets: {target}")
                        continue
                    # print(f"z_scores: {z_scores}")
                    target = z_scores
                    trend, seasonal, resid = perform_stl_decomposition(target, dataset_name)
                    windows_list = []
                    for i in range(0, len(target) - window_size + 1):
                        window_target = target[i:i + window_size]
                        x = {'target': window_target[:x_size]}
                        y = {'target': window_target[x_size:]}
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
                        windows_dict[data_id + '_' + str(i)] = windows_list
                    else:
                        print(f"Skipping data_id: {data_id} due to missing target")
                        continue
            else:
                # import pdb; pdb.set_trace()
                print(f"Processing data_id: {data_id} with length: {len(targets)}")

                if len(targets) < window_size:
                    # import pdb; pdb.set_trace()
                    if len(targets) < 2*y_size:
                        print(f"Skipping data_id: {data_id} due to insufficient data with length: {len(targets)}")
                        continue
                    print(f"Processing: data_id: {data_id} due to insufficient data with length: {len(targets)}")
                    # continue
                    # continue
                    padding_length = window_size - len(targets)
                    # 在目标序列末尾补充0
                    targets = np.pad(targets, (padding_length, 0), 'constant', constant_values=0)
                    #ecl_w
                    # front_padding = 136 #152
                    # # 在目标序列后面补充72个0
                    # back_padding = 88 #72
                    #car
                    # front_padding = 297 #152
                    # # 在目标序列后面补充72个0
                    # back_padding = 84 #72
                    #covid
                    # front_padding = 154 #152
                    # # 在目标序列后面补充72个0
                    # back_padding = 66 #72
                    # targets = np.pad(targets, (front_padding, back_padding), 'constant', constant_values=0)
                    # print()
                    # 将填充后的目标更新回原数组
                    # targets[i] = padded_target
                    print(f"Padding target with {padding_length} zeros for data_id: {data_id} with total length: {len(targets)}")

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
            # import pdb; pdb.set_trace()
            file_path = f"dataset/gift_eval_more/{dataset_name}.pkl"
        
            # 获取目录路径
            directory = os.path.dirname(file_path)
            
            # 创建所有必要的目录
            os.makedirs(directory, exist_ok=True)
            
            # 保存文件
            with open(file_path, "wb") as f:
                pickle.dump(sliding_windows, f)
    except Exception as e:
        print(f"Failed to process dataset {dataset_name}: {e}")

