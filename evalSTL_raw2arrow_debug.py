import pickle
import datasets
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from datasets import load_from_disk
import os

import os
import gc
import json
import pickle
import numpy as np
import pyarrow as pa
from tqdm import tqdm
# from datasets import Hasher
from datasets.fingerprint import Hasher


class DatasetStateGenerator:
    def __init__(self, dataset_name, version="1.0.0"):
        self.dataset_name = dataset_name
        self.version = version
        
    def _generate_fingerprint(self, data_files):
        """生成数据集指纹"""
        hasher = Hasher()
        
        # 添加数据文件路径到哈希
        for file_path in sorted(data_files):
            hasher.update(file_path)
            
            # 可选：添加文件内容的哈希
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
                    
        return hasher.hexdigest()

    def _get_split_dict(self, data_files, split_names=['train', 'validation', 'test']):
        """生成split信息"""
        split_dict = {}
        total_files = len(data_files)
        
        # 默认分割比例
        split_ratios = {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1
        }
        
        start_idx = 0
        for split_name in split_names:
            n_files = int(total_files * split_ratios[split_name])
            if split_name == split_names[-1]:  # 最后一个split获取所有剩余文件
                split_files = data_files[start_idx:]
            else:
                split_files = data_files[start_idx:start_idx + n_files]
            
            split_dict[split_name] = {
                "name": split_name,
                "num_bytes": sum(os.path.getsize(f) for f in split_files),
                "num_examples": self._count_examples(split_files),
                "dataset_name": self.dataset_name,
                "fingerprint": self._generate_fingerprint(split_files)
            }
            start_idx += n_files
            
        return split_dict

    def _count_examples(self, files):
        """计算样本数量"""
        total_examples = 0
        for file in files:
            # 对于Arrow文件
            if file.endswith('.arrow'):
                import pyarrow as pa
                with pa.memory_map(file, 'r') as source:
                    reader = pa.ipc.RecordBatchFileReader(source)
                    total_examples += reader.num_record_batches
            # 对于Parquet文件
            elif file.endswith('.parquet'):
                import pyarrow.parquet as pq
                total_examples += pq.read_metadata(file).num_rows
        return total_examples

    def generate_state_json(self, data_dir, output_dir):
        """生成state.json文件"""
        # 获取所有数据文件
        # import pdb; pdb.set_trace()
        data_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.arrow', '.parquet')):
                    data_files.append(os.path.join(root, file))
        
        # 创建state字典
        state = {
            "splits": self._get_split_dict(data_files),
            "_fingerprint": self._generate_fingerprint(data_files),
            "transforms": {
                "fingerprint": self._generate_fingerprint(data_files),
                "transform_history": []
            }
        }
        
        # 保存state.json
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'state.json'), 'w') as f:
            json.dump(state, f, indent=2)
            
        return state

    
    def generate_dataset_info(self, data_dir, output_dir, samples = 1000):
        """生成dataset_info.json文件"""
        # 获取特征信息
        arrow_files = [f for f in os.listdir(data_dir) if f.endswith('.arrow')]
        if not arrow_files:
            raise ValueError(f"No arrow files found in {data_dir}")
            
        first_arrow_file = os.path.join(data_dir, arrow_files[0])
        
        import pyarrow as pa
        with pa.memory_map(first_arrow_file, 'r') as source:
            schema = pa.ipc.RecordBatchFileReader(source).schema

        # 获取特征信息
        features = {
            field.name: {
                "dtype": str(field.type),
                "_type": "Value",
                "shape": [-1] if pa.types.is_list(field.type) else None
            }
            for field in schema
        }

        # 计算数据集大小
        dataset_size = sum(
            os.path.getsize(os.path.join(data_dir, f))
            for f in os.listdir(data_dir)
            if f.endswith('.arrow')
        )

        dataset_info = {
            "description": f"Monash Time Series Dataset - {self.dataset_name}",
            "citation": "",
            "homepage": "",
            "license": "",
            "features": features,
            "post_processed": None,
            "supervised_keys": None,
            "task_templates": None,
            "builder_name": self.dataset_name,
            "config_name": "default",
            "version": {
                "version_str": self.version,
                "major": int(self.version.split('.')[0]),
                "minor": int(self.version.split('.')[1]),
                "patch": int(self.version.split('.')[2]),
            },
            "splits": None,
            "download_checksums": None,
            "download_size": 0,
            "dataset_size": dataset_size,
            "size_in_bytes": dataset_size,
            "num_sample": samples,
        }

        # 保存dataset_info.json
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return dataset_info



period_map = {

    # # Jena Weather
    'jena_weather/10T': 144,  # 10分钟数据，一天144个点
    'jena_weather/H': 24,
    'jena_weather/D': 30,
    
    # BizITObs
    'bizitobs_application': 360,  # 10秒数据，一小时360个点
    'bizitobs_service': 360,
    'bizitobs_l2c/5T': 288,   # 5分钟数据，一天288个点
    'bizitobs_l2c/H': 24,
    
    # # # Bitbrains
    # 'bitbrains_fast_storage/5T': 288,
    # 'bitbrains_fast_storage/H': 24,
    # 'bitbrains_rnd/5T': 288,
    # 'bitbrains_rnd/H': 24,
    
    # # Restaurant
    # 'restaurant/D': 7,
    
    # # ETT
    # 'ett1/15T': 96,
    # 'ett1/H': 24,
    # 'ett1/D': 30,
    # 'ett1/W': 52,
    # 'ett2/15T': 96,
    # 'ett2/H': 24,
    # 'ett2/D': 30,
    # 'ett2/W': 52,
    
    # # Transport -24 done
    # 'LOOP_SEATTLE/5T': 288,
    # 'LOOP_SEATTLE/H': 24,
    # 'LOOP_SEATTLE/D': 7,
    # 'SZ_TAXI/15T': 96,
    # 'SZ_TAXI/H': 24,
    # 'M_DENSE/H': 24,
    # 'M_DENSE/D': 30,
    
    # # Solar
    # 'solar/10T': 144,
    # 'solar/H': 24,
    # 'solar/D': 30,
    # 'solar/W': 52,
    
    # # Sales
    # 'hierarchical_sales/D': 7,
    # 'hierarchical_sales/W': 52,
    
    # M4
    # 'm4_yearly': 2,
    # 'm4_quarterly': 4,
    # 'm4_monthly': 12,
    # 'm4_weekly': 52,
    # 'm4_daily': 7,
    # 'm4_hourly': 24,
    
    # # Healthcare
    # 'hospital': 12,
    # 'covid_deaths': 7,
    # 'us_births/D': 30,
    # 'us_births/W': 52,
    # 'us_births/M': 12,
    
    # # Nature
    # 'saugeenday/D': 30,
    # 'saugeenday/W': 52,
    # 'saugeenday/M': 12,
    # 'temperature_rain_with_missing': 30,
    # 'kdd_cup_2018/H': 24,
    # 'kdd_cup_2018/D': 30,
    
    # #Sales
    # 'car_parts_with_missing': 12,
    
    # # Electricity
    # 'electricity/15T': 96,
    # 'electricity/H': 24,
    # 'electricity/D': 30,
    # 'electricity/W': 52
}

# all_names = [
#     # 'temperature_rain_with_missing'
#     # 'electricity/15T',
#     # 'electricity/H',
#     # 'electricity/D',
#     # 'covid_deaths',
#     # 'electricity/W',
#     # 'car_parts_with_missing',
#     'LOOP_SEATTLE/5T'

# ]
all_names = period_map.keys()
# [ 
# #  "electricity/W" 
#  "car_parts_with_missing"      
# ]
##period_map.keys()
all_names = [
    'ett1/15T'
]

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


def create_sliding_windows_chunked(dataset_name, ds, output_dir,  x_size=336, y_size=96, 
                                   chunk_size=10000, max_memory_mb=1000):
    """
    创建滑动窗口并按批次生成Arrow文件
    
    参数:
        dataset_name: 数据集名称
        ds: 数据集
        output_dir: 输出目录
        window_size: 窗口大小
        x_size: 输入序列长度
        y_size: 输出序列长度
        chunk_size: 每个Arrow文件中的样本数量
        max_memory_mb: 内存阈值(MB)，超过此值将触发写入磁盘
    """

    window_size = x_size + y_size
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建Arrow Schema
    schema = pa.schema([
        ('x_target', pa.list_(pa.float32())),
        ('y_target', pa.list_(pa.float32())),
        ('x_trend', pa.list_(pa.float32())),
        ('x_seasonal', pa.list_(pa.float32())),
        ('x_resid', pa.list_(pa.float32())),
        # ('y_trend', pa.list_(pa.float32())),
        # ('y_seasonal', pa.list_(pa.float32())),
        # ('y_resid', pa.list_(pa.float32())),
        # ('mean', pa.float32()),
        # ('std', pa.float32()),
    ])
    
    # 初始化变量
    i = 0
    file_index = 0
    total_samples = 0
    windows_list = []
    current_memory = 0
    max_memory_bytes = max_memory_mb * 1024 * 1024
    
    # 辅助函数：写入Arrow文件
    def write_arrow_chunk(data, schema, output_dir, file_index):
        if not data:
            return
            
        output_path = os.path.join(output_dir, f'{dataset_name}/{file_index}.arrow')

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)


        
        # 转换为Arrow表
        arrays = []
        for field in schema.names:
            arrays.append(pa.array([item[field] for item in data]))
        
        batch = pa.record_batch(arrays, schema=schema)
        
        # 写入文件
        with pa.OSFile(output_path, 'wb') as sink:
            writer = pa.RecordBatchFileWriter(sink, schema)
            writer.write_batch(batch)
            writer.close()
        
        print(f"Wrote {len(data)} samples to {output_path}")
    
    # 处理每个数据条目
    for entry in ds:
        data_id = entry['item_id']
        
        if 'target' in entry:
            targets = np.array(entry['target'])
            import pdb; pdb.set_trace()
            if len(targets.shape) > 1:
                # 处理多序列数据
                for i in range(targets.shape[0]):
                    target = targets[i]
                    if len(target) < window_size:
                        if len(targets) < 2*y_size:
                            print(f"Skipping data_id: {data_id} due to insufficient data with length: {len(targets)}")
                            continue
                        padding_length = window_size - len(target)
                        padded_target = np.pad(target, (padding_length, 0), 'constant', constant_values=0)
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
                        print(f"Skip: NaN values in z_scores for data_id: {data_id}")
                        print(f"targets: {target}")
                        continue
                    target = z_scores
                    trend, seasonal, resid = perform_stl_decomposition(target, dataset_name)
                    
                    for k in range(0, len(target) - window_size + 1, y_size):
                        window_target = target[k:k + window_size]
                        x_target = window_target[:x_size].astype(np.float32).tolist()
                        y_target = window_target[x_size:].astype(np.float32).tolist()
                        x_trend = trend[k:k + x_size].astype(np.float32).tolist()
                        x_seasonal = seasonal[k:k + x_size].astype(np.float32).tolist()
                        x_resid = resid[k:k + x_size].astype(np.float32).tolist()
                        # y_trend = trend[k + x_size:k + window_size].astype(np.float32).tolist()
                        # y_seasonal = seasonal[k + x_size:k + window_size].astype(np.float32).tolist()
                        # y_resid = resid[k + x_size:k + window_size].astype(np.float32).tolist()
                        
                        processed_item = {
                            'x_target': x_target,
                            'y_target': y_target,
                            'x_trend': x_trend, 
                            'x_seasonal': x_seasonal, 
                            'x_resid': x_resid,
                            # 'y_trend': y_trend, 
                            # 'y_seasonal': y_seasonal, 
                            # 'y_resid': y_resid,
                            # 'mean': float(mean), 
                            # 'std': float(std)
                        }
                        
                        windows_list.append(processed_item)
                        current_memory += sum(len(str(v)) for v in processed_item.values())
                        total_samples += 1
                        
                        # 检查是否需要写入磁盘
                        if len(windows_list) >= chunk_size or current_memory >= max_memory_bytes:
                            write_arrow_chunk(windows_list, schema, output_dir, file_index)
                            file_index += 1
                            windows_list = []
                            current_memory = 0
                            gc.collect()
            else:
                # 处理单序列数据
                print(f"Processing data_id: {data_id} with length: {len(targets)}")

                if len(targets) < window_size:
                    if len(targets) < 2*y_size:
                        print(f"Skipping data_id: {data_id} due to insufficient data with length: {len(targets)}")
                        continue
                    padding_length = window_size - len(targets)
                    targets = np.pad(targets, (padding_length, 0), 'constant', constant_values=0)
                    print(f"Padding target with {padding_length} zeros for data_id: {data_id} with total length: {len(targets)}")

                # 数据预处理
                mean = np.nanmean(targets)
                std = np.nanstd(targets)
                targets = np.where(np.isnan(targets), mean, targets)
                z_scores = (targets - mean) / std
                
                if np.isnan(z_scores).any():
                    print(f"NaN values in z_scores for data_id: {data_id}")
                    print(f"targets: {targets}")
                    continue
                
                targets = z_scores
                trend, seasonal, resid = perform_stl_decomposition(targets, dataset_name)

                # for i in range(0, len(targets) - window_size + 1):
                # import pdb; pdb.set_trace()
                for i in range(0, len(targets) - window_size + 1, y_size):
                    window_targets = targets[i:i + window_size]
                    x_target = window_targets[:x_size].astype(np.float32).tolist()
                    y_target = window_targets[x_size:].astype(np.float32).tolist()
                    x_trend = trend[i:i + x_size].astype(np.float32).tolist()
                    x_seasonal = seasonal[i:i + x_size].astype(np.float32).tolist()
                    x_resid = resid[i:i + x_size].astype(np.float32).tolist()
                    # y_trend = trend[i + x_size:i + window_size].astype(np.float32).tolist()
                    # y_seasonal = seasonal[i + x_size:i + window_size].astype(np.float32).tolist()
                    # y_resid = resid[i + x_size:i + window_size].astype(np.float32).tolist()
                    
                    processed_item = {
                        'x_target': x_target,
                        'y_target': y_target,
                        'x_trend': x_trend, 
                        'x_seasonal': x_seasonal, 
                        'x_resid': x_resid,
                        # 'y_trend': y_trend, 
                        # 'y_seasonal': y_seasonal, 
                        # 'y_resid': y_resid,
                        # 'mean': float(mean), 
                        # 'std': float(std)
                    }
                    
                    windows_list.append(processed_item)
                    current_memory += sum(len(str(v)) for v in processed_item.values())
                    total_samples += 1
                    
                    # 检查是否需要写入磁盘
                    if len(windows_list) >= chunk_size or current_memory >= max_memory_bytes:
                        write_arrow_chunk(windows_list, schema, output_dir, file_index)
                        file_index += 1
                        windows_list = []
                        current_memory = 0
                        gc.collect()
        else:
            # 处理没有target字段的数据
            for key in entry.keys():
                print(f"Key: {key}")
                
                try:
                    targets = np.array(entry[key])
                    if len(targets) < window_size or key == 'timestamp':
                        print(f"Skipping data_id: {data_id}'s {key} due to insufficient data with length: {len(targets)}")
                        continue
                except:
                    print('error on:' + key)
                    continue
                    
                if np.isnan(targets).any():
                    print(f"NaN values in targets for data_id: {data_id}")
                    print(f"targets: {len(targets)}")
                    targets = targets[~np.isnan(targets)]
                    print(f"After de-NAN targets: {len(targets)}")

                mean = np.nanmean(targets)
                std = np.nanstd(targets)
                targets = np.where(np.isnan(targets), mean, targets)
                z_scores = (targets - mean) / std
                
                if np.isnan(z_scores).any():
                    print(f"NaN values in z_scores for data_id: {data_id}")
                    continue
                
                targets = z_scores
                trend, seasonal, resid = perform_stl_decomposition(targets, dataset_name)

                for i in range(0, len(targets) - window_size + 1, y_size):
                    window_targets = targets[i:i + window_size]
                    x_target = window_targets[:x_size].astype(np.float32).tolist()
                    y_target = window_targets[x_size:].astype(np.float32).tolist()
                    x_trend = trend[i:i + x_size].astype(np.float32).tolist()
                    x_seasonal = seasonal[i:i + x_size].astype(np.float32).tolist()
                    x_resid = resid[i:i + x_size].astype(np.float32).tolist()
                    # y_trend = trend[i + x_size:i + window_size].astype(np.float32).tolist()
                    # y_seasonal = seasonal[i + x_size:i + window_size].astype(np.float32).tolist()
                    # y_resid = resid[i + x_size:i + window_size].astype(np.float32).tolist()
                    
                    processed_item = {
                        'x_target': x_target,
                        'y_target': y_target,
                        'x_trend': x_trend, 
                        'x_seasonal': x_seasonal, 
                        'x_resid': x_resid,
                        # 'y_trend': y_trend, 
                        # 'y_seasonal': y_seasonal, 
                        # 'y_resid': y_resid,
                        # 'mean': float(mean), 
                        # 'std': float(std)
                    }
                    
                    windows_list.append(processed_item)
                    current_memory += sum(len(str(v)) for v in processed_item.values())
                    total_samples += 1
                    
                    # 检查是否需要写入磁盘
                    if len(windows_list) >= chunk_size or current_memory >= max_memory_bytes:
                        write_arrow_chunk(windows_list, schema, output_dir, file_index)
                        file_index += 1
                        windows_list = []
                        current_memory = 0
                        gc.collect()
    
    if total_samples > 0:
        # 处理剩余的窗口
        if windows_list:
            write_arrow_chunk(windows_list, schema, output_dir, file_index)
            file_index += 1
        
        # 生成数据集状态文件
        # import pdb; pdb.set_trace()
        output_dir = os.path.join(output_dir, f'{dataset_name}')

        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        state_generator = DatasetStateGenerator(dataset_name)
        state = state_generator.generate_state_json(output_dir, os.path.join(output_dir, 'dataset_files'))
        dataset_info = state_generator.generate_dataset_info(output_dir, os.path.join(output_dir, 'dataset_files'), samples=total_samples)
        
        print(f"处理完成，总样本数: {total_samples}，生成的Arrow文件数: {file_index}")
    return total_samples, file_index

all_data_windows = {}

for dataset_name in all_names:
    # try:

    print(f"Processing dataset: {dataset_name}")
    print("====================================="*2)
    ds = read_dataset(dataset_name)
    if ds is not None:
        total_samples, file_index = create_sliding_windows_chunked(dataset_name, ds, output_dir=f"dataset/gift_eval_skip_48/", y_size=48)
        # if len(sliding_windows) ==0:
        #     print(f"Skipping dataset: {dataset_name} due to insufficient data")
        #     continue
        # all_data_windows[dataset_name] = sliding_windows
        #
        # with open(f"dataset/gift_eval/{dataset_name}.pkl", "wb") as f:
        #     pickle.dump(sliding_windows, f)
        # import pdb; pdb.set_trace()
        # file_path = f"dataset/gift_eval_more/{dataset_name}.pkl"
    
        # # 获取目录路径
        # directory = os.path.dirname(file_path)
        
        # # 创建所有必要的目录
        # os.makedirs(directory, exist_ok=True)
        
        # # 保存文件
        # with open(file_path, "wb") as f:
        #     pickle.dump(sliding_windows, f)
    # except Exception as e:
    #     print(f"Failed to process dataset {dataset_name}: {e}")

