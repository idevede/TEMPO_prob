import os
import gc
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from datasets.fingerprint import Hasher
from sklearn.preprocessing import StandardScaler
import torch

class TSDecompositionProcessor:
    def __init__(self, dataset_name, version="1.0.0"):
        self.dataset_name = dataset_name
        self.version = version
        
    def process_dataset(self, data_path, trend_path, seasonal_path, residual_path, output_dir, 
                        x_size=336, y_size=48, chunk_size=10000, max_memory_mb=1000):
        """
        处理时间序列数据及其分解结果
        
        参数:
            data_path: 原始数据CSV路径
            trend_path: 趋势分量路径
            seasonal_path: 季节性分量路径
            residual_path: 残差分量路径
            output_dir: 输出目录
            x_size: 输入序列长度
            y_size: 输出序列长度
            chunk_size: 每个Arrow文件中的样本数量
            max_memory_mb: 内存阈值(MB)
        """
        window_size = x_size + y_size
        
        # 创建输出目录
        dataset_output_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 读取数据
        print(f"Reading data from {data_path}")
        df_raw = pd.read_csv(data_path)
        timestamps = df_raw.iloc[:, 0].values  # 假设第一列是时间戳
        features = df_raw.columns[1:]  # 跳过时间戳列
        scaler = StandardScaler()
        

       

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[features]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)

        train_data = df_raw[0:num_train]
        scaler.fit(train_data.values)
        data = scaler.transform(df_raw.values)
       
        

            
        
        def tensor_to_numpy(data):
            """将PyTorch张量转换为NumPy数组"""
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            return data

        with open(trend_path, 'rb') as f:
            trend_data = pickle.load(f)
            trend_data = tensor_to_numpy(trend_data)

        with open(seasonal_path, 'rb') as f:
            seasonal_data = pickle.load(f)
            seasonal_data = tensor_to_numpy(seasonal_data)
            
        with open(residual_path, 'rb') as f:
            residual_data = pickle.load(f)
            residual_data = tensor_to_numpy(residual_data)
                
        # 验证数据形状
        if trend_data.shape != (len(df_raw), len(features)):
            raise ValueError(f"Trend data shape {trend_data.shape} doesn't match original data shape {(len(df), len(features))}")
        
        if seasonal_data.shape != (len(df_raw), len(features)):
            raise ValueError(f"Seasonal data shape {seasonal_data.shape} doesn't match original data shape {(len(df), len(features))}")
            
        if residual_data.shape != (len(df_raw), len(features)):
            raise ValueError(f"Residual data shape {residual_data.shape} doesn't match original data shape {(len(df), len(features))}")
        
        total_samples = 0
        file_indices = {}
        
        # 为每个特征处理数据
        for i, feature in enumerate(tqdm(features, desc="Processing features")):
            print(f"Processing feature: {feature}")
            # import pdb; pdb.set_trace()
            # 获取该特征的数据
            normalized_data = data[:, i]
            #df_raw[feature].values
            
            trend = trend_data[:, i]
            seasonal = seasonal_data[:, i]
            residual = residual_data[:, i]
            
            # # 数据预处理
            # mean = np.nanmean(original_data)
            # std = np.nanstd(original_data)
            # # 填充缺失值
            # original_data = np.where(np.isnan(original_data), mean, original_data)
            # # 标准化
            # normalized_data = (original_data - mean) / std

            # import pdb; pdb.set_trace()
            
            if np.isnan(normalized_data).any():
                print(f"Warning: NaN values in normalized data for feature {feature}")
                continue
            
            # 创建特征输出目录
            feature_dir = os.path.join(dataset_output_dir, f"feature_{feature}")
            os.makedirs(feature_dir, exist_ok=True)
            
            # 创建滑动窗口
            windows_list = []
            current_memory = 0
            max_memory_bytes = max_memory_mb * 1024 * 1024
            file_index = 0
            feature_samples = 0
            
            schema = pa.schema([
                ('x_target', pa.list_(pa.float32())),
                ('y_target', pa.list_(pa.float32())),
                ('x_trend', pa.list_(pa.float32())),
                ('x_seasonal', pa.list_(pa.float32())),
                ('x_resid', pa.list_(pa.float32())),
                # ('timestamp', pa.list_(pa.int64())),
            ])
            
            for j in range(0, len(normalized_data) - window_size + 1, y_size):
                window_data = normalized_data[j:j + window_size]
                window_trend = trend[j:j + window_size]
                window_seasonal = seasonal[j:j + window_size]
                window_residual = residual[j:j + window_size]
                # window_timestamp = timestamps[j:j + window_size]
                
                x_target = window_data[:x_size].astype(np.float32).tolist()
                y_target = window_data[x_size:].astype(np.float32).tolist()
                x_trend = window_trend[:x_size].astype(np.float32).tolist()
                x_seasonal = window_seasonal[:x_size].astype(np.float32).tolist()
                x_resid = window_residual[:x_size].astype(np.float32).tolist()
                # x_timestamp = window_timestamp[:x_size].astype(np.int64).tolist()
                
                processed_item = {
                    'x_target': x_target,
                    'y_target': y_target,
                    'x_trend': x_trend, 
                    'x_seasonal': x_seasonal, 
                    'x_resid': x_resid,
                    # 'timestamp': x_timestamp,
                }
                
                windows_list.append(processed_item)
                current_memory += sum(len(str(v)) for v in processed_item.values())
                feature_samples += 1
                
                # 检查是否需要写入磁盘
                if len(windows_list) >= chunk_size or current_memory >= max_memory_bytes:
                    self._write_arrow_chunk(windows_list, schema, feature_dir, file_index)
                    file_index += 1
                    windows_list = []
                    current_memory = 0
                    gc.collect()
            
            # 处理剩余的窗口
            if windows_list:
                self._write_arrow_chunk(windows_list, schema, feature_dir, file_index)
                file_index += 1
            
            file_indices[feature] = file_index
            total_samples += feature_samples
            
            # 为每个特征生成状态文件
            state_generator = DatasetStateGenerator(f"{self.dataset_name}_{feature}")
            state = state_generator.generate_state_json(feature_dir, os.path.join(feature_dir, 'dataset_files'))
            dataset_info = state_generator.generate_dataset_info(feature_dir, os.path.join(feature_dir, 'dataset_files'), 
                                                                 samples=feature_samples)
            
            print(f"Feature {feature} completed: {feature_samples} samples, {file_index} files")
            
        # 为整个数据集生成状态文件
        state_generator = DatasetStateGenerator(self.dataset_name)
        combined_state = state_generator.generate_state_json(dataset_output_dir, os.path.join(dataset_output_dir, 'dataset_files'))
        combined_info = state_generator.generate_dataset_info(dataset_output_dir, os.path.join(dataset_output_dir, 'dataset_files'), 
                                                           samples=total_samples)
        
        print(f"Dataset processing completed: Total samples={total_samples}")
        return total_samples, file_indices
    
    def _write_arrow_chunk(self, data, schema, output_dir, file_index):
        """写入Arrow文件"""
        if not data:
            return
            
        output_path = os.path.join(output_dir, f"{file_index}.arrow")
        
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

    def generate_dataset_info(self, data_dir, output_dir, samples=1000):
        """生成dataset_info.json文件"""
        # 获取特征信息
        arrow_files = [f for f in os.listdir(data_dir) if f.endswith('.arrow')]
        if not arrow_files:
            raise ValueError(f"No arrow files found in {data_dir}")
            
        first_arrow_file = os.path.join(data_dir, arrow_files[0])
        
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
            "description": f"Time Series Decomposition Dataset - {self.dataset_name}",
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


# 示例用法
def process_datasets(datasets_config, output_base_dir="processed_datasets"):
    """
    处理多个数据集
    
    参数:
        datasets_config: 包含数据集配置的字典，格式为:
            {
                "dataset_name": {
                    "data_path": "path/to/data.csv",
                    "trend_path": "path/to/trend.pk",
                    "seasonal_path": "path/to/seasonal.pk",
                    "residual_path": "path/to/residual.pk",
                    "period": 24  # 可选，时间序列的周期性
                }
            }
        output_base_dir: 输出基础目录
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    for dataset_name, config in datasets_config.items():
        print(f"Processing dataset: {dataset_name}")
        print("=" * 50)
        
        processor = TSDecompositionProcessor(dataset_name)
        
        try:
            total_samples, file_indices = processor.process_dataset(
                config["data_path"],
                config["trend_path"],
                config["seasonal_path"],
                config["residual_path"],
                output_base_dir,
                x_size=336,
                y_size=48
            )
            
            print(f"Dataset {dataset_name} processed successfully.")
            print(f"Total samples: {total_samples}")
            print(f"Files per feature: {file_indices}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            print("-" * 50)


# 配置数据集
datasets_config = {
    # "electricity": {
    #     "data_path": "/workspace/defucao/datasets/all_datasets/electricity/electricity.csv",
    #     "trend_path": "/workspace/defucao/electricity/trend.pk",
    #     "seasonal_path": "/workspace/defucao/electricity/seasonal.pk",
    #     "residual_path": "/workspace/defucao/electricity/resid.pk"
    # },
    "traffic":{
        "data_path": "/workspace/defucao/datasets/all_datasets/traffic/traffic.csv",
        "trend_path": "/workspace/defucao/traffic/trend.pk",
        "seasonal_path": "/workspace/defucao/traffic/seasonal.pk",
        "residual_path": "/workspace/defucao/traffic/resid.pk"
    },
    "weather":{
        "data_path": "/workspace/defucao/datasets/all_datasets/weather/weather.csv",
        "trend_path": "/workspace/defucao/weather/trend.pk",
        "seasonal_path": "/workspace/defucao/weather/seasonal.pk",
        "residual_path": "/workspace/defucao/weather/resid.pk"
    },
    "exchange":{
        "data_path": "/workspace/defucao/datasets/all_datasets/exchange_rate/exchange_rate.csv",
        "trend_path": "/workspace/defucao/exchange/trend.pk",
        "seasonal_path": "/workspace/defucao/exchange/seasonal.pk",
        "residual_path": "/workspace/defucao/exchange/resid.pk"
    },

    # 可以添加更多数据集...
}

# 运行处理
process_datasets(datasets_config, "dataset/tempo/")