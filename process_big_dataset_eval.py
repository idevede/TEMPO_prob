import pickle
import pyarrow as pa
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
import json
from datasets import Dataset
import gc
import json
import os
import hashlib
from datasets.fingerprint import Hasher
from datasets.info import DatasetInfo
from datasets.splits import SplitInfo
import numpy as np

class MonashDatasetConverter:
    def __init__(self, chunk_size_mb=200):
        self.chunk_size_mb = chunk_size_mb * 1024 * 1024
        
    def _create_arrow_schema(self):
        """创建Arrow Schema"""
        return pa.schema([
            ('x_target', pa.list_(pa.float32())),
            ('y_target', pa.list_(pa.float32())),
            ('x_trend', pa.list_(pa.float32())),
            ('x_seasonal', pa.list_(pa.float32())),
            ('x_resid', pa.list_(pa.float32()))
        ])

    def process_pkl_file(self, pkl_path: str) -> dict:
        """读取单个PKL文件并预处理数据"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded {len(data)} items from {pkl_path}")
            processed_data = []
            
            for key in data.keys():
                for item in data[key]:
                    processed_item = {
                        'x_target': item['x']['target'].astype(np.float32).tolist(),
                        'y_target': item['y']['target'].astype(np.float32).tolist(),
                        'x_trend': item['x_trend'].astype(np.float32).tolist(),
                        'x_seasonal': item['x_seasonal'].astype(np.float32).tolist(),
                        'x_resid': item['x_resid'].astype(np.float32).tolist()
                    }
                    processed_data.append(processed_item)
            print(f"Processed {len(processed_data)} items")
            #shuffle
            np.random.shuffle(processed_data)
            return processed_data

    def convert_to_arrow(self, pkl_path: str, output_dir: str, dataset_name: str = "monash"):
        """将PKL文件转换为Arrow格式"""
        os.makedirs(output_dir, exist_ok=True)
        schema = self._create_arrow_schema()
        
        # 流式处理PKL文件
        buffer = []
        current_size = 0
        file_index = 0
        
        print(f"Processing PKL file: {pkl_path}")
        processed_data = self.process_pkl_file(pkl_path)
        samples = len(processed_data)
        
        for item in tqdm(processed_data):
            buffer.append(item)
            # 估算当前大小
            current_size += sum(len(str(v)) for v in item.values())
            
            if current_size >= self.chunk_size_mb:
                self._write_arrow_chunk(buffer, schema, output_dir, file_index, dataset_name=dataset_name)
                buffer = []
                current_size = 0
                file_index += 1
                gc.collect()
        
        # 处理剩余数据
        if buffer:
            self._write_arrow_chunk(buffer, schema, output_dir, file_index, dataset_name=dataset_name)
            
        return file_index + 1, samples  # 返回总的文件数量

    def _write_arrow_chunk(self, data, schema, output_dir, file_index, dataset_name="monash"):
        """写入一个Arrow数据块"""
        output_path = os.path.join(output_dir, f'{dataset_name}_{file_index}.arrow')
        
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

def convert_and_prepare_dataset(pkl_path: str, output_base_dir: str, dataset_name: str):
    """完整的转换和准备过程"""
    # 创建必要的目录
    arrow_dir = os.path.join(output_base_dir, 'arrow_files')
    dataset_files_dir = os.path.join(output_base_dir, 'dataset_files')
    os.makedirs(arrow_dir, exist_ok=True)
    os.makedirs(dataset_files_dir, exist_ok=True)

    # 1. 转换PKL到Arrow
    converter = MonashDatasetConverter(chunk_size_mb=200)
    num_files, sample = converter.convert_to_arrow(pkl_path, arrow_dir, dataset_name=dataset_name)
    print(f"Created {num_files} Arrow files")

    # 2. 生成数据集状态文件
    state_generator = DatasetStateGenerator(dataset_name)
    state = state_generator.generate_state_json(arrow_dir, dataset_files_dir)
    dataset_info = state_generator.generate_dataset_info(arrow_dir, dataset_files_dir, samples = sample)

    return arrow_dir, dataset_files_dir

import os

def get_file_info(file_path):
    # 获取文件名（不含路径）
    filename = os.path.basename(file_path)
    # 获取不含扩展名的文件名
    base_name = os.path.splitext(filename)[0]
    # 获取上一级目录名
    parent_dir = os.path.basename(os.path.dirname(file_path))
    # 组合目录名和文件名
    dataset_name = f"{parent_dir}_{base_name}"
    
    pkl_path = file_path
    output_dir = f"dataset/gift_eval_arrow/{dataset_name}/"
    
    return pkl_path, output_dir, dataset_name

# 递归查找所有.pkl文件
def find_pkl_files(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


# 使用示例
if __name__ == "__main__":
    # 配置

   
    # 指定目录
    directory = "dataset/gift_eval"

    pkl_files = find_pkl_files(directory)

    for full_path in pkl_files:
        PKL_PATH, OUTPUT_DIR, DATASET_NAME = get_file_info(full_path)
        print(f"PKL_PATH = {PKL_PATH}")
        print(f"OUTPUT_DIR = {OUTPUT_DIR}")
        print(f"DATASET_NAME = {DATASET_NAME}")
        print("---")
            
        if DATASET_NAME == "mexico_city_bikes" \
            or DATASET_NAME == "m4_yearly" or DATASET_NAME == "m4_weekly" or DATASET_NAME == "ercot": 
            continue
        if "m4_hourly" in DATASET_NAME or "m4_daily" in DATASET_NAME:
            continue
        # 转换并准备数据集
        try:
            arrow_dir, dataset_files_dir = convert_and_prepare_dataset(
                pkl_path=PKL_PATH,
                output_base_dir=OUTPUT_DIR,
                dataset_name=DATASET_NAME
            )
            gc.collect()
        except Exception as e:
            print(f"Error processing {PKL_PATH}: {e}")
            gc.collect()
            continue
