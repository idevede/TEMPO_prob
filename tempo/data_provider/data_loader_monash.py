import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import json
import gc
import fcntl
import time
from filelock import FileLock
from typing import List, Tuple, Dict, Optional, Union

class Dataset_Monash(Dataset):
    def __init__(self, 
                 root_path: str,
                 flag: str = 'train',
                 size: Optional[Tuple[int, int, int]] = None,
                 features: str = 'S',
                 data_path: str = 'ETTh1.csv',
                 target: str = 'OT',
                 scale: bool = True,
                 timeenc: int = 0,
                 freq: str = 'h',
                 percent: int = 100,
                 data_name: str = 'etth2',
                 max_len: int = -1,
                 train_all: bool = False):
        
        super().__init__()
        
        self.seq_len = size[0] if size else 24 * 4 * 4
        self.label_len = size[1] if size else 24 * 4
        self.pred_len = size[2] if size else 24 * 4
        
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        
        self.cache_dir = Path(root_path) / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建文件锁
        self.lock_file = self.cache_dir / f'dataset_cache_{self.set_type}.lock'
        self._initialize_data()
        
    def _initialize_data(self):
        cache_file = self.cache_dir / f'dataset_cache_{self.set_type}.h5'
        index_file = self.cache_dir / f'dataset_index_{self.set_type}.json'
        
        # 使用文件锁来确保只有一个进程在处理数据
        with FileLock(str(self.lock_file)):
            if cache_file.exists() and index_file.exists():
                self._load_from_cache(cache_file, index_file)
            else:
                self._process_and_cache_data(cache_file, index_file)

    def _load_from_cache(self, cache_file: Path, index_file: Path):
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with open(index_file, 'r') as f:
                    self.data_index = json.load(f)
                self.h5_file = h5py.File(str(cache_file), 'r', libver='latest')
                self.dataset = self.h5_file['data']
                break
            except (BlockingIOError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{max_retries} loading cache file")
                time.sleep(retry_delay)

    def _process_and_cache_data(self, cache_file: Path, index_file: Path):
        try:
            # 确保文件不存在
            cache_file.unlink(missing_ok=True)
            index_file.unlink(missing_ok=True)
            
            datasets = []
            chunk_size = 1000
            
            # 收集数据
            for directory in ['dataset/chronos', 'dataset/chronos_2']:
                if not Path(directory).exists():
                    continue
                    
                for file_path in Path(directory).glob('*.pkl'):
                    # if self._is_valid_file(file_path):
                    datasets.extend(self._load_file(file_path, chunk_size))
            
            if not datasets:
                raise ValueError("No valid data found")
            
            # 数据集划分
            train_ratio, val_ratio = 0.7, 0.1
            n_samples = len(datasets)
            splits = [
                int(n_samples * train_ratio),
                int(n_samples * (train_ratio + val_ratio))
            ]
            
            # 选择相应的数据集部分
            if self.set_type == 0:
                datasets = datasets #[:splits[0]]
            elif self.set_type == 1:
                datasets = datasets[splits[0]:splits[1]]
            else:
                datasets = datasets[splits[1]:]
            
            # 创建HDF5文件
            with h5py.File(str(cache_file), 'w', libver='latest') as f:
                data_shape = (len(datasets), self.seq_len + self.pred_len, 7)
                dataset = f.create_dataset('data', 
                                         shape=data_shape,
                                         dtype='float32',
                                         chunks=(min(100, len(datasets)), 
                                                self.seq_len + self.pred_len, 
                                                7),
                                         compression='gzip')
                
                for i in range(0, len(datasets), chunk_size):
                    chunk = datasets[i:i + chunk_size]
                    batch_data = [self._process_single_sample(data) for data in chunk]
                    dataset[i:i + len(chunk)] = batch_data
                    del batch_data
                    gc.collect()
            
            # 保存索引
            self.data_index = list(range(len(datasets)))
            with open(index_file, 'w') as f:
                json.dump(self.data_index, f)
            
            # 打开用于读取的文件
            self.h5_file = h5py.File(str(cache_file), 'r', libver='latest')
            self.dataset = self.h5_file['data']
            
        except Exception as e:
            # 清理可能部分创建的文件
            cache_file.unlink(missing_ok=True)
            index_file.unlink(missing_ok=True)
            raise e

    def _is_valid_file(self, file_path: Path) -> bool:
        return any(keyword in file_path.name for keyword in [
            'monash_pedestrian_counts',
            'm4_weekly', 'm4_yearly', 'm4_hourly', 'm4_monthly',
            'ercot'
        ])

    def _load_file(self, file_path: Path, chunk_size: int) -> List:
        datasets = []
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                for key in data:
                    current_chunk = []
                    for item in data[key]:
                        current_chunk.append(item)
                        if len(current_chunk) >= chunk_size:
                            datasets.extend(current_chunk)
                            current_chunk = []
                    if current_chunk:
                        datasets.extend(current_chunk)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return datasets

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        data = self.dataset[self.data_index[index]]
        
        seq_x = data[:self.seq_len, 0]
        seq_y = data[self.seq_len:, 0]
        seq_x_mark = data[:self.seq_len, 1:5]
        seq_y_mark = data[self.seq_len:, 1:5]
        seq_trend = data[:self.seq_len, 4]
        seq_seasonal = data[:self.seq_len, 5]
        seq_resid = data[:self.seq_len, 6]
        
        return (
            np.expand_dims(seq_x, axis=-1),
            np.expand_dims(seq_y, axis=-1),
            seq_x_mark,
            seq_y_mark,
            torch.from_numpy(np.expand_dims(seq_trend, axis=-1).copy()),
            torch.from_numpy(np.expand_dims(seq_seasonal, axis=-1).copy()),
            torch.from_numpy(np.expand_dims(seq_resid, axis=-1).copy())
        )

    def __len__(self) -> int:
        return len(self.data_index)

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()