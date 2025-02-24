from datasets import load_dataset, interleave_datasets
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_from_disk
import datasets
from torch.utils.data import get_worker_info
import torch.distributed as dist

class StreamingMonashDataset(IterableDataset):
    def __init__(
        self,
        root_dirs: List[str],
        split: str = 'train',
        batch_size: int = 32,
        buffer_size: int = 10000,
        seed: int = 42,
        num_proc: int = 4,
        verbose: bool = True,
        num_workers: int = 4,
        cache_dir: str = None,
        prefetch_factor: int = 2
    ):
        """
        流式Arrow数据集加载器
        
        Args:
            root_dirs: 根目录列表
            split: 数据集分割
            batch_size: 批次大小
            buffer_size: 随机打乱缓冲区大小
            seed: 随机种子
            num_proc: 预处理使用的进程数
            verbose: 是否显示详细信息
        """
        super().__init__()
        self.root_dirs = root_dirs
        self.split = split
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.prefetch_factor = prefetch_factor
        
        
        
        self.num_proc = num_proc
        self.verbose = verbose
        
        # 设置日志
        self._setup_logging()

        # self.cache_dir = cache_dir
        
        # 从dataset_info.json获取大小
        self.length = self._get_dataset_size()
        
        # 生成缓存键
        # self.cache_key = self._generate_cache_key()
        

        # 加载数据集
        self.dataset = self._load_datasets()

        
        
        # # 添加数据处理流水线
        # self.dataset = self._setup_pipeline()

        

        # 从dataset_info.json获取数据集大小
        self.length = self._get_dataset_size()

    def _get_dataset_size(self):
        """从dataset_info.json获取数据集大小"""
        total_size = 0
        for root_dir in self.root_dirs:
            try:
                info_path = Path(root_dir) / 'dataset_files/dataset_info.json'
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        total_size += info.get("num_sample", 0)
                print(f"Loaded dataset size from {info_path}: {total_size}")
            except Exception as e:
                print(f"Error reading dataset_info.json from {root_dir}: {e}")
        return total_size

    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_directory(self, root_dir: str) -> bool:
        """验证目录结构"""
        arrow_dir = Path(root_dir) / 'arrow_files'
        if not arrow_dir.exists():
            self.logger.warning(f"Arrow directory not found: {arrow_dir}")
            return False
        
        arrow_files = list(arrow_dir.glob('*.arrow'))
        if not arrow_files:
            self.logger.warning(f"No arrow files found in {arrow_dir}")
            return False
            
        return True
    
    def _load_datasets(self):
        """加载所有数据集"""
        datasets = []
        valid_dirs = [
            root_dir for root_dir in self.root_dirs
            if self._validate_directory(root_dir)
        ]
        
        if not valid_dirs:
            raise ValueError("No valid data directories found")
        
        for root_dir in valid_dirs:
            try:
                arrow_dir = Path(root_dir) / 'arrow_files'
                dataset = load_dataset(
                    "arrow",
                    data_dir=str(arrow_dir),
                    split=self.split,
                    streaming=True
                ).with_format("torch")
                datasets.append(dataset)
                self.logger.info(f"Loaded dataset from {arrow_dir}")
                
            except Exception as e:
                self.logger.error(f"Error loading dataset from {root_dir}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        # 合并数据集
        return interleave_datasets(
            datasets,
            probabilities=None,
            seed=self.seed,
            stopping_strategy="all_exhausted"
            # stopping_strategy="first_exhausted"
        )

   
    def _preprocess_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """数据预处理函数"""
        try:
            processed = {
                'x_target': np.array(example['x_target'], dtype=np.float32),
                'y_target': np.array(example['y_target'], dtype=np.float32),
                # 添加其他预处理步骤
                # 'batch_x_mark': np.array(example['x_target'], dtype=np.float32),
                # 'batch_y_mark': np.array(example['y_target'], dtype=np.float32),
                'x_trend': np.array(example['x_trend'], dtype=np.float32),
                'x_seasonal': np.array(example['x_seasonal'], dtype=np.float32),
                'x_resid': np.array(example['x_resid'], dtype=np.float32)

            }
            
            # 数据验证
            if np.isnan(processed['x_target']).any() or np.isnan(processed['y_target']).any():
                self.logger.warning("Found NaN values in sample")
                return None
                
            return processed
            
        except Exception as e:
            self.logger.warning(f"Error preprocessing sample: {e}")
            return None
    
    
    def __iter__(self):
        return iter(self.dataset) #self.dataset.__iter__()

    # def __iter__(self):
    #     # 获取 worker 信息
    #     worker_info = get_worker_info()
        
    #     # 获取分布式信息 
    #     rank = dist.get_rank() if dist.is_initialized() else 0
    #     world_size = dist.get_world_size() if dist.is_initialized() else 1

    #     # 计算这个 worker 的分片
    #     if worker_info is None:
    #         # 单worker时
    #         worker_id = 0
    #         num_workers = 1
    #     else:
    #         # 多worker时
    #         worker_id = worker_info.id
    #         num_workers = worker_info.num_workers
            
    #     # 总分片数 = world_size * num_workers
    #     total_shards = world_size * num_workers
    #     # 当前分片索引 = rank * num_workers + worker_id
    #     shard_idx = rank * num_workers + worker_id
        
    #     # 加载和分片数据
    #     dataset = self._load_datasets()
    #     dataset = dataset.shard(
    #         num_shards=total_shards,
    #         index=shard_idx
    #     )
        
    #     # 打乱和批处理
    #     dataset = dataset.shuffle(
    #         buffer_size=self.buffer_size,
    #         seed=self.seed + shard_idx
    #     ).batch(self.batch_size)
        
    #     return iter(dataset)
    
    def __len__(self):
        # # 注意：对于流式数据集，这个长度是预估的
        # return sum(
        #     len(list(Path(root_dir).glob('arrow_files/*.arrow')))
        #     for root_dir in self.root_dirs
        # ) * 1000  # 假设每个文件平均有1000个样本
        return self.length//self.batch_size

import os
from pathlib import Path

def get_dataset_dirs(base_path: str) -> List[str]:
    """
    获取指定路径下的所有数据集目录
    
    Args:
        base_path: 基础路径 ("/home/defucao/workspace/TEMPO/dataset/chronos_arrow")
    
    Returns:
        包含所有子目录完整路径的列表
    """
    base_path = Path(base_path)
    
    # 获取所有子目录
    dataset_dirs = [
        str(d) for d in base_path.iterdir() 
        if d.is_dir()
    ]
    
    if not dataset_dirs:
        raise ValueError(f"No subdirectories found in {base_path}")
        
    print(f"Found {len(dataset_dirs)} datasets:")
    for dir_path in dataset_dirs:
        print(f"  - {os.path.basename(dir_path)}")
        
    return dataset_dirs

# 使用示例
# 使用示例
if __name__ == "__main__":
    # 定义数据目录
    root_dirs = get_dataset_dirs("/home/defucao/workspace/TEMPO/dataset/chronos_arrow")

    
    # 创建数据集
    dataset = StreamingMonashDataset(
        root_dirs=root_dirs,
        batch_size=1024,
        buffer_size=10000,
        seed=42,
        verbose=True,
        num_workers=4,  # 调整为CPU核心数
        cache_dir='./cache',  # 添加缓存
        prefetch_factor=2
    )
    
    # def custom_collate_fn(batch):
    #     # 调整 batch 中数据的维度顺序
    #     batch = torch.stack(batch)
    #     batch = batch.transpose(0, 1)  # 或者使用 permute
    #     return batch

    dataloader = DataLoader(
    dataset,
    batch_size=1000,
    num_workers=0,  # 多进程加载
    prefetch_factor=None,  # 预加载的batch数
    pin_memory=True,  # 使用固定内存，加快GPU传输
    drop_last=True,  # 丢弃不完整的批次
    # collate_fn=custom_collate_fn
    )

    # 训练循环
    for  batch in tqdm(dataloader): #enumerate(
        # 处理批次数据
        # print(batch_idx, np.array(batch['x_target']).shape)
        pass

  
    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=1024,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=4
    # )
    # # 训练循环示例
    # for batch_idx, batch in enumerate(tqdm(dataset)):
    #     # 处理批次数据
    #     print(batch_idx, np.array(batch['x_target']).shape)
        # import pdb; pdb.set_trace()
        # print()
        # np.array(batch['x_target']).shape: (32, 336)
        # np.array(batch['y_target']).shape: (32, 96)
        # np.array(batch['x_trend']).shape: (32, 336)

    
        # pass