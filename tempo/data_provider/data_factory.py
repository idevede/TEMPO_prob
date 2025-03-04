from tempo.data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from tempo.data_provider.data_loader_monash import Dataset_Monash
from torch.utils.data import DataLoader
from tempo.data_provider.data_loader_monash_stream import StreamingMonashDataset

data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'monash': StreamingMonashDataset,
}

import os
from pathlib import Path

def get_dataset_dirs(base_path: str):
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


def data_provider(args, flag, drop_last_test=True, train_all=False):
    Data = data_dict[args.data]
    if args.data == 'monash':
         # 定义数据目录
        # root_dirs = get_dataset_dirs("./dataset/chronos_ready")
        # root_dirs = get_dataset_dirs("./dataset/chronos_arrow")
        root_dirs = get_dataset_dirs("./dataset/gift_eval_arrow")

    
        data_set = Data(
            root_dirs=root_dirs,
            batch_size=args.batch_size,
            buffer_size=10000,
            seed=42,
            verbose=True,
            # num_workers=4,  # 调整为CPU核心数
            # cache_dir='./cache',  # 添加缓存
            # prefetch_factor=2
        )
        # from datasets import load_dataset
        # data_set = load_dataset(
        #     "arrow",
        #     data_dir='dataset/chronos_arrow/m4_weekly/arrow_files', 
        #     # data_dir='dataset/chronos_arrow/m4_yearly/arrow_files', 
        #     split="train",
        #     streaming=True
        # )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            # collate_fn=collate_fn,
            num_workers=4
        )

    

        # from tqdm import tqdm
        # for batch in tqdm(data_loader):




        # data_loader = DataLoader(
        #     data_set,
        #     batch_size=None,  # batch在dataset中已处理
        #     shuffle=False,    # 不需要shuffle
        #     num_workers=args.num_workers,
        #     # prefetch_factor=args.prefetch_factor
        #     # prefetch_factor=None,  # 预加载的batch数
        #     pin_memory=True,  # 使用固定内存，加快GPU传输
        #     # drop_last=True,  # 丢弃不完整的批次
        # )
        return data_set, data_loader
    
    else:
        timeenc = 0 if args.embed != 'timeF' else 1
        percent = args.percent
        max_len = args.max_len

        if flag == 'test':
            shuffle_flag = False
            drop_last = drop_last_test
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
            Data = Dataset_Pred
        elif flag == 'val':
            shuffle_flag = True
            drop_last = drop_last_test
            batch_size = args.batch_size
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
            data_name = args.data_name
        )
        # print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
