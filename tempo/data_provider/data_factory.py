from tempo.data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_TSF, \
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_GIFT
from tempo.data_provider.data_loader_monash import Dataset_Monash
from torch.utils.data import DataLoader
from tempo.data_provider.data_loader_monash_stream import StreamingMonashDataset
from datasets import load_dataset, interleave_datasets
import pickle
import os
import torch 
data_dict = {
    'custom': Dataset_Custom,
    'tsf_data': Dataset_TSF,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'monash': StreamingMonashDataset,
    'gift': Dataset_GIFT
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
    if args.data == 'gift':
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
        # data_name='custom', seq_len=96, label_len=48, pred_len=96, 
        #         features='S', target='OT', scale=True, timeenc=0, freq='h',
        #         stl_position='./stl_data/', flag='train', term = 'short'

        # try:
            # import pdb; pdb.set_trace()
            # data_set = torch.load(f'./TEMPO_data_loader/{flag}/{args.data_name.replace('/', '_')}_{args.root_path}_{args.seq_len}_{args.pred_len}.pth')
            # filename = f'{data_name.replace("/", "_")}_{root_path}_{seq_len}_{pred_len}.pkl'
        filename = f'{args.data_name.replace("/", "_")}_{args.root_path}_{args.seq_len}_{args.pred_len}.pkl'
            
        load_path = f'./TEMPO_data_loader/{flag}/{filename}'
    
        try:
            # import pdb; pdb.set_trace()
            with open(load_path, 'rb') as f:
                data_set = pickle.load(f)
        except:
            data_set = Data(
                data_name = args.data_name,
                term=args.root_path,
                flag=flag,
                seq_len = args.seq_len,
                label_len = args.label_len,
                pred_len = args.pred_len,
                features=args.features,
                target=args.target,
                timeenc=0,
                freq=args.freq,
            )
            

            # 确保目录存在
            save_dir = f'./TEMPO_data_loader/{flag}/'
            os.makedirs(save_dir, exist_ok=True)

            # 构建文件名，替换'/'为'_'
            filename = f'{args.data_name.replace("/", "_")}_{args.root_path}_{args.seq_len}_{args.pred_len}.pkl'
            save_path = os.path.join(save_dir, filename)

            # 保存数据集到 pkl 文件
            with open(save_path, 'wb') as f:
                pickle.dump(data_set, f)

            print(f"数据集已保存到: {save_path}")
            # torch.save(data_set, f'./TEMPO_data_loader/{flag}/{args.data_name.replace('/', '_')}_{args.root_path}_{args.seq_len}_{args.pred_len}.pth')
       
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last_test)
        return data_set, data_loader
    
    elif args.data == 'monash':
         # 定义数据目录
        # root_dirs = get_dataset_dirs("./dataset/chronos_ready")
        # root_dirs = get_dataset_dirs("./dataset/chronos_arrow")
        # root_dirs = get_dataset_dirs("./dataset/gift_eval_arrow")
        # root_dirs = get_dataset_dirs("./dataset/chronos_0.01_together/arrow_files")


        # data_set = Data(
        #     root_dirs=root_dirs,
        #     batch_size=args.batch_size,
        #     buffer_size=10000,
        #     seed=42,
        #     verbose=True,
        # )

        # data_set = load_dataset(
        #             "arrow",
        #             data_dir=str(Path('./dataset/chronos_0.2_together/arrow_files')),
        #             # data_dir=str(Path('./dataset/chronos_0.01_together/arrow_files')),
        #             # data_dir=str(Path('./dataset/gift_eval_arrow_together/arrow_files')),
        #             split='train',
        #             # streaming=True
        #             streaming=False
        #         ).with_format("torch")

        print(f"加载数据集: {args.data_path}")
        if flag == 'train':
            data_set = load_dataset(
                        "arrow",
                        # data_dir=str(Path('./dataset/chronos_0.1_together/arrow_files')),
                        # data_dir=str(Path('./dataset/chronos_0.2_together/arrow_files')),
                        # data_dir=str(Path('./dataset/gift_eval_big_arrow_together/arrow_files')),
                        # data_dir=str(Path('./dataset/finetune_gift_eval/arrow_files')),
                        # data_dir=str(Path('./dataset/gift_eval_more_arrow/temperature_rain_with_missing')),
                        # data_dir=str(Path('./dataset/gift_eval_more_arrow/electricity/H')),
                        # data_dir=str(Path('./dataset/gift_eval_more_arrow/electricity/15T')),
                        # gift_eval_skip_48_together
                        data_dir=str(Path('./dataset/gift_eval_skip_48_together/arrow_files')),

                        
                        split='train',
                        # streaming=True
                        streaming=False
                    ).with_format("torch")
            num_samples = len(data_set)
            print(f"数据集中的样本数量: {num_samples}")
            # import pdb; pdb.set_trace()
        else:
            data_set = load_dataset(
                        "arrow",
                        # data_dir=str(Path('./dataset/chronos_0.1_together/arrow_files')),
                        # data_dir=str(Path('./dataset/chronos_0.01_together/arrow_files')),
                        data_dir=str(Path('./dataset/finetune_gift_eval/arrow_files')),

                        split='train',
                        # streaming=True
                        streaming=False
                    ).with_format("torch")
    
        

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            # shuffle=False,
            shuffle=True,
            num_workers=4
        )

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
