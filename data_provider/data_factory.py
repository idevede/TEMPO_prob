from .data_loader import Dataset_Custom, Dataset_Pred,Dataset_ETT_hour, Dataset_ETT_minute#,  #Dataset_GDELT, Dataset_PSM
# from .ci_dataset import Dataset_AQ_CI, Dataset_Energy_CI, Dataset_Stock_CI
from .dataset_CSDI import Dataset_ECL, Dataset_Solar, Dataset_Traffic, Dataset_Wiki, \
    Dataset_Taxi, Dataset_Exchange, Dataset_Traffic_862, Dataset_AQ, Dataset_nasdaq, \
        Dataset_mimic, Dataset_physionet, Dataset_M5
from .dataset_lagllama import Dataset_PM25, Dataset_Physics, Dataset_Cloud
from torch.utils.data import DataLoader, ConcatDataset
# from .dataset_pde import Dataset_PDE

data_dict = {
    'aq': Dataset_AQ,
    'ecl': Dataset_ECL,
    'solar': Dataset_Solar,
    'traffic': Dataset_Traffic,
    'wiki': Dataset_Wiki,
    'taxi': Dataset_Taxi,
    'exchange': Dataset_Exchange,
    'pm25': Dataset_PM25,
    'weather': Dataset_PM25, # weather3010
    'phy': Dataset_Physics,
    'cloud': Dataset_Cloud,
    'traffic_862': Dataset_Traffic_862,
    'nasdaq': Dataset_nasdaq,
    'mimic': Dataset_mimic,
    'physionet': Dataset_physionet,
    'm5': Dataset_M5,
    'ett_h': Dataset_ETT_hour,
    'ett_m': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    # 'gdelt': Dataset_GDELT,


    # 'pde': Dataset_PDE
    
}


# def data_provider(args, flag="train", drop_last_test=False, train_all=False,overwrite_shuffle=None,max_sen_len=None):
#     Data = data_dict[args.data]
#     timeenc = 0 

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = drop_last_test
#         batch_size = args.batch_size
#     elif flag == 'pred':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = 1
#         Data = Dataset_Pred
#     elif flag == 'val':
#         shuffle_flag = True
#         drop_last = drop_last_test
#         batch_size = args.batch_size
#     else:
#         shuffle_flag = True
#         drop_last = False
#         batch_size = args.batch_size
#     if overwrite_shuffle is not None:
#         shuffle_flag = overwrite_shuffle
#         print("overwriting shuffle to ", overwrite_shuffle)
#     if args.data_path is not None:
#         data_set = Data(
#             size=[args.seq_len, 0, 0],
#             timeenc=timeenc,
#             split=flag,
#             data_path=args.data_path,
#             max_sen_len=max_sen_len
#         )
#     else:
#         data_set = Data(
#             size=[args.seq_len, 0, 0],
#             timeenc=timeenc,
#             split=flag,
#             max_sen_len=max_sen_len
#         )
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader

def data_provider(args, flag="train", drop_last_test=False, train_all=False,overwrite_shuffle=None):
    all_datasets = []
    all_dataloader = []
    
    # for i in range(len(args.data)):
    # d = args.data[i]
    d = args.data
    # import pdb; pdb.set_trace()
    print("loading data: ", d)
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
    if overwrite_shuffle is not None:
        shuffle_flag = overwrite_shuffle
        print("overwriting shuffle to ", overwrite_shuffle)
    if d in ['exchange', 'wiki', 'taxi', 'traffic_862', 'aq', 'nasdaq']:
        data_set = Data(
            size=[args.seq_len, 0, args.pred_len],
            split=flag,
            root_path = args.root_path,
            data_path=args.data_path,
            txt_path = args.txt_path,
            text_condition=args.text_condition,
        )
    else:
        data_set = Data(
            size=[args.seq_len, 0, args.pred_len],
            split=flag,
            root_path = args.root_path,
            data_path=args.data_path,
            # txt_path = args.txt_path,
            # text_condition=args.text_condition,
        )
        # before 11/16
        # data_set = Data(
        #     size=[args.seq_len, 0, args.pred_len],
        #     split=flag,
        #     # root_path = args.root_path,
        #     data_path=args.data_path,
        #     txt_path = args.txt_path,
        #     text_condition=args.text_condition,
        # )
    print("length of data_sets: ", len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,)
    all_datasets.append(data_set)
    all_dataloader.append(data_loader)
    return all_datasets, all_dataloader

