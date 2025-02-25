import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tempo.utils.timefeatures import time_features
from tempo.utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
import pickle
from statsmodels.tsa.seasonal import STL
# import mmap
from typing import List, Tuple, Dict, Optional
import json

warnings.filterwarnings('ignore')

stl_position = 'stl/'

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
        
        # 初始化序列长度参数
        self.seq_len = size[0] if size else 24 * 4 * 4
        self.label_len = size[1] if size else 24 * 4
        self.pred_len = size[2] if size else 24 * 4
        
        # 验证和设置数据集类型
        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        
        # 保存其他参数
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.data_name = data_name
        
        # 设置缓存
        self.cache_dir = Path(root_path) / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # 初始化数据
        self._initialize_data()
        
    def _initialize_data(self):
        """初始化数据集，包括缓存机制"""
        cache_file = self.cache_dir / f'dataset_cache_{self.set_type}.mmap'
        index_file = self.cache_dir / f'dataset_index_{self.set_type}.json'
        
        # 如果缓存存在，直接加载
        if cache_file.exists() and index_file.exists():
            self._load_from_cache(cache_file, index_file)
            return
            
        # 否则重新处理数据
        self._process_and_cache_data(cache_file, index_file)
        
    def _load_from_cache(self, cache_file: Path, index_file: Path):
        """从缓存加载数据"""
        with open(index_file, 'r') as f:
            self.data_index = json.load(f)
        self.mmap_file = np.memmap(cache_file, dtype='float32', mode='r', shape=(len(self.data_index), self.seq_len + self.pred_len, 7))
        
    def _process_and_cache_data(self, cache_file: Path, index_file: Path):
        """处理数据并创建缓存"""
        datasets = []
        directories = ['dataset/chronos', 'dataset/chronos_2']
        
        # 收集数据
        for directory in directories:
            datasets.extend(self._load_directory(directory))
            
        # 确定数据集的划分
        train_ratio, val_ratio = 0.7, 0.1
        n_samples = len(datasets)
        splits = [
            int(n_samples * train_ratio),
            int(n_samples * (train_ratio + val_ratio))
        ]
        
        # 选择相应的数据集部分
        if self.set_type == 0:  # train
            datasets = datasets[:splits[0]]
        elif self.set_type == 1:  # val
            datasets = datasets[splits[0]:splits[1]]
        else:  # test
            datasets = datasets[splits[1]:]
            
        # 创建内存映射文件
        data_shape = (len(datasets), self.seq_len + self.pred_len, 7)
        mmap_array = np.memmap(cache_file, dtype='float32', mode='w+', shape=data_shape)
        
        # 处理并保存数据
        for i, data in enumerate(datasets):
            processed_data = self._process_single_sample(data)
            mmap_array[i] = processed_data
            
        mmap_array.flush()
        
        # 保存索引
        self.data_index = list(range(len(datasets)))
        with open(index_file, 'w') as f:
            json.dump(self.data_index, f)
            
        self.mmap_file = np.memmap(cache_file, dtype='float32', mode='r', shape=data_shape)
        
    def _load_directory(self, directory: str) -> List[Dict]:
        """加载指定目录中的数据"""
        datasets = []
        allowed_keywords = {
            'monash_pedestrian_counts',
            'm4_weekly', 'm4_yearly', 'm4_hourly', 'm4_monthly',
            'ercot'
        }
        
        for file_path in Path(directory).glob('*.pkl'):
            # if any(keyword in file_path.name for keyword in allowed_keywords):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                for key in data:
                    datasets.extend(data[key])
                        
        return datasets
        
    def _process_single_sample(self, data: Dict) -> np.ndarray:
        """处理单个样本数据"""
        seq_x = data['x']['target']
        seq_y = data['y']['target']
        seq_marks = np.zeros((len(seq_x), 4))
        
        processed_data = np.concatenate([
            np.expand_dims(seq_x, axis=-1),
            np.expand_dims(seq_y, axis=-1),
            seq_marks,
            np.expand_dims(data['x_trend'], axis=-1),
            np.expand_dims(data['x_seasonal'], axis=-1),
            np.expand_dims(data['x_resid'], axis=-1)
        ], axis=-1)
        
        return processed_data
        
    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        """获取单个样本"""
        data = self.mmap_file[self.data_index[index]]
        
        # 解包数据
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
            torch.tensor(np.expand_dims(seq_trend, axis=-1)),
            torch.tensor(np.expand_dims(seq_seasonal, axis=-1)),
            torch.tensor(np.expand_dims(seq_resid, axis=-1))
        )
        
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.data_index)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反向转换数据（如果需要）"""
        if hasattr(self, 'scaler'):
            return self.scaler.inverse_transform(data)
        return data
    
'''
class Dataset_Monash(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, data_name = 'etth2', max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()

       


    def __read_data__(self):
        
        def save_large_list(data_list, cache_file, chunk_size=1000):
            with open(cache_file, 'wb') as f:
                for i in range(0, len(data_list), chunk_size):
                    chunk = data_list[i:i + chunk_size]
                    pickle.dump(chunk, f)

        def load_large_list(cache_file):
            result = []
            with open(cache_file, 'rb') as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                        result.extend(chunk)
                    except EOFError:
                        break
            return result

        def load_all_datasets(directory):

            cache_file = os.path.join(directory, 'cache/all_datasets_cache.pkl')
    
            # # Try to load from cache first
            # if os.path.exists(cache_file):
            #     print("Loading from cache file...")
            #     # with open(cache_file, 'rb') as f:
            #     #     return pickle.load(f)
            #     load_large_list(cache_file)
    
            all_datasets_list = []

            # 遍历指定目录中的所有文件
            for filename in os.listdir(directory):
                if filename.endswith('.pkl'):
                    if any(keyword in filename for keyword in [
                        'monash_pedestrian_counts',
                        'm4_weekly', 'm4_yearly', 'm4_hourly', 'm4_monthly',
                        'ercot', 
                    # 'm4_weekly', 'm4_yearly', 'm4_hourly', 'm4_monthly',
                    ]):
                        print(filename)
                        file_path = os.path.join(directory, filename)
                        with open(file_path, 'rb') as file:
                            data = pickle.load(file)
                            
                            
                            for key in list(data.keys()):
                                all_datasets_list.extend(data[key])           
            return all_datasets_list

        # 使用示例
        directory = 'datasets/chronos'
        self.all_datasets_list = load_all_datasets(directory)
        print(len(self.all_datasets_list))
       

    def __getitem__(self, index):
        
        seq_x = self.all_datasets_list[index]['x']['target']
        seq_y = self.all_datasets_list[index]['y']['target']
        seq_x_mark = np.zeros_like(seq_x).reshape(-1, 1)
        seq_y_mark = np.zeros_like(seq_y).reshape(-1, 1)
        seq_trend = self.all_datasets_list[index]['x_trend']
        seq_seasonal = self.all_datasets_list[index]['x_seasonal']
        seq_resid = self.all_datasets_list[index]['x_resid']

        return np.expand_dims(seq_x, axis=-1), \
        np.expand_dims(seq_y, axis=-1), \
        np.repeat(seq_x_mark, 4, axis=1),\
        np.repeat(seq_y_mark, 4, axis=1), \
        torch.tensor(np.expand_dims(seq_trend, axis=-1)), \
        torch.tensor(np.expand_dims(seq_seasonal, axis=-1)), \
        torch.tensor(np.expand_dims(seq_resid, axis=-1))
    #     seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid
    # , seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        return len(self.all_datasets_list)-1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
'''

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, data_name = 'etth2', max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        # print("self.enc_in = {}".format(self.enc_in))
        # print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
       
    def stl_resolve(self, data_raw, data_name):
        """
        STL Global Decomposition
        """
        # self.data_name = 'etth1'
        self.data_name = data_name
        save_stl = stl_position + self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name: #== :
                    res = STL(df, period = 7).fit()
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                else:
                    res = STL(df, period = 24*2).fit()

                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw, data_name=self.data_name)
        # end -dove

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, data_name = 'ettm2', train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def stl_resolve(self, data_raw, data_name):
        """
        STL Global Decomposition
        """
        # self.data_name = 'etth1'
        self.data_name = data_name
        save_stl = stl_position + self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name:
                    res = STL(df).fit() # , period = 7 52？
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period = 24*4).fit()
                else:
                    res = STL(df).fit()
                
                
                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw, data_name=self.data_name)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, data_name = 'weather', max_len=-1, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        # self.save_stl = 'stl/' 
     

    def stl_resolve(self, data_raw):
        """
        STL Global Decomposition
        """
        
        save_stl = stl_position +  self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                
                
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name:
                    res = STL(df).fit() # , period = 7 52？
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period = 24*4).fit()
                elif 'traffic' in self.data_name  or 'electricity' in self.data_name:
                    res = STL(df, period = 24).fit()
                else:
                    res = STL(df).fit()
                
                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

       

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        
       

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # After we get data, we do the stl resolve
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw)
        
        

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_trend = self.trend_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_resid = self.resid_stamp[s_begin:s_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_seasonal, seq_resid

    def __len__(self):
        # return 1000 #(
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 percent=None, train_all=False, period = 24, max_len=-1, data_name = 'weather'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.period = period
        self.data_name  = data_name  
        self.__read_data__()

    def stl_resolve(self, data_raw, period = 24):
        """
        STL Global Decomposition
        """
        
        save_stl = stl_position +  self.data_name   
        # save_stl = 'stl/' + 'weather'   

        self.save_stl = save_stl
        trend_pk = self.save_stl + '/trend.pk'
        seasonal_pk = self.save_stl + '/seasonal.pk'
        resid_pk = self.save_stl + '/resid.pk'
        if os.path.isfile(trend_pk) and os.path.isfile(seasonal_pk) and os.path.isfile(resid_pk):
            with open(trend_pk, 'rb') as f:
                trend_stamp = pickle.load(f)
            with open(seasonal_pk, 'rb') as f:
                seasonal_stamp = pickle.load(f)
            with open(resid_pk, 'rb') as f:
                resid_stamp = pickle.load(f)
        else:
            os.makedirs(self.save_stl, exist_ok=True)
            data_raw['date'] = pd.to_datetime(data_raw['date'])
            data_raw.set_index('date', inplace=True)

            [n,m] = data_raw.shape

            trend_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            seasonal_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)
            resid_stamp = torch.zeros([len(data_raw), m], dtype=torch.float32)

            cols = data_raw.columns
            for i, col in enumerate(cols):
                df = data_raw[col]
                # df = df.resample(self.args.freq).mean().ffill()
                
                
                if 'weather' in self.data_name: # == 'weather':
                    res = STL(df, period = 24*6).fit()
                elif 'ill' in self.data_name:
                    res = STL(df).fit() # 
                elif 'etth1' in self.data_name or 'etth2' in self.data_name:
                    res = STL(df, period = 24).fit()
                elif 'ettm1' in self.data_name or 'ettm2' in self.data_name:
                    res = STL(df, period = 24*4).fit()
                elif 'traffic' in self.data_name  or 'electricity' in self.data_name:
                    res = STL(df, period = 24).fit()
                else:
                    res = STL(df, period = period).fit()
                
                trend_stamp[:, i] = torch.tensor(np.array(res.trend.values), dtype=torch.float32)
                seasonal_stamp[:, i] = torch.tensor(np.array(res.seasonal.values), dtype=torch.float32)
                resid_stamp[:, i] = torch.tensor(np.array(res.resid.values), dtype=torch.float32)
            with open(trend_pk, 'wb') as f:
                pickle.dump(trend_stamp, f)
            with open(seasonal_pk, 'wb') as f:
                pickle.dump(seasonal_stamp, f)
            with open(resid_pk, 'wb') as f:
                pickle.dump(resid_stamp, f)
        return trend_stamp, seasonal_stamp, resid_stamp


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = 0 #len(df_raw) - self.seq_len
        border2 = int(0.1*len(df_raw)) - self.seq_len + 1

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        
        col_date = df_raw.columns[:1]
        df_time = df_raw[col_date]
        data_raw = pd.DataFrame.join(df_time, pd.DataFrame(data))#[border1:border2]
        trend_stamp, seasonal_stamp, resid_stamp = self.stl_resolve(data_raw=data_raw, period = self.period)
        
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.trend_stamp = trend_stamp[border1:border2]
        self.seasonal_stamp = seasonal_stamp[border1:border2]
        self.resid_stamp = resid_stamp[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_trend = self.trend_stamp[s_begin:s_end]
        seq_seasonal = self.seasonal_stamp[s_begin:s_end]
        seq_resid = self.resid_stamp[s_begin:s_end]

        return seq_x.reshape(-1, 1), seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, seq_trend.reshape(-1, 1), seq_seasonal.reshape(-1, 1), seq_resid.reshape(-1, 1)

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='Daily',
                 percent=10, max_len=-1, train_all=False):
        
        self.train_all = train_all
        
        self.seq_len = size[0]
        self.pred_len = size[2]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.max_len = max_len
        if self.max_len == -1:
            self.max_len = 1e8

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(os.path.join(self.root_path,
                                                                                                                              self.data_path))
        self.freq = frequency
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            if self.train_all:
                border1s = [0,          0,          train_len-self.seq_len]
                border2s = [train_len,  train_len,  _len]
            else:
                border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
                border2s = [train_len - self.pred_len,  train_len,                                _len]
            border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            # print("_len = {}".format(_len))
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        if self.train_all:
            border1s = [0,          0,          train_len-self.seq_len]
            border2s = [train_len,  train_len,  _len]
        else:
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
        border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.set_type == 2:
            s_end = -self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        # if self.set_type == 2:
        #     print("data_x.shape = {}, data_y.shape = {}".format(data_x.shape, data_y.shape))

        return data_x, data_y, data_x, data_y

    def __len__(self):
        if self.set_type == 0:
            # return self.tot_len
            return min(self.max_len, self.tot_len)
        else:
            return self.tot_len
