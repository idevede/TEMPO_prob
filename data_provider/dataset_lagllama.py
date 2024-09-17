import os
# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from .timefeatures import time_features
import re
import warnings
from .datautils import calcute_lags
import pickle
import csv

from transformers import GPT2Tokenizer, GPT2Model
llm = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True) 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    pad_token = '[PAD]'
    tokenizer.add_special_tokens({'pad_token': pad_token})
    tokenizer.pad_token = pad_token
for param in llm.parameters():
    param.requires_grad = False
warnings.filterwarnings('ignore')


class Dataset_PM25(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/',
                 data_name = 'pm25'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 192 # 168 +24
            self.label_len = 0
            self.pred_len = 24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        if data_name == 'pm25':
            self.test_length= 24*1
            self.valid_length = 24*14
        elif data_name == 'weather3010':
            self.test_length= 30*1
            self.valid_length = 30*2
        # elif data_name == 'request':
        #     self.test_length= 30*1
        #     self.valid_length = 30*2
        self.data_name = data_name
            
        self.seq_length = self.seq_len  #+ self.pred_length

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]
        self.max_sen_len = 140
        self.text_condition = text_condition
        if self.text_condition:

            self.tokenizer = tokenizer
            self.llm = llm
        
        self.__read_data__()
        if self.text_condition:
            self.max_sen_len = self.txt_embeddings.shape[1]
        else:
            self.max_sen_len = 1048

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path, self.txt_path, 'solar.txt')
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]

        if self.data_name == 'pm25': 
            paths= self.root_path + self.data_path + '/BeijinPM2.5_clean_one_column.csv' 
            csv_data = pd.read_csv(paths).set_index('time').values

            self.main_data = csv_data.astype(np.float32)
            self.mask_data = np.ones_like(self.main_data)
            self.mean_data = np.mean(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
            self.std_data = np.std(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
            if self.scale:
                self.main_data = (self.main_data - self.mean_data) / self.std_data
            self.dataset_description = 'The Beijing PM2.5 dataset contains hourly data of PM2.5 levels recorded by the US Embassy in Beijing. The dataset also includes meteorological data from Beijing Capital International Airport.'

            
        elif self.data_name == 'weather3010':
            datafolder = self.root_path + self.data_path 
            paths=datafolder+'weather_dataset.csv' 
            csv_data = pd.read_csv(paths)
            csv_data = csv_data.values.T[1:,:]
            self.main_data = csv_data.astype(np.float32)
            self.mask_data = np.ones_like(self.main_data)
            self.min_data = np.min(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
            self.max_data = np.max(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
           
            if self.scale:
                # Min-Max scaling
                self.scaler = MinMaxScaler()
                self.scaler.fit(self.main_data[:-(self.test_length)])

                self.main_data = self.scaler.transform(self.main_data)#(self.main_data - self.min_data) / (self.max_data - self.min_data)
            # import pdb;pdb.set_trace()

            self.dataset_description = 'The Solar dataset comprises 6000 simulated time series for 5-minute solar power and hourly forecasts of photovoltaic power plants in the U.S. in 2006. It includes 137 time series reflecting solar power production every 10 minutes in Alabama during 2006'


        
        total_length = len(self.main_data)
        df_raw = pd.DataFrame(self.main_data)

        if self.features == 'M' or self.features == 'MS':
            # cols_data = df_raw.columns#[1:]
            df_data = df_raw #[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        data = df_data.values
        if self.set_type == 0:
            start = 0
            end = total_length - self.seq_length - self.valid_length - self.test_length + 1
            self.use_index = np.arange(start,end,1)
        elif self.set_type == 1:
            start = total_length - self.seq_length - self.valid_length - self.test_length + self.pred_len
            end = total_length - self.seq_length - self.test_length + self.pred_len
            self.use_index = np.arange(start,end,self.pred_len)
        else:
            start = total_length - self.seq_length - self.test_length + self.pred_len
            end = total_length - self.seq_length + self.pred_len
            self.use_index = np.arange(start,end,self.pred_len)

        self.data_x = data[self.use_index]

        if self.text_condition:
            all_txts = []
            #maybe think about how missing value affects the calculation of input statistics
            for i in list(self.use_index): #range(self.data_x.shape[0]-self.seq_len+1):
                seq_x = self.main_data[i:i+self.seq_len-self.pred_len,:]
                # import pdb; pdb.set_trace()
                seq_x = torch.tensor(seq_x, dtype=torch.float32)
                seq_x = seq_x.permute(1,0)
                min_values = torch.min(seq_x, dim=1)[0]
                min_values = [round(a,5) for a in min_values.tolist()]
                max_values = torch.max(seq_x, dim=1)[0]
                max_values = [round(a,5) for a in max_values.tolist()]
                medians = torch.median(seq_x, dim=1).values
                medians = [round(a,5) for a in medians.tolist()]
                lags = calcute_lags(seq_x)
                lags = lags.tolist()
                trends = seq_x.diff(dim=1).sum(dim=1)
                trends = ["upward" if a > 0 else "downward" for a in trends]
                #print("min_values = {}".format(min_values), "max_values = {}".format(max_values), "medians = {}".format(medians), "lags = {}".format(lags), "trends = {}".format(trends))
                stats = ("Input statistics: "
                        f"min values {min_values}, "
                        f"max values {max_values}, "
                        f"median values {medians}, "
                        f"the trend of input are {trends}, "
                        f"top 5 lags are {lags}")
                all_txts.append(self.dataset_description + " "+stats)
            # import pdb; pdb.set_trace()
            self.all_txts = all_txts 
            # import pdb; pdb.set_trace()
            all_txts_token = self.tokenizer(all_txts, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
            # all_txts_token = self.tokenizer(all_txts, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
            # self.max_sen_len = all_txts.shape[1]
            txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
            self.txt_embeddings = txt_embeddings
            # txt_embedding_np = np.array(txt_embeddings)
            # embeding_path = os.path.join(self.root_path,
            #              "ettm1.pt")
            # torch.save(txt_embeddings, embeding_path)
            # # open(embeding_path, "wb").write(txt_embeddings)
            # import pdb; pdb.set_trace()
            # if os.exists(embeding_path):
            #     txt_embeddings = torch.load(embeding_path)
            # else:
            #     self.txt_embeddings = txt_embeddings 
            print("txt_embeddings.shape = {}".format(txt_embeddings.shape))
            del self.llm
            del self.tokenizer
            # del txt_embeddings
            del all_txts_token
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
      
        if self.text_condition:
            # text_embedding = self.get_txt_embeddings(self.all_txts[s_begin])
            text_embedding = self.txt_embeddings[orgindex]
            if len(text_embedding.shape) == 1:
                text_embedding = torch.unsqueeze(text_embedding,dim = 0)
            text_embedding = torch.max(text_embedding, dim = 0)[0]
            text_embedding = torch.unsqueeze(text_embedding,dim = 0)
        else:
            text_embedding = torch.zeros(1,768)
       
        observed_mask = self.mask_data[index:index+self.seq_len]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_x = seq_x.permute(1,0)
        observed_mask = torch.tensor(observed_mask, dtype=torch.long)
        observed_mask = observed_mask.permute(1,0)
        return seq_x, text_embedding, observed_mask
    
    def __len__(self):
        return len(self.use_index)



class Dataset_Cloud(Dataset):
    def __init__(self, root_path='./dataset/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/',
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 180 # 168 +24
            self.label_len = 0
            self.pred_len = 60 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        print(size)

        
            
        self.seq_length = self.seq_len  #+ self.pred_length

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]
        self.max_sen_len = 140
        self.text_condition = text_condition
        if self.text_condition:

            self.tokenizer = tokenizer
            self.llm = llm

        # self.test_length= 60*14
        self.valid_length = 60*14 #24*14
        
        self.__read_data__()
        if self.text_condition:
            self.max_sen_len = self.txt_embeddings.shape[1]
        else:
            self.max_sen_len = 1048

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path, self.txt_path, 'solar.txt')
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]

        
        paths= self.root_path + self.data_path + '_train.csv' 
        csv_data = pd.read_csv(paths, header= None)
        cols_data = csv_data.columns[1:]
        df_data = csv_data[cols_data].values
        print(df_data.shape)
       

        self.main_data = df_data.astype(np.float32)
        self.mask_data = np.ones_like(self.main_data)
        self.mean_data = np.mean(self.main_data, axis=0)
        self.std_data = np.std(self.main_data, axis=0)
        if self.scale:
            if self.set_type == 0:
                self.main_data = (self.main_data - self.mean_data) / self.std_data
            if self.set_type == 1:
                self.main_data = (self.main_data - self.mean_data) / self.std_data
            if self.set_type == 2:
                paths= self.root_path + self.data_path + '_test.csv' 
                csv_data = pd.read_csv(paths, header= None)
                cols_data = csv_data.columns[1:]
                df_data = csv_data[cols_data].values #[-180:]
                self.main_data = df_data.astype(np.float32)
                self.mask_data = np.ones_like(self.main_data)
                self.main_data = (self.main_data - self.mean_data) / self.std_data
        self.dataset_description = 'The Huawei cloud dataset contain serverless traces. We select 8 series containing metrics based on the minute-frequency occurrences of the top 10 functions by median occurrences over 141 days: function delay, platform delay, cpu usage, memory usage, cpu limit, memory limit, instances. platform delay, requests.'

            
        
           
            

        
        total_length = len(self.main_data)
        df_raw = pd.DataFrame(self.main_data)



        # if self.features == 'M' or self.features == 'MS':
        #     # cols_data = df_raw.columns#[1:]
        #     df_data = df_raw #[cols_data]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]
        print(self.set_type)
        data = df_raw.values
        if self.set_type == 0:
            start = 0
            end = total_length - self.seq_length + 1
            self.use_index = np.arange(start,end,1)

        elif self.set_type == 1:
            start = total_length - self.seq_length - self.valid_length
            end = total_length - self.seq_length + 1
            self.use_index = np.arange(start,end,1)
        else:
            start = 0
            end = total_length - self.seq_length + self.pred_len
            self.use_index = np.arange(start,end,self.pred_len)
            # print(self.use_index)

        # self.data_x = data[self.use_index]
        self.data_x = self.main_data[self.use_index]

        if self.text_condition:
            all_txts = []
            #maybe think about how missing value affects the calculation of input statistics
            for i in list(self.use_index): #range(self.data_x.shape[0]-self.seq_len+1):
                seq_x = self.main_data[i:i+self.seq_len-self.pred_len,:]
                # import pdb; pdb.set_trace()
                seq_x = torch.tensor(seq_x, dtype=torch.float32)
                seq_x = seq_x.permute(1,0)
                min_values = torch.min(seq_x, dim=1)[0]
                min_values = [round(a,5) for a in min_values.tolist()]
                max_values = torch.max(seq_x, dim=1)[0]
                max_values = [round(a,5) for a in max_values.tolist()]
                medians = torch.median(seq_x, dim=1).values
                medians = [round(a,5) for a in medians.tolist()]
                lags = calcute_lags(seq_x)
                lags = lags.tolist()
                trends = seq_x.diff(dim=1).sum(dim=1)
                trends = ["upward" if a > 0 else "downward" for a in trends]
                #print("min_values = {}".format(min_values), "max_values = {}".format(max_values), "medians = {}".format(medians), "lags = {}".format(lags), "trends = {}".format(trends))
                stats = ("Input statistics: "
                        f"min values {min_values}, "
                        f"max values {max_values}, "
                        f"median values {medians}, "
                        f"the trend of input are {trends}, "
                        f"top 5 lags are {lags}")
                all_txts.append(self.dataset_description + " "+stats)
            # import pdb; pdb.set_trace()
            self.all_txts = all_txts 
            # import pdb; pdb.set_trace()
            all_txts_token = self.tokenizer(all_txts, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
            # all_txts_token = self.tokenizer(all_txts, return_tensors="pt", padding=True, truncation=True, max_length=1024).input_ids
            # self.max_sen_len = all_txts.shape[1]
            txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
            self.txt_embeddings = txt_embeddings
            # txt_embedding_np = np.array(txt_embeddings)
            # embeding_path = os.path.join(self.root_path,
            #              "ettm1.pt")
            # torch.save(txt_embeddings, embeding_path)
            # # open(embeding_path, "wb").write(txt_embeddings)
            # import pdb; pdb.set_trace()
            # if os.exists(embeding_path):
            #     txt_embeddings = torch.load(embeding_path)
            # else:
            #     self.txt_embeddings = txt_embeddings 
            print("txt_embeddings.shape = {}".format(txt_embeddings.shape))
            del self.llm
            del self.tokenizer
            # del txt_embeddings
            del all_txts_token
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        # print("get item:", seq_x.shape, self.seq_len)
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
      
        if self.text_condition:
            # text_embedding = self.get_txt_embeddings(self.all_txts[s_begin])
            text_embedding = self.txt_embeddings[orgindex]
            if len(text_embedding.shape) == 1:
                text_embedding = torch.unsqueeze(text_embedding,dim = 0)
            text_embedding = torch.max(text_embedding, dim = 0)[0]
            text_embedding = torch.unsqueeze(text_embedding,dim = 0)
        else:
            text_embedding = torch.zeros(1,768)
       
        observed_mask = self.mask_data[index:index+self.seq_len]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_x = seq_x.permute(1,0)
        observed_mask = torch.tensor(observed_mask, dtype=torch.long)
        observed_mask = observed_mask.permute(1,0)
        return seq_x, text_embedding, observed_mask
    
    def __len__(self):
        return len(self.use_index)


class Dataset_Physics(Dataset):
    def __init__(self, root_path='./', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/',
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 192 # 168 +24
            self.label_len = 0
            self.pred_len = 24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        print(size)

        
            
        self.seq_length = self.seq_len  #+ self.pred_length

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]
        self.max_sen_len = 140
        self.text_condition = text_condition
        if self.text_condition:

            self.tokenizer = tokenizer
            self.llm = llm

        # self.test_length= 60*14
        self.valid_length = 60*14 #24*14
        
        self.__read_data__()
        if self.text_condition:
            self.max_sen_len = self.txt_embeddings.shape[1]
        else:
            self.max_sen_len = 1048

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path, self.txt_path, 'solar.txt')
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]

        
        # paths= self.root_path + self.data_path + '1dvorticity.npy' 
        # paths=  './dataset/1dvorticity.npy' 
        paths=  './dataset/1dadvection.npy' 
        paths='./dataset/1dBurgers.npy'
        # paths=  './dataset/1dNavier-Stokes.npy'
        print(paths)
        df_data = np.load(paths)
       

        self.main_data = df_data.astype(np.float32).transpose((0, 2, 1))
        self.mask_data = np.ones_like(self.main_data)
        self.mean_data = 0 #np.mean(self.main_data, axis=0)
        self.std_data = 1 #np.std(self.main_data, axis=0)
        if self.scale:
            if self.set_type == 0:
                self.main_data = (self.main_data - self.mean_data) / self.std_data
            if self.set_type == 1:
                self.main_data = (self.main_data - self.mean_data) / self.std_data
            if self.set_type == 2:
                self.main_data = (self.main_data - self.mean_data) / self.std_data
                
        self.dataset_description = 'To simulate a 1-dimensional Navier-Stokes equation focusing on the vorticity transport and generate multivariate time series data from it, we can use numerical methods like finite difference or spectral methods for spatial discretization and a simple time-stepping method like Euler or Runge-Kutta for time integration.'

        total_length = len(self.main_data)
        self.data_x =  self.main_data
        if self.text_condition:
            all_txts = []
            #maybe think about how missing value affects the calculation of input statistics
            for i in range(total_length):
                seq_x = self.mean_data[i, 0:self.seq_len-self.pred_len]
                # import pdb; pdb.set_trace()
                seq_x = torch.tensor(seq_x, dtype=torch.float32)
                seq_x = seq_x.permute(1,0)
                min_values = torch.min(seq_x, dim=1)[0]
                min_values = [round(a,5) for a in min_values.tolist()]
                max_values = torch.max(seq_x, dim=1)[0]
                max_values = [round(a,5) for a in max_values.tolist()]
                medians = torch.median(seq_x, dim=1).values
                medians = [round(a,5) for a in medians.tolist()]
                lags = calcute_lags(seq_x)
                lags = lags.tolist()
                trends = seq_x.diff(dim=1).sum(dim=1)
                trends = ["upward" if a > 0 else "downward" for a in trends]
                #print("min_values = {}".format(min_values), "max_values = {}".format(max_values), "medians = {}".format(medians), "lags = {}".format(lags), "trends = {}".format(trends))
                stats = ("Input statistics: "
                        f"min values {min_values}, "
                        f"max values {max_values}, "
                        f"median values {medians}, "
                        f"the trend of input are {trends}, "
                        f"top 5 lags are {lags}")
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(all_txts, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                txt_embeddings = torch.max(text_embedding, dim = 0)[0]
                self.txt_embeddings.append(torch.unsqueeze(txt_embeddings,dim = 0))

            # print("txt_embeddings.shape = {}".format(txt_embeddings.shape))
            del self.llm
            del self.tokenizer  
        
           
            

        
       
        
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        
        seq_x = torch.tensor(self.main_data[orgindex], dtype=torch.float32)
      
        if self.text_condition:
            # text_embedding = self.get_txt_embeddings(self.all_txts[s_begin])
            text_embedding = self.txt_embeddings[orgindex]
            # if len(text_embedding.shape) == 1:
            #     text_embedding = torch.unsqueeze(text_embedding,dim = 0)
            # text_embedding = torch.max(text_embedding, dim = 0)[0]
            # text_embedding = torch.unsqueeze(text_embedding,dim = 0)
        else:
            text_embedding = torch.zeros(1,768)
       
        observed_mask = self.mask_data[orgindex]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_x = seq_x.permute(1,0)
        observed_mask = torch.tensor(observed_mask, dtype=torch.long)
        observed_mask = observed_mask.permute(1,0)
        return seq_x, text_embedding, observed_mask
    
    def __len__(self):
        return len(self.main_data)
