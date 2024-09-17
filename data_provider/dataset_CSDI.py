import os
# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

class Dataset_ECL(Dataset):
    def __init__(self, root_path='../CSDI/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
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
        
        self.test_length= 24*7
        self.valid_length = 24*5
            
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
            self.max_sen_len = 512 #self.txt_embeddings.shape[1]
            self.tokenizer = tokenizer
            self.llm = llm
        else:
            self.max_sen_len = 1048
        self.__read_data__()
       
        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path='../CSDI/data/', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'hourly electricity consumption of 370 customers.'
        # electricity
        paths= self.root_path + self.data_path + '/data.pkl' 
        with open(paths, 'rb') as f:
            self.main_data, self.mask_data = pickle.load(f)
        paths= self.root_path + self.data_path + '/meanstd.pkl'
        with open(paths, 'rb') as f:
            self.mean_data, self.std_data = pickle.load(f)
        if self.scale:
            self.main_data = (self.main_data - self.mean_data) / self.std_data

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
        
        self.txt_embeddings = []
        
        directory = self.root_path + 'txt_embeddings/' + self.data_path + '/' +str(self.set_type) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.text_condition:
            #maybe think about how missing value affects the calculation of input statistics
            save_embs = []
            if os.path.exists(directory):  
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        save_embs.append(int(file[:-4]))
            if len(save_embs) > 0:
                save_embs.sort()
                print(save_embs)
                # for i in range(len(save_embs)):
                txt_embeddings = np.load(directory + str(save_embs[-1]) + '.npy')
                txt_embeddings = txt_embeddings.tolist()
                self.txt_embeddings += txt_embeddings
                # last_embs = save_embs[-1]
                print(len(self.txt_embeddings))
              
            print(directory)
            print(len(self.use_index))
            for i in range(len(self.txt_embeddings), len(self.use_index)):
                
                i = self.use_index[i]       
                
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
                # all_txts.append(self.dataset_description + " "+stats)
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(data_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                # print(txt_embeddings.shape)
                txt_embeddings = torch.max(txt_embeddings, dim = 0)[0]
                
                txt_embeddings = txt_embeddings.detach().cpu().numpy()
                self.txt_embeddings.append(np.expand_dims(txt_embeddings,axis=0))
                if len(self.txt_embeddings) % 10000 == 0 :
                    print('len(self.txt_embeddings)', len(self.txt_embeddings))
                    
                    np.save(directory + str(i) + '.npy', np.array(self.txt_embeddings))
            np.save(directory + str(len(self.use_index)) + '.npy', np.array(self.txt_embeddings))
            print("txt_embeddings.shape = {}".format(len(self.txt_embeddings)))
            
            del self.llm
            del self.tokenizer
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)

        if self.text_condition:
            text_embedding = self.txt_embeddings[orgindex]
            text_embedding = torch.from_numpy(np.array(text_embedding)).float()
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


class Dataset_Solar(Dataset):
    def __init__(self, root_path='../CSDI/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
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
        
        self.test_length= 24*7
        self.valid_length = 24*5
            
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
        self.text_condition = text_condition
        if self.text_condition:
            self.max_sen_len = 512 #self.txt_embeddings.shape[1]
            self.tokenizer = tokenizer
            self.llm = llm
        else:
            self.max_sen_len = 1048
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path='../CSDI/data/', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
        self.dataset_description = 'hourly solar power production records of 137 stations in Alabama State.'
        csv_data = []
        paths= self.root_path + self.data_path #+ '/train.csv' 
        with open(paths, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                csv_data.append(row)
        csv_data = np.array(csv_data)
        self.main_data = csv_data.astype(np.float32)
        self.mask_data = np.ones_like(self.main_data)
        self.mean_data = np.mean(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        self.std_data = np.std(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        if self.scale:
            self.main_data = (self.main_data - self.mean_data) / self.std_data

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
        
        self.txt_embeddings = []
        
        directory = self.root_path + 'txt_embeddings/' + self.data_path + '/' +str(self.set_type) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.text_condition:
            #maybe think about how missing value affects the calculation of input statistics
            save_embs = []
            if os.path.exists(directory):  
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        save_embs.append(int(file[:-4]))
            if len(save_embs) > 0:
                save_embs.sort()
                print(save_embs)
                # for i in range(len(save_embs)):
                txt_embeddings = np.load(directory + str(save_embs[-1]) + '.npy')
                txt_embeddings = txt_embeddings.tolist()
                self.txt_embeddings += txt_embeddings
                # last_embs = save_embs[-1]
                print(len(self.txt_embeddings))
              
            print(directory)
            print(len(self.use_index))
            for i in range(len(self.txt_embeddings), len(self.use_index)):
                
                i = self.use_index[i]       
                
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
                # all_txts.append(self.dataset_description + " "+stats)
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(data_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                # print(txt_embeddings.shape)
                txt_embeddings = torch.max(txt_embeddings, dim = 0)[0]
                
                txt_embeddings = txt_embeddings.detach().cpu().numpy()
                self.txt_embeddings.append(np.expand_dims(txt_embeddings,axis=0))
                if len(self.txt_embeddings) % 10000 == 0 :
                    print('len(self.txt_embeddings)', len(self.txt_embeddings))
                    
                    np.save(directory + str(i) + '.npy', np.array(self.txt_embeddings))
            np.save(directory + str(len(self.use_index)) + '.npy', np.array(self.txt_embeddings))
            print("txt_embeddings.shape = {}".format(len(self.txt_embeddings)))
            
            del self.llm
            del self.tokenizer
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
      
        if self.text_condition:
            text_embedding = self.txt_embeddings[orgindex]
            text_embedding = torch.from_numpy(np.array(text_embedding)).float()
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


class Dataset_Traffic(Dataset):
    def __init__(self, root_path='../CSDI/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
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
        
        self.test_length= 24*7
        self.valid_length = 24*5
            
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
        self.text_condition = text_condition
        if self.text_condition:
            self.max_sen_len = 512 #self.txt_embeddings.shape[1]
            self.tokenizer = tokenizer
            self.llm = llm
        else:
            self.max_sen_len = 1048
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path='../CSDI/data/', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
        
        self.dataset_description = 'hourly occupancy rate of 963 San Fancisco freeway car lanes.'
        csv_data = []
        paths= self.root_path + self.data_path + '/train.csv' 
        with open(paths, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                csv_data.append(row)
        csv_data = np.array(csv_data)
        self.main_data = csv_data.astype(np.float32)
        self.mask_data = np.ones_like(self.main_data)
        self.mean_data = 0 #np.mean(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        self.std_data = 1 #np.std(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        if self.scale:
            self.main_data = (self.main_data - self.mean_data) / self.std_data

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
        
        self.txt_embeddings = []
        
        directory = self.root_path + 'txt_embeddings/' + self.data_path + '/' +str(self.set_type) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.text_condition:
            #maybe think about how missing value affects the calculation of input statistics
            save_embs = []
            if os.path.exists(directory):  
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        save_embs.append(int(file[:-4]))
            if len(save_embs) > 0:
                save_embs.sort()
                print(save_embs)
                # for i in range(len(save_embs)):
                txt_embeddings = np.load(directory + str(save_embs[-1]) + '.npy')
                txt_embeddings = txt_embeddings.tolist()
                self.txt_embeddings += txt_embeddings
                # last_embs = save_embs[-1]
                print(len(self.txt_embeddings))
              
            print(directory)
            print(len(self.use_index))
            for i in range(len(self.txt_embeddings), len(self.use_index)):
                
                i = self.use_index[i]       
                
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
                # all_txts.append(self.dataset_description + " "+stats)
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(data_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                # print(txt_embeddings.shape)
                txt_embeddings = torch.max(txt_embeddings, dim = 0)[0]
                
                txt_embeddings = txt_embeddings.detach().cpu().numpy()
                self.txt_embeddings.append(np.expand_dims(txt_embeddings,axis=0))
                if len(self.txt_embeddings) % 10000 == 0 :
                    print('len(self.txt_embeddings)', len(self.txt_embeddings))
                    
                    np.save(directory + str(i) + '.npy', np.array(self.txt_embeddings))
            np.save(directory + str(len(self.use_index)) + '.npy', np.array(self.txt_embeddings))
            print("txt_embeddings.shape = {}".format(len(self.txt_embeddings)))
            
            del self.llm
            del self.tokenizer
            
    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
      
        if self.text_condition:
            text_embedding = self.txt_embeddings[orgindex]
            text_embedding = torch.from_numpy(np.array(text_embedding)).float()
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


class Dataset_Wiki(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 120 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 30 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.test_length= 30*5
        self.valid_length = 30*2
            
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
        self.text_condition = text_condition
        if self.text_condition:
            self.max_sen_len = 512 #self.txt_embeddings.shape[1]
            self.tokenizer = tokenizer
            self.llm = llm
        else:
            self.max_sen_len = 1048
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path, self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'daily page views of 2000 Wikipedia pages'
        # paths= self.root_path + self.data_path + '/train.csv' 
        paths= self.root_path + self.data_path + '/test.csv' 
        csv_data = pd.read_csv(paths)
        csv_data = csv_data.values
        csv_data = np.array(csv_data)
        self.main_data = csv_data.astype(np.float32)
        self.mask_data = np.ones_like(self.main_data)
        self.mean_data = np.mean(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        self.std_data = np.std(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        if self.scale:
            self.main_data = (self.main_data - self.mean_data) / self.std_data

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
        
        self.txt_embeddings = []
        
        directory = self.root_path + 'txt_embeddings/' + self.data_path + '/' +str(self.set_type) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.text_condition:
            #maybe think about how missing value affects the calculation of input statistics
            save_embs = []
            if os.path.exists(directory):  
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        save_embs.append(int(file[:-4]))
            if len(save_embs) > 0:
                save_embs.sort()
                print(save_embs)
                # for i in range(len(save_embs)):
                txt_embeddings = np.load(directory + str(save_embs[-1]) + '.npy')
                txt_embeddings = txt_embeddings.tolist()
                self.txt_embeddings += txt_embeddings
                # last_embs = save_embs[-1]
                print(len(self.txt_embeddings))
              
            print(directory)
            print(len(self.use_index))
            for i in range(len(self.txt_embeddings), len(self.use_index)):
                
                i = self.use_index[i]       
                
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
                # all_txts.append(self.dataset_description + " "+stats)
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(data_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                # print(txt_embeddings.shape)
                txt_embeddings = torch.max(txt_embeddings, dim = 0)[0]
                
                txt_embeddings = txt_embeddings.detach().cpu().numpy()
                self.txt_embeddings.append(np.expand_dims(txt_embeddings,axis=0))
                if len(self.txt_embeddings) % 10000 == 0 :
                    print('len(self.txt_embeddings)', len(self.txt_embeddings))
                    
                    np.save(directory + str(i) + '.npy', np.array(self.txt_embeddings))
            np.save(directory + str(len(self.use_index)) + '.npy', np.array(self.txt_embeddings))
            print("txt_embeddings.shape = {}".format(len(self.txt_embeddings)))
            
            del self.llm
            del self.tokenizer
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
      
        if self.text_condition:
            text_embedding = self.txt_embeddings[orgindex]
            text_embedding = torch.from_numpy(np.array(text_embedding)).float()
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

    
class Dataset_Taxi(Dataset):
    def __init__(self, root_path='../CSDI/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 72 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # self.test_length= 30*5
        self.valid_length = 24*2
            
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
        self.text_condition = text_condition
        if self.text_condition:
            self.max_sen_len = 512 #self.txt_embeddings.shape[1]
            self.tokenizer = tokenizer
            self.llm = llm
        else:
            self.max_sen_len = 1048
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path='../CSDI/data/', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'half hourly traffic time series of New York taxi rides taken at 1214 locations in the months of January 2015 for training and January 2016 for test.'
        # paths= self.root_path + self.data_path + '/train.csv' 
        # datafolder = './data/taxi_30min/'
        paths=self.root_path+'train/train.csv' 
        csv_data = pd.read_csv(paths)
        csv_data = csv_data.values
        paths=self.root_path+'test/test.csv' 
        csv_data_test = pd.read_csv(paths)
        csv_data_test = csv_data_test.values
        self.main_data = csv_data.astype(np.float32)
        self.mask_data = np.ones_like(self.main_data)

        self.mean_data = np.mean(self.main_data, axis=0)
        self.std_data = np.std(self.main_data, axis=0)
        
        self.test_data = csv_data_test.astype(np.float32)
        self.mask_test = np.ones_like(self.test_data)
        if self.scale:
            self.main_data = (self.main_data - self.mean_data) / self.std_data
            self.test_data = (self.test_data - self.mean_data) / self.std_data

        total_length = len(self.main_data)
        df_raw = pd.DataFrame(self.main_data)

        total_length_test = len(self.test_data)
        df_raw_test = pd.DataFrame(self.test_data)

        if self.features == 'M' or self.features == 'MS':
            # cols_data = df_raw.columns#[1:]
            df_data = df_raw #[cols_data]
            df_data_test = df_raw_test
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        data = df_data.values
        if self.set_type == 0:
            start = 0
            end = total_length- self.seq_length +1 #total_length - self.seq_length - self.valid_length + 1
            self.use_index = np.arange(start,end,1)
        elif self.set_type == 1:
            start = total_length - self.seq_length - self.valid_length + self.pred_len   #0 #total_length_test - self.seq_length - self.valid_length - self.test_length + self.pred_length
            end = total_length - self.seq_length + self.pred_len  #0 #total_length_test - self.seq_length - self.valid_length - self.test_length + self.pred_length
            
            self.use_index = np.arange(start,end,self.pred_len)
        else:
            start = 0 #total_length_test - 24*56 - self.seq_length +1 #total_length - self.seq_length - self.test_length + self.pred_length
            end = total_length_test - self.seq_length + self.pred_len
            self.use_index = np.arange(start,end,self.pred_len)

            self.use_index = self.use_index[:57]

        self.data_x = data[self.use_index]
        
        self.txt_embeddings = []
        
        directory = self.root_path + 'txt_embeddings/' + self.data_path + '/' +str(self.set_type) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.text_condition:
            #maybe think about how missing value affects the calculation of input statistics
            save_embs = []
            if os.path.exists(directory):  
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        save_embs.append(int(file[:-4]))
            if len(save_embs) > 0:
                save_embs.sort()
                print(save_embs)
                # for i in range(len(save_embs)):
                txt_embeddings = np.load(directory + str(save_embs[-1]) + '.npy')
                txt_embeddings = txt_embeddings.tolist()
                self.txt_embeddings += txt_embeddings
                # last_embs = save_embs[-1]
                print(len(self.txt_embeddings))
              
            print(directory)
            print(len(self.use_index))
            for i in range(len(self.txt_embeddings), len(self.use_index)):
                
                i = self.use_index[i]       
                
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
                # all_txts.append(self.dataset_description + " "+stats)
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(data_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                # print(txt_embeddings.shape)
                txt_embeddings = torch.max(txt_embeddings, dim = 0)[0]
                
                txt_embeddings = txt_embeddings.detach().cpu().numpy()
                self.txt_embeddings.append(np.expand_dims(txt_embeddings,axis=0))
                if len(self.txt_embeddings) % 10000 == 0 :
                    print('len(self.txt_embeddings)', len(self.txt_embeddings))
                    
                    np.save(directory + str(i) + '.npy', np.array(self.txt_embeddings))
            np.save(directory + str(len(self.use_index)) + '.npy', np.array(self.txt_embeddings))
            print("txt_embeddings.shape = {}".format(len(self.txt_embeddings)))
            
            del self.llm
            del self.tokenizer
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len

        index = self.use_index[orgindex]
        if self.set_type==0 or self.set_type==1:
            seq_x = self.main_data[index:index+self.seq_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            observed_mask = self.mask_data[index:index+self.seq_len]
        else:
            seq_x = self.test_data[index:index+self.seq_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            observed_mask = self.mask_test[index:index+self.seq_len]
      
        if self.text_condition:
            text_embedding = self.txt_embeddings[orgindex]
            text_embedding = torch.from_numpy(np.array(text_embedding)).float()
        else:
            text_embedding = torch.zeros(1,768)

       
        

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_x = seq_x.permute(1,0)
        observed_mask = torch.tensor(observed_mask, dtype=torch.long)
        observed_mask = observed_mask.permute(1,0)
        return seq_x, text_embedding, observed_mask
    
    def __len__(self):
        return len(self.use_index)


class Dataset_Exchange(Dataset):
    def __init__(self, root_path='../CSDI/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 120 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 30 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.test_length= 30*5
        self.valid_length = 30*2
            
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
        self.text_condition = text_condition
        if self.text_condition:
            self.max_sen_len = 512 #self.txt_embeddings.shape[1]
            self.tokenizer = tokenizer
            self.llm = llm
        else:
            self.max_sen_len = 1048
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path='../CSDI/data/', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'
            
        paths= self.root_path + self.data_path + '/train.csv' 
        csv_data = pd.read_csv(paths)
        csv_data = csv_data.values
        csv_data = np.array(csv_data)
        self.main_data = csv_data.astype(np.float32)
        self.mask_data = np.ones_like(self.main_data)
        self.mean_data = 0 #np.mean(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        self.std_data = 1 #np.std(self.main_data[:-(self.test_length + self.valid_length)], axis=0)
        if self.scale:
            self.main_data = (self.main_data - self.mean_data) / self.std_data

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
        
        self.txt_embeddings = []
        
        directory = self.root_path + 'txt_embeddings/' + self.data_path + '/' +str(self.set_type) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.text_condition:
            #maybe think about how missing value affects the calculation of input statistics
            save_embs = []
            if os.path.exists(directory):  
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        save_embs.append(int(file[:-4]))
            if len(save_embs) > 0:
                save_embs.sort()
                print(save_embs)
                # for i in range(len(save_embs)):
                txt_embeddings = np.load(directory + str(save_embs[-1]) + '.npy')
                txt_embeddings = txt_embeddings.tolist()
                self.txt_embeddings += txt_embeddings
                # last_embs = save_embs[-1]
                print(len(self.txt_embeddings))
              
            print(directory)
            print(len(self.use_index))
            for i in range(len(self.txt_embeddings), len(self.use_index)):
                
                i = self.use_index[i]       
                
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
                # all_txts.append(self.dataset_description + " "+stats)
                data_text = self.dataset_description + " "+stats
                all_txts_token = self.tokenizer(data_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_sen_len).input_ids
                txt_embeddings = torch.squeeze(self.llm.get_input_embeddings()(all_txts_token))
                # print(txt_embeddings.shape)
                txt_embeddings = torch.max(txt_embeddings, dim = 0)[0]
                
                txt_embeddings = txt_embeddings.detach().cpu().numpy()
                self.txt_embeddings.append(np.expand_dims(txt_embeddings,axis=0))
                if len(self.txt_embeddings) % 10000 == 0 :
                    print('len(self.txt_embeddings)', len(self.txt_embeddings))
                    
                    np.save(directory + str(i) + '.npy', np.array(self.txt_embeddings))
            np.save(directory + str(len(self.use_index)) + '.npy', np.array(self.txt_embeddings))
            print("txt_embeddings.shape = {}".format(len(self.txt_embeddings)))
            
            del self.llm
            del self.tokenizer
            

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        index = self.use_index[orgindex]
        seq_x = self.main_data[index:index+self.seq_len]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
      
        if self.text_condition:
            text_embedding = self.txt_embeddings[orgindex]
            text_embedding = torch.from_numpy(np.array(text_embedding)).float()
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

    

