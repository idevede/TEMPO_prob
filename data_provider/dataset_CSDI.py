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
import pandas as pd
from statsmodels.tsa.seasonal import STL

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

root_path_all = './csdi_data/' #/u/dcao1/workspace/CSDI_miss_value/data
root_path_all = '/u/dcao1/workspace/CSDI_miss_value/data/'
stl_position = 'stl/'
class Dataset_ECL(Dataset):
    def __init__(self, root_path='./csdi_data', split='train', size=None,
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
        

        self.root_path = root_path_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
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
    def __init__(self, root_path='./csdi_data', split='train', size=None,
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

        self.root_path = root_path_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
        self.dataset_description = 'hourly solar power production records of 137 stations in Alabama State.'
        csv_data = []
        paths= self.root_path + self.data_path + '/train.csv' 
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
    def __init__(self, root_path='./csdi_data', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = None):
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
        self.txt_path = txt_path
            
        self.seq_length = self.seq_len  #+ self.pred_length

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
        
        self.dataset_description = 'hourly occupancy rate of 963 San Fancisco freeway car lanes.'
        csv_data = []
        paths= self.root_path + self.data_path # + '/traffic_missing_data_5.csv' 

        csv_data = pd.read_csv(paths, index_col=0)
        csv_data.fillna(0, inplace=True)
        # with open(paths, 'r', newline='') as csvfile:
        #     csvreader = csv.reader(csvfile)
        #     for row in csvreader:
        #         csv_data.append(row)
        csv_data = csv_data.values
        
        self.main_data = csv_data.astype(np.float32)
        if self.txt_path is not None:
            paths= self.root_path + self.txt_path #+ '/traffic_missing_mask_5.csv'
            csv_data = pd.read_csv(paths, index_col=0) 
            self.mask_data = csv_data.values #np.ones_like(self.main_data)
            # import pdb; pdb.set_trace()
        else:
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

        self.root_path = root_path_all
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
    def __init__(self, root_path='./csdi_data', split='train', size=None,
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

        self.root_path = root_path_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'half hourly traffic time series of New York taxi rides taken at 1214 locations in the months of January 2015 for training and January 2016 for test.'
        # paths= self.root_path + self.data_path + '/train.csv' 
        # datafolder = './data/taxi_30min/'
        paths=self.root_path + self.data_path +'/train/train.csv' 
        csv_data = pd.read_csv(paths)
        csv_data = csv_data.values
        paths=self.root_path+ self.data_path +'/test/test.csv' 
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
    def __init__(self, root_path='./csdi_data', split='train', size=None,
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
        self.txt_path = txt_path
        self.root_path = root_path #_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path + self.data_path # + '/traffic_missing_data_5.csv' 
        print('Data path:', paths)
        # self.root_path = self.root_path[0]
        # self.data_path = self.data_path[0]
        csv_data = pd.read_csv(paths, index_col=0)
        # csv_data = csv_data.drop(csv_data.columns[0], axis=1)
        csv_data.fillna(0, inplace=True)
        # with open(paths, 'r', newline='') as csvfile:
        #     csvreader = csv.reader(csvfile)
        #     for row in csvreader:
        #         csv_data.append(row)
        csv_data = csv_data.values
        
        self.main_data = csv_data.astype(np.float32)
        # import pdb; pdb.set_trace()
        # self.mask_data = np.ones_like(self.main_data)
        # We don't have taxt information in this dataset
        if self.txt_path is not None:
            paths= self.root_path + self.txt_path #+ '/traffic_missing_mask_5.csv'
            # import pdb; pdb.set_trace()
            csv_data = pd.read_csv(paths, index_col=0) 
            # csv_data = csv_data.drop(csv_data.columns[0], axis=1)
            self.mask_data = csv_data.values #np.ones_like(self.main_data)
            # import pdb; pdb.set_trace()
        else:
            self.mask_data = np.ones_like(self.main_data)

        # self.main_data = self.main_data * self.mask_data
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

    
class Dataset_Traffic_862(Dataset):
    def __init__(self, root_path='./csdi_data', split='train', size=None,
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
        self.txt_path = txt_path
        self.root_path = root_path #_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path + self.data_path # + '/traffic_missing_data_5.csv' 
        print('Data path:', paths)
        # self.root_path = self.root_path[0]
        # self.data_path = self.data_path[0]
        csv_data = pd.read_csv(paths, index_col=0)
        # Set unobserved positions to NaN
        #####################################
        # # CSV 重新制作train data
        # paths_mask_matrix= self.root_path + self.txt_path #+ '/traffic_missing_mask_5.csv'
        # import pdb; pdb.set_trace()
        csv_data = csv_data.drop(csv_data.columns[0], axis=1)
        # mask_matrix = pd.read_csv(paths_mask_matrix, index_col=0) 
        # mask_matrix = mask_matrix.drop(mask_matrix.columns[0], axis=1)
        # import pdb; pdb.set_trace()
        # csv_data = csv_data.mask(mask_matrix == 0, np.nan)
        csv_data.replace(0, np.nan, inplace=True)
        csv_data.fillna(method='ffill', inplace=True)
        ######################################
        
        csv_data.fillna(0, inplace=True)
        # with open(paths, 'r', newline='') as csvfile:
        #     csvreader = csv.reader(csvfile)
        #     for row in csvreader:
        #         csv_data.append(row)
        csv_data = csv_data.values
        
        self.main_data = csv_data.astype(np.float32)
        # import pdb; pdb.set_trace()
        # self.mask_data = np.ones_like(self.main_data)
        # We don't have taxt information in this dataset
        if self.txt_path is not None:
            paths= self.root_path + self.txt_path #+ '/traffic_missing_mask_5.csv'
            # import pdb; pdb.set_trace()
            csv_data = pd.read_csv(paths, index_col=0) 
            csv_data = csv_data.drop(csv_data.columns[0], axis=1)
            self.mask_data = csv_data.values #np.ones_like(self.main_data)
            # import pdb; pdb.set_trace()
        else:
            self.mask_data = np.ones_like(self.main_data)

        # self.main_data = self.main_data * self.mask_data
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



class Dataset_AQ(Dataset):
    def __init__(self, root_path='/u/dcao1/workspace/CSDI_miss_value/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 168 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.test_length= self.pred_len*5
        self.valid_length = self.pred_len*2
            
        self.seq_length = self.seq_len  + self.pred_len

        self.seq_len = self.seq_length # 跟TimeDIT写的不一样，这里是seq_length，是History， Timedit里加了furture

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.txt_path = txt_path
        self.root_path = root_path #_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path + self.data_path # + '/traffic_missing_data_5.csv' 
        csv_data = pd.read_csv(paths)
        # csv_data.drop(columns=['Unnamed: 0.1'], inplace=True)
        # import pdb; pdb.set_trace()
        ######################
        csv_data.replace(0, np.nan, inplace=True)
        csv_data.fillna(method='ffill', inplace=True)
        ######################
        csv_data.fillna(0, inplace=True)
        csv_data = csv_data.values
        main_data = csv_data.astype(np.float32)
        self.main_data = main_data
        # self.mask_data = np.ones_like(self.main_data)
        # We don't have taxt information in this dataset
        if self.txt_path is not None:
            paths= self.root_path + self.txt_path #+ '/traffic_missing_mask_5.csv'
            # import pdb; pdb.set_trace()
            csv_data = pd.read_csv(paths) 
          
            self.mask_data = csv_data.values #np.ones_like(self.main_data)
            # import pdb; pdb.set_trace()
        else:
            self.mask_data = np.ones_like(self.main_data)

        self.main_data = self.main_data * self.mask_data

        # Mask the data where mask_data is 0
        masked_data = np.where(self.mask_data != 0, self.main_data, np.nan)
        # import pdb; pdb.set_trace()
        # Calculate mean and std, ignoring NaN values
        self.mean_data = np.nanmean(masked_data[:-(self.test_length + self.valid_length)], axis=0)
        self.std_data = np.nanstd(masked_data[:-(self.test_length + self.valid_length)], axis=0)
        
        self.main_data = (self.main_data - self.mean_data) / self.std_data
        # import pdb; pdb.set_trace()
        # if self.scale:
        #     self.main_data = (self.main_data - self.mean_data) / self.std_data

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
        # import pdb; pdb.set_trace()
        self.enc_in = self.data_x.shape[-1]
        
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        if self.set_type == 2:
            index = self.use_index[orgindex]
            seq_x = self.main_data[index:index+self.seq_len-self.pred_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index+self.seq_len-self.pred_len:index+self.seq_len]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index+self.seq_len-self.pred_len:index+self.seq_len]
            # import pdb; pdb.set_trace()
        else:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            # s_begin = index % self.tot_len
            index = self.use_index[index]
            seq_x = self.main_data[index:index+self.seq_len-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index+self.seq_len-self.pred_len:index+self.seq_len, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index+self.seq_len-self.pred_len:index+self.seq_len, feat_id:feat_id+1]
            # import pdb; pdb.set_trace()
        
        return seq_x, seq_y, observed_mask, observed_mask
    
    def __len__(self):
        if self.set_type == 2:
            return len(self.use_index)
        return len(self.use_index)*self.enc_in   



class Dataset_nasdaq(Dataset):
    def __init__(self, root_path='/u/dcao1/workspace/CSDI_miss_value/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 168 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.test_length= self.pred_len*5
        self.valid_length = self.pred_len*2
            
        self.seq_length = self.seq_len  + self.pred_len

        self.seq_len = self.seq_length # 跟TimeDIT写的不一样，这里是seq_length，是History， Timedit里加了furture

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.txt_path = txt_path
        self.root_path = root_path #_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path + self.data_path # + '/traffic_missing_data_5.csv' 
        csv_data = pd.read_csv(paths, index_col=0)
        # Drop columns that include '1month' in their names
        csv_data = csv_data.drop(columns=[col for col in csv_data.columns if '1month' in col])
        mask_matrix = (~csv_data.isna()).astype(int)

        # csv_data.drop(columns=['Unnamed: 0.1'], inplace=True)
        # import pdb; pdb.set_trace()
        ######################
        csv_data.replace(0, np.nan, inplace=True)
        csv_data.fillna(method='ffill', inplace=True)
        ######################
        csv_data.fillna(0, inplace=True)
        csv_data = csv_data.values
        main_data = csv_data.astype(np.float32)
        self.main_data = main_data
        main_data = csv_data.astype(np.float32)
        self.main_data = main_data
        
        self.mask_data = mask_matrix.values #n
        self.main_data = self.main_data * self.mask_data

        self.main_data = self.main_data * self.mask_data

        # Mask the data where mask_data is 0
        masked_data = np.where(self.mask_data != 0, self.main_data, np.nan)
        # import pdb; pdb.set_trace()
        # Calculate mean and std, ignoring NaN values
        self.mean_data = 0 #np.nanmean(masked_data[:-(self.test_length + self.valid_length)], axis=0)
        self.std_data = 1 #np.nanstd(masked_data[:-(self.test_length + self.valid_length)], axis=0)
        
        self.main_data = (self.main_data - self.mean_data) / self.std_data
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
        # import pdb; pdb.set_trace()
        self.enc_in = self.data_x.shape[-1]
        
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        if self.set_type == 2:
            index = self.use_index[orgindex]
            seq_x = self.main_data[index:index+self.seq_len-self.pred_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index+self.seq_len-self.pred_len:index+self.seq_len]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index+self.seq_len-self.pred_len:index+self.seq_len]
            # import pdb; pdb.set_trace()
        else:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            # s_begin = index % self.tot_len
            index = self.use_index[index]
            seq_x = self.main_data[index:index+self.seq_len-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index+self.seq_len-self.pred_len:index+self.seq_len, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index+self.seq_len-self.pred_len:index+self.seq_len, feat_id:feat_id+1]
            # import pdb; pdb.set_trace()
        
        return seq_x, seq_y, observed_mask, observed_mask
    
    def __len__(self):
        if self.set_type == 2:
            return len(self.use_index)
        return len(self.use_index)*self.enc_in     




def normalize_and_fill(df):
    """Normalizes the dataframe, fills NaN values, and creates a mask matrix."""

    # Calculate mean and std for non-NaN values
    mean = df.mean(axis=0, skipna=True)
    std = df.std(axis=0, skipna=True)

    # Normalize the data
    normalized_data = (df - mean) / std

    # Forward fill NaN values
    normalized_data.fillna(method='ffill', inplace=True)

    # Fill remaining NaN values with 0
    normalized_data.fillna(0, inplace=True)

    # Create a mask matrix where NaN values are 0 and other values are 1
    mask_matrix = (~df.isna()).astype(int)

    # Adjust the length of the data
    if len(df) < 30:
        padding_length = 30 - len(df)

        # Pad with zeros at the beginning
        padding = pd.DataFrame(0, index=range(padding_length), columns=df.columns)
        normalized_data = pd.concat([padding, normalized_data], ignore_index=True)
        mask_matrix = pd.concat([padding, mask_matrix], ignore_index=True)
    # else:
    #     # Use only the final 30 entries
    #     normalized_data = normalized_data.iloc[-30:].reset_index(drop=True)
    #     mask_matrix = mask_matrix.iloc[-30:].reset_index(drop=True)

    return normalized_data, mask_matrix

import pickle as pkl
class Dataset_mimic(Dataset):
    def __init__(self, root_path='/u/dcao1/workspace/CSDI_miss_value/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 30 #168 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 3 #24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # self.test_length= self.pred_len*5
        # self.valid_length = self.pred_len*2
            
        self.seq_length = self.seq_len  #+ self.pred_length

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.txt_path = txt_path
        self.root_path = root_path #_all
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

        # self.enc_in = self.data_x.shape[-1]
        # print("self.enc_in = {}".format(self.enc_in))
        # print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path #+ 
        datafolder = self.root_path + 'mimic/'
        if 'vital' in self.data_path:
            try:
                with open(datafolder +'mimic_vital_data.pkl', 'rb') as f:
                    self.main_data = pkl.load(f)
                with open(datafolder + 'mimic_vital_mask.pkl', 'rb') as f:
                    self.mask_data = pkl.load(f)
            except:
                paths=datafolder+'vital.csv' 
                csv_data = pd.read_csv(paths)
                icu_counts = csv_data.groupby('icustay_id').size().to_dict()
                # Filter the dictionary to keep only values greater than 10 and less than 40
                filtered_icu_counts = {key: value for key, value in icu_counts.items() if 10 < value < 40}
                for key in filtered_icu_counts.keys():
                    vital_use_columns = csv_data.drop(['charttime', 'icustay_id'], axis = 1)
                    df = vital_use_columns[csv_data['icustay_id'] == key]
                    normalized_data, mask_matrix = normalize_and_fill(df[-30:])
                    self.main_data.append(normalized_data.values)
                    self.mask_data.append(mask_matrix.values)
                
                
                with open(datafolder +'mimic_vital_data.pkl', 'wb') as f:
                    pkl.dump(self.main_data, f)
                with open(datafolder + 'mimic_vital_mask.pkl', 'wb') as f:
                    pkl.dump(self.mask_data, f)
                # import pdb; pdb.set_trace()

        else:
            try:
                with open(datafolder +'mimic_lab_data.pkl', 'rb') as f:
                    self.main_data = pkl.load(f)
                with open(datafolder + 'mimic_lab_mask.pkl', 'rb') as f:
                    self.mask_data = pkl.load(f)
            except:
                paths=datafolder+'lab.csv' 
                csv_data = pd.read_csv(paths)
                icu_counts = csv_data.groupby('icustay_id').size().to_dict()
                # Filter the dictionary to keep only values greater than 10 and less than 40
                filtered_icu_counts = {key: value for key, value in icu_counts.items() if 10 < value < 40}
                for key in filtered_icu_counts.keys():
                    lab_use_columns = csv_data.drop(['charttime', 'icustay_id', 'hadm_id', 'subject_id'], axis = 1)
                    df = lab_use_columns[csv_data['icustay_id'] == key]
                    normalized_data, mask_matrix = normalize_and_fill(df[-30:])
                    self.main_data.append(normalized_data.values)
                    self.mask_data.append(mask_matrix.values)
                
                
                with open(datafolder +'mimic_lab_data.pkl', 'wb') as f:
                    pkl.dump(self.main_data, f)
                with open(datafolder + 'mimic_lab_mask.pkl', 'wb') as f:
                    pkl.dump(self.mask_data, f)
        
        # import pdb; pdb.set_trace()
        self.test_data = self.main_data[-5:].copy()
        self.mask_test = self.mask_data[-5:].copy()
        self.val_data = self.main_data[-int((len(self.main_data) - 5)*0.1):-5].copy()
        self.mask_val = self.mask_data[-int((len(self.main_data) - 5)*0.1):-5].copy()
        self.mask_data = self.mask_data[:-int((len(self.main_data) - 5)*0.1)].copy()
        self.main_data = self.main_data[:-int((len(self.main_data) - 5)*0.1)].copy()
        self.mmean_data = 0
        self.std_data = 1
        
        if self.set_type == 0:
            self.use_index = np.arange(len(self.main_data))
            self.data_x = self.main_data#[self.use_index]
        elif self.set_type == 1:
            self.use_index = np.arange(len(self.val_data))
            self.data_x = self.val_data#[self.use_index]
        else:
            self.use_index = np.arange(len(self.test_data))
            self.data_x = self.test_data#[self.use_index]

        # import pdb; pdb.set_trace()
        self.enc_in = self.data_x[0].shape[-1]
        
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        if self.set_type == 2:
            index = self.use_index[orgindex]
            seq_x = self.test_data[index][:-self.pred_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.test_data[index][-self.pred_len:]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_test[index][-self.pred_len:]
            # import pdb; pdb.set_trace()
        elif self.set_type == 1:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            # s_begin = index % self.tot_len
            index = self.use_index[index]
            seq_x = self.val_data[index][:-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.val_data[index][-self.pred_len:, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_val[index][-self.pred_len:, feat_id:feat_id+1]
        else:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            # s_begin = index % self.tot_len
            index = self.use_index[index]
            seq_x = self.main_data[index][:-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index][-self.pred_len:, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index][-self.pred_len:, feat_id:feat_id+1]
            # import pdb; pdb.set_trace()
        
        return seq_x, seq_y, observed_mask, observed_mask
    
    def __len__(self):
        if self.set_type == 2:
            return len(self.use_index)
        return len(self.use_index)*self.enc_in    
        


class Dataset_physionet(Dataset):
    def __init__(self, root_path='/u/dcao1/workspace/CSDI_miss_value/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 30 #168 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 3 #24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # self.test_length= self.pred_len*5
        # self.valid_length = self.pred_len*2
            
        self.seq_length = self.seq_len  #+ self.pred_length

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.txt_path = txt_path
        self.root_path = root_path #_all
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

        # txt_info_path = os.path.join(self.root_path='./csdi_data', self.txt_path)
        # with open(txt_info_path, "r") as f:
        #     self.dataset_description = f.read().split("\n")
        #     # self.variable_description = self.dataset_description[1:]
        #     self.dataset_description = self.dataset_description[0]
            
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path #+ 
        datafolder = self.root_path + 'physionet_2012/'
        datatype = self.data_path
        self.main_data =[]
        self.mask_data = []
        try:
            with open(datafolder + datatype + '_'+str(self.pred_len) + '_7col_data.pkl', 'rb') as f:
                self.main_data = pkl.load(f)
            with open(datafolder + datatype + '_'+str(self.pred_len) + '_7col_mask.pkl', 'rb') as f:
                self.mask_data = pkl.load(f)
        except:
            paths=datafolder + datatype +'.csv'  
            csv_data = pd.read_csv(paths)
            icu_counts = csv_data.groupby('RecordID').size().to_dict()
            # Filter the dictionary to keep only values greater than 10 and less than 40
            filtered_icu_counts = {key: value for key, value in icu_counts.items() if value>10}
            for key in filtered_icu_counts.keys():
                vital_use_columns = csv_data.drop(['RecordID', 'Time'], axis = 1)
                # vital_use_columns = vital_use_columns[['DiasABP', 'HR', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Urine', 'Weight']]
                vital_use_columns = vital_use_columns[['DiasABP', 'HR', 'MAP', 'NIDiasABP',  'RespRate', 'SysABP', 'Urine']]
                
                df = vital_use_columns[csv_data['RecordID'] == key]
                normalized_data, mask_matrix = normalize_and_fill(df[-30:])
                self.main_data.append(normalized_data.values)
                self.mask_data.append(mask_matrix.values)
            
            
            with open(datafolder + datatype + '_'+str(self.pred_len)  + '_7col_data.pkl', 'wb') as f:
                pkl.dump(self.main_data, f)
            with open(datafolder + datatype + '_'+str(self.pred_len)  + '_7col_mask.pkl', 'wb') as f:
                pkl.dump(self.mask_data, f)
                # import pdb; pdb.set_trace()

        
        
        # import pdb; pdb.set_trace()
        self.test_data = self.main_data[-5:].copy()
        self.mask_test = self.mask_data[-5:].copy()
        self.val_data = self.main_data[-int((len(self.main_data) - 5)*0.1):-5].copy()
        self.mask_val = self.mask_data[-int((len(self.main_data) - 5)*0.1):-5].copy()
        self.mask_data = self.mask_data[:-int((len(self.main_data) - 5)*0.1)].copy()
        self.main_data = self.main_data[:-int((len(self.main_data) - 5)*0.1)].copy()
        self.mean_data = 0
        self.std_data = 1
        
        if self.set_type == 0:
            self.use_index = np.arange(len(self.main_data))
        elif self.set_type == 1:
            self.use_index = np.arange(len(self.val_data))
        else:
            self.use_index = np.arange(len(self.test_data))

        self.data_x = self.main_data[0]
     # import pdb; pdb.set_trace()
        self.enc_in = self.data_x[0].shape[-1]
        
        

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        if self.set_type == 2:
            index = self.use_index[orgindex]
            seq_x = self.test_data[index][:-self.pred_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.test_data[index][-self.pred_len:]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_test[index][-self.pred_len:]
            # import pdb; pdb.set_trace()
        elif self.set_type == 1:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            # s_begin = index % self.tot_len
            index = self.use_index[index]
            seq_x = self.val_data[index][:-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.val_data[index][-self.pred_len:, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_val[index][-self.pred_len:, feat_id:feat_id+1]
        else:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            # s_begin = index % self.tot_len
            index = self.use_index[index]
            seq_x = self.main_data[index][:-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index][-self.pred_len:, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index][-self.pred_len:, feat_id:feat_id+1]
            # import pdb; pdb.set_trace()
        
        return seq_x, seq_y, observed_mask, observed_mask
    
    def __len__(self):
        if self.set_type == 2:
            return len(self.use_index)
        return len(self.use_index)*self.enc_in    

def process_data(data):
    n, m = data.shape
    processed_data = data.copy()
    mask = np.ones_like(data)

    for col in range(0, m, 2):  # 每隔一列处理
        # 处理数据
        for i in range(0, n, 7):
            end = min(i + 7, n)
            sum_val = np.sum(data[i:end, col])
            processed_data[i:end, col] = 0
            if i + 6 < n:
                processed_data[i + 6, col] = sum_val

        # 创建掩码
        mask[:, col] = 0
        mask[6::7, col] = 1

    return processed_data, mask

def calculate_stats(data, mask):
    means = []
    stds = []

    for col in range(data.shape[1]):
        if np.all(mask[:, col] == 1):  # 未处理的列
            col_data = data[:, col]
        else:  # 处理过的列
            col_data = data[mask[:, col] == 1, col]
        
        means.append(np.mean(col_data))
        stds.append(np.std(col_data))

    return np.array(means), np.array(stds)
      

class Dataset_M5(Dataset):
    def __init__(self, root_path='/u/dcao1/workspace/CSDI_miss_value/data/', split='train', size=None,
                 features='M', data_path='electricity_nips/', 
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False,use_time_features=False,
                 text_condition = False, txt_path = 'electricity_nips/'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 170 #168 #90 # 168 +24
            self.label_len = 0
            self.pred_len = 28 #24 
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # self.test_length= self.pred_len*5
        # self.valid_length = self.pred_len*2
            
        self.seq_length = self.seq_len  + self.pred_len # 跟TimeDIT写的不一样，这里是seq_length，是History + Future， Timedit里加了furture

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.txt_path = txt_path
        self.root_path = root_path #_all
        self.data_path = data_path
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]
        self.text_condition = text_condition
        
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len  + 1


    def __read_data__(self):
   
        self.dataset_description = 'Exchange consists of daily exchange rates of eight countries including Australia, \
            British, Canada, Switzerland, China, Japan, New Zealand, and Singapore from 1990 to 2016.8'

        # import pdb; pdb.set_trace()    
        paths= self.root_path #+ 
        datafolder = self.root_path + 'M5/sales_train_evaluation'
        datatype = self.data_path.split('_')[0]
        area = self.data_path.split('_')[1]
        self.main_data =[]
        self.mask_data = []
        try:
            with open(datafolder + datatype + '_'+str(self.pred_len) + '_7col_data.pkl', 'rb') as f:
                self.main_data = pkl.load(f)
            with open(datafolder + datatype + '_'+str(self.pred_len) + '_7col_mask.pkl', 'rb') as f:
                self.mask_data = pkl.load(f)
        except:
            paths=datafolder + '.csv'  
            csv_data = pd.read_csv(paths, index_col=0)
            dept_ids = csv_data['dept_id'].unique()
            store_ids = csv_data['store_id'].unique()

            # foods_depts = [dept for dept in dept_ids if 'FOODS' in dept] #HOBBIES
            foods_depts = [dept for dept in dept_ids if datatype in dept] #HOBBIES

            foods_stores = [store for store in store_ids if area in store]
            self.test_data = []
            self.mask_test = []
            self.test_mean = []
            self.test_std = []
            self.val_data = []
            self.mask_val = []
            self.main_data = []
            self.mask_data = []
            self.trend_train = []
            self.trend_val = []
            self.season_train = []
            self.season_val = []
            self.resid_train = []
            self.resid_val = []

            for foods_store in foods_stores:
                print(foods_store)
                df = csv_data[(csv_data['store_id'] == foods_store) & (csv_data['dept_id'].isin(foods_depts))]
                d_columns = [f'd_{i}' for i in range(1, 1942)]
                # filtered_data_d = df[d_columns].T.values
                # filtered_data_d = df[d_columns].T.rolling(window=4, min_periods=1).mean().to_numpy()
                filtered_data_d = df[d_columns].T.rolling(window=7, min_periods=1).sum().to_numpy()
                print("filtered_data_d mean", filtered_data_d.mean())
                # filtered_data_d, mask = process_data(filtered_data_d)
                # mean, std = calculate_stats(filtered_data_d[:-self.pred_len], mask[:-self.pred_len])
                mask = np.ones_like(filtered_data_d)
                mean = np.mean(filtered_data_d[:-self.pred_len], axis=0)
                std = np.std(filtered_data_d[:-self.pred_len], axis=0)
                normalized_data = (filtered_data_d - mean) / std
                # import pdb; pdb.set_trace()
                # Initialize arrays to store decomposition results
                trend_all = np.zeros_like(normalized_data)
                seasonal_all = np.zeros_like(normalized_data)
                residual_all = np.zeros_like(normalized_data)
                if self.set_type == 0 or self.set_type ==1:
                    # Perform STL decomposition for each sample
                    for i in range(normalized_data.shape[0]):
                        # Convert each time series to pandas Series
                        ts = pd.Series(normalized_data[i, :])
                        
                        # Perform STL decomposition
                        # Adjust period based on your data's seasonal pattern
                        stl = STL(ts, period=24)  # For example, if daily data with yearly seasonality, period=365
                        result = stl.fit()
                        
                        # Store results
                        trend_all[i, :] = result.trend
                        seasonal_all[i, :] = result.seasonal
                        residual_all[i, :] = result.resid

                # Components are now available as numpy arrays
                trend_np = trend_all #.to_numpy()
                seasonal_np = seasonal_all #.to_numpy()
                residual_np = residual_all #.to_numpy()

                start = ((len(normalized_data) - self.seq_length -self.seq_length) -(self.seq_length-self.pred_len))//self.pred_len
                end = len(normalized_data) - self.seq_length -self.seq_length + 1
                self.test_data.append(normalized_data[-self.seq_length:].copy())
                self.mask_test.append(mask[-self.seq_length:])
                # self.mask_test.append(np.ones_like(normalized_data[-self.seq_length:]).copy())
                self.test_mean.append(mean)
                self.test_std.append(std)

                self.use_index = np.arange(start,end,self.pred_len)
                # self.use_index = np.arange(start,end,1)

                for index in self.use_index[:-3]:
                    self.mask_data.append(mask[index:index+self.seq_length])
                    # self.mask_data.append(np.ones_like(normalized_data[index:index+self.seq_length]))
                    self.main_data.append(normalized_data[index:index+self.seq_length])
                    self.trend_train.append(trend_np[index:index+self.seq_length])
                    self.season_train.append(seasonal_np[index:index+self.seq_length])
                    self.resid_train.append(residual_np[index:index+self.seq_length])
                    
                    
                for index in self.use_index[-3:]:
                    self.mask_val.append(mask[index:index+self.seq_length])
                    # self.mask_val.append(np.ones_like(normalized_data[index:index+self.seq_length]))
                    self.val_data.append(normalized_data[index:index+self.seq_length])
                    self.trend_val.append(trend_np[index:index+self.seq_length])
                    self.season_val.append(seasonal_np[index:index+self.seq_length])
                    self.resid_val.append(residual_np[index:index+self.seq_length])

            
            '''
            with open(datafolder + datatype + '_'+str(self.pred_len)  + '_7col_data.pkl', 'wb') as f:
                pkl.dump(self.main_data, f)
            with open(datafolder + datatype + '_'+str(self.pred_len)  + '_7col_mask.pkl', 'wb') as f:
                pkl.dump(self.mask_data, f)
                # import pdb; pdb.set_trace()
            '''

        
        
       
        self.mean_data = 0
        self.std_data = 1
        
        if self.set_type == 0:
            self.use_index = np.arange(len(self.main_data))
        elif self.set_type == 1:
            self.use_index = np.arange(len(self.val_data))
        else:
            self.use_index = np.arange(len(self.test_data))

        self.data_x = self.main_data[0]
            

    def __getitem__(self, orgindex):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        # index = self.use_index[orgindex]
        if self.set_type == 0:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            seq_x = self.main_data[index][:-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.main_data[index][-self.pred_len:, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_data[index][-self.pred_len:, feat_id:feat_id+1]
            train_trend = self.trend_train[index][:-self.pred_len, feat_id:feat_id+1]
            train_season = self.season_train[index][:-self.pred_len, feat_id:feat_id+1]
            train_resid = self.resid_train[index][:-self.pred_len, feat_id:feat_id+1]
            trend = torch.tensor(train_trend, dtype=torch.float32)
            season = torch.tensor(train_season, dtype=torch.float32)
            resid = torch.tensor(train_resid, dtype=torch.float32)
        # seq_x = self.main_data[index:index+self.seq_len]
        elif self.set_type == 1:
            index = orgindex//self.enc_in
            feat_id = orgindex%self.enc_in
            seq_x = self.val_data[index][:-self.pred_len, feat_id:feat_id+1]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            seq_y = self.val_data[index][-self.pred_len:, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y, dtype=torch.float32)
            observed_mask = self.mask_val[index][-self.pred_len:, feat_id:feat_id+1]

            val_trend = self.trend_val[index][:-self.pred_len, feat_id:feat_id+1]
            val_season = self.season_val[index][:-self.pred_len, feat_id:feat_id+1]
            val_resid = self.resid_val[index][:-self.pred_len, feat_id:feat_id+1]
            trend = torch.tensor(val_trend, dtype=torch.float32)
            season = torch.tensor(val_season, dtype=torch.float32)
            resid = torch.tensor(val_resid, dtype=torch.float32)
        else:
            index = self.use_index[orgindex]
            seq_x = self.test_data[index][:-self.pred_len]
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
            observed_mask = self.mask_test[index][-self.pred_len:]
            seq_y = self.test_data[index][-self.pred_len:]
            means = self.test_mean[index]
            stds = self.test_std[index]
        
      
        
        if self.set_type == 0 or self.set_type == 1:
            return seq_x, seq_y, observed_mask, observed_mask, trend, season, resid
        else:
            return seq_x, seq_y, observed_mask, means, stds
        # return seq_x, text_embedding, observed_mask
    
    def __len__(self):
        if self.set_type == 2:
            return len(self.use_index)
        print(self.enc_in )
        return len(self.use_index)*self.enc_in   



   


