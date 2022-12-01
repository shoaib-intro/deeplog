# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:36:40 2022

@author: ashoaib
"""
import pandas as pd
import numpy as np
import json, re, os, time
import progressbar
from .model import Model

# Import as a variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict, defaultdict

class Trainer():
    def __init__(self, options):
        self.device = options['device']
        self.input_dir = options['input_dir']
        self.output_dir = options['output_dir']
        self.log_file = options['log_file']
        
        # model
        self.window_size = options['window_size']
        self.num_candidate = self.window_size - 1
        self.num_epochs = options['num_epochs']
        self.batch_size = options['batch_size']
        self.input_size = options['input_size']
        self.num_layers = 2
        self.hidden_size = 64
        
        with open(self.output_dir + "log_keys.json", "r") as f:
            event_num = json.load(f)
        self.num_classes = len(event_num) +1
        
        self.model = Model(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)
    
    
    
    def mapping(self):
        log_templates_file = self.output_dir + self.log_file + "_templates.pkl"
        print('Mapping Log Keys from ', log_templates_file)
    
        log_temp = pd.read_pickle(log_templates_file)
        log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
        log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
        print(f'total {len(log_temp_dict)} logkeys found.')
        
        #[print(key, ' : ', value) for key, val in log_temp_dict.items()]
        with open (self.output_dir + "log_keys.json", "w") as f:
            json.dump(log_temp_dict, f)
        return len(log_temp_dict) + 1
    
    def data_sampling(self):
        log_structured_file = self.output_dir + self.log_file + "_structured.pkl"
        print("Loading", log_structured_file)
    
        #df = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})
        df = pd.read_pickle(log_structured_file)
    
        with open(self.output_dir + "log_keys.json", "r") as f:
            event_num = json.load(f)
            df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))
            #keys = df[(df.EventId < 0).all(1)]['EventTemplate']
            #if keys: print(f'found new logkeys: \n {keys}')
            
            #grouping by unique machine/pod/process Id using group function to speedup instead dictionary
            df_eventId = df.groupby(['Source'])['EventId'].apply(pd.Series.tolist).reset_index(name='sequences')
            df_index = df.groupby(['Source'])['LineId'].apply(pd.Series.tolist).reset_index(name='index')
            data_df = pd.merge(df_eventId, df_index, on='Source')
            
            print('grouping done. cleaning now! ')
            data_df['EventSequence'] = data_df['sequences'].apply(lambda x: re.sub(r'[\[\]\'\,]', '', str(x)))
            np.savetxt(self.output_dir + self.log_file+ "_sequence.txt", data_df['EventSequence'].values, fmt='%s')
            
            data_df.to_pickle(self.output_dir + self.log_file+ "_sample_data.pkl")
            print(f'{self.log_file} sampling done.')
    
    def generate_train(self):
        name = self.output_dir + self.log_file + '_sequence.txt'
        num_sessions = 0
        inputs = []
        outputs = []
        with open( name, 'r') as f:
            for line in f.readlines():
                num_sessions += 1 
                line = tuple(map(int, line.strip().split()))
                for i in range(len(line) - self.window_size):
                    inputs.append(line[i:i + self.window_size])
                    outputs.append(line[i + self.window_size])
        print('Number of sessions({}): {}'.format(name, num_sessions))
        print('Number of seqs({}): {}'.format(name, len(inputs)))
        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
        return dataset

    
    def train(self):
        start_time = time.time()
        self.mapping()
        self.data_sampling() #sampling
        
        if not os.path.isdir(self.output_dir + 'model'): os.makedirs(self.output_dir + 'model')
        
        #log = '{}_size={}_epoch={}_window={}_layers={}_classes={}.pt'.format(str(self.log_file),str(self.batch_size), str(self.num_epochs), str(self.window_size), str(self.num_layers), str(self.num_classes))
        #model_path = f'model/{log}'
        model_path = 'model/{}_size={}_epoch={}_window={}_layers={}_classes={}.pt'.format(str(self.log_file),str(self.batch_size), str(self.num_epochs), str(self.window_size), str(self.num_layers), str(self.num_classes))
        
        print('generating sequence batch.....')
        seq_dataset = self.generate_train()
        dataloader = DataLoader(seq_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True) # shuffles sequences along labels
    
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
    
        print('Training started.....')
        total_step = len(dataloader)
        
        for epoch in range(self.num_epochs):  # Loop over the dataset multiple times
            train_loss = 0
            bar = progressbar.ProgressBar(maxval=total_step).start()
        
            for step, (seq, label) in enumerate(dataloader):
                bar.update(step)
                # Forward pass
                seq = seq.clone().detach().view(-1, self.window_size, self.input_size)
            
                output = self.model(seq)
                loss = criterion(output, label)
        
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, self.num_epochs, train_loss / total_step))
            if epoch%10 == 0:
                tmp_model_path = 'model/{}_size={}_epoch={}_window={}_layers={}_classes={}.pt'.format(str(self.log_file),str(self.batch_size), str(epoch), str(self.window_size), str(self.num_layers), str(self.num_classes))
                torch.save(self.model.state_dict(), self.output_dir + tmp_model_path)
    
        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))
        torch.save(self.model.state_dict(), self.output_dir + model_path)
        print('Finished Training')