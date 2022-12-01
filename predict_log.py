# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:17:19 2022

@author: ashoaib
"""


import pandas as pd
import numpy as np
import json, re, os, progressbar, glob, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from .model import Model


    
class Predictor():
    def __init__(self, options):
        self.device = options['device']
        self.input_dir = options['input_dir']
        self.output_dir = options['output_dir']
        self.log_file = options['log_file']
        
        # model
        self.window_size = options['window_size']
        self.num_candidates = self.window_size - 1
        self.num_epochs = options['num_epochs']
        self.batch_size = options['batch_size']
        self.input_size = 1
        self.num_layers = 2
        self.hidden_size = 64
        
        with open(self.output_dir + "log_keys.json", "r") as f:
            event_num = json.load(f)
        self.num_classes = len(event_num) +1
    
    def load_model(self):
        with open(self.output_dir + "log_keys.json", "r") as f:
            event_num = json.load(f)
        num_classes = len(event_num) +1

        log = '{}_size={}_epoch={}_window={}_layers={}_classes={}.pt'.format(str(self.log_file),str(self.batch_size), str(self.num_epochs), str(self.window_size), str(self.num_layers), str(self.num_classes))
        model_path = f'model/{log}'
        #model_path = 'model/' + ''.join(os.listdir(self.output_dir + 'model/'))
        
        model = Model(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)
        model.load_state_dict(torch.load(self.output_dir + model_path))
        #model.eval()
        print('model_path: {}'.format(model_path))
        return model

    def generate(self):
        logs = list() #set() to speedup used set()
        name = self.output_dir + self.log_file + '_sequence.txt'
        with open( name, 'r') as f:
            for ln in f.readlines():
                #ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
                ln = list(map(int, ln.strip().split()))
                ln = ln + [-1] * (self.window_size + 1 - len(ln))
                print(f' sequence lengths: {len(ln)}')
                logs.append(tuple(ln))
                print('Number of sessions({}): {}'.format(name, len(logs)))
        return logs

    def marking_anomalies(self):
        result = self.output_dir + self.log_file + '_anomalies.csv'
        print(f'saving results into {result}....!')
        all_files = glob.glob(os.path.join(self.output_dir + 'temp',"*.pkl"))
        df = pd.read_pickle(self.output_dir + self.log_file +  '_structured.pkl')
        df['seq_path'] = 'no'
    
        for file in all_files:
            print(f'Processing {file}')
            data = pd.read_pickle(file)
            for idx in data['index']:
                df.loc[df['LineId'] == idx, 'seq_path'] = 'yes'
        df.to_csv(result)
    
        dirpath = self.output_dir + 'temp'
        if os.path.exists(dirpath) and os.path.isdir(dirpath): shutil.rmtree(dirpath)    
    
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
    
    
    def predict(self):
        
        self.data_sampling()
        if not os.path.exists(self.output_dir + 'temp'): os.makedirs(self.output_dir + 'temp')
        
        print('Generating sequence batches......!')
        test_normal_loader = self.generate()
    
        #print('Normal loader',type(test_normal_loader))
        data = pd.read_pickle(self.output_dir + self.log_file+ "_sample_data.pkl")
        
        model = self.load_model()
 
        FP=0
        with torch.no_grad():
            for idx, line in zip(data['index'], test_normal_loader):
                logkey, logkey_trace, index, index_traces= [],[],[],[]
            
                #print(f'line with length {len(line)}---- {line} and index with length {len(idx)}---{idx}')
                print(f'line with length {len(line)} and index with length {len(idx)}')
                bar = progressbar.ProgressBar(maxval=len(line)).start()
            
                for i in range(len(line) - int(self.window_size+1)):
                    bar.update(i)

                    seq = line[i:i + self.window_size] #sequence [2,31,22]
                    label = line[i + self.window_size] #label [3]

                    seq = torch.tensor(seq, dtype=torch.float).view(-1, self.window_size, self.input_size)
                    label = torch.tensor(label).view(-1)

                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
            
                    if label not in predicted:
                        logkey.append(line[i + self.window_size]);  logkey_trace.append(line[i:i +self. window_size+3]) #logkeys
                        index.append(idx[i + self.window_size]);    index_traces.append(idx[i:i + self.window_size+3])  #indexes
                        FP += 1
            
                print(f'Total Anomalies found ---> {FP}')
                result = pd.DataFrame({'index':index, 'index_traces':index_traces, 'logkey':logkey, 'logkey_trace':logkey_trace})
                result.to_pickle(f'{self.output_dir}temp/line_{len(line)}_.pkl')
                result = ''
        
        print('Prediction done, Marking anomalies.')
        self.marking_anomalies()



