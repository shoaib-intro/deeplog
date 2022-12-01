# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:52:54 2022

@author: ashoaib
"""
from logparser import Spell, Drain
import pandas as pd
import json, re
import numpy as np

class Parser():
    def __init__(self, options):
        self.device = options['device']
        self.input_dir = options['input_dir']
        self.output_dir = options['output_dir']
        self.log_file = options['log_file']
        self.logkeys = options['logkeys']
        
        self.log_format = options['log_format']
        self.parser_algo = options['parser_algo'] 
        
    
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
    
    def parse(self):
        if self.parser_algo == 'spell':
            tau        = 0.5  # Message type threshold (default: 0.5)
            regex      = [
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r"(/[-\w]+)+",  # file path
            #"(?<=blk_)[-\d]+" #replace block_id with *
            #r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
            ] 
        
            parser = Spell.LogParser(indir=self.input_dir, outdir=self.output_dir, log_format=self.log_format, tau=tau, rex=regex, keep_para=True)
            parser.parse(self.log_file)

        elif self.parser_algo == 'drain':
            regex = [
                #r"(?<=blk_)[-\d]+", # block_id
                r'\d+\.\d+\.\d+\.\d+',  # IP
                r"(/[-\w]+)+"  # file path
                #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
                ]
            # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
            st = 0.4  # Similarity threshold
            depth = 4  # Depth of all leaf nodes


            parser = Drain.LogParser(self.log_format, indir=self.input_dir, outdir=self.output_dir, depth=depth, st=st, rex=regex, keep_para=True)
            parser.parse(self.log_file)
        
        if self.logkeys: self.mapping()