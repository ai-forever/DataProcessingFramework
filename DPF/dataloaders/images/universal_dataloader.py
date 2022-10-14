import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import tarfile
import torch
import itertools
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import BatchSampler, DataLoader

from .utils import default_preprocess
from .raw_dataset import RawDataset
from .shards_dataset import ShardsDataset

FORMAT_TO_DATASET = {
    'raw': RawDataset,
    'shards': ShardsDataset
}


class UniversalT2IDataloader:
    def __init__(self, 
                 df, 
                 cols_to_return=[], 
                 preprocess_f=default_preprocess,
                 **dataloader_kwargs):
        
        self.df = df
        self.df_formats = df['data_format'].unique().tolist()
        assert all([f in FORMAT_TO_DATASET for f in self.df_formats]), "Unknown data format in dataloader"
        self.cols_to_return = cols_to_return
        self.preprocess_f = preprocess_f
        self.dataloader_kwargs = dataloader_kwargs
        self.len = None
        
    def test(self):
        for data_format in self.df_formats:
            DatasetClass = FORMAT_TO_DATASET[data_format]
            dataset = DatasetClass(self.df[self.df['data_format'] == data_format], self.cols_to_return, self.preprocess_f)
            print(f'"{data_format}" dataset created')
            dataloader = DataLoader(dataset, **self.dataloader_kwargs)
            print(f'"{data_format}" dataloader created')
            for i in dataloader:
                break
            print(f'"{data_format}" iteration tested. Format "{data_format}" is ok!')
        
    def __len__(self):
        if 'batch_size' in self.dataloader_kwargs:
            bs = self.dataloader_kwargs['batch_size']
        else:
            bs = 1
            
        format_counts = self.df['data_format'].value_counts().to_dict()
        batched_len = sum([
            count//bs if count % bs == 0 else (count//bs)+1 for count in format_counts.values()
        ])
        return batched_len
    
    def __iter__(self):
        for data_format in self.df_formats:
            DatasetClass = FORMAT_TO_DATASET[data_format]
            dataset = DatasetClass(self.df[self.df['data_format'] == data_format], self.cols_to_return, self.preprocess_f)
            dataloader = DataLoader(dataset, **self.dataloader_kwargs)
            for item in dataloader:
                yield item