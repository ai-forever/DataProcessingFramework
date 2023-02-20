import abc
import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import string

from tqdm import tqdm
import torch

from DPF.dataloaders.images import UniversalT2IDataloader
from DPF.filesystems.filesystem import FileSystem
from DPF.filters import Filter


class ImageFilter(Filter):
    
    def __init__(
            self, 
            pbar: bool
        ):
        super(ImageFilter, self).__init__()
        
        self.pbar = pbar
        
        self.schema = [] # fill with your columns
        self.dataloader_kwargs = {} # Insert your params
        
    @abc.abstractmethod
    def preprocess(self, img_bytes: bytes, data: dict):
        pass
        
    @abc.abstractmethod
    def process_batch(self, batch) -> dict:
        pass
        
    @staticmethod
    def _add_values_from_batch(main_dict: dict, batch_dict: dict):
        for k, v in batch_dict.items():
            main_dict[k].extend(v)
       
    def _generate_dict_from_schema(self):
        return {i: [] for i in self.schema}
            
    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        dataloader = UniversalT2IDataloader(filesystem, df, **self.dataloader_kwargs)
        
        df_labels = self._generate_dict_from_schema()
        
        for batch in tqdm(dataloader, disable=not self.pbar):
            df_batch_labels = self.process_batch(batch)
            self._add_values_from_batch(df_labels, df_batch_labels)
        
        df_result = pd.DataFrame(df_labels)
        df = pd.merge(df, df_result, on='image_path')
        
        return df