import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import string

from tqdm import tqdm
import torch

from DPF.dataloaders.images import UniversalT2IDataloader


class ImageFilter:
    
    def __init__(self, task_name: str, save_parquets: str, save_parquets_dir: str, pbar: bool):
        self.save_parquets_dir = save_parquets_dir
        self.save_parquets = save_parquets
        self.task_name = task_name if task_name else ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.pbar = pbar
        
        if self.save_parquets:
            assert save_parquets_dir is not None and type(save_parquets_dir) == str, \
                f"save_parquets_dir parameters should be correct path"
            self.save_parquets_dir = save_parquets_dir.rstrip('/')
            os.makedirs(self.save_parquets_dir, exist_ok=True)
        
    def preprocess(self, img_bytes, data):
        raise NotImplementedError(
                f'Implement preprocess in {self.__class__.__name__}'
        )
        
    def process_batch(self, batch) -> dict:
        raise NotImplementedError(
                f'Implement process_batch in {self.__class__.__name__}'
        )
        
    @staticmethod
    def _add_values_from_batch(main_dict, batch_dict):
        for k, v in batch_dict.items():
            main_dict[k].extend(v)
       
    def _generate_dict_from_schema(self):
        return {i: [] for i in self.schema}
            
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        dataloader = UniversalT2IDataloader(df, **self.dataloader_kwargs)
        
        df_labels = self._generate_dict_from_schema()
        
        for batch in tqdm(dataloader, disable=not self.pbar):
            df_batch_labels = self.process_batch(batch)
            self._add_values_from_batch(df_labels, df_batch_labels)
        
        df_result = pd.DataFrame(df_labels)
        df = pd.merge(df, df_result, on='image_path')
        
        if self.save_parquets:
            parquet_path = f'{self.save_parquets_dir}/{self.task_name}.parquet'
            df.to_parquet(
                parquet_path,
                index=False
            )
        
        return df
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.run(df)