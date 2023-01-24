import os
import pandas as pd
from PIL import Image
import numpy as np
import random
import string
from pandarallel import pandarallel

from tqdm import tqdm
import torch

from DPF.dataloaders.images import UniversalT2IDataloader
from DPF.filesystems.filesystem import FileSystem

class TextFilter:
    
    def __init__(self, caption_name:str='caption'):
        self.caption_name = caption_name
        
          
    def filter_text(self, row):
        pass
        
    
    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        pandarallel.initialize(nb_workers=16)
        # print(np.array(list(df.parallel_apply(self.filter_text, axis=1))))
        df[self.result_columns] = np.array(list(df.parallel_apply(self.filter_text, axis=1)))
        return df
        
    def __call__(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        return self.run(df, filesystem)