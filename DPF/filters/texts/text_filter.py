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
from DPF.filters import Filter


class TextFilter(Filter):
    
    def __init__(
            self, 
            text_column_name: str = 'caption',
            workers: int = 16
        ):
        super(TextFilter, self).__init__()
        
        self.text_column_name = text_column_name
        self.workers = workers
        
        self.result_columns = []
           
    def process(self, row):
        raise NotImplementedError()
        
    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        pandarallel.initialize(nb_workers=self.workers)
        res = np.array(list(df.parallel_apply(self.process, axis=1)))
        df[self.result_columns] = res
        return df
        
    def __call__(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        return self.run(df, filesystem)