import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from pandarallel import pandarallel

from DPF.filesystems.filesystem import FileSystem


class T2IValidator:
    
    def __init__(
            self, 
            filesystem: FileSystem,
            caption_column: str
        ):
        self.filesystem = filesystem
        self.caption_column = caption_column

    def validate_caption_df(self, df: pd.DataFrame):
        errors = {}
        error2count = {}
        
        df_caption_isna = df[self.caption_column].isna()
        df_caption_small = df[self.caption_column].str.strip().str.len() <= 2
        if df_caption_isna.any():
            errors['ok'] = False
            errname = 'provided caption column has None values'
            errors[errname] = sum(df_caption_isna)
            error2count[errname] = sum(df_caption_isna)
        elif df_caption_small.any():
            errors['ok'] = False
            errname = 'provided caption column has values with length less than 2'
            errors[errname] = sum(df_caption_small)
            error2count[errname] = sum(df_caption_small)
        
        return errors, error2count