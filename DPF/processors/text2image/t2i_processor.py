import pandas as pd
import numpy as np
import os
import glob
import csv
import tarfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import functools

from DPF.validators.t2i_validator import validate_caption
from DPF.processors.utils.shards_generator import ShardsGenerator
from DPF.utils.utils import save_dataframe, read_dataframe


class T2IProcessor:
    def __init__(self, df, dataset_path,
                 datafiles_ext, imagename_column,
                 caption_column, image_ext):
        self.df = df
        self.init_shape = df.shape
        self.dataset_path = dataset_path
        
        self.datafiles_ext = datafiles_ext
        self.imagename_column = imagename_column
        self.caption_column = caption_column
        self.image_ext = image_ext

    def validate(self, check_folders=True, mark_duplicated_image_names=False, 
                 mark_bad_caption=True) -> None:
        raise NotImplementedError()
        
    def update_data(self, columns_to_add, processes=1, force=False, overwrite_columns=True):
        assert force or len(self.df) == self.init_shape[0], \
            f"Dataframe length changed after initialisation. Was {self.init_shape[0]}, now {len(self.df)}. Set force=True to ignore this."
        assert set(columns_to_add).issubset(set(self.df.columns))
        
        raise NotImplementedError()
    
    def rebuild(self, force=False):
        assert not force or len(self.df) == self.init_shape[0], \
            f"Dataframe length didn`t changed after initialisation. Set force=True to ignore this and force rebuild dataset."
        raise NotImplementedError()
        
    def apply_filter(self, filter_func):
        self.df = filter_func(self.df)