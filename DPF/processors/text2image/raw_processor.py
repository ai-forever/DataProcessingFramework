from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from PIL import Image
import os
import csv
import tarfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import functools

from DPF.filesystems import FileSystem
from DPF.processors.text2image.t2i_processor import T2IProcessor
from DPF.processors.utils.shards_generator import ShardsGenerator


class RawProcessor(T2IProcessor):
    """
    Class that describes all interactions with text2image dataset in raw format (folders with images and dataframes).
    It is recommended to use T2IFormatter to create RawProcessor instead of directly initialiasing a RawProcessor class.
    """
    
    def __init__(
            self, 
            filesystem: FileSystem, 
            df: pd.DataFrame, 
            dataset_path: str,
            datafiles_ext: str, 
            imagename_column: str,
            caption_column: str, 
            image_ext: str
        ):
        super().__init__(
            filesystem, df, dataset_path, 
            datafiles_ext, imagename_column, 
            caption_column, image_ext
        )
        
    def get_random_samples(
            self, 
            df: Optional[pd.DataFrame] = None, 
            n: int = 1
        ) -> list:
        if df is None:
            df = self.df
            
        df_samples = df.sample(n)
        
        samples = []
        for item in df_samples.to_dict('records'):
            filepath = item['image_path']
            image_bytes = self.filesystem.read_file(filepath, binary=True)
            img = Image.open(image_bytes)
            samples.append((img, item))
        return samples