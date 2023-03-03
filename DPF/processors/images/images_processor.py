from abc import abstractmethod
from typing import List, Dict, Optional, Callable, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
import csv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from DPF.filesystems import LocalFileSystem, FileSystem
from DPF.utils import get_file_extension


def preprocessing_for_convert(img_bytes, data):
    return img_bytes, data

    
class ImagesProcessor:
    """
    Class that describes all interactions with image datasets.
    It is recommended to use ImagesFormatter to create Processor instead of directly initialiasing a Processor class.
    """
    
    def __init__(
            self, 
            filesystem: FileSystem, 
            df: pd.DataFrame, 
        ):
        self.filesystem = filesystem
        
        self.df = df
        self.init_shape = df.shape
        
    def get_filesystem(self):
        """
        Get a FileSystem object
        
        Returns
        -------
        DPF.filesystems.FileSystem
            FileSystem of that dataset
        """
        return self.filesystem
        
    def get_random_samples(
            self, 
            df: Optional[pd.DataFrame] = None, 
            n: int = 1
        ) -> list:
        """
        Get N random samples from dataset
        
        Parameters
        ----------
        df: pd.DataFrame | None
            DataFrame to sample from. If none, processor.df is used
        n: int = 1
            Number of samples to return
            
        Returns
        -------
        list
            List of tuples with PIL images and dataframe data
        """

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
        
    def apply_filter(
            self, 
            filter_func: Callable[[pd.DataFrame, FileSystem], pd.DataFrame]
        ):
        """
        Applies a function to processor.df and stores result to processor.df:
        self.df = filter_func(self.df, self.filesystem)
        
        Parameters
        ----------
        filter_func: Callable[[pd.DataFrame, FileSystem], pd.DataFrame]
            Function to apply
        """
        self.df = filter_func(self.df, self.filesystem)