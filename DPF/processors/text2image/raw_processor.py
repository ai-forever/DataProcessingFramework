import pandas as pd
import numpy as np
from PIL import Image
import os
import csv
import tarfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import functools

from DPF.processors.text2image.t2i_processor import T2IProcessor
from DPF.processors.utils.shards_generator import ShardsGenerator


class RawProcessor(T2IProcessor):
    def __init__(self, filesystem, df, dataset_path,
                 datafiles_ext, imagename_column,
                 caption_column, image_ext):
        self.filesystem = filesystem
        
        self.df = df
        self.init_shape = df.shape
        self.dataset_path = dataset_path.rstrip('/')
        
        self.datafiles_ext = datafiles_ext.lstrip('.')
        self.imagename_column = imagename_column
        self.caption_column = caption_column
        self.image_ext = image_ext
    
    def rebuild(self, force=False):
        assert not force or len(self.df) == self.init_shape[0], \
            f"Dataframe length didn`t changed after initialisation. Set force=True to ignore this and force rebuild dataset."
        raise NotImplementedError()
        
    def get_random_samples(self, df=None, n=1):
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
        
    def to_shards(self, save_path, processes=8, images_per_tar=1000, force=False, rename_images=False,
                  imagename_column="image_name", caption_column="caption", columns_to_add=[]):
        
        shards_gen = ShardsGenerator(
            self.df, save_path, processes=processes, images_per_tar=images_per_tar, force=force, 
            rename_images=rename_images, save_csv=True, imagename_column=imagename_column, 
            columns_to_add=[caption_column]+columns_to_add
        )
        shards_gen.run()