import pandas as pd
import numpy as np
import os
import glob
import csv
import tarfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import functools

from DPF.filesystems.localfilesystem import LocalFileSystem
from DPF.dataloaders.images import UniversalT2IDataloader
from DPF.processors.writers.shardsfilewriter import ShardsFileWriter
from DPF.utils.utils import get_file_extension

def preprocessing_for_convert(img_bytes, data):
    return img_bytes, data

class T2IProcessor:
    def __init__(self, filesystem, df, dataset_path,
                 datafiles_ext, imagename_column,
                 caption_column, image_ext):
        self.filesystem = filesystem
        
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
        self.df = filter_func(self.df, self.filesystem)
        
    def to_shards(
            self, 
            save_dir, 
            target_filesystem=LocalFileSystem(), 
            max_files_in_shard=1000,
            columns_to_save=[],
            processes=1,
            pbar=True
        ) -> None:
        
        dataloader = UniversalT2IDataloader(
            self.filesystem, self.df, 
            cols_to_return=columns_to_save,
            preprocess_f=preprocessing_for_convert,
            num_workers=processes, batch_size=1,
            collate_fn=lambda x: x, drop_last=False
        )
        
        fw = ShardsFileWriter(
            target_filesystem, save_dir,
            max_files_in_shard=max_files_in_shard,
            image_ext=self.image_ext, 
            datafiles_ext='csv', archive_ext='tar'
        )
        
        with fw as filewriter:
            for items in tqdm(dataloader, disable=not pbar):
                img_bytes, data = items[0]

                image_ext = None
                if self.image_ext is None:
                    image_ext = get_file_extension(data['image_path'])
                data.pop('image_path')
                
                filewriter.save_file(img_bytes, image_ext=image_ext, file_data=data)