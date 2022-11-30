import pandas as pd
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from DPF.processors.text2image.t2i_processor import T2IProcessor
    
    
class ShardsProcessor(T2IProcessor):
    def __init__(self, filesystem, df, dataset_path, archive_ext,
                 datafiles_ext, imagename_column,
                 caption_column, image_ext):
        self.filesystem = filesystem
        
        self.df = df
        self.init_shape = df.shape
        self.dataset_path = dataset_path.rstrip('/')
        
        self.archive_ext = archive_ext.lstrip('.')
        self.datafiles_ext = datafiles_ext.lstrip('.')
        self.imagename_column = imagename_column
        self.caption_column = caption_column
        self.image_ext = image_ext
    
    def rebuild(self, force=False):
        assert not force or len(self.df) == self.init_shape[0], \
            f"Dataframe length didn`t changed after initialisation. Set force=True to ignore this and force rebuild dataset."
        raise NotImplementedError()        