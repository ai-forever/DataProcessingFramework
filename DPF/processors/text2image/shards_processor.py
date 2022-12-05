import pandas as pd
from PIL import Image
import os
import random
import tarfile
import io
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
        
    def get_random_samples(self, df=None, n=1, from_tars=1):
        if df is None:
            df = self.df
            
        archives = random.sample(df['archive_path'].unique().tolist(), from_tars)
        df_samples = df[df['archive_path'].isin(archives)].sample(n)

        archive_to_samples = df_samples.groupby('archive_path').apply(
            lambda x: x.to_dict('records')
        )
        
        samples = []
        for archive_path, data in archive_to_samples.to_dict().items():
            tar_bytes = self.filesystem.read_file(archive_path, binary=True)
            with tarfile.open('r', fileobj=tar_bytes) as tar:
                for item in data:
                    filename = item[self.imagename_column]
                    img = Image.open(io.BytesIO(tar.extractfile(filename).read()))
                    samples.append((img, item))
        return samples