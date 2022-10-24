import pandas as pd
import numpy as np
import os
import glob
import csv
import tarfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import functools

from DPF.processors.text2image.t2i_processor import T2IProcessor
from DPF.validators.t2i_validator import validate_caption
from DPF.processors.utils.shards_generator import ShardsGenerator
from DPF.utils.utils import save_dataframe, read_dataframe


class RawProcessor(T2IProcessor):
    def __init__(self, df, dataset_path,
                 datafiles_ext, imagename_column,
                 caption_column, image_ext):
        self.df = df
        self.init_shape = df.shape
        self.dataset_path = dataset_path.rstrip('/')
        
        self.datafiles_ext = datafiles_ext.lstrip('.')
        self.imagename_column = imagename_column
        self.caption_column = caption_column
        self.image_ext = image_ext

    def validate(self, exactly_match=True, mark_duplicated_image_names=False, 
                 mark_bad_caption=True) -> None:
        assert (self.df['data_format']=='raw').all(), "All data should be in raw format"
        
        # check folders with images
        table_paths = self.df['table_path'].unique().tolist()
        folders_not_existing = []
        folder_paths = []
        for datafile_path in table_paths:
            folder_path = datafile_path[:-len(self.datafiles_ext)-1]+'/'
            folder_paths.append(folder_path)
            if not os.path.exists(folder_path):
                folders_not_existing.append(folder_path)
        assert len(folders_not_existing) == 0, f"Folders {folders_not_existing} are not existing! Check dataset format"
        
        if exactly_match:
            folders = glob.glob(f'{self.dataset_path}/*/')
            assert set(folders) == set(folder_paths), f"Folders {set(folders).difference(set(folder_paths))} don't have data file."
        
        self.df['validate_status'] = True
        # check for duplicate path
        self.df['duplicate_image_path'] = False
        condition = self.df['image_path'].duplicated()
        self.df.loc[condition, 'validate_status'] = False
        self.df.loc[condition, 'duplicate_image_path'] = True
        # check for duplicate image
        if mark_duplicated_image_names:
            self.df['duplicate_image_name'] = False
            condition = self.df['image_path'].duplicated()
            self.df.loc[condition, 'validate_status'] = False
            self.df.loc[condition, 'duplicate_image_name'] = True
        # check captions
        if mark_bad_caption:
            self.df = validate_caption(self.df)
        # check for existing
        self.df['file_not_exists'] = False
        condition = self.df['image_path'].apply(lambda x: not os.path.exists(x))
        self.df.loc[condition, 'validate_status'] = False
        self.df.loc[condition, 'file_not_exists'] = True
        
    def _merge_and_write_table(self, table_path, df_to_add, overwrite_columns=True):
        if self.image_ext:
            image_ext = self.image_ext.lstrip('.')
            df_to_add['image_name'] = df_to_add['image_name'].str.slice(0, -len(image_ext)-1)
        df_to_add.rename(columns={'image_name': self.imagename_column}, inplace=True)
        
        df = read_dataframe(table_path, self.datafiles_ext)
        columns = [i for i in df.columns if i != self.imagename_column]
        columns_to_add = [i for i in df_to_add.columns if i != self.imagename_column]
        columns_intersection = set(columns).intersection(set(columns_to_add))
        if overwrite_columns:
            df.drop(columns=list(columns_intersection), inplace=True)
        else:
            df_to_add.drop(columns=list(columns_intersection), inplace=True)

        if df_to_add.shape[1] > 1:
            df = pd.merge(df, df_to_add, on=self.imagename_column)
            save_dataframe(df, table_path, self.datafiles_ext, index=False)
            
    def _merge_and_write_table_mp(self, data):
        return self._merge_and_write_table(*data)
        
    def update_data(self, columns_to_add, processes=1, force=False, overwrite_columns=True):
        assert force or len(self.df) == self.init_shape[0], \
            f"Dataframe length changed after initialisation. Was {self.init_shape[0]}, now {len(self.df)}. Set force=True to ignore this."
        assert set(columns_to_add).issubset(set(self.df.columns))
        
        table_to_new_data = self.df.groupby('table_path').apply(
            lambda x: tuple([v for v in x[['image_name']+columns_to_add].to_dict('records')])
        )
        
        def gen():
            for table_path in table_to_new_data.keys():
                yield (table_path, pd.DataFrame(table_to_new_data[table_path]), overwrite_columns)
                
        params_iter = gen()
        process_map(self._merge_and_write_table_mp, iter(params_iter), 
                    max_workers=processes, chunksize=1, total=len(table_to_new_data))
    
    def rebuild(self, force=False):
        assert not force or len(self.df) == self.init_shape[0], \
            f"Dataframe length didn`t changed after initialisation. Set force=True to ignore this and force rebuild dataset."
        raise NotImplementedError()
    
    def to_raw(self, save_path):
        assert os.path.abspath(save_path) != os.path.abspath(self.dataset_path)
        raise NotImplementedError()
        
    def to_shards(self, save_path, processes=8, images_per_tar=1000, force=False, rename_images=False,
                  imagename_column="image_name", caption_column="caption", columns_to_add=[]):
        
        shards_gen = ShardsGenerator(
            self.df, save_path, processes=processes, images_per_tar=images_per_tar, force=force, 
            rename_images=rename_images, save_csv=True, imagename_column=imagename_column, 
            columns_to_add=[caption_column]+columns_to_add
        )
        shards_gen.run()