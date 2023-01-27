from typing import List
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


class ProcessorHelper:
    def __init__(self, filesystem, imagename_column, image_ext):
        self.filesystem = filesystem
        
        self.imagename_column = imagename_column
        self.image_ext = image_ext
        
    def _replace_and_write_table(self, table_path, col2newcol) -> List[str]:
        df = self.filesystem.read_dataframe(table_path)
        df.rename(columns=col2newcol, inplace=True)
        
        errors = []
        errname = None
        try:
            self.filesystem.save_dataframe(df, table_path, index=False)
        except Exception as err:
            errname = f"error during saving: {err}"
        if errname:
            errors.append(errname)
        
        return errors
    
    def _replace_and_write_table_mp(self, data):
        return self._replace_and_write_table(*data)
        
    def _delete_and_write_table(self, table_path, columns_to_delete) -> List[str]:
        df = self.filesystem.read_dataframe(table_path)
        df.drop(columns=columns_to_delete, inplace=True)
        
        errors = []
        errname = None
        try:
            self.filesystem.save_dataframe(df, table_path, index=False)
        except Exception as err:
            errname = f"error during saving: {err}"
        if errname:
            errors.append(errname)
        
        return errors
    
    def _delete_and_write_table_mp(self, data):
        return self._delete_and_write_table(*data)
    
    def _merge_and_write_table(self, table_path, df_to_add, overwrite_columns=True) -> List[str]:
        if self.image_ext:
            image_ext = self.image_ext.lstrip('.')
            df_to_add['image_name'] = df_to_add['image_name'].str.slice(0, -len(image_ext)-1)
        df_to_add.rename(columns={'image_name': self.imagename_column}, inplace=True)
        
        df = self.filesystem.read_dataframe(table_path)
        columns = [i for i in df.columns if i != self.imagename_column]
        columns_to_add = [i for i in df_to_add.columns if i != self.imagename_column]
        columns_intersection = set(columns).intersection(set(columns_to_add))
        if overwrite_columns:
            df.drop(columns=list(columns_intersection), inplace=True)
        else:
            df_to_add.drop(columns=list(columns_intersection), inplace=True)
        
        errors = []
        if df_to_add.shape[1] > 1:
            image_names_orig = set(df[self.imagename_column])
            shape_orig = len(df)
            df = pd.merge(df, df_to_add, on=self.imagename_column)
            
            errname = None
            if len(df) != shape_orig:
                errname = f'Shape of dataframe {table_path} changed after merging. Skipping this dataframe. Check for errors'
                print('[WARNING]', errname)
            elif set(df[self.imagename_column]) != image_names_orig:
                errname = f'Image names from dataframe {table_path} changed after merging. Skipping this dataframe. Check for errors'
                print('[WARNING]', errname)
            else:
                try:
                    self.filesystem.save_dataframe(df, table_path, index=False)
                except Exception as err:
                    errname = f"error during saving: {err}"
            if errname:
                errors.append(errname)
                
        return errors
            
    def _merge_and_write_table_mp(self, data):
        return self._merge_and_write_table(*data)

    
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
        
    def get_filesystem(self):
        return self.filesystem

    def validate(self, check_folders=True, mark_duplicated_image_names=False, 
                 mark_bad_caption=True) -> None:
        raise NotImplementedError()
        
    def rename_columns(self, col2newcol, processes=1, force=False):
        assert force or len(self.df) == self.init_shape[0], \
            f"Dataframe length changed after initialisation. Was {self.init_shape[0]}, now {len(self.df)}. Set force=True to ignore this."
        assert set(col2newcol.keys()).difference(self.df.columns) == set(), \
            f"Some provided columns not presented in dataset: {set(col2newcol.keys()).difference(self.df.columns)}"
        
        table_paths = self.df['table_path'].unique()
        
        def gen():
            for table_path in table_paths:
                yield (table_path, col2newcol)
                
        params_iter = gen()
        helper = ProcessorHelper(
            filesystem=self.filesystem, 
            imagename_column=self.imagename_column, 
            image_ext=self.image_ext
        )
        
        errors = process_map(helper._replace_and_write_table_mp, iter(params_iter), 
                             max_workers=processes, chunksize=1, total=len(table_paths))
        errors_flatten = [item for sublist in errors for item in sublist]
        if len(errors_flatten) == 0:
            self.df.rename(columns=col2newcol, inplace=True)
        else:
            print("[WARNING] Errors occured, please create new processor")
        return errors_flatten
        
    def delete_columns(self, columns_to_delete, processes=1, force=False):
        assert force or len(self.df) == self.init_shape[0], \
            f"Dataframe length changed after initialisation. Was {self.init_shape[0]}, now {len(self.df)}. Set force=True to ignore this."
        assert set(columns_to_delete).difference(self.df.columns) == set(), \
            f"Some provided columns not presented in dataset: {set(columns_to_delete).difference(self.df.columns)}"
        assert self.imagename_column not in columns_to_delete, f"Can`t delete image name column: {self.imagename_column}"
        assert self.caption_column not in columns_to_delete, f"Can`t delete caption column: {self.caption_column}"
        
        table_paths = self.df['table_path'].unique()
        
        def gen():
            for table_path in table_paths:
                yield (table_path, columns_to_delete)
                
        params_iter = gen()
        helper = ProcessorHelper(
            filesystem=self.filesystem, 
            imagename_column=self.imagename_column, 
            image_ext=self.image_ext
        )
        
        errors = process_map(helper._delete_and_write_table_mp, iter(params_iter), 
                             max_workers=processes, chunksize=1, total=len(table_paths))
        errors_flatten = [item for sublist in errors for item in sublist]
        if len(errors_flatten) == 0:
            self.df.drop(columns=columns_to_delete, inplace=True)
        else:
            print("[WARNING] Errors occured, please create new processor")
        return errors_flatten
        
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
        helper = ProcessorHelper(
            filesystem=self.filesystem, 
            imagename_column=self.imagename_column, 
            image_ext=self.image_ext
        )
        
        errors = process_map(helper._merge_and_write_table_mp, iter(params_iter), 
                             max_workers=processes, chunksize=1, total=len(table_to_new_data))
        errors_flatten = [item for sublist in errors for item in sublist]
        return errors_flatten
    
    def rebuild(self, force=False):
        assert not force or len(self.df) == self.init_shape[0], \
            f"Dataframe length didn`t changed after initialisation. Set force=True to ignore this and force rebuild dataset."
        raise NotImplementedError()
        
    def get_random_samples(self, df=None, n=1):
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