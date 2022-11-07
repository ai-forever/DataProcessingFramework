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
        
    def _merge_and_write_table(self, table_path, df_to_add, overwrite_columns=True):
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

        if df_to_add.shape[1] > 1:
            df = pd.merge(df, df_to_add, on=self.imagename_column)
            self.filesystem.save_dataframe(df, table_path, index=False)
            
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