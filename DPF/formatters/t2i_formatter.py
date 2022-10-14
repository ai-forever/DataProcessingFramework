import pandas as pd
import os
import glob
from tqdm import tqdm

from DPF.utils.utils import get_file_extension


class T2IFormatter:
    
    def _read_dataframe(self, filepath: str, filetype: str, **kwargs) -> pd.DataFrame:
        if filetype == 'csv':
            return pd.read_csv(filepath, **kwargs)
        elif filetype == 'parquet':
            return pd.read_parquet(filepath, **kwargs)
        else:
            raise NotImplementedError(f"Unknown file format: {filetype}")
    
    def _postprocess_dataframe(self, df: pd.DataFrame):
        columns = ['image_name', 'image_path', 'table_path', 'archive_path', 'data_format', 'caption']
        columns = [i for i in columns if i in df.columns]
        orig_columns = [i for i in df.columns if i not in columns]
        columns.extend(list(orig_columns))
        return df[columns]
    
    def from_shards(
        self,
        dataset_path: str,
        archive_ext: str = 'tar',
        datafiles_ext: str = 'csv', 
        imagename_column: str = 'image_name',
        caption_column: str = 'caption',
        image_ext: str = None,
        use_abs_path: bool = False,
        progress_bar: bool = False
    ) -> pd.DataFrame:
        
        dataset_path = dataset_path.rstrip('/')
        if use_abs_path:
            dataset_path = os.path.abspath(dataset_path)
        datafiles_ext = datafiles_ext.lstrip('.')
        archive_ext = archive_ext.lstrip('.')
        
        datafiles = glob.glob(f'{dataset_path}/*.{datafiles_ext}')
        
        dataframes = []
        df_needed_columns = None
        for datafile in tqdm(datafiles, disable=not progress_bar):
            df = self._read_dataframe(datafile, datafiles_ext)
            
            if df_needed_columns is None:
                df_needed_columns = set(df.columns)
                
            assert set(df.columns) == df_needed_columns, \
                f'Dataframe {datafile} have different columns. Expected {df_needed_columns}, got {set(df.columns)}'
            assert imagename_column in df.columns, f'Dataframe {datafile} does not have "{imagename_column}" column'
            assert caption_column in df.columns, f'Dataframe {datafile} does not have "{caption_column}" column'

            df['table_path'] = datafile
            df['caption'] = df[caption_column]
            df['image_name'] = df[imagename_column]
            if image_ext:
                image_ext = image_ext.lstrip('.')
                df['image_name'] += '.'+image_ext

            df['archive_path'] = df['table_path'].str.rstrip(datafiles_ext)+archive_ext
            df['image_path'] = df['archive_path']+'/'+df['image_name']
            dataframes.append(df)
        
        df = pd.concat(dataframes, ignore_index=True)
        df['data_format'] = 'shards'
        df = self._postprocess_dataframe(df)
        return df
    
    def from_raw(
        self,
        dataset_path: str, 
        datafiles_ext: str = 'csv', 
        imagename_column: str = 'image_name',
        caption_column: str = 'caption',
        image_ext: str = None,
        use_abs_path: bool = False,
        progress_bar: bool = False
    ) -> pd.DataFrame:
        
        dataset_path = dataset_path.rstrip('/')
        if use_abs_path:
            dataset_path = os.path.abspath(dataset_path)
        datafiles_ext = datafiles_ext.lstrip('.')
        
        datafiles = glob.glob(f'{dataset_path}/*.{datafiles_ext}')
        
        dataframes = []
        df_needed_columns = None
        for datafile in tqdm(datafiles, disable=not progress_bar):
            df = self._read_dataframe(datafile, datafiles_ext)
            
            if df_needed_columns is None:
                df_needed_columns = set(df.columns)
                
            assert set(df.columns) == df_needed_columns, \
                f'Dataframe {datafile} have different columns. Expected {df_needed_columns}, got {set(df.columns)}'
            assert imagename_column in df.columns, f'Dataframe {datafile} does not have "{imagename_column}" column'
            assert caption_column in df.columns, f'Dataframe {datafile} does not have "{caption_column}" column'
            
            df['table_path'] = datafile
            df['caption'] = df[caption_column]
            df['image_name'] = df[imagename_column]
            if image_ext:
                image_ext = image_ext.lstrip('.')
                df['image_name'] += '.'+image_ext

            df['image_path'] = df['table_path'].str.slice(0,-(len(datafiles_ext)+1))+'/'+df['image_name']
            dataframes.append(df)
        
        df = pd.concat(dataframes, ignore_index=True)
        df['data_format'] = 'raw'
        df = self._postprocess_dataframe(df)
        return df