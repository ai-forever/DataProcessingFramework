import pandas as pd
import os
from typing import List, Set, Optional
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from DPF.processors.text2image.raw_processor import RawProcessor
from DPF.processors.text2image.shards_processor import ShardsProcessor
from DPF.filesystems import LocalFileSystem, S3FileSystem
from DPF.utils.utils import get_file_extension


class DataframeReader:
    def __init__(
            self, 
            filesystem, 
            read_params: dict, 
            df_needed_columns: Set[str],
            check_same_columns: bool = True,
        ):
        self.filesystem = filesystem
        self.read_params = read_params
        
        self.check_same_columns = check_same_columns
        self.df_needed_columns = df_needed_columns
        
    def _add_base_columns(self, df, filepath, caption_column, imagename_column, image_ext):
        if self.check_same_columns:
            assert set(df.columns) == self.df_needed_columns, \
                f'Dataframe {filepath} have different columns. Expected {self.df_needed_columns}, got {set(df.columns)}'
        assert imagename_column in df.columns, f'Dataframe {filepath} does not have "{imagename_column}" column'
        assert caption_column in df.columns, f'Dataframe {filepath} does not have "{caption_column}" column'

        df['table_path'] = filepath
        df['caption'] = df[caption_column]
        df['image_name'] = df[imagename_column]
        if image_ext:
            image_ext = image_ext.lstrip('.')
            df['image_name'] += '.'+image_ext
        
    def load_shards_df(self, filepath):
        datafiles_ext = self.read_params["datafiles_ext"]
        archive_ext = self.read_params["archive_ext"]
        image_ext = self.read_params["image_ext"]
        caption_column = self.read_params["caption_column"]
        imagename_column = self.read_params["imagename_column"]
        
        df = self.filesystem.read_dataframe(filepath)

        self._add_base_columns(df, filepath, caption_column, imagename_column, image_ext)

        df['archive_path'] = df['table_path'].str.rstrip(datafiles_ext)+archive_ext
        df['image_path'] = df['archive_path']+'/'+df['image_name']
        return df
        
    def load_raw_df(self, filepath):
        datafiles_ext = self.read_params["datafiles_ext"]
        image_ext = self.read_params["image_ext"]
        caption_column = self.read_params["caption_column"]
        imagename_column = self.read_params["imagename_column"]
        
        df = self.filesystem.read_dataframe(filepath)
            
        self._add_base_columns(df, filepath, caption_column, imagename_column, image_ext)

        df['image_path'] = df['table_path'].str.slice(0,-(len(datafiles_ext)+1))+'/'+df['image_name']
        return df
    
        
class T2IFormatter:

    def __init__(
            self,
            filesystem: str = 'local',
            **filesystem_kwargs
        ):
        if filesystem == 'local':
            self.filesystem = LocalFileSystem()
        elif filesystem == 's3':
            self.filesystem = S3FileSystem(**filesystem_kwargs)
        else:
            raise NotImplementedError(f"Unknown filesystem format: {filesystem}")

    def _postprocess_dataframe(self, df: pd.DataFrame):
        columns = ['image_name', 'image_path', 'table_path', 'archive_path', 'data_format', 'caption']
        columns = [i for i in columns if i in df.columns]
        orig_columns = [i for i in df.columns if i not in columns]
        columns.extend(list(orig_columns))
        
        df_has_null_caption = df['caption'].isnull().values.any()
        if df_has_null_caption:
            print(f'[WARNING] Column with captions has NaN values.')
            df.fillna(value={'caption': ''}, inplace=True)
            
        df['caption'] = df['caption'].astype("string")
        
        return df[columns]
    
    def from_shards(
        self,
        dataset_path: str,
        archive_ext: str = 'tar',
        datafiles_ext: str = 'csv', 
        imagename_column: str = 'image_name',
        caption_column: str = 'caption',
        check_same_columns: bool = True,
        image_ext: Optional[str] = None,
        processes: int = 1,
        progress_bar: bool = False
    ) -> pd.DataFrame:
        
        dataset_path = dataset_path.rstrip('/')
        datafiles_ext = datafiles_ext.lstrip('.')
        archive_ext = archive_ext.lstrip('.')
        
        datafiles = self.filesystem.listdir_with_ext(dataset_path, ext=datafiles_ext)
        
        dataframes = []
        if len(datafiles) > 0:
            reader = DataframeReader(
                self.filesystem,
                {
                    "datafiles_ext": datafiles_ext, "archive_ext": archive_ext, "image_ext": image_ext,
                    "imagename_column": imagename_column, "caption_column": caption_column
                },
                check_same_columns = check_same_columns,
                df_needed_columns = set(self.filesystem.read_dataframe(datafiles[0]).columns)
            )
            dataframes = process_map(
                reader.load_shards_df, datafiles, disable=not progress_bar, 
                max_workers=processes, chunksize=1
            )
        
        df = pd.concat(dataframes, ignore_index=True)
        df['data_format'] = 'shards'
        df = self._postprocess_dataframe(df)
        
        processor = ShardsProcessor(
            self.filesystem,
            df,
            dataset_path,
            archive_ext=archive_ext,
            datafiles_ext=datafiles_ext, 
            imagename_column=imagename_column,
            caption_column=caption_column,
            image_ext=image_ext
        )
        return processor
    
    def from_raw(
        self,
        dataset_path: str, 
        datafiles_ext: str = 'csv', 
        imagename_column: str = 'image_name',
        caption_column: str = 'caption',
        check_same_columns: bool = True,
        image_ext: Optional[str] = None,
        processes: int = 1,
        progress_bar: bool = False,
    ) -> pd.DataFrame:
        
        dataset_path = dataset_path.rstrip('/')
        datafiles_ext = datafiles_ext.lstrip('.')
        
        datafiles = self.filesystem.listdir_with_ext(dataset_path, ext=datafiles_ext)
        
        dataframes = []
        if len(datafiles) > 0:
            reader = DataframeReader(
                self.filesystem,
                {
                    "datafiles_ext": datafiles_ext, "image_ext": image_ext,
                    "imagename_column": imagename_column, "caption_column": caption_column
                },
                check_same_columns = check_same_columns,
                df_needed_columns = set(self.filesystem.read_dataframe(datafiles[0]).columns)
            )
            dataframes = process_map(
                reader.load_raw_df, datafiles, disable=not progress_bar, 
                max_workers=processes, chunksize=1
            )
        
        df = pd.concat(dataframes, ignore_index=True)
        df['data_format'] = 'raw'
        df = self._postprocess_dataframe(df)
        
        processor = RawProcessor(
            self.filesystem,
            df,
            dataset_path,
            datafiles_ext=datafiles_ext, 
            imagename_column=imagename_column,
            caption_column=caption_column,
            image_ext=image_ext
        )
        return processor
