import pandas as pd
import os
from typing import List, Set, Optional
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from DPF.processors.text2image.raw_processor import RawProcessor
from DPF.processors.text2image.shards_processor import ShardsProcessor
from DPF.processors.text2image.t2i_processor import T2IProcessor
from DPF.filesystems import LocalFileSystem, S3FileSystem, FileSystem
from DPF.utils.utils import get_file_extension
from .formatter import Formatter


class DataframeReader:
    """
    DataframeReader is used to read and preprocess dataframes
    """
    
    def __init__(
            self, 
            filesystem: FileSystem, 
            read_params: dict, 
            df_needed_columns: Set[str],
            check_same_columns: bool = True,
        ):
        self.filesystem = filesystem
        self.read_params = read_params
        
        self.check_same_columns = check_same_columns
        self.df_needed_columns = df_needed_columns
        
    def _add_base_columns(
            self, 
            df: pd.DataFrame, 
            filepath: str, 
            caption_column: str, 
            imagename_column: str, 
            image_ext: str
        ):
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
        
    def load_shards_df(self, filepath: str) -> pd.DataFrame:
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
        
    def load_raw_df(self, filepath: str) -> pd.DataFrame:
        datafiles_ext = self.read_params["datafiles_ext"]
        image_ext = self.read_params["image_ext"]
        caption_column = self.read_params["caption_column"]
        imagename_column = self.read_params["imagename_column"]
        
        df = self.filesystem.read_dataframe(filepath)
            
        self._add_base_columns(df, filepath, caption_column, imagename_column, image_ext)

        df['image_path'] = df['table_path'].str.slice(0,-(len(datafiles_ext)+1))+'/'+df['image_name']
        return df
    
        
class T2IFormatter(Formatter):
    """
    Formatter for text-to-image datasets. 
    Formatter is used to read and create a Processor class for a dataset.
    """

    def __init__(
            self,
            filesystem: str = 'local',
            **filesystem_kwargs
        ):
        super().__init__(filesystem, **filesystem_kwargs)

    def _postprocess_dataframe(self, df: pd.DataFrame):
        # TODO: do not create a copy of df, use inplace operations
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
    ) -> ShardsProcessor:
        """
        Reads a dataset in shards (images in archives and dataframes) and creates a ShardsProcessor for dataset.
        
        Parameters
        ----------
        dataset_path: str
            Path to shards
        archive_ext: str = 'tar'
            Extension of archives
        datafiles_ext: str = 'csv'
            Extension of tables with data
        imagename_column: str = 'image_name'
            Name of column with image names
        caption_column: str = 'caption'
            Name of column with captions
        check_same_columns: bool = True
            Check that the columns in all tables are the same
        image_ext: Optional[str] = None
            Extension of images in tars if there is no extensions in imagename_column
        processes: int = 1
            Number of parallel processes to read a dataset data
        progress_bar: bool = False
            Progress bar to track dataset reading process
            
        Returns
        -------
        ShardsProcessor
            ShardsProcessor object for given dataset
        """
        
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
    ) -> RawProcessor:
        """
        Reads a dataset in raw format (images in folders and dataframes) and creates a RawProcessor for dataset.
        
        Parameters
        ----------
        dataset_path: str
            Path to dataset
        datafiles_ext: str = 'csv'
            Extension of tables with data
        imagename_column: str = 'image_name'
            Name of column with image names
        caption_column: str = 'caption'
            Name of column with captions
        check_same_columns: bool = True
            Check that the columns in all tables are the same
        image_ext: Optional[str] = None
            Extension of images in folders if there is no extensions in imagename_column
        processes: int = 1
            Number of parallel processes to read a dataset data
        progress_bar: bool = False
            Progress bar to track dataset reading process
            
        Returns
        -------
        RawProcessor
            RawProcessor object for given dataset
        """
        
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
