import os.path
from typing import Union, Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from DPF.filesystems import FileSystem, LocalFileSystem
from DPF.filters import DataFilter, ColumnFilter
from DPF.processors.writers import ABSWriter, ShardedFilesWriter, ShardsWriter
from DPF.dataloaders.utils import default_preprocess, default_collate
from DPF.datatypes import ColumnDataType
from DPF.modalities import MODALITIES
from DPF.configs import DatasetConfig, config2format
from DPF.validators import ValidationResult


class DatasetProcessor(ABC):

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        config: DatasetConfig,
    ):
        self.filesystem = filesystem
        self._df = df
        self.config = config

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def columns(self) -> List[str]:
        return self._df.columns.tolist()

    def __getitem__(self, column_name: str) -> pd.Series:
        return self._df[column_name]

    def __setitem__(self, key: str, value: Union[List[str], pd.Series]):
        self._df[key] = value

    def summary(self):
        """Prints summary info about dataset"""
        print('Dataset format:', config2format(self.config))
        print('Path:', self.config.path)
        print('Modalities:', list(self.config.modality2datatype.keys()))

        cols = self.columns
        print('Columns:', len(cols))
        last_s = f'Total samples: {len(self.df)}'
        print(last_s)

        width_columns = ['width', 'WIDTH', 'w']
        width_col = None
        for col in width_columns:
            if col in cols:
                width_col = col
                break
        height_columns = ['height', 'HEIGHT', 'h']
        height_col = None
        for col in height_columns:
            if col in cols:
                height_col = col
                break
        if width_col is not None and height_col is not None:
            data_width = self.df[width_col].describe()
            data_height = self.df[height_col].describe()
            data_aspect_ratio = (self.df[width_col]/self.df[height_col]).describe()
            data_aspect_ratio = {k:round(v, 2) for k,v in data_aspect_ratio.to_dict().items()}
            print('-'*len(last_s))
            print('width:', {k:round(v, 2) for k,v in data_width.to_dict().items()})
            print('height:', {k:round(v, 2) for k,v in data_height.to_dict().items()})
            last_s = f'aspect ratio: {data_aspect_ratio}'
            print(last_s)
            print('-'*len(last_s))

    @abstractmethod
    def rename_columns(self, column_map: Dict[str, str], workers: int = 16) -> List[str]:
        """Renames columns in files of a dataset

        Parameters
        ----------
        column_map: Dict[str, str]
            Mapping of old column name into new column name
        workers: int = 16
            Number of parallel threads

        Returns
        -------
        List[str]
            List of errors
        """
        pass

    @abstractmethod
    def delete_columns(self, columns: List[str], workers: int = 16) -> List[str]:
        """Deletes columns in files of a dataset

        Parameters
        ----------
        columns: List[str]
            List of column names to delete
        workers: int = 16
            Number of parallel threads

        Returns
        -------
        List[str]
            List of errors
        """
        pass

    @abstractmethod
    def update_columns(self, columns: List[str], workers: int = 16) -> List[str]:
        """Updates info in columns or adds new columns in files of a dataset

        Parameters
        ----------
        columns: List[str]
            List of column names to add or update
        workers: int = 16
            Number of parallel threads

        Returns
        -------
        List[str]
            List of errors
        """
        pass

    @abstractmethod
    def get_torch_dataset(
    self,
    modalities: List[str],
    meta_columns: Optional[List[str]] = None,
    preprocess_f: Callable[[dict, dict], Any] = default_preprocess,
    return_none_on_error: bool = False
    ) -> Dataset:
        pass

    def apply_data_filter(
    self,
    datafilter: DataFilter,
    validate_filter_result: bool = True,
        return_none_on_error: bool = False
    ):
        """Applies a data filter to dataset

        Parameters
        ----------
        datafilter: DataFilter
            Instance of a DataFilter
        validate_filter_result: bool = True
            Whether to check the correctness of datafilter result (data integrity)
        return_none_on_error: bool = False
            Whether to return None on sample if there is error in dataloader
        """
        dataset = self.get_torch_dataset(
            modalities=datafilter.modalities,
            meta_columns=datafilter.metadata_columns+[datafilter.key_column],
            preprocess_f=datafilter.preprocess,
            return_none_on_error=return_none_on_error
        )
        df_result = datafilter.run(dataset)

        if validate_filter_result:
            assert set(self._df[datafilter.key_column]) == set(df_result[datafilter.key_column]), \
                f"Result dataframe after filter have different values in key column {datafilter.key_column}"
            assert len(df_result) == len(self._df), \
                f"Length of resulted dataframe changed after filtering. Old length = {len(self._df)}, new = {len(df_result)}"

        self._df = pd.merge(self._df, df_result, on=datafilter.key_column, how='left')

    def apply_column_filter(self, column_filter: ColumnFilter, validate_filter_result: bool = True):
        """Applies a column filter to dataset

        Parameters
        ----------
        column_filter: ColumnFilter
            Instance of a DataFilter
        validate_filter_result: bool = True
            Whether or not to check the correctness of datafilter result (data integrity)
        """
        filter_res = column_filter(self._df)

        if validate_filter_result:
            assert len(filter_res) == len(self._df), \
                f"Length of resulted dataframe changed after filtering. Old length = {len(self._df)}, new = {len(filter_res)}"

        self._df[column_filter.schema] = filter_res

    @abstractmethod
    def validate(
        self,
        validate_filestructure: bool = True,
        columns_to_check: List[str] = [],
        workers: int = 1,
        pbar: bool = True
    ) -> ValidationResult:
        """Validates a dataset to match the data format

        Parameters
        ----------
        validate_filestructure: bool = True
            Whether to validate the filestructure of a dataset
        columns_to_check: List[str] = []
            List of column names that should be in a dataset
        workers: int = 1
            Number of parallel threads for validation
        pbar: bool = True
            Whether to show progress bar or not

        Returns
        -------
        ValidationResult
            A dataclass with all validation errors
        """
        pass

    def get_random_sample(
        self,
        df_filter: Optional[pd.Series] = None
    ) -> (Dict[str, bytes], Dict[str, str]):
        """Returns a random sample from dataset

        Parameters
        ----------
        df_filter: Optional[pd.Series] = None
            Condition for dataframe to filter, df[df_filter] will be used for sampling. If None, uses original dataframe.

        Returns
        -------
        Dict[str, bytes]
            Mapping from modality name to bytes
        Dict[str, str]
            Mapping from column name to its value
        """
        if df_filter is not None:
            df_to_sample = self.df[df_filter]
        else:
            df_to_sample = self.df

        sample = df_to_sample.sample(1).iloc[0].to_dict()
        modality2bytes = self._read_files_from_sample(sample)
        return modality2bytes, sample

    @abstractmethod
    def _read_files_from_sample(
        self,
        sample: Dict[str, str]
    ) -> Dict[str, bytes]:
        pass

    def filter_df(
        self,
        condition: pd.Series
    ):
        self._df = self._df[condition]

    def convert(
        self,
        writer: ABSWriter,
        meta_columns: Optional[List[str]] = None,
        dataloader_kwargs: Optional[dict] = None,
        pbar: bool = True
    ):
        meta_columns = meta_columns or []
        dataloader_kwargs = dataloader_kwargs or {}

        new_dataloader_kwargs = {
            'num_workers': 8,
            'batch_size': 1,
            'collate_fn': default_collate,
            'drop_last': False,
        }
        new_dataloader_kwargs.update(dataloader_kwargs)

        dataset = self.get_torch_dataset(
            list(self.config.modality2datatype.keys()),
            preprocess_f=default_preprocess,
            meta_columns=meta_columns
        )
        dataloader = DataLoader(dataset, **new_dataloader_kwargs)

        with writer as writer:
            for batch in tqdm(dataloader, disable=not pbar):
                modality2bytes, metadata = batch[0]

                modality2sample_data = {}
                for modality, bytes_data in modality2bytes.items():
                    datatype = self.config.modality2datatype[modality]
                    if isinstance(datatype, ColumnDataType):
                        pass
                    else:
                        path_col = MODALITIES[modality].path_column
                        extension = os.path.splitext(os.path.basename(metadata[path_col]))[-1]
                        modality2sample_data[modality] = (extension, bytes_data)
                        metadata.pop(path_col)

                writer.save_sample(modality2sample_data, metadata)

    def to_sharded_files(
        self,
        destination_dir: str,
        filesystem: FileSystem = LocalFileSystem(),
        max_files_in_shard: int = 1000,
        datafiles_ext: str = "csv",
        filenaming: str = "counter",
        meta_columns: Optional[List[str]] = None,
        keys_mapping: Optional[Dict[str, str]] = None,
        workers: int = 8,
        pbar: bool = True
    ):
        """Converts dataset to sharded files format

        Parameters
        ----------
        destination_dir: str
            Path to directory
        filesystem: FileSystem = LocalFileSystem()
            The FileSystem where this path is located
        max_files_in_shard: int = 1000
            Maximum number of files in shard
        datafiles_ext: str = "csv"
            Extension of files with data tables
        filenaming: str = "counter"
            File naming type. "counter" and "uuid" are available
        meta_columns: Optional[List[str]] = None
            Column names to write in new dataset
        keys_mapping: Optional[Dict[str, str]] = None
            Mapping used to rename columns
        workers: int = 8
            Number of parallel processes
        pbar: bool = True
            Whether to show a progress bar
        """
        writer = ShardedFilesWriter(
            filesystem,
            destination_dir,
            keys_mapping=keys_mapping,
            max_files_in_shard=max_files_in_shard,
            datafiles_ext=datafiles_ext,
            filenaming=filenaming
        )
        self.convert(
            writer,
            meta_columns=meta_columns,
            dataloader_kwargs={'num_workers': workers},
            pbar=pbar
        )

    def to_shards(
        self,
        destination_dir: str,
        filesystem: FileSystem = LocalFileSystem(),
        max_files_in_shard: int = 1000,
        datafiles_ext: str = "csv",
        archives_ext: Optional[str] = "tar",
        filenaming: str = "counter",
        meta_columns: Optional[List[str]] = None,
        keys_mapping: Optional[Dict[str, str]] = None,
        workers: int = 8,
        pbar: bool = True
    ):
        """Converts dataset to sharded files format

        Parameters
        ----------
        destination_dir: str
            Path to directory
        filesystem: FileSystem = LocalFileSystem()
            The FileSystem where this path is located
        max_files_in_shard: int = 1000
            Maximum number of files in shard
        datafiles_ext: str = "csv"
            Extension of files with data tables
        archives_ext: Optional[str] = "tar"
            Extension of archives with data
        filenaming: str = "counter"
            File naming type. "counter" and "uuid" are available
        meta_columns: Optional[List[str]] = None
            Column names to write in new dataset
        keys_mapping: Optional[Dict[str, str]] = None
            Mapping used to rename columns
        workers: int = 8
            Number of parallel processes
        pbar: bool = True
            Whether to show a progress bar
        """
        writer = ShardsWriter(
            filesystem,
            destination_dir,
            keys_mapping=keys_mapping,
            max_files_in_shard=max_files_in_shard,
            datafiles_ext=datafiles_ext,
            archives_ext=archives_ext,
            filenaming=filenaming
        )
        self.convert(
            writer,
            meta_columns=meta_columns,
            dataloader_kwargs={'num_workers': workers},
            pbar=pbar
        )

    def __len__(self) -> int:
        return len(self.df)