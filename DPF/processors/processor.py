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
from DPF.configs import DatasetConfig
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

    @abstractmethod
    def rename_columns(self, column_map: Dict[str, str], workers: int = 16) -> List[str]:
        pass

    @abstractmethod
    def delete_columns(self, columns: List[str], workers: int = 16) -> List[str]:
        pass

    @abstractmethod
    def update_columns(self, columns: List[str], workers: int = 16) -> List[str]:
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

    def apply_data_filter(self, datafilter: DataFilter, validate_filter_result: bool = True):
        dataset_kwargs = datafilter.get_dataset_kwargs()
        dataset = self.get_torch_dataset(**dataset_kwargs)
        df_result = datafilter.run(dataset)

        if validate_filter_result:
            assert set(self._df[datafilter.key_column]) == set(df_result[datafilter.key_column]), \
                f"Result dataframe after filter have different values in key column {datafilter.key_column}"
            assert len(df_result) == len(self._df), \
                f"Length of resulted dataframe changed after filtering. Old length = {len(self._df)}, new = {len(df_result)}"

        self._df = pd.merge(self._df, df_result, on=datafilter.key_column)

    def apply_column_filter(self, column_filter: ColumnFilter, validate_filter_result: bool = True):
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
        pass

    def get_random_sample(
        self,
        df_filter: Optional[pd.Series] = None
    ) -> (Dict[str, bytes], Dict[str, str]):
        if df_filter:
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