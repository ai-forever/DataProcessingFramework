import os.path
from typing import Union, Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from DPF.filesystems import FileSystem
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
    def rename_columns(self, column_map: Dict[str, str]) -> List[str]:
        pass

    @abstractmethod
    def delete_columns(self, columns: List[str]) -> List[str]:
        pass

    @abstractmethod
    def update_columns(self, columns: List[str]) -> List[str]:
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

    @abstractmethod
    def validate(
        self,
        validate_filestructure: bool = True,
        validate_dataframes: bool = True,
        columns_to_check: List[str] = [],
        workers: int = 1,
        pbar: bool = True
    ) -> ValidationResult:
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
        max_files_in_shard: int = 1000,
        datafiles_ext: str = "csv",
        meta_columns: Optional[List[str]] = None,
        dataloader_kwargs: Optional[dict] = None,
        pbar: bool = True
    ):
        writer = ShardedFilesWriter(
            self.filesystem,
            destination_dir,
            max_files_in_shard=max_files_in_shard,
            datafiles_ext=datafiles_ext
        )
        self.convert(
            writer,
            meta_columns=meta_columns,
            dataloader_kwargs=dataloader_kwargs,
            pbar=pbar
        )

    def to_shards(
        self,
        destination_dir: str,
        max_files_in_shard: int = 1000,
        datafiles_ext: str = "csv",
        archives_ext: Optional[str] = "tar",
        meta_columns: Optional[List[str]] = None,
        dataloader_kwargs: Optional[dict] = None,
        pbar: bool = True
    ):
        writer = ShardsWriter(
            self.filesystem,
            destination_dir,
            max_files_in_shard=max_files_in_shard,
            datafiles_ext=datafiles_ext,
            archives_ext=archives_ext
        )
        self.convert(
            writer,
            meta_columns=meta_columns,
            dataloader_kwargs=dataloader_kwargs,
            pbar=pbar
        )

    def __len__(self) -> int:
        return len(self.df)