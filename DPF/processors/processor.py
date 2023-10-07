import os.path
from typing import Union, Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from DPF.filesystems import FileSystem
from DPF.processors.writers import ABSWriter
from DPF.dataloaders.utils import default_preprocess, default_collate
from DPF.datatypes import ColumnDataType
from DPF.modalities import MODALITIES
from DPF.configs import DatasetConfig


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

    def convert(
        self,
        writer: ABSWriter,
        meta_columns: Optional[List[str]] = None,
        dataloader_kwargs: Optional[dict] = None
    ):
        meta_columns = meta_columns or []

        dataloader_kwargs = dataloader_kwargs or {
            'num_workers': 8,
            'batch_size': 1,
            'collate_fn': default_collate,
            'drop_last': False,
        }
        dataset = self.get_torch_dataset(
            list(self.config.modality2datatype.keys()),
            preprocess_f=default_preprocess,
            meta_columns=meta_columns
        )
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        with writer as writer:
            for batch in dataloader:
                modality2bytes, metadata = batch[0]

                modality2sample_data = {}
                for modality, bytes_data in modality2bytes.items():
                    datatype = self.config.modality2datatype[modality]
                    if isinstance(datatype, ColumnDataType):
                        pass
                    else:
                        path_col = MODALITIES[modality].path_column
                        modality2sample_data[modality] = (
                            os.path.splitext(os.path.basename(metadata[path_col]))[-1],
                            bytes_data
                        )
                        metadata.pop(path_col)

                writer.save_sample(modality2sample_data, metadata)
                break

    # TODO
    # @abstractmethod
    # def validate(self):
