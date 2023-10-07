from typing import Dict, List, Optional, Union, Callable, Any
import pandas as pd
from torch.utils.data import Dataset

from DPF.filesystems import FileSystem
from DPF.modalities import MODALITIES
from DPF.configs import DatasetConfig, ShardedFilesDatasetConfig
from DPF.dataloaders import FilesDataset, default_preprocess
from .sharded_processor import ShardedDatasetProcessor


class ShardedFilesDatasetProcessor(ShardedDatasetProcessor):

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        config: ShardedFilesDatasetConfig
    ):
        super().__init__(filesystem, df, config)

    def get_container_path(self, split_name: str) -> str:
        return self.config.path + '/' + split_name + '/'

    def get_torch_dataset(
        self,
        modalities: List[str],
        meta_columns: Optional[List[str]] = None,
        preprocess_f: Callable[[dict, dict], Any] = default_preprocess,
        return_none_on_error: bool = False
    ) -> FilesDataset:
        assert len(set(modalities)) == len(list(modalities))
        datatypes_to_load = [self.config.modality2datatype[m] for m in modalities]
        return FilesDataset(
            self.filesystem,
            self._df,
            datatypes_to_load,
            meta_columns=meta_columns,
            preprocess_f=preprocess_f
        )