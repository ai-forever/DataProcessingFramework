from typing import Dict, List, Optional, Union, Callable, Any
import pandas as pd
from torch.utils.data import Dataset

from DPF.filesystems import FileSystem
from DPF.modalities import MODALITIES
from DPF.configs import DatasetConfig, ShardedDatasetConfig
from DPF.dataloaders import ShardsDataset, default_preprocess
from .sharded_processor import ShardedDatasetProcessor


class ShardsDatasetProcessor(ShardedDatasetProcessor):

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        config: ShardedDatasetConfig
    ):
        super().__init__(filesystem, df, config)

    def get_container_path(self, split_name: str) -> str:
        return self.config.path + '/' + split_name + '.' + self.config.archives_ext

    def get_torch_dataset(
        self,
        modalities: List[str],
        meta_columns: Optional[List[str]] = None,
        preprocess_f: Callable[[dict, dict], Any] = default_preprocess,
        return_none_on_error: bool = False
    ) -> ShardsDataset:
        assert len(set(modalities)) == len(list(modalities))
        split2archive_path = {
            i: self.get_container_path(i) for i in self.df['split_name'].unique().tolist()
        }
        datatypes_to_load = [self.config.modality2datatype[m] for m in modalities]
        return ShardsDataset(
            self.filesystem,
            self._df,
            split2archive_path,
            datatypes_to_load,
            meta_columns=meta_columns,
            preprocess_f=preprocess_f
        )