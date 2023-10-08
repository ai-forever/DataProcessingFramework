from typing import Dict, List, Optional, Union, Callable, Any
import pandas as pd

from DPF.filesystems import FileSystem
from DPF.configs import ShardedFilesDatasetConfig
from DPF.dataloaders import FilesDataset, default_preprocess
from .sharded_processor import ShardedDatasetProcessor
from DPF.validators.format_validators import ShardedValidationResult, ShardedFilesValidator


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

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_dataframes: bool = True,
        columns_to_check: List[str] = [],
        workers: int = 1,
        pbar: bool = True
    ) -> ShardedValidationResult:
        validator = ShardedFilesValidator(
            self.df,
            self.filesystem,
            self.config,
            columns_to_check
        )
        return validator.validate(
            validate_filestructure=validate_filestructure,
            validate_dataframes=validate_dataframes,
            workers=workers,
            pbar=pbar
        )

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