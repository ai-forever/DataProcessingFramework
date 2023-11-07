from typing import Dict, List, Optional, Union, Callable, Any
import pandas as pd

from DPF.filesystems import FileSystem
from DPF.configs import ShardedFilesDatasetConfig
from DPF.dataloaders import FilesDataset, default_preprocess
from .sharded_processor import ShardedDatasetProcessor
from DPF.validators.format_validators import ShardedValidationResult, ShardedFilesValidator
from DPF.datatypes import ColumnDataType, ShardedDataType


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
        validate_shards: bool = True,
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
            validate_shards=validate_shards,
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

    def _read_files_from_sample(
        self,
        sample: Dict[str, str]
    ) -> Dict[str, bytes]:
        path_column2modality = {}
        column2modality = {}
        for d in self.config.datatypes:
            if isinstance(d, ColumnDataType):
                column2modality[d.modality.column] = d.modality.key
            elif isinstance(d, ShardedDataType):
                path_column2modality[d.modality.path_column] = d.modality.key
            else:
                raise ValueError()

        modality2data = {}
        # read files
        for col in path_column2modality.keys():
            modality = path_column2modality[col]
            file_bytes = self.filesystem.read_file(sample[col], binary=True).getvalue()
            modality2data[modality] = file_bytes
        # read data from columns
        for col in column2modality.keys():
            modality = column2modality[col]
            modality2data[modality] = sample[col]
        return modality2data