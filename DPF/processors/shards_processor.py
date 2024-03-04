import os
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from DPF.configs import ShardedDatasetConfig
from DPF.dataloaders import ShardsDataset, identical_preprocess_function
from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.filesystems import FileSystem
from DPF.validators.format_validators import ShardedValidationResult, ShardsValidator

from ..transforms import BaseFilesTransforms
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

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_shards: bool = True,
        columns_to_check: List[str] = [],
        workers: int = 1,
        pbar: bool = True
    ) -> ShardedValidationResult:
        validator = ShardsValidator(
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
        preprocess_f: Callable[[dict, dict], Any] = identical_preprocess_function,
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
            preprocess_function=preprocess_f,
            return_none_on_error=return_none_on_error
        )

    def _read_files_from_sample(
        self,
        sample: Dict[str, str]
    ) -> Dict[str, bytes]:
        tar_path = self.get_container_path(sample['split_name'])
        path_column2modality = {}
        column2modality = {}
        for d in self.config.datatypes:
            if isinstance(d, ColumnDataType):
                column2modality[d.modality.column] = d.modality.key
            elif isinstance(d, ShardedDataType):
                path_column2modality[d.modality.path_column] = d.modality.key
            else:
                raise ValueError()

        tar = self.filesystem.read_tar(tar_path)

        modality2data = {}
        # read files
        for col in path_column2modality.keys():
            modality = path_column2modality[col]
            filename = os.path.basename(sample[col])
            file_bytes = tar.extractfile(filename).read()
            modality2data[modality] = file_bytes
        # read data from columns
        for col in column2modality.keys():
            modality = column2modality[col]
            modality2data[modality] = sample[col]
        return modality2data

    def apply_transform(self, transforms: Union[BaseFilesTransforms]):
        raise NotImplementedError()
