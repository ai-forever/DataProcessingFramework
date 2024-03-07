import os
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from DPF.configs import ShardedDatasetConfig
from DPF.dataloaders import ShardsDataset, identical_preprocess_function
from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.filesystems import FileSystem
from DPF.validators.format_validators import ShardedValidationResult, ShardsValidator

from .sharded_processor import ShardedDatasetProcessor
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping


class ShardsDatasetProcessor(ShardedDatasetProcessor):

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        config: ShardedDatasetConfig
    ):
        super().__init__(filesystem, df, config)

    def get_container_path(self, split_name: str) -> str:
        return self.config.path + '/' + split_name + '.' + self.config.archives_ext  # type: ignore

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_shards: bool = True,
        columns_to_check: Optional[List[str]] = None,
        workers: int = 1,
        pbar: bool = True
    ) -> ShardedValidationResult:
        if columns_to_check is None:
            columns_to_check = []
        validator = ShardsValidator(  # type: ignore
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

    def _get_torch_dataset(
        self,
        modalities: List[ModalityName],
        columns_to_use: Optional[List[str]] = None,
        preprocess_f: Callable[[ModalityToDataMapping, Dict[str, str]], Any] = identical_preprocess_function,
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
            datatypes_to_load,  # type: ignore
            metadata_columns=columns_to_use,
            preprocess_function=preprocess_f,
            return_none_on_error=return_none_on_error
        )

    def _read_sample_data(
        self,
        sample: Dict[str, str]
    ) -> ModalityToDataMapping:
        tar_path = self.get_container_path(sample['split_name'])
        path_column2modality: Dict[str, ModalityName] = {}
        column2modality: Dict[str, ModalityName] = {}
        for d in self.config.datatypes:
            if isinstance(d, ColumnDataType):
                column2modality[d.column_name] = d.modality.name
            elif isinstance(d, ShardedDataType):
                path_column2modality[d.modality.path_column] = d.modality.name
            else:
                raise ValueError()

        tar = self.filesystem.read_tar(tar_path)

        modality2data: ModalityToDataMapping = {}
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
