from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from DPF.configs import ShardedFilesDatasetConfig
from DPF.dataloaders import FilesDataset, identical_preprocess_function
from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.filesystems import FileSystem
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping
from DPF.validators import ValidationResult
from DPF.validators.format_validators import ShardedFilesValidator

from .processor_mixins import ApplyTransformProcessorMixin
from .sharded_processor import ShardedDatasetProcessor


class ShardedFilesDatasetProcessor(ShardedDatasetProcessor, ApplyTransformProcessorMixin):
    filesystem: FileSystem
    df: pd.DataFrame
    config: ShardedFilesDatasetConfig

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
        validate_metadata: bool = True,
        columns_to_check: Optional[List[str]] = None,
        workers: int = 1,
        pbar: bool = True
    ) -> ValidationResult:
        if columns_to_check is None:
            columns_to_check = []

        validator = ShardedFilesValidator(
            self.df,
            self.filesystem,
            self.config,
            columns_to_check
        )
        return validator.validate(
            validate_filestructure=validate_filestructure,
            validate_metadata=validate_metadata,
            workers=workers,
            pbar=pbar
        )

    def _get_torch_dataset(
        self,
        modalities: List[ModalityName],
        columns_to_use: Optional[List[str]] = None,
        preprocess_f: Callable[[ModalityToDataMapping, Dict[str, str]], Any] = identical_preprocess_function,
        return_none_on_error: bool = False
    ) -> FilesDataset:
        assert len(set(modalities)) == len(list(modalities))
        datatypes_to_load = [self.config.modality2datatype[m] for m in modalities]
        return FilesDataset(
            self.filesystem,
            self._df,
            datatypes_to_load,  # type: ignore
            metadata_columns=columns_to_use,
            preprocess_function=preprocess_f,
            return_none_on_error=return_none_on_error
        )

    def _read_sample_data(
        self,
        sample: Dict[str, str]
    ) -> ModalityToDataMapping:
        path_column2modality: Dict[str, ModalityName] = {}
        column2modality: Dict[str, ModalityName] = {}
        for d in self.config.datatypes:
            if isinstance(d, ColumnDataType):
                column2modality[d.column_name] = d.modality.name
            elif isinstance(d, ShardedDataType):
                path_column2modality[d.modality.path_column] = d.modality.name
            else:
                raise ValueError()

        modality2data: ModalityToDataMapping = {}
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
