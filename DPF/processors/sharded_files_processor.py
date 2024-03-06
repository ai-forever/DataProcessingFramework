from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from DPF.configs import ShardedFilesDatasetConfig
from DPF.dataloaders import FilesDataset, identical_preprocess_function
from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.filesystems import FileSystem
from DPF.modalities import MODALITIES
from DPF.transforms import BaseFilesTransforms
from DPF.validators.format_validators import (
    ShardedFilesValidator,
    ShardedValidationResult,
)

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

        validator = ShardedFilesValidator(  # type: ignore
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
    ) -> FilesDataset:
        assert len(set(modalities)) == len(list(modalities))
        datatypes_to_load = [self.config.modality2datatype[m] for m in modalities]
        return FilesDataset(
            self.filesystem,
            self._df,
            datatypes_to_load,
            meta_columns=meta_columns,
            preprocess_function=preprocess_f,
            return_none_on_error=return_none_on_error
        )

    def _read_files_from_sample(
        self,
        sample: Dict[str, str]
    ) -> Dict[str, Union[bytes, str]]:
        path_column2modality: Dict[str, str] = {}
        column2modality: Dict[str, str] = {}
        for d in self.config.datatypes:
            if isinstance(d, ColumnDataType):
                column2modality[d.column_name] = d.modality.name
            elif isinstance(d, ShardedDataType):
                path_column2modality[d.modality.path_column] = d.modality.name
            else:
                raise ValueError()

        modality2data: Dict[str, Union[bytes, str]] = {}
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

    def apply_transform(self, transforms: Union[BaseFilesTransforms]) -> None:
        assert transforms.modality in self.config.modality2datatype

        filepath_column = MODALITIES[transforms.modality].path_column
        filepaths = self.df[filepath_column].tolist()

        metadata_lists = None
        if len(transforms.required_metadata) > 0:
            metadata_lists = {
                col: self.df[col].tolist()
                for col in transforms.required_metadata
            }

        transformed_metadata = transforms.run(filepaths, metadata_lists=metadata_lists)
        for data in transformed_metadata:
            data.metadata[filepath_column] = data.filepath
        df_to_merge = pd.DataFrame([data.metadata for data in transformed_metadata])

        # drop metadata columns from original df to replace them
        self._df.drop(columns=transforms.metadata_to_change, errors='ignore', inplace=True)

        self._df = pd.merge(self._df, df_to_merge, on=filepath_column, how='left')
