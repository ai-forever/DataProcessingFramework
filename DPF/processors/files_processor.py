from typing import Dict, List, Optional, Union, Callable, Any
import pandas as pd
from abc import ABC, abstractmethod
import os

from DPF.filesystems import FileSystem
from DPF.datatypes import FileDataType, ColumnDataType
from DPF.configs import DatasetConfig, FilesDatasetConfig
from DPF.processors.helpers import DataFramesChanger
from DPF.filters.data_filter import DataFilter
from DPF.processors.writers import ABSWriter
from .processor import DatasetProcessor
from DPF.dataloaders import default_preprocess, FilesDataset
from DPF.validators.format_validators import FilesValidator, FilesValidationResult


class FilesDatasetProcessor(DatasetProcessor):

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        config: FilesDatasetConfig,
    ):
        super().__init__(filesystem, df, config)

    def rename_columns(self, column_map: Dict[str, str], workers: int = 1) -> List[str]:
        df = self.filesystem.read_dataframe(self.config.table_path)
        for col_old, col_new in column_map.items():
            assert col_old in df.columns, f'Dataframe dont have "{col_old}" column'
            assert col_new not in df.columns, f'Dataframe already have "{col_new}" column'
        df.rename(columns=column_map, inplace=True)
        self.filesystem.save_dataframe(df, self.config.table_path, index=False)

        self._df.rename(columns=column_map, inplace=True)
        return []

    def delete_columns(self, columns: List[str], workers: int = 1) -> List[str]:
        df = self.filesystem.read_dataframe(self.config.table_path)
        for col in columns:
            assert col in df.columns, f'Dataframe dont have "{col}" column'
        df.drop(columns=columns, inplace=True)
        self.filesystem.save_dataframe(df, self.config.table_path, index=False)

        self._df.drop(columns=columns, inplace=True)
        return []

    def update_columns(self, columns: List[str], workers: int = 1) -> List[str]:
        key_column = None
        for d in self.config.datatypes:
            if isinstance(d, FileDataType):
                key_column = d.modality.path_column
                break
        assert key_column is not None, "Cant find key column to use for update"
        assert key_column not in columns, f'Cant update key column "{key_column}"'

        df_old = self.filesystem.read_dataframe(self.config.table_path)
        df_new = self._df[[key_column] + columns]
        df_new.loc[:,key_column] = df_new[key_column].str.slice(len(self.config.base_path)+1)
        assert key_column in df_old.columns, f'Dataframe dont have "{key_column}" column'
        assert set(df_old[key_column]) == set(df_new[key_column]), \
            f'Dataframe has different values in "{key_column}"'

        duplicates = df_old[df_old[key_column].duplicated()][key_column].tolist()
        assert len(duplicates) == 0, f'Dataframe has duplicates in "{key_column}" column: {duplicates}'

        duplicates = df_new[df_new[key_column].duplicated()][key_column].tolist()
        assert len(duplicates) == 0, f'New dataframe has duplicates in "{key_column}" column: {duplicates}'

        assert len(df_old) == len(df_new), f'Length of dataframe is changed'

        columns_to_add = [i for i in df_new.columns if i != key_column]
        columns_intersection = set(df_old.columns).intersection(set(columns_to_add))

        if len(columns_intersection) > 0:
            df_old.drop(columns=list(columns_intersection), inplace=True)

        df = pd.merge(df_old, df_new, on=key_column)
        self.filesystem.save_dataframe(df, self.config.table_path, index=False)
        return []

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_dataframes: bool = True,
        columns_to_check: List[str] = [],
        workers: int = 1,
        pbar: bool = True
    ) -> FilesValidationResult:
        validator = FilesValidator(
            self.df,
            self.filesystem,
            self.config,
            columns_to_check
        )
        return validator.validate(
            validate_filestructure=validate_filestructure,
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
            elif isinstance(d, FileDataType):
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
