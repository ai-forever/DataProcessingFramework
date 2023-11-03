from typing import Dict, List, Optional, Union, Callable, Any
import pandas as pd
from abc import ABC, abstractmethod
import os

from DPF.filesystems import FileSystem
from DPF.datatypes import ShardedDataType
from DPF.configs import DatasetConfig, ShardedDatasetConfig
from DPF.processors.helpers import DataFramesChanger
from DPF.filters.data_filter import DataFilter
from DPF.processors.writers import ABSWriter
from .processor import DatasetProcessor


class ShardedDatasetProcessor(DatasetProcessor, ABC):

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        config: ShardedDatasetConfig,
    ):
        super().__init__(filesystem, df, config)
        assert 'split_name' in self.columns

    @abstractmethod
    def get_container_path(self, split_name: str) -> str:
        pass

    def get_datafile_path(self, split_name: str) -> str:
        return self.config.path+'/'+split_name+'.'+self.config.datafiles_ext

    def rename_columns(self, column_map: Dict[str, str], workers: int = 16) -> List[str]:
        splits = self.df['split_name'].unique().tolist()
        datafile_paths = [self.get_datafile_path(split) for split in splits]

        helper = DataFramesChanger(
            datafile_paths, self.filesystem, self.config
        )
        errors = helper.rename_columns(column_map, max_threads=workers)
        self._df.rename(columns=column_map, inplace=True)
        return errors

    def delete_columns(self, columns: List[str], workers: int = 16) -> List[str]:
        for col in columns:
            assert col not in self.config.columns_mapping.keys(), \
                f'Column "{col}" is required column for "{self.config.columns_mapping[col]}"'

        splits = self.df['split_name'].unique().tolist()
        datafile_paths = [self.get_datafile_path(split) for split in splits]

        helper = DataFramesChanger(
            datafile_paths, self.filesystem, self.config
        )
        errors = helper.delete_columns(columns, max_threads=workers)
        self._df.drop(columns=columns, inplace=True)
        return errors

    def update_columns(self, columns: List[str], workers: int = 16) -> List[str]:
        key_column = None
        path_column = None
        for d in self.config.datatypes:
            if isinstance(d, ShardedDataType):
                key_column = d.user_basename_column_name
                path_column = d.modality.path_column
                break
        assert key_column is not None, "Cant find key column to use for update"
        assert key_column not in columns, f'Cant update key column "{key_column}"'

        def _add_key_column(data):
            data[key_column] = data[path_column].apply(os.path.basename)
            return data

        table_to_new_data = self.df.groupby("split_name").apply(
            lambda x: list(v for v in _add_key_column(x[[path_column]+columns]).to_dict("records"))
        )
        table_to_new_data.index = [self.get_datafile_path(i) for i in table_to_new_data.index]

        helper = DataFramesChanger(
            list(table_to_new_data.keys()), self.filesystem, self.config
        )
        errors = helper.update_columns(key_column, dict(table_to_new_data), max_threads=workers)
        return errors

