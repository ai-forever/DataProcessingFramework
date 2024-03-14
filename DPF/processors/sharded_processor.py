import os
from abc import ABC, abstractmethod

import pandas as pd

from DPF.configs import ShardedDatasetConfig
from DPF.connectors import Connector
from DPF.datatypes import ShardedDataType
from DPF.processors.helpers import DataFramesChanger

from .processor import DatasetProcessor


class ShardedDatasetProcessor(DatasetProcessor, ABC):

    def __init__(
        self,
        connector: Connector,
        df: pd.DataFrame,
        config: ShardedDatasetConfig,
    ):
        super().__init__(connector, df, config)
        assert 'split_name' in self.columns

    @abstractmethod
    def get_shard_path(self, split_name: str) -> str:
        pass

    def get_datafile_path(self, split_name: str) -> str:
        return self.config.path+'/'+split_name+'.'+self.config.datafiles_ext  # type: ignore

    def rename_columns(self, column_map: dict[str, str], workers: int = 16) -> list[str]:
        splits = self.df['split_name'].unique().tolist()
        datafile_paths = [self.get_datafile_path(split) for split in splits]

        helper = DataFramesChanger(
            datafile_paths, self.connector, self.config
        )
        errors = helper.rename_columns(column_map, max_threads=workers)
        self._df.rename(columns=column_map, inplace=True)
        return errors

    def delete_columns(self, columns: list[str], workers: int = 16) -> list[str]:
        for col in columns:
            assert col not in self.config.user_column2default_column.keys(), \
                f'Column "{col}" is required column for "{self.config.user_column2default_column[col]}"'

        splits = self.df['split_name'].unique().tolist()
        datafile_paths = [self.get_datafile_path(split) for split in splits]

        helper = DataFramesChanger(
            datafile_paths, self.connector, self.config
        )
        errors = helper.delete_columns(columns, max_threads=workers)
        self._df.drop(columns=columns, inplace=True)
        return errors

    def update_columns(self, columns: list[str], workers: int = 16) -> list[str]:
        key_column = None
        path_column = None
        for d in self.config.datatypes:
            if isinstance(d, ShardedDataType):
                key_column = d.user_basename_column_name
                path_column = d.modality.path_column
                break
        assert key_column is not None, "Cant find key column to use for update"
        assert key_column not in columns, f'Cant update key column "{key_column}"'

        def _add_key_column(data: pd.DataFrame) -> pd.DataFrame:
            data[key_column] = data[path_column].apply(os.path.basename)
            return data

        table_to_new_data = self.df.groupby("split_name").apply(
            lambda x: list(v for v in _add_key_column(x[[path_column]+columns]).to_dict("records"))  # noqa
        )
        table_to_new_data.index = [self.get_datafile_path(i) for i in table_to_new_data.index]

        helper = DataFramesChanger(
            list(table_to_new_data.keys()), self.connector, self.config
        )
        errors = helper.update_columns(key_column, dict(table_to_new_data), max_threads=workers)
        return errors

