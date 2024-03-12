from abc import ABC, abstractmethod

import pandas as pd
from tqdm.contrib.concurrent import thread_map

from DPF.configs import ShardedDatasetConfig
from DPF.connectors import Connector
from DPF.datatypes import ShardedDataType
from DPF.validators import ValidationResult, Validator
from DPF.validators.errors import (
    DataFrameErrorType,
    DuplicatedValuesError,
    FileStructureErrorType,
    IsNotKeyError,
    MissedColumnsError,
)


class ShardedValidator(Validator, ABC):

    def __init__(
        self,
        merged_df: pd.DataFrame,
        connector: Connector,
        config: ShardedDatasetConfig,
        columns_to_check: list[str]
    ):
        self.merged_df = merged_df
        self.connector = connector
        self.config = config
        self.columns_to_check = columns_to_check

    @abstractmethod
    def _validate_files(self, filepaths: list[str]) -> list[FileStructureErrorType]:
        pass

    def _validate_filestructure(self, filepaths: list[str]) -> list[FileStructureErrorType]:
        errors: list[FileStructureErrorType] = []

        for datatype in self.config.datatypes:
            if isinstance(datatype, ShardedDataType):
                has_duplicates = self.merged_df[datatype.modality.path_column].str.split('/').str[-1].duplicated().any()
                if has_duplicates:
                    errors.append(IsNotKeyError(datatype.user_basename_column_name))

        errors.extend(self._validate_files(filepaths))
        return errors

    @abstractmethod
    def _validate_shard_files(
        self,
        dataframe_path: str,
        df: pd.DataFrame
    ) -> tuple[list[FileStructureErrorType], list[DataFrameErrorType]]:
        pass

    def _validate_shard(self, path: str) -> tuple[str, list[DataFrameErrorType], list[FileStructureErrorType]]:
        df = self.connector.read_dataframe(path)
        dataframe_errors: list[DataFrameErrorType] = []

        # validate dataframe
        missed_columns = set(self.columns_to_check).difference(set(df.columns))
        if len(missed_columns) > 0:
            dataframe_errors.append(MissedColumnsError(path, list(missed_columns)))
        for datatype in self.config.datatypes:
            if isinstance(datatype, ShardedDataType):
                filenames = df[datatype.user_basename_column_name]
                has_duplicates = filenames.duplicated().any()
                if has_duplicates:
                    dataframe_errors.append(DuplicatedValuesError(path, datatype.user_basename_column_name))

        # validate contents
        filestructure_errors, df_errors = self._validate_shard_files(path, df)
        #
        dataframe_errors.extend(df_errors)
        return path, dataframe_errors, filestructure_errors

    def _validate_shards(
        self,
        filepaths: list[str],
        workers: int = 4,
        pbar: bool = True
    ) -> tuple[dict[str, list[DataFrameErrorType]], list[FileStructureErrorType]]:
        datafiles = [f for f in filepaths if f.endswith('.'+self.config.datafiles_ext)]

        results = thread_map(
            self._validate_shard,
            datafiles,
            max_workers=workers,
            disable=not pbar,
            chunksize=1
        )
        dataframe2errors = {}
        filestructure_errors = []
        for res in results:
            dataframe2errors[res[0]] = res[1]
            filestructure_errors.extend(res[2])
        return dataframe2errors, filestructure_errors

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_metadata: bool = True,
        workers: int = 4,
        pbar: bool = True
    ) -> ValidationResult:
        filepaths = self.connector.listdir(self.config.path)
        filestructure_errors: list[FileStructureErrorType] = []
        dataframe2errors: dict[str, list[DataFrameErrorType]] = {}

        if validate_filestructure:
            filestructure_errors.extend(self._validate_filestructure(filepaths))
        if validate_filestructure or validate_metadata:
            _dataframe2errors, _filestructure_errors = self._validate_shards(filepaths, workers, pbar)
            if validate_filestructure:
                filestructure_errors.extend(_filestructure_errors)
            if validate_metadata:
                for path, errors in _dataframe2errors.items():
                    if len(errors) > 0:
                        dataframe2errors[path] = errors

        return ValidationResult(filestructure_errors, dataframe2errors)
