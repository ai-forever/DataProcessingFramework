from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from DPF.configs import FilesDatasetConfig
from DPF.datatypes import FileDataType
from DPF.filesystems import FileSystem
from DPF.validators import ValidationResult, Validator
from DPF.validators.format_validators import (
    DataFrameError,
    DuplicatedValuesError,
    FileStructureError,
    IsNotKeyError,
    MissedColumnsError,
)


@dataclass
class FilesValidationResult(ValidationResult):
    filestructure_errors: List[FileStructureError]
    dataframe_errors: Dict[str, List[DataFrameError]]

    def __repr__(self):
        s = "FilesValidationResult:"
        s += f"\nfilestructure_errors = {self.filestructure_errors}"
        s += f"\ndataframe_errors = {self.dataframe_errors}"
        return s

    @property
    def total_errors(self) -> int:
        return len(self.filestructure_errors) + sum(map(len, self.dataframe_errors.values()))


class FilesValidator(Validator):

    def __init__(
        self,
        merged_df: pd.DataFrame,
        filesystem: FileSystem,
        config: FilesDatasetConfig,
        columns_to_check: List[str]
    ):
        self.merged_df = merged_df
        self.filesystem = filesystem
        self.config = config
        self.columns_to_check = columns_to_check

    def _validate_filestructure(self) -> List[FileStructureError]:
        errors = []

        for datatype in self.config.datatypes:
            if isinstance(datatype, FileDataType):
                has_duplicates = self.merged_df[datatype.modality.path_column].str.split('/').str[-1].duplicated().any()
                if has_duplicates:
                    errors.append(IsNotKeyError(datatype.user_path_column_name))

        return errors

    def _validate_df(self, df: pd.DataFrame) -> (List[DataFrameError], List[FileStructureError]):
        dataframe_errors = []

        # validate dataframe
        missed_columns = set(self.columns_to_check).difference(set(df.columns))
        if len(missed_columns) > 0:
            dataframe_errors.append(
                MissedColumnsError(self.config.table_path, list(missed_columns))
            )
        for datatype in self.config.datatypes:
            if isinstance(datatype, FileDataType):
                filepaths = df[datatype.user_path_column_name]
                has_duplicates = filepaths.duplicated().any()
                if has_duplicates:
                    dataframe_errors.append(
                        DuplicatedValuesError(self.config.table_path, datatype.user_basename_column_name)
                    )

        return dataframe_errors

    def validate(
        self,
        validate_filestructure: bool = True,
        workers: int = 4,
        pbar: bool = True
    ) -> FilesValidationResult:
        filestructure_errors = []
        dataframe2errors = {}

        if validate_filestructure:
            filestructure_errors.extend(self._validate_filestructure())

        df = self.filesystem.read_dataframe(self.config.table_path)
        dataframe_errors = self._validate_df(df)
        if len(dataframe_errors) > 0:
            dataframe2errors[self.config.table_path] = dataframe_errors

        return FilesValidationResult(filestructure_errors, dataframe2errors)
