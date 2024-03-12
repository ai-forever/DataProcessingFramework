
import pandas as pd

from DPF.configs import FilesDatasetConfig
from DPF.connectors import Connector
from DPF.datatypes import FileDataType
from DPF.validators import ValidationResult, Validator
from DPF.validators.errors import (
    DataFrameErrorType,
    DuplicatedValuesError,
    FileStructureErrorType,
    IsNotKeyError,
    MissedColumnsError,
)


class FilesValidator(Validator):

    def __init__(
        self,
        merged_df: pd.DataFrame,
        connector: Connector,
        config: FilesDatasetConfig,
        columns_to_check: list[str]
    ):
        self.merged_df = merged_df
        self.connector = connector
        self.config = config
        self.columns_to_check = columns_to_check

    def _validate_filestructure(self) -> list[FileStructureErrorType]:
        errors: list[FileStructureErrorType] = []

        for datatype in self.config.datatypes:
            if isinstance(datatype, FileDataType):
                has_duplicates = self.merged_df[datatype.modality.path_column].str.split('/').str[-1].duplicated().any()
                if has_duplicates:
                    errors.append(IsNotKeyError(datatype.user_path_column_name))

        return errors

    def _validate_df(self, df: pd.DataFrame) -> list[DataFrameErrorType]:
        dataframe_errors: list[DataFrameErrorType] = []

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
                        DuplicatedValuesError(self.config.table_path, datatype.user_path_column_name)
                    )

        return dataframe_errors

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_metadata: bool = True,
        workers: int = 4,
        pbar: bool = True
    ) -> ValidationResult:
        filestructure_errors: list[FileStructureErrorType] = []
        dataframe2errors: dict[str, list[DataFrameErrorType]] = {}

        if validate_filestructure:
            filestructure_errors.extend(self._validate_filestructure())
        if validate_metadata:
            df = self.connector.read_dataframe(self.config.table_path)
            dataframe_errors = self._validate_df(df)
            if len(dataframe_errors) > 0:
                dataframe2errors[self.config.table_path] = dataframe_errors

        return ValidationResult(filestructure_errors, dataframe2errors)
