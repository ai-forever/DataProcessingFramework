import os

import pandas as pd

from DPF.configs import ShardedFilesDatasetConfig
from DPF.datatypes import ShardedDataType
from DPF.filesystems import FileSystem
from DPF.validators.errors import (
    DataFrameErrorType,
    FileStructureErrorType,
    MissingValueError,
    NoSuchFileError,
)
from DPF.validators.format_validators.sharded_validator import ShardedValidator


class ShardedFilesValidator(ShardedValidator):

    def __init__(
        self,
        merged_df: pd.DataFrame,
        filesystem: FileSystem,
        config: ShardedFilesDatasetConfig,
        columns_to_check: list[str]
    ):
        super().__init__(merged_df, filesystem, config, columns_to_check)

    def _validate_files(self, filepaths: list[str]) -> list[FileStructureErrorType]:
        datafiles_ext = '.' + self.config.datafiles_ext.lstrip('.')
        datafiles_set = {f for f in filepaths if f.endswith(datafiles_ext)}
        folders_set = {f.rstrip('/') for f in filepaths if f not in datafiles_set}

        errors: list[FileStructureErrorType] = []
        for datafile in datafiles_set:
            folder_path = datafile.replace(datafiles_ext, '')
            if folder_path not in folders_set:
                errors.append(NoSuchFileError(folder_path))

        for folder_path in folders_set:
            datafile_path = folder_path+datafiles_ext
            if datafile_path not in datafiles_set:
                errors.append(NoSuchFileError(datafile_path))
        return errors

    def _validate_shard_files(
        self,
        dataframe_path: str,
        df: pd.DataFrame
    ) -> tuple[list[FileStructureErrorType], list[DataFrameErrorType]]:
        errors: list[FileStructureErrorType] = []
        errors_df: list[DataFrameErrorType] = []
        folder_path = dataframe_path.replace('.'+self.config.datafiles_ext.lstrip('.'), '')

        filenames_in_folder_set = set(self.filesystem.listdir(folder_path, filenames_only=True))

        filename_columns = []
        for datatype in self.config.datatypes:
            if isinstance(datatype, ShardedDataType):
                filename_columns.append(datatype.user_basename_column_name)

        for filename_col in filename_columns:
            filenames_in_table = df[filename_col].tolist()
            filenames_in_table_set = set(filenames_in_table)

            files_in_table_but_not_in_tar = filenames_in_table_set.difference(
                filenames_in_folder_set
            )
            files_in_tar_but_not_in_csv = filenames_in_folder_set.difference(
                filenames_in_table_set
            )

            if len(files_in_tar_but_not_in_csv) > 0:
                for file in files_in_table_but_not_in_tar:
                    errors.append(NoSuchFileError(os.path.join(folder_path, file)))

            if len(files_in_table_but_not_in_tar) > 0:
                for file in files_in_table_but_not_in_tar:
                    errors_df.append(MissingValueError(dataframe_path, filename_col, file))

        return errors, errors_df
