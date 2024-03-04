import os
from typing import List

import pandas as pd

from DPF.datatypes import ShardedDataType
from DPF.validators.format_validators import (
    DataFrameError,
    FileStructureError,
    MissingValueError,
    NoSuchFileError,
    ShardedValidator,
)


class ShardsValidator(ShardedValidator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validate_files(self, filepaths: List[str]) -> List[FileStructureError]:
        datafiles_ext = '.' + self.config.datafiles_ext
        archives_ext = '.' + self.config.archives_ext
        datafiles_set = set([f for f in filepaths if f.endswith(datafiles_ext)])
        archives_set = set([f for f in filepaths if f.endswith(archives_ext)])

        errors = []
        for datafile in datafiles_set:
            archive_path = datafile.replace(datafiles_ext, archives_ext)
            if archive_path not in archives_set:
                errors.append(NoSuchFileError(archive_path))

        for archive in archives_set:
            datafile_path = archive.replace(archives_ext, datafiles_ext)
            if datafile_path not in datafiles_set:
                errors.append(NoSuchFileError(datafile_path))
        return errors

    def _validate_shard_files(
        self,
        dataframe_path: str,
        df: pd.DataFrame
    ) -> (List[FileStructureError], List[DataFrameError]):
        errors = []
        errors_df = []
        archive_path = dataframe_path.replace(self.config.datafiles_ext, self.config.archives_ext)

        tar = self.filesystem.read_tar(archive_path)
        filenames_in_tar = []
        for c, member in enumerate(tar):
            filenames_in_tar.append(member.name)
        tar.close()
        filenames_in_tar_set = set(filenames_in_tar)

        filename_columns = []
        for datatype in self.config.datatypes:
            if isinstance(datatype, ShardedDataType):
                filename_columns.append(datatype.user_basename_column_name)

        for filename_col in filename_columns:
            filenames_in_table = df[filename_col].tolist()
            filenames_in_table_set = set(filenames_in_table)

            files_in_table_but_not_in_tar = filenames_in_table_set.difference(
                filenames_in_tar_set
            )
            files_in_tar_but_not_in_csv = filenames_in_tar_set.difference(
                filenames_in_table_set
            )

            if len(files_in_tar_but_not_in_csv) > 0:
                for file in files_in_table_but_not_in_tar:
                    errors.append(NoSuchFileError(os.path.join(archive_path, file)))

            if len(files_in_table_but_not_in_tar) > 0:
                for file in files_in_table_but_not_in_tar:
                    errors_df.append(MissingValueError(dataframe_path, filename_col, file))

        return errors, errors_df
