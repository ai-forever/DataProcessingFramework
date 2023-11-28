import os
import uuid
from typing import Optional, Dict, List, Tuple
import traceback
import pandas as pd

from DPF.modalities import MODALITIES
from DPF.filesystems.filesystem import FileSystem
from .filewriter import ABSWriter
from .utils import rename_dict_keys


class ShardedFilesWriter(ABSWriter):
    """
    RawFileWriter
    """

    def __init__(
        self,
        filesystem: FileSystem,
        destination_dir: str,
        keys_mapping: Optional[Dict[str, str]] = None,
        max_files_in_shard: int = 1000,
        datafiles_ext: str = "csv",
        filenaming: str = "counter"
    ) -> None:
        self.filesystem = filesystem
        self.destination_dir = destination_dir
        self.keys_mapping = keys_mapping
        self.max_files_in_shard = max_files_in_shard
        self.datafiles_ext = "." + datafiles_ext.lstrip(".")
        self.filenaming = filenaming
        assert self.filenaming in ["counter", "uuid"], "Invalid files naming"

        self.df_raw = []
        self.shard_index, self.last_file_index = self._init_writer_from_last_uploaded_file()
        self.last_path_to_dir = None

    def save_sample(
        self,
        modality2sample_data: Dict[str, Tuple[str, bytes]],
        table_data: Dict[str, str] = {},
    ) -> None:
        # creating directory
        path_to_dir = self.filesystem.join(
            self.destination_dir, self._calculate_current_dirname()
        )
        if (self.last_path_to_dir is None) or (self.last_path_to_dir != path_to_dir):
            self.last_path_to_dir = path_to_dir
            self.filesystem.mkdir(path_to_dir)

        # writing to file
        for modality, (extension, file_bytes) in modality2sample_data.items():
            filename = self.get_current_filename(extension)
            table_data[MODALITIES[modality].sharded_file_name_column] = filename
            path_to_file = self.filesystem.join(path_to_dir, filename)
            self.filesystem.save_file(file_bytes, path_to_file, binary=True)

        if self.keys_mapping:
            table_data = rename_dict_keys(table_data, self.keys_mapping)

        self.df_raw.append(table_data)
        self._try_close_batch()

    def __enter__(self) -> "FileWriter":
        return self

    def __exit__(
        self,
        exception_type: Optional[type],
        exception_value: Optional[Exception],
        exception_traceback: traceback,
    ) -> None:
        if len(self.df_raw) != 0:
            self._flush(self._calculate_current_dirname())
        self.last_file_index = 0

    def _init_writer_from_last_uploaded_file(self) -> (int, int):
        self.filesystem.mkdir(self.destination_dir)
        list_dirs = [
            int(os.path.basename(filename[: -len(self.datafiles_ext)]))
            for filename in self.filesystem.listdir(self.destination_dir)
            if filename.endswith(self.datafiles_ext)
        ]
        if len(list_dirs) == 0:
            return 0, 0

        last_dir = str(sorted(list_dirs)[-1])
        dir_path = self.filesystem.join(self.destination_dir, last_dir)

        filenames = self.filesystem.listdir(dir_path, filenames_only=True)
        names = [os.path.splitext(f)[0] for f in filenames if not f.startswith('.')]
        if len(names) == 0:
            return int(last_dir), int(last_dir)*self.max_files_in_shard

        if self.filenaming == "counter":
            if all([name.isdigit() for name in names]):
                index = int(sorted(names)[-1]) + 1
            else:
                raise ValueError(f'Could read index from {dir_path}. Check filenames')
        else:
            index = len(names)
        return int(last_dir), index

    def get_current_filename(self, extension: str) -> str:
        extension = extension.lstrip('.')
        if self.filenaming == "counter":
            return f"{self.last_file_index}.{extension}"
        elif self.filenaming == "uuid":
            return f"{uuid.uuid4().hex}.{extension}"

    def _calculate_current_dirname(self) -> str:
        return str(self.shard_index)

    def _try_close_batch(self) -> None:
        old_dirname = self._calculate_current_dirname()

        self.last_file_index += 1
        if self.last_file_index % self.max_files_in_shard == 0:
            self.shard_index += 1

        new_dirname = self._calculate_current_dirname()
        if old_dirname != new_dirname:
            self._flush(old_dirname)

    def _flush(self, dirname: str) -> None:
        if len(self.df_raw) > 0:
            df_to_save = pd.DataFrame(
                self.df_raw,
                columns=self._rearrange_cols(list(self.df_raw[0].keys()))
            )
            path_to_csv_file = self.filesystem.join(
                self.destination_dir, f"{dirname}{self.datafiles_ext}"
            )
            self.filesystem.save_dataframe(df_to_save, path_to_csv_file, index=False)
        self.df_raw = []

    def _rearrange_cols(self, columns: List[str]) -> List[str]:
        cols_first = []
        for modality in MODALITIES.values():
            if modality.sharded_file_name_column:
                cols_first.append(modality.sharded_file_name_column)
        for modality in MODALITIES.values():
            if modality.path_column:
                cols_first.append(modality.path_column)
        for modality in MODALITIES.values():
            if modality.column:
                cols_first.append(modality.column)

        cols_first = [col for col in cols_first if col in columns]
        cols_end = [col for col in columns if col not in cols_first]
        return cols_first+cols_end
