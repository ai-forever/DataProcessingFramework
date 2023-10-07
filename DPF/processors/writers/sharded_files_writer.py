import os
from typing import Optional, Dict, List, Tuple
import traceback
import pandas as pd

from DPF.modalities import MODALITIES
from DPF.filesystems.filesystem import FileSystem
from .filewriter import ABSWriter


class ShardedFilesWriter(ABSWriter):
    """
    RawFileWriter
    """

    def __init__(
        self,
        filesystem: FileSystem,
        destination_dir: str,
        max_files_in_folder: Optional[int] = 1000,
        datafiles_ext: Optional[str] = "csv",
    ) -> None:
        self.filesystem = filesystem
        self.destination_dir = destination_dir
        self.max_files_in_folder = max_files_in_folder
        self.datafiles_ext = "." + datafiles_ext.lstrip(".")

        self.df_raw = []
        self.last_file_index = self._init_writer_from_last_uploaded_file()
        self.last_path_to_dir = None

    def save_sample(
        self,
        modality2sample_data: Dict[str, Tuple[str, bytes]],
        table_data: Dict[str, str] = {},
    ) -> None:
        # creating directory
        path_to_dir = os.path.join(
            self.destination_dir, self._calculate_current_dirname()
        )
        if (self.last_path_to_dir is None) or (self.last_path_to_dir != path_to_dir):
            self.last_path_to_dir = path_to_dir
            self.filesystem.mkdir(path_to_dir)

        # writing to file
        for modality, (extension, file_bytes) in modality2sample_data.items():
            filename = self.get_current_filename(extension)
            table_data[MODALITIES[modality].sharded_file_name_column] = filename
            path_to_file = os.path.join(path_to_dir, filename)
            self.filesystem.save_file(file_bytes, path_to_file, binary=True)

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

    def _init_writer_from_last_uploaded_file(self) -> int:
        self.filesystem.mkdir(self.destination_dir)
        list_dirs = [
            int(os.path.basename(filename[: -len(self.datafiles_ext)]))
            for filename in self.filesystem.listdir(self.destination_dir)
            if filename.endswith(self.datafiles_ext)
        ]
        if len(list_dirs) == 0:
            return 0

        last_dir = str(sorted(list_dirs)[-1])
        dir_path = os.path.join(self.destination_dir, last_dir)

        filenames = self.filesystem.listdir(dir_path, filenames_only=True)
        names = [os.path.splitext(f)[0] for f in filenames if not f.startswith('.')]
        if len(names) == 0:
            return int(last_dir)*self.max_files_in_folder

        if all([name.isdigit() for name in names]):
            index = int(sorted(names)[-1]) + 1
        else:
            raise ValueError(f'Could read index from {dir_path}. Check filenames')
        return index

    def get_current_filename(self, extension: str) -> str:
        extension = extension.lstrip('.')
        return f"{self.last_file_index}.{extension}"

    def _calculate_current_dirname(self) -> str:
        return str(self.last_file_index // self.max_files_in_folder)

    def _try_close_batch(self) -> None:
        old_dirname = self._calculate_current_dirname()
        self.last_file_index += 1
        new_dirname = self._calculate_current_dirname()
        if old_dirname != new_dirname:
            self._flush(old_dirname)

    def _flush(self, dirname: str) -> None:
        if len(self.df_raw) > 0:
            df_to_save = pd.DataFrame(self.df_raw)
            path_to_csv_file = os.path.join(
                self.destination_dir, f"{dirname}{self.datafiles_ext}"
            )
            self.filesystem.save_dataframe(df_to_save, path_to_csv_file, index=False)
        self.df_raw = []
