import os
from typing import Optional, Dict
import traceback

import pandas as pd

from DPF.filesystems.filesystem import FileSystem
from .filewriter import FileWriter


class RawFileWriter(FileWriter):

    def __init__(
        self,
        filesystem: FileSystem,
        destination_dir: str,
        max_files_in_folder: Optional[int] = 1000,
        image_ext: Optional[str] = None,
        datafiles_ext: Optional[str] = 'csv',
    ) -> None:
        self.filesystem = filesystem
        self.destination_dir = destination_dir
        self.max_files_in_folder = max_files_in_folder
        self.image_ext = '.'+image_ext.lstrip('.') if image_ext is not None else None
        self.datafiles_ext = '.'+datafiles_ext.lstrip('.')

        self.df_raw = []
        self.last_file_index = self._init_writer_from_last_uploaded_file()
        self.last_path_to_dir = None

    def save_file(
        self,
        file_bytes: bytes,
        image_ext: Optional[str] = None,
        file_data: Optional[Dict[str, str]] = None
    ) -> None:
        # creating directory
        path_to_dir = os.path.join(self.destination_dir, self._calculate_current_dirname())
        if (self.last_path_to_dir is None) or (self.last_path_to_dir != path_to_dir):
            self.last_path_to_dir = path_to_dir
            self.filesystem.mkdir(path_to_dir)

        # writing to file
        filename = self._calculate_current_filename(image_ext=image_ext)
        path_to_file = os.path.join(path_to_dir, filename)
        self.filesystem.save_file(file_bytes, path_to_file, binary=True)

        save_data = {
            "image_name": filename,
        }
        if file_data is not None:
            save_data.update(file_data)

        self.df_raw.append(save_data)
        self._try_close_batch()

    def __enter__(self) -> "FileWriter":
        return self

    def __exit__(
        self, exception_type: Optional[type],
        exception_value: Optional[Exception],
        exception_traceback: traceback
    ) -> None:
        if len(self.df_raw) != 0:
            self._flush(self._calculate_current_dirname())
        self.last_file_index = 0

    def _init_writer_from_last_uploaded_file(self) -> int:
        self.filesystem.mkdir(self.destination_dir)
        list_dirs = [
            int(os.path.basename(filename[:-len(self.datafiles_ext)]))
            for filename in self.filesystem.listdir(self.destination_dir)
            if filename.endswith('.csv')
        ]
        if len(list_dirs) < 1:
            return 0

        last_dir = str(sorted(list_dirs)[-1])
        self.df_raw = self.filesystem.read_dataframe(
            os.path.join(self.destination_dir, last_dir + self.datafiles_ext)
        ).to_dict("records")
        list_files = [
            int(os.path.splitext(data['image_name'])[0])
            for data in self.df_raw
        ]
        last_file = sorted(list_files)[-1]

        return last_file + 1

    def _calculate_current_dirname(self) -> str:
        return str(self.last_file_index // self.max_files_in_folder)

    def _calculate_current_filename(self, image_ext: str) -> str:
        if image_ext is None:
            return f"{self.last_file_index}{self.image_ext}"
        else:
            image_ext = image_ext.lstrip('.')
            return f"{self.last_file_index}.{image_ext}"

    def _try_close_batch(self) -> None:
        old_dirname = self._calculate_current_dirname()
        self.last_file_index += 1
        new_dirname = self._calculate_current_dirname()
        if old_dirname != new_dirname:
            self._flush(old_dirname)

    def _flush(self, dirname: str) -> None:
        df_to_save = pd.DataFrame(self.df_raw)
        path_to_csv_file = os.path.join(self.destination_dir, f"{dirname}{self.datafiles_ext}")
        self.filesystem.save_dataframe(df_to_save, path_to_csv_file, index=False)
        self.df_raw = []
