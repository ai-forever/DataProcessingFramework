from typing import Optional, Dict, Tuple, List
import os
import io
import tarfile
import traceback
import pandas as pd

from DPF.filesystems.filesystem import FileSystem
from .filewriter import ABSWriter
from DPF.modalities import MODALITIES


class ShardsWriter(ABSWriter):
    """
    ShardsFileWriter
    """

    def __init__(
        self,
        filesystem: FileSystem,
        destination_dir: str,
        max_files_in_shard: Optional[int] = 1000,
        datafiles_ext: Optional[str] = "csv",
        archives_ext: Optional[str] = "tar",
    ) -> None:
        self.filesystem = filesystem
        self.destination_dir = destination_dir
        self.max_files_in_shard = max_files_in_shard
        self.datafiles_ext = "." + datafiles_ext.lstrip(".")
        self.archives_ext = "." + archives_ext.lstrip(".")

        self.df_raw = []
        self.tar_bytes = io.BytesIO()
        self.tar = None
        self.last_file_index = self._init_writer_from_last_uploaded_file()

    def save_sample(
        self,
        modality2sample_data: Dict[str, Tuple[str, bytes]],
        table_data: Dict[str, str] = {},
    ) -> None:
        # check tar
        if self.tar is None:
            self.tar = tarfile.open(mode="w", fileobj=self.tar_bytes)

        # writing to file
        for modality, (extension, file_bytes) in modality2sample_data.items():
            filename = self.get_current_filename(extension)
            table_data[MODALITIES[modality].sharded_file_name_column] = filename
            img_tar_info, fp = self._prepare_image_for_tar_format(file_bytes, filename)
            self.tar.addfile(img_tar_info, fp)

        self.df_raw.append(table_data)
        self._try_close_batch()

    @staticmethod
    def _prepare_image_for_tar_format(
        file_bytes: bytes, filename: str
    ) -> Tuple[tarfile.TarInfo, io.BytesIO]:
        fp = io.BytesIO(file_bytes)
        img_tar_info = tarfile.TarInfo(name=filename)
        img_tar_info.size = len(fp.getvalue())
        return img_tar_info, fp

    def __enter__(self) -> "FileWriter":
        return self

    def __exit__(
        self,
        exception_type,
        exception_value: Optional[Exception],
        exception_traceback: traceback,
    ) -> None:
        if len(self.df_raw) != 0:
            self._flush(self._calculate_current_tarname())
        self.last_file_index = 0

    def _init_writer_from_last_uploaded_file(self) -> int:
        self.filesystem.mkdir(self.destination_dir)
        list_csv = [
            int(os.path.basename(filename[: -len(self.datafiles_ext)]))
            for filename in self.filesystem.listdir(self.destination_dir)
            if filename.endswith(".csv")
        ]
        if len(list_csv) < 1:
            return 0

        last_csv = str(sorted(list_csv)[-1])
        self.df_raw = self.filesystem.read_dataframe(
            os.path.join(self.destination_dir, last_csv + self.datafiles_ext)
        ).to_dict("records")
        #
        self.tar_bytes = self.filesystem.read_file(
            os.path.join(self.destination_dir, last_csv + self.archives_ext), binary=True
        )
        self.tar = tarfile.open(mode="a", fileobj=self.tar_bytes)
        #
        list_files = [
            int(os.path.splitext(data["image_name"])[0]) for data in self.df_raw
        ]
        last_file = sorted(list_files)[-1]

        return last_file + 1

    def _calculate_current_tarname(self) -> str:
        return str(self.last_file_index // self.max_files_in_shard) + self.archives_ext

    def get_current_filename(self, extension: str) -> str:
        extension = extension.lstrip('.')
        return f"{self.last_file_index}.{extension}"

    def _try_close_batch(self) -> None:
        old_tarname = self._calculate_current_tarname()
        self.last_file_index += 1
        new_tarname = self._calculate_current_tarname()
        if old_tarname != new_tarname:
            self._flush(old_tarname)

    def _flush_and_upload_datafile(self, filename: str) -> None:
        df_to_save = pd.DataFrame(
            self.df_raw,
            columns=self._rearrange_cols(list(self.df_raw[0].keys()))
        )
        path_to_csv_file = os.path.join(self.destination_dir, filename)
        self.filesystem.save_dataframe(df_to_save, path_to_csv_file, index=False)
        self.df_raw = []

    def _flush_and_upload_tar(self, filename: str) -> None:
        self.tar.close()
        self.tar_bytes.seek(0)
        self.filesystem.save_file(
            self.tar_bytes, os.path.join(self.destination_dir, filename), binary=True
        )
        self.tar = None
        self.tar_bytes = io.BytesIO()

    def _flush(self, tarname: str) -> None:
        self._flush_and_upload_datafile(tarname[:-4] + self.datafiles_ext)
        self._flush_and_upload_tar(tarname)

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
