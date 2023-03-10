from typing import Optional, Dict, Tuple
import os
import io
import tarfile
import traceback
import pandas as pd

from DPF.filesystems.filesystem import FileSystem
from .filewriter import FileWriter


class ShardsFileWriter(FileWriter):
    """
    ShardsFileWriter
    """

    def __init__(
        self,
        filesystem: FileSystem,
        destination_dir: str,
        max_files_in_shard: Optional[int] = 1000,
        image_ext: Optional[str] = None,
        datafiles_ext: Optional[str] = 'csv',
        archive_ext: Optional[str] = 'tar',
    ) -> None:
        self.filesystem = filesystem
        self.destination_dir = destination_dir
        self.max_files_in_shard = max_files_in_shard
        self.image_ext = '.'+image_ext.lstrip('.') if image_ext is not None else None
        self.datafiles_ext = '.'+datafiles_ext.lstrip('.')
        self.archive_ext = '.'+archive_ext.lstrip('.')

        self.df_raw = []
        self.tar_bytes = io.BytesIO()
        self.tar = None
        self.last_file_index = self._init_writer_from_last_uploaded_file()

    def save_file(
        self,
        file_bytes: bytes,
        image_ext: Optional[str] = None,
        file_data: Optional[Dict[str, str]] = None
    ) -> None:
        # check tar
        path_to_tar = os.path.join(self.destination_dir, self._calculate_current_tarname())
        if self.tar is None:
            self.tar = tarfile.open(mode='w', fileobj=self.tar_bytes)

        # writing to file
        filename = self._calculate_current_filename(image_ext=image_ext)
        img_tar_info, fp = self._prepare_image_for_tar_format(file_bytes, filename)
        self.tar.addfile(img_tar_info, fp)

        save_data = {
            "image_name": filename,
        }
        if file_data is not None:
            save_data.update(file_data)

        self.df_raw.append(save_data)
        self._try_close_batch()

    @staticmethod
    def _prepare_image_for_tar_format(file_bytes: bytes,
                                      filename: str) -> Tuple[tarfile.TarInfo, io.BytesIO]:
        fp = io.BytesIO(file_bytes)
        img_tar_info = tarfile.TarInfo(name=filename)
        img_tar_info.size = len(fp.getvalue())
        return img_tar_info, fp

    def __enter__(self) -> "FileWriter":
        return self

    def __exit__(
        self, exception_type, exception_value: Optional[Exception], exception_traceback: traceback
    ) -> None:
        if len(self.df_raw) != 0:
            self._flush(self._calculate_current_tarname())
        self.last_file_index = 0

    def _download_to_fileobj(self, s3_path: str) -> io.BytesIO:
        # todo move method to CloudS3
        data = io.BytesIO()
        self.connector.client.download_fileobj(Bucket=self.connector.bucket,
                                               Key=s3_path, Fileobj=data)
        data.seek(0)
        return data

    def _init_writer_from_last_uploaded_file(self) -> int:
        self.filesystem.mkdir(self.destination_dir)
        list_csv = [
            int(os.path.basename(filename[:-len(self.datafiles_ext)]))
            for filename in self.filesystem.listdir(self.destination_dir)
            if filename.endswith('.csv')
        ]
        if len(list_csv) < 1:
            return 0

        last_csv = str(sorted(list_csv)[-1])
        self.df_raw = self.filesystem.read_dataframe(
            os.path.join(self.destination_dir, last_csv + self.datafiles_ext)
        ).to_dict("records")
        #
        self.tar_bytes = self.filesystem.read_file(os.path.join(self.destination_dir,
                                                                last_csv + self.archive_ext),
                                                   binary=True)
        self.tar = tarfile.open(mode='a', fileobj=self.tar_bytes)
        #
        list_files = [
            int(os.path.splitext(data['image_name'])[0])
            for data in self.df_raw
        ]
        last_file = sorted(list_files)[-1]

        return last_file + 1

    def _calculate_current_tarname(self) -> str:
        return str(self.last_file_index // self.max_files_in_shard)+self.archive_ext

    def _calculate_current_filename(self, image_ext: str) -> str:
        if image_ext is None:
            return f"{self.last_file_index}{self.image_ext}"
        else:
            image_ext = image_ext.lstrip('.')
            return f"{self.last_file_index}.{image_ext}"

    def _try_close_batch(self) -> None:
        old_tarname = self._calculate_current_tarname()
        self.last_file_index += 1
        new_tarname = self._calculate_current_tarname()
        if old_tarname != new_tarname:
            self._flush(old_tarname)

    def _flush_and_upload_datafile(self, filename: str) -> None:
        df_to_save = pd.DataFrame(self.df_raw)
        path_to_csv_file = os.path.join(self.destination_dir, filename)
        self.filesystem.save_dataframe(df_to_save, path_to_csv_file, index=False)
        self.df_raw = []

    def _flush_and_upload_tar(self, filename: str) -> None:
        self.tar.close()
        self.tar_bytes.seek(0)
        self.filesystem.save_file(self.tar_bytes, os.path.join(self.destination_dir, filename),
                                  binary=True)
        self.tar = None
        self.tar_bytes = io.BytesIO()

    def _flush(self, tarname: str) -> None:
        self._flush_and_upload_datafile(tarname[:-4]+self.datafiles_ext)
        self._flush_and_upload_tar(tarname)
