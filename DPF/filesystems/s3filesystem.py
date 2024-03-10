import io
import os
from collections.abc import Iterable
from typing import Optional, Union

import fsspec

from .filesystem import FileData, FileSystem


class S3FileSystem(FileSystem):
    """
    Class that wraps interaction with S3.
    """

    def __init__(self, key: str, secret: str, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.key = key
        self.secret = secret
        self.storage_options = {
            "anon": False,
            "key": self.key,
            "secret": self.secret,
            "client_kwargs": {"endpoint_url": self.endpoint_url},
        }

    def read_file(self, filepath: str, binary: bool) -> io.BytesIO:
        mode = "rb" if binary else "rt"
        with fsspec.open(
            filepath, s3=self.storage_options, mode=mode, skip_instance_cache=True
        ) as f:
            if mode == "rb":
                res = io.BytesIO(f.read())
                res.seek(0)
            else:
                res = f.read()
        return res

    def save_file(
        self, data: Union[str, bytes, io.BytesIO], filepath: str, binary: bool
    ) -> None:
        mode = "wb" if binary else "wt"

        with fsspec.open(
            f"simplecache::{filepath}", s3=self.storage_options, mode=mode
        ) as f:
            if isinstance(data, io.BytesIO):
                data.seek(0)
                f.write(data.read())
            else:
                f.write(data)

    def listdir(
        self, folder_path: str, filenames_only: Optional[bool] = False
    ) -> list[str]:
        folder_path = folder_path.lstrip("s3://").rstrip("/") + "/"  # noqa
        s3 = fsspec.filesystem("s3", **self.storage_options)
        files: list[str] = s3.ls(folder_path)
        if folder_path in files:
            files.remove(folder_path)  # remove parent dir
        if filenames_only:
            files = [os.path.basename(f) for f in files]
        else:
            files = ["s3://" + f for f in files]
        return files

    def listdir_meta(self, folder_path: str) -> list[FileData]:
        folder_path = folder_path.lstrip("s3://").rstrip("/") + "/"  # noqa
        s3 = fsspec.filesystem("s3", **self.storage_options)
        files_data = s3.ls(folder_path, detail=True)

        results = []
        for file_data in files_data:
            if file_data['Key'] == folder_path:
                continue
            path = "s3://"+file_data['Key']
            filetype = file_data['type']
            size = None
            last_modified = file_data.get('LastModified', None)
            if filetype == 'file':
                size = file_data.get('Size', None)
            results.append(FileData(path, filetype, last_modified, size))
        return results

    def mkdir(self, folder_path: str) -> None:
        folder_path = folder_path.rstrip("/") + "/"
        s3 = fsspec.filesystem("s3", **self.storage_options)
        # for some reason doesn't create directory
        # but it's ok because directories being created automatically when upload files
        s3.makedirs(folder_path, exist_ok=True)

    def walk(self, folder_path: str) -> Iterable[tuple[str, list[str], list[str]]]:
        fs = fsspec.filesystem("s3", **self.storage_options)

        yield from fs.walk(folder_path)

    def join(self, *args: str) -> str:
        path = ''
        for arg in args:
            if arg.endswith('/'):
                path += arg
            else:
                path += arg+'/'
        return path[:-1]
