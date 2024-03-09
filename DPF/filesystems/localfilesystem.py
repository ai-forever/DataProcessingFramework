import io
import os
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union

from .filesystem import FileData, FileSystem


class LocalFileSystem(FileSystem):
    """
    Class that wraps interaction with local filesystem.
    """

    # TODO(review) - дубль кода с filesystem.py
    def read_file(self, filepath: str, binary: bool) -> io.BytesIO:
        mode = "rb" if binary else "rt"
        with open(filepath, mode) as f:
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

        with open(filepath, mode) as f:
            if isinstance(data, io.BytesIO):
                data.seek(0)
                f.write(data.read())
            else:
                f.write(data)

    def listdir(
        self, folder_path: str, filenames_only: Optional[bool] = False
    ) -> List[str]:
        folder_path = folder_path.rstrip("/") + "/"
        files = os.listdir(folder_path)
        if not filenames_only:
            files = [folder_path + f for f in files]
        return files

    def listdir_meta(self, folder_path: str) -> List[FileData]:
        folder_path = folder_path.rstrip("/") + "/"
        results = []
        for fd in os.scandir(folder_path):
            path = fd.path
            type_ = 'directory' if fd.is_dir() else 'file'
            stats = fd.stat()
            size = stats.st_size
            last_modified = datetime.fromtimestamp(stats.st_mtime)
            results.append(FileData(path, type_, last_modified, size))
        return results

    def mkdir(self, folder_path: str) -> None:
        folder_path = folder_path.rstrip("/") + "/"
        os.makedirs(folder_path, exist_ok=True)

    def walk(self, folder_path: str) -> Iterable[Tuple[str, List[str], List[str]]]:
        yield from os.walk(folder_path)

    def join(self, *args: str) -> str:
        return os.path.join(*args)
