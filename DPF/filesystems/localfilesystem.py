import os
import io
from typing import Union, List, Optional, Tuple, Iterable

from .filesystem import FileSystem


class LocalFileSystem(FileSystem):
    """
    Class that wraps interaction with local filesystem.
    """

    def __init__(self):
        super(LocalFileSystem).__init__()

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

    def mkdir(self, folder_path: str) -> None:
        folder_path = folder_path.rstrip("/") + "/"
        os.makedirs(folder_path, exist_ok=True)

    def walk(self, folder_path: str) -> Iterable[Tuple[str, List[str], List[str]]]:
        yield from os.walk(folder_path)
