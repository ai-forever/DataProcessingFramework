import io
from typing import Union

from fsconnectors import S3Connector as S3Client

from .connector import Connector


class S3Connector(Connector):
    """
    Class that wraps interaction with S3.
    """

    def __init__(self, key: str, secret: str, endpoint_url: str):
        """
        Parameters
        ----------
        key: str
            Access key to s3 storage
        secret: str
            Secret key to s3 storage
        endpoint_url: str
            Endpoint for s3 storage
        """
        self.endpoint_url = endpoint_url
        self.key = key
        self.secret = secret
        self.s3client = S3Client(
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.key,
            aws_secret_access_key=self.secret
        )

    @staticmethod
    def _preprocess_filepath(path: str) -> str:
        return path[5:]

    def read_file(self, filepath: str, binary: bool) -> io.BytesIO:
        mode = "rb" if binary else "rt"
        with self.s3client.open(self._preprocess_filepath(filepath), mode=mode) as f:
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

        with self.s3client.open(self._preprocess_filepath(filepath), mode=mode) as f:
            if isinstance(data, io.BytesIO):
                data.seek(0)
                f.write(data.read())
            else:
                f.write(data)

    def listdir(self, folder_path: str) -> list[str]:
        folder_path = self._preprocess_filepath(folder_path).rstrip("/") + "/"  # noqa
        files: list[str] = self.s3client.listdir(folder_path)
        if folder_path in files:
            files.remove(folder_path)  # remove parent dir
        files = ["s3://" + folder_path + f for f in files]
        return files

    def mkdir(self, folder_path: str) -> None:
        folder_path = self._preprocess_filepath(folder_path).rstrip("/") + "/"
        # for some reason doesn't create directory
        # but it's ok because directories being created automatically when upload files
        self.s3client.mkdir(folder_path)

    def join(self, *args: str) -> str:
        path = ''
        for arg in args:
            if arg.endswith('/'):
                path += arg
            else:
                path += arg+'/'
        return path[:-1]
