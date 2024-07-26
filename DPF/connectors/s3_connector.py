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
        if '.tar' in filepath and '?tar_offset=' in filepath and '?size=' in filepath:
            filepath = self._preprocess_filepath(filepath)
            offset = filepath.split('?tar_offset=')[1].split('?size=')[0]
            size = filepath.split('?size=')[1]
            filepath = filepath.split('?')[0]
            offset = int(offset)
            size = int(size)
            s3 = self.s3client._get_client()
            range_header = "bytes=%d-%d" % (offset, offset + size - 1)
            bucket_name = filepath.split('/')[0]
            tar_key = filepath.replace(bucket_name, '')[1:]
            video_obj = s3.get_object(Bucket=bucket_name, Key=tar_key, Range=range_header)
            res = video_obj["Body"].read()
            if mode == "rb":
                res = io.BytesIO(res)
                res.seek(0)
        else:
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