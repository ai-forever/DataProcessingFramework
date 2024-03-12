import io
import os
import tarfile
from abc import ABC, abstractmethod
from typing import Any, Union

import pandas as pd

from DPF.connectors.errors import UnknownFileFormatException


class Connector(ABC):
    """
    Abstract class for all filesystems
    """

    @abstractmethod
    def read_file(self, filepath: str, binary: bool) -> io.BytesIO:
        """
        Reads file content

        Parameters
        ----------
        filepath: str
            Path to file
        binary: bool
            Read file in binary mode or in text mode

        Returns
        -------
        io.BytesIO | str
            io.BytesIO object if binary, string otherwise
        """
        pass

    @abstractmethod
    def save_file(
        self, data: Union[str, bytes, io.BytesIO], filepath: str, binary: bool
    ) -> None:
        """
        Saves data to file

        Parameters
        ----------
        data: Union[str, bytes, io.BytesIO]
            Data to save
        filepath: str
            Path to file
        binary: bool
            Write file in binary mode or in text mode
        """
        pass

    def read_tar(self, filepath: str) -> tarfile.TarFile:
        """
        Reads a tar file like tarfile.open

        Parameters
        ----------
        filepath: str
            Path to file
        """
        tar_bytes = self.read_file(filepath, binary=True)
        return tarfile.open(fileobj=tar_bytes, mode="r")

    def read_dataframe(self, filepath: str, **kwargs: Any) -> pd.DataFrame:
        """
        Reads dataframe

        Parameters
        ----------
        filepath: str
            Path to dataframe file (csv, parquet, etc.)
        **kwargs
            kwargs for pandas read function

        Returns
        -------
        pd.DataFrame
            Pandas dataframe
        """
        filetype = os.path.splitext(filepath)[1]  # get extension
        filetype = filetype.lstrip(".")
        data = self.read_file(filepath, binary=True)
        if filetype == "csv":
            return pd.read_csv(data, **kwargs)
        if filetype == "parquet":
            return pd.read_parquet(data, **kwargs)
        else:
            raise UnknownFileFormatException(f"Unknown file format: {filetype}")

    def save_dataframe(self, df: pd.DataFrame, filepath: str, **kwargs: Any) -> None:
        """
        Saves dataframe

        Parameters
        ----------
        df: pd.DataFrame
            Pandas dataframe to save
        filepath: str
            Path to file
        **kwargs
            kwargs for pandas save function
        """
        filetype = os.path.splitext(filepath)[1]  # get extension
        filetype = filetype.lstrip(".")
        data = io.BytesIO()
        if filetype == "csv":
            df.to_csv(data, **kwargs)
        elif filetype == "parquet":
            df.to_parquet(data, **kwargs)
        else:
            raise UnknownFileFormatException(f"Unknown file format: {filetype}")
        self.save_file(data=data, filepath=filepath, binary=True)

    @abstractmethod
    def listdir(self, folder_path: str) -> list[str]:
        """
        Returns the contents of folder

        Parameters
        ----------
        folder_path: str
            Path to folder

        Returns
        -------
        List[str]
            List of filepaths (filenames if filenames_only)
        """
        pass

    @abstractmethod
    def mkdir(self, folder_path: str) -> None:
        """
        Creates a directory

        Parameters
        ----------
        folder_path: str
            Path to folder to create
        """
        pass

    @abstractmethod
    def join(self, *args: str) -> str:
        """
        Join paths like os.path.join

        Parameters
        ----------
        *args: str
            List of strings - subfolders, etc

        Returns
        -------
        str
            Joined full path
        """
        pass
