from abc import ABC, abstractmethod
import os
import io
import tarfile
import datetime
from typing import Union, List, Dict, Optional, Tuple, Iterable
import pandas as pd


class FileData:
    """Class that represents a file with his metadata"""

    def __init__(
        self,
        path: str,
        type: str,
        last_modified: Optional[datetime.datetime] = None,
        file_size: Optional[int] = None
    ):
        self.path = path
        # TODO(review) - на вход, как и ранее, нужно уже подавать чистые пути, желательно избавиться от таких чисток
        self.name = os.path.basename(self.path.rstrip('/'))
        assert type in {'directory', 'file'}, \
            "param 'type' must be one of {'directory', 'file'}, got "+str(type)
        self.type = type
        self.last_modified = last_modified
        self.file_size = file_size

    def __repr__(self) -> str:
        return f'File(path="{self.path}, size={self.file_size}, last_modified={self.last_modified}")'


class FileSystem(ABC):
    """
    Abstract class for all filesystems
    """

    # TODO(review) - очень желательно не забывать ключ.слово pass в абстрактных методах (одно из правил чистого кода)
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

    def read_tar(self, filepath: str):
        """
        Reads a tar file like tarfile.open

        Parameters
        ----------
        filepath: str
            Path to file
        """
        tar_bytes = self.read_file(filepath, binary=True)
        return tarfile.open(fileobj=tar_bytes, mode="r")

    def read_dataframe(self, filepath: str, **kwargs) -> pd.DataFrame:
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
            # TODO(review) - лучше под эту ошибку завести кастомное исключение (UnknownFileFormatException, например)
            raise NotImplementedError(f"Unknown file format: {filetype}")

    def save_dataframe(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
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
            raise NotImplementedError(f"Unknown file format: {filetype}")
        self.save_file(data=data, filepath=filepath, binary=True)

    # TODO(review) - нет смысла листить директорию, вернув только имена файлов, это разные операции. На выходе из листинга должны быть абсолютные пути до файлов
    # Логику с именами файлов (без абсолютного пути) лучше вынести в отдельный метод
    @abstractmethod
    def listdir(self, folder_path: str, filenames_only: bool = False) -> List[str]:
        """
        Returns the contents of folder

        Parameters
        ----------
        folder_path: str
            Path to folder
        filenames_only: bool = False
            Returns only filenames if True

        Returns
        -------
        List[str]
            List of filepaths (filenames if filenames_only)
        """
        pass

    # TODO(review) - не совсем понятна логика листинга вместе с расширением файла, оно должно возвращаться всегда, иначе операцию листинга нельза назвать операцией листинга директории
    def listdir_with_ext(
        self, folder_path: str, ext: str, filenames_only: bool = False
    ) -> List[str]:
        """
        Returns all files in folder with provided extinsion

        Parameters
        ----------
        folder_path: str
            Path to folder
        ext: str
            Extension of files
        filenames_only: bool = False
            Returns only filenames if True

        Returns
        -------
        List[str]
            List of filepaths (filenames if filenames_only)
        """
        ext = "." + ext.lstrip(".")
        return [
            f
            for f in self.listdir(folder_path, filenames_only=filenames_only)
            if f.endswith(ext)
        ]

     # TODO(review) - третья логика листинга в классе, нужно это дело совместить в 1 метод, и параметризовать вывод (с помощью bool-значений на входе в метод)
    @abstractmethod
    def listdir_meta(self, folder_path: str) -> List[FileData]:
        """
        Returns the contents of folder with meta information (datetime created, etc)

        Parameters
        ----------
        folder_path: str
            Path to folder

        Returns
        -------
        List[FileData]
            List of FileData objects
        """
        pass

    # TODO(review) - метод с постоянным поведением, реализация должна быть одна, единственная и здесь
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
    def walk(self, folder_path: str) -> Iterable[Tuple[str, List[str], List[str]]]:
        """
        Recursively get contents of folder in os.walk style

        Parameters
        ----------
        folder_path: str
            Path to folder

        Returns
        -------
        Iterable[Tuple[str, List[str], List[str]]]
            Iterable oТf tuples with 3 elements: root, dirs, files
        """
        pass

    @abstractmethod
    def join(self, *args) -> str:
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
