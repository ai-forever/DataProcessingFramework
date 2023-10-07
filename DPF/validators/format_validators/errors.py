from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Dict


class DataFrameError:

    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class MissedColumnsError(DataFrameError):
    path: str
    missed_columns: List[str]

    def __repr__(self) -> str:
        return f"Dataframe {self.path} has missed columns: {self.missed_columns}"


@dataclass
class DuplicatedValuesError(DataFrameError):
    path: str
    column_name: str

    def __repr__(self) -> str:
        return f'Dataframe {self.path} has duplicated values in column "{self.column_name}"'


@dataclass
class MissingValueError(DataFrameError):
    path: str
    column_name: str
    value: str

    def __repr__(self) -> str:
        return f'Dataframe {self.path} missing value "{self.value}" in column "{self.column_name}"'


class FileStructureError(ABC):

    @abstractmethod
    def __repr__(self) -> str:
        pass


@dataclass
class NoSuchFileError(FileStructureError):
    path: str

    def __repr__(self) -> str:
        return f'Expected file "{self.path}"'


@dataclass
class FileNotInDataError(FileStructureError):
    path: str

    def __repr__(self) -> str:
        return f'Expected file "{self.path}" to be in datafile'


@dataclass
class IsNotKeyError(FileStructureError):
    key_column: str

    def __repr__(self) -> str:
        return f"Expected column {self.key_column} to be absolute key"
