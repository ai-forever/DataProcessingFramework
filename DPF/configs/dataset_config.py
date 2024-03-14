from abc import ABC, abstractmethod

from DPF.datatypes import DataType
from DPF.modalities import ModalityName


class DatasetConfig(ABC):
    """Config for a dataset"""

    def __init__(self, path: str):
        assert not path.endswith('/')
        self.path = path

    @property
    @abstractmethod
    def datatypes(self) -> list[DataType]:
        """List of datatypes of a dataset"""
        pass

    @property
    @abstractmethod
    def modality2datatype(self) -> dict[ModalityName, DataType]:
        """Mapping modality to its datatype"""
        pass

    @property
    @abstractmethod
    def user_column2default_column(self) -> dict[str, str]:
        pass

    @property
    def user_column_names(self) -> list[str]:
        return list(self.user_column2default_column.keys())

    @property
    def user_columns_to_rename(self) -> dict[str, str]:
        columns_to_rename = {}
        for k, v in self.user_column2default_column.items():
            if k != v:
                columns_to_rename[k] = v
        return columns_to_rename
