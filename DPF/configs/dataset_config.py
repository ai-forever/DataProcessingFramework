from abc import ABC, abstractmethod
from typing import Dict, List

from DPF.datatypes import DataType


class DatasetConfig(ABC):
    """Config for a dataset"""

    def __init__(self, path: str):
        assert not path.endswith('/')
        self.path = path

    @property
    @abstractmethod
    def datatypes(self) -> List[DataType]:
        pass

    @property
    @abstractmethod
    def modality2datatype(self) -> Dict[str, DataType]:
        pass

    @property
    @abstractmethod
    def user_column2default_column(self) -> Dict[str, str]:
        pass

    @property
    def user_column_names(self) -> List[str]:
        return list(self.user_column2default_column.keys())

    @property
    def user_columns_to_rename(self) -> Dict[str, str]:
        columns_to_rename = {}
        for k, v in self.user_column2default_column.items():
            if k != v:
                columns_to_rename[k] = v
        return columns_to_rename
