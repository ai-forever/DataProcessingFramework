from abc import ABC, abstractmethod
from typing import Dict, List

from DPF.datatypes import DataType


class DatasetConfig(ABC):
    """Config for a dataset"""

    def __init__(
        self,
        path: str,
        datatypes: List[DataType],
    ):
        """
        Parameters
        ----------
        path: str
            Path to dataset
        datatypes: List[DataType]
            List of datatypes in dataset
        """
        assert len({d.modality.key for d in datatypes}) == len(datatypes)
        assert not path.endswith('/')
        self.datatypes = datatypes
        self.path = path

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

    def __repr__(self) -> str:
        s = "DatasetConfig(\n\t"
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s
