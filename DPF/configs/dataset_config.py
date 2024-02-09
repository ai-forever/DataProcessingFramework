from typing import List, Dict, Optional, Union
from abc import abstractmethod, ABC

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
        assert len(set([d.modality.key for d in datatypes])) == len(datatypes)
        assert not path.endswith('/')
        self.datatypes = datatypes
        self.path = path

    @property
    @abstractmethod
    def modality2datatype(self) -> Dict[str, DataType]:
        pass

    @property
    @abstractmethod
    def columns_mapping(self) -> Dict[str, str]:
        pass

    def __repr__(self) -> str:
        s = "DatasetConfig(\n\t"
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s