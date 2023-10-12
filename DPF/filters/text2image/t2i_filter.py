from typing import List, Dict, Union
from abc import abstractmethod, ABC

from DPF.filters import DataFilter
from DPF.modalities import MODALITIES


class T2IFilter(DataFilter, ABC):
    """
    Base class for all text-to-image filters.
    """

    def __init__(self, pbar: bool):
        super().__init__(pbar)

        self.pbar = pbar
        self.schema = []  # fill with your columns
        self.dataloader_kwargs = {}  # Insert your params

    @property
    def modalities(self) -> List[str]:
        return ['image', 'text']

    @property
    def key_column(self) -> str:
        return MODALITIES['image'].path_column

    @property
    def metadata_columns(self) -> List[str]:
        return []
