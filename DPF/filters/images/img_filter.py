from typing import List
from abc import ABC

from DPF.filters.data_filter import DataFilter
from DPF.modalities import MODALITIES


class ImageFilter(DataFilter, ABC):
    """
    Abstract class for all image filters.
    """

    def __init__(self, pbar: bool):
        super().__init__(pbar)

        self.pbar = pbar

        self.schema = []  # fill with your columns
        self.dataloader_kwargs = {}  # Insert your params

    @property
    def modalities(self) -> List[str]:
        return ['image']

    @property
    def key_column(self) -> str:
        return MODALITIES['image'].path_column

    @property
    def metadata_columns(self) -> List[str]:
        return []
