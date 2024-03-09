from abc import ABC
from typing import List

from DPF.filters import DataFilter
from DPF.modalities import MODALITIES, ModalityName


class T2IFilter(DataFilter, ABC):
    """
    Base class for all text-to-image filters.
    """

    @property
    def modalities(self) -> List[ModalityName]:
        return ['image', 'text']

    @property
    def key_column(self) -> str:
        return MODALITIES['image'].path_column

    @property
    def metadata_columns(self) -> List[str]:
        return []
