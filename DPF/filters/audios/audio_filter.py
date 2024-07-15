from abc import ABC

from DPF.filters.data_filter import DataFilter
from DPF.modalities import MODALITIES, ModalityName


class AudioFilter(DataFilter, ABC):
    """
    Abstract class for all audio filters.
    """

    @property
    def modalities(self) -> list[ModalityName]:
        return ['audio']

    @property
    def key_column(self) -> str:
        return MODALITIES['audio'].path_column

    @property
    def metadata_columns(self) -> list[str]:
        return []
