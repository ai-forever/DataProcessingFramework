from abc import ABC

from DPF.filters.data_filter import DataFilter
from DPF.modalities import MODALITIES, ModalityName


# TODO(review) - дважды абстрактный класс (наследуется от ABC, и класса, отнаследованного от ABC). Абстрактный класс должен быть один, от него наследуются конкретные реализации
class VideoFilter(DataFilter, ABC):
    """
    Abstract class for all image filters.
    """

    @property
    def modalities(self) -> list[ModalityName]:
        return ['video']

    @property
    def key_column(self) -> str:
        return MODALITIES['video'].path_column

    @property
    def metadata_columns(self) -> list[str]:
        return []
