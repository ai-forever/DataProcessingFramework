from typing import List
from abc import ABC

from DPF.filters.data_filter import DataFilter
from DPF.modalities import MODALITIES


# TODO(review) - дважды абстрактный класс (наследуется от ABC, и класса, отнаследованного от ABC). Абстрактный класс должен быть один, от него наследуются конкретные реализации
class VideoFilter(DataFilter, ABC):
    """
    Abstract class for all image filters.
    """

    # TODO(review) - дубль инициализации, она повторяет инициализацию класса-родителя
    def __init__(self, pbar: bool):
        super().__init__(pbar)

        self.pbar = pbar

        self.schema = []  # fill with your columns
        self.dataloader_kwargs = {}  # Insert your params

    @property
    def modalities(self) -> List[str]:
        return ['video']

    @property
    def key_column(self) -> str:
        return MODALITIES['video'].path_column

    @property
    def metadata_columns(self) -> List[str]:
        return []
