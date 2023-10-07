from abc import ABC, abstractmethod

from DPF.modalities import MODALITIES, Modality


class DataType(ABC):

    def __init__(self, modality: Modality):
        assert modality.key in MODALITIES
        assert (modality.can_be_file and self.is_file) or (modality.can_be_column and not self.is_file)
        self.modality = modality

    @property
    @abstractmethod
    def is_file(self) -> bool:
        pass


class ColumnDataType(DataType):

    def __init__(
        self,
        modality: Modality,
        user_column_name: str
    ):
        super().__init__(modality)
        self.user_column_name = user_column_name

    @property
    def is_file(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f'ColumnDataType(modality={self.modality}, user_column_name="{self.user_column_name}")'


class FileDataType(DataType):

    def __init__(
        self,
        modality: Modality,
        user_path_column_name: str
    ):
        super().__init__(modality)
        self.user_path_column_name = user_path_column_name

    @property
    def is_file(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'FileDataType(modality={self.modality}, user_path_column_name="{self.user_path_column_name}")'


class ShardedDataType(DataType):

    def __init__(
        self,
        modality: Modality,
        user_basename_column_name: str,
    ):
        super().__init__(modality)
        self.user_basename_column_name = user_basename_column_name

    @property
    def is_file(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'ShardedDataType(modality={self.modality}, user_basename_column_name="{self.user_basename_column_name}")'
