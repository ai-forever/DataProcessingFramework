from abc import ABC, abstractmethod

from DPF.modalities import MODALITIES, Modality


class DataType(ABC):
    """Represents modality in a specific storage format"""

    def __init__(self, modality: Modality):
        assert modality.key in MODALITIES
        assert self.is_file or (modality.can_be_column and not self.is_file)
        self.modality = modality

    @property
    @abstractmethod
    def is_file(self) -> bool:
        pass


class ColumnDataType(DataType):
    """Represents modality in the column of table"""

    def __init__(
        self,
        modality: Modality,
        user_column_name: str
    ):
        """
        Parameters
        ----------
        modality: Modality
            instance of DPF.modalities.Modality
        user_column_name: str
            Name of column with data of this modality
        """
        super().__init__(modality)
        self.user_column_name = user_column_name

    @property
    def is_file(self) -> bool:
        return False

    @property
    def column_name(self) -> str:
        return self.modality.column  # type: ignore

    def __repr__(self) -> str:
        return f'ColumnDataType(modality={self.modality}, user_column_name="{self.user_column_name}")'


class FileDataType(DataType):
    """Represents data with modality in file"""

    def __init__(
        self,
        modality: Modality,
        user_path_column_name: str
    ):
        """
        Parameters
        ----------
        modality: Modality
            instance of DPF.modalities.Modality
        user_path_column_name: str
            Name of column with paths to files of this modality
        """
        super().__init__(modality)
        self.user_path_column_name = user_path_column_name

    @property
    def is_file(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'FileDataType(modality={self.modality}, user_path_column_name="{self.user_path_column_name}")'


class ShardedDataType(DataType):
    """Represents data with modality in files in dataset with sharded format"""

    def __init__(
        self,
        modality: Modality,
        user_basename_column_name: str,
    ):
        """
        Parameters
        ----------
        modality: Modality
            instance of DPF.modalities.Modality
        user_basename_column_name: str
            Column name with file names of this modality
        """
        super().__init__(modality)
        self.user_basename_column_name = user_basename_column_name

    @property
    def is_file(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'ShardedDataType(modality={self.modality}, user_basename_column_name="{self.user_basename_column_name}")'
