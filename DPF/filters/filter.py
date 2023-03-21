from abc import ABC, abstractmethod
import pandas as pd
from DPF.filesystems import FileSystem


class Filter(ABC):
    """
    Abstract class for all filters
    """

    @abstractmethod
    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        pass

    def __call__(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        return self.run(df, filesystem)
