from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pandarallel import pandarallel

from DPF.filesystems.filesystem import FileSystem
from DPF.filters import Filter


class TextFilter(Filter):
    """
    Base class for all text filters.
    """

    def __init__(
            self,
            text_column_name: str = 'caption',
            workers: int = 16
    ):
        super().__init__()

        self.text_column_name = text_column_name
        self.workers = workers

        self.schema = []

    @abstractmethod
    def process(self, row):
        pass

    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        pandarallel.initialize(nb_workers=self.workers)
        res = np.array(list(df.parallel_apply(self.process, axis=1)))
        df[self.schema] = res
        return df
