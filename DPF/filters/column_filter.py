from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from pandarallel import pandarallel


class ColumnFilter(ABC):
    """
    Abstract class for all filters that use only DataFrame.
    """

    def __init__(self, workers: int, pbar: bool = True):
        self.workers = workers
        self.pbar = pbar

    @property
    @abstractmethod
    def columns_to_process(self) -> List[str]:
        """List of columns in DataFrame that should be processed and passed to "process" method"""
        pass

    @property
    @abstractmethod
    def schema(self) -> List[str]:
        """List of result columns that filter adds to a DataFrame"""
        pass

    @abstractmethod
    def process(self, row: dict) -> tuple:
        pass

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        pandarallel.initialize(nb_workers=self.workers)
        res = np.array(list(df[self.columns_to_process].parallel_apply(self.process, axis=1)))
        return res

