from typing import Dict, Tuple, List, Union, Any
from abc import abstractmethod, ABC

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
        self.df_columns = [] # fill needed cols
        self.schema = []

    @abstractmethod
    def process(self, row: dict) -> tuple:
        pass

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        pandarallel.initialize(nb_workers=self.workers)
        res = np.array(list(df[self.df_columns].parallel_apply(self.process, axis=1)))
        return res

