from abc import ABC, abstractmethod
from typing import Any

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
    def columns_to_process(self) -> list[str]:
        """List of columns in DataFrame that should be processed and passed to "process_sample" method"""
        pass

    @property
    @abstractmethod
    def result_columns(self) -> list[str]:
        """List of result columns that filter adds to a DataFrame"""
        pass

    @property
    def schema(self) -> list[str]:
        """List of all columns of a DataFrame returned by filter.
        Also includes system columns needed to merge (they are not added)
        """
        return self.result_columns

    @abstractmethod
    def process_sample(self, sample: dict[str, Any]) -> list[Any]:
        """Method that processes one sample

        Parameters
        ----------
        sample: dict[str, Any]
            Sample from dataset dataframe. Mapping from column name to its value

        Returns
        -------
        list[Any]
            Results to add (length of list is equal to length of a "schema" property)
        """
        pass

    def __call__(self, df: pd.DataFrame) -> list[list[Any]]:
        """Run filter. Rusn process_sample method on full dataframe

        Parameters
        ----------
        df: pd.DataFrame
            Dataset to be processed

        Returns
        -------
        list[list[Any]]
            List of results
        """
        pandarallel.initialize(nb_workers=self.workers)
        res = list(df[self.columns_to_process].parallel_apply(self.process_sample, axis=1))
        return res

