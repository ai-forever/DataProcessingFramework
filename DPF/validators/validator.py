from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class Validator(ABC):
    """
    Abstract class for all validators.
    """

    @abstractmethod
    def validate_df(
            self,
            df: pd.DataFrame
        ) -> (dict, Dict[str, int]):
        # validates only dataframe
        # should return (dict, Dict[str, int])
        # dictionary with errors description and dictionary with number of occured errors
        errors = {"ok": True}
        error2count = {}

    @abstractmethod
    def validate(
            self,
            dataset_dir: str,
            processes: int = 1
        ) -> (dict, Dict[str, int], bool):
        # validates a dataset
        # should return (dict, Dict[str, int], bool)
        # dictionary with errors description, dictionary with number of occured errors
        # and status code (True if there was no errors)
        errors = {"ok": True}
        error2count = {}
        all_ok = True
