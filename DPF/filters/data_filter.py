from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DPF.dataloaders.dataloader_utils import identical_collate_fn
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping


class DataFilter(ABC):
    """
    Abstract class for all data filters.
    """

    def __init__(self, pbar: bool, _pbar_position: int = 0):
        super().__init__()
        self.pbar = pbar
        self.pbar_position = _pbar_position

    @property
    @abstractmethod
    def schema(self) -> list[str]:
        """List of result columns that filter adds to a DataFrame"""
        pass

    @property
    @abstractmethod
    def dataloader_kwargs(self) -> dict[str, Any]:
        """Parameters for dataloader"""
        pass

    @property
    @abstractmethod
    def modalities(self) -> list[ModalityName]:
        """List of modalities used in filter. For example, ["image"] or ["video", "text"]."""
        pass

    @property
    @abstractmethod
    def key_column(self) -> str:
        """Column name used to merge results"""
        pass

    @property
    @abstractmethod
    def metadata_columns(self) -> list[str]:
        """Additional column names needed by filter (will be passed to preprocess method)"""
        pass

    @abstractmethod
    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        """Method that preprocesses data for filter.
        This method is passed to dataloader and runs in parallel processes.

        Parameters
        ----------
        modality2data: ModalityToDataMapping
            Mapping of modality name to its data
        metadata: dict[str, Any]
            Sample data from columns, column names used from metadata_columns property

        Returns
        -------
        Any
            Any preprocessed data. Used in process_batch method
        """
        pass

    @abstractmethod
    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        """Method that processes preprocessed data.

        Parameters
        ----------
        batch: list[Any]
            List of preprocessed items from preprocess_data method

        Returns
        -------
        dict[str, list[Any]]
            Mapping from column to its data for batch samples. Used to create new dataframe
        """
        pass

    @staticmethod
    def _add_values_from_batch(
        main_dict: dict[str, list[Any]],
        batch_dict: dict[str, list[Any]]
    ) -> None:
        for k, v in batch_dict.items():
            main_dict[k].extend(v)

    def _get_dict_from_schema(self) -> dict[str, list[Any]]:
        return {i: [] for i in self.schema}

    def run(self, dataset: Dataset[tuple[bool, Any]]) -> pd.DataFrame:
        """Runs datafilter on a dataset and returns a dataframe with results

        Parameters
        ----------
        dataset: Dataset[tuple[bool, Any]]
            torch.Dataset instance with samples to process

        Returns
        -------
        pd.DataFrame
            Dataframe with columns from schema property
        """
        dataloader = DataLoader(dataset, collate_fn=identical_collate_fn, **self.dataloader_kwargs)
        filter_results = self._get_dict_from_schema()

        for batch in tqdm(dataloader, disable=not self.pbar, position=self.pbar_position):
            # drop Nans
            batch_filtered = [b[1] for b in batch if b[0]]
            if len(batch_filtered) == 0:
                continue

            filter_results_batch = self.process_batch(batch_filtered)
            self._add_values_from_batch(filter_results, filter_results_batch)

        return pd.DataFrame(filter_results)
