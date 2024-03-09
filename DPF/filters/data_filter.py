from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from DPF.dataloaders.dataloader_utils import identical_collate_fn
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping


class DataFilter(ABC):
    """
    Abstract class for all filters that use datalaaders.
    """

    def __init__(self, pbar: bool, _pbar_position: int = 0):
        super().__init__()
        self.pbar = pbar
        self.pbar_position = _pbar_position

    @property
    @abstractmethod
    def schema(self) -> List[str]:
        """List of result columns that filter adds to a DataFrame"""
        pass

    @property
    @abstractmethod
    def dataloader_kwargs(self) -> Dict[str, Any]:
        """Parameters for dataloader"""
        pass

    @property
    @abstractmethod
    def modalities(self) -> List[ModalityName]:
        """List of modalities used in filter. For example, ["image"] or ["video", "text"]."""
        pass

    @property
    @abstractmethod
    def key_column(self) -> str:
        """Column name to use to merge results"""
        pass

    @property
    @abstractmethod
    def metadata_columns(self) -> List[str]:
        """Additional column names needed by filter (will be passed to preprocess method)"""
        pass

    @abstractmethod
    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: Dict[str, str]
    ) -> Any:
        pass

    @abstractmethod
    def process_batch(self, batch: List[Any]) -> Dict[str, List[Any]]:
        pass

    @staticmethod
    def _add_values_from_batch(
        main_dict: Dict[str, List[Any]],
        batch_dict: Dict[str, List[Any]]
    ) -> None:
        for k, v in batch_dict.items():
            main_dict[k].extend(v)

    def _get_dict_from_schema(self) -> Dict[str, List[Any]]:
        return {i: [] for i in self.schema}

    def run(self, dataset: Dataset[Tuple[bool, Any]]) -> pd.DataFrame:
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
