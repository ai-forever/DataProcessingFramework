from typing import Dict, List, Union, Any
from abc import abstractmethod, ABC
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from DPF.filters.utils import identical_collate_fn


class DataFilter(ABC):
    """
    Abstract class for all filters that use datalaaders.
    """

    def __init__(self, pbar: bool):
        super().__init__()
        self.pbar = pbar

        self.schema = []
        self.dataloader_kwargs = {}  # Insert your params

    @property
    @abstractmethod
    def modalities(self) -> List[str]:
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
    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        pass

    @abstractmethod
    def process_batch(self, batch) -> dict:
        pass

    @staticmethod
    def _add_values_from_batch(main_dict: dict, batch_dict: dict):
        for k, v in batch_dict.items():
            main_dict[k].extend(v)

    def _generate_dict_from_schema(self):
        return {i: [] for i in self.schema}

    def run(self, dataset: Dataset) -> pd.DataFrame:
        dataloader = DataLoader(dataset, collate_fn=identical_collate_fn, **self.dataloader_kwargs)
        df_labels = self._generate_dict_from_schema()

        for batch in tqdm(dataloader, disable=not self.pbar):
            # drop Nans
            batch_filtered = [b[1] for b in batch if b[0]]
            if len(batch_filtered) == 0:
                continue

            df_batch_labels = self.process_batch(batch_filtered)
            self._add_values_from_batch(df_labels, df_batch_labels)

        return pd.DataFrame(df_labels)
