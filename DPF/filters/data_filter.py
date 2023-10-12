from typing import Dict, List, Union, Any
from abc import abstractmethod, ABC
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


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
        pass

    @property
    @abstractmethod
    def key_column(self) -> str:
        pass

    @property
    @abstractmethod
    def metadata_columns(self) -> List[str]:
        pass

    def get_dataset_kwargs(self) -> dict:
        return {
            "modalities": self.modalities,
            "meta_columns": self.metadata_columns+[self.key_column],
            "preprocess_f": self.preprocess
        }

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
        dataloader = DataLoader(dataset, **self.dataloader_kwargs)
        df_labels = self._generate_dict_from_schema()

        for batch in tqdm(dataloader, disable=not self.pbar):
            df_batch_labels = self.process_batch(batch)
            self._add_values_from_batch(df_labels, df_batch_labels)

        return pd.DataFrame(df_labels)
