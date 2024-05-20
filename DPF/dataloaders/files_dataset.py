from typing import Any, Callable, Optional, Union

import pandas as pd
from torch.utils.data import Dataset

from DPF.connectors import Connector
from DPF.dataloaders.dataloader_utils import (
    get_columns_to_modality_mapping,
    get_paths_columns_to_modality_mapping,
    identical_preprocess_function,
)
from DPF.datatypes import ColumnDataType, FileDataType, ShardedDataType
from DPF.types import ModalityToDataMapping


class FilesDataset(Dataset[tuple[bool, Any]]):
    """
    Dataset class to read "raw" files
    """

    def __init__(
        self,
        connector: Connector,
        df: pd.DataFrame,
        datatypes: list[Union[ShardedDataType, FileDataType, ColumnDataType]],
        metadata_columns: Optional[list[str]] = None,
        preprocess_function: Callable[[ModalityToDataMapping, dict[str, str]], Any] = identical_preprocess_function,
        # TODO(review) - на ошибке надо выбрасывать ошибку, а не возвращать None, и в дальнейшем эту ошибку обрабатывать прикладом, использующим этот класс
        return_none_on_error: bool = False
    ):
        """
        Parameters
        ----------
        connector: Connector
            Object of a DPF.connectors.Connector type
        df: pd.DataFrame
            Dataset dataframe from DatasetProcessor
        datatypes: List[Union[ShardedDataType, FileDataType, ColumnDataType]]
            List of datatypes to read
        metadata_columns: Optional[List[str]] = None
            List of dataframe columns to return from dataloader
        preprocess_function: Callable[[ModalityToDataMapping, Any] = default_preprocess
            Preprocessing function for data. First argument of the preprocess_f is mapping from modality name to bytes
            and the second argument is mapping from meta_column name to its value.
        return_none_on_error: bool = False
            Whether to return None if error during reading file occurs
        """
        self.connector = connector

        self.datatypes = datatypes
        assert all(isinstance(d, (ShardedDataType, FileDataType, ColumnDataType)) for d in self.datatypes)

        self.meta_columns = metadata_columns if metadata_columns else []

        # mapping columns with path to modality name
        self.path_column2modality = get_paths_columns_to_modality_mapping(
            [datatype for datatype in self.datatypes if isinstance(datatype, (FileDataType, ShardedDataType))]
        )
        # mapping column name to modality name (if datatype is ColumnDataType)
        self.column2modality = get_columns_to_modality_mapping(
            [datatype for datatype in self.datatypes if isinstance(datatype, ColumnDataType)]
        )
        self.all_columns = list(set(
            list(self.path_column2modality.keys()) + list(self.column2modality.keys()) + self.meta_columns
        ))

        self.data_to_iterate = df[self.all_columns].values
        self.preprocess_f = preprocess_function
        self.return_none_on_error = return_none_on_error

    def __len__(self) -> int:
        return len(self.data_to_iterate)

    def __getitem__(self, idx: int) -> tuple[bool, Any]:
        row_sample_data = {
            self.all_columns[c]: item for c, item in enumerate(self.data_to_iterate[idx])
        }
        modality2data = {}
        is_ok = True

        # read data from files
        for col in self.path_column2modality.keys():
            modality = self.path_column2modality[col]
            if self.return_none_on_error:
                try:
                    file_bytes = self.connector.read_file(row_sample_data[col], binary=True).getvalue()
                except Exception:
                    file_bytes = None
                    is_ok = False
            else:
                file_bytes = self.connector.read_file(row_sample_data[col], binary=True).getvalue()
            modality2data[modality] = file_bytes

        # read data from columns
        for col in self.column2modality.keys():
            modality = self.column2modality[col]
            modality2data[modality] = row_sample_data[col]

        preprocessed_data = None
        if self.return_none_on_error and is_ok:
            try:
                preprocessed_data = self.preprocess_f(modality2data, row_sample_data)
            except Exception:
                is_ok = False
        elif is_ok:
            preprocessed_data = self.preprocess_f(modality2data, row_sample_data)
        return is_ok, preprocessed_data
