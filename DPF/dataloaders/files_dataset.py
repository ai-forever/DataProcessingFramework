from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from torch.utils.data import Dataset

from DPF.dataloaders.dataloader_utils import identical_preprocess_function
from DPF.datatypes import ColumnDataType, FileDataType, ShardedDataType
from DPF.filesystems.filesystem import FileSystem


class FilesDataset(Dataset):
    """
    Dataset class to read "raw" files
    """

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        datatypes: List[Union[ShardedDataType, FileDataType, ColumnDataType]],
        meta_columns: Optional[List[str]] = None,
        preprocess_function: Callable[[Dict[str, bytes], Dict[str, str]], Any] = identical_preprocess_function,
        # TODO(review) - на ошибке надо выбрасывать ошибку, а не возвращать None, и в дальнейшем эту ошибку обрабатывать прикладом, использующим этот класс
        return_none_on_error: bool = False
    ):
        """
        Parameters
        ----------
        filesystem: FileSystem
            Object of a DPF.filesystems.Filesystem type
        df: pd.DataFrame
            Dataset dataframe from DatasetProcessor
        datatypes: List[Union[ShardedDataType, FileDataType, ColumnDataType]]
            List of datatypes to read
        meta_columns: Optional[List[str]] = None
            List of dataframe columns to return from dataloader
        preprocess_function: Callable[[Dict[str, bytes], Dict[str, str]], Any] = default_preprocess
            Preprocessing function for data. First argument of the preprocess_f is mapping from modality name to bytes
            and the second argument is mapping from meta_column name to its value.
        return_none_on_error: bool = False
            Whether to return None if error during reading file occures
        """
        self.filesystem = filesystem

        self.datatypes = datatypes
        self.meta_columns = meta_columns if meta_columns else []

        # configuring columns
        self.path_column2modality = {}
        self.column2modality = {}
        for d in self.datatypes:
            if isinstance(d, ColumnDataType):
                self.column2modality[d.modality.column] = d.modality.key
            elif isinstance(d, (ShardedDataType, FileDataType)):
                self.path_column2modality[d.modality.path_column] = d.modality.key
            else:
                raise ValueError()
        self.columns = list(set(
            list(self.path_column2modality.keys()) + list(self.column2modality.keys()) + self.meta_columns
        ))

        #
        self.data_to_iterate = df[self.columns].values
        self.preprocess_f = preprocess_function
        self.return_none_on_error = return_none_on_error

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        data = {
            self.columns[c]: item for c, item in enumerate(self.data_to_iterate[idx])
        }
        modality2data = {}
        is_ok = True
        # read files
        for col in self.path_column2modality.keys():
            modality = self.path_column2modality[col]
            if self.return_none_on_error:
                try:
                    file_bytes = self.filesystem.read_file(data[col], binary=True).getvalue()
                except Exception:
                    file_bytes = None
                    is_ok = False
            else:
                file_bytes = self.filesystem.read_file(data[col], binary=True).getvalue()
            modality2data[modality] = file_bytes

        # read data from columns
        for col in self.column2modality.keys():
            modality = self.column2modality[col]
            modality2data[modality] = data[col]

        preprocessed_data = None
        if self.return_none_on_error and is_ok:
            try:
                preprocessed_data = self.preprocess_f(modality2data, data)
            except Exception:
                is_ok = False
        elif is_ok:
            preprocessed_data = self.preprocess_f(modality2data, data)
        return is_ok, preprocessed_data
