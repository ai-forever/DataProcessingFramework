import itertools
import os
import tarfile
from collections.abc import Iterator
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from DPF.dataloaders.dataloader_utils import (
    get_columns_to_modality_mapping,
    get_paths_columns_to_modality_mapping,
    identical_preprocess_function,
)
from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.filesystems.filesystem import FileSystem
from DPF.types import ModalityToDataMapping


class ShardsDataset(IterableDataset[tuple[bool, Any]]):
    """
    Dataset class for shards format (files in tar archives)
    """

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        split2archive_path: dict[str, str],
        datatypes: list[Union[ShardedDataType, ColumnDataType]],
        metadata_columns: Optional[list[str]] = None,
        preprocess_function: Callable[[ModalityToDataMapping, dict[str, str]], Any] = identical_preprocess_function,
        return_none_on_error: bool = False
    ):
        """
        Parameters
        ----------
        filesystem: FileSystem
            Object of a DPF.filesystems.Filesystem type
        df: pd.DataFrame
            Dataset dataframe from DatasetProcessor
        split2archive_path: Dict[str, str]
            Mapping of the shard index (e.g. split index) to the tar path
        datatypes: List[Union[ShardedDataType, FileDataType, ColumnDataType]]
            List of datatypes to read
        metadata_columns: Optional[List[str]] = None
            List of dataframe columns to return from dataloader
        preprocess_function: Callable[[ModalityToDataMapping, Dict[str, str]], Any] = default_preprocess
            Preprocessing function for data. First argument of the preprocess_f is mapping from modality name to bytes
            and the second argument is mapping from meta_column name to its value.
        return_none_on_error: bool = False
            Whether to return None if error during reading file occurs
        """
        super().__init__()
        self.filesystem = filesystem

        self.datatypes = datatypes
        self.meta_columns = metadata_columns or []

        # mapping columns with path to modality name
        self.path_column2modality = get_paths_columns_to_modality_mapping(
            [datatype for datatype in self.datatypes if isinstance(datatype, ShardedDataType)]
        )
        # mapping column name to modality name (if datatype is ColumnDataType)
        self.column2modality = get_columns_to_modality_mapping(
            [datatype for datatype in self.datatypes if isinstance(datatype, ColumnDataType)]
        )
        self.all_columns = list(set(
            list(self.path_column2modality.keys()) + list(self.column2modality.keys()) + self.meta_columns
        ))

        #
        self.tar_to_data = df.groupby("split_name").apply(
            lambda x: [tuple(v.values()) for v in x[self.all_columns].to_dict("records")]
        )
        self.tar_to_data.index = [split2archive_path[i] for i in self.tar_to_data.index]

        self.total_samples = len(df)
        self.preprocess_f = preprocess_function
        self.return_none_on_error = return_none_on_error

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self) -> Iterator[tuple[bool, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_total_num = worker_info.num_workers if worker_info is not None else None
        worker_id = worker_info.id if worker_info is not None else None

        for tar_path in itertools.islice(
            self.tar_to_data.keys(), worker_id, None, worker_total_num
        ):
            data_all = self.tar_to_data[tar_path]
            tar_bytes = self.filesystem.read_file(tar_path, binary=True)
            tar = tarfile.open(fileobj=tar_bytes, mode="r")
            for data in data_all:
                is_ok = True
                row_sample_data = {self.all_columns[i]: item for i, item in enumerate(data)}
                modality2data = {}

                # read data from files
                for col in self.path_column2modality.keys():
                    modality = self.path_column2modality[col]
                    filename = os.path.basename(row_sample_data[col])
                    if self.return_none_on_error:
                        try:
                            file_bytes = tar.extractfile(filename).read()  # type: ignore
                        except Exception:
                            file_bytes = None
                            is_ok = False
                    else:
                        file_bytes = tar.extractfile(filename).read()  # type: ignore
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

                yield is_ok, preprocessed_data
            tar.close()
