from typing import List, Dict, Optional, Callable, Union, Any
import os
import tarfile
import itertools
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from DPF.filesystems.filesystem import FileSystem
from DPF.dataloaders.utils import default_preprocess
from DPF.datatypes import ShardedDataType, ColumnDataType


class ShardsDataset(IterableDataset):
    """
    Dataset class for sharded dataformat
    """

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        split2archive_path: Dict[str, str],
        datatypes: List[Union[ShardedDataType, ColumnDataType]],
        meta_columns: Optional[List[str]] = None,
        preprocess_f: Callable[[dict, dict], Any] = default_preprocess,
        return_none_on_error: bool = False
    ):
        super(ShardsDataset).__init__()
        self.filesystem = filesystem

        self.datatypes = datatypes
        self.meta_columns = meta_columns or []
        self.configure_columns()

        self.tar_to_data = df.groupby("split_name").apply(
            lambda x: [tuple(v.values()) for v in x[self.columns].to_dict("records")]
        )
        self.tar_to_data.index = [split2archive_path[i] for i in self.tar_to_data.index]

        self.total_samples = len(df)
        self.preprocess_f = preprocess_f
        self.return_none_on_error = return_none_on_error

    def configure_columns(self):
        self.path_column2modality = {}
        self.column2modality = {}
        for d in self.datatypes:
            if isinstance(d, ColumnDataType):
                self.column2modality[d.modality.column] = d.modality.key
            elif isinstance(d, ShardedDataType):
                self.path_column2modality[d.modality.path_column] = d.modality.key
            else:
                raise ValueError()
        self.columns = list(set(
            list(self.path_column2modality.keys()) + list(self.column2modality.keys()) + self.meta_columns
        ))

    def __len__(self):
        return self.total_samples

    def __iter__(self):
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
                data = {self.columns[i]: item for i, item in enumerate(data)}
                modality2data = {}
                # read files
                for col in self.path_column2modality.keys():
                    modality = self.path_column2modality[col]
                    filename = os.path.basename(data[col])
                    if self.return_none_on_error:
                        try:
                            file_bytes = tar.extractfile(filename).read()
                        except Exception as err:
                            file_bytes = None
                    else:
                        file_bytes = tar.extractfile(filename).read()
                    modality2data[modality] = file_bytes
                # read data from columns
                for col in self.column2modality.keys():
                    modality = self.column2modality[col]
                    modality2data[modality] = data[col]

                yield self.preprocess_f(modality2data, data)
            tar.close()
