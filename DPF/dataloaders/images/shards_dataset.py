from typing import List, Optional
import os
import tarfile
import itertools
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from DPF.filesystems.filesystem import FileSystem
from .utils import default_preprocess


class ShardsDataset(IterableDataset):
    """
    Dataset class for sharded dataformat
    """

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        cols_to_return: Optional[List[str]] = None,
        preprocess_f=default_preprocess,
    ):
        super(ShardsDataset).__init__()
        if cols_to_return is None:
            cols_to_return = []
        self.filesystem = filesystem
        self.columns = ["image_path"] + cols_to_return
        self.tar_to_data = df.groupby("archive_path").apply(
            lambda x: [tuple(v.values()) for v in x[self.columns].to_dict("records")]
        )
        self.total_samples = len(df)
        self.preprocess_f = preprocess_f

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
                filename = os.path.basename(data["image_path"])
                img_bytes = tar.extractfile(filename).read()
                yield self.preprocess_f(img_bytes, data)
            tar.close()
