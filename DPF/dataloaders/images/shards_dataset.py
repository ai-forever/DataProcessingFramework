from typing import List
import numpy as np
import pandas as pd
import os
from PIL import Image
from io import BytesIO
import tarfile
import torch
import itertools
from torch.utils.data import IterableDataset

from .utils import default_preprocess
from DPF.filesystems.filesystem import FileSystem


class ShardsDataset(IterableDataset):
    
    def __init__(
            self, 
            filesystem: FileSystem, 
            df: pd.DataFrame, 
            cols_to_return: List[str] = [], 
            preprocess_f = default_preprocess
        ):
        super(ShardsDataset).__init__()
        self.filesystem = filesystem
        self.columns = ['image_path']+cols_to_return
        self.tar_to_data = df.groupby('archive_path').apply(
            lambda x: [tuple(v.values()) for v in x[self.columns].to_dict('records')]
        )
        self.total_samples = len(df)
        self.preprocess_f = preprocess_f
        
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_total_num = worker_info.num_workers if worker_info is not None else None
        worker_id = worker_info.id if worker_info is not None else None
        
        for tar_path in itertools.islice(self.tar_to_data.keys(), worker_id, None, worker_total_num):
            data_all = self.tar_to_data[tar_path]
            tar_bytes = self.filesystem.read_file(tar_path, binary=True)
            tar = tarfile.open(fileobj=tar_bytes, mode='r')
            for c, data in enumerate(data_all):
                data = {self.columns[i]: item for i, item in enumerate(data)}
                filename = os.path.basename(data['image_path'])
                img_bytes = tar.extractfile(filename).read()
                yield self.preprocess_f(img_bytes, data)
            tar.close()