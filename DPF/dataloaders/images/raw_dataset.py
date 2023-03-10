from typing import List, Optional
import pandas as pd
from torch.utils.data import Dataset
from DPF.filesystems.filesystem import FileSystem
from .utils import default_preprocess


class RawDataset(Dataset):

    def __init__(
            self,
            filesystem: FileSystem,
            df: pd.DataFrame,
            cols_to_return: Optional[List[str]] = None,
            preprocess_f = default_preprocess
        ):
        super(RawDataset).__init__()
        if cols_to_return is None:
            cols_to_return = []
        self.filesystem = filesystem
        self.columns = ['image_path'] + cols_to_return
        self.data_to_iterate = df[self.columns].values
        self.preprocess_f = preprocess_f

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        data = {self.columns[c]: item for c, item in enumerate(self.data_to_iterate[idx])}
        image_path = data['image_path']
        image_bytes = self.filesystem.read_file(image_path, binary=True).getvalue()
        return self.preprocess_f(image_bytes, data)
