import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset

from .utils import default_preprocess


class RawDataset(Dataset):
    def __init__(self, df, cols_to_return=[], preprocess_f=default_preprocess):
        super(RawDataset).__init__()
        self.data_to_iterate = df[['image_path']+cols_to_return].values
        self.preprocess_f = preprocess_f
        
    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        data = self.data_to_iterate[idx]
        image_path = data[0]
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.preprocess_f(image_bytes, data)