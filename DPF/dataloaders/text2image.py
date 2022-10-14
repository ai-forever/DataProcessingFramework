import pandas as pd
import numpy as np
import os
from PIL import Image
from io import BytesIO
import tarfile
import torch
import itertools
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import BatchSampler, DataLoader

from DPF.utils.image_utils import read_image_rgb

            
def default_preprocess(img_bytes, data):
    image_path = data[0]
    image = Image.open(BytesIO(img_bytes))
    image = np.array(image.resize((256, 256)))
    return image, data
    
    
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


class ShardsDataset(IterableDataset):
    def __init__(self, df, cols_to_return=[], preprocess_f=default_preprocess):
        super(ShardsDataset).__init__()
        self.tar_to_data = df.groupby('archive_path').apply(
            lambda x: [tuple(v.values()) for v in x[['image_path']+cols_to_return].to_dict('records')]
        )
        self.total_samples = len(df)
        self.preprocess_f = preprocess_f
        
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_total_num = worker_info.num_workers if worker_info is not None else None
        worker_id = worker_info.id if worker_info is not None else None
        #print(f'worker_id: {worker_id}, worker_total_num: {worker_total_num}')
        
        for tar_path in itertools.islice(self.tar_to_data.keys(), worker_id, None, worker_total_num):
            data = self.tar_to_data[tar_path]
            filenames = [os.path.basename(i[0]) for i in data]
            tar = tarfile.open(tar_path, mode='r')
            for c in range(len(data)):
                filename = os.path.basename(data[c][0])
                img_bytes = tar.extractfile(filename).read()
                yield self.preprocess_f(img_bytes, data[c])
            tar.close()


FORMAT_TO_DATASET = {
    'raw': RawDataset,
    'shards': ShardsDataset
}

            
class UniversalT2IDataloader:
    def __init__(self, 
                 df, 
                 cols_to_return=[], 
                 preprocess_f=default_preprocess,
                 **dataloader_kwargs):
        
        self.df = df
        self.df_formats = df['data_format'].unique().tolist()
        assert all([f in FORMAT_TO_DATASET for f in self.df_formats]), "Unknown data format in dataloader"
        self.cols_to_return = cols_to_return
        self.preprocess_f = preprocess_f
        self.dataloader_kwargs = dataloader_kwargs
        self.len = None
        
    def test(self):
        for data_format in self.df_formats:
            DatasetClass = FORMAT_TO_DATASET[data_format]
            dataset = DatasetClass(self.df[self.df['data_format'] == data_format], self.cols_to_return, self.preprocess_f)
            print(f'"{data_format}" dataset created')
            dataloader = DataLoader(dataset, **self.dataloader_kwargs)
            print(f'"{data_format}" dataloader created')
            for i in dataloader:
                break
            print(f'"{data_format}" iteration tested. Format "{data_format}" is ok!')
        
    def __len__(self):
        if 'batch_size' in self.dataloader_kwargs:
            bs = self.dataloader_kwargs['batch_size']
        else:
            bs = 1
            
        format_counts = self.df['data_format'].value_counts().to_dict()
        batched_len = sum([
            count//bs if count % bs == 0 else (count//bs)+1 for count in format_counts.values()
        ])
        return batched_len
    
    def __iter__(self):
        for data_format in self.df_formats:
            DatasetClass = FORMAT_TO_DATASET[data_format]
            dataset = DatasetClass(self.df[self.df['data_format'] == data_format], self.cols_to_return, self.preprocess_f)
            dataloader = DataLoader(dataset, **self.dataloader_kwargs)
            for item in dataloader:
                yield item