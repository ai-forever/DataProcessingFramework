import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
from io import BytesIO
import string
import random
import numpy as np
from argparse import ArgumentParser
import multiprocessing as mp

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import BatchSampler, DataLoader

from .t2ifilter import T2IFilter
from DPF.dataloaders.text2image import UniversalT2IDataloader
from DPF.utils import init_logger


def get_image_info(img_bytes, data):
    path = data[0]
    
    is_correct = True
    width, height, channels = None, None, None
    err_str = None

    try:
        pil_img = Image.open(BytesIO(img_bytes))
        pil_img.load()

        arr = np.array(pil_img)

        width = pil_img.width
        height = pil_img.height
        if len(arr.shape) == 2:
            channels = 1
        else:
            channels = arr.shape[2]
    except Exception as err:
        is_correct = False
        err_str = str(err)
    
    return path, is_correct, width, height, channels, err_str
    
    
class ImageInfoGatherer(T2IFilter):
    
    def __init__(self, save_parquets_dir='parquets/', task_name=None, save_parquets=True, 
                 workers=16, log_filename=None, logging_dir='./logs/', logger=None):
        
        self.save_parquets_dir = save_parquets_dir.rstrip('/')
        self.save_parquets = save_parquets
        self.num_workers = workers
        self.collate_fn = lambda x: x
        self.task_name = task_name if task_name else ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        
        os.makedirs(self.save_parquets_dir, exist_ok=True)
        
        logfile = f'log_gather_images_info.log' if log_filename is None else log_filename
        self.logger = logger or init_logger(logfile, logging_dir=logging_dir)
        self.logger.info(f'Using {self.num_workers} workers')
        if self.save_parquets:
            self.logger.info(f'Saving dataframes to: {self.save_parquets_dir}')
        
    def process_batch(self, batch) -> dict:
        df_batch_labels = {
            'image_path': [], 
            'is_correct': [], 
            'width': [], 
            'height': [], 
            'channels': [],
            'error': []
        }
        
        for data in batch:
            image_path, is_correct, width, height, channels, error = data
            df_batch_labels['image_path'].append(image_path)
            df_batch_labels['is_correct'].append(is_correct)
            df_batch_labels['width'].append(width)
            df_batch_labels['height'].append(height)
            df_batch_labels['channels'].append(channels)
            df_batch_labels['error'].append(error)
        return df_batch_labels
        
    def preprocess(self, img_bytes, data):
        return get_image_info(img_bytes, data)
        
    def run(self, df):
        self.logger.info(f'Starting task {self.task_name}')
        self.logger.info(f'Files to process: {len(df)}')
        
        dataloader = UniversalT2IDataloader(
            df, num_workers=self.num_workers, batch_size=1,
            preprocess_f=self.preprocess, collate_fn=self.collate_fn
        )
        
        df_labels = {
            'image_path': [], 
            'is_correct': [], 
            'width': [], 
            'height': [], 
            'channels': [],
            'error': []
        }

        for batch in tqdm(dataloader):
            df_batch_labels = self.process_batch(batch)
            self.add_values_from_batch(df_labels, df_batch_labels)
        
        df_result = pd.DataFrame(df_labels)
        invalid_images_count = len(df_result[df_result['is_correct'] == False])
        self.logger.info(f'Task finished. Invalid images count: {invalid_images_count}')
        
        df = pd.merge(df, df_result, on='image_path')
        
        if self.save_parquets:
            df_result.to_parquet(
                f'{self.save_parquets_dir}/{self.task_name}.parquet',
                index=False
            )
        
        return df