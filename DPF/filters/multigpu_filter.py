from typing import List
import pandas as pd
import numpy as np
import torch

import multiprocessing
from multiprocessing import Process, Manager

from DPF.filesystems import FileSystem
from .filter import Filter


def run_one_process(df, fs, index, results, filter_class, filter_kwargs, device):
    imgfilter = filter_class(**filter_kwargs, device=device)
    res = imgfilter(df, fs)
    res.set_index(index, inplace=True)
    results.append(res)
            

class MultiGPUFilter(Filter):
    """
    Class for multi-gpu inference
    """
    def __init__(
            self,
            devices: List[torch.device],
            filter_class,
            **filter_kwargs
        ):
        super(MultiGPUFilter, self).__init__()
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.devices = devices
        self.num_parts = len(devices)
    
    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        manager = Manager()
        shared_results = manager.list()
        
        df_splits = np.array_split(df, self.num_parts)
        params = []
        for i in range(self.num_parts):
            params.append((
                df_splits[i], filesystem, df_splits[i].index, shared_results,
                self.filter_class, self.filter_kwargs, self.devices[i]
            ))
            
        processes = []
        for param in params:
            p = Process(target=run_one_process, args=param)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        res_df = pd.concat(shared_results)
        res_df.sort_index(inplace=True)
        return res_df