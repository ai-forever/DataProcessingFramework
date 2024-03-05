from typing import List, Dict, Union, Any, Type
from multiprocessing import Process, Manager
import pandas as pd
import numpy as np
import torch

from DPF.configs import DatasetConfig
from DPF.filesystems import FileSystem
from DPF.dataset_reader import DatasetReader
from .data_filter import DataFilter


# TODO(review) - один вызов в MultiGPUFilter, нужно перенести его внутрь класса
def run_one_process(
    config: DatasetConfig,
    fs: FileSystem,
    df: pd.DataFrame,
    i: int,
    index: pd.Series,
    results: List[pd.DataFrame],
    filter_class: Type[DataFilter],
    filter_kwargs: dict,
    device: str,
    filter_run_kwargs: dict
):
    reader = DatasetReader(filesystem=fs)
    processor = reader.from_df(config, df)
    datafilter = filter_class(**filter_kwargs, _pbar_position=i, device=device)
    processor.apply_data_filter(datafilter, **filter_run_kwargs)
    res = processor.df
    res.set_index(index, inplace=True)
    results.append(res)


class MultiGPUDataFilter:
    """
    Class for multi-gpu inference
    """

    def __init__(
        self,
        devices: List[Union[torch.device | str]],
        filter_class: type,
        filter_params: dict
    ):
        self.filter_class = filter_class
        self.filter_params = filter_params
        self.devices = devices
        self.num_parts = len(devices)

    def run(self, df: pd.DataFrame, config: DatasetConfig, fs: FileSystem, filter_run_kwargs: dict) -> pd.DataFrame:
        manager = Manager()
        shared_results = manager.list()

        df_splits = np.array_split(df, self.num_parts)
        params = []
        for i in range(self.num_parts):
            params.append(
                (
                    config,
                    fs,
                    df_splits[i],
                    i,
                    df_splits[i].index,
                    shared_results,
                    self.filter_class,
                    self.filter_params,
                    self.devices[i],
                    filter_run_kwargs
                )
            )

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
