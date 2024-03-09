from multiprocessing import Manager, Process
from typing import List, Type, Union, Dict, Any

import numpy as np
import pandas as pd
import torch

from DPF.configs import DatasetConfig
from DPF.dataset_reader import DatasetReader
from DPF.filesystems import FileSystem

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
    filter_kwargs: Dict[str, Any],
    device: Union[str, torch.device],
    filter_run_kwargs: Dict[str, Any]
) -> None:
    reader = DatasetReader(filesystem=fs)
    processor = reader.from_df(config, df)
    datafilter = filter_class(**filter_kwargs, _pbar_position=i, device=device)  # type: ignore
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
        devices: List[Union[torch.device, str]],
        datafilter_class: Type[DataFilter],
        datafilter_params: Dict[str, Any]
    ):
        self.filter_class = datafilter_class
        self.filter_params = datafilter_params
        self.devices = devices
        self.num_parts = len(devices)

    def run(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        fs: FileSystem,
        filter_run_kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
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
                    df_splits[i].index,  # type: ignore
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
