import multiprocessing
from multiprocessing import Manager
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from DPF.configs import DatasetConfig
from DPF.connectors import Connector
from DPF.dataset_reader import DatasetReader

from .data_filter import DataFilter


# TODO(review) - один вызов в MultiGPUFilter, нужно перенести его внутрь класса
def run_one_process(
    config: DatasetConfig,
    connector: Connector,
    df: pd.DataFrame,
    i: int,
    index: pd.Series,
    results: list[pd.DataFrame],
    filter_class: Optional[type[DataFilter]],
    filter_kwargs: Optional[dict[str, Any]],
    datafilter_init_fn: Optional[Callable[[int, Union[str, torch.device], dict[str, Any]], DataFilter]],
    datafilter_init_fn_kwargs: dict[str, Any],
    device: Union[str, torch.device],
    filter_run_kwargs: dict[str, Any]
) -> None:
    reader = DatasetReader(connector=connector)
    processor = reader.from_df(config, df)
    if datafilter_init_fn:
        datafilter = datafilter_init_fn(i, device, datafilter_init_fn_kwargs)
    else:
        datafilter = filter_class(**filter_kwargs, _pbar_position=i, device=device)  # type: ignore

    datafilter._created_by_multigpu_data_filter = True
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
        devices: list[Union[torch.device, str]],
        datafilter_class: Optional[type[DataFilter]] = None,
        datafilter_params: Optional[dict[str, Any]] = None,
        datafilter_init_fn: Optional[Callable[[int, Union[str, torch.device], dict[str, Any]], DataFilter]] = None,
        datafilter_init_fn_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        devices: list[Union[torch.device, str]]
            List of devices to run datafilter on
        datafilter_class: Optional[type[DataFilter]] = None
            Class of datafilter to use
        datafilter_params: Optional[dict[str, Any]] = None
            Parameters for datafilter_class initialization
        datafilter_init_fn: Optional[Callable[[int, Union[str, torch.device], dict[str, Any]], DataFilter]] = None
            Initialization function for a datafilter. Takes _pbar_position as first arg and device as a second arg
        datafilter_init_fn_kwargs: Optional[dict[str, Any]] = None
            Additional parameters for datafilter_init_fn
        """
        self.filter_class = datafilter_class
        self.filter_params = datafilter_params
        self.datafilter_init_fn = datafilter_init_fn
        self.datafilter_init_fn_kwargs = datafilter_init_fn_kwargs if datafilter_init_fn_kwargs is not None else {}
        assert self.datafilter_init_fn or self.filter_class, "One method of filter initialization should be specified"
        self.devices = devices
        self.num_parts = len(devices)

        # getting result columns names
        if self.datafilter_init_fn:
            datafilter = self.datafilter_init_fn(0, devices[0], self.datafilter_init_fn_kwargs)
        else:
            datafilter = self.filter_class(**self.filter_params, device=devices[0]) # type: ignore
        self._result_columns = datafilter.result_columns
        del datafilter
        torch.cuda.empty_cache()

    @property
    def result_columns(self) -> list[str]:
        return self._result_columns

    def run(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        connector: Connector,
        filter_run_kwargs: dict[str, Any]
    ) -> pd.DataFrame:
        """Renames columns in files of a dataset

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to process with datafilter
        config: DatasetConfig
            Config of that dataset
        connector: Connector
            Connector to use
        filter_run_kwargs: dict[str, Any]
            Parameters for datafilter.run method

        Returns
        -------
        pd.DataFrame
            Dataframe with new columns added
        """
        manager = Manager()
        shared_results = manager.list()

        df_splits = np.array_split(df, self.num_parts)
        params = []
        for i in range(self.num_parts):
            params.append(
                (
                    config,
                    connector,
                    df_splits[i],
                    i,
                    df_splits[i].index,  # type: ignore
                    shared_results,
                    self.filter_class,
                    self.filter_params,
                    self.datafilter_init_fn,
                    self.datafilter_init_fn_kwargs,
                    self.devices[i],
                    filter_run_kwargs
                )
            )

        processes = []
        context = multiprocessing.get_context('spawn')
        for param in params:
            p = context.Process(target=run_one_process, args=param)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        res_df = pd.concat(shared_results)
        res_df.sort_index(inplace=True)
        return res_df
