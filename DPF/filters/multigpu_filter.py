import multiprocessing
from multiprocessing import Manager
from typing import Any, Union

import numpy as np
import pandas as pd
import torch

from DPF.configs import DatasetConfig
from DPF.connectors import Connector
from DPF.dataset_reader import DatasetReader

from .data_filter import DataFilter


class MultiGPUDataFilter:
    """
    Class for multi-gpu inference
    """

    def __init__(
        self,
        devices: list[Union[torch.device, str]],
        datafilter_class: type[DataFilter],
        datafilter_params: dict[str, Any]
    ):
        """
        Parameters
        ----------
        devices: list[Union[torch.device, str]]
            List of devices to run datafilter on
        datafilter_class: type[DataFilter]
            Class of datafilter to use
        datafilter_params: dict[str, Any]
            Parameters for datafilter_class initialization
        """
        self.filter_class = datafilter_class
        self.filter_params = datafilter_params
        self.devices = devices
        self.num_parts = len(devices)

        self.filters = []
        for i in range(self.num_parts):
            self.filters.append(datafilter_class(**datafilter_params, _pbar_position=i, device=devices[i]))
            self.filters[i]._created_by_multigpu_data_filter = True

        # getting result columns names
        self._result_columns = self.filters[0].result_columns
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
                    filter_run_kwargs
                )
            )

        processes = []
        context = multiprocessing.get_context('spawn')
        for param in params:
            p = context.Process(target=self.run_one_process, args=param)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        res_df = pd.concat(shared_results)
        res_df.sort_index(inplace=True)
        return res_df
    
    
    def run_one_process(
        self,
        config: DatasetConfig,
        connector: Connector,
        df: pd.DataFrame,
        i: int,
        index: pd.Series,
        results: list[pd.DataFrame],
        filter_run_kwargs: dict[str, Any]
    ) -> None:
        reader = DatasetReader(connector=connector)
        processor = reader.from_df(config, df)
        processor.apply_data_filter(self.filters[i], **filter_run_kwargs)
        res = processor.df
        res.set_index(index, inplace=True)
        results.append(res)
