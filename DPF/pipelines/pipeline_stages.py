import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import pandas as pd

from DPF.filters import ColumnFilter, DataFilter
from DPF.filters.multigpu_filter import MultiGPUDataFilter
from DPF.processors import DatasetProcessor

from .types import FilterTypes


class PipelineStage(ABC):

    @property
    @abstractmethod
    def stage_name(self) -> str:
        pass

    @abstractmethod
    def run(self, processor: DatasetProcessor, logger: logging.Logger) -> None:
        pass


class ShufflePipelineStage(PipelineStage):

    @property
    def stage_name(self) -> str:
        return "ShufflePipelineStage"

    def run(self, processor: DatasetProcessor, logger: logging.Logger) -> None:
        processor._df = processor.df.sample(frac=1).reset_index(drop=True)


class DataFramePipelineStage(PipelineStage):

    def __init__(self, filter_func: Callable[[pd.DataFrame], pd.DataFrame]):
        self.filter_func = filter_func

    @property
    def stage_name(self) -> str:
        return "ConditionFilterPipelineStage"

    def run(self, processor: DatasetProcessor, logger: logging.Logger) -> None:
        processor._df = self.filter_func(processor.df)


class DeduplicationPipelineStage(PipelineStage):

    def __init__(self, columns_to_dedup: list[str]):
        self.columns_to_dedup = columns_to_dedup

    @property
    def stage_name(self) -> str:
        return f"DeduplicationPipelineStage(columns={self.columns_to_dedup})"

    def run(self, processor: DatasetProcessor, logger: logging.Logger) -> None:
        processor._df = processor.df.drop_duplicates(self.columns_to_dedup)


class FilterPipelineStage(PipelineStage):

    def __init__(
        self,
        filter_type: FilterTypes,
        filter_class: Union[type[DataFilter], type[ColumnFilter], type[MultiGPUDataFilter]],
        filter_kwargs: dict[str, Any],
        processor_run_kwargs: dict[str, Any],
        skip_if_columns_exist: bool = True
    ):
        self.filter_type = filter_type
        self.filter_class = filter_class
        self.filter_kwargs = filter_kwargs
        self.processor_run_kwargs = processor_run_kwargs
        self.skip_if_columns_exist = skip_if_columns_exist

    @property
    def stage_name(self) -> str:
        return f"FilterPipelineStage(filter_class={self.filter_class}, filter_kwargs={self.filter_kwargs})"

    def run(self, processor: DatasetProcessor, logger: logging.Logger) -> None:
        filter_obj = self.filter_class(**self.filter_kwargs)

        columns_to_be_added = filter_obj.result_columns
        columns_intersection = set(processor.columns).intersection(set(columns_to_be_added))
        if columns_intersection == set(columns_to_be_added):
            if self.skip_if_columns_exist:
                logger.info(f"All columns are presented in a dataset, skipping filtering")
                return
            else:
                logger.info(f"All columns are presented in a dataset, force rewriting them")

        if len(columns_intersection) > 0:
            logger.info(f"Dropping existing columns: {list(columns_intersection)}")
            processor.df.drop(columns=columns_to_be_added, inplace=True, errors='ignore')

        if self.filter_type == 'datafilter':
            processor.apply_data_filter(filter_obj, **self.processor_run_kwargs)  # type: ignore
        elif self.filter_type == 'columnfilter':
            processor.apply_column_filter(filter_obj, **self.processor_run_kwargs)  # type: ignore
        elif self.filter_type == 'multigpufilter':
            processor.apply_multi_gpu_data_filter(filter_obj, **self.processor_run_kwargs)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
