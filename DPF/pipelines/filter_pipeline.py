from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import pandas as pd
import torch

from DPF.filters import ColumnFilter, DataFilter
from DPF.filters.multigpu_filter import MultiGPUDataFilter
from DPF.processors import DatasetProcessor
from DPF.transforms import BaseFilesTransforms
from DPF.utils.logger import init_logger, init_stdout_logger

from .pipeline_stages import (
    DataFramePipelineStage,
    DeduplicationPipelineStage,
    FilterPipelineStage,
    PipelineStage,
    ShufflePipelineStage,
    TransformPipelineStage,
)
from .types import OnErrorOptions


@dataclass
class PipelineStageRunner:
    stage: PipelineStage
    on_error: OnErrorOptions


class FilterPipeline:

    def __init__(self, pipeline_name: str, logs_dir: Optional[str] = None):
        self.pipeline_name = pipeline_name
        self.stages: list[PipelineStageRunner] = []
        if logs_dir:
            self.logger = init_logger(pipeline_name, logging_dir=logs_dir)
        else:
            self.logger = init_stdout_logger()

    def add_datafilter(
        self,
        datafilter: Optional[type[DataFilter]] = None,
        datafilter_kwargs: dict[str, Any] = None,
        devices: Optional[list[str]] = None,
        processor_apply_kwargs: Optional[dict[str, Any]] = None,
        on_error: OnErrorOptions = "stop",
        skip_if_columns_exist: bool = True,
        constant_gpu: bool = False
    ) -> None:
        if processor_apply_kwargs is None:
            processor_apply_kwargs = {}

        if devices is None:
            stage = FilterPipelineStage(
                'datafilter', filter_class=datafilter,
                filter_kwargs=datafilter_kwargs, processor_apply_kwargs=processor_apply_kwargs,
                skip_if_columns_exist=skip_if_columns_exist, constant_gpu=constant_gpu
            )
        elif len(devices) == 0:
            new_kwargs = datafilter_kwargs.copy()
            new_kwargs['device'] = devices[0]
            stage = FilterPipelineStage(
                'datafilter', filter_class=datafilter,
                filter_kwargs=new_kwargs, processor_apply_kwargs=processor_apply_kwargs,
                skip_if_columns_exist=skip_if_columns_exist, constant_gpu=constant_gpu
            )
        else:
            stage = FilterPipelineStage(
                'multigpufilter', filter_class=MultiGPUDataFilter,
                filter_kwargs={
                    "devices": devices,
                    "datafilter_class": datafilter,
                    "datafilter_params": datafilter_kwargs
                    },
                processor_apply_kwargs=processor_apply_kwargs,
                skip_if_columns_exist=skip_if_columns_exist,
                constant_gpu=constant_gpu
            )

        self.stages.append(
            PipelineStageRunner(stage, on_error=on_error)
        )

    def add_columnfilter(
        self,
        columnfilter: type[ColumnFilter],
        columnfilter_kwargs: dict[str, Any],
        processor_apply_kwargs: Optional[dict[str, Any]] = None,
        on_error: OnErrorOptions = "stop",
        skip_if_columns_exist: bool = True
    ) -> None:
        if processor_apply_kwargs is None:
            processor_apply_kwargs = {}

        stage = FilterPipelineStage(
            'columnfilter', filter_class=columnfilter,
            filter_kwargs=columnfilter_kwargs, processor_apply_kwargs=processor_apply_kwargs,
            skip_if_columns_exist=skip_if_columns_exist
        )

        self.stages.append(
            PipelineStageRunner(stage, on_error=on_error)
        )

    def add_shuffle(self) -> None:
        stage = ShufflePipelineStage()
        self.stages.append(
            PipelineStageRunner(stage, on_error="stop")
        )

    def add_deduplication(
        self,
        columns: list[str],
        on_error: OnErrorOptions = "stop"
    ) -> None:
        stage = DeduplicationPipelineStage(columns)
        self.stages.append(
            PipelineStageRunner(stage, on_error=on_error)
        )

    def add_dataframe_filter(
        self,
        filter_func: Callable[[pd.DataFrame], pd.DataFrame],
        on_error: OnErrorOptions = "stop"
    ) -> None:
        stage = DataFramePipelineStage(filter_func)
        self.stages.append(
            PipelineStageRunner(stage, on_error=on_error)
        )

    def add_transforms(
        self,
        transforms_class: type[BaseFilesTransforms],
        transforms_kwargs: dict[str, Any],
        processor_apply_kwargs: Optional[dict[str, Any]] = None,
        on_error: OnErrorOptions = "stop"
    ) -> None:
        stage = TransformPipelineStage(
            transforms_class, transforms_kwargs,
            processor_apply_kwargs=processor_apply_kwargs
        )
        self.stages.append(
            PipelineStageRunner(stage, on_error=on_error)
        )

    def _log_dataset_info(self, processor: DatasetProcessor) -> None:
        self.logger.info(f'Dataset path: {processor.config.path}')
        self.logger.info(f'Dataset modalities: {processor.modalities}')
        self.logger.info(f'Dataset size: {processor.df.shape}')
        self.logger.info(f'Dataset columns: {processor.df.columns}')

    def run(self, processor: DatasetProcessor) -> None:
        self.logger.info(f'Starting filtering dataset {processor.config.path} with {self.pipeline_name} pipeline')
        self._log_dataset_info(processor)
        for i, stage_runner in enumerate(self.stages):
            self.logger.info("-"*16)
            self.logger.info(f"Starting stage {i}: {stage_runner.stage.stage_name}")
            try:
                stage_runner.stage.run(processor, self.logger)
            except Exception as err:
                self.logger.exception(f"Error occured during filtering: {err}")
                if stage_runner.on_error == "stop":
                    self.logger.warning('Stopping pipeline')
                    raise err
                else:
                    self.logger.warning('Continue')
            else:
                self.logger.info(f"Pipeline stage finished. New dataframe shape: {processor.df.shape}")
