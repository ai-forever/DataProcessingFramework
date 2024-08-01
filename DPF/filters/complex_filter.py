from dataclasses import dataclass
from typing import Any

from DPF.filters.data_filter import DataFilter
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping


@dataclass
class ComplexFilterPreprocessedData:
    key: str
    preprocessed_values: dict[int, Any]


class ComplexDataFilter(DataFilter):

    def __init__(
        self,
        datafilters: list[DataFilter],
        workers: int,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.datafilters = datafilters
        self.workers = workers

        assert len(self.datafilters) > 0
        assert all(
            i.key_column == self.datafilters[0].key_column for i in self.datafilters
        )  # check all filters have same key col

    @property
    def modalities(self) -> list[ModalityName]:
        modals = []
        for datafilter in self.datafilters:
            modals.extend(datafilter.modalities)
        return list(set(modals))

    @property
    def key_column(self) -> str:
        return self.datafilters[0].key_column

    @property
    def metadata_columns(self) -> list[str]:
        meta_cols = []
        for datafilter in self.datafilters:
            meta_cols.extend(datafilter.metadata_columns)
        return list(set(meta_cols))

    @property
    def result_columns(self) -> list[str]:
        result_cols = []
        for datafilter in self.datafilters:
            result_cols.extend(datafilter.result_columns)
        return list(set(result_cols))

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        preprocessed_results = {}
        for ind, datafilter in enumerate(self.datafilters):
            preprocessed_results[ind] = datafilter.preprocess_data(modality2data, metadata)
        return ComplexFilterPreprocessedData(key, preprocessed_results)

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        results = {}
        preprocessed_data: ComplexFilterPreprocessedData = batch[0]
        for ind, datafilter in enumerate(self.datafilters):
            filter_results_data = datafilter.process_batch([preprocessed_data.preprocessed_values[ind]])
            for col, value in filter_results_data.items():
                results[col] = value
        results[self.key_column] = [preprocessed_data.key]

        return results
