from typing import Any, Union

import torch

from DPF.types import ModalityToDataMapping

from .img_filter import ImageFilter


class DummyGPUFilter(ImageFilter):
    """
    DummyGPUFilter class for testing purposes
    """

    def __init__(
        self,
        workers: int = 16,
        device: Union[str, torch.device] = "cuda",
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.device = device

    @property
    def result_columns(self) -> list[str]:
        return ["dummy_label",]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        return key, 1

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, dummy_labels = list(zip(*batch))
        df_batch_labels[self.key_column].extend(keys)
        df_batch_labels[self.result_columns[0]].extend(dummy_labels)

        return df_batch_labels
