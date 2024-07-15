from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import soundfile as sf

from DPF.types import ModalityToDataMapping

from .audio_filter import AudioFilter


@dataclass
class AudioInfo:
    key: str
    is_correct: bool
    duration: Optional[float]
    sample_rate: Optional[int]
    error: Optional[str]


def get_audio_info(audio_bytes: bytes, data: dict[str, Any], key_column: str) -> AudioInfo:
    """
    Get info about audio
    """
    key = data[key_column]

    is_correct = True
    sample_rate, duration = None, None
    err_str = None

    try:
        file = sf.SoundFile(BytesIO(audio_bytes))

        sample_rate = file.samplerate
        duration = len(file) / sample_rate
    except Exception as err:
        is_correct = False
        err_str = str(err)

    return AudioInfo(key, is_correct, duration, sample_rate, err_str)


class AudioInfoFilter(AudioFilter):
    """
    Filter for gathering basic info about audios (width, height, number of channels)

    Parameters
    ----------
    workers: int = 16
        Number of parallel dataloader workers
    pbar: bool = True
        Whether to show progress bar
    """

    def __init__(self, workers: int = 16, pbar: bool = True, _pbar_position: int = 0):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers

    @property
    def result_columns(self) -> list[str]:
        return [
            "is_correct", "duration", "sample_rate", "error",
        ]

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
        return get_audio_info(modality2data['audio'], metadata, self.key_column)

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for image_info in batch:
            df_batch_labels[self.key_column].append(image_info.key)
            df_batch_labels["is_correct"].append(image_info.is_correct)
            df_batch_labels["duration"].append(image_info.duration)
            df_batch_labels["sample_rate"].append(image_info.sample_rate)
            df_batch_labels["error"].append(image_info.error)
        return df_batch_labels
