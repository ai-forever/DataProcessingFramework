import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import imageio.v3 as iio

from .video_filter import VideoFilter


@dataclass
class VideoInfo:
    key: str
    is_correct: bool
    width: int
    height: int
    fps: float
    duration: float
    error: Optional[str]


def get_video_info(video_bytes, data, key_column) -> VideoInfo:
    """
    Get image path, read status, width, height, num channels, read error
    """
    key = data[key_column]

    is_correct = True
    width, height, fps, duration = None, None, None, None
    err_str = None

    try:
        meta = iio.immeta(io.BytesIO(video_bytes), plugin="pyav")
        frame = iio.imread(io.BytesIO(video_bytes), index=0, plugin="pyav")
        fps = meta['fps']
        duration = meta['duration']
        height, width = frame.shape[:2]
    except Exception as err:
        is_correct = False
        err_str = str(err)

    return VideoInfo(key, is_correct, width, height, fps, duration, err_str)


class VideoInfoFilter(VideoFilter):
    """
    VideoInfoFilter class
    """

    def __init__(self, workers: int = 16, pbar: bool = True, _pbar_position: int = 0):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers

    @property
    def schema(self) -> List[str]:
        return [
            self.key_column, "is_correct", "error",
            "width", "height", "fps", "duration"
        ]

    @property
    def dataloader_kwargs(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict) -> VideoInfo:
        return get_video_info(modality2data['video'], metadata, self.key_column)

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        for video_info in batch:
            df_batch_labels[self.key_column].append(video_info.name)
            df_batch_labels["is_correct"].append(video_info.is_correct)
            df_batch_labels["error"].append(video_info.error)
            df_batch_labels["width"].append(video_info.width)
            df_batch_labels["height"].append(video_info.height)
            df_batch_labels["fps"].append(video_info.fps)
            df_batch_labels["duration"].append(video_info.duration)
        return df_batch_labels
