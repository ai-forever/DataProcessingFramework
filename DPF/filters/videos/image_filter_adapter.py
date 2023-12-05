from typing import Dict, Union, List
import imageio.v3 as iio
import io

from DPF.filters.images.img_filter import ImageFilter
from DPF.filters.utils import identical_collate_fn
from .video_filter import VideoFilter


def get_video_info(video_bytes, data, key_column):
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

    return key, is_correct, width, height, fps, duration, err_str


class VideoInfoFilter(VideoFilter):
    """
    VideoInfoFilter class
    """

    def __init__(
        self,
        image_filter: ImageFilter,
        video_frames: List[float],
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(pbar)
        self.image_filter = image_filter
        self.num_workers = workers

        self.schema = [
            self.key_column,
            "is_correct",
            "error",
            "width",
            "height",
            "fps",
            "duration"
        ]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "collate_fn": identical_collate_fn,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        return get_video_info(modality2data['video'], metadata, self.key_column)

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        for data in batch:
            key, is_correct, width, height, fps, duration, error = data
            df_batch_labels[self.key_column].append(key)
            df_batch_labels["is_correct"].append(is_correct)
            df_batch_labels["error"].append(error)
            df_batch_labels["width"].append(width)
            df_batch_labels["height"].append(height)
            df_batch_labels["fps"].append(fps)
            df_batch_labels["duration"].append(duration)
        return df_batch_labels
