import io
from collections.abc import Iterable
from typing import Any, Callable

import imageio.v3 as iio
from PIL import Image

from DPF.filters.images.img_filter import ImageFilter
from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter

Image.MAX_IMAGE_PIXELS = None 

class ImageFilterAdapter(VideoFilter):
    """
    Runs an ImageFilter on one frame from video

    Parameters
    ----------
    image_filter: ImageFilter
        Image filter to apply
    video_frame: float
        Position of frame to use
        For example 0 means first frame, 0.5 means central frame and 1 means last frame
    workers: int = 8
        Number of pytorch dataloader workers
    pbar: bool = True
        Whether to show progress bar
    """

    def __init__(
        self,
        image_filter: ImageFilter,
        video_frame: float,
        workers: int = 8,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.image_filter = image_filter
        self.video_frame = video_frame
        self.num_workers = workers

    @property
    def result_columns(self) -> list[str]:
        return self.image_filter.result_columns

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

        video_bytes = modality2data['video']
        meta = iio.immeta(io.BytesIO(video_bytes), plugin="pyav")
        fps = meta['fps']
        duration = meta['duration']
        total_frames = int(fps*duration)
        frame_index = min(int(total_frames*self.video_frame), total_frames-1)
        frame = iio.imread(io.BytesIO(video_bytes), index=frame_index, plugin="pyav")

        buff = io.BytesIO()
        Image.fromarray(frame).convert('RGB').save(buff, format='JPEG', quality=95)  # type: ignore
        modality2data['image'] = buff.getvalue()
        metadata[self.image_filter.key_column] = ''
        return key, self.image_filter.preprocess_data(modality2data, metadata)

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        for key, data in batch:
            df_batch_labels_images = self.image_filter.process_batch([data])
            df_batch_labels[self.key_column].append(key)
            for colname in self.schema[1:]:
                df_batch_labels[colname].extend(df_batch_labels_images[colname])
        return df_batch_labels


def chunks(lst: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MultiFrameImageFilterAdapter(VideoFilter):
    """
    Runs an ImageFilter on several frames from video

    Parameters
    ----------
    image_filter: ImageFilter
        Image filter to apply
    video_frames: list[float]
        List of positions of frames to use
        For example 0 means first frame, 0.5 means central frame and 1 means last frame
    reduce_results_fn: Callable[[str, list[Any]], Any]
        Function to convert an array of outputs into a single output.
        Takes column name as a first arg and a list of outputs for every frame as a second argument.
        Returns one value that will be used in that column.
    batch_size: int = 8
        Batch size for model (will only create batches within one video sample)
    workers: int = 8
        Number of pytorch dataloader workers
    pbar: bool = True
        Whether to show progress bar
    """

    def __init__(
        self,
        image_filter: ImageFilter,
        video_frames: list[float],
        reduce_results_fn: Callable[[str, list[Any]], Any],
        batch_size: int = 8,
        workers: int = 8,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.image_filter = image_filter
        self.video_frames = video_frames
        self.reduce_results_fn = reduce_results_fn
        self.batch_size = batch_size
        self.num_workers = workers

    @property
    def result_columns(self) -> list[str]:
        return self.image_filter.result_columns

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

        video_bytes = modality2data['video']
        meta = iio.immeta(io.BytesIO(video_bytes), plugin="pyav")
        fps = meta['fps']
        duration = meta['duration']
        total_frames = int(fps*duration)

        preprocessed_data = []
        for video_frame_pos in self.video_frames:
            frame_index = min(int(total_frames*video_frame_pos), total_frames-1)
            frame = iio.imread(io.BytesIO(video_bytes), index=frame_index, plugin="pyav")

            buff = io.BytesIO()
            Image.fromarray(frame).convert('RGB').save(buff, format='JPEG', quality=95)  # type: ignore
            modality2data['image'] = buff.getvalue()
            metadata[self.image_filter.key_column] = ''

            preprocessed_data.append(self.image_filter.preprocess_data(modality2data, metadata))

        return key, preprocessed_data

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        key, data = batch[0]
        df_batch_labels_images = self._get_dict_from_schema()
        for batched_preprocessed_data in chunks(data, self.batch_size):
            df_batch_labels_images_batch = self.image_filter.process_batch(batched_preprocessed_data)
            for colname in self.result_columns:
                df_batch_labels_images[colname].extend(df_batch_labels_images_batch[colname])

        df_batch_labels[self.key_column].append(key)
        for colname in self.result_columns:
            df_batch_labels[colname].extend([self.reduce_results_fn(colname, df_batch_labels_images[colname])])
        return df_batch_labels
