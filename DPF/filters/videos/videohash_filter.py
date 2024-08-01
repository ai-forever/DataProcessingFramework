import os
import shutil
from typing import Any
from uuid import uuid4

from videohash import VideoHash  # type: ignore

from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter


class VideohashFilter(VideoFilter):

    def __init__(
        self,
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers

    @property
    def result_columns(self) -> list[str]:
        return ["video_hash"]

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
        video_file = modality2data['video']

        uid = str(uuid4())
        tmp_dir = os.path.join(os.path.abspath(os.getcwd()), uid) + os.path.sep
        os.makedirs(tmp_dir, exist_ok=True)

        video_path = os.path.join(tmp_dir, 'video.mp4')
        with open(video_path, 'wb') as f:
            f.write(video_file)

        hash_obj = VideoHash(path=video_path, storage_path=tmp_dir)
        shutil.rmtree(hash_obj.storage_path)
        os.remove(video_path)
        os.rmdir(tmp_dir)
        return key, hash_obj.hash_hex

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()
        for key, hash_hex in batch:
            df_batch_labels[self.key_column].append(key)
            df_batch_labels[self.result_columns[0]].append(hash_hex)
        return df_batch_labels
