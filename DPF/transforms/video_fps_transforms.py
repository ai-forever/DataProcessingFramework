import shutil
import subprocess
import uuid

from DPF.transforms.base_file_transforms import (
    BaseFilesTransforms,
    PoolOptions,
    TransformsFileData,
)
from DPF.transforms.video_resize_transforms import is_ffmpeg_installed


class VideoFPSTransforms(BaseFilesTransforms):

    def __init__(
        self,
        fps: int,
        eps: float = 0.1,
        pool_type: PoolOptions = 'threads',
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(pool_type, workers, pbar)
        self.fps = fps
        self.eps = eps

        assert is_ffmpeg_installed(), "Install ffmpeg first"

    @property
    def required_metadata(self) -> list[str]:
        return ['fps']

    @property
    def metadata_to_change(self) -> list[str]:
        return ['fps']

    @property
    def modality(self) -> str:
        return 'video'

    def _process_filepath(self, data: TransformsFileData) -> TransformsFileData:
        filepath = data.filepath
        ext = filepath.split('.')[-1]
        video_fps = data.metadata['fps']

        if (video_fps < (self.fps - self.eps)) or (video_fps > (self.fps + self.eps)):
            temp_filename = str(uuid.uuid4()) + '.' + ext
            ffmpeg_command = f'ffmpeg -hide_banner -i {filepath} -vf fps={self.fps} {temp_filename} -y'
            subprocess.run(ffmpeg_command, shell=True, capture_output=True, check=True)
            shutil.move(temp_filename, filepath)

        return TransformsFileData(filepath, {'fps': self.fps})
