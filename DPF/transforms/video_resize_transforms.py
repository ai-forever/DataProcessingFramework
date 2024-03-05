import shutil
import subprocess
import uuid
from typing import List

from DPF.transforms.base_file_transforms import BaseFilesTransforms, TransformsFileData
from DPF.transforms.image_video_resizer import Resizer


def is_ffmpeg_installed():
    try:
        subprocess.run('ffmpeg -version', shell=True, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


class VideoResizeTransforms(BaseFilesTransforms):

    def __init__(
        self,
        resizer: Resizer,
        ffmpeg_preset: str = 'fast',
        pool_type: str = 'processes',
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(pool_type, workers, pbar)
        self.resizer = resizer
        self.ffmpeg_preset = ffmpeg_preset

        assert is_ffmpeg_installed(), "Please install ffmpeg"

    @property
    def required_metadata(self) -> List[str]:
        return ['width', 'height']

    @property
    def metadata_to_change(self) -> List[str]:
        return ['width', 'height']

    @property
    def modality(self) -> str:
        return 'video'

    def _process_filepath(self, data: TransformsFileData) -> TransformsFileData:
        filepath = data.filepath
        ext = filepath.split('.')[-1]
        width = data.metadata['width']
        height = data.metadata['height']

        new_width, new_height = self.resizer.get_new_size(width, height)
        if (new_width, new_height) != (width, height):
            new_width += new_width % 2
            new_height += new_height % 2
            temp_filename = str(uuid.uuid4())+'.'+ext
            ffmpeg_command = f'ffmpeg -i {filepath} -preset {self.ffmpeg_preset} -vf "scale={new_width}:{new_height}" {temp_filename} -y'
            subprocess.run(ffmpeg_command, shell=True, capture_output=True, check=True)
            shutil.move(temp_filename, filepath)

        return TransformsFileData(filepath, {'width': new_width, 'height': new_height})
