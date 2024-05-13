import shutil
import subprocess
import uuid
from typing import Any, Optional

from DPF.transforms.base_file_transforms import (
    BaseFilesTransforms,
    PoolOptions,
    TransformsFileData,
)
from DPF.transforms.resizer import Resizer


def is_ffmpeg_installed() -> bool:
    try:
        subprocess.run('ffmpeg -version', shell=True, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def convert_ffmpeg_args_to_str(args: dict[str, list[str]]) -> str:
    arg_strings = []
    for k, v in args.items():
        s = ','.join(v)
        arg_strings.append(f"{k} {s}")
    return ' '.join(arg_strings)


class VideoFFMPEGTransforms(BaseFilesTransforms):

    def __init__(
        self,
        resizer: Optional[Resizer] = None,
        fps: Optional[int] = None,
        fps_eps: float = 0.1,
        cut_start_col: Optional[str] = None,
        cut_duration_col: Optional[str] = None,
        copy_stream: bool = False,
        preset: Optional[str] = None,
        crf: Optional[int] = None,
        copy_audio_stream: bool = True,
        pool_type: PoolOptions = 'threads',
        workers: int = 16,
        pbar: bool = True,
    ):
        super().__init__(pool_type, workers, pbar)
        self.resizer = resizer
        self.fps = fps
        self.fps_eps = fps_eps
        self.cut_start_col = cut_start_col
        self.cut_duration_col = cut_duration_col
        if self.cut_duration_col or self.cut_start_col:
            assert self.cut_duration_col and self.cut_start_col, f"Both {self.cut_duration_col} and {self.cut_start_col} must be specified"
        self.copy_when_cut = copy_stream
        if self.copy_when_cut:
            assert self.copy_when_cut and not (self.fps or self.resizer), "Copy stream can be used only for cutting videos"

        self.preset = preset
        self.crf = crf
        self.copy_audio_stream = copy_audio_stream

        self.default_args = ' '.join(self.get_default_ffmpeg_args())

        assert is_ffmpeg_installed(), "Install ffmpeg first"
        assert self.resizer or self.fps or self.cut_start_col, "At least one transform should be specified"

    def get_default_ffmpeg_args(self) -> list[str]:
        args = []
        if self.preset:
            args.append(f"-preset {self.preset}")
        if self.crf:
            args.append(f'-crf {self.crf}')
        if self.copy_audio_stream:
            args.append('-c:a copy')
        return args

    @property
    def required_metadata(self) -> list[str]:
        meta = []
        if self.resizer:
            meta += ['width', 'height']
        if self.fps:
            meta += ['fps']
        if self.cut_duration_col and self.cut_start_col:
            meta += [self.cut_start_col, self.cut_duration_col]
        return meta

    @property
    def metadata_to_change(self) -> list[str]:
        meta = []
        if self.resizer:
            meta += ['width', 'height']
        if self.fps:
            meta += ['fps']
        return meta

    @property
    def modality(self) -> str:
        return 'video'

    def _process_filepath(self, data: TransformsFileData) -> TransformsFileData:
        filepath = data.filepath
        ext = filepath.split('.')[-1]
        ffmpeg_args_start: list[str] = []
        ffmpeg_args_map: dict[str, list[str]] = {}
        result_metadata: dict[str, Any] = {}

        if self.resizer:
            width, height = data.metadata['width'], data.metadata['height']
            new_width, new_height = self.resizer.get_new_size(width, height)
            if (new_width, new_height) != (width, height):
                new_width += new_width % 2
                new_height += new_height % 2
                ffmpeg_args_map['-vf'] = ffmpeg_args_map.get('-vf', []) + [f"scale={new_width}:{new_height}"]
            result_metadata['width'] = new_width
            result_metadata['height'] = new_height

        if self.fps:
            video_fps = data.metadata['fps']
            if (video_fps < (self.fps - self.fps_eps)) or (video_fps > (self.fps + self.fps_eps)):
                ffmpeg_args_map['-vf'] = ffmpeg_args_map.get('-vf', []) + [f"fps={self.fps}"]
                video_fps = float(self.fps)
            result_metadata['fps'] = video_fps

        if self.cut_start_col and self.cut_duration_col and data.metadata[self.cut_start_col] is not None:
            start = data.metadata[self.cut_start_col]
            cut_duration = data.metadata[self.cut_duration_col]
            ffmpeg_args_start.append(f'-ss {start}')
            ffmpeg_args_map['-t'] = [str(cut_duration)]
            if self.copy_when_cut:
                ffmpeg_args_map['-c'] = ['copy']
                ffmpeg_args_map['-avoid_negative_ts'] = ['1']

        if len(ffmpeg_args_map) > 0:
            args_str = convert_ffmpeg_args_to_str(ffmpeg_args_map)
            args_start_str = ' '.join(ffmpeg_args_start)
            temp_filename = str(uuid.uuid4()) + '.' + ext
            ffmpeg_command = f'ffmpeg -hide_banner {args_start_str} -i {filepath} {args_str} {self.default_args} {temp_filename} -y'
            print(ffmpeg_command)
            subprocess.run(ffmpeg_command, shell=True, capture_output=True, check=True)
            shutil.move(temp_filename, filepath)

        return TransformsFileData(filepath, result_metadata)
