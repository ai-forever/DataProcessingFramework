from typing import List
from PIL import Image

from DPF.transforms.base_file_transforms import BaseFilesTransforms, TransformsFileArguments
from DPF.transforms.image_video_resizer import Resizer


class ImageResizeTransforms(BaseFilesTransforms):

    def __init__(
        self,
        resizer: Resizer,
        img_format: str = 'JPEG',
        pool_type: str = 'processes',
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(pool_type, workers, pbar)
        self.resizer = resizer
        self.img_format = img_format

    @property
    def required_metadata(self) -> List[str]:
        return []

    @property
    def modality(self) -> str:
        return 'image'

    def _process_filepath(self, data: TransformsFileArguments):
        filepath = data.filepath
        img = Image.open(filepath)
        width, height = self.resizer.get_new_size(img.width, img.height)

        if (width, height) != (img.width, img.height):
            img = img.resize((width, height))
            img.save(filepath, format=self.img_format)
