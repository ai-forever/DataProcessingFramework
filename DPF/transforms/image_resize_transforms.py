
from PIL import Image

from DPF.transforms.base_file_transforms import (
    BaseFilesTransforms,
    PoolOptions,
    TransformsFileData,
)
from DPF.transforms.resizer import Resizer

Image.MAX_IMAGE_PIXELS = None 

class ImageResizeTransforms(BaseFilesTransforms):

    def __init__(
        self,
        resizer: Resizer,
        img_format: str = 'JPEG',
        pool_type: PoolOptions = 'processes',
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(pool_type, workers, pbar)
        self.resizer = resizer
        self.img_format = img_format

    @property
    def required_metadata(self) -> list[str]:
        return []

    @property
    def metadata_to_change(self) -> list[str]:
        return ['width', 'height']

    @property
    def modality(self) -> str:
        return 'image'

    def _process_filepath(self, data: TransformsFileData) -> TransformsFileData:
        filepath = data.filepath
        img = Image.open(filepath)
        width, height = self.resizer.get_new_size(img.width, img.height)

        if (width, height) != (img.width, img.height):
            img = img.resize((width, height))
            img.save(filepath, format=self.img_format)

        return TransformsFileData(filepath, {'width': width, 'height': height})
