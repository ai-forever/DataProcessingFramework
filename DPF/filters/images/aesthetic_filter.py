import os
from urllib.request import urlretrieve
import torch
from torch import nn
import clip
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from .img_filter import ImageFilter


def get_aesthetic_model(clip_model, cache_folder):
    """
    Load the aethetic model
    """

    if clip_model == 'ViT-L/14':
        clip_model = 'vit_l_14'
        m = nn.Linear(768, 1)
    elif clip_model == 'ViT-B/32':
        clip_model = 'vit_b_32'
        m = nn.Linear(512, 1)
    else:
        raise ValueError("Unsupported clip model")

    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model + "_linear.pth?raw=true"
        )
        # TODO rework download
        urlretrieve(url_model, path_to_model)

    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


class AestheticFilter(ImageFilter):
    """
    AestheticFilter class
    """

    def __init__(
            self,
            clip_model: str,
            weights_folder: str,
            device: str = 'cuda:0',
            workers: int = 16,
            batch_size: int = 64,
            pbar: bool = True
    ):
        super().__init__(pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.weights_folder = weights_folder
        self.clip_model, self.clip_transforms = clip.load(clip_model, device=self.device,
                                                          download_root=weights_folder)
        self.aesthetic_model = get_aesthetic_model(clip_model, weights_folder)
        self.aesthetic_model.to(self.device)

        self.schema = ['image_path', 'aesthetic_score']
        self.dataloader_kwargs = {
            'num_workers': self.num_workers, 'batch_size': self.batch_size,
            'preprocess_f': self.preprocess, 'collate_fn': identical_collate_fn,
            'drop_last': False
        }

    def preprocess(self, img_bytes, data):
        image_path = data['image_path']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.clip_transforms(pil_img)
        return image_path, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        image_paths, image_tensors = list(zip(*batch))
        batch = default_collate(image_tensors).to(self.device)

        with torch.no_grad():
            inputs = self.clip_model.encode_image(batch)
            outputs = self.aesthetic_model(inputs.float())
        df_batch_labels['aesthetic_score'].extend(outputs.cpu().reshape(-1).tolist())
        df_batch_labels['image_path'].extend(image_paths)

        return df_batch_labels
