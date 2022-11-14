import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
from .img_filter import ImageFilter
from DPF.utils import read_image_rgb_from_bytes
import clip
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate


def get_aesthetic_model(clip_model, cache_folder):
    """load the aethetic model"""
    if clip_model == 'ViT-L/14':
        clip_model = 'vit_l_14'
    elif clip_model == 'ViT-B/32':
        clip_model = 'vit_b_32'
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == 'vit_l_14':
        m = nn.Linear(768, 1)
    elif clip_model == 'vit_b_32':
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


class AestheticFilter(ImageFilter):

    def __init__(self, clip_model, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, workers=16, batch_size=64, device='cuda:0'):
        super(AestheticFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.clip_model, self.clip_transforms = clip.load(clip_model, device=self.device)
        self.aesthetic_model = get_aesthetic_model(clip_model, weights_folder)
        self.aesthetic_model.to(self.device)

        self.schema = ['image_path', 'aesthetic_score']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=self.collate_fn,
            drop_last=False
        )

    @staticmethod
    def collate_fn(x):
        return x

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
