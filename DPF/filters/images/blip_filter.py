from .img_filter import ImageFilter
from PIL import Image
import torch
import urllib
import os
import sys
import git
from DPF.utils import read_image_rgb_from_bytes
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate


class BLIPFilter(ImageFilter):

    def __init__(self, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, workers=16, batch_size=64, device='cuda:0'):
        super(BLIPFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.weights_folder = weights_folder
        
        if not os.path.exists(os.path.join(self.weights_folder, 'BLIP')):
            os.path.makedirs(self.weights_folder, exist_ok=True)
            git.Git(self.weights_folder).clone('https://github.com/salesforce/BLIP.git')
        if not os.path.exists(os.path.join(self.weights_folder, 'BLIP_model_large.pth')):
            urllib.request.urlretrieve('https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
                                       filename=os.path.join(self.weights_folder, 'BLIP_model_large.pth'), reporthook=None, data=None)
        sys.path.append(os.path.join(self.weights_folder, 'BLIP'))
        from BLIP.models.blip import blip_decoder

        self.blip_model = blip_decoder(pretrained=os.path.join(self.weights_folder, 'BLIP_model_large.pth'), image_size=384, vit='large')
        self.blip_model.eval()
        self.blip_model = self.blip_model.to(device)
        self.blip_processor = transforms.Compose([transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                       (0.26862954, 0.26130258, 0.27577711))])

        self.schema = ['image_path', 'blip_caption']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=self.collate_fn,
            drop_last=False
        )

    def preprocess(self, img_bytes, data):
        image_path = data['image_path']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.blip_processor(pil_img)
        return image_path, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        image_paths, image_tensors = list(zip(*batch))

        with torch.no_grad():
            batch = default_collate(image_tensors).to(self.device)
            captions = self.blip_model.generate(batch, sample=False, num_beams=3, max_length=50, min_length=20)

        df_batch_labels['blip_caption'].extend(captions)
        df_batch_labels['image_path'].extend(image_paths)

        return df_batch_labels
