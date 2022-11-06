import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

import ruclip

from .t2ifilter import T2IFilter
from DPF.utils import read_image_rgb_from_bytes

def get_similarity(predictor, inputs, text_latents):
    with torch.no_grad():
        logit_scale = predictor.clip_model.logit_scale.exp()
        image_latents = predictor.clip_model.encode_image(inputs['pixel_values'])
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        logits_per_text = torch.matmul(text_latents.to(predictor.device), image_latents.t()) * logit_scale
        logits_per_text = logits_per_text.cpu().numpy()
    
    return np.diag(logits_per_text)


class RuCLIPFilter(T2IFilter):
    
    def __init__(self, ruclip_version, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, device='cuda:0', workers=16, batch_size=64,
                 templates=['{}', 'изображение с {}', 'фото с {}']):
        super(RuCLIPFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.templates = templates
            
        self.ruclip_version = ruclip_version
        self.weights_folder = weights_folder
        self.ruclip_model, self.ruclip_processor = ruclip.load(ruclip_version, device=device, cache_dir=weights_folder)
        self.ruclip_predictor = ruclip.Predictor(self.ruclip_model, self.ruclip_processor, 
                                                 device, bs=self.batch_size, templates=self.templates)
        
        self.schema = ['image_path', 'ruclip_similarity']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=lambda x: x,
            drop_last=False, cols_to_return=['caption']
        )
        
    def preprocess(self, img_bytes, data):
        image_path = data['image_path']
        text = data['caption']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.ruclip_processor.image_transform(pil_img)
        return image_path, img_tensor, text
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        image_paths, image_tensors, batch_labels = list(zip(*batch))
        image_tensors = [t.to(self.device) for t in image_tensors]
        
        with torch.no_grad():
            inputs = {}
            inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True)
            text_latents = self.ruclip_predictor.get_text_latents(batch_labels)
            batch_similarity = get_similarity(self.ruclip_predictor, inputs, text_latents).tolist()
            df_batch_labels['ruclip_similarity'].extend(batch_similarity)
            df_batch_labels['image_path'].extend(image_paths)
                
        return df_batch_labels