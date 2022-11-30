import os
import pandas as pd
from PIL import Image
import numpy as np
from clip_onnx import clip_onnx, attention
import numpy as np
    
from torch import nn

import torch
from torch.nn.utils.rnn import pad_sequence

import clip

from .t2ifilter import T2IFilter
from DPF.utils import read_image_rgb_from_bytes


class CLIPFilter(T2IFilter):
    
    def __init__(self, clip_version, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, device='cuda:0', workers=16, batch_size=64,
                 templates=['{}', 'photo of a {}', 'picture of a {}'], use_onnx=False, logit_scale=None):
        super(CLIPFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.templates = templates
        self.onnx = use_onnx
            
        self.clip_version = clip_version
        self.weights_folder = weights_folder
        
        if self.onnx:
            visual_path = os.path.join(self.weights_folder, f'{clip_version.replace("/", "_")}_visual.onnx')
            textual_path = os.path.join(self.weights_folder, f'{clip_version.replace("/", "_")}_textual.onnx')
            try:
                self.logit_scale = logit_scale
                _, self.clip_processor = clip.load(clip_version, device="cpu", jit=False, download_root=self.weights_folder)
                self.clip_model = clip_onnx(None)
                self.clip_model.load_onnx(visual_path=visual_path,
                                          textual_path=textual_path,
                                          logit_scale=self.logit_scale)
                self.clip_model.start_sessions(providers=['CUDAExecutionProvider'])
            except:
                self.clip_model, self.clip_processor = clip.load(clip_version, device="cpu", jit=False, download_root=self.weights_folder)
                self.logit_scale = self.clip_model.logit_scale.detach().cpu().numpy()
                image = np.random.rand(100, 100, 3) * 255
                image = self.clip_processor(Image.fromarray(image.astype('uint8')).convert('RGBA')).unsqueeze(0).cpu()
                text = clip.tokenize(['picture']).cpu()
                self.clip_model = clip_onnx(self.clip_model, visual_path=visual_path, textual_path=textual_path)
                self.clip_model.convert2onnx(image, text, verbose=True)
                self.clip_model.start_sessions(providers=['CUDAExecutionProvider'])
        else:
            self.clip_model, self.clip_processor = clip.load(clip_version, device=self.device, download_root=self.weights_folder)
        
        self.schema = ['image_path', 'clip_similarity']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=lambda x: x,
            drop_last=False, cols_to_return=['caption']
        )
        
    def preprocess(self, img_bytes, data):
        image_path = data['image_path']
        text = data['caption']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.clip_processor(pil_img)
        return image_path, img_tensor, text
    
    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()
        
        image_paths, image_tensors, batch_labels = list(zip(*batch))
        
        with torch.no_grad():
            if self.onnx:
                image_tensors = [t for t in image_tensors]
                inputs = {}
                inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True).detach().cpu().numpy().astype(np.float32)
                text_latents = []
                for template in self.templates:
                    for template in self.templates:
                        texts = clip.tokenize([template.format(class_label.lower().strip()) for class_label in batch_labels], truncate=True)
                        text_latents.append(self.clip_model.encode_text(texts.detach().cpu().numpy().astype(np.int32)))
                text_latents = np.stack(text_latents).mean(0)
                text_latents = text_latents / np.linalg.norm(text_latents, axis=-1, keepdims=True)
                batch_similarity = self.get_similarity(inputs, text_latents).tolist()          
            else:
                image_tensors = [t.to(self.device) for t in image_tensors]
                inputs = {}
                inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True)
                text_latents = []
                for template in self.templates:
                    texts = clip.tokenize([template.format(class_label.lower().strip()) for class_label in batch_labels], truncate=True)
                    text_latents.append(self.clip_model.encode_text(texts.to(self.device)))
                text_latents = torch.stack(text_latents).mean(0)
                text_latents = text_latents / text_latents.norm(dim=-1, keepdim=True)
                batch_similarity = self.get_similarity(inputs, text_latents).tolist()
                
                
        df_batch_labels['clip_similarity'].extend(batch_similarity)
        df_batch_labels['image_path'].extend(image_paths)
                
        return df_batch_labels
    
    def get_similarity(self, inputs, text_latents):
        if self.onnx:
            logit_scale = np.exp(self.logit_scale)
            image_latents = self.clip_model.encode_image(inputs['pixel_values'])
            image_latents = image_latents / np.linalg.norm(image_latents, axis=-1, keepdims=True)
            logits_per_image = np.matmul(image_latents, text_latents.T) * logit_scale
            probs = logits_per_image.tolist()
        else:
            with torch.no_grad():
                logit_scale = self.clip_model.logit_scale.exp()
                image_latents = self.clip_model.encode_image(inputs['pixel_values'])
                image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
                logits_per_image = torch.matmul(image_latents, text_latents.t()) * logit_scale
                probs = logits_per_image.cpu().numpy().tolist()

        return np.diag(probs)
