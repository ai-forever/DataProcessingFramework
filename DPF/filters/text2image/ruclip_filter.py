import os
import pandas as pd
from PIL import Image
import numpy as np
from clip_onnx import clip_onnx, attention
import numpy as np
    
from torch import nn

import torch
from torch.nn.utils.rnn import pad_sequence

import ruclip

from .t2ifilter import T2IFilter
from DPF.utils import read_image_rgb_from_bytes


class RuCLIPFilter(T2IFilter):
    
    def __init__(self, ruclip_version, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, device='cuda:0', workers=16, batch_size=64,
                 templates=['{}', 'изображение с {}', 'фото с {}'], use_onnx=False, logit_scale=None):
        super(RuCLIPFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)
        
        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        self.templates = templates
        self.onnx = use_onnx
            
        self.ruclip_version = ruclip_version
        self.weights_folder = weights_folder
        
        if self.onnx:
            
            DEFAULT_EXPORT = dict(input_names=['input'], output_names=['output'],
                                  export_params=True, verbose=False, opset_version=15,
                                  do_constant_folding=True,
                                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})            
            
            visual_path = os.path.join(self.weights_folder, f'{ruclip_version.replace("/", "_")}_visual.onnx')
            textual_path = os.path.join(self.weights_folder, f'{ruclip_version.replace("/", "_")}_textual.onnx')
            try:
                self.logit_scale = logit_scale
                _, self.ruclip_processor = ruclip.load(ruclip_version, device='cpu', cache_dir=self.weights_folder)
                self.ruclip_model = clip_onnx(None)
                self.ruclip_model.load_onnx(visual_path=visual_path,
                                            textual_path=textual_path,
                                            logit_scale=self.logit_scale)
                self.ruclip_model.start_sessions(providers=['CUDAExecutionProvider'])
            except:
                self.ruclip_model, self.ruclip_processor = ruclip.load(ruclip_version, device='cpu', cache_dir=self.weights_folder)
                self.logit_scale = self.ruclip_model.logit_scale.detach().cpu().numpy()
                image = np.random.rand(100, 100, 3) * 255
                images = [Image.fromarray(image.astype('uint8')).convert('RGBA')]
                labels = ['изображение']
                dummy_input = self.ruclip_processor(text=labels, images=images, return_tensors='pt', padding=True)
                image = dummy_input["pixel_values"].cpu()
                text = dummy_input["input_ids"].cpu()

                self.ruclip_model = clip_onnx(self.ruclip_model, visual_path=visual_path, textual_path=textual_path)
                self.ruclip_model.convert2onnx(image, text, verbose=True,
                                               textual_wrapper=Textual,
                                               textual_export_params=DEFAULT_EXPORT,
                                               visual_export_params=DEFAULT_EXPORT)
                self.ruclip_model.start_sessions(providers=["CUDAExecutionProvider"])
        else:
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
        
        with torch.no_grad():
            if self.onnx:
                image_tensors = [t for t in image_tensors]
                inputs = {}
                inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True).detach().cpu().numpy().astype(np.float32)
                text_latents = []
                for template in self.templates:
                    texts = self.ruclip_processor(text=[template.format(class_label.lower().strip()) for class_label in batch_labels],
                                                  return_tensors='pt', padding=True)
                    text_latents.append(self.ruclip_model.encode_text(texts["input_ids"].detach().cpu().numpy().astype(np.int64)))
                text_latents = np.stack(text_latents).mean(0)
                text_latents = text_latents / np.linalg.norm(text_latents, axis=-1, keepdims=True)
                batch_similarity = self.get_similarity(inputs, text_latents).tolist()          
            else:
                image_tensors = [t.to(self.device) for t in image_tensors]
                inputs = {}
                inputs['pixel_values'] = pad_sequence(image_tensors, batch_first=True)
                text_latents = self.ruclip_predictor.get_text_latents(batch_labels)
                batch_similarity = self.get_similarity(inputs, text_latents).tolist()
                
                
        df_batch_labels['ruclip_similarity'].extend(batch_similarity)
        df_batch_labels['image_path'].extend(image_paths)
                
        return df_batch_labels
    
    def get_similarity(self, inputs, text_latents):
        if self.onnx:
            logit_scale = np.exp(self.logit_scale)
            image_latents = self.ruclip_model.encode_image(inputs['pixel_values'])
            image_latents = image_latents / np.linalg.norm(image_latents, axis=-1, keepdims=True)
            logits_per_text = np.matmul(text_latents, image_latents.T) * logit_scale
        else:
            with torch.no_grad():
                logit_scale = self.ruclip_model.logit_scale.exp()
                image_latents = self.ruclip_model.encode_image(inputs['pixel_values'])
                image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
                logits_per_text = torch.matmul(text_latents.to(self.ruclip_predictor.device), image_latents.t()) * logit_scale
                logits_per_text = logits_per_text.cpu().numpy()

        return np.diag(logits_per_text)

class Textual(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), torch.where(text == 3)[1]] @ self.text_projection
        return x
