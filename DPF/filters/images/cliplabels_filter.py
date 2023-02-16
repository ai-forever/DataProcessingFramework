from typing import List, Optional
from PIL import Image
import numpy as np
import torch
import clip
import os
from clip_onnx import clip_onnx, attention

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from .img_filter import ImageFilter
from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn


class CLIPLabelsFilter(ImageFilter):
    """
    Filter for perfoming zero-shot with CLIP model.
    
    Parameters
    ----------
    clip_model: str
        Version of model to use. Check available version here: https://github.com/openai/CLIP"
    labels: List[str]
        List of classes for detecting
    weights_folder: str
        Path to folder with weights
    templates: List[str] = ['{}', 'photo of a {}']
        TODO
    use_onnx: bool = False
        TODO
    device: str = 'cuda:0'
        Torch device to use
    workers: int = 16
        Number of processes for use in dataloader
    batch_size: int = 64
        Batch size for model
    pbar: bool = True
        Flag for displaying progress bar
        
    Attributes
    ----------
    schema: List[str]
        List of columns to be added with this filter.
    dataloader_kwargs: dict:
        Parameters for dataloader (batch_size, num_workers, collate_fn, etc.)
    """

    def __init__(
            self, 
            clip_model: str, 
            labels: List[str], 
            weights_folder: str, 
            device: str = 'cuda:0', 
            use_onnx: bool = False,
            templates: List[str] = ['{}', 'photo of a {}'], 
            workers: int = 16, 
            batch_size: int = 64, 
            pbar: bool = True
        ):
        super(CLIPLabelsFilter, self).__init__(pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        
        self.clip_version = clip_model
        self.templates = templates
        self.labels = labels
        self.weights_folder = weights_folder
        self.onnx = use_onnx
        
        if self.onnx:
            visual_path = os.path.join(self.weights_folder, f'{clip_model.replace("/", "_")}_visual.onnx')
            textual_path = os.path.join(self.weights_folder, f'{clip_model.replace("/", "_")}_textual.onnx')
            try:
                ### where is clip_model?
                _, self.clip_processor = clip.load(clip_model, device="cpu", jit=False, download_root=weights_folder) # why device="cpu"?
                self.clip_model = clip_onnx(None)
                self.clip_model.load_onnx(visual_path=visual_path,
                                          textual_path=textual_path,
                                          logit_scale=100.0000) # why 100?
                self.clip_model.start_sessions(providers=['CUDAExecutionProvider'])
            except Exception as err:
                print(err)
                self.clip_model, self.clip_processor = clip.load(clip_model, device="cpu", jit=False, download_root=weights_folder)
                image = np.random.rand(100, 100, 3) * 255
                image = self.clip_processor(Image.fromarray(image.astype('uint8')).convert('RGBA')).unsqueeze(0).cpu()
                text = clip.tokenize(['picture']).cpu()
                self.clip_model = clip_onnx(self.clip_model, visual_path=visual_path, textual_path=textual_path)
                self.clip_model.convert2onnx(image, text, verbose=True)
                self.clip_model.start_sessions(providers=['CUDAExecutionProvider'])
        else:
            self.clip_model, self.clip_processor = clip.load(clip_model, device=self.device, download_root=weights_folder)
        #
        self.text_features = self.get_text_features()
        #
        self.label2column = {l: f'{self.clip_version} clip score "{l}"' for l in self.labels}
        self.schema = ['image_path'] + [self.label2column[l] for l in self.labels]
        #
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=identical_collate_fn,
            drop_last=False
        )

    def get_text_features(self):
        text_features = []
        if self.onnx:
            for template in self.templates:
                texts = clip.tokenize([template.format(class_label.strip()) for class_label in self.labels])
                text_features.append(self.clip_model.encode_text(texts.detach().cpu().numpy().astype(np.int32)))
            text_features = np.stack(text_features).mean(0)
            text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        else:
            for template in self.templates:
                texts = clip.tokenize([template.format(class_label.strip()) for class_label in self.labels])
                text_features.append(self.clip_model.encode_text(texts.to(self.device)))
            text_features = torch.stack(text_features).mean(0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def preprocess(self, img_bytes: bytes, data: dict):
        image_path = data['image_path']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        if self.onnx:
            img_tensor = self.clip_processor(pil_img)
            img_tensor = img_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            img_tensor = self.clip_processor(pil_img)
        return image_path, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        image_paths, image_tensors = list(zip(*batch))

        with torch.no_grad():
            if self.onnx:
                batch = default_collate(image_tensors).cpu().numpy()
                image_features = self.clip_model.encode_image(batch)
                image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
                logits_per_image = np.matmul(image_features, self.text_features.T)
                probs = logits_per_image.tolist()
            else:
                batch = default_collate(image_tensors).to(self.device)
                image_features = self.clip_model.encode_image(batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits_per_image = torch.matmul(image_features, self.text_features.t())
                probs = logits_per_image.cpu().numpy().tolist()

        for c, label in enumerate(self.labels):
            df_batch_labels[self.label2column[label]] += [i[c] for i in probs]
        df_batch_labels['image_path'].extend(image_paths)

        return df_batch_labels
