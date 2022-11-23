from .img_filter import ImageFilter
from PIL import Image
import numpy as np
import clip
import os
from clip_onnx import clip_onnx, attention
from DPF.utils import read_image_rgb_from_bytes

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate
import torch


class CLIPLabelsFilter(ImageFilter):

    def __init__(self, clip_model, labels, weights_folder, templates=['{}', 'photo of a {}'], task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, workers=16, batch_size=64, device='cuda:0', use_onnx=False):
        super(CLIPLabelsFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.templates = templates
        self.labels = labels
        self.weights_folder = weights_folder
        self.onnx = use_onnx
        
        if self.onnx:
            visual_path = os.path.join(self.weights_folder, f'{clip_model.replace("/", "_")}_visual.onnx')
            textual_path = os.path.join(self.weights_folder, f'{clip_model.replace("/", "_")}_textual.onnx')
            try:
                _, self.clip_processor = clip.load(clip_model, device="cpu", jit=False, download_root=weights_folder)
                self.clip_model = clip_onnx(None)
                self.clip_model.load_onnx(visual_path=visual_path,
                                          textual_path=textual_path,
                                          logit_scale=100.0000)
                self.clip_model.start_sessions(providers=['CUDAExecutionProvider'])
            except:
                self.clip_model, self.clip_processor = clip.load(clip_model, device="cpu", jit=False, download_root=weights_folder)
                image = np.random.rand(100, 100, 3) * 255
                image = self.clip_processor(Image.fromarray(image.astype('uint8')).convert('RGBA')).unsqueeze(0).cpu()
                text = clip.tokenize(['picture']).cpu()
                self.clip_model = clip_onnx(self.clip_model, visual_path=visual_path, textual_path=textual_path)
                self.clip_model.convert2onnx(image, text, verbose=True)
                self.clip_model.start_sessions(providers=['CUDAExecutionProvider'])
        else:
            self.clip_model, self.clip_processor = clip.load(clip_model, device=self.device, download_root=weights_folder)
            
        self.text_features = self.get_text_features()

        self.schema = ['image_path'] + self.labels
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=self.collate_fn,
            drop_last=False
        )

    def get_text_features(self):
        text_features = []
        if self.onnx:
            for template in self.templates:
                texts = clip.tokenize([template.format(class_label.lower().strip()) for class_label in self.labels])
                text_features.append(self.clip_model.encode_text(texts.detach().cpu().numpy().astype(np.int64)))
            text_features = np.stack(text_features).mean(0)
            text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
        else:
            for template in self.templates:
                texts = clip.tokenize([template.format(class_label.lower().strip()) for class_label in self.labels])
                text_features.append(self.clip_model.encode_text(texts.to(self.device)))
            text_features = torch.stack(text_features).mean(0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def preprocess(self, img_bytes, data):
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
            df_batch_labels[label] += [i[c] for i in probs]
        df_batch_labels['image_path'].extend(image_paths)

        return df_batch_labels
