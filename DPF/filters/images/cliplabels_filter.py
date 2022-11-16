from .img_filter import ImageFilter
import clip
from DPF.utils import read_image_rgb_from_bytes

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate
import torch


class CLIPLabelsFilter(ImageFilter):

    def __init__(self, clip_model, labels, weights_folder, templates=['{}', 'photo of a {}'], task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, workers=16, batch_size=64, device='cuda:0'):
        super(CLIPLabelsFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.templates = templates
        self.labels = labels
        self.weights_folder = weights_folder
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
        for template in self.templates:
            texts = clip.tokenize([template.format(class_label.lower().strip()) for class_label in self.labels]).to(
                self.device)
            text_features.append(self.clip_model.encode_text(texts))
        text_features = torch.stack(text_features).mean(0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def preprocess(self, img_bytes, data):
        image_path = data['image_path']
        pil_img = read_image_rgb_from_bytes(img_bytes)
        img_tensor = self.clip_processor(pil_img)
        return image_path, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        image_paths, image_tensors = list(zip(*batch))
        batch = default_collate(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_per_image = torch.matmul(image_features, self.text_features.t())
            probs = logits_per_image.cpu().numpy().tolist()

        for c, label in enumerate(self.labels):
            df_batch_labels[label] += [i[c] for i in probs]
        df_batch_labels['image_path'].extend(image_paths)

        return df_batch_labels
