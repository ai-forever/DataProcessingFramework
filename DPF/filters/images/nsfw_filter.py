import os
import numpy as np
from .img_filter import ImageFilter
import clip
from urllib.request import urlretrieve
import zipfile
from DPF.utils import read_image_rgb_from_bytes
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.models import load_model


def load_safety_model(clip_model, cache_folder, device):
    """load the safety model"""
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
        url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
    else:
        raise ValueError("Unsupported clip model")
        
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)
        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        urlretrieve(url_model, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)
            
    with tf.device(device):
        loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)

    return loaded_model


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class NSFWFilter(ImageFilter):

    def __init__(self, clip_model, weights_folder, task_name=None, save_parquets_dir=None,
                 save_parquets=False, pbar=True, workers=16, batch_size=64, device='cuda:0'):
        super(NSFWFilter, self).__init__(task_name, save_parquets, save_parquets_dir, pbar)

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device
        
        self.weights_folder = weights_folder
        self.clip_model, self.clip_transforms = clip.load(clip_model, device=self.device, download_root=weights_folder)
        self.safety_model = load_safety_model(clip_model, weights_folder, device=self.device.lower().replace('cuda', 'gpu'))

        self.schema = ['image_path', 'nsfw']
        self.dataloader_kwargs = dict(
            num_workers=self.num_workers, batch_size=self.batch_size,
            preprocess_f=self.preprocess, collate_fn=self.collate_fn,
            drop_last=False
        )

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
            image_features = self.clip_model.encode_image(batch)
            emb = np.asarray(normalized(image_features.detach().cpu()))
        nsfw_values = self.safety_model.predict(emb, batch_size=self.batch_size, verbose=0).reshape(-1).tolist()
        df_batch_labels['nsfw'].extend(nsfw_values)
        df_batch_labels['image_path'].extend(image_paths)

        return df_batch_labels