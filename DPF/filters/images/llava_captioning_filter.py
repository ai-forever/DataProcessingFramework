from typing import Dict, List, Union
import torch
import time

from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch
import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from .img_filter import ImageFilter


class LLaVaCaptioningFilter(ImageFilter):
    """
    LLaVa captioning filter
    """

    def __init__(
        self, 
        model_path: str = "4bit/llava-v1.5-13b-3GB", 
        prompt: str = 'detailed-long', 
        workers: int = 16, 
        device="cuda:0", 
        pbar=True
    ):
        super().__init__(pbar)

        self.num_workers = workers
        self.device = device
        
        #
        self.prompt_to_use = prompt
        prompts = {
            'detailed-long': 'Please provide a caption for this image. Describe it as if it were in a dataset for an image generation model. Speak confidently and describe everything clearly. Caption should be short but detailed. Please, dont lie and describe only what you can see'
        }
        self.prompt = prompts[self.prompt_to_use]
        #
        self.model_path = model_path
        kwargs = {"device_map": self.device}
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        self.model = LlavaLlamaForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        #
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device)
        self.image_processor = vision_tower.image_processor
        #
        # preprocessing prompt
        disable_torch_init()
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        inp = f"{roles[0]}: {self.prompt}"
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        raw_prompt = conv.get_prompt()
        self.input_ids = tokenizer_image_token(
            raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

        self.schema = [self.key_column, f"caption {self.model_path} prompt {self.prompt_to_use}"]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": 1,
            "collate_fn": identical_collate_fn,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        try:
            pil_img = read_image_rgb_from_bytes(modality2data['image']).convert('RGB')
            img_tensor = self.image_processor.preprocess(pil_img, return_tensors='pt')['pixel_values'].half()
            return key, img_tensor
        except Exception as err:
            return key, None

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        keys, image_tensors = list(zip(*batch))
        key = keys[0]
        if image_tensors[0] is not None:
            image_tensor = image_tensors[0].to(self.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    self.input_ids, images=image_tensor, do_sample=True, temperature=0.2, 
                    max_new_tokens=1024, use_cache=True, stopping_criteria=[self.stopping_criteria]
                )
            output = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()[:-4]
        else:
            output = None
            print('error:', key)

        df_batch_labels[self.schema[1]].append(output)
        df_batch_labels[self.key_column].append(key)

        return df_batch_labels
