from typing import Any, Dict, List, Union

import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from transformers import AutoTokenizer

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes

from .img_filter import ImageFilter


class LLaVaCaptioningFilter(ImageFilter):
    """
    LLaVa captioning filter
    """

    def __init__(
        self,
        model_path: str = 'liuhaotian/llava-v1.5-13b',
        prompt: str = 'detailed-long',
        workers: int = 16,
        batch_size: int = 16,
        device="cuda:0",
        pbar=True
    ):
        super().__init__(pbar)
        self.batch_size = batch_size
        self.num_workers = workers
        self.device = device

        #
        self.prompt_to_use = prompt
        prompts = {
            'detailed-long': 'Please provide a caption for this image. Speak confidently and describe everything clearly. Do not lie and describe only what you can see',
            'pixart': 'Describe this image and its style in a very detailed manner',
            'short': 'Describe this image very shortly in 1-2 short sentences',
            'short-video': 'Describe this video very shortly in 1-2 short sentences. Describe what is happening in this video.'
        }
        self.prompt = prompts[self.prompt_to_use]
        print(self.prompt)
        #
        self.model_path = model_path
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path).to(self.device).half()
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
        #
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

    @property
    def schema(self) -> List[str]:
        return [self.key_column, f"caption {self.model_path} prompt {self.prompt_to_use}"]

    @property
    def dataloader_kwargs(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        pil_img = read_image_rgb_from_bytes(modality2data['image']).convert('RGB')
        img_tensor = self.image_processor.preprocess(pil_img, return_tensors='pt')['pixel_values'].half()
        return key, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        keys, image_tensors = list(zip(*batch))
        image_tensors = default_collate(image_tensors).to(self.device)

        input_ids_batch = self.input_ids.repeat_interleave(image_tensors.shape[0], 0).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_batch, images=image_tensors, do_sample=True, temperature=0.2, top_p=0.7,
                max_new_tokens=512, use_cache=True, stopping_criteria=[self.stopping_criteria]
            )

        all_outputs = []
        for i in range(output_ids.shape[0]):
            output = self.tokenizer.decode(output_ids[i, self.input_ids.shape[1]:]).strip().split('</s>')[0]
            all_outputs.append(output)

        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
