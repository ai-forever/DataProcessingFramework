import torch
from PIL import Image
import requests
from io import BytesIO
import time
import subprocess
from threading import Thread
from typing import Union, Iterator, List
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from transformers.generation.streamers import TextIteratorStreamer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from typing import Dict, List, Union
import torch
import time

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate

from DPF.utils import read_image_rgb_from_bytes
from DPF.filters.utils import identical_collate_fn
from DPF.filters.images.img_filter import ImageFilter

PROMPTS = {
    'instruct-prompt': """As a vital member of the image generation team, your primary task is to craft detailed captions for images. Adhere to these guidelines for optimal results:
1. **Concise Descriptions:** Focus solely on generating the image caption, omitting introductory phrases.
2. **Single Output:** Provide a single image description in response to each user request.
3. **Incorporate Text:** If the image contains text, integrate it seamlessly into your description.
4. **Word Limit:** Maintain image descriptions within the 15-80 word range; any extra words beyond this limit will be disregarded.

Your ability to produce clear and captivating image captions plays a crucial role in enhancing user experience. Keep the descriptions engaging and within the specified constraints.
""",
    'detailed-long': 'Please provide a caption for this image. Speak confidently and describe everything clearly. Do not lie and describe only what you can see',
}


class LLaVa16_CaptioningFilter(ImageFilter):
    """
    LLaVa captioning filter
    """

    def __init__(
            self,
            model_path: str = 'liuhaotian/llava-v1.6-34b',
            prompt_name: str = 'instruct-prompt',
            workers: int = 16,
            batch_size: int = 16,
            device="cuda:0",
            pbar=True
    ):
        super().__init__(pbar)
        self.batch_size = batch_size
        self.num_workers = workers
        self.device = device

        # setting prompts
        self.prompt_name = prompt_name
        assert self.prompt_name in PROMPTS.keys(), f"Invalid prompt name, available options: {PROMPTS.keys()}"
        self.prompt = PROMPTS[self.prompt_name]
        print(self.prompt)

        # loading model
        self.model_path = model_path
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )

        # model parameters
        self.top_p = 1.0
        self.temperature = 0.2
        self.max_tokens = 240
        self.conv_mode = "llava_v1"
        self.conv = conv_templates[self.conv_mode].copy()

        # preprocessing prompt
        inp = DEFAULT_IMAGE_TOKEN + '\n' + self.prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        raw_prompt = self.conv.get_prompt()

        self.input_ids = tokenizer_image_token(
            raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

        self.schema = [self.key_column, f"caption {self.model_path} prompt {self.prompt_name}"]
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": batch_size,
            "drop_last": False,
        }

    def preprocess(self, modality2data: Dict[str, Union[bytes, str]], metadata: dict):
        key = metadata[self.key_column]
        pil_img = read_image_rgb_from_bytes(modality2data['image']).convert('RGB')
        img_tensor = self.image_processor.preprocess(pil_img, return_tensors='pt')['pixel_values'].squeeze(0).half()
        return key, img_tensor

    def process_batch(self, batch) -> dict:
        df_batch_labels = self._generate_dict_from_schema()

        keys, image_tensors = list(zip(*batch))
        image_tensors = default_collate(image_tensors).to(self.device)
        input_ids_batch = self.input_ids.repeat_interleave(image_tensors.shape[0], 0).to(self.device)

        with torch.inference_mode():
            print(image_tensors.shape)
            print(input_ids_batch.shape)
            output_ids = self.model.generate(
                inputs=input_ids_batch, images=image_tensors,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_tokens,
                use_cache=True,
            )

        all_outputs = []
        for i in range(output_ids.shape[0]):
            output = self.tokenizer.decode(output_ids[i, self.input_ids.shape[1]:]).strip().split('</s>')[0]
            # preprocess output
            output = output.split('USER:')[0].split('ASSISTANT:')[0]
            all_outputs.append(output)

        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels