import re
from typing import Any

import torch
from torchvision import transforms as T
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from DPF.filters.images.img_filter import ImageFilter
from DPF.types import ModalityToDataMapping
from DPF.utils import read_image_rgb_from_bytes


class Llava34b_Filter(ImageFilter):
    """
    The filter implements a description of the images supplied to the input using a model llava-v1.6-34b-hf.
    """

    def __init__(
        self,
        model_path: str = 'llava-hf/llava-v1.6-34b-hf',
        workers: int = 16,
        batch_size: int = 8,
        prompt: str = 'detailed-long',
        device: str = "cuda:0",
        pbar: bool = True,
        crop_size_x: int = 336,
        crop_size_y: int = 336,
        resize: int = 336,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.batch_size = batch_size
        self.num_workers = workers
        self.device = device
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        self.resize = resize
        self.model_path = model_path
        self.prompt_to_use = prompt
        prompts = {
            'detailed-long': 'Please provide a caption for this image. Speak confidently and describe everything clearly. Do not lie and describe only what you can see',
            'pixart': 'Describe this image and its style in a very detailed manner',
            'short': 'Describe this image very shortly in 1-2 short sentences',
            'short-video': 'Describe this video very shortly in 1-2 short sentences. Describe what is happening in this video.'
        }
        self.input_ids = prompts[self.prompt_to_use]
        print(self.input_ids)
        self.prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n" +  f"{self.input_ids}" + "<|im_end|><|im_start|>assistant\n"
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map=self.device
        )

    @property
    def result_columns(self) -> list[str]:
        return [f"caption {self.model_path}"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "drop_last": False,
        }

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        pil_img = read_image_rgb_from_bytes(
            modality2data['image']).convert('RGB')
        transform = T.Compose([
                    T.Resize(self.resize),
                    T.CenterCrop((self.crop_size_x,self.crop_size_y))
                    ])
        cropped_image = transform(pil_img)
        return key, cropped_image

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()
        keys, images = list(zip(*batch))
        prompts = [self.prompt for _ in range(self.batch_size)]
        inputs = self.processor(prompts, list(
            images), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=512, use_cache=True)

        all_outputs = []
        for i in range(output_ids.shape[0]):
            output = self.processor.decode(
                output_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output = re.sub(r'.*?assistant', '', output, flags=re.DOTALL)
            output = re.sub(r'\n', '', output, count=1)
            all_outputs.append(output)

        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)

        return df_batch_labels
