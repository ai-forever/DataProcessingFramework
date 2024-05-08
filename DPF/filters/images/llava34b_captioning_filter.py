from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import re
try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate
from typing import Any
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
        device: str = "cuda:0",
        pbar: bool = True,
        crop_size_x: int = 336,
        crop_size_y: int = 336,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.batch_size = batch_size
        self.num_workers = workers
        self.device = device
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        self.prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nDescribe this image and its style in a very detailed manner<|im_end|><|im_start|>assistant\n"
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        device_map=self.device
        )

    @property
    def result_columns(self) -> list[str]:
        return ["llava34b_caption"]

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
        pil_img = read_image_rgb_from_bytes(modality2data['image']).convert('RGB')
        width, height = pil_img.size  
        left = int((width - self.crop_size_x)/2)
        top = int((height - self.crop_size_y)/2)
        right = int((width + self.crop_size_x)/2)
        bottom = int((height + self.crop_size_y)/2)
        cropped_image = pil_img.crop((left, top, right, bottom))
        cropped_image = cropped_image.resize((self.crop_size_x, self.crop_size_y))
        return key, cropped_image
    
    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()
        keys, images = list(zip(*batch))
        prompts = [self.prompt for _ in range(self.batch_size)]
        inputs = self.processor(prompts, list(images), return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=512, use_cache=True)
            
        all_outputs = []
        for i in range(output_ids.shape[0]):
            output = self.processor.decode(output_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output = re.sub(r'.*?assistant', '', output, flags=re.DOTALL)
            all_outputs.append(output) 
            
        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys) 
        
        return df_batch_labels     
    