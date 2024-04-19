import os
import warnings
from io import BytesIO
from typing import Any, Optional

import gdown
import torch
from lita.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    TIME_TOKEN_TEMPLATE,
)
from lita.model.language_model.lita_llama import LitaLlamaForCausalLM
from lita.utils import load_video
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate


def load_pretrained_model(model_path: str,
                          model_base: str,
                          model_name: str,
                          load_8bit: bool = False,
                          load_4bit: bool = False,
                          device_map: str = "auto",
                          device: str = "cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}  # type: ignore

    if load_8bit:
        kwargs['load_in_8bit'] = True  # type: ignore
    elif load_4bit:
        kwargs['load_in_4bit'] = True  # type: ignore
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16  # type: ignore

    if 'lita' not in model_name.lower():
        warnings.warn("this function is for loading LITA models", stacklevel=2)
    if 'lora' in model_name.lower():
        warnings.warn("lora is currently not supported for LITA", stacklevel=2)
    if 'mpt' in model_name.lower():
        warnings.warn("mpt is currently not supported for LITA", stacklevel=2)

    if model_base is not None:
        print('Loading LITA from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = LitaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items() if 'mm_projector' in k}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LitaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # time tokens and embeddings
    num_time_tokens = getattr(model.config, "num_time_tokens", 0)
    if num_time_tokens > 0:
        time_tokens = [TIME_TOKEN_TEMPLATE.format(t=x) for x in range(num_time_tokens)]
        num_new_tokens = tokenizer.add_tokens(time_tokens)

        if model_base is None:
            assert num_new_tokens == 0, "time tokens should already be in the tokenizer for full finetune model"

        if num_new_tokens > 0:
            warnings.warn("looking for weights in mm_projector.bin", stacklevel=2)
            assert num_new_tokens == num_time_tokens
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            assert 'model.embed_tokens.weight' in weights and 'lm_head.weight' in weights

            dtype = input_embeddings.dtype
            device = input_embeddings.device

            tokenizer_time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
            time_token_ids = getattr(model.config, 'time_token_ids', tokenizer_time_token_ids)
            input_embeddings[tokenizer_time_token_ids] = weights['model.embed_tokens.weight'][time_token_ids].to(dtype).to(device)
            output_embeddings[tokenizer_time_token_ids] = weights['lm_head.weight'][time_token_ids].to(dtype).to(device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    return tokenizer, model, image_processor, context_len


def disable_torch_init() -> None:
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    torch.nn.Linear.reset_parameters = lambda self: None  # type: ignore
    torch.nn.LayerNorm.reset_parameters = lambda self: None  # type: ignore


class LITAFilter(VideoFilter):
    """
    LITA inference class to get captions for auto-labeling videos.
    More info about the model here: https://github.com/NVlabs/LITA
    """
    def __init__(
        self,
        weights_path: str = "./lita-vicuna-v1-3-13b-finetune",
        model_base: Optional[str] = None,
        prompt: str = "detailed_video",
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
        load_4bit: bool = False,
        load_8bit: bool = False,
        device: str = "cuda:0",
        workers: int = 16,
        batch_size: int = 8,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.model_name = get_model_name_from_path(weights_path)
        self.prompt_to_use = prompt
        prompt_templates = {
            'detailed_video': 'Describe this video and its style in a very detailed manner',
            'short_video': 'Describe this video and its style briefly'
        }

        self.num_workers = workers
        self.batch_size = batch_size
        self.device = device

        self.inp = prompt_templates[self.prompt_to_use]
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        weights_url = "https://drive.google.com/drive/folders/1-P7p-tq5aXZzSoefEJx4PSFKH8jt8KWy"
        if not os.path.exists(weights_path):
            gdown.download_folder(weights_url)

        disable_torch_init()

        pretrainers = load_pretrained_model(weights_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)  # type: ignore
        self.tokenizer, self.model, self.processor, self.context_len = pretrainers

        self.model_num_frames = self.model.config.num_frames

        self.conv_mode = "llava_v1"
        self.conv = conv_templates[self.conv_mode].copy()

        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + self.inp
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()
        self.input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

    @property
    def result_columns(self) -> list[str]:
        return [f"caption {self.model_name} prompt {self.prompt_to_use}"]

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
        video_file = BytesIO(modality2data['video'])
        video_file = load_video(video_file, self.processor, self.model_num_frames).unsqueeze(0).half()
        return key, video_file

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, video_tensors = list(zip(*batch))

        video_tensors = default_collate(video_tensors).to(self.device)  # type: ignore
        input_ids_batch = self.input_ids.repeat_interleave(video_tensors.shape[0], 0).to(self.device)  # type: ignore

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_batch,
                images=video_tensors[:, 0],  # type: ignore
                do_sample=True,
                temperature=self.temperature,
                top_p=0.85,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True
            )

        all_outputs: list[Optional[str]] = []
        for i in range(output_ids.shape[0]):
            caption = self.tokenizer.decode(output_ids[i, self.input_ids.shape[1]:]).strip().split('</s>')[0]
            all_outputs.append(caption)
        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)
        return df_batch_labels
