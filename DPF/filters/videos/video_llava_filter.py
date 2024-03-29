from io import BytesIO
from typing import Any, Optional

import torch
from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import SeparatorStyle, conv_templates
from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from videollava.model.builder import load_pretrained_model

from DPF.types import ModalityToDataMapping

from .video_filter import VideoFilter

try:
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data import default_collate


def disable_torch_init() -> None:
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    torch.nn.Linear.reset_parameters = lambda self: None  # type: ignore
    torch.nn.LayerNorm.reset_parameters = lambda self: None  # type: ignore


def check_caption(caption: str) -> Optional[str]:
    sentences_dict = {}
    sentences = caption.split('.')
    for sentence in sentences:
        if sentence not in sentences_dict:
            sentences_dict[sentence] = 1
        else:
            sentences_dict[sentence] += 1

    if max(sentences_dict.values()) == 1:
        return caption
    else:
        return ""


class VideoLLaVAFilter(VideoFilter):
    """
    Video-LLaVA inference class to get captions for auto-labeling videos.
    More info about the model here: https://github.com/PKU-YuanGroup/Video-LLaVA
    """
    def __init__(
        self,
        model_path: str = "LanguageBind/Video-LLaVA-7B",
        model_name: str = "Video-LLaVA-7B",
        model_base: Optional[str] = None,
        cache_path: Optional[str] = None,
        prompt: str = "detailed_video",
        temperature: float = 0.8,
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
        self.model_name = model_name
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

        disable_torch_init()

        pretrainers = load_pretrained_model(
            model_path, model_base, model_name,
            load_8bit, load_4bit,
            device=self.device, cache_dir=cache_path
        )
        self.tokenizer, self.model, self.processor, self.context_len = pretrainers
        self.video_processor = self.processor['video']

        self.conv_mode = "llava_v1"
        self.conv = conv_templates[self.conv_mode].copy()

        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + self.inp
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
    def schema(self) -> list[str]:
        return [self.key_column, f"caption {self.model_name} prompt {self.prompt_to_use}"]

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
        video_file = self.video_processor(video_file, return_tensors='pt')['pixel_values'][0].half()
        return key, video_file

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, video_tensors = list(zip(*batch))
        video_tensors = default_collate(video_tensors).to(self.device)  # type: ignore

        input_ids_batch = self.input_ids.repeat_interleave(video_tensors.shape[0], 0).to(self.device)  # type: ignore

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_batch,
                images=video_tensors,  # video as fake images
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                no_repeat_ngram_size=2,
                use_cache=True,
                stopping_criteria=[self.stopping_criteria])

        all_outputs: list[Optional[str]] = []
        for i in range(output_ids.shape[0]):
            caption = self.tokenizer.decode(output_ids[i, self.input_ids.shape[1]:]).strip().split('</s>')[0]
            all_outputs.append(check_caption(caption))
        df_batch_labels[self.schema[1]].extend(all_outputs)
        df_batch_labels[self.key_column].extend(keys)
        return df_batch_labels
